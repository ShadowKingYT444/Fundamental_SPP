#!/usr/bin/env python3
"""Walk-forward training harness for the Fundamental_SPP reproduction.

Usage (from ~/workspace/fundamental_spp):
    python src/train.py --model miss --regime fund63 --year 2021
    python src/train.py --model gnn --regime tech63 --year 2023 --synthetic

Walk-forward per SPEC: for test year Y, train = 2010-01-01..(Y-2)-12-31
(expanding), val = full year (Y-1) scored DAILY. Training samples use stride 21
trading days. Adam lr 1e-3, batch 256, <=10 epochs, early stop (patience 2) on
val RankIC = mean over val dates of Spearman(score, target).

PANEL CONTRACT (data/panel_<regime>.parquet, written by workstream 2):
  - columns: ticker (str), date (str YYYY-MM-DD), sector (str), close (float),
             target_63d (float), target_5d (float),
             feat_0 .. feat_{L*F-1} (float32).
  - feat_{t*F+f} = feature f at window position t, t=0 oldest, t=L-1 label date,
    with features in REGIMES[regime]['features'] order (see below).
  - one row per (ticker, label date); rows sorted by date ASC (required for the
    chunked date-group streamer).
  - rows with NaN features/targets should already be dropped; a defensive drop
    is applied anyway.

The loader streams the parquet in chunks (never loads the full panel into RAM):
  - pass 1: collect unique dates in the train/val ranges (date column only).
  - pass 2 (unless --skip-normalize): sample train rows -> per-feature 1%/99%
    winsorization bounds from TRAINING data only.
  - per epoch: stream date groups; winsorize + cross-sectional z-score per date
    (computed on that date's own cross-section, no leakage); train / validate.

Checkpoints: checkpoints/<model>_<regime>_<year>.pt with metadata.
Logs: logs/ws3.log (plus stdout).
"""

import argparse
import json
import logging
import os
import random
import sys
import time
from collections import deque

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models import build_model, count_parameters, build_graph  # noqa: E402
from losses import combined_loss  # noqa: E402
import synthetic as synth  # noqa: E402

# ---------------------------------------------------------------------------
# regimes / features
# ---------------------------------------------------------------------------

TECH_FEATURES = [
    'ret_5', 'ret_21', 'ret_63', 'ret_252',
    'vol_21', 'vol_63',
    'ma21_dist', 'ma63_dist', 'ma200_dist',
    'rsi_14', 'macd', 'macd_signal',
    'atr_14', 'vol_surprise', 'pv_corr63',
]
FUND_FEATURES = [
    'roa', 'op_margin', 'rev_growth', 'earn_growth', 'earn_yield',
    'fcf_yield', 'leverage', 'liquidity', 'accruals',
]
REGIMES = {
    'fund63': {'features': TECH_FEATURES + FUND_FEATURES, 'target': 'target_63d'},
    'tech63': {'features': TECH_FEATURES, 'target': 'target_63d'},
    'tech5': {'features': TECH_FEATURES, 'target': 'target_5d'},
}
SEQ_LEN = 252
PRICE_WINDOW = 63
GRAPH_REBUILD_EVERY = 21  # trading days ("recomputed periodically")

# models that must be scored on whole-date cross-sections (cross-stock mixing)
CROSS_SECTIONAL_MODELS = {'stockmixer', 'gnn'}

log = logging.getLogger('ws3')


# ---------------------------------------------------------------------------
# logging
# ---------------------------------------------------------------------------

def setup_logging(log_path):
    os.makedirs(os.path.dirname(log_path) or '.', exist_ok=True)
    log.setLevel(logging.INFO)
    fmt = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    fh = logging.FileHandler(log_path)
    fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    log.handlers = [fh, sh]


# ---------------------------------------------------------------------------
# chunked panel streaming
# ---------------------------------------------------------------------------

def _chunk_rows(pf, n_feat_cols, target_bytes=64 * 1024 * 1024):
    n_cols = n_feat_cols + 8
    rows = max(2000, int(target_bytes / (n_cols * 4)))
    return rows


def iter_date_groups(path, columns, date_set, n_feat_cols, chunk_rows=None):
    """Yield (date, feat_mat, meta) for dates in date_set, chronological.

    feat_mat: np.float32 (N, L*F); meta: dict of per-row metadata arrays
    (ticker/sector/close/target_col). Uses pyarrow directly -- a pandas
    DataFrame with 6000+ columns costs GBs of RAM per date group; the numpy
    path costs megabytes. Requires rows sorted by date ascending
    (see PANEL CONTRACT). The last group of each chunk is held back in case
    it continues in the next chunk.
    """
    pf = pq.ParquetFile(path)
    feat_cols = [c for c in columns if c.startswith('feat_')]
    meta_cols = [c for c in columns if not c.startswith('feat_') and c != 'date']
    if chunk_rows is None:
        chunk_rows = _chunk_rows(pf, n_feat_cols)
    want = set(date_set) if date_set is not None else None

    carry_date, carry_feat, carry_meta = None, None, None

    def take_carry():
        nonlocal carry_date, carry_feat, carry_meta
        out = None
        if carry_date is not None:
            out = (carry_date, carry_feat,
                   {k: np.concatenate(v) for k, v in carry_meta.items()})
            carry_date, carry_feat, carry_meta = None, None, None
        return out

    for batch in pf.iter_batches(batch_size=chunk_rows, columns=columns):
        tbl = batch
        dates = np.array([str(d) for d in tbl.column('date').to_pylist()])
        if want is not None:
            mask = np.fromiter((d in want for d in dates), dtype=bool,
                               count=len(dates))
            if not mask.any():
                continue
            tbl = tbl.filter(pa.array(mask))
            dates = dates[mask]
        order = np.argsort(dates, kind='mergesort')
        dates = dates[order]
        feats = np.stack([tbl.column(c).to_numpy()[order] for c in feat_cols],
                         axis=1).astype(np.float32, copy=False)
        meta = {}
        for c in meta_cols:
            col = tbl.column(c)
            try:
                vals = col.to_numpy()
            except Exception:
                vals = np.array(col.to_pylist())
            meta[c] = vals[order]
        del tbl
        bounds = np.flatnonzero(dates[1:] != dates[:-1]) + 1
        starts = np.concatenate([[0], bounds])
        ends = np.concatenate([bounds, [len(dates)]])
        for i, s in enumerate(starts):
            e = int(ends[i])
            d = str(dates[s])
            fmat = feats[s:e]
            m = {k: v[s:e] for k, v in meta.items()}
            if carry_date is not None and d == carry_date:
                carry_feat = np.concatenate([carry_feat, fmat], axis=0)
                for k in carry_meta:
                    carry_meta[k].append(m[k])
            else:
                out = take_carry()
                if out is not None:
                    yield out
                carry_date = d
                carry_feat = fmat
                carry_meta = {k: [v] for k, v in m.items()}
    out = take_carry()
    if out is not None:
        yield out


def panel_dates_in_range(path, lo, hi):
    """Streaming unique-date collection (reads only the date column)."""
    pf = pq.ParquetFile(path)
    dates = set()
    for batch in pf.iter_batches(batch_size=200_000, columns=['date']):
        d = batch.to_pandas()['date']
        dates.update(d[(d >= lo) & (d <= hi)].unique().tolist())
    return sorted(dates)


# ---------------------------------------------------------------------------
# normalization
# ---------------------------------------------------------------------------

def fit_winsor_bounds(path, read_cols, train_dates, max_rows=10_000, seed=42):
    """Sample train rows -> per-feature 1st/99th percentiles (train data only).

    Bounded memory: reads 10k-row chunks, keeps float32, caps at max_rows.
    """
    pf = pq.ParquetFile(path)
    cols = ['date'] + read_cols
    train_set = set(train_dates)
    chunks = []
    n = 0
    rng = np.random.default_rng(seed)
    for batch in pf.iter_batches(batch_size=10_000, columns=cols):
        df = batch.to_pandas()
        df = df[df['date'].isin(train_set)]
        if df.empty:
            del df
            continue
        m = df[read_cols].to_numpy(dtype=np.float32)
        del df
        if n + len(m) > max_rows:
            take = max_rows - n
            idx = rng.choice(len(m), take, replace=False)
            chunks.append(m[idx])
            n = max_rows
            break
        chunks.append(m)
        n += len(m)
    if not chunks:
        raise SystemExit('no train rows found for winsor bounds')
    mat = np.concatenate(chunks, axis=0)
    del chunks
    lo = np.nanpercentile(mat, 1, axis=0)
    hi = np.nanpercentile(mat, 99, axis=0)
    del mat
    log.info('winsor bounds fit on %d train rows', n)
    return lo.astype(np.float32), hi.astype(np.float32)


def group_to_tensors(feat_mat, meta, target_col, w_lo, w_hi, L=SEQ_LEN, F_=None):
    """(feat_mat (N,L*F) float32, meta dict) -> (X (N,L,F), y (N,), ...).

    Winsorize with train bounds, then cross-sectional z-score per date using
    that date's own cross-section. Rows with NaN target/features are dropped.
    """
    if F_ is None:
        F_ = feat_mat.shape[1] // L
    mat = np.ascontiguousarray(feat_mat, dtype=np.float32)
    y = np.asarray(meta[target_col], dtype=np.float32)
    valid = np.isfinite(y) & np.isfinite(mat).all(axis=1)
    mat, y = mat[valid], y[valid]
    tickers = np.asarray(meta['ticker'])[valid].tolist()
    sectors = np.asarray(meta['sector'])[valid]
    closes = np.asarray(meta['close'], dtype=np.float64)[valid]
    if w_lo is not None:
        mat = np.clip(mat, w_lo, w_hi)
    mu = mat.mean(axis=0, keepdims=True)
    sd = mat.std(axis=0, keepdims=True)
    sd = np.where(sd < 1e-8, 1.0, sd)
    mat = (mat - mu) / sd
    X = mat.reshape(len(mat), L, F_)
    return (torch.from_numpy(X), torch.from_numpy(y),
            tickers, sectors, closes)


def maybe_rebuild_graph(graph_state, tickers, sectors, price_hist):
    """Rebuild the stock graph every GRAPH_REBUILD_EVERY trading days, or
    whenever the ticker set changes (node indices must match batch order)."""
    tickers = list(tickers)
    n = graph_state['n'] + 1
    graph_state['n'] = n
    if (n == 1 or n % GRAPH_REBUILD_EVERY == 0
            or tickers != graph_state.get('tickers')):
        graph = build_graph(sectors, price_hist.matrix(tickers))
        graph_state['graph'] = graph
        graph_state['tickers'] = tickers
        graph_state['e'] = graph.shape[1]
    return graph_state['graph']


def rank_ic(dates_scores):
    """Mean over dates of Spearman(score, target). dates_scores: list of (s, y)."""
    ics = []
    for s, y in dates_scores:
        s = np.asarray(s, dtype=float)
        y = np.asarray(y, dtype=float)
        m = np.isfinite(s) & np.isfinite(y)
        if m.sum() < 10:
            continue
        r = spearmanr(s[m], y[m]).statistic
        if np.isfinite(r):
            ics.append(r)
    return float(np.mean(ics)) if ics else float('nan')


# ---------------------------------------------------------------------------
# GNN price history / graph
# ---------------------------------------------------------------------------

class PriceHistory:
    """Per-ticker trailing close buffers for rolling correlation graphs."""

    def __init__(self, window=PRICE_WINDOW):
        self.window = window
        self.buf = {}

    def update(self, tickers, closes):
        for t, c in zip(tickers, closes):
            d = self.buf.get(t)
            if d is None:
                d = deque(maxlen=self.window)
                self.buf[t] = d
            d.append(float(c))

    def matrix(self, tickers):
        W = self.window
        out = np.full((len(tickers), W), np.nan)
        for i, t in enumerate(tickers):
            d = self.buf.get(t)
            if d:
                arr = np.array(d, dtype=float)
                out[i, -len(arr):] = arr
        # daily returns
        with np.errstate(divide='ignore', invalid='ignore'):
            rets = out[:, 1:] / out[:, :-1] - 1.0
        return rets


# ---------------------------------------------------------------------------
# training / evaluation
# ---------------------------------------------------------------------------

def train_one_epoch(model, opt, path, read_cols, target_col, train_dates,
                    w_lo, w_hi, batch_size, epoch, model_name, price_hist=None,
                    graph=None, graph_state=None, seed=42):
    model.train()
    gen = torch.Generator().manual_seed(seed + epoch)
    total_loss, n_batches = 0.0, 0
    comp_h, comp_r = 0.0, 0.0
    cols = ['ticker', 'date', 'sector', 'close', target_col] + read_cols

    if model_name in CROSS_SECTIONAL_MODELS:
        # whole-date cross-section batches; chronological (GNN price history)
        for date, feat_mat, meta in iter_date_groups(path, cols, set(train_dates), len(read_cols)):
            X, y, tickers, sectors, closes = group_to_tensors(
                feat_mat, meta, target_col, w_lo, w_hi, F_=len(read_cols) // SEQ_LEN)
            if len(X) == 0:
                continue
            if model_name == 'gnn':
                price_hist.update(tickers, closes)
                graph = maybe_rebuild_graph(graph_state, tickers, sectors,
                                            price_hist)
                scores = model(X, graph)
            else:
                scores = model(X)
            loss, comps = combined_loss(scores, y, generator=gen)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total_loss += loss.item()
            comp_h += comps['huber'].item()
            comp_r += comps['rank'].item()
            n_batches += 1
    else:
        # shuffle-buffer batching across date groups (bounded memory)
        buf_X, buf_y = [], []
        for date, feat_mat, meta in iter_date_groups(path, cols, set(train_dates), len(read_cols)):
            X, y, *_ = group_to_tensors(feat_mat, meta, target_col, w_lo, w_hi,
                                                 F_=len(read_cols) // SEQ_LEN)
            if len(X) == 0:
                continue
            buf_X.append(X)
            buf_y.append(y)
            pool_X = torch.cat(buf_X)
            pool_y = torch.cat(buf_y)
            if len(pool_X) >= 2048:
                perm = torch.randperm(len(pool_X), generator=gen)
                pool_X, pool_y = pool_X[perm], pool_y[perm]
                n_full = (len(pool_X) // batch_size) * batch_size
                for b in range(0, n_full, batch_size):
                    xb, yb = pool_X[b:b + batch_size], pool_y[b:b + batch_size]
                    scores = model(xb)
                    loss, comps = combined_loss(scores, yb, generator=gen)
                    opt.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    opt.step()
                    total_loss += loss.item()
                    comp_h += comps['huber'].item()
                    comp_r += comps['rank'].item()
                    n_batches += 1
                buf_X = [pool_X[n_full:]] if n_full < len(pool_X) else []
                buf_y = [pool_y[n_full:]] if n_full < len(pool_y) else []
        # flush remainder
        if buf_X:
            pool_X = torch.cat(buf_X)
            pool_y = torch.cat(buf_y)
            perm = torch.randperm(len(pool_X), generator=gen)
            pool_X, pool_y = pool_X[perm], pool_y[perm]
            for b in range(0, len(pool_X), batch_size):
                xb, yb = pool_X[b:b + batch_size], pool_y[b:b + batch_size]
                if len(xb) < 8:
                    continue
                scores = model(xb)
                loss, comps = combined_loss(scores, yb, generator=gen)
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                total_loss += loss.item()
                comp_h += comps['huber'].item()
                comp_r += comps['rank'].item()
                n_batches += 1
    if n_batches == 0:
        return float('nan'), float('nan'), float('nan')
    return total_loss / n_batches, comp_h / n_batches, comp_r / n_batches


@torch.no_grad()
def evaluate(model, path, read_cols, target_col, val_dates, w_lo, w_hi,
             model_name, pre_dates=()):
    """Daily scoring on val dates -> RankIC. Single streaming pass."""
    model.eval()
    date_set = set(pre_dates) | set(val_dates)
    val_set = set(val_dates)
    cols = ['ticker', 'date', 'sector', 'close', target_col] + read_cols
    price_hist = PriceHistory()
    graph, eval_state = None, {'n': 0}
    per_date = []
    for date, feat_mat, meta in iter_date_groups(path, cols, date_set, len(read_cols)):
        X, y, tickers, sectors, closes = group_to_tensors(
            feat_mat, meta, target_col, w_lo, w_hi, F_=len(read_cols) // SEQ_LEN)
        if model_name == 'gnn':
            price_hist.update(tickers, closes)
            graph = maybe_rebuild_graph(eval_state, tickers, sectors, price_hist)
        if date not in val_set or len(X) == 0:
            continue
        if model_name == 'gnn':
            scores = model(X, graph).cpu().numpy()
        else:
            scores = model(X).cpu().numpy()
        per_date.append((scores, y.numpy()))
    ic = rank_ic(per_date)
    return ic, len(per_date)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def parse_args(argv=None):
    p = argparse.ArgumentParser(description='Walk-forward training harness')
    p.add_argument('--model', required=True, choices=['miss', 'lstm', 'stockmixer', 'gnn'])
    p.add_argument('--regime', required=True, choices=list(REGIMES))
    p.add_argument('--year', required=True, type=int)
    p.add_argument('--data-dir', default='data')
    p.add_argument('--panel-path', default=None, help='override panel parquet path')
    p.add_argument('--checkpoint-dir', default='checkpoints')
    p.add_argument('--epochs', type=int, default=10)
    p.add_argument('--batch-size', type=int, default=256)
    p.add_argument('--stride', type=int, default=21)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--patience', type=int, default=2)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--skip-normalize', action='store_true')
    p.add_argument('--log', default='logs/ws3.log')
    p.add_argument('--synthetic', action='store_true',
                   help='generate a synthetic panel on the fly and train on it')
    p.add_argument('--synth-stocks', type=int, default=200)
    p.add_argument('--synth-dpy', type=int, default=63)
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    setup_logging(args.log)
    t0 = time.time()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    regime = REGIMES[args.regime]
    feat_cols = regime['features']
    target_col = regime['target']
    F_ = len(feat_cols)
    Y = args.year
    train_lo, train_hi = '2010-01-01', f'{Y - 2}-12-31'
    val_year = Y - 1

    if args.synthetic:
        panel_path = os.path.join(args.data_dir, 'synth',
                                  f'panel_{args.regime}.parquet')
        os.makedirs(os.path.dirname(panel_path), exist_ok=True)
        log.info('generating synthetic panel -> %s', panel_path)
        meta = synth.generate_panel(panel_path, n_stocks=args.synth_stocks,
                                    dpy=args.synth_dpy, L=SEQ_LEN, F=F_,
                                    seed=args.seed)
        log.info('synthetic panel: %s', json.dumps(meta))
    else:
        panel_path = args.panel_path or os.path.join(
            args.data_dir, f'panel_{args.regime}.parquet')
    if not os.path.exists(panel_path):
        log.error('panel not found: %s (workstream 2 has not built it yet; '
                  'or pass --synthetic / --panel-path)', panel_path)
        raise SystemExit(2)
    log.info('panel: %s', panel_path)

    # physical window columns: feat_{t*F+f}, t=0 oldest (see PANEL CONTRACT).
    # Logical feature order for this regime is REGIMES[regime]['features'].
    read_cols = [f'feat_{i}' for i in range(SEQ_LEN * F_)]
    _schema_names = set(pq.ParquetFile(panel_path).schema.names)
    missing = [c for c in read_cols if c not in _schema_names]
    if missing:
        raise SystemExit(
            f'panel {panel_path} is missing {len(missing)} window columns '
            f'(e.g. {missing[:3]}). Expected flat columns feat_0..feat_{SEQ_LEN * F_ - 1} '
            f'per the PANEL CONTRACT in src/train.py.')
    for c in ('ticker', 'date', 'sector', 'close', target_col):
        if c not in _schema_names:
            raise SystemExit(f'panel {panel_path} is missing required column {c!r}')

    log.info('collecting panel dates ...')
    train_dates_all = panel_dates_in_range(panel_path, train_lo, train_hi)
    val_dates = panel_dates_in_range(panel_path, f'{val_year}-01-01', f'{val_year}-12-31')
    if not train_dates_all or not val_dates:
        raise SystemExit(f'no train/val dates found for year {Y}')
    train_dates = train_dates_all[::args.stride]
    log.info('train range %s..%s: %d dates -> %d with stride %d; val %d: %d dates',
             train_lo, train_hi, len(train_dates_all), len(train_dates),
             args.stride, val_year, len(val_dates))

    if args.skip_normalize:
        w_lo = w_hi = None
        log.info('normalization skipped (--skip-normalize)')
    else:
        w_lo, w_hi = fit_winsor_bounds(panel_path, read_cols, train_dates_all,
                                       seed=args.seed)

    model = build_model(args.model, F_)
    n_params = count_parameters(model)
    log.info('model=%s regime=%s year=%d params=%d (~%.3fM)',
             args.model, args.regime, Y, n_params, n_params / 1e6)
    assert 50_000 <= n_params <= 500_000, f'param count {n_params} out of ~0.2M range'
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    # 63 trading days before val start, to warm the GNN price history
    pre_dates = train_dates_all[-PRICE_WINDOW:] if args.model == 'gnn' else []
    price_hist = PriceHistory() if args.model == 'gnn' else None
    graph_state = {'n': 0, 'e': 0}

    best_ic, best_epoch, wait = float('-inf'), -1, 0
    ckpt_path = os.path.join(args.checkpoint_dir,
                             f'{args.model}_{args.regime}_{Y}.pt')
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    for epoch in range(args.epochs):
        ep0 = time.time()
        tr_loss, huber_c, rank_c = train_one_epoch(
            model, opt, panel_path, read_cols, target_col, train_dates,
            w_lo, w_hi, args.batch_size, epoch, args.model, price_hist,
            None, graph_state, seed=args.seed)
        val_ic, n_vdates = evaluate(model, panel_path, read_cols, target_col,
                                    val_dates, w_lo, w_hi, args.model, pre_dates)
        log.info('epoch %d/%d  train_loss=%.4f (huber=%.4f rank=%.4f)  '
                 'val_RankIC=%.4f (%d dates)  %.1fs',
                 epoch + 1, args.epochs, tr_loss, huber_c, rank_c,
                 val_ic, n_vdates, time.time() - ep0)
        improved = np.isfinite(val_ic) and val_ic > best_ic
        if improved:
            best_ic, best_epoch, wait = val_ic, epoch, 0
            torch.save({
                'state_dict': model.state_dict(),
                'meta': {
                    'model': args.model, 'regime': args.regime, 'year': Y,
                    'val_rankic': best_ic, 'epoch': epoch + 1,
                    'feature_cols': feat_cols, 'target_col': target_col,
                    'n_params': n_params, 'stride': args.stride,
                    'saved_at': time.strftime('%Y-%m-%d %H:%M:%S'),
                }}, ckpt_path)
            log.info('new best -> checkpoint %s', ckpt_path)
        else:
            wait += 1
            if wait >= args.patience:
                log.info('early stop: no val RankIC improvement for %d epochs',
                         args.patience)
                break

    wall = time.time() - t0
    log.info('DONE model=%s regime=%s year=%d best_val_RankIC=%.4f at epoch %d '
             'wall=%.1fs checkpoint=%s',
             args.model, args.regime, Y, best_ic, best_epoch + 1, wall, ckpt_path)
    print(json.dumps({'model': args.model, 'regime': args.regime, 'year': Y,
                      'best_val_rankic': best_ic, 'best_epoch': best_epoch + 1,
                      'wall_s': wall, 'checkpoint': ckpt_path}))


if __name__ == '__main__':
    main()
