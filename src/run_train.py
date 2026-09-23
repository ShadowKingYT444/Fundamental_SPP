"""Walk-forward training on long-format panels with on-the-fly windows.

Usage:
    python src/run_train.py --model miss --regime fund63 --year 2021

Reuses src/models.py, src/losses.py, src/seqdata.py. Panels are already
winsorized + cross-sectionally z-scored by workstream 2 (thresholds from
2010-2019, strictly past data -- no lookahead), so no further normalization
is applied here.
"""

import argparse
import json
import logging
import os
import random
import resource
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models import build_model, build_graph, count_parameters  # noqa: E402
from losses import combined_loss  # noqa: E402
from seqdata import PanelStore, PriceMatrix, TradingCalendar, SEQ_LEN  # noqa: E402

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')
log = logging.getLogger('run_train')

FUND_FEATURES = ['roa', 'op_margin', 'rev_growth', 'earn_growth', 'earn_yield',
                 'fcf_yield', 'leverage', 'liquidity', 'accruals']
TECH_FEATURES = ['ret_5', 'ret_21', 'ret_63', 'ret_252', 'vol_21', 'vol_63',
                 'ma_dist_21', 'ma_dist_63', 'ma_dist_200', 'rsi_14', 'macd',
                 'macd_signal', 'atr_14', 'vol_surprise', 'pv_trend_63']
REGIMES = {
    # fund63 uses fundamental features only (information-isolation design,
    # SPEC.md: "fund63 (fundamental features -> target_63d)").
    'fund63': {'panel': 'panel_fund63.parquet', 'features': FUND_FEATURES},
    'tech63': {'panel': 'panel_tech63.parquet', 'features': TECH_FEATURES},
    'tech5': {'panel': 'panel_tech5.parquet', 'features': TECH_FEATURES},
}
CROSS_SECTIONAL = {'stockmixer', 'gnn'}


def _rankdata(a):
    tmp = np.argsort(a, kind='mergesort')
    r = np.empty_like(tmp, dtype=float)
    r[tmp] = np.arange(len(a), dtype=float)
    # average ties
    _, inv, cnt = np.unique(a, return_inverse=True, return_counts=True)
    sums = np.bincount(inv, weights=r)
    r = sums[inv] / cnt[inv]
    return r


def rank_ic(dates_scores):
    ics = []
    for s, y in dates_scores:
        s = np.asarray(s, dtype=float)
        y = np.asarray(y, dtype=float)
        m = np.isfinite(s) & np.isfinite(y)
        if m.sum() < 10:
            continue
        rs, ry = _rankdata(s[m]), _rankdata(y[m])
        cs, cy = rs - rs.mean(), ry - ry.mean()
        denom = np.sqrt((cs ** 2).sum() * (cy ** 2).sum())
        r = (cs * cy).sum() / denom if denom > 0 else np.nan
        if np.isfinite(r):
            ics.append(r)
    return float(np.mean(ics)) if ics else float('nan')


def gather_batch(panel, samples, idx):
    Xs = np.stack([panel.get_window(ti, pos)[0] for ti, pos in
                   (samples[i] for i in idx)])
    ys = np.array([panel.get_window(ti, pos)[1] for ti, pos in
                   (samples[i] for i in idx)], dtype=np.float32)
    return torch.from_numpy(Xs), torch.from_numpy(ys)


def build_train_arrays(panel, samples):
    """Materialize all training windows once: X (N,L,F), y (N,) torch tensors.

    NOTE (2026-09-23): no longer used for training -- the 867 MB materialized
    array kept the process near the OOM-killer edge on the 7.7 GiB box.
    Training now streams windows per batch via ``batch_windows`` (identical
    values, a pure implementation detail). Kept for reference/tests.
    """
    N = len(samples)
    L, F = SEQ_LEN, panel.F
    X = torch.empty(N, L, F, dtype=torch.float32)
    y = torch.empty(N, dtype=torch.float32)
    for i, (ti, pos) in enumerate(samples):
        w, yv = panel.get_window(ti, pos)
        X[i] = torch.from_numpy(w)
        y[i] = float(yv)
    return X, y


def batch_windows(panel, samples, idx):
    """Stack (B, L, F) windows for one batch directly from the panel.

    Same values as fancy-indexing the materialized array, without the
    ~867 MB resident copy. ``idx`` is an iterable of positions into
    ``samples`` (already permuted by the caller).
    """
    ws, ys = [], []
    for j in idx:
        ti, pos = samples[int(j)]
        w, yv = panel.get_window(ti, pos)
        ws.append(w)
        ys.append(yv)
    Xb = np.stack(ws).astype(np.float32, copy=False)
    yb = np.asarray(ys, dtype=np.float32)
    return torch.from_numpy(Xb), torch.from_numpy(yb)


def _mem_gb():
    """(process RSS GB, system available GB) for the training log."""
    try:
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
    except Exception:
        rss = float('nan')
    try:
        avail = None
        with open('/proc/meminfo') as f:
            for line in f:
                if line.startswith('MemAvailable'):
                    avail = int(line.split()[1]) / 1e6
                    break
        avail = float('nan') if avail is None else avail
    except Exception:
        avail = float('nan')
    return rss, avail


def train_one_epoch(model, opt, panel, samples, batch_size, epoch, seed,
                    model_name, price_mat=None, train_arrays=None,
                    device='cpu'):
    device = torch.device(device)
    model.train()
    gen = torch.Generator(device=device).manual_seed(seed + epoch)
    rng = np.random.default_rng(seed + epoch)
    total, hub, rnk, nb = 0.0, 0.0, 0.0, 0
    if model_name in CROSS_SECTIONAL:
        # whole-date cross-section batches, chronological
        for date in samples:  # samples = date list here
            X, y, tickers, sectors = panel.date_batch(date)
            if len(X) < 8:
                continue
            Xt = torch.from_numpy(X).to(device)
            yt = torch.from_numpy(y).to(device)
            if model_name == 'gnn':
                rets = price_mat.trailing_returns(tickers, date, 63)
                ei = build_graph(np.asarray(sectors), rets, top_k=5).to(device)
                scores = model(Xt, ei)
            else:
                scores = model(Xt)
            loss, comps = combined_loss(scores, yt, generator=gen)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += loss.item()
            hub += comps['huber'].item()
            rnk += comps['rank'].item()
            nb += 1
    else:
        order = torch.randperm(len(samples), generator=gen)
        for b in range(0, len(order), batch_size):
            idx = order[b:b + batch_size]
            if len(idx) < 8:
                continue
            # Stream windows from the panel instead of fancy-indexing a
            # materialized 867 MB array (OOM insurance; identical values).
            xb, yb = batch_windows(panel, samples, idx)
            xb, yb = xb.to(device), yb.to(device)
            scores = model(xb)
            loss, comps = combined_loss(scores, yb, generator=gen)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += loss.item()
            hub += comps['huber'].item()
            rnk += comps['rank'].item()
            nb += 1
            if nb % 50 == 0:
                rss, avail = _mem_gb()
                log.info('batch %d/%d rss=%.2fGB mem_avail=%.2fGB',
                         nb, (len(order) + batch_size - 1) // batch_size,
                         rss, avail)
    if nb == 0:
        return float('nan'), float('nan'), float('nan')
    return total / nb, hub / nb, rnk / nb


@torch.no_grad()
def evaluate(model, panel, val_dates, model_name, price_mat=None,
             device='cpu'):
    device = torch.device(device)
    model.eval()
    per_date = []
    for date in val_dates:
        X, y, tickers, sectors = panel.date_batch(date)
        if len(X) == 0:
            continue
        Xt = torch.from_numpy(X).to(device)
        if model_name == 'gnn':
            rets = price_mat.trailing_returns(tickers, date, 63)
            ei = build_graph(np.asarray(sectors), rets, top_k=5).to(device)
            scores = model(Xt, ei).cpu().numpy()
        else:
            scores = model(Xt).cpu().numpy()
        per_date.append((scores, y))
    return rank_ic(per_date), len(per_date)


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument('--model', required=True,
                   choices=['miss', 'lstm', 'stockmixer', 'gnn'])
    p.add_argument('--regime', required=True, choices=list(REGIMES))
    p.add_argument('--year', required=True, type=int)
    p.add_argument('--data-dir', default='data')
    p.add_argument('--checkpoint-dir', default='checkpoints')
    p.add_argument('--epochs', type=int, default=10)
    p.add_argument('--batch-size', type=int, default=512)
    p.add_argument('--stride', type=int, default=21)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--patience', type=int, default=2)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--threads', type=int, default=2)
    p.add_argument('--device', default='auto',
                   help="torch device: 'auto' (cuda if available, else cpu), "
                        "'cpu', or 'cuda'. CPU numerics are unchanged.")
    args = p.parse_args(argv)

    torch.set_num_threads(args.threads)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = args.device
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    log.info('device=%s (cuda_available=%s)', device,
             torch.cuda.is_available())
    t0 = time.time()

    reg = REGIMES[args.regime]
    Y = args.year
    data_dir = args.data_dir
    log.info('loading calendar + panel %s ...', reg['panel'])
    cal = TradingCalendar(os.path.join(data_dir, 'prices.parquet'))
    panel = PanelStore(os.path.join(data_dir, reg['panel']), reg['features'],
                       'target', cal)
    price_mat = PriceMatrix(os.path.join(data_dir, 'prices.parquet'), cal) \
        if args.model == 'gnn' else None
    log.info('panel: %d tickers, F=%d', len(panel.tickers), panel.F)

    train_lo, train_hi = '2010-01-01', f'{Y - 2}-12-31'
    train_dates_all = panel.panel_dates_in_range(train_lo, train_hi)
    val_dates_all = panel.panel_dates_in_range(f'{Y - 1}-01-01', f'{Y - 1}-12-31')
    # CPU-budget simplification: checkpoint-selection RankIC is computed on
    # every 2nd val date (~126 dates, still a robust RankIC estimate).
    # Test-year scoring (run_score.py) remains daily.
    val_dates = val_dates_all[::2]
    if not train_dates_all or not val_dates:
        raise SystemExit(f'no train/val dates for year {Y}')
    train_dates = train_dates_all[::args.stride]
    log.info('train %s..%s: %d dates -> %d stride-%d; val %d: %d dates',
             train_lo, train_hi, len(train_dates_all), len(train_dates),
             args.stride, Y - 1, len(val_dates))

    log.info('building sample index ...')
    train_arrays = None
    if args.model in CROSS_SECTIONAL:
        samples = train_dates  # per-date batches
        log.info('cross-sectional mode: %d date batches', len(samples))
    else:
        samples = panel.sample_index(train_dates)
        log.info('%d train samples', len(samples))
        # Windows are streamed per batch from the panel (see batch_windows);
        # the old materialized 867 MB array is skipped as OOM insurance.
    if not samples:
        raise SystemExit('no training samples')

    model = build_model(args.model, panel.F)
    model.to(device)
    n_params = count_parameters(model)
    log.info('model=%s regime=%s year=%d params=%d (~%.3fM)',
             args.model, args.regime, Y, n_params, n_params / 1e6)
    assert 50_000 <= n_params <= 500_000, f'param count {n_params} out of range'
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    ckpt_path = os.path.join(args.checkpoint_dir,
                             f'{args.model}_{args.regime}_{Y}.pt')
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    # Crash-safety (2026-09-23): a previous attempt may have died mid-training
    # after saving an early "best" checkpoint. Training always starts from
    # scratch, so a leftover checkpoint can only be partial -- remove it so
    # ckpt_ok() can never mistake it for a completed training run.
    if os.path.exists(ckpt_path):
        log.info('removing stale checkpoint from a prior attempt: %s',
                 ckpt_path)
        os.remove(ckpt_path)
    best_ic, best_epoch, wait = float('-inf'), -1, 0
    for epoch in range(args.epochs):
        ep0 = time.time()
        tr, hub, rnk = train_one_epoch(model, opt, panel, samples,
                                       args.batch_size, epoch, args.seed,
                                       args.model, price_mat, train_arrays,
                                       device)
        val_ic, n_v = evaluate(model, panel, val_dates, args.model, price_mat,
                               device)
        log.info('epoch %d/%d loss=%.4f (huber=%.4f rank=%.4f) '
                 'val_RankIC=%.4f (%d dates) %.1fs',
                 epoch + 1, args.epochs, tr, hub, rnk, val_ic, n_v,
                 time.time() - ep0)
        if np.isfinite(val_ic) and val_ic > best_ic:
            best_ic, best_epoch, wait = val_ic, epoch, 0
            torch.save({'state_dict': model.state_dict(),
                        'meta': {'model': args.model, 'regime': args.regime,
                                 'year': Y, 'val_rankic': best_ic,
                                 'epoch': epoch + 1,
                                 'feature_cols': reg['features'],
                                 'n_params': n_params, 'stride': args.stride,
                                 'saved_at': time.strftime('%Y-%m-%d %H:%M:%S')}},
                       ckpt_path)
            log.info('new best -> %s', ckpt_path)
        else:
            wait += 1
            if wait >= args.patience:
                log.info('early stop after %d epochs w/o improvement', wait)
                break
    wall = time.time() - t0
    log.info('DONE %s %s %d best_val_RankIC=%.4f epoch=%d wall=%.1fs -> %s',
             args.model, args.regime, Y, best_ic, best_epoch + 1, wall,
             ckpt_path)
    # Stamp completion: ckpt_ok() in run_all.py requires this flag, so a
    # checkpoint from a run that died mid-training is never treated as done.
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        ckpt['meta']['training_completed'] = True
        torch.save(ckpt, ckpt_path)
        log.info('stamped training_completed on %s', ckpt_path)
    print(json.dumps({'model': args.model, 'regime': args.regime, 'year': Y,
                      'best_val_rankic': best_ic, 'best_epoch': best_epoch + 1,
                      'wall_s': wall, 'checkpoint': ckpt_path}))


if __name__ == '__main__':
    main()
