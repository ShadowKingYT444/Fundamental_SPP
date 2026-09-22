"""Synthetic data generator for validation and benchmarking.

Produces flat-format parquet panels (see train.py PANEL CONTRACT) with a planted
linear signal: the 63d/5d forward return loads on a fixed linear combination of
the label-date feature vector plus noise, so a sufficiently expressive model can
overfit / learn it (RankIC > 0 achievable).

Also provides generate_tensors() for quick in-memory overfit tests.
"""

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

N_SECTORS = 11


def _ar1(rng, shape, phi=0.9):
    """shape (n_stocks, T, F) AR(1) with unit stationary variance."""
    n, T, F = shape
    out = np.empty(shape, dtype=np.float64)
    out[:, 0, :] = rng.normal(0, 1, (n, F))
    scale = np.sqrt(max(1e-12, 1 - phi ** 2))
    for t in range(1, T):
        out[:, t, :] = phi * out[:, t - 1, :] + scale * rng.normal(0, 1, (n, F))
    return out


def generate_panel(path, n_stocks=200, dpy=63, first_year=2010, last_year=2025,
                   L=252, F=24, seed=42, signal_strength=1.0):
    """Write a synthetic flat panel parquet to ``path``. Returns metadata dict.

    Label dates: dpy business-ish dates per year from first_year..last_year.
    """
    rng = np.random.default_rng(seed)
    n_years = last_year - first_year + 1
    # dpy evenly-spaced business days per calendar year (walk-forward needs
    # real year structure: train <= Y-2, val = Y-1)
    _labels = []
    for y in range(first_year, last_year + 1):
        bd = pd.bdate_range(f'{y}-01-01', f'{y}-12-31', freq='B')
        step = max(1, len(bd) // dpy)
        _labels.extend(bd[::step][:dpy])
    label_dates = pd.DatetimeIndex(_labels)
    n_labels = len(label_dates)
    T = L + n_labels + 63
    date_strs = label_dates.strftime('%Y-%m-%d').to_numpy()

    tickers = np.array([f'STK{i:04d}' for i in range(n_stocks)])
    sectors = rng.integers(0, N_SECTORS, n_stocks)
    w = rng.normal(0, 1, F)
    w /= np.linalg.norm(w)

    feats = _ar1(rng, (n_stocks, T, F))                       # (stock, time, feat)
    # planted signal on the label-date feature vector
    sig = np.einsum('ktf,f->kt', feats[:, L:L + n_labels, :], w)  # (stock, label)
    r63 = signal_strength * 0.03 * sig + rng.normal(0, 0.02, sig.shape)
    r5 = signal_strength * 0.015 * sig + rng.normal(0, 0.01, sig.shape)
    # sector-neutralize targets
    t63 = r63.copy()
    t5 = r5.copy()
    for s in range(N_SECTORS):
        m = sectors == s
        if m.any():
            t63[m] -= r63[m].mean(axis=0, keepdims=True)
            t5[m] -= r5[m].mean(axis=0, keepdims=True)
    # closes: plain random walk (only used for GNN correlation graph history)
    logc = np.cumsum(rng.normal(0, 0.01, (n_stocks, T)), axis=1)
    closes = 100.0 * np.exp(logc)

    feat_cols = [f'feat_{i}' for i in range(L * F)]
    cols = {
        'ticker': [], 'date': [], 'sector': [], 'close': [],
        'target_63d': [], 'target_5d': [],
    }
    feat_blocks = []
    for k in range(n_labels):
        win = feats[:, k:L + k, :].reshape(n_stocks, L * F)   # t*F+f, t=0 oldest
        feat_blocks.append(win.astype(np.float32))
        cols['ticker'].append(tickers)
        cols['date'].append(np.full(n_stocks, date_strs[k]))
        cols['sector'].append(sectors.astype(str))
        cols['close'].append(closes[:, L + k])
        cols['target_63d'].append(t63[:, k])
        cols['target_5d'].append(t5[:, k])
    table_dict = {c: np.concatenate(v) for c, v in cols.items()}
    feat_mat = np.concatenate(feat_blocks, axis=0)
    for i, fc in enumerate(feat_cols):
        table_dict[fc] = feat_mat[:, i]
    table = pa.table(table_dict)
    # Row groups of ~5k rows: critical for streaming reads. A single row group
    # forces pyarrow to decode whole 6000-col column chunks (GBs of RAM);
    # row groups let iter_batches stream in bounded memory.
    pq.write_table(table, path, row_group_size=5000)
    return {
        'path': path, 'n_stocks': n_stocks, 'dpy': dpy,
        'first_year': first_year, 'last_year': last_year,
        'L': L, 'F': F, 'rows': n_labels * n_stocks,
    }


def generate_tensors(n=400, L=252, F=24, seed=0, signal_strength=1.0):
    """In-memory (X, y): X (n, L, F), y (n,) with planted linear signal on the
    last timestep's features. Features roughly standardized."""
    rng = np.random.default_rng(seed)
    X = _ar1(rng, (n, L, F), phi=0.9).astype(np.float32)
    w = rng.normal(0, 1, F).astype(np.float32)
    w /= np.linalg.norm(w)
    sig = X[:, -1, :] @ w
    y = (signal_strength * 0.05 * sig + rng.normal(0, 0.02, n)).astype(np.float32)
    return torch_X_y(X, y)


def torch_X_y(X, y):
    import torch
    return torch.from_numpy(X), torch.from_numpy(y)


if __name__ == '__main__':
    import sys
    meta = generate_panel(sys.argv[1] if len(sys.argv) > 1 else '/tmp/synth.parquet',
                          n_stocks=64, dpy=21, first_year=2018, last_year=2021)
    print(meta)
