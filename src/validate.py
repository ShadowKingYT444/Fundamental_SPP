#!/usr/bin/env python3
"""Synthetic validation for workstream 3 (run from ~/workspace/fundamental_spp):
    python src/validate.py

Checks:
  1. param counts ~0.2M for all 4 models (F=24 and F=15)
  2. parallel associative scan == sequential reference scan
  3. each model overfits a tiny synthetic set: loss decreases, IC > 0
  4. checkpoint save/load roundtrip preserves outputs
  5. loss sanity: constant targets -> rank term 0; perfect ranking -> rank term 0
Logs to logs/ws3.log.
"""

import os
import sys
import time

import numpy as np
import torch
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models import (build_model, count_parameters, build_graph,  # noqa: E402
                    _assoc_scan, _sequential_scan)
from losses import combined_loss, pairwise_margin_ranking_loss  # noqa: E402
from synthetic import generate_tensors  # noqa: E402
import train as T  # noqa: E402  (logging setup)


def check_param_counts():
    print('=== 1. param counts ===')
    rows = []
    for F in (24, 15):
        for name in ('miss', 'lstm', 'stockmixer', 'gnn'):
            m = build_model(name, F)
            n = count_parameters(m)
            ok = 50_000 <= n <= 500_000
            rows.append((name, F, n, ok))
            print(f'{name:12s} F={F:2d} params={n:>7,} (~{n/1e6:.3f}M) {"OK" if ok else "OUT OF RANGE"}')
            assert ok, f'{name} F={F}: {n} params out of range'
    return rows


def check_scan():
    print('=== 2. parallel scan vs sequential reference ===')
    torch.manual_seed(0)
    for L in (1, 7, 252, 300):
        a = torch.sigmoid(torch.randn(5, L, 48))          # (0,1): stable dynamics
        c = torch.randn(5, L, 48)
        h_par = _assoc_scan(a, c)
        h_seq = _sequential_scan(a, c)
        err = (h_par - h_seq).abs().max().item()
        print(f'L={L:3d} max|diff|={err:.2e}')
        assert err < 1e-4, f'scan mismatch at L={L}'
    print('scan OK')


def check_losses():
    print('=== 3. loss sanity ===')
    torch.manual_seed(0)
    s = torch.randn(64)
    y_const = torch.ones(64)
    r = pairwise_margin_ranking_loss(s, y_const)
    print(f'constant targets -> rank loss = {r.item():.6f}')
    assert r.item() == 0.0
    y = torch.arange(64, dtype=torch.float32)  # pairwise |diff| >= 1 > margin
    r2 = pairwise_margin_ranking_loss(y, y)  # perfect ranking, all diffs > margin -> 0
    print(f'perfect ranking (separated) -> rank loss = {r2.item():.6f}')
    assert r2.item() == 0.0
    tot, comps = combined_loss(s, torch.randn(64))
    print(f'combined loss = {tot.item():.4f} (huber={comps["huber"]:.4f}, rank={comps["rank"]:.4f})')
    assert torch.isfinite(tot)
    print('losses OK')


def overfit_one(name, F=24, n=400, epochs=20, seed=0):
    torch.manual_seed(seed)
    np.random.seed(seed)
    X, y = generate_tensors(n=n, L=252, F=F, seed=seed)
    model = build_model(name, F)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    gen = torch.Generator().manual_seed(seed)
    edge_index = None
    if name == 'gnn':
        sectors = np.random.randint(0, 4, n)
        edge_index = build_graph(sectors, np.random.randn(n, 63))
    bs = n if name == 'gnn' else 128  # GNN: whole graph at once (edge_index aligns)
    losses = []
    model.train()
    for ep in range(epochs):
        perm = torch.randperm(n, generator=gen)
        ep_loss = 0.0
        nb = 0
        for b in range(0, n, bs):
            idx = perm[b:b + bs]
            xb, yb = X[idx], y[idx]
            scores = model(xb, edge_index) if name == 'gnn' else model(xb)
            loss, _ = combined_loss(scores, yb, generator=gen)
            opt.zero_grad()
            loss.backward()
            opt.step()
            ep_loss += loss.item()
            nb += 1
        losses.append(ep_loss / nb)
    model.eval()
    with torch.no_grad():
        s_all = []
        for b in range(0, n, bs):
            xb = X[b:b + bs]
            s_all.append(model(xb, edge_index) if name == 'gnn' else model(xb))
        s_all = torch.cat(s_all).numpy()
    ic = spearmanr(s_all, y.numpy()).statistic
    return losses[0], losses[-1], ic, model


def check_overfit():
    print('=== 4. overfit tiny synthetic set (planted signal) ===')
    # model-specific configs verified to overfit on this box (MISS is slow on
    # CPU; StockMixer needs more epochs for the Huber term)
    cfgs = {
        'miss': dict(n=200, epochs=10),
        'lstm': dict(n=200, epochs=10),
        'stockmixer': dict(n=200, epochs=25),
        'gnn': dict(n=200, epochs=15),
    }
    for name, kw in cfgs.items():
        t0 = time.time()
        l0, l1, ic, _ = overfit_one(name, **kw)
        dt = time.time() - t0
        ok = (l1 < 0.5 * l0) and ic > 0.3
        print(f'{name:12s} loss {l0:.4f} -> {l1:.4f}   train IC={ic:+.3f}   '
              f'{dt:.1f}s  {"OK" if ok else "FAIL"}')
        assert ok, f'{name} failed to overfit (loss {l0:.3f}->{l1:.3f}, IC {ic:.3f})'
    print('overfit OK')


def check_checkpoint():
    print('=== 5. checkpoint save/load roundtrip ===')
    torch.manual_seed(1)
    m1 = build_model('miss', 24)
    m1.eval()
    x = torch.randn(16, 252, 24)
    with torch.no_grad():
        y1 = m1(x)
    ckpt = '/tmp/ws3_ckpt_test.pt'
    torch.save({'state_dict': m1.state_dict(),
                'meta': {'val_rankic': 0.123, 'epoch': 3}}, ckpt)
    m2 = build_model('miss', 24)
    blob = torch.load(ckpt, map_location='cpu', weights_only=False)
    m2.load_state_dict(blob['state_dict'])
    m2.eval()
    with torch.no_grad():
        y2 = m2(x)
    err = (y1 - y2).abs().max().item()
    print(f'max|dy| after reload = {err:.2e}; meta={blob["meta"]}')
    assert err == 0.0
    print('checkpoint OK')


def main():
    T.setup_logging('logs/ws3.log')
    t0 = time.time()
    rows = check_param_counts()
    check_scan()
    check_losses()
    fit = check_overfit()
    check_checkpoint()
    print(f'\nALL VALIDATION CHECKS PASSED in {time.time()-t0:.1f}s')
    T.log.info('validate.py: all checks passed')


if __name__ == '__main__':
    main()
