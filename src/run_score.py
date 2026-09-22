"""Score a test year daily with a trained checkpoint -> score CSV.

Usage:
    python src/run_score.py --model miss --regime fund63 --year 2021

Writes scores/<model>_<regime>_<year>.csv per SCORE_CONTRACT.md:
columns ticker, date (YYYY-MM-DD), sector, score.
"""

import argparse
import logging
import os
import sys
import time

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models import build_model  # noqa: E402
from models import build_graph  # noqa: E402
from seqdata import PanelStore, PriceMatrix, TradingCalendar  # noqa: E402
from run_train import REGIMES  # noqa: E402

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')
log = logging.getLogger('run_score')


@torch.no_grad()
def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument('--model', required=True,
                   choices=['miss', 'lstm', 'stockmixer', 'gnn'])
    p.add_argument('--regime', required=True, choices=list(REGIMES))
    p.add_argument('--year', required=True, type=int)
    p.add_argument('--data-dir', default='data')
    p.add_argument('--checkpoint-dir', default='checkpoints')
    p.add_argument('--out-dir', default='scores')
    p.add_argument('--threads', type=int, default=2)
    args = p.parse_args(argv)
    torch.set_num_threads(args.threads)
    t0 = time.time()

    reg = REGIMES[args.regime]
    ckpt_path = os.path.join(args.checkpoint_dir,
                             f'{args.model}_{args.regime}_{args.year}.pt')
    if not os.path.exists(ckpt_path):
        raise SystemExit(f'checkpoint not found: {ckpt_path}')
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    feat_cols = ckpt['meta'].get('feature_cols', reg['features'])
    model = build_model(args.model, len(feat_cols))
    model.load_state_dict(ckpt['state_dict'])
    model.eval()
    log.info('loaded %s (val_RankIC=%.4f epoch=%s)', ckpt_path,
             ckpt['meta'].get('val_rankic', float('nan')),
             ckpt['meta'].get('epoch'))

    cal = TradingCalendar(os.path.join(args.data_dir, 'prices.parquet'))
    panel = PanelStore(os.path.join(args.data_dir, reg['panel']), feat_cols,
                       'target', cal)
    price_mat = PriceMatrix(os.path.join(args.data_dir, 'prices.parquet'), cal) \
        if args.model == 'gnn' else None

    test_dates = panel.panel_dates_in_range(f'{args.year}-01-01',
                                            f'{args.year}-12-31')
    log.info('scoring %d test dates ...', len(test_dates))
    rows = []
    for date in test_dates:
        X, y, tickers, sectors = panel.date_batch(date)
        if len(X) == 0:
            continue
        Xt = torch.from_numpy(X)
        if args.model == 'gnn':
            rets = price_mat.trailing_returns(tickers, date, 63)
            ei = build_graph(np.asarray(sectors), rets, top_k=5)
            scores = model(Xt, ei).cpu().numpy()
        else:
            scores = model(Xt).cpu().numpy()
        for t, s, sc in zip(tickers, sectors, scores):
            rows.append((t, date, s, float(sc)))
    df = pd.DataFrame(rows, columns=['ticker', 'date', 'sector', 'score'])
    os.makedirs(args.out_dir, exist_ok=True)
    out = os.path.join(args.out_dir,
                       f'{args.model}_{args.regime}_{args.year}.csv')
    df.to_csv(out, index=False)
    log.info('wrote %s (%d rows) in %.1fs', out, len(df), time.time() - t0)
    print(out)


if __name__ == '__main__':
    main()
