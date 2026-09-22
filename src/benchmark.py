#!/usr/bin/env python3
"""Runtime benchmark for workstream 3 (run from ~/workspace/fundamental_spp):
    python src/benchmark.py

Builds a synthetic panel at realistic scale (~30k training samples, L=252,
F=24) and times full harness runs (train + daily val scoring) per model via
subprocess calls to src/train.py. Extrapolates to the 60-run grid
(4 models x 3 regimes x 5 years) and prints budget recommendations.
Logs to logs/ws3.log.
"""

import json
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import synthetic as synth  # noqa: E402
import train as T  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = sys.executable  # ~/workspace/venv/bin/python

# (model, epochs_to_time)
PLAN = [('miss', 3), ('lstm', 2), ('stockmixer', 2), ('gnn', 2)]
AVG_EPOCHS_PER_RUN = 5.0  # early stop (patience 2) -> typically 3-7 epochs


def run_harness(model, panel_path, epochs, stride=7, year=2021):
    cmd = [PY, 'src/train.py', '--model', model, '--regime', 'fund63',
           '--year', str(year), '--panel-path', panel_path,
           '--epochs', str(epochs), '--stride', str(stride),
           '--skip-normalize', '--log', 'logs/ws3.log',
           '--checkpoint-dir', '/tmp/ws3_bench_ckpt']
    t0 = time.time()
    proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True,
                          timeout=7200)
    wall = time.time() - t0
    out = proc.stdout.strip().splitlines()
    meta = json.loads(out[-1]) if out else {}
    if proc.returncode != 0:
        print(proc.stderr[-3000:])
        raise RuntimeError(f'train.py failed for {model}')
    return wall, meta


def main():
    T.setup_logging('logs/ws3.log')
    os.makedirs('data/synth', exist_ok=True)
    panel_path = 'data/synth/bench_panel_fund63.parquet'
    if not os.path.exists(panel_path):
        print('generating benchmark panel (~30k train samples, L=252, F=24) ...')
        t0 = time.time()
        # year 2021: train = 2010..2019 = 10y * 63 dpy = 630 dates;
        # stride 7 -> 90 dates x 300 stocks = 27k samples
        meta = synth.generate_panel(panel_path, n_stocks=300, dpy=63,
                                    first_year=2010, last_year=2025,
                                    L=252, F=24, seed=7)
        print(f'panel written in {time.time()-t0:.1f}s: {meta["rows"]} rows')

    results = {}
    for model, epochs in PLAN:
        print(f'\n--- timing {model} ({epochs} epochs) ---', flush=True)
        wall, meta = run_harness(model, panel_path, epochs)
        per_epoch = wall / epochs
        results[model] = {'wall': wall, 'epochs': epochs,
                          'per_epoch_s': per_epoch,
                          'val_rankic': meta.get('best_val_rankic')}
        print(f'{model}: {wall:.1f}s total / {epochs} epochs = '
              f'{per_epoch:.1f}s per epoch; val RankIC={meta.get("best_val_rankic"):.4f}')

    print('\n=== EXTRAPOLATION to 60 runs (4 models x 3 regimes x 5 years) ===')
    print(f'assumption: avg {AVG_EPOCHS_PER_RUN} epochs/run (early stop patience 2)')
    total = 0.0
    for model, _ in PLAN:
        per_epoch = results[model]['per_epoch_s']
        per_run = per_epoch * AVG_EPOCHS_PER_RUN
        sub = per_run * 3 * 5  # 3 regimes x 5 years
        total += sub
        print(f'{model:12s} per-epoch {per_epoch:7.1f}s  per-run ~{per_run/60:6.1f} min  '
              f'15 runs subtotal ~{sub/3600:5.2f} h')
    print(f'TOTAL ~{total/3600:.2f} hours (~{total/3600/24:.2f} days) on this 2-vCPU box')
    print('\nnote: fund63 (F=24) is the most expensive regime; tech63/tech5 (F=15) '
          'run ~10-20% faster. Val scoring (daily, full val year) is included above.')
    T.log.info('benchmark results: %s', json.dumps(results))


if __name__ == '__main__':
    main()
