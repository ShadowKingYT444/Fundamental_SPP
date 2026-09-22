"""Run pending (model, regime, year) configs in the FOREGROUND, sequentially.

Each train/score step runs as its own subprocess (memory isolation: the OS
reclaims everything when the child exits). Designed to be driven by the
hourly monitor cron: each tick runs as many configs as fit in the time
budget, then exits. Fully resumable -- a killed run simply retries the
current config next time (checkpoints + score CSVs are the resume markers).

Usage:
    python src/run_next.py [--time-budget SECONDS]   (default 3000)

Prints DONE when all 60 configs are complete. Exit code always 0 unless
invoked incorrectly.
"""

import json
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_all import (MODELS, REGIMES, YEARS, STATUS_PATH, ckpt_ok,  # noqa: E402
                     save_status, score_ok)


def next_pending():
    for model in MODELS:
        for regime in REGIMES:
            for year in YEARS:
                if not (ckpt_ok(model, regime, year)
                        and score_ok(model, regime, year)):
                    return model, regime, year
    return None


def main(argv=None):
    args = argv if argv is not None else sys.argv[1:]
    budget = 3000.0
    if '--time-budget' in args:
        budget = float(args[args.index('--time-budget') + 1])
    if os.path.exists(STATUS_PATH):
        with open(STATUS_PATH) as f:
            status = json.load(f)
    else:
        status = {}
    t0 = time.time()
    ran = 0
    while time.time() - t0 < budget:
        nxt = next_pending()
        if nxt is None:
            print('DONE: all 60 configs complete', flush=True)
            break
        model, regime, year = nxt
        key = f'{model}_{regime}_{year}'
        print(f'[{key}] starting', flush=True)
        t1 = time.time()
        try:
            if not ckpt_ok(model, regime, year):
                subprocess.run(
                    [sys.executable, 'src/run_train.py', '--model', model,
                     '--regime', regime, '--year', str(year)], check=True)
            if not score_ok(model, regime, year):
                subprocess.run(
                    [sys.executable, 'src/run_score.py', '--model', model,
                     '--regime', regime, '--year', str(year)], check=True)
            st = {'status': 'ok', 'wall_s': time.time() - t1,
                  'finished': time.strftime('%Y-%m-%d %H:%M:%S')}
        except subprocess.CalledProcessError as e:
            st = {'status': 'failed', 'error': f'exit={e.returncode}',
                  'wall_s': time.time() - t1}
        status[key] = st
        save_status(status)
        ran += 1
        print(f'[{key}] {st["status"]} in {st["wall_s"]:.0f}s', flush=True)
    print(f'run_next: {ran} configs in {(time.time() - t0) / 60:.1f} min',
          flush=True)


if __name__ == '__main__':
    main()
