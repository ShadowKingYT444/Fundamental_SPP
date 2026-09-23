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
import fcntl
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_all import (MODELS, REGIMES, YEARS, STATUS_PATH, ckpt_ok,  # noqa: E402
                     save_status, score_ok)

LOCK_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         '..', 'results', '.run_next.lock')

# OOM mitigation (2026-09-23): miss_tech63_2024 was SIGKILLed 5x on the
# 7.7 GiB box (4x at batch 512, 1x at batch 256). Two fixes: (a) fall back to
# batch 256 for the remaining MISS tech63 configs, halving per-batch
# activations; (b) the selective-scan backward chunk is now 64 (was 128),
# halving the peak (B, L, d_inner, d_state) state tensor to ~198 MB -- a pure
# implementation detail, bit-identical numerics. Both are paper-unspecified
# and will be documented in REPORT.md.
BATCH_OVERRIDES = {'miss_tech63_2024': 256, 'miss_tech63_2025': 256}

# Long configs (2026-09-23): at batch 256 these need ~100-330 min of
# UNINTERRUPTED training (33 min/epoch, patience 2, up to 10 epochs), which
# exceeds the hourly monitor tick's 3600 s timeout. A tick that starts one
# burns ~50 min and is killed mid-training; run_train.py then deletes the
# partial checkpoint, so the work is pure waste. The monitor tick must NOT
# start these -- skip them unless --allow-long is passed. They are handled
# by a dedicated long driver holding the lockfile.
LONG_CONFIGS = {'miss_tech63_2024', 'miss_tech63_2025'}


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
    allow_long = '--allow-long' in args
    # Single-driver lock (2026-09-23): the hourly monitor cron and ad-hoc
    # foreground runs must never overlap -- two drivers each materialize the
    # ~0.9 GB training windows and the OOM killer takes one of them. Whoever
    # holds the lock drives; the other exits quietly.
    lock_path = os.path.normpath(LOCK_PATH)
    os.makedirs(os.path.dirname(lock_path), exist_ok=True)
    lock_fh = open(lock_path, 'w')
    try:
        fcntl.flock(lock_fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        print('run_next: another driver holds the lock; exiting quietly',
              flush=True)
        return
    lock_fh.write(str(os.getpid()))
    lock_fh.flush()
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
        if key in LONG_CONFIGS and not allow_long:
            print(f'[{key}] needs >1h uninterrupted training; skipping '
                  f'(rerun with --allow-long from a long driver)', flush=True)
            break
        print(f'[{key}] starting', flush=True)
        t1 = time.time()
        try:
            if not ckpt_ok(model, regime, year):
                train_cmd = [sys.executable, 'src/run_train.py', '--model', model,
                             '--regime', regime, '--year', str(year)]
                if key in BATCH_OVERRIDES:
                    train_cmd += ['--batch-size', str(BATCH_OVERRIDES[key])]
                    print(f'[{key}] OOM fallback: batch_size={BATCH_OVERRIDES[key]}',
                          flush=True)
                subprocess.run(train_cmd, check=True)
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
