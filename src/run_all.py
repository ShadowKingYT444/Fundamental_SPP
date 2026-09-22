"""Driver: run all 60 (model, regime, year) train+score jobs sequentially.

Resumable: skips configs whose checkpoint exists and loads cleanly, and
skips scoring when the score CSV already exists. Safe to relaunch after an
interruption -- it continues where it left off.

Usage:
    setsid nohup python -u src/run_all.py > logs/run_all.log 2>&1 < /dev/null &
"""

import gc
import json
import logging
import os
import sys
import time
import traceback

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_train  # noqa: E402
import run_score  # noqa: E402

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')
log = logging.getLogger('run_all')

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(BASE)

MODELS = ['miss', 'lstm', 'stockmixer', 'gnn']
REGIMES = ['fund63', 'tech63', 'tech5']
YEARS = [2021, 2022, 2023, 2024, 2025]
STATUS_PATH = 'results/run_status.json'


def ckpt_ok(model, regime, year):
    p = f'checkpoints/{model}_{regime}_{year}.pt'
    if not os.path.exists(p):
        return False
    try:
        ckpt = torch.load(p, map_location='cpu', weights_only=False)
        m = ckpt.get('meta', {})
        return (m.get('model') == model and m.get('regime') == regime
                and m.get('year') == year
                and abs(m.get('val_rankic', float('nan'))) >= 0)
    except Exception:
        return False


def score_ok(model, regime, year):
    p = f'scores/{model}_{regime}_{year}.csv'
    return os.path.exists(p) and os.path.getsize(p) > 1000


def save_status(status):
    tmp = STATUS_PATH + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(status, f, indent=1)
    os.replace(tmp, STATUS_PATH)


def main():
    if os.path.exists(STATUS_PATH):
        with open(STATUS_PATH) as f:
            status = json.load(f)
    else:
        status = {}
    total = len(MODELS) * len(REGIMES) * len(YEARS)
    done = 0
    t_all = time.time()
    for model in MODELS:
        for regime in REGIMES:
            for year in YEARS:
                key = f'{model}_{regime}_{year}'
                st = status.get(key, {})
                if ckpt_ok(model, regime, year) and score_ok(model, regime, year):
                    log.info('[%s] already done, skipping', key)
                    done += 1
                    continue
                log.info('[%s] starting (%d/%d)', key, done + 1, total)
                t0 = time.time()
                try:
                    if not ckpt_ok(model, regime, year):
                        run_train.main(['--model', model, '--regime', regime,
                                        '--year', str(year)])
                    else:
                        log.info('[%s] checkpoint exists, skipping train', key)
                    if not score_ok(model, regime, year):
                        run_score.main(['--model', model, '--regime', regime,
                                        '--year', str(year)])
                    else:
                        log.info('[%s] scores exist, skipping score', key)
                    st = {'status': 'ok', 'wall_s': time.time() - t0,
                          'finished': time.strftime('%Y-%m-%d %H:%M:%S')}
                except Exception as e:  # noqa: BLE001 - keep the sweep alive
                    log.error('[%s] FAILED: %s\n%s', key, e,
                              traceback.format_exc())
                    st = {'status': 'failed', 'error': str(e)[:500],
                          'wall_s': time.time() - t0}
                status[key] = st
                save_status(status)
                done += 1
                gc.collect()
                log.info('[%s] %s in %.0fs (elapsed total %.1fh)',
                         key, st['status'], st['wall_s'],
                         (time.time() - t_all) / 3600)
    log.info('ALL DONE: %d/%d configs, total %.1fh', done, total,
             (time.time() - t_all) / 3600)


if __name__ == '__main__':
    main()
