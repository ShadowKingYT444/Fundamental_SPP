## 2026-09-23 ~09:10 UTC -- monitor killed the long driver; hardened
- The 09:02 monitor tick killed the freshly launched long driver
  within ~3 min (no live-lock check existed in the cron instructions).
  State stayed clean: no checkpoint, no partial scores (by design).
- Cron `fundamental-spp-sweep-monitor` updated via cron.update:
  step 1 now checks `results/.run_next.lock` for a live driver PID
  and ends the tick quietly if one holds it; never kills a
  lock-holding process.
- `run_next.py`: new LONG_CONFIGS guard -- `miss_tech63_2024` /
  `miss_tech63_2025` are skipped unless `--allow-long` is passed.
  A 3600 s monitor tick cannot finish their ~2-5.5 h training, and
  run_train.py deletes partial checkpoints, so starting one in a
  tick is pure wasted CPU. Dedicated long driver handles them.
- Long driver relaunched via nohup (`logs/long_driver_20260923.log`),
  `--time-budget 20000 --allow-long`; retraining `miss_tech63_2024`
  (batch 256 + chunk-64 backward). Repo remote main = c5c10d0.

## 2026-09-23 ~08:55 UTC -- incident + hardening
- `miss_tech63_2024` at batch 256 was SIGKILLed a 5th time, ~20 min into
  epoch 2 (epoch 1 had completed and saved a checkpoint). Peak suspect:
  `_scan_backward`'s `(B, L, d_inner, d_state)` state tensor
  (128*252*192*16 float32 = 396 MB per chunk) on top of the 867 MB
  materialized windows + panel on the 7.7 GiB box.
- The 08:53 hourly-monitor tick then launched a SECOND driver which,
  because `ckpt_ok()` accepted the epoch-1 partial checkpoint, skipped
  retraining and started SCORING a half-trained model. Killed it
  (driver + scorer) before any score CSV was written -- no junk data.
- Fixes committed: (1) selective-scan backward chunk 128 -> 64
  (~198 MB peak; bit-identical numerics, pure impl detail);
  (2) `run_train.py` deletes any stale checkpoint at start and stamps
  `training_completed=True` in meta at DONE; (3) `ckpt_ok()` requires
  the stamp, so partial checkpoints always trigger retrain;
  (4) `run_next.py` takes an exclusive lockfile -- monitor ticks and
  foreground runs can no longer overlap and double memory.
- Stamped `training_completed` on the 8 already-finished configs;
  quarantined the epoch-1 partial `miss_tech63_2024` checkpoint to
  `checkpoints/quarantine/`.
- Monitor cron stays enabled: with the lock + resume fixes it is now a
  safe unattended driver.

## 2026-09-23 ~08:20 UTC
- Reviewed chenliu-1996/figures4papers; adopted its house style for the
  report visuals: sans-serif typography, minimal spines, frameless legends,
  semantic palette (blue #0F4D92 = key method MISS/fund63, red = contrasts,
  neutrals = baselines), black-edged bars, dpi=300 PNG + vector PDF export.
- Added `src/plot_style.py` (style preset) and `src/make_figures.py`
  (Fig 1 equity curves, Fig 2 architecture x regime Sharpe bars,
   Fig 3 cost sensitivity, Fig 4 bootstrap of Sharpe difference).
  All four templates smoke-tested with synthetic data (renders verified).
- `src/evaluate.py` now also writes daily net-return series to
  `results/equity/<model>_<regime>.csv` so Figure 1 can be built at the end.

# Full sweep — status (updated 2026-09-22 22:50 UTC)

## Strategy (current)
Foreground, resumable: each tick runs pending configs via
`python -u src/run_next.py --time-budget 3000`. Each config trains/scores as an
isolated subprocess; progress recorded in `results/run_status.json`.
Checkpoints (`checkpoints/*.pt`) and score CSVs (`scores/*.csv`) are the resume
markers — a killed tick simply resumes the current config next time.
The old detached `run_all.py` driver is retired (its session processes were
silently terminated).

Order: miss → lstm → stockmixer → gnn; regimes fund63 → tech63 → tech5; years 2021–2025.

## Progress: 6 / 60 configs ok
- miss_fund63_2021 … miss_fund63_2025: ok (MISS fundamental-63d regime complete)
- miss_tech63_2021: ok
- miss_tech63_2022: FAILED once (SIGKILL, exit -9 — likely OOM; box has ~7 GiB
  RAM and training + scoring can peak it). Retry re-trained fine (checkpoint
  written), but the scorer was killed when the supervising tick died before
  finishing. Next tick retries from the checkpoint.

## Measured per-iteration cost (batch 512, F=15, 2 vCPU, no GPU)
- MISS: 19.39 s/iter (scripted sequential selective scan + manual adjoint backward,
  backward chunked over batch dim in 128s to bound the (B,L,192,16) state tensor)
- LSTM: 4.51 s/iter
- StockMixer: 1.32 s/iter
- GNN: 3.07 s/iter

## Simplifications in force (all documented for REPORT.md)
- Batch size 512 (vs paper-unspecified; smoke-tested at 256).
- Train dates sampled every 21 trading days (not all dates).
- Validation scored every 2nd date (127 dates/yr); test scoring daily.
- Winsor thresholds fixed from 2010–2019 (not refit per fold).
- Static best-effort sectors; trade events = entries + exits.
- Max 10 epochs, early stopping patience 2, checkpoint = best val RankIC.
- CPU-only, seed 0, Adam 1e-3, Huber + 0.25 × pairwise rank loss.
- OOM fallback (2026-09-23): batch 256 for miss_tech63_2024/2025 only, after
  miss_tech63_2024 was SIGKILLed 4× at batch 512 on the 7.7 GiB box.
  (miss_tech63_2022/2023 each died 1–2× at 512 then recovered.)

## Validation completed
- Score → backtest → evaluate chain: OK (lstm_tech5_2021 1-epoch smoke).
- MISS scan: gradcheck passed; assoc-scan rewrite attempted then reverted
  (14× slower on this box — low memory bandwidth favors sequential scan).
- evaluate.py table writer fixed for missing configs.
