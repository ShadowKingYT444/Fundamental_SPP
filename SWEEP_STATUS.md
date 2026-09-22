# Full sweep — launch status (2026-09-22 09:52 UTC)

## Launched
`python src/run_all.py` (detached, logs to `logs/run_all.log`, status in `results/run_status.json`).
Resumable: re-running `run_all.py` skips configs with a valid checkpoint + score CSV.

Order: miss → lstm → stockmixer → gnn; regimes fund63 → tech63 → tech5; years 2021–2025.
First config: miss_fund63_2021 (started 09:52:23 UTC).

## Measured per-iteration cost (batch 512, F=15, 2 vCPU, no GPU)
- MISS: 19.39 s/iter (scripted sequential selective scan + manual adjoint backward,
  backward chunked over batch dim in 128s to bound the (B,L,192,16) state tensor)
- LSTM: 4.51 s/iter
- StockMixer: 1.32 s/iter
- GNN: 3.07 s/iter

## Estimated budget (~50 h total ≈ 2 days)
- MISS fund63: 5 runs × ~45 min ≈ 4 h (first result ~09:52 + 4 h ≈ 14:00 UTC)
- MISS tech63/tech5: 10 runs × ~2.7 h ≈ 27 h
- LSTM all: ~7 h; StockMixer all: ~2.3 h; GNN all: ~4.7 h
- Scoring: 60 × ~5 min ≈ 5 h

## Simplifications in force (all documented for REPORT.md)
- Batch size 512 (vs paper-unspecified; smoke-tested at 256).
- Train dates sampled every 21 trading days (not all dates).
- Validation scored every 2nd date (127 dates/yr); test scoring daily.
- Winsor thresholds fixed from 2010–2019 (not refit per fold).
- Static best-effort sectors; trade events = entries + exits.
- Max 10 epochs, early stopping patience 2, checkpoint = best val RankIC.
- CPU-only, seed 0, Adam 1e-3, Huber + 0.25 × pairwise rank loss.

## Validation completed
- Score → backtest → evaluate chain: OK (lstm_tech5_2021 1-epoch smoke).
- MISS scan: gradcheck passed; assoc-scan rewrite attempted then reverted
  (14× slower on this box — low memory bandwidth favors sequential scan).
- evaluate.py table writer fixed for missing configs.

## Smoke artifacts removed before launch
checkpoints/lstm_tech5_2021.pt, checkpoints/miss_fund63_2021.pt,
scores/lstm_tech5_2021.csv, results/run_status.json (all 1-epoch).
