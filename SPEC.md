# Fundamental_SPP Reproduction — Shared Spec (v1)

Reproduction of Ding (2026) "Fundamental Information for Low-Turnover Equity ML".
All workstreams MUST follow these conventions. Working dir: `~/workspace/fundamental_spp/`.

## Layout
- `data/` — all datasets (parquet). Raw downloads in `data/raw/` (prune after processing).
- `src/` — shared library code (`src/common.py`, `src/models.py`, `src/train.py`, `src/backtest.py`, ...).
- `checkpoints/` — trained model weights, one file per (model, regime, test_year).
- `results/` — metrics JSON/CSV, figures.
- `logs/` — run logs.
- `SPEC.md` (this file), `REPORT.md` (final verification report, written at the end by coordinator).

## Universe & dates
- Universe: S&P 500 constituents. Build a membership table `universe.parquet` with columns
  `ticker, cik, sector, start_date, end_date` (dates as YYYY-MM-DD strings; end_date null = still in).
  Source: Wikipedia current + former constituent lists ("date added"); sector = GICS sector.
  CIK from SEC ticker→CIK mapping file (https://www.sec.gov/files/company_tickers.json).
  Handle obvious ticker changes (e.g. FB→META) via a manual map; document gaps.
- OHLCV: Yahoo Finance daily, adjusted close + volume, from **2008-01-01 to 2026-04-30**
  (labels need +63 trading days after 2025-12-31). `prices.parquet`: `ticker, date, open, high, low, close, adj_close, volume`.
- Trading calendar: union of dates present in prices (NYSE). All joins use string dates `YYYY-MM-DD`.

## Features
Computed per (ticker, date), stored in `features_<regime>.parquet` with columns
`ticker, date, <feature cols...>, target_63d, target_5d, sector, in_universe (bool)`.

Technical features (from adj_close/volume), lookbacks in trading days:
- returns: ret_5, ret_21, ret_63, ret_252
- realized vol: vol_21, vol_63 (std of daily log returns × sqrt)
- MA distance: close vs SMA(21), SMA(63), SMA(200): (close/sma - 1)
- rsi_14, macd (12/26/9: macd line & signal diff), atr_14 (normalized by close)
- volume surprise: volume / SMA(volume, 63) - 1
- price-volume trend: correlation(close, volume, 63d)
Fundamental features (SEC EDGAR companyfacts, 10-Q/10-K, POINT-IN-TIME by filing
acceptance datetime `filed`; feature at date t = latest filing with filed <= t):
- roa = NetIncomeTTM / Assets; op_margin = OperatingIncomeTTM / RevenueTTM
- rev_growth, earn_growth = TTM YoY growth (vs 4 quarters ago)
- earn_yield = NetIncomeTTM / MarketCap; fcf_yield = (OpCashFlowTTM - CapExTTM) / MarketCap
- leverage = TotalLiabilities / Assets (or Debt/Equity fallback); liquidity = CurrentAssets / CurrentLiabilities
- accruals = (NetIncomeTTM - OpCashFlowTTM) / Assets
Companyfacts XBRL tags (us-gaap): Assets; Liabilities; Revenues (fallback RevenueFromContractWithCustomerExcludingAssessedTax); NetIncomeLoss; OperatingIncomeLoss; NetCashProvidedByUsedInOperatingActivities; PaymentsToAcquirePropertyPlantAndEquipment; CurrentAssets; CurrentLiabilities; StockholdersEquity. MarketCap = adj_close × shares outstanding (shares from 10-Q/10-K facts `CommonStockSharesOutstanding` or EntityCommonStockSharesOutstanding; forward-filled point-in-time).
Minimum history: require ≥ 4 quarters of fundamentals before first fundamental feature is valid (else NaN → row dropped from fundamental regime only).

## Normalization & targets
- Winsorize each feature at [1st, 99th] percentiles computed on TRAINING data only; then
  cross-sectional z-score per date (mean/std across universe stocks that day).
- Targets: `target_63d = fwd_ret_63d(ticker) - mean sector fwd_ret_63d` (sector-neutral),
  `target_5d` analogous. fwd_ret_N = adj_close[t+N]/adj_close[t] - 1.
- Regimes: `fund63` (fundamental features → target_63d), `tech63` (technical features → target_63d), `tech5` (technical features → target_5d).

## Models (all ~0.2M params, input seq len L=252 daily feature vectors; predict score for label date)
- MISS: Linear proj → 2× (selective SSM block + residual proj + LayerNorm) → scoring head (Linear→1).
  Selective SSM implemented in pure PyTorch (no CUDA): input-dependent Δ, B, C; parallel associative scan or sequential; state dim ~16, d_model ~64.
- LSTM: 2-layer LSTM (~0.2M) → head.
- StockMixer: MLP mixing across indicators/time/stocks per Fan & Shen (indicator mixer, time mixer, stock mixer on cross-sectional batch) → head.
- GNN: per-stock encoder + 1–2 message-passing layers over graph with sector edges + rolling 63d return-correlation edges (top-k per stock, recomputed periodically) → head.
- Loss: Huber(delta=1.0) + 0.25 × pairwise margin ranking loss (pairs sampled within batch).
- Optimizer: Adam, lr 1e-3, batch 256, ≤10 epochs, early stop on val RankIC.

## Walk-forward
- Test years 2021..2025. For test year Y: train = 2010-01-01..(Y-2)-12-31 expanding; val = (Y-1) full year; test = Y.
  Retrain at each boundary using only data available at that point (features/targets must respect PIT).
- Training samples: (stock, date) with stride 21 trading days (monthly sampling) to fit CPU budget; val/test scored DAILY.
- Checkpoint selection: max validation RankIC (Spearman of score vs target, pooled per date then averaged).
- 4 models × 3 regimes × 5 years = 60 runs. Save best checkpoint per run.

## Portfolio
- Score daily for all universe stocks in test year (NaN features → no score).
- Rebalance: 63d regimes → first trading day of each month; 5d regime → every Monday.
- Within each GICS sector, rank scores: LONG top quintile, SHORT bottom quintile.
- Sizing: each sector gets equal gross weight (1/11 of gross); within sector, equal weight per name;
  scale so total long notional = total short notional (dollar neutral). Position cap 2% of NAV.
- Hysteresis (lower turnover for 63d): keep existing position unless its score rank crosses exit band
  (long exits if rank < 60th pct of sector; short exits if rank > 40th pct).
- Costs: 15 bps one-way on traded notional (also run 0/5/10/25/50 bps sensitivity).
- Metrics per year: annual return (compounded daily net), ann. Sharpe (×sqrt(252), rf=0),
  turnover (annual one-way traded notional / NAV), trade events/yr (count of position open+close events /5yrs... per year: entries+exits), max drawdown.
- Report 5-yr means to compare with paper Tables 1–2.

## Robustness
- Moving-block bootstrap (10k resamples, block len 21d) of Sharpe difference fund63 vs tech5 (MISS).
- Year-level one-sided sign tests.
- Deflated Sharpe Ratio (Bailey & Lopez de Prado) with trials = # configs evaluated.

## Constraints
- CPU only (torch CPU). Small batches, checkpoint each run. Watch disk; prune raw downloads.
- NEVER fabricate metrics. Missing/unverifiable pieces must be reported as such.
- Dates as strings; seeds fixed (42); all randomness via torch/numpy seeded.
