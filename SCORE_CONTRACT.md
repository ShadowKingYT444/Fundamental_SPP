# Score-file contract (workstream 4: backtest & evaluation)

Workstream 3 (training) must emit one CSV per (model, regime, test year).
Workstream 4 consumes them with `python src/evaluate.py --scores_dir <dir> --out results/`.

## Filename

`<model>_<regime>_<year>.csv`, e.g. `miss_fund63_2021.csv`

- `model`: `miss` | `lstm` | `stockmixer` | `gnn` (lowercase; may itself contain
  underscores — parsing takes `<regime>` as the second-to-last `_`-separated token)
- `regime`: `fund63` | `tech63` | `tech5`
  - `fund63`, `tech63` → portfolio rebalances on the **first trading day of each month**
  - `tech5` → portfolio rebalances **every Monday** (trading day)
- `year`: one test year per file: `2021` … `2025`

## Columns

| column | type | meaning |
|---|---|---|
| `ticker` | str | stock ticker, must match `prices.parquet` tickers |
| `date` | str `YYYY-MM-DD` | scoring date; **daily** coverage of the whole test year (val/test are scored daily per SPEC) |
| `sector` | str | GICS sector name (11 standard sectors; used for within-sector ranking and equal 1/11 sector gross weights) |
| `score` | float | model output for (ticker, date); higher = more attractive. Rows with NaN score/sector are dropped (stock uninvestable that day). |

Notes:
- Exactly one row per (ticker, date); duplicates → last wins.
- Scores must be point-in-time (no lookahead): the score for date *t* may only use
  information available at *t*.
- The score's horizon should match the regime: `fund63`/`tech63` scores predict the
  63-day sector-neutral forward return; `tech5` scores predict the 5-day one.

## Prices (not in the score file)

Daily returns are computed from `data/prices.parquet` (columns `ticker, date,
adj_close`; dates as `YYYY-MM-DD` strings), overridable via
`--prices <path>` or the `prices` argument of `run_backtest()`.

## What the backtest does with the file (summary)

At each rebalance date, within each sector, stocks are ranked by `score`
(percentile, 1.0 = best): long top quintile (≥ 0.80), short bottom quintile
(≤ 0.20), with hysteresis exit bands (long kept unless rank < 0.60, short kept
unless rank > 0.40). Each sector gets 1/11 of gross; long side and short side of
a sector each get NAV/22 equal-weighted per name; scaled to exact dollar
neutrality; 2% of NAV single-name cap. Trades execute at the close; costs are
`costs_bps` one-way on traded notional (default 15). See `src/backtest.py`
docstring for full rules and resolved ambiguities.
