# WS2 — Fundamental features: methodology notes

## Source
- SEC EDGAR companyfacts JSON (`https://data.sec.gov/api/xbrl/companyfacts/CIK##########.json`),
  fetched per CIK with `User-Agent: FundamentalSPP research contact@example.com`, ~4 req/s.
- JSON processed in memory; never written to disk (disk constraint). Manifest:
  `data/raw/facts_manifest.csv`.
- Forms restricted to **10-Q and 10-K only**.

## Concepts (us-gaap; ONLY these tags + documented fallbacks)
| Feature | Tags |
|---|---|
| Assets | Assets |
| Liabilities | Liabilities |
| Revenues | Revenues → RevenueFromContractWithCustomerExcludingAssessedTax |
| NetIncomeLoss | NetIncomeLoss |
| OperatingIncomeLoss | OperatingIncomeLoss |
| OpCashFlow | NetCashProvidedByUsedInOperatingActivities |
| CapEx paid | PaymentsToAcquirePropertyPlantAndEquipment |
| CurrentAssets | **AssetsCurrent** (taxonomy variant of CurrentAssets) → CurrentAssets |
| CurrentLiabilities | **LiabilitiesCurrent** (taxonomy variant) → CurrentLiabilities |
| Shares out | EntityCommonStockSharesOutstanding → CommonStockSharesOutstanding |

Rationale for fallbacks: SPEC says "Extract ONLY the us-gaap concepts in SPEC". The
taxonomy-variant tags (AssetsCurrent/LiabilitiesCurrent) are the same economic concepts
under alternate tag names. Note: an earlier build also used `SalesRevenueNet` as a
pre-2018 revenue fallback; this was REMOVED in the final rebuild for strict SPEC
compliance (WS2 final report, 2026-09-22). Documented here as a deviation that was
reverted.

## PIT semantics
- Per (concept, fiscal period end): **earliest-filed observation wins** (original
  release; later restatements ignored → no lookahead from restated numbers).
- Flows: within a filing, the **shortest-duration (latest-start) fact** is the discrete
  quarter; annual facts with dur ≥ 300d are split out. **Q4 = annual − (Q1+Q2+Q3)**
  when all three quarters present.
- TTM = rolling 4-quarter sum; YoY = TTM / TTM(4q ago) − 1.
- Feature visibility date = filing date of its inputs; daily value at date t =
  latest observation with **filed ≤ t** (date-granular; companyfacts gives filing
  dates, not intraday acceptance timestamps).
- Gate: **≥ 4 quarters of NetIncome history** before ANY fundamental row is valid.

## CIK quirks
- XOM: universe CIK 0002115436 has only recent facts; predecessor CIK 0000034088
  merged (229 NI entries) → both fetched, entries merged.
- CBS: universe CIK is NaN; true CIK 0000813828 fetched separately.
- LLTC, RTN: universe leaves CIK NaN (mapped to acquirer in earlier draft); excluded —
  no fundamentals rather than wrong ones.
- ABK: NaN CIK in todo → expected 404 (removed 2008, pre-XBRL).

## Daily panel
- Business-day grid 2008-01-01..2026-04-30; quarterly PIT features forward-filled.
- `data/features_fund_raw.parquet`: ticker, date (YYYY-MM-DD str), sector,
  9 features + fcf_ttm/shares_v/ni_ttm intermediates.

## Normalization & targets (`src/normalize.py`)
- Rows restricted to historical S&P 500 membership spells BEFORE any statistics.
- MarketCap = adj_close × PIT shares; earn_yield, fcf_yield computed after price join.
- Winsorize [1%, 99%] on 2010-01-01..2019-12-31 → `data/winsor_thresholds.parquet`
  (columns: set, feature, p01, p99).
- Cross-sectional z-score per date (universe stocks that day).
- Targets: fwd returns over +63/+5 trading days, minus same-date GICS-sector mean
  (sector means over in-universe stocks only).
- Panels: `panel_fund63`, `panel_tech63`, `panel_tech5`
  (ticker, date str, sector, normalized features, target); rows require all features
  + target valid.
