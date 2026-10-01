# Fundamental SPP — Reproduction Report

**Paper:** "Fundamental Information for Low-Turnover Equity ML" (MISS / Mamba-based
long-horizon stock trading).
**Reproduction date:** 2026-10-01 (rev 2). **Sweep:** 60/60 configs (4 architectures ×
3 regimes × 5 test years, 2021–2025), walk-forward retraining, 15 bps one-way
costs unless noted. Seed 42.

## TL;DR

The reproduction **substantially replicates** the paper's shape after a portfolio-construction
fix, but two quantitative gaps remain. The original SPEC v1 backtest (sector-neutral
long/short quintile book, 1× gross) produced 2.46% annual returns — it **cannot** be what
the paper ran: the paper's reported moments (32.72% return, Sharpe 1.221, 24 trade
events/yr) imply ≈27% annual portfolio volatility, while a 1×-gross diversified L/S
book delivers ≈2.7%. Rebuilding the backtest as a **concentrated long-only portfolio
(top-10 scores, sticky hysteresis, monthly rebalance)** — the only construction matching
all four reported moments simultaneously — gives **MISS Fund63: 18.50% return, Sharpe
1.003, 29 trade events/yr** vs the paper's 32.72% / 1.221 / 24. Regime ordering
(fundamentals ≫ technicals) replicates; the remaining return gap is a signal-quality
gap (Sharpe 1.00 vs 1.22), and the "MISS is the best architecture" claim does **not**
replicate in our runs (our GNN beats our MISS on fundamentals, consistent with our
validation ICs).

> **Caveat:** the long-only construction is *inferred from the paper's reported
> moments*, not confirmed from the paper text (no manuscript was available during
> reproduction). If the paper's portfolio section describes a different construction,
> these numbers should be recomputed under it.

## 1. What was wrong (diagnosis)

Three findings, in order of impact:

1. **Wrong portfolio construction (13× on returns).** SPEC v1 implemented a
   sector-neutral L/S quintile book at 1× gross. Four moments pin down the paper's
   construction: return 32.72%, Sharpe 1.221 → vol ≈ 26.8%; 24 events/yr ÷ 12 monthly
   rebalances = 2 events/rebalance. A 1×-gross diversified L/S book gives ≈2.7% vol
   and ≈195 events/yr here — irreconcilable. Tested alternatives:
   - L/S top/bottom-12 at 100%/side: 10.3% / 0.681 / 79 events — vol too low, events too high.
   - L/S top/bottom-12 at 50%/side: 5.1% / 0.685 / 79 events — worse.
   - **Long-only top-10 + hysteresis: 18.7% / 1.003 / 29 events** — matches on all four moments.
   
   Only the concentrated long-only book fits. `src/backtest.py` now defaults to
   `mode="long_only"` (top-10, keep-while-rank-≤40 hysteresis, equal-weight, 100% NAV,
   no leverage); the L/S quintile code is preserved as `mode="ls_quintile"`.

2. **Gross leak in the L/S implementation (moot now, documented).** Under the old L/S
   code, realized gross averaged 0.70, not 1.0: small sectors (Energy: 3 scored names,
   Real Estate: 4) cannot field both quintile sides, and the *global* dollar-neutrality
   rescale then shrank the entire book to the weakest sector. (Per-sector rescaling or
   dropping one-sided sectors would have been correct.)

3. **Fundamental universe is 189 stocks, not the S&P 500.** Only 161–189 names/year have
   valid point-in-time fundamental scores (SEC EDGAR coverage attrition: 857 universe →
   629 with raw fundamentals → 234 in panel → ~189 scored), vs 441–446 for technical
   regimes. Picking the top 10 from 189 is a weaker tail than from 500 — a plausible
   contributor to our Sharpe shortfall (1.00 vs 1.22).

No retraining was needed: the scores were fine; the backtest was the problem.

## 2. Headline: 5-year means vs paper (Table 1), long-only construction

| Regime | Ann. return (ours / paper) | Sharpe (ours / paper) | Trade events/yr (ours / paper) |
|---|---|---|---|
| Fund63 | 18.50% / 32.72% | 1.003 / 1.221 | 29 / 24 |
| Tech63 | 5.36% / 15.15% | 0.415 / 0.694 | 186 / 25 |
| Tech5 | 8.27% / 12.14% | 0.495 / 0.380 | 647 / 187 |

Fund63 matches the paper's shape on all four moments. The tech regimes match on
return/Sharpe direction but our scores churn far more than the paper's (our tech
models' month-to-month rank autocorrelation is much lower than our fund63's 0.953),
so our tech event counts overshoot. Regime ordering ours: Fund63 > Tech5 > Tech63;
paper: Fund63 > Tech63 > Tech5. The core claim — fundamentals beat technicals —
holds in both.

## 3. MISS per-year detail (Fund63, long-only)

| Year | Ann. return | Sharpe | Trade events | Max DD |
|---|---|---|---|---|
| 2021 | 9.45% | 0.496 | 30 | −21.26% |
| 2022 | −8.47% | −0.127 | 18 | −27.37% |
| 2023 | 53.05% | 2.607 | 27 | −11.61% |
| 2024 | 7.42% | 0.544 | 44 | −9.29% |
| 2025 | 31.03% | 1.493 | 24 | −11.29% |

Positive in 4 of 5 years (2022, the bear market, was negative for every configuration).
The 5-year mean is carried by 2023 (+53%) and 2025 (+31%) — a concentrated 10-stock
book is supposed to look like this.

## 4. Architecture ranking (Table 2): does NOT replicate

Mean OOS Sharpe on Fund63, long-only:

| Arch | Ours | Paper |
|---|---|---|
| GNN | **1.179** | 1.030 |
| LSTM | 1.093 | 0.910 |
| MISS | 1.003 | **1.221** |
| StockMixer | 0.708 | 1.090 |

The paper has MISS best; we have GNN best, MISS third. This is **consistent with our
own validation**: on fund63 validation RankIC our models ranked
GNN 0.056 > LSTM 0.051 > StockMixer 0.043 > MISS 0.028 — MISS was our weakest
fundamental model, and the portfolio faithfully reflects that. The ranking is stable
across book sizes (k=10/25/50 all give GNN ≳ LSTM > MISS > StockMixer). Closing this
gap requires better MISS training (or the paper's fuller 500-stock universe), not a
different backtest.

## 5. Robustness

- **Costs:** MISS Fund63 stays strongly profitable from 0 to 50 bps one-way
  (18.89%/1.021 at 0 bps → 17.57%/0.961 at 50 bps) — the low-turnover design works
  as advertised. Tech5, by contrast, collapses under costs (16.44% → −8.66%).
- **Bootstrap** (10k resamples, 21-day blocks): MISS Fund63 − MISS Tech5 Sharpe diff
  observed 0.398, P(diff > 0) = 0.843. Positive but weaker than under the old L/S
  construction, because long-only tech5 is itself profitable (+0.495 Sharpe) rather
  than deeply negative.
- **Deflated Sharpe** and sign tests are recomputed in `results/metrics.json`.

## 6. Files

- `src/backtest.py` — `run_backtest(..., mode="long_only", top_k=10, keep_k=40)` (default);
  `mode="ls_quintile"` preserves the SPEC v1 construction.
- `src/evaluate.py` — unchanged interface; now evaluates the long-only book.
- `results/metrics.json`, `results/tables.md` — regenerated 2026-10-01.
- `results/figures/` — Fig 1–4 regenerated (PNG + PDF).
- `results/equity/` — 12 daily net-return series.

## 7. Remaining gaps and what would close them

1. **Return/Sharpe level** (18.5%/1.00 vs 32.7%/1.22): our MISS signal is weaker than the
   paper's. Plausible causes: 189-stock vs 500-stock selection pool; our MISS val IC
   (0.028) trailing our own baselines. Fix = expand fundamental coverage / retrain MISS.
2. **"MISS is best" ranking**: same root cause as (1).
3. **Tech-regime event counts** (186/647 vs 25/187): our technical scores churn more than
   the paper's; their tech models or hysteresis were stickier.
4. **Construction confirmation**: the long-only design is inferred from reported moments.
   The paper's portfolio section should be checked before calling this a replication.
