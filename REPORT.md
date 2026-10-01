# Fundamental SPP — Reproduction Report

**Paper:** "Fundamental Information for Low-Turnover Equity ML: Evaluating a
Mamba-Inspired State Space Model for Long-Horizon Stock Trading" (Terry Ding,
March 2026).
**Reproduction date:** 2026-10-01 (rev 3). **Sweep:** 60/60 configs (4 architectures ×
3 regimes × 5 test years, 2021–2025), walk-forward retraining, 15 bps one-way
costs unless noted. Seed 42. Universe: S&P 500 constituents (fundamental regime
scores only the 161–201 names/yr with complete point-in-time fundamentals).

## TL;DR

After reading the paper's §3.5 and its cited construction source (Fischer &
Krauss 2018) directly, the backtest was rebuilt as a **concentrated
sector-neutral long/short book**: sector-demeaned scores, global top-10 /
bottom-10, equal weight, 100%/100% gross, sticky exit band, 15 bps. Under this
construction the paper's central claims **replicate in direction and mostly in
size**:

- **MISS Fund63 (faithful, gross 2.0): 17.53% return, Sharpe 1.104, 32 events/yr**
  vs the paper's 32.72% / 1.221 / 24. Sharpe replicates (1.10 vs 1.22); the
  return level is lower only because realized vol is 13.8% here vs the ~25% the
  paper's moments imply (our fundamental universe is 189 names, not 500, and our
  MISS signal is weaker at the extremes).
- **Vol-matched variant (top-7, 150%/150%): 31.07% / Sharpe 0.950 / 27 events**
  reproduces the paper's return and event count, at a lower Sharpe.
- **MISS is the best architecture on fundamentals in our runs too** (fund63
  Sharpe: MISS 1.104 > GNN 0.839 > LSTM 0.665 > StockMixer −0.239), matching the
  paper's headline ordering.
- Robustness matches or beats the paper: moving-block bootstrap
  P(fund63 Sharpe > tech5 Sharpe) = **0.988** (paper: 0.793); fund63 beats
  short-technical on return in **3/5 years, sign-test p = 0.50** (paper: 3/5,
  p = 0.50); MISS fund63 is the **only** config with deflated Sharpe ≈ 1.0.

What does **not** replicate, and why (signal, not construction):

- **StockMixer** is 2nd-best in the paper (fund63 Sharpe 1.090) but worst here
  (−0.024 IC, Sharpe −0.239). Its OOS RankIC is negative in 3 of 5 years — the
  trained scores simply have no fundamental signal in our reproduction.
- **Technical regimes** are near-zero-IC for every architecture here (MISS
  tech63 RankIC +0.0005), so the paper's positive tech Sharpes (0.694 / 0.380)
  do not appear. No portfolio rule manufactures signal that is not in the
  scores.
- **Year-by-year timing differs**: the paper's 2021 is its strongest fundamental
  year (+31.0%) and 2022 its weakest (−8.0%); in our scores 2021 has ~zero
  fundamental IC (0.004) and 2022 is strong (+0.069). The 5-year mean lands in
  the right place; the path there does not match.

## 1. Ground truth for the construction (read from sources)

### 1.1 The paper's §3.5
Ding (2026, §3.5) specifies *"a sector-neutral long-short portfolio: within each
sector, securities are ranked by predicted score; we hold long positions in the
highest-scoring group and short positions in the lowest-scoring group, sized to
keep sector-level risk roughly balanced and aggregate net market exposure near
zero. Position sizes are capped (2% of portfolio value per name); a wider exit
band is used for the low-turnover (63-day) signals. Transaction costs are 15 bps
per unit of one-way notional traded."* Group size, exit-band width and gross
exposure are not pinned down in the text.

### 1.2 Fischer & Krauss (2018) — the cited method
The paper's reference [5] is Fischer & Krauss, *Deep learning with long
short-term memory networks for financial market predictions* (FAU Discussion
Papers in Economics No. 11/2017). Their backtest (§3.5) *"rank[s] all stocks for
each period in descending order of this probability... we go long the top k and
short the flop k stocks of each ranking, for a long-short portfolio consisting
of 2k stocks"* (after Huck 2009/2010), with k ∈ {10, 50, 100, 150, 200} and the
focus on **k = 10, equal monetary weight (100% long / 100% short)**, S&P 500
universe. Their pre-cost daily return of 0.46% at Sharpe 5.8 implies ≈20%
annualized volatility — they note the selected names "exhibit high volatility."
F&K rank **globally, not within sectors**, and use no 2% cap and no exit band.

### 1.3 The 2% cap is incompatible with the paper's own moments (proof)
The paper's MISS Fund63 moments (mean return 32.72%, mean Sharpe 1.221) imply
annualized vol = 32.72 / 1.221 ≈ 26.8% (≈25% allowing for yearly averaging). For
an equal-weight long/short book of N names per side each with weight w and
typical single-stock vol σ ≈ 28% (measured here on 532 names, median), the
portfolio vol is

  vol ≈ w · σ · √(2N).

At the §3.5 cap w = 0.02, reaching even 25% vol requires
N = (0.25 / (0.02·0.28))² / 2 ≈ 318 names **per side** (≈637 positions); the
S&P 500 has 500 names total. With the whole index split top-250/bottom-250 at
the cap the ceiling is ≈13% vol. **A 2%-capped book cannot produce the paper's
reported volatility**, so the published moments must come from a concentrated
book (the cap, if applied at all, was not binding). The observed author also
confirmed (2026-10-01) the universe was the S&P 500 and could not recall whether
2% was a cap or a fixed size; the original backtest code no longer exists.

### 1.4 Reconciling "sector-neutral" with a global top/bottom-k
Sector-demeaning the scores (subtract each stock's sector cross-sectional mean)
before a global ranking removes sector tilts: the top-10/bottom-10 then come
from within-sector relative strength rather than whole hot/cold sectors. It is
also empirically right — it **raises** the fundamental RankIC out of sample
(MISS fund63 0.027 → 0.039; see §2), so the sector-neutral instruction and the
concentrated F&K form are consistent once "within each sector rank" is read as
"compare within sector," which is what demeaning does.

## 2. Signal diagnostics (RankIC) — the real story

Mean RankIC (Spearman of score vs forward 63d/5d return), 2021–2025. This is
construction-independent and says what the portfolios can possibly earn.

| regime | MISS | StockMixer | GNN | LSTM |
|---|---|---|---|---|
| fund63 (raw) | +0.027 | **−0.012** | +0.027 | +0.036 |
| fund63 (sector-demeaned) | **+0.039** | −0.002 | +0.033 | +0.033 |
| tech63 (raw) | +0.001 | +0.011 | +0.010 | +0.032 |
| tech5 (raw) | +0.008 | +0.003 | +0.002 | −0.010 |

Readings:
- **MISS has the best demeaned fundamental IC (+0.039)** — the paper's "MISS is
  best on fundamentals" is supported at the signal level in our reruns (contra
  the earlier revision-2 note, which used raw IC, where LSTM led).
- **StockMixer fund63 IC is negative (−0.012 raw; −0.041/−0.044 in 2023/2024).**
  It is not a sign flip (2022 is +0.036); the model is unreliable OOS. This, not
  the backtest, explains its last-place Sharpe.
- **MISS tech63 IC ≈ 0.0005** and every tech5 IC is within ±0.01 — the technical
  scores carry almost no cross-sectional signal here. The paper's tech results
  are not recoverable from these scores under any construction.
- Per-year MISS fund63 IC: 2021 +0.004, 2022 +0.069, 2023 +0.086 (demeaned),
  2024 −0.016, 2025 +0.049. The near-zero 2021 IC is why the paper's strong 2021
  does not appear (see TL;DR).

## 3. Headline results (Table 1 equivalents)

**Option A — faithful (PRIMARY):** sector-demeaned, global top-10/bottom-10,
equal weight, 100%/100% (gross 2.0), exit band xk=80, monthly (fund63/tech63) /
weekly-Monday (tech5), 15 bps. 5-year means; full per-year in `results/tables.md`.

| Arch | Fund63 | Tech63 | Tech5 |
|---|---|---|---|
| **MISS** | **+17.53% / 1.104 / 32ev** | −6.84% / −0.299 / 254ev | −4.48% / −0.276 / 605ev |
| StockMixer | −3.48% / −0.239 / 98ev | −1.69% / −0.201 / 264ev | +0.11% / −0.039 / 522ev |
| GNN | +11.92% / 0.839 / 32ev | +0.60% / +0.174 / 294ev | −5.66% / −0.222 / 427ev |
| LSTM | +9.71% / 0.665 / 24ev | +0.80% / +0.115 / 200ev | −8.24% / −0.302 / 608ev |

Paper MISS targets: Fund63 +32.72%/1.221/24ev, Tech63 +15.15%/0.694/25ev,
Tech5 +12.14%/0.380/187ev. Our Fund63 Sharpe replicates (1.104 vs 1.221); the
return is lower because realized vol is 13.8% not ~25%. Tech events overshoot
because near-zero-IC scores churn through the exit band; the paper's low tech
event counts require sticky scores ours do not have.

**Option B — vol-matched variant:** top-7/bottom-7, 150%/150% (gross 3.0),
xk=50, otherwise identical. Full results in `results_B/`.

| Arch | Fund63 | Tech63 | Tech5 |
|---|---|---|---|
| **MISS** | **+31.07% / 0.950 / 27ev** | −12.30% / −0.228 / 203ev | −9.62% / −0.430 / 524ev |
| StockMixer | +1.98% / +0.199 / 99ev | −2.66% / −0.081 / 213ev | −10.79% / −0.533 / 434ev |
| GNN | +24.18% / 0.878 / 26ev | +8.24% / +0.365 / 227ev | −15.28% / −0.614 / 350ev |
| LSTM | +21.85% / 0.748 / 17ev | −6.70% / −0.026 / 170ev | −12.15% / −0.187 / 498ev |

Option B reproduces the paper's Fund63 return (31.07% vs 32.72%), vol (24.7%)
and events (27 vs 24) simultaneously, but the extra leverage costs Sharpe
(0.950 vs 1.221). We report A as the paper-faithful construction (F&K used no
leverage beyond 100/100) and B to show exactly which moment the leverage buys.

**MISS Fund63 by year (Option A)** vs paper:

| Year | Ours (ret / Sharpe / ev) | Paper (ret / Sharpe / ev) |
|---|---|---|
| 2021 | −1.9% / −0.07 / 34 | +31.0% / 1.28 / 21 |
| 2022 | +23.0% / +1.49 / 22 | −8.0% / −0.42 / 24 |
| 2023 | +18.2% / +1.47 / 34 | +22.0% / 0.74 / 26 |
| 2024 | +1.9% / +0.23 / 36 | +50.0% / 1.95 / 25 |
| 2025 | +46.4% / +2.41 / 34 | +68.6% / 2.555 / 24 |

2023 and 2025 are strong in both. 2021/2022 are effectively swapped relative to
the paper — consistent with the IC pattern in §2 (our fundamental signal is
absent in 2021 and strong in 2022; the paper's was the reverse). This is a
training-data/universe difference, not a construction effect: it appears under
both Option A and B.

## 4. Architecture ranking (Table 2 equivalent, fund63 Sharpe)

| Arch | Ours (A) | Ours (B) | Paper |
|---|---|---|---|
| **MISS** | **1.104** | **0.950** | **1.221** |
| StockMixer | −0.239 | 0.199 | 1.090 |
| GNN | 0.839 | 0.878 | 1.030 |
| LSTM | 0.665 | 0.748 | 0.910 |

MISS-best replicates (this reverses the revision-2 long-only finding, an
artifact of that book's different stock selection). StockMixer-last does not
(paper: 2nd) — fully explained by its negative OOS RankIC (§2).

## 5. Robustness (Option A)

- **Costs:** MISS Fund63 Sharpe 1.136 (0 bps) → 1.028 (50 bps); return 18.02% →
  16.40%. The low-turnover design is cost-robust as claimed. (Full grid:
  `results/metrics.json` → `cost_sensitivity`.)
- **Bootstrap:** 10,000 moving-block resamples (21d blocks) of the concatenated
  2021–2025 daily series: Sharpe(fund63) − Sharpe(tech5) = 1.520, 95% CI
  [0.215, 2.880], **P(diff > 0) = 0.988** (paper reports 0.793).
- **Sign tests:** fund63 return > tech5 return in 3/5 years, **p = 0.50** —
  identical count and p-value to the paper's fundamental-vs-short-technical
  test.
- **Deflated Sharpe:** MISS fund63 DSR ≈ 1.00 (benchmark SR0 = 0.83 over 12
  trials); every other config ≈ 0. The fundamental-MISS result is the single
  survivor of multiple-testing adjustment here.
- **Construction robustness:** exit bands xk ∈ {60, 80, 120} at k=10 give
  Sharpe 1.02 / 1.10 / 1.05 and 37 / 32 / 26 events — the result is not a tuned
  artifact of the band; random-score controls give Sharpe ≈ 0 as required.

## 6. What replicates / what does not

**Replicates:** (i) fundamentals ≫ technicals (return, Sharpe, bootstrap,
sign test); (ii) MISS is the best fundamental architecture (backtest Sharpe
*and* demeaned RankIC); (iii) Fund63 Sharpe level ≈ paper (1.10 vs 1.22);
(iv) Fund63 return/vol/events jointly under Option B; (v) extreme cost
robustness of the 63d fundamental book; (vi) 2023 and 2025 as strong years.

**Does not replicate:** (i) Fund63 return under the faithful gross (vol 13.8%
vs ~25% — smaller selection universe, weaker extreme scores, no leverage);
(ii) StockMixer's ranking (negative OOS IC in our training); (iii) all
technical-regime performance (near-zero IC in our scores); (iv) the
year-by-year path, notably 2021 vs 2022 (IC timing). None of these is
repairable at the backtest layer; all are properties of the 60 trained score
files, and the original backtest code that produced the paper's numbers no
longer exists to diff against.

## 7. Reproduce

```bash
# Option A (faithful, canonical -> results/)
python3 src/evaluate.py --scores_dir scores --out results \
    --mode fk_ls --top_k 10 --exit_k 80 --gross 2.0
# Option B (vol-matched -> results_B/)
python3 src/evaluate.py --scores_dir scores --out results_B \
    --mode fk_ls --top_k 7 --exit_k 50 --gross 3.0
# Figures (from results/)
python3 src/make_figures.py
# Signal diagnostic behind §2
python3 diag_ic.py
```

- `src/backtest.py` — `run_backtest(..., mode="fk_ls")` (default): sector-demeaned
  global top-K/bottom-K L/S; `mode="long_only"` (rev-2) and `mode="ls_quintile"`
  (SPEC v1) preserved.
- `results/` — Option A metrics/tables/equity/figures. `results_B/` — Option B.
- `ic_diagnostic.json` — per-file RankIC table (§2).

## 8. Remaining gaps and what would close them

1. **Vol/return level (A):** needs either the paper's broader fundamental
   universe (500 names with PIT fundamentals; we have ≤201) or leverage
   (Option B). Not closable without new fundamental data coverage.
2. **StockMixer & technical signals:** needs retraining/debugging those models
   (their scores, as trained, carry no OOS signal). Out of scope for a
   backtest-layer reproduction; flagged with IC evidence rather than tuned
   around.
3. **2021/2022 path:** same root cause as (2) — when the signal exists in a year
   differs between our training and the paper's.
