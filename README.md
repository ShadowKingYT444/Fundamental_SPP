# Fundamental SPP — Paper Reproduction

End-to-end reproduction of **"Fundamental Information for Low-Turnover Equity
ML"** (Ding 2026): Mamba-based MISS plus LSTM / StockMixer / GNN baselines,
walk-forward backtest 2021–2025 across three feature regimes (Fund63, Tech63,
Tech5).

**Status:** 60/60 sweep configs complete; backtest rebuilt to the paper's
sector-neutral long/short construction (revision 3). See [REPORT.md](REPORT.md)
for the full verdict — the fundamental ranking, MISS-best-on-fundamentals, and
the Fund63 Sharpe replicate; the technical-regime and StockMixer numbers do
not, for signal reasons documented with RankIC evidence.

## Layout

```
src/                training + evaluation pipeline
  models.py         MISS (selective SSM), LSTM, StockMixer, GNN
  train.py          walk-forward training harness (--model/--regime/--year)
  backtest.py       portfolio backtest: mode="fk_ls" (default, sector-demeaned
                    concentrated L/S), "long_only" (rev 2), "ls_quintile" (v1)
  evaluate.py       metrics: per-year, 5y means, cost sensitivity,
                    moving-block bootstrap, sign tests, deflated Sharpe;
                    --mode/--top_k/--exit_k/--gross/--no_demean select the book
  make_figures.py   paper figures 1–4 (300-DPI PNG + editable PDF)
  plot_style.py     figures4papers house style
diag_ic.py          RankIC signal diagnostic behind REPORT §2
ic_diagnostic.json  per-file RankIC table
colab/
  sweep_gpu.ipynb   the 60-config Colab GPU sweep (resume-capable, Drive-synced)
data/               prepared panels, prices, universe (not all in git)
results/            Option A (faithful): metrics.json, tables.md, equity/, figures/
results_B/          Option B (vol-matched variant): metrics.json, tables.md, equity/
paper/              LaTeX reproduction edition (main.tex) + rendered main.pdf,
                    Figures 1-4 in paper/figures/
REPORT.md           reproduction report with honest paper comparison
```

## Quickstart

```bash
# Option A — faithful F&K-style book (canonical -> results/)
python3 src/evaluate.py --scores_dir scores --out results \
    --mode fk_ls --top_k 10 --exit_k 80 --gross 2.0

# Option B — vol-matched variant (-> results_B/)
python3 src/evaluate.py --scores_dir scores --out results_B \
    --mode fk_ls --top_k 7 --exit_k 50 --gross 3.0

# Paper figures (from results/)
python3 src/make_figures.py
```

Training one config (needs the data panels + a GPU for MISS):

```bash
python3 src/train.py --model miss --regime fund63 --year 2021 --device auto
```

The full 60-config sweep runs from `colab/sweep_gpu.ipynb`: it stages data from
the Drive tarball, resumes from `results/run_status.json`, trains each
config, scores it, syncs checkpoints/scores/results to Drive, and finishes with
the evaluation + figure cells. Score CSVs (277 MB total) live on Drive, not in
git — only the compact metrics/tables/figures are committed.

## Key numbers (net of 15 bps, 5y means)

Portfolio construction (REPORT §1): scores are sector-demeaned, then a global
top-K/bottom-K long/short, equal weight, sticky exit band — the Fischer &
Krauss (2018) form the paper cites, reconciled with its "sector-neutral"
wording. The §3.5 "2% cap" is provably incompatible with the paper's own
reported volatility and was therefore not binding in whatever produced the
paper's tables (proof in REPORT §1.3).

| MISS Fund63 | Return (ours / paper) | Sharpe (ours / paper) | Events/yr (ours / paper) |
|---|---|---|---|
| **Option A** (top-10, 100%/100%) | 17.53% / 32.72% | **1.104 / 1.221** | 32 / 24 |
| **Option B** (top-7, 150%/150%) | **31.07% / 32.72%** | 0.950 / 1.221 | **27 / 24** |

Architecture ranking on Fund63 replicates (MISS best in ours and the paper);
StockMixer does not (negative out-of-sample RankIC here). Details and caveats:
[REPORT.md](REPORT.md).
