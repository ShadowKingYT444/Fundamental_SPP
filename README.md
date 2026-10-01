# Fundamental SPP — Paper Reproduction

End-to-end reproduction of **"Fundamental Information for Low-Turnover Equity
ML"**: Mamba-based MISS plus LSTM / StockMixer / GNN baselines, walk-forward
backtest 2021–2025 across three feature regimes (Fund63, Tech63, Tech5).

**Status:** 60/60 sweep configs complete. See [REPORT.md](REPORT.md) for the
honest verdict — regime ordering and MISS-on-fundamentals replicate; absolute
return levels and trade-event counts do not.

## Layout

```
src/                training + evaluation pipeline
  models.py         MISS (selective SSM), LSTM, StockMixer, GNN
  train.py          walk-forward training harness (--model/--regime/--year)
  backtest.py       portfolio backtest (15 bps one-way default)
  evaluate.py       metrics: per-year, 5y means, cost sensitivity,
                    moving-block bootstrap, sign tests, deflated Sharpe
  make_figures.py   paper figures 1–4 (300-DPI PNG + editable PDF)
  plot_style.py     figures4papers house style
colab/
  sweep_gpu.ipynb   the 60-config Colab GPU sweep (resume-capable, Drive-synced)
data/               prepared panels, prices, universe (not all in git)
results/
  metrics.json      full evaluation output (60 configs)
  tables.md         all tables
  equity/           12 daily net-return series (arch × regime)
  figures/          fig1..fig4.{png,pdf}
REPORT.md           reproduction report with honest paper comparison
```

## Quickstart

```bash
# 1. Evaluate trained score files -> metrics, tables, equity curves
python3 src/evaluate.py --scores_dir scores --out results

# 2. Build the paper figures
python3 src/make_figures.py        # real figures -> results/figures/
python3 src/make_figures.py --smoke  # synthetic-data smoke test -> /tmp/fig_smoke
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

| MISS regime | Return (ours / paper) | Sharpe (ours / paper) | Events/yr (ours / paper) |
|---|---|---|---|
| Fund63 | 2.46% / 32.72% | 0.935 / 1.221 | 195 / 24 |
| Tech63 | −1.58% / 15.15% | −0.310 / 0.694 | 2,255 / 25 |
| Tech5 | −3.61% / 12.14% | −1.161 / 0.380 | 6,184 / 187 |

Details and caveats: [REPORT.md](REPORT.md).
