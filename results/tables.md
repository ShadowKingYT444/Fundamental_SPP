# Fundamental_SPP — Evaluation results

Generated (UTC): 2026-10-01T22:23:49.166163+00:00 | seed=42 | costs=15.0bps | bootstrap: 10000 resamples, block=21d

## Score-file contract

Columns: `ticker, date, sector, score` (`date` as YYYY-MM-DD). Filename: `<model>_<regime>_<year>.csv`, e.g. `miss_fund63_2021.csv`. `model` ∈ {miss, lstm, stockmixer, gnn}; `regime` ∈ {fund63, tech63, tech5} (fund63/tech63 → monthly rebalance, tech5 → weekly Monday rebalance); one test year per file. Prices come from `data/prices.parquet` (`ticker, date, adj_close`) unless `--prices` is given.

## Per-year metrics

| config | year | ann. return | Sharpe | turnover | trade events | max DD |
|---|---|---|---|---|---|---|
| gnn_fund63 | 2021 | 7.67% | 0.665 | 4.22 | 30 | -8.36% |
| gnn_fund63 | 2022 | 38.55% | 2.378 | 5.88 | 40 | -5.36% |
| gnn_fund63 | 2023 | 1.08% | 0.147 | 4.78 | 28 | -18.01% |
| gnn_fund63 | 2024 | 12.47% | 0.898 | 4.84 | 34 | -6.94% |
| gnn_fund63 | 2025 | -0.18% | 0.107 | 4.25 | 26 | -25.38% |
| gnn_tech5 | 2021 | 28.44% | 1.790 | 55.49 | 540 | -8.31% |
| gnn_tech5 | 2022 | -2.62% | 0.023 | 36.00 | 318 | -24.06% |
| gnn_tech5 | 2023 | -25.71% | -1.868 | 46.24 | 430 | -27.41% |
| gnn_tech5 | 2024 | 0.42% | 0.107 | 56.49 | 536 | -9.24% |
| gnn_tech5 | 2025 | -28.82% | -1.164 | 35.72 | 310 | -34.60% |
| gnn_tech63 | 2021 | 4.11% | 0.309 | 20.98 | 202 | -13.23% |
| gnn_tech63 | 2022 | 2.03% | 0.200 | 38.38 | 380 | -15.67% |
| gnn_tech63 | 2023 | 20.30% | 1.130 | 24.02 | 232 | -11.57% |
| gnn_tech63 | 2024 | 1.60% | 0.179 | 37.47 | 366 | -21.05% |
| gnn_tech63 | 2025 | -25.02% | -0.947 | 30.65 | 292 | -33.40% |
| lstm_fund63 | 2021 | -8.81% | -0.480 | 4.47 | 26 | -22.49% |
| lstm_fund63 | 2022 | 29.21% | 1.852 | 5.11 | 32 | -10.92% |
| lstm_fund63 | 2023 | 16.63% | 1.183 | 3.93 | 22 | -7.78% |
| lstm_fund63 | 2024 | -12.61% | -0.688 | 4.39 | 22 | -17.96% |
| lstm_fund63 | 2025 | 24.11% | 1.456 | 3.77 | 20 | -11.06% |
| lstm_tech5 | 2021 | 10.29% | 0.746 | 62.11 | 600 | -6.78% |
| lstm_tech5 | 2022 | 5.88% | 0.352 | 38.59 | 350 | -21.14% |
| lstm_tech5 | 2023 | -30.45% | -1.492 | 43.72 | 396 | -30.41% |
| lstm_tech5 | 2024 | -5.93% | -0.330 | 99.57 | 976 | -12.17% |
| lstm_tech5 | 2025 | -20.96% | -0.784 | 75.29 | 720 | -35.03% |
| lstm_tech63 | 2021 | 13.90% | 0.687 | 13.29 | 122 | -17.54% |
| lstm_tech63 | 2022 | 8.55% | 0.480 | 13.97 | 120 | -20.41% |
| lstm_tech63 | 2023 | 12.38% | 0.607 | 30.24 | 292 | -25.02% |
| lstm_tech63 | 2024 | -9.57% | -0.276 | 22.87 | 212 | -28.08% |
| lstm_tech63 | 2025 | -21.26% | -0.921 | 26.64 | 252 | -28.86% |
| miss_fund63 | 2021 | -1.88% | -0.069 | 4.70 | 34 | -14.37% |
| miss_fund63 | 2022 | 23.01% | 1.488 | 3.98 | 22 | -10.52% |
| miss_fund63 | 2023 | 18.18% | 1.466 | 4.92 | 34 | -6.46% |
| miss_fund63 | 2024 | 1.90% | 0.228 | 5.15 | 36 | -14.29% |
| miss_fund63 | 2025 | 46.44% | 2.407 | 4.83 | 34 | -7.51% |
| miss_tech5 | 2021 | 12.64% | 0.884 | 53.03 | 514 | -13.52% |
| miss_tech5 | 2022 | -14.26% | -0.798 | 68.78 | 658 | -21.73% |
| miss_tech5 | 2023 | -31.75% | -2.353 | 70.85 | 678 | -29.61% |
| miss_tech5 | 2024 | 24.81% | 1.620 | 60.06 | 578 | -7.85% |
| miss_tech5 | 2025 | -13.84% | -0.731 | 62.37 | 598 | -20.53% |
| miss_tech63 | 2021 | -24.21% | -1.527 | 24.75 | 226 | -30.14% |
| miss_tech63 | 2022 | -3.48% | -0.003 | 27.91 | 274 | -22.66% |
| miss_tech63 | 2023 | 10.62% | 0.770 | 33.58 | 328 | -13.74% |
| miss_tech63 | 2024 | -15.12% | -0.731 | 17.47 | 162 | -25.99% |
| miss_tech63 | 2025 | -2.01% | -0.004 | 28.59 | 278 | -17.90% |
| stockmixer_fund63 | 2021 | -3.83% | -0.306 | 10.55 | 94 | -11.68% |
| stockmixer_fund63 | 2022 | 2.35% | 0.258 | 11.24 | 100 | -11.74% |
| stockmixer_fund63 | 2023 | -4.85% | -0.430 | 11.94 | 106 | -16.05% |
| stockmixer_fund63 | 2024 | -6.16% | -0.466 | 9.06 | 78 | -14.04% |
| stockmixer_fund63 | 2025 | -4.92% | -0.251 | 12.34 | 112 | -17.73% |
| stockmixer_tech5 | 2021 | 11.17% | 0.906 | 48.72 | 464 | -8.93% |
| stockmixer_tech5 | 2022 | -1.80% | -0.098 | 61.09 | 588 | -13.06% |
| stockmixer_tech5 | 2023 | -24.68% | -2.177 | 49.83 | 472 | -24.31% |
| stockmixer_tech5 | 2024 | -5.26% | -0.452 | 58.47 | 562 | -15.37% |
| stockmixer_tech5 | 2025 | 21.15% | 1.628 | 55.17 | 522 | -8.24% |
| stockmixer_tech63 | 2021 | -11.00% | -1.035 | 33.73 | 324 | -18.79% |
| stockmixer_tech63 | 2022 | -17.68% | -1.617 | 23.57 | 226 | -20.95% |
| stockmixer_tech63 | 2023 | 4.33% | 0.431 | 29.73 | 288 | -9.99% |
| stockmixer_tech63 | 2024 | 0.01% | 0.070 | 22.38 | 216 | -16.99% |
| stockmixer_tech63 | 2025 | 15.88% | 1.145 | 27.12 | 266 | -8.70% |

## 5-year means vs paper targets (Table 1, MISS)

| config | ann. return (ours / paper) | Sharpe (ours / paper) | trade events/yr (ours / paper) |
|---|---|---|---|
| gnn_fund63 | 11.92% / n/a | 0.839 / n/a | 32 / n/a |
| gnn_tech5 | -5.66% / n/a | -0.222 / n/a | 427 / n/a |
| gnn_tech63 | 0.60% / n/a | 0.174 / n/a | 294 / n/a |
| lstm_fund63 | 9.71% / n/a | 0.665 / n/a | 24 / n/a |
| lstm_tech5 | -8.24% / n/a | -0.302 / n/a | 608 / n/a |
| lstm_tech63 | 0.80% / n/a | 0.115 / n/a | 200 / n/a |
| miss_fund63 | 17.53% / 32.72% | 1.104 / 1.221 | 32 / 24 |
| miss_tech5 | -4.48% / 12.14% | -0.276 / 0.380 | 605 / 187 |
| miss_tech63 | -6.84% / 15.15% | -0.299 / 0.694 | 254 / 25 |
| stockmixer_fund63 | -3.48% / n/a | -0.239 / n/a | 98 / n/a |
| stockmixer_tech5 | 0.11% / n/a | -0.039 / n/a | 522 / n/a |
| stockmixer_tech63 | -1.69% / n/a | -0.201 / n/a | 264 / n/a |

## Table 2 — Sharpe grid (targets TBD from paper)

| config | Sharpe ours | Sharpe paper |
|---|---|---|
| gnn_fund63 | 0.839 | 1.030 |
| gnn_tech5 | -0.222 | 0.410 |
| gnn_tech63 | 0.174 | 0.660 |
| lstm_fund63 | 0.665 | 0.910 |
| lstm_tech5 | -0.302 | 0.330 |
| lstm_tech63 | 0.115 | 0.580 |
| miss_fund63 | 1.104 | 1.221 |
| miss_tech5 | -0.276 | 0.380 |
| miss_tech63 | -0.299 | 0.694 |
| stockmixer_fund63 | -0.239 | 1.090 |
| stockmixer_tech5 | -0.039 | 0.440 |
| stockmixer_tech63 | -0.201 | 0.730 |

## Cost sensitivity (5y means)

| config | 0bps ret / Sharpe | 5bps ret / Sharpe | 10bps ret / Sharpe | 25bps ret / Sharpe | 50bps ret / Sharpe |
|---|---|---|---|---|---|
| gnn_fund63 | 12.41% / 0.869 | 12.25% / 0.859 | 12.08% / 0.849 | 11.59% / 0.819 | 10.77% / 0.770 |
| gnn_tech5 | 0.97% / 0.151 | -1.29% / 0.027 | -3.50% / -0.097 | -9.83% / -0.472 | -19.50% / -1.090 |
| gnn_tech63 | 4.90% / 0.386 | 3.45% / 0.316 | 2.02% / 0.245 | -2.18% / 0.034 | -8.84% / -0.309 |
| lstm_fund63 | 10.11% / 0.689 | 9.97% / 0.681 | 9.84% / 0.673 | 9.44% / 0.648 | 8.77% / 0.608 |
| lstm_tech5 | 0.72% / 0.202 | -2.37% / 0.034 | -5.35% / -0.134 | -13.73% / -0.632 | -25.98% / -1.415 |
| lstm_tech63 | 3.71% / 0.239 | 2.73% / 0.197 | 1.76% / 0.156 | -1.10% / 0.034 | -5.71% / -0.169 |
| miss_fund63 | 18.02% / 1.136 | 17.86% / 1.125 | 17.69% / 1.115 | 17.21% / 1.082 | 16.40% / 1.028 |
| miss_tech5 | 4.48% / 0.298 | 1.41% / 0.106 | -1.58% / -0.085 | -10.04% / -0.654 | -22.62% / -1.558 |
| miss_tech63 | -3.27% / -0.102 | -4.48% / -0.167 | -5.66% / -0.233 | -9.15% / -0.430 | -14.72% / -0.751 |
| stockmixer_fund63 | -2.17% / -0.126 | -2.61% / -0.164 | -3.05% / -0.201 | -4.34% / -0.314 | -6.48% / -0.498 |
| stockmixer_tech5 | 8.32% / 0.626 | 5.52% / 0.405 | 2.78% / 0.183 | -5.03% / -0.476 | -16.81% / -1.503 |
| stockmixer_tech63 | 2.09% / 0.114 | 0.82% / 0.008 | -0.44% / -0.097 | -4.15% / -0.407 | -10.08% / -0.891 |

## Robustness

Moving-block bootstrap of Sharpe(miss_fund63) − Sharpe(miss_tech5): 10000 resamples, block 21d, 1250 days. Mean diff 1.530 [0.215, 2.880], P(diff > 0) = 0.9879.

Year-level one-sided sign tests (H1: P(win) > 0.5):

- gnn_fund63__ret_gt_0: 4/5 wins, p = 0.1875
- gnn_tech5__ret_gt_0: 2/5 wins, p = 0.8125
- gnn_tech63__ret_gt_0: 4/5 wins, p = 0.1875
- lstm_fund63__ret_gt_0: 3/5 wins, p = 0.5000
- lstm_tech5__ret_gt_0: 2/5 wins, p = 0.8125
- lstm_tech63__ret_gt_0: 3/5 wins, p = 0.5000
- miss_fund63__beats__miss_tech5: 3/5 wins, p = 0.5000
- miss_fund63__ret_gt_0: 4/5 wins, p = 0.1875
- miss_tech5__ret_gt_0: 2/5 wins, p = 0.8125
- miss_tech63__ret_gt_0: 1/5 wins, p = 0.9688
- stockmixer_fund63__ret_gt_0: 1/5 wins, p = 0.9688
- stockmixer_tech5__ret_gt_0: 2/5 wins, p = 0.8125
- stockmixer_tech63__ret_gt_0: 3/5 wins, p = 0.5000

Deflated Sharpe Ratios (Bailey & Lopez de Prado; trials = #configs = 12):

| config | DSR | SR0 benchmark |
|---|---|---|
| gnn_fund63 | 0.0060 | 0.828 |
| gnn_tech5 | 0.0000 | 0.828 |
| gnn_tech63 | 0.0000 | 0.828 |
| lstm_fund63 | 0.0000 | 0.828 |
| lstm_tech5 | 0.0000 | 0.828 |
| lstm_tech63 | 0.0000 | 0.828 |
| miss_fund63 | 1.0000 | 0.828 |
| miss_tech5 | 0.0000 | 0.828 |
| miss_tech63 | 0.0000 | 0.828 |
| stockmixer_fund63 | 0.0000 | 0.828 |
| stockmixer_tech5 | 0.0000 | 0.828 |
| stockmixer_tech63 | 0.0000 | 0.828 |

