# Fundamental_SPP — Evaluation results

Generated (UTC): 2026-10-01T17:22:14.014840+00:00 | seed=42 | costs=15.0bps | bootstrap: 10000 resamples, block=21d

## Score-file contract

Columns: `ticker, date, sector, score` (`date` as YYYY-MM-DD). Filename: `<model>_<regime>_<year>.csv`, e.g. `miss_fund63_2021.csv`. `model` ∈ {miss, lstm, stockmixer, gnn}; `regime` ∈ {fund63, tech63, tech5} (fund63/tech63 → monthly rebalance, tech5 → weekly Monday rebalance); one test year per file. Prices come from `data/prices.parquet` (`ticker, date, adj_close`) unless `--prices` is given.

## Per-year metrics

| config | year | ann. return | Sharpe | turnover | trade events | max DD |
|---|---|---|---|---|---|---|
| gnn_fund63 | 2021 | 27.88% | 1.846 | 3.93 | 41 | -8.21% |
| gnn_fund63 | 2022 | -17.78% | -0.521 | 4.48 | 42 | -26.99% |
| gnn_fund63 | 2023 | 35.57% | 2.127 | 2.69 | 23 | -8.94% |
| gnn_fund63 | 2024 | 16.32% | 1.271 | 3.31 | 33 | -8.39% |
| gnn_fund63 | 2025 | 18.24% | 1.175 | 2.04 | 15 | -9.03% |
| gnn_tech5 | 2021 | 23.53% | 1.069 | 36.28 | 468 | -11.70% |
| gnn_tech5 | 2022 | -7.30% | -0.089 | 17.38 | 229 | -27.40% |
| gnn_tech5 | 2023 | 11.15% | 0.684 | 45.82 | 600 | -15.66% |
| gnn_tech5 | 2024 | -1.06% | -0.007 | 35.91 | 444 | -15.08% |
| gnn_tech5 | 2025 | 28.21% | 1.064 | 48.28 | 654 | -17.31% |
| gnn_tech63 | 2021 | 21.81% | 1.468 | 11.09 | 142 | -8.42% |
| gnn_tech63 | 2022 | 0.92% | 0.167 | 20.36 | 218 | -23.81% |
| gnn_tech63 | 2023 | 30.56% | 1.248 | 14.58 | 200 | -17.40% |
| gnn_tech63 | 2024 | -5.25% | -0.255 | 21.87 | 222 | -12.40% |
| gnn_tech63 | 2025 | 7.76% | 0.425 | 19.51 | 215 | -20.75% |
| lstm_fund63 | 2021 | 18.99% | 0.999 | 3.68 | 36 | -13.58% |
| lstm_fund63 | 2022 | -8.90% | -0.190 | 3.61 | 33 | -24.00% |
| lstm_fund63 | 2023 | 45.77% | 2.425 | 2.56 | 20 | -12.69% |
| lstm_fund63 | 2024 | 17.84% | 1.151 | 3.45 | 31 | -6.85% |
| lstm_fund63 | 2025 | 22.31% | 1.080 | 2.73 | 19 | -16.09% |
| lstm_tech5 | 2021 | 31.18% | 1.749 | 40.16 | 527 | -7.02% |
| lstm_tech5 | 2022 | -12.18% | -0.574 | 30.22 | 417 | -20.38% |
| lstm_tech5 | 2023 | 4.17% | 0.443 | 40.09 | 554 | -9.37% |
| lstm_tech5 | 2024 | -8.74% | -0.516 | 69.75 | 821 | -12.50% |
| lstm_tech5 | 2025 | -3.30% | -0.149 | 61.28 | 770 | -15.35% |
| lstm_tech63 | 2021 | 60.82% | 1.704 | 10.46 | 135 | -17.83% |
| lstm_tech63 | 2022 | -19.45% | -0.352 | 12.74 | 167 | -33.06% |
| lstm_tech63 | 2023 | 41.44% | 1.411 | 18.30 | 200 | -26.94% |
| lstm_tech63 | 2024 | -11.42% | -0.556 | 15.51 | 183 | -16.89% |
| lstm_tech63 | 2025 | 5.16% | 0.323 | 14.50 | 175 | -30.09% |
| miss_fund63 | 2021 | 9.45% | 0.496 | 3.55 | 30 | -21.26% |
| miss_fund63 | 2022 | -8.47% | -0.127 | 2.71 | 18 | -27.37% |
| miss_fund63 | 2023 | 53.05% | 2.607 | 2.90 | 27 | -11.61% |
| miss_fund63 | 2024 | 7.42% | 0.544 | 4.17 | 44 | -9.29% |
| miss_fund63 | 2025 | 31.03% | 1.493 | 2.92 | 24 | -11.29% |
| miss_tech5 | 2021 | 31.73% | 2.010 | 33.66 | 452 | -8.93% |
| miss_tech5 | 2022 | -10.82% | -0.462 | 55.78 | 715 | -16.93% |
| miss_tech5 | 2023 | -16.45% | -0.914 | 57.79 | 707 | -31.51% |
| miss_tech5 | 2024 | 32.97% | 1.563 | 53.66 | 688 | -9.87% |
| miss_tech5 | 2025 | 3.92% | 0.278 | 52.24 | 674 | -19.15% |
| miss_tech63 | 2021 | 21.79% | 1.064 | 16.36 | 196 | -17.36% |
| miss_tech63 | 2022 | -25.54% | -0.503 | 12.69 | 150 | -39.73% |
| miss_tech63 | 2023 | 27.72% | 1.150 | 18.46 | 208 | -19.18% |
| miss_tech63 | 2024 | -2.38% | 0.024 | 12.77 | 163 | -15.00% |
| miss_tech63 | 2025 | 5.22% | 0.338 | 19.50 | 214 | -18.56% |
| stockmixer_fund63 | 2021 | 27.87% | 1.828 | 8.95 | 107 | -5.99% |
| stockmixer_fund63 | 2022 | -20.94% | -0.847 | 9.72 | 112 | -24.26% |
| stockmixer_fund63 | 2023 | 26.25% | 1.594 | 9.93 | 130 | -11.21% |
| stockmixer_fund63 | 2024 | 3.67% | 0.347 | 8.37 | 109 | -9.30% |
| stockmixer_fund63 | 2025 | 9.55% | 0.617 | 11.02 | 138 | -12.90% |
| stockmixer_tech5 | 2021 | 40.20% | 1.960 | 38.24 | 503 | -7.37% |
| stockmixer_tech5 | 2022 | -20.90% | -0.871 | 51.05 | 672 | -28.73% |
| stockmixer_tech5 | 2023 | 1.29% | 0.163 | 38.15 | 501 | -15.05% |
| stockmixer_tech5 | 2024 | -3.90% | -0.169 | 41.82 | 547 | -10.15% |
| stockmixer_tech5 | 2025 | 22.95% | 1.140 | 40.97 | 533 | -11.78% |
| stockmixer_tech63 | 2021 | 33.79% | 1.991 | 19.07 | 217 | -8.03% |
| stockmixer_tech63 | 2022 | -17.46% | -0.636 | 15.83 | 182 | -29.62% |
| stockmixer_tech63 | 2023 | 21.92% | 1.252 | 17.31 | 205 | -17.14% |
| stockmixer_tech63 | 2024 | 9.04% | 0.716 | 14.51 | 172 | -8.26% |
| stockmixer_tech63 | 2025 | 11.72% | 0.674 | 16.42 | 199 | -20.11% |

## 5-year means vs paper targets (Table 1, MISS)

| config | ann. return (ours / paper) | Sharpe (ours / paper) | trade events/yr (ours / paper) |
|---|---|---|---|
| gnn_fund63 | 16.05% / n/a | 1.179 / n/a | 31 / n/a |
| gnn_tech5 | 10.91% / n/a | 0.544 / n/a | 479 / n/a |
| gnn_tech63 | 11.16% / n/a | 0.611 / n/a | 199 / n/a |
| lstm_fund63 | 19.20% / n/a | 1.093 / n/a | 28 / n/a |
| lstm_tech5 | 2.23% / n/a | 0.190 / n/a | 618 / n/a |
| lstm_tech63 | 15.31% / n/a | 0.506 / n/a | 172 / n/a |
| miss_fund63 | 18.50% / 32.72% | 1.003 / 1.221 | 29 / 24 |
| miss_tech5 | 8.27% / 12.14% | 0.495 / 0.380 | 647 / 187 |
| miss_tech63 | 5.36% / 15.15% | 0.415 / 0.694 | 186 / 25 |
| stockmixer_fund63 | 9.28% / n/a | 0.708 / n/a | 119 / n/a |
| stockmixer_tech5 | 7.93% / n/a | 0.445 / n/a | 551 / n/a |
| stockmixer_tech63 | 11.80% / n/a | 0.800 / n/a | 195 / n/a |

## Table 2 — Sharpe grid (targets TBD from paper)

| config | Sharpe ours | Sharpe paper |
|---|---|---|
| gnn_fund63 | 1.179 | 1.030 |
| gnn_tech5 | 0.544 | 0.410 |
| gnn_tech63 | 0.611 | 0.660 |
| lstm_fund63 | 1.093 | 0.910 |
| lstm_tech5 | 0.190 | 0.330 |
| lstm_tech63 | 0.506 | 0.580 |
| miss_fund63 | 1.003 | 1.221 |
| miss_tech5 | 0.495 | 0.380 |
| miss_tech63 | 0.415 | 0.694 |
| stockmixer_fund63 | 0.708 | 1.090 |
| stockmixer_tech5 | 0.445 | 0.440 |
| stockmixer_tech63 | 0.800 | 0.730 |

## Cost sensitivity (5y means)

| config | 0bps ret / Sharpe | 5bps ret / Sharpe | 10bps ret / Sharpe | 25bps ret / Sharpe | 50bps ret / Sharpe |
|---|---|---|---|---|---|
| gnn_fund63 | 16.44% / 1.201 | 16.31% / 1.194 | 16.18% / 1.187 | 15.79% / 1.165 | 15.14% / 1.128 |
| gnn_tech5 | 17.19% / 0.812 | 15.06% / 0.723 | 12.96% / 0.634 | 6.91% / 0.365 | -2.45% / -0.081 |
| gnn_tech63 | 13.86% / 0.732 | 12.95% / 0.691 | 12.05% / 0.651 | 9.39% / 0.530 | 5.08% / 0.330 |
| lstm_fund63 | 19.60% / 1.111 | 19.47% / 1.105 | 19.33% / 1.099 | 18.94% / 1.081 | 18.29% / 1.051 |
| lstm_tech5 | 9.65% / 0.673 | 7.12% / 0.512 | 4.64% / 0.351 | -2.43% / -0.131 | -13.14% / -0.917 |
| lstm_tech63 | 17.61% / 0.581 | 16.84% / 0.556 | 16.07% / 0.531 | 13.80% / 0.456 | 10.10% / 0.331 |
| miss_fund63 | 18.89% / 1.021 | 18.76% / 1.015 | 18.63% / 1.009 | 18.23% / 0.991 | 17.57% / 0.961 |
| miss_tech5 | 16.44% / 0.885 | 13.65% / 0.755 | 10.93% / 0.625 | 3.14% / 0.236 | -8.66% / -0.403 |
| miss_tech63 | 7.80% / 0.506 | 6.98% / 0.475 | 6.17% / 0.445 | 3.76% / 0.354 | -0.15% / 0.202 |
| stockmixer_fund63 | 10.69% / 0.788 | 10.22% / 0.761 | 9.75% / 0.735 | 8.35% / 0.654 | 6.04% / 0.520 |
| stockmixer_tech5 | 14.67% / 0.791 | 12.38% / 0.676 | 10.13% / 0.560 | 3.65% / 0.214 | -6.36% / -0.358 |
| stockmixer_tech63 | 14.46% / 0.936 | 13.57% / 0.891 | 12.68% / 0.845 | 10.06% / 0.708 | 5.80% / 0.477 |

## Robustness

Moving-block bootstrap of Sharpe(miss_fund63) − Sharpe(miss_tech5): 10000 resamples, block 21d, 1250 days. Mean diff 0.401 [-0.406, 1.178], P(diff > 0) = 0.8434.

Year-level one-sided sign tests (H1: P(win) > 0.5):

- gnn_fund63__ret_gt_0: 4/5 wins, p = 0.1875
- gnn_tech5__ret_gt_0: 3/5 wins, p = 0.5000
- gnn_tech63__ret_gt_0: 4/5 wins, p = 0.1875
- lstm_fund63__ret_gt_0: 4/5 wins, p = 0.1875
- lstm_tech5__ret_gt_0: 2/5 wins, p = 0.8125
- lstm_tech63__ret_gt_0: 3/5 wins, p = 0.5000
- miss_fund63__beats__miss_tech5: 3/5 wins, p = 0.5000
- miss_fund63__ret_gt_0: 4/5 wins, p = 0.1875
- miss_tech5__ret_gt_0: 3/5 wins, p = 0.5000
- miss_tech63__ret_gt_0: 3/5 wins, p = 0.5000
- stockmixer_fund63__ret_gt_0: 4/5 wins, p = 0.1875
- stockmixer_tech5__ret_gt_0: 3/5 wins, p = 0.5000
- stockmixer_tech63__ret_gt_0: 4/5 wins, p = 0.1875

Deflated Sharpe Ratios (Bailey & Lopez de Prado; trials = #configs = 12):

| config | DSR | SR0 benchmark |
|---|---|---|
| gnn_fund63 | 1.0000 | 0.378 |
| gnn_tech5 | 1.0000 | 0.378 |
| gnn_tech63 | 1.0000 | 0.378 |
| lstm_fund63 | 1.0000 | 0.378 |
| lstm_tech5 | 0.0000 | 0.378 |
| lstm_tech63 | 1.0000 | 0.378 |
| miss_fund63 | 1.0000 | 0.378 |
| miss_tech5 | 0.8376 | 0.378 |
| miss_tech63 | 0.0001 | 0.378 |
| stockmixer_fund63 | 1.0000 | 0.378 |
| stockmixer_tech5 | 0.7579 | 0.378 |
| stockmixer_tech63 | 1.0000 | 0.378 |

