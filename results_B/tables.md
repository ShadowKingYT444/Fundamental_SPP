# Fundamental_SPP — Evaluation results

Generated (UTC): 2026-10-01T22:23:46.950459+00:00 | seed=42 | costs=15.0bps | bootstrap: 10000 resamples, block=21d

## Score-file contract

Columns: `ticker, date, sector, score` (`date` as YYYY-MM-DD). Filename: `<model>_<regime>_<year>.csv`, e.g. `miss_fund63_2021.csv`. `model` ∈ {miss, lstm, stockmixer, gnn}; `regime` ∈ {fund63, tech63, tech5} (fund63/tech63 → monthly rebalance, tech5 → weekly Monday rebalance); one test year per file. Prices come from `data/prices.parquet` (`ticker, date, adj_close`) unless `--prices` is given.

## Per-year metrics

| config | year | ann. return | Sharpe | turnover | trade events | max DD |
|---|---|---|---|---|---|---|
| gnn_fund63 | 2021 | 4.63% | 0.317 | 6.33 | 20 | -16.74% |
| gnn_fund63 | 2022 | 89.98% | 2.670 | 11.65 | 42 | -11.77% |
| gnn_fund63 | 2023 | 13.58% | 0.653 | 6.83 | 16 | -21.39% |
| gnn_fund63 | 2024 | 8.10% | 0.442 | 9.62 | 32 | -13.28% |
| gnn_fund63 | 2025 | 4.61% | 0.311 | 6.47 | 18 | -32.41% |
| gnn_tech5 | 2021 | 49.31% | 1.673 | 95.98 | 438 | -13.17% |
| gnn_tech5 | 2022 | 0.55% | 0.216 | 63.82 | 260 | -35.87% |
| gnn_tech5 | 2023 | -54.07% | -2.667 | 84.64 | 358 | -53.73% |
| gnn_tech5 | 2024 | -22.83% | -0.905 | 100.45 | 448 | -26.39% |
| gnn_tech5 | 2025 | -49.37% | -1.385 | 62.40 | 246 | -52.47% |
| gnn_tech63 | 2021 | -5.18% | -0.004 | 34.46 | 154 | -24.11% |
| gnn_tech63 | 2022 | 8.91% | 0.426 | 60.42 | 278 | -27.17% |
| gnn_tech63 | 2023 | 54.96% | 1.574 | 41.24 | 188 | -20.17% |
| gnn_tech63 | 2024 | 28.87% | 0.938 | 61.75 | 282 | -25.30% |
| gnn_tech63 | 2025 | -46.37% | -1.108 | 52.69 | 232 | -53.77% |
| lstm_fund63 | 2021 | -9.96% | -0.235 | 7.05 | 18 | -29.70% |
| lstm_fund63 | 2022 | 81.55% | 2.524 | 7.47 | 22 | -11.89% |
| lstm_fund63 | 2023 | 12.12% | 0.591 | 5.80 | 14 | -19.50% |
| lstm_fund63 | 2024 | -40.41% | -1.301 | 9.56 | 18 | -47.37% |
| lstm_fund63 | 2025 | 65.97% | 2.163 | 5.02 | 14 | -10.17% |
| lstm_tech5 | 2021 | 17.76% | 0.773 | 110.24 | 500 | -16.46% |
| lstm_tech5 | 2022 | -8.70% | 0.012 | 61.85 | 254 | -36.40% |
| lstm_tech5 | 2023 | -45.79% | -1.501 | 77.99 | 326 | -46.46% |
| lstm_tech5 | 2024 | 10.89% | 0.510 | 172.58 | 790 | -16.62% |
| lstm_tech5 | 2025 | -34.93% | -0.731 | 137.89 | 622 | -51.94% |
| lstm_tech63 | 2021 | 11.27% | 0.479 | 25.79 | 112 | -24.81% |
| lstm_tech63 | 2022 | 8.43% | 0.405 | 21.42 | 84 | -31.82% |
| lstm_tech63 | 2023 | 10.34% | 0.454 | 54.60 | 246 | -36.83% |
| lstm_tech63 | 2024 | -20.36% | -0.358 | 43.23 | 186 | -41.30% |
| lstm_tech63 | 2025 | -43.16% | -1.107 | 49.83 | 220 | -52.87% |
| miss_fund63 | 2021 | -4.65% | -0.016 | 9.66 | 34 | -35.35% |
| miss_fund63 | 2022 | 14.43% | 0.665 | 6.44 | 16 | -24.91% |
| miss_fund63 | 2023 | 22.18% | 1.068 | 8.91 | 28 | -9.36% |
| miss_fund63 | 2024 | -8.08% | -0.329 | 9.68 | 32 | -25.83% |
| miss_fund63 | 2025 | 131.47% | 3.360 | 7.28 | 24 | -9.90% |
| miss_tech5 | 2021 | 5.05% | 0.322 | 98.70 | 442 | -24.99% |
| miss_tech5 | 2022 | -21.50% | -0.679 | 121.64 | 544 | -32.25% |
| miss_tech5 | 2023 | -48.70% | -2.344 | 130.62 | 578 | -46.36% |
| miss_tech5 | 2024 | 47.29% | 1.639 | 115.38 | 526 | -11.36% |
| miss_tech5 | 2025 | -30.22% | -1.089 | 118.27 | 528 | -33.87% |
| miss_tech63 | 2021 | -14.72% | -0.447 | 40.54 | 174 | -34.41% |
| miss_tech63 | 2022 | -13.86% | -0.109 | 47.82 | 218 | -43.84% |
| miss_tech63 | 2023 | 16.07% | 0.671 | 54.12 | 248 | -24.50% |
| miss_tech63 | 2024 | -27.27% | -0.716 | 35.91 | 156 | -33.19% |
| miss_tech63 | 2025 | -21.74% | -0.537 | 48.62 | 218 | -34.89% |
| stockmixer_fund63 | 2021 | 3.95% | 0.291 | 23.46 | 102 | -15.90% |
| stockmixer_fund63 | 2022 | 30.02% | 1.376 | 19.71 | 88 | -10.21% |
| stockmixer_fund63 | 2023 | 1.37% | 0.170 | 26.65 | 118 | -17.78% |
| stockmixer_fund63 | 2024 | -15.34% | -0.621 | 17.68 | 72 | -24.18% |
| stockmixer_fund63 | 2025 | -10.10% | -0.219 | 26.61 | 116 | -21.80% |
| stockmixer_tech5 | 2021 | 5.83% | 0.363 | 86.40 | 382 | -20.00% |
| stockmixer_tech5 | 2022 | -22.28% | -1.070 | 112.12 | 504 | -29.22% |
| stockmixer_tech5 | 2023 | -39.03% | -2.167 | 89.64 | 394 | -37.67% |
| stockmixer_tech5 | 2024 | -6.12% | -0.239 | 103.26 | 464 | -24.83% |
| stockmixer_tech5 | 2025 | 7.64% | 0.449 | 96.17 | 428 | -15.19% |
| stockmixer_tech63 | 2021 | -14.72% | -0.756 | 58.35 | 262 | -21.59% |
| stockmixer_tech63 | 2022 | -18.32% | -0.833 | 40.35 | 180 | -23.12% |
| stockmixer_tech63 | 2023 | 3.15% | 0.254 | 50.55 | 232 | -19.96% |
| stockmixer_tech63 | 2024 | 6.97% | 0.410 | 38.99 | 178 | -22.63% |
| stockmixer_tech63 | 2025 | 9.64% | 0.519 | 46.28 | 212 | -20.31% |

## 5-year means vs paper targets (Table 1, MISS)

| config | ann. return (ours / paper) | Sharpe (ours / paper) | trade events/yr (ours / paper) |
|---|---|---|---|
| gnn_fund63 | 24.18% / n/a | 0.878 / n/a | 26 / n/a |
| gnn_tech5 | -15.28% / n/a | -0.614 / n/a | 350 / n/a |
| gnn_tech63 | 8.24% / n/a | 0.365 / n/a | 227 / n/a |
| lstm_fund63 | 21.85% / n/a | 0.748 / n/a | 17 / n/a |
| lstm_tech5 | -12.15% / n/a | -0.187 / n/a | 498 / n/a |
| lstm_tech63 | -6.70% / n/a | -0.026 / n/a | 170 / n/a |
| miss_fund63 | 31.07% / 32.72% | 0.950 / 1.221 | 27 / 24 |
| miss_tech5 | -9.62% / 12.14% | -0.430 / 0.380 | 524 / 187 |
| miss_tech63 | -12.30% / 15.15% | -0.228 / 0.694 | 203 / 25 |
| stockmixer_fund63 | 1.98% / n/a | 0.199 / n/a | 99 / n/a |
| stockmixer_tech5 | -10.79% / n/a | -0.533 / n/a | 434 / n/a |
| stockmixer_tech63 | -2.66% / n/a | -0.081 / n/a | 213 / n/a |

## Table 2 — Sharpe grid (targets TBD from paper)

| config | Sharpe ours | Sharpe paper |
|---|---|---|
| gnn_fund63 | 0.878 | 1.030 |
| gnn_tech5 | -0.614 | 0.410 |
| gnn_tech63 | 0.365 | 0.660 |
| lstm_fund63 | 0.748 | 0.910 |
| lstm_tech5 | -0.187 | 0.330 |
| lstm_tech63 | -0.026 | 0.580 |
| miss_fund63 | 0.950 | 1.221 |
| miss_tech5 | -0.430 | 0.380 |
| miss_tech63 | -0.228 | 0.694 |
| stockmixer_fund63 | 0.199 | 1.090 |
| stockmixer_tech5 | -0.533 | 0.440 |
| stockmixer_tech63 | -0.081 | 0.730 |

## Cost sensitivity (5y means)

| config | 0bps ret / Sharpe | 5bps ret / Sharpe | 10bps ret / Sharpe | 25bps ret / Sharpe | 50bps ret / Sharpe |
|---|---|---|---|---|---|
| gnn_fund63 | 25.27% / 0.911 | 24.91% / 0.900 | 24.54% / 0.889 | 23.46% / 0.856 | 21.66% / 0.801 |
| gnn_tech5 | -4.35% / -0.215 | -8.14% / -0.348 | -11.78% / -0.481 | -21.88% / -0.878 | -36.25% / -1.521 |
| gnn_tech63 | 16.02% / 0.571 | 13.37% / 0.503 | 10.78% / 0.434 | 3.30% / 0.228 | -8.20% / -0.111 |
| lstm_fund63 | 22.59% / 0.772 | 22.34% / 0.764 | 22.10% / 0.756 | 21.37% / 0.733 | 20.15% / 0.693 |
| lstm_tech5 | 4.21% / 0.316 | -1.58% / 0.148 | -7.03% / -0.020 | -21.52% / -0.521 | -40.53% / -1.323 |
| lstm_tech63 | -1.71% / 0.114 | -3.40% / 0.067 | -5.06% / 0.021 | -9.89% / -0.118 | -17.44% / -0.345 |
| miss_fund63 | 32.09% / 0.986 | 31.75% / 0.974 | 31.41% / 0.962 | 30.39% / 0.926 | 28.72% / 0.866 |
| miss_tech5 | 6.83% / 0.197 | 1.05% / -0.013 | -4.42% / -0.222 | -19.20% / -0.841 | -39.12% / -1.813 |
| miss_tech63 | -6.40% / -0.036 | -8.40% / -0.100 | -10.37% / -0.164 | -16.07% / -0.355 | -24.90% / -0.664 |
| stockmixer_fund63 | 5.05% / 0.331 | 4.02% / 0.287 | 3.00% / 0.243 | -0.03% / 0.112 | -4.92% / -0.106 |
| stockmixer_tech5 | 2.70% / 0.133 | -2.00% / -0.090 | -6.49% / -0.312 | -18.84% / -0.966 | -36.06% / -1.968 |
| stockmixer_tech63 | 3.90% / 0.232 | 1.67% / 0.127 | -0.51% / 0.023 | -6.83% / -0.286 | -16.64% / -0.764 |

## Robustness

Moving-block bootstrap of Sharpe(miss_fund63) − Sharpe(miss_tech5): 10000 resamples, block 21d, 1250 days. Mean diff 1.462 [0.189, 2.733], P(diff > 0) = 0.9883.

Year-level one-sided sign tests (H1: P(win) > 0.5):

- gnn_fund63__ret_gt_0: 5/5 wins, p = 0.0312
- gnn_tech5__ret_gt_0: 2/5 wins, p = 0.8125
- gnn_tech63__ret_gt_0: 3/5 wins, p = 0.5000
- lstm_fund63__ret_gt_0: 3/5 wins, p = 0.5000
- lstm_tech5__ret_gt_0: 2/5 wins, p = 0.8125
- lstm_tech63__ret_gt_0: 3/5 wins, p = 0.5000
- miss_fund63__beats__miss_tech5: 3/5 wins, p = 0.5000
- miss_fund63__ret_gt_0: 3/5 wins, p = 0.5000
- miss_tech5__ret_gt_0: 2/5 wins, p = 0.8125
- miss_tech63__ret_gt_0: 1/5 wins, p = 0.9688
- stockmixer_fund63__ret_gt_0: 3/5 wins, p = 0.5000
- stockmixer_tech5__ret_gt_0: 2/5 wins, p = 0.8125
- stockmixer_tech63__ret_gt_0: 3/5 wins, p = 0.5000

Deflated Sharpe Ratios (Bailey & Lopez de Prado; trials = #configs = 12):

| config | DSR | SR0 benchmark |
|---|---|---|
| gnn_fund63 | 0.0664 | 0.877 |
| gnn_tech5 | 0.0000 | 0.877 |
| gnn_tech63 | 0.0000 | 0.877 |
| lstm_fund63 | 0.0000 | 0.877 |
| lstm_tech5 | 0.0000 | 0.877 |
| lstm_tech63 | 0.0000 | 0.877 |
| miss_fund63 | 0.9937 | 0.877 |
| miss_tech5 | 0.0000 | 0.877 |
| miss_tech63 | 0.0000 | 0.877 |
| stockmixer_fund63 | 0.0000 | 0.877 |
| stockmixer_tech5 | 0.0000 | 0.877 |
| stockmixer_tech63 | 0.0000 | 0.877 |

