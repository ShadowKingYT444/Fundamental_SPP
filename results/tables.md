# Fundamental_SPP — Evaluation results

Generated (UTC): 2026-10-01T15:02:36.720598+00:00 | seed=42 | costs=15.0bps | bootstrap: 10000 resamples, block=21d

## Score-file contract

Columns: `ticker, date, sector, score` (`date` as YYYY-MM-DD). Filename: `<model>_<regime>_<year>.csv`, e.g. `miss_fund63_2021.csv`. `model` ∈ {miss, lstm, stockmixer, gnn}; `regime` ∈ {fund63, tech63, tech5} (fund63/tech63 → monthly rebalance, tech5 → weekly Monday rebalance); one test year per file. Prices come from `data/prices.parquet` (`ticker, date, adj_close`) unless `--prices` is given.

## Per-year metrics

| config | year | ann. return | Sharpe | turnover | trade events | max DD |
|---|---|---|---|---|---|---|
| gnn_fund63 | 2021 | -0.14% | -0.050 | 2.46 | 182 | -1.91% |
| gnn_fund63 | 2022 | 5.79% | 1.978 | 2.81 | 206 | -2.35% |
| gnn_fund63 | 2023 | -1.61% | -0.637 | 2.91 | 201 | -2.91% |
| gnn_fund63 | 2024 | 2.92% | 0.822 | 2.74 | 197 | -2.47% |
| gnn_fund63 | 2025 | -3.09% | -0.614 | 2.58 | 182 | -8.94% |
| gnn_tech5 | 2021 | -0.19% | -0.030 | 26.66 | 5127 | -5.24% |
| gnn_tech5 | 2022 | -6.76% | -1.358 | 17.63 | 3203 | -7.36% |
| gnn_tech5 | 2023 | -4.87% | -1.611 | 21.12 | 4279 | -4.87% |
| gnn_tech5 | 2024 | -4.56% | -1.742 | 27.26 | 5832 | -4.05% |
| gnn_tech5 | 2025 | -5.65% | -0.904 | 22.85 | 4865 | -8.82% |
| gnn_tech63 | 2021 | -0.89% | -0.179 | 9.71 | 1917 | -3.81% |
| gnn_tech63 | 2022 | -2.25% | -0.529 | 15.48 | 3143 | -4.85% |
| gnn_tech63 | 2023 | 0.64% | 0.160 | 9.39 | 1903 | -6.86% |
| gnn_tech63 | 2024 | -1.26% | -0.334 | 15.27 | 3166 | -5.43% |
| gnn_tech63 | 2025 | -7.75% | -1.344 | 12.47 | 2690 | -9.50% |
| lstm_fund63 | 2021 | -1.75% | -0.676 | 2.44 | 167 | -4.03% |
| lstm_fund63 | 2022 | 7.21% | 2.539 | 1.94 | 149 | -1.46% |
| lstm_fund63 | 2023 | -3.27% | -1.244 | 2.14 | 146 | -4.69% |
| lstm_fund63 | 2024 | 0.34% | 0.131 | 2.01 | 153 | -2.22% |
| lstm_fund63 | 2025 | 0.28% | 0.101 | 1.76 | 123 | -3.97% |
| lstm_tech5 | 2021 | -3.67% | -1.103 | 28.57 | 5607 | -5.11% |
| lstm_tech5 | 2022 | -5.88% | -1.193 | 24.20 | 4711 | -6.88% |
| lstm_tech5 | 2023 | -8.12% | -1.911 | 26.64 | 5505 | -7.97% |
| lstm_tech5 | 2024 | -4.52% | -1.498 | 48.16 | 10731 | -5.87% |
| lstm_tech5 | 2025 | -6.54% | -0.940 | 37.46 | 8191 | -9.89% |
| lstm_tech63 | 2021 | 3.69% | 0.707 | 7.37 | 1432 | -4.58% |
| lstm_tech63 | 2022 | -3.90% | -0.911 | 9.77 | 2020 | -7.07% |
| lstm_tech63 | 2023 | 1.87% | 0.446 | 13.47 | 2775 | -5.35% |
| lstm_tech63 | 2024 | -3.21% | -0.696 | 11.29 | 2371 | -5.30% |
| lstm_tech63 | 2025 | -0.01% | 0.019 | 11.71 | 2562 | -4.44% |
| miss_fund63 | 2021 | -0.50% | -0.137 | 2.62 | 190 | -3.99% |
| miss_fund63 | 2022 | 7.67% | 2.636 | 2.45 | 182 | -1.70% |
| miss_fund63 | 2023 | 2.70% | 1.458 | 2.41 | 188 | -1.06% |
| miss_fund63 | 2024 | -1.49% | -0.639 | 2.49 | 210 | -2.75% |
| miss_fund63 | 2025 | 3.93% | 1.355 | 2.50 | 204 | -1.39% |
| miss_tech5 | 2021 | -2.30% | -0.745 | 25.69 | 4823 | -3.45% |
| miss_tech5 | 2022 | -2.12% | -0.461 | 32.76 | 6864 | -4.95% |
| miss_tech5 | 2023 | -6.19% | -2.525 | 34.71 | 7401 | -6.21% |
| miss_tech5 | 2024 | -3.57% | -1.485 | 28.07 | 6110 | -4.24% |
| miss_tech5 | 2025 | -3.84% | -0.589 | 25.89 | 5722 | -6.21% |
| miss_tech63 | 2021 | -3.83% | -1.012 | 9.76 | 1899 | -5.21% |
| miss_tech63 | 2022 | -2.70% | -0.722 | 13.02 | 2627 | -5.40% |
| miss_tech63 | 2023 | 2.03% | 0.766 | 13.13 | 2755 | -1.83% |
| miss_tech63 | 2024 | -0.30% | -0.066 | 8.46 | 1658 | -3.49% |
| miss_tech63 | 2025 | -3.10% | -0.516 | 10.96 | 2337 | -5.49% |
| stockmixer_fund63 | 2021 | -3.20% | -1.524 | 5.38 | 521 | -3.80% |
| stockmixer_fund63 | 2022 | 1.31% | 0.651 | 5.31 | 593 | -1.59% |
| stockmixer_fund63 | 2023 | -1.21% | -0.689 | 6.01 | 702 | -2.52% |
| stockmixer_fund63 | 2024 | -3.03% | -1.354 | 5.24 | 557 | -3.76% |
| stockmixer_fund63 | 2025 | -5.56% | -2.114 | 6.43 | 705 | -6.72% |
| stockmixer_tech5 | 2021 | -2.01% | -0.925 | 24.27 | 4780 | -3.23% |
| stockmixer_tech5 | 2022 | -5.00% | -2.538 | 32.19 | 6672 | -5.80% |
| stockmixer_tech5 | 2023 | -6.15% | -3.000 | 25.15 | 5201 | -6.20% |
| stockmixer_tech5 | 2024 | -3.73% | -2.072 | 28.90 | 6142 | -3.86% |
| stockmixer_tech5 | 2025 | -0.53% | -0.243 | 27.70 | 5871 | -1.55% |
| stockmixer_tech63 | 2021 | -3.25% | -1.762 | 13.49 | 2622 | -4.04% |
| stockmixer_tech63 | 2022 | -1.04% | -0.478 | 9.90 | 2010 | -3.29% |
| stockmixer_tech63 | 2023 | -1.66% | -0.900 | 12.40 | 2566 | -3.23% |
| stockmixer_tech63 | 2024 | 3.60% | 1.552 | 9.22 | 1956 | -1.60% |
| stockmixer_tech63 | 2025 | -0.98% | -0.521 | 11.68 | 2452 | -2.24% |

## 5-year means vs paper targets (Table 1, MISS)

| config | ann. return (ours / paper) | Sharpe (ours / paper) | trade events/yr (ours / paper) |
|---|---|---|---|
| gnn_fund63 | 0.78% / n/a | 0.300 / n/a | 194 / n/a |
| gnn_tech5 | -4.40% / n/a | -1.129 / n/a | 4661 / n/a |
| gnn_tech63 | -2.30% / n/a | -0.445 / n/a | 2564 / n/a |
| lstm_fund63 | 0.56% / n/a | 0.170 / n/a | 148 / n/a |
| lstm_tech5 | -5.75% / n/a | -1.329 / n/a | 6949 / n/a |
| lstm_tech63 | -0.31% / n/a | -0.087 / n/a | 2232 / n/a |
| miss_fund63 | 2.46% / 32.72% | 0.935 / 1.221 | 195 / 24 |
| miss_tech5 | -3.61% / 12.14% | -1.161 / 0.380 | 6184 / 187 |
| miss_tech63 | -1.58% / 15.15% | -0.310 / 0.694 | 2255 / 25 |
| stockmixer_fund63 | -2.34% / n/a | -1.006 / n/a | 616 / n/a |
| stockmixer_tech5 | -3.49% / n/a | -1.756 / n/a | 5733 / n/a |
| stockmixer_tech63 | -0.67% / n/a | -0.422 / n/a | 2321 / n/a |

## Table 2 — Sharpe grid (targets TBD from paper)

| config | Sharpe ours | Sharpe paper |
|---|---|---|
| gnn_fund63 | 0.300 | 1.030 |
| gnn_tech5 | -1.129 | 0.410 |
| gnn_tech63 | -0.445 | 0.660 |
| lstm_fund63 | 0.170 | 0.910 |
| lstm_tech5 | -1.329 | 0.330 |
| lstm_tech63 | -0.087 | 0.580 |
| miss_fund63 | 0.935 | 1.221 |
| miss_tech5 | -1.161 | 0.380 |
| miss_tech63 | -0.310 | 0.694 |
| stockmixer_fund63 | -1.006 | 1.090 |
| stockmixer_tech5 | -1.756 | 0.440 |
| stockmixer_tech63 | -0.422 | 0.730 |

## Cost sensitivity (5y means)

| config | 0bps ret / Sharpe | 5bps ret / Sharpe | 10bps ret / Sharpe | 25bps ret / Sharpe | 50bps ret / Sharpe |
|---|---|---|---|---|---|
| gnn_fund63 | 1.08% / 0.402 | 0.98% / 0.368 | 0.88% / 0.334 | 0.57% / 0.231 | 0.06% / 0.061 |
| gnn_tech5 | -1.17% / -0.229 | -2.26% / -0.530 | -3.34% / -0.831 | -6.50% / -1.709 | -11.55% / -2.999 |
| gnn_tech63 | -0.61% / -0.049 | -1.18% / -0.183 | -1.74% / -0.315 | -3.42% / -0.696 | -6.15% / -1.250 |
| lstm_fund63 | 0.77% / 0.245 | 0.70% / 0.220 | 0.63% / 0.195 | 0.42% / 0.120 | 0.07% / -0.005 |
| lstm_tech5 | -1.10% / -0.138 | -2.67% / -0.541 | -4.22% / -0.940 | -8.72% / -2.059 | -15.74% / -3.539 |
| lstm_tech63 | 1.15% / 0.238 | 0.66% / 0.130 | 0.17% / 0.021 | -1.27% / -0.300 | -3.65% / -0.801 |
| miss_fund63 | 2.74% / 1.045 | 2.65% / 1.008 | 2.56% / 0.971 | 2.28% / 0.861 | 1.81% / 0.676 |
| miss_tech5 | 0.59% / 0.150 | -0.83% / -0.295 | -2.23% / -0.734 | -6.31% / -1.959 | -12.74% / -3.533 |
| miss_tech63 | -0.08% / 0.109 | -0.59% / -0.030 | -1.08% / -0.170 | -2.57% / -0.583 | -5.01% / -1.196 |
| stockmixer_fund63 | -1.60% / -0.659 | -1.85% / -0.776 | -2.09% / -0.892 | -2.82% / -1.227 | -4.04% / -1.723 |
| stockmixer_tech5 | 0.45% / 0.220 | -0.88% / -0.458 | -2.19% / -1.123 | -6.03% / -2.878 | -12.10% / -4.739 |
| stockmixer_tech63 | 0.88% / 0.368 | 0.36% / 0.100 | -0.15% / -0.165 | -1.69% / -0.895 | -4.20% / -1.770 |

## Robustness

Moving-block bootstrap of Sharpe(miss_fund63) − Sharpe(miss_tech5): 10000 resamples, block 21d, 1250 days. Mean diff 1.815 [0.674, 2.983], P(diff > 0) = 0.9993.

Year-level one-sided sign tests (H1: P(win) > 0.5):

- gnn_fund63__ret_gt_0: 2/5 wins, p = 0.8125
- gnn_tech5__ret_gt_0: 0/5 wins, p = 1.0000
- gnn_tech63__ret_gt_0: 1/5 wins, p = 0.9688
- lstm_fund63__ret_gt_0: 3/5 wins, p = 0.5000
- lstm_tech5__ret_gt_0: 0/5 wins, p = 1.0000
- lstm_tech63__ret_gt_0: 2/5 wins, p = 0.8125
- miss_fund63__beats__miss_tech5: 5/5 wins, p = 0.0312
- miss_fund63__ret_gt_0: 3/5 wins, p = 0.5000
- miss_tech5__ret_gt_0: 0/5 wins, p = 1.0000
- miss_tech63__ret_gt_0: 1/5 wins, p = 0.9688
- stockmixer_fund63__ret_gt_0: 1/5 wins, p = 0.9688
- stockmixer_tech5__ret_gt_0: 0/5 wins, p = 1.0000
- stockmixer_tech63__ret_gt_0: 1/5 wins, p = 0.9688

Deflated Sharpe Ratios (Bailey & Lopez de Prado; trials = #configs = 12):

| config | DSR | SR0 benchmark |
|---|---|---|
| gnn_fund63 | 0.0000 | 1.232 |
| gnn_tech5 | 0.0000 | 1.232 |
| gnn_tech63 | 0.0000 | 1.232 |
| lstm_fund63 | 0.0000 | 1.232 |
| lstm_tech5 | 0.0000 | 1.232 |
| lstm_tech63 | 0.0000 | 1.232 |
| miss_fund63 | 0.0000 | 1.232 |
| miss_tech5 | 0.0000 | 1.232 |
| miss_tech63 | 0.0000 | 1.232 |
| stockmixer_fund63 | 0.0000 | 1.232 |
| stockmixer_tech5 | 0.0000 | 1.232 |
| stockmixer_tech63 | 0.0000 | 1.232 |

