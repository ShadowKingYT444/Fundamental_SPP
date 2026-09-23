"""Generate the paper's figures and tables from reproduced results.

Inputs (no invented data):
  results/metrics.json        -- from src/evaluate.py
  results/equity/<cfg>.csv    -- daily net returns per config (date,ret)

Outputs: results/figures/fig{1..4}.{png,pdf}

Style follows the figures4papers house conventions (see src/plot_style.py),
distilled from https://github.com/ChenLiu-1996/figures4papers.

  python3 src/make_figures.py            -- build all figures from results/
  python3 src/make_figures.py --smoke    -- render with synthetic data into
                                          /tmp/fig_smoke/ (template check only)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot_style import (  # noqa: E402
    PALETTE, REGIME_COLORS, ARCH_COLORS, ARCH_LABELS, REGIME_LABELS,
    apply_style, save_pub,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = PROJECT_ROOT / "results" / "figures"
ARCH_ORDER = ["miss", "stockmixer", "gnn", "lstm"]
REGIME_ORDER = ["fund63", "tech63", "tech5"]

apply_style()


# ------------------------------------------------------------- Figure 1 ----
def fig1_equity(metrics: dict, outdir: Path, smoke: bool = False) -> None:
    """Cumulative NAV for MISS x {fund63, tech63, tech5}, 2021-2025."""
    fig = plt.figure(figsize=(14, 6.5))
    ax = fig.add_subplot(1, 1, 1)
    for regime in REGIME_ORDER:
        if smoke:
            rng = np.random.default_rng(hash(regime) % 2**32)
            nav = np.cumprod(1 + rng.normal(0.0004, 0.01, 1250))
            dates = pd.date_range("2021-01-04", periods=1250, freq="B")
        else:
            df = pd.read_csv(PROJECT_ROOT / "results" / "equity"
                             / f"miss_{regime}.csv", parse_dates=["date"])
            df = df.dropna(subset=["ret"])
            nav = np.cumprod(1.0 + df["ret"].to_numpy(dtype=float))
            dates = df["date"]
        ax.plot(dates, nav, lw=2.5, color=REGIME_COLORS[regime],
                label=f"{REGIME_LABELS[regime]}")
    ax.set_ylabel("Cumulative net asset value")
    ax.set_ylim(bottom=0.6)
    ax.legend(loc="upper left")
    ax.set_title("MISS equity curves, 2021-2025 (net of 15 bps one-way costs)"
                 if not smoke else "Fig 1 template (smoke)")
    save_pub(fig, outdir / "fig1_equity")


# ------------------------------------------------------------- Figure 2 ----
def fig2_arch_bars(metrics: dict, outdir: Path, smoke: bool = False) -> None:
    """Mean OOS Sharpe by architecture x regime; one panel per regime."""
    data, errs = {}, {}
    if smoke:
        data = {
            "fund63": [1.0, 0.9, 0.8, 0.7],
            "tech63": [0.6, 0.65, 0.55, 0.5],
            "tech5": [0.35, 0.4, 0.38, 0.3],
        }
        errs = {r: [0.15] * 4 for r in REGIME_ORDER}
    else:
        table2 = metrics["table2_sharpe"]
        for regime in REGIME_ORDER:
            vals, sds = [], []
            for arch in ARCH_ORDER:
                entry = table2.get(f"{arch}_{regime}") or {}
                v = entry.get("ours_sharpe")
                if v is None:
                    v = entry.get("mean_sharpe", np.nan)
                vals.append(v if v is not None else np.nan)
                sd = entry.get("sharpe_std")
                sds.append(sd if sd is not None else 0.0)
            data[regime], errs[regime] = vals, sds

    fig = plt.figure(figsize=(16, 5.5))
    for i, regime in enumerate(REGIME_ORDER):
        ax = fig.add_subplot(1, 4, i + 1)
        x = np.arange(len(ARCH_ORDER))
        colors = [ARCH_COLORS[a] for a in ARCH_ORDER]
        ax.bar(x, data[regime], yerr=errs[regime], capsize=5,
               color=colors, edgecolor="black", linewidth=1.5,
               ecolor=PALETTE["ink"])
        ax.set_xticks([])
        ax.set_ylabel("Mean OOS Sharpe" if i == 0 else "")
        ax.set_title(REGIME_LABELS[regime])
    ax = fig.add_subplot(1, 4, 4)  # dedicated legend panel
    handles = [plt.Rectangle((0, 0), 1, 1, fc=ARCH_COLORS[a], ec="black")
               for a in ARCH_ORDER]
    ax.legend(handles, [ARCH_LABELS[a] for a in ARCH_ORDER],
              loc="center left", fontsize=16)
    ax.set_axis_off()
    save_pub(fig, outdir / "fig2_arch_sharpe")


# ------------------------------------------------------------- Figure 3 ----
def fig3_cost_sensitivity(metrics: dict, outdir: Path, smoke: bool = False) -> None:
    """Sharpe and annual return vs one-way cost for MISS regimes."""
    fig = plt.figure(figsize=(14, 5.5))
    for p, (stat, ylabel) in enumerate(
            [("sharpe_mean", "Mean Sharpe"),
             ("annual_return_mean", "Mean annual return")]):
        ax = fig.add_subplot(1, 2, p + 1)
        for regime in REGIME_ORDER:
            if smoke:
                costs = np.array([0, 5, 10, 25, 50])
                base = {"fund63": 1.2, "tech63": 0.7, "tech5": 0.4}[regime]
                vals = base - costs * (0.02 if regime == "tech5" else 0.006)
            else:
                cs = metrics["cost_sensitivity"].get(f"miss_{regime}", {})
                costs = np.array(sorted(int(c) for c in cs))
                vals = np.array([cs[str(c)][stat] for c in costs])
            ax.plot(costs, vals, lw=2.5, marker="o",
                    color=REGIME_COLORS[regime], label=REGIME_LABELS[regime])
        ax.axvline(15, color=PALETTE["neutral_dark"], ls="--", lw=1.5)
        ymin, ymax = ax.get_ylim()
        ax.text(15.6, ymin + 0.03 * (ymax - ymin), "15 bps",
                color=PALETTE["neutral_dark"], fontsize=12)
        ax.set_xlabel("One-way transaction cost (bps)")
        ax.set_ylabel(ylabel)
        ax.set_title("Sharpe vs cost" if p == 0 else "Return vs cost")
    ax.legend()
    save_pub(fig, outdir / "fig3_cost_sensitivity")


# ------------------------------------------------------------- Figure 4 ----
def fig4_bootstrap(metrics: dict, outdir: Path, smoke: bool = False) -> None:
    """Moving-block bootstrap of Sharpe(fund63) - Sharpe(tech5), MISS."""
    if smoke:
        rng = np.random.default_rng(0)
        diffs = rng.normal(0.45, 0.35, 1000)
        observed = 0.45
    else:
        b = metrics["bootstrap_sharpe_diff"]["miss_fund63__minus__miss_tech5"]
        diffs = np.asarray(b["diffs"], dtype=float)
        observed = float(b["observed"])

    fig = plt.figure(figsize=(10, 5.5))
    ax = fig.add_subplot(1, 1, 1)
    ax.hist(diffs, bins=40, color=PALETTE["blue_light"],
            edgecolor=PALETTE["blue_main"], linewidth=1.2)
    ax.axvline(0, color=PALETTE["red_strong"], lw=2.5, ls="--",
               label="No difference")
    ax.axvline(observed, color=PALETTE["blue_main"], lw=2.5,
               label=f"Observed diff = {observed:.2f}")
    ax.set_xlabel("Sharpe difference (fund63 − tech5), MISS")
    ax.set_ylabel("Bootstrap count")
    ax.legend()
    save_pub(fig, outdir / "fig4_bootstrap")


# ---------------------------------------------------------------- main -----
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true",
                    help="render templates with synthetic data into /tmp/fig_smoke")
    args = ap.parse_args()
    outdir = Path("/tmp/fig_smoke") if args.smoke else FIG_DIR
    outdir.mkdir(parents=True, exist_ok=True)

    metrics: dict = {}
    if not args.smoke:
        mpath = PROJECT_ROOT / "results" / "metrics.json"
        if not mpath.exists():
            sys.exit("results/metrics.json missing; run src/evaluate.py first")
        metrics = json.loads(mpath.read_text())

    fig1_equity(metrics, outdir, args.smoke)
    fig2_arch_bars(metrics, outdir, args.smoke)
    fig3_cost_sensitivity(metrics, outdir, args.smoke)
    fig4_bootstrap(metrics, outdir, args.smoke)
    print("wrote figures to", outdir)


if __name__ == "__main__":
    main()
