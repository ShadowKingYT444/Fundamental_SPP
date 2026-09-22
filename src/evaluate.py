"""Evaluation + robustness diagnostics for the Fundamental_SPP reproduction.

Reads per-run score CSVs and produces portfolio metrics, paper comparisons,
cost-sensitivity analysis, and robustness diagnostics.

Score-file contract:
    Columns : ticker, date (YYYY-MM-DD), sector (GICS sector), score (float)
    Filename: <model>_<regime>_<year>.csv   (e.g. miss_fund63_2021.csv)
    model   : miss | lstm | stockmixer | gnn
    regime  : fund63 | tech63 | tech5        (drives rebalance frequency)
    year    : one test year per file (2021..2025)

CLI:
    python src/evaluate.py --scores_dir <dir> --out results/
        [--prices data/prices.parquet] [--costs 15] [--n_boot 10000]
        [--block 21] [--seed 42]

Outputs:
    <out>/metrics.json   all computed metrics (JSON-serializable)
    <out>/tables.md      human-readable comparison tables

Run from the project root (~/workspace/fundamental_spp/) so relative paths
resolve; absolute paths also work.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from backtest import run_backtest  # noqa: E402

log = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[1]

TRADING_DAYS = 252
EULER_GAMMA = 0.5772156649015329

# Paper targets (Ding 2026, Table 1, MISS rows) given in the task brief.
PAPER_TARGETS = {
    ("miss", "fund63"): {"annual_return": 0.3272, "sharpe": 1.221, "trade_events": 24},
    ("miss", "tech63"): {"annual_return": 0.1515, "sharpe": 0.694, "trade_events": 25},
    ("miss", "tech5"): {"annual_return": 0.1214, "sharpe": 0.380, "trade_events": 187},
}
# Table 2 = Sharpe grid (models x regimes), Ding 2026 Table 2.
TABLE2_TARGETS: dict[tuple[str, str], float] = {
    ("miss", "fund63"): 1.221, ("miss", "tech63"): 0.694, ("miss", "tech5"): 0.380,
    ("stockmixer", "fund63"): 1.090, ("stockmixer", "tech63"): 0.730,
    ("stockmixer", "tech5"): 0.440,
    ("gnn", "fund63"): 1.030, ("gnn", "tech63"): 0.660, ("gnn", "tech5"): 0.410,
    ("lstm", "fund63"): 0.910, ("lstm", "tech63"): 0.580, ("lstm", "tech5"): 0.330,
}

COST_LEVELS = [0, 5, 10, 25, 50]


# ---------------------------------------------------------------- metrics ---
def sharpe_ann(rets: pd.Series) -> float:
    r = np.asarray(rets, dtype=float)
    r = r[np.isfinite(r)]
    if len(r) < 2:
        return float("nan")
    sd = r.std(ddof=1)
    if sd <= 0 or not np.isfinite(sd):
        return float("nan")
    return float(r.mean() / sd * math.sqrt(TRADING_DAYS))


def max_drawdown(rets: pd.Series) -> float:
    r = np.asarray(rets, dtype=float)
    r = r[np.isfinite(r)]
    if len(r) == 0:
        return float("nan")
    nav = np.cumprod(1.0 + r)
    peak = np.maximum.accumulate(nav)
    dd = nav / peak - 1.0
    return float(dd.min())


def annual_return(rets: pd.Series) -> float:
    r = np.asarray(rets, dtype=float)
    r = r[np.isfinite(r)]
    if len(r) == 0:
        return float("nan")
    return float(np.prod(1.0 + r) - 1.0)


def year_metrics(result) -> dict:
    return {
        "annual_return": annual_return(result.returns),
        "sharpe": sharpe_ann(result.returns),
        "turnover": result.turnover,
        "trade_events": result.trade_events,
        "n_entries": result.n_entries,
        "n_exits": result.n_exits,
        "max_drawdown": max_drawdown(result.returns),
        "n_rebalances": result.n_rebalances,
        "traded_notional": result.traded_notional,
        "costs_paid": result.costs_paid,
        "n_days": int(len(result.returns)),
    }


# ---------------------------------------------------------------- loading ---
def parse_score_filename(path: Path) -> tuple[str, str, int]:
    stem = path.stem
    parts = stem.split("_")
    if len(parts) < 3:
        raise ValueError(f"filename {path.name!r} does not match <model>_<regime>_<year>.csv")
    year = int(parts[-1])
    regime = parts[-2]
    model = "_".join(parts[:-2])
    if regime not in ("fund63", "tech63", "tech5"):
        raise ValueError(f"unknown regime {regime!r} in {path.name!r}")
    return model, regime, year


def load_scores(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    req = {"ticker", "date", "sector", "score"}
    missing = req - set(df.columns)
    if missing:
        raise ValueError(f"{path.name}: missing columns {sorted(missing)}")
    return df


# ------------------------------------------------------- robustness stats ---
def block_bootstrap_sharpe_diff(a: pd.Series, b: pd.Series,
                                n_boot: int = 10_000, block: int = 21,
                                seed: int = 42) -> dict:
    """P(SR_a - SR_b > 0) via paired circular moving-block bootstrap.

    a, b: daily net returns (same calendar). Blocks are drawn jointly so the
    paired correlation structure is preserved.
    """
    idx = a.index.intersection(b.index).sort_values()
    ra = a.reindex(idx).to_numpy(dtype=float)
    rb = b.reindex(idx).to_numpy(dtype=float)
    ok = np.isfinite(ra) & np.isfinite(rb)
    ra, rb = ra[ok], rb[ok]
    n = len(ra)
    if n < block:
        raise ValueError("series shorter than block length")
    rng = np.random.default_rng(seed)
    n_blocks = math.ceil(n / block)
    diffs = np.empty(n_boot)
    ar = np.arange(block)
    for i in range(n_boot):
        starts = rng.integers(0, n, size=n_blocks)
        sel = ((starts[:, None] + ar[None, :]) % n).ravel()[:n]
        sa = sharpe_ann(pd.Series(ra[sel]))
        sb = sharpe_ann(pd.Series(rb[sel]))
        diffs[i] = sa - sb
    diffs = diffs[np.isfinite(diffs)]
    return {
        "n_boot": int(n_boot),
        "block": int(block),
        "n_days": int(n),
        "mean_diff": float(np.mean(diffs)),
        "ci_2.5": float(np.quantile(diffs, 0.025)),
        "ci_97.5": float(np.quantile(diffs, 0.975)),
        "p_diff_gt_0": float(np.mean(diffs > 0)),
        "p_value_onesided": float(np.mean(diffs <= 0)),  # H0: diff <= 0
    }


def binom_sf(k: int, n: int, p: float = 0.5) -> float:
    """P(X >= k) for X ~ Binomial(n, p), exact sum (no scipy needed)."""
    from math import comb
    return float(sum(comb(n, j) * p**j * (1 - p) ** (n - j) for j in range(k, n + 1)))


def sign_test(wins: int, n: int) -> dict:
    """One-sided sign test: H0 P(win)=0.5 vs H1 P(win)>0.5."""
    return {"wins": int(wins), "n": int(n),
            "p_value_onesided": binom_sf(wins, n) if n else float("nan")}


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _norm_ppf(p: float) -> float:
    # Acklam's approximation, good to ~1e-9; avoids scipy dependency.
    if not 0.0 < p < 1.0:
        raise ValueError("p must be in (0,1)")
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104411687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    plow, phigh = 0.02425, 1 - 0.02425
    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
               ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    if p > phigh:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
                 ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    q = p - 0.5
    r = q * q
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / \
           (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)


def deflated_sharpe_ratio(sr_ann: float, skew: float, kurt_pearson: float,
                          n_obs: int, trial_srs: list[float]) -> dict | None:
    """Deflated Sharpe Ratio (Bailey & Lopez de Prado, 2014).

    sr_ann: observed annualized Sharpe; skew/kurt_pearson: of daily returns;
    n_obs: # daily observations; trial_srs: annualized SRs of all K configs tried.
    Returns None when K < 2 (no selection-bias correction possible).
    """
    k = len(trial_srs)
    if k < 2 or not np.isfinite(sr_ann):
        return None
    var_sr = float(np.var(trial_srs, ddof=1))
    if var_sr <= 0 or not np.isfinite(var_sr):
        return None
    sr0 = math.sqrt(var_sr) * ((1 - EULER_GAMMA) * _norm_ppf(1 - 1 / k)
                               + EULER_GAMMA * _norm_ppf(1 - 1 / (k * math.e)))
    denom = 1 - skew * sr_ann + (kurt_pearson - 1) / 4 * sr_ann ** 2
    if denom <= 0 or n_obs < 2:
        return None
    dsr = _norm_cdf((sr_ann - sr0) * math.sqrt(n_obs - 1) / math.sqrt(denom))
    return {"dsr": float(dsr), "sr0_benchmark": float(sr0), "n_trials": int(k),
            "var_trial_sr": var_sr}


# ------------------------------------------------------------------ main ---
def _jsonable(o):
    if isinstance(o, float) and (math.isnan(o) or math.isinf(o)):
        return None
    if isinstance(o, (np.floating, np.integer)):
        o = o.item()
        return _jsonable(o)
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    return o


def run_evaluation(scores_dir: Path, prices: pd.DataFrame | None,
                   costs_bps: float = 15.0, n_boot: int = 10_000,
                   block: int = 21, seed: int = 42) -> dict:
    np.random.seed(seed)
    files = sorted(scores_dir.glob("*.csv"))
    if not files:
        raise ValueError(f"no score CSVs in {scores_dir}")
    log.info("found %d score files", len(files))

    per_year: dict[str, dict] = {}
    by_config: dict[tuple[str, str], dict[int, dict]] = {}
    by_config_returns: dict[tuple[str, str], dict[int, pd.Series]] = {}

    for f in files:
        model, regime, year = parse_score_filename(f)
        key = f"{model}_{regime}_{year}"
        log.info("backtesting %s (regime=%s, year=%d, costs=%sbps)",
                 key, regime, year, costs_bps)
        scores = load_scores(f)
        res = run_backtest(scores, costs_bps=costs_bps, regime=regime, prices=prices)
        m = year_metrics(res)
        per_year[key] = {"model": model, "regime": regime, "year": year, **m}
        by_config.setdefault((model, regime), {})[year] = m
        by_config_returns.setdefault((model, regime), {})[year] = res.returns

    # ---- 5-year means ----
    metric_names = ["annual_return", "sharpe", "turnover", "trade_events",
                    "max_drawdown", "n_rebalances"]
    five_year = {}
    for (model, regime), ym in sorted(by_config.items()):
        years = sorted(ym)
        agg = {"years": years, "n_years": len(years)}
        for mn in metric_names:
            vals = [ym[y][mn] for y in years
                    if ym[y][mn] is not None and np.isfinite(ym[y][mn])]
            agg[mn + "_mean"] = float(np.mean(vals)) if vals else None
        five_year[f"{model}_{regime}"] = {"model": model, "regime": regime, **agg}

    # ---- paper comparison ----
    comparison = {}
    for (model, regime), tgt in PAPER_TARGETS.items():
        ck = f"{model}_{regime}"
        got = five_year.get(ck)
        row = {"target": tgt, "ours": None, "delta": None}
        if got:
            ours = {k: got.get(k + "_mean") for k in tgt}
            row["ours"] = ours
            row["delta"] = {k: (ours[k] - tgt[k]) if ours[k] is not None else None
                            for k in tgt}
        comparison[ck] = row
    table2 = {}
    for (model, regime), tgt_sr in TABLE2_TARGETS.items():
        ck = f"{model}_{regime}"
        got = five_year.get(ck, {}).get("sharpe_mean")
        table2[ck] = {"target_sharpe": tgt_sr, "ours_sharpe": got}

    # ---- cost sensitivity (5y means per cost level) ----
    cost_sensitivity = {}
    for (model, regime), years_d in sorted(by_config.items()):
        ck = f"{model}_{regime}"
        log.info("cost sensitivity for %s", ck)
        sens = {}
        for cbps in COST_LEVELS:
            rets_y, sh_y = [], []
            for year in sorted(years_d):
                f = scores_dir / f"{model}_{regime}_{year}.csv"
                res = run_backtest(load_scores(f), costs_bps=cbps,
                                   regime=regime, prices=prices)
                m = year_metrics(res)
                rets_y.append(m["annual_return"])
                sh_y.append(m["sharpe"])
            sens[str(cbps)] = {
                "annual_return_mean": float(np.nanmean(rets_y)),
                "sharpe_mean": float(np.nanmean(sh_y)),
            }
        cost_sensitivity[ck] = sens

    # ---- robustness: bootstrap Sharpe diff (MISS fund63 vs tech5) ----
    bootstrap = None
    a_key, b_key = ("miss", "fund63"), ("miss", "tech5")
    if a_key in by_config_returns and b_key in by_config_returns:
        ra = pd.concat([by_config_returns[a_key][y]
                        for y in sorted(by_config_returns[a_key])])
        rb = pd.concat([by_config_returns[b_key][y]
                        for y in sorted(by_config_returns[b_key])])
        log.info("block bootstrap: %d resamples, block=%d", n_boot, block)
        bootstrap = {"a": "miss_fund63", "b": "miss_tech5",
                     **block_bootstrap_sharpe_diff(ra, rb, n_boot, block, seed)}
    else:
        log.warning("skipping bootstrap: miss_fund63 or miss_tech5 missing")

    # ---- year-level one-sided sign tests ----
    sign_tests = {}
    for ck, ym in sorted(five_year.items()):
        years = ym["years"]
        rets = [per_year[f"{ck}_{y}"]["annual_return"] for y in years]
        wins = sum(1 for r in rets if r is not None and r > 0)
        sign_tests[ck + "__ret_gt_0"] = {"config": ck, "test": "annual_return>0",
                                        **sign_test(wins, len(years))}
    if a_key in by_config and b_key in by_config:
        yrs = sorted(set(by_config[a_key]) & set(by_config[b_key]))
        wins = sum(1 for y in yrs
                   if by_config[a_key][y]["annual_return"]
                   > by_config[b_key][y]["annual_return"])
        sign_tests["miss_fund63__beats__miss_tech5"] = {
            "test": "yearly annual_return fund63 > tech5 (MISS)",
            **sign_test(wins, len(yrs))}

    # ---- deflated Sharpe ratios (trials = # configs) ----
    trial_srs = []
    cfg_series = {}
    for (model, regime), yrs_d in by_config.items():
        rs = pd.concat([by_config_returns[(model, regime)][y]
                        for y in sorted(yrs_d)])
        cfg_series[f"{model}_{regime}"] = rs
        trial_srs.append(sharpe_ann(rs))
    dsr = {}
    for ck, rs in sorted(cfg_series.items()):
        r = rs.to_numpy(dtype=float)
        r = r[np.isfinite(r)]
        res_dsr = deflated_sharpe_ratio(
            sharpe_ann(rs), float(pd.Series(r).skew()),
            float(pd.Series(r).kurtosis() + 3.0), len(r), trial_srs)
        dsr[ck] = res_dsr

    return {
        "meta": {
            "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "seed": seed, "costs_bps": costs_bps,
            "n_boot": n_boot, "block": block,
            "n_files": len(files),
        },
        "per_year": per_year,
        "five_year": five_year,
        "paper_comparison": comparison,
        "table2_sharpe": table2,
        "cost_sensitivity": cost_sensitivity,
        "bootstrap_sharpe_diff": bootstrap,
        "sign_tests": sign_tests,
        "deflated_sharpe": dsr,
    }


def write_tables_md(res: dict) -> str:
    L = []
    A = L.append
    meta = res["meta"]
    A("# Fundamental_SPP — Evaluation results")
    A("")
    A(f"Generated (UTC): {meta['generated_utc']} | seed={meta['seed']} | "
      f"costs={meta['costs_bps']}bps | bootstrap: {meta['n_boot']} resamples, "
      f"block={meta['block']}d")
    A("")
    A("## Score-file contract")
    A("")
    A("Columns: `ticker, date, sector, score` (`date` as YYYY-MM-DD). "
      "Filename: `<model>_<regime>_<year>.csv`, e.g. `miss_fund63_2021.csv`. "
      "`model` ∈ {miss, lstm, stockmixer, gnn}; `regime` ∈ {fund63, tech63, tech5} "
      "(fund63/tech63 → monthly rebalance, tech5 → weekly Monday rebalance); "
      "one test year per file. Prices come from `data/prices.parquet` "
      "(`ticker, date, adj_close`) unless `--prices` is given.")
    A("")
    A("## Per-year metrics")
    A("")
    A("| config | year | ann. return | Sharpe | turnover | trade events | max DD |")
    A("|---|---|---|---|---|---|---|")
    for key in sorted(res["per_year"]):
        p = res["per_year"][key]
        A(f"| {p['model']}_{p['regime']} | {p['year']} | "
          f"{p['annual_return']:.2%} | {p['sharpe']:.3f} | {p['turnover']:.2f} | "
          f"{p['trade_events']} | {p['max_drawdown']:.2%} |")
    A("")
    A("## 5-year means vs paper targets (Table 1, MISS)")
    A("")
    A("| config | ann. return (ours / paper) | Sharpe (ours / paper) | "
      "trade events/yr (ours / paper) |")
    A("|---|---|---|---|")
    for ck in sorted(res["five_year"]):
        fy = res["five_year"][ck]
        comp = res["paper_comparison"].get(ck)
        if comp and comp["ours"]:
            o, tgt = comp["ours"], comp["target"]
            A(f"| {ck} | {o['annual_return']:.2%} / {tgt['annual_return']:.2%} | "
              f"{o['sharpe']:.3f} / {tgt['sharpe']:.3f} | "
              f"{o['trade_events']:.0f} / {tgt['trade_events']} |")
        else:
            A(f"| {ck} | {fy['annual_return_mean']:.2%} / n/a | "
              f"{fy['sharpe_mean']:.3f} / n/a | {fy['trade_events_mean']:.0f} / n/a |")
    A("")
    A("## Table 2 — Sharpe grid (targets TBD from paper)")
    A("")
    if res["table2_sharpe"]:
        A("| config | Sharpe ours | Sharpe paper |")
        A("|---|---|---|")
        for ck, row in sorted(res["table2_sharpe"].items()):
            o = f"{row['ours_sharpe']:.3f}" if row['ours_sharpe'] is not None else "n/a"
            t = f"{row['target_sharpe']:.3f}" if row['target_sharpe'] is not None else "n/a"
            A(f"| {ck} | {o} | {t} |")
    else:
        A("_No Table 2 targets supplied yet — fill TABLE2_TARGETS in evaluate.py "
          "from Ding (2026). Ours (5y mean Sharpe):_")
        A("")
        for ck in sorted(res["five_year"]):
            A(f"- {ck}: {res['five_year'][ck]['sharpe_mean']:.3f}")
    A("")
    A("## Cost sensitivity (5y means)")
    A("")
    A("| config | " + " | ".join(
        f"{c}bps ret / Sharpe" for c in COST_LEVELS) + " |")
    A("|---|" + "---|" * len(COST_LEVELS))
    for ck in sorted(res["cost_sensitivity"]):
        cells = []
        for c in COST_LEVELS:
            s = res["cost_sensitivity"][ck][str(c)]
            cells.append(f"{s['annual_return_mean']:.2%} / {s['sharpe_mean']:.3f}")
        A(f"| {ck} | " + " | ".join(cells) + " |")
    A("")
    A("## Robustness")
    A("")
    bb = res["bootstrap_sharpe_diff"]
    if bb:
        A(f"Moving-block bootstrap of Sharpe({bb['a']}) − Sharpe({bb['b']}): "
          f"{bb['n_boot']} resamples, block {bb['block']}d, {bb['n_days']} days. "
          f"Mean diff {bb['mean_diff']:.3f} "
          f"[{bb['ci_2.5']:.3f}, {bb['ci_97.5']:.3f}], "
          f"P(diff > 0) = {bb['p_diff_gt_0']:.4f}.")
    else:
        A("Block bootstrap skipped (miss_fund63 or miss_tech5 missing).")
    A("")
    A("Year-level one-sided sign tests (H1: P(win) > 0.5):")
    A("")
    for name in sorted(res["sign_tests"]):
        st = res["sign_tests"][name]
        A(f"- {name}: {st['wins']}/{st['n']} wins, p = {st['p_value_onesided']:.4f}")
    A("")
    A("Deflated Sharpe Ratios (Bailey & Lopez de Prado; "
      f"trials = #configs = {len(res['deflated_sharpe'])}):")
    A("")
    A("| config | DSR | SR0 benchmark |")
    A("|---|---|---|")
    for ck in sorted(res["deflated_sharpe"]):
        d = res["deflated_sharpe"][ck]
        if d:
            A(f"| {ck} | {d['dsr']:.4f} | {d['sr0_benchmark']:.3f} |")
        else:
            A(f"| {ck} | n/a (<2 trials) | n/a |")
    A("")
    return "\n".join(L) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(description="Evaluate Fundamental_SPP score files.")
    ap.add_argument("--scores_dir", required=True, help="dir of <model>_<regime>_<year>.csv")
    ap.add_argument("--out", required=True, help="output dir for metrics.json, tables.md")
    ap.add_argument("--prices", default=None,
                    help="prices parquet (default: data/prices.parquet under project root)")
    ap.add_argument("--costs", type=float, default=15.0, help="one-way costs in bps")
    ap.add_argument("--n_boot", type=int, default=10_000)
    ap.add_argument("--block", type=int, default=21)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args(argv)

    logdir = PROJECT_ROOT / "logs"
    logdir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[logging.FileHandler(logdir / "ws4.log"),
                  logging.StreamHandler(sys.stdout)],
    )
    np.random.seed(args.seed)

    scores_dir = Path(args.scores_dir)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    prices = pd.read_parquet(args.prices) if args.prices else None

    res = run_evaluation(scores_dir, prices, costs_bps=args.costs,
                         n_boot=args.n_boot, block=args.block, seed=args.seed)
    (out / "metrics.json").write_text(json.dumps(_jsonable(res), indent=2))
    (out / "tables.md").write_text(write_tables_md(res))
    log.info("wrote %s and %s", out / "metrics.json", out / "tables.md")


if __name__ == "__main__":
    main()
