#!/usr/bin/env python3
"""Workstream 2 — normalization + targets + regime panels.

Inputs (whatever exists):
  data/features_fund_raw.parquet   (this workstream)
  data/features_tech_raw.parquet   (workstream 1, optional)
  data/prices.parquet              (workstream 1, optional)
  data/universe.parquet
  data/raw/facts_manifest.csv      (file_ticker -> CIK map for ticker expansion)

Steps:
  1. Expand fundamentals: one row per CIK in fund raw -> replicate to ALL
     universe tickers sharing that CIK (handles FB/META-style renames).
  2. If prices exist: MarketCap = adj_close * shares_v (PIT ffill);
     earn_yield = ni_ttm / MarketCap; fcf_yield = fcf_ttm / MarketCap.
  3. Winsorize each feature at [1%,99%] with thresholds from TRAINING period
     2010-01-01..2019-12-31; save thresholds to data/winsor_thresholds.parquet.
  4. Cross-sectional z-score per date.
  5. If prices exist: target_63d / target_5d = fwd_ret_N - sector mean fwd_ret_N.
  6. Write data/panel_fund63.parquet, data/panel_tech63.parquet,
     data/panel_tech5.parquet (ticker, date, sector, features..., target);
     only rows with valid features AND valid target.
Without prices: writes data/features_fund_norm.parquet (normalized fund
features, no targets) and stops with a documented note.
"""
import os
import numpy as np
import pandas as pd

ROOT = os.path.expanduser("~/workspace/fundamental_spp")
DATA = os.path.join(ROOT, "data")
LOG = os.path.join(ROOT, "logs", "ws2.log")

FUND_FEATS = ["roa", "op_margin", "rev_growth", "earn_growth", "earn_yield",
              "fcf_yield", "leverage", "liquidity", "accruals"]
# SPEC technical features (from adj_close/volume); intersected with available cols.
# Actual names in workstream-1 features_tech_raw.parquet:
TECH_FEATS = ["ret_5", "ret_21", "ret_63", "ret_252", "vol_21", "vol_63",
              "ma_dist_21", "ma_dist_63", "ma_dist_200",
              "rsi_14", "macd", "macd_signal", "atr_14",
              "vol_surprise", "pv_trend_63"]

def restrict_to_membership(df, uni):
    """Keep only (ticker, date) rows inside a historical S&P 500 membership spell.
    Handles tickers with multiple spells (row kept if in ANY spell)."""
    u = uni[["ticker", "start_date", "end_date"]].copy()
    u["start_date"] = pd.to_datetime(u["start_date"])
    u["end_date"] = pd.to_datetime(u["end_date"].fillna("2100-01-01"))
    df = df.copy()
    df["date_dt"] = pd.to_datetime(df["date"])
    df["_idx"] = np.arange(len(df))
    m = df.merge(u, on="ticker", how="left")
    keep = (m["date_dt"] >= m["start_date"]) & (m["date_dt"] <= m["end_date"])
    kept_idx = m.loc[keep.fillna(False), "_idx"].unique()
    out = df[df["_idx"].isin(kept_idx)].drop(columns=["date_dt", "_idx"])
    return out.reset_index(drop=True)
TRAIN_START, TRAIN_END = "2010-01-01", "2019-12-31"

def log(msg):
    line = f"{pd.Timestamp.now():%Y-%m-%d %H:%M:%S} [normalize] {msg}"
    print(line, flush=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")

# Universe CIK quirks (workstream 1 maps some acquired tickers to the acquirer's
# CIK). These tickers must NOT inherit the survivor's fundamentals:
#   LLTC (Linear Technology, acq. 2017) -> universe CIK is MU's
#   RTN  (Raytheon, merged 2020)        -> universe CIK is UTX/RTX's
TICKER_FUND_EXCLUDE = {"LLTC", "RTN"}
# CBS was mapped to VIAC/PARA's CIK; its own CIK 0000813828 is fetched separately
# (see fund_q/CBS.parquet) and mapped here.
CIK_TICKERS_OVERRIDE = {"0000813828": ["CBS"]}

def load_fund_expanded():
    """Fund raw expanded from per-CIK rows to per-ticker rows."""
    fund = pd.read_parquet(os.path.join(DATA, "features_fund_raw.parquet"))
    uni = pd.read_parquet(os.path.join(DATA, "universe.parquet"))
    man = pd.read_csv(os.path.join(DATA, "raw", "facts_manifest.csv"), dtype=str)
    file_cik = dict(zip(man["ticker"], man["cik"]))
    # CIK -> tickers (universe). manifest cik may be "cikA+cikB" for remaps; use first.
    cik_of_file = {t: (c.split("+")[0] if isinstance(c, str) else None)
                   for t, c in file_cik.items()}
    uni_tickers = uni.dropna(subset=["cik"])
    cik_tickers = uni_tickers.groupby("cik")["ticker"].unique().to_dict()
    cik_tickers.update(CIK_TICKERS_OVERRIDE)
    # drop excluded tickers from every CIK's ticker list
    dedicated = {t for v in CIK_TICKERS_OVERRIDE.values() for t in v}  # have own CIK
    for cik in list(cik_tickers):
        kept = [t for t in cik_tickers[cik]
                if t not in TICKER_FUND_EXCLUDE and t not in dedicated]
        if kept:
            cik_tickers[cik] = kept
        else:
            del cik_tickers[cik]
    sector_of = uni.dropna(subset=["sector"]).drop_duplicates("ticker").set_index("ticker")["sector"].to_dict()
    # fill missing sectors from same-CIK siblings
    cik_sector = uni.dropna(subset=["sector"]).drop_duplicates("cik").set_index("cik")["sector"].to_dict()

    fund["cik"] = fund["ticker"].map(cik_of_file)
    chunks = []
    for cik, grp in fund.groupby("cik"):
        tickers = cik_tickers.get(cik, [])
        if len(tickers) == 0:
            continue
        for t in tickers:
            g = grp.copy()
            g["ticker"] = t
            chunks.append(g)
    out = pd.concat(chunks, ignore_index=True)
    out["sector"] = out["ticker"].map(sector_of)
    miss = out["sector"].isna()
    out.loc[miss, "sector"] = out.loc[miss, "cik"].map(cik_sector)
    out = out.drop(columns=["cik"])
    return out

def add_market_cap_features(fund, prices):
    px = prices[["ticker", "date", "adj_close"]].copy()
    m = fund.merge(px, on=["ticker", "date"], how="inner")
    log(f"fund x prices inner merge: {m.shape} (fund rows {len(fund)})")
    m["mcap"] = m["adj_close"] * m["shares_v"]
    m["earn_yield"] = m["ni_ttm"] / m["mcap"]
    m["fcf_yield"] = m["fcf_ttm"] / m["mcap"]
    # clean infinities from zero/negative denominators
    for c in ["earn_yield", "fcf_yield"]:
        m[c] = m[c].replace([np.inf, -np.inf], np.nan)
    return m

def winsorize(df, feats, set_name="fund"):
    """Clip at [1%,99%] from training-period thresholds; save thresholds."""
    tr = df[(df["date"] >= TRAIN_START) & (df["date"] <= TRAIN_END)]
    rows = []
    for f in feats:
        if f not in df.columns:
            rows.append({"feature": f, "p01": np.nan, "p99": np.nan})
            continue
        v = tr[f].dropna()
        if len(v) == 0:
            p01, p99 = np.nan, np.nan
        else:
            p01, p99 = v.quantile(0.01), v.quantile(0.99)
        rows.append({"feature": f, "p01": p01, "p99": p99})
        if np.isfinite(p01) and np.isfinite(p99) and p99 > p01:
            df[f] = df[f].clip(p01, p99)
    th = pd.DataFrame(rows)
    th["set"] = set_name
    return df, th

def zscore_per_date(df, feats):
    for f in feats:
        if f not in df.columns:
            continue
        g = df.groupby("date")[f]
        mu = g.transform("mean")
        sd = g.transform("std")  # ddof=1
        z = (df[f] - mu) / sd
        # constant feature (sd==0) -> z=0; NaN inputs (sd NaN) stay NaN
        z = z.mask(sd == 0, 0.0)
        df[f] = z
    return df

def add_targets(df, prices, uni):
    """Sector-neutral forward-return targets. Needs +63 trading days of prices.
    Guards against price gaps (workstream-1 data): the row N positions ahead must
    actually be N business days later, else the forward return is NaN."""
    px = prices[["ticker", "date", "adj_close"]].copy().sort_values(["ticker", "date"])
    px["date"] = px["date"].astype(str)
    px["_dt"] = pd.to_datetime(px["date"])
    for N, col in [(63, "fwd63"), (5, "fwd5")]:
        px[col] = px.groupby("ticker")["adj_close"].transform(
            lambda s: s.shift(-N) / s - 1)
        # gap guard: Nth-next row must be ~N business days ahead.
        # busday_count >= N always (it counts holidays; trading days don't).
        # Allow up to 10 extra days for holidays; true gaps are 100s of days.
        d0 = px["_dt"].to_numpy(dtype="datetime64[D]")
        d1 = px.groupby("ticker")["_dt"].shift(-N)
        d1_np = d1.to_numpy(dtype="datetime64[D]")
        m = ~pd.isna(d1_np)
        ok = pd.Series(False, index=px.index)
        if m.any():
            nbus = np.busday_count(d0[m], d1_np[m])
            ok.loc[m] = (nbus - N) <= 10
        px.loc[~ok.to_numpy(), col] = np.nan
    px = px.drop(columns=["_dt"])
    sector_of = uni.dropna(subset=["sector"]).drop_duplicates("ticker").set_index("ticker")["sector"].to_dict()
    px["sector"] = px["ticker"].map(sector_of)
    # sector means over in-universe stocks only (SPEC: "across universe stocks that day")
    px = restrict_to_membership(px, uni)
    for col, tcol in [("fwd63", "target_63d"), ("fwd5", "target_5d")]:
        valid = px[col].notna() & px["sector"].notna()
        secmean = px.loc[valid].groupby(["date", "sector"])[col].transform("mean")
        px[tcol] = np.nan
        px.loc[valid, tcol] = px.loc[valid, col].values - secmean.values
    keep = px[["ticker", "date", "target_63d", "target_5d"]]
    return df.merge(keep, on=["ticker", "date"], how="left")

def main():
    fund = load_fund_expanded()
    log(f"fund expanded: {fund.shape}, tickers={fund['ticker'].nunique()}, "
        f"dates {fund['date'].min()}..{fund['date'].max()}")
    uni_full = pd.read_parquet(os.path.join(DATA, "universe.parquet"))
    # restrict to historical membership spells BEFORE any statistics
    fund = restrict_to_membership(fund, uni_full)
    log(f"fund in-membership: {fund.shape}, tickers={fund['ticker'].nunique()}")
    prices_path = os.path.join(DATA, "prices.parquet")
    tech_path = os.path.join(DATA, "features_tech_raw.parquet")
    has_prices = os.path.exists(prices_path)
    has_tech = os.path.exists(tech_path)
    log(f"has_prices={has_prices} has_tech={has_tech}")

    tech = None
    if has_tech:
        tech = pd.read_parquet(tech_path)
        tech["date"] = tech["date"].astype(str)
        tech = restrict_to_membership(tech, uni_full)
        log(f"tech raw in-membership: {tech.shape}, cols={tech.columns.tolist()[:14]}")

    if has_prices:
        prices = pd.read_parquet(prices_path)
        prices["date"] = prices["date"].astype(str)
        fund = add_market_cap_features(fund, prices)
    else:
        log("NOTE: no prices.parquet — earn_yield/fcf_yield stay NaN; no targets yet.")

    # ---- winsorize + z-score (fundamentals; tech too if present) ----
    fund, th_fund = winsorize(fund, FUND_FEATS)
    fund = zscore_per_date(fund, FUND_FEATS)
    fund.to_parquet(os.path.join(DATA, "features_fund_norm.parquet"), index=False)
    log(f"wrote features_fund_norm.parquet {fund.shape}")

    thresholds = th_fund
    if has_tech and tech is not None:
        tech_feats = [c for c in TECH_FEATS if c in tech.columns]
        log(f"tech features used: {tech_feats}")
        tech, th_tech = winsorize(tech, tech_feats, set_name="tech")
        tech = zscore_per_date(tech, tech_feats)
        thresholds = pd.concat([thresholds, th_tech], ignore_index=True)
    thresholds.to_parquet(os.path.join(DATA, "winsor_thresholds.parquet"), index=False)
    log(f"wrote winsor_thresholds.parquet ({len(thresholds)} features)")

    if not has_prices:
        log("STOPPING before panels: targets need prices.parquet (workstream 1). "
            "Normalized fundamentals saved; rerun when prices exist.")
        return

    uni = uni_full
    fund = add_targets(fund, prices, uni)
    n_t63 = fund["target_63d"].notna().sum()
    n_t5 = fund["target_5d"].notna().sum()
    log(f"targets: valid target_63d={n_t63}, target_5d={n_t5}")

    # ---- regime panels ----
    fund_feats_present = [f for f in FUND_FEATS if f in fund.columns]
    panel_fund63 = fund.dropna(subset=fund_feats_present + ["target_63d"]).copy()
    panel_fund63 = panel_fund63[["ticker", "date", "sector"] + fund_feats_present].copy()
    panel_fund63["target"] = fund.loc[panel_fund63.index, "target_63d"]
    panel_fund63.to_parquet(os.path.join(DATA, "panel_fund63.parquet"), index=False)
    log(f"wrote panel_fund63.parquet {panel_fund63.shape}")

    if has_tech and tech is not None:
        tech_feats = [c for c in TECH_FEATS if c in tech.columns]
        tech = add_targets(tech, prices, uni)
        for name, tcol in [("panel_tech63", "target_63d"), ("panel_tech5", "target_5d")]:
            p = tech.dropna(subset=tech_feats + [tcol]).copy()
            out = p[["ticker", "date", "sector"] + tech_feats].copy()
            out["target"] = p[tcol].values
            out.to_parquet(os.path.join(DATA, f"{name}.parquet"), index=False)
            log(f"wrote {name}.parquet {out.shape}")
    else:
        log("NOTE: no features_tech_raw.parquet — tech regime panels skipped.")

if __name__ == "__main__":
    main()
