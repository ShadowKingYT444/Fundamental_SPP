#!/usr/bin/env python3
"""Workstream 2 — SEC EDGAR fundamental features (point-in-time).

Phase 1: download companyfacts JSON per CIK, extract quarterly PIT series
         -> data/raw/fund_q/{ticker}.parquet (resumable checkpoint).
Phase 2: assemble daily (business-day) panel -> data/features_fund_raw.parquet.

PIT rules:
  - Each concept observation visible at its `filed` acceptance date
    (visible on the filing date: filed_date <= t).
  - Per (concept, period end): keep the EARLIEST-filed observation (original
    release; later restatements ignored -> strictly no lookahead).
  - Flow variables quarterlyized: within a filing, for a given end date keep the
    entry with the latest start date (discrete quarter, e.g. 3-month in a 10-Q).
    10-K full-year values are split: Q4 = annual - Q1 - Q2 - Q3.
  - TTM(end) = sum of last 4 discrete quarterly values ON THE CONCEPT'S OWN
    quarterly series; visible at max(filed) of the 4 quarters.
  - First valid fundamental row requires >= 4 quarters of NetIncomeLoss history.
"""
import math, time, os
import numpy as np
import pandas as pd
import requests

ROOT = os.path.expanduser("~/workspace/fundamental_spp")
DATA = os.path.join(ROOT, "data")
RAWQ = os.path.join(DATA, "raw", "fund_q")
LOG = os.path.join(ROOT, "logs", "ws2.log")
UA = "FundamentalSPP research contact@example.com"
HEADERS = {"User-Agent": UA, "Accept-Encoding": "gzip", "Host": "data.sec.gov"}
os.makedirs(RAWQ, exist_ok=True)

def log(msg):
    line = f"{pd.Timestamp.now():%Y-%m-%d %H:%M:%S} {msg}"
    with open(LOG, "a") as f:  # stdout is also redirected here; don't print to avoid dupes
        f.write(line + "\n")

CONCEPTS = {
    # cname -> ([candidate (ns,tag) in priority order], kind)
    # for multi-candidate concepts we quarterlyize each candidate and keep the
    # one with the most non-null quarterly observations (handles tag variants
    # like AssetsCurrent vs CurrentAssets, Revenues vs RevenueFromContract...)
    "assets":      ([("us-gaap", "Assets")], "pit"),
    "liab":        ([("us-gaap", "Liabilities")], "pit"),
    "cur_assets":  ([("us-gaap", "CurrentAssets"),
                     ("us-gaap", "AssetsCurrent")], "pit"),
    "cur_liab":    ([("us-gaap", "CurrentLiabilities"),
                     ("us-gaap", "LiabilitiesCurrent")], "pit"),
    "equity":      ([("us-gaap", "StockholdersEquity")], "pit"),
    "shares":      ([("us-gaap", "CommonStockSharesOutstanding"),
                     ("dei", "EntityCommonStockSharesOutstanding")], "pit"),
    "rev":         ([("us-gaap", "Revenues"),
                     ("us-gaap", "RevenueFromContractWithCustomerExcludingAssessedTax"),
                     ("us-gaap", "SalesRevenueNet")], "flow"),  # SalesRevenueNet: pre-2018 filers (ext.)
    # NetIncome: some filers (e.g. BSX, ITW, MNST, SPG) tag net income as
    # us-gaap:ProfitLoss instead of NetIncomeLoss for most of their history.
    # Merge both tags (same economic concept); earliest-filed wins per period.
    "ni":          ([("us-gaap", "NetIncomeLoss"), ("us-gaap", "ProfitLoss")], "flow"),
    "opinc":       ([("us-gaap", "OperatingIncomeLoss")], "flow"),
    "ocf":         ([("us-gaap", "NetCashProvidedByUsedInOperatingActivities")], "flow"),
    "capex_pay":   ([("us-gaap", "PaymentsToAcquirePropertyPlantAndEquipment")], "flow"),
}
FORMS = {"10-Q", "10-K"}

# Tickers whose filing history moved to a new CIK: fetch predecessor CIK(s) too
# and merge entries (dedup per period-end keeps earliest filed -> PIT-safe).
CIK_REMAP = {
    "XOM": ["0002115436", "0000034088"],  # ExxonMobil: new CIK (2024+) + historic CIK
}
FEAT_COLS = ["roa", "op_margin", "rev_growth", "earn_growth",
             "leverage", "liquidity", "accruals"]

def fetch_companyfacts(cik10):
    url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik10}.json"
    for attempt in range(3):
        try:
            r = requests.get(url, headers=HEADERS, timeout=90)
            if r.status_code == 200:
                return r.json()
            if r.status_code in (429, 403, 503):
                time.sleep(5 * (attempt + 1)); continue
            return {"_error": f"http {r.status_code}"}
        except Exception:
            time.sleep(3 * (attempt + 1))
    return {"_error": "fetch failed after retries"}

def extract_entries(facts, ns, tag):
    """Raw entries list for one (namespace, tag)."""
    out = []
    node = facts.get(ns, {}).get(tag)
    if not node:
        return out
    for unit, entries in node.get("units", {}).items():
        if unit not in ("USD", "shares"):
            continue
        for e in entries:
            if e.get("form") not in FORMS:
                continue
            try:
                val = float(e["val"])
            except (KeyError, TypeError, ValueError):
                continue
            if not math.isfinite(val):
                continue
            try:
                filed = pd.to_datetime(e["filed"], utc=True)
            except Exception:
                continue
            out.append({"end": e.get("end"), "start": e.get("start"),
                        "filed": filed, "val": val, "form": e.get("form")})
    return out

def extract_concept(facts_list, ns_tag_list, kind):
    """Quarterly series for a concept: MERGE entries across all candidate tags
    (they are taxonomy variants of the same concept), then quarterlyize once.
    Per period-end the earliest-filed observation wins (PIT-safe)."""
    entries = []
    for ns, tag in ns_tag_list:
        for facts in facts_list:
            entries += extract_entries(facts, ns, tag)
    return quarterlyize(entries, kind)

def quarterlyize(entries, kind):
    """Quarterly series DataFrame(end, val, filed) for one concept."""
    if not entries:
        return pd.DataFrame(columns=["end", "val", "filed"])
    df = pd.DataFrame(entries)
    df["end"] = pd.to_datetime(df["end"], errors="coerce")
    df["start"] = pd.to_datetime(df["start"], errors="coerce")
    df = df.dropna(subset=["end"]).copy()
    if kind == "flow":
        df = df.sort_values(["form", "filed", "end", "start"])
        df = df.groupby(["form", "filed", "end"], as_index=False).last()  # latest start = discrete qtr
        df["dur"] = (df["end"] - df["start"]).dt.days
        df["is_annual"] = df["dur"] >= 300
        df = df.sort_values(["end", "filed"]).drop_duplicates("end", keep="first")  # earliest filed
        df = df.sort_values("end").reset_index(drop=True)
        q = df[~df["is_annual"]][["end", "val", "filed"]].copy()
        derived = []
        for _, a in df[df["is_annual"]].iterrows():
            inyr = q[(q["end"] > a["start"]) & (q["end"] < a["end"])]
            if len(inyr) == 3 and inyr["val"].notna().all():
                derived.append({"end": a["end"], "val": a["val"] - inyr["val"].sum(),
                                "filed": a["filed"]})
        if derived:
            q = pd.concat([q, pd.DataFrame(derived)], ignore_index=True)
        q = q.sort_values("end").drop_duplicates("end", keep="first").reset_index(drop=True)
        return q
    df = df.sort_values(["end", "filed"]).drop_duplicates("end", keep="first")
    return df[["end", "val", "filed"]].sort_values("end").reset_index(drop=True)

def ttm_series(q):
    """TTM + YoY on a concept's OWN quarterly series."""
    q = q.sort_values("end").reset_index(drop=True)
    v = q["val"]
    ttm = v.rolling(4, min_periods=4).sum()
    # visibility: max filed (POSIX seconds; unit-proof via Timestamp.timestamp)
    fsec = q["filed"].map(lambda t: t.timestamp() if pd.notna(t) else np.nan)
    fmax = pd.to_datetime(fsec.rolling(4, min_periods=4).max(), unit="s", utc=True)
    out = pd.DataFrame({"end": q["end"], "ttm": ttm, "ttm_f": fmax})
    out["yoy"] = ttm / ttm.shift(4) - 1
    out["yoy_f"] = fmax
    return out

def build_quarterly_panel(cfacts_list):
    facts_list = [cf.get("facts", {}) for cf in cfacts_list]
    series = {}
    for cname, (tags, kind) in CONCEPTS.items():
        series[cname] = extract_concept(facts_list, tags, kind)
    if all(len(s) == 0 for s in series.values()):
        return None
    ends = sorted({e for s in series.values() for e in s["end"]})
    qp = pd.DataFrame({"end": pd.to_datetime(ends)})
    ttm_of = {}
    for cname, (tags, kind) in CONCEPTS.items():
        s = series[cname]
        if len(s):
            m = s.rename(columns={"val": f"{cname}_v", "filed": f"{cname}_f"})
            qp = qp.merge(m, on="end", how="left")
        if kind == "flow" and len(s):
            t = ttm_series(s)
            ttm_of[cname] = t.rename(columns={"ttm": f"{cname}_ttm", "ttm_f": f"{cname}_ttm_f",
                                              "yoy": f"{cname}_yoy", "yoy_f": f"{cname}_yoy_f"})
    for cname, t in ttm_of.items():
        qp = qp.merge(t, on="end", how="left")
    qp = qp.sort_values("end").reset_index(drop=True)

    def col(c):
        return qp[c] if c in qp else pd.Series(np.nan, index=qp.index)
    feats = pd.DataFrame({"end": qp["end"]})
    feats["roa"]        = col("ni_ttm") / col("assets_v")
    feats["roa_f"]      = col("ni_ttm_f").combine_first(col("assets_f"))
    feats["op_margin"]  = col("opinc_ttm") / col("rev_ttm")
    feats["op_margin_f"]= col("opinc_ttm_f").combine_first(col("rev_ttm_f"))
    feats["rev_growth"]  = col("rev_yoy")
    feats["rev_growth_f"]= col("rev_yoy_f")
    feats["earn_growth"] = col("ni_yoy")
    feats["earn_growth_f"] = col("ni_yoy_f")
    feats["leverage"]   = col("liab_v") / col("assets_v")
    feats["leverage_f"] = col("liab_f").combine_first(col("assets_f"))
    feats["liquidity"]  = col("cur_assets_v") / col("cur_liab_v")
    feats["liquidity_f"]= col("cur_assets_f").combine_first(col("cur_liab_f"))
    feats["accruals"]   = (col("ni_ttm") - col("ocf_ttm")) / col("assets_v")
    feats["accruals_f"] = col("ni_ttm_f").combine_first(col("ocf_ttm_f")).combine_first(col("assets_f"))
    # keep ni_ttm for earn_yield = ni_ttm / MarketCap (computed once prices exist)
    feats["ni_ttm"] = col("ni_ttm")
    feats["ni_ttm_f"] = col("ni_ttm_f")
    if "capex_pay_ttm" in qp:
        feats["fcf_ttm"]   = col("ocf_ttm") - (-col("capex_pay_ttm"))
        feats["fcf_ttm_f"] = col("ocf_ttm_f").combine_first(col("capex_pay_ttm_f"))
    feats["shares_v"] = col("shares_v")
    feats["shares_f"] = col("shares_f")
    # gate: >=4 quarters NI history before ANY fundamental feature is valid
    gate = col("ni_ttm").notna()
    for c in FEAT_COLS:
        feats.loc[~gate, c] = np.nan
    feats["any_valid"] = feats[FEAT_COLS].notna().any(axis=1)
    return feats

def pit_ffill_daily(feats, grid):
    """Forward-fill quarterly PIT features onto daily grid (filed_date <= t)."""
    out = pd.DataFrame({"date": pd.to_datetime(grid).astype("datetime64[s]")})
    for c in FEAT_COLS + ["fcf_ttm", "shares_v", "ni_ttm"]:
        fc = f"{c}_f" if f"{c}_f" in feats.columns else None
        use = ["end", c] + ([fc] if fc else [])
        sub = feats[use].dropna(subset=[c]).copy() if c in feats.columns else pd.DataFrame()
        if len(sub) == 0:
            out[c] = np.nan
            continue
        if fc:
            sub["fdate"] = pd.to_datetime(sub[fc].dt.tz_convert("UTC").dt.date).astype("datetime64[s]")
        else:
            sub["fdate"] = pd.to_datetime(sub["end"]).astype("datetime64[s]")
        sub = sub.sort_values("fdate")
        m = pd.merge_asof(out[["date"]], sub[["fdate", c]],
                          left_on="date", right_on="fdate", direction="backward")
        out[c] = m[c].values
    return out

def process_ticker(ticker, cik_list, grid):
    qpath = os.path.join(RAWQ, f"{ticker}.parquet")
    if os.path.exists(qpath):
        feats = pd.read_parquet(qpath)
    else:
        cfacts_list, errors = [], []
        for cik10 in cik_list:
            cf = fetch_companyfacts(cik10)
            if "_error" in cf:
                errors.append(f"{cik10}:{cf['_error']}")
            else:
                cfacts_list.append(cf)
            time.sleep(0.25)  # ~4 req/sec politeness (per CIK fetch)
        if not cfacts_list:
            return None, "; ".join(errors) or "no facts"
        feats = build_quarterly_panel(cfacts_list)
        del cfacts_list
        if feats is None or not feats["any_valid"].any():
            (feats if feats is not None else pd.DataFrame()).to_parquet(qpath)
            return None, "no valid quarters"
        feats.to_parquet(qpath)
    daily = pit_ffill_daily(feats, grid)
    daily.insert(0, "ticker", ticker)
    return daily, "ok"

def main():
    uni = pd.read_parquet(os.path.join(DATA, "universe.parquet")).drop_duplicates("cik").copy()
    log(f"universe: {len(uni)} unique CIKs")
    grid = pd.bdate_range("2008-01-01", "2026-04-30")
    manifest_path = os.path.join(DATA, "raw", "facts_manifest.csv")
    manifest = []
    if os.path.exists(manifest_path):
        # seed from previous run so cached tickers keep their cik mapping
        old = pd.read_csv(manifest_path, dtype=str)
        manifest = old.to_dict("records")
    chunks = []
    done = {f[:-8] for f in os.listdir(RAWQ) if f.endswith(".parquet")}
    todo = uni[~uni["ticker"].isin(done)]
    log(f"todo: {len(todo)} tickers ({len(done)} cached)")
    for i, row in enumerate(todo.itertuples()):
        cik_list = CIK_REMAP.get(row.ticker, [str(row.cik).zfill(10)])
        daily, status = process_ticker(row.ticker, cik_list, grid)
        nq = 0
        try:
            qf = pd.read_parquet(os.path.join(RAWQ, f"{row.ticker}.parquet"))
            nq = int(qf["any_valid"].sum()) if "any_valid" in qf.columns else 0
        except Exception:
            pass
        manifest.append({"ticker": row.ticker, "cik": "+".join(cik_list), "status": status,
                         "n_valid_quarters": nq})
        if daily is not None and len(daily):
            chunks.append(daily)
        if (i + 1) % 25 == 0:
            log(f"progress {i+1}/{len(todo)}")
            pd.DataFrame(manifest).to_csv(manifest_path, index=False)
    for t in sorted(done):  # reload cached tickers' daily panels
        try:
            feats = pd.read_parquet(os.path.join(RAWQ, t + ".parquet"))
            if "any_valid" in feats.columns and feats["any_valid"].any():
                d = pit_ffill_daily(feats, grid)
                d.insert(0, "ticker", t)
                chunks.append(d)
        except Exception as e:
            log(f"cached {t} failed: {e}")
    if manifest:
        pd.DataFrame(manifest).to_csv(manifest_path, index=False)
    if chunks:
        panel = pd.concat(chunks, ignore_index=True)
        panel["date"] = panel["date"].dt.strftime("%Y-%m-%d")
        panel["sector"] = panel["ticker"].map(dict(zip(uni["ticker"], uni["sector"])))
        cols = ["ticker", "date", "sector"] + FEAT_COLS + ["fcf_ttm", "shares_v", "ni_ttm"]
        for c in cols:
            if c not in panel:
                panel[c] = np.nan
        panel = panel[cols]
        panel.to_parquet(os.path.join(DATA, "features_fund_raw.parquet"), index=False)
        log(f"WROTE features_fund_raw.parquet: {panel.shape}")
        log("coverage:\n" + panel[FEAT_COLS].notna().mean().to_string())
    else:
        log("no chunks produced")

if __name__ == "__main__":
    main()
