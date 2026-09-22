"""Assemble data/features_fund_raw.parquet from data/raw/fund_q/*.parquet.
Used after re-runs/one-offs update quarterly files."""
import os, sys
import pandas as pd
import numpy as np
sys.path.insert(0, os.path.expanduser("~/workspace/fundamental_spp/src"))
from fundamentals import DATA, RAWQ, FEAT_COLS, pit_ffill_daily, log

GRID = pd.bdate_range("2008-01-01", "2026-04-30")

def main():
    uni = pd.read_parquet(os.path.join(DATA, "universe.parquet"))
    chunks = []
    files = sorted(f for f in os.listdir(RAWQ) if f.endswith(".parquet"))
    log(f"assembling from {len(files)} quarterly files")
    for i, f in enumerate(files):
        t = f[:-8]
        try:
            feats = pd.read_parquet(os.path.join(RAWQ, f))
            if "any_valid" in feats.columns and feats["any_valid"].any():
                d = pit_ffill_daily(feats, GRID)
                d.insert(0, "ticker", t)
                chunks.append(d)
        except Exception as e:
            log(f"assemble {t} failed: {e}")
        if (i + 1) % 100 == 0:
            log(f"assemble {i+1}/{len(files)}")
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

if __name__ == "__main__":
    main()
