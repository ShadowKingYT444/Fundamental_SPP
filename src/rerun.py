"""Re-run extraction for tickers in /tmp/rerun_tickers.txt (ProfitLoss fix).
Usage: rerun.py <worker_id> <n_workers>  -- each worker takes its slice.
Writes data/raw/facts_manifest_rerun<W>.csv; quarterly files are idempotent."""
import os, sys, csv
import pandas as pd
sys.path.insert(0, os.path.expanduser("~/workspace/fundamental_spp/src"))
from fundamentals import (DATA, RAWQ, CIK_REMAP, process_ticker, log)

GRID = pd.bdate_range("2008-01-01", "2026-04-30")

def main():
    wid = int(sys.argv[1]); nw = int(sys.argv[2])
    tickers = [l.strip() for l in open("/tmp/rerun_tickers.txt") if l.strip()]
    uni = pd.read_parquet(os.path.join(DATA, "universe.parquet")).drop_duplicates("cik")
    uni = uni.set_index("ticker")
    mine = [t for i, t in enumerate(tickers) if i % nw == wid]
    log(f"rerun worker {wid}/{nw}: {len(mine)} tickers")
    man_path = os.path.join(DATA, "raw", f"facts_manifest_rerun{wid}.csv")
    if not os.path.exists(man_path):
        with open(man_path, "w", newline="") as f:
            csv.writer(f).writerow(["ticker", "cik", "status", "n_valid_quarters"])
    for i, t in enumerate(mine):
        try:
            cik_list = CIK_REMAP.get(t, [str(uni.loc[t, "cik"]).zfill(10)])
        except KeyError:
            # ticker deduped from universe (shares CIK with another); find via manifest
            cik_list = None
        if cik_list is None or "nan" in cik_list[0]:
            continue
        daily, status = process_ticker(t, cik_list, GRID)
        nq = 0
        try:
            qf = pd.read_parquet(os.path.join(RAWQ, f"{t}.parquet"))
            nq = int(qf["any_valid"].sum()) if "any_valid" in qf.columns else 0
        except Exception:
            pass
        with open(man_path, "a", newline="") as f:
            csv.writer(f).writerow([t, "+".join(cik_list), status, nq])
        if (i + 1) % 25 == 0:
            log(f"rerun worker {wid}: {i+1}/{len(mine)}")
    log(f"rerun worker {wid}: done")

if __name__ == "__main__":
    main()
