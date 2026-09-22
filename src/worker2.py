"""Worker 2: process the BACK HALF of the fundamentals todo list in reverse.
Disjoint from worker 1 (which goes forward from index 0). Writes its own
manifest; quarterly files are idempotent so a mid-point meeting is harmless.
Same SEC politeness (0.25s sleep per request) and User-Agent as worker 1."""
import os, sys, csv
import pandas as pd
sys.path.insert(0, os.path.expanduser("~/workspace/fundamental_spp/src"))
from fundamentals import (DATA, RAWQ, CIK_REMAP, process_ticker, log)

GRID = pd.bdate_range("2008-01-01", "2026-04-30")

SPLIT = 315
MAN2 = os.path.join(DATA, "raw", "facts_manifest_w2.csv")

def main():
    uni = pd.read_parquet(os.path.join(DATA, "universe.parquet")).drop_duplicates("cik").copy()
    grid = GRID
    done = {f[:-8] for f in os.listdir(RAWQ) if f.endswith(".parquet")}
    todo = uni[~uni["ticker"].isin(done)]
    back = todo.iloc[SPLIT:].iloc[::-1]  # back half, reversed
    log(f"worker2: {len(back)} tickers (back half reversed)")
    if not os.path.exists(MAN2):
        with open(MAN2, "w", newline="") as f:
            csv.writer(f).writerow(["ticker", "cik", "status", "n_valid_quarters"])
    for i, row in enumerate(back.itertuples()):
        # re-check: worker 1 may have cached it meanwhile
        if os.path.exists(os.path.join(RAWQ, f"{row.ticker}.parquet")):
            continue
        cik_list = CIK_REMAP.get(row.ticker, [str(row.cik).zfill(10)])
        daily, status = process_ticker(row.ticker, cik_list, grid)
        nq = 0
        try:
            qf = pd.read_parquet(os.path.join(RAWQ, f"{row.ticker}.parquet"))
            nq = int(qf["any_valid"].sum()) if "any_valid" in qf.columns else 0
        except Exception:
            pass
        with open(MAN2, "a", newline="") as f:
            csv.writer(f).writerow([row.ticker, "+".join(cik_list), status, nq])
        if (i + 1) % 25 == 0:
            log(f"worker2 progress {i+1}/{len(back)}")
    log("worker2: done")

if __name__ == "__main__":
    main()
