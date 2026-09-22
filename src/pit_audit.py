"""PIT audit (independent): for each ticker/feature, every daily value change
must occur exactly on a quarterly observation's filing date, and at each
filing date the daily value must equal the newly-filed observation.
Verifies filed <= t semantics without reusing the production merge."""
import os, sys
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.expanduser("~/workspace/fundamental_spp/src"))
from fundamentals import pit_ffill_daily, FEAT_COLS

ROOT = os.path.expanduser("~/workspace/fundamental_spp")
QDIR = os.path.join(ROOT, "data", "raw", "fund_q")
GRID = pd.bdate_range("2008-01-01", "2026-04-30")

def audit_ticker(ticker):
    feats = pd.read_parquet(os.path.join(QDIR, f"{ticker}.parquet"))
    daily = pit_ffill_daily(feats, GRID)
    daily["date"] = pd.to_datetime(daily["date"]).dt.tz_localize(None)
    errs = 0
    for c in FEAT_COLS:
        fc = f"{c}_f"
        if fc not in feats.columns:
            continue
        q = feats[["end", c, fc]].dropna(subset=[c]).copy()
        if len(q) < 2:
            continue
        q["fdate"] = pd.to_datetime(q[fc].dt.tz_convert("UTC").dt.date).dt.tz_localize(None)
        q = q.sort_values("fdate").reset_index(drop=True)
        d = daily[["date", c]].dropna(subset=[c]).reset_index(drop=True)
        if len(d) == 0:
            continue
        # 1) every daily change date must be a filing date
        chg = d[c].to_numpy() != np.roll(d[c].to_numpy(), 1)
        chg[0] = False
        change_dates = set(d.loc[chg, "date"])
        filing_dates = set(q["fdate"])
        bad_changes = change_dates - filing_dates
        # 2) at each filing date, daily value must equal that filing's observation
        #    (use the LAST observation filed that day)
        last_obs = q.groupby("fdate").last()
        merr = 0
        for fdate, row in last_obs.iterrows():
            dv = d.loc[d["date"] == fdate, c]
            if len(dv) and not np.isclose(dv.iloc[0], row[c]):
                merr += 1
        if bad_changes or merr:
            errs += len(bad_changes) + merr
            print(f"  {ticker}.{c}: {len(bad_changes)} non-filing change dates, "
                  f"{merr} filing-date value mismatches")
    return errs

if __name__ == "__main__":
    tickers = sys.argv[1:] or ["AAPL", "XOM", "JPM", "MSFT", "BRK-B"]
    total = 0
    for t in tickers:
        try:
            e = audit_ticker(t)
        except FileNotFoundError:
            print(f"{t}: not ready"); continue
        print(f"{t}: {'PIT OK' if e == 0 else f'{e} ERRORS'}")
        total += e
    print("TOTAL ERRORS:", total)
