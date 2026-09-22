"""WS1 recovery: download prices for tickers missing from data/prices.parquet.

Targets: tickers whose spells overlap the 2008-01-01..2026-04-30 window and which
are not pure-rename old tickers (those are covered via the new ticker). Fixes the
yfinance MultiIndex column layout ((Price, Ticker)) that broke ws1_finalize's
recovery step. Appends recovered rows to data/prices.parquet (spell-filtered).
"""
import os
import sys
import time

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yfinance as yf

UNI = "data/universe.parquet"
OUT = "data/prices.parquet"
WINDOW_START = "2008-01-01"
WINDOW_END = "2026-04-30"
FINAL_COLS = ["ticker", "date", "open", "high", "low", "close", "adj_close", "volume"]
# old tickers covered via new-ticker downloads; skip
SKIP = {"FB", "PCLN", "UTX", "CBS", "VIAC", "PARA", "WLP", "ANTM", "LUK",
        "KORS", "FLT", "RE", "JEC", "WLTW", "KFT"}


def flatten_columns(df):
    if isinstance(df.columns, pd.MultiIndex):
        lv0 = list(df.columns.get_level_values(0))
        # yfinance layouts: (Ticker, Price) or (Price, Ticker)
        if "Open" in lv0 or "open" in lv0:
            df.columns = df.columns.get_level_values(0)
        else:
            df.columns = df.columns.get_level_values(1)
    return df


def norm(df, t):
    df = flatten_columns(df)
    df = df.reset_index().rename(columns={
        "Date": "date", "index": "date",
        "Open": "open", "High": "high", "Low": "low", "Close": "close",
        "Adj Close": "adj_close", "Volume": "volume",
        "open": "open", "high": "high", "low": "low", "close": "close",
        "adj_close": "adj_close", "volume": "volume"})
    if "date" not in df.columns:
        return None
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    missing = [c for c in FINAL_COLS if c not in df.columns and c != "ticker"]
    if missing:
        return None
    df["ticker"] = t
    return df[FINAL_COLS]


def main():
    uni = pd.read_parquet(UNI)
    px = pd.read_parquet(OUT)
    have = set(px["ticker"].unique())
    # candidates: in universe, not in prices, spell overlaps window, not rename-old
    cands = []
    for t, g in uni.groupby("ticker"):
        if t in have or t in SKIP:
            continue
        spells = [(r["start_date"], r["end_date"] if pd.notna(r["end_date"]) else None)
                  for _, r in g.iterrows()]
        if any(s <= WINDOW_END and (e is None or e >= WINDOW_START) for s, e in spells):
            cands.append(t)
    print(f"recovery candidates: {len(cands)}", flush=True)

    frames = []
    failed = []
    for i, t in enumerate(sorted(cands)):
        sym = t.replace(".", "-")
        try:
            df = yf.download(sym, start="2007-01-01", end="2026-05-01",
                             progress=False, auto_adjust=False)
            nd = norm(df, t) if df is not None and len(df) else None
            if nd is not None and len(nd):
                # spell filter
                keep = pd.Series(False, index=nd.index)
                for s, e in [(r["start_date"], r["end_date"] if pd.notna(r["end_date"]) else None)
                             for _, r in uni[uni["ticker"] == t].iterrows()]:
                    m = (nd["date"] >= s)
                    if e is not None:
                        m &= (nd["date"] <= e)
                    keep |= m
                nd = nd[keep]
                if len(nd):
                    frames.append(nd)
                    print(f"[{i+1}/{len(cands)}] {t}: {len(nd)} rows", flush=True)
                else:
                    failed.append(t)
            else:
                failed.append(t)
        except Exception as e:
            failed.append(t)
        time.sleep(2)
    print(f"recovered {len(frames)} tickers, {len(failed)} still missing", flush=True)
    if frames:
        rec = pd.concat(frames, ignore_index=True)
        px2 = pd.concat([px, rec], ignore_index=True)
        px2 = px2.drop_duplicates(subset=["ticker", "date"]).sort_values(["ticker", "date"])
        px2 = px2.reset_index(drop=True)
        px2.to_parquet(OUT, index=False, compression="zstd")
        print(f"updated {OUT}: {len(px2)} rows, {px2['ticker'].nunique()} tickers", flush=True)
    pd.Series(sorted(failed)).to_csv("logs/ws1_no_price_tickers.csv", index=False)


if __name__ == "__main__":
    main()
