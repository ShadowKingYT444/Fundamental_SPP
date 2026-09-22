"""WS1: download daily OHLCV prices for all universe tickers.

Strategy:
- Batched yf.download (40 tickers/batch, single request per batch) for speed and
  to avoid per-request throttling. Per-batch retries with backoff.
- Pure-rename chains (same company, e.g. FB->META) share one download symbol;
  rows are split by rename date into historical tickers.
- Corporate-action cases that are NOT clean renames (ACT/AGN, Q/IQV, DOW/DWDP/DD)
  download under their own symbols.
- Symbols with no Yahoo data (truly delisted/acquired) are recorded as missing;
  Stooq fallback was attempted but is proxy-blocked in this environment.
- Rows kept only within membership spells; date stored as YYYY-MM-DD.
- Per-batch chunk parquet -> data/prices_chunks/ (concatenated later).
"""
import os, sys, time
import pandas as pd
import yfinance as yf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

UNI = pd.read_parquet("data/universe.parquet")
OUT_CHUNKS = "data/prices_chunks"
os.makedirs(OUT_CHUNKS, exist_ok=True)

WINDOW_START = "2008-01-01"
DL_START = "2007-01-01"
DL_END = "2026-05-01"
BATCH = 40

CHAINS = {
    "META": [("FB", "2008-01-01", "2022-06-08"), ("META", "2022-06-09", None)],
    "BKNG": [("PCLN", "2008-01-01", "2018-02-26"), ("BKNG", "2018-02-27", None)],
    "RTX":  [("UTX", "2008-01-01", "2020-04-02"), ("RTX", "2020-04-03", None)],
    "PSKY": [("CBS", "2008-01-01", "2019-12-04"), ("VIAC", "2019-12-05", "2022-02-16"),
             ("PARA", "2022-02-17", "2025-08-06"), ("PSKY", "2025-08-07", None)],
    "ELV":  [("WLP", "2008-01-01", "2014-12-01"), ("ANTM", "2014-12-02", "2022-06-27"),
             ("ELV", "2022-06-28", None)],
    "JEF":  [("LUK", "2008-01-01", "2018-05-17"), ("JEF", "2018-05-18", "2019-09-25")],
    "CPRI": [("KORS", "2008-01-01", "2019-01-01"), ("CPRI", "2019-01-02", "2020-05-11")],
    "CPAY": [("FLT", "2008-01-01", "2024-03-24"), ("CPAY", "2024-03-25", None)],
    "EG":   [("RE", "2008-01-01", "2023-07-09"), ("EG", "2023-07-10", None)],
    "J":    [("JEC", "2008-01-01", "2019-12-09"), ("J", "2019-12-10", None)],
    "CTRA": [("COG", "2008-01-01", "2021-09-30"), ("CTRA", "2021-10-01", None)],
    "DAY":  [("CDAY", "2008-01-01", "2024-01-31"), ("DAY", "2024-02-01", None)],
    "WTW":  [("WLTW", "2008-01-01", "2022-01-09"), ("WTW", "2022-01-10", None)],
    "MDLZ": [("KFT", "2008-01-01", "2012-10-01"), ("MDLZ", "2012-10-02", None)],
    "QEP":  [("STR", "2008-01-01", "2010-06-29"), ("QEP", "2010-06-30", None)],
}

def yf_sym(t):
    return t.replace(".", "-")

def build_plan():
    plan, chained = {}, set()
    for dl, segs in CHAINS.items():
        plan[dl] = segs
        chained.update(t for t, _, _ in segs)
    for t, g in UNI.groupby("ticker"):
        if t in chained:
            continue
        segs = []
        for _, r in g.iterrows():
            s = max(r["start_date"], WINDOW_START)
            e = r["end_date"] if pd.notna(r["end_date"]) else None
            if e is None or e >= WINDOW_START:
                segs.append((t, s, e))
        if segs:
            plan[t] = segs
    return plan

def extract(df, ysym):
    """Extract one symbol's OHLCV from a batch download frame."""
    if df is None or len(df) == 0:
        return None
    try:
        if isinstance(df.columns, pd.MultiIndex):
            lvl0 = set(df.columns.get_level_values(0))
            if ysym in lvl0:
                sub = df.xs(ysym, axis=1, level=0)
            else:
                return None
        else:
            sub = df
        sub = sub.rename(columns={"Open": "open", "High": "high", "Low": "low",
                                  "Close": "close", "Adj Close": "adj_close",
                                  "Volume": "volume"})
        cols = [c for c in ["open", "high", "low", "close", "adj_close", "volume"]
                if c in sub.columns]
        sub = sub[cols].dropna(subset=["close"])
        if len(sub) == 0:
            return None
        sub = sub.reset_index().rename(columns={"index": "date", "Date": "date"})
        sub["date"] = pd.to_datetime(sub["date"]).dt.strftime("%Y-%m-%d")
        return sub[["date"] + cols]
    except Exception:
        return None

def main():
    plan = build_plan()
    syms = sorted(plan)
    done = {f[:-8] for f in os.listdir(OUT_CHUNKS) if f.endswith(".parquet")}
    syms = [s for s in syms if s not in done]
    print(f"download symbols: {len(plan)}, remaining: {len(syms)}", flush=True)
    results = []
    nb = (len(syms) + BATCH - 1) // BATCH
    for bi in range(nb):
        batch = syms[bi * BATCH:(bi + 1) * BATCH]
        ysyms = [yf_sym(s) for s in batch]
        df, err = None, None
        for attempt in range(3):
            try:
                df = yf.download(ysyms, start=DL_START, end=DL_END, auto_adjust=False,
                                 progress=False, threads=True, group_by="ticker")
                break
            except Exception as e:
                err = f"{type(e).__name__}: {str(e)[:80]}"
                time.sleep(4 * (2 ** attempt))
        for sym, ysym in zip(batch, ysyms):
            sub = extract(df, ysym)
            rows = []
            if sub is not None:
                for ticker, s0, e0 in plan[sym]:
                    seg = sub[sub["date"] >= s0]
                    if e0:
                        seg = seg[seg["date"] <= e0]
                    if len(seg):
                        seg = seg.copy()
                        seg["ticker"] = ticker
                        rows.append(seg)
            results.append({"symbol": sym, "ok": bool(rows),
                            "n_rows": sum(len(r) for r in rows),
                            "err": err if sub is None else None})
            if rows:
                pd.concat(rows, ignore_index=True).to_parquet(
                    f"{OUT_CHUNKS}/{sym}.parquet", index=False, compression="zstd")
        print(f"  batch {bi + 1}/{nb}: "
              f"{sum(1 for r in results[-len(batch):] if r['ok'])}/{len(batch)} ok", flush=True)
        time.sleep(3)
    res = pd.DataFrame(results)
    res.to_csv("logs/ws1_download_report.csv", index=False)
    ok = res[res["ok"]]
    print(f"downloaded {len(ok)}/{len(res)} symbols, {int(ok['n_rows'].sum())} rows", flush=True)
    bad = res[~res["ok"]]
    if len(bad):
        print(f"MISSING ({len(bad)}): " + ", ".join(bad["symbol"]), flush=True)

if __name__ == "__main__":
    main()
