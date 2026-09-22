"""WS1 finalization: assemble data/prices.parquet from downloaded chunks.

Takes the yfinance batch chunks in data/raw/prices_chunk_*.parquet (downloaded for
the 877/858 universe; all 640 tickers present are also in the current universe file)
and:
  1. Maps pure-rename old tickers to the NEW ticker's download (Yahoo serves the
     full company history under the new ticker; the old ticker either 404s or
     returns mislabeled redirected data). Segments are cut by UNIVERSE spells.
  2. Probes tickers with no chunk data for transient yfinance failures and
     downloads full history for those that succeed.
  3. Spell-filters every ticker to its universe membership spells.
  4. Concats, dedupes (ticker,date), sorts, writes data/prices.parquet.
  5. Prunes raw chunk dirs.

Rename chains here mirror RENAME_MAP in src/ws1_repair.py plus index-event
renames (FLT->CPAY, COG->CTRA, WLP->ANTM->ELV, ...). COG/CDAY have no source
data (CTRA/DAY downloads failed) and are documented gaps.
"""
import glob
import os
import sys
import time

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yfinance as yf

UNI = "data/universe.parquet"
RAW_CHUNKS = sorted(glob.glob("data/raw/prices_chunk_*.parquet"))
NEW_CHUNKS = "data/prices_chunks"
OUT = "data/prices.parquet"
WINDOW_START = "2008-01-01"
WINDOW_END = "2026-04-30"  # inclusive

# old ticker -> new ticker whose Yahoo history covers the old ticker's spells
RENAME_SOURCE = {
    "FB": "META", "PCLN": "BKNG", "UTX": "RTX",
    "CBS": "PSKY", "VIAC": "PSKY", "PARA": "PSKY",
    "WLP": "ELV", "ANTM": "ELV", "LUK": "JEF", "KORS": "CPRI",
    "FLT": "CPAY", "RE": "EG", "JEC": "J", "WLTW": "WTW", "KFT": "MDLZ",
}

FINAL_COLS = ["ticker", "date", "open", "high", "low", "close", "adj_close", "volume"]


def load_chunks():
    data = {}
    for f in RAW_CHUNKS:
        df = pd.read_parquet(f)
        for t, g in df.groupby("ticker"):
            data.setdefault(t, []).append(g)
    return {t: pd.concat(gs, ignore_index=True) for t, gs in data.items()}


def spell_filter(df, spells):
    """Keep rows within any of the ticker's (start, end) spells (inclusive)."""
    keep = pd.Series(False, index=df.index)
    for s, e in spells:
        m = df["date"] >= s
        if e is not None:
            m &= df["date"] <= e
        keep |= m
    return df[keep]


def main():
    uni = pd.read_parquet(UNI)
    spells = {}
    for t, g in uni.groupby("ticker"):
        spells[t] = [(r["start_date"], r["end_date"] if pd.notna(r["end_date"]) else None)
                     for _, r in g.iterrows()]

    chunks = load_chunks()
    print(f"loaded {len(chunks)} tickers from {len(RAW_CHUNKS)} raw chunks", flush=True)

    # --- probe tickers with no data for transient yfinance failures ---
    missing = [t for t in spells if t not in chunks and t not in RENAME_SOURCE]
    # rename-old tickers use the new ticker's data; COG/CDAY have no source -> gaps
    recoverable, failed = [], []
    if missing:
        print(f"probing {len(missing)} missing tickers for transient failures...", flush=True)
        for i in range(0, len(missing), 40):
            batch = missing[i:i + 40]
            try:
                df = yf.download(batch, start="2025-01-01", end="2025-02-01",
                                 progress=False, auto_adjust=False, group_by="ticker")
            except Exception:
                failed.extend(batch)
                continue
            ok_here = set()
            if len(batch) == 1:
                ok_here = set(batch) if df is not None and len(df) else set()
            else:
                for t in batch:
                    try:
                        sub = df[t] if t in df.columns.get_level_values(0) else None
                        if sub is not None and len(sub.dropna(how="all")):
                            ok_here.add(t)
                    except Exception:
                        pass
            recoverable.extend(sorted(ok_here))
            failed.extend([t for t in batch if t not in ok_here])
            time.sleep(2)
        print(f"probe: {len(recoverable)} recoverable, {len(failed)} still missing", flush=True)

    # --- full download for recoverable tickers ---
    for t in recoverable:
        sym = t.replace(".", "-")
        try:
            df = yf.download(sym, start="2007-01-01", end="2026-05-01",
                             progress=False, auto_adjust=False)
            if df is None or not len(df):
                failed.append(t)
                continue
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(-1)
            df = df.reset_index().rename(columns={"Date": "date", "Open": "open",
                "High": "high", "Low": "low", "Close": "close",
                "Adj Close": "adj_close", "Volume": "volume"})
            df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
            df["ticker"] = t
            chunks[t] = df[FINAL_COLS]
            print(f"recovered {t}: {len(df)} rows", flush=True)
        except Exception as e:
            print(f"recover {t} failed: {e}", flush=True)
            failed.append(t)
        time.sleep(1.5)

    # --- assemble with rename mapping + spell filter ---
    out_frames, no_data = [], []
    for t in sorted(spells):
        src = RENAME_SOURCE.get(t, t)
        df = chunks.get(src)
        if df is None or not len(df):
            no_data.append(t)
            continue
        sub = spell_filter(df, spells[t])
        if len(sub):
            sub = sub.copy()
            sub["ticker"] = t
            out_frames.append(sub[FINAL_COLS])
        else:
            no_data.append(t)

    px = pd.concat(out_frames, ignore_index=True)
    px["date"] = px["date"].astype(str)
    px = px[(px["date"] >= WINDOW_START) & (px["date"] <= WINDOW_END)]
    px = px.drop_duplicates(subset=["ticker", "date"]).sort_values(["ticker", "date"])
    px = px.reset_index(drop=True)
    for c in ["open", "high", "low", "close", "adj_close", "volume"]:
        px[c] = pd.to_numeric(px[c], errors="coerce")
    px.to_parquet(OUT, index=False, compression="zstd")
    print(f"wrote {OUT}: {len(px)} rows, {px['ticker'].nunique()} tickers, "
          f"{px['date'].min()}..{px['date'].max()}", flush=True)
    print(f"tickers with no price data: {len(no_data)}", flush=True)
    pd.Series(sorted(no_data)).to_csv("logs/ws1_no_price_tickers.csv", index=False)

    # --- prune raw chunks ---
    for f in RAW_CHUNKS:
        os.remove(f)
    if os.path.isdir(NEW_CHUNKS):
        import shutil
        shutil.rmtree(NEW_CHUNKS)
    print("pruned raw chunk dirs", flush=True)


if __name__ == "__main__":
    main()
