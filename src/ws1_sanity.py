"""Workstream 1e: sanity checks for prices.parquet and features_tech_raw.parquet.

- row counts, date ranges
- per-ticker coverage: fraction of expected trading days (union NYSE calendar
  from prices, intersected with membership spells); report tickers < 80%
- AAPL adj_close spot check on 2020-12-31 (expect ~128.816757)
- feature NaN rates
Appends summary to logs/ws1.log. Run from ~/workspace/fundamental_spp.
"""
import logging

import pandas as pd

logging.basicConfig(filename="logs/ws1.log", level=logging.INFO,
                    format="%(asctime)s %(message)s")
log = logging.getLogger("ws1_sanity")


def main():
    out = []
    p = pd.read_parquet("data/prices.parquet")
    u = pd.read_parquet("data/universe.parquet")
    out.append(f"prices: {len(p):,} rows, {p['ticker'].nunique()} tickers, "
               f"{p['date'].min()}..{p['date'].max()}")
    out.append(f"dup (ticker,date): {p.duplicated(['ticker', 'date']).sum()}")
    out.append(f"null closes: {p['close'].isna().sum()}, null adj_close: "
               f"{p['adj_close'].isna().sum()}, null volume: {p['volume'].isna().sum()}")

    # AAPL spot check
    a = p[(p["ticker"] == "AAPL") & (p["date"] == "2020-12-31")]
    out.append(f"AAPL 2020-12-31 adj_close: "
               f"{a['adj_close'].iloc[0] if len(a) else 'MISSING'} (expect ~128.816757)")
    # trading calendar = union of dates in prices
    cal = sorted(p["date"].unique())
    # AAPL 2020 detail: max date, gaps vs NYSE calendar, sample values
    a20 = p[(p["ticker"] == "AAPL") & (p["date"] >= "2020-01-01") & (p["date"] <= "2020-12-31")]
    cal20 = sorted(d for d in cal if "2020-01-01" <= d <= "2020-12-31")
    gaps20 = [d for d in cal20 if d not in set(a20["date"])]
    out.append(f"AAPL 2020: {len(a20)} rows, max date {a20['date'].max() if len(a20) else 'n/a'}, "
               f"missing trading days: {len(gaps20)} {gaps20[:10]}")
    for d in ["2020-01-02", "2020-03-23", "2020-08-31", "2020-12-31"]:
        r = a20[a20["date"] == d]
        out.append(f"  AAPL {d}: adj_close="
                   f"{round(float(r['adj_close'].iloc[0]), 4) if len(r) else 'MISSING'}")

    # coverage vs expected trading days within spells (cal defined above)
    calset = set(cal)
    rows = []
    for t, g in u.groupby("ticker"):
        exp = set()
        for _, r in g.iterrows():
            e = r["end_date"] if pd.notna(r["end_date"]) else "2026-04-30"
            exp.update(d for d in cal if r["start_date"] <= d <= e)
        got = set(p.loc[p["ticker"] == t, "date"])
        rows.append((t, len(got), len(exp), len(got) / len(exp) if exp else 0.0))
    cov = pd.DataFrame(rows, columns=["ticker", "got", "exp", "frac"])
    thin = cov[cov["frac"] < 0.80].sort_values("frac")
    out.append(f"tickers with <80% expected trading days: {len(thin)}")
    for _, r in thin.iterrows():
        out.append(f"  {r['ticker']}: {r['got']}/{r['exp']} = {r['frac']:.2%}")
    zero = sorted(set(u["ticker"]) - set(p["ticker"].unique()))
    out.append(f"tickers with zero price rows: {len(zero)}: {zero}")

    f = pd.read_parquet("data/features_tech_raw.parquet")
    out.append(f"features_tech_raw: {len(f):,} rows, {f['ticker'].nunique()} tickers")
    feats = [c for c in f.columns if c not in ("ticker", "date", "sector")]
    nr = f[feats].isna().mean().round(4)
    out.append("feature NaN rates: " + ", ".join(f"{c}={v:.2%}" for c, v in nr.items()))

    msg = "SANITY\n" + "\n".join(out)
    print(msg, flush=True)
    log.info(msg)


if __name__ == "__main__":
    main()
