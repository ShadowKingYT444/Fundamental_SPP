"""Workstream 1d: technical features from prices.parquet.

Per SPEC (lookbacks in trading days, computed on adj_close unless noted):
- ret_5/21/63/252: adj_close[t]/adj_close[t-N] - 1
- vol_21/63: rolling std of daily log returns x sqrt(252) (ddof=1)
- ma_dist_21/63/200: adj_close / SMA(adj_close, N) - 1
- rsi_14: Wilder's RSI(14) on adj_close
- macd_line = EMA12 - EMA26, macd_sig_diff = macd_line - EMA9(macd_line)
  (raw price units; downstream cross-sectional z-scoring handles scale)
- atr_14: Wilder ATR(14) from unadjusted H/L/C, normalized by close
- vol_surprise: volume / SMA(volume, 63) - 1
- pv_trend_63: rolling 63d Pearson correlation(adj_close, volume)

Output: data/features_tech_raw.parquet (ticker, date, 15 features, sector).
Leading NaNs from lookbacks are kept (downstream drops NaN-feature rows).
Deterministic. Run from ~/workspace/fundamental_spp.
"""
import logging
from datetime import datetime

import numpy as np
import pandas as pd

IN = "data/prices.parquet"
UNI = "data/universe.parquet"
OUT = "data/features_tech_raw.parquet"
LOG = "logs/ws1.log"

logging.basicConfig(filename=LOG, level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger("ws1_tech")

FEATURES = ["ret_5", "ret_21", "ret_63", "ret_252",
            "vol_21", "vol_63",
            "ma_dist_21", "ma_dist_63", "ma_dist_200",
            "rsi_14", "macd", "macd_signal",
            "atr_14", "vol_surprise", "pv_trend_63"]


def tech_features(g):
    g = g.sort_values("date").reset_index(drop=True)
    c = g["adj_close"].astype(float)
    v = g["volume"].astype(float)

    out = pd.DataFrame({"ticker": g["ticker"], "date": g["date"]})
    for n in (5, 21, 63, 252):
        out[f"ret_{n}"] = c / c.shift(n) - 1.0

    logret = np.log(c / c.shift(1))
    out["vol_21"] = logret.rolling(21).std() * np.sqrt(252)
    out["vol_63"] = logret.rolling(63).std() * np.sqrt(252)

    for n in (21, 63, 200):
        out[f"ma_dist_{n}"] = c / c.rolling(n).mean() - 1.0

    # Wilder RSI(14)
    delta = c.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    ag = gain.ewm(alpha=1.0 / 14, adjust=False).mean()
    al = loss.ewm(alpha=1.0 / 14, adjust=False).mean()
    rs = ag / al.replace(0.0, np.nan)
    out["rsi_14"] = (100.0 - 100.0 / (1.0 + rs)).where(al > 0, 100.0)

    # MACD 12/26/9
    ema12 = c.ewm(span=12, adjust=False).mean()
    ema26 = c.ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    out["macd"] = macd_line
    out["macd_signal"] = macd_line - macd_line.ewm(span=9, adjust=False).mean()

    # Wilder ATR(14) from unadjusted OHLC, normalized by close
    h, l, cc = g["high"].astype(float), g["low"].astype(float), g["close"].astype(float)
    pc = cc.shift(1)
    tr = pd.concat([h - l, (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1.0 / 14, adjust=False).mean()
    out["atr_14"] = atr / cc

    out["vol_surprise"] = v / v.rolling(63).mean() - 1.0
    out["pv_trend_63"] = c.rolling(63).corr(v)
    return out


def main():
    t0 = datetime.now()
    print(f"[{t0.strftime('%H:%M:%S')}] loading {IN}", flush=True)
    p = pd.read_parquet(IN)
    uni = pd.read_parquet(UNI)
    sector_of = uni.drop_duplicates("ticker").set_index("ticker")["sector"]

    feats = []
    tickers = sorted(p["ticker"].unique())
    for i, t in enumerate(tickers):
        feats.append(tech_features(p[p["ticker"] == t]))
        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{len(tickers)}", flush=True)
    f = pd.concat(feats, ignore_index=True)
    # sector: former tickers may legitimately have unknown (NaN) sector; keep NaN
    f["sector"] = f["ticker"].map(sector_of)
    n_nosec = int(f["sector"].isna().sum())
    if n_nosec:
        print(f"  note: {n_nosec} rows with unknown sector (former tickers)", flush=True)
    f = f[["ticker", "date"] + FEATURES + ["sector"]]
    f = f.sort_values(["ticker", "date"]).reset_index(drop=True)
    f.to_parquet(OUT, index=False, compression="snappy")

    nan_rates = f[FEATURES].isna().mean().round(4).to_dict()
    msg = (f"wrote {OUT}: {len(f):,} rows x {len(FEATURES)} features, "
           f"{f['ticker'].nunique()} tickers, {f['date'].min()}..{f['date'].max()}, "
           f"NaN rates: {nan_rates}")
    print(msg, flush=True)
    log.info(msg)
    print(f"done in {datetime.now() - t0}", flush=True)


if __name__ == "__main__":
    main()
