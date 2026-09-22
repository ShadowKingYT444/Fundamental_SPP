"""Portfolio backtest for the Fundamental_SPP reproduction.

Score-file contract (see also SCORE_CONTRACT.md):
    Columns : ticker (str), date (YYYY-MM-DD str), sector (GICS sector str), score (float)
    Filename: <model>_<regime>_<year>.csv, e.g. miss_fund63_2021.csv
    model   : miss | lstm | stockmixer | gnn
    regime  : fund63 | tech63 | tech5
    year    : 2021 .. 2025 (one test year per file)

Prices (for return computation) are NOT part of the score file. They are passed
via the ``prices`` argument (DataFrame with columns ticker, date, adj_close) or,
if omitted, loaded from ``data/prices.parquet`` relative to the project root
(~/workspace/fundamental_spp/).

Portfolio rules (per SPEC.md):
  * Rebalance: 63d regimes (fund63, tech63) -> first trading day of each month;
    5d regime (tech5) -> every Monday (trading day).
  * Within each GICS sector, rank scores cross-sectionally:
    LONG  = top quintile (rank_pct >= 0.80), SHORT = bottom quintile (<= 0.20).
  * Hysteresis: an existing long is kept unless rank_pct < 0.60;
    an existing short is kept unless rank_pct > 0.40. (Applies to all regimes;
    disable with hysteresis=False.)
  * Sizing: each sector gets equal gross weight (1/11 of gross, fixed 11 GICS
    sectors); within a sector the long side and short side each get NAV/22,
    equal-weighted per name; scaled so total long notional == total short
    notional (dollar neutral). Single-name cap: 2% of NAV (excess stays in cash).
  * Costs: ``costs_bps`` one-way on traded notional, deducted from NAV.
  * Trades execute at the CLOSE of the rebalance day.
  * Year end: positions are marked to market, NOT force-liquidated.

Definitions:
  * "Trade events/yr" = (# position entries + # position exits) during the year.
    A long->short flip counts as 2 events (1 exit + 1 entry). Resizing an
    existing position is NOT an event.
  * Turnover = (annual one-way traded notional) / (average daily NAV).

Only pandas/numpy are used. Deterministic: no unseeded randomness anywhere.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

N_SECTORS = 11          # fixed GICS sector count used for equal sector gross weights
LONG_ENTRY = 0.80       # top quintile entry threshold (rank percentile)
SHORT_ENTRY = 0.20      # bottom quintile entry threshold
LONG_EXIT = 0.60        # hysteresis exit band for longs
SHORT_EXIT = 0.40       # hysteresis exit band for shorts
POSITION_CAP = 0.02     # max |weight| per name as fraction of NAV
NAV0 = 1.0              # starting NAV (unit NAV; all stats are scale-free)

PROJECT_ROOT = __import__("pathlib").Path(__file__).resolve().parents[1]


@dataclass
class BacktestResult:
    """Container for one (model, regime, year) backtest run."""

    returns: pd.Series            # daily net returns, indexed by date string
    positions: pd.DataFrame       # daily positions: date, ticker, sector, side, weight
    turnover: float               # annual one-way traded notional / avg daily NAV
    traded_notional: float        # annual one-way traded notional (NAV units)
    costs_paid: float             # total costs paid (NAV units)
    n_entries: int                # # position entries in the year
    n_exits: int                  # # position exits in the year
    n_rebalances: int             # # rebalance dates
    rebalance_dates: list = field(default_factory=list)

    @property
    def trade_events(self) -> int:
        return self.n_entries + self.n_exits


def _load_prices(prices, year: int) -> pd.DataFrame:
    """Return price matrix (date x ticker) of adj_close for the given year."""
    if prices is None:
        ppath = PROJECT_ROOT / "data" / "prices.parquet"
        log.info("loading prices from %s", ppath)
        prices = pd.read_parquet(ppath)
    req = {"ticker", "date", "adj_close"}
    missing = req - set(prices.columns)
    if missing:
        raise ValueError(f"prices is missing columns: {sorted(missing)}")
    df = prices[prices["date"].astype(str).str.slice(0, 4) == str(year)].copy()
    if df.empty:
        raise ValueError(f"no prices found for year {year}")
    px = df.pivot_table(index="date", columns="ticker", values="adj_close",
                        aggfunc="last").sort_index()
    # forward-fill within the year so a missing print does not kill a position;
    # back-fill leading NaNs (stock listed mid-year) so it can be traded later.
    px = px.ffill().bfill()
    return px


def _prepare_scores(scores: pd.DataFrame, year: int):
    """Validate scores; return (score matrix date x ticker, sector Series)."""
    req = {"ticker", "date", "sector", "score"}
    missing = req - set(scores.columns)
    if missing:
        raise ValueError(f"scores is missing columns: {sorted(missing)}")
    df = scores.copy()
    df["date"] = df["date"].astype(str)
    df = df[df["date"].str.slice(0, 4) == str(year)]
    if df.empty:
        raise ValueError(f"no scores found for year {year}")
    df = df.dropna(subset=["sector", "score"])
    df = df.sort_values("date").drop_duplicates(["ticker", "date"], keep="last")
    sm = df.pivot_table(index="date", columns="ticker", values="score",
                        aggfunc="last").sort_index()
    sector = df.drop_duplicates("ticker").set_index("ticker")["sector"]
    return sm, sector


def _rebalance_dates(dates: pd.DatetimeIndex, regime: str) -> pd.DatetimeIndex:
    """Monthly first-trading-day (63d regimes) or weekly Monday (5d regime)."""
    d = pd.DatetimeIndex(dates)
    if regime in ("fund63", "tech63"):
        # first trading day of each calendar month
        rb = d.to_series().groupby([d.year, d.month]).first()
        rb = pd.DatetimeIndex(rb.values)
    elif regime == "tech5":
        # every Monday that is a trading day
        rb = d[d.weekday == 0]
    else:
        raise ValueError(f"unknown regime {regime!r}; expected fund63/tech63/tech5")
    # always include the first trading day of the year so positions initialize
    rb = rb.union(pd.DatetimeIndex([d[0]])).sort_values()
    return rb


def run_backtest(scores: pd.DataFrame, costs_bps: float = 15.0,
                 regime: str = "fund63", prices: pd.DataFrame | None = None,
                 hysteresis: bool = True) -> BacktestResult:
    """Run the SPEC portfolio backtest for one test year.

    Parameters
    ----------
    scores : DataFrame with columns (ticker, date, sector, score), one test year.
    costs_bps : one-way transaction cost in basis points on traded notional.
    regime : 'fund63' | 'tech63' (monthly rebalance) | 'tech5' (weekly rebalance).
    prices : optional DataFrame (ticker, date, adj_close); defaults to
        data/prices.parquet under the project root.
    hysteresis : apply exit-band hysteresis (default True).

    Returns
    -------
    BacktestResult
    """
    if len(scores) == 0:
        raise ValueError("scores is empty")
    year = int(str(scores["date"].astype(str).iloc[0])[:4])

    sm, sector = _prepare_scores(scores, year)
    px = _load_prices(prices, year)
    sm.index = pd.DatetimeIndex(sm.index)
    px.index = pd.DatetimeIndex(px.index)

    # common trading-day grid: score dates with price coverage
    dates = sm.index.intersection(px.index).sort_values()
    if len(dates) < 2:
        raise ValueError("fewer than 2 common trading days between scores and prices")
    dates = pd.DatetimeIndex(dates)  # normalize to datetimes for rebalance logic
    sm = sm.reindex(dates)
    px = px.reindex(dates)
    tickers = sm.columns.intersection(px.columns)
    sm = sm[tickers]
    px = px[tickers]
    sector = sector.reindex(tickers)

    # sector codes aligned to tickers
    sectors = sorted(sector.dropna().unique())
    sec_code = sector.map({s: i for i, s in enumerate(sectors)}).to_numpy()

    score_m = sm.to_numpy(dtype=float)          # (T, N), NaN where unscored
    price_m = px.to_numpy(dtype=float)          # (T, N)
    T, N = score_m.shape

    rb_dates = _rebalance_dates(dates, regime)
    rb_pos = dates.get_indexer(rb_dates)

    shares = np.zeros(N)                        # signed share holdings
    cash = NAV0
    nav = np.full(T, np.nan)
    pos_w = np.zeros((T, N))                    # signed weights for output
    cur_long = np.zeros(N, dtype=bool)
    cur_short = np.zeros(N, dtype=bool)

    n_entries = n_exits = 0
    traded_notional = 0.0
    costs_paid = 0.0

    cost_rate = costs_bps / 1e4
    side_gross = NAV0 / (2 * N_SECTORS)          # NAV/22 per sector-side (at NAV0 scale;
                                                # scaled by current NAV below)

    for t in range(T):
        p = price_m[t]
        if t in rb_pos:
            nav_t = cash + float(np.nansum(shares * p))
            s = score_m[t]
            ok = np.isfinite(s) & np.isfinite(p)

            new_long = np.zeros(N, dtype=bool)
            new_short = np.zeros(N, dtype=bool)

            # --- within-sector ranking with hysteresis ---
            for sc in range(len(sectors)):
                idx = np.flatnonzero((sec_code == sc) & ok)
                if len(idx) == 0:
                    continue
                ranks = pd.Series(s[idx]).rank(pct=True).to_numpy()  # 1.0 = top
                if hysteresis:
                    keep_l = cur_long[idx] & (ranks >= LONG_EXIT)
                    keep_s = cur_short[idx] & (ranks <= SHORT_EXIT)
                else:
                    keep_l = np.zeros(len(idx), dtype=bool)
                    keep_s = np.zeros(len(idx), dtype=bool)
                enter_l = (~cur_long[idx]) & (ranks >= LONG_ENTRY)
                enter_s = (~cur_short[idx]) & (ranks <= SHORT_ENTRY)
                sel_l = idx[keep_l | enter_l]
                sel_s = idx[keep_s | enter_s]
                new_long[sel_l] = True
                new_short[sel_s] = True

            # --- trade events ---
            # (names that lost score/price coverage cannot be selected above,
            #  so they fall out of new_long/new_short and count as exits)
            entries = (new_long & ~cur_long) | (new_short & ~cur_short)
            exits = ((cur_long & ~new_long) | (cur_short & ~new_short))
            n_entries += int(entries.sum())
            n_exits += int(exits.sum())

            # --- sizing: equal sector gross (1/11), dollar neutral, 2% cap ---
            # target gross per sector-side scales with current NAV
            scale = nav_t / NAV0
            tgt = np.zeros(N)
            for sc in range(len(sectors)):
                idx_l = np.flatnonzero((sec_code == sc) & new_long)
                idx_s = np.flatnonzero((sec_code == sc) & new_short)
                if len(idx_l):
                    w = min(side_gross * scale / len(idx_l), POSITION_CAP * nav_t)
                    tgt[idx_l] = w
                if len(idx_s):
                    w = min(side_gross * scale / len(idx_s), POSITION_CAP * nav_t)
                    tgt[idx_s] = -w
            # exact dollar neutrality: scale the larger side down to the smaller
            gl, gs = tgt[tgt > 0].sum(), -tgt[tgt < 0].sum()
            if gl > 0 and gs > 0 and not np.isclose(gl, gs):
                if gl > gs:
                    tgt[tgt > 0] *= gs / gl
                else:
                    tgt[tgt < 0] *= gl / gs

            target_shares = np.zeros(N)
            tradeable = np.isfinite(p) & (p > 0)
            target_shares[tradeable] = tgt[tradeable] / p[tradeable]

            d_shares = target_shares - shares
            tn = float(np.nansum(np.abs(d_shares) * np.where(tradeable, p, 0.0)))
            cost = cost_rate * tn
            # cash settlement at close prices
            cash = cash - float(np.nansum(d_shares * np.where(tradeable, p, 0.0))) - cost
            shares = target_shares
            traded_notional += tn
            costs_paid += cost
            cur_long, cur_short = new_long, new_short
            nav_t = cash + float(np.nansum(shares * p))  # post-trade NAV

        nav[t] = cash + float(np.nansum(shares * price_m[t]))
        pos_w[t] = shares * price_m[t] / nav[t] if nav[t] > 0 else 0.0

    rets = pd.Series(nav, index=dates).pct_change().dropna()
    rets.index = rets.index.astype(str)

    # daily positions (only rows with nonzero weight)
    nz = np.flatnonzero(np.abs(pos_w).sum(axis=0) > 0)
    pos_rows = []
    date_str = dates.strftime("%Y-%m-%d").to_numpy()
    for j in nz:
        wj = pos_w[:, j]
        m = wj != 0
        if m.any():
            pos_rows.append(pd.DataFrame({
                "date": date_str[m],
                "ticker": tickers[j],
                "sector": sector.iloc[j],
                "side": np.sign(wj[m]).astype(int),
                "weight": wj[m],
            }))
    positions = (pd.concat(pos_rows, ignore_index=True)
                 if pos_rows else
                 pd.DataFrame(columns=["date", "ticker", "sector", "side", "weight"]))

    avg_nav = float(np.nanmean(nav))
    turnover = traded_notional / avg_nav if avg_nav > 0 else np.nan

    return BacktestResult(
        returns=rets,
        positions=positions,
        turnover=float(turnover),
        traded_notional=float(traded_notional),
        costs_paid=float(costs_paid),
        n_entries=int(n_entries),
        n_exits=int(n_exits),
        n_rebalances=int(len(rb_dates)),
        rebalance_dates=[str(d.date()) for d in rb_dates],
    )
