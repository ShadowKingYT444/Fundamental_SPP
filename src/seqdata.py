"""On-the-fly sequence windows from long-format panels.

The workstream-2 panels are long format: one row per (ticker, date) with F
feature columns and a single 'target' column. Features are already winsorized
(1st/99th percentiles from 2010-2019 training data) and cross-sectionally
z-scored per date -- see SPEC.md and data/WS2_NOTES.md. This module builds
(L=252)-day lag windows on demand instead of materializing wide panels.

A sample at (ticker, date) is valid iff the ticker has 252 consecutive trading
days ending at `date` (checked against the trading calendar), all feature
values in the window are finite, and the target is finite.
"""

import numpy as np
import pandas as pd

SEQ_LEN = 252


class TradingCalendar:
    def __init__(self, price_path):
        dates = pd.read_parquet(price_path, columns=['date'])['date'].unique()
        self.dates = np.array(sorted(str(d) for d in dates))
        self.ord = {d: i for i, d in enumerate(self.dates)}

    def trading_days(self, lo, hi):
        return [d for d in self.dates if lo <= d <= hi]


class PanelStore:
    """In-memory long panel with per-ticker sorted series."""

    def __init__(self, panel_path, feat_cols, target_col='target',
                 calendar=None):
        df = pd.read_parquet(panel_path,
                             columns=['ticker', 'date', 'sector', target_col]
                             + feat_cols)
        df['date'] = df['date'].astype(str)
        df = df.sort_values(['ticker', 'date']).reset_index(drop=True)
        self.feat_cols = list(feat_cols)
        self.F = len(feat_cols)
        self.target_col = target_col
        self.calendar = calendar
        self.tickers = []  # filled in groupby order (sorted)
        self.dates = []      # per ticker: np array of date strings
        self.ords = []       # per ticker: np int32 calendar ordinals
        self.X = []          # per ticker: (T, F) float32
        self.y = []          # per ticker: (T,) float32
        self.sector = []     # per ticker: str
        self.date_pos = []   # per ticker: dict date -> position
        Xmat = df[feat_cols].to_numpy(dtype=np.float32)
        yv = df[target_col].to_numpy(dtype=np.float32)
        dates_all = df['date'].to_numpy()
        sectors_all = df['sector'].to_numpy()
        for t, idx in df.groupby('ticker', sort=True).indices.items():
            idx = np.asarray(idx)
            d = dates_all[idx]
            self.tickers.append(t)
            self.dates.append(d)
            if calendar is not None:
                self.ords.append(np.array([calendar.ord.get(str(x), -1)
                                           for x in d], dtype=np.int32))
            else:
                self.ords.append(np.arange(len(d), dtype=np.int32))
            self.X.append(np.ascontiguousarray(Xmat[idx]))
            self.y.append(yv[idx].astype(np.float32))
            self.sector.append(str(sectors_all[idx][0]))
            self.date_pos.append({str(x): i for i, x in enumerate(d)})
        self.ti = {t: i for i, t in enumerate(self.tickers)}
        # precompute per-position window validity (vectorized):
        # 252 consecutive trading days + all-finite window + finite target
        self.ok = []
        L = SEQ_LEN
        for ti in range(len(self.tickers)):
            T = len(self.dates[ti])
            ok = np.zeros(T, dtype=bool)
            if T >= L:
                o = self.ords[ti]
                consec = (o[L - 1:] >= 0) & (o[:T - L + 1] >= 0) & \
                    (o[L - 1:] - o[:T - L + 1] == L - 1)
                bad = ~np.isfinite(self.X[ti]).all(axis=1) | \
                    ~np.isfinite(self.y[ti])
                cs = np.concatenate([[0], np.cumsum(bad.astype(np.int32))])
                nowin = cs[L:] - cs[:T - L + 1] == 0
                ok[L - 1:] = consec & nowin
            self.ok.append(ok)
        del df, Xmat, yv

    def panel_dates_in_range(self, lo, hi):
        out = set()
        for d in self.dates:
            m = (d >= lo) & (d <= hi)
            out.update(d[m].tolist())
        return sorted(out)

    def _valid_pos(self, ti, pos):
        ok = self.ok[ti]
        return bool(ok[pos]) if pos < len(ok) else False

    def sample_index(self, dates):
        """List of (ticker_idx, pos) valid samples for the given dates."""
        out = []
        for d in dates:
            for ti in range(len(self.tickers)):
                pos = self.date_pos[ti].get(d)
                if pos is not None and self._valid_pos(ti, pos):
                    out.append((ti, pos))
        return out

    def get_window(self, ti, pos):
        L = SEQ_LEN
        return self.X[ti][pos - L + 1:pos + 1], self.y[ti][pos]

    def date_batch(self, date):
        """All valid (ticker) windows for one date -> X (N,L,F), y (N,),
        tickers, sectors. Empty arrays if none."""
        Xs, ys, tickers, sectors = [], [], [], []
        for ti, t in enumerate(self.tickers):
            pos = self.date_pos[ti].get(date)
            if pos is not None and self._valid_pos(ti, pos):
                w, yv = self.get_window(ti, pos)
                Xs.append(w)
                ys.append(yv)
                tickers.append(t)
                sectors.append(self.sector[ti])
        if not Xs:
            z = np.zeros((0, SEQ_LEN, self.F), dtype=np.float32)
            return z, np.zeros(0, dtype=np.float32), [], []
        return (np.stack(Xs).astype(np.float32),
                np.array(ys, dtype=np.float32), tickers, sectors)


class PriceMatrix:
    """Per-ticker adj_close series for rolling-return graphs (GNN)."""

    def __init__(self, price_path, calendar):
        df = pd.read_parquet(price_path, columns=['ticker', 'date', 'adj_close'])
        df['date'] = df['date'].astype(str)
        df = df.sort_values(['ticker', 'date'])
        self.cal = calendar
        self.series = {}
        for t, g in df.groupby('ticker', sort=False):
            d = g['date'].to_numpy()
            c = g['adj_close'].to_numpy(dtype=np.float64)
            self.series[str(t)] = (d, c)
        del df

    def trailing_returns(self, tickers, date, window=63):
        """(N, window) daily returns ending at `date` (NaN where unavailable)."""
        di = self.cal.ord.get(date)
        out = np.full((len(tickers), window), np.nan)
        if di is None:
            return out
        for i, t in enumerate(tickers):
            s = self.series.get(t)
            if s is None:
                continue
            d, c = s
            # positions of the last `window+1` calendar closes <= date
            pos = np.searchsorted(d, date, side='right') - 1
            if pos < window:
                continue
            cc = c[pos - window:pos + 1]
            if np.all(np.isfinite(cc)) and np.all(cc > 0):
                out[i] = cc[1:] / cc[:-1] - 1.0
        return out
