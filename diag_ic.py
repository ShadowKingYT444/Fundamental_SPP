"""RankIC diagnostic for all 60 score files.
IC = mean Spearman corr(score_t, forward H-day return) across dates.
H = 63 for fund63/tech63, 5 for tech5.
Also reports IC on sector-demeaned scores.
"""
import pandas as pd, numpy as np, sys, os, json
from pathlib import Path

BASE = str(Path(__file__).resolve().parent)
HORIZON = {'fund63': 63, 'tech63': 63, 'tech5': 5}

px_long = pd.read_parquet(os.path.join(BASE, 'data/prices.parquet'))
PW = px_long.pivot(index='date', columns='ticker', values='adj_close').sort_index()
PW.index = pd.DatetimeIndex(PW.index)
FRET = PW.pct_change(HORIZON['fund63']).shift(-HORIZON['fund63'])  # placeholder, recomputed per horizon
FRETS = {}
for h in (63, 5):
    FRETS[h] = PW.pct_change(h).shift(-h)


def score_matrix(path):
    s = pd.read_csv(path)
    sm = s.pivot_table(index='date', columns='ticker', values='score', aggfunc='last').sort_index()
    sm.index = pd.DatetimeIndex(sm.index)
    sector = s.drop_duplicates('ticker').set_index('ticker')['sector']
    return sm, sector


def spearman_by_row(A, B):
    """Row-wise Spearman corr between two aligned DataFrames."""
    a = A.rank(axis=1)
    b = B.rank(axis=1)
    av = a.to_numpy(float); bv = b.to_numpy(float)
    ics = []
    for i in range(av.shape[0]):
        x, y = av[i], bv[i]
        m = np.isfinite(x) & np.isfinite(y)
        if m.sum() < 10:
            continue
        x = x[m] - x[m].mean(); y = y[m] - y[m].mean()
        d = np.sqrt((x * x).sum() * (y * y).sum())
        if d > 0:
            ics.append((x * y).sum() / d)
    return float(np.mean(ics)) if ics else float('nan'), len(ics)


out = {}
for f in sorted(os.listdir(os.path.join(BASE, 'scores'))):
    if not f.endswith('.csv'):
        continue
    name = f[:-4]
    arch, regime, yr = name.rsplit('_', 2)
    h = HORIZON[regime]
    sm, sector = score_matrix(os.path.join(BASE, 'scores', f))
    fr = FRETS[h].reindex(sm.index)
    cols = sm.columns.intersection(fr.columns)
    S = sm[cols]; F = fr[cols]
    ic_raw, n = spearman_by_row(S, F)
    # sector-demeaned
    sec = np.array([sector.get(c, 'UNK') for c in cols])
    Sv = S.to_numpy(float); Sd = Sv.copy()
    for s in np.unique(sec):
        m = sec == s
        Sd[:, m] = Sv[:, m] - np.nanmean(Sv[:, m], axis=1, keepdims=True)
    Sd = pd.DataFrame(Sd, index=S.index, columns=S.columns)
    ic_dm, _ = spearman_by_row(Sd, F)
    out[name] = dict(ic_raw=round(ic_raw, 4), ic_demeaned=round(ic_dm, 4), n_dates=n)

with open(os.path.join(BASE, 'ic_diagnostic.json'), 'w') as fp:
    json.dump(out, fp, indent=1)

# print compact table: mean IC over years
archs = ['miss', 'stockmixer', 'gnn', 'lstm']
print(f"{'regime':8s} {'metric':10s} " + ' '.join(f'{a:>10s}' for a in archs))
for rg in ('fund63', 'tech63', 'tech5'):
    for met in ('ic_raw', 'ic_demeaned'):
        vals = []
        for a in archs:
            vv = [out[f'{a}_{rg}_{y}'][met] for y in range(2021, 2026)]
            vals.append(f"{np.mean(vv):+.4f}")
        print(f"{rg:8s} {met:10s} " + ' '.join(f'{v:>10s}' for v in vals))
    print()
