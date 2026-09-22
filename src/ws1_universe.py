"""Workstream 1: build S&P 500 point-in-time membership universe.

Sources:
  - Wikipedia "List of S&P 500 companies" (current constituents: ticker, GICS sector, Date added)
  - Wikipedia "Historical components of the S&P 500" (add/remove events 1976-2026)
  - SEC company_tickers.json (ticker -> CIK)
  - Manual rename map for pure ticker renames not recorded as index add/remove events.

Output: data/universe.parquet (ticker, cik, sector, start_date, end_date),
        data/universe_notes.md
Dates are YYYY-MM-DD strings. end_date null/empty = still in index.
A ticker may have multiple rows (multiple membership spells).
"""
import json
import re
from datetime import date, timedelta

import pandas as pd

CUR_HTML = "data/raw/sp500_current.html"
HIST_HTML = "data/raw/sp500_hist.html"
SEC_JSON = "data/raw/company_tickers.json"
OUT_PARQUET = "data/universe.parquet"
NOTES_MD = "data/universe_notes.md"

WINDOW_START = "2008-01-01"
WINDOW_END = "2026-04-30"

# Pure ticker renames NOT recorded as index add/remove events: old -> (new, rename_effective_date)
# rename_effective_date = first trading day under the NEW ticker.
# Dates marked (approx) are best-effort from company announcements; boundaries are
# second-order for monthly-rebalanced backtests but years must be right.
RENAME_MAP = {
    "FB": ("META", "2022-06-09"),    # Facebook -> Meta Platforms
    "PCLN": ("BKNG", "2018-02-27"),  # Priceline -> Booking Holdings
    "UTX": ("RTX", "2020-04-03"),    # United Technologies -> RTX (Raytheon merger)
    "CBS": ("VIAC", "2019-12-05"),   # CBS+Viacom merger -> ViacomCBS
    "VIAC": ("PARA", "2022-02-17"),   # ViacomCBS -> Paramount Global
    "PARA": ("PSKY", "2025-08-07"),   # Paramount+Skydance merger -> Paramount Skydance (approx)
    "WLP": ("ANTM", "2014-12-02"),   # WellPoint -> Anthem (approx)
    "ANTM": ("ELV", "2022-06-28"),   # Anthem -> Elevance Health (approx)
    "LUK": ("JEF", "2018-05-18"),    # Leucadia -> Jefferies Financial (approx)
    "KORS": ("CPRI", "2019-01-02"),  # Michael Kors -> Capri Holdings (approx)
    "FLT": ("CPAY", "2024-03-25"),   # FleetCor -> Corpay (approx)
    "RE": ("EG", "2023-07-10"),      # Everest Re -> Everest Group
    "JEC": ("J", "2019-12-10"),      # Jacobs Engineering -> Jacobs Solutions
    "COG": ("CTRA", "2021-10-01"),   # Cabot+Cimarex merger -> Coterra
    "ACT": ("AGN", "2015-03-17"),    # Actavis -> Allergan plc (acquired Allergan, took name/ticker)
    "Q": ("IQV", "2017-11-15"),      # QuintilesIMS -> IQVIA (also in hist as add/remove pair)
    "CDAY": ("DAY", "2024-02-01"),   # Ceridian ticker change (also in hist)
    "WLTW": ("WTW", "2022-01-10"),   # Willis Towers Watson ticker change (also in hist)
    "KFT": ("MDLZ", "2012-10-02"),   # Kraft Foods -> Mondelez (also in hist)
    "STR": ("QEP", "2010-06-30"),    # Questar -> QEP Resources (also in hist)
    "DOW": ("DWDP", "2017-09-01"),   # Dow Chemical -> DowDuPont (also in hist)
    "DWDP": ("DD", "2019-06-03"),    # DowDuPont -> DuPont de Nemours (also in hist)
    "FSR": ("USB", "2001-02-27"),    # Firstar -> U.S. Bancorp (pre-2008; spell drops out of window)
    "CCR": ("CFC", "1999-06-30"),    # Countrywide Credit -> Countrywide Financial (approx, pre-2008)
}

# Delisted/acquired tickers whose index removal is missing from the Wikipedia changes
# table: manual end_date (acquisition-close based; exact effective day approximate).
MANUAL_END_DATES = {
    "BUD": "2008-11-18",   # InBev completed Anheuser-Busch acquisition 2008-11-19
    "NCC": "2008-12-30",   # PNC completed National City acquisition 2008-12-31
    "HRS": "2019-06-28",   # Harris+L3 merger closed 2019-06-28 (LHX from 2019-07-01)
    "JOYG": "2017-06-30",  # Joy Global removed 2017 (Komatsu acquisition; moved to MidCap 400) (approx)
    "DLPH": "2020-09-30",  # BorgWarner completed Delphi Technologies acquisition 2020-10-01
    "TSO": "2018-09-28",   # Marathon Petroleum completed Tesoro acquisition 2018-10-01
}


def parse_current():
    cur = pd.read_html(CUR_HTML)[0]
    cur = cur.rename(columns={"Symbol": "ticker", "GICS Sector": "sector",
                              "Date added": "date_added", "Security": "name"})
    cur["ticker"] = cur["ticker"].astype(str).str.strip()
    cur["date_added"] = pd.to_datetime(cur["date_added"], errors="coerce").dt.strftime("%Y-%m-%d")
    assert cur["date_added"].notna().all(), "unparseable Date added values"
    assert cur["ticker"].is_unique, "duplicate tickers in current table"
    return cur[["ticker", "name", "sector", "date_added"]]


def parse_hist():
    h = pd.read_html(HIST_HTML)[0]
    h.columns = ["date", "added_t", "added_name", "rem_t", "rem_name", "reason", "refs"]
    h["dt"] = pd.to_datetime(h["date"], errors="coerce")
    assert h["dt"].notna().all()
    h["date"] = h["dt"].dt.strftime("%Y-%m-%d")
    return h


def build_events(hist):
    """ticker -> sorted list of (date, kind) with kind in {'+','-'}; same-day +/- pairs cancelled."""
    def clean(t):
        # strip footnote artifacts like "ALLE |"
        t = re.sub(r"\s*\|$", "", t.strip())
        return t
    ev = {}
    for _, r in hist.iterrows():
        d = r["date"]
        for col, kind in (("added_t", "+"), ("rem_t", "-")):
            t = r[col]
            if isinstance(t, str) and t.strip() and t.strip().lower() != "nan":
                t = clean(t)
                if t:
                    ev.setdefault(t, []).append((d, kind))
    # cancel same-day add+remove pairs (e.g. FOXA 2019-03-19, GAS 2011-12-12)
    for t in ev:
        by_date = {}
        for d, k in ev[t]:
            by_date.setdefault(d, set()).add(k)
        ev[t] = sorted((d, k) for d, ks in by_date.items() for k in ks
                       if not ({"+", "-"} <= ks))
    return ev


def walk_spells(ticker, events):
    """Event walk -> list of (start, end) spells. end None = open."""
    spells, start = [], None
    for d, kind in sorted(events):
        if kind == "+":
            if start is None:
                start = d
            # add while IN: ignore (duplicate)
        else:  # remove
            if start is None:
                # remove with no known add: assume member since window start (in-window approx)
                start = WINDOW_START
            spells.append((start, (pd.to_datetime(d) - timedelta(days=1)).strftime("%Y-%m-%d")))
            start = None
    if start is not None:
        spells.append((start, None))
    return spells


def main():
    cur = parse_current()
    hist = parse_hist()
    events = build_events(hist)
    # manual end dates (delisted tickers missing from changes table) as synthetic removals
    for t, ed in MANUAL_END_DATES.items():
        rdate = (pd.to_datetime(ed) + timedelta(days=1)).strftime("%Y-%m-%d")
        events.setdefault(t, []).append((rdate, "-")
        )
        # re-cancel same-day pairs not needed; just re-sort later in walk
    curset = set(cur["ticker"])
    date_added = dict(zip(cur["ticker"], cur["date_added"]))
    sector_of = dict(zip(cur["ticker"], cur["sector"]))

    # --- walk all tickers seen anywhere ---
    all_tickers = set(events) | curset | set(RENAME_MAP) | {v[0] for v in RENAME_MAP.values()}
    spells = {t: walk_spells(t, events.get(t, [])) for t in all_tickers}

    anomalies = []

    # --- rename overrides ---
    # date on which each ticker itself became the ticker (for rename chains)
    became = {}
    for o2, (n2, rd2) in RENAME_MAP.items():
        became[n2] = rd2
    for old, (new, rd) in RENAME_MAP.items():
        rd_prev = (pd.to_datetime(rd) - timedelta(days=1)).strftime("%Y-%m-%d")
        # OLD ticker: end any spell at rd-1; else create [became_date or WINDOW_START, rd-1]
        old_covers_pre = False
        new_osp = []
        for s, e in spells.setdefault(old, []):
            if e is not None and e < rd:
                new_osp.append((s, e))
                old_covers_pre = True
            elif s >= rd:
                new_osp.append((s, e))  # unexpected; keep
            else:
                new_osp.append((s, rd_prev))
                old_covers_pre = True
        if not old_covers_pre:
            new_osp.append((became.get(old, WINDOW_START), rd_prev))
            anomalies.append(f"rename {old}->{new}: no covering spell, created "
                             f"[{became.get(old, WINDOW_START)},{rd_prev}]")
        spells[old] = new_osp
        # NEW ticker: split any spell straddling rd into (s, rd-1) + (rd, e);
        # drop the pre-rd part when the old ticker covers it (same company).
        new_nsp = []
        for s, e in spells.setdefault(new, []):
            if (e is not None and e < rd) or s >= rd:
                new_nsp.append((s, e))
            else:
                if not old_covers_pre:
                    new_nsp.append((s, rd_prev))
                new_nsp.append((rd, e))
        if new in curset and not any(e is None for _, e in new_nsp):
            new_nsp.append((rd, None))
        spells[new] = sorted(set(new_nsp))

    # --- reconcile with current table ---
    for t in curset:
        sp = spells.get(t, [])
        if not sp:
            spells[t] = [(date_added[t], None)]
        elif sp[-1][1] is not None:
            # walk ended closed but ticker is current: re-added (event missing) -> new spell
            da = date_added[t]
            if da > sp[-1][1]:
                spells[t] = sp + [(da, None)]
            else:
                spells[t] = sp[:-1] + [(sp[-1][0], None)]
                anomalies.append(f"{t}: current but walk closed; reopened last spell")
        # ensure sorted, merge dupes
        spells[t] = sorted(set(spells[t]))

    # --- drop spells entirely before the data window; KEEP spells starting after
    # WINDOW_END (membership truth; the price window filter handles dates) ---
    # also drop invalid spells (start > end), e.g. from remove-while-out edge cases
    final = {}
    for t, sp in spells.items():
        kept = [(s, e) for s, e in sp
                if (e is None or s <= e) and (e is None or e >= WINDOW_START)]
        if kept:
            final[t] = kept

    # --- merge overlapping or contiguous spells within each ticker ---
    def merge(sp):
        sp = sorted(set(sp))
        out = []
        for s, e in sp:
            if out:
                ps, pe = out[-1]
                pe_d = date.max if pe is None else pd.to_datetime(pe).date()
                if pd.to_datetime(s).date() <= pe_d + timedelta(days=1):
                    out[-1] = (ps, None if (pe is None or e is None) else max(pe, e))
                    continue
            out.append((s, e))
        return out
    for t in final:
        final[t] = merge(final[t])

    # --- anomaly: non-current tickers with open spells ---
    for t, sp in final.items():
        if t not in curset and any(e is None for _, e in sp):
            anomalies.append(f"{t}: not current but has open spell {sp} -> closing at {WINDOW_END}")
            final[t] = [(s, WINDOW_END if e is None else e) for s, e in sp]

    # --- CIK lookup ---
    sec = json.load(open(SEC_JSON))
    sec_map = {v["ticker"]: str(v["cik_str"]).zfill(10) for v in sec.values()}
    # for renamed old tickers, use the new ticker's CIK (same company)
    def cik_for(t):
        for cand in (t, t.replace(".", "-")):
            if cand in sec_map:
                return sec_map[cand]
        if t in RENAME_MAP:
            return sec_map.get(RENAME_MAP[t][0])
        return None

    rows = []
    unresolved_cik = []
    for t in sorted(final):
        sec_ticker = t  # wikipedia symbol
        cik = cik_for(t)
        if cik is None:
            unresolved_cik.append(t)
        # sector: current table; rename-old inherits new ticker's sector
        sector = sector_of.get(t)
        if sector is None and t in RENAME_MAP:
            sector = sector_of.get(RENAME_MAP[t][0])
        for s, e in final[t]:
            rows.append({"ticker": t, "cik": cik, "sector": sector,
                         "start_date": s, "end_date": e})
    uni = pd.DataFrame(rows).sort_values(["ticker", "start_date"]).reset_index(drop=True)

    uni.to_parquet(OUT_PARQUET, index=False, compression="snappy")
    print(f"universe rows: {len(uni)}, distinct tickers: {uni['ticker'].nunique()}")
    print(f"current members: {(uni['end_date'].isna()).sum()} spells open")
    print(f"spells per ticker: {(uni.groupby('ticker').size() > 1).sum()} tickers with >1 spell")
    print(f"unresolved CIK: {len(unresolved_cik)}")
    print(f"missing sector (former tickers): {uni['sector'].isna().sum()} spell-rows")

    notes = []
    notes.append("# Universe build notes\n")
    notes.append(f"- Sources: Wikipedia 'List of S&P 500 companies' ({len(cur)} current constituents, "
                 f"scraped 2026-09-21) + Wikipedia 'Historical components of the S&P 500' "
                 f"({len(hist)} change events, 1976-07-01..2026-09-21) + SEC company_tickers.json.\n")
    notes.append(f"- Distinct tickers: {uni['ticker'].nunique()} ({len(uni)} membership spells). "
                 f"Window of interest: {WINDOW_START}..{WINDOW_END}.\n")
    notes.append("- Membership semantics: add effective date = first day IN index; "
                 "remove effective date -> end_date = effective date - 1 day (S&P changes are "
                 "'effective prior to the open' on the stated date).\n")
    notes.append("- Known limitation: the Wikipedia changes table is missing some add events "
                 "(e.g. AAPL 1982, NVDA). Former tickers with a remove event but no add event get "
                 f"start_date = {WINDOW_START} (in-window approximation). Current tickers always "
                 "use the current table's 'Date added' as fallback.\n")
    notes.append("- Manual rename map (pure ticker renames not recorded as index events):\n")
    for old, (new, rd) in RENAME_MAP.items():
        notes.append(f"  - {old} -> {new} effective {rd}\n")
    notes.append("- Manual end dates for delisted tickers missing from the changes table "
                 "(acquisition-close based; exact effective day approximate):\n")
    for t, ed in MANUAL_END_DATES.items():
        notes.append(f"  - {t}: end {ed}\n")
    notes.append("- yfinance download mapping: 'BRK.B'->'BRK-B', 'BF.B'->'BF-B'.\n")
    notes.append("- Special case AGN: the ticker traded continuously through the Actavis acquisition "
                 "(old Allergan removed effective 2015-03-23; new Allergan from 2015-03-17). The universe "
                 "keeps AGN as one continuous spell [2008-01-01, 2020-05-11]; ACT ends 2015-03-16.\n")
    notes.append(f"\n## Unresolved CIKs ({len(unresolved_cik)})\n")
    for t in unresolved_cik:
        notes.append(f"- {t}\n")
    notes.append(f"\n## Spells with unknown sector ({uni['sector'].isna().sum()} spell-rows, "
                 "former tickers not in current table)\n")
    ms = uni[uni['sector'].isna()][['ticker','start_date','end_date']].drop_duplicates()
    for _, r in ms.iterrows():
        notes.append(f"- {r['ticker']}: {r['start_date']}..{r['end_date']}\n")
    if anomalies:
        notes.append(f"\n## Anomalies ({len(anomalies)})\n")
        for a in anomalies[:60]:
            notes.append(f"- {a}\n")
    open(NOTES_MD, "w").write("".join(notes))
    print(f"notes written to {NOTES_MD}")


if __name__ == "__main__":
    main()
