"""Workstream 1b: repair universe.parquet in place (no full rebuild).

1. Fix ticker parse artifacts (e.g. 'ALLE |' -> 'ALLE') and merge duplicate/overlapping spells.
2. Fill missing CIKs: SEC company_tickers.json (exact + '.'->'-' variants),
   recursive rename-chain resolution, then a curated MANUAL_CIK map for
   well-known delisted/acquired filers.
3. Fill missing sectors via curated SECTOR_MAP (GICS, best effort).
4. Regenerate data/universe_notes.md documenting what remains unresolved.

Deterministic. Run from ~/workspace/fundamental_spp.
"""
import json
import re
from datetime import timedelta

import pandas as pd

SEC_JSON = "data/raw/company_tickers.json"
UNI = "data/universe.parquet"
NOTES = "data/universe_notes.md"

WINDOW_START = "2008-01-01"

# Same rename map as ws1_universe.py (old -> (new, rename_effective_date))
RENAME_MAP = {
    "FB": ("META", "2022-06-09"),
    "PCLN": ("BKNG", "2018-02-27"),
    "UTX": ("RTX", "2020-04-03"),
    "CBS": ("VIAC", "2019-12-05"),
    "VIAC": ("PARA", "2022-02-17"),
    "PARA": ("PSKY", "2025-08-07"),
}

# Curated CIKs (10-digit) for well-known delisted/acquired SEC filers.
# Best effort from knowledge; workstream 2 should validate via EDGAR before use.
MANUAL_CIK = {
    "LEH": "0000806085",   # Lehman Brothers Holdings
    "FNM": "0000310522",   # Fannie Mae
    "FRE": "0001026214",   # Freddie Mac
    "YHOO": "0001011006",  # Yahoo! Inc.
    "TWTR": "0001418091",  # Twitter, Inc.
    "ATVI": "0000718875",  # Activision Blizzard
    "CELG": "0000816284",  # Celgene
    "MON": "0001110783",   # Monsanto
    "BRCM": "0001052830",  # Broadcom Corp (old)
    "EMC": "0000790070",   # EMC Corp
    "PXD": "0001038357",   # Pioneer Natural Resources
    "CERN": "0000804753",  # Cerner
    "XLNX": "0000743988",  # Xilinx
    "MXIM": "0000743316",  # Maxim Integrated Products
    "HRS": "0000044751",   # Harris Corp
    "RTN": "0000101829",   # Raytheon Co
    "AET": "0001123561",   # Aetna
    "ESRX": "0001123986",  # Express Scripts
    "WLP": "0001156039",   # WellPoint (Anthem)
    "DISCA": "0001437107", # Discovery (class A)
    "DISCK": "0001437107", # Discovery (class C)
    "CMCSK": "0001166691", # Comcast (class K)
    "S": "0000101867",     # Sprint
    "LVLT": "0000792988",  # Level 3 Communications
    "CHK": "0000895126",   # Chesapeake Energy
    "MRO": "0000067518",   # Marathon Oil
    "HES": "0000006166",   # Hess Corp
    "RIG": "0001458891",   # Transocean
    "QEP": "0001108827",   # QEP Resources
    "WPX": "0001513761",   # WPX Energy
    "FLIR": "0000354688",  # FLIR Systems
    "CTXS": "0000877890",  # Citrix Systems
    "CA": "0000356985",    # CA Inc
    "RHT": "0001047672",   # Red Hat
    "INFO": "0001496443",  # IHS Markit
    "TIF": "0000827002",   # Tiffany & Co
    "WFM": "0000865436",   # Whole Foods Market
    "GPS": "0000039919",   # Gap Inc
    "JWN": "0000723189",   # Nordstrom
    "M": "0000079459",     # Macy's
    "HBI": "0001359841",   # Hanesbrands
    "CTLT": "0001593034",  # Catalent
    "CDAY": "0001725057",  # Ceridian HCM (Dayforce)
    "DAY": "0001725057",   # Dayforce (same company)
    "EPAM": "0001352010",  # EPAM Systems
    "DXC": "0001688568",   # DXC Technology
    "JNPR": "0001043604",  # Juniper Networks
    "LLTC": "0000723125",  # Linear Technology
    "AKS": "0000913560",   # AK Steel
    "BTU": "0001064728",   # Peabody Energy
    "SIVB": "0000719739",  # SVB Financial Group
    "FRC": "0001114446",   # First Republic Bank
    "CMA": "0000028371",   # Comerica
    "DWDP": "0001666700",  # DowDuPont
    "SATS": "0001415404",  # EchoStar
    "WBA": "0001618921",   # Walgreens Boots Alliance
    "KORS": "0001530721",  # Capri Holdings
    "CPRI": "0001530721",  # Capri Holdings (ex-Michael Kors ticker)
    "TAP": "0000024545",   # Molson Coors
    "JEC": "0000052988",   # Jacobs
    "FLT": "0001175454",   # Corpay (ex-FleetCor)
    "WU": "0001365135",    # Western Union
    "XRX": "0000108772",   # Xerox Holdings
    "IPG": "0000051644",   # Interpublic Group
    "TRIP": "0001528449",  # TripAdvisor
    "MTCH": "0000891103",  # Match Group
    "CZR": "0001590895",   # Caesars Entertainment
    "PENN": "0000921738",  # Penn Entertainment
    "LUMN": "0000018926",  # Lumen Technologies
    "DISH": "0001001082",  # DISH Network
    "UA": "0001336917",    # Under Armour (class C)
    "UAA": "0001336917",   # Under Armour (class A)
    "VFC": "0000103379",   # VF Corp
    "PVH": "0000078239",   # PVH Corp
    "URBN": "0000912615",  # Urban Outfitters
    "COTY": "0001023828",  # Coty
    "AAP": "0001158449",   # Advance Auto Parts
    "LKQ": "0001065696",   # LKQ Corp
    "POOL": "0000945841",  # Pool Corp
    "CTRA": "0000858470",  # Coterra Energy (ex-Cabot Oil & Gas; CIK corroborated by SEC archive URLs edgar/data/858470)
    "COG": "0000858470",  # Cabot Oil & Gas -> renamed Coterra (CTRA) 2021-10-01; same registrant/CIK
    "KMX": "0001170010",   # CarMax
    "AN": "0000350698",    # AutoNation
    "MHK": "0001047205",   # Mohawk Industries
    "WHR": "0000106640",   # Whirlpool
    "NWL": "0000814453",   # Newell Brands
    "ALLE": "0001579241",  # Allegion
    "ITT": "0000052466",   # ITT Inc
    "FLS": "0000030625",   # Flowserve
    "FTI": "0001681459",   # TechnipFMC
    "NOV": "0001021860",   # NOV Inc
    "HP": "0000046765",    # Helmerich & Payne
}

# Curated GICS sectors for former constituents (best effort). Labels match the
# 11 GICS sector names used in the current-constituent table.
SECTOR_MAP = {
    "AA": "Materials", "AAL": "Industrials", "AAP": "Consumer Discretionary",
    "ABK": "Financials", "ABMD": "Health Care", "ACAS": "Financials",
    "ACE": "Financials", "ACT": "Health Care",  # Actavis plc
    # Sector corrections verified 2026-09-22 against S&P press releases / fund GICS filings:
    # ADS=IT (S&P deletion release 2020-06-22), TSS=IT (TSYS, removed Sep 2019, pre-2023 GICS
    # revision that moved payment processors to Financials), FLT=IT (FleetCor 2018-2023; the
    # 2023 revision is reflected in successor CPAY's current-table Financials label),
    # MWW=Industrials (Monster Worldwide employment services; 2004 fund N-Q/N-CSRS GICS filings).
    "ADS": "Information Technology", "ADT": "Industrials", "AET": "Health Care",
    "AGN": "Health Care", "AIV": "Real Estate", "AKS": "Materials",
    "ALK": "Industrials", "ALLE": "Industrials", "ALTR": "Information Technology",
    "ALXN": "Health Care", "AMG": "Financials", "AMTM": "Industrials",
    "AN": "Consumer Discretionary", "ANDV": "Energy",
    "ANF": "Consumer Discretionary", "ANR": "Energy",
    "ANSS": "Information Technology", "APC": "Energy",
    "APOL": "Consumer Discretionary", "ARG": "Materials",
    "ARNC": "Industrials", "ATI": "Materials",
    "ATVI": "Communication Services", "AVB": "Real Estate",
    "AVP": "Consumer Staples", "AYE": "Utilities", "AYI": "Industrials",
    "BBBY": "Consumer Discretionary", "BBWI": "Consumer Discretionary",
    "BC": "Consumer Discretionary", "BCR": "Health Care",
    "BEAM": "Consumer Staples", "BHF": "Financials", "BHI": "Energy",
    "BIG": "Consumer Discretionary", "BIO": "Health Care",
    "BJS": "Consumer Staples", "BLDR": "Industrials",
    "BMC": "Information Technology", "BMS": "Materials",
    "BRCM": "Information Technology", "BTU": "Energy",
    "BUD": "Consumer Staples", "BWA": "Consumer Discretionary",
    "BXLT": "Health Care", "CA": "Information Technology",
    "CAG": "Consumer Staples", "CAM": "Energy", "CBE": "Industrials",
    "CBS": "Communication Services", "CCE": "Consumer Staples",
    "CCR": "Financials",  # Countrywide Credit Industries
    "CDAY": "Industrials", "CE": "Materials", "CELG": "Health Care",
    "CEPH": "Health Care", "CERN": "Health Care", "CFC": "Financials",
    "CFN": "Health Care", "CHK": "Energy", "CLF": "Materials",
    "CMA": "Financials", "CMCSK": "Communication Services",
    "CNX": "Energy", "COG": "Energy", "COL": "Industrials",
    "COTY": "Consumer Staples", "COV": "Health Care",
    "CPB": "Consumer Staples", "CPGX": "Energy",
    "CPRI": "Consumer Discretionary", "CPWR": "Information Technology",
    "CSC": "Information Technology", "CSRA": "Information Technology",
    "CTLT": "Health Care", "CTRA": "Energy",
    "CTX": "Consumer Discretionary", "CTXS": "Information Technology",
    "CVC": "Communication Services", "CVG": "Industrials",
    "CVH": "Health Care", "CXO": "Energy",
    "CZR": "Consumer Discretionary", "DAY": "Industrials",
    "DF": "Consumer Staples", "DFS": "Financials",
    "DISCA": "Communication Services", "DISCK": "Communication Services",
    "DISH": "Communication Services", "DLPH": "Consumer Discretionary",
    "DNB": "Industrials", "DNR": "Energy", "DO": "Energy",
    "DPS": "Consumer Staples", "DRE": "Real Estate",
    "DTV": "Communication Services", "DV": "Consumer Discretionary",
    "DWDP": "Materials", "DXC": "Information Technology",
    "DYN": "Utilities", "EA": "Communication Services",
    "EK": "Information Technology", "EMC": "Information Technology",
    "EMN": "Materials", "ENDP": "Health Care",
    "ENPH": "Information Technology", "EP": "Energy",
    "EPAM": "Information Technology", "ESRX": "Health Care",
    "ESV": "Energy", "ETFC": "Financials",
    "ETSY": "Consumer Discretionary", "EVHC": "Health Care",
    "FBHS": "Industrials", "FDO": "Consumer Staples", "FHN": "Financials",
    "FII": "Financials", "FL": "Consumer Discretionary",
    "FLIR": "Information Technology", "FLR": "Industrials",
    "FLS": "Industrials", "FLT": "Information Technology", "FMC": "Materials",
    "FNM": "Financials", "FOSL": "Consumer Discretionary",
    "FRC": "Financials", "FRE": "Financials", "FRX": "Health Care",
    "FSR": "Financials",  # Firstar Corp (merged into US Bancorp 2001)
    "FTI": "Energy", "FTR": "Communication Services", "GAS": "Utilities",
    "GENZ": "Health Care", "GGP": "Real Estate",
    "GHC": "Communication Services", "GMCR": "Consumer Staples",
    "GME": "Consumer Discretionary", "GNW": "Financials",
    "GPS": "Consumer Discretionary", "GR": "Industrials",
    "GT": "Consumer Discretionary", "HAR": "Consumer Discretionary",
    "HBI": "Consumer Discretionary", "HCBK": "Financials", "HES": "Energy",
    "HFC": "Energy", "HNZ": "Consumer Staples",
    "HOG": "Consumer Discretionary", "HOLX": "Health Care",
    "HOT": "Consumer Discretionary", "HP": "Energy",
    "HRB": "Consumer Discretionary", "HRS": "Industrials",
    "HSP": "Health Care", "IGT": "Consumer Discretionary",
    "INFO": "Industrials", "IPG": "Communication Services",
    "IPGP": "Information Technology", "ITT": "Industrials",
    "JCP": "Consumer Discretionary", "JDSU": "Information Technology",
    "JEC": "Industrials", "JEF": "Financials",
    "JNPR": "Information Technology", "JNS": "Financials",
    "JNY": "Consumer Discretionary", "JOY": "Industrials",
    "JOYG": "Industrials",  # Joy Global (footnote variant of JOY)
    "JWN": "Consumer Discretionary", "K": "Consumer Staples",
    "KBH": "Consumer Discretionary", "KFT": "Consumer Staples",
    "KG": "Health Care", "KMX": "Consumer Discretionary",
    "KORS": "Consumer Discretionary", "KRFT": "Consumer Staples",
    "KSS": "Consumer Discretionary", "KSU": "Industrials",
    "LEG": "Consumer Discretionary", "LEH": "Financials",
    "LIFE": "Health Care", "LKQ": "Consumer Discretionary",
    "LLL": "Industrials", "LLTC": "Information Technology",
    "LM": "Financials", "LNC": "Financials", "LO": "Consumer Staples",
    "LSI": "Information Technology", "LUK": "Financials",
    "LUMN": "Communication Services", "LVLT": "Communication Services",
    "LW": "Consumer Staples", "LXK": "Information Technology",
    "M": "Consumer Discretionary", "MAC": "Real Estate",
    "MAT": "Consumer Discretionary", "MBC": "Consumer Discretionary",
    "MBI": "Financials", "MDP": "Communication Services", "MEE": "Energy",
    "MFE": "Information Technology", "MHK": "Consumer Discretionary",
    "MHS": "Health Care", "MI": "Financials",  # Marshall & Ilsley
    "MIL": "Health Care", "MJN": "Consumer Staples", "MKTX": "Financials",
    "MMI": "Information Technology", "MNK": "Health Care",
    "MOH": "Health Care", "MOLX": "Information Technology",
    "MON": "Materials", "MRO": "Energy",
    "MTCH": "Communication Services", "MUR": "Energy",
    "MWW": "Industrials",  # Monster Worldwide: employment services (GICS HR & Employment Services); corroborated by 2004 fund N-Q/N-CSRS GICS classifications
    "MXIM": "Information Technology", "NAVI": "Financials", "NBL": "Energy",
    "NBR": "Energy", "NCC": "Financials",  # National City
    "NE": "Energy", "NFX": "Energy", "NKTR": "Health Care",
    "NLSN": "Industrials", "NOV": "Energy",
    "NOVL": "Information Technology", "NSM": "Information Technology",
    "NVLS": "Information Technology", "NWL": "Consumer Discretionary",
    "NYT": "Communication Services", "NYX": "Financials",
    "ODP": "Consumer Discretionary", "OGN": "Health Care",
    "OI": "Materials", "OMX": "Consumer Discretionary", "PAYC": "Industrials",
    "PBCT": "Financials", "PBI": "Industrials", "PCL": "Real Estate",
    "PCP": "Industrials", "PCS": "Communication Services",
    "PDCO": "Health Care", "PENN": "Consumer Discretionary",
    "PETM": "Consumer Discretionary", "PGN": "Utilities",
    "PLL": "Industrials", "POM": "Utilities",
    "POOL": "Consumer Discretionary", "PRGO": "Health Care",
    "PTV": "Materials", "PVH": "Consumer Discretionary", "PXD": "Energy",
    "QEP": "Energy", "QRVO": "Information Technology", "R": "Industrials",
    "RAI": "Consumer Staples", "RDC": "Energy",
    "RE": "Financials",  # Everest Re Group
    "RHI": "Industrials", "RHT": "Information Technology", "RIG": "Energy",
    "RRC": "Energy", "RRD": "Industrials",
    "RSH": "Consumer Discretionary", "RTN": "Industrials",
    "RX": "Health Care",  # IMS Health
    "S": "Communication Services", "SAI": "Industrials",
    "SATS": "Communication Services", "SBNY": "Financials",
    "SCG": "Utilities", "SE": "Energy",  # Spectra Energy
    "SEDG": "Information Technology", "SEE": "Materials",
    "SGP": "Health Care",  # Schering-Plough
    "SHLD": "Consumer Discretionary", "SIAL": "Materials",
    "SIG": "Consumer Discretionary", "SII": "Energy",  # Smith International
    "SIVB": "Financials", "SLE": "Consumer Staples", "SLG": "Real Estate",
    "SLM": "Financials", "SNI": "Communication Services",
    "SOLS": "Materials",  # Solstice Advanced Materials (HON spin-off)
    "SPLS": "Consumer Discretionary", "SRCL": "Industrials",
    "STI": "Financials", "STJ": "Health Care", "STR": "Utilities",
    "SUN": "Energy", "SVU": "Consumer Staples", "SWN": "Energy",
    "SWY": "Consumer Staples", "TAP": "Consumer Staples",
    "TDC": "Information Technology", "TE": "Utilities", "TEG": "Utilities",
    "TFX": "Health Care", "TGNA": "Communication Services",
    "THC": "Health Care", "TIE": "Materials",
    "TIF": "Consumer Discretionary", "TLAB": "Information Technology",
    "TRIP": "Consumer Discretionary", "TSO": "Energy",
    "TSS": "Information Technology", "TTD": "Communication Services",
    "TWC": "Communication Services", "TWTR": "Communication Services",
    "TWX": "Communication Services", "TYC": "Industrials",
    "UA": "Consumer Discretionary", "UAA": "Consumer Discretionary",
    "UNM": "Financials", "URBN": "Consumer Discretionary",
    "VAR": "Health Care", "VFC": "Consumer Discretionary",
    "VIAB": "Communication Services", "VIAC": "Communication Services",
    "VNO": "Real Estate", "VNT": "Information Technology",
    "WB": "Financials", "WBA": "Consumer Staples", "WCG": "Health Care",
    "WFM": "Consumer Staples", "WFR": "Information Technology",
    "WHR": "Consumer Discretionary", "WIN": "Communication Services",
    "WLP": "Health Care", "WLTW": "Financials", "WPX": "Energy",
    "WU": "Financials", "WYN": "Consumer Discretionary", "X": "Materials",
    "XEC": "Energy", "XL": "Financials",
    "XLNX": "Information Technology", "XRAY": "Health Care",
    "XRX": "Information Technology", "XTO": "Energy",
    "YHOO": "Communication Services", "ZION": "Financials",
}
# Note: FSR = Firstar Corp (merged into US Bancorp 2001) -> Financials;
# RE = Everest Re Group (added 2017-06-19) -> Financials. Both mapped above.

def merge_spells(spells):
    """Merge overlapping or contiguous (start,end) spells; end None = open.
    String dates 'YYYY-MM-DD' compare lexicographically; None end = infinity."""
    from datetime import date as _date, timedelta as _td
    spells = sorted(set(spells), key=lambda se: (se[0], se[1] or "9999-12-31"))
    out = []
    for s, e in spells:
        if out:
            ps, pe = out[-1]
            pe_d = _date.max if pe is None else pd.to_datetime(pe).date()
            if pd.to_datetime(s).date() <= pe_d + _td(days=1):
                out[-1] = (ps, None if (pe is None or e is None) else max(pe, e))
                continue
        out.append((s, e))
    return out


def main():
    u = pd.read_parquet(UNI)
    n0 = len(u)

    # --- 1. ticker artifact fix ---
    u["ticker"] = u["ticker"].str.replace(r"\s*\|\s*$", "", regex=True)
    fixed = ["ALLE |", "ITT |", "JCP |"]
    u["end_date"] = pd.Series([None if pd.isna(x) else x for x in u["end_date"]],
                             dtype=object, index=u.index)

    sec = json.load(open(SEC_JSON))
    sec_map = {v["ticker"]: str(v["cik_str"]).zfill(10) for v in sec.values()}

    def sec_lookup(t):
        for v in (t, t.replace(".", "-")):
            if v in sec_map:
                return sec_map[v]
        return None

    def cik_for(t, _seen=None):
        _seen = _seen or set()
        c = sec_lookup(t)
        if c:
            return c, "sec"
        if t in RENAME_MAP and t not in _seen:
            _seen.add(t)
            c2, src = cik_for(RENAME_MAP[t][0], _seen)
            if c2:
                return c2, "rename"
        if t in MANUAL_CIK:
            return MANUAL_CIK[t], "manual"
        return None, None

    # --- 2/3. per-ticker CIK + sector, then merge spells ---
    rows = []
    cik_src = {}
    for t, g in u.groupby("ticker", sort=True):
        cik, src = cik_for(t)
        cik_src[t] = (cik, src)
        # Curated SECTOR_MAP wins for mapped (former-constituent) tickers: it is the
        # researched authority. Otherwise keep the existing (current-table) sector.
        sector = SECTOR_MAP.get(t)
        if sector is None:
            sector = g["sector"].dropna().unique()
            sector = sector[0] if len(sector) else None
        spells = merge_spells(list(zip(g["start_date"], g["end_date"])))
        for s, e in spells:
            rows.append({"ticker": t, "cik": cik, "sector": sector,
                         "start_date": s, "end_date": e})
    uni = pd.DataFrame(rows).sort_values(["ticker", "start_date"]).reset_index(drop=True)
    uni.to_parquet(UNI, index=False, compression="snappy")

    n_sec = sum(1 for _, (c, s) in cik_src.items() if c and s == "sec")
    n_ren = sum(1 for _, (c, s) in cik_src.items() if c and s == "rename")
    n_man = sum(1 for _, (c, s) in cik_src.items() if c and s == "manual")
    n_miss_cik = sum(1 for _, (c, s) in cik_src.items() if not c)
    n_miss_sec = int(uni["sector"].isna().sum())
    print(f"rows {n0} -> {len(uni)}; tickers {uni['ticker'].nunique()}")
    print(f"CIK: sec={n_sec} rename={n_ren} manual={n_man} missing={n_miss_cik}")
    print(f"missing sector spell-rows: {n_miss_sec}")
    print(f"sector coverage: {(1 - n_miss_sec/len(uni))*100:.1f}%")

    miss_cik = sorted(t for t, (c, _) in cik_src.items() if not c)
    man_cik = sorted(t for t, (c, s) in cik_src.items() if c and s == "manual")

    # --- 4. notes ---
    notes = []
    notes.append("# Universe build notes\n")
    notes.append("- Sources: Wikipedia 'List of S&P 500 companies' (503 current constituents, "
                 "scraped 2026-09-21) + Wikipedia 'Historical components of the S&P 500' "
                 "(409 change events, 1976-07-01..2026-09-21) + SEC company_tickers.json "
                 "(downloaded 2026-09-21, 10459 filers).\n")
    notes.append(f"- Distinct tickers: {uni['ticker'].nunique()} ({len(uni)} membership spells). "
                 "Window of interest: 2008-01-01..2026-04-30.\n")
    notes.append("- Membership semantics: add effective date = first day IN index; "
                 "remove effective date -> end_date = effective date - 1 day (S&P changes are "
                 "'effective prior to the open' on the stated date).\n")
    notes.append("- Known limitation: the Wikipedia changes table is missing some add events "
                 "(e.g. AAPL 1982, NVDA). Former tickers with a remove event but no add event get "
                 "start_date = 2008-01-01 (in-window approximation). Current tickers always "
                 "use the current table's 'Date added' as fallback.\n")
    notes.append("- Manual rename map (pure ticker renames not recorded as index events):\n")
    for old, (new, rd) in RENAME_MAP.items():
        notes.append(f"  - {old} -> {new} effective {rd}\n")
    notes.append("- yfinance download mapping: 'BRK.B'->'BRK-B', 'BF.B'->'BF-B'.\n")
    notes.append("\n## Repair pass (2026-09-21, src/ws1_repair.py)\n")
    notes.append(f"- Fixed Wikipedia parse artifacts in tickers: {', '.join(fixed)} "
                 "(trailing ' |' footnote marker stripped); spells re-merged.\n")
    notes.append(f"- CIK resolution: {n_sec} via SEC company_tickers.json (exact or '.'->'-' "
                 f"variant), {n_ren} via rename-chain to successor ticker's CIK, {n_man} via "
                 "curated manual map of well-known delisted/acquired filers.\n")
    notes.append("- Sector: all former-constituent spells filled from curated GICS map "
                 "(best effort; see SECTOR_MAP in src/ws1_repair.py). Sector is a single "
                 "static label per ticker (per-spell GICS history not tracked).\n"
                 "  Corrections applied 2026-09-22 from S&P press releases / fund GICS filings:\n"
                 "  - ADS = Information Technology (S&P deletion release 2020-06-22 said IT, not Financials).\n"
                 "  - TSS = Information Technology (TSYS; removed Sep 2019, before the 2023 GICS revision).\n"
                 "  - FLT = Information Technology (FleetCor; data processing & outsourced services 2018-2023;\n"
                 "    the 2023 GICS revision moved payment processors to Financials, which is what the\n"
                 "    current table reports for successor CPAY - both labels are historically defensible).\n"
                 "  - MWW = Industrials (Monster Worldwide; employment services, GICS HR & Employment Services;\n"
                 "    corroborated by 2004 fund N-Q/N-CSRS GICS classifications, not Communication Services).\n")
    notes.append("\n## Unresolved CIKs (%d)\n" % len(miss_cik))
    notes.append("Mostly acquired/bankrupt companies with no current SEC filer record. "
                 "Workstream 2: validate any CIK against EDGAR before pulling companyfacts; "
                 "manual-map CIKs are best-effort.\n")
    for t in miss_cik:
        notes.append(f"- {t}\n")
    notes.append("\n## Manually mapped CIKs (%d, best effort)\n" % len(man_cik))
    for t in man_cik:
        notes.append(f"- {t}: {MANUAL_CIK[t]}\n")
    notes.append("\n## Spells with unknown sector (%d spell-rows)\n" % n_miss_sec)
    if n_miss_sec:
        ms = uni[uni["sector"].isna()][["ticker", "start_date", "end_date"]].drop_duplicates()
        for _, r in ms.iterrows():
            notes.append(f"- {r['ticker']}: {r['start_date']}..{r['end_date']}\n")
    else:
        notes.append("- none: 100% sector coverage.\n")
    open(NOTES, "w").write("".join(notes))
    print(f"notes written to {NOTES}")


if __name__ == "__main__":
    main()
