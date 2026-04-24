"""
Industry Group Relative Strength
Groups NSE stocks into 25 sectors and ranks each by 3-month RS vs Nifty.
Zero extra NSE API calls — uses bhavcopy cache only.
"""
import time
import pandas as pd
from data_fetcher import _weekdays_back, _download_one_day

_cache    = {"data": None, "ts": 0}
CACHE_TTL = 3600   # 1 hour

_NIFTY_SYMS = [
    "RELIANCE", "TCS", "HDFCBANK", "ICICIBANK", "INFOSYS",
    "HINDUNILVR", "SBIN", "BHARTIARTL", "ITC", "KOTAKBANK",
    "AXISBANK", "WIPRO", "HCLTECH", "MARUTI", "BAJFINANCE",
    "TITAN", "NTPC", "POWERGRID", "NESTLEIND", "SUNPHARMA",
]

INDUSTRY_GROUPS: dict[str, list[str]] = {
    "Banks - Private":        ["HDFCBANK", "ICICIBANK", "KOTAKBANK", "AXISBANK",
                               "INDUSINDBK", "BANDHANBNK", "FEDERALBNK", "IDFCFIRSTB"],
    "Banks - PSU":            ["SBIN", "BANKBARODA", "PNB", "CANBK",
                               "UNIONBANK", "INDIANB", "BANKINDIA"],
    "IT - Large Cap":         ["TCS", "INFOSYS", "WIPRO", "HCLTECH",
                               "TECHM", "LTIM", "MPHASIS"],
    "IT - Midcap":            ["PERSISTENT", "COFORGE", "LTTS", "KPITTECH",
                               "TATAELXSI", "MASTEK"],
    "Auto & OEM":             ["MARUTI", "TATAMOTORS", "EICHERMOT", "BAJAJ-AUTO",
                               "HEROMOTOCO", "M&M", "ASHOKLEY"],
    "Auto Ancillary":         ["MOTHERSON", "BOSCHLTD", "EXIDEIND",
                               "BALKRISIND", "APOLLOTYRE"],
    "Pharma - Large":         ["SUNPHARMA", "DRREDDY", "CIPLA",
                               "DIVISLAB", "AUROPHARMA", "LUPIN"],
    "Pharma - Midcap":        ["ALKEM", "TORNTPHARM", "IPCALAB",
                               "AJANTPHARM", "GLENMARK"],
    "FMCG":                   ["HINDUNILVR", "ITC", "NESTLEIND", "DABUR",
                               "MARICO", "BRITANNIA", "COLPAL", "GODREJCP"],
    "Oil & Gas":              ["RELIANCE", "ONGC", "IOC", "BPCL",
                               "HINDPETRO", "GAIL", "OIL"],
    "Metals & Mining":        ["TATASTEEL", "JSWSTEEL", "HINDALCO",
                               "VEDL", "SAIL", "NMDC", "NATIONALUM"],
    "Power & Utilities":      ["NTPC", "POWERGRID", "ADANIPOWER",
                               "TATAPOWER", "CESC", "TORNTPOWER"],
    "Cement":                 ["ULTRACEMCO", "GRASIM", "AMBUJACEM",
                               "ACC", "SHREECEM", "RAMCOCEM"],
    "Capital Goods":          ["SIEMENS", "ABB", "BHEL", "THERMAX",
                               "CUMMINSIND", "GMRINFRA"],
    "Real Estate":            ["DLF", "GODREJPROP", "PRESTIGE",
                               "OBEROIRLTY", "BRIGADE", "SOBHA"],
    "Consumer Discretionary": ["TITAN", "TRENT", "ABFRL",
                               "PAGEIND", "BATAINDIA"],
    "NBFCs":                  ["BAJFINANCE", "BAJAJFINSV", "CHOLAFIN",
                               "MUTHOOTFIN", "M&MFIN", "SHRIRAMFIN"],
    "Insurance":              ["HDFCLIFE", "SBILIFE", "ICICIGI", "LICI"],
    "Specialty Chemicals":    ["PIDILITIND", "DEEPAKNITR", "ATUL",
                               "NAVINFLUOR", "GALAXYSURF"],
    "Defence & Aerospace":    ["HAL", "BEL", "MIDHANI", "BEML", "COCHINSHIP"],
    "Railways & Infra":       ["RVNL", "IRCTC", "IRFC", "RAILTEL", "TITAGARH"],
    "Healthcare Services":    ["APOLLOHOSP", "FORTIS", "MAXHEALTH", "NARAYANA"],
    "Retail & E-Commerce":    ["DMART", "TRENT", "NYKAA", "ZOMATO"],
    "Telecom":                ["BHARTIARTL", "INDUSTOWER", "VODAFONEIDEA"],
    "Paints & Adhesives":     ["ASIANPAINT", "BERGERPAINTS", "KANSAINER", "PIDILITIND"],
}


# ── Data loader ───────────────────────────────────────────────────────────────

def _load_stocks(progress_callback=None) -> dict[str, pd.DataFrame]:
    dates  = _weekdays_back(300)
    total  = len(dates)
    frames = []
    for i, dt in enumerate(dates):
        df = _download_one_day(dt)
        if df is not None:
            frames.append(df)
        if progress_callback and i % 50 == 0:
            progress_callback(i, total, f"Loading data… {i}/{total} days")
    if not frames:
        return {}
    combined = pd.concat(frames, ignore_index=True).sort_values("Date")
    stocks: dict[str, pd.DataFrame] = {}
    for sym, grp in combined.groupby("Symbol"):
        g = grp.set_index("Date")[["Close", "Volume"]]
        g = g[~g.index.duplicated(keep="last")].sort_index()
        if len(g) >= 60:
            stocks[sym] = g
    return stocks


def _build_nifty(stocks: dict) -> pd.Series | None:
    closes = []
    for sym in _NIFTY_SYMS:
        df = stocks.get(sym)
        if df is not None and len(df) >= 63:
            closes.append(df["Close"].dropna())
    if not closes:
        return None
    combined = pd.concat(closes, axis=1).dropna(how="all")
    bench = combined.mean(axis=1)
    return bench if len(bench) >= 20 else None


# ── Group RS computation ──────────────────────────────────────────────────────

def _group_rs(stocks: dict, nifty: pd.Series) -> list[dict]:
    results = []
    for group_name, symbols in INDUSTRY_GROUPS.items():
        group_closes = []
        found_syms   = []
        for sym in symbols:
            df = stocks.get(sym)
            if df is not None and len(df) >= 63:
                group_closes.append(df["Close"].dropna())
                found_syms.append(sym)

        if len(group_closes) < 2:
            continue

        combined  = pd.concat(group_closes, axis=1).dropna(how="all")
        group_idx = combined.mean(axis=1)
        if len(group_idx) < 63:
            continue

        idx = group_idx.index.intersection(nifty.index)
        if len(idx) < 63:
            continue

        g = group_idx[idx]
        n = nifty[idx]

        r1m = round((float(g.iloc[-1]) / float(g.iloc[-22])  - 1) * 100, 2) if len(g) > 22  else 0.0
        r3m = round((float(g.iloc[-1]) / float(g.iloc[-63])  - 1) * 100, 2) if len(g) > 63  else 0.0
        r6m = round((float(g.iloc[-1]) / float(g.iloc[-126]) - 1) * 100, 2) if len(g) > 126 else 0.0

        nifty_r3m   = round((float(n.iloc[-1]) / float(n.iloc[-63]) - 1) * 100, 2) if len(n) > 63 else 0.0
        rs_vs_nifty = round(r3m - nifty_r3m, 2)

        results.append({
            "group":        group_name,
            "r1m":          r1m,
            "r3m":          r3m,
            "r6m":          r6m,
            "rs_vs_nifty":  rs_vs_nifty,
            "member_count": len(found_syms),
            "symbols":      found_syms,
        })

    results.sort(key=lambda x: x["rs_vs_nifty"], reverse=True)
    for i, r in enumerate(results):
        r["rank"] = i + 1
    return results


# ── Main entry ────────────────────────────────────────────────────────────────

def run_industry_analysis(progress_callback=None) -> dict:
    if _cache["data"] and time.time() - _cache["ts"] < CACHE_TTL:
        return _cache["data"]

    stocks = _load_stocks(progress_callback)
    if not stocks:
        return {"groups": [], "computed_at": int(time.time()), "total_groups": 0}

    if progress_callback:
        progress_callback(0, 1, "Computing industry group RS…")

    nifty = _build_nifty(stocks)
    if nifty is None:
        return {"groups": [], "computed_at": int(time.time()), "total_groups": 0}

    groups = _group_rs(stocks, nifty)

    out = {
        "groups":       groups,
        "computed_at":  int(time.time()),
        "total_groups": len(groups),
    }
    _cache["data"] = out
    _cache["ts"]   = time.time()

    if progress_callback:
        progress_callback(1, 1, f"Done — {len(groups)} industry groups ranked by RS")
    return out
