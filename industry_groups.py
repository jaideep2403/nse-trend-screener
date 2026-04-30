"""
Industry Group Relative Strength
Groups NSE stocks into 25 sectors and ranks each by 3-month RS vs Nifty.
Zero extra NSE API calls — uses bhavcopy cache only.
"""
import math
import time
import pandas as pd
from data_fetcher import _weekdays_back, _download_one_day

_cache        = {"data": None, "ts": 0}
_stocks_cache = {"data": None, "ts": 0}
CACHE_TTL     = 3600   # 1 hour
STOCKS_TTL    = 3600   # reuse loaded stocks for 1 hour

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


# ── Cached stock loader (shared between industry analysis + RRG) ──────────────

def _get_stocks(progress_callback=None) -> dict:
    if _stocks_cache["data"] and time.time() - _stocks_cache["ts"] < STOCKS_TTL:
        return _stocks_cache["data"]
    stocks = _load_stocks(progress_callback)
    _stocks_cache["data"] = stocks
    _stocks_cache["ts"]   = time.time()
    return stocks


# ── RRG computation ───────────────────────────────────────────────────────────

def _compute_rrg(stocks: dict, nifty: pd.Series) -> list[dict]:
    """
    Compute Relative Rotation Graph positions for each industry group.

    Algorithm (approximates JdK RRG):
    1. Build a daily price index for each group (equal-weight mean of constituents)
    2. Compute raw RS line = group_index / nifty_index
    3. Resample to weekly (last trading day of each ISO week)
    4. RS-Ratio  = weekly RS / 26-week rolling mean × 100  (>100 = outperforming)
    5. RS-Momentum = (RS-Ratio[t] / RS-Ratio[t-4]) × 100   (>100 = improving)
    6. Return the last 5 weekly (x=RS-Ratio, y=RS-Momentum) as a trail

    Quadrants (centred at 100, 100):
      Leading   : x≥100, y≥100   Weakening : x≥100, y<100
      Improving : x<100,  y≥100  Lagging   : x<100,  y<100
    """
    # Ensure nifty has a proper DatetimeIndex for weekly resampling
    nifty = nifty.copy()
    if not isinstance(nifty.index, pd.DatetimeIndex):
        nifty.index = pd.to_datetime(nifty.index)

    results = []

    for group_name, symbols in INDUSTRY_GROUPS.items():
        group_closes = []
        for sym in symbols:
            df = stocks.get(sym)
            if df is not None and len(df) >= 60:
                c = df["Close"].dropna()
                if not isinstance(c.index, pd.DatetimeIndex):
                    c.index = pd.to_datetime(c.index)
                group_closes.append(c)

        if len(group_closes) < 2:
            continue

        # Daily group price index
        combined  = pd.concat(group_closes, axis=1).dropna(how="all")
        group_idx = combined.mean(axis=1)

        # Align with nifty
        common = group_idx.index.intersection(nifty.index)
        if len(common) < 80:
            continue

        g = group_idx[common]
        n = nifty[common]

        # Raw relative-strength line
        rs_daily = g / n

        # Weekly resampling (ISO week, last bar)
        rs_weekly = rs_daily.resample("W").last().dropna()
        if len(rs_weekly) < 12:
            continue

        # RS-Ratio: normalise to 100 via 26-week rolling mean
        rolling_mean = rs_weekly.rolling(window=26, min_periods=10).mean()
        rs_ratio     = (rs_weekly / rolling_mean * 100).dropna()
        if len(rs_ratio) < 8:
            continue

        # RS-Momentum: 4-week rate of change of RS-Ratio, anchored at 100
        rs_mom = (rs_ratio / rs_ratio.shift(4) * 100).dropna()
        if len(rs_mom) < 3:
            continue

        # Build trail from last 5 common weekly points
        common_idx = rs_ratio.index.intersection(rs_mom.index)
        rs_ratio   = rs_ratio[common_idx]
        rs_mom     = rs_mom[common_idx]

        n_trail = min(5, len(rs_ratio))
        trail   = [
            {"x": round(float(rs_ratio.iloc[i]), 2),
             "y": round(float(rs_mom.iloc[i]),   2)}
            for i in range(-n_trail, 0)
        ]
        if not trail:
            continue

        def _quad(p):
            return ("Leading"   if p["x"] >= 100 and p["y"] >= 100 else
                    "Weakening" if p["x"] >= 100 and p["y"] <  100 else
                    "Improving" if p["x"] <  100 and p["y"] >= 100 else
                    "Lagging")

        cur          = trail[-1]
        quadrant     = _quad(cur)
        prev_quad    = _quad(trail[-2]) if len(trail) >= 2 else quadrant
        # Distance from RRG centre (100, 100) — proxy for "how deep in this quadrant"
        distance     = round(math.sqrt((cur["x"] - 100) ** 2 + (cur["y"] - 100) ** 2), 2)

        # Tail direction — vector from previous to current point
        if len(trail) >= 2:
            dx = cur["x"] - trail[-2]["x"]
            dy = cur["y"] - trail[-2]["y"]
            if   dx > 0.3 and dy > 0.3:    tail_dir = "↗ accelerating"
            elif dx > 0.3 and dy < -0.3:   tail_dir = "↘ losing momentum"
            elif dx < -0.3 and dy > 0.3:   tail_dir = "↖ recovering"
            elif dx < -0.3 and dy < -0.3:  tail_dir = "↙ deteriorating"
            elif dx > 0.3:                 tail_dir = "→ stronger RS"
            elif dx < -0.3:                tail_dir = "← weaker RS"
            elif dy > 0.3:                 tail_dir = "↑ momentum up"
            elif dy < -0.3:                tail_dir = "↓ momentum down"
            else:                          tail_dir = "• stable"
        else:
            tail_dir = "—"

        # Pattern — quadrant transition tells the story
        if quadrant == "Leading"   and prev_quad == "Improving": pattern = "🚀 Fresh entry to Leading"
        elif quadrant == "Weakening" and prev_quad == "Leading":   pattern = "⚠ Topping out"
        elif quadrant == "Lagging"   and prev_quad == "Weakening": pattern = "📉 Deepening weakness"
        elif quadrant == "Improving" and prev_quad == "Lagging":   pattern = "🌱 Bottoming"
        elif quadrant == "Leading":                                pattern = "✓ Holding strength"
        elif quadrant == "Improving":                              pattern = "📈 Building strength"
        elif quadrant == "Weakening":                              pattern = "⚠ Losing strength"
        else:                                                      pattern = "❌ Weak"

        # Top 3 stocks in this sector by 3-month return
        sector_returns = []
        for sym in symbols:
            df = stocks.get(sym)
            if df is not None:
                c = df["Close"].dropna()
                if len(c) >= 66:
                    r3m_val = (float(c.iloc[-1]) / float(c.iloc[-66]) - 1) * 100
                    sector_returns.append({"symbol": sym, "r3m": round(r3m_val, 1)})
        sector_returns.sort(key=lambda x: -x["r3m"])
        top_stocks = sector_returns[:3]

        results.append({
            "group":          group_name,
            "trail":          trail,
            "quadrant":       quadrant,
            "prev_quadrant":  prev_quad,
            "rs_ratio":       cur["x"],
            "rs_momentum":    cur["y"],
            "distance":       distance,
            "tail_direction": tail_dir,
            "pattern":        pattern,
            "top_stocks":     top_stocks,
        })

    # Sort for deterministic rendering (Leading first, then Weakening, Improving, Lagging)
    order = {"Leading": 0, "Weakening": 1, "Improving": 2, "Lagging": 3}
    results.sort(key=lambda r: order.get(r["quadrant"], 4))
    return results


def _build_action_buckets(rrg_results: list) -> dict:
    """
    Categorise sectors into 4 action buckets:
      🟢 BUY        — fresh entry into Leading (Improving → Leading transition)
      🔵 ACCUMULATE — Leading or Improving with momentum > 100 (still strengthening)
      🟡 TRIM       — Weakening, or Leading with falling momentum
      🔴 AVOID      — Lagging, or Improving with falling momentum
    """
    buy, accumulate, trim, avoid = [], [], [], []

    for r in rrg_results:
        q    = r["quadrant"]
        prev = r["prev_quadrant"]
        mom  = r["rs_momentum"]

        if q == "Leading" and prev == "Improving":
            buy.append(r)
        elif q == "Improving" and prev == "Lagging":
            buy.append(r)            # bottoming — early entry
        elif q == "Leading" and mom >= 100:
            accumulate.append(r)
        elif q == "Improving" and mom >= 100:
            accumulate.append(r)
        elif q == "Weakening":
            trim.append(r)
        elif q == "Leading" and mom < 100:
            trim.append(r)            # losing momentum at the top
        elif q == "Lagging":
            avoid.append(r)
        elif q == "Improving" and mom < 100:
            avoid.append(r)            # weak and not building
        else:
            avoid.append(r)            # safe default

    # Sort each bucket: most-actionable first (highest RS-momentum for buy/accumulate,
    # furthest distance from centre for trim/avoid → most extreme)
    buy.sort(key=lambda r: -r["rs_momentum"])
    accumulate.sort(key=lambda r: -r["rs_momentum"])
    trim.sort(key=lambda r: -r["distance"])
    avoid.sort(key=lambda r: -r["distance"])

    return {"buy": buy, "accumulate": accumulate, "trim": trim, "avoid": avoid}


def run_rrg_analysis() -> dict:
    """Load stocks (from cache) and compute RRG data. Zero new NSE API calls."""
    stocks = _get_stocks()
    if not stocks:
        return {"sectors": [], "actions": {}, "computed_at": int(time.time())}
    nifty = _build_nifty(stocks)
    if nifty is None:
        return {"sectors": [], "actions": {}, "computed_at": int(time.time())}
    sectors = _compute_rrg(stocks, nifty)
    actions = _build_action_buckets(sectors)
    return {"sectors": sectors, "actions": actions, "computed_at": int(time.time())}


# ── Main entry ────────────────────────────────────────────────────────────────

def run_industry_analysis(progress_callback=None) -> dict:
    if _cache["data"] and time.time() - _cache["ts"] < CACHE_TTL:
        return _cache["data"]

    stocks = _get_stocks(progress_callback)
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
