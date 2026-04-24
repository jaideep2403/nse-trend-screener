"""
Market Breadth & Timing Dashboard
- % NSE stocks above 50-day / 200-day MA (breadth)
- New 52-week highs vs lows
- Advance / Decline ratio
- Nifty50 trend status + Stage
- Distribution Day Count on Nifty (IBD method)
- Market timing signal: Bull / Correction / Bear

Data: NSE bhavcopy — zero Yahoo calls
"""
import time
import numpy as np
import pandas as pd
from data_fetcher import _weekdays_back, _download_one_day
from analysis_utils import stage_analysis, stage_label

_NIFTY50_SYMS = [
    "RELIANCE", "TCS", "HDFCBANK", "ICICIBANK", "INFOSYS",
    "HINDUNILVR", "SBIN", "BHARTIARTL", "ITC", "KOTAKBANK",
    "AXISBANK", "WIPRO", "HCLTECH", "MARUTI", "BAJFINANCE",
    "TITAN", "NTPC", "POWERGRID", "NESTLEIND", "SUNPHARMA",
    "ADANIENT", "ADANIPORTS", "LTIM", "BAJAJFINSV", "EICHERMOT",
    "TECHM", "TATASTEEL", "JSWSTEEL", "ULTRACEMCO", "GRASIM",
]

MIN_BARS = 60
_cache   = {"data": None, "ts": 0}
CACHE_TTL = 1800   # 30 min


# ── Data loader ───────────────────────────────────────────────────────────────

def _load_all_stocks(progress_callback=None) -> dict[str, pd.DataFrame]:
    dates  = _weekdays_back(400)
    total  = len(dates)
    frames = []
    for i, dt in enumerate(dates):
        df = _download_one_day(dt)
        if df is not None:
            frames.append(df)
        if progress_callback and i % 50 == 0:
            progress_callback(i, total, f"Loading bhavcopy… {i}/{total} days")
    if not frames:
        return {}
    combined = pd.concat(frames, ignore_index=True).sort_values("Date")
    stocks = {}
    for sym, grp in combined.groupby("Symbol"):
        g = grp.set_index("Date")[["Open", "High", "Low", "Close", "Volume"]]
        g = g[~g.index.duplicated(keep="last")].sort_index()
        if len(g) >= MIN_BARS:
            stocks[sym] = g
    return stocks


def _build_nifty(stocks: dict) -> pd.Series | None:
    closes = []
    for sym in _NIFTY50_SYMS:
        df = stocks.get(sym)
        if df is not None and len(df) >= 63:
            closes.append(df["Close"].dropna())
    if not closes:
        return None
    combined = pd.concat(closes, axis=1).dropna(how="all")
    bench = combined.mean(axis=1)
    return bench if len(bench) >= 20 else None


# ── Breadth metrics ───────────────────────────────────────────────────────────

def _compute_breadth(stocks: dict) -> dict:
    """Compute breadth metrics across all stocks."""
    above50 = above200 = new_highs = new_lows = up_today = down_today = total = 0

    for sym, df in stocks.items():
        cl = df["Close"].dropna()
        if len(cl) < 22:
            continue
        total += 1
        cur = float(cl.iloc[-1])
        prev = float(cl.iloc[-2]) if len(cl) >= 2 else cur

        # MA breadth
        if len(cl) >= 52:
            ma50 = float(cl.rolling(50).mean().iloc[-1])
            if cur > ma50:
                above50 += 1
        if len(cl) >= 210:
            ma200 = float(cl.rolling(200).mean().iloc[-1])
            if cur > ma200:
                above200 += 1

        # 52-week highs/lows
        lookback = min(252, len(cl))
        hi52 = float(cl.iloc[-lookback:].max())
        lo52 = float(cl.iloc[-lookback:].min())
        if cur >= hi52 * 0.995:
            new_highs += 1
        if cur <= lo52 * 1.005:
            new_lows += 1

        # Advance/Decline today
        if cur > prev:
            up_today += 1
        elif cur < prev:
            down_today += 1

    pct50  = round(above50  / total * 100, 1) if total else 0
    pct200 = round(above200 / total * 100, 1) if total else 0
    adr    = round(up_today / down_today, 2) if down_today else 9.9

    return {
        "total_stocks": total,
        "pct_above_50ma":  pct50,
        "pct_above_200ma": pct200,
        "new_highs":   new_highs,
        "new_lows":    new_lows,
        "advance":     up_today,
        "decline":     down_today,
        "adv_decl_ratio": adr,
    }


def _distribution_days(nifty: pd.Series) -> int:
    """
    Count Nifty distribution days (close DOWN on higher volume than prior day)
    in the last 25 sessions. IBD counts ≥ 5 as a market under pressure.
    Note: we approximate with price alone since we don't have Nifty futures volume.
    We use the equal-weighted index change directly.
    """
    try:
        if len(nifty) < 26:
            return 0
        pct_chg  = nifty.pct_change().iloc[-25:]
        down_big = (pct_chg < -0.002).sum()   # > 0.2% drop = distribution day
        return int(down_big)
    except Exception:
        return 0


def _market_timing_signal(breadth: dict, dist_days: int, nifty_stage: int) -> dict:
    """
    Returns market timing status: Bull / Caution / Correction / Bear
    """
    p50  = breadth["pct_above_50ma"]
    p200 = breadth["pct_above_200ma"]
    hl_ratio = breadth["new_highs"] / (breadth["new_highs"] + breadth["new_lows"] + 1)

    score = 0
    # Breadth
    if p50  >= 60: score += 2
    elif p50 >= 40: score += 1
    if p200 >= 55: score += 2
    elif p200 >= 35: score += 1
    # Highs/Lows
    if hl_ratio >= 0.7: score += 2
    elif hl_ratio >= 0.5: score += 1
    # Distribution days
    if dist_days <= 3:   score += 2
    elif dist_days <= 5: score += 1
    # Nifty stage
    if nifty_stage == 2:  score += 2
    elif nifty_stage == 1: score += 1

    if score >= 8:
        status = "Bull Market"; cls = "pos"
    elif score >= 5:
        status = "Uptrend (Caution)"; cls = "neutral"
    elif score >= 3:
        status = "Correction"; cls = "neg"
    else:
        status = "Bear Market"; cls = "neg"

    return {"status": status, "score": score, "max": 10, "cls": cls}


# ── Historical breadth trend (last 10 weeks) ─────────────────────────────────

def _breadth_trend(stocks: dict) -> list[dict]:
    """
    Returns weekly % above 50-day MA for last 10 weeks.
    Used for a sparkline / trend chart.
    """
    weeks = []
    for w in range(10, 0, -1):
        above = total = 0
        for sym, df in stocks.items():
            cl = df["Close"].dropna()
            if len(cl) < 55 + w * 5:
                continue
            idx   = -(w * 5)
            cur   = float(cl.iloc[idx])
            ma50  = float(cl.iloc[:idx].rolling(50).mean().iloc[-1]) if len(cl[:idx]) >= 50 else 0
            if ma50 > 0:
                total += 1
                if cur > ma50:
                    above += 1
        pct = round(above / total * 100, 1) if total else 0
        weeks.append({"week": f"-{w}w", "pct_above_50": pct})
    return weeks


# ── Top 10 stocks hitting 52-week highs today ─────────────────────────────────

def _new_high_stocks(stocks: dict) -> list[dict]:
    results = []
    for sym, df in stocks.items():
        cl  = df["Close"].dropna()
        vol = df["Volume"].dropna()
        if len(cl) < 63:
            continue
        cur  = float(cl.iloc[-1])
        lookback = min(252, len(cl))
        hi52 = float(cl.iloc[-lookback:].max())
        if cur < hi52 * 0.995:
            continue
        avg_vol = float(vol.rolling(20).mean().iloc[-1]) if len(vol) >= 20 else 0
        adtv_cr = round(avg_vol * cur / 1e7, 2)
        if adtv_cr < 0.5:
            continue
        vr = round(float(vol.iloc[-1]) / avg_vol, 2) if avg_vol > 0 else 1.0
        r3m = round((cur / float(cl.iloc[-63]) - 1) * 100, 2) if len(cl) > 63 else 0.0
        results.append({
            "symbol":    sym,
            "price":     round(cur, 2),
            "adtv_cr":   adtv_cr,
            "vol_ratio": vr,
            "r3m":       r3m,
            "pct_ath":   0.0,
        })
    results.sort(key=lambda x: x["vol_ratio"], reverse=True)
    return results[:20]


# ── Follow-Through Day (IBD) ──────────────────────────────────────────────────

def _follow_through_day(nifty: pd.Series) -> dict:
    """
    IBD Follow-Through Day detection.
    After a market decline (≥8%), a FTD is a Nifty move ≥1.7% on day 4+ of rally.
    Signals potential new market uptrend beginning.
    """
    result = {
        "rally_attempt": False, "days_since_trough": 0,
        "trough_to_now_pct": 0.0, "ftd_today": False,
        "today_chg_pct": 0.0, "ftd_in_window": False,
    }
    try:
        if len(nifty) < 30:
            return result
        window     = nifty.iloc[-60:]
        peak_val   = float(window.max())
        trough_val = float(window.min())
        cur        = float(nifty.iloc[-1])
        if peak_val <= 0 or (peak_val - trough_val) / peak_val < 0.08:
            return result
        if cur <= trough_val * 1.01:
            return result
        trough_loc       = int(window.values.argmin())
        days_from_trough = len(window) - 1 - trough_loc
        result["rally_attempt"]      = True
        result["days_since_trough"]  = days_from_trough
        result["trough_to_now_pct"]  = round((cur - trough_val) / trough_val * 100, 2)
        if len(nifty) >= 2 and float(nifty.iloc[-2]) > 0:
            today_chg = (float(nifty.iloc[-1]) - float(nifty.iloc[-2])) / float(nifty.iloc[-2]) * 100
            result["today_chg_pct"] = round(today_chg, 2)
            result["ftd_today"]     = (days_from_trough >= 3 and today_chg >= 1.7)
        for i in range(-10, 0):
            try:
                prev = float(nifty.iloc[i - 1])
                if prev <= 0:
                    continue
                if (float(nifty.iloc[i]) - prev) / prev * 100 >= 1.7:
                    result["ftd_in_window"] = True
                    break
            except Exception:
                continue
    except Exception:
        pass
    return result


# ── Main entry ────────────────────────────────────────────────────────────────

def run_market_breadth(progress_callback=None) -> dict:
    if (_cache["data"]
            and time.time() - _cache["ts"] < CACHE_TTL):
        return _cache["data"]

    stocks = _load_all_stocks(progress_callback)
    if not stocks:
        return {"error": "No data available"}

    if progress_callback:
        progress_callback(0, 1, "Computing market breadth…")

    nifty       = _build_nifty(stocks)
    breadth     = _compute_breadth(stocks)
    dist_days   = _distribution_days(nifty) if nifty is not None else 0
    nifty_stage = stage_analysis(nifty) if nifty is not None else 0
    timing      = _market_timing_signal(breadth, dist_days, nifty_stage)
    new_highs_l = _new_high_stocks(stocks)
    ftd         = _follow_through_day(nifty) if nifty is not None else {}

    # Nifty returns
    nifty_r1m = nifty_r3m = nifty_r6m = None
    if nifty is not None and len(nifty) > 22:
        nifty_r1m = round((float(nifty.iloc[-1]) / float(nifty.iloc[-22]) - 1) * 100, 2)
    if nifty is not None and len(nifty) > 63:
        nifty_r3m = round((float(nifty.iloc[-1]) / float(nifty.iloc[-63]) - 1) * 100, 2)
    if nifty is not None and len(nifty) > 126:
        nifty_r6m = round((float(nifty.iloc[-1]) / float(nifty.iloc[-126]) - 1) * 100, 2)

    out = {
        "breadth":       breadth,
        "dist_days":     dist_days,
        "nifty_stage":   nifty_stage,
        "nifty_stage_label": stage_label(nifty_stage),
        "nifty_r1m":     nifty_r1m,
        "nifty_r3m":     nifty_r3m,
        "nifty_r6m":     nifty_r6m,
        "timing":        timing,
        "ftd":           ftd,
        "new_highs_list": new_highs_l,
        "computed_at":   int(time.time()),
    }

    _cache["data"] = out
    _cache["ts"]   = time.time()

    if progress_callback:
        progress_callback(1, 1, "Market breadth computed")
    return out
