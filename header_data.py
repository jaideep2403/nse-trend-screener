"""
Lightweight market header — Nifty 50 level + Adv/Dec + market trend.

Reads only the latest 2 bhavcopy pkl files, no full stock universe load.
Cached for 30 seconds so the header can poll cheaply every minute.
"""
import io
import os
import pickle
import time
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import requests

BHAV_DIR = Path(os.getenv("BHAV_DIR", "/tmp/nse_bhav_days"))
DATA_DIR = Path(os.getenv("DATA_DIR", os.path.dirname(__file__) or "."))

_NIFTY50_URL = "https://archives.nseindia.com/content/indices/ind_nifty50list.csv"
_NIFTY50_CACHE = DATA_DIR / ".nifty50_cache.pkl"
_NIFTY50_TTL   = 7 * 86400  # weekly

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 Chrome/122.0.0.0 Safari/537.36"
    ),
}

# Hardcoded fallback (current Nifty 50 constituents)
_NIFTY50_FALLBACK = [
    "ADANIENT","ADANIPORTS","APOLLOHOSP","ASIANPAINT","AXISBANK",
    "BAJAJ-AUTO","BAJAJFINSV","BAJFINANCE","BEL","BHARTIARTL",
    "CIPLA","COALINDIA","DRREDDY","EICHERMOT","GRASIM",
    "HCLTECH","HDFCBANK","HDFCLIFE","HEROMOTOCO","HINDALCO",
    "HINDUNILVR","ICICIBANK","INDUSINDBK","INFY","ITC",
    "JIOFIN","JSWSTEEL","KOTAKBANK","LT","M&M",
    "MARUTI","NESTLEIND","NTPC","ONGC","POWERGRID",
    "RELIANCE","SBILIFE","SBIN","SHRIRAMFIN","SUNPHARMA",
    "TATACONSUM","TATAMOTORS","TATASTEEL","TCS","TECHM",
    "TITAN","TRENT","ULTRACEMCO","WIPRO","ZOMATO",
]

_cache = {"data": None, "ts": 0.0}
_CACHE_TTL = 30  # seconds — only caches the heavy bhavcopy read; trend re-checked each call


def _load_nifty50_symbols() -> list[str]:
    """Load Nifty 50 constituents (cached weekly)."""
    try:
        if _NIFTY50_CACHE.exists():
            with open(_NIFTY50_CACHE, "rb") as f:
                d = pickle.load(f)
            if time.time() - d["ts"] < _NIFTY50_TTL and d.get("symbols"):
                return d["symbols"]
    except Exception:
        pass
    try:
        r = requests.get(_NIFTY50_URL, headers=_HEADERS, timeout=10)
        if r.status_code == 200:
            df = pd.read_csv(io.BytesIO(r.content))
            syms = df["Symbol"].dropna().str.strip().tolist()
            if len(syms) >= 40:
                _NIFTY50_CACHE.parent.mkdir(parents=True, exist_ok=True)
                with open(_NIFTY50_CACHE, "wb") as f:
                    pickle.dump({"ts": time.time(), "symbols": syms}, f)
                return syms
    except Exception:
        pass
    return list(_NIFTY50_FALLBACK)


def _list_bhav_pkls() -> list[Path]:
    """All bhavcopy pkls sorted ascending (oldest first)."""
    if not BHAV_DIR.exists():
        return []
    return sorted(BHAV_DIR.glob("*.pkl"))


def _load_bhav(p: Path) -> pd.DataFrame | None:
    try:
        with open(p, "rb") as f:
            return pickle.load(f)
    except Exception:
        return None


def _trend_label(change_pct: float | None, adr: float | None) -> tuple[str, str]:
    """
    Classify market state from today's % change + adv/decl ratio.
    Returns (label, color_key).
    """
    if change_pct is None or adr is None:
        return ("—", "muted")
    # Strong upmove
    if change_pct >= 0.5 and adr >= 1.5:
        return ("Strong Upmove", "green")
    if change_pct >= 0.2 and adr >= 1.0:
        return ("Upmove", "green")
    # Correction
    if change_pct <= -0.5 and adr <= 0.7:
        return ("Correction", "red")
    if change_pct <= -0.2 and adr < 1.0:
        return ("Weakness", "red")
    # Sideways everything else
    return ("Sideways", "yellow")


def _resolve_trend(change_pct, adr) -> tuple[str, str, str]:
    """
    Return (trend_label, color_key, source). ONLY uses Market Breadth's signal.
    Quick adv/dec estimate is no longer shown — it conflicted with the breadth
    tab. If breadth not yet computed, returns 'Computing…' so the user knows
    the system is working on the authoritative signal.
    """
    try:
        from market_breadth import (_cache as mb_cache,
                                    PERSISTED_CACHE_TTL as MB_TTL)
        if (mb_cache.get("data")
                and (time.time() - mb_cache.get("ts", 0)) < MB_TTL
                and mb_cache["data"].get("timing")):
            t = mb_cache["data"]["timing"]
            color_map = {"pos": "green", "neutral": "yellow", "neg": "red"}
            # Prefer smoothed status (5d MA) if available — more stable
            label = t.get("smoothed_status") or t.get("status", "—")
            cls   = t.get("smoothed_cls")    or t.get("cls", "")
            return (label, color_map.get(cls, "muted"), "breadth")
    except Exception:
        pass
    return ("Computing…", "muted", "computing")


def get_market_header() -> dict:
    """
    Compute Nifty 50 level, day change %, advances, declines, trend label.
    Reads only 2 latest bhavcopy pkls — fast (<300ms typical).
    """
    # Use cached heavy data (bhavcopy read) but ALWAYS re-resolve trend so it
    # picks up the latest Market Breadth result the moment that scan finishes.
    if _cache["data"] and (time.time() - _cache["ts"] < _CACHE_TTL):
        cached = dict(_cache["data"])
        trend, tcolor, tsrc = _resolve_trend(cached.get("change_pct"), cached.get("adv_decl_ratio"))
        cached["trend"] = trend
        cached["trend_color"] = tcolor
        cached["trend_source"] = tsrc
        return cached

    pkls = _list_bhav_pkls()
    if len(pkls) < 2:
        return {
            "error": "insufficient_bhavcopy",
            "nifty": None, "change_pct": None,
            "advances": None, "declines": None,
            "trend": "—", "trend_color": "muted",
            "computed_at": int(time.time()),
        }

    today_df = _load_bhav(pkls[-1])
    prev_df  = _load_bhav(pkls[-2])
    if today_df is None or prev_df is None:
        return {"error": "bhavcopy_load_failed", "computed_at": int(time.time())}

    # ── Adv/Dec across whole universe ──
    today  = today_df.set_index("Symbol")["Close"]
    prev   = prev_df.set_index("Symbol")["Close"]
    common = today.index.intersection(prev.index)
    chg    = (today.loc[common] - prev.loc[common]) / prev.loc[common]
    advances = int((chg > 0).sum())
    declines = int((chg < 0).sum())
    unchanged = int((chg == 0).sum())
    total = len(common)
    adr = round(advances / declines, 2) if declines > 0 else None

    # ── Nifty 50 level (equal-weighted average of 50 constituents) ──
    n50 = _load_nifty50_symbols()
    n50_today = today.reindex(n50).dropna()
    n50_prev  = prev.reindex(n50).dropna()
    common50  = n50_today.index.intersection(n50_prev.index)

    nifty_level = None
    change_pct  = None
    if len(common50) >= 30:  # need at least 30 of 50 for reliability
        # Equal-weighted index: scaled to look ~Nifty-like (use ratio change)
        avg_today = float(n50_today.loc[common50].mean())
        avg_prev  = float(n50_prev.loc[common50].mean())
        change_pct = round((avg_today / avg_prev - 1) * 100, 2)
        # Display value: show today's avg constituent price (a proxy level)
        nifty_level = round(avg_today, 2)

    trend, trend_color, trend_source = _resolve_trend(change_pct, adr)

    # Date
    date_str = pkls[-1].stem  # YYYYMMDD
    try:
        d = date(int(date_str[:4]), int(date_str[4:6]), int(date_str[6:8]))
        date_label = d.strftime("%d-%b-%Y")
    except Exception:
        date_label = date_str

    out = {
        "nifty":         nifty_level,
        "change_pct":    change_pct,
        "advances":      advances,
        "declines":      declines,
        "unchanged":     unchanged,
        "total":         total,
        "adv_decl_ratio": adr,
        "trend":         trend,
        "trend_color":   trend_color,
        "trend_source":  trend_source,
        "trading_date":  date_label,
        "components_used": int(len(common50)),
        "computed_at":   int(time.time()),
    }

    _cache["data"] = out
    _cache["ts"]   = time.time()
    return out
