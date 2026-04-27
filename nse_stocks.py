"""
NSE stock universe — Nifty 50 + Nifty Next 50 + Nifty 500 + Nifty Smallcap 250.

Downloads 4 index CSVs from NSE archives (no auth, no rate limits).
Cached locally for 7 days so we never hammer NSE on every startup.

Combined unique universe ≈ 500 stocks (Nifty Smallcap 250 is the smallcap
tier *inside* Nifty 500, so the 4 lists deduplicate to ~500 stocks).
"""

import os
import time
import pickle
import requests
import io
import pandas as pd

_CACHE_PATH = os.path.join(os.environ.get("DATA_DIR", os.path.dirname(__file__)), ".nse_universe_cache.pkl")
_CACHE_TTL  = 7 * 86400   # refresh weekly

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,*/*",
}

# All 4 index CSV URLs — publicly available on NSE archives, no login needed
_INDEX_URLS = {
    "Nifty50":          "https://archives.nseindia.com/content/indices/ind_nifty50list.csv",
    "NiftyNext50":      "https://archives.nseindia.com/content/indices/ind_niftynext50list.csv",
    "Nifty500":         "https://archives.nseindia.com/content/indices/ind_nifty500list.csv",
    "NiftySmallcap250": "https://archives.nseindia.com/content/indices/ind_niftysmallcap250list.csv",
}

# Fallback Nifty 100 hardcoded — used only if ALL 4 NSE downloads fail
_FALLBACK = [
    "ADANIENT","ADANIPORTS","APOLLOHOSP","ASIANPAINT","AXISBANK",
    "BAJAJ-AUTO","BAJAJFINSV","BAJFINANCE","BHARTIARTL","BPCL",
    "BRITANNIA","CIPLA","COALINDIA","DIVISLAB","DLF",
    "DRREDDY","EICHERMOT","GRASIM","HCLTECH","HDFCBANK",
    "HDFCLIFE","HEROMOTOCO","HINDALCO","HINDUNILVR","ICICIBANK",
    "ICICIGI","INDUSINDBK","INFOSYS","ITC","JSWSTEEL",
    "KOTAKBANK","LT","M&M","MARUTI","NESTLEIND",
    "NTPC","ONGC","POWERGRID","RELIANCE","SBILIFE",
    "SBIN","SHRIRAMFIN","SUNPHARMA","TATAMOTORS","TATASTEEL",
    "TCS","TECHM","TITAN","TVSMOTOR","ULTRACEMCO","WIPRO",
    "ABB","AMBUJACEM","AUBANK","BAJAJHFL","BANKBARODA",
    "BEL","BERGEPAINT","BOSCHLTD","BSE","CANBK",
    "CGPOWER","CHOLAFIN","COFORGE","COLPAL","CUMMINSIND",
    "DABUR","DMART","FEDERALBNK","FORTIS","GODREJCP",
    "GODREJPROP","HAL","HAVELLS","ICICIAMC","INDUSTOWER",
    "IRCTC","JINDALSTEL","JUBLFOOD","LICI","LODHA",
    "LTTS","LUPIN","M&MFIN","MARICO","MAXHEALTH",
    "MPHASIS","NAUKRI","OBEROIRLTY","OFSS","PFC",
    "PERSISTENT","PIDILITIND","RECLTD","SCHAEFFLER","SIEMENS",
    "TATACHEM","TATACONSUM","TIINDIA","TORNTPHARM","TRENT",
    "VEDL","ZOMATO",
]


# ── Cache helpers ──────────────────────────────────────────────────────────────

def _load_cache() -> list[str] | None:
    try:
        if os.path.exists(_CACHE_PATH):
            with open(_CACHE_PATH, "rb") as f:
                data = pickle.load(f)
            if time.time() - data["ts"] < _CACHE_TTL:
                return data["symbols"]
    except Exception:
        pass
    return None


def _save_cache(symbols: list[str]) -> None:
    try:
        with open(_CACHE_PATH, "wb") as f:
            pickle.dump({"ts": time.time(), "symbols": symbols}, f)
    except Exception:
        pass


# ── Downloader ─────────────────────────────────────────────────────────────────

def _fetch_index(name: str, url: str) -> list[str]:
    """Download one NSE index CSV. Returns [] on any failure."""
    try:
        r = requests.get(url, headers=_HEADERS, timeout=15)
        if r.status_code != 200:
            return []
        df = pd.read_csv(io.BytesIO(r.content))
        if "Symbol" not in df.columns:
            return []
        return df["Symbol"].dropna().str.strip().tolist()
    except Exception:
        return []


def _build_universe() -> list[str]:
    """
    Download all 4 index lists, deduplicate, return sorted symbol list.
    Uses whatever indices are available — at least one must succeed.
    """
    seen: dict[str, None] = {}   # ordered-set via dict
    fetched_any = False

    for name, url in _INDEX_URLS.items():
        syms = _fetch_index(name, url)
        if syms:
            fetched_any = True
            for s in syms:
                seen[s] = None   # dedup, preserve first-seen order

    if not fetched_any:
        return list(_FALLBACK)

    return list(seen.keys())


# ── Public API ─────────────────────────────────────────────────────────────────

def get_nifty500_symbols() -> list[str]:
    """
    Return deduplicated universe: Nifty50 ∪ NiftyNext50 ∪ Nifty500 ∪ NiftySmallcap250.
    Uses local 7-day cache. Falls back to hardcoded Nifty100 if NSE unreachable.
    """
    cached = _load_cache()
    if cached:
        return cached

    universe = _build_universe()
    if universe:
        _save_cache(universe)
    return universe


def get_nse_tickers() -> list[str]:
    """
    Return universe in .NS suffix format used by all scanners and the screener.
    """
    return [f"{s}.NS" for s in get_nifty500_symbols()]
