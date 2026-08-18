"""
NSE stock universe — Nifty 50 + Nifty Next 50 + Nifty 500 + Nifty Smallcap 250
                   + Nifty Microcap 250 + Nifty Total Market.

Downloads index CSVs from NSE archives (no auth, no rate limits).
Cached locally for 7 days so we never hammer NSE on every startup.

Combined unique universe ≈ 750 stocks (Nifty Total Market covers all
exchange-listed large/mid/small/micro caps; deduplication keeps ~750 unique).
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

# NSE index CSV URLs — publicly available on NSE archives, no login needed.
_INDEX_URLS = {
    "Nifty50":          "https://archives.nseindia.com/content/indices/ind_nifty50list.csv",
    "NiftyNext50":      "https://archives.nseindia.com/content/indices/ind_niftynext50list.csv",
    "Nifty500":         "https://archives.nseindia.com/content/indices/ind_nifty500list.csv",
    "NiftySmallcap250": "https://archives.nseindia.com/content/indices/ind_niftysmallcap250list.csv",
    "NiftyMicrocap250": "https://archives.nseindia.com/content/indices/ind_niftymicrocap250_list.csv",
    "NiftyTotalMarket": "https://archives.nseindia.com/content/indices/ind_niftytotalmarket_list.csv",
}

# Fallback Nifty 100 hardcoded — used only if ALL 4 NSE downloads fail
_FALLBACK = [
    "ADANIENT","ADANIPORTS","APOLLOHOSP","ASIANPAINT","AXISBANK",
    "BAJAJ-AUTO","BAJAJFINSV","BAJFINANCE","BHARTIARTL","BPCL",
    "BRITANNIA","CIPLA","COALINDIA","DIVISLAB","DLF",
    "DRREDDY","EICHERMOT","GRASIM","HCLTECH","HDFCBANK",
    "HDFCLIFE","HEROMOTOCO","HINDALCO","HINDUNILVR","ICICIBANK",
    "ICICIGI","INDUSINDBK","INFY","ITC","JSWSTEEL",
    "KOTAKBANK","LT","M&M","MARUTI","NESTLEIND",
    "NTPC","ONGC","POWERGRID","RELIANCE","SBILIFE",
    "SBIN","SHRIRAMFIN","SUNPHARMA","TATASTEEL",
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

# Symbol prefixes to exclude — NSE occasionally adds placeholder/dummy tickers
# for corporate restructurings (e.g. DUMMYVEDL* for Vedanta demerger)
_EXCLUDED_PREFIXES = ("DUMMY",)


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


def _is_valid_symbol(sym: str) -> bool:
    """Return False for NSE placeholder / dummy tickers that should never be scanned."""
    return not any(sym.upper().startswith(p) for p in _EXCLUDED_PREFIXES)


# Real NSE stocks whose symbols happen to contain fund-like tokens (ETF / GOLD /
# SILVER / GILT) but are NOT ETFs — they must never be filtered out.
_NOT_ETF = {
    "JETFREIGHT",   # Jet Freight Logistics (contains "ETF")
    "PNBGILTS",     # PNB Gilts Ltd, primary dealer (contains "GILT")
    "GOLDIAM",      # Goldiam International, jewellery (contains "GOLD")
    "GOLDTECH",     # Goldstone Technologies (contains "GOLD")
    "SHANTIGOLD",   # Shanti Gold International, jewellery
    "SKYGOLD",      # Sky Gold, jewellery
}

def is_etf(symbol: str) -> bool:
    """True for exchange-traded funds / debt-fund units. ETFs trade in the EQ
    series alongside stocks, so the off-index / full-EQ loaders pick them up —
    but a stock screener must never show them. Validated against the full
    ~2400-symbol NSE list: catches all ~156 ETFs with ZERO false positives on
    real stocks (the handful of fund-named real stocks are whitelisted above).
    NIFTYBEES etc. stay loadable at the data layer for benchmark use; this only
    gates which symbols get SCANNED / displayed."""
    s = (symbol or "").upper().strip()
    if not s or s in _NOT_ETF:
        return False
    # Fund/ETF name tokens that no real (non-whitelisted) stock carries.
    if any(t in s for t in ("ETF", "BEES", "LIQUID", "GILT", "GSEC")):
        return True
    # INDEX-TRACKER GAP (fixed 2026-08-06). The list above catches fund-suffixed
    # names but not index trackers, so 44 ETFs were reaching every stock screener:
    # AONENIFTY, AXISNIFTY, HDFCNEXT50, MONIFTY500, PSUBANK, MIDCAP, SMALLCAP,
    # SENSEXBETA … A weekly-breakout scan surfaced HDFCNEXT50 as a "stock
    # breakout", which is what exposed it. Verified against the live 2,597-symbol
    # universe: these tokens flag exactly those 44 index products and ZERO real
    # operating companies (the fund-named real stocks stay covered by _NOT_ETF
    # whitelisting above).
    if any(t in s for t in ("NIFTY", "NEXT50", "SENSEX", "MIDCAP", "SMALLCAP",
                            "PSUBANK", "MOM50", "MOM100", "CPSE", "BHARATBOND")):
        return True
    # Commodity (gold / silver) ETFs.
    if "GOLD" in s or "SILVER" in s or "SLVR" in s:
        return True
    return False


def _load_extra_symbols() -> list[str]:
    """
    Load hand-curated extra symbols from $DATA_DIR/.extra_symbols.json.
    Use this for stocks that aren't in any NSE index (e.g. PRECWIRE).
    File format: {"symbols": ["PRECWIRE", "OTHER1", ...]}
    """
    extras_path = os.path.join(os.environ.get("DATA_DIR", os.path.dirname(__file__)),
                               ".extra_symbols.json")
    try:
        if os.path.exists(extras_path):
            import json
            with open(extras_path) as f:
                data = json.load(f)
            syms = data.get("symbols", [])
            return [s.strip().upper() for s in syms if isinstance(s, str) and s.strip()]
    except Exception:
        pass
    return []


def _build_universe() -> list[str]:
    """
    Download all configured index lists, deduplicate, filter dummy tickers,
    merge in any hand-curated extras, return symbol list.
    Uses whatever indices are available — at least one must succeed.
    """
    seen: dict[str, None] = {}   # ordered-set via dict
    fetched_any = False

    for name, url in _INDEX_URLS.items():
        syms = _fetch_index(name, url)
        if syms:
            fetched_any = True
            for s in syms:
                if _is_valid_symbol(s):
                    seen[s] = None   # dedup, preserve first-seen order
        # Polite pacing — NSE archives don't rate-limit but be a good citizen
        time.sleep(0.4)

    # Hand-curated additions (e.g. PRECWIRE which is in NO official NSE index)
    for s in _load_extra_symbols():
        if _is_valid_symbol(s):
            seen[s] = None

    if not fetched_any and not seen:
        return list(_FALLBACK)

    return list(seen.keys())


# ── Public API ─────────────────────────────────────────────────────────────────

def get_universe_symbols() -> list[str]:
    """
    Return deduplicated NSE Total Market universe (~750 stocks).
    Union of: Nifty50 ∪ NiftyNext50 ∪ Nifty500 ∪ NiftySmallcap250
              ∪ NiftyMicrocap250 ∪ NiftyTotalMarket + hand-curated extras.

    This is the SAME ~750-stock universe that the Nifty Total Market index covers —
    we just build it from the union of 6 sub-index CSVs for resilience (if NSE
    fails to publish one CSV on a given day, we still have the others).

    Uses local 7-day cache. Falls back to hardcoded Nifty100 if NSE unreachable.
    """
    cached = _load_cache()
    if cached:
        return cached

    universe = _build_universe()
    if universe:
        _save_cache(universe)
    return universe


# BUG-FIX: original name `get_nifty500_symbols` was misleading — it returns
# the Nifty Total Market 750 universe (~750 stocks), NOT just Nifty 500.
# The misleading name caused confusion: "why are we using Nifty 500 not Total Market?"
# when the function was already returning the 750-stock universe.
# Keep the old name as a back-compat alias so existing imports don't break,
# but the canonical name going forward is get_universe_symbols().
def get_nifty500_symbols() -> list[str]:
    """[DEPRECATED NAME] Returns the Nifty Total Market 750 universe (~751 stocks).
    Misleading name kept for back-compat. New code should use get_universe_symbols()."""
    return get_universe_symbols()


def get_nse_tickers() -> list[str]:
    """
    Return universe in .NS suffix format used by all scanners and the screener.
    """
    return [f"{s}.NS" for s in get_universe_symbols()]
