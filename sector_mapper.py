"""
sector_mapper.py — Auto-maps all 751 NSE stocks to sectors using NSE's own
TotalMarket index CSV (static file, no auth, no yfinance, no rate limits).

Strategy:
  • INDUSTRY_GROUPS (523 stocks) = ground truth, hand-curated fine-grained sectors.
  • The remaining ~241 stocks → mapped via NSE's broad Industry label from the
    TotalMarket CSV → then bucketed into our existing sector names.
  • Result cached in .sector_cache.json (30-day TTL, gitignored).
  • get_enriched_industry_groups() returns a complete dict covering all 751 stocks.

LOCAL-ONLY — gitignored, never pushed to GitHub.
"""

import io
import json
import time
import threading
import requests
import pandas as pd
from pathlib import Path
from industry_groups import INDUSTRY_GROUPS

# ── Config ────────────────────────────────────────────────────────────────────

CACHE_FILE = Path(__file__).parent / ".sector_cache.json"  # BUG-011 FIX: absolute path relative to module, not cwd
CACHE_TTL  = 30 * 86_400          # 30 days — NSE industry classification rarely changes

NSE_TOTAL_MARKET_URL = (
    "https://archives.nseindia.com/content/indices/ind_niftytotalmarket_list.csv"
)

_NSE_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Accept":          "text/html,*/*",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer":         "https://www.nseindia.com/",
}

# ── NSE broad Industry → our existing fine-grained sector names ───────────────
# NSE uses only 22 broad labels — we map each to the most representative sector.
# Stocks already in INDUSTRY_GROUPS keep their precise hand-curated sub-sector.

NSE_INDUSTRY_TO_SECTOR: dict[str, str] = {
    "Automobile and Auto Components": "Auto Ancillary",
    "Capital Goods":                  "Capital Goods",
    "Chemicals":                      "Specialty Chemicals",
    "Construction":                   "Infrastructure & EPC",
    "Construction Materials":         "Building Materials",
    "Consumer Durables":              "Consumer Durables",
    "Consumer Services":              "Consumer Discretionary",
    "Diversified":                    "Conglomerates",
    "Fast Moving Consumer Goods":     "FMCG",
    "Financial Services":             "NBFCs",
    "Forest Materials":               "Paper & Packaging",
    "Healthcare":                     "Healthcare Services",
    # IT split is handled per-symbol via _classify_it_sector() below — large-cap
    # IT names (TCS, INFY, WIPRO, HCLTECH) are not "IT - Midcap".
    "Information Technology":         "IT - Midcap",
    "Media Entertainment & Publication": "Media & Entertainment",
    "Metals & Mining":                "Metals & Mining",
    "Oil Gas & Consumable Fuels":     "Oil & Gas",
    "Power":                          "Power & Utilities",
    "Realty":                         "Real Estate",
    "Services":                       "Services",
    "Telecommunication":              "Telecom",
    "Textiles":                       "Textiles",
    "Utilities":                      "Power & Utilities",
}

# Large-cap IT names from Nifty IT / Nifty 50 / known LC universe — these
# should NOT be tagged as "IT - Midcap" just because NSE only publishes the
# broad "Information Technology" label. Anything else IT gets "IT - Midcap".
_IT_LARGECAP_SYMS = {
    "TCS", "INFY", "WIPRO", "HCLTECH", "TECHM", "LTIM", "MPHASIS",
    "PERSISTENT", "OFSS", "COFORGE",
}


def _classify_it_sector(symbol: str) -> str:
    return "IT - Largecap" if symbol in _IT_LARGECAP_SYMS else "IT - Midcap"

# ── Internals ─────────────────────────────────────────────────────────────────

_lock   = threading.Lock()
_mem    = {"data": None, "ts": 0}


def _load_cache() -> dict:
    if _mem["data"] and time.time() - _mem["ts"] < CACHE_TTL:
        return _mem["data"]
    try:
        if CACHE_FILE.exists():
            raw = json.loads(CACHE_FILE.read_text())
            if time.time() - raw.get("_ts", 0) < CACHE_TTL:
                _mem.update({"data": raw, "ts": time.time()})
                return raw
    except Exception:
        pass
    return {}


def _save_cache(data: dict):
    try:
        data["_ts"] = time.time()
        CACHE_FILE.write_text(json.dumps(data, indent=2))
        _mem.update({"data": data, "ts": time.time()})
    except Exception:
        pass


def _fetch_nse_industry_map() -> dict[str, str]:
    """
    Fetch NSE TotalMarket CSV → {symbol: nse_industry_label}.
    Returns empty dict on failure (graceful degradation).
    """
    try:
        r = requests.get(NSE_TOTAL_MARKET_URL, headers=_NSE_HEADERS, timeout=15)
        if r.status_code != 200 or len(r.content) < 1000:
            return {}
        df = pd.read_csv(io.BytesIO(r.content))
        df.columns = [c.strip() for c in df.columns]
        if "Symbol" not in df.columns or "Industry" not in df.columns:
            return {}
        df["Symbol"]   = df["Symbol"].str.strip()
        df["Industry"] = df["Industry"].str.strip()
        return dict(zip(df["Symbol"], df["Industry"]))
    except Exception:
        return {}


def _build_cache() -> dict:
    """
    Build the complete {symbol: our_sector} cache for all symbols in
    NSE TotalMarket that are NOT already in INDUSTRY_GROUPS.
    """
    # Build base set from INDUSTRY_GROUPS (already mapped)
    ig_syms: set[str] = set()
    for syms in INDUSTRY_GROUPS.values():
        ig_syms.update(syms)

    nse_map = _fetch_nse_industry_map()
    if not nse_map:
        return {}

    result: dict[str, str] = {}
    for sym, nse_ind in nse_map.items():
        if sym in ig_syms:
            continue   # already has precise hand-curated sector
        our_sector = NSE_INDUSTRY_TO_SECTOR.get(nse_ind)
        if our_sector:
            # Split Information Technology into LC vs MC by symbol membership
            # so giants like TCS/INFY aren't lumped under "IT - Midcap".
            if nse_ind == "Information Technology":
                our_sector = _classify_it_sector(sym)
            result[sym] = our_sector
        # else: unmapped → leave out (extremely rare)

    return result


# ── Public API ────────────────────────────────────────────────────────────────

def refresh_sector_cache(background: bool = True) -> None:
    """
    Refresh the sector cache from NSE TotalMarket CSV.
    Runs in background by default so it never blocks a scan.
    """
    def _do():
        with _lock:
            cached = _load_cache()
            # Check if we already have a fresh cache
            if cached and time.time() - cached.get("_ts", 0) < CACHE_TTL:
                return
            new_data = _build_cache()
            if new_data:
                _save_cache(new_data)

    if background:
        threading.Thread(target=_do, daemon=True).start()
    else:
        _do()


def get_enriched_sector_map() -> dict[str, str]:
    """
    Returns {symbol: sector_name} for ALL stocks:
      - INDUSTRY_GROUPS stocks → their precise hand-curated sector
      - Extra stocks from NSE TotalMarket → mapped via NSE_INDUSTRY_TO_SECTOR
    """
    # Start with INDUSTRY_GROUPS (ground truth)
    sector_map: dict[str, str] = {}
    for sector, syms in INDUSTRY_GROUPS.items():
        for sym in syms:
            sector_map[sym] = sector

    # Overlay cached auto-mapped extras
    cached = _load_cache()
    for sym, sector in cached.items():
        if sym != "_ts" and sym not in sector_map:
            sector_map[sym] = sector

    return sector_map


def get_enriched_industry_groups() -> dict[str, list[str]]:
    """
    Returns a complete INDUSTRY_GROUPS-style dict covering all 751 stocks.
    INDUSTRY_GROUPS entries are preserved exactly.
    Extra stocks from cache are appended to their mapped sector.
    """
    # Deep-copy INDUSTRY_GROUPS
    groups: dict[str, list[str]] = {k: list(v) for k, v in INDUSTRY_GROUPS.items()}

    cached = _load_cache()
    ig_syms: set[str] = set()
    for syms in INDUSTRY_GROUPS.values():
        ig_syms.update(syms)

    for sym, sector in cached.items():
        if sym == "_ts" or sym in ig_syms:
            continue
        if sector not in groups:
            groups[sector] = []
        if sym not in groups[sector]:
            groups[sector].append(sym)

    return groups
