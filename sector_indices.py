"""
Sector / thematic index intelligence — constituents, daily levels, and a
canonical symbol→industry map for the full ~750-stock universe.

All data comes from NSE's PUBLIC archives (no auth, no rate limits):
  - Constituent lists : https://archives.nseindia.com/content/indices/<file>.csv
                        columns ['Company Name','Industry','Symbol','Series','ISIN Code']
  - Daily all-indices : https://archives.nseindia.com/content/indices/ind_close_all_<DDMMYYYY>.csv
                        columns incl. ['Index Name','Closing Index Value','Change(%)','P/E','P/B','Div Yield']

DESIGN — this is a real-money app and NSE flakes out, so every public accessor:
  * NEVER raises — returns the last-good cache, or {} if there is nothing cached.
  * Caches to a JSON file next to this module, written ATOMICALLY (temp + os.replace).
  * Honours a TTL (weekly for slow-changing constituents; per-bhav-date for levels).
  * Logs a single line to stderr on any fetch failure (not silent).

Public API (return contracts at bottom of file):
  INDEX_FILES                         -> dict[str, str]   (validated at import)
  get_sector_constituents(force=False)-> dict[str, list[str]]
  get_constituent_industry_map()      -> dict[str, str]
  get_sector_levels(force=False)      -> dict[str, dict]
  get_canonical_sector_map(force=False)-> dict[str, str]
"""

from __future__ import annotations

import io
import os
import sys
import json
import time
import tempfile
from pathlib import Path

import requests
import pandas as pd

import nse_stocks  # for _HEADERS (verified-working request headers)
from data_fetcher import _latest_bhavcopy_date

# ── Constants ────────────────────────────────────────────────────────────────
_HEADERS = nse_stocks._HEADERS
_TIMEOUT = 15
_ARCHIVE_BASE = "https://archives.nseindia.com/content/indices/"

_DIR = Path(__file__).parent
_CONSTITUENTS_CACHE = _DIR / ".sector_indices_cache.json"
_LEVELS_CACHE       = _DIR / ".sector_levels_cache.json"
_CANONICAL_CACHE    = _DIR / ".sector_canonical_cache.json"

_CONSTITUENTS_TTL = 7 * 86400          # weekly — constituents change semi-annually
_CANONICAL_TTL    = 7 * 86400          # weekly — total-market membership is slow-moving

_CANONICAL_FILE   = "ind_niftytotalmarket_list.csv"   # ~753 stocks

# Sectoral + thematic indices we want to track. Values are the candidate CSV
# filenames; each is VALIDATED at import (see _validate_index_files) and any that
# 404 / lack a 'Symbol' column are dropped. For a handful we list known filename
# variants (NSE is inconsistent about underscores) and keep the first that works.
_CANDIDATE_INDEX_FILES: dict[str, list[str]] = {
    "Nifty Bank":               ["ind_niftybanklist.csv"],
    "Nifty IT":                 ["ind_niftyitlist.csv"],
    "Nifty Pharma":             ["ind_niftypharmalist.csv"],
    "Nifty Auto":               ["ind_niftyautolist.csv"],
    "Nifty FMCG":               ["ind_niftyfmcglist.csv"],
    "Nifty Metal":              ["ind_niftymetallist.csv"],
    "Nifty Realty":             ["ind_niftyrealtylist.csv"],
    "Nifty Media":              ["ind_niftymedialist.csv"],
    "Nifty PSU Bank":           ["ind_niftypsubanklist.csv"],
    "Nifty Private Bank":       ["ind_nifty_privatebanklist.csv",
                                 "ind_niftyprivatebanklist.csv"],
    "Nifty Financial Services": ["ind_niftyfinancelist.csv",
                                 "ind_niftyfinancialserviceslist.csv"],
    "Nifty Healthcare":         ["ind_niftyhealthcarelist.csv"],
    "Nifty Consumer Durables":  ["ind_niftyconsumerdurableslist.csv"],
    "Nifty Oil & Gas":          ["ind_niftyoilgaslist.csv"],
    "Nifty India Defence":      ["ind_niftyindiadefence_list.csv",
                                 "ind_niftyindiadefencelist.csv",
                                 "ind_nifty_india_defence_list.csv"],
    "Nifty Energy":             ["ind_niftyenergylist.csv"],
    "Nifty Infrastructure":     ["ind_niftyinfralist.csv"],
    "Nifty Commodities":        ["ind_niftycommoditieslist.csv"],
    "Nifty MNC":                ["ind_niftymnclist.csv"],
    "Nifty PSE":                ["ind_niftypselist.csv"],
    "Nifty CPSE":               ["ind_niftycpselist.csv"],
}


# ── Tiny helpers ─────────────────────────────────────────────────────────────
def _log(msg: str) -> None:
    """One-line, non-fatal log to stderr (never silent on fetch failure)."""
    print(f"[sector_indices] {msg}", file=sys.stderr)


def _atomic_write_json(path: Path, obj) -> None:
    """Write JSON atomically: temp file in the same dir + os.replace().
    Never raises — a cache-write failure must not break the accessor."""
    try:
        fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=path.name + ".", suffix=".tmp")
        try:
            with os.fdopen(fd, "w") as f:
                json.dump(obj, f)
            os.replace(tmp, path)
        finally:
            if os.path.exists(tmp):
                try:
                    os.remove(tmp)
                except OSError:
                    pass
    except Exception as e:                       # pragma: no cover - defensive
        _log(f"cache write failed for {path.name}: {e}")


def _read_cache(path: Path) -> dict | None:
    """Return parsed cache envelope {'ts','key','data'} or None. Never raises."""
    try:
        if path.exists():
            with open(path) as f:
                return json.load(f)
    except Exception as e:
        _log(f"cache read failed for {path.name}: {e}")
    return None


def _fetch_csv(filename: str) -> pd.DataFrame | None:
    """Download one indices CSV → DataFrame, or None on any failure (logged).
    Uses the verified-working fetch pattern (requests + pd.read_csv(BytesIO))."""
    url = _ARCHIVE_BASE + filename
    try:
        r = requests.get(url, headers=_HEADERS, timeout=_TIMEOUT)
        if r.status_code != 200:
            _log(f"HTTP {r.status_code} for {filename}")
            return None
        return pd.read_csv(io.BytesIO(r.content))
    except Exception as e:
        _log(f"fetch failed for {filename}: {e}")
        return None


def _symbols_from_df(df: pd.DataFrame) -> list[str]:
    """Extract clean Symbol list from a constituent CSV (strip + dropna)."""
    if df is None or "Symbol" not in df.columns:
        return []
    return [s for s in df["Symbol"].dropna().astype(str).str.strip().tolist() if s]


def _parse_pct(val) -> float | None:
    """Parse NSE Change(%) strings like '-.48' / '.08' / '1.23' → float.
    Returns None for non-numeric / blank values."""
    try:
        if val is None:
            return None
        s = str(val).strip()
        if not s or s.lower() in ("-", "na", "nan", "n.a.", ""):
            return None
        return float(s)                          # float() handles leading-dot forms
    except (ValueError, TypeError):
        return None


def _parse_num(val) -> float | None:
    """Parse a generic numeric field (close / P-E / P-B) → float or None."""
    try:
        if val is None:
            return None
        s = str(val).replace(",", "").strip()
        if not s or s.lower() in ("-", "na", "nan", "n.a.", ""):
            return None
        return float(s)
    except (ValueError, TypeError):
        return None


# ── INDEX_FILES — validated at module build time ─────────────────────────────
def _validate_index_files() -> dict[str, str]:
    """Fetch each candidate once; keep only filenames returning HTTP 200 with a
    'Symbol' column. Tries listed variants in order, records which worked.
    Logs any display name dropped entirely. Never raises."""
    resolved: dict[str, str] = {}
    dropped: list[str] = []
    for display, candidates in _CANDIDATE_INDEX_FILES.items():
        chosen = None
        for fname in candidates:
            df = _fetch_csv(fname)
            if df is not None and "Symbol" in df.columns:
                chosen = fname
                break
            time.sleep(0.25)                     # polite pacing between probes
        if chosen:
            resolved[display] = chosen
        else:
            dropped.append(display)
        time.sleep(0.25)
    if dropped:
        _log(f"dropped {len(dropped)} index(es) at validation: {', '.join(dropped)}")
    _log(f"validated {len(resolved)}/{len(_CANDIDATE_INDEX_FILES)} sectoral/thematic indices")
    return resolved


# Built once at import. If NSE is fully unreachable this may be empty/partial;
# the accessors below still work off cache and never raise.
INDEX_FILES: dict[str, str] = _validate_index_files()


# ── (2) Constituents + industry map ──────────────────────────────────────────
def _download_constituents() -> tuple[dict[str, list[str]], dict[str, str]]:
    """Download every validated INDEX_FILES CSV.
    Returns ({display: [symbols]}, {symbol: Industry}). Skips indices whose
    fetch fails (so a partial NSE outage still yields what it can)."""
    constituents: dict[str, list[str]] = {}
    industry: dict[str, str] = {}
    for display, fname in INDEX_FILES.items():
        df = _fetch_csv(fname)
        if df is None:
            continue
        syms = _symbols_from_df(df)
        if syms:
            constituents[display] = syms
        if "Symbol" in df.columns and "Industry" in df.columns:
            for _, row in df[["Symbol", "Industry"]].dropna().iterrows():
                sym = str(row["Symbol"]).strip()
                ind = str(row["Industry"]).strip()
                if sym and ind:
                    industry[sym] = ind
        time.sleep(0.3)
    return constituents, industry


def get_sector_constituents(force: bool = False) -> dict[str, list[str]]:
    """{display_name: [symbols]} for each tracked sectoral/thematic index.

    Cached (weekly TTL, atomic write). On ANY fetch failure falls back to the
    last-good cache; returns {} only if nothing is cached. Never raises.
    `force=True` bypasses the TTL and re-downloads."""
    try:
        cache = _read_cache(_CONSTITUENTS_CACHE)
        if (not force and cache and isinstance(cache.get("data"), dict)
                and (time.time() - cache.get("ts", 0)) < _CONSTITUENTS_TTL):
            return cache["data"]

        constituents, industry = _download_constituents()
        if constituents:
            _atomic_write_json(_CONSTITUENTS_CACHE, {
                "ts": time.time(),
                "data": constituents,
                "industry": industry,
            })
            return constituents

        # Fetch produced nothing — fall back to last-good cache regardless of TTL.
        if cache and isinstance(cache.get("data"), dict):
            _log("constituents fetch empty — serving last-good cache")
            return cache["data"]
        _log("constituents fetch empty and no cache available")
        return {}
    except Exception as e:                       # pragma: no cover - defensive
        _log(f"get_sector_constituents fatal-guard: {e}")
        cache = _read_cache(_CONSTITUENTS_CACHE)
        if cache and isinstance(cache.get("data"), dict):
            return cache["data"]
        return {}


def get_constituent_industry_map() -> dict[str, str]:
    """{symbol: Industry} aggregated across the tracked index CSVs.
    Reads from the constituents cache (populating it first if needed).
    Never raises; returns {} if unavailable."""
    try:
        cache = _read_cache(_CONSTITUENTS_CACHE)
        if not cache or "industry" not in cache:
            get_sector_constituents()            # populate cache (incl. industry)
            cache = _read_cache(_CONSTITUENTS_CACHE)
        if cache and isinstance(cache.get("industry"), dict):
            return cache["industry"]
        return {}
    except Exception as e:                       # pragma: no cover - defensive
        _log(f"get_constituent_industry_map fatal-guard: {e}")
        return {}


# ── (3) Daily index levels ───────────────────────────────────────────────────
def _download_levels(bhav_key: str) -> dict[str, dict]:
    """Download ind_close_all_<DDMMYYYY>.csv and build
    {Index Name: {close, chg_pct, pe, pb}} for our tracked display names.
    Matching is on the 'Index Name' column (NSE names already match ours)."""
    fname = f"ind_close_all_{bhav_key}.csv"
    df = _fetch_csv(fname)
    if df is None or "Index Name" not in df.columns:
        return {}

    wanted = set(INDEX_FILES.keys()) or None     # if INDEX_FILES empty, take all
    levels: dict[str, dict] = {}
    for _, row in df.iterrows():
        name = str(row.get("Index Name", "")).strip()
        if not name:
            continue
        if wanted is not None and name not in wanted:
            continue
        levels[name] = {
            "close":   _parse_num(row.get("Closing Index Value")),
            "chg_pct": _parse_pct(row.get("Change(%)")),
            "pe":      _parse_num(row.get("P/E")),
            "pb":      _parse_num(row.get("P/B")),
        }
    return levels


def get_sector_levels(force: bool = False) -> dict[str, dict]:
    """{display_name: {"close","chg_pct","pe","pb"}} from the latest daily
    ind_close_all CSV. Cache is keyed by the bhav date, so a new trading day
    auto-invalidates it. On ANY failure falls back to last-good cache; {} if
    none. Never raises. `force=True` re-downloads for the current bhav date."""
    try:
        cache = _read_cache(_LEVELS_CACHE)
        bhav = _latest_bhavcopy_date()
        bhav_key = bhav.strftime("%d%m%Y") if bhav else None

        if (not force and cache and isinstance(cache.get("data"), dict)
                and bhav_key is not None and cache.get("key") == bhav_key):
            return cache["data"]

        if bhav_key is None:
            _log("no latest bhavcopy date — serving last-good levels cache")
            if cache and isinstance(cache.get("data"), dict):
                return cache["data"]
            return {}

        levels = _download_levels(bhav_key)
        if levels:
            _atomic_write_json(_LEVELS_CACHE, {
                "ts": time.time(),
                "key": bhav_key,
                "data": levels,
            })
            return levels

        if cache and isinstance(cache.get("data"), dict):
            _log("levels fetch empty — serving last-good cache")
            return cache["data"]
        _log("levels fetch empty and no cache available")
        return {}
    except Exception as e:                       # pragma: no cover - defensive
        _log(f"get_sector_levels fatal-guard: {e}")
        cache = _read_cache(_LEVELS_CACHE)
        if cache and isinstance(cache.get("data"), dict):
            return cache["data"]
        return {}


# ── (4) Canonical symbol→industry map (Total Market, ~753 stocks) ────────────
def get_canonical_sector_map(force: bool = False) -> dict[str, str]:
    """{symbol: Industry} for the full Nifty Total Market list (~753 stocks).
    Source for Path B sector classification. Cached (weekly TTL, atomic write),
    falls back to last-good cache on any failure; {} if none. Never raises."""
    try:
        cache = _read_cache(_CANONICAL_CACHE)
        if (not force and cache and isinstance(cache.get("data"), dict)
                and (time.time() - cache.get("ts", 0)) < _CANONICAL_TTL):
            return cache["data"]

        df = _fetch_csv(_CANONICAL_FILE)
        if df is not None and "Symbol" in df.columns and "Industry" in df.columns:
            mapping: dict[str, str] = {}
            for _, row in df[["Symbol", "Industry"]].dropna().iterrows():
                sym = str(row["Symbol"]).strip()
                ind = str(row["Industry"]).strip()
                if sym and ind:
                    mapping[sym] = ind
            if mapping:
                _atomic_write_json(_CANONICAL_CACHE, {"ts": time.time(), "data": mapping})
                return mapping

        if cache and isinstance(cache.get("data"), dict):
            _log("canonical map fetch empty — serving last-good cache")
            return cache["data"]
        _log("canonical map fetch empty and no cache available")
        return {}
    except Exception as e:                       # pragma: no cover - defensive
        _log(f"get_canonical_sector_map fatal-guard: {e}")
        cache = _read_cache(_CANONICAL_CACHE)
        if cache and isinstance(cache.get("data"), dict):
            return cache["data"]
        return {}


if __name__ == "__main__":   # manual smoke test
    c = get_sector_constituents()
    print("indices:", len(c),
          "| Nifty Bank:", len(c.get("Nifty Bank", [])),
          "| Nifty India Defence:", len(c.get("Nifty India Defence", [])))
    L = get_sector_levels()
    print("levels:", len(L), "| Nifty India Defence:", L.get("Nifty India Defence"))
    print("canonical map size:", len(get_canonical_sector_map()))
