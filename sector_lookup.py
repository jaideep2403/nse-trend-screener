"""
On-demand industry lookup for the ~1,550 tail stocks NSE's free data doesn't classify
(so the Peers panel shows REAL sector peers instead of a random market-cap cohort).

When a chart is opened for an unclassified name, we fetch its screener.in page ONCE
(polite, one request), read its industry breadcrumb (e.g. Universal Cables →
"Cables - Electricals"), and map that to our own sector taxonomy by keyword overlap
against our existing sector names ("Cables - Electricals" → "Wires & Cables"). The
result is cached to disk forever, so it's a one-time cost per stock. Any failure
returns None → the caller simply shows no peers, never wrong ones.
"""
from __future__ import annotations

import json
import os
import re
import time

_CACHE = os.path.join(os.environ.get("DATA_DIR", os.path.dirname(__file__)), "sector_lookup.json")
_mem: dict | None = None
_last_fetch = [0.0]
_MIN_GAP = 1.0        # ≥1s between screener hits — stay polite
# Circuit breaker: if screener.in starts refusing us (rate-limit / timeouts), STOP hitting
# it for a while so a throttled source can never hang a chart request on the peers lookup.
_fail_streak = [0]
_cooldown_until = [0.0]
_FAIL_TRIP = 4        # this many consecutive failures →
_COOLDOWN_S = 600     # …back off from screener for 10 minutes
_UA = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                     "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0 Safari/537.36"}

_STOP = {"and", "the", "of", "other", "others", "products", "product", "ltd", "misc",
         "general", "diversified", "services", "goods"}


def _load() -> dict:
    global _mem
    if _mem is None:
        try:
            with open(_CACHE) as f:
                _mem = json.load(f)
        except Exception:
            _mem = {}
    return _mem


def _save():
    try:
        with open(_CACHE, "w") as f:
            json.dump(_mem, f)
    except Exception:
        pass


def _kw(s: str) -> set:
    return {w for w in re.split(r"[^a-z0-9]+", (s or "").lower()) if len(w) > 2 and w not in _STOP}


def cached_map() -> dict:
    """{symbol: our-sector} for every tail stock already looked up on-demand (skips the
    cached '' misses). Free to fold into any sector view — no new network."""
    return {k: v for k, v in _load().items() if isinstance(v, str) and v}


def _our_sectors() -> list[str]:
    try:
        import sector_mapper as sm
        return sorted(set(sm.get_enriched_sector_map().values()))
    except Exception:
        return []


def _screener_industries(sym: str) -> list[str]:
    """The stock's classification breadcrumb from screener.in (finest last), or []."""
    import requests
    if time.time() < _cooldown_until[0]:
        raise RuntimeError("screener cooldown")      # circuit open → don't hit the network
    gap = time.time() - _last_fetch[0]
    if gap < _MIN_GAP:
        time.sleep(_MIN_GAP - gap)
    _last_fetch[0] = time.time()
    try:
        r = requests.get(f"https://www.screener.in/company/{sym}/", headers=_UA, timeout=6)
    except Exception:
        _fail_streak[0] += 1
        if _fail_streak[0] >= _FAIL_TRIP:
            _cooldown_until[0] = time.time() + _COOLDOWN_S
        raise
    if r.status_code in (429, 503):                  # explicit rate-limit → trip the breaker
        _fail_streak[0] += 1
        if _fail_streak[0] >= _FAIL_TRIP:
            _cooldown_until[0] = time.time() + _COOLDOWN_S
        raise RuntimeError(f"screener {r.status_code}")
    _fail_streak[0] = 0                               # a good response resets the streak
    if r.status_code != 200:
        return []
    html = r.text
    # The Sector/Industry breadcrumb links: <a ...Sector">Industrials</a> …
    #                                        <a ...Industry">Cables - Electricals</a>
    vals = [m.group(2).strip() for m in
            re.finditer(r'(Sector|Industry)"[^>]*>([^<]{2,50})<', html)]
    # de-dup, keep order (broad → fine)
    seen, out = set(), []
    for v in vals:
        if v and v not in seen:
            seen.add(v)
            out.append(v)
    return out


def industry_of(sym: str) -> str | None:
    """Our sector name for a tail stock, via screener.in — cached forever. None on fail."""
    sym = (sym or "").upper()
    m = _load()
    if sym in m:                        # cached (including a cached miss "")
        return m[sym] or None
    # Consult the BULK sector scrape first — it already classified ~1,830 tail stocks, so
    # this resolves without ANY network hit (and stops the chart's peers lookup from
    # hanging on a throttled screener.in for names we've already scraped).
    try:
        import bulk_sector_scrape as _bss
        _b = _bss.get_sector_map().get(sym)
        if _b:
            m[sym] = _b
            _save()
            return _b
    except Exception:
        pass
    try:
        inds = _screener_industries(sym)
    except Exception:
        return None                     # transient failure → don't cache, allow retry
    our = _our_sectors()
    best, best_score = None, 0
    # Prefer the FINEST industry (last breadcrumb) — it's the most specific.
    for rank, ind in enumerate(reversed(inds)):
        ikw = _kw(ind)
        for sec in our:
            score = len(ikw & _kw(sec)) * 10 - rank      # finer breadcrumb wins ties
            if score > best_score:
                best, best_score = sec, score
        if best:                        # a fine-level match beats anything broader
            break
    m[sym] = best or ""                 # cache the result (empty = no confident match)
    _save()
    return best
