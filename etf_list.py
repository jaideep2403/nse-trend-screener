"""
Authoritative NSE ETF exclusion list.

NSE publishes the full list of exchange-traded funds at /api/etf. ETFs trade in
the EQ series alongside stocks, so the full-universe loaders would otherwise pick
them up and display them as "stocks". Symbol-pattern matching is leaky — index
trackers like ABSLMSCIN / AONETOTAL / ALPHA / BFSI / CONSUMER / DEFENCE carry no
fund token — so we use NSE's own list as the definitive set.

ONE polite call, cached to disk (weekly). The hot path (`etf_symbols()`, used inside
is_etf on every symbol) NEVER touches the network: it reads memory, then disk, and
returns an empty set only if neither exists. Refresh happens out-of-band via
`refresh()` (called once at app startup), so a stale weekly cache never blocks a
scan. Falls back to the last-good cache when NSE is unhappy.
"""
from __future__ import annotations

import json
import os
import time

_CACHE_PATH = os.path.join(os.environ.get("DATA_DIR", os.path.dirname(__file__)), "nse_etf_list.json")
_TTL = 7 * 24 * 3600
_URL = "https://www.nseindia.com/api/etf"
_mem: dict = {"data": None, "ts": 0.0}


def _load_disk():
    try:
        with open(_CACHE_PATH) as f:
            return json.load(f)
    except Exception:
        return None


def etf_symbols() -> set[str]:
    """The authoritative ETF symbol set. Memory → disk → empty. Never blocks."""
    if _mem["data"] is not None:
        return _mem["data"]
    blob = _load_disk()
    data = set(blob.get("data", [])) if blob else set()
    _mem.update(data=data, ts=(blob or {}).get("ts", 0.0))
    return data


def _fetch() -> list[str]:
    import fo_data
    sess = fo_data._get_seeded_session()   # seeds nseindia.com cookies politely, once
    time.sleep(1.0)                        # a human pause before the data call
    r = sess.get(_URL, timeout=25)
    r.raise_for_status()
    j = r.json()
    rows = j.get("data", j if isinstance(j, list) else [])
    return sorted({(x.get("symbol") or "").upper() for x in rows
                   if isinstance(x, dict) and x.get("symbol")})


def refresh(force: bool = False) -> set[str]:
    """Refresh from NSE if the disk cache is older than a week. Safe to call at
    startup; on any failure the last-good cache is retained. Returns the set."""
    now = time.time()
    blob = _load_disk()
    if not force and blob and now - blob.get("ts", 0) < _TTL:
        _mem.update(data=set(blob.get("data", [])), ts=blob.get("ts", 0.0))
        return _mem["data"]
    try:
        syms = _fetch()
        if not syms:
            raise RuntimeError("empty ETF list")
        try:
            with open(_CACHE_PATH, "w") as f:
                json.dump({"ts": now, "data": syms}, f)
        except Exception:
            pass
        _mem.update(data=set(syms), ts=now)
        return _mem["data"]
    except Exception:
        if blob:                           # NSE unhappy → keep last good
            _mem.update(data=set(blob.get("data", [])), ts=blob.get("ts", 0.0))
            return _mem["data"]
        return set()
