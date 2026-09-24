"""
Earnings / results dates from NSE's bulk financial-results feed.

ONE call per day, cached to disk — the polite, human way. NSE returns every listed
company's recent quarterly-result announcements in a single response (~3,800 rows),
so we never hammer per-stock. {symbol: 'YYYY-MM-DD'} = the LATEST result date.

Used by the PEAD screen (post-earnings drift = a fresh uptrend that started after a
recent results announcement). Falls back to the last good cache if NSE is unhappy.
"""
from __future__ import annotations

import datetime as _dt
import json
import os
import time

_CACHE_PATH = os.path.join(os.environ.get("DATA_DIR", os.path.dirname(__file__)), "earnings_dates.json")
_TTL = 24 * 3600
_mem = {"data": None, "ts": 0}
_URL = "https://www.nseindia.com/api/corporates-financial-results?index=equities&period=Quarterly"


def _parse_date(s: str):
    try:
        return _dt.datetime.strptime((s or "").split()[0], "%d-%b-%Y").date().isoformat()
    except Exception:
        return None


def _fetch() -> dict:
    import fo_data
    sess = fo_data._get_seeded_session()   # seeds nseindia.com cookies politely, once
    time.sleep(1.0)                        # a human pause before the data call
    r = sess.get(_URL, timeout=25)
    r.raise_for_status()
    rows = r.json()
    if isinstance(rows, dict):
        rows = rows.get("data", [])
    latest: dict[str, str] = {}
    for x in rows:
        sym = x.get("symbol")
        bd = _parse_date(x.get("broadCastDate") or x.get("filingDate") or "")
        if not sym or not bd:
            continue
        if sym not in latest or bd > latest[sym]:
            latest[sym] = bd
    return latest


def _load_disk():
    try:
        with open(_CACHE_PATH) as f:
            return json.load(f)
    except Exception:
        return None


def get_earnings_dates(force: bool = False) -> dict:
    """{symbol: 'YYYY-MM-DD'} latest results date. Cached 24h in memory + on disk."""
    now = time.time()
    if not force and _mem["data"] and now - _mem["ts"] < _TTL:
        return _mem["data"]
    if not force:
        blob = _load_disk()
        if blob and now - blob.get("ts", 0) < _TTL:
            _mem.update(data=blob["data"], ts=blob["ts"])
            return blob["data"]
    try:
        data = _fetch()
        if not data:
            raise RuntimeError("empty result set")
    except Exception:
        blob = _load_disk()          # NSE unhappy → serve last good cache
        if blob:
            _mem.update(data=blob["data"], ts=blob.get("ts", 0))
            return blob["data"]
        return {}
    _mem.update(data=data, ts=now)
    try:
        with open(_CACHE_PATH, "w") as f:
            json.dump({"ts": now, "data": data}, f)
    except Exception:
        pass
    return data
