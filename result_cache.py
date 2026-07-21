"""
Disk-persisted scan-result cache, keyed by the latest bhavcopy date.

WHY: every scanner recomputes its result from scratch on a cold cache (load
~284 bhavcopy day-files + analyse ~800 stocks = several seconds, much longer on
a throttled box). The result was only held in memory for 1 hour, so EVERY cold
start — a deploy, the first visitor, or hourly TTL expiry — paid the full
recompute. That is the recurring "scans are slow" complaint.

WHAT: each scanner persists its computed result to disk keyed by the current
bhavcopy date. A cold start then loads the last result from disk in ~50 ms
instead of recomputing. Because NSE data only changes once per day, the cache is
valid for the whole trading day and auto-invalidates the moment a newer bhavcopy
arrives (the date tag changes). The daily bhavcopy refresh recomputes + re-
persists, so the expensive work happens ONCE per day and every user — first or
not, before or after a restart — is served a warm result.

Safe by construction: atomic writes (tmp + os.replace), bhav-date tagging for
freshness, and every operation is best-effort (a cache miss just recomputes).
"""
from __future__ import annotations

import glob
import os
import pickle
import threading

_BASE = os.path.dirname(os.path.abspath(__file__))
_DIR = os.path.join(os.environ.get("DATA_DIR", _BASE), ".scan_cache")
_LOCK = threading.Lock()


def _bhav_tag() -> str:
    """A token that changes only when a newer bhavcopy is available."""
    try:
        from data_fetcher import _latest_bhavcopy_date
        d = _latest_bhavcopy_date()
        return d.isoformat() if d else "nodate"
    except Exception:
        return "nodate"


def _code_version() -> str:
    """Newest source-file mtime across the app's Python modules, snapshotted ONCE
    per process at import. Combined into the cache tag so a CODE change (deploy,
    bug fix, logic change) invalidates EVERY persisted scan result.

    WHY (fix 2026-07-08): the tag was the bhavcopy DATE only, so after a code
    change without a new bhavcopy the disk cache still matched — every scanner
    served results computed by the OLD code until the next trading day (or a
    manual `rm .scan_cache/*.pkl`). A restart carrying new code now bumps this
    version and forces a clean recompute."""
    try:
        mtimes = [os.path.getmtime(p) for p in glob.glob(os.path.join(_BASE, "*.py"))]
        return str(int(max(mtimes))) if mtimes else "0"
    except Exception:
        return "0"


_CODE_VERSION = _code_version()   # stable within a process; changes across deploys


def _tag() -> str:
    """Cache-validity token = bhavcopy date + code version. Either one changing
    invalidates the persisted result (fresh data OR fresh code → recompute)."""
    return f"{_bhav_tag()}|{_CODE_VERSION}"


def get(name: str):
    """Return the persisted result for `name` if it was computed against the
    CURRENT bhavcopy date, else None. Never raises."""
    try:
        path = os.path.join(_DIR, f"{name}.pkl")
        if not os.path.exists(path):
            return None
        with open(path, "rb") as f:
            blob = pickle.load(f)
        if blob.get("tag") == _tag():
            return blob.get("data")
    except Exception:
        pass
    return None


def put(name: str, data) -> None:
    """Persist `data` for `name`, tagged with the current bhavcopy date.
    Atomic (tmp file + os.replace). Best-effort — never raises."""
    if data is None:
        return
    try:
        with _LOCK:
            os.makedirs(_DIR, exist_ok=True)
            tmp = os.path.join(_DIR, f"{name}.pkl.tmp")
            final = os.path.join(_DIR, f"{name}.pkl")
            with open(tmp, "wb") as f:
                pickle.dump({"tag": _tag(), "data": data}, f, protocol=pickle.HIGHEST_PROTOCOL)
            os.replace(tmp, final)
    except Exception:
        pass


def invalidate(name: str | None = None) -> None:
    """Drop the persisted cache for `name` (or all of them when name is None)."""
    try:
        if name is None:
            for fn in os.listdir(_DIR):
                if fn.endswith(".pkl"):
                    os.remove(os.path.join(_DIR, fn))
        else:
            p = os.path.join(_DIR, f"{name}.pkl")
            if os.path.exists(p):
                os.remove(p)
    except Exception:
        pass
