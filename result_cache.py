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
_PREF_DIR = os.path.join(os.environ.get("DATA_DIR", _BASE), ".scan_cache")


def _pick_writable_dir(preferred: str) -> str:
    """Pick a cache dir we can actually WRITE to. If the preferred one (usually the
    /data volume) is not writable — the classic root-owned-volume permission bug —
    fall back to a writable path and SHOUT about it, because a silently-unwritable
    cache means every scan recomputes cold forever (the exact 'scans are slow'
    symptom). Persistence across redeploys still needs /data fixed, but at least the
    running container caches and stays fast, and the log names the real problem."""
    import tempfile
    candidates = [preferred,
                  os.path.join(_BASE, ".scan_cache"),
                  os.path.join(tempfile.gettempdir(), "ascent_scan_cache")]
    for cand in candidates:
        try:
            os.makedirs(cand, exist_ok=True)
            _t = os.path.join(cand, ".wtest")
            with open(_t, "w") as f:
                f.write("ok")
            os.remove(_t)
            if cand != preferred:
                print(f"[result_cache] WARNING: {preferred!r} is NOT writable — scan "
                      f"results cannot persist there, so every scan was recomputing "
                      f"cold. Falling back to {cand!r} (fast within this container, but "
                      f"lost on redeploy). FIX: chown {preferred} to the container user "
                      f"on the EC2 box so the cache survives deploys.", flush=True)
            return cand
        except Exception:
            continue
    print("[result_cache] CRITICAL: no writable cache dir found — scans will recompute "
          "on EVERY request. Check DATA_DIR / volume permissions.", flush=True)
    return preferred


_DIR = _pick_writable_dir(_PREF_DIR)
_LOCK = threading.Lock()

# ── Stale-while-revalidate ───────────────────────────────────────────────────
# A client must NEVER block on a cold scan. When the cache tag has moved (a deploy
# bumped the code version, or a new bhavcopy landed), get_or_stale() serves the
# LAST-GOOD result instantly and kicks a background refresh, so the page paints
# immediately with data that is at most one compute-cycle old and then updates.
# Prewarm runs in "force mode" so IT recomputes fresh instead of serving stale.
import time as _time
_force_local = threading.local()
_refresh_hook = None
_last_refresh_kick = [0.0]


def set_refresh_hook(cb):
    """app wires this to spawn _prewarm_all_scans in the background."""
    global _refresh_hook
    _refresh_hook = cb


def force_mode(on: bool = True):
    """Prewarm sets this around its compute so get_or_stale forces a real recompute
    (returns None) rather than serving the stale copy back to the warmer itself."""
    _force_local.force = on


def _in_force_mode() -> bool:
    return getattr(_force_local, "force", False)


def _kick_refresh():
    """Fire the background prewarm at most once every 20s (it self-serializes via
    its own lock anyway; this just avoids spawning a burst of no-op threads)."""
    if _refresh_hook is None:
        return
    now = _time.time()
    if now - _last_refresh_kick[0] < 20:
        return
    _last_refresh_kick[0] = now
    try:
        _refresh_hook()
    except Exception:
        pass


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
            data = blob.get("data")
            if not blob.get("breakdown_done"):
                _schedule_breakdown(name, data)
            return data
    except Exception:
        pass
    return None


def stats() -> dict:
    """Health snapshot for /api/diag/cache — is the cache persisting, and is each
    scan fresh/stale/missing? Turns 'scans are slow' into a fact instead of a guess."""
    import time as _t
    out = {"dir": _DIR, "preferred_dir": _PREF_DIR,
           "using_fallback": _DIR != _PREF_DIR,
           "writable": os.access(_DIR, os.W_OK),
           "current_tag": _tag(), "scans": {}}
    try:
        for pth in sorted(glob.glob(os.path.join(_DIR, "*.pkl"))):
            nm = os.path.basename(pth)[:-4]
            try:
                with open(pth, "rb") as f:
                    blob = pickle.load(f)
                data = blob.get("data")
                rows = None
                if isinstance(data, dict):
                    for k in ("results", "stocks", "ranked", "picks", "leaders"):
                        if isinstance(data.get(k), list):
                            rows = len(data[k]); break
                out["scans"][nm] = {
                    "fresh": blob.get("tag") == _tag(),
                    "age_min": round((_t.time() - os.path.getmtime(pth)) / 60, 1),
                    "rows": rows,
                }
            except Exception as e:
                out["scans"][nm] = {"error": str(e)}
    except Exception as e:
        out["error"] = str(e)
    return out


def get_or_stale(name: str):
    """Client-facing read. Fresh result if the tag matches; otherwise the last-good
    result served INSTANTLY plus a background refresh (stale-while-revalidate); None
    only when nothing was ever computed. In force mode (prewarm) always returns None
    so the caller recomputes fresh."""
    try:
        path = os.path.join(_DIR, f"{name}.pkl")
        if not os.path.exists(path):
            return None
        with open(path, "rb") as f:
            blob = pickle.load(f)
    except Exception:
        return None
    data = blob.get("data")
    if blob.get("tag") == _tag():
        if not blob.get("breakdown_done"):
            _schedule_breakdown(name, data)
        return data                       # fresh
    if _in_force_mode() or data is None:
        return None                       # prewarm must recompute (or nothing cached)
    _kick_refresh()                       # stale: serve now, refresh behind the scenes
    if not blob.get("breakdown_done"):
        _schedule_breakdown(name, data)
    return data


# ── Universal breakdown annotation ───────────────────────────────────────────
# Every scanner's result funnels through put(), which makes this the ONE place a
# breakdown score can be attached to all 27 tabs without touching 27 modules.
#
# Why it earns its place (measured, not assumed — 76,773 observations over 6y):
#   score 0-1 -> +2.63% forward 21d return      score 4+ -> +0.83%   (-1.80pp)
#   and the edge SURVIVES inside every r6m momentum quintile (+0.49 to +1.88pp),
#   so it is not momentum in disguise.
# What it does NOT do: predict drawdown DEPTH. Forward maxDD is -9.04% at score
#   0-1 vs -9.82% at 4+ — essentially flat. It is a trend-deterioration signal,
#   not a crash predictor, and the UI must not claim otherwise.
#
# COST — and why this is NO LONGER done inside put() (2026-08-03):
#   ~2.4ms/row. put() is called once per scan, which sounded cheap. But when a new
#   bhavcopy lands, EVERY cache is invalidated and _prewarm_all_scans() re-runs all
#   ~20 scans back-to-back in one thread. Annotating inside put() therefore added
#   ~2s x 20 scans of GIL-holding Python work to the exact window in which the app
#   is already rebuilding everything — measured live on 2026-08-03, a static
#   /static/favicon.svg took 32s while that ran, and tabs sat on skeletons because
#   their requests never got scheduled.
#
#   It is now deferred: put() stores the raw result immediately, and the annotation
#   runs on a background daemon thread the first time the result is READ, then
#   re-persists itself with breakdown_done=True. No request and no prewarm ever
#   blocks on it; the badge simply appears a few seconds after a tab's first load.
_BREAKDOWN_KEYS = ("results", "stocks", "ranked", "top", "all", "picks")

_bd_lock = threading.Lock()
_bd_inflight: set[str] = set()


def _schedule_breakdown(name, data):
    """Annotate `data` off the request/prewarm path, once, then re-persist it."""
    if not isinstance(data, dict):
        return
    with _bd_lock:
        if name in _bd_inflight:
            return
        _bd_inflight.add(name)

    def _work():
        try:
            annotated = _annotate_breakdown(data)
            _write(name, annotated, breakdown_done=True)
        except Exception:
            pass
        finally:
            with _bd_lock:
                _bd_inflight.discard(name)

    threading.Thread(target=_work, daemon=True, name=f"breakdown:{name}").start()


def _annotate_breakdown(data):
    if not isinstance(data, dict):
        return data
    lists = [(k, v) for k, v in data.items()
             if k in _BREAKDOWN_KEYS and isinstance(v, list) and v
             and isinstance(v[0], dict) and "symbol" in v[0]]
    if not lists:
        return data
    try:
        import breakdown_detector as _bd
        import benchmark as _bm
        nifty = _bm.get_benchmark(days=900)
        # Use the WIDE universe. industry_groups._get_stocks() carries only ~810
        # curated names, but scanners range over ~2,588 — measured, that left 154
        # of Emerging Leaders' 292 rows unannotated (53% coverage on that tab).
        # load_base_universe() is cached after the first call (1.5s cold, 0.00s warm).
        try:
            import shared_universe as _su
            stocks = _su.load_base_universe(days=400)
        except Exception:
            import industry_groups as _ig
            stocks = _ig._get_stocks()
        if not stocks:
            return data
    except Exception:
        return data
    for _k, rows in lists:
        try:
            _bd.annotate(rows, stocks, nifty)
        except Exception:
            continue
    return data


def _write(name: str, data, breakdown_done: bool = False) -> None:
    """Atomic persist (tmp file + os.replace). Best-effort — never raises."""
    try:
        with _LOCK:
            os.makedirs(_DIR, exist_ok=True)
            tmp = os.path.join(_DIR, f"{name}.pkl.tmp")
            final = os.path.join(_DIR, f"{name}.pkl")
            with open(tmp, "wb") as f:
                pickle.dump({"tag": _tag(), "data": data,
                             "breakdown_done": breakdown_done},
                            f, protocol=pickle.HIGHEST_PROTOCOL)
            os.replace(tmp, final)
    except Exception:
        pass


def put(name: str, data) -> None:
    """Persist `data` for `name`, tagged with the current bhavcopy date.

    Deliberately does NO computation — see the note above _BREAKDOWN_KEYS. This is
    called once per scan, and during a post-bhavcopy rebuild that is ~20 calls in a
    row on one thread; anything expensive here stalls the whole server."""
    if data is None:
        return
    _write(name, data, breakdown_done=False)


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
