"""
Coiling scan — stocks that are BUILDING a base right now and pressing up under their
pivot, i.e. the pre-breakout side of the Breakout tab. Uses the same multi-base
detector the charts draw: a name qualifies when its MOST-RECENT base is still forming
(not yet broken out), price sits within a few percent below the pivot, the base has
lasted at least a few weeks, and it's doing this in an uptrend (above the 50-DMA).

Ranked by "readiness": closest to the pivot, tightest, and strongest wins — those are
the ones most likely to attempt the breakout next. Liquidity-gated (ADTV ≥ ₹2cr) so
the list stays tradeable. Cached like every other scan. Nothing is asserted — a coil
can still fail; this is a watchlist, not a signal.
"""
from __future__ import annotations

import time

import numpy as np

import result_cache

MIN_ADTV_CR = 2.0        # liquid enough to trade
MIN_WEEKS   = 3          # a real consolidation, not a 3-day pause
NEAR_PIVOT  = 12.0       # within this % BELOW the pivot ("pressing up under it")
_cache = {"data": None, "ts": 0.0}
CACHE_TTL = 3600


def _compute() -> dict:
    import shared_universe as su
    import base_detector as bd
    U = su.load_base_universe(days=400)
    if not U:
        return {"results": [], "computed_at": int(time.time()), "total_scanned": 0}

    rows = []
    scanned = 0
    for sym, df in U.items():
        c = df["Close"].to_numpy(dtype=float)
        if len(c) < 60:
            continue
        v = df["Volume"].to_numpy(dtype=float)
        # liquidity floor
        look = min(20, len(c))
        adtv = float((c[-look:] * v[-look:]).mean()) / 1e7
        if adtv < MIN_ADTV_CR:
            continue
        scanned += 1
        o = df["Open"].to_numpy(dtype=float)
        h = df["High"].to_numpy(dtype=float)
        l = df["Low"].to_numpy(dtype=float)
        dts = [str(t.date()) for t in df.index]
        bases = bd.detect_bases(dts, o.tolist(), h.tolist(), l.tolist(), c.tolist(), v.tolist())
        if not bases:
            continue
        last = bases[-1]
        if not last.get("forming"):
            continue                       # already broke out or no live base
        pivot = last["pivot"]
        px = float(c[-1])
        to_pivot = (pivot / px - 1.0) * 100 if px > 0 else 999   # % move needed to clear the pivot
        if not (0 < to_pivot <= NEAR_PIVOT):
            continue                       # too far under the pivot (or already through it)
        if last["weeks"] < MIN_WEEKS:
            continue
        ma50 = float(c[-50:].mean())
        if px < ma50:
            continue                       # base must be in an uptrend, not a downtrend
        rows.append({
            "symbol":       sym,
            "price":        round(px, 2),
            "pivot":        round(pivot, 2),
            "to_pivot_pct": round(to_pivot, 2),       # how far under the pivot it sits
            "weeks":        last["weeks"],
            "depth_pct":    last["depth_pct"],
            "type":         last["type"],
            "tightening":   last.get("tightening"),
            "vol_dryup":    last.get("vol_dryup"),
            "adtv_cr":      round(adtv, 1),
            "since_date":   last["start_date"],
            "rs":           None,                       # enriched by the endpoint
        })

    # Readiness: closest to the pivot first, then tightest, then deepest-dried-up.
    def _ready(r):
        tight = r["tightening"] if r["tightening"] is not None else 1.0
        return (r["to_pivot_pct"], tight)
    rows.sort(key=_ready)
    return {"results": rows, "computed_at": int(time.time()), "total_scanned": scanned}


def run_coiling_scan(force: bool = False) -> dict:
    if not force and _cache["data"] and time.time() - _cache["ts"] < CACHE_TTL:
        return _cache["data"]
    if not force:
        disk = result_cache.get_or_stale("coiling")
        if disk is not None:
            _cache.update(data=disk, ts=time.time())
            return disk
    data = _compute()
    _cache.update(data=data, ts=time.time())
    try:
        result_cache.put("coiling", data)
    except Exception:
        pass
    return data


def invalidate_cache():
    _cache.update(data=None, ts=0)
