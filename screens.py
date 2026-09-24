"""
PEAD and Deep Value screens — alphayantra-style, on our own data.

PEAD (post-earnings-announcement drift): uptrend stocks whose EARNINGS are accelerating
(eps_accel > 0) — the mechanical proxy for "results beat, price is drifting up". Where
NSE's results feed lines up (earnings_dates), the actual result date is shown; note the
data window here runs ahead of that feed, so the date is context, not a hard filter.

DEEP VALUE: uptrend stocks that are still cheap — trailing P/E in (0, DV_PE_MAX].

Both carry the same Donchian trend fields as every other tab (trend, since date, days,
return since signal) so a stock reads identically wherever it appears. Cached.
"""
from __future__ import annotations

import os
import sqlite3
import time

import result_cache
import sector_rotation as sr

DV_PE_MAX = 20.0
FRESH_MAX_DAYS = 400          # "still drifting" — trend started within the data window
_cache = {"data": None, "ts": 0}
CACHE_TTL = 3600


def _fund_map() -> dict:
    try:
        dbp = os.path.join(os.environ.get("DATA_DIR", os.path.dirname(__file__)), "fundamentals.db")
        if not os.path.exists(dbp):
            dbp = os.path.join(os.path.dirname(__file__), "fundamentals.db")
        con = sqlite3.connect(dbp)
        out = {}
        for r in con.execute("SELECT symbol, pe_ratio, eps_accel, promoter_holding, roe FROM fundamentals"):
            out[r[0]] = {"pe": r[1], "eps_accel": r[2], "promoter": r[3], "roe": r[4]}
        con.close()
        return out
    except Exception:
        return {}


def _sector_of() -> dict:
    try:
        import industry_groups as ig
        m = {}
        for name, syms in ig.INDUSTRY_GROUPS.items():
            for s in syms:
                m[s] = name
        return m
    except Exception:
        return {}


def _compute() -> dict:
    import shared_universe as su
    U = su.load_base_universe(days=800)
    if not U:
        return {"pead": [], "deep_value": [], "computed_at": int(time.time()), "error": "universe empty"}
    fmap = _fund_map()
    mcap = sr._mcap_map()
    sect = _sector_of()
    try:
        import earnings_dates as ed
        edates = ed.get_earnings_dates()
    except Exception:
        edates = {}

    last_date = next(iter(U.values())).index[-1]
    pead, dv = [], []
    for sym, df in U.items():
        c = df["Close"].to_numpy(dtype=float)
        h = df["High"].to_numpy(dtype=float)
        l = df["Low"].to_numpy(dtype=float)
        if len(c) < sr.DONCHIAN_N + 5:
            continue
        don = sr._donchian(c, h, l)
        if not don or don["state"] != "up":
            continue
        last = float(c[-1])
        since_idx = don["since"]
        since_date = df.index[since_idx]
        since_days = (last_date - since_date).days
        base = {
            "symbol": sym,
            "sector": sect.get(sym, "—"),
            "price": round(last, 2),
            "mcap_cr": round(float(mcap[sym])) if sym in mcap and mcap[sym] else None,
            "d1_pct": round((last / c[-2] - 1) * 100, 2) if len(c) > 1 else None,
            "trend": "up",
            "breakout_level": round(float(don["level"]), 2),
            "vs_breakout_pct": round((last / don["level"] - 1) * 100, 2),
            "since_date": str(since_date.date()),
            "since_days": since_days,
            "since_signal_pct": round((last / c[since_idx] - 1) * 100, 2),
            "spark": sr._spark(c[-90:]),
            "spark_up": bool(c[-1] >= c[max(0, len(c) - 90)]),
        }
        f = fmap.get(sym, {})
        # PEAD — accelerating earnings + uptrend
        ea = f.get("eps_accel")
        if ea is not None and ea > 0:
            pead.append({**base, "eps_accel": round(float(ea), 1),
                         "result_date": edates.get(sym), "pe": f.get("pe")})
        # DEEP VALUE — cheap + uptrend
        pe = f.get("pe")
        if pe is not None and 0 < pe <= DV_PE_MAX:
            dv.append({**base, "pe": round(float(pe), 1), "roe": f.get("roe"),
                       "eps_accel": round(float(ea), 1) if ea is not None else None})

    pead.sort(key=lambda x: x["since_signal_pct"], reverse=True)
    dv.sort(key=lambda x: (x["pe"] if x["pe"] else 999))     # cheapest first
    return {"pead": pead[:80], "deep_value": dv[:150],
            "pead_total": len(pead), "dv_total": len(dv),
            "computed_at": int(time.time()), "as_of": str(last_date.date()),
            "earnings_dates_loaded": len(edates)}


def run_screens(force: bool = False) -> dict:
    if not force and _cache["data"] and time.time() - _cache["ts"] < CACHE_TTL:
        return _cache["data"]
    if not force:
        disk = result_cache.get_or_stale("screens_pead_dv")
        if disk is not None:
            _cache["data"] = disk
            _cache["ts"] = time.time()
            return disk
    data = _compute()
    _cache["data"] = data
    _cache["ts"] = time.time()
    try:
        result_cache.put("screens_pead_dv", data)
    except Exception:
        pass
    return data


def invalidate_cache():
    _cache["data"] = None
    _cache["ts"] = 0
