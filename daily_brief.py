"""
Daily Brief — the alphayantra-style home digest.

  • STRONG SECTORS — top uptrend sectors ranked by momentum of member stocks
    (straight from sector_rotation).
  • STRATEGIC ALPHA FRAMEWORK — uptrend stocks that carry at least one tag, with the
    days since the trend started and the return since. Tags, all from data we hold:
      52W HIGH   — within 2% of the 1-year high.
      PEAD       — a results date in the last ~120 days, positive EPS acceleration,
                   and the breakout came AFTER the result (post-earnings drift).
      LOW FLOAT  — promoter holding ≥ 60% (little free float).
      DEEP VALUE — trailing P/E between 0 and 15.

Your-Watchlist trend changes are added on the client from /api/watchlist. Everything
here is mechanical and cached; the trend rule is the SAME Donchian used everywhere.
"""
from __future__ import annotations

import datetime as _dt
import os
import sqlite3
import time

import numpy as np

import result_cache
import sector_rotation as sr

_cache = {"data": None, "ts": 0}
CACHE_TTL = 3600


def _fund_map() -> dict:
    try:
        dbp = os.path.join(os.environ.get("DATA_DIR", os.path.dirname(__file__)), "fundamentals.db")
        if not os.path.exists(dbp):
            dbp = os.path.join(os.path.dirname(__file__), "fundamentals.db")
        con = sqlite3.connect(dbp)
        out = {}
        for r in con.execute("SELECT symbol, pe_ratio, promoter_holding, result_date, eps_accel FROM fundamentals"):
            out[r[0]] = {"pe": r[1], "promoter": r[2], "result_date": r[3], "eps_accel": r[4]}
        con.close()
        return out
    except Exception:
        return {}


def _compute() -> dict:
    import shared_universe as su
    U = su.load_base_universe(days=800)
    if not U:
        return {"strong_sectors": [], "alpha_framework": [], "computed_at": int(time.time()),
                "error": "universe empty"}
    fmap = _fund_map()
    mcap = sr._mcap_map()

    # Strong sectors — reuse the sector rotation (already ranked by momentum).
    sec = sr.run_sector_rotation()
    strong_sectors = [{
        "sector": s["sector"], "since_date": s["since_date"], "trend": s["trend"],
        "member_count": s["member_count"], "stocks_rising": s["stocks_rising"],
        "avg_momentum_3m": s["avg_momentum_3m"], "spark": s.get("spark", []),
    } for s in sec.get("sectors", []) if s["trend"] == "up"][:8]

    last_date = next(iter(U.values())).index[-1]
    picks = []
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
        since_ret = round((last / c[since_idx] - 1.0) * 100, 2)
        since_days = (last_date - since_date).days

        tags = []
        w = c[-252:] if len(c) >= 252 else c
        if last >= float(w.max()) * 0.98:
            tags.append("52W HIGH")
        f = fmap.get(sym, {})
        rd = f.get("result_date")
        if rd and f.get("eps_accel") and f["eps_accel"] > 0:
            try:
                rdt = _dt.date.fromisoformat(str(rd)[:10])
                dsr = (last_date.date() - rdt).days
                if 0 <= dsr <= 120 and since_date.date() >= rdt:
                    tags.append("PEAD")
            except Exception:
                pass
        ph = f.get("promoter")
        if ph is not None and ph >= 60:
            tags.append("LOW FLOAT")
        pe = f.get("pe")
        if pe is not None and 0 < pe < 15:
            tags.append("DEEP VALUE")

        if not tags:
            continue
        picks.append({
            "symbol": sym, "price": round(last, 2),
            "mcap_cr": round(float(mcap[sym])) if sym in mcap and mcap[sym] else None,
            "tags": tags, "since_days": since_days, "since_ret": since_ret,
            "since_date": str(since_date.date()),
        })

    picks.sort(key=lambda x: x["since_ret"], reverse=True)
    picks = picks[:20]
    return {"strong_sectors": strong_sectors, "alpha_framework": picks,
            "computed_at": int(time.time()), "as_of": str(last_date.date()),
            "n_uptrend": sec.get("n_uptrend", 0), "total_sectors": sec.get("total_sectors", 0)}


def run_daily_brief(force: bool = False) -> dict:
    if not force and _cache["data"] and time.time() - _cache["ts"] < CACHE_TTL:
        return _cache["data"]
    if not force:
        disk = result_cache.get_or_stale("daily_brief")
        if disk is not None:
            _cache["data"] = disk
            _cache["ts"] = time.time()
            return disk
    data = _compute()
    _cache["data"] = data
    _cache["ts"] = time.time()
    try:
        result_cache.put("daily_brief", data)
    except Exception:
        pass
    return data


def invalidate_cache():
    _cache["data"] = None
    _cache["ts"] = 0
