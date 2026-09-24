"""
Sector Rotation — the alphayantra-style list + drill-down, built on our own data.

For every sector in INDUSTRY_GROUPS:
  • current trend (up/down) and the date it STARTED — a mechanical Donchian breakout
    on the sector's equal-weight index (same rule as the Breakout-tab trend chart).
  • momentum sparkline (the sector index).
  • stocks-rising count — how many member stocks are currently in an uptrend.
  • lagging↔leading position — the sector's 3-month relative strength vs Nifty,
    rank-normalised across sectors.

Drill-down (member stocks), matching their columns:
  symbol · price · mcap ₹Cr · 1D% · breakout level · vs-breakout% · since-signal% · spark
where breakout level = the stock's current Donchian level (trailing support in an
uptrend, resistance in a downtrend); vs-breakout = price vs that level; since-signal =
return since the current trend started.

Nothing is asserted — the trend, the dates and the returns are all rule-based, and an
uptrend that has since faded shows a negative since-signal. Cached via result_cache.
"""
from __future__ import annotations

import os
import sqlite3
import time

import numpy as np

import result_cache

DONCHIAN_N = 44          # ~2 trading months; new N-day high => uptrend, new N-day low => downtrend
SPARK_POINTS = 32
_cache = {"data": None, "ts": 0}
CACHE_TTL = 3600


def _mcap_map() -> dict:
    try:
        dbp = os.path.join(os.environ.get("DATA_DIR", os.path.dirname(__file__)), "fundamentals.db")
        if not os.path.exists(dbp):
            dbp = os.path.join(os.path.dirname(__file__), "fundamentals.db")
        con = sqlite3.connect(dbp)
        m = {r[0]: r[1] for r in con.execute(
            "SELECT symbol, market_cap FROM fundamentals WHERE market_cap IS NOT NULL")}
        con.close()
        return m
    except Exception:
        return {}


def _donchian(close: np.ndarray, high: np.ndarray, low: np.ndarray, N: int = DONCHIAN_N) -> dict | None:
    """Current trend state, the bar it started, and the trailing level series.
    Mirrors the JS awlComputeTrend so the tab and the chart never disagree."""
    n = len(close)
    if n < N + 2:
        return None
    state = None
    since = 0
    level_last = None
    for i in range(1, n):
        lb = min(N, i)
        ph = float(high[i - lb:i].max())
        pl = float(low[i - lb:i].min())
        if state != "up" and close[i] >= ph:
            state, since = "up", i
        elif state != "down" and close[i] <= pl:
            state, since = "down", i
        level_last = pl if state == "up" else ph
    if state is None:
        return None
    return {"state": state, "since": since, "level": level_last}


def _spark(arr: np.ndarray, k: int = SPARK_POINTS) -> list:
    if len(arr) <= k:
        vals = arr
    else:
        idx = np.linspace(0, len(arr) - 1, k).round().astype(int)
        vals = arr[idx]
    return [round(float(x), 2) for x in vals]


def _pct(a, b):
    return round((a / b - 1.0) * 100, 2) if (b and b > 0) else None


def _compute() -> dict:
    import shared_universe as su
    # Use the ENRICHED groups (hand-curated INDUSTRY_GROUPS + NSE-auto-mapped extras
    # from the current TotalMarket classification) so each sector carries its latest
    # components, not just the ~523 hand-listed names. Falls back to the raw groups.
    try:
        import sector_mapper as _sm
        GROUPS = _sm.get_enriched_industry_groups()
    except Exception:
        import industry_groups as ig
        GROUPS = ig.INDUSTRY_GROUPS
    U = su.load_base_universe(days=800)
    if not U:
        return {"sectors": [], "computed_at": int(time.time()), "error": "universe empty"}
    mcap = _mcap_map()

    # Nifty proxy for relative strength (equal-weight mean of all members' returns is
    # noisy; use the benchmark for a stable lagging/leading axis).
    try:
        import benchmark as bm
        nb = bm.get_benchmark(days=800)
        nifty_r3 = float(nb.iloc[-1] / nb.iloc[-64] - 1.0) if nb is not None and len(nb) > 64 else 0.0
    except Exception:
        nifty_r3 = 0.0

    sectors = []
    for name, syms in GROUPS.items():
        members = []
        idx_frames = []
        rising = 0
        r3_list = []
        for sym in syms:
            df = U.get(sym)
            if df is None or len(df) < DONCHIAN_N + 5:
                continue
            c = df["Close"].to_numpy(dtype=float)
            h = df["High"].to_numpy(dtype=float)
            l = df["Low"].to_numpy(dtype=float)
            dt = df.index
            don = _donchian(c, h, l)
            if don is None:
                continue
            last = c[-1]
            d1 = _pct(last, c[-2]) if len(c) > 1 else None
            r3 = _pct(last, c[-64]) if len(c) > 64 else None
            if r3 is not None:
                r3_list.append(r3)
            if don["state"] == "up":
                rising += 1
            members.append({
                "symbol": sym,
                "price": round(float(last), 2),
                "mcap_cr": round(float(mcap[sym])) if sym in mcap and mcap[sym] else None,  # DB stores crores
                "d1_pct": d1,
                "trend": don["state"],
                "breakout_level": round(float(don["level"]), 2),
                "vs_breakout_pct": _pct(last, don["level"]),
                "since_date": str(dt[don["since"]].date()),
                "since_signal_pct": _pct(last, c[don["since"]]),
                "spark": _spark(c[-90:]),
                "spark_up": bool(c[-1] >= c[max(0, len(c) - 90)]),
            })
            idx_frames.append(df["Close"])

        if len(members) < 1:
            continue

        # Equal-weight sector index (normalise each member to its own start, then mean).
        import pandas as pd
        idx = None
        if idx_frames:
            al = pd.concat(idx_frames, axis=1).dropna()
            if len(al) > DONCHIAN_N + 5:
                norm = al / al.iloc[0]
                idx = norm.mean(axis=1)
        sec_don = None
        spark = []
        since_date = None
        if idx is not None:
            ic = idx.to_numpy(dtype=float)
            sec_don = _donchian(ic, ic, ic)
            spark = _spark(ic[-90:])
            if sec_don:
                since_date = str(idx.index[sec_don["since"]].date())

        members.sort(key=lambda m: (m["since_signal_pct"] if m["since_signal_pct"] is not None else -9999), reverse=True)
        avg_r3 = round(float(np.mean(r3_list)), 2) if r3_list else 0.0
        sectors.append({
            "sector": name,
            "member_count": len(members),
            "trend": (sec_don or {}).get("state", "up"),
            "since_date": since_date,
            "stocks_rising": rising,
            "avg_momentum_3m": avg_r3,
            "rs_vs_nifty": round(avg_r3 - nifty_r3 * 100, 2),
            "spark": spark,
            "stocks": members,
        })

    # Rank by momentum of member stocks (their mean 3-month return), like the site.
    sectors.sort(key=lambda s: s["avg_momentum_3m"], reverse=True)
    # lagging↔leading position = rank percentile of rs_vs_nifty
    rss = sorted(s["rs_vs_nifty"] for s in sectors)
    for s in sectors:
        r = sum(1 for x in rss if x <= s["rs_vs_nifty"])
        s["position_pct"] = round(r / len(rss), 3) if rss else 0.5
    return {"sectors": sectors, "computed_at": int(time.time()),
            "total_sectors": len(sectors),
            "n_uptrend": sum(1 for s in sectors if s["trend"] == "up")}


def run_sector_rotation(force: bool = False) -> dict:
    if not force and _cache["data"] and time.time() - _cache["ts"] < CACHE_TTL:
        return _cache["data"]
    if not force:
        disk = result_cache.get_or_stale("sector_rotation")
        if disk is not None:
            _cache["data"] = disk
            _cache["ts"] = time.time()
            return disk
    data = _compute()
    _cache["data"] = data
    _cache["ts"] = time.time()
    try:
        result_cache.put("sector_rotation", data)
    except Exception:
        pass
    return data


def invalidate_cache():
    _cache["data"] = None
    _cache["ts"] = 0
