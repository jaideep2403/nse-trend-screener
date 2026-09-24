"""
Monster Candidate score — one ranked list that fuses the signals big winners share
BEFORE the move, so early leaders stop hiding across separate tabs.

CALIBRATED FROM DATA (not folklore): a base-rate study of all 464 fresh 52-week-high
breakouts in liquid names (2025-04 → 2026-04) split by what happened next found only
~2.6% became ≥100% monsters within 120d. The features that actually SEPARATED the 🚀
monsters from the 💀 duds, in order of strength:
  1. SMALL CAP — the dominant lever. Monster hit-rate by cap band: <₹1k cr 16.7%,
     ₹1-3k 7.0%, ₹3-7k 4.5%, ₹7-15k 1.2%, ₹15-50k 1.8%, >₹50k 0.0% (monotonic).
  2. VOLUME BUILDING into the breakout (base back-half 2.4× the front-half vs 1.0× for
     duds) + a big IGNITION-day surge (5.5× base avg vs 3.0×). NOT dry-up.
  3. RELATIVE-STRENGTH leadership vs the index before the move (+61% vs +45%).
  4. LEADING sector/theme (capital goods, auto/defence, pharma over-indexed 3-7×).
  5. Accelerating earnings (kept from cache).
What did NOT separate them: base *tightness* — monsters had slightly DEEPER, not tighter,
bases (22.5% vs 18%). So the old tight-VCP-with-dry-up emphasis is de-weighted here.

Everything is computed from data we already have: OHLCV+delivery (bhavcopy), point-in-time
market cap (sector_rotation), the RS line vs Nifty (rs_line), the base detector, sector
rotation, and cached fundamentals (no new scraping — the SCRAPE_OFF sentinel stays honoured).
"""
from __future__ import annotations

import time

import numpy as np

import result_cache

MIN_ADTV_CR = 3.0        # tradeable
MAX_EXTENDED = 30.0      # reject names already >30% above the 50-DMA (too late)
CACHE_TTL = 3600
_cache = {"data": None, "ts": 0.0}


# ── factor helpers ───────────────────────────────────────────────────────────
def _episodic_pivot(o, h, l, c, v, look: int = 45):
    """Most recent 'episodic pivot': a gap-up (≥4%) or power day (≥8%) on ≥1.8× volume
    within `look` bars, that price has since HELD (didn't close back below the pivot
    day). This is the Qullamaggie-style day-one ignition of many big moves."""
    n = len(c)
    if n < 30:
        return None
    best = None
    for i in range(max(1, n - look), n):
        base = c[i - 1]
        if base <= 0:
            continue
        gap = (o[i] - base) / base
        day = (c[i] - base) / base
        lo = max(0, i - 20)
        avgv = float(np.mean(v[lo:i])) if i > lo else float(v[i])
        vr = (v[i] / avgv) if avgv > 0 else 0.0
        if (gap >= 0.04 or day >= 0.08) and vr >= 1.8:
            held = True
            if i < n - 1:
                held = float(np.min(c[i + 1:])) >= c[i] * 0.93   # held above the pivot close
            best = {"days_ago": n - 1 - i, "gain": round(day * 100, 1),
                    "gap": round(gap * 100, 1), "vr": round(vr, 1), "held": held}
    return best


def _group_strength() -> dict:
    """{sector: position_pct 0..1} — how strongly a sector is leading (from sector
    rotation). Missing sectors default to neutral (0.5)."""
    try:
        import sector_rotation as sr
        data = sr.run_sector_rotation()
        return {s["sector"]: s.get("position_pct", 0.5) for s in data.get("sectors", [])}
    except Exception:
        return {}


def _fund_factor(f: dict | None) -> tuple[float, dict]:
    """0..14 from cached fundamentals (EPS+sales YoY growth + acceleration flag)."""
    if not f:
        return 0.0, {}
    eps = f.get("eps_growth_yoy")
    sal = f.get("sales_growth_yoy")
    acc = f.get("eps_accel")
    score = 0.0
    if eps is not None:
        score += max(0.0, min(1.0, eps / 40.0)) * 7.0      # 40%+ EPS growth = full
    if sal is not None:
        score += max(0.0, min(1.0, sal / 25.0)) * 4.0      # 25%+ sales growth = full
    if acc:
        score += 3.0                                        # quarterly EPS accelerating
    return score, {"eps_g": eps, "sales_g": sal, "eps_accel": bool(acc)}


def _mcap_score(mc: float | None) -> float:
    """0..24 small-cap tilt — THE dominant separator. Mirrors the measured monster
    hit-rate ladder by cap band (see module docstring). Unknown cap → mild-neutral."""
    if mc is None:
        return 8.0
    if mc < 1000:    return 24.0
    if mc < 3000:    return 20.0
    if mc < 7000:    return 14.0
    if mc < 15000:   return 8.0
    if mc < 50000:   return 4.0
    return 0.0


def _vol_confirmation(v: np.ndarray) -> tuple[float, float]:
    """(build, surge): base back-half volume expansion, and the recent ignition-day
    surge vs base average. Monsters ran ~2.4× build and ~5.5× surge; duds ~1.0×/3.0×."""
    n = len(v)
    if n < 60:
        return 1.0, 1.0
    front = float(np.mean(v[-60:-40])) or 0.0
    back = float(np.mean(v[-20:]))
    build = (back / front) if front > 0 else 1.0
    base_avg = float(np.mean(v[-60:-10])) or 0.0
    surge = (float(np.max(v[-10:])) / base_avg) if base_avg > 0 else 1.0
    return build, surge


# ── main scan ────────────────────────────────────────────────────────────────
def _compute() -> dict:
    import shared_universe as su
    import base_detector as bd
    import rs_line as rl
    try:
        import sector_mapper as sm
        smap = sm.get_enriched_sector_map()
    except Exception:
        smap = {}
    try:
        import fundamentals as F
        allf = F.load_all_fundamentals()
    except Exception:
        allf = {}
    gstr = _group_strength()
    try:
        import sector_rotation as sr
        mcap_map = sr._mcap_map()
    except Exception:
        mcap_map = {}

    U = su.load_base_universe(days=400)
    if not U:
        return {"results": [], "computed_at": int(time.time()), "total_scanned": 0}
    rl.benchmark_series()   # warm the RS benchmark once

    rows, scanned = [], 0
    for sym, df in U.items():
        c = df["Close"].to_numpy(dtype=float)
        if len(c) < 120:
            continue
        v = df["Volume"].to_numpy(dtype=float)
        look = min(20, len(c))
        adtv = float((c[-look:] * v[-look:]).mean()) / 1e7
        if adtv < MIN_ADTV_CR:
            continue
        scanned += 1
        o = df["Open"].to_numpy(dtype=float)
        h = df["High"].to_numpy(dtype=float)
        l = df["Low"].to_numpy(dtype=float)
        px = c[-1]

        ma50 = float(c[-50:].mean())
        ma200 = float(c[-200:].mean()) if len(c) >= 200 else float(c.mean())
        ext = (px / ma50 - 1.0) * 100 if ma50 > 0 else 0.0
        uptrend = px >= ma50 and ma50 >= ma200

        # bases
        dts = [str(t.date()) for t in df.index]
        bases = bd.detect_bases(dts, o.tolist(), h.tolist(), l.tolist(),
                                c.tolist(), v.tolist())
        base = bases[-1] if bases else None
        forming = bool(base and base.get("forming"))
        to_pivot = base.get("now_vs_pivot_pct") if base else None   # +ve = above pivot
        tightening = base.get("tightening") if base else None       # lower = tighter
        vol_dryup = bool(base.get("vol_dryup")) if base else False

        # RS line vs Nifty
        rsf = rl.rs_signals_series(df["Close"])
        ep = _episodic_pivot(o, h, l, c, v)

        # ── gate: keep only EARLY setups ─────────────────────────────────────
        recent_ep = bool(ep and ep["held"] and ep["days_ago"] <= 30)
        has_leadership = rsf["above_zero"] or rsf["new_high"] or recent_ep
        early = (ext <= MAX_EXTENDED) and (uptrend or forming) and has_leadership
        if not early:
            continue

        # ── score the cluster (0..100), weights CALIBRATED to the monster study ──
        mc = mcap_map.get(sym)
        build, surge = _vol_confirmation(v)
        bars = len(c)
        sec = smap.get(sym)
        gpos = gstr.get(sec, 0.5)

        S = 0.0
        # 1. SMALL-CAP tilt (max 24) — the dominant separator
        S += _mcap_score(mc)
        # 2. VOLUME confirmation (max 22) — build into the base + ignition surge
        build_score = max(0.0, min(1.0, (build - 1.0) / 1.4)) * 12.0   # 2.4× -> full
        surge_score = max(0.0, min(1.0, (surge - 2.5) / 3.0)) * 10.0   # 5.5× -> full
        S += build_score + surge_score
        # 3. RS-line leadership (max 16)
        if rsf["above_zero"]:        S += 5
        if rsf["slope_up"]:          S += 3
        if rsf["new_high"]:          S += 5
        if rsf["high_before_price"]: S += 3
        # 4. Fundamentals accel (max 14) — from cache, no scraping
        fscore, fmeta = _fund_factor(allf.get(sym))
        S += fscore
        # 5. Leading sector/theme (max 12)
        S += max(0.0, (gpos - 0.5) / 0.5) * 12.0
        # 6. Readiness (max 8) — early, not extended: coiling under pivot / fresh Stage-2,
        #    a nod to youth, and a held episodic-pivot ignition
        if forming and to_pivot is not None and -10.0 <= to_pivot <= 2.0:
            S += 4.0 * (1.0 - min(1.0, abs(to_pivot) / 10.0))
        elif uptrend and 0 < ext <= 10:
            S += 2.0
        S += max(0.0, min(1.0, (500 - bars) / 400.0)) * 2.0
        if recent_ep and ep.get("held"):
            S += 2.0
        # 7. Tightness residual (max 4) — kept small; it did NOT separate winners
        if tightening is not None:
            S += max(0.0, min(1.0, (1.0 - tightening))) * 4.0

        rows.append({
            "symbol": sym,
            "score": round(min(100.0, S), 1),
            "price": round(px, 2),
            "mcap_cr": round(float(mc)) if mc else None,
            "vol_build": round(build, 1),
            "ign_surge": round(surge, 1),
            "rs_lead": bool(rsf["high_before_price"]),
            "rs_new_high": bool(rsf["new_high"]),
            "rs_above0": bool(rsf["above_zero"]),
            "eps_g": fmeta.get("eps_g"),
            "sales_g": fmeta.get("sales_g"),
            "eps_accel": fmeta.get("eps_accel", False),
            "tightening": tightening,
            "vol_dryup": vol_dryup,
            "base_type": base.get("type") if base else None,
            "base_weeks": base.get("weeks") if base else None,
            "to_pivot_pct": to_pivot,
            "forming": forming,
            "sector": sec,
            "grp_pos": round(gpos, 2),
            "ep_days": ep["days_ago"] if recent_ep else None,
            "ep_gain": ep["gain"] if recent_ep else None,
            "bars": bars,
            "ext_pct": round(ext, 1),
            "adtv_cr": round(adtv, 1),
            "rs": None,           # market-wide RS rating enriched by the endpoint
        })

    rows.sort(key=lambda r: r["score"], reverse=True)
    return {"results": rows[:80], "computed_at": int(time.time()),
            "total_scanned": scanned, "qualified": len(rows)}


def run_monster_scan(force: bool = False) -> dict:
    if not force and _cache["data"] and time.time() - _cache["ts"] < CACHE_TTL:
        return _cache["data"]
    if not force:
        disk = result_cache.get_or_stale("monster_candidate")
        if disk is not None:
            _cache.update(data=disk, ts=time.time())
            return disk
    data = _compute()
    _cache.update(data=data, ts=time.time())
    try:
        result_cache.put("monster_candidate", data)
    except Exception:
        pass
    return data


def invalidate_cache():
    _cache.update(data=None, ts=0)
