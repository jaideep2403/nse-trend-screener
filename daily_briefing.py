"""
Top-down daily briefing for FortuneX.

Three shelves, all recomputed once per bhavcopy from our own EOD universe:

  1. MARKET REGIME   — participation read: % of rated names above their 30-week (150-day)
                       line now vs ~3 weeks ago, with a plain label and direction.
  2. LEADING SECTORS — each group's median RS, its day-over-day rotation (rank change),
                       and PARTICIPATION (% of members above their own 30-week line).
  3. WHAT CHANGED    — a session-over-session DIFF of every rated name's Weinstein-stage /
                       10-week / breakout state: Turned Stage 2, Reclaimed 10-week, Fresh
                       breakout, Lost 10-week, Turned down a stage.

Everything here is descriptive — a state, dated; never a buy/sell call. The compute is a
single pass over the cached universe (rolling 50/150-day means per name), so it is cheap.
"""
from __future__ import annotations

import json
import os
import statistics
import time

import result_cache

# Persist the last two sector rankings so "rotation" is a real day-over-day rank change.
_RANK_PATH = os.path.join(os.getenv("DATA_DIR", os.path.dirname(os.path.abspath(__file__))),
                          ".briefing_sector_rank.json")

_cache = {"data": None, "ts": 0}
_CACHE_TTL = 3600


# ── Per-name state (this session vs the prior session) ────────────────────────
def _state(close):
    """State of ONE name for the last bar and the bar before it, from a single close
    series. Weinstein stage (1 Basing / 2 Advancing / 3 Topping / 4 Declining) inlined
    from the shared method (analysis_utils.stage_analysis) but computed for BOTH bars off
    one pair of rolling means, so the whole universe diffs in one cheap pass.
    Returns None when there isn't enough history (need ~176 bars for a 150-MA + slope)."""
    if close is None or len(close) < 176:
        return None
    ma50 = close.rolling(50).mean()
    ma150 = close.rolling(150).mean()

    def _stage(i):
        # i is a negative index for the bar we're classifying; the slope look-back is the
        # 30-week MA 22 bars earlier (same ±0.5% hysteresis as analysis_utils.stage_analysis).
        try:
            cur = float(close.iloc[i])
            m50 = float(ma50.iloc[i]); m150 = float(ma150.iloc[i])
            m50p = float(ma50.iloc[i - 22]); m150p = float(ma150.iloc[i - 22])
        except Exception:
            return 0
        s50 = m50 > m50p * 1.005
        s150 = m150 > m150p * 1.005
        if cur > m50 and cur > m150 and s50 and s150:
            return 2
        if cur > m50 and not s150:
            return 3
        if cur < m150 and not s150:
            return 4
        return 1

    def _above(i, ma):
        try:
            return float(close.iloc[i]) > float(ma.iloc[i])
        except Exception:
            return None

    try:
        ext150 = float(close.iloc[-1]) / float(ma150.iloc[-1]) - 1.0
    except Exception:
        ext150 = None
    try:
        r3m = float(close.iloc[-1]) / float(close.iloc[-64]) - 1.0 if len(close) >= 64 else None
    except Exception:
        r3m = None
    return {
        "stage_now": _stage(-1),
        "stage_prev": _stage(-2),
        "a50_now": _above(-1, ma50),
        "a50_prev": _above(-2, ma50),
        "a150_now": _above(-1, ma150),
        "a150_3w": _above(-16, ma150),   # ~3 weeks ago, for the breadth delta
        "ext150": ext150,                # extension above the 30-week line (how stretched)
        "r3m": r3m,                      # ~3-month return (is it genuinely advancing?)
    }


# Weinstein strength ordering for "turned up/down a stage": Advancing is strongest, then
# Basing (constructive), then Topping, then Declining (weakest).
_STRENGTH = {2: 3, 1: 2, 3: 1, 4: 0}


def _load_ranks():
    try:
        with open(_RANK_PATH) as f:
            return json.load(f)
    except Exception:
        return {}


def _rotation_and_persist(secs, tag):
    """Attach `rot` (rank change vs the previous SESSION) to each sector and roll the
    two-slot store forward. Positive rot = the sector climbed the strength ranking."""
    store = _load_ranks()
    cur_slot = store.get("cur") or {}
    prev_rank = {}
    if cur_slot.get("tag") and cur_slot.get("tag") != tag:
        prev_rank = cur_slot.get("rank") or {}          # yesterday becomes the baseline
    else:
        prev_rank = (store.get("prev") or {}).get("rank") or {}   # same-day recompute
    for i, s in enumerate(secs):
        pr = prev_rank.get(s["sector"])
        s["rot"] = (pr - (i + 1)) if pr else 0
    cur_rank = {s["sector"]: i + 1 for i, s in enumerate(secs)}
    try:
        new_store = {"cur": {"tag": tag, "rank": cur_rank}}
        if cur_slot.get("tag") and cur_slot.get("tag") != tag:
            new_store["prev"] = cur_slot
        else:
            new_store["prev"] = store.get("prev") or cur_slot
        with open(_RANK_PATH, "w") as f:
            json.dump(new_store, f)
    except Exception:
        pass


def compute(rs_map: dict, sector_map: dict, breakout_syms=None,
            universe_days: int = 400) -> dict:
    """Build the briefing. `rs_map` {sym: rs 1-99} defines the rated universe (mirrors the
    screener); `sector_map` {sym: sector}; `breakout_syms` an iterable of names that just
    broke out (from the breakout scan)."""
    import shared_universe as su
    try:
        from data_fetcher import _latest_bhavcopy_date
        d = _latest_bhavcopy_date()
        tag = d.isoformat() if d else "nodate"
    except Exception:
        tag = "nodate"

    U = su.load_base_universe(days=universe_days)
    bo = set(breakout_syms or [])

    turned_s2, reclaimed, lost, turned_down, aligned = [], [], [], [], []
    n = above150_now = above150_3w = 0
    sectors: dict[str, dict] = {}

    for sym, df in U.items():
        rs = rs_map.get(sym)
        if rs is None:                       # only rated names — the screener's universe
            continue
        cl = df.get("Close") if hasattr(df, "get") else df["Close"]
        st = _state(cl)
        if st is None:
            continue

        # ── Alignment (glass-box composite): a leader that is advancing, still-early, and
        #    taking fresh money — scored on three lenses so the read shows its working.
        #      Trend  = RS leadership (gated to a Weinstein advancing stage)
        #      Entry  = still-early: how little it is stretched above its 30-week line
        #      Money  = smart-money inflow: 5-day vs 22-day turnover surge
        #    We only score names that clear the floor (Stage 2 · RS 70+ · positive 3-month ·
        #    within 15% of the 30-week line · money coming in) — a description, not a call.
        if (st["stage_now"] == 2 and rs >= 70 and st["ext150"] is not None
                and st["ext150"] <= 0.15 and st["r3m"] is not None and st["r3m"] > 0):
            amt = 0.0
            try:
                vol = df["Volume"].astype(float)
                turn = cl.astype(float) * vol
                a5 = float(turn.iloc[-5:].mean()); a22 = float(turn.iloc[-22:].mean())
                amt = (a5 / a22 - 1.0) if a22 > 0 else 0.0
            except Exception:
                amt = 0.0
            if amt > 0:
                trend_s = int(rs)
                entry_s = max(0, min(100, round(100 - max(0.0, st["ext150"] - 0.05) * 450)))
                money_s = max(0, min(100, round(50 + amt * 80)))
                align = round(0.40 * trend_s + 0.35 * entry_s + 0.25 * money_s)
                aligned.append({"symbol": sym, "rs": rs, "align": align,
                                "trend": trend_s, "entry": entry_s, "money": money_s,
                                "ext": round(st["ext150"] * 100, 1)})
        n += 1
        if st["a150_now"]:
            above150_now += 1
        if st["a150_3w"]:
            above150_3w += 1

        sec = sector_map.get(sym) or ""
        if sec:
            sm = sectors.setdefault(sec, {"rs": [], "above": 0, "n": 0})
            sm["rs"].append(rs)
            sm["n"] += 1
            if st["a150_now"]:
                sm["above"] += 1

        pill = {"symbol": sym, "rs": rs}
        if st["stage_now"] == 2 and st["stage_prev"] != 2:
            turned_s2.append(pill)
        if st["a50_now"] and (st["a50_prev"] is False):
            reclaimed.append(pill)
        if (st["a50_now"] is False) and st["a50_prev"]:
            lost.append(pill)
        s_now, s_prev = st["stage_now"], st["stage_prev"]
        if s_now and s_prev and _STRENGTH.get(s_now, 9) < _STRENGTH.get(s_prev, 9):
            turned_down.append(pill)

    fresh_bo = [{"symbol": s, "rs": rs_map.get(s)}
                for s in bo if rs_map.get(s) is not None]

    for g in (turned_s2, reclaimed, lost, turned_down, fresh_bo):
        g.sort(key=lambda p: (p["rs"] if p["rs"] is not None else -1), reverse=True)
    aligned.sort(key=lambda p: p["align"], reverse=True)

    # ── Regime (participation) ──
    pct_now = round(100 * above150_now / n) if n else 0
    pct_3w = round(100 * above150_3w / n) if n else 0
    delta = pct_now - pct_3w
    if pct_now >= 66:
        label, tone = "Strong", "up"
    elif pct_now >= 40:
        label, tone = "Mixed", "flat"
    else:
        label, tone = "Weak", "down"
    direction = "improving" if delta > 2 else "deteriorating" if delta < -2 else "steady"

    # ── Leading sectors (median RS · participation · rotation) ──
    secs = []
    for sec, sm in sectors.items():
        if sm["n"] < 5:      # skip thin groups so a 3-stock niche can't headline leadership
            continue
        secs.append({
            "sector": sec,
            "med_rs": round(statistics.median(sm["rs"]), 1),
            "participation": round(100 * sm["above"] / sm["n"]),
            "n": sm["n"],
        })
    secs.sort(key=lambda s: s["med_rs"], reverse=True)
    _rotation_and_persist(secs, tag)

    return {
        "as_of": tag,
        "regime": {"label": label, "tone": tone, "pct_above_30w": pct_now,
                   "pct_3w_ago": pct_3w, "delta_3w": delta, "direction": direction,
                   "n": n},
        "leading_sectors": secs[:8],
        "all_sectors": secs,
        "aligned": aligned[:12],
        "changed": {
            "turned_s2":   turned_s2,
            "reclaimed":   reclaimed,
            "fresh_bo":    fresh_bo,
            "lost":        lost,
            "turned_down": turned_down,
        },
        "computed_at": int(time.time()),
    }


def _bhav_tag() -> str:
    try:
        from data_fetcher import _latest_bhavcopy_date
        d = _latest_bhavcopy_date()
        return d.isoformat() if d else "nodate"
    except Exception:
        return "nodate"


# ── "All stocks" screener table ───────────────────────────────────────────────
_STAGE_LABEL = {1: "Basing", 2: "Advancing", 3: "Topping", 4: "Declining"}
_ALL_CACHE = {"data": None, "ts": 0}


def all_stocks(rs_map: dict, sector_map: dict, bo_map=None,
               universe_days: int = 400, force: bool = False) -> dict:
    """Every rated name as one screener row: RS · Weinstein stage · setup · money-flow
    (5d/22d turnover surge) · extension vs 30-week · 52-week-range position · 3-month
    return · a 3-month trend sparkline. Cached per bhavcopy; a single universe pass."""
    tag = _bhav_tag()
    c = _ALL_CACHE["data"]
    if not force and c and c.get("as_of") == tag and time.time() - _ALL_CACHE["ts"] < _CACHE_TTL:
        return c
    if not force:
        disk = result_cache.get_or_stale("all_stocks")
        if disk and disk.get("as_of") == tag:
            _ALL_CACHE.update(data=disk, ts=time.time())
            return disk

    import shared_universe as su
    U = su.load_base_universe(days=universe_days)
    bo_map = bo_map or {}
    rows = []
    for sym, df in U.items():
        rs = rs_map.get(sym)
        if rs is None:
            continue
        cl = df.get("Close") if hasattr(df, "get") else df["Close"]
        if cl is None or len(cl) < 176:
            continue
        try:
            c = cl.astype(float)
            ma50 = c.rolling(50).mean()
            ma150 = c.rolling(150).mean()
            last = float(c.iloc[-1])
            m50 = float(ma50.iloc[-1]); m150 = float(ma150.iloc[-1])
            m50p = float(ma50.iloc[-22]); m150p = float(ma150.iloc[-22])
            s50 = m50 > m50p * 1.005
            s150 = m150 > m150p * 1.005
            if last > m50 and last > m150 and s50 and s150:
                stg = 2
            elif last > m50 and not s150:
                stg = 3
            elif last < m150 and not s150:
                stg = 4
            else:
                stg = 1
            ext150 = last / m150 - 1.0 if m150 > 0 else 0.0
            r3m = last / float(c.iloc[-64]) - 1.0 if len(c) >= 64 else 0.0
            win = c.iloc[-252:]
            hi = float(win.max()); lo = float(win.min())
            pos52 = round(100 * (last - lo) / (hi - lo)) if hi > lo else 50
            a50 = last > m50
        except Exception:
            continue

        money = None
        try:
            vol = df["Volume"].astype(float)
            turn = c * vol
            a5 = float(turn.iloc[-5:].mean()); a22 = float(turn.iloc[-22:].mean())
            money = round(a5 / a22, 1) if a22 > 0 else None
        except Exception:
            money = None

        bo = bo_map.get(sym) or {}
        bd = bo.get("breakout_days_ago")
        if bo.get("from_base") and (bd is None or bd <= 3):
            setup = "Fresh BO"
        elif ext150 >= 0.20 or r3m >= 0.35:
            setup = "Extended"
        elif not a50:
            setup = "Slipping"
        elif pos52 >= 92:
            setup = "Near highs"
        elif stg == 2:
            setup = "Pullback"
        else:
            setup = "—"

        # 3-month sparkline — ~63 bars down-sampled to ~14 points, normalised 0-100.
        seg = c.iloc[-63:] if len(c) >= 63 else c
        step = max(1, len(seg) // 14)
        pts = list(seg.iloc[::step])[-14:]
        smn = min(pts); smx = max(pts); rg = (smx - smn) or 1.0
        spark = [round((v - smn) / rg * 100) for v in pts]

        rows.append({
            "symbol": sym, "sector": sector_map.get(sym) or "",
            "rs": rs, "stage": stg, "stage_label": _STAGE_LABEL.get(stg, "—"),
            "setup": setup, "money": money, "ext": round(ext150 * 100, 1),
            "pos52": pos52, "r3m": round(r3m * 100, 1), "spark": spark,
        })

    rows.sort(key=lambda r: (r["rs"] if r["rs"] is not None else -1), reverse=True)
    out = {"as_of": tag, "rows": rows, "n": len(rows), "computed_at": int(time.time())}
    _ALL_CACHE.update(data=out, ts=time.time())
    result_cache.put("all_stocks", out)
    return out


def run(rs_map, sector_map, breakout_syms=None, force: bool = False) -> dict:
    """Cached briefing, keyed by bhavcopy date so it refreshes the moment new data lands
    (compute is ~0.2s, so a lazy first hit after a new session is cheap)."""
    tag = _bhav_tag()
    c = _cache["data"]
    if not force and c and c.get("as_of") == tag and time.time() - _cache["ts"] < _CACHE_TTL:
        return c
    if not force:
        disk = result_cache.get_or_stale("briefing")
        if disk and disk.get("as_of") == tag:
            _cache["data"] = disk
            _cache["ts"] = time.time()
            return disk
    data = compute(rs_map, sector_map, breakout_syms)
    _cache["data"] = data
    _cache["ts"] = time.time()
    result_cache.put("briefing", data)
    return data


def invalidate_cache():
    _cache["data"] = None
    _cache["ts"] = 0
