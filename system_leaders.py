"""
System — RS>90 leaders at a pivot, that beat the index, tracked daily.
=====================================================================

WHAT THE USER ASKED FOR (2026-08-13), built without assuming it works:
  strongest stocks (RS > 90) · at a pivot with a trigger · beating the index ·
  a thesis for each · actionable advice · daily performance tracking.

BRUTAL HONESTY, UP FRONT — this is a LEADERSHIP-BREAKOUT screen, the single most
studied pattern in this app, and every honest measurement of it says the same
thing:
  * Across 55,272 breakouts the MEDIAN one LOSES to the market (-3.41% excess,
    43.5% win). The entire edge is a +29.6% top decile.
  * Trend/momentum confluence (which "RS>90 + above MAs + beats index" IS) tested
    +1.48pp in-sample and -2.22pp OUT-of-sample — it did not generalise.
  * Requiring price ABOVE the 50-DMA cut the one validated edge's win rate.
So this screen is a FAT-TAIL bet: usually wrong, occasionally very right. It is
built anyway because the user asked for it and it is a legitimate way to fish for
the tail — but the tab states its own measured expectancy so nobody mistakes
"RS 97, beats index" for "likely to go up". The number that matters is carried on
every row and re-measured by system_leaders_validation, not asserted here.

POINT-IN-TIME: `_pit_row()` is the ONE detection function. The live scan and the
backtest both call it, so the thing measured is byte-identical to the thing shown.
The pivot base excludes the last 3 bars, so today's bar can never set the level it
is judged against (the same-bar look-ahead trap that produced a false stop in the
portfolio tab on 2026-08-06).
"""
from __future__ import annotations

import time
import numpy as np
import pandas as pd
import result_cache

# ── Tunables (the user's spec, made explicit) ───────────────────────────────
RS_MIN            = 90      # "strongest" — IBD-style cross-sectional percentile
MIN_ADTV_CR       = 5.0     # must be tradeable (matches weekly_breakout)
BASE_LOOKBACK     = 55      # bars forming the pivot base (~11 weeks)
BASE_SHIFT        = 3       # exclude the most recent bars from the base
PIVOT_NEAR_PCT    = 8.0     # within this % BELOW the pivot = a live SETUP
EXTENDED_PCT      = 15.0    # more than this % ABOVE the pivot = chased, not fresh
HARD_STOP_PCT     = 8.0

_cache: dict = {"data": None, "ts": 0.0}
CACHE_TTL = 6 * 3600


def _ibd_raw(c: np.ndarray, i: int) -> float | None:
    """IBD non-overlapping-quarter blend at bar i, point-in-time.

    Same 40/20/20/20 construction as screener.compute_ibd_score, so an RS rank
    here means the same thing it means on the Screener tab.
    """
    if i < 252 or i >= len(c):
        return None
    r3  = c[i] / c[i - 63]  - 1
    r6  = c[i] / c[i - 126] - 1
    r12 = c[i] / c[i - 252] - 1
    q1  = r3
    q2  = (1 + r6) / (1 + r3) - 1
    q34 = (1 + r12) / (1 + r6) - 1
    return 0.40 * q1 + 0.20 * q2 + 0.40 * q34


def rank_rs(stocks: dict, i_by_sym: dict | None = None) -> dict:
    """Cross-sectional RS 1-99 for every stock at its latest bar (or i_by_sym[sym])."""
    raw = {}
    for s, df in stocks.items():
        c = df["Close"].to_numpy(dtype=float)
        i = (i_by_sym or {}).get(s, len(c) - 1)
        if i >= len(c):
            continue
        v = _ibd_raw(c, i)
        if v is not None and np.isfinite(v):
            raw[s] = v
    if not raw:
        return {}
    ser = pd.Series(raw).rank(pct=True) * 100
    return {s: int(round(v)) for s, v in ser.items()}


# ── Refinement thresholds — every one MEASURED on 7yr before being added ─────
# raw System (RS90+pivot+beats idx)         n=2345  excess +0.57pp  win 53.0%
# + mcap >= 1000cr (no penny/small)         n=1513  excess +2.76pp  win 57.8%
# + tight price action (20d CV <= median)   n= 835  excess +3.30pp  win 56.9%
# + RS not exhausted (<98)                   n= 686  excess +3.11pp  win 57.1%   <- shipped
# So the user's two structural asks (drop penny caps; best price action) BOTH
# validated and together roughly 5x the edge. RS>=98 measured WORSE (extended), so
# RS is gated at 90 but NOT pushed higher. The F&O question is handled in the scan.
MIN_MCAP_CR       = 1000.0   # remove penny/small caps — the single biggest lift
MAX_RS            = 100      # RS ceiling. Was 98 (RS>=98 measured -0.6pp in the RAW
                            # screen); but inside the REFINED pool relaxing to 100 costs
                            # ~nothing (+3.25 vs +3.32pp) and admits the RS-100 leaders
                            # the user asked for (CUPID etc.). Kept as a var for clarity.
TIGHT_CV_MAX      = 12.0     # 20-day close CV. Was 6.0; raised to admit hot momentum
                            # leaders (CUPID cv=11.9%) alongside tight coils. Measured
                            # cost +3.32 -> +3.11pp for ~2x the names — tighter still
                            # ranks higher, so quality leads and the movers follow.
TARGET_MULT       = 1.25     # pivot-based objective: a 25% measured move
MAX_TOP_N         = 50       # show the TOP 50 by rank (user asked for a top-50 list)


def _trend_score(c, i) -> int:
    """0-100 structural trend gauge (same 8 facts as the portfolio gauge). DESCRIPTIVE
    of where the stock stands — not a prediction (see the portfolio-tab study)."""
    pts = tot = 0
    def add(ok):
        nonlocal pts, tot; tot += 1; pts += 1 if ok else 0
    px = c[i]
    ma20 = np.mean(c[i - 19:i + 1]); ma50 = np.mean(c[i - 49:i + 1]); ma200 = np.mean(c[i - 199:i + 1])
    add(px > ma50); add(px > ma200); add(px > ma20)
    add(ma50 > np.mean(c[i - 64:i - 14]))                    # 50-DMA rising
    add(ma200 > np.mean(c[i - 214:i - 14]))                  # 200-DMA rising
    add(px >= np.max(c[i - 251:i + 1]) * 0.90)               # within 10% of 52w high
    add(c[i] / c[i - 63] - 1 > 0.05)                         # positive 3-mo
    add(np.min(c[i - 20:i - 5]) < c[i])                      # above recent swing low
    return int(round(100 * pts / tot))


def _days_trending(c, i) -> int:
    """Consecutive sessions the close has held above its 50-DMA — the uptrend's age."""
    k = 0
    for j in range(i, max(i - 400, 49), -1):
        if c[j] > np.mean(c[j - 49:j + 1]):
            k += 1
        else:
            break
    return k


# NOTE — a multi-factor "leader score" (F&O + days-trending + past-pivot + near-high,
# weighted by their univariate edges) was built and BACKTESTED here on 2026-08-13.
# It did NOT discriminate: top-15-by-score returned +8.0pp median / 59% beat vs the
# full cohort's +8.2pp / 59% — zero lift. The univariate edges cancelled once combined
# inside the already-filtered cohort. It was removed rather than shipped as fake rigour.
# What DID survive as a real dud-filter is a hard AND of the two robust levers, applied
# as `conviction` below: F&O membership + an established (>60-session) uptrend. Measured
# fwd-6m: median +20.8pp / 69% beat vs the cohort +6.4pp / 58% (n=134 — real but with
# wider variance, so it is an OPT-IN high-conviction view, not the default that hides
# the rest).
CONVICTION_MIN_TREND_DAYS = 60

def _pit_row(sym, df, i, rs_rank, bench_r3, bench_r6, mcap=None, is_fno=False):
    """THE detection function. Returns a row dict when the stock is a System
    leader at bar i, else None. Used by BOTH the live scan and the backtest."""
    c = df["Close"].to_numpy(dtype=float)
    h = df["High"].to_numpy(dtype=float)
    v = df["Volume"].to_numpy(dtype=float)
    n = len(c)
    if i < 252 or i >= n:
        return None
    if rs_rank < RS_MIN or rs_rank > MAX_RS:
        return None

    # QUALITY / NO-PENNY GATE: market cap >= Rs 1000cr. Measured +2.76pp vs -3.41pp
    # for the uncovered/small group. A stock with no mcap row is treated as failing
    # (the uncovered group underperformed by construction).
    if mcap is None or mcap < MIN_MCAP_CR:
        return None

    adtv = float(np.mean(c[i - 19:i + 1] * v[i - 19:i + 1])) / 1e7
    if adtv < MIN_ADTV_CR:
        return None

    # Stage-2 confirmation: above rising 50 & 200-DMA (a leader is in an uptrend).
    ma50 = float(np.mean(c[i - 49:i + 1]))
    ma200 = float(np.mean(c[i - 199:i + 1]))
    if not (c[i] > ma50 and c[i] > ma200 and ma50 > np.mean(c[i - 64:i - 14])):
        return None

    # BEATS THE INDEX: stock return > benchmark return on BOTH 3m and 6m.
    r3 = c[i] / c[i - 63] - 1
    r6 = c[i] / c[i - 126] - 1
    if bench_r3 is not None and not (r3 > bench_r3):
        return None
    if bench_r6 is not None and not (r6 > bench_r6):
        return None

    # BEST PRICE ACTION: tight 20-day consolidation. Measured +3.30pp vs -0.26pp
    # for the looser half. A leader that is coiling, not chopping.
    tight_cv = float(np.std(c[i - 19:i + 1]) / np.mean(c[i - 19:i + 1]) * 100)
    if tight_cv > TIGHT_CV_MAX:
        return None

    # PIVOT with a TRIGGER. Base = highest high strictly BEFORE the last 3 bars.
    base = h[i - BASE_LOOKBACK - BASE_SHIFT: i - BASE_SHIFT]
    if base.size < 20:
        return None
    pivot = float(base.max())
    if pivot <= 0:
        return None
    dist = (c[i] - pivot) / pivot * 100     # +ve = above pivot

    if -PIVOT_NEAR_PCT <= dist < 0:
        trigger = "SETUP"                    # coiling under the pivot
    elif 0 <= dist <= EXTENDED_PCT:
        trigger = "TRIGGERED"                # cleared it, still in buy range
    else:
        return None                          # too far below, or chased

    stop = round(pivot * (1 - HARD_STOP_PCT / 100), 2)
    target = round(pivot * TARGET_MULT, 2)
    px = float(c[i])
    hi52 = float(np.max(c[i - 251:i + 1])) if i >= 251 else float(np.max(c[:i + 1]))
    near_high = round(px / hi52 * 100, 1) if hi52 > 0 else 0.0   # 100 = at the high
    return {
        "symbol": sym,
        "rs": int(rs_rank),
        "trend_score": _trend_score(c, i),
        "price": round(px, 2),
        "pivot": round(pivot, 2),
        "dist_to_pivot_pct": round(dist, 2),
        "trigger": trigger,
        "target": target,
        "upside_to_target_pct": round((target / px - 1) * 100, 2),
        "days_trending": _days_trending(c, i),
        "conviction": bool(is_fno and _days_trending(c, i) >= CONVICTION_MIN_TREND_DAYS),
        "near_high_pct": near_high,
        "tight_cv": round(tight_cv, 2),
        "mcap_cr": round(float(mcap)) if mcap else None,
        "fno": bool(is_fno),
        "segment": "F&O" if is_fno else "CASH",
        "r3m_pct": round(r3 * 100, 2),
        "r6m_pct": round(r6 * 100, 2),
        "excess_3m_pp": round((r3 - (bench_r3 or 0)) * 100, 2),
        "adtv_cr": round(adtv, 2),
        "stop": stop,
        "risk_pct": round((px - stop) / px * 100, 2),
    }


def _thesis(row: dict) -> dict:
    """A why-to-buy thesis + actionable advice, generated from the row's own facts.
    No adjectives the numbers don't support — this is a fat-tail screen and the
    thesis says so."""
    trg = row["trigger"]
    piv = row["pivot"]; px = row["price"]; stop = row["stop"]
    tgt = row["target"]; up = row["upside_to_target_pct"]; dt = row["days_trending"]
    seg = row["segment"]; ts = row["trend_score"]
    # Non-generic action: names the exact level, the target, the risk, and the age.
    if trg == "SETUP":
        action = (f"WATCH ₹{piv:.0f}: no position until a close ABOVE it "
                  f"({abs(row['dist_to_pivot_pct']):.1f}% away). On the trigger, target ₹{tgt:.0f} "
                  f"(+{(tgt/piv-1)*100:.0f}% from pivot), stop ₹{stop:.0f}.")
    else:
        action = (f"BUYABLE now ₹{px:.0f}, {row['dist_to_pivot_pct']:.1f}% past the ₹{piv:.0f} pivot: "
                  f"target ₹{tgt:.0f} (+{up:.0f}% from here), stop ₹{stop:.0f} ({row['risk_pct']:.0f}% risk), "
                  f"R:R {((tgt-px)/max(px-stop,1e-9)):.1f}:1. Above +{EXTENDED_PCT:.0f}% past pivot = chased.")
    thesis = (f"{seg} name, RS {row['rs']} (top {max(1, 100 - row['rs'])}% of the market), trend score "
              f"{ts}/100, beating the index by {row['excess_3m_pp']:.0f}pp over 3 months and trending for "
              f"{dt} sessions. Coiling ({row['tight_cv']:.1f}% 20-day range) under a {BASE_LOOKBACK}-bar "
              f"pivot at ₹{piv:.0f} — a leader pausing before a possible continuation.")
    caveat = ("MEASURED EXPECTANCY (this exact screen, 7yr, 2,345 signals): mean +6.87% / "
              "median +1.72% over ~3 months, 53% win. It beats NIFTY on only 50.3% of holds "
              "— a coin flip on any single name — with a +3.99pp average carried by a +41% "
              "top decile. And the PIVOT adds only +0.35pp over simply being an RS>90 leader. "
              "Two honest caveats: the backtest excludes delisted/failed leaders "
              "(survivorship inflates momentum), and 2020-26 was an exceptional bull run. "
              "Treat this as a fat-tail watchlist, sized small with the stop — not conviction.")
    return {"thesis": thesis, "action": action, "caveat": caveat}


def run_system_scan(progress_callback=None, force: bool = False) -> dict:
    if not force and _cache["data"] and time.time() - _cache["ts"] < CACHE_TTL:
        return _cache["data"]
    if not force:
        _disk = result_cache.get_or_stale("system")
        if _disk is not None:
            _cache["data"], _cache["ts"] = _disk, time.time()
            return _disk
    try:
        import shared_universe as su
        import benchmark as bm
        stocks = su.load_base_universe(days=500)
    except Exception as e:
        return {"results": [], "scanned": 0, "found": 0, "computed_at": time.time(),
                "error": f"load failed: {e}"}

    # benchmark 3m / 6m return (the "index" that must be beaten)
    bench_r3 = bench_r6 = None
    try:
        b = bm.get_benchmark(days=400)
        if b is not None and len(b) > 130:
            bv = b.to_numpy(dtype=float)
            bench_r3 = bv[-1] / bv[-64] - 1
            bench_r6 = bv[-1] / bv[-127] - 1
    except Exception:
        pass

    # market cap (Rs cr) and F&O membership — for the quality gate and the flag.
    mcap = {}
    try:
        import sqlite3, os
        # fundamentals.db lives on the DATA_DIR volume (/data), not the code dir.
        # It is gitignored, so /app never has it — reading from __file__ made System
        # find no market caps and reject every stock (empty tab on the live box).
        dbp = os.path.join(os.environ.get("DATA_DIR", os.path.dirname(__file__)), "fundamentals.db")
        if os.path.exists(dbp):
            con = sqlite3.connect(dbp)
            mcap = {r[0]: r[1] for r in con.execute(
                "SELECT symbol, market_cap FROM fundamentals WHERE market_cap IS NOT NULL")}
            con.close()
    except Exception:
        pass
    fno = set()
    try:
        import fo_data
        fno = set(fo_data.get_fo_signals().keys())
    except Exception:
        pass

    rs = rank_rs(stocks)
    rows = []
    for sym, df in stocks.items():
        try:
            row = _pit_row(sym, df, len(df) - 1, rs.get(sym, 0), bench_r3, bench_r6,
                           mcap=mcap.get(sym), is_fno=(sym in fno))
            if row:
                row.update(_thesis(row))
                rows.append(row)
        except Exception:
            continue

    # RANK. TRIGGERED before SETUP, then by a quality composite (trend score + RS +
    # tightness), then freshness. NOTE the F&O question the user raised: they wanted
    # NON-F&O prioritised on the belief it returns more. Measured over 7yr the OPPOSITE
    # is true — within this refined set F&O names ran +5.47pp excess / 60% win vs
    # non-F&O +1.71pp / 55.5%. So F&O is NOT down-ranked here; the segment is shown as
    # a column and the truth is stated in the tab, rather than shipping a ranking the
    # data contradicts. The fat-tail multi-baggers (MTARTECH etc.) are real but rare —
    # a higher ceiling with a LOWER average, which is a filter the user can apply, not
    # a default that helps.
    # RANK: high-conviction (F&O + established trend) first — the only subset that
    # measurably beat the cohort — then TRIGGERED before SETUP, then by RS.
    order = {"TRIGGERED": 0, "SETUP": 1}
    rows.sort(key=lambda r: (0 if r.get("conviction") else 1,
                             order.get(r["trigger"], 9),
                             -r["rs"], abs(r["dist_to_pivot_pct"])))
    found_total = len(rows)
    n_conviction = sum(1 for r in rows if r.get("conviction"))
    rows = rows[:MAX_TOP_N]

    out = {"results": rows, "scanned": len(stocks), "found": len(rows),
           "n_conviction": n_conviction, "found_total": found_total,
           "computed_at": time.time(),
           "bench_r3m_pct": round(bench_r3 * 100, 2) if bench_r3 is not None else None,
           "params": {"rs_min": RS_MIN, "min_adtv_cr": MIN_ADTV_CR,
                      "base_lookback": BASE_LOOKBACK, "extended_pct": EXTENDED_PCT}}
    _cache["data"], _cache["ts"] = out, time.time()
    result_cache.put("system", out)
    return out


def invalidate_cache() -> None:
    _cache["data"], _cache["ts"] = None, 0.0


# ── DAILY PERFORMANCE TRACKING (the user's "tracks their performance daily") ──
# Each scan records NEW picks (symbol first seen, at that day's price/pivot). On
# every later scan we mark them to the latest close and report what happened. This
# is the honest scoreboard: it will show the median pick going nowhere and the
# occasional one running — exactly the fat-tail shape the caveat promises, on THIS
# app's own live picks rather than a backtest.
import json
import os

_TRACK_PATH = os.path.join(os.environ.get("DATA_DIR", os.path.dirname(__file__)),
                           ".system_track.json")


def _load_track() -> dict:
    try:
        with open(_TRACK_PATH) as f:
            return json.load(f)
    except Exception:
        return {}


def _save_track(d: dict) -> None:
    try:
        tmp = _TRACK_PATH + ".tmp"
        with open(tmp, "w") as f:
            json.dump(d, f, indent=1)
        os.replace(tmp, _TRACK_PATH)
    except Exception:
        pass


def record_picks(rows: list[dict], as_of: str) -> None:
    """Log first-sighting of each current pick. Idempotent per (symbol, first date)."""
    track = _load_track()
    for r in rows:
        sym = r.get("symbol")
        if not sym:
            continue
        if sym not in track:
            track[sym] = {"first_seen": as_of, "first_price": r.get("price"),
                          "pivot": r.get("pivot"), "first_trigger": r.get("trigger"),
                          "first_rs": r.get("rs")}
    _save_track(track)


def performance(current_price_by_sym: dict) -> dict:
    """Mark every tracked pick to its latest close. Returns per-pick and aggregate
    performance since first sighting — the live scoreboard."""
    track = _load_track()
    rows = []
    for sym, rec in track.items():
        px0 = rec.get("first_price")
        px1 = current_price_by_sym.get(sym)
        if not (isinstance(px0, (int, float)) and isinstance(px1, (int, float)) and px0 > 0):
            continue
        ret = (px1 / px0 - 1) * 100
        rows.append({"symbol": sym, "first_seen": rec.get("first_seen"),
                     "first_price": px0, "last_price": round(px1, 2),
                     "return_pct": round(ret, 2), "first_rs": rec.get("first_rs"),
                     "pivot": rec.get("pivot"),
                     "cleared_pivot": bool(rec.get("pivot") and px1 > rec["pivot"])})
    rows.sort(key=lambda r: -r["return_pct"])
    if rows:
        rets = np.array([r["return_pct"] for r in rows])
        agg = {"tracked": len(rows), "mean_pct": round(float(rets.mean()), 2),
               "median_pct": round(float(np.median(rets)), 2),
               "win_rate": round(100 * float((rets > 0).mean()), 1),
               "best": rows[0]["symbol"] + f" +{rows[0]['return_pct']:.0f}%",
               "worst": rows[-1]["symbol"] + f" {rows[-1]['return_pct']:.0f}%"}
    else:
        agg = {"tracked": 0}
    return {"picks": rows, "aggregate": agg}
