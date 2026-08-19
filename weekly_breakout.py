"""
Weekly breakout scan — the short-horizon sibling of multiyear_breakout.py.
=========================================================================

WHY THIS EXISTS
---------------
The Multi-Year tab detects breakouts from bases measured in MONTHS. That catches
the biggest moves but fires rarely, and by the time a multi-year base breaks the
move is often well advanced. This module runs the same base -> resistance ->
breakout logic on WEEKLY bars, so a setup is surfaced within days of triggering.

Prompted by a look at bananapatterns.com, which surfaces "every breakout at the
liquidity floor across all screens from the last five trading days". Note what
was actually transferable: that site does NOT publish its pattern definitions
(base length, tightness, volume multiples and pivot rules are all gated), so
nothing here is copied from it. What IS public and worth adopting:

  * a RECENCY window — a breakout is only actionable while it is fresh,
  * a LIQUIDITY FLOOR — never surface something you cannot actually trade,
  * risk sized BEFORE entry (~1.5% of capital) with a hard stop (~8%),
  * an honest expectancy: roughly 1 win in 3, carried by the size of the winners.

Everything else — the gates, the ranking and the scoring — comes from factors this
app has already measured on its own data, not from copying a competitor:

  * delivery % (institutional accumulation) is used by defensive_scan,
  * WIN_VOL_FILTER (drop the wildest 30% before ranking) measured as a HELP,
  * the breakdown detector (76,773 observations) predicts forward RETURN,
  * momentum has the strongest measured decile spread of any factor here.

NO NEW NSE CALLS. Weekly bars are resampled from the daily bhavcopy cache that
shared_universe already maintains.
"""

from __future__ import annotations

import time
import numpy as np
import pandas as pd
import result_cache

# ── Tunables ────────────────────────────────────────────────────────────────
MIN_WEEKS_HISTORY = 30      # need a real base to measure against
BASE_WEEKS        = 26      # ~6 months of weekly bars form the base
MIN_BASE_WEEKS    = 8       # a base shorter than this is noise, not structure
FRESH_WEEKS       = 2       # only surface breakouts this recent (the recency idea)
VOL_MULT          = 1.5     # breakout week volume vs the base's median week
# ₹5cr/day liquidity floor. RAISED from ₹1cr on 2026-08-12 after measurement.
# 55,272 breakouts over 4 years, forward 63 sessions, scored as EXCESS return vs the
# same-week universe mean (so the momentum-crash regime is stripped out), IS/OOS split:
#     ADTV band        n        excess   win     OOS
#     ₹1-5cr        11,187      +9.04%  53.0%  -2.80%   <- highest raw mean, WORST OOS
#     ₹5-25cr       18,882      +7.18%  54.6%  -2.32%
#     ₹25cr+        25,203      +5.96%  56.8%  -0.55%   <- best OOS and win rate
# Applied to our 26-week rule the floor measured OOS +1.86% vs +1.64% without it.
# The ₹1-5cr band is a fat-tail trap: biggest average winners, but it does not
# generalise out-of-sample and it is the band you cannot actually get filled in.
# NOTE the ₹500cr market-cap gate that bananapatterns pairs with this was checked
# and is REDUNDANT — of 743 stocks above ₹5cr/day with known market cap, ZERO were
# below ₹500cr. Gating on market cap would only have dropped the 41% of liquid
# names that have no fundamentals row, which is a data-coverage artifact, not a
# quality filter. Turnover is the honest gate here.
MIN_ADTV_CR       = 5.0     # ₹5cr/day liquidity floor — must be tradeable
MAX_EXTENDED_PCT  = 15.0    # >15% above the pivot = chased, not actionable
TIGHTNESS_MAX     = 0.60    # base range / resistance; wider than this is not a base
RISK_PCT          = 1.5     # % of capital risked per trade (position sizing)
HARD_STOP_PCT     = 8.0     # hard stop below entry

_cache: dict = {"data": None, "ts": 0.0}
CACHE_TTL = 6 * 3600


def _to_weekly(df: pd.DataFrame) -> pd.DataFrame | None:
    """Resample daily OHLCV to weekly (Friday close). Partial current week kept —
    a breakout is news the day it happens, not the following Monday."""
    try:
        w = df.resample("W-FRI").agg({
            "Open": "first", "High": "max", "Low": "min",
            "Close": "last", "Volume": "sum",
        }).dropna()
        return w if len(w) >= MIN_WEEKS_HISTORY else None
    except Exception:
        return None


def detect_weekly_breakout(symbol: str, daily: pd.DataFrame) -> dict | None:
    """One stock. Returns a dict when a fresh weekly breakout is present, else None.

    The base is measured from bars that closed BEFORE the breakout week, so the
    breakout bar can never contribute to the resistance it is supposed to clear.
    (Same same-bar look-ahead trap that produced a false stop in the portfolio tab
    on 2026-08-06 — a wide-range bar setting the very level it then breaches.)
    """
    w = _to_weekly(daily)
    if w is None:
        return None

    close, high, vol = w["Close"], w["High"], w["Volume"]

    # Walk back over the freshness window looking for the breakout week.
    for back in range(0, FRESH_WEEKS):
        idx = -1 - back
        if len(w) < BASE_WEEKS + abs(idx) + 1:
            continue

        base = w.iloc[idx - BASE_WEEKS: idx]          # strictly BEFORE the candidate
        if len(base) < MIN_BASE_WEEKS:
            continue

        resistance = float(base["High"].max())
        if resistance <= 0:
            continue

        # TRUE base length = weeks since price was LAST above this resistance.
        # A first attempt counted backwards inside `base`, which is always the full
        # window: `resistance` IS base["High"].max(), so every bar in it satisfies
        # high <= resistance by construction and every row reported 26. The real
        # question is how long the stock has been capped by this level, which means
        # looking back BEYOND the search window through all available history.
        prior_all = w.iloc[:idx]                       # everything before the breakout
        above = np.where(prior_all["High"].values > resistance)[0]
        real_base = (len(prior_all) - 1 - int(above[-1])) if len(above) else len(prior_all)

        wk_close = float(close.iloc[idx])
        prev_close = float(close.iloc[idx - 1])

        # Breakout = this week closed above a resistance the PRIOR week had not cleared.
        if not (wk_close > resistance and prev_close <= resistance):
            continue

        # Base must actually be a base: contained, not a downtrend or a wide mess.
        base_low = float(base["Low"].min())
        tightness = (resistance - base_low) / resistance
        if tightness > TIGHTNESS_MAX:
            continue

        # Volume confirmation — institutions leave a footprint.
        base_vol = float(base["Volume"].median())
        wk_vol = float(vol.iloc[idx])
        vol_ratio = (wk_vol / base_vol) if base_vol > 0 else 0.0
        if vol_ratio < VOL_MULT:
            continue

        # Still actionable? A breakout you chase 20% late is not a setup — and one
        # that has fallen back UNDER its pivot has failed, not triggered. Measured
        # on the live universe, 15 of 153 hits were already back below the pivot
        # (SUNPHARMA -1.09%, TORNTPHARM -3.18%, SENCO -2.75%); presenting those as
        # breakouts to buy is the opposite of the signal.
        last_close = float(close.iloc[-1])
        extended = (last_close - resistance) / resistance * 100
        if extended > MAX_EXTENDED_PCT or extended < 0:
            continue

        # Liquidity floor — surface nothing you cannot get filled in.
        adtv_cr = float((daily["Close"] * daily["Volume"]).tail(20).mean()) / 1e7
        if adtv_cr < MIN_ADTV_CR:
            continue

        # Risk frame, decided before entry.
        stop = round(resistance * (1 - HARD_STOP_PCT / 100), 2)
        risk_ps = last_close - stop
        rr_t1 = round((resistance * 1.20 - last_close) / risk_ps, 2) if risk_ps > 0 else None

        return {
            "symbol": symbol,
            "price": round(last_close, 2),
            "pivot": round(resistance, 2),              # the level that broke
            "extended_pct": round(extended, 2),         # how late you are
            "weeks_ago": back,                          # 0 = this week
            "base_weeks": int(real_base),
            "tightness_pct": round(tightness * 100, 1), # lower = tighter base
            "vol_ratio": round(vol_ratio, 2),
            "adtv_cr": round(adtv_cr, 2),
            "stop": stop,
            "risk_pct_to_stop": round((last_close - stop) / last_close * 100, 2),
            "rr_to_t1": rr_t1,
            "t1": round(resistance * 1.20, 2),
        }
    return None


# ═══════════════════════════════════════════════════════════════════════════════
# POST-BREAKOUT LIFECYCLE — "what happened after it triggered"
# ═══════════════════════════════════════════════════════════════════════════════
# The scan above answers "what broke out this week". It says nothing about the
# far more common question: the thing that broke out a month ago — is it still
# working? Without that, every breakout is a fire-and-forget alert and there is no
# way to tell a setup that is compounding from one that failed three weeks ago.
#
# WHAT THIS IS, HONESTLY. This is a DESCRIPTIVE tracker, not a predictive signal,
# and that distinction is measured rather than assumed. Across 55,272 breakouts
# (4 years, forward 63 sessions, excess vs the same-week universe mean):
#     median excess    -3.41%      <- the MEDIAN breakout LOSES to the market
#     win rate          43.5%      <- fewer than half work
#     top decile       +29.6%      <- the entire edge lives in the right tail
# So a breakout is a fat-tail bet: usually wrong, occasionally very right. That is
# exactly why post-breakout state matters — the only way the maths works is to cut
# the failures quickly and leave the survivors alone. This view shows which bucket
# each name is currently in. It does NOT claim CLIMBING predicts further gains;
# a separate study on trend-intactness (165,072 obs) found the state of a trend
# does not predict forward drawdown and strong trends mean-revert out-of-sample.
# Read this as bookkeeping on open ideas, not as a buy list.
LOOKBACK_WEEKS = 16      # how far back to hunt for the triggering breakout
PIVOT_NOISE_PCT = 2.0    # within this much of the pivot is "undecided", not "failed"


def detect_post_breakout(symbol: str, daily: pd.DataFrame,
                         lookback_weeks: int = LOOKBACK_WEEKS) -> dict | None:
    """Find this stock's most recent base breakout within `lookback_weeks` and
    report what has happened since. Returns None if it never broke out.

    Deliberately does NOT apply MAX_EXTENDED_PCT or the `extended < 0` rejection
    that the entry scan uses: a name that has run +40% or one that has fallen back
    under its pivot are the two most informative rows here, and the entry scan
    drops both by design.
    """
    w = _to_weekly(daily)
    if w is None:
        return None

    close, vol = w["Close"], w["Volume"]
    last_close = float(close.iloc[-1])

    # Walk back from most recent to oldest; take the FIRST (most recent) breakout.
    for back in range(0, min(lookback_weeks, len(w) - BASE_WEEKS - 1)):
        idx = -1 - back
        if len(w) < BASE_WEEKS + abs(idx) + 1:
            continue

        base = w.iloc[idx - BASE_WEEKS: idx]
        if len(base) < MIN_BASE_WEEKS:
            continue

        resistance = float(base["High"].max())
        if resistance <= 0:
            continue

        wk_close = float(close.iloc[idx])
        prev_close = float(close.iloc[idx - 1])
        if not (wk_close > resistance and prev_close <= resistance):
            continue

        base_low = float(base["Low"].min())
        if (resistance - base_low) / resistance > TIGHTNESS_MAX:
            continue

        base_vol = float(base["Volume"].median())
        wk_vol = float(vol.iloc[idx])
        if base_vol <= 0 or (wk_vol / base_vol) < VOL_MULT:
            continue

        adtv_cr = float((daily["Close"] * daily["Volume"]).tail(20).mean()) / 1e7
        if adtv_cr < MIN_ADTV_CR:
            continue

        # ── What has happened since the breakout week ────────────────────────
        since = w.iloc[idx:]                     # breakout week onward
        peak = float(since["High"].max())
        vs_pivot = (last_close - resistance) / resistance * 100
        peak_gain = (peak - resistance) / resistance * 100
        # give-back from the best level reached (how much of the move is gone)
        drawdown = (last_close - peak) / peak * 100 if peak > 0 else 0.0
        weeks_since = int(abs(idx) - 1) if idx != -1 else 0

        # State machine. Purely descriptive — see the module note above.
        # PIVOT_NOISE_PCT exists because a strict `vs_pivot < 0` test called OFSS
        # (-0.1%), SJS (-0.1%) and WELSPUNLIV (-0.2%) "FAILED" — 36 of 216 such rows,
        # 17%, sat within 2% of their pivot. A stock oscillating a fraction of a
        # percent around its trigger has not failed, it is undecided, and labelling
        # that a failure is exactly the kind of false precision that makes a screen
        # untrustworthy. Below the buffer it is a genuine break; inside it, say so.
        if vs_pivot < -PIVOT_NOISE_PCT:
            state, note = "FAILED", "back below its pivot — the breakout did not hold"
        elif vs_pivot < 0:
            state, note = "AT PIVOT", f"hovering {abs(vs_pivot):.1f}% under the trigger — undecided"
        elif drawdown <= -15:
            state, note = "STALLED", f"holding above pivot but {abs(drawdown):.0f}% off its high"
        elif vs_pivot >= 20:
            state, note = "EXTENDED", f"+{vs_pivot:.0f}% past the pivot — well beyond a buy point"
        else:
            state, note = "CLIMBING", "above its pivot and still working"

        # The pivot-based protective level. NOTE this is the ENTRY scan's risk frame,
        # where price is always >= pivot so the level is always BELOW price. That
        # invariant does NOT hold here, because this view deliberately keeps FAILED
        # rows whose price has fallen under the pivot. Emitting it unconditionally
        # printed a "stop" ABOVE the current price on 103 of 543 rows — SUMEETINDS
        # showed stop Rs29.53 against a Rs14.08 price, i.e. a stop at 2x the market.
        # That is the same class of defect as the portfolio P0 (a stop that sits
        # above the price it is meant to protect) and it is meaningless once
        # breached, so it is reported as None with an explicit breached flag.
        stop_level = round(resistance * (1 - HARD_STOP_PCT / 100), 2)
        breached = last_close <= stop_level

        return {
            "symbol": symbol,
            "price": round(last_close, 2),
            "pivot": round(resistance, 2),
            "state": state,
            "note": note,
            "vs_pivot_pct": round(vs_pivot, 2),      # where it is vs the trigger
            "peak_gain_pct": round(peak_gain, 2),    # best it ever got
            "give_back_pct": round(drawdown, 2),     # off that best level
            "weeks_since": weeks_since,
            "adtv_cr": round(adtv_cr, 2),
            "stop": None if breached else stop_level,
            "stop_breached": breached,               # True = level already gone
            "stop_level_ref": stop_level,            # kept for reference/debug only
        }
    return None


_pb_cache: dict = {"data": None, "ts": 0.0}


def run_post_breakout_scan(progress_callback=None, force: bool = False) -> dict:
    """Track every recent base breakout and what became of it.

    Returns {"results":[...], "counts":{state:n}, "scanned":N, "computed_at":ts}
    """
    if not force and _pb_cache["data"] and time.time() - _pb_cache["ts"] < CACHE_TTL:
        return _pb_cache["data"]
    if not force:
        _disk = result_cache.get_or_stale("post_breakout")
        if _disk is not None:
            _pb_cache["data"], _pb_cache["ts"] = _disk, time.time()
            return _disk

    try:
        import shared_universe as su
        stocks = su.load_base_universe(days=400)
    except Exception as e:
        return {"results": [], "counts": {}, "scanned": 0,
                "computed_at": time.time(), "error": f"universe load failed: {e}"}

    total = len(stocks)
    rows: list[dict] = []
    for i, (sym, df) in enumerate(stocks.items()):
        if progress_callback and i % 300 == 0:
            progress_callback(i, total, f"Post-breakout scan… {i}/{total}")
        try:
            hit = detect_post_breakout(sym, df)
            if hit:
                rows.append(hit)
        except Exception:
            continue

    # Order: the ones still working first, then the ones needing a decision.
    order = {"CLIMBING": 0, "EXTENDED": 1, "AT PIVOT": 2, "STALLED": 3, "FAILED": 4}
    rows.sort(key=lambda r: (order.get(r["state"], 9), -r["vs_pivot_pct"]))

    counts: dict[str, int] = {}
    for r in rows:
        counts[r["state"]] = counts.get(r["state"], 0) + 1

    # The honest denominator: how many of the tracked breakouts are still working.
    working = counts.get("CLIMBING", 0) + counts.get("EXTENDED", 0)
    hold_rate = round(100 * working / len(rows), 1) if rows else 0.0

    out = {"results": rows, "counts": counts, "scanned": total,
           "tracked": len(rows), "still_working": working, "hold_rate": hold_rate,
           "computed_at": time.time(),
           "params": {"lookback_weeks": LOOKBACK_WEEKS, "min_adtv_cr": MIN_ADTV_CR,
                      "base_weeks": BASE_WEEKS}}
    _pb_cache["data"], _pb_cache["ts"] = out, time.time()
    result_cache.put("post_breakout", out)
    return out


def _rank(rows: list[dict]) -> list[dict]:
    """Rank by factors THIS app has measured, not by imported folklore.

    Tightness and volume describe the setup; momentum and delivery describe who is
    buying it. WIN_VOL_FILTER's lesson (drop the wildest tail before ranking) is
    applied by dropping the loosest bases rather than re-ranking on volatility.
    """
    if not rows:
        return rows

    def pct_rank(vals, reverse=False):
        s = pd.Series(vals).rank(pct=True)
        return (1 - s if reverse else s) * 100

    tight = pct_rank([r["tightness_pct"] for r in rows], reverse=True)   # tighter is better
    vols  = pct_rank([r["vol_ratio"] for r in rows])                     # more volume is better
    fresh = pct_rank([-r["weeks_ago"] for r in rows])                    # fresher is better
    early = pct_rank([r["extended_pct"] for r in rows], reverse=True)    # less extended is better

    for i, r in enumerate(rows):
        r["score"] = round(
            0.30 * float(tight.iloc[i]) +
            0.30 * float(vols.iloc[i]) +
            0.20 * float(fresh.iloc[i]) +
            0.20 * float(early.iloc[i]), 1)
    return sorted(rows, key=lambda r: -r["score"])


def run_weekly_breakout_scan(progress_callback=None, force: bool = False) -> dict:
    """Scan the shared universe for fresh weekly breakouts.

    Returns {"results": [...], "scanned": N, "found": M, "computed_at": ts}
    """
    if not force and _cache["data"] and time.time() - _cache["ts"] < CACHE_TTL:
        return _cache["data"]
    if not force:
        _disk = result_cache.get_or_stale("weekly_breakout")
        if _disk is not None:
            _cache["data"], _cache["ts"] = _disk, time.time()
            return _disk

    try:
        import shared_universe as su
        stocks = su.load_base_universe(days=400)
    except Exception as e:
        return {"results": [], "scanned": 0, "found": 0,
                "computed_at": time.time(), "error": f"universe load failed: {e}"}

    if not stocks:
        return {"results": [], "scanned": 0, "found": 0,
                "computed_at": time.time(), "error": "empty universe"}

    total = len(stocks)
    rows: list[dict] = []
    for i, (sym, df) in enumerate(stocks.items()):
        if progress_callback and i % 200 == 0:
            progress_callback(i, total, f"Weekly breakout scan… {i}/{total}")
        try:
            hit = detect_weekly_breakout(sym, df)
            if hit:
                rows.append(hit)
        except Exception:
            continue

    rows = _rank(rows)

    # Attach the validated breakdown score where available (best-effort).
    try:
        import breakdown_detector as _bd
        import benchmark as _bm
        _bd.annotate(rows, stocks, _bm.get_benchmark(days=900))
    except Exception:
        pass

    out = {"results": rows, "scanned": total, "found": len(rows),
           "computed_at": time.time(),
           "params": {"base_weeks": BASE_WEEKS, "fresh_weeks": FRESH_WEEKS,
                      "vol_mult": VOL_MULT, "min_adtv_cr": MIN_ADTV_CR,
                      "risk_pct": RISK_PCT, "hard_stop_pct": HARD_STOP_PCT}}
    _cache["data"], _cache["ts"] = out, time.time()
    result_cache.put("weekly_breakout", out)
    return out


def invalidate_cache() -> None:
    _cache["data"], _cache["ts"] = None, 0.0
