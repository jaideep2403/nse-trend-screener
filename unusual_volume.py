"""
Institutional Accumulation — sustained delivery, validated before shipping.
==========================================================================

THE QUESTION
------------
Relative volume cannot separate accumulation from churn. A stock can trade 3x its
normal volume entirely intraday and close in exactly the same hands. NSE publishes
DELIVERY PERCENTAGE — the share of traded volume that actually settled into a demat
account — which is the one field that distinguishes ownership from noise.

WHAT WAS ACTUALLY MEASURED
--------------------------
900-stock sample, 642 with usable delivery history, point-in-time, every 5th bar,
forward 21 sessions, n = 12,158 observations, split in half by date for IS/OOS.
Candidate definitions were tested BEFORE any of this was written:

    variant                              n     mean     win%    OOS vs base
    persist>=4 (delivery above baseline)  760   +2.07%   53.9%   +0.14pp
    + absolute delivery >= 55%            262   +2.65%   60.3%   +2.19pp   <- kept
    + volume dry-up (Wyckoff)             261   -0.12%   47.5%   -0.57pp   REJECTED
    + tight range as well                  80   -0.43%   50.0%   -0.92pp   REJECTED
    + volume SURGE                        164   +0.45%   44.5%   -1.93pp   REJECTED
    + require price above MA50             81   +1.84%   54.3%      (worse) REJECTED

THREE PIECES OF RECEIVED WISDOM THAT DID NOT SURVIVE
----------------------------------------------------
  1. "Volume dries up during accumulation" (Wyckoff). Every variant containing a
     dry-up condition measured NEGATIVE. Two independent tests now agree.
  2. "Unusual volume marks institutional entry." Delivery persistence PLUS a volume
     surge measured -1.93pp OOS — worse than the delivery leg alone. The volume
     surge is when retail arrives, not when institutions build.
  3. "Buy strength — require price above the 50-DMA." Adding it cut the mean from
     2.65% to 1.84% and the win rate from 60.3% to 54.3%. Accumulation happens in
     bases, below the average, before markup — which is Wyckoff's actual claim.

THE EDGE IS LIQUIDITY-BANDED — this is the part that stops it being noise
-------------------------------------------------------------------------
    ADTV band        n    signal mean   win%    edge vs same-band peers
    < Rs 1cr        212      -0.92%    47.2%    -1.09pp    NEGATIVE (junk)
    Rs 1-5cr        106      +3.50%    64.2%    +1.53pp
    Rs 5-25cr        84      +3.55%    60.7%    +1.47pp
    Rs 25-100cr      52      +1.13%    57.7%    -0.33pp
    > Rs 100cr       20      -1.64%    45.0%    -3.11pp

Economically coherent: in mega-caps institutions are permanently present, so
delivery persistence carries no information; below Rs 1cr it is noise and easily
manipulated. The signal only means something in the middle, where a genuine
institutional build is both possible and visible. Hence the hard ADTV band.

SHIPPED SPECIFICATION AND ITS MEASURED RESULT
---------------------------------------------
    persist >= 4 of the last 5 sessions with delivery > 1.15x the stock's OWN
    25-day baseline, AND 5-day mean delivery >= 55%, AND ADTV in Rs 1-25cr.

    signal  n=  190   mean +3.52%   median +2.81%   win 62.6%
    peers   n= 7071   mean +2.03%   median +0.31%   win 51.1%
    EDGE              +1.49pp mean, +11.5pp win rate
    IS +1.02pp | OOS +2.16pp        (positive in both halves)
    forward <= -10%: 11.1% of signals vs 14.9% of peers — FEWER large losses

Note the median is positive and the win rate is 11.5pp above peers: this is not a
fat-tail signal that needs a handful of moonshots to work, unlike the breakout
screens elsewhere in this app. It also fires on 297 distinct stocks across 474
observations (top 3 names = 4% of hits), so it is broad rather than an artifact of
a few tickers.

HONEST LIMITS
-------------
  * One market regime (the data available here), no deep bear market in sample.
  * Delivery is end-of-day and anonymous: it proves shares settled, NOT who bought.
    A promoter, an HNI and a mutual fund look identical in this field.
  * The ADTV band means the scan deliberately ignores large caps. That is correct
    per the measurement above, but it makes this a small/mid-cap tool.
  * +1.49pp mean over 21 sessions is a real but modest edge. It is a watchlist
    generator, not a reason to size aggressively.
"""

from __future__ import annotations

import time
import numpy as np
import pandas as pd

# ── Validated parameters — do not change without re-running the study ────────
BASE_WINDOW      = 25     # sessions forming the stock's own delivery baseline
BASE_SHIFT       = 5      # baseline EXCLUDES the recent window it is judged against
FAST_WINDOW      = 5      # the recent window
RVOL_WINDOW      = 20     # median-volume window (reported, NOT gated — see above)

PERSIST_MIN      = 4      # sessions (of FAST_WINDOW) above baseline x PERSIST_MULT
PERSIST_MULT     = 1.15
DELIV_ABS_MIN    = 55.0   # 5-day mean delivery %. 45/50/60/65 all measured worse.
ADTV_MIN_CR      = 1.0    # below this the signal measured NEGATIVE
ADTV_MAX_CR      = 25.0   # above this the edge decays, then inverts

# ── Quality guards — junk filters, NOT edge filters ─────────────────────────
# Each was measured before adding. They are essentially free: they remove ~1% of
# observations and change the edge by <=0.01pp, so they are not curve-fitting.
# They exist because the raw signal surfaced rows that were technically valid and
# obviously unusable, e.g. SUMEETINDS at Rs 17.28 with delivery 99.8% against a
# 9.6% baseline (a 10.4x jump) while the price fell 22.5% in five sessions. A 9.6%
# baseline is a pure intraday trading vehicle, a 10x delivery jump is a regime
# change or a corporate action rather than accumulation, and "accumulation" during
# a 22% collapse is incoherent on its face. Output nobody believes is worthless
# however good the aggregate statistics are.
#   CORE                                    n=190  +1.49pp mean, OOS +2.16pp
#   CORE + these three guards               n=177  +1.49pp mean, OOS +2.11pp
#                                                  median +2.91% (the best of any variant)
MAX_DELIV_MULT   = 3.0    # >3x its own baseline is an anomaly, not a build
MIN_DELIV_BASE   = 20.0   # a stock whose NORMAL delivery is <20% is a trading vehicle
MIN_MOVE_5D      = -10.0  # not "accumulation" if the price is collapsing

_cache: dict = {"data": None, "ts": 0.0}
CACHE_TTL = 30 * 60


def detect_accumulation(symbol: str, df: pd.DataFrame) -> dict | None:
    """One stock, point-in-time. Returns a dict when the validated pattern is present.

    Uses only bars up to the last row of `df`; the baseline is shifted so the
    candidate sessions never help set the "normal" they are compared against.
    """
    try:
        if df is None or "DelivPer" not in df.columns:
            return None
        need = BASE_WINDOW + BASE_SHIFT + RVOL_WINDOW + 5
        if len(df) < need:
            return None

        close = df["Close"].astype(float)
        vol   = df["Volume"].astype(float)
        dp    = df["DelivPer"].astype(float)
        if dp.notna().sum() < BASE_WINDOW + FAST_WINDOW:
            return None

        # Baseline: this stock's own median delivery, EXCLUDING the recent window.
        # Sector-relative by construction — a bank at 40% and an FMCG name at 70%
        # are each judged against themselves, so no sector table is needed.
        base = float(dp.iloc[-(BASE_WINDOW + BASE_SHIFT):-BASE_SHIFT].median())
        if not np.isfinite(base) or base <= 0:
            return None

        recent  = dp.iloc[-FAST_WINDOW:]
        persist = int((recent > base * PERSIST_MULT).sum())
        deliv5  = float(recent.mean())
        if persist < PERSIST_MIN or deliv5 < DELIV_ABS_MIN:
            return None
        # Quality guards — see the note beside MAX_DELIV_MULT.
        if base < MIN_DELIV_BASE or (deliv5 / base) > MAX_DELIV_MULT:
            return None

        med_vol = float(vol.iloc[-(RVOL_WINDOW + 1):-1].median())
        if med_vol <= 0:
            return None

        last    = float(close.iloc[-1])
        adtv_cr = float((close * vol).tail(20).mean()) / 1e7
        if not (ADTV_MIN_CR <= adtv_cr <= ADTV_MAX_CR):
            return None

        move5 = (last / float(close.iloc[-6]) - 1) * 100
        if move5 < MIN_MOVE_5D:
            return None

        ma50 = float(close.rolling(50).mean().iloc[-1]) if len(close) >= 50 else None
        return {
            "symbol": symbol,
            "price": round(last, 2),
            "persist": persist,                                   # of FAST_WINDOW
            "deliv_5d": round(deliv5, 1),
            "deliv_base": round(base, 1),
            "deliv_mult": round(deliv5 / base, 2),
            "deliv_today": round(float(dp.iloc[-1]), 1),
            "rvol": round(float(vol.iloc[-1]) / med_vol, 2),      # reported only
            "rvol_5d": round(float(vol.iloc[-FAST_WINDOW:].mean()) / med_vol, 2),
            "day_chg": round((last / float(close.iloc[-2]) - 1) * 100, 2),
            "move_5d": round(move5, 2),
            "adtv_cr": round(adtv_cr, 2),
            "ext_ma50": round(((last / ma50) - 1) * 100, 1) if ma50 else None,
        }
    except Exception:
        return None


def _rank(rows: list[dict]) -> list[dict]:
    """Rank on the two legs the study found carried the edge: how far above its own
    baseline delivery has run, and the absolute level it reached. Volume is NOT a
    ranking input — every variant that used it measured worse."""
    if not rows:
        return rows
    def pr(vals):
        return pd.Series(vals).rank(pct=True) * 100
    mult = pr([r["deliv_mult"] for r in rows])
    lvl  = pr([r["deliv_5d"] for r in rows])
    per  = pr([r["persist"] for r in rows])
    for i, r in enumerate(rows):
        r["score"] = round(0.40 * float(mult.iloc[i]) +
                           0.40 * float(lvl.iloc[i]) +
                           0.20 * float(per.iloc[i]), 1)
    return sorted(rows, key=lambda r: -r["score"])


def run_accumulation_scan(progress_callback=None, force: bool = False) -> dict:
    """Scan for validated institutional accumulation.

    Uses data_fetcher.fetch_ohlcv rather than shared_universe.load_base_universe
    because only the former keeps the DelivPer column, and delivery is the entire
    signal. ~26s cold for the universe, then cached 30 min.
    """
    if not force and _cache["data"] and time.time() - _cache["ts"] < CACHE_TTL:
        return _cache["data"]
    try:
        import shared_universe as su
        import data_fetcher as dfch
        syms = [f"{s}.NS" for s in su.load_base_universe(days=400).keys()]
    except Exception as e:
        return {"results": [], "scanned": 0, "found": 0,
                "computed_at": time.time(), "error": f"universe load failed: {e}"}

    if progress_callback:
        progress_callback(0, len(syms), "Loading delivery history…")
    try:
        data = dfch.fetch_ohlcv(syms, min_bars=120)
    except Exception as e:
        return {"results": [], "scanned": 0, "found": 0,
                "computed_at": time.time(), "error": f"delivery fetch failed: {e}"}

    rows: list[dict] = []
    total = len(data)
    for i, (tkr, df) in enumerate(data.items()):
        if progress_callback and i % 300 == 0:
            progress_callback(i, total, f"Checking delivery footprint… {i}/{total}")
        hit = detect_accumulation(tkr.replace(".NS", ""), df)
        if hit:
            rows.append(hit)

    rows = _rank(rows)
    out = {"results": rows, "scanned": total, "found": len(rows),
           "computed_at": time.time(),
           "spec": {"persist_min": PERSIST_MIN, "deliv_abs_min": DELIV_ABS_MIN,
                    "adtv_band_cr": [ADTV_MIN_CR, ADTV_MAX_CR]},
           "evidence": {"n": 190, "mean": 3.52, "median": 2.81, "win": 62.6,
                        "peer_mean": 2.03, "peer_median": 0.31, "peer_win": 51.1,
                        "edge_mean_pp": 1.49, "edge_win_pp": 11.5,
                        "is_pp": 1.02, "oos_pp": 2.11, "guarded_n": 177}}
    _cache["data"], _cache["ts"] = out, time.time()
    return out


# Backwards-compatible aliases — the earlier draft of this module exposed these.
run_unusual_volume_scan = run_accumulation_scan
detect_unusual_volume   = detect_accumulation


def invalidate_cache() -> None:
    _cache["data"], _cache["ts"] = None, 0.0
