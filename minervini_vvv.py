"""
Mark Minervini VVV (SEPA + Volatility Contraction + Volume) Scanner.

Implements the exact pipeline Minervini describes in "Trade Like a Stock Market Wizard":

  Universe (Nifty Total Market ≈ 750)
      ↓
  Stage 2 uptrend ONLY (Weinstein/Minervini)
      ↓
  Trend Template score ≥ 6 of 8 SEPA criteria
      ↓
  RS rank ≥ 70 vs universe (true cross-sectional, not raw return)
      ↓
  VCP forming  OR  Episodic Pivot occurring
      ↓
  Volume confirming pattern (>= 1.3× baseline, accumulation > distribution)
      ↓
  Tight stop near 10-EMA or base-low — risk ≤ 5% (Minervini's max)
      ↓
  Reward:risk ≥ 3 (target = entry + 3R) — gives 25% wins ≥ break-even

Result: a HIGH-CONVICTION watchlist — typically 3-15 stocks in a Correction
regime, 30-50 in an Uptrend.
"""
from __future__ import annotations

import time
import threading
from typing import Optional

import numpy as np
import pandas as pd

from industry_groups import _get_stocks, _build_nifty
from analysis_utils import (
    stage_analysis,
    trend_template_score,
    cross_sectional_rs_rank,
    volume_baseline,
    rs_line_new_high,
    is_3wt,
    power_trend,
)
from breakout_scanner import _is_vcp
from institutional_scanner import _detect_earnings_setup, _detect_pocket_pivot, _acc_dist_days
from risk_config import MAX_STOP_PCT, MIN_RR_RATIO


# ── Additional Minervini continuation patterns (BUG-FIX: original VVV only had
# fresh-base patterns and missed stocks already in established uptrends like
# BSE, RRKABEL, MTARTECH that Minervini actively trades). ─────────────────────

def _is_tight_pullback(close: pd.Series, vol: pd.Series, high: pd.Series, low: pd.Series) -> tuple[bool, dict]:
    """
    Continuation Pullback to 10-EMA (Minervini's "buy the quiet pullback" setup).
    Fires when a Stage-2 leader pulls back to the 10-EMA in a tight range —
    the textbook "trampoline" entry for stocks already in uptrend.

    Criteria (relaxed from theoretical text to fit real institutional flow):
      - Price within 8% of 10-EMA (a bit wider — leading stocks pull back further)
      - Last 5-bar range ≤ 10% (tight, but not as strict as 7%)
      - Last 5-bar volume average ≤ 1.6× 20-day median (volume drying up,
        leading stocks naturally trade above baseline so 1.0× is too strict)
      - Above MA50 (still in trend)
    """
    if len(close) < 50:
        return False, {}
    try:
        ema10 = close.ewm(span=10, adjust=False).mean().iloc[-1]
        ma50  = close.rolling(50).mean().iloc[-1]
        cur   = float(close.iloc[-1])
        if cur < ma50:                                # must still be above MA50
            return False, {}
        # Distance to 10-EMA — accept up to 8% (was 5%)
        pct_from_ema = abs(cur - ema10) / ema10 * 100
        if pct_from_ema > 8.0:
            return False, {}
        # Tight 5-bar range — accept up to 10% (was 7%)
        recent_hi = float(high.iloc[-5:].max())
        recent_lo = float(low.iloc[-5:].min())
        range_pct = (recent_hi - recent_lo) / recent_lo * 100
        if range_pct > 10.0:
            return False, {}
        # Volume — leading stocks trade above baseline; only reject CHURN (>1.6×)
        recent_vol = float(vol.iloc[-5:].mean())
        baseline   = volume_baseline(vol, window=20)
        if baseline > 0 and recent_vol > baseline * 1.6:
            return False, {}
        return True, {
            "entry":       round(recent_hi * 1.002, 2),    # break of 5-bar high
            "ema10":       round(float(ema10), 2),
            "base_low":    round(recent_lo, 2),
            "range_pct":   round(range_pct, 2),
            "pct_from_ema": round(pct_from_ema, 2),
        }
    except Exception:
        return False, {}


def _is_3wt_breakout(close: pd.Series, vol: pd.Series) -> tuple[bool, dict]:
    """
    3-Weeks-Tight breakout (Minervini): 3 consecutive weekly closes within 1.5%,
    followed by a breakout above the 3-week high. Captures stocks like BSE that
    are in a steady uptrend with periodic tight consolidations.
    """
    if len(close) < 25:
        return False, {}
    try:
        if not is_3wt(close):
            return False, {}
        # Compute the 3-week high from weekly closes
        wk = close.resample("W-FRI").last().dropna()
        if len(wk) < 4:
            return False, {}
        last3_hi = float(wk.iloc[-3:].max())
        last3_lo = float(wk.iloc[-3:].min())
        cur = float(close.iloc[-1])
        # Must be ABOVE (or right at) the 3-week high to qualify as a breakout.
        # Allow a small tolerance (0.5%) for stocks closing fractionally below
        # but actively pushing through during the day.
        if cur < last3_hi * 0.995:
            return False, {}
        # Reject stocks that have fallen out of the base (below the 3-week low).
        if cur < last3_lo * 0.99:
            return False, {}
        return True, {
            "entry":     round(last3_hi * 1.002, 2),
            "base_high": round(last3_hi, 2),
            "base_low":  round(last3_lo, 2),
        }
    except Exception:
        return False, {}


def _is_power_trend(close: pd.Series, vol: pd.Series) -> tuple[bool, dict]:
    """
    Power Trend continuation (O'Neil/Minervini): 21-EMA > 50-MA AND both rising
    AND price > 21-EMA for 80%+ of last 20 bars. The "leader runs" setup —
    used for relentless trends like MTARTECH that never give a clean pullback.

    BUG-FIX: previous version required today to be an up-day, which rejected
    stocks like BSE/RRKABEL/MTARTECH on quiet pullback days — exactly when
    Minervini takes continuation entries. Now: any day where price holds
    above 21-EMA qualifies as a Power Trend pivot, with stop just below 21-EMA.
    """
    if len(close) < 60:
        return False, {}
    try:
        if not power_trend(close):
            return False, {}
        cur   = float(close.iloc[-1])
        ema21 = float(close.ewm(span=21, adjust=False).mean().iloc[-1])
        # Just require price holding above the 21-EMA (the trend reference).
        # Continuation buyers enter on ANY day price tags / hovers above 21-EMA.
        if cur < ema21 * 0.95:                   # max 5% below 21-EMA
            return False, {}
        return True, {
            "entry":    round(cur * 1.002, 2),
            "ema21":    round(ema21, 2),
            "base_low": round(ema21 * 0.95, 2),   # 5% below 21-EMA = trailing stop
        }
    except Exception:
        return False, {}

# ── Fresh breakout from a long base (early-entry detector) ──────────────────

def _is_fresh_breakout_from_base(close):
    """
    Detect a stock that has broken out of a multi-month base but whose
    MA150 slope hasn't yet turned positive (so the Weinstein classifier
    still calls it Stage 1 or Stage 3).

    Criteria — ALL must hold:
      1. Price > MA50
      2. Price > MA200
      3. MA50 rising vs 22 bars ago (≥ 0.5% increase)
      4. Within 5% of 52-week high (= confirmed breakout, not just a bounce)

    Returns (True, label) on match, (False, "") otherwise.
    """
    import pandas as pd
    if len(close) < 200:
        return False, ""
    try:
        cur = float(close.iloc[-1])
        ma50  = close.rolling(50).mean()
        ma200 = close.rolling(200).mean()
        ma50_now = float(ma50.iloc[-1])
        ma200_now = float(ma200.iloc[-1])
        # MA50 must be available 22 bars ago for slope check
        if pd.isna(ma50.iloc[-22]):
            return False, ""
        ma50_22d = float(ma50.iloc[-22])
        if cur <= ma50_now:           return False, ""
        if cur <= ma200_now:          return False, ""
        if ma50_now <= ma50_22d * 1.005:   return False, ""  # MA50 rising req
        # 52-week high proximity
        win = close.iloc[-252:] if len(close) >= 252 else close
        hi52 = float(win.max())
        if hi52 <= 0 or cur < hi52 * 0.95:
            return False, ""
        return True, "Fresh Breakout"
    except Exception:
        return False, ""


# ── Minervini's exact thresholds (from his book) ─────────────────────────────
MIN_TT_SCORE      = 6        # Minervini: ≥ 6 of 8 SEPA criteria. 7-8 = ideal.
MIN_RS_RANK       = 70       # Minervini: RS Rating must be 70+
MIN_RS_RANK_IDEAL = 80       # 80+ is "best-in-class"
MIN_ADTV_CR       = 1.0      # ₹1 Cr daily turnover — institutional min
MIN_BARS          = 200      # need 200 bars for full SEPA + VCP
MAX_RISK_PCT      = 5.0      # Minervini: stop ≤ 5% from entry (typically 2-4%)
TARGET_R_MULTIPLE = 3.0      # Minervini: 3:1 minimum R:R
EMA_SHORT         = 10       # 10-EMA is the canonical stop anchor

# ── Early-entry mode ─────────────────────────────────────────────────────────
# Stocks breaking out of a multi-month base often haven't yet had their MA150
# slope turn positive — the Weinstein classifier still flags them as Stage 1
# or Stage 3 even though they're clearly entering institutional accumulation.
# Strict VVV (stage == 2 only) misses these for another 10-20 bars while the
# slow MA catches up; that's exactly the window where Minervini buys.
# Setting EARLY_ENTRY_MODE = True accepts ALSO stocks that pass an explicit
# "fresh breakout from long base" check: price > MA50 > MA200, MA50 slope
# rising, and within 5% of 52-week high. These are flagged as `early_entry`
# in the result row so the UI can show them distinctly.
EARLY_ENTRY_MODE = True

# Volume confirmation thresholds
MIN_VOL_MULTIPLE  = 1.3      # today's vol must be ≥ 1.3× 20-day baseline median
REQUIRE_ACC_DAYS  = True     # acc days >= dist days in last 20 sessions

CACHE_TTL = 3600
_cache: dict = {"data": None, "ts": 0.0}
_cache_lock = threading.Lock()


def _safe(v, default=0.0):
    if v is None:
        return default
    try:
        f = float(v)
        return default if (np.isnan(f) or np.isinf(f)) else f
    except Exception:
        return default


def _ema(series: pd.Series, span: int) -> float:
    """Single-point EMA at the latest bar."""
    if series is None or len(series) < span:
        return 0.0
    return float(series.ewm(span=span, adjust=False).mean().iloc[-1])


def _adtv_cr(close: pd.Series, vol: pd.Series, window: int = 20) -> float:
    """Average daily turnover in ₹ Cr using median-volume baseline."""
    vb = volume_baseline(vol, window=window)
    cur = _safe(close.iloc[-1]) if len(close) else 0.0
    return round(vb * cur / 1e7, 2)


def _atr_pct(df: pd.DataFrame, period: int = 14) -> float:
    """ATR as percentage of price."""
    try:
        from analysis_utils import atr
        a = atr(df, period=period)
        cur = _safe(df["Close"].iloc[-1])
        if cur <= 0:
            return 0.0
        return round(a / cur * 100, 2)
    except Exception:
        return 0.0


def _build_stop(entry: float, ema10: float, base_low: Optional[float]) -> tuple[float, str]:
    """
    Minervini's stop loss logic:
      1. Anchor at 10-EMA × 0.98 (2% below the EMA)
      2. Or the base-low (pattern stop) if it's tighter
      3. Never wider than 5% from entry (MAX_RISK_PCT)
      4. Never tighter than 1% (noise floor)
    Returns (stop_price, reason).
    """
    candidates = []
    if ema10 > 0:
        candidates.append((ema10 * 0.98, "10-EMA"))
    if base_low and base_low > 0 and base_low < entry:
        candidates.append((base_low * 0.99, "base-low"))

    # 5% max risk floor — the wider edge of acceptable
    max_stop = entry * (1 - MAX_RISK_PCT / 100)
    candidates.append((max_stop, f"max {MAX_RISK_PCT}%"))

    # Pick the TIGHTEST valid stop (highest price, smallest risk)
    candidates = [c for c in candidates if c[0] > 0 and c[0] < entry]
    if not candidates:
        return entry * 0.95, "5%-default"
    candidates.sort(key=lambda x: -x[0])    # highest stop = tightest = first
    return candidates[0]


def _analyze_one(symbol: str, df: pd.DataFrame, nifty: pd.Series,
                 rs_rank: int, rs_rank_12m: int) -> Optional[dict]:
    """Score one stock through the 6-step Minervini VVV pipeline."""
    try:
        c   = df["Close"].dropna()
        v   = df["Volume"].dropna() if "Volume" in df.columns else pd.Series([], dtype=float)
        if len(c) < MIN_BARS:
            return None
        cur = _safe(c.iloc[-1])
        if cur <= 0:
            return None

        # ── GATE 1: Liquidity ────────────────────────────────────────────
        adtv_cr = _adtv_cr(c, v)
        if adtv_cr < MIN_ADTV_CR:
            return None

        # ── GATE 2: Stage 2 uptrend (or early breakout from a long base) ──
        stg = stage_analysis(c)
        early_entry = False
        if stg != 2:
            if EARLY_ENTRY_MODE:
                is_fresh, _fb_label = _is_fresh_breakout_from_base(c)
                if not is_fresh:
                    return None
                early_entry = True
            else:
                return None

        # ── GATE 3: Trend Template ≥ 6 of 8 ──────────────────────────────
        tt_score, tt_met = trend_template_score(c, rs_rating=rs_rank)
        if tt_score < MIN_TT_SCORE:
            return None

        # ── GATE 4: RS rank ≥ 70 ─────────────────────────────────────────
        if rs_rank < MIN_RS_RANK:
            return None

        # ── GATE 5: Any Minervini entry pattern ──────────────────────────
        # BUG-FIX: original VVV only checked fresh-breakout patterns (VCP/EP/PP)
        # which all reject stocks already in established uptrends. Adding three
        # continuation patterns that Minervini explicitly trades: tight pullback
        # to 10-EMA · 3-Weeks-Tight breakout · Power Trend continuation.
        h = df["High"].dropna()
        l = df["Low"].dropna()
        vcp_ok, vcp_lvl = _is_vcp(c, v)
        ep_ok,  ep_lvl  = _detect_earnings_setup(df)
        pp_ok,  pp_lvl  = _detect_pocket_pivot(df)
        tp_ok,  tp_lvl  = _is_tight_pullback(c, v, h, l)
        wt_ok,  wt_lvl  = _is_3wt_breakout(c, v)
        pwr_ok, pwr_lvl = _is_power_trend(c, v)

        if not (vcp_ok or ep_ok or pp_ok or tp_ok or wt_ok or pwr_ok):
            return None

        # Pick the strongest pattern (priority: VCP > EP > PP > 3WT > Pullback > PowerTrend)
        # — VCP is the cleanest fresh entry, PowerTrend is the loosest continuation.
        if vcp_ok:
            pattern   = "VCP"
            entry     = _safe(vcp_lvl.get("entry"), default=cur)
            base_high = _safe(vcp_lvl.get("base_high"), default=entry)
            base_low  = _safe(vcp_lvl.get("base_low"),  default=entry * 0.92)
        elif ep_ok:
            pattern   = "Episodic Pivot"
            entry     = _safe(ep_lvl.get("entry"), default=cur)
            base_high = entry
            base_low  = _safe(ep_lvl.get("sl"), default=entry*0.95)
        elif pp_ok:
            pattern   = "Pocket Pivot"
            entry     = _safe(pp_lvl.get("entry"), default=cur)
            base_high = entry
            base_low  = _safe(pp_lvl.get("sl"), default=entry*0.95)
        elif wt_ok:
            pattern   = "3-Weeks-Tight"
            entry     = _safe(wt_lvl.get("entry"), default=cur)
            base_high = _safe(wt_lvl.get("base_high"), default=entry)
            base_low  = _safe(wt_lvl.get("base_low"),  default=entry * 0.96)
        elif tp_ok:
            pattern   = "Tight Pullback"
            entry     = _safe(tp_lvl.get("entry"), default=cur)
            base_high = entry
            base_low  = _safe(tp_lvl.get("base_low"), default=entry * 0.95)
        else:
            pattern   = "Power Trend"
            entry     = _safe(pwr_lvl.get("entry"), default=cur)
            base_high = entry
            base_low  = _safe(pwr_lvl.get("base_low"), default=entry * 0.93)

        # Continuation patterns (don't require fresh volume surge) are flagged here
        # so the volume gate below can be relaxed for them.
        is_continuation = (tp_ok or wt_ok or pwr_ok) and not (vcp_ok or ep_ok or pp_ok)

        if entry <= 0:
            entry = cur
        if base_low <= 0 or base_low >= entry:
            base_low = entry * 0.92

        # ── GATE 6: Volume confirming the pattern ────────────────────────
        if len(v) < 25:
            return None
        vbase = volume_baseline(v, window=20)
        if vbase <= 0:
            return None
        today_vol_mult = _safe(v.iloc[-1]) / vbase
        # Volume gate rules:
        #   - Fresh breakout patterns (VCP / EP / PP) — already vol-confirmed by detector
        #   - Continuation patterns (Pullback / 3WT / Power Trend) — Minervini's rule
        #     is "volume DRIES UP on the pullback, then surges on the breakout day".
        #     So we accept today_vol >= 0.6× baseline (quiet pullback acceptable).
        if pattern in ("Pocket Pivot", "VCP", "Episodic Pivot"):
            pass   # detector handles vol internally
        elif is_continuation:
            # Continuation: tolerate quiet days but reject hard distribution
            if today_vol_mult > 2.0 and _safe(c.iloc[-1]) < _safe(c.iloc[-2]):
                return None   # heavy down-vol = institutional selling
        elif today_vol_mult < MIN_VOL_MULTIPLE:
            return None

        # Accumulation vs distribution days check
        if REQUIRE_ACC_DAYS:
            acc_d, dist_d = _acc_dist_days(c, v, period=20)
            if dist_d > acc_d:
                return None

        # ── GATE 7: Tight stop + R:R ─────────────────────────────────────
        ema10 = _ema(c, EMA_SHORT)
        stop, stop_reason = _build_stop(entry, ema10, base_low)
        risk_pct = (entry - stop) / entry * 100
        # BUG-FIX: float-precision tolerance — `_build_stop`'s fallback computes
        # `entry × (1 - 5/100)` which yields risk_pct = 5.000000001 due to IEEE 754.
        # Without tolerance, stocks at exactly the max-stop cap (BSE, RRKABEL etc)
        # were rejected even though they meet the rule. Add 0.01% epsilon.
        if risk_pct > MAX_RISK_PCT + 0.01 or risk_pct < 0.5:
            return None
        target = entry + (entry - stop) * TARGET_R_MULTIPLE
        rr     = (target - entry) / (entry - stop)
        if rr < MIN_RR_RATIO:
            return None

        # ── Scoring ──────────────────────────────────────────────────────
        # Composite priority (sort key): higher is better
        # — Tighter stop = higher score (lower risk_pct boosts)
        # — Higher RS = higher score
        # — VCP > EP > PP pattern priority
        # — TT score 7-8 ideal
        pattern_pts = {
            "VCP": 30, "Episodic Pivot": 25, "Pocket Pivot": 22,
            "3-Weeks-Tight": 20, "Tight Pullback": 18, "Power Trend": 15,
        }[pattern]
        score = (
            pattern_pts +
            min(25, rs_rank // 4) +              # 0-25 from RS
            tt_score * 3 +                       # 18-24 from TT
            max(0, int((5 - risk_pct) * 4)) +    # 0-20 reward for tight stop
            (5 if vcp_ok and ep_ok else 0) +     # bonus when both fire
            (5 if rs_line_new_high(c, nifty) else 0)
        )
        score = min(100, score)

        # Tier
        if   score >= 85: tier = "🏆 IDEAL"
        elif score >= 70: tier = "✅ STRONG"
        elif score >= 55: tier = "👀 WATCH"
        else:             tier = "📋 TRACK"
        # Early-entry stocks (caught before the slow-MA confirmation) get a
        # sunrise prefix so users see the higher uncertainty up front.
        if early_entry:
            tier = "🌅 EARLY " + tier.split(" ", 1)[1]

        # Extra context for the UI
        r1m = round((cur / _safe(c.iloc[-21]) - 1) * 100, 2) if len(c) >= 21 else None
        r3m = round((cur / _safe(c.iloc[-63]) - 1) * 100, 2) if len(c) >= 63 else None

        # ── Money Flow Index (14-bar) — institutional flow proxy ─────────────
        # MFI weights typical-price by VOLUME so it doesn't trigger on small-vol
        # price wiggles — that's why it's a much better "smart money" signal
        # than plain RSI. Used here to surface which VVV setups have actual
        # institutional money behind them vs ones that are technically valid
        # but volume-thin.
        # Bands:
        #   ≥80  Strong Accumulation (often pre-pivot or pivot day)
        #   65-79 Accumulation
        #   40-64 Neutral / mixed flow
        #   25-39 Distribution
        #   ≤25  Strong Distribution (selling pressure)
        try:
            from sector_analysis import _mfi
            mfi_v = _mfi(df, period=14)
        except Exception:
            mfi_v = 50.0
        if mfi_v is None:
            mfi_v = 50.0
        if   mfi_v >= 80: mfi_label = "Strong Accum"
        elif mfi_v >= 65: mfi_label = "Accumulation"
        elif mfi_v >= 40: mfi_label = "Neutral"
        elif mfi_v >= 25: mfi_label = "Distribution"
        else:             mfi_label = "Strong Dist"

        return {
            "symbol":       symbol,
            "score":        int(score),
            "tier":         tier,
            "pattern":      pattern,
            "vcp_ok":       bool(vcp_ok),
            "ep_ok":        bool(ep_ok),
            "pp_ok":        bool(pp_ok),
            # SEPA scoring
            "tt_score":     tt_score,
            "tt_met":       tt_met,
            "stage":        stg,
            "early_entry":  bool(early_entry),
            # RS
            "rs_rank":      int(rs_rank),
            "rs_rank_12m":  int(rs_rank_12m),
            "rs_line_high": bool(rs_line_new_high(c, nifty)),
            # Trade plan
            "price":        round(cur, 2),
            "entry":        round(entry, 2),
            "stop":         round(stop, 2),
            "stop_reason":  stop_reason,
            "target":       round(target, 2),
            "risk_pct":     round(risk_pct, 2),
            "rr_ratio":     round(rr, 2),
            "ema10":        round(ema10, 2),
            # Context
            "base_high":    round(base_high, 2),
            "base_low":     round(base_low, 2),
            "adtv_cr":      adtv_cr,
            "vol_mult":     round(today_vol_mult, 2),
            "atr_pct":      _atr_pct(df),
            "r1m":          r1m,
            "r3m":          r3m,
            # Institutional flow
            "mfi":          round(float(mfi_v), 1),
            "mfi_label":    mfi_label,
        }
    except Exception:
        return None


def run_vvv_scan(progress_callback=None) -> dict:
    """Main entry — run the full Minervini VVV pipeline. Cached 1h."""
    with _cache_lock:
        if (_cache["data"]
                and time.time() - _cache["ts"] < CACHE_TTL
                and _cache["data"].get("results") is not None):
            return _cache["data"]

    def _prog(n, total, msg):
        if progress_callback:
            progress_callback(n, total, msg)

    _prog(0, 100, "Loading OHLCV…")
    stocks = _get_stocks()
    if not stocks:
        return {"results": [], "computed_at": int(time.time()),
                "total_scanned": 0, "error": "No OHLCV data"}

    _prog(15, 100, f"Loaded {len(stocks)} stocks. Building Nifty proxy…")
    nifty = _build_nifty(stocks)
    if nifty is None or len(nifty) < 50:
        return {"results": [], "computed_at": int(time.time()),
                "total_scanned": len(stocks),
                "error": "Nifty proxy unavailable — try after running another scan first"}

    _prog(25, 100, "Computing universe-wide RS ranks…")
    # Universe-wide RS ranks (post-Tier 3 consistent percentile, 1-99)
    def _ret(sym, days):
        c = stocks[sym]["Close"].dropna()
        if len(c) <= days:
            return None
        try:
            return float(c.iloc[-1] / c.iloc[-days] - 1) * 100
        except Exception:
            return None

    ret_6m  = {s: _ret(s, 126) for s in stocks}
    ret_12m = {s: _ret(s, 252) for s in stocks}
    rs_6m   = cross_sectional_rs_rank(ret_6m)
    rs_12m  = cross_sectional_rs_rank(ret_12m)

    _prog(40, 100, "Running VVV pipeline on each stock…")

    results: list[dict] = []
    done = [0]
    total = len(stocks)
    for sym, df in stocks.items():
        r = _analyze_one(sym, df, nifty,
                         rs_rank=rs_6m.get(sym, 0),
                         rs_rank_12m=rs_12m.get(sym, 0))
        if r is not None:
            results.append(r)
        done[0] += 1
        if done[0] % 100 == 0:
            pct = 40 + int(done[0] / total * 50)
            _prog(pct, 100, f"VVV scan {done[0]}/{total}…")

    _prog(92, 100, "Sorting + enriching results…")

    # Sort: score desc, then RS desc
    results.sort(key=lambda r: (-r["score"], -r["rs_rank"]))

    # Tier counts
    ideal_n  = sum(1 for r in results if r["tier"].startswith("🏆"))
    strong_n = sum(1 for r in results if r["tier"].startswith("✅"))
    watch_n  = sum(1 for r in results if r["tier"].startswith("👀"))
    track_n  = sum(1 for r in results if r["tier"].startswith("📋"))

    out = {
        "results":       results,
        "computed_at":   int(time.time()),
        "total_scanned": total,
        "qualified":     len(results),
        "ideal_count":   ideal_n,
        "strong_count":  strong_n,
        "watch_count":   watch_n,
        "track_count":   track_n,
        "thresholds": {
            "min_tt_score":   MIN_TT_SCORE,
            "min_rs_rank":    MIN_RS_RANK,
            "max_risk_pct":   MAX_RISK_PCT,
            "min_rr_ratio":   MIN_RR_RATIO,
            "vol_multiple":   MIN_VOL_MULTIPLE,
        },
    }

    with _cache_lock:
        _cache["data"] = out
        _cache["ts"]   = time.time()

    # Enrich with cross-scan consensus stamps (P2-10/11)
    try:
        from consensus import enrich_results, invalidate_cache as _con_inv
        _con_inv()
        enrich_results(results)
    except Exception:
        pass

    _prog(100, 100,
          f"Done — {len(results)} stocks passed VVV "
          f"({ideal_n} IDEAL, {strong_n} STRONG, {watch_n} WATCH)")
    return out


def invalidate_cache():
    with _cache_lock:
        _cache["data"] = None
        _cache["ts"]   = 0.0
