"""
Breakout Scanner — scans all 2500 NSE EQ stocks for breakouts
Timeframes : D (20-day high), W (13-week high), M (6-month high), Y (52-week / ATH)
Patterns   : ATH, VCP, Box, Rectangular
Data source: NSE bhavcopy (already cached, zero Yahoo calls)
"""
import time
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from data_fetcher import _weekdays_back, _download_one_day
from nse_stocks import is_etf
from analysis_utils import (
    trend_template_score, stage_analysis, stage_label,
    is_high_tight_flag, detect_candle_signals,
)
from industry_groups import INDUSTRY_GROUPS

# ── Symbol → Group lookup (built once at import) ──────────────────────────────
_SYM_TO_GROUP: dict[str, str] = {}
for _grp, _syms in INDUSTRY_GROUPS.items():
    for _s in _syms:
        _SYM_TO_GROUP[_s] = _grp

MIN_BARS     = 60      # minimum trading days required
MIN_ADTV_CR  = 0.5    # Minimal liquidity guard only — universe filtered by Nifty500 membership
# Curated 750 PLUS any HIGHLY-liquid off-index stock at/above this turnover
# (keeps actively-traded off-index names e.g. AEROFLEX, while keeping the
# universe ~base-size for speed — ₹2Cr nearly doubled it to ~1450 → 2x slow).
UNIVERSE_OFFINDEX_ADTV_CR = 50.0
SCAN_WORKERS = 8
_cache = {"data": None, "ts": 0}
CACHE_TTL = 3600       # 1 hour


# ── Split / Bonus backward-adjustment (BUG-001) ───────────────────────────────

def _adjust_for_splits(df):
    """Delegate to canonical analysis_utils.adjust_for_splits."""
    from analysis_utils import adjust_for_splits
    return adjust_for_splits(df)


# ── Load all NSE EQ stock OHLCV from cached bhavcopy files ────────────────────

# Full-EQ ratings basis (MarketSmith "vs all stocks") — populated during load.
_UNIV_RS: dict[str, float] = {}
_UNIV_AD: dict[str, float] = {}


def _load_all_stocks(progress_callback=None) -> dict[str, pd.DataFrame]:
    """Return {symbol: OHLCV_df} for all NSE EQ stocks from cached bhavcopy days."""
    dates = _weekdays_back(400)
    total = len(dates)
    frames = []

    for i, dt in enumerate(dates):
        df = _download_one_day(dt)
        if df is not None:
            frames.append(df)
        if progress_callback and i % 40 == 0:
            progress_callback(i, total, f"Loading historical data… {i}/{total} days")

    if not frames:
        return {}

    # Universe: Nifty Total Market 750 (Nifty50 ∪ Next50 ∪ Nifty500 ∪ Smallcap250 ∪ Microcap250 ∪ TotalMarket)
    try:
        from nse_stocks import get_universe_symbols
        _universe = set(get_universe_symbols())
    except Exception:
        _universe = set()
    combined = pd.concat(frames, ignore_index=True).sort_values("Date")

    stocks = {}
    univ_rs: dict[str, float] = {}   # MarketSmith RS metric for EVERY EQ stock
    univ_ad: dict[str, float] = {}   # 13-week Acc/Dis score for every EQ stock
    for sym, grp in combined.groupby("Symbol"):
        if is_etf(sym): continue
        g = grp.set_index("Date")[["Open", "High", "Low", "Close", "Volume"]]
        g = g[~g.index.duplicated(keep="last")].sort_index()
        g = _adjust_for_splits(g)
        if len(g) < MIN_BARS:
            continue
        # Ratings basis: MarketSmith ranks RS / Acc-Dis vs ALL publicly traded
        # companies, not a curated 750. Compute for every EQ name (cheap), so the
        # 1-99 RS Rating and A-E Demand grade are vs the whole market like theirs.
        m = _ms_rs_metric(g["Close"].dropna())
        if m is not None:
            univ_rs[sym] = m
        univ_ad[sym] = _buyer_demand(g)
        # Scan the curated 750 (all current coverage) PLUS any LIQUID off-index
        # stock (≥₹2Cr ADTV) — recent IPOs not yet in the index (e.g. AEROFLEX)
        # were never scanned for breakouts even after they broke out. Now they are.
        if sym not in _universe:
            cv   = g[["Close", "Volume"]].dropna()
            look = min(20, len(cv))
            adtv = float((cv["Close"].iloc[-look:] * cv["Volume"].iloc[-look:]).mean()) / 1e7 if look else 0.0
            if adtv < UNIVERSE_OFFINDEX_ADTV_CR:
                continue
        stocks[sym] = g

    global _UNIV_RS, _UNIV_AD
    _UNIV_RS, _UNIV_AD = univ_rs, univ_ad
    return stocks


# ── Utility ───────────────────────────────────────────────────────────────────

def _vol_ratio(vol: pd.Series, period: int = 20) -> float:
    # BUG-039 FIX: enforce min_periods so partial windows don't yield misleading ratios.
    if len(vol) < period + 1:
        return 1.0
    avg_s = vol.iloc[-(period + 1):-1].rolling(period, min_periods=20).mean()
    avg = float(avg_s.iloc[-1]) if len(avg_s) and not pd.isna(avg_s.iloc[-1]) else 0.0
    return round(float(vol.iloc[-1]) / avg, 2) if avg > 0 else 1.0


def _atr(df: pd.DataFrame, period: int = 14) -> float:
    """Wilder's ATR via canonical analysis_utils.atr (single source of truth)."""
    from analysis_utils import atr as _canonical_atr
    return _canonical_atr(df, period=period)


def _week_pct_range(close: pd.Series, weeks_ago_start: int, weeks_ago_end: int) -> float:
    """Price range (%) over a window in the past.
    weeks_ago_start > weeks_ago_end (e.g. start=10, end=7 → bars[-50:-35]).
    """
    s = len(close) - weeks_ago_start * 5   # older end
    e = len(close) - weeks_ago_end * 5     # more-recent end
    s, e = max(0, s), max(0, e)
    if s >= e:          # nothing to slice
        return 0.0
    slc = close.iloc[s:e]
    lo, hi = float(slc.min()), float(slc.max())
    return (hi - lo) / lo * 100 if lo > 0 else 0.0


# ── Timeframe detection ───────────────────────────────────────────────────────

def _detect_timeframes(close: pd.Series, vol: pd.Series) -> list[str]:
    """Return timeframe labels where price is at a new high (D / W / M / Y)."""
    tf  = []
    cur = float(close.iloc[-1])

    # D — new 20-day closing high
    if len(close) >= 21:
        if cur >= float(close.iloc[-22:-1].max()):
            tf.append("D")

    # W — new 13-week high (weekly resampled)
    if len(close) >= 65:
        try:
            wk = close.resample("W-FRI").last().dropna()
            if len(wk) >= 14 and float(wk.iloc[-1]) >= float(wk.iloc[-14:-1].max()):
                tf.append("W")
        except Exception:
            pass

    # M — new 6-month high (monthly resampled)
    if len(close) >= 126:
        try:
            mo = close.resample("ME").last().dropna()
            if len(mo) >= 7 and float(mo.iloc[-1]) >= float(mo.iloc[-7:-1].max()):
                tf.append("M")
        except Exception:
            pass

    # Y — new 52-week high or near ATH
    # BUG-008 FIX: exclude current bar from the lookback so a "new high" really
    # means the bar broke ABOVE prior 251 sessions, not just tied with itself.
    if len(close) >= 252:
        if cur >= float(close.iloc[-252:-1].max()) * 0.995:
            tf.append("Y")
    elif len(close) >= 2 and cur >= float(close.iloc[:-1].max()) * 0.995:
        tf.append("Y")   # ATH even with < 1yr data

    return tf


# ── Pattern detection — returns (matched, levels_dict) ────────────────────────

def _is_ath(close: pd.Series) -> bool:
    # BUG-008 FIX: ATH means current bar > all PRIOR bars — exclude self from max().
    if len(close) < 2:
        return False
    return float(close.iloc[-1]) >= float(close.iloc[:-1].max()) * 0.995


def _is_vcp(close: pd.Series, vol: pd.Series):
    """
    Volatility Contraction Pattern (Minervini):
    Returns (True, {"entry": ..., "base_high": ..., "base_low": ...}) or (False, {})
    """
    if len(close) < 80:
        return False, {}
    cur   = float(close.iloc[-1])
    lo_yr = float(close.iloc[-252:].min()) if len(close) >= 252 else float(close.min())
    if cur < lo_yr * 1.15:
        return False, {}

    r1 = _week_pct_range(close, 10, 7)
    r2 = _week_pct_range(close, 6,  4)
    r3 = _week_pct_range(close, 3,  0)

    if not (r1 > 0 and r2 > 0 and r3 > 0):
        return False, {}
    if not (r2 < r1 * 0.85 and r3 < r2 * 0.85):
        return False, {}
    if r3 > 12:
        return False, {}

    # Final base pivot high (15-day) — this is the breakout entry
    base_high = float(close.iloc[-15:].max())
    if cur < base_high * 0.95:
        return False, {}

    # Volume declining
    if len(vol) >= 20:
        if float(vol.iloc[-10:].mean()) > float(vol.iloc[-20:-10].mean()) * 1.2:
            return False, {}

    # Base low = lowest close in last 15 days
    base_low = float(close.iloc[-15:].min())

    levels = {
        "entry":     round(base_high * 1.003, 2),   # just above pivot high
        "base_high": round(base_high, 2),
        "base_low":  round(base_low, 2),
    }
    return True, levels


def _is_box(close: pd.Series, high: pd.Series, low: pd.Series):
    """
    Flat Base / Box Pattern.
    Returns (True, {"entry": ..., "base_high": ..., "base_low": ...}) or (False, {})
    """
    if len(close) < 25:
        return False, {}

    window   = min(45, max(15, len(close) - 1))
    box_hi   = float(high.iloc[-window:-1].max())
    box_lo   = float(low.iloc[-window:-1].min())
    if box_lo <= 0:
        return False, {}

    pct_range = (box_hi - box_lo) / box_lo * 100
    if pct_range > 18:
        return False, {}

    cur = float(close.iloc[-1])
    if cur < box_hi * 0.97 or cur > box_hi * 1.08:
        return False, {}

    # Confirm consolidation (not trending)
    early = float(close.iloc[-window])
    mid   = float(close.iloc[-window // 2])
    if early > 0 and abs(mid - early) / early > 0.12:
        return False, {}

    levels = {
        "entry":     round(box_hi * 1.003, 2),
        "base_high": round(box_hi, 2),
        "base_low":  round(box_lo, 2),
    }
    return True, levels


def _is_rectangular(close: pd.Series, high: pd.Series, vol: pd.Series):
    """
    Rectangular Breakout.
    Returns (True, {"entry": ..., "base_high": ..., "base_low": ...}) or (False, {})
    """
    if len(close) < 53:
        return False, {}

    resistance = float(close.iloc[-53:-1].max())
    cur        = float(close.iloc[-1])

    if cur < resistance * 0.99 or cur > resistance * 1.06:
        return False, {}
    if _vol_ratio(vol, 20) < 1.3:
        return False, {}

    # Base low = lowest close in consolidation window
    base_low = float(close.iloc[-53:-1].min())

    levels = {
        "entry":     round(resistance * 1.003, 2),
        "base_high": round(resistance, 2),
        "base_low":  round(base_low, 2),
    }
    return True, levels


def _detect_patterns(close, high, low, vol):
    """Returns (patterns_list, combined_levels_dict)."""
    patterns = []
    levels   = {}

    if _is_ath(close):
        patterns.append("ATH")
        win = min(20, len(close) - 1)
        levels = {
            "entry":     round(float(close.iloc[-1]) * 1.001, 2),
            "base_high": round(float(close.iloc[-1]), 2),
            "base_low":  round(float(low.iloc[-win:].min()), 2),
        }

    vcp_ok, vcp_lvl = _is_vcp(close, vol)
    if vcp_ok:
        patterns.append("VCP")
        if not levels:
            levels = vcp_lvl

    box_ok, box_lvl = _is_box(close, high, low)
    if box_ok:
        patterns.append("Box")
        if not levels:
            levels = box_lvl

    if not patterns:
        rect_ok, rect_lvl = _is_rectangular(close, high, vol)
        if rect_ok:
            patterns.append("Rectangular")
            levels = rect_lvl

    # High Tight Flag (rare, but most powerful)
    htf_ok, htf_lvl = is_high_tight_flag(close, high)
    if htf_ok:
        patterns.append("HTF")
        if not levels:
            levels = {
                "entry":     round(htf_lvl["base_high"] * 1.003, 2),
                "base_high": htf_lvl["base_high"],
                "base_low":  htf_lvl["flag_lo"],
            }

    return patterns, levels


# ── Level computation ─────────────────────────────────────────────────────────

def _compute_levels(df: pd.DataFrame, entry: float, base_high: float, base_low: float,
                    ath: float) -> dict:
    """
    Compute SL and T1/T2/T3 given pattern levels.

    SL strategy:
      - ATR-based: entry − 2×ATR14
      - Base-low based: base_low × 0.985 (just under base low)
      - Take the HIGHER of the two (tighter stop = better risk control)
      - Hard cap: never more than 8% below entry

    Targets:
      T1 = measured move (base height projected above entry)
      T2 = 2× risk from entry  (2R)
      T3 = max(ATH, 3× risk from entry)  (3R or ATH, whichever higher)
    """
    atr14 = _atr(df)
    # ── Real stop = below the base structure (true invalidation), NOT clamped to
    # a fixed 8%. The old `max(sl, entry*0.92)` forced every volatile breakout to
    # exactly -8%, hiding each stock's real risk. Prefer the base-low structure
    # stop; fall back to a 2× ATR volatility stop; sanity-bound risk to 2-15%.
    sl_base = base_low * 0.985 if (base_low and base_low < entry) else None
    sl_atr  = entry - 2.0 * atr14 if atr14 > 0 else None
    # Take the TIGHTER (higher) of the structure vs volatility stop — better risk
    # control on a breakout entry — but NOT clamped to a fixed 8%.
    opts = [v for v in (sl_base, sl_atr) if v is not None and v < entry]
    sl = max(opts) if opts else entry * 0.92
    sl = max(sl, entry * 0.85)    # sanity: never risk more than 15%
    sl = min(sl, entry * 0.98)    # sanity: at least ~2% below entry
    sl = round(sl, 2)

    risk = entry - sl
    if risk <= 0:
        risk = entry * 0.04

    # ── Targets from the base's MEASURED MOVE (projected base height) + ATH —
    # per-stock, structural, not fixed multiples of an 8% risk.
    base_height = (base_high - base_low) if base_high > base_low else 2 * risk
    t1 = round(entry + base_height, 2)                       # 1× measured move
    t2 = round(max(entry + 1.6 * base_height, t1 + risk), 2) # 1.6× projection
    t3 = round(max(ath * 1.02, entry + 2.4 * base_height, t2 + risk), 2)  # ATH / extension
    t2 = max(t2, round(t1 + 0.01, 2)); t3 = max(t3, round(t2 + 0.01, 2))

    risk_pct = round(risk / entry * 100, 2)
    # R:R to T1 (the real first/measured-move target). The old code measured to
    # T2 which was DEFINED as 2R, so R:R was tautologically 2.0 for every stock.
    cur = float(df["Close"].iloc[-1]) if len(df) > 0 else entry
    effective_entry = max(entry, cur)
    effective_risk  = effective_entry - sl
    rr = round((t1 - effective_entry) / effective_risk, 2) if effective_risk > 0 else 0

    return {
        "sl":       sl,
        "t1":       t1,
        "t2":       t2,
        "t3":       t3,
        "risk_pct": risk_pct,
        "rr":       rr,
    }


# ── Buyer Demand rating (A–E) ─────────────────────────────────────────────────

def _buyer_demand(df: pd.DataFrame) -> float:
    """
    Raw 13-week (65-session) IBD-style Accumulation/Distribution SCORE — each
    session weighted by volume relative to average × up/down direction.

    Returns the raw float; the A–E letter grade is assigned CROSS-SECTIONALLY
    after the scan (A = top 20% of accumulation, E = bottom 20%). The old code
    used absolute thresholds (score>0.3 → 'A'), but every breakout stock broke
    out ON volume, so the score cleared 0.3 → literally every row showed 'A'.
    A relative grade (like IBD's real A/D rating) actually discriminates.
    """
    try:
        if len(df) < 20:
            return 0.0
        r = df.iloc[-65:].copy() if len(df) >= 65 else df.copy()
        hi, lo, cl, vol = r["High"], r["Low"], r["Close"], r["Volume"]
        avg_vol = float(vol.mean())
        if avg_vol <= 0:
            return 0.0
        # MarketSmith's A/D is built on the Chaikin money-flow multiplier (Close
        # Location Value): where the close sits in the day's range. A close at the
        # HIGH on big volume = strong accumulation even if it's flat day-over-day;
        # a weak close on an up day is NOT. (Our old up/down-sign score missed
        # this — it ranked TDPOWERSYS above PARAS; MarketSmith does the opposite.)
        rng = (hi - lo).replace(0, np.nan)
        clv = (((cl - lo) - (hi - cl)) / rng).fillna(0.0)   # -1 (weak) … +1 (strong)
        mfv = clv * vol                                      # money-flow volume
        return float(mfv.sum()) / (avg_vol * len(r))         # normalized
    except Exception:
        return 0.0


def _ms_rs_metric(close: pd.Series) -> float | None:
    """MarketSmith / IBD Relative-Strength raw metric: blends the stock's
    3/6/9/12-month price performance, weighting the most recent quarter 2×
    (IBD's 40/20/20/20). Percentile-ranked across the universe → the 1-99 RS
    Rating. (Our old Price Str ranked 3-month return ALONE — this adds the
    9-/12-month context MarketSmith uses.)"""
    n = len(close)
    if n < 64:
        return None
    cur = float(close.iloc[-1])
    def perf(k):
        return cur / float(close.iloc[-k]) if (n > k and float(close.iloc[-k]) > 0) else None
    parts = [(0.40, perf(63)), (0.20, perf(126)), (0.20, perf(189)), (0.20, perf(252))]
    avail = [(w, p) for w, p in parts if p is not None]
    if not avail or perf(63) is None:
        return None
    tw = sum(w for w, _ in avail)
    return sum(w * p for w, p in avail) / tw


# ── Industry Group Ranks (computed from all loaded stocks) ────────────────────

def _compute_group_ranks(stocks: dict) -> dict[str, int]:
    """
    Rank each industry group 1–N by 3-month RS vs Nifty proxy.
    Uses only the already-loaded bhavcopy data — zero extra API calls.
    Returns {group_name: rank}  (1 = strongest group).
    """
    try:
        # Canonical 20-stock Nifty proxy basket (single source of truth).
        from analysis_utils import NIFTY_PROXY_SYMS, equal_weight_index
        nifty_closes = [stocks[s]["Close"].dropna()
                        for s in NIFTY_PROXY_SYMS if s in stocks and len(stocks[s]) >= 63]
        if not nifty_closes:
            return {}
        nifty = equal_weight_index(pd.concat(nifty_closes, axis=1).dropna(how="all"))
        nifty_r3m  = (float(nifty.iloc[-1]) / float(nifty.iloc[-63]) - 1) * 100 if len(nifty) >= 63 else 0.0

        group_rs: dict[str, float] = {}
        for grp, syms in INDUSTRY_GROUPS.items():
            closes = [stocks[s]["Close"].dropna()
                      for s in syms if s in stocks and len(stocks[s]) >= 63]
            if len(closes) < 2:
                continue
            grp_idx = equal_weight_index(pd.concat(closes, axis=1).dropna(how="all"))
            if len(grp_idx) < 63:
                continue
            r3m = (float(grp_idx.iloc[-1]) / float(grp_idx.iloc[-63]) - 1) * 100
            group_rs[grp] = r3m - nifty_r3m

        sorted_grps = sorted(group_rs, key=lambda x: group_rs[x], reverse=True)
        return {grp: i + 1 for i, grp in enumerate(sorted_grps)}
    except Exception:
        return {}


# ── Per-stock analysis ────────────────────────────────────────────────────────

def _analyze(symbol: str, df: pd.DataFrame) -> dict | None:
    try:
        close = df["Close"].dropna()
        high  = df["High"].dropna()
        low   = df["Low"].dropna()
        vol   = df["Volume"].dropna()

        if len(close) < MIN_BARS:
            return None

        cur = float(close.iloc[-1])
        if cur <= 0:
            return None

        # Liquidity filter
        avg_vol  = float(vol.rolling(20).mean().iloc[-1]) if len(vol) >= 20 else 0
        adtv_cr  = round(avg_vol * cur / 1e7, 2)
        if adtv_cr < MIN_ADTV_CR:
            return None

        timeframes           = _detect_timeframes(close, vol)
        patterns, pat_levels = _detect_patterns(close, high, low, vol)

        # VCP/Box setups included even before breakout — they're actionable
        if not timeframes and not patterns:
            return None
        if not timeframes and patterns:
            timeframes = ["Setup"]   # pattern forming, breakout not yet triggered
        if not patterns:
            # Pure breakout — no pattern label, use pivot high as base
            patterns = ["Breakout"]
            win = min(20, len(close) - 1)
            pat_levels = {
                "entry":     round(cur * 1.001, 2),
                "base_high": round(cur, 2),
                "base_low":  round(float(low.iloc[-win:].min()), 2),
            }

        ath       = float(close.max())                  # incl. today → pct_ath 0% = at ATH
        # PRIOR ATH (exclude the current bar) for the T3 target, so a stock
        # printing a fresh ATH projects BEYOND its prior high instead of
        # collapsing T3's ATH term to cur×1.02.
        ath_prior = float(close.iloc[:-1].max()) if len(close) > 1 else ath
        pct_ath   = round((cur - ath) / ath * 100, 2)
        vr        = _vol_ratio(vol, 20)

        r1m = round((cur / float(close.iloc[-21])  - 1) * 100, 2) if len(close) >= 21 else 0.0
        r3m = round((cur / float(close.iloc[-63])  - 1) * 100, 2) if len(close) >= 63 else 0.0

        ma50  = round(float(close.rolling(50).mean().iloc[-1]), 2)  if len(close) >= 50  else None
        ma200 = round(float(close.rolling(200).mean().iloc[-1]), 2) if len(close) >= 200 else None

        # Entry / SL / Targets
        entry      = pat_levels.get("entry",     round(cur * 1.003, 2))
        base_high  = pat_levels.get("base_high", round(cur, 2))
        base_low   = pat_levels.get("base_low",  round(float(low.iloc[-20:].min()), 2))

        lvls = _compute_levels(df, entry, base_high, base_low, ath)

        # Trend Template (RS placeholder, updated after full scan)
        tt_score, tt_met = trend_template_score(close, rs_rating=0)
        stage            = stage_analysis(close)
        candles          = detect_candle_signals(df)
        bd               = _buyer_demand(df)
        grp_name         = _SYM_TO_GROUP.get(symbol, "Other")

        return {
            "symbol":      symbol,
            "price":       round(cur, 2),
            "adtv_cr":     adtv_cr,
            "timeframes":  timeframes,
            "patterns":    patterns,
            "vol_ratio":   vr,
            "pct_ath":     pct_ath,
            "r1m":         r1m,
            "r3m":         r3m,
            "above_ma50":  (cur > ma50)  if ma50  else False,
            "above_ma200": (cur > ma200) if ma200 else False,
            # ── Institutional quality ────────────────────────────────
            "tt_score":    tt_score,
            "tt_met":      tt_met,
            "stage":       stage,
            "stage_label": stage_label(stage),
            "candles":     candles,
            "rs_rating":   50,    # updated after full scan
            # ── MarketSmith-style ratings (updated after full scan) ──
            "price_str":    50,   # RS Rating 0-99 (updated after scan)
            "_ad_score":    bd,   # raw A/D score — graded A-E cross-sectionally post-scan
            "buyer_demand": "C",  # placeholder; real grade assigned after full scan
            "group_name":   grp_name,
            "group_rank":   0,    # updated after group RS computed
            "total_groups": 0,
            # ── Trading levels ───────────────────────────────────────
            "entry":       entry,
            "sl":          lvls["sl"],
            "t1":          lvls["t1"],
            "t2":          lvls["t2"],
            "t3":          lvls["t3"],
            "risk_pct":    lvls["risk_pct"],
            "rr":          lvls["rr"],
        }
    except Exception:
        return None


# ── Follow-Through Score (backtest-validated) ─────────────────────────────────

def _assign_follow_through(results: list, stocks: dict) -> None:
    """Attach a 0-100 Follow-Through Score = probability the breakout SUSTAINS
    rather than reverses. Weights were derived from a point-in-time backtest of
    5,285 historical breakouts (10-day forward, success = held a -7% stop AND
    reached +6%): momentum-into-breakout 40% · relative strength 35% · breakout
    volume 17% · close strength 8%. (Base follow-through rate was 32%; the top
    quintile of this score followed through 42% vs the bottom quintile's 17%.)
    The four factors are percentile-ranked across the current breakout set."""
    raw = []
    for r in results:
        df = stocks.get(r["symbol"])
        if df is None or len(df) < 64:
            raw.append(None); continue
        c = df["Close"].astype(float).values; h = df["High"].astype(float).values
        l = df["Low"].astype(float).values;  v = df["Volume"].astype(float).values
        # len >= 64 guaranteed above, so these windows are always valid.
        prior_run = c[-1] / c[-21] - 1                              # momentum into the breakout
        r3m       = c[-1] / c[-64] - 1                              # ~3-month relative strength
        volr      = v[-1] / (v[-51:-1].mean() + 1e-9)              # breakout volume vs 50d avg
        rng       = (h[-1] - l[-1]) or 1e-9
        clv       = ((c[-1] - l[-1]) - (h[-1] - c[-1])) / rng       # close location value
        raw.append((prior_run, r3m, volr, clv))

    valid = [(i, x) for i, x in enumerate(raw) if x is not None]
    if not valid:
        for r in results:
            r["follow_through"] = None; r["ft_tier"] = ""
        return
    dff = pd.DataFrame([x for _, x in valid], columns=["pr", "r3m", "vol", "clv"])
    rk = {col: dff[col].rank(pct=True).values for col in dff.columns}
    fts = (0.40 * rk["pr"] + 0.35 * rk["r3m"] + 0.17 * rk["vol"] + 0.08 * rk["clv"]) * 100
    score_of = {i: float(fts[k]) for k, (i, _) in enumerate(valid)}
    for i, r in enumerate(results):
        s = score_of.get(i)
        if s is None:
            r["follow_through"] = None; r["ft_tier"] = ""
        else:
            r["follow_through"] = round(s)
            r["ft_tier"] = "High" if s >= 65 else "Moderate" if s >= 45 else "Low"


# ── Main entry ────────────────────────────────────────────────────────────────

def run_breakout_scan(progress_callback=None) -> dict:
    if (_cache["data"]
            and time.time() - _cache["ts"] < CACHE_TTL
            and _cache["data"].get("results")):
        return _cache["data"]

    # 1. Load all NSE OHLCV from bhavcopy cache
    stocks = _load_all_stocks(progress_callback)
    if not stocks:
        return {"results": [], "computed_at": int(time.time()), "total_scanned": 0}

    total   = len(stocks)
    done    = [0]
    results = []

    if progress_callback:
        progress_callback(0, total, f"Scanning {total} stocks for breakout patterns…")

    # 2. Analyse in parallel
    def _job(item):
        sym, df = item
        r = _analyze(sym, df)
        done[0] += 1
        if progress_callback and done[0] % 200 == 0:
            progress_callback(done[0], total,
                              f"Scanning {done[0]}/{total} stocks…")
        return r

    with ThreadPoolExecutor(max_workers=SCAN_WORKERS) as ex:
        for r in ex.map(_job, stocks.items()):
            if r:
                results.append(r)

    # 3. Assign RS Rating from r3m rank across the full universe
    # BUG-019 / BUG-027 FIX: prefer ranking across the FULL loaded universe
    # (industry_groups._get_stocks output via `stocks`) rather than just the
    # breakout subset, so RS=99 means top 1% of the whole investable universe.
    # If the universe is unavailable we fall back to subset ranking and flag it.
    if results:
        try:
            # Rank vs the FULL EQ universe (MarketSmith basis: "vs all publicly
            # traded companies") — computed for every EQ name during load.
            universe_r3m = dict(_UNIV_RS)
            universe_ad  = dict(_UNIV_AD)
            if len(universe_r3m) >= len(results):
                u_series = pd.Series(universe_r3m)
                u_ranks  = (u_series.rank(pct=True) * 99).round(0).astype(int)
                rs_arr = [int(u_ranks.get(r["symbol"], 50)) for r in results]
            else:
                # Subset fallback (still directionally correct)
                r3m_s  = pd.Series([r["r3m"] for r in results])
                rs_arr = (r3m_s.rank(pct=True) * 99).round(0).astype(int).tolist()
        except Exception:
            r3m_s  = pd.Series([r["r3m"] for r in results])
            rs_arr = (r3m_s.rank(pct=True) * 99).round(0).astype(int).tolist()
        for r, rs in zip(results, rs_arr):
            r["rs_rating"]  = int(rs)
            r["price_str"]  = int(rs)   # Price Strength = RS Rating 0-99
            # Recompute TT with real RS
            r["tt_score"], r["tt_met"] = trend_template_score(
                stocks[r["symbol"]]["Close"].dropna(), rs_rating=rs
            )

        # Buyer Demand (Acc/Dis) A+ to E — graded vs the WHOLE universe, like
        # MarketSmith (each stock's 13-week A/D rank among ALL stocks). Breakouts
        # skew to A/B because they ARE accumulating more than the average stock —
        # but no longer ALL 'A'. (Old code used an absolute threshold on the
        # breakout subset → everything cleared it.)
        try:
            u_ad = pd.Series(universe_ad)
            if len(u_ad) < 50:
                raise ValueError("universe A/D unavailable")
            for r in results:
                v = u_ad.get(r["symbol"])
                p = float((u_ad <= v).mean()) if v is not None else 0.5
                # MarketSmith-style A+ … E grade bands on the money-flow percentile
                r["buyer_demand"] = ("A+" if p >= 0.90 else "A" if p >= 0.75 else
                                     "B+" if p >= 0.60 else "B" if p >= 0.45 else
                                     "C"  if p >= 0.25 else "D" if p >= 0.10 else "E")
                r.pop("_ad_score", None)
        except Exception:
            ad_pct = pd.Series([r.get("_ad_score", 0.0) for r in results]).rank(pct=True)
            for r, p in zip(results, ad_pct):
                r["buyer_demand"] = ("A" if p >= 0.80 else "B" if p >= 0.60 else
                                     "C" if p >= 0.40 else "D" if p >= 0.20 else "E")
                r.pop("_ad_score", None)

    # 3b. Compute industry group ranks from already-loaded stock data
    if results:
        group_ranks  = _compute_group_ranks(stocks)
        total_groups = len(group_ranks)
        for r in results:
            grp = r.get("group_name", "Other")
            r["group_rank"]   = group_ranks.get(grp, 0)
            r["total_groups"] = total_groups

    # 3c. Follow-Through Score — backtest-validated probability a breakout sustains
    if results:
        _assign_follow_through(results, stocks)

    # 4. Sort: multi-timeframe first, then HTF > ATH > VCP > Box > Rectangular, then TT score
    def _sort_key(r):
        pattern_score = (5 if "HTF"        in r["patterns"] else
                         4 if "ATH"         in r["patterns"] else
                         3 if "VCP"         in r["patterns"] else
                         2 if "Box"         in r["patterns"] else
                         1 if "Rectangular" in r["patterns"] else 0)
        return (len(r["timeframes"]), pattern_score, r["tt_score"], r["vol_ratio"])

    results.sort(key=_sort_key, reverse=True)

    out = {
        "results":       results,
        "computed_at":   int(time.time()),
        "total_scanned": total,
    }
    _cache["data"] = out
    _cache["ts"]   = time.time()

    if progress_callback:
        progress_callback(total, total,
                          f"Done — {len(results)} breakouts in {total} stocks")
    return out
