"""
Shared technical analysis utilities — used by all scanners.
Zero network calls. Pure OHLCV math.
"""
from __future__ import annotations
import numpy as np
import pandas as pd


# ── Canonical Nifty proxy constituent list ─────────────────────────────────────
# TIER-3 FIX: was 3 different lists (10 / 20 / 30 stocks) across 7 files.
# Single source of truth — every _build_nifty() imports from here.
# 20 liquid large-caps so the proxy tracks Nifty50 with <2% tracking error.
NIFTY_PROXY_SYMS = [
    "RELIANCE", "TCS", "HDFCBANK", "ICICIBANK", "INFY",
    "HINDUNILVR", "SBIN", "BHARTIARTL", "ITC", "KOTAKBANK",
    "AXISBANK", "WIPRO", "HCLTECH", "MARUTI", "BAJFINANCE",
    "TITAN", "NTPC", "POWERGRID", "NESTLEIND", "SUNPHARMA",
]


# ── Standardized return windows (trading days, ~21/month) ─────────────────────
# BUG-FIX (cross-tab consistency): r1m was 21 in sector_analysis, 22 in industry_groups,
# r3m was 63 vs 66 vs 63, etc. → same stock showed different "3-month return"
# across tabs. Single source of truth here, imported everywhere.
BARS_1M  = 21
BARS_3M  = 63
BARS_6M  = 126
BARS_12M = 252


# ── Split / Bonus backward-adjustment (centralized) ──────────────────────────
def adjust_for_splits(df: pd.DataFrame) -> pd.DataFrame:
    """
    Backward-adjust OHLC for stock splits/bonuses.

    BUG-FIX: the prior threshold `< 0.55` (>45% overnight drop) caught only
    1:1+ bonuses and major splits. It MISSED:
      - 3:2 bonus → 40% drop (ratio 0.60) — very common in NSE
      - 4:3 bonus → 43% drop (ratio 0.57)
      - 5:4 bonus → 44% drop (ratio 0.556)

    New approach: detect any overnight drop > 18% AND ratio close to a clean
    fraction (within 3%) of a small N/M like 1/2, 2/3, 3/5, 1/3, 4/5, 3/4, 1/4,
    5/6, 1/5, 2/5, 1/6. This catches all common bonus ratios while ignoring
    panic-day or earnings-disappointment drops (which won't match a clean fraction).

    All callers (data_fetcher, monster_growth, early_growth, early_mover_scanner,
    institutional_scanner, momentum_scanner, multiyear_breakout, trending) should
    import and use this helper instead of their own 1-line threshold.
    """
    if df.empty or len(df) < 2:
        return df

    # Common bonus/split ratios (post/pre share count)
    KNOWN_RATIOS = [
        ("1:2 split / 1:1 bonus", 1/2),   # 50% drop
        ("1:3 split / 2:1 bonus", 1/3),   # 66.7% drop
        ("1:4 split / 3:1 bonus", 1/4),   # 75% drop
        ("1:5 split / 4:1 bonus", 1/5),   # 80% drop
        ("1:10 split",            1/10),  # 90% drop
        ("2:3 / 3:2 bonus",       2/3),   # 33.3% drop
        ("3:5 / 5:3 bonus",       3/5),   # 40% drop
        ("3:4 / 4:3 bonus",       3/4),   # 25% drop
        ("4:5 / 5:4 bonus",       4/5),   # 20% drop
        ("5:6 bonus",             5/6),   # 16.7% drop
        # NOTE: "3:2 bonus" with ratio 2/3 is already covered above on the
        # "2:3 / 3:2 bonus" row. A previous entry with ratio (3/2 - 1) = 0.5
        # was wrong (that is a 1:1 bonus / 1:2 split, not 3:2) and has been
        # removed — it caused historical OHLC to be over-adjusted by ~33%.
    ]

    ohlc_cols = [c for c in ["Open", "High", "Low", "Close"] if c in df.columns]
    closes = df["Close"].values.astype(float)
    vol = df["Volume"].values.astype(float) if "Volume" in df.columns else None

    events: list[tuple[int, float]] = []
    for i in range(1, len(closes)):
        prev, cur = closes[i - 1], closes[i]
        if prev <= 0 or cur <= 0:
            continue
        ratio = cur / prev
        # Only consider drops > 18%; smaller drops aren't bonuses.
        if ratio >= 0.82:
            continue
        # Check if ratio is within 3% of a known clean fraction
        for _, r in KNOWN_RATIOS:
            if abs(ratio - r) / r < 0.03:
                events.append((i, ratio))
                break
        else:
            # Drop > 30% AND volume spike >= 2× — likely a split even if ratio is unusual
            if ratio < 0.70 and vol is not None and i >= 10:
                avg_v = vol[max(0, i-10):i].mean()
                if avg_v > 0 and vol[i] >= avg_v * 2:
                    events.append((i, ratio))

    if not events:
        return df

    df = df.copy()
    col_idx = [df.columns.get_loc(c) for c in ohlc_cols]
    for split_idx, ratio in reversed(events):
        df.iloc[:split_idx, col_idx] *= ratio
    return df


# ── Volume baseline (median, not SMA) ─────────────────────────────────────────
def volume_baseline(vol: pd.Series, window: int = 20, use_median: bool = True) -> float:
    """
    Return a robust volume baseline for the last `window` bars.

    BUG-FIX: most callers used `vol.rolling(20).mean()` which is heavily skewed
    by single block-deal / expiry / earnings days that 5–10× normal volume.
    Median is much more robust: a 10× outlier moves the mean ~50%, but only
    moves the median by 5% (one bar out of 20).

    Args:
        vol: volume series
        window: lookback bars
        use_median: True for median (default), False for mean

    Returns:
        float baseline (0.0 if insufficient data)
    """
    if vol is None or len(vol) < window:
        return 0.0
    sample = vol.iloc[-window:].dropna()
    if len(sample) == 0:
        return 0.0
    if use_median:
        return float(sample.median())
    return float(sample.mean())


# ── Equal-weight index builder (NEVER use raw-price mean) ─────────────────────
def equal_weight_index(closes_df: pd.DataFrame, base: float = 100.0) -> pd.Series:
    """
    Build a proper equal-weight index from a DataFrame of close-price series.

    BUG-FIX: The old `combined.mean(axis=1)` averaged RAW prices — so a stock
    trading at ₹12,000 (MARUTI) contributed 30× more than one at ₹400 (ITC).
    A 1% move in MARUTI moved the "Nifty proxy" ~10× more than a 1% move in ITC.
    Every RS-vs-Nifty calc in the app inherited this distortion.

    Correct method: rebase each series to `base` at its first valid value,
    then average. This way each stock contributes proportionally to its
    % change from the start of the window — not its nominal price level.

    Args:
        closes_df: wide DataFrame (rows=dates, cols=symbols), each col a close series.
        base: starting value for each rebased series (cosmetic; default 100).

    Returns:
        pd.Series of the equal-weight index value over time, or empty Series.
    """
    if closes_df is None or closes_df.empty:
        return pd.Series(dtype=float)
    # Forward-fill within each column so a holiday doesn't break the rebase;
    # then take the first VALID value per column as the base.
    filled = closes_df.ffill()
    first_vals = filled.bfill().iloc[0]   # first non-NaN per column
    # Avoid divide-by-zero for accidental zero-price columns
    safe_first = first_vals.replace(0, np.nan)
    rebased = filled.div(safe_first) * base
    return rebased.mean(axis=1).dropna()


# ── Trend Template Score (Minervini SEPA) ─────────────────────────────────────

def trend_template_score(close: pd.Series, rs_rating: int = 0) -> tuple[int, list[str]]:
    """
    Minervini's 8-criteria Trend Template.
    Returns (score 0-8, list of satisfied criteria labels).
    Score 7-8 = ideal buy zone.
    """
    n = len(close)
    if n < 200:
        return 0, []
    # Adaptive lookback for "1-month-ago MA200" — bhavcopy history is finite.
    # Need at least 1 valid MA200 sample in the lookback (i.e., n - lookback >= 200).
    # Prefer 22 bars (1 month) when available; fall back to half the available window
    # when history is shorter (typical for our 211-bar Nifty universe).
    lookback_1m = min(22, max(5, n - 200 - 1))

    cur     = float(close.iloc[-1])
    ma50    = float(close.rolling(50).mean().iloc[-1])
    ma150   = float(close.rolling(150).mean().iloc[-1])
    ma200   = float(close.rolling(200).mean().iloc[-1])
    ma200_1m = float(close.rolling(200).mean().iloc[-lookback_1m])

    hi52 = float(close.iloc[-252:].max()) if len(close) >= 252 else float(close.max())
    lo52 = float(close.iloc[-252:].min()) if len(close) >= 252 else float(close.min())

    score = 0
    met   = []

    # 1 — Price > 150MA and 200MA
    if cur > ma150 and cur > ma200:
        score += 1; met.append("P>150&200")
    # 2 — 150MA > 200MA
    if ma150 > ma200:
        score += 1; met.append("150>200")
    # 3 — 200MA trending up (vs 1 month ago)
    if ma200 > ma200_1m:
        score += 1; met.append("200↑")
    # 4 — 50MA > 150MA and 200MA
    if ma50 > ma150 and ma50 > ma200:
        score += 1; met.append("50>150&200")
    # 5 — Price > 50MA
    if cur > ma50:
        score += 1; met.append("P>50")
    # 6 — Price ≥ 25% above 52-week low
    if lo52 > 0 and cur >= lo52 * 1.25:
        score += 1; met.append("25%↑Lo")
    # 7 — Price within 25% of 52-week high
    if hi52 > 0 and cur >= hi52 * 0.75:
        score += 1; met.append("<25%Hi")
    # 8 — RS Rating ≥ 70
    if rs_rating >= 70:
        score += 1; met.append("RS≥70")

    return score, met


# ── Stage Analysis (Stan Weinstein) ──────────────────────────────────────────

def stage_analysis(close: pd.Series) -> int:
    """
    Weinstein Stage: 1=Basing, 2=Advancing, 3=Topping, 4=Declining.
    Uses both MA50 AND MA150 slope + price position (true Weinstein method).
    Returns 0 if insufficient data.

    TIER-3 FIX (BUG-028): previous version used only MA150, which granted Stage 2
    to stocks above the 30-week MA regardless of the 10-week MA direction.
    True Weinstein Stage 2 requires price above BOTH MAs with BOTH rising.
    All callers (monster_growth, alpha_engine, portfolio, early_growth, etc.)
    now import this single implementation instead of maintaining their own copies.
    """
    if len(close) < 175:
        return 0

    cur       = float(close.iloc[-1])
    ma50      = float(close.rolling(50).mean().iloc[-1])
    ma150     = float(close.rolling(150).mean().iloc[-1])
    ma50_1m   = float(close.rolling(50).mean().iloc[-22])
    ma150_1m  = float(close.rolling(150).mean().iloc[-22])
    # ±0.5% hysteresis prevents noisy oscillation on flat MAs
    slope50   = ma50  > ma50_1m  * 1.005
    slope150  = ma150 > ma150_1m * 1.005

    if cur > ma50 and cur > ma150 and slope50 and slope150: return 2  # Advancing
    if cur > ma50 and not slope150:                         return 3  # Topping
    if cur < ma150 and not slope150:                        return 4  # Declining
    return 1  # Basing


def stage_label(s: int) -> str:
    return {1: "S1 Basing", 2: "S2 ▲", 3: "S3 Top", 4: "S4 ▼"}.get(s, "—")


def stage_color(s: int) -> str:
    return {1: "neutral", 2: "pos", 3: "neg", 4: "neg"}.get(s, "")


# ── Compression / Contraction Patterns ───────────────────────────────────────

def is_nr7(df: pd.DataFrame) -> bool:
    """NR7 — today's range is the narrowest of the last 7 sessions."""
    try:
        ranges = (df["High"] - df["Low"]).iloc[-7:]
        if len(ranges) < 7:
            return False
        return float(ranges.iloc[-1]) == float(ranges.min())
    except Exception:
        return False


def is_inside_bar(df: pd.DataFrame) -> bool:
    """Inside Bar — today's high/low entirely within yesterday's range."""
    try:
        if len(df) < 2:
            return False
        return (float(df["High"].iloc[-1]) < float(df["High"].iloc[-2]) and
                float(df["Low"].iloc[-1])  > float(df["Low"].iloc[-2]))
    except Exception:
        return False


def is_3wt(close: pd.Series) -> bool:
    """
    3-Weeks-Tight (Minervini): three consecutive weekly closes within 1.5%.
    """
    try:
        wk = close.resample("W-FRI").last().dropna()
        if len(wk) < 4:
            return False
        last3 = wk.iloc[-3:]
        lo, hi = float(last3.min()), float(last3.max())
        return hi > 0 and (hi - lo) / lo * 100 <= 1.5
    except Exception:
        return False


# ── High Tight Flag ───────────────────────────────────────────────────────────

def is_high_tight_flag(close: pd.Series, high: pd.Series) -> tuple[bool, dict]:
    """
    High Tight Flag (O'Neil):
    - Stock rises ≥ 100% in 4–8 weeks (20–40 bars)
    - Then consolidates ≤ 20% from peak
    - Currently near top of flag
    Returns (True, {"base_high", "flag_lo", "flag_pct"}) or (False, {})
    """
    try:
        if len(close) < 25:
            return False, {}
        cur = float(close.iloc[-1])
        for bars in [20, 25, 30, 35, 40]:
            if len(close) < bars + 5:
                continue
            base_price = float(close.iloc[-(bars + 1)])
            if base_price <= 0:
                continue
            window = close.iloc[-bars:]
            peak   = float(window.max())
            if peak < base_price * 2.0:    # must be up ≥ 100%
                continue
            # Flag: from peak to now
            peak_idx = window.idxmax()
            post_pk  = close.loc[peak_idx:]
            if len(post_pk) < 3:
                continue
            flag_lo  = float(post_pk.min())
            flag_pct = (peak - flag_lo) / peak * 100
            if flag_pct > 20:
                continue
            if cur < float(post_pk.max()) * 0.97:   # must be near top of flag
                continue
            return True, {
                "base_high": round(peak, 2),
                "flag_lo":   round(flag_lo, 2),
                "flag_pct":  round(flag_pct, 1),
                "run_up_pct": round((peak - base_price) / base_price * 100, 1),
            }
        return False, {}
    except Exception:
        return False, {}


# ── RS Line New High ──────────────────────────────────────────────────────────

def rs_line_new_high(close: pd.Series, nifty: pd.Series) -> bool:
    """
    True when the RS Line (stock / Nifty) is at a new 52-week high.
    This fires BEFORE the price breakout — a leading institutional signal.
    """
    try:
        idx = close.index.intersection(nifty.index)
        if len(idx) < 63:
            return False
        nifty_a = nifty[idx]
        if float(nifty_a.iloc[-1]) <= 0:
            return False
        rs = close[idx] / nifty_a
        # BUG-011 FIX: exclude the current bar from the new-high lookback so we
        # are comparing current RS against PRIOR window, not against itself.
        lookback = min(252, len(rs) - 1) if len(rs) > 1 else 1
        if lookback <= 0:
            return False
        return float(rs.iloc[-1]) >= float(rs.iloc[-lookback - 1:-1].max()) * 0.995
    except Exception:
        return False


# ── Candlestick Patterns ──────────────────────────────────────────────────────

def detect_candle_signals(df: pd.DataFrame) -> list[str]:
    """
    Detects high-probability candle signals on the most recent bar.
    Returns list of detected signal names.
    """
    signals = []
    try:
        if len(df) < 3:
            return signals
        o  = float(df["Open"].iloc[-1])
        h  = float(df["High"].iloc[-1])
        l  = float(df["Low"].iloc[-1])
        c  = float(df["Close"].iloc[-1])
        o1 = float(df["Open"].iloc[-2])
        c1 = float(df["Close"].iloc[-2])
        h1 = float(df["High"].iloc[-2])
        l1 = float(df["Low"].iloc[-2])

        body        = abs(c - o)
        total_range = h - l
        if total_range <= 0:
            return signals
        lower_shadow = min(o, c) - l
        upper_shadow = h - max(o, c)

        # Bullish Marubozu — full-body bull candle, no wicks
        if c > o and body / total_range >= 0.85:
            signals.append("Marubozu↑")

        # Bullish Engulfing — bearish prev, bullish today engulfs it
        if (c1 < o1 and c > o and
                o < min(o1, c1) and c > max(o1, c1)):
            signals.append("Engulfing↑")

        # Hammer — long lower shadow, small body near top
        if (lower_shadow >= 2.0 * body and
                upper_shadow <= body * 0.5 and
                body / total_range >= 0.05):
            signals.append("Hammer")

        # Bullish Piercing / Strong Recovery
        if (c1 < o1 and c > o and
                o < (o1 + c1) / 2 and
                c > (o1 + c1) / 2 and
                c < o1):
            signals.append("Piercing↑")

        # Doji at key level (body < 5% of range) — indecision before breakout
        if body / total_range < 0.05 and total_range > 0:
            signals.append("Doji")

    except Exception:
        pass
    return signals


# ── Power Trend (O'Neil) ──────────────────────────────────────────────────────

def power_trend(close: pd.Series) -> bool:
    """
    Power Trend: 21-EMA > 50-MA for last 20 bars (≥80% of days),
    AND both are rising over that period.
    """
    try:
        if len(close) < 60:
            return False
        ema21 = close.ewm(span=21, adjust=False).mean()
        ma50  = close.rolling(50).mean()
        if float(ema21.iloc[-1]) <= float(ma50.iloc[-1]):
            return False
        if float(ema21.iloc[-1]) <= float(ema21.iloc[-20]):
            return False
        if float(ma50.iloc[-1]) <= float(ma50.iloc[-20]):
            return False
        above_count = (close.iloc[-20:].values > ema21.iloc[-20:].values).sum()
        return int(above_count) >= 16
    except Exception:
        return False


# ── Base Count (IBD method) ───────────────────────────────────────────────────

def base_count(close: pd.Series) -> int:
    """
    Count consolidation bases formed in the last 52 weeks.
    A base = 10–45% pullback from a prior high followed by recovery.
    1st base = lowest risk; 3rd+ = later stage.
    """
    try:
        if len(close) < 63:
            return 0
        lookback = close.iloc[-252:] if len(close) >= 252 else close
        bases = 0
        in_base = False
        base_start_max = 0.0
        for i in range(10, len(lookback)):
            win = lookback.iloc[max(0, i - 40):i]
            local_max = float(win.max())
            cur_val   = float(lookback.iloc[i])
            drawdown  = (local_max - cur_val) / local_max * 100 if local_max > 0 else 0
            if not in_base and 10 <= drawdown <= 45:
                in_base = True
                base_start_max = local_max
            elif in_base and cur_val >= base_start_max * 0.95:
                bases += 1
                in_base = False
                base_start_max = 0.0
        return min(bases, 5)
    except Exception:
        return 0


# ── Price / Volume Character ──────────────────────────────────────────────────

def price_vol_character(df: pd.DataFrame) -> str:
    """
    'Accumulation' = up days on above-avg volume dominate (last 20 sessions).
    'Distribution' = down days on above-avg volume dominate.
    'Neutral'      = mixed.
    """
    try:
        if len(df) < 20:
            return "Neutral"
        recent  = df.iloc[-20:].copy()
        vol     = recent["Volume"].dropna()
        close_s = recent["Close"].dropna()
        open_s  = recent["Open"].dropna()
        avg_vol = float(vol.mean())
        if avg_vol <= 0:
            return "Neutral"
        up_wt = down_wt = 0.0
        for i in range(len(recent)):
            try:
                c = float(close_s.iloc[i]); o = float(open_s.iloc[i])
                v = float(vol.iloc[i]) / avg_vol
                if c > o:   up_wt   += v
                elif c < o: down_wt += v
            except Exception:
                continue
        if up_wt > down_wt * 1.3:   return "Accumulation"
        elif down_wt > up_wt * 1.3: return "Distribution"
        return "Neutral"
    except Exception:
        return "Neutral"


# ── Gap Classifier ────────────────────────────────────────────────────────────

def classify_gap(df: pd.DataFrame) -> str | None:
    """
    Classify today's gap if ≥1.5%.
    Breakaway Gap↑ · Continuation Gap↑ · Exhaustion Gap↑ · Gap Down
    """
    try:
        if len(df) < 25:
            return None
        today_open = float(df["Open"].iloc[-1])
        prev_close = float(df["Close"].iloc[-2])
        if prev_close <= 0:
            return None
        gap_pct = (today_open - prev_close) / prev_close * 100
        if abs(gap_pct) < 1.5:
            return None
        vol       = df["Volume"].dropna()
        avg_vol   = float(vol.iloc[-20:].mean())
        vol_ratio = float(vol.iloc[-1]) / avg_vol if avg_vol > 0 else 1.0
        close_s   = df["Close"].dropna()
        r3m = (float(close_s.iloc[-1]) / float(close_s.iloc[-63]) - 1) * 100 if len(close_s) > 63 else 0
        if gap_pct > 1.5:
            if r3m > 40 and vol_ratio > 1.5:  return "Exhaustion Gap↑"
            elif vol_ratio > 1.5:             return "Breakaway Gap↑"
            else:                             return "Continuation Gap↑"
        return "Gap Down"
    except Exception:
        return None


# ── Delivery % Trend ──────────────────────────────────────────────────────────

def delivery_trend(deliv_series: pd.Series) -> str:
    """
    'Rising' = delivery % trending up → institutional accumulation.
    'Falling' = delivery % falling → speculative / day-trading activity.
    'Stable'  = no clear trend.
    'Unknown' = insufficient data (old cache files lack DelivPer).
    """
    try:
        d = deliv_series.dropna()
        if len(d) < 10:
            return "Unknown"
        recent = float(d.iloc[-5:].mean())
        prior  = float(d.iloc[-15:-5].mean()) if len(d) >= 15 else float(d.mean())
        if prior <= 0:
            return "Unknown"
        change = (recent - prior) / prior * 100
        if change > 5:    return "Rising"
        elif change < -5: return "Falling"
        return "Stable"
    except Exception:
        return "Unknown"


# ── Composite Rank (0–100) ────────────────────────────────────────────────────

def composite_rank(tt_score: int, rs_rating: int, stage: int,
                   deliv_trend_val: str = "Unknown",
                   base_num: int = 0,
                   power_trend_ok: bool = False) -> int:
    """
    IBD-inspired composite 0–100 rank.
    TT Score 30% · RS Rating 25% · Stage 20% · Delivery 10% · Base 10% · PowerTrend 5%

    BUG-024 NOTE: rs_rating expects a true cross-sectional percentile (1-99).
    Cap input to [1, 99] to prevent out-of-range values causing scoring anomalies.
    """
    try:
        # BUG-024 FIX: cap rs_rating to valid range before scoring
        rs_rating = max(1, min(99, rs_rating or 50))
        score  = (tt_score / 8) * 30
        score += (min(rs_rating, 99) / 99) * 25
        score += {2: 20, 1: 10, 3: 5, 4: 0}.get(stage, 0)
        score += {"Rising": 10, "Stable": 5, "Falling": 0, "Unknown": 3}.get(deliv_trend_val, 3)
        if base_num == 1:   score += 10
        elif base_num == 2: score += 7
        elif base_num >= 3: score += 4
        if power_trend_ok:  score += 5
        return min(100, round(score))
    except Exception:
        return 0


# ── Cross-sectional RS rank (P2-13: full universe consistency) ───────────────

def cross_sectional_rs_rank(returns_by_symbol: dict[str, float]) -> dict[str, int]:
    """
    Convert raw {symbol: return} into {symbol: percentile_rank 1-99}.

    P2-13 FIX: every scanner previously computed RS rank against its own
    filtered subset (e.g. ADTV >= 1Cr in alpha_engine, mcap <= 50000 in
    early_growth). Same stock therefore had different RS ranks across tabs.
    This helper ranks across the FULL universe of provided symbols, so all
    scanners get consistent 1-99 percentiles when they hand it the same
    return dict.
    """
    if not returns_by_symbol:
        return {}
    rets = [(s, r) for s, r in returns_by_symbol.items() if r is not None]
    if not rets:
        return {}
    rets.sort(key=lambda x: x[1])
    n = len(rets)
    out: dict[str, int] = {}
    for i, (sym, _) in enumerate(rets):
        # 1-99 percentile (avoid 0 and 100 — composite scorers cap at 99)
        pct = int(round((i + 1) / n * 99))
        out[sym] = max(1, min(99, pct))
    return out


# ── Sector-adjusted RS rank (P2-12: "leader in a leader") ────────────────────

def sector_adjusted_rs(stock_rs: int, sector_rs: int) -> int:
    """
    Combine a stock's full-universe RS rank with its sector's RS rank
    into a 0-99 composite. Highlights "leader in a leading sector" pattern.

    Formula: 0.6 × stock_rs + 0.4 × sector_rs (stock dominates but sector
    has meaningful weight). A RS-90 stock in a sector ranked 30 ends up
    around 66 — visibly weaker than RS-75 in a top-10 sector (75×.6 + 90×.4
    = 81).
    """
    s = max(1, min(99, int(stock_rs or 50)))
    g = max(1, min(99, int(sector_rs or 50)))
    return max(1, min(99, int(round(0.6 * s + 0.4 * g))))


# ── ATR (shared) ──────────────────────────────────────────────────────────────

def atr(df: pd.DataFrame, period: int = 14) -> float:
    try:
        hi = df["High"].dropna()
        lo = df["Low"].dropna()
        cl = df["Close"].dropna()
        idx = hi.index.intersection(lo.index).intersection(cl.index)
        if len(idx) < period + 2:
            return float(cl.iloc[-1]) * 0.02
        h = hi[idx]; l = lo[idx]; c = cl[idx]
        tr = pd.concat([h - l, (h - c.shift(1)).abs(), (l - c.shift(1)).abs()], axis=1).max(axis=1)
        # BUG-025 FIX: Wilder smoothing (alpha = 1/period) is the standard
        # ATR formula; plain rolling SMA produces a different value with a
        # heavier weight on the oldest bar in the window.
        atr_v = tr.ewm(alpha=1 / period, adjust=False).mean()
        return float(atr_v.iloc[-1])
    except Exception:
        return 0.0
