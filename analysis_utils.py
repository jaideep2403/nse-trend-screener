"""
Shared technical analysis utilities — used by all scanners.
Zero network calls. Pure OHLCV math.
"""
import numpy as np
import pandas as pd


# ── Trend Template Score (Minervini SEPA) ─────────────────────────────────────

def trend_template_score(close: pd.Series, rs_rating: int = 0) -> tuple[int, list[str]]:
    """
    Minervini's 8-criteria Trend Template.
    Returns (score 0-8, list of satisfied criteria labels).
    Score 7-8 = ideal buy zone.
    """
    if len(close) < 220:
        return 0, []

    cur     = float(close.iloc[-1])
    ma50    = float(close.rolling(50).mean().iloc[-1])
    ma150   = float(close.rolling(150).mean().iloc[-1])
    ma200   = float(close.rolling(200).mean().iloc[-1])
    ma200_1m = float(close.rolling(200).mean().iloc[-22])   # 1 month ago

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
    Uses 30-week (≈150-day) MA slope + price position.
    Returns 0 if insufficient data.
    """
    if len(close) < 160:
        return 0

    ma150      = close.rolling(150).mean()
    cur        = float(close.iloc[-1])
    ma_now     = float(ma150.iloc[-1])
    ma_1m_ago  = float(ma150.iloc[-22])   # ~1 month
    ma_rising  = ma_now > ma_1m_ago * 1.001
    ma_falling = ma_now < ma_1m_ago * 0.999
    above_ma   = cur > ma_now

    if above_ma and ma_rising:  return 2  # Advancing — buy
    if above_ma and ma_falling: return 3  # Topping   — caution
    if not above_ma and ma_falling: return 4  # Declining — avoid
    return 1  # Basing — watch


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
        wk = close.resample("W").last().dropna()
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
        lookback = min(252, len(rs))
        return float(rs.iloc[-1]) >= float(rs.iloc[-lookback:].max()) * 0.995
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
    """
    try:
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
        return float(tr.rolling(period).mean().iloc[-1])
    except Exception:
        return 0.0
