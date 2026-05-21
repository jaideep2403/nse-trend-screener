"""
Early Growth Scanner — catches stocks like ACUTAAS 2-4 quarters BEFORE they become obvious.

The monster move starts at "Earnings Inflection" — the quarter when operating leverage
kicks in for the first time. By the time 130%+ profit growth is visible, most of the
initial re-rating has already happened. This scanner targets the EARLIER signal:

  Stage 1 base (or very early Stage 2) + earnings just starting to accelerate
  + small market cap + promoter buying + delivery % rising quietly

Scoring (0–100):
  Factor                        Max   Rationale
  ─────────────────────────    ────  ──────────────────────────────────────────────
  1. Base Quality               35   Length × tightness × volume decline = coiled spring
  2. Earnings Acceleration      25   First signs of operating leverage kicking in
  3. Market Cap (early phase)   20   < ₹500Cr can 10×; > ₹5000Cr the easy move is done
  4. Smart Money / Promoter     15   Insiders + delivery % = accumulation before crowd
  5. Revenue Growth Quality      5   Revenue accelerating confirms the earnings story
  ──────────────────────────   ────
  Total                        100

Tiers:
  ≥ 75  → 🌱 EARLY MONSTER  (base + fundamentals = rare early setup)
  60-74 → 💡 WATCH           (promising but incomplete — monitor for 1-2 more quarters)
  45-59 → 📋 TRACK           (early signals only — needs fundamental confirmation)
  < 45  → skip

Hard filters before scoring:
  - Stage must be 1 (basing) OR Stage 2 with MA150 cross < 40 bars ago
  - Profit growth ≥ 15% (first signs of acceleration, not full confirmation)
  - PE > 0 (profitable — avoid turnaround stories at this stage)
  - Market cap ≤ ₹8000 Cr (beyond this, the easy phase is over)
  - Base length ≥ 5 weeks (at least 5 weeks of consolidation)
  - Base depth ≤ 40% (not a wild volatile stock)

Data: all local — bhavcopy OHLCV + DelivPer + fundamentals.db
"""
from __future__ import annotations

import time
import threading
from typing import Optional

import numpy as np
import pandas as pd

from fundamentals import load_all_fundamentals
from data_fetcher import _weekdays_back, _download_one_day

# ── Constants ──────────────────────────────────────────────────────────────────
MIN_BARS          = 150    # need MA150
MIN_PROFIT_PCT    = 15.0   # first signs of acceleration (lower than Monster's 25%)
# Universe for "monstrous moves" = small + mid + small-large caps.
# Genuine 2x-10x moves come from this range; pure mega-caps rarely double.
# Pre-fix this was 8000 Cr — killed 19/20 valid accelerators in current run.
MAX_MKTCAP_CR     = 50000  # ₹50,000 Cr — covers small + mid + small-large caps
MIN_BASE_WEEKS    = 5      # at least 5 weeks consolidation
MAX_BASE_DEPTH    = 40.0   # base can't be wider than 40% (volatile = not institutional)
MAX_CROSS_BARS    = 40     # "early Stage 2" = crossed MA150 within last 40 bars
# PE sanity for "monstrous moves" — allow growth premium (up to 100) but
# block bubble valuations (PE > 150) where re-rating downside outweighs upside.
MIN_PE            = 5.0    # below this = likely cyclical bottom / data error
MAX_PE            = 150.0  # above this = bubble territory, mean reversion risk
# Data sanity: profit growth above this is typically a turnaround / recovery
# from negative EPS (e.g. EDELWEISS +964% TTM but 3-yr CAGR = -43%). Cap at 500%
# to filter out base-rate artifacts; genuine "monstrous movers" almost never
# exceed 300% sustained growth.
MAX_PROFIT_PCT    = 500.0
# SUSTAINED-ONLY filter: require 3-year CAGR >= 0 to exclude pure turnaround
# plays where the recent acceleration follows multi-year decline.
# Set to None to disable; a positive threshold (e.g. 5.0) requires real growth.
MIN_3Y_CAGR       = 0.0
CACHE_TTL         = 3600

_cache:      dict = {"data": None, "ts": 0.0}
_cache_lock  = threading.Lock()


# ── Split adjustment ───────────────────────────────────────────────────────────

def _adjust_for_splits(df):
    """Delegate to canonical analysis_utils.adjust_for_splits."""
    from analysis_utils import adjust_for_splits
    return adjust_for_splits(df)


# ── Universe loader (includes DelivPer) ───────────────────────────────────────

def _load_universe(progress_callback=None) -> dict[str, pd.DataFrame]:
    """
    Load full NSE universe OHLCV + DelivPer from bhavcopy cache.
    We scan ALL Nifty Total Market 750 stocks because early-phase stocks are
    typically small/mid cap and only show up in the wider Total Market universe.
    Split-adjusted. Zero live API calls.
    """
    dates  = _weekdays_back(300)
    total  = len(dates)
    frames = []

    for i, dt in enumerate(dates):
        df = _download_one_day(dt)
        if df is not None:
            frames.append(df)
        if progress_callback and i % 50 == 0:
            progress_callback(i, total, f"Loading OHLCV… {i}/{total} days")

    if not frames:
        return {}

    combined = pd.concat(frames, ignore_index=True).sort_values("Date")
    stocks   = {}

    for sym, grp in combined.groupby("Symbol"):
        # Skip illiquid / very small stocks — must have volume data
        want_cols = [c for c in ["Open","High","Low","Close","Volume","DelivPer"]
                     if c in grp.columns]
        g = grp.set_index("Date")[want_cols]
        g = g[~g.index.duplicated(keep="last")].sort_index()
        g = _adjust_for_splits(g)
        if len(g) >= MIN_BARS:
            stocks[sym] = g

    if progress_callback:
        progress_callback(total, total, f"Loaded {len(stocks)} stocks ✓")
    return stocks


# ── Nifty proxy ────────────────────────────────────────────────────────────────

# Canonical 20-stock Nifty proxy basket (was a local 10-stock duplicate
# that gave Early Growth a different RS-vs-Nifty than every other tab).
from analysis_utils import NIFTY_PROXY_SYMS as _NIFTY_SYMS

def _build_nifty(stocks: dict) -> Optional[pd.Series]:
    series = []
    for sym in _NIFTY_SYMS:
        df = stocks.get(sym)
        if df is not None and len(df) >= 63:
            c = df["Close"].dropna()
            if float(c.iloc[0]) > 0:
                series.append(c / float(c.iloc[0]))
    if not series:
        return None
    combined = pd.concat(series, axis=1).dropna(how="all")
    from analysis_utils import equal_weight_index
    return equal_weight_index(combined) if len(combined) >= 20 else None


# ── Technical helpers ──────────────────────────────────────────────────────────

def _safe(v, default: float = 0.0) -> float:
    if v is None:
        return default
    try:
        f = float(v)
        return default if (np.isnan(f) or np.isinf(f)) else f
    except Exception:
        return default


def _stage_signal(c: pd.Series) -> tuple[int, float]:
    """
    Returns (stage, ma150_slope_pct).
    Stage: 1=basing, 2=advancing, 3=topping, 4=declining.
    ma150_slope_pct: % change in MA150 over last 22 bars (positive = rising).
    """
    if len(c) < 160:
        return 0, 0.0
    ma150   = c.rolling(150).mean()
    ma50    = c.rolling(50).mean()
    cur     = float(c.iloc[-1])
    ma_now  = float(ma150.iloc[-1])
    ma_prev = float(ma150.iloc[-22])
    slope   = (ma_now - ma_prev) / max(ma_prev, 1) * 100
    rising  = ma_now > ma_prev
    above   = cur > ma_now
    above50 = cur > float(ma50.iloc[-1])

    if above and above50 and rising:  return 2, slope
    if above and above50 and not rising: return 3, slope
    if not above and not rising:      return 4, slope
    return 1, slope


def _bars_since_ma150_cross(c: pd.Series) -> Optional[int]:
    """
    How many bars ago did price cross ABOVE MA150?
    Returns None if still below MA150 (Stage 1) or if cross was > MAX_CROSS_BARS ago.
    Returns 0 if cross was exactly yesterday.
    """
    if len(c) < 152:
        return None
    ma150 = c.rolling(150).mean()
    cur_above = float(c.iloc[-1]) > float(ma150.iloc[-1])
    if not cur_above:
        return None   # still in Stage 1 — caller uses this to confirm Stage 1
    for i in range(1, min(MAX_CROSS_BARS + 1, len(c) - 150)):
        was_below = float(c.iloc[-i - 1]) <= float(ma150.iloc[-i - 1])
        if was_below:
            return i  # crossed i bars ago
    return None  # has been above MA150 for > MAX_CROSS_BARS bars → too late


def _base_analysis(c: pd.Series, h: pd.Series, lo: pd.Series,
                   v: pd.Series) -> dict:
    """
    Analyse the current consolidation/base.

    Algorithm:
      Walk backward from today, tracking range (high-low).
      Stop when range exceeds MAX_BASE_DEPTH or we've gone back 260 bars (1 year).
      Base = the longest window where price stayed within MAX_BASE_DEPTH%.

    Returns dict with:
      base_weeks        : length in weeks
      base_depth_pct    : (base_high - base_low) / base_high * 100
      vol_declining     : True if volume trended down during the base
      near_high         : True if current price within 8% of base high
      pos_in_base       : 0.0–1.0 (0=at base low, 1=at base high)
      base_high, base_low : price levels
    """
    n = len(c)
    if n < 20:
        return {"base_weeks": 0, "base_depth_pct": 100, "vol_declining": False,
                "near_high": False, "pos_in_base": 0.5,
                "base_high": 0.0, "base_low": 0.0}

    cur        = float(c.iloc[-1])
    running_hi = float(h.iloc[-1])
    running_lo = float(lo.iloc[-1])
    base_bars  = 1

    for i in range(1, min(260, n)):
        day_hi = float(h.iloc[-i])
        day_lo = float(lo.iloc[-i])
        cand_hi = max(running_hi, day_hi)
        cand_lo = min(running_lo, day_lo)
        # TIER-3 FIX: symmetric midpoint formula avoids asymmetry where the
        # same ₹50 range looks "deeper" at ₹100 than at ₹500.
        mid   = (cand_hi + cand_lo) / 2
        depth = (cand_hi - cand_lo) / mid * 100 if mid > 0 else 100
        if depth > MAX_BASE_DEPTH:
            break
        running_hi = cand_hi
        running_lo = cand_lo
        base_bars  = i + 1

    mid_price  = (running_hi + running_lo) / 2
    base_depth = (running_hi - running_lo) / mid_price * 100 if mid_price > 0 else 100
    base_weeks  = base_bars // 5
    near_high   = cur >= running_hi * 0.92

    pos = ((cur - running_lo) / (running_hi - running_lo)
           if running_hi > running_lo else 0.5)
    pos = max(0.0, min(1.0, round(pos, 2)))

    # Volume: compare first half vs second half of base
    vol_declining = False
    if base_bars >= 20 and len(v) >= base_bars:
        half      = base_bars // 2
        early_vol = float(v.iloc[-base_bars:-half].mean()) if half > 0 else 0
        late_vol  = float(v.iloc[-half:].mean())           if half > 0 else 0
        if early_vol > 0:
            vol_declining = late_vol < early_vol * 0.88   # 12%+ volume decline = drying up

    return {
        "base_weeks":    base_weeks,
        "base_depth_pct": round(base_depth, 1),
        "vol_declining":  vol_declining,
        "near_high":      near_high,
        "pos_in_base":    pos,
        "base_high":      round(running_hi, 2),
        "base_low":       round(running_lo, 2),
    }


def _delivery_trend(df: pd.DataFrame) -> Optional[float]:
    """
    Delivery % trend during last 40 trading days.
    Returns positive value if delivery % is rising (accumulation),
    negative if falling (distribution), None if data unavailable.
    Rising delivery % during a quiet base = smart money quietly loading.
    """
    if "DelivPer" not in df.columns:
        return None
    dp = df["DelivPer"].dropna()
    if len(dp) < 40:
        return None
    try:
        # Compare last 20d avg vs prior 20d avg
        recent = float(dp.iloc[-20:].mean())
        prior  = float(dp.iloc[-40:-20].mean())
        if prior <= 0:
            return None
        return round((recent - prior) / prior * 100, 1)   # % change in delivery %
    except Exception:
        return None


def _rs_pct(c: pd.Series, nifty: Optional[pd.Series], bars: int = 66) -> float:
    """N-bar excess return vs Nifty proxy."""
    if nifty is None or len(c) < bars:
        return 0.0
    try:
        combined = pd.concat([c.rename("s"), nifty.rename("n")], axis=1).dropna()
        if len(combined) < bars:
            return 0.0
        s = (float(combined["s"].iloc[-1]) / float(combined["s"].iloc[-bars]) - 1) * 100
        n = (float(combined["n"].iloc[-1]) / float(combined["n"].iloc[-bars]) - 1) * 100
        return round(s - n, 2)
    except Exception:
        return 0.0


# ── Core scoring ───────────────────────────────────────────────────────────────

def _score_stock(sym: str, df: pd.DataFrame, fund: dict,
                 nifty: Optional[pd.Series],
                 universe_r3m: dict[str, float]) -> Optional[dict]:
    """
    Score one stock for Early Growth. Returns None if it fails hard filters.

    Stage acceptance:
      Stage 1  → forming base (best — still before the big move)
      Stage 2 + MA150 cross ≤ MAX_CROSS_BARS bars ago → early breakout (still actionable)
      Stage 3 / 4 → too late / wrong direction → skip
      Stage 0 → insufficient data → skip
    """
    try:
        c   = df["Close"].dropna()
        h   = df["High"].dropna()
        lo  = df["Low"].dropna()
        v   = df["Volume"].dropna()
        if len(c) < MIN_BARS:
            return None

        cur = _safe(c.iloc[-1])
        if cur <= 0:
            return None

        # ── Fundamental checks ───────────────────────────────────────────────
        if not fund:
            return None

        # Raw values preserve None vs 0.0 distinction — `_safe()` would coerce
        # both to 0.0, making "genuine 0% growth" indistinguishable from
        # "missing data" and over-promoting stagnant companies via the
        # higher 3Y CAGR fallback. Same fix that monster_growth already has.
        profit_ttm_raw = fund.get("growth_ttm")
        profit_3y_raw  = fund.get("growth_3y_cagr")
        profit_yoy_raw = fund.get("eps_growth_yoy")
        if profit_ttm_raw is not None:
            profit_gr = float(profit_ttm_raw)
        elif profit_3y_raw is not None:
            profit_gr = float(profit_3y_raw)
        elif profit_yoy_raw is not None:
            profit_gr = float(profit_yoy_raw)
        else:
            profit_gr = 0.0
        profit_ttm = _safe(profit_ttm_raw)
        profit_3y  = _safe(profit_3y_raw)
        profit_yoy = _safe(profit_yoy_raw)

        sales_ttm_raw = fund.get("sales_growth_ttm")
        sales_3y_raw  = fund.get("sales_growth_3y_cagr")
        sales_yoy_raw = fund.get("sales_growth_yoy")
        if sales_ttm_raw is not None:
            sales_gr = float(sales_ttm_raw)
        elif sales_3y_raw is not None:
            sales_gr = float(sales_3y_raw)
        elif sales_yoy_raw is not None:
            sales_gr = float(sales_yoy_raw)
        else:
            sales_gr = 0.0
        sales_ttm = _safe(sales_ttm_raw)
        sales_3y  = _safe(sales_3y_raw)
        sales_yoy = _safe(sales_yoy_raw)

        pe          = _safe(fund.get("pe_ratio"))
        roe         = _safe(fund.get("roe"))
        market_cap  = _safe(fund.get("market_cap"))   # ₹ Crore
        promoter    = _safe(fund.get("promoter_holding"))
        prom_delta  = fund.get("promoter_delta")      # may be None
        eps_accel   = fund.get("eps_accel")           # QoQ (0/1/None)
        eps_accel_y = fund.get("eps_accel_yoy")       # YoY (0/1/None)
        eps_q1      = _safe(fund.get("eps_q1"))       # most recent quarter profit
        eps_q2      = _safe(fund.get("eps_q2"))
        eps_q3      = _safe(fund.get("eps_q3"))

        # ── Hard filters ─────────────────────────────────────────────────────
        if profit_gr < MIN_PROFIT_PCT:
            return None
        # Data sanity: screener.in returns garbage for recovery cases
        # (e.g. eps -1 → +100 = 10,000% growth). Cap to avoid false positives.
        if profit_gr > MAX_PROFIT_PCT:
            return None
        # SUSTAINED-ONLY filter: require non-negative 3-year CAGR to exclude
        # turnaround plays (recent burst masking multi-year decline).
        # profit_3y_raw is None when screener.in didn't show the 3y column.
        if MIN_3Y_CAGR is not None:
            if profit_3y_raw is None:
                return None   # can't confirm sustained — exclude
            if profit_3y_raw < MIN_3Y_CAGR:
                return None   # negative 3y = turnaround, not sustained
        # PE sanity: must be > 0 (positive earnings), < MAX_PE (no bubbles)
        if pe < MIN_PE or pe > MAX_PE:
            return None
        if market_cap > 0 and market_cap > MAX_MKTCAP_CR:
            return None   # too big — monstrous moves are unlikely from mega caps

        # ── Earnings acceleration hard filter ─────────────────────────────
        # Only include stocks with confirmed non-seasonal EPS acceleration.
        # eps_accel==1 means 2/3 recent quarters beat same quarter prior year.
        # If eps_q1 is None (data not yet scraped), skip — can't confirm.
        eps_q1_val = fund.get("eps_q1")
        eps_accel_val = fund.get("eps_accel")
        if eps_q1_val is None:
            return None   # quarterly data not yet scraped — skip
        if eps_accel_val != 1:
            return None   # confirmed non-seasonal acceleration required

        # ── Stage filter ─────────────────────────────────────────────────────
        stage, ma150_slope = _stage_signal(c)

        if stage == 0 or stage == 4 or stage == 3:
            return None   # insufficient data / declining / topping → skip

        cross_bars  = None
        is_stage1   = (stage == 1)
        is_early_s2 = False

        if stage == 2:
            cross_bars = _bars_since_ma150_cross(c)
            if cross_bars is None:
                return None    # Stage 2 but cross was > MAX_CROSS_BARS ago → too late
            is_early_s2 = True

        # ── Base analysis ─────────────────────────────────────────────────────
        base = _base_analysis(c, h, lo, v)

        if base["base_weeks"] < MIN_BASE_WEEKS:
            return None
        if base["base_depth_pct"] > MAX_BASE_DEPTH:
            return None

        # ── Factor 1: Base Quality (35 pts) ──────────────────────────────────
        # Base length — longer base = more supply absorbed = bigger coil
        bw = base["base_weeks"]
        if bw >= 52:    f1_len = 17
        elif bw >= 26:  f1_len = 14
        elif bw >= 16:  f1_len = 11
        elif bw >= 12:  f1_len = 8
        elif bw >= 8:   f1_len = 5
        else:           f1_len = 2

        # Base tightness — depth < 15% = very controlled = institutional
        # Indian markets tend to have wider bases than US, so 25-35% is still acceptable
        bd = base["base_depth_pct"]
        if bd <= 10:    f1_tight = 8
        elif bd <= 15:  f1_tight = 7
        elif bd <= 20:  f1_tight = 6
        elif bd <= 25:  f1_tight = 4
        elif bd <= 30:  f1_tight = 2
        elif bd <= 35:  f1_tight = 1   # acceptable for small-caps
        else:           f1_tight = 0

        # Volume drying up during base = sellers exhausted
        f1_vol  = 6 if base["vol_declining"] else 0

        # Near base high = coiled spring, ready
        f1_near = 4 if base["near_high"] else (2 if base["pos_in_base"] >= 0.6 else 0)

        f1 = f1_len + f1_tight + f1_vol + f1_near   # max 35

        # ── Factor 2: Earnings Acceleration (25 pts) ──────────────────────────
        # This is the KEY differentiator vs early_mover_scanner
        f2 = 0

        # eps_accel is already confirmed == 1 by hard filter above (10 pts guaranteed)
        f2 += 10

        # YoY acceleration: most recent Q > same Q one year ago (removes seasonality)
        if eps_accel_y == 1:
            f2 += 10
        elif eps_accel_y == 0:
            f2 += 0
        else:
            f2 += 3

        # Profit growth magnitude (but not too extreme — base effect distorts >300%)
        capped_pg = min(profit_gr, 300)
        if capped_pg >= 50:    f2 += 5
        elif capped_pg >= 30:  f2 += 4
        elif capped_pg >= 20:  f2 += 2
        else:                  f2 += 0

        # Margin signal: q1 > q2 > q3 (most recent first) = expanding margins
        if eps_q1 > 0 and eps_q2 > 0 and eps_q3 > 0:
            if eps_q1 > eps_q2 and eps_q2 > eps_q3:
                # Compute margin expansion proxy (latest Q profit vs 3 quarters ago)
                margin_expand = (eps_q1 / max(eps_q3, 0.001) - 1) * 100
                if margin_expand >= 50:
                    f2 = min(f2 + 3, 25)   # significant operating leverage

        # ── Factor 3: Market Cap — Stage of opportunity (20 pts) ─────────────
        # Graduated: tiny caps have biggest potential, but ≤8000Cr still meaningful
        # F3: smaller cap = bigger upside potential. Rescaled for MAX_MKTCAP=50000 Cr.
        # Tiny micro caps can 10×, mega caps rarely 2×.
        f3 = 0
        if market_cap > 0:
            if market_cap <= 300:    f3 = 20   # micro — 10× potential
            elif market_cap <= 750:  f3 = 18
            elif market_cap <= 1500: f3 = 16
            elif market_cap <= 3000: f3 = 14
            elif market_cap <= 5000: f3 = 12
            elif market_cap <= 8000: f3 = 10   # small-cap
            elif market_cap <= 15000:f3 = 8    # mid-cap
            elif market_cap <= 30000:f3 = 6    # upper-mid
            elif market_cap <= 50000:f3 = 4    # small-large (still room to double)
            else:                    f3 = 0    # blocked by hard filter
        else:
            f3 = 10   # unknown cap → assume mid-cap eligible

        # ── Factor 4: Smart Money / Promoter (15 pts) ────────────────────────
        f4 = 0

        # Promoter increasing stake = best inside signal
        if prom_delta is not None:
            pd_val = _safe(prom_delta, default=None)
            if pd_val is not None:
                if pd_val > 1.0:    f4 += 8   # buying aggressively
                elif pd_val > 0.0:  f4 += 5   # buying modestly
                elif pd_val < -2.0: f4 -= 2   # selling = caution

        # High promoter holding = confident management
        if promoter >= 65:  f4 += 4
        elif promoter >= 50: f4 += 2

        # ROE signal: profitable business even when small
        if roe >= 20:       f4 += 3
        elif roe >= 15:     f4 += 2
        elif roe >= 10:     f4 += 1

        # Delivery % rising during base = quiet accumulation
        deliv_chg = _delivery_trend(df)
        deliv_bonus = 0
        if deliv_chg is not None:
            if deliv_chg >= 15:   deliv_bonus = 3   # delivery rising sharply = accumulation
            elif deliv_chg >= 5:  deliv_bonus = 2
            f4 = min(f4 + deliv_bonus, 15)

        # ── Factor 5: Revenue Growth Quality (5 pts) ─────────────────────────
        f5 = 0
        if sales_gr >= 25:    f5 = 5
        elif sales_gr >= 15:  f5 = 3
        elif sales_gr >= 10:  f5 = 2
        elif sales_gr >= 5:   f5 = 1

        # Bonus: revenue growth accelerating (TTM > 3Y CAGR)
        if sales_ttm and sales_3y and sales_ttm > sales_3y * 1.2:
            f5 = min(f5 + 2, 5)

        # ── Total ─────────────────────────────────────────────────────────────
        total = max(0, min(100, f1 + f2 + f3 + f4 + f5))

        if total < 38:
            return None

        tier = ("EARLY MONSTER" if total >= 75
                else "WATCH"    if total >= 60
                else "TRACK"    if total >= 45
                else "MONITOR")

        # PEG
        peg = None
        if pe > 0 and profit_gr > 0:
            raw = pe / profit_gr
            if raw > 0:
                peg = round(raw, 3) if raw < 0.1 else round(raw, 2)

        # 52W high distance
        hi_window = c.iloc[-252:] if len(c) >= 252 else c
        pct_from_hi = round((cur / float(hi_window.max()) - 1) * 100, 2) if float(hi_window.max()) > 0 else 0.0

        # RS rank (cross-sectional)
        rs_rank = 50
        if universe_r3m:
            sym_r3m  = (float(c.iloc[-1]) / float(c.iloc[-63]) - 1) * 100 if len(c) >= 63 else 0.0
            u_series = pd.Series(universe_r3m)
            rs_rank  = int((u_series < sym_r3m).sum() / max(len(u_series), 1) * 99)

        adtv_cr = round(float(v.rolling(20).mean().iloc[-1] if len(v) >= 20 else 0) * cur / 1e7, 2)

        return {
            "symbol":          sym,
            "score":           total,
            "tier":            tier,
            # Stage context
            "stage":           stage,
            "is_stage1":       is_stage1,
            "cross_bars":      cross_bars,      # bars since MA150 breakout (None=Stage1)
            "ma150_slope":     round(ma150_slope, 2),
            # Base
            "base_weeks":      base["base_weeks"],
            "base_depth_pct":  base["base_depth_pct"],
            "vol_declining":   base["vol_declining"],
            "near_high":       base["near_high"],
            "pos_in_base":     base["pos_in_base"],
            "base_high":       base["base_high"],
            "base_low":        base["base_low"],
            # Fundamentals
            "profit_gr":       round(profit_gr, 1),
            "sales_gr":        round(sales_gr, 1),
            "pe":              round(pe, 1),
            "peg":             peg,
            "roe":             round(roe, 1),
            "market_cap":      round(market_cap, 0) if market_cap > 0 else None,
            "eps_accel":       eps_accel,
            "eps_accel_yoy":   eps_accel_y,
            "promoter_pct":    round(promoter, 1),
            "promoter_delta":  round(_safe(prom_delta, default=0.0), 2) if prom_delta is not None else None,
            "deliv_chg":       deliv_chg,
            # Factor breakdown
            "f1_base":         f1,
            "f2_accel":        f2,
            "f3_mktcap":       f3,
            "f4_smart":        f4,
            "f5_revenue":      f5,
            # Technical
            "price":           round(cur, 2),
            "rs_rank":         rs_rank,
            "pct_from_hi":     pct_from_hi,
            "adtv_cr":         adtv_cr,
        }
    except Exception:
        return None


# ── Main entry ─────────────────────────────────────────────────────────────────

def run_early_growth_scan(progress_callback=None) -> dict:
    """
    Full Early Growth scan — Nifty Total Market 750 universe, all local data.
    Returns dict with results list + metadata.
    """
    with _cache_lock:
        if (_cache["data"]
                and (time.time() - _cache["ts"]) < CACHE_TTL
                and _cache["data"].get("results")):
            return _cache["data"]

    if progress_callback:
        progress_callback(0, 100, "Loading OHLCV + DelivPer from bhavcopy cache…")

    stocks = _load_universe(progress_callback)
    if not stocks:
        return {
            "results": [], "computed_at": int(time.time()), "total_scanned": 0,
            "error": "No OHLCV data — bhavcopy not yet downloaded",
        }

    if progress_callback:
        progress_callback(60, 100, "Loading fundamentals from local DB…")
    all_funds = load_all_fundamentals()

    nifty = _build_nifty(stocks)

    # Cross-sectional RS ranking denominator
    universe_r3m: dict[str, float] = {}
    for sym, df in stocks.items():
        c = df["Close"].dropna()
        if len(c) >= 63:
            universe_r3m[sym] = (float(c.iloc[-1]) / float(c.iloc[-63]) - 1) * 100

    if progress_callback:
        progress_callback(70, 100, f"Scoring {len(stocks)} stocks for early growth…")

    results = []
    for sym, df in stocks.items():
        fund = all_funds.get(sym)
        if not fund:
            continue
        r = _score_stock(sym, df, fund, nifty, universe_r3m)
        if r is not None:
            results.append(r)

    # Sort: Early Monster first, then by score desc, then by base_weeks desc
    results.sort(key=lambda x: (-x["score"], -x["base_weeks"]))

    s1_count  = sum(1 for r in results if r["is_stage1"])
    es2_count = sum(1 for r in results if not r["is_stage1"])
    em_count  = sum(1 for r in results if r["tier"] == "EARLY MONSTER")
    w_count   = sum(1 for r in results if r["tier"] == "WATCH")
    t_count   = sum(1 for r in results if r["tier"] == "TRACK")
    m_count   = sum(1 for r in results if r["tier"] == "MONITOR")

    # P2-14: update stage transition log
    try:
        from stage_transitions import update_all as _stg_upd
        from analysis_utils import stage_analysis as _stg
        stg_map = {}
        for r in results:
            sym = r.get("symbol")
            if sym and sym in stocks:
                s = _stg(stocks[sym]["Close"].dropna())
                if s in (1, 2, 3, 4):
                    stg_map[sym] = s
        _stg_upd(stg_map)
    except Exception:
        pass

    out = {
        "results":              results,
        "computed_at":          int(time.time()),
        "total_scanned":        len(stocks),
        "fundamental_coverage": sum(1 for s in stocks if s in all_funds),
        "eps_data_coverage":    sum(1 for f in all_funds.values() if f.get("eps_q1") is not None),
        "eps_data_total":       len(all_funds),
        "stage1_count":         s1_count,
        "early_s2_count":       es2_count,
        "early_monster_count":  em_count,
        "watch_count":          w_count,
        "track_count":          t_count,
        "monitor_count":        m_count,
    }

    with _cache_lock:
        _cache["data"] = out
        _cache["ts"]   = time.time()

    # P2-10/11: enrich AFTER cache is set so Early Growth's own results are
    # visible to build_consensus when it scans every scanner's _cache.
    try:
        from consensus import enrich_results, invalidate_cache as _con_inv
        _con_inv()
        enrich_results(results)
    except Exception:
        pass

    if progress_callback:
        progress_callback(100, 100,
            f"Done — {len(results)} early-phase stocks "
            f"({em_count} Early Monster, {w_count} Watch, {t_count} Track, {m_count} Monitor) "
            f"| {s1_count} in Stage 1 base, {es2_count} just broke out")
    return out


def invalidate_cache():
    """Bust cache when new bhavcopy arrives."""
    with _cache_lock:
        _cache["data"] = None
        _cache["ts"]   = 0.0
