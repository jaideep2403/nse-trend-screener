"""
Trending Stocks Scanner — Enhanced with 10 features
=====================================================
Scans the full Nifty Total Market 750 universe for stocks in a clean, sustained uptrend.
Uses only local bhavcopy OHLCV + Delivery data — no external API calls.

Score (0–10): 4 original MA/RS criteria + 4 new binary signals
  1.  Price > MA20 & MA50          — short-term trend intact
  2.  MA20 > MA50                  — momentum alignment
  3.  MA50 slope rising            — medium-term trend climbing
  4.  Within 20% of 52W high       — near highs, not broken
  5.  Beating Nifty 1M             — outperforming index short-term
  6.  Beating Nifty 3M             — outperforming index medium-term
  7.  Volume expanding             — OBV rising OR vol > 20-day avg
  8.  ADX ≥ 25                     — strong directional trend
  9.  Higher Highs + Higher Lows   — price structure is bullish
  10. MA200 aligned                — price above long-term average

Display-only extras (not in score):
  - New 52W High flag (past 5 sessions)
  - Trend Age (days continuously above MA50)
  - Pullback at MA20 support (buy-the-dip flag)
  - Avg Delivery %
  - Trend Consistency R²

Plus: Sector RS Heatmap (separate panel in UI).
"""

import time
import numpy as np
import pandas as pd
from industry_groups import _get_stocks, _build_nifty, INDUSTRY_GROUPS, _group_rs
from nse_stocks import get_universe_symbols

try:
    from sector_mapper import get_enriched_sector_map as _enriched_sector_map, refresh_sector_cache
    _MAPPER_OK = True
except ImportError:
    _MAPPER_OK = False

_cache    = {"data": None, "ts": 0}
CACHE_TTL = 1800   # 30 min
TREND_MIN_SCORE = 5   # show stocks with at least 5/10


# ── Sector map ────────────────────────────────────────────────────────────────

def _sector_map():
    """Returns {symbol: sector} for all 751 stocks (INDUSTRY_GROUPS + NSE auto-mapped extras)."""
    if _MAPPER_OK:
        return _enriched_sector_map()
    return {s: g for g, syms in INDUSTRY_GROUPS.items() for s in syms}


# ── Split/corporate-action cleaner ────────────────────────────────────────────

def _clean_df(df):
    """
    Detect and strip pre-corporate-action (split/bonus/demerger) price history.
    A >35% single-day price drop is treated as a split event.
    Aligns all columns to the same clean index.
    """
    c = df["Close"].dropna()
    if len(c) < 2:
        return df
    pct_chg   = c.pct_change().abs()
    anomalies = pct_chg[pct_chg > 0.35].index
    if len(anomalies) == 0:
        return df
    last_event = anomalies[-1]
    clean = df[df.index > last_event]
    return clean if len(clean) >= 30 else df


# ── Feature 2: ADX (Average Directional Index) ───────────────────────────────

def _adx(high, low, close, period=14):
    """
    Compute ADX from numpy arrays. Returns ADX value or None if insufficient data.
    ADX ≥ 25 = trending. ADX ≥ 40 = very strong trend.
    """
    n = len(close)
    if n < period * 2 + 1:
        return None

    tr_arr, pdm_arr, mdm_arr = [], [], []
    for i in range(1, n):
        h, l, pc = float(high[i]), float(low[i]), float(close[i - 1])
        ph, pl   = float(high[i - 1]), float(low[i - 1])
        tr  = max(h - l, abs(h - pc), abs(l - pc))
        up  = h - ph
        dn  = pl - l
        pdm = up if up > dn and up > 0 else 0.0
        mdm = dn if dn > up and dn > 0 else 0.0
        tr_arr.append(tr); pdm_arr.append(pdm); mdm_arr.append(mdm)

    # Wilder smoothing (same as EMA with alpha = 1/period)
    def wilder(arr):
        s = [sum(arr[:period])]
        for v in arr[period:]:
            s.append(s[-1] - s[-1] / period + v)
        return s

    atr  = wilder(tr_arr)
    pDMs = wilder(pdm_arr)
    mDMs = wilder(mdm_arr)

    dx_list = []
    for i in range(len(atr)):
        a = atr[i]
        if a == 0:
            dx_list.append(0.0)
            continue
        pDI = 100 * pDMs[i] / a
        mDI = 100 * mDMs[i] / a
        s   = pDI + mDI
        dx_list.append(100 * abs(pDI - mDI) / s if s > 0 else 0.0)

    # BUG-023 FIX: ADX is biased high right after the Wilder-seed window
    # because the first `period` DX values use partially seeded ATR/DI.
    # Skip the first `period` DX values so the seed mean is computed on
    # fully-smoothed bars, matching the standard Wilder ADX definition.
    if len(dx_list) < 2 * period:
        return None
    seed_slice = dx_list[period:2 * period]
    adx = sum(seed_slice) / period
    for v in dx_list[2 * period:]:
        adx = (adx * (period - 1) + v) / period
    return round(adx, 1)


# ── Feature 3: New 52W High flag ─────────────────────────────────────────────

def _new_52w_high(c):
    """
    Returns (flag, days_ago) where flag=True if price hit a new 52W high
    in the last 5 sessions.
    """
    n   = len(c)
    w52 = c.iloc[-252:] if n >= 252 else c
    h52 = float(w52.max())
    for offset in range(1, 6):
        idx = n - offset
        if idx < 0:
            break
        if float(c.iloc[idx]) >= h52 * 0.995:
            return True, offset
    return False, None


# ── Feature 4: Trend Age ──────────────────────────────────────────────────────

def _trend_age(c):
    """Count consecutive days (from today backwards) that price was above MA50.

    BUG-FIX: prior code computed `ma50 = prices[i-50:i].mean()` which is the
    average of the PRIOR 50 bars excluding bar `i` — disagrees with the standard
    `close.rolling(50).mean()` (which INCLUDES the current bar) used elsewhere
    in the same file's `c1` ("Price > MA50") check. Off-by-one made trend-age
    sometimes contradict the trend signal in the same scan.
    """
    if len(c) < 51:
        return 0
    import pandas as _pd
    s = _pd.Series(c.values.astype(float))
    ma50 = s.rolling(50).mean()
    n = len(s)
    count = 0
    for i in range(n - 1, 48, -1):
        if _pd.isna(ma50.iloc[i]):
            break
        if s.iloc[i] > ma50.iloc[i]:
            count += 1
        else:
            break
    return count


# ── Feature 5: Pullback at MA20 support ───────────────────────────────────────

def _at_ma20_support(c):
    """
    Flag if the stock dipped within 2% of MA20 in the last 5 sessions
    while the overall trend (MA20 > MA50) is still bullish.
    """
    if len(c) < 55:
        return False
    prices = c.values.astype(float)
    n = len(prices)
    ma20_now = prices[-20:].mean()
    ma50_now = prices[-50:].mean()
    if ma20_now <= ma50_now:          # trend not bullish
        return False
    for offset in range(1, 6):
        idx = n - offset
        if idx < 19:
            break
        price = prices[idx]
        # MA20 INCLUDING the bar at idx — was prices[idx-20:idx] which is
        # the MA ending one bar before, comparing today's price to yesterday's MA.
        ma20  = prices[idx - 19:idx + 1].mean()
        if abs(price / ma20 - 1) <= 0.02:   # within 2% of MA20
            return True
    return False


# ── Feature 6: Delivery % ─────────────────────────────────────────────────────

def _avg_delivery(df, n=20):
    """Average delivery % over last n sessions. Returns None if not available."""
    if "DelivPer" not in df.columns:
        return None
    d = df["DelivPer"].dropna().iloc[-n:]
    return round(float(d.mean()), 1) if len(d) >= 5 else None


# ── Feature 7: Higher Highs + Higher Lows ────────────────────────────────────

def _higher_highs_lows(c: pd.Series, weeks: int = 10) -> bool:
    """
    Higher weekly highs AND higher weekly lows over `weeks` calendar weeks.
    60% majority required.
    TIER-3 FIX: was using artificial 5-bar slices which shift with holidays
    (a 4-day week counted as 5 bars, misaligning all subsequent windows).
    resample("W-FRI") uses actual Mon-Fri calendar boundaries so holiday
    weeks collapse correctly and the comparison is always apples-to-apples.
    """
    if not isinstance(c.index, pd.DatetimeIndex) or len(c) < weeks * 4:
        return False
    try:
        wk_hi = c.resample("W-FRI").max().dropna()
        wk_lo = c.resample("W-FRI").min().dropna()
        if len(wk_hi) < weeks:
            return False
        wk_hi = wk_hi.iloc[-weeks:]
        wk_lo = wk_lo.iloc[-weeks:]
        hh = int(sum(wk_hi.iloc[i] > wk_hi.iloc[i - 1] for i in range(1, weeks)))
        hl = int(sum(wk_lo.iloc[i] > wk_lo.iloc[i - 1] for i in range(1, weeks)))
        return hh >= int(weeks * 0.6) and hl >= int(weeks * 0.6)
    except Exception:
        return False


# ── Feature 8: Sector RS Heatmap ─────────────────────────────────────────────
# Computed separately in run_trending_scan() using _group_rs() from industry_groups


# ── Feature 9: Trend Consistency R² ──────────────────────────────────────────

def _r_squared(c, n=66):
    """
    R² of log(price) vs time over the last n days.
    1.0 = perfectly linear uptrend. 0.0 = random walk.
    """
    if len(c) < n:
        n = len(c)
    if n < 20:
        return None
    prices = np.log(c.iloc[-n:].values.astype(float))
    x = np.arange(n, dtype=float)
    coeffs = np.polyfit(x, prices, 1)
    fitted = np.polyval(coeffs, x)
    ss_res = float(((prices - fitted) ** 2).sum())
    ss_tot = float(((prices - prices.mean()) ** 2).sum())
    if ss_tot == 0:
        return 1.0
    return round(max(0.0, 1 - ss_res / ss_tot), 3)


# ── Feature 7 (volume): OBV trend + volume ratio ─────────────────────────────

def _volume_expanding(c, v, n=20):
    """
    Returns True if volume is expanding:
      - Current 5-day avg volume > 20-day avg volume (accumulation), OR
      - OBV is higher now than it was n days ago.
    """
    if v is None or len(v) < n + 1:
        return None
    vols = v.values.astype(float)
    # Volume ratio: recent 5-day vs 20-day avg
    recent_vol = vols[-5:].mean()
    avg_vol    = vols[-n:].mean()
    vol_ratio  = recent_vol / avg_vol if avg_vol > 0 else 1.0

    # OBV
    closes = c.values.astype(float)
    obv = 0.0
    obv_series = []
    for i in range(1, len(closes)):
        if closes[i] > closes[i - 1]:
            obv += vols[i]
        elif closes[i] < closes[i - 1]:
            obv -= vols[i]
        obv_series.append(obv)

    obv_rising = (len(obv_series) >= n) and (obv_series[-1] > obv_series[-n])
    return vol_ratio > 1.0 or obv_rising, round(vol_ratio, 2)


# ── Core scorer ───────────────────────────────────────────────────────────────

def _score_stock(df, nifty):
    """
    Compute all 10 features + display metrics for one stock DataFrame.
    df must already be cleaned (split-adjusted via _clean_df).
    Returns None if insufficient data (< 20 bars).
    """
    c  = df["Close"].dropna()
    v  = df["Volume"].dropna() if "Volume" in df.columns else None
    n  = len(c)
    if n < 20:
        return None

    price  = float(c.iloc[-1])
    prices = c.values.astype(float)

    # Moving averages
    ma20  = float(prices[-20:].mean())
    ma50  = float(prices[-50:].mean())  if n >= 50  else None
    ma200 = float(prices[-200:].mean()) if n >= 200 else None
    # BUG-015 FIX: MA50 slope uses 20-bar lookback (vs the previous 5-bar slice in ma50_20d).
    # Compare current MA50 to MA50 from 20 trading days ago for a stable slope signal.
    ma50_20d = float(prices[-70:-20].mean()) if n >= 70 else None  # MA50 as of 20 bars ago

    # 52W window
    w52    = c.iloc[-252:] if n >= 252 else c
    high52 = float(w52.max())
    low52  = float(w52.min())

    # Returns
    r1m = round((price / float(c.iloc[-21])  - 1) * 100, 1) if n >= 21  else None
    r3m = round((price / float(c.iloc[-63])  - 1) * 100, 1) if n >= 63  else None
    r6m = round((price / float(c.iloc[-126]) - 1) * 100, 1) if n >= 126 else None

    # RS vs Nifty
    def rs(period):
        if nifty is None or n < period:
            return None
        idx = c.index.intersection(nifty.index)
        if len(idx) < period:
            return None
        return round(
            (float(c[idx].iloc[-1]) / float(c[idx].iloc[-period]) - 1) * 100 -
            (float(nifty[idx].iloc[-1]) / float(nifty[idx].iloc[-period]) - 1) * 100,
            1)

    # Match the return windows used elsewhere (21/63/126) so the RS score
    # and the displayed return columns are computed over the SAME period.
    rs1m = rs(21); rs3m = rs(63); rs6m = rs(126)
    # BUG-016 FIX: skip None components and reweight rather than defaulting to 0.
    # If rs1m is None, use rs3m 0.6 + rs6m 0.4. If only rs3m exists, use it fully.
    if rs1m is not None and rs3m is not None and rs6m is not None:
        rsc = round(rs1m * 0.4 + rs3m * 0.4 + rs6m * 0.2, 1)
    elif rs1m is None and rs3m is not None and rs6m is not None:
        rsc = round(rs3m * 0.6 + rs6m * 0.4, 1)
    elif rs3m is not None and rs6m is None:
        rsc = round(rs3m * 1.0, 1)
    elif rs3m is None and rs6m is not None:
        rsc = round(rs6m * 1.0, 1)
    elif rs1m is not None:
        rsc = round(rs1m * 1.0, 1)
    else:
        rsc = 0.0
    pct_from_high = round((price / high52 - 1) * 100, 1)

    # ── 10 scored criteria ────────────────────────────────────────────────────

    # 1. Price > MA20 & MA50
    c1 = (price > ma20) and (ma50 is not None and price > ma50)
    # 2. MA20 > MA50
    c2 = ma50 is not None and ma20 > ma50
    # 3. MA50 slope rising
    c3 = ma50 is not None and ma50_20d is not None and ma50 > ma50_20d
    # 4. Within 20% of 52W high
    c4 = price >= high52 * 0.80
    # 5. Beating Nifty 1M
    c5 = rs1m is not None and rs1m > 0
    # 6. Beating Nifty 3M
    c6 = rs3m is not None and rs3m > 0

    # 7. Volume expanding (OBV or vol ratio)
    vol_result = _volume_expanding(c, v)
    if isinstance(vol_result, tuple):
        vol_flag, vol_ratio = vol_result
    else:
        vol_flag, vol_ratio = vol_result, None
    c7 = bool(vol_flag) if vol_flag is not None else False

    # 8. ADX ≥ 25
    adx_val = None
    if "High" in df.columns and "Low" in df.columns and n >= 30:
        h = df["High"].dropna().values.astype(float)
        l = df["Low"].dropna().values.astype(float)
        min_len = min(len(h), len(l), n)
        adx_val = _adx(h[-min_len:], l[-min_len:], prices[-min_len:])
    c8 = adx_val is not None and adx_val >= 25

    # 9. Higher Highs + Higher Lows
    c9 = _higher_highs_lows(c)

    # 10. MA200 aligned (price > MA200)
    c10 = ma200 is not None and price > ma200

    # Force all criteria to Python bool (numpy.bool_ breaks JSON serialisation)
    criteria = [bool(x) for x in [c1, c2, c3, c4, c5, c6, c7, c8, c9, c10]]
    score    = int(sum(criteria))

    # ── Display-only extras ───────────────────────────────────────────────────
    new_high_flag, new_high_days = _new_52w_high(c)
    trend_age   = int(_trend_age(c))
    at_support  = bool(_at_ma20_support(c))
    avg_deliv   = _avg_delivery(df)
    r2          = _r_squared(c)

    return {
        # Core
        "score":          score,
        "criteria":       criteria,
        "price":          round(float(price), 2),
        "ma20":           round(float(ma20), 2),
        "ma50":           round(float(ma50), 2)  if ma50  is not None else None,
        "ma200":          round(float(ma200), 2) if ma200 is not None else None,
        "high52":         round(float(high52), 2),
        "low52":          round(float(low52), 2),
        "pct_from_high":  float(pct_from_high),
        # Returns
        "r1m": float(r1m) if r1m is not None else None,
        "r3m": float(r3m) if r3m is not None else None,
        "r6m": float(r6m) if r6m is not None else None,
        # RS
        "rs1m": float(rs1m) if rs1m is not None else None,
        "rs3m": float(rs3m) if rs3m is not None else None,
        "rs6m": float(rs6m) if rs6m is not None else None,
        "rs_composite": float(rsc),
        # New features
        "adx":            float(adx_val) if adx_val is not None else None,
        "vol_ratio":      float(vol_ratio) if vol_ratio is not None else None,
        "r_squared":      float(r2) if r2 is not None else None,
        "avg_deliv":      float(avg_deliv) if avg_deliv is not None else None,
        "trend_age":      trend_age,
        "new_high_flag":  bool(new_high_flag),
        "new_high_days":  int(new_high_days) if new_high_days is not None else None,
        "at_support":     at_support,
        "hh_hl":          bool(c9),
        "data_days":      int(n),
    }


# ── Main scan ─────────────────────────────────────────────────────────────────

def run_trending_scan(progress_callback=None):
    global _cache
    if _cache["data"] and time.time() - _cache["ts"] < CACHE_TTL:
        return _cache["data"]

    if progress_callback:
        progress_callback(0, 100, "Loading market data…")

    stocks   = _get_stocks()
    nifty    = _build_nifty(stocks)
    universe = set(get_universe_symbols())  # Nifty Total Market 750
    sec_map  = _sector_map()

    if not stocks:
        return {"error": "No bhavcopy data. Run any other scan first."}

    nifty_3m = None
    if nifty is not None and len(nifty) >= 66:
        nifty_3m = round((float(nifty.iloc[-1]) / float(nifty.iloc[-63]) - 1) * 100, 1)

    results = []
    syms    = sorted(s for s in stocks if s in universe)
    total   = len(syms)

    for i, sym in enumerate(syms):
        if progress_callback and i % 50 == 0:
            progress_callback(i, total, f"Scanning {sym}… {i}/{total}")

        df  = _clean_df(stocks[sym].copy())
        m   = _score_stock(df, nifty)
        if m is None or m["score"] < TREND_MIN_SCORE:
            continue

        m["symbol"] = sym
        m["sector"] = sec_map.get(sym, "Other")
        results.append(m)

    if progress_callback:
        progress_callback(total, total, f"Done — {len(results)} trending stocks found")

    # Sort: score desc → rs_composite desc
    results.sort(key=lambda x: (-x["score"], -(x["rs_composite"] or -999)))
    for i, r in enumerate(results):
        r["rank"] = i + 1

    # Sector RS heatmap
    sector_rs = _group_rs(stocks, nifty) if nifty is not None else []

    try:
        from data_fetcher import _latest_bhavcopy_date as _lbd
        _bd = _lbd()
        bhavcopy_date = str(_bd) if _bd else None
    except Exception:
        bhavcopy_date = None

    result = {
        "stocks":          results,
        "universe_count":  total,
        "trending_count":  len(results),
        "nifty_3m":        nifty_3m,
        "computed_at":     time.time(),
        "bhavcopy_date":   bhavcopy_date,
        "sector_rs":       sector_rs,
        "criteria_labels": [
            "Price > MA20 & MA50",
            "MA20 > MA50",
            "MA50 Slope ↑",
            "Within 20% of 52W High",
            "Beating Nifty 1M",
            "Beating Nifty 3M",
            "Volume Expanding",
            "ADX ≥ 25",
            "Higher Highs & Lows",
            "Price > MA200",
        ],
    }

    _cache.update({"data": result, "ts": time.time()})
    return result
