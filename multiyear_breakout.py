"""
Multi-Year Breakout Scanner
============================
Detects stocks breaking above multi-year resistance after a prolonged base.

Data source: NSE bhavcopy archive (works back to Dec 2019, ~6 years).
We sample one bhavcopy per month (last trading day) to build a long-history
monthly OHLCV time series — much faster than downloading daily files for 6 years
(72 files vs ~1500), and reuses the already-tested `_download_one_day()`
infrastructure with proper cookie seeding. ZERO yfinance dependency.

Algorithm:
  For each lookback window N in [1, 2, 3, 5] years:
    - resistance = max(monthly_high[ -N*12 - 6 : -6 ])   ← prior N-yr high
    - breakout window = last 6 months
    - if current_close > resistance AND was below 6 months ago:
        - base_pct = fraction of lookback months that closed BELOW resistance
        - if base_pct >= 0.50 → valid base
    - pick the longest valid base window
"""

import os
import time
import pickle
import logging
import warnings
import calendar
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import date, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed

from data_fetcher import _download_one_day, _bhav_cache_path, BHAV_DIR

warnings.filterwarnings("ignore")
log = logging.getLogger(__name__)

# ── Directories ───────────────────────────────────────────────────────────────
DATA_DIR       = Path(os.getenv("DATA_DIR", "/tmp"))
MULTIYEAR_DIR  = DATA_DIR / "multiyear_cache"
MULTIYEAR_DIR.mkdir(parents=True, exist_ok=True)

CACHE_TTL          = 7 * 86400      # 7 days for per-stock monthly cache
SCAN_CACHE_TTL     = 6 * 3600       # 6 hours for full scan results
MIN_MONTHS         = 12             # need ≥12 monthly bars for any breakout detection
DOWNLOAD_WORKERS   = 10             # matches data_fetcher norm; >10 risks NSE anti-bot wall on cold /tmp.
                                    # (Real speedup is the single-groupby monthly resample, not worker count.)
HISTORY_YEARS      = 6              # how far back to fetch (NSE archive limit ~Dec 2019)
BREAKOUT_WINDOW    = 6              # consider breakouts in last N months
BASE_PCT_THRESHOLD = 0.50           # ≥50% of lookback months must be below resistance

# Base duration windows (months, label)
WINDOWS = [
    (60,  "5yr+"),
    (36,  "3yr+"),
    (24,  "2yr+"),
    (12,  "1yr+"),
]

_scan_cache      = {"data": None, "ts": 0}
_near_scan_cache = {"data": None, "ts": 0}


# ── Split / Bonus backward-adjustment (BUG-003) ───────────────────────────────

def _adjust_for_splits(df):
    """Delegate to canonical analysis_utils.adjust_for_splits."""
    from analysis_utils import adjust_for_splits
    return adjust_for_splits(df)


# ── Monthly sample dates ──────────────────────────────────────────────────────

def _month_sample_dates(years_back: int = HISTORY_YEARS) -> list[date]:
    """
    BUG-FIX: was returning ONE weekday per month (~72 dates for 6 years).
    Then resample("ME") on those single-day samples gave monthly High = monthly
    Low = monthly Close (because each "month" had only 1 bar). So multi-year
    resistance levels were just month-end close prices — NOT actual intra-month
    peaks. A stock that printed ₹1,200 intra-month but closed ₹1,050 had its
    "resistance" recorded as ₹1,050; today at ₹1,080 fired a false breakout.

    Now: return EVERY weekday in the history window. _download_one_day caches
    each bhavcopy locally, so most calls are cache hits after first run.
    Resampled monthly High = max of all daily Highs that fell in that month.
    """
    today = date.today()
    dates: list[date] = []
    cutoff = today - timedelta(days=years_back * 365 + 30)
    d = today
    while d >= cutoff:
        if d.weekday() < 5:    # Mon-Fri only
            dates.append(d)
        d -= timedelta(days=1)
    return dates


# ── Download monthly bhavcopy samples ─────────────────────────────────────────

def _download_monthly_samples(progress_callback=None) -> list[pd.DataFrame]:
    """
    Download bhavcopy for the last trading day of each month for HISTORY_YEARS.
    If a date is a holiday (returns None), walk back up to 5 days.
    """
    target_dates = _month_sample_dates(HISTORY_YEARS)
    total = len(target_dates)
    frames = []
    done = [0]

    def _fetch(target_dt: date) -> pd.DataFrame | None:
        # Try target date, then walk backwards up to 5 days for holiday/missing
        d = target_dt
        for _ in range(6):
            df = _download_one_day(d)
            if df is not None and len(df) > 100:
                return df
            d -= timedelta(days=1)
        return None

    if progress_callback:
        progress_callback(0, total, f"Fetching {total} monthly bhavcopy samples from NSE archives…")

    with ThreadPoolExecutor(max_workers=DOWNLOAD_WORKERS) as ex:
        futs = [ex.submit(_fetch, dt) for dt in target_dates]
        for fut in as_completed(futs):
            df = fut.result()
            done[0] += 1
            if df is not None:
                frames.append(df)
            if progress_callback and done[0] % 5 == 0:
                progress_callback(done[0], total,
                                  f"Downloaded {done[0]}/{total} months · {len(frames)} successful")

    if progress_callback:
        progress_callback(total, total,
                          f"Downloaded {len(frames)}/{total} monthly bhavcopy files ✓")
    return frames


# ── Build per-stock monthly OHLCV ─────────────────────────────────────────────

def _build_monthly_ohlcv(frames: list[pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """
    Combine bhavcopy frames into per-stock monthly OHLCV.

    BUG-002 FIX: instead of treating each bhavcopy sample as its own
    "month bar" (which gave a single-day snapshot, not a real monthly High/Low),
    we resample whatever daily data is available with `.resample("M").agg(...)`
    so monthly High = max of all daily Highs that fell in the month, etc.

    BUG-003 FIX: backward-adjust for stock splits/bonuses before resampling so
    pre-split prices don't corrupt the monthly aggregation.

    PERF: replaced 2637 per-symbol resample() calls with a single vectorized
    groupby(['Symbol','YM']).agg(...) after collecting all adjusted daily frames
    into one DataFrame.  Saves ~40% of build time vs the old loop.
    """
    if not frames:
        return {}

    combined = pd.concat(frames, ignore_index=True)
    combined["Date"] = pd.to_datetime(combined["Date"])
    combined = combined.sort_values(["Symbol", "Date"]).drop_duplicates(
        subset=["Symbol", "Date"], keep="last"
    )

    # Per-symbol: backward-adjust for splits, then collect into a single list.
    # adjust_for_splits must run on DAILY data (split events appear as overnight
    # drops that may span two calendar months and would be invisible in monthly bars).
    all_daily: list[pd.DataFrame] = []
    for sym, g in combined.groupby("Symbol", sort=False):
        if len(g) < MIN_MONTHS:
            continue
        daily = g.set_index("Date")[["Open", "High", "Low", "Close", "Volume"]].copy()
        daily = _adjust_for_splits(daily)
        daily["Symbol"] = sym
        all_daily.append(daily)

    if not all_daily:
        return {}

    # Single vectorized monthly aggregation across all symbols at once.
    big = pd.concat(all_daily).reset_index()
    big["YM"] = big["Date"].dt.to_period("M")
    monthly_all = big.groupby(["Symbol", "YM"], sort=False).agg(
        Open=("Open",   "first"),
        High=("High",   "max"),
        Low=("Low",     "min"),
        Close=("Close", "last"),
        Volume=("Volume","sum"),
    )

    # Fast conversion to {symbol: monthly_df} dict.
    monthly_reset = monthly_all.reset_index()
    monthly_reset["Date"] = monthly_reset["YM"].dt.to_timestamp(how="end")
    monthly_reset = (
        monthly_reset.drop(columns=["YM"])
        .dropna(subset=["Close"])
        .set_index("Date")
    )

    out: dict[str, pd.DataFrame] = {}
    for sym, g in monthly_reset.groupby("Symbol", sort=False):
        g2 = g.drop(columns=["Symbol"]).sort_index()
        if len(g2) >= MIN_MONTHS:
            out[sym] = g2
    return out


def _get_or_build_monthly_data(progress_callback=None) -> dict[str, pd.DataFrame]:
    """
    Top-level: returns {symbol: monthly_df} for all stocks.
    Uses a single combined-cache file (rebuilt every 7 days).
    """
    combined_cache = MULTIYEAR_DIR / "_all_monthly.pkl"

    # Try cache
    if combined_cache.exists():
        age = time.time() - combined_cache.stat().st_mtime
        if age < CACHE_TTL:
            try:
                with open(combined_cache, "rb") as f:
                    cached = pickle.load(f)
                if progress_callback:
                    progress_callback(100, 100,
                                      f"Loaded {len(cached)} stocks from monthly cache ⚡")
                return cached
            except Exception:
                pass

    # Build fresh
    frames = _download_monthly_samples(progress_callback)
    if not frames:
        return {}

    data = _build_monthly_ohlcv(frames)

    try:
        with open(combined_cache, "wb") as f:
            pickle.dump(data, f)
    except Exception:
        pass

    return data


# ── Breakout detection ────────────────────────────────────────────────────────

def detect_multiyear_breakout(symbol: str, monthly: pd.DataFrame) -> dict | None:
    """
    Return breakout dict if the stock has broken above a multi-year resistance
    within the last BREAKOUT_WINDOW months — else None.
    """
    if monthly is None or len(monthly) < MIN_MONTHS:
        return None

    closes = monthly["Close"].values.astype(float)
    highs  = monthly["High"].values.astype(float)
    vols   = monthly["Volume"].values.astype(float)

    n = len(closes)
    current_close = closes[-1]

    # Volume ratio: last 1 month vs 12-month avg
    if n >= 13:
        vol_avg = np.mean(vols[-13:-1])
    else:
        vol_avg = np.mean(vols[:-1]) if n > 1 else vols[-1]
    vol_ratio = (vols[-1] / vol_avg) if vol_avg > 0 else 1.0

    best = None
    best_window = 0

    for win_months, win_label in WINDOWS:
        # Need: win_months lookback + BREAKOUT_WINDOW recent months + 1 current
        required = win_months + BREAKOUT_WINDOW + 1
        if n < required:
            continue

        # Resistance = max high in lookback (excluding the recent breakout window)
        lookback_highs  = highs[-(win_months + BREAKOUT_WINDOW): -BREAKOUT_WINDOW]
        lookback_closes = closes[-(win_months + BREAKOUT_WINDOW): -BREAKOUT_WINDOW]
        resistance      = float(np.max(lookback_highs))

        if resistance <= 0:
            continue
        if current_close <= resistance:
            continue

        # Breakout must be RECENT — was below resistance at start of breakout window
        close_window_start = closes[-BREAKOUT_WINDOW]
        if close_window_start > resistance:
            continue   # already above before breakout window — old breakout, skip

        # Base quality check
        below_count = int(np.sum(lookback_closes < resistance))
        base_pct = below_count / max(len(lookback_closes), 1)
        if base_pct < BASE_PCT_THRESHOLD:
            continue

        # When did the breakout actually happen? (count months above resistance from end)
        months_above = 0
        for i in range(n - 1, -1, -1):
            if closes[i] > resistance:
                months_above += 1
            else:
                break
        if months_above == 0 or months_above > BREAKOUT_WINDOW:
            continue

        pct_above   = (current_close / resistance - 1) * 100
        # Months spent in the base BEFORE the breakout. The base ends `months_above`
        # months before the latest sample, so the base count is `n - months_above`
        # (not `n - 1 - months_above` — that was an extra -1 shift that under-counted
        # the base length by one month, demoting genuine long-base breakouts in the
        # tier ranking).
        base_months = n - months_above

        # ATH check
        is_ath = bool(current_close >= np.max(closes[:-1]))

        # ── Measured-move target (base-height projection) ──────────────────
        # Classic TA: after breakout, price tends to travel at least the height of the base.
        # base_low  = support level during base period
        # target    = resistance + (resistance - base_low)
        base_low        = float(np.min(lookback_closes))
        base_height     = resistance - base_low
        measured_target = resistance + base_height
        remaining_upside_pct = (measured_target / current_close - 1) * 100

        result = {
            "symbol":              symbol,
            "cmp":                 round(current_close, 2),
            "resistance":          round(resistance, 2),
            "pct_above":           round(pct_above, 1),
            "base_label":          win_label,
            "base_months":         int(base_months),
            "breakout_age":        int(months_above),  # months since breakout
            "vol_ratio":           round(vol_ratio, 2),
            "is_ath":              is_ath,
            "data_months":         int(n),
            "base_pct":            round(base_pct, 2),
            "base_low":            round(base_low, 2),
            "measured_target":     round(measured_target, 2),
            "remaining_upside_pct": round(remaining_upside_pct, 1),
        }

        # Prefer longest valid base
        if win_months > best_window:
            best = result
            best_window = win_months

    return best


# ── Main scan ─────────────────────────────────────────────────────────────────

def run_multiyear_scan(min_base_years: int = 1,
                       progress_callback=None) -> dict:
    """
    Scan Nifty Total Market 750 for multi-year breakouts.
    Returns {"results": [...], "scanned": N, "found": M, "computed_at": ts}
    """
    # Scan-level cache (results are not pre-filtered, so reuse for any request)
    cached = _scan_cache["data"]
    if cached and time.time() - _scan_cache["ts"] < SCAN_CACHE_TTL:
        return cached

    # Universe
    try:
        from nse_stocks import get_universe_symbols
        symbols = get_universe_symbols()
    except Exception:
        symbols = []

    if not symbols:
        return {"results": [], "scanned": 0, "found": 0,
                "computed_at": time.time(),
                "error": "Could not load Nifty Total Market 750 universe"}

    # Sector lookup
    from industry_groups import INDUSTRY_GROUPS
    sym_to_sector: dict[str, str] = {}
    for grp, syms in INDUSTRY_GROUPS.items():
        for s in syms:
            sym_to_sector[s] = grp

    # Get all monthly data (cache → download)
    monthly_data = _get_or_build_monthly_data(progress_callback)

    if not monthly_data:
        return {"results": [], "scanned": 0, "found": 0,
                "computed_at": time.time(),
                "error": "Failed to download bhavcopy archive data from NSE"}

    # Run detection on universe
    total = len(symbols)
    if progress_callback:
        progress_callback(0, total,
                          f"Scanning {total} stocks for multi-year breakouts…")

    universe = set(symbols)
    results = []
    scanned_count = 0

    for i, sym in enumerate(symbols):
        df = monthly_data.get(sym)
        if df is None or len(df) < MIN_MONTHS:
            continue
        scanned_count += 1
        bo = detect_multiyear_breakout(sym, df)
        if bo:
            bo["sector"] = sym_to_sector.get(sym, "Other")
            results.append(bo)

        if progress_callback and i % 50 == 0:
            progress_callback(i, total,
                              f"Scanning… {i}/{total} ({len(results)} breakouts so far)")

    # Always return all breakouts — frontend pills handle base filtering.
    # min_base_years param kept for backward compatibility but no longer filters.
    results.sort(key=lambda x: (-x["base_months"], x["breakout_age"], -x["pct_above"]))
    for i, r in enumerate(results, 1):
        r["rank"] = i

    out = {
        "results":        results,
        "all_breakouts":  len(results),
        "scanned":        scanned_count,
        "universe_size":  total,
        "found":          len(results),
        "min_base_years": min_base_years,
        "computed_at":    time.time(),
    }
    _scan_cache["data"] = out
    _scan_cache["ts"]   = time.time()
    return out


def invalidate_cache():
    """Force next scan to rebuild monthly data + re-run detection."""
    _scan_cache["data"]      = None
    _scan_cache["ts"]        = 0
    _near_scan_cache["data"] = None
    _near_scan_cache["ts"]   = 0
    combined_cache = MULTIYEAR_DIR / "_all_monthly.pkl"
    if combined_cache.exists():
        combined_cache.unlink()


# ── Near-Breakout detection ───────────────────────────────────────────────────

NEAR_BREAKOUT_MAX_GAP = 10.0   # % below resistance to qualify as "approaching"

def detect_near_breakout(symbol: str, monthly: pd.DataFrame) -> dict | None:
    """
    Return a near-breakout dict if the stock is currently within 10% below
    a multi-year resistance level (approaching but not yet broken out) — else None.
    """
    if monthly is None or len(monthly) < MIN_MONTHS:
        return None

    closes = monthly["Close"].values.astype(float)
    highs  = monthly["High"].values.astype(float)
    vols   = monthly["Volume"].values.astype(float)

    n = len(closes)
    current_close = closes[-1]

    # Volume ratio: last 1 month vs 12-month avg
    if n >= 13:
        vol_avg = np.mean(vols[-13:-1])
    else:
        vol_avg = np.mean(vols[:-1]) if n > 1 else vols[-1]
    vol_ratio = (vols[-1] / vol_avg) if vol_avg > 0 else 1.0

    best        = None
    best_window = 0

    for win_months, win_label in WINDOWS:
        required = win_months + BREAKOUT_WINDOW + 1
        if n < required:
            continue

        # Resistance = max high in lookback (excluding the recent window)
        lookback_highs  = highs[-(win_months + BREAKOUT_WINDOW): -BREAKOUT_WINDOW]
        lookback_closes = closes[-(win_months + BREAKOUT_WINDOW): -BREAKOUT_WINDOW]
        resistance      = float(np.max(lookback_highs))

        if resistance <= 0:
            continue

        # Must be BELOW resistance (not broken out yet)
        if current_close >= resistance:
            continue

        # Must be within NEAR_BREAKOUT_MAX_GAP % below resistance
        gap_pct = (resistance - current_close) / resistance * 100
        if gap_pct > NEAR_BREAKOUT_MAX_GAP:
            continue

        # Base quality: ≥50% of lookback months closed below resistance
        below_count = int(np.sum(lookback_closes < resistance))
        base_pct = below_count / max(len(lookback_closes), 1)
        if base_pct < BASE_PCT_THRESHOLD:
            continue

        # Count consecutive months below resistance from most recent bar
        consecutive_below = 0
        for i in range(n - 1, -1, -1):
            if closes[i] < resistance:
                consecutive_below += 1
            else:
                break

        # ── Measured-move target ───────────────────────────────────────────
        # base_low  = lowest close in lookback (support)
        # After a breakout, price typically travels base_height above resistance.
        # We show the projected target + total upside from current price.
        base_low        = float(np.min(lookback_closes))
        base_height     = resistance - base_low
        measured_target = resistance + base_height
        total_upside_pct = (measured_target / current_close - 1) * 100

        result = {
            "symbol":            symbol,
            "cmp":               round(current_close, 2),
            "resistance":        round(resistance, 2),
            "gap_pct":           round(gap_pct, 1),
            "base_label":        win_label,
            "base_months":       win_months,
            "consecutive_below": consecutive_below,
            "vol_ratio":         round(vol_ratio, 2),
            "data_months":       int(n),
            "base_pct":          round(base_pct, 2),
            "base_low":          round(base_low, 2),
            "measured_target":   round(measured_target, 2),
            "total_upside_pct":  round(total_upside_pct, 1),
        }

        # Prefer longest valid base
        if win_months > best_window:
            best        = result
            best_window = win_months

    return best


def run_near_breakout_scan(progress_callback=None) -> dict:
    """
    Scan Nifty Total Market 750 for stocks approaching multi-year resistance (within 10% below).
    Reuses the same monthly data cache as run_multiyear_scan() — if that ran first,
    this scan completes instantly from cache.
    Returns {"results": [...], "scanned": N, "found": M, "computed_at": ts}
    """
    cached = _near_scan_cache["data"]
    if cached and time.time() - _near_scan_cache["ts"] < SCAN_CACHE_TTL:
        return cached

    # Universe
    try:
        from nse_stocks import get_universe_symbols
        symbols = get_universe_symbols()
    except Exception:
        symbols = []

    if not symbols:
        return {"results": [], "scanned": 0, "found": 0,
                "computed_at": time.time(),
                "error": "Could not load Nifty Total Market 750 universe"}

    # Sector lookup
    from industry_groups import INDUSTRY_GROUPS
    sym_to_sector: dict[str, str] = {}
    for grp, syms in INDUSTRY_GROUPS.items():
        for s in syms:
            sym_to_sector[s] = grp

    # Reuse already-downloaded monthly data (fast if MBO scan ran first)
    monthly_data = _get_or_build_monthly_data(progress_callback)

    if not monthly_data:
        return {"results": [], "scanned": 0, "found": 0,
                "computed_at": time.time(),
                "error": "Failed to download bhavcopy archive data from NSE"}

    total = len(symbols)
    if progress_callback:
        progress_callback(0, total,
                          f"Scanning {total} stocks for near-breakout candidates…")

    results      = []
    scanned_count = 0

    for i, sym in enumerate(symbols):
        df = monthly_data.get(sym)
        if df is None or len(df) < MIN_MONTHS:
            continue
        scanned_count += 1
        nb = detect_near_breakout(sym, df)
        if nb:
            nb["sector"] = sym_to_sector.get(sym, "Other")
            results.append(nb)

        if progress_callback and i % 50 == 0:
            progress_callback(i, total,
                              f"Scanning… {i}/{total} ({len(results)} near-breakout candidates so far)")

    # Sort: closest to breakout first, then longest base
    results.sort(key=lambda x: (x["gap_pct"], -x["base_months"]))
    for i, r in enumerate(results, 1):
        r["rank"] = i

    out = {
        "results":       results,
        "scanned":       scanned_count,
        "universe_size": total,
        "found":         len(results),
        "computed_at":   time.time(),
    }
    _near_scan_cache["data"] = out
    _near_scan_cache["ts"]   = time.time()
    return out
