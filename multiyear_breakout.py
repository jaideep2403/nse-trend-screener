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
DOWNLOAD_WORKERS   = 8              # parallel bhavcopy downloads
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

_scan_cache = {"data": None, "ts": 0}


# ── Monthly sample dates ──────────────────────────────────────────────────────

def _month_sample_dates(years_back: int = HISTORY_YEARS) -> list[date]:
    """
    Return the last weekday of each month going back `years_back` years.
    These are the dates we'll fetch bhavcopy for (1 per month = ~72 dates total).
    """
    today = date.today()
    dates = []
    # Start from current month, walk back month by month
    y, m = today.year, today.month
    for _ in range(years_back * 12 + 1):
        # Last day of month (calendar.monthrange returns (weekday, last_day))
        last_day = calendar.monthrange(y, m)[1]
        d = date(y, m, last_day)
        # Walk back to a weekday
        while d.weekday() >= 5:  # 5=Sat, 6=Sun
            d -= timedelta(days=1)
        # If we're in the current month, use the latest cached/today's bhavcopy
        if (y, m) == (today.year, today.month):
            d = today
            while d.weekday() >= 5 or d > today:
                d -= timedelta(days=1)
        dates.append(d)
        # Decrement month
        m -= 1
        if m == 0:
            m = 12
            y -= 1
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
    Combine monthly bhavcopy frames into per-stock monthly OHLCV.
    Each row in result represents one month-end snapshot (Open=Close).
    """
    if not frames:
        return {}

    combined = pd.concat(frames, ignore_index=True).sort_values("Date")
    combined["Date"] = pd.to_datetime(combined["Date"])

    out: dict[str, pd.DataFrame] = {}
    for sym, g in combined.groupby("Symbol"):
        g = g.sort_values("Date").drop_duplicates(subset=["Date"], keep="last")
        if len(g) < MIN_MONTHS:
            continue
        df = g.set_index("Date")[["Open", "High", "Low", "Close", "Volume"]].copy()
        out[sym] = df
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
        base_months = (n - 1) - months_above   # months in base before breakout

        # ATH check
        is_ath = bool(current_close >= np.max(closes[:-1]))

        result = {
            "symbol":       symbol,
            "cmp":          round(current_close, 2),
            "resistance":   round(resistance, 2),
            "pct_above":    round(pct_above, 1),
            "base_label":   win_label,
            "base_months":  int(base_months),
            "breakout_age": int(months_above),  # months since breakout
            "vol_ratio":    round(vol_ratio, 2),
            "is_ath":       is_ath,
            "data_months":  int(n),
            "base_pct":     round(base_pct, 2),
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
    Scan Nifty 500 for multi-year breakouts.
    Returns {"results": [...], "scanned": N, "found": M, "computed_at": ts}
    """
    # Scan-level cache
    cached = _scan_cache["data"]
    if (cached and time.time() - _scan_cache["ts"] < SCAN_CACHE_TTL
            and cached.get("min_base_years") == min_base_years):
        return cached

    # Universe
    try:
        from nse_stocks import get_nifty500_symbols
        symbols = get_nifty500_symbols()
    except Exception:
        symbols = []

    if not symbols:
        return {"results": [], "scanned": 0, "found": 0,
                "computed_at": time.time(),
                "error": "Could not load Nifty 500 universe"}

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

    # Filter by min base years
    min_months = min_base_years * 12
    filtered = []
    for r in results:
        win_label_months = next((w[0] for w in WINDOWS if w[1] == r["base_label"]), 0)
        if win_label_months >= min_months:
            filtered.append(r)

    # Rank: longest base, then most recent breakout, then biggest % above
    filtered.sort(key=lambda x: (-x["base_months"], x["breakout_age"], -x["pct_above"]))
    for i, r in enumerate(filtered, 1):
        r["rank"] = i

    out = {
        "results":        filtered,
        "all_breakouts":  len(results),
        "scanned":        scanned_count,
        "universe_size":  total,
        "found":          len(filtered),
        "min_base_years": min_base_years,
        "computed_at":    time.time(),
    }
    _scan_cache["data"] = out
    _scan_cache["ts"]   = time.time()
    return out


def invalidate_cache():
    """Force next scan to rebuild monthly data + re-run detection."""
    _scan_cache["data"] = None
    _scan_cache["ts"]   = 0
    combined_cache = MULTIYEAR_DIR / "_all_monthly.pkl"
    if combined_cache.exists():
        combined_cache.unlink()
