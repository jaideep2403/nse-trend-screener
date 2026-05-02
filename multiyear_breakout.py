"""
Multi-Year Breakout Scanner
============================
Detects stocks that have broken above a multi-year resistance level.

Data strategy (priority order):
  1. Per-stock monthly cache (pickle, 7-day TTL) — instant
  2. yfinance 10yr monthly download — ~0.5s per stock, rate-limited
  3. Bhavcopy daily data (resampled to monthly) — fallback, limited to ~1yr

Algorithm:
  For each lookback window N in [1, 2, 3, 5, 10] years:
    - resistance = max(monthly_high[ -N*12 - 3 : -3 ])   ← prior N-yr high
    - if current_close > resistance:
        - base_pct = fraction of lookback months that closed BELOW resistance
        - if base_pct >= 0.60 and stock was below resistance ≥ 3 months ago → BREAKOUT
    → pick the longest valid base window
"""

import os
import time
import pickle
import logging
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta

warnings.filterwarnings("ignore")
log = logging.getLogger(__name__)

# ── Directories ───────────────────────────────────────────────────────────────
DATA_DIR       = Path(os.getenv("DATA_DIR", "/tmp"))
BHAV_DIR       = Path(os.getenv("BHAV_DIR", "/tmp/nse_bhav_days"))
OHLCV_DIR      = Path(os.getenv("OHLCV_DIR", "/tmp/nse_ohlcv_pkl"))
MULTIYEAR_DIR  = DATA_DIR / "multiyear_cache"
MULTIYEAR_DIR.mkdir(parents=True, exist_ok=True)

MONTHLY_CACHE_TTL = 7 * 86400   # 7 days — monthly data barely changes
SCAN_CACHE_TTL    = 6 * 3600    # 6 hours — scan results
MIN_MONTHS        = 14          # need at least 14 monthly bars to detect a 1yr base
YF_DELAY          = 1.2         # seconds between yfinance requests
YF_BATCH          = 20          # download this many tickers at once with yf.download
YF_BATCH_DELAY    = 5.0         # seconds between batches

# Base duration windows (months)
WINDOWS = [
    (120, "10yr+"),
    (60,  "5yr+"),
    (36,  "3yr+"),
    (24,  "2yr+"),
    (12,  "1yr+"),
]

_scan_cache = {"data": None, "ts": 0}


# ── Monthly cache helpers ─────────────────────────────────────────────────────

def _monthly_cache_path(symbol: str) -> Path:
    return MULTIYEAR_DIR / f"{symbol}.pkl"


def _load_monthly_cache(symbol: str) -> pd.DataFrame | None:
    p = _monthly_cache_path(symbol)
    if not p.exists():
        return None
    age = time.time() - p.stat().st_mtime
    if age > MONTHLY_CACHE_TTL:
        return None
    try:
        with open(p, "rb") as f:
            return pickle.load(f)
    except Exception:
        return None


def _save_monthly_cache(symbol: str, df: pd.DataFrame):
    p = _monthly_cache_path(symbol)
    try:
        with open(p, "wb") as f:
            pickle.dump(df, f)
    except Exception:
        pass


# ── Data fetching ─────────────────────────────────────────────────────────────

def _fetch_from_bhavcopy(symbol: str) -> pd.DataFrame | None:
    """Load daily bhavcopy pkl and resample to monthly OHLCV."""
    pkl = OHLCV_DIR / f"{symbol}.NS.pkl"
    if not pkl.exists():
        return None
    try:
        with open(pkl, "rb") as f:
            df = pickle.load(f)
        if df is None or len(df) < 20:
            return None
        df.index = pd.to_datetime(df.index)
        monthly = df.resample("ME").agg(
            Open=("Open", "first"),
            High=("High", "max"),
            Low=("Low", "min"),
            Close=("Close", "last"),
            Volume=("Volume", "sum"),
        ).dropna()
        return monthly if len(monthly) >= 2 else None
    except Exception:
        return None


def _fetch_yfinance_batch(symbols: list[str]) -> dict[str, pd.DataFrame]:
    """Download monthly data for a batch of symbols from yfinance."""
    try:
        import yfinance as yf
        tickers = [f"{s}.NS" for s in symbols]
        raw = yf.download(
            tickers,
            period="10y",
            interval="1mo",
            progress=False,
            auto_adjust=True,
            group_by="ticker",
            timeout=30,
        )
        result = {}
        for sym, ticker in zip(symbols, tickers):
            try:
                if len(symbols) == 1:
                    df = raw.copy()
                else:
                    df = raw[ticker].copy() if ticker in raw.columns.get_level_values(0) else pd.DataFrame()
                df = df.dropna(how="all")
                if len(df) >= MIN_MONTHS:
                    df.index = pd.to_datetime(df.index).tz_localize(None)
                    result[sym] = df[["Open", "High", "Low", "Close", "Volume"]]
            except Exception:
                continue
        return result
    except Exception as e:
        log.debug(f"yfinance batch failed: {e}")
        return {}


def get_monthly_data(symbol: str) -> pd.DataFrame | None:
    """Return monthly OHLCV for symbol (cache → yfinance → bhavcopy fallback)."""
    # 1. Cache hit
    cached = _load_monthly_cache(symbol)
    if cached is not None and len(cached) >= MIN_MONTHS:
        return cached

    # 2. Try yfinance (single stock — called per-stock to control rate)
    try:
        import yfinance as yf
        import warnings; warnings.filterwarnings("ignore")
        df = yf.download(
            f"{symbol}.NS",
            period="10y",
            interval="1mo",
            progress=False,
            auto_adjust=True,
            timeout=20,
        )
        df = df.dropna(how="all")
        if len(df) >= MIN_MONTHS:
            df.index = pd.to_datetime(df.index).tz_localize(None)
            df = df[["Open", "High", "Low", "Close", "Volume"]]
            _save_monthly_cache(symbol, df)
            return df
    except Exception as e:
        log.debug(f"yfinance failed for {symbol}: {e}")

    # 3. Bhavcopy fallback
    df = _fetch_from_bhavcopy(symbol)
    if df is not None and len(df) >= MIN_MONTHS:
        _save_monthly_cache(symbol, df)
    return df


# ── Breakout detection ────────────────────────────────────────────────────────

def detect_multiyear_breakout(symbol: str, monthly: pd.DataFrame) -> dict | None:
    """
    Return breakout dict if symbol has broken above a multi-year resistance,
    else None.
    """
    if monthly is None or len(monthly) < MIN_MONTHS:
        return None

    closes = monthly["Close"].values
    highs  = monthly["High"].values
    vols   = monthly["Volume"].values

    n = len(closes)
    current_close = closes[-1]
    current_high  = highs[-1]

    # Volume ratio: last 1 month vs 12-month avg
    vol_avg = np.mean(vols[-13:-1]) if n >= 13 else np.mean(vols[:-1])
    vol_ratio = (vols[-1] / vol_avg) if vol_avg > 0 else 1.0

    best = None

    for win_months, win_label in WINDOWS:
        if n < win_months + 3:
            continue  # not enough history for this window

        # Resistance = max high in the lookback window, excluding last 3 months
        lookback_highs  = highs[-(win_months + 3):-3]
        lookback_closes = closes[-(win_months + 3):-3]
        resistance      = float(np.max(lookback_highs))

        if resistance <= 0:
            continue

        # Breakout condition: current close > resistance
        if current_close <= resistance:
            continue

        # Was it below resistance 3 months ago? (confirms recent breakout, not old one)
        close_3m_ago = closes[-4] if n >= 4 else closes[0]
        if close_3m_ago > resistance:
            continue  # already above — not a fresh breakout

        # Base quality: ≥60% of lookback months closed below resistance
        below_resistance = np.sum(lookback_closes < resistance)
        base_pct = below_resistance / len(lookback_closes) if len(lookback_closes) > 0 else 0
        if base_pct < 0.60:
            continue  # not a proper base (too many closes above resistance)

        # How far above resistance?
        pct_above = (current_close / resistance - 1) * 100

        # Duration of base = consecutive months below resistance before breakout
        # Walk back from month -3 counting contiguous sub-resistance months
        base_months = 0
        for i in range(n - 4, max(n - win_months - 3, -1), -1):
            if closes[i] < resistance:
                base_months += 1
            else:
                break  # first month that was above → base ends

        # ATH check: is this also an all-time high?
        is_ath = bool(current_close >= np.max(closes[:-1]))

        result = {
            "symbol":        symbol,
            "cmp":           round(current_close, 2),
            "resistance":    round(resistance, 2),
            "pct_above":     round(pct_above, 1),
            "base_label":    win_label,
            "base_months":   base_months,
            "vol_ratio":     round(vol_ratio, 2),
            "is_ath":        is_ath,
            "data_months":   n,
        }

        # Keep the result with the longest base window
        if best is None or win_months > WINDOWS[[w[0] for w in WINDOWS].index(
                next(w[0] for w in WINDOWS if w[1] == best["base_label"]))][0]:
            best = result

    return best


# ── Main scan ─────────────────────────────────────────────────────────────────

def run_multiyear_scan(min_base_years: int = 1,
                       progress_callback=None) -> dict:
    """
    Scan Nifty 500 for multi-year breakouts.
    Returns {"results": [...], "scanned": N, "found": M, "computed_at": ts}
    """
    # Check scan-level cache
    cached = _scan_cache["data"]
    if (cached and time.time() - _scan_cache["ts"] < SCAN_CACHE_TTL
            and cached.get("min_base_years") == min_base_years):
        return cached

    # Load universe
    try:
        from nse_stocks import get_nifty500_symbols
        symbols = get_nifty500_symbols()
    except Exception:
        symbols = []

    if not symbols:
        return {"results": [], "scanned": 0, "found": 0,
                "computed_at": time.time(), "error": "Could not load symbol list"}

    # Sector lookup
    from industry_groups import INDUSTRY_GROUPS
    sym_to_sector: dict[str, str] = {}
    for grp, syms in INDUSTRY_GROUPS.items():
        for s in syms:
            sym_to_sector[s] = grp

    total   = len(symbols)
    results = []
    yf_needed = []

    # Phase 1: serve cached + bhavcopy-only stocks instantly
    if progress_callback:
        progress_callback(0, total, f"Checking cache for {total} symbols…")

    for i, sym in enumerate(symbols):
        cached_df = _load_monthly_cache(sym)
        if cached_df is not None and len(cached_df) >= MIN_MONTHS:
            bo = detect_multiyear_breakout(sym, cached_df)
            if bo:
                bo["sector"] = sym_to_sector.get(sym, "Other")
                results.append(bo)
        else:
            # Try bhavcopy first (instant, no network)
            df = _fetch_from_bhavcopy(sym)
            if df is not None and len(df) >= MIN_MONTHS:
                _save_monthly_cache(sym, df)
                bo = detect_multiyear_breakout(sym, df)
                if bo:
                    bo["sector"] = sym_to_sector.get(sym, "Other")
                    results.append(bo)
            else:
                yf_needed.append(sym)

        if progress_callback and i % 50 == 0:
            progress_callback(i, total, f"Phase 1: {i}/{total} — {len(results)} breakouts found so far")

    # Phase 2: fetch remaining from yfinance in batches
    if yf_needed:
        if progress_callback:
            progress_callback(len(symbols) - len(yf_needed), total,
                              f"Fetching {len(yf_needed)} stocks from Yahoo Finance…")

        import yfinance as _yf

        done = 0
        for i in range(0, len(yf_needed), YF_BATCH):
            batch = yf_needed[i: i + YF_BATCH]
            try:
                tickers_ns = [f"{s}.NS" for s in batch]
                raw = _yf.download(
                    tickers_ns,
                    period="10y",
                    interval="1mo",
                    progress=False,
                    auto_adjust=True,
                    group_by="ticker",
                    timeout=45,
                )
                for sym in batch:
                    ticker = f"{sym}.NS"
                    try:
                        if len(batch) == 1:
                            df = raw.copy()
                        else:
                            lvl0 = raw.columns.get_level_values(0)
                            df = raw[ticker] if ticker in lvl0 else pd.DataFrame()
                        df = df.dropna(how="all")
                        if len(df) >= MIN_MONTHS:
                            df.index = pd.to_datetime(df.index).tz_localize(None)
                            df = df[["Open", "High", "Low", "Close", "Volume"]]
                            _save_monthly_cache(sym, df)
                            bo = detect_multiyear_breakout(sym, df)
                            if bo:
                                bo["sector"] = sym_to_sector.get(sym, "Other")
                                results.append(bo)
                    except Exception:
                        continue
            except Exception as e:
                log.debug(f"yf batch {i} failed: {e}")
                # Per-stock fallback with individual requests
                for sym in batch:
                    try:
                        time.sleep(YF_DELAY)
                        df = _yf.download(f"{sym}.NS", period="10y", interval="1mo",
                                          progress=False, auto_adjust=True, timeout=15)
                        df = df.dropna(how="all")
                        if len(df) >= MIN_MONTHS:
                            df.index = pd.to_datetime(df.index).tz_localize(None)
                            df = df[["Open", "High", "Low", "Close", "Volume"]]
                            _save_monthly_cache(sym, df)
                            bo = detect_multiyear_breakout(sym, df)
                            if bo:
                                bo["sector"] = sym_to_sector.get(sym, "Other")
                                results.append(bo)
                    except Exception:
                        continue

            done += len(batch)
            if progress_callback:
                progress_callback(
                    len(symbols) - len(yf_needed) + done, total,
                    f"Phase 2: {done}/{len(yf_needed)} Yahoo stocks processed"
                )
            if i + YF_BATCH < len(yf_needed):
                time.sleep(YF_BATCH_DELAY)

    # Filter by min_base_years
    min_label_months = min_base_years * 12
    filtered = []
    for r in results:
        win_months = next((w[0] for w in WINDOWS if w[1] == r["base_label"]), 0)
        if win_months >= min_label_months or r["base_months"] >= min_label_months:
            filtered.append(r)

    # Sort: longest base first, then % above resistance
    filtered.sort(key=lambda x: (-x["base_months"], -x["pct_above"]))

    # Add rank
    for i, r in enumerate(filtered, 1):
        r["rank"] = i

    out = {
        "results":        filtered,
        "all_breakouts":  len(results),
        "scanned":        total,
        "found":          len(filtered),
        "min_base_years": min_base_years,
        "computed_at":    time.time(),
    }
    _scan_cache["data"] = out
    _scan_cache["ts"]   = time.time()
    return out
