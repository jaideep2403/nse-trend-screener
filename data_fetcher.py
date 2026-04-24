"""
NSE Bhavcopy-based OHLCV fetcher
- Downloads official NSE daily bhavcopy CSVs (free, no rate limits)
- URL: https://archives.nseindia.com/products/content/sec_bhavdata_full_DDMMYYYY.csv
- Caches each day's file locally (never re-downloads old days)
- Builds per-stock DataFrames covering 1 year of history
"""
import io
import pickle
import time
import requests
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import date, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed

# ── Directories ────────────────────────────────────────────────────────────────
import os
BHAV_DIR   = Path(os.getenv("BHAV_DIR",  "/tmp/nse_bhav_days"))
OHLCV_DIR  = Path(os.getenv("OHLCV_DIR", "/tmp/nse_ohlcv_pkl"))
OHLCV_TTL  = 6 * 3600                      # reuse stock pkl for 6 hours
BHAV_DIR.mkdir(exist_ok=True)
OHLCV_DIR.mkdir(exist_ok=True)

DOWNLOAD_WORKERS = 10   # parallel bhavcopy downloads — NSE handles this fine
LOOKBACK_DAYS    = 400  # calendar days to cover (~270 trading days / ~1yr OHLCV)

_session = None


# ── HTTP session (reused across calls) ────────────────────────────────────────

def _get_session() -> requests.Session:
    global _session
    if _session is None:
        s = requests.Session()
        s.headers.update({
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                          "AppleWebKit/537.36 (KHTML, like Gecko) "
                          "Chrome/124.0.0.0 Safari/537.36",
            "Accept":          "text/html,application/xhtml+xml,*/*",
            "Accept-Language": "en-US,en;q=0.9",
            "Referer":         "https://www.nseindia.com/",
        })
        _session = s
    return _session


# ── Bhavcopy download helpers ──────────────────────────────────────────────────

def _bhav_url(dt: date) -> str:
    return (f"https://archives.nseindia.com/products/content/"
            f"sec_bhavdata_full_{dt.strftime('%d%m%Y')}.csv")


def _bhav_cache_path(dt: date) -> Path:
    return BHAV_DIR / f"{dt.strftime('%Y%m%d')}.pkl"


def _download_one_day(dt: date) -> pd.DataFrame | None:
    """Download and cache one day's bhavcopy. Returns None for holidays/futures."""
    cache = _bhav_cache_path(dt)
    if cache.exists():
        try:
            with open(cache, "rb") as f:
                return pickle.load(f)
        except Exception:
            cache.unlink(missing_ok=True)

    try:
        r = _get_session().get(_bhav_url(dt), timeout=15)
        if r.status_code != 200 or len(r.content) < 5_000:
            return None   # holiday or future date

        df = pd.read_csv(io.BytesIO(r.content))
        df.columns = [c.strip() for c in df.columns]

        # Keep EQ series only
        if "SERIES" in df.columns:
            df = df[df["SERIES"].str.strip() == "EQ"]

        if df.empty:
            return None

        df["SYMBOL"] = df["SYMBOL"].str.strip()
        df["DATE"]   = pd.Timestamp(dt)

        # Normalise numeric columns
        for col in ["OPEN_PRICE", "HIGH_PRICE", "LOW_PRICE",
                    "CLOSE_PRICE", "TTL_TRD_QNTY", "DELIV_PER"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        out = df[["SYMBOL", "DATE", "OPEN_PRICE", "HIGH_PRICE",
                  "LOW_PRICE", "CLOSE_PRICE", "TTL_TRD_QNTY"]].copy()
        out.columns = ["Symbol", "Date", "Open", "High", "Low", "Close", "Volume"]
        # Delivery % — present in bhavcopy; old cached files lack this column (→ NaN)
        if "DELIV_PER" in df.columns:
            out["DelivPer"] = df["DELIV_PER"].values

        with open(cache, "wb") as f:
            pickle.dump(out, f)
        return out

    except Exception:
        return None


def _weekdays_back(n: int) -> list[date]:
    """Generate Mon–Fri dates going back n calendar days from today."""
    today  = date.today()
    result = []
    d = today - timedelta(days=1)  # start from yesterday (today not published yet)
    cutoff = today - timedelta(days=n)
    while d >= cutoff:
        if d.weekday() < 5:   # 0=Mon … 4=Fri
            result.append(d)
        d -= timedelta(days=1)
    return result


# ── Per-stock pickle helpers (reuse across screener & sector analysis) ─────────

def _stock_pkl_path(ticker: str) -> Path:
    return OHLCV_DIR / (ticker.replace("/", "_") + ".pkl")


def _stock_pkl_load(ticker: str) -> pd.DataFrame | None:
    p = _stock_pkl_path(ticker)
    if not p.exists():
        return None
    if time.time() - p.stat().st_mtime > OHLCV_TTL:
        return None
    try:
        with open(p, "rb") as f:
            return pickle.load(f)
    except Exception:
        return None


def _stock_pkl_save(ticker: str, df: pd.DataFrame):
    try:
        with open(_stock_pkl_path(ticker), "wb") as f:
            pickle.dump(df, f)
    except Exception:
        pass


def _pkl_stats() -> tuple[int, int]:
    now   = time.time()
    files = list(OHLCV_DIR.glob("*.pkl"))
    fresh = sum(1 for p in files if now - p.stat().st_mtime <= OHLCV_TTL)
    return len(files), fresh


# ── Main API ───────────────────────────────────────────────────────────────────

def fetch_ohlcv(tickers: list[str], min_bars: int = 200,
                progress_callback=None) -> dict[str, pd.DataFrame]:
    """
    Return {ticker → OHLCV DataFrame} for the given .NS tickers.

    Strategy:
      1. Serve from per-stock pickle cache (instant, no network)
      2. For misses: download bhavcopy CSVs (one per trading day, parallel)
         then assemble per-stock DataFrames and save to cache
    """
    symbols = [t.replace(".NS", "") for t in tickers]

    # 1. Check per-stock cache
    result  = {}
    missing_symbols = []
    for sym, ticker in zip(symbols, tickers):
        df = _stock_pkl_load(ticker)
        if df is not None and len(df) >= min_bars:
            result[ticker] = df
        else:
            missing_symbols.append(sym)

    if not missing_symbols:
        if progress_callback:
            progress_callback(len(tickers), len(tickers),
                              f"All {len(tickers)} stocks served from cache ⚡")
        return result

    # 2. Download bhavcopy days in parallel
    dates        = _weekdays_back(LOOKBACK_DAYS)
    total_dates  = len(dates)
    downloaded   = []
    done_count   = [0]

    def _dl(dt):
        df = _download_one_day(dt)
        done_count[0] += 1
        if progress_callback:
            progress_callback(
                done_count[0], total_dates,
                f"Downloading NSE data: day {done_count[0]}/{total_dates}"
            )
        return df

    if progress_callback:
        progress_callback(0, total_dates, f"Fetching {total_dates} trading days from NSE…")

    with ThreadPoolExecutor(max_workers=DOWNLOAD_WORKERS) as ex:
        futs = [ex.submit(_dl, d) for d in dates]
        for fut in as_completed(futs):
            df = fut.result()
            if df is not None:
                downloaded.append(df)

    if not downloaded:
        return result   # nothing fetched — return whatever cache had

    # 3. Combine all days into one big DataFrame
    combined = pd.concat(downloaded, ignore_index=True)
    combined = combined.sort_values("Date")

    # 4. Build per-stock DataFrames for missing symbols
    for sym in missing_symbols:
        ticker = f"{sym}.NS"
        sdf    = combined[combined["Symbol"] == sym].copy()
        if len(sdf) < min_bars:
            continue
        sdf = sdf.set_index("Date")[["Open", "High", "Low", "Close", "Volume"]]
        sdf = sdf[~sdf.index.duplicated(keep="last")]   # dedupe (market holidays reshuffling)
        sdf = sdf.sort_index()
        _stock_pkl_save(ticker, sdf)
        result[ticker] = sdf

    if progress_callback:
        progress_callback(total_dates, total_dates,
                          f"Done — {len(result)} stocks loaded from NSE bhavcopy ✓")
    return result
