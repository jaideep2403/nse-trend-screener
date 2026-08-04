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
from datetime import date, datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed

# ── Directories ────────────────────────────────────────────────────────────────
import os
BHAV_DIR   = Path(os.getenv("BHAV_DIR",  "/tmp/nse_bhav_days"))
OHLCV_DIR  = Path(os.getenv("OHLCV_DIR", "/tmp/nse_ohlcv_pkl"))
OHLCV_TTL  = 6 * 3600                      # reuse stock pkl for 6 hours


def _atomic_pickle_dump(obj, path) -> None:
    """Write a pickle ATOMICALLY: dump to a temp file in the same directory, then
    os.replace() it into place (atomic on POSIX). A concurrent reader therefore
    always sees either the old complete file or the new complete file — never a
    half-written one. Fixes the read-during-write race where a scanner/portfolio
    reading a day-pkl while the scheduler was rewriting it got a truncated file
    (pickle.load → EOFError → row dropped → '<30 rows' → 'No bhavcopy data')."""
    import tempfile
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=path.name + ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as f:
            pickle.dump(obj, f)
        os.replace(tmp, path)          # atomic rename — readers never see a partial file
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
BHAV_DIR.mkdir(exist_ok=True)
OHLCV_DIR.mkdir(exist_ok=True)

DOWNLOAD_WORKERS = 10   # parallel bhavcopy downloads — NSE handles this fine
LOOKBACK_DAYS    = 400  # calendar days to cover (~270 trading days / ~1yr OHLCV)

_session = None
_DAY_MEM: dict = {}   # in-memory memo: date → parsed DataFrame (never None)


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
    if dt in _DAY_MEM:
        return _DAY_MEM[dt]
    cache = _bhav_cache_path(dt)
    if cache.exists():
        try:
            with open(cache, "rb") as f:
                df = pickle.load(f)
            _DAY_MEM[dt] = df
            return df
        except Exception as e:
            print(f"[data_fetcher] corrupt bhavcopy cache {cache.name} — deleting and re-downloading: {e}", flush=True)
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

        _atomic_pickle_dump(out, cache)
        _DAY_MEM[dt] = out
        return out

    except Exception as e:
        print(f"[data_fetcher] bhavcopy download/parse failed for {dt}: {e}", flush=True)
        return None


def _weekdays_back(n: int) -> list[date]:
    """Generate Mon–Fri dates going back n calendar days from today.
    Always tries today first — _download_one_day() returns None gracefully
    if today's bhavcopy isn't published yet (NSE typically publishes ~6–7 PM IST).
    """
    today  = date.today()
    result = []
    d = today  # try today; skipped automatically if not a weekday or file not live yet
    cutoff = today - timedelta(days=n)
    while d >= cutoff:
        if d.weekday() < 5:   # 0=Mon … 4=Fri
            result.append(d)
        d -= timedelta(days=1)
    return result


# ── Latest available bhavcopy date ────────────────────────────────────────────

def _latest_bhavcopy_date() -> date | None:
    """
    Return the most recent date for which a bhavcopy pkl is cached locally.
    Checks today first, then walks back up to 10 trading days.
    Returns None only if no cached bhavcopy exists at all.
    """
    for dt in _weekdays_back(20):
        if _bhav_cache_path(dt).exists():
            return dt
    return None


# ── Auto-refresh: proactively pull today's bhavcopy ──────────────────────────

import threading as _threading

_refresh_lock  = _threading.Lock()
# State tracked separately so we can distinguish:
#   last_checked — when we last attempted (success OR failure)
#   last_attempt_msg — what happened on the last attempt (visible via /api/bhavcopy/status)
#   last_success_ts — when we last SUCCESSFULLY downloaded any bhavcopy
#   consecutive_failures — count since last success (drives session reset + escalation)
# The pre-fix code only had `last_checked`, which couldn't distinguish "we
# checked recently and it failed" from "we have fresh data" — the scheduler
# kept throttling itself even though it had never succeeded for the current day.
_refresh_state = {
    "last_checked":         0.0,
    "last_attempt_msg":     "",
    "last_success_ts":      0.0,
    "consecutive_failures": 0,
    "last_new_date":        None,
}

def _seed_last_success_from_disk() -> float:
    """Recover `last_success_ts` from the newest cached bhavcopy's mtime.

    WHY (2026-08-04): this state was in-memory only, so EVERY restart reset it to
    0.0 → `since_success` became inf → the stuck-detector fired immediately on any
    weekday where today's file wasn't in yet, and the log read
    `since last success: infh` even though the last download had succeeded hours
    earlier. The newest file's mtime IS the time of the last successful download,
    so it survives restarts for free — no new state file to keep in sync.
    """
    try:
        files = sorted(BHAV_DIR.glob("*.pkl"), key=lambda p: p.stat().st_mtime)
        return files[-1].stat().st_mtime if files else 0.0
    except Exception:
        return 0.0


_refresh_state["last_success_ts"] = _seed_last_success_from_disk()

_CHECK_INTERVAL_HAVE    = 1800   # 30 min — once we already have today's file
_CHECK_INTERVAL_MISSING = 300    # 5 min  — aggressive retry when today's file missing
_CHECK_INTERVAL_STUCK   = 60     # 1 min  — emergency retry when scheduler appears stuck

# NSE publishes the EQ bhavcopy AFTER market close (15:30 IST) — in practice from
# ~18:00 IST. Every attempt before that is guaranteed to fail, and each failure is
# not free: measured 2026-08-04, 49 attempts over 8h produced 43 session resets,
# because a miss triggers internal retries and a fresh handshake. Almost all of
# those ran hours before publication was even possible. Waiting until the window
# opens turns a day of futile scraping into a handful of real attempts — and keeps
# us a polite client, which is the whole point.
# Local clock is assumed to be IST (this box runs IST); `force=True` always bypasses.
_PUBLISH_WINDOW_OPENS = (17, 45)   # HH, MM local time


def _publish_window_open(when: datetime | None = None) -> bool:
    """True once NSE could plausibly have published today's file."""
    n = when or datetime.now()
    return (n.hour, n.minute) >= _PUBLISH_WINDOW_OPENS
# If we've gone this long without ANY successful download, treat the scheduler as
# stuck and bypass the throttle. NSE publishes between ~6 PM IST and midnight,
# so 4 hours of "could not get today's file during a weekday" is well past normal
# delay and signals a real problem (session expired, NSE format change, etc.).
_STUCK_THRESHOLD_SECS   = 4 * 3600


def auto_refresh_bhavcopy(force: bool = False) -> dict:
    """
    Proactively try to download today's (or the latest weekday's) bhavcopy.
    Called from the background scheduler — safe to call frequently; internal
    throttling ensures we only hit NSE at most every 5 min when today's data
    is missing (4–9 PM IST window), or every 30 min once already cached.

    NEW: throttle relaxes automatically if we've gone too long without a
    successful download. This prevents the scheduler from sitting in a
    "throttled" state for hours when something has gone wrong (network
    timeout, NSE format change, requests session expired, etc.).

    Pass force=True to bypass throttle entirely — used by the /api/bhavcopy/
    refresh endpoint when a user clicks "Force Refresh".

    Returns {"downloaded": bool, "date": date|None, "already_had": bool,
             "msg": str, "since_success": float}
    """
    now = time.time()
    today = date.today()

    # Stuck detection: if we've gone too long without a successful download
    # during a trading day, ignore the throttle and aggressively retry. This
    # prevents the scheduler from sitting "throttled" for hours when the
    # download path is silently broken.
    since_success = now - _refresh_state["last_success_ts"] if _refresh_state["last_success_ts"] else float("inf")
    # "Stuck" must mean "we should have been able to download and couldn't". Before
    # the publication window that is never true — the file simply doesn't exist yet —
    # so requiring the window keeps the [STUCK — escalating] log line honest instead
    # of firing on every restart during market hours.
    is_stuck = (today.weekday() < 5
                and not _bhav_cache_path(today).exists()
                and _publish_window_open()
                and since_success > _STUCK_THRESHOLD_SECS)

    # Use short interval when today's file is missing, long interval once we
    # have it, or 1 min "emergency" when stuck.
    if _bhav_cache_path(today).exists():
        interval = _CHECK_INTERVAL_HAVE
    elif is_stuck:
        interval = _CHECK_INTERVAL_STUCK
    else:
        interval = _CHECK_INTERVAL_MISSING

    if not force and (now - _refresh_state["last_checked"] < interval):
        return {"downloaded": False, "date": None, "already_had": True,
                "msg": f"throttled (next check in {int(interval - (now - _refresh_state['last_checked']))}s)",
                "since_success": since_success if since_success != float("inf") else None}

    with _refresh_lock:
        # Re-check after acquiring lock (another thread may have just done it)
        if not force and (time.time() - _refresh_state["last_checked"] < interval):
            return {"downloaded": False, "date": None, "already_had": True,
                    "msg": "throttled (lost the race)",
                    "since_success": since_success if since_success != float("inf") else None}

        _refresh_state["last_checked"] = time.time()
        # Only weekdays — NSE doesn't publish on weekends
        if today.weekday() >= 5:
            _refresh_state["last_attempt_msg"] = "weekend — no bhavcopy"
            return {"downloaded": False, "date": None, "already_had": False,
                    "msg": "weekend — no bhavcopy",
                    "since_success": since_success if since_success != float("inf") else None}

        # Check if we already have today's file AND validate it's non-empty.
        # The bare exists() check could see a partially-written zero-byte file
        # from a crashed download and incorrectly think we're done.
        cache_path = _bhav_cache_path(today)
        if cache_path.exists() and cache_path.stat().st_size > 1024:
            _refresh_state["last_attempt_msg"] = "already have today"
            _refresh_state["consecutive_failures"] = 0
            _refresh_state["last_success_ts"] = now
            return {"downloaded": False, "date": today, "already_had": True,
                    "msg": "already have today",
                    "since_success": 0}

        # Publication window — see _PUBLISH_WINDOW_OPENS. Before NSE could plausibly
        # have published, a request is guaranteed to fail, so don't make it at all.
        # This is NOT counted as a failure: consecutive_failures stays put, so it
        # neither triggers a session reset nor trips the stuck-detector.
        if not force:
            _oh, _om = _PUBLISH_WINDOW_OPENS
            _n = datetime.now()
            if (_n.hour, _n.minute) < (_oh, _om):
                msg = (f"waiting for NSE publication window "
                       f"(opens {_oh:02d}:{_om:02d}, now {_n.strftime('%H:%M')})")
                _refresh_state["last_attempt_msg"] = msg
                return {"downloaded": False, "date": None, "already_had": False,
                        "msg": msg,
                        "since_success": since_success if since_success != float("inf") else None}

        # Try to fetch today's bhavcopy from NSE (typically published ~6–7 PM IST)
        # On repeated failures, reset the requests session — NSE's anti-bot wall
        # sometimes expires the seeded cookies, and a fresh session recovers.
        if _refresh_state["consecutive_failures"] >= 3:
            try:
                global _session
                _session = None
                print(f"[bhavcopy] 🔄 Resetting session after "
                      f"{_refresh_state['consecutive_failures']} failures",
                      flush=True)
            except Exception:
                pass

        result = _download_one_day(today)
        if result is not None:
            _refresh_state["last_new_date"] = today
            print(f"[bhavcopy] ✅ Auto-downloaded {today} bhavcopy "
                  f"({len(result)} rows)", flush=True)
            # Bust ALL in-memory caches so next scan uses fresh data. NOTE: we
            # clear the WORKING snapshot but deliberately KEEP `last_good` — if the
            # fresh load is mid-write/incomplete, _get_stocks falls back to it
            # instead of caching a broken partial. last_attempt=0 forces a prompt
            # reload now that new data has landed (bypasses the retry throttle).
            try:
                from industry_groups import _stocks_cache, _cache as _ig_cache
                _stocks_cache["data"]         = None   # raw OHLCV stocks
                _stocks_cache["ts"]           = 0
                _stocks_cache["complete"]     = False
                _stocks_cache["last_attempt"] = 0.0
                _ig_cache["data"]             = None   # RS scan results
                _ig_cache["ts"]               = 0
            except Exception:
                pass
            try:
                from sector_analysis import _cache as _sa_cache
                _sa_cache["data"] = None       # sector heatmap results
                _sa_cache["ts"]   = 0
            except Exception:
                pass
            try:
                from market_breadth import _cache as _mb_cache, _CACHE_PATH as _mb_disk
                _mb_cache["data"] = None       # market breadth in-memory
                _mb_cache["ts"]   = 0
                # Delete disk-based breadth cache (6h TTL — stale after new bhavcopy)
                if _mb_disk.exists():
                    _mb_disk.unlink(missing_ok=True)
            except Exception:
                pass
            try:
                from trending import _cache as _tr_cache
                _tr_cache["data"] = None       # trending scan results
                _tr_cache["ts"]   = 0
            except Exception:
                pass
            # BUG-FIX: previously these 4 caches were NOT busted → momentum, institutional,
            # multi-year breakout, early-mover all served stale data after a new bhavcopy.
            for mod_name, cache_name in [
                ("momentum_scanner", "_cache"),
                ("institutional_scanner", "_cache"),
                ("multiyear_breakout", "_scan_cache"),
                ("early_mover_scanner", "_cache"),
                ("early_growth", "_cache"),
                ("monster_growth", "_cache"),
            ]:
                try:
                    mod = __import__(mod_name)
                    c = getattr(mod, cache_name, None)
                    if c is not None:
                        c["data"] = None
                        c["ts"] = 0
                except Exception:
                    pass
            _refresh_state["last_success_ts"]      = time.time()
            _refresh_state["consecutive_failures"] = 0
            _refresh_state["last_attempt_msg"]     = f"downloaded {today}"
            return {"downloaded": True, "date": today, "already_had": False,
                    "msg": f"downloaded {today}", "since_success": 0}
        else:
            # Failure — increment counter so subsequent attempts know to reset
            # the session. Log every Nth failure for visibility (don't log
            # every 5-min retry, but do log when the situation gets concerning).
            _refresh_state["consecutive_failures"] += 1
            n_fail = _refresh_state["consecutive_failures"]
            msg = f"not yet published by NSE ({today}) — attempt #{n_fail}"
            _refresh_state["last_attempt_msg"] = msg
            # Log on 1st failure (normal — file not yet published) and every
            # 12th after that (every ~1 hour at 5 min cadence) so the log
            # contains a heartbeat without being spammy.
            if n_fail == 1 or n_fail % 12 == 0 or is_stuck:
                stuck_note = " [STUCK — escalating]" if is_stuck else ""
                print(f"[bhavcopy] ❌ {msg}{stuck_note} "
                      f"(since last success: {since_success/3600:.1f}h)",
                      flush=True)
            return {"downloaded": False, "date": None, "already_had": False,
                    "msg": msg,
                    "since_success": since_success if since_success != float("inf") else None,
                    "consecutive_failures": n_fail}


# ── Per-stock pickle helpers (reuse across screener & sector analysis) ─────────

def _stock_pkl_path(ticker: str) -> Path:
    return OHLCV_DIR / (ticker.replace("/", "_") + ".pkl")


def _stock_pkl_load(ticker: str) -> pd.DataFrame | None:
    """
    Load per-stock OHLCV pkl from disk.

    Invalidates (returns None) if:
      - File doesn't exist or is corrupt
      - File is older than OHLCV_TTL (time-based staleness)
      - DataFrame's last row date is older than the latest available
        bhavcopy — means a new day's data exists that isn't in the pkl
    """
    p = _stock_pkl_path(ticker)
    if not p.exists():
        return None
    if time.time() - p.stat().st_mtime > OHLCV_TTL:
        return None
    try:
        with open(p, "rb") as f:
            df = pickle.load(f)
        # ── Key check: reject pkl if it's missing a newer bhavcopy day ──────
        # BUG-012 FIX: Strip timezone info before calling .date() to avoid
        # "can't compare offset-naive and offset-aware datetimes" TypeError.
        latest = _latest_bhavcopy_date()
        if latest is not None:
            last_idx = df.index[-1]
            if hasattr(last_idx, "tzinfo") and last_idx.tzinfo is not None:
                last_idx = last_idx.tz_localize(None)
            pkl_last = last_idx.date() if hasattr(last_idx, "date") else last_idx
            if pkl_last < latest:
                return None   # stale — a newer bhavcopy day exists, force rebuild
        return df
    except Exception as e:
        print(f"[data_fetcher] corrupt stock pkl for {ticker} — forcing rebuild: {e}", flush=True)
        return None


def _stock_pkl_save(ticker: str, df: pd.DataFrame):
    try:
        _atomic_pickle_dump(df, _stock_pkl_path(ticker))
    except Exception as e:
        print(f"[data_fetcher] failed to save stock pkl for {ticker}: {e}", flush=True)


def _pkl_stats() -> tuple[int, int]:
    now   = time.time()
    files = list(OHLCV_DIR.glob("*.pkl"))
    fresh = sum(1 for p in files if now - p.stat().st_mtime <= OHLCV_TTL)
    return len(files), fresh


# ── Split / Bonus adjustment ──────────────────────────────────────────────────

def _adjust_for_splits(df, symbol=None):
    """Delegates to the centralised analysis_utils.adjust_for_splits which
    handles the full set of NSE bonus ratios (3:2, 4:3, 5:4, etc.)."""
    from analysis_utils import adjust_for_splits
    return adjust_for_splits(df, symbol)


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
        # BUG-FIX: was dropping DelivPer (NSE delivery %). Every downstream consumer
        # (analysis_utils.delivery_trend, institutional/early-mover scoring, composite_rank)
        # got None for delivery → all institutional-accumulation signals defaulted neutral.
        cols = ["Open", "High", "Low", "Close", "Volume"]
        if "DelivPer" in sdf.columns:
            cols.append("DelivPer")
        sdf = sdf.set_index("Date")[cols]
        sdf = sdf[~sdf.index.duplicated(keep="last")]   # dedupe (market holidays reshuffling)
        sdf = sdf.sort_index()
        sdf = _adjust_for_splits(sdf, ticker)     # backward-adjust unadjusted bhavcopy prices
        _stock_pkl_save(ticker, sdf)
        result[ticker] = sdf

    if progress_callback:
        progress_callback(total_dates, total_dates,
                          f"Done — {len(result)} stocks loaded from NSE bhavcopy ✓")
    return result
