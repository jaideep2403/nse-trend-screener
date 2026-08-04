"""
F&O Open Interest data from NSE static archive bhavcopy ZIPs.

URL (current, post-2024 unified format):
  https://archives.nseindia.com/content/fo/BhavCopy_NSE_FO_0_0_0_{YYYYMMDD}_F_0000.csv.zip

BUG-FIX: Was using the old `fo{DDMMYYYY}bhav.csv.zip` URL which NSE retired
in mid-2024 — every download returned 404, the entire F&O signal pipeline
returned empty for every stock. The new unified format works and is the same
file structure NSE now uses for both cash and derivatives bhavcopies.

Column names ALSO changed: SYMBOL→TckrSymb, INSTRUMENT→FinInstrmTp,
CLOSE→ClsPric, OPEN_INT→OpnIntrst, CHG_IN_OI→ChngInOpnIntrst, EXPIRY_DT→XpryDt.
FinInstrmTp values: STF (stock future), IDF (index future), STO/IDO (options).

These are static files on NSE archives — no session seeding required, no rate limit.
We download the last 7 trading days' F&O bhavcopy files, extract STF (stock futures)
rows, and compute per-symbol OI signals:

  Signal logic (price direction + OI direction):
  ┌──────────────┬─────────────┬──────────────────────────────────┐
  │ OI Change    │ Price Chg   │ Signal                           │
  ├──────────────┼─────────────┼──────────────────────────────────┤
  │ Rising (+)   │ Rising (+)  │ LONG_BUILDUP  — most bullish     │
  │ Rising (+)   │ Falling (-) │ SHORT_BUILDUP — bearish          │
  │ Falling (-)  │ Rising (+)  │ SHORT_COVER   — near-term bull   │
  │ Falling (-)  │ Falling (-) │ LONG_UNWIND   — bearish exit     │
  │ Flat         │ Any         │ NEUTRAL                          │
  └──────────────┴─────────────┴──────────────────────────────────┘

Cache: 24 hours per trading day file. Static file → safe to cache indefinitely.
"""
from __future__ import annotations

import io
import os
import pickle
import time
import zipfile
from datetime import date, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd
import requests

# ── Directories ────────────────────────────────────────────────────────────────
# PERSISTENT (2026-08-02): this lived in /tmp and macOS purged it — 631 cached
# daily F&O archives fell to 9, silently gutting the institutional-flow factor.
# These are rate-limited NSE scrapes; losing them costs hours and NSE goodwill.
_FO_DIR = Path(os.getenv("FO_DIR", os.path.join(os.path.expanduser("~"), ".ascent_cache", "nse_fo_bhav")))
_FO_DIR.mkdir(exist_ok=True)

_CACHE_PATH = _FO_DIR / "_fo_signal_cache.pkl"
_CACHE_TTL  = 24 * 3600   # 24-hour cache — static files, never change

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept":          "application/octet-stream, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer":         "https://www.nseindia.com/",
}

_MEM: dict = {"data": None, "ts": 0.0}

# Month abbreviations used by NSE in URLs
_MONTHS = ["JAN","FEB","MAR","APR","MAY","JUN","JUL","AUG","SEP","OCT","NOV","DEC"]


# ── URL builder ───────────────────────────────────────────────────────────────

def _fo_url(dt: date) -> str:
    """BUG-FIX: NSE retired the old fo{DDMMYYYY}bhav.csv.zip path in mid-2024.
    Now uses the unified BhavCopy_NSE_FO archive (same shape as cash bhavcopy).
    """
    return (
        f"https://archives.nseindia.com/content/fo/"
        f"BhavCopy_NSE_FO_0_0_0_{dt.strftime('%Y%m%d')}_F_0000.csv.zip"
    )


def _fo_cache_path(dt: date) -> Path:
    return _FO_DIR / f"fo_{dt.strftime('%Y%m%d')}.pkl"


# ── Download one day's F&O bhavcopy ──────────────────────────────────────────

def _download_fo_day(dt: date) -> Optional[pd.DataFrame]:
    """
    Download one trading day's F&O bhavcopy ZIP and parse it.
    Returns DataFrame with FUTSTK rows only, or None on failure.
    File is cached permanently (static NSE archive, never changes).
    """
    cache = _fo_cache_path(dt)
    if cache.exists():
        try:
            with open(cache, "rb") as f:
                return pickle.load(f)
        except Exception:
            cache.unlink(missing_ok=True)

    url = _fo_url(dt)
    try:
        # NSE archives sometimes need session cookies; seed politely once per process.
        sess = _get_seeded_session()
        r = sess.get(url, timeout=20)
        if r.status_code != 200 or len(r.content) < 1000:
            return None   # holiday / future date / file not published yet

        with zipfile.ZipFile(io.BytesIO(r.content)) as z:
            csv_name = [n for n in z.namelist() if n.lower().endswith(".csv")]
            if not csv_name:
                return None
            raw = z.read(csv_name[0])

        df = pd.read_csv(io.BytesIO(raw))
        df.columns = [c.strip() for c in df.columns]

        # BUG-FIX: NSE renamed every column in 2024.
        # Map: SYMBOL→TckrSymb, INSTRUMENT→FinInstrmTp, CLOSE→ClsPric,
        #      OPEN_INT→OpnIntrst, CHG_IN_OI→ChngInOpnIntrst, EXPIRY_DT→XpryDt.
        # FUTSTK → STF (stock future).
        rename = {
            "TckrSymb":         "SYMBOL",
            "FinInstrmTp":      "INSTRUMENT",
            "ClsPric":          "CLOSE",
            "OpnIntrst":        "OPEN_INT",
            "ChngInOpnIntrst":  "CHG_IN_OI",
            "XpryDt":           "EXPIRY_DT",
        }
        for new, old in rename.items():
            if new in df.columns and old not in df.columns:
                df[old] = df[new]

        if "INSTRUMENT" not in df.columns or "SYMBOL" not in df.columns:
            return None
        # New code uses "STF" for stock futures; keep backward-compat with "FUTSTK".
        inst_vals = df["INSTRUMENT"].astype(str).str.strip()
        df = df[inst_vals.isin(["STF", "FUTSTK"])].copy()
        if df.empty:
            return None

        df["SYMBOL"] = df["SYMBOL"].astype(str).str.strip()
        for col in ["CLOSE", "OPEN_INT", "CHG_IN_OI"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

        # Front-month only: take the nearest expiry per symbol
        if "EXPIRY_DT" in df.columns:
            df["EXPIRY_DT"] = pd.to_datetime(df["EXPIRY_DT"], errors="coerce")
            df = df.sort_values("EXPIRY_DT")
            df = df.groupby("SYMBOL", sort=False).first().reset_index()

        keep_cols = ["SYMBOL", "CLOSE", "OPEN_INT", "CHG_IN_OI"]
        if "EXPIRY_DT" in df.columns:
            keep_cols.append("EXPIRY_DT")
        out = df[keep_cols].copy()
        out["date"] = dt

        with open(cache, "wb") as f:
            pickle.dump(out, f)
        return out

    except Exception:
        return None


# Seeded session shared across calls (cookies needed for nseindia.com archives)
_session: Optional[requests.Session] = None
def _get_seeded_session() -> requests.Session:
    global _session
    if _session is not None:
        return _session
    s = requests.Session()
    s.headers.update(_HEADERS)
    try:
        s.get("https://www.nseindia.com", timeout=10)
        s.get("https://www.nseindia.com/api/marketStatus", timeout=10)
    except Exception:
        pass
    _session = s
    return s


# ── Load last N trading days ──────────────────────────────────────────────────

def _weekdays_back(n: int = 20) -> list[date]:
    today  = date.today()
    result = []
    d      = today
    cutoff = today - timedelta(days=n)
    while d >= cutoff and len(result) < 10:
        if d.weekday() < 5:
            result.append(d)
        d -= timedelta(days=1)
    return result


def _load_fo_frames(n_days: int = 10) -> list[pd.DataFrame]:
    """Download last n_days worth of F&O data. Returns list of frames."""
    frames: list[pd.DataFrame] = []
    for dt in _weekdays_back(n_days * 2):   # walk back calendar days
        df = _download_fo_day(dt)
        if df is not None and not df.empty:
            frames.append(df)
            if len(frames) >= n_days:
                break
        # Polite pacing: only for network hits (cached files are instant)
        if not _fo_cache_path(dt).exists():
            time.sleep(0.5)
    return frames


# ── OI signal computation ─────────────────────────────────────────────────────

def _compute_signals(frames: list[pd.DataFrame]) -> dict[str, dict]:
    """
    Compute per-symbol OI signal from the last 2 available trading days.

    Returns {symbol: {
        "signal":    "LONG_BUILDUP" | "SHORT_BUILDUP" | "SHORT_COVER" |
                     "LONG_UNWIND" | "NEUTRAL",
        "oi_today":  int,
        "oi_prev":   int,
        "oi_chg":    int,      # today minus 5 days ago
        "oi_chg_pct": float,
        "price":     float,    # today's settle price
        "price_chg_pct": float # % change latest vs 5 days ago
    }}
    """
    if len(frames) < 2:
        return {}

    latest = frames[0].set_index("SYMBOL")
    # Use a 5-day lag frame for OI trend confirmation. The OI/price thresholds
    # below (5% / 0.5%) are calibrated for FIVE-DAY moves; if we only have 1-2
    # days of cached F&O history (cold start, post-cache-clear, or after a long
    # holiday weekend) we cannot meaningfully say "OI is rising over 5 days".
    # Returning {} here makes Alpha Engine show NEUTRAL until the cache warms up,
    # rather than firing false LONG_BUILDUP/SHORT_BUILDUP signals.
    if len(frames) < 5:
        return {}
    lag_idx = min(5, len(frames) - 1)
    lag     = frames[lag_idx].set_index("SYMBOL")

    signals: dict[str, dict] = {}
    for sym in latest.index:
        try:
            oi_now   = float(latest.loc[sym, "OPEN_INT"])
            price_now = float(latest.loc[sym, "CLOSE"])

            if sym not in lag.index:
                continue

            oi_old    = float(lag.loc[sym, "OPEN_INT"])
            price_old = float(lag.loc[sym, "CLOSE"])

            oi_chg     = oi_now - oi_old
            price_chg  = price_now - price_old if price_old > 0 else 0

            # BUG-035 FIX: 2% OI move is well within ordinary noise; 5% gives
            # a directional signal closer to what professional desks watch for.
            oi_chg_pct    = (oi_chg / oi_old * 100)    if oi_old > 0 else 0.0
            price_chg_pct = (price_chg / price_old * 100) if price_old > 0 else 0.0

            # BUG-004 FIX: if the front-month contract rolled between the lag
            # snapshot and today, comparing OI across DIFFERENT contracts is
            # meaningless — flag as ROLLED instead of LONG/SHORT_BUILDUP etc.
            rolled = False
            try:
                exp_now = latest.loc[sym, "EXPIRY_DT"] if "EXPIRY_DT" in latest.columns else None
                exp_old = lag.loc[sym,    "EXPIRY_DT"] if "EXPIRY_DT" in lag.columns    else None
                if exp_now is not None and exp_old is not None:
                    if pd.notna(exp_now) and pd.notna(exp_old) and exp_now != exp_old:
                        rolled = True
            except Exception:
                pass

            oi_up    = oi_chg_pct > 5.0
            oi_down  = oi_chg_pct < -5.0
            price_up = price_chg_pct > 0.5
            price_dn = price_chg_pct < -0.5

            if rolled:
                sig = "ROLLED"
            elif oi_up and price_up:
                sig = "LONG_BUILDUP"
            elif oi_up and price_dn:
                sig = "SHORT_BUILDUP"
            elif oi_down and price_up:
                sig = "SHORT_COVER"
            elif oi_down and price_dn:
                sig = "LONG_UNWIND"
            else:
                sig = "NEUTRAL"

            signals[sym] = {
                "signal":        sig,
                "oi_today":      int(oi_now),
                "oi_prev":       int(oi_old),
                "oi_chg":        int(oi_chg),
                "oi_chg_pct":    round(oi_chg_pct, 2),
                "price":         round(price_now, 2),
                "price_chg_pct": round(price_chg_pct, 2),
            }
        except Exception:
            continue

    return signals


# ── Public API ────────────────────────────────────────────────────────────────

def get_fo_signals() -> dict[str, dict]:
    """
    Return OI signals for all F&O stocks. Cached 24 hours.
    Keys are NSE symbols (without .NS suffix).

    Example: get_fo_signals().get("BEL") → {"signal": "LONG_BUILDUP", "oi_chg_pct": 8.2, ...}
    """
    if _MEM["data"] and time.time() - _MEM["ts"] < _CACHE_TTL:
        return _MEM["data"]

    # Disk cache
    if _CACHE_PATH.exists():
        try:
            with open(_CACHE_PATH, "rb") as f:
                cached = pickle.load(f)
            if time.time() - cached.get("ts", 0) < _CACHE_TTL:
                _MEM.update({"data": cached["signals"], "ts": cached["ts"]})
                return cached["signals"]
        except Exception:
            _CACHE_PATH.unlink(missing_ok=True)

    # Download and compute
    frames  = _load_fo_frames(n_days=7)
    signals = _compute_signals(frames)

    if signals:
        try:
            with open(_CACHE_PATH, "wb") as f:
                pickle.dump({"signals": signals, "ts": time.time()}, f)
        except Exception:
            pass
        _MEM.update({"data": signals, "ts": time.time()})

    return signals
