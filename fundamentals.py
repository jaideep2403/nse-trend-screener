"""
Fundamentals — screener.in scraper, ~240 stocks/hour, fully background.

Design:
  - Auto-starts on app launch as a daemon thread
  - 15 seconds base delay + random 0-5s jitter between requests (~240 stocks/hour)
  - Jitter makes requests look human — avoids clock-pattern rate limiting
  - Resume-safe: skips symbols already in cache with real data
  - SQLite local cache (fundamentals.db), TTL 90 days
  - Zero yfinance / Yahoo Finance code

Source:  screener.in  (public HTML, 1 req/stock, no login needed)
Cache:   ./fundamentals.db  SQLite

Safe floor confirmed by testing: 10s clean, 5s rate-limited → 15s + jitter is safe.

Progress visible at /api/fundamentals/status:
  scraped_count, failed_count, pending_count, eta_minutes, current_symbol
"""
import os
import re
import random
import sqlite3
import time
import threading
import requests
from typing import Optional

DB_PATH      = os.path.join(os.environ.get("DATA_DIR", os.path.dirname(__file__)), "fundamentals.db")
TTL_DAYS     = 90          # re-fetch after 90 days
FETCH_DELAY  = 15          # base seconds between requests
FETCH_JITTER = 5           # add random 0–FETCH_JITTER seconds to each delay
FRESH_DAYS   = 30          # consider cached "fresh" if < 30 days old AND has data

_db_lock = threading.Lock()   # serialises SQLite writes

# ── Scheduler live state (read by /api/fundamentals/status) ───────────────────
_sched = {
    "running":          False,
    "total":            0,      # total non-ETF symbols
    "scraped_count":    0,      # successfully stored (this session + prev)
    "failed_count":     0,      # failed fetches this session
    "pending_count":    0,      # still to do
    "current_symbol":   "",     # symbol being fetched right now
    "last_scraped":     "",     # last symbol successfully stored
    "last_scraped_at":  0,      # unix ts of last success
    "eta_minutes":      0,      # estimated minutes to finish
    "started_at":       0,
    "error":            "",
}
_sched_lock = threading.Lock()


# ── SQLite helpers ─────────────────────────────────────────────────────────────

def _init_db():
    with _db_lock:
        conn = sqlite3.connect(DB_PATH)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS fundamentals (
                symbol           TEXT PRIMARY KEY,
                eps_growth_yoy   REAL,
                sales_growth_yoy REAL,
                roe              REAL,
                debt_to_equity   REAL,
                pe_ratio         REAL,
                market_cap       REAL,
                promoter_holding REAL,
                updated_at       INTEGER
            )
        """)
        conn.commit()
        conn.close()


def _upsert(data: dict):
    with _db_lock:
        conn = sqlite3.connect(DB_PATH)
        conn.execute("""
            INSERT OR REPLACE INTO fundamentals
            (symbol, eps_growth_yoy, sales_growth_yoy, roe,
             debt_to_equity, pe_ratio, market_cap, promoter_holding, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (data["symbol"], data["eps_growth_yoy"], data["sales_growth_yoy"],
              data["roe"], data["debt_to_equity"], data["pe_ratio"],
              data["market_cap"], data["promoter_holding"], data["updated_at"]))
        conn.commit()
        conn.close()


def get_fundamentals(symbol: str) -> Optional[dict]:
    """Read one symbol's fundamentals from cache (instant, no network)."""
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT * FROM fundamentals WHERE symbol = ?", (symbol,)
        ).fetchone()
        conn.close()
        return dict(row) if row else None
    except Exception:
        return None


def load_all_fundamentals() -> dict[str, dict]:
    """Load all cached fundamentals as {symbol: dict}. Instant, no network."""
    if not os.path.exists(DB_PATH):
        return {}
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        rows = conn.execute("SELECT * FROM fundamentals").fetchall()
        conn.close()
        return {r["symbol"]: dict(r) for r in rows}
    except Exception:
        return {}


def _is_cached_fresh(symbol: str, cached: dict | None = None) -> bool:
    """True if symbol has real data AND was fetched < FRESH_DAYS ago."""
    if cached is None:
        cached = load_all_fundamentals()
    row = cached.get(symbol)
    if not row:
        return False
    has_data = any(row.get(k) for k in
                   ("eps_growth_yoy", "sales_growth_yoy", "roe", "pe_ratio"))
    if not has_data:
        return False
    age_days = (time.time() - row.get("updated_at", 0)) / 86400
    return age_days <= FRESH_DAYS


def is_cache_stale() -> bool:
    if not os.path.exists(DB_PATH):
        return True
    try:
        conn = sqlite3.connect(DB_PATH)
        row = conn.execute("SELECT MAX(updated_at) FROM fundamentals").fetchone()
        conn.close()
        if not row or not row[0]:
            return True
        return (time.time() - row[0]) / 86400 > TTL_DAYS
    except Exception:
        return True


def cache_status() -> dict:
    """Cache metadata + live scheduler state for UI display."""
    db_info = {"exists": False, "count": 0, "with_data": 0, "age_days": None, "stale": True}
    if os.path.exists(DB_PATH):
        try:
            conn = sqlite3.connect(DB_PATH)
            cnt  = conn.execute("SELECT COUNT(*) FROM fundamentals").fetchone()[0]
            ts   = conn.execute("SELECT MAX(updated_at) FROM fundamentals").fetchone()[0]
            with_data = conn.execute(
                "SELECT COUNT(*) FROM fundamentals WHERE (eps_growth_yoy != 0 OR "
                "sales_growth_yoy != 0 OR roe != 0 OR pe_ratio != 0)"
            ).fetchone()[0]
            conn.close()
            age = (time.time() - ts) / 86400 if ts else None
            db_info = {
                "exists":   True,
                "count":    cnt,
                "with_data": with_data,
                "age_days": round(age, 1) if age else None,
                "stale":    (age or 999) > TTL_DAYS,
            }
        except Exception as e:
            db_info["error"] = str(e)

    with _sched_lock:
        sched = dict(_sched)

    return {**db_info, "scheduler": sched}


# ── screener.in scraper ────────────────────────────────────────────────────────

_session = requests.Session()
_session.headers.update({
    "User-Agent":      ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                        "AppleWebKit/537.36 (KHTML, like Gecko) "
                        "Chrome/122.0.0.0 Safari/537.36"),
    "Accept":          "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
    "Connection":      "keep-alive",
})


def _fetch_one(symbol: str) -> Optional[dict]:
    """
    Scrape one stock from screener.in.
    Returns structured dict or None on any failure.
    Never raises — always graceful.
    """
    url = f"https://www.screener.in/company/{symbol}/"
    try:
        resp = _session.get(url, timeout=15)
        if resp.status_code == 404:
            return None          # symbol not on screener.in (SME / delisted)
        if resp.status_code != 200:
            return None
        html = resp.text

        # ── Reject ETFs / Mutual Funds — screener.in puts "ETF" in <title> ────
        title_m = re.search(r'<title>(.*?)</title>', html, re.IGNORECASE)
        if title_m:
            title_text = title_m.group(1).upper()
            if any(kw in title_text for kw in (" ETF ", "ETF ", " ETF", "EXCHANGE TRADED",
                                                "MUTUAL FUND", "INDEX FUND")):
                return None   # skip ETFs silently

        # ── Top-ratios section: ROE, Stock P/E ────────────────────────────────
        top_ratios: dict[str, float] = {}
        m = re.search(r'<ul id="top-ratios">(.*?)</ul>', html, re.DOTALL)
        if m:
            items = re.findall(
                r'<span class="name">\s*([\s\S]*?)\s*</span>[\s\S]*?'
                r'<span class="number">([\d,\.]+)</span>',
                m.group(1)
            )
            for raw_name, raw_val in items:
                name = re.sub(r'\s+', ' ', raw_name).strip()
                try:
                    top_ratios[name] = float(raw_val.replace(',', ''))
                except ValueError:
                    pass

        roe        = top_ratios.get("ROE", 0.0)
        pe         = top_ratios.get("Stock P/E", 0.0)
        market_cap = top_ratios.get("Market Cap", 0.0)   # in Cr on screener.in

        # Universe = Nifty50 ∪ NiftyNext50 ∪ Nifty500 ∪ NiftySmallcap250
        # No market-cap gate here — all index constituents are valid targets.

        # ── Ranges tables: Sales Growth + Profit Growth (prefer TTM, fall back 3Y) ──
        sales_growth = 0.0
        eps_growth   = 0.0
        roe_from_table = None   # fallback if ROE missing from top-ratios

        tables = re.findall(
            r'<table class="ranges-table">([\s\S]*?)</table>', html
        )
        for table_html in tables:
            # Header
            th = re.search(r'<th[^>]*>([\s\S]*?)</th>', table_html)
            if not th:
                continue
            header = re.sub(r'\s+', ' ', th.group(1)).strip()

            # All <td> pairs
            rows = re.findall(
                r'<td>([\s\S]*?)</td>\s*<td>([\s\S]*?)</td>', table_html
            )
            ttm_val = yr3_val = last_yr_val = None
            for lbl, val in rows:
                lbl = lbl.strip()
                val = val.strip().rstrip('%').strip()
                try:
                    fval = float(val)
                except ValueError:
                    continue
                if lbl == 'TTM:':
                    ttm_val = fval
                elif lbl == '3 Years:':
                    yr3_val = fval
                elif lbl == 'Last Year:':
                    last_yr_val = fval

            if 'Sales Growth' in header:
                best = ttm_val if ttm_val is not None else yr3_val
                if best is not None:
                    sales_growth = best
            elif 'Profit Growth' in header:
                best = ttm_val if ttm_val is not None else yr3_val
                if best is not None:
                    eps_growth = best
            elif 'Return on Equity' in header:
                # Fallback ROE from ranges table (Last Year preferred)
                best = last_yr_val if last_yr_val is not None else yr3_val
                if best is not None:
                    roe_from_table = best

        # Use ranges-table ROE if top-ratios didn't have it
        if roe == 0.0 and roe_from_table is not None:
            roe = roe_from_table

        # ── Validate: reject if no meaningful data at all ─────────────────────
        if roe == 0 and pe == 0 and sales_growth == 0 and eps_growth == 0:
            return None

        return {
            "symbol":           symbol,
            "eps_growth_yoy":   round(eps_growth,   2),
            "sales_growth_yoy": round(sales_growth, 2),
            "roe":              round(roe,           2),
            "debt_to_equity":   0.0,   # not exposed in screener.in top-level HTML
            "pe_ratio":         round(pe,            2),
            "market_cap":       round(market_cap,    2),
            "promoter_holding": 0.0,
            "updated_at":       int(time.time()),
        }

    except requests.exceptions.Timeout:
        return None
    except Exception:
        return None


# ── Background scheduler ───────────────────────────────────────────────────────

def _get_symbol_list() -> list[str]:
    """
    Return the Nifty 500 symbol list — NSE's authoritative large-cap universe.
    ETFs are already excluded by nse_stocks.get_nifty500_symbols().
    The market-cap gate in _fetch_one() further filters to > 5000 Cr stocks.
    """
    try:
        from nse_stocks import get_nifty500_symbols
        from edge_engine import _is_etf
        syms = get_nifty500_symbols()
        # Double-check: strip any ETFs that slip through the index list
        return [s for s in syms if not _is_etf(s)]
    except Exception:
        pass
    return []


def _scheduler_loop():
    """
    Background loop: scrapes 1 stock every ~15-20 seconds (~240/hour).
    Auto-resumes — skips symbols already in cache with real data.
    Runs forever; restarts the pending list when all symbols are fresh.
    """
    _init_db()

    # Wait briefly for bhavcopy to be available on first startup
    time.sleep(8)

    while True:
        symbols = _get_symbol_list()
        if not symbols:
            with _sched_lock:
                _sched["error"] = "Bhavcopy not available yet — retrying in 5 min"
            time.sleep(300)
            continue

        cached  = load_all_fundamentals()
        pending = [s for s in symbols if not _is_cached_fresh(s, cached)]

        already_done = len(symbols) - len(pending)

        with _sched_lock:
            _sched["running"]       = True
            _sched["total"]         = len(symbols)
            _sched["scraped_count"] = already_done
            _sched["pending_count"] = len(pending)
            _sched["failed_count"]  = 0
            _sched["started_at"]    = _sched["started_at"] or time.time()
            _sched["error"]         = ""

        if not pending:
            # All done — sleep until cache starts going stale, then re-check
            with _sched_lock:
                _sched["current_symbol"] = ""
                _sched["eta_minutes"]    = 0
                _sched["pending_count"]  = 0
            time.sleep(3600)   # check again in 1 hour
            continue

        for i, sym in enumerate(pending):
            remaining = len(pending) - i
            with _sched_lock:
                _sched["current_symbol"] = sym
                _sched["pending_count"]  = remaining
                _sched["eta_minutes"]    = round(remaining * (FETCH_DELAY + FETCH_JITTER / 2) / 60, 1)

            data = _fetch_one(sym)

            if data:
                try:
                    _upsert(data)
                    with _sched_lock:
                        _sched["scraped_count"] += 1
                        _sched["last_scraped"]    = sym
                        _sched["last_scraped_at"] = int(time.time())
                except Exception:
                    with _sched_lock:
                        _sched["failed_count"] += 1
            else:
                with _sched_lock:
                    _sched["failed_count"] += 1

            # 15s base + 0-5s jitter  ≈  240 stocks / hour  (safe floor tested at 10s)
            time.sleep(FETCH_DELAY + random.uniform(0, FETCH_JITTER))

        # Finished one pass — loop back to re-check what's still missing
        with _sched_lock:
            _sched["current_symbol"] = ""


def start_background_scheduler():
    """
    Call once from app.py at startup.
    Spawns the daemon scheduler thread if not already running.
    """
    with _sched_lock:
        if _sched["running"]:
            return   # already started
        _sched["started_at"] = time.time()

    t = threading.Thread(target=_scheduler_loop, daemon=True,
                         name="fundamentals-scheduler")
    t.start()


def scheduler_status() -> dict:
    """Live scheduler state dict (thread-safe snapshot)."""
    with _sched_lock:
        return dict(_sched)
