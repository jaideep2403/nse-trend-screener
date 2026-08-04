"""
Fundamentals — screener.in scraper, ~400 stocks/hour, fully background.

Design:
  - Auto-starts on app launch as a daemon thread
  - 8 seconds base delay + random 0-3s jitter between requests (~400 stocks/hour)
  - Jitter makes requests look human — avoids clock-pattern rate limiting
  - Resume-safe: skips symbols already in cache with real data
  - SQLite local cache (fundamentals.db), TTL 90 days
  - Zero yfinance / Yahoo Finance code

Source:  screener.in  (public HTML, 1 req/stock, no login needed)
Cache:   ./fundamentals.db  SQLite

Safe floor confirmed by testing: 10s clean, 5s rate-limited → 8s + jitter is safe.

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
FETCH_DELAY  = 8           # base seconds between requests (tested clean; 5s gets rate-limited)
FETCH_JITTER = 3           # add random 0–FETCH_JITTER seconds — looks human to screener.in
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
                symbol             TEXT PRIMARY KEY,
                eps_growth_yoy     REAL,
                sales_growth_yoy   REAL,
                roe                REAL,
                debt_to_equity     REAL,
                pe_ratio           REAL,
                market_cap         REAL,
                promoter_holding   REAL,
                updated_at         INTEGER,
                -- BUG-029: split TTM vs 3Y CAGR (early_growth.py reads these)
                growth_ttm         REAL,
                growth_3y_cagr     REAL,
                sales_growth_ttm   REAL,
                sales_growth_3y_cagr REAL,
                -- F2: quarterly EPS acceleration (8 quarters for YoY comparison)
                eps_q1             REAL,
                eps_q2             REAL,
                eps_q3             REAL,
                eps_q4             REAL,
                eps_q5             REAL,
                eps_q6             REAL,
                eps_q7             REAL,
                eps_q8             REAL,
                eps_accel          INTEGER,
                result_date        TEXT,
                -- F3: promoter holding delta
                promoter_prev      REAL,
                promoter_delta     REAL
            )
        """)
        # Add new columns to existing DB (safe on re-run)
        for col_def in [
            "eps_q1 REAL", "eps_q2 REAL", "eps_q3 REAL", "eps_q4 REAL",
            # YoY comparison quarters (same quarter, prior year) — added later
            "eps_q5 REAL", "eps_q6 REAL", "eps_q7 REAL", "eps_q8 REAL",
            # BUG-029: TTM vs 3Y CAGR split (early_growth.py depends on these)
            "growth_ttm REAL", "growth_3y_cagr REAL",
            "sales_growth_ttm REAL", "sales_growth_3y_cagr REAL",
            "eps_accel INTEGER", "result_date TEXT",
            "promoter_prev REAL", "promoter_delta REAL", "opm REAL",
            # BUG-SEASONAL FIX: YoY quarterly profit acceleration (q1 vs q5)
            "eps_accel_yoy INTEGER",
        ]:
            col_name = col_def.split()[0]
            try:
                conn.execute(f"ALTER TABLE fundamentals ADD COLUMN {col_def}")
            except Exception:
                pass   # column already exists
        conn.commit()
        conn.close()


def _upsert(data: dict):
    with _db_lock:
        conn = sqlite3.connect(DB_PATH)
        conn.execute("""
            INSERT OR REPLACE INTO fundamentals
            (symbol, eps_growth_yoy, sales_growth_yoy, roe,
             debt_to_equity, pe_ratio, market_cap, promoter_holding, updated_at,
             growth_ttm, growth_3y_cagr, sales_growth_ttm, sales_growth_3y_cagr,
             eps_q1, eps_q2, eps_q3, eps_q4, eps_q5, eps_q6, eps_q7, eps_q8,
             eps_accel, result_date,
             promoter_prev, promoter_delta, eps_accel_yoy, opm)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (data["symbol"], data["eps_growth_yoy"], data["sales_growth_yoy"],
              data["roe"], data["debt_to_equity"], data["pe_ratio"],
              data["market_cap"], data["promoter_holding"], data["updated_at"],
              data.get("growth_ttm"), data.get("growth_3y_cagr"),
              data.get("sales_growth_ttm"), data.get("sales_growth_3y_cagr"),
              data.get("eps_q1"), data.get("eps_q2"), data.get("eps_q3"), data.get("eps_q4"),
              data.get("eps_q5"), data.get("eps_q6"), data.get("eps_q7"), data.get("eps_q8"),
              data.get("eps_accel"), data.get("result_date"),
              data.get("promoter_prev"), data.get("promoter_delta"),
              data.get("eps_accel_yoy"), data.get("opm")))
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
    """
    True if symbol has real data AND was fetched < FRESH_DAYS ago AND
    quarterly data is populated (eps_q1 not None).

    Stocks scraped before the eps_q1 regex fix will have eps_q1=None —
    treat them as stale so the scheduler re-scrapes them at the normal
    15-20s rate. No extra API hammering; just re-queues missing data.
    """
    if cached is None:
        cached = load_all_fundamentals()
    row = cached.get(symbol)
    if not row:
        return False
    has_data = any(row.get(k) for k in
                   ("eps_growth_yoy", "sales_growth_yoy", "roe", "pe_ratio"))
    if not has_data:
        return False
    # If quarterly data is missing, or YoY quarters missing (scraped before q5-q8 columns),
    # treat as stale so scheduler re-scrapes with full 8-quarter history.
    if row.get("eps_q1") is None or row.get("eps_q5") is None:
        return False
    # BUG-029: TTM/3Y growth columns added later — force re-scrape if missing
    if row.get("growth_ttm") is None and row.get("growth_3y_cagr") is None:
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
        if resp.status_code == 429:
            # Rate-limited — back off 5 minutes then retry once.
            # 5 min is enough for screener.in's sliding window to clear.
            print(f"[fundamentals] 429 rate-limited for {symbol}, sleeping 5 min…", flush=True)
            time.sleep(300)
            resp2 = _session.get(url, timeout=15)
            if resp2.status_code != 200:
                # Still blocked — skip this stock, will retry next scheduler pass
                print(f"[fundamentals] still blocked after backoff, skipping {symbol}", flush=True)
                return None
            html = resp2.text
        elif resp.status_code == 503:
            # Server overloaded — wait 3 min
            print(f"[fundamentals] 503 for {symbol}, sleeping 3 min…", flush=True)
            time.sleep(180)
            return None
        elif resp.status_code != 200:
            return None
        else:
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

        # Universe = Nifty Total Market 750 (Nifty50 ∪ Next50 ∪ Nifty500 ∪ Smallcap250 ∪ Microcap250 ∪ TotalMarket)
        # No market-cap gate here — all index constituents are valid targets.

        # ── Ranges tables: Sales Growth + Profit Growth (prefer TTM, fall back 3Y) ──
        # BUG-029 FIX: previously TTM and 3Y CAGR were conflated into a
        # single growth number ("best available"). Now we keep them
        # separate so consumers can pick the right one (TTM = momentum,
        # 3Y CAGR = sustained growth). `eps_growth_yoy` / `sales_growth_yoy`
        # continue to reflect the TTM-preferred number for back-compat,
        # but `*_ttm` and `*_3y_cagr` are exposed as well.
        sales_growth = 0.0
        eps_growth   = 0.0
        sales_growth_ttm    = None
        sales_growth_3y     = None
        eps_growth_ttm      = None
        eps_growth_3y       = None
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
                sales_growth_ttm = ttm_val
                sales_growth_3y  = yr3_val
                best = ttm_val if ttm_val is not None else yr3_val
                if best is not None:
                    sales_growth = best
            elif 'Profit Growth' in header:
                eps_growth_ttm = ttm_val
                eps_growth_3y  = yr3_val
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

        # ── OPM % (operating profit margin) — the "M" in the SMR rating ───────
        # screener.in doesn't expose OPM in top-ratios; it lives as an "OPM %" row
        # in the quarterly/annual P&L tables. Take the most recent column.
        # CRITICAL: parse this WITHOUT a nested-quantifier regex. The obvious
        # `<tr>…<td>OPM %</td>((?:\s*<td[^>]*>[\s\S]*?</td>)+)</tr>` pattern
        # catastrophically backtracks on screener.in's whitespace-heavy tables —
        # it doesn't fail, it HANGS, which would freeze the whole scraper thread
        # on the first stock. Locate the label, take a BOUNDED slice, then a flat
        # findall: same result, 0.4 ms, no backtracking possible.
        opm = None
        try:
            m_lbl = re.search(r'<td[^>]*>\s*OPM\s*%\s*</td>', html, re.IGNORECASE)
            if m_lbl:
                seg = html[m_lbl.end(): m_lbl.end() + 4000]
                cut = seg.find('</tr>')
                if cut != -1:
                    seg = seg[:cut]
                for v in reversed(re.findall(
                        r'<td[^>]*>\s*(-?[\d,\.]+)\s*%?\s*</td>', seg)):
                    try:
                        opm = float(v.replace(',', ''))   # most recent column
                        break
                    except ValueError:
                        continue
        except Exception:
            opm = None

        # ── F2: Quarterly NET PROFIT (NOT EPS!) — parse from #quarters section ─
        # BUG-014 FIX / NOTE: the fields below are mislabelled "eps_q1..4"
        # for historical reasons but are actually QUARTERLY NET PROFIT in
        # rupees crore — share-count is NOT normalised, so this signal
        # under-/over-states true per-share EPS when shares outstanding
        # change (buybacks, fresh issuance, splits). Treated here as a
        # profit-acceleration proxy. Aliased to profit_q1..4 in the return
        # dict for new consumers; old `eps_q*` keys remain for back-compat.
        eps_quarters: list[float] = []
        result_date_str = None
        m_qtr = re.search(r'<section[^>]+id=["\']quarters["\'][^>]*>([\s\S]*?)</section>', html)
        if m_qtr:
            qtr_html = m_qtr.group(1)
            # Extract header dates (e.g. "Sep 2024", "Dec 2024")
            headers = re.findall(r'<th[^>]*>([\w\s]+\d{4})</th>', qtr_html)
            if headers:
                result_date_str = headers[0].strip()   # most recent quarter
            # Find Net Profit row
            np_match = re.search(
                r'<tr[^>]*>[\s\S]*?Net Profit[\s\S]*?((?:<td[^>]*>[\s\S]*?</td>\s*)+)</tr>',
                qtr_html, re.IGNORECASE
            )
            if np_match:
                # BUG-FIX: screener.in wraps cell values in whitespace/newlines;
                # must strip \s* around the value or regex returns empty list.
                td_vals = re.findall(r'<td[^>]*>\s*([\d,\.\-]+)\s*</td>', np_match.group(1))
                # BUG-SEASONAL FIX: parse up to 8 quarters (was 4) so we can
                # compare the most-recent quarter (q1) against the same quarter
                # one year ago (q5) — a YoY signal that is immune to seasonal bias.
                for v in td_vals[:8]:
                    try:
                        eps_quarters.append(float(v.replace(',', '')))
                    except ValueError:
                        pass

        # EPS acceleration: each quarter higher than the previous
        eps_q1 = eps_quarters[0] if len(eps_quarters) > 0 else None
        eps_q2 = eps_quarters[1] if len(eps_quarters) > 1 else None
        eps_q3 = eps_quarters[2] if len(eps_quarters) > 2 else None
        eps_q4 = eps_quarters[3] if len(eps_quarters) > 3 else None
        # SEASONAL FIX: Q1>Q2>Q3 confuses seasonal cycles with genuine acceleration
        # (e.g. PGIL garments: Dec always > Sep always > Jun — that's seasonality, not growth).
        # True acceleration = at least 2 of the last 3 quarters improving vs same quarter YoY.
        # q1 vs q5, q2 vs q6, q3 vs q7 (each vs same quarter prior year).
        eps_q5 = eps_quarters[4] if len(eps_quarters) > 4 else None
        eps_q6 = eps_quarters[5] if len(eps_quarters) > 5 else None
        eps_q7 = eps_quarters[6] if len(eps_quarters) > 6 else None

        eps_accel = None
        yoy_beats = []
        if eps_q1 is not None and eps_q5 is not None:
            yoy_beats.append(1 if eps_q1 > eps_q5 else 0)
        if eps_q2 is not None and eps_q6 is not None:
            yoy_beats.append(1 if eps_q2 > eps_q6 else 0)
        if eps_q3 is not None and eps_q7 is not None:
            yoy_beats.append(1 if eps_q3 > eps_q7 else 0)
        if len(yoy_beats) >= 2:
            # Accelerating = majority of recent quarters beating same quarter prior year
            eps_accel = 1 if sum(yoy_beats) >= 2 else 0
        elif len(yoy_beats) == 1:
            eps_accel = yoy_beats[0]   # only one comparable quarter available

        # YoY acceleration on most recent quarter alone (single-quarter signal)
        eps_accel_yoy = None
        if eps_q1 is not None and eps_q5 is not None:
            eps_accel_yoy = 1 if eps_q1 > eps_q5 else 0

        # ── F3: Promoter Holding Delta — parse shareholding section ───────────
        promoter_holding  = 0.0
        promoter_prev     = None
        promoter_delta    = None
        m_sh = re.search(r'<section[^>]+id=["\']shareholding["\'][^>]*>([\s\S]*?)</section>', html)
        if m_sh:
            sh_html = m_sh.group(1)
            prom_match = re.search(
                r'<tr[^>]*>[\s\S]*?Promoters[\s\S]*?((?:<td[^>]*>[\s\S]*?</td>\s*)+)</tr>',
                sh_html, re.IGNORECASE
            )
            if prom_match:
                # BUG-FIX: screener.in wraps values in whitespace AND appends '%'
                # e.g. <td>66.56%</td> — must strip \s* and allow optional %
                td_vals = re.findall(r'<td[^>]*>\s*([\d,\.]+)%?\s*</td>', prom_match.group(1))
                promo_vals = []
                for v in td_vals[:4]:
                    try:
                        promo_vals.append(float(v.replace(',', '')))
                    except ValueError:
                        pass
                if len(promo_vals) >= 1:
                    promoter_holding = promo_vals[0]
                if len(promo_vals) >= 2:
                    promoter_prev  = promo_vals[1]
                    promoter_delta = round(promo_vals[0] - promo_vals[1], 2)

        # Fallback: promoter from top_ratios if not found in shareholding table
        if promoter_holding == 0.0:
            promoter_holding = top_ratios.get("Promoter holding", 0.0)

        # ── Validate: reject ONLY when every metric is missing from the page.
        # The previous check `roe == 0 and pe == 0 and sales_growth == 0 and
        # eps_growth == 0` rejected legitimate turnaround/loss-recovery names
        # whose actual values happen to be 0. Distinguish "not found in HTML"
        # (top_ratios/missing) from "found and equal to 0.0".
        found_any = (
            "ROE" in top_ratios
            or "Stock P/E" in top_ratios
            or sales_growth_ttm is not None
            or sales_growth_3y  is not None
            or eps_growth_ttm   is not None
            or eps_growth_3y    is not None
            or roe_from_table   is not None
        )
        if not found_any:
            return None

        # BUG-004 FIX: D/E is not reliably present in screener.in top-level HTML.
        # Return None so the UI can display "—" instead of misleading "0.0".
        # TODO: Parse D/E from the "Debt to equity" row in the top-ratios section
        #       if screener.in ever exposes it consistently.
        de_ratio = top_ratios.get("Debt to equity") or top_ratios.get("D/E") or None

        return {
            "symbol":           symbol,
            "eps_growth_yoy":   round(eps_growth,   2),
            "sales_growth_yoy": round(sales_growth, 2),
            # BUG-029: split TTM vs 3Y CAGR
            "growth_ttm":         round(eps_growth_ttm, 2)   if eps_growth_ttm   is not None else None,
            "growth_3y_cagr":     round(eps_growth_3y,  2)   if eps_growth_3y    is not None else None,
            "sales_growth_ttm":   round(sales_growth_ttm, 2) if sales_growth_ttm is not None else None,
            "sales_growth_3y_cagr": round(sales_growth_3y, 2) if sales_growth_3y is not None else None,
            "roe":              round(roe,           2),
            "debt_to_equity":   round(de_ratio, 2) if de_ratio is not None else None,
            "pe_ratio":         round(pe,            2),
            "market_cap":       round(market_cap,    2),
            "promoter_holding": round(promoter_holding, 2),
            "updated_at":       int(time.time()),
            # F2 — quarterly EPS
            # BUG-014 — keep historical eps_q* keys (DB-backed) and expose
            # profit_q* aliases so new code can use the correct name.
            "eps_q1":           eps_q1,
            "eps_q2":           eps_q2,
            "eps_q3":           eps_q3,
            "eps_q4":           eps_q4,
            # YoY comparison quarters (same quarter, one year ago)
            "eps_q5":           eps_q5,
            "eps_q6":           eps_q6,
            "eps_q7":           eps_q7,
            "eps_q8":           eps_quarters[7] if len(eps_quarters) > 7 else None,
            "profit_q1":        eps_q1,
            "profit_q2":        eps_q2,
            "profit_q3":        eps_q3,
            "profit_q4":        eps_q4,
            "eps_accel":        eps_accel,
            "profit_accel":     eps_accel,
            # BUG-SEASONAL FIX: YoY quarterly acceleration (q1 vs same quarter prior year)
            "eps_accel_yoy":    eps_accel_yoy,
            "profit_accel_yoy": eps_accel_yoy,
            "result_date":      result_date_str,
            # F3 — promoter delta
            "promoter_prev":    promoter_prev,
            "promoter_delta":   promoter_delta,
            # SMR "M" leg
            "opm":              round(opm, 2) if opm is not None else None,
        }

    except requests.exceptions.Timeout:
        return None
    except Exception:
        return None


# ── Background scheduler ───────────────────────────────────────────────────────

def _get_symbol_list() -> list[str]:
    """
    Return the Nifty Total Market 750 symbol list (~751 stocks).
    ETFs are already excluded by nse_stocks.get_universe_symbols().
    The market-cap gate in _fetch_one() further filters to > 5000 Cr stocks.
    """
    try:
        from nse_stocks import get_universe_symbols
        from edge_engine import _is_etf
        syms = get_universe_symbols()
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
    BUG-010 FIX: Set _sched["running"] = True INSIDE the lock BEFORE spawning
    the thread. Previously _sched["running"] was only set inside the thread
    (after the initial 8-second sleep), allowing a second caller to race in
    and spawn a duplicate thread during that window.
    """
    with _sched_lock:
        if _sched["running"]:
            return   # already started — guard under lock prevents race
        # Mark running NOW (before spawn) so concurrent callers see it immediately
        _sched["running"]    = True
        _sched["started_at"] = time.time()

    t = threading.Thread(target=_scheduler_loop, daemon=True,
                         name="fundamentals-scheduler")
    t.start()


def scheduler_status() -> dict:
    """Live scheduler state dict (thread-safe snapshot)."""
    with _sched_lock:
        return dict(_sched)
