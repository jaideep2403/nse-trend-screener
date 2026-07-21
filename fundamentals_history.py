"""
Point-in-time fundamentals capture.

WHY THIS EXISTS
---------------
`fundamentals.db` holds only the LATEST snapshot per stock. Applying today's ROE /
growth / quality to a HISTORICAL rebalance is therefore look-ahead biased — which
is exactly why the All-Weather "quality tilt" backtest is only an optimistic upper
bound, not a survivorship-free proof.

This module fixes that GOING FORWARD: it periodically snapshots the current
fundamentals into a dated history table (`fundamentals_history`), so we accumulate
a genuine point-in-time record — what each stock's fundamentals WERE on each
capture date. Once enough snapshots exist (a few quarters is ideal, since
fundamentals change quarterly), `as_of(symbol, date)` lets a backtest look up the
fundamentals KNOWN AS OF each rebalance, with zero look-ahead. Cheap, no network:
it just copies rows the screener.in scraper already stored.

Cadence: WEEKLY is plenty for a quarterly-changing factor and keeps the table tiny
(~750 rows/week). Snapshots are idempotent per (symbol, snapshot_date).
"""
from __future__ import annotations

import os
import sqlite3
import time
import threading
from datetime import date, datetime

DB_PATH = os.path.join(os.environ.get("DATA_DIR", os.path.dirname(__file__)),
                       "fundamentals.db")
TABLE = "fundamentals_history"
CAPTURE_EVERY_DAYS = 7

# Raw fields copied verbatim from the live fundamentals table (point-in-time record).
FIELDS = ["roe", "growth_3y_cagr", "growth_ttm", "eps_growth_yoy", "sales_growth_yoy",
          "eps_accel_yoy", "promoter_holding", "promoter_delta", "pe_ratio", "market_cap"]

_lock = threading.Lock()
_sched = {"running": False, "last_capture": None, "snapshots": 0, "rows": 0, "error": ""}


def _init_db():
    with _lock:
        conn = sqlite3.connect(DB_PATH)
        cols = ", ".join(f"{f} REAL" for f in FIELDS)
        conn.execute(f"""
            CREATE TABLE IF NOT EXISTS {TABLE} (
                symbol        TEXT,
                snapshot_date TEXT,
                {cols},
                quality       REAL,
                captured_at   INTEGER,
                PRIMARY KEY (symbol, snapshot_date)
            )
        """)
        conn.execute(f"CREATE INDEX IF NOT EXISTS idx_{TABLE}_date ON {TABLE}(snapshot_date)")
        conn.commit()
        conn.close()


def capture_snapshot(snapshot_date: str | None = None) -> dict:
    """Copy the current fundamentals table into the dated history table. Idempotent
    per (symbol, date) — re-running the same day overwrites, never duplicates."""
    _init_db()
    d = snapshot_date or date.today().isoformat()
    # Read the live snapshot.
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        live = [dict(r) for r in conn.execute(
            f"SELECT symbol, {', '.join(FIELDS)} FROM fundamentals").fetchall()]
        conn.close()
    except Exception as e:
        return {"error": f"read fundamentals failed: {e}", "date": d, "rows": 0}
    if not live:
        return {"error": "no fundamentals to snapshot", "date": d, "rows": 0}

    # Quality (computed with the CURRENT formula, stored for convenience; the raw
    # fields are also stored so quality can be re-derived if the formula changes).
    try:
        import quality as _q
        qmap = _q.load_quality_map(refresh=True)
    except Exception:
        qmap = {}

    now = int(time.time())
    placeholders = ", ".join(["?"] * (len(FIELDS) + 4))  # symbol, date, fields, quality, captured_at
    cols = f"symbol, snapshot_date, {', '.join(FIELDS)}, quality, captured_at"
    with _lock:
        conn = sqlite3.connect(DB_PATH)
        for r in live:
            vals = [r["symbol"], d] + [r.get(f) for f in FIELDS] + [qmap.get(r["symbol"]), now]
            conn.execute(f"INSERT OR REPLACE INTO {TABLE} ({cols}) VALUES ({placeholders})", vals)
        conn.commit()
        conn.close()
    return {"date": d, "rows": len(live)}


def snapshot_dates() -> list[str]:
    if not os.path.exists(DB_PATH):
        return []
    try:
        conn = sqlite3.connect(DB_PATH)
        rows = conn.execute(
            f"SELECT DISTINCT snapshot_date FROM {TABLE} ORDER BY snapshot_date").fetchall()
        conn.close()
        return [r[0] for r in rows]
    except Exception:
        return []


def latest_snapshot_date() -> str | None:
    ds = snapshot_dates()
    return ds[-1] if ds else None


def as_of(symbol: str, on_date: str) -> dict | None:
    """Most recent snapshot of `symbol` on or BEFORE `on_date` — the point-in-time
    lookup a bias-free backtest uses. None if we had no data for it yet by then."""
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            f"SELECT * FROM {TABLE} WHERE symbol = ? AND snapshot_date <= ? "
            f"ORDER BY snapshot_date DESC LIMIT 1", (symbol, on_date)).fetchone()
        conn.close()
        return dict(row) if row else None
    except Exception:
        return None


def coverage() -> dict:
    ds = snapshot_dates()
    rows = 0
    if ds:
        try:
            conn = sqlite3.connect(DB_PATH)
            rows = conn.execute(f"SELECT COUNT(*) FROM {TABLE}").fetchone()[0]
            conn.close()
        except Exception:
            rows = 0
    span_days = None
    if len(ds) >= 2:
        span_days = (datetime.fromisoformat(ds[-1]) - datetime.fromisoformat(ds[0])).days
    with _lock:
        sched = dict(_sched)
    return {"snapshots": len(ds), "rows": rows, "first": ds[0] if ds else None,
            "last": ds[-1] if ds else None, "span_days": span_days,
            "dates": ds, "scheduler": sched,
            "ready_for_validation": len(ds) >= 8 and (span_days or 0) >= 90}


def _days_since(d: str | None) -> int:
    if not d:
        return 10 ** 6
    try:
        return (date.today() - date.fromisoformat(d)).days
    except Exception:
        return 10 ** 6


def _loop():
    _init_db()
    time.sleep(30)   # let the fundamentals scraper populate on a cold boot
    while True:
        try:
            last = latest_snapshot_date()
            if _days_since(last) >= CAPTURE_EVERY_DAYS:
                res = capture_snapshot()
                with _lock:
                    _sched["last_capture"] = res.get("date")
                    _sched["rows"] = res.get("rows", 0)
                    _sched["snapshots"] = len(snapshot_dates())
                    _sched["error"] = res.get("error", "")
        except Exception as e:
            with _lock:
                _sched["error"] = str(e)
        time.sleep(24 * 3600)   # check once a day; captures at most weekly


def start_snapshot_scheduler():
    """Idempotent — spawn the daily-check / weekly-capture daemon once."""
    with _lock:
        if _sched["running"]:
            return
        _sched["running"] = True
    threading.Thread(target=_loop, daemon=True, name="fundamentals-history").start()
