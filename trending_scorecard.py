"""
Trending Scorecard — the honesty loop for the Trending tab.

Every scan logs the ranked trending list (one row per stock per BHAVCOPY date,
idempotent — re-scans of the same trading day never duplicate or overwrite).
On subsequent scans, forward 5/10/20-trading-day returns and benchmark-adjusted
alpha are filled in once enough future bars exist. summary() then answers the
only question that matters: did the stocks this tab surfaced actually beat
the market, and does the Conviction rank separate winners from losers?

Measurement convention (documented so numbers are interpretable):
  - Entry price  = the scan-date CLOSE (the price the score was computed on).
  - Forward ret  = close-to-close over N TRADING bars of that stock's series.
  - Alpha        = forward ret − NIFTYBEES return over the same dates.
No costs are subtracted here — this measures signal quality, not a backtest.
"""
from __future__ import annotations

import sqlite3
import time
from pathlib import Path

import pandas as pd

DB_PATH = Path(__file__).parent / "trending_scorecard.db"

_HORIZONS = (5, 10, 20)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS scorecard (
    scan_date    TEXT NOT NULL,      -- bhavcopy date the scan was computed on
    symbol       TEXT NOT NULL,
    rank         INTEGER,
    score        REAL,               -- 0-10 trend-structure score
    conviction   REAL,               -- 0-100 evidence-based conviction rank
    price        REAL,               -- scan-date close (entry for measurement)
    adtv_cr      REAL,
    sector       TEXT,
    entry_window TEXT,
    logged_at    INTEGER,
    fwd5    REAL, fwd10  REAL, fwd20  REAL,
    alpha5  REAL, alpha10 REAL, alpha20 REAL,
    PRIMARY KEY (scan_date, symbol)
);
CREATE INDEX IF NOT EXISTS idx_scorecard_pending
    ON scorecard (scan_date) WHERE alpha20 IS NULL;
"""


def _conn() -> sqlite3.Connection:
    c = sqlite3.connect(DB_PATH, timeout=10)
    c.executescript(_SCHEMA)
    return c


def log_scan(results: list[dict], scan_date: str) -> int:
    """Idempotent insert of the ranked trending list for one bhavcopy date.
    INSERT OR IGNORE: history is append-only, a re-scan never rewrites it."""
    if not results or not scan_date:
        return 0
    rows = [(
        scan_date,
        r.get("symbol"),
        r.get("rank"),
        r.get("score"),
        r.get("conviction"),
        r.get("price"),
        r.get("adtv_cr"),
        r.get("sector"),
        r.get("entry_window"),
        int(time.time()),
    ) for r in results if r.get("symbol")]
    with _conn() as c:
        cur = c.executemany(
            "INSERT OR IGNORE INTO scorecard "
            "(scan_date, symbol, rank, score, conviction, price, adtv_cr, "
            " sector, entry_window, logged_at) "
            "VALUES (?,?,?,?,?,?,?,?,?,?)", rows)
        return cur.rowcount


def fill_forward_returns(stocks: dict, bench: pd.Series | None) -> int:
    """Fill fwd/alpha columns for logged rows whose horizons have elapsed.
    `stocks` = {symbol: OHLCV df} (split-adjusted), `bench` = NIFTYBEES closes.
    Only fills NULL cells; never overwrites a previously measured value."""
    if not stocks:
        return 0
    with _conn() as c:
        pending = c.execute(
            "SELECT scan_date, symbol, price FROM scorecard "
            "WHERE alpha20 IS NULL").fetchall()
        if not pending:
            return 0
        filled = 0
        for scan_date, symbol, entry_price in pending:
            df = stocks.get(symbol)
            if df is None or entry_price is None or entry_price <= 0:
                continue
            close = df["Close"].dropna()
            if not isinstance(close.index, pd.DatetimeIndex):
                continue
            try:
                ts = pd.Timestamp(scan_date)
                # position of the scan date bar (exact match expected; asof-safe)
                pos_arr = close.index.get_indexer([ts])
                pos = int(pos_arr[0])
                if pos < 0:
                    aligned = close.index.asof(ts)
                    if pd.isna(aligned):
                        continue
                    pos = int(close.index.get_loc(aligned))
            except Exception:
                continue

            updates: dict[str, float] = {}
            for h in _HORIZONS:
                if pos + h >= len(close):
                    continue           # horizon not elapsed yet
                exit_p = float(close.iloc[pos + h])
                fwd = (exit_p / float(entry_price) - 1) * 100
                updates[f"fwd{h}"] = round(fwd, 2)
                if bench is not None and len(bench) > 2:
                    b0 = bench.asof(close.index[pos])
                    b1 = bench.asof(close.index[pos + h])
                    if not (pd.isna(b0) or pd.isna(b1)) and float(b0) > 0:
                        bret = (float(b1) / float(b0) - 1) * 100
                        updates[f"alpha{h}"] = round(fwd - bret, 2)
            if not updates:
                continue
            sets = ", ".join(f"{k} = COALESCE({k}, ?)" for k in updates)
            c.execute(f"UPDATE scorecard SET {sets} "
                      "WHERE scan_date = ? AND symbol = ?",
                      (*updates.values(), scan_date, symbol))
            filled += 1
        return filled


def summary() -> dict:
    """Bucketed honest performance of past trending picks.

    Buckets by conviction quartile (the v3 rank we now sort by) AND by
    legacy score band, each with n / avg fwd20 / avg alpha20 / win rate.
    """
    with _conn() as c:
        total, measured = c.execute(
            "SELECT COUNT(*), COUNT(alpha20) FROM scorecard").fetchone()
        dates = c.execute(
            "SELECT COUNT(DISTINCT scan_date), MIN(scan_date), MAX(scan_date) "
            "FROM scorecard").fetchone()

        def _bucket(where: str, label: str) -> dict | None:
            row = c.execute(
                f"SELECT COUNT(alpha20), AVG(fwd20), AVG(alpha20), "
                f"       AVG(CASE WHEN alpha20 > 0 THEN 1.0 ELSE 0.0 END), "
                f"       AVG(fwd5), AVG(alpha5) "
                f"FROM scorecard WHERE alpha20 IS NOT NULL AND ({where})"
            ).fetchone()
            n = row[0] or 0
            if n == 0:
                return None
            return {
                "bucket":     label,
                "n":          n,
                "avg_fwd20":  round(row[1], 2),
                "avg_alpha20": round(row[2], 2),
                "beat_bench_pct": round(row[3] * 100, 1),
                "avg_fwd5":   round(row[4], 2) if row[4] is not None else None,
                "avg_alpha5": round(row[5], 2) if row[5] is not None else None,
            }

        by_conviction = [b for b in (
            _bucket("conviction >= 75", "Conviction 75-100"),
            _bucket("conviction >= 50 AND conviction < 75", "Conviction 50-75"),
            _bucket("conviction >= 25 AND conviction < 50", "Conviction 25-50"),
            _bucket("conviction < 25", "Conviction 0-25"),
        ) if b]
        by_score = [b for b in (
            _bucket("score >= 8", "Score 8-10"),
            _bucket("score >= 6 AND score < 8", "Score 6-8"),
            _bucket("score < 6", "Score < 6"),
        ) if b]

    return {
        "rows_logged":    total,
        "rows_measured":  measured,
        "scan_dates":     dates[0],
        "first_scan":     dates[1],
        "last_scan":      dates[2],
        "by_conviction":  by_conviction,
        "by_score":       by_score,
        "note": ("Forward returns are close-to-close vs scan-date close; "
                 "alpha vs NIFTYBEES; no costs. Fills in as future data "
                 "arrives — needs ≥20 trading days of history to populate."),
    }
