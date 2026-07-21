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

# Columns added after the initial schema shipped. sqlite has no
# ADD COLUMN IF NOT EXISTS, so each is applied best-effort on connect.
_MIGRATIONS = [
    "ALTER TABLE scorecard ADD COLUMN retrace_10d REAL",   # fade state at scan time
]


def _conn() -> sqlite3.Connection:
    c = sqlite3.connect(DB_PATH, timeout=10)
    c.executescript(_SCHEMA)
    for mig in _MIGRATIONS:
        try:
            c.execute(mig)
        except sqlite3.OperationalError:
            pass   # column already exists
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
        r.get("retrace_10d"),
        int(time.time()),
    ) for r in results if r.get("symbol")]
    with _conn() as c:
        cur = c.executemany(
            "INSERT OR IGNORE INTO scorecard "
            "(scan_date, symbol, rank, score, conviction, price, adtv_cr, "
            " sector, entry_window, retrace_10d, logged_at) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?)", rows)
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


# Buckets with fewer measured rows than this are flagged low_sample — the UI
# must present them as "insufficient data", not as evidence.
MIN_BUCKET_N = 30


def summary() -> dict:
    """Bucketed honest performance of past trending picks — the report card.

    Four bucket groups (entry_window, rank band, conviction band, score band),
    each measured at EVERY horizon (5/10/20 trading days) with:
      n, avg fwd ret, avg alpha, win rate (alpha>0), and p10 alpha (the
      worst-decile outcome — the tail-risk stat the GVT&D episode taught us
      matters more than the mean).
    """
    with _conn() as c:
        df = pd.read_sql_query(
            "SELECT rank, score, conviction, entry_window, "
            "       fwd5, fwd10, fwd20, alpha5, alpha10, alpha20 "
            "FROM scorecard", c)
        dates = c.execute(
            "SELECT COUNT(DISTINCT scan_date), MIN(scan_date), MAX(scan_date) "
            "FROM scorecard").fetchone()

    total = int(len(df))
    measured_by_h = {h: int(df[f"alpha{h}"].notna().sum()) for h in _HORIZONS}

    def _stats(sub: pd.DataFrame, label: str) -> dict | None:
        if not len(sub):
            return None
        out = {"bucket": label, "horizons": {}}
        best_n = 0
        for h in _HORIZONS:
            a = sub[f"alpha{h}"].dropna()
            f = sub[f"fwd{h}"].dropna()
            n = int(len(a))
            best_n = max(best_n, n)
            if n == 0:
                out["horizons"][str(h)] = None
                continue
            out["horizons"][str(h)] = {
                "n":          n,
                "avg_fwd":    round(float(f.mean()), 2),
                "avg_alpha":  round(float(a.mean()), 2),
                "win_pct":    round(float((a > 0).mean()) * 100, 1),
                "p10_alpha":  round(float(a.quantile(0.10)), 2),
                "low_sample": n < MIN_BUCKET_N,
            }
        return out if best_n > 0 else None

    ew_order = ["pullback", "fresh_breakout", "in_range", "extended", "weakening"]
    seen_ew = [e for e in ew_order if e in set(df["entry_window"].dropna())]
    seen_ew += sorted(set(df["entry_window"].dropna()) - set(ew_order))
    by_entry_window = [b for b in (
        _stats(df[df["entry_window"] == e], e) for e in seen_ew) if b]

    by_rank = [b for b in (
        _stats(df[df["rank"] <= 10], "Rank 1-10"),
        _stats(df[(df["rank"] > 10) & (df["rank"] <= 25)], "Rank 11-25"),
        _stats(df[(df["rank"] > 25) & (df["rank"] <= 50)], "Rank 26-50"),
        _stats(df[df["rank"] > 50], "Rank 51+"),
    ) if b]

    by_conviction = [b for b in (
        _stats(df[df["conviction"] >= 75], "Conviction 75-100"),
        _stats(df[(df["conviction"] >= 50) & (df["conviction"] < 75)], "Conviction 50-75"),
        _stats(df[(df["conviction"] >= 25) & (df["conviction"] < 50)], "Conviction 25-50"),
        _stats(df[df["conviction"] < 25], "Conviction 0-25"),
    ) if b]

    by_score = [b for b in (
        _stats(df[df["score"] >= 8], "Score 8-10"),
        _stats(df[(df["score"] >= 6) & (df["score"] < 8)], "Score 6-8"),
        _stats(df[df["score"] < 6], "Score < 6"),
    ) if b]

    return {
        "rows_logged":    total,
        "rows_measured":  measured_by_h.get(20, 0),   # back-compat: 20d count
        "measured_by_horizon": {str(h): n for h, n in measured_by_h.items()},
        "min_bucket_n":   MIN_BUCKET_N,
        "scan_dates":     dates[0],
        "first_scan":     dates[1],
        "last_scan":      dates[2],
        "by_entry_window": by_entry_window,
        "by_rank":         by_rank,
        "by_conviction":   by_conviction,
        "by_score":        by_score,
        "note": ("All picks logged since first_scan. Forward returns are "
                 "close-to-close vs scan-date close; alpha vs NIFTYBEES; no "
                 "costs. p10 = worst-decile alpha (tail risk). Buckets under "
                 f"{MIN_BUCKET_N} measured rows are low-sample — not evidence."),
    }


def rank_deltas(current_scan_date: str, lookback_days: int = 5) -> dict[str, dict]:
    """{symbol: {rank_prev, delta_rank, delta_score, prev_date}} comparing the
    latest logged scan at/before `current_scan_date` with the scan closest to
    `lookback_days` TRADING days earlier (by distinct scan_date ordering).
    Symbols absent from the earlier scan (new entrants) are omitted."""
    with _conn() as c:
        ds = [r[0] for r in c.execute(
            "SELECT DISTINCT scan_date FROM scorecard "
            "WHERE scan_date <= ? ORDER BY scan_date DESC "
            "LIMIT ?", (current_scan_date, lookback_days + 1)).fetchall()]
        if len(ds) < 2:
            return {}
        cur_d, prev_d = ds[0], ds[-1]
        rows = c.execute(
            "SELECT a.symbol, a.rank, b.rank, a.score, b.score "
            "FROM scorecard a JOIN scorecard b "
            "  ON a.symbol = b.symbol AND a.scan_date = ? AND b.scan_date = ?",
            (cur_d, prev_d)).fetchall()
    out = {}
    for sym, r_now, r_prev, s_now, s_prev in rows:
        if r_now is None or r_prev is None:
            continue
        out[sym] = {
            "rank_prev":   int(r_prev),
            # positive = improved (moved UP the list), negative = slipping
            "delta_rank":  int(r_prev) - int(r_now),
            "delta_score": (round(float(s_now) - float(s_prev), 1)
                            if s_now is not None and s_prev is not None else None),
            "prev_date":   prev_d,
        }
    return out
