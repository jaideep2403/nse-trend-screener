"""
Per-stock Stage Transition Log.

P2-14: tracks when each stock LAST transitioned between Weinstein stages.
Persisted to a small SQLite table so we can answer questions like:
  - "Did SYRMA just become Stage 2 today, or has it been S2 for 6 months?"
  - "Which stocks crossed into Stage 2 within the last 5 days?" (fresh signals)
  - "Which Stage 4 stocks just turned Stage 1 (basing)?" (early bottom candidates)

Schema:
  symbol      TEXT   PK part
  stage       INT    (1, 2, 3, 4)
  since_date  TEXT   "YYYY-MM-DD" of first day in this stage
  prev_stage  INT    stage before this one
  bars_in     INT    bars in current stage (recomputed on update)

Update strategy: called once per day during the EOD scan refresh.
"""
from __future__ import annotations

import os
import sqlite3
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd


_DB_PATH = Path(os.getenv("DATA_DIR", os.path.dirname(__file__) or ".")) / "stage_transitions.db"
_lock    = threading.Lock()


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(str(_DB_PATH), timeout=10)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS stage_log (
            symbol      TEXT PRIMARY KEY,
            stage       INTEGER NOT NULL,
            since_date  TEXT    NOT NULL,
            prev_stage  INTEGER,
            bars_in     INTEGER DEFAULT 1,
            updated_at  INTEGER NOT NULL
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_stage     ON stage_log(stage)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_since     ON stage_log(since_date)")
    conn.commit()
    return conn


def update_stage(symbol: str, new_stage: int, today: str | None = None) -> dict:
    """
    Record today's stage for `symbol`. If different from yesterday's stored
    stage, log the transition. Returns {symbol, stage, since_date, prev_stage,
    bars_in, transitioned: bool}.
    """
    if not symbol or new_stage not in (1, 2, 3, 4):
        return {"symbol": symbol, "stage": new_stage, "transitioned": False}

    today = today or datetime.now().strftime("%Y-%m-%d")
    with _lock:
        conn = _connect()
        try:
            row = conn.execute(
                "SELECT stage, since_date, prev_stage, bars_in FROM stage_log WHERE symbol=?",
                (symbol,)
            ).fetchone()
            if row is None:
                # First time seeing this symbol — initialize.
                # There is no prior stage to transition FROM, so this is NOT
                # a transition event. Flagging True here would cause every
                # symbol to appear as a fresh signal on the first scan after
                # a database reset, generating false buy/exit alerts.
                conn.execute(
                    "INSERT INTO stage_log VALUES (?, ?, ?, ?, ?, ?)",
                    (symbol, new_stage, today, None, 1, int(time.time()))
                )
                conn.commit()
                return {"symbol": symbol, "stage": new_stage, "since_date": today,
                        "prev_stage": None, "bars_in": 1, "transitioned": False}
            old_stage, since_date, prev_stage, bars_in = row
            if new_stage != old_stage:
                # Transition
                conn.execute(
                    "UPDATE stage_log SET stage=?, since_date=?, prev_stage=?, "
                    "bars_in=1, updated_at=? WHERE symbol=?",
                    (new_stage, today, old_stage, int(time.time()), symbol)
                )
                conn.commit()
                return {"symbol": symbol, "stage": new_stage, "since_date": today,
                        "prev_stage": old_stage, "bars_in": 1, "transitioned": True}
            else:
                # Same stage — bump bars_in
                conn.execute(
                    "UPDATE stage_log SET bars_in=bars_in+1, updated_at=? WHERE symbol=?",
                    (int(time.time()), symbol)
                )
                conn.commit()
                return {"symbol": symbol, "stage": new_stage, "since_date": since_date,
                        "prev_stage": prev_stage, "bars_in": bars_in + 1,
                        "transitioned": False}
        finally:
            conn.close()


def update_all(stage_map: dict[str, int], today: str | None = None) -> dict:
    """Bulk update — called once per day from app.py scheduler or scanner."""
    today = today or datetime.now().strftime("%Y-%m-%d")
    transitioned = []
    for sym, stg in stage_map.items():
        r = update_stage(sym, stg, today=today)
        if r.get("transitioned"):
            transitioned.append(r)
    return {"updated": len(stage_map), "transitioned": len(transitioned),
            "transitions": transitioned}


def _compute_historical_transition(close, today_stage: int):
    """
    Walk back through cached bhavcopy history to estimate when this stock
    entered its CURRENT stage. Without this, a first-time populate would
    stamp every stock with `since_date = today`, making the "Stage Transitions
    in last 10d" view show 100% of the universe (all 0d-ago) — meaningless.

    Approach: compute stage_analysis at progressively older anchor points
    (5d, 10d, 22d, 44d, 66d back). The first anchor where the historical
    stage differs from today's stage bounds when the transition happened.
    The `since_date` is the date of the most recent SAME-stage anchor
    (= our best estimate of when today's stage began).

    Returns (since_date_str, prev_stage_or_None).
    """
    from analysis_utils import stage_analysis
    n = len(close)
    if n < 200 or today_stage not in (1, 2, 3, 4):
        return None, None

    anchors = [5, 10, 22, 44, 66]   # ~1w, 2w, 1mo, 2mo, 3mo
    last_same_back = 0
    detected_prev_stage = None
    detected_back = None

    for back in anchors:
        # stage_analysis needs ≥175 bars; we have `n - back` bars at this anchor
        if n - back < 175:
            break
        try:
            s_past = stage_analysis(close.iloc[:n - back])
        except Exception:
            break
        if s_past == 0:
            break   # insufficient history at this point — older anchors won't be better
        if s_past == today_stage:
            last_same_back = back
        else:
            detected_prev_stage = s_past
            detected_back = back
            break

    if detected_prev_stage is not None:
        # Transition is between bar -(detected_back) and bar -(last_same_back).
        # Use last_same_back + 1 (≈ first day in new stage) as the estimate.
        target_back = (last_same_back + 1) if last_same_back > 0 else max(1, detected_back // 2)
        target_back = max(1, min(target_back, n - 1))
        try:
            since_idx = close.index[-target_back]
            since_str = since_idx.strftime("%Y-%m-%d") if hasattr(since_idx, "strftime") else None
        except Exception:
            since_str = None
        return since_str, detected_prev_stage

    # No transition found in lookback window — stock has been in current stage
    # for at LEAST `last_same_back` days. Record that as since_date and leave
    # prev_stage = None (we don't know what it was before our window).
    if last_same_back > 0:
        try:
            since_idx = close.index[-last_same_back]
            since_str = since_idx.strftime("%Y-%m-%d") if hasattr(since_idx, "strftime") else None
            return since_str, None
        except Exception:
            pass
    return None, None


def _seed_with_history(symbol: str, stage: int, since_date: str,
                       prev_stage: int | None) -> None:
    """Direct INSERT for the historical bootstrap path — bypasses the normal
    update_stage flow because that always stamps today's date as since_date."""
    with _lock:
        conn = _connect()
        try:
            conn.execute(
                "INSERT OR REPLACE INTO stage_log VALUES (?, ?, ?, ?, ?, ?)",
                (symbol, stage, since_date, prev_stage, 1, int(time.time()))
            )
            conn.commit()
        finally:
            conn.close()


def populate_from_universe(today: str | None = None) -> dict:
    """
    Self-populate the stage_log by classifying every loaded bhavcopy stock
    via analysis_utils.stage_analysis. On FIRST populate (DB empty), each
    symbol's since_date and prev_stage are estimated by walking backward
    through bhavcopy history — so the "Stage Transitions in last Nd" view
    shows real recent transitions, not "everything stamped today".

    On subsequent populates (DB already has rows for these symbols),
    update_stage is used normally — which only changes since_date if the
    stage has actually changed since the last recorded entry.

    Returns {updated, transitioned, transitions} same shape as update_all.
    """
    try:
        from industry_groups import _get_stocks
        from analysis_utils import stage_analysis
    except Exception as e:
        return {"updated": 0, "transitioned": 0, "transitions": [], "error": str(e)}

    try:
        stocks = _get_stocks()
    except Exception as e:
        return {"updated": 0, "transitioned": 0, "transitions": [], "error": str(e)}

    if not stocks:
        return {"updated": 0, "transitioned": 0, "transitions": [],
                "error": "no bhavcopy data loaded"}

    today_str = today or datetime.now().strftime("%Y-%m-%d")

    # Which symbols already have a row? If yes, we update normally. If no,
    # we seed with historical lookback so since_date reflects reality.
    with _lock:
        conn = _connect()
        try:
            existing = {r[0] for r in conn.execute("SELECT symbol FROM stage_log")}
        finally:
            conn.close()

    new_stage_map: dict[str, int] = {}
    seeded = 0
    for sym, df in stocks.items():
        try:
            c = df["Close"].dropna()
            if len(c) < 175:
                continue
            s_today = int(stage_analysis(c))
            if s_today not in (1, 2, 3, 4):
                continue
        except Exception:
            continue

        if sym in existing:
            new_stage_map[sym] = s_today
            continue

        # New symbol — try to estimate real since_date from history
        since_str, prev_st = _compute_historical_transition(c, s_today)
        if not since_str:
            since_str = today_str   # fallback: no usable history
        _seed_with_history(sym, s_today, since_str, prev_st)
        seeded += 1

    # Now do normal updates for any symbols that already existed
    upd = update_all(new_stage_map, today=today_str) if new_stage_map else \
          {"updated": 0, "transitioned": 0, "transitions": []}
    upd["seeded"] = seeded
    upd["updated"] = upd.get("updated", 0) + seeded
    return upd


def get_stage_info(symbol: str) -> dict:
    """Query: how long has this stock been in its current stage?"""
    with _lock:
        conn = _connect()
        try:
            row = conn.execute(
                "SELECT stage, since_date, prev_stage, bars_in FROM stage_log WHERE symbol=?",
                (symbol.upper(),)
            ).fetchone()
            if not row:
                return {"symbol": symbol.upper(), "stage": None, "label": "Unknown"}
            stage, since, prev, bars = row
            try:
                days_ago = (datetime.now().date() - datetime.strptime(since, "%Y-%m-%d").date()).days
            except Exception:
                days_ago = 0
            return {
                "symbol":      symbol.upper(),
                "stage":       stage,
                "since_date":  since,
                "days_in":     days_ago,
                "bars_in":     bars,
                "prev_stage":  prev,
                "fresh":       days_ago <= 10,   # < 2 weeks = fresh transition
                "label":       (f"S{stage} fresh ({days_ago}d ago)" if days_ago <= 10
                                 else f"S{stage} since {since}"),
            }
        finally:
            conn.close()


def recent_transitions(into_stage: int, within_days: int = 10) -> list[dict]:
    """
    P2-14 KEY VIEW: list stocks that JUST transitioned INTO `into_stage`
    within the last `within_days` calendar days. Most actionable view.
    """
    cutoff = (datetime.now() - timedelta(days=within_days)).strftime("%Y-%m-%d")
    with _lock:
        conn = _connect()
        try:
            rows = conn.execute(
                "SELECT symbol, stage, since_date, prev_stage, bars_in "
                "FROM stage_log WHERE stage=? AND since_date >= ? "
                "ORDER BY since_date DESC",
                (into_stage, cutoff)
            ).fetchall()
            out = []
            for sym, stg, since, prev, bars in rows:
                try:
                    days_ago = (datetime.now().date() -
                                datetime.strptime(since, "%Y-%m-%d").date()).days
                except Exception:
                    days_ago = 0
                out.append({
                    "symbol":     sym,
                    "stage":      stg,
                    "prev_stage": prev,
                    "since_date": since,
                    "days_ago":   days_ago,
                    "bars_in":    bars,
                })
            return out
        finally:
            conn.close()


def stats() -> dict:
    """Aggregate counts per stage — quick health check."""
    with _lock:
        conn = _connect()
        try:
            rows = conn.execute(
                "SELECT stage, COUNT(*) FROM stage_log GROUP BY stage"
            ).fetchall()
            counts = {1: 0, 2: 0, 3: 0, 4: 0}
            for stg, cnt in rows:
                counts[stg] = cnt
            total = sum(counts.values())
            return {
                "total_tracked": total,
                "stage_1": counts[1],
                "stage_2": counts[2],
                "stage_3": counts[3],
                "stage_4": counts[4],
            }
        finally:
            conn.close()
