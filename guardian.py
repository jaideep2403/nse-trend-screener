"""
Position Guardian — the app chases YOU, not the other way round.

WHY (the GVT&D lesson): a held position flipped to 🔴 weakening on 2026-06-24
at ₹5,052 and the exit engine said EXIT — but that intelligence only rendered
if the user happened to open the Portfolio tab. The stock bottomed 13% lower.
The Guardian closes that gap: every scan (and a daily scheduler pass) sweeps
all portfolio holdings + watchlist symbols, persists their alert state, and
the global header strip — visible on EVERY tab — surfaces anything that fired.

DESIGN
  • run_sweep() is the only heavy call (loads price history). It runs after
    each trending scan and from the daily scheduler — never from /api/header.
  • get_active_alerts() is a pure sqlite read (<5ms) — safe for the header
    poll. One row per symbol = current state; alerts dedupe by state change.
  • Severity ladder: exit > trim > watch. Portfolio symbols use the full
    ladder (exits.evaluate_exit + weakening flip); watchlist symbols cap at
    'watch' (you can't exit what you don't hold).

SCHEMA (guardian.db)
    guardian_state(
        symbol      TEXT PRIMARY KEY,
        kind        TEXT,      -- 'position' | 'watchlist'
        bhav_date   TEXT,      -- bhavcopy date of the last sweep
        severity    TEXT,      -- 'exit' | 'trim' | 'watch' | 'ok'
        reason      TEXT,
        price       REAL,
        stop        REAL,
        entry       REAL,
        entry_window TEXT,
        since_date  TEXT,      -- first bhav_date this severity appeared
        dismissed_on TEXT,     -- bhav_date the user dismissed the alert (or NULL)
        updated_at  INTEGER
    )
"""
from __future__ import annotations

import os
import sqlite3
import threading
import time
from pathlib import Path

_DB_PATH = Path(os.getenv("DATA_DIR", os.path.dirname(__file__) or ".")) / "guardian.db"
_lock = threading.Lock()

_SEV_RANK = {"ok": 0, "watch": 1, "trim": 2, "exit": 3}


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(str(_DB_PATH), timeout=10)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS guardian_state (
            symbol       TEXT PRIMARY KEY,
            kind         TEXT,
            bhav_date    TEXT,
            severity     TEXT,
            reason       TEXT,
            price        REAL,
            stop         REAL,
            entry        REAL,
            entry_window TEXT,
            since_date   TEXT,
            dismissed_on TEXT,
            updated_at   INTEGER
        )
    """)
    conn.commit()
    return conn


# ── Sweep (heavy — call from scan completion / scheduler only) ────────────────

def run_sweep() -> dict:
    """Evaluate every portfolio holding + watchlist symbol; persist states.
    Returns {checked, alerts, bhav_date}. Never raises — returns error dict."""
    try:
        return _run_sweep_inner()
    except Exception as e:
        return {"error": str(e), "checked": 0, "alerts": []}


def _run_sweep_inner() -> dict:
    from industry_groups import _get_stocks
    from trending import _clean_df, _score_stock

    try:
        from data_fetcher import _latest_bhavcopy_date
        bd = _latest_bhavcopy_date()
        bhav_date = bd.isoformat() if bd else None
    except Exception:
        bhav_date = None

    # ── Collect targets: portfolio holdings (full analysis incl. the SAME
    # evaluate_exit the Portfolio tab renders) + watchlist symbols. ──
    targets: dict[str, dict] = {}   # symbol -> {kind, entry, sl, exit_action, exit_reason}
    try:
        import portfolio
        for p in portfolio.list_positions():
            sym = (p.get("symbol") or "").upper()
            if not sym:
                continue
            ex = p.get("exit") or {}
            targets[sym] = {
                "kind":        "position",
                "entry":       p.get("entry_price"),
                "sl":          p.get("sl"),
                "exit_action": (ex.get("action") or "HOLD").upper(),
                "exit_reason": ex.get("reason") or "",
            }
    except Exception:
        pass
    try:
        import watchlist
        for sym in watchlist.get_symbols():
            if sym not in targets:   # a held symbol outranks its watchlist entry
                targets[sym] = {"kind": "watchlist", "entry": None, "sl": None,
                                "exit_action": None, "exit_reason": ""}
    except Exception:
        pass

    if not targets:
        return {"checked": 0, "alerts": [], "bhav_date": bhav_date}

    stocks = _get_stocks()
    try:
        from benchmark import get_benchmark
        nifty = get_benchmark(days=420)
    except Exception:
        nifty = None

    alerts = []
    now = int(time.time())
    with _lock:
        conn = _connect()
        try:
            for sym, t in targets.items():
                df = stocks.get(sym)
                ew, price, reasons = None, None, []
                if df is not None and len(df) >= 20:
                    try:
                        m = _score_stock(_clean_df(df.copy()), nifty)
                    except Exception:
                        m = None
                    if m:
                        ew = m.get("entry_window")
                        price = m.get("price")
                        if ew == "weakening":
                            fade = m.get("retrace_10d")
                            reasons.append(
                                f"weakening — {fade:+.0f}% off 10d high" if fade is not None
                                else "weakening — strength fading")

                # Severity ladder
                severity = "ok"
                if t["kind"] == "position":
                    act = t.get("exit_action")
                    if act == "EXIT":
                        severity = "exit"
                        reasons.insert(0, t.get("exit_reason") or "exit signal")
                    elif act == "TRIM":
                        severity = "trim"
                        reasons.insert(0, t.get("exit_reason") or "trim signal")
                    elif ew == "weakening":
                        severity = "trim"   # held + fading = act, not just watch
                elif ew == "weakening":
                    severity = "watch"      # watchlist cap

                reason = " · ".join(r for r in reasons if r) or "ok"

                prev = conn.execute(
                    "SELECT severity, since_date, dismissed_on FROM guardian_state "
                    "WHERE symbol=?", (sym,)).fetchone()
                prev_sev = prev[0] if prev else "ok"
                since = (prev[1] if prev and prev_sev == severity and prev[1]
                         else bhav_date)
                # A dismissal only silences the CURRENT episode: escalation or
                # a fresh severity after an 'ok' spell re-arms the alert.
                dismissed_on = prev[2] if prev else None
                if _SEV_RANK.get(severity, 0) > _SEV_RANK.get(prev_sev, 0):
                    dismissed_on = None
                if severity == "ok":
                    dismissed_on = None

                conn.execute(
                    "INSERT INTO guardian_state (symbol, kind, bhav_date, severity, "
                    " reason, price, stop, entry, entry_window, since_date, "
                    " dismissed_on, updated_at) "
                    "VALUES (?,?,?,?,?,?,?,?,?,?,?,?) "
                    "ON CONFLICT(symbol) DO UPDATE SET "
                    " kind=excluded.kind, bhav_date=excluded.bhav_date, "
                    " severity=excluded.severity, reason=excluded.reason, "
                    " price=excluded.price, stop=excluded.stop, entry=excluded.entry, "
                    " entry_window=excluded.entry_window, since_date=excluded.since_date, "
                    " dismissed_on=excluded.dismissed_on, updated_at=excluded.updated_at",
                    (sym, t["kind"], bhav_date, severity, reason, price,
                     t.get("sl"), t.get("entry"), ew, since, dismissed_on, now))

                if severity != "ok":
                    alerts.append({"symbol": sym, "severity": severity,
                                   "reason": reason, "kind": t["kind"]})

            # Symbols no longer held/watched drop out of the table
            conn.execute(
                "DELETE FROM guardian_state WHERE symbol NOT IN ({})".format(
                    ",".join("?" * len(targets))), tuple(targets.keys()))
            conn.commit()
        finally:
            conn.close()

    return {"checked": len(targets), "alerts": alerts, "bhav_date": bhav_date}


# ── Fast reads (safe for the /api/header poll) ────────────────────────────────

def get_active_alerts() -> list[dict]:
    """Current non-ok, non-dismissed states — a pure sqlite read."""
    with _lock:
        conn = _connect()
        try:
            rows = conn.execute(
                "SELECT symbol, kind, severity, reason, price, stop, entry, "
                "       entry_window, since_date, bhav_date FROM guardian_state "
                "WHERE severity != 'ok' AND dismissed_on IS NULL "
                "ORDER BY CASE severity WHEN 'exit' THEN 0 WHEN 'trim' THEN 1 "
                "ELSE 2 END, symbol").fetchall()
        finally:
            conn.close()
    return [{
        "symbol": r[0], "kind": r[1], "severity": r[2], "reason": r[3],
        "price": r[4], "stop": r[5], "entry": r[6], "entry_window": r[7],
        "since": r[8], "bhav_date": r[9],
    } for r in rows]


def dismiss(symbol: str) -> bool:
    """Silence the symbol's CURRENT alert episode (re-arms on escalation)."""
    sym = (symbol or "").upper().strip()
    if not sym:
        return False
    with _lock:
        conn = _connect()
        try:
            cur = conn.execute(
                "UPDATE guardian_state SET dismissed_on = bhav_date "
                "WHERE symbol = ? AND severity != 'ok'", (sym,))
            conn.commit()
            return cur.rowcount > 0
        finally:
            conn.close()
