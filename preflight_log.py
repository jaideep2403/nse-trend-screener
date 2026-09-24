"""
Pre-trade check ledger — makes the Add-form checks accountable (2026-09-18).

WHY. The checks in `portfolio.preflight_position()` are rules. Rules that nobody scores
drift into decoration: you override them, nothing measures the result, and a year later
nobody can say whether the box helped or just added a click. This records EVERY decision
— added, half size, sized to risk, or cancelled — with the checks that fired, then marks
each one to market so the scorecard can answer two questions with your own trades:

    • when you OVERRODE a check, what happened next?
    • when you CANCELLED, what did the trade you skipped do?

Both matter. A check that fires on trades that went on to work is costing you money;
one that fires on trades you were right to skip is earning its place.

Outcomes are computed ON READ from the price history (no background job, nothing to go
stale). Honest by construction: with a handful of trades the scorecard says so rather
than dressing up n=2 as evidence — `verdict` stays "too few" until MIN_N decisions.

The DB (`preflight.db`) is gitignored — it is your trading record, not code.
"""
from __future__ import annotations

import json
import os
import sqlite3
import threading
import time
from pathlib import Path

DB_PATH = Path(os.getenv("DATA_DIR", os.path.dirname(__file__) or ".")) / "preflight.db"
_lock = threading.Lock()

# Below this many decisions a per-check number is noise, and the scorecard says so.
MIN_N = 8

ACTIONS = ("added", "added_anyway", "half_size", "risk_size", "cancelled")


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(str(DB_PATH), timeout=10)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS preflight_log (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            ts            INTEGER,
            logged_on     TEXT,      -- bhavcopy/basis date the checks were computed on
            symbol        TEXT,
            entry_date    TEXT,
            qty           REAL,
            entry_price   REAL,
            action        TEXT,      -- added | added_anyway | half_size | risk_size | cancelled
            warn_count    INTEGER,
            size_count    INTEGER,
            checks        TEXT,      -- JSON [{id, level, title}]
            trigger_price REAL,
            risk_abs      REAL,
            book_pct      REAL,
            price_at_log  REAL
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS ix_pf_symbol ON preflight_log (symbol)")
    conn.commit()
    return conn


def log_decision(payload: dict) -> dict:
    """Record one decision. Never raises — a logging failure must not block a trade."""
    try:
        action = str(payload.get("action") or "").strip()
        sym = str(payload.get("symbol") or "").upper().strip()
        if not sym or action not in ACTIONS:
            return {"logged": False, "error": f"bad symbol/action ({sym!r}, {action!r})"}
        checks = [{"id": str(c.get("id", ""))[:60], "level": str(c.get("level", ""))[:10],
                   "title": str(c.get("title", ""))[:160]}
                  for c in (payload.get("checks") or []) if isinstance(c, dict)][:16]
        with _lock:
            conn = _connect()
            try:
                cur = conn.execute(
                    "INSERT INTO preflight_log (ts, logged_on, symbol, entry_date, qty, entry_price, "
                    " action, warn_count, size_count, checks, trigger_price, risk_abs, book_pct, "
                    " price_at_log) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (int(time.time()), payload.get("basis_date"), sym, payload.get("entry_date"),
                     _f(payload.get("qty")), _f(payload.get("entry_price")), action,
                     int(payload.get("warn_count") or 0), int(payload.get("size_count") or 0),
                     json.dumps(checks), _f(payload.get("trigger")), _f(payload.get("risk_abs")),
                     _f(payload.get("book_pct")), _f(payload.get("last_close"))))
                conn.commit()
                return {"logged": True, "id": cur.lastrowid}
            finally:
                conn.close()
    except Exception as e:
        return {"logged": False, "error": str(e)}


def _f(v):
    try:
        return float(v) if v is not None else None
    except (TypeError, ValueError):
        return None


def _forward_return(symbol: str, from_date: str | None, ref_price: float | None) -> dict:
    """Return since the decision, marked to the latest close. Point of reference is the
    close on the decision's basis date (what you saw), not the entry you typed."""
    out = {"ret_pct": None, "bars": None, "last_close": None}
    try:
        import portfolio as _pf
        df = _pf._build_history(symbol)
        if df is None or len(df) < 2:
            return out
        import pandas as pd
        last = float(df["Close"].iloc[-1])
        out["last_close"] = round(last, 2)
        base = None
        if from_date:
            seg = df.loc[:pd.Timestamp(from_date)]
            if len(seg):
                base = float(seg["Close"].iloc[-1])
                out["bars"] = int(len(df) - len(seg))
        if base is None:
            base = _f(ref_price)
        if base:
            out["ret_pct"] = round((last / base - 1) * 100, 2)
    except Exception:
        pass
    return out


def recent(limit: int = 100) -> list[dict]:
    with _lock:
        conn = _connect()
        try:
            rows = conn.execute(
                "SELECT id, ts, logged_on, symbol, entry_date, qty, entry_price, action, "
                "       warn_count, size_count, checks, trigger_price, risk_abs, book_pct, "
                "       price_at_log FROM preflight_log ORDER BY id DESC LIMIT ?",
                (max(1, min(int(limit or 100), 500)),)).fetchall()
        finally:
            conn.close()
    out = []
    for r in rows:
        fwd = _forward_return(r[3], r[2], r[14])
        out.append({"id": r[0], "at": r[1], "basis_date": r[2], "symbol": r[3],
                    "entry_date": r[4], "qty": r[5], "entry_price": r[6], "action": r[7],
                    "warn_count": r[8], "size_count": r[9],
                    "checks": json.loads(r[10] or "[]"), "trigger": r[11],
                    "risk_abs": r[12], "book_pct": r[13], "price_at_log": r[14], **fwd})
    return out


def scorecard(limit: int = 500) -> dict:
    """Per-check outcomes of YOUR decisions, marked to the latest close.

    overridden = you bought anyway (added_anyway / half_size / risk_size)
    cancelled  = you did not buy; the return shown is what the skipped trade did
    A check is only judged once MIN_N decisions carry it.
    """
    rows = recent(limit)
    per: dict[str, dict] = {}
    clean = {"n": 0, "rets": []}
    for r in rows:
        ids = [c.get("id") for c in (r.get("checks") or []) if c.get("level") in ("warn", "size")]
        bought = r["action"] in ("added", "added_anyway", "half_size", "risk_size")
        if not ids:
            if bought and r.get("ret_pct") is not None:
                clean["n"] += 1
                clean["rets"].append(r["ret_pct"])
            continue
        for cid in ids:
            e = per.setdefault(cid, {"check": cid, "fired": 0, "overridden": 0, "cancelled": 0,
                                     "over_rets": [], "cancel_rets": []})
            e["fired"] += 1
            if bought:
                e["overridden"] += 1
                if r.get("ret_pct") is not None:
                    e["over_rets"].append(r["ret_pct"])
            else:
                e["cancelled"] += 1
                if r.get("ret_pct") is not None:
                    e["cancel_rets"].append(r["ret_pct"])
    def _avg(xs):
        return round(sum(xs) / len(xs), 2) if xs else None
    checks = []
    for e in per.values():
        n = e["fired"]
        checks.append({
            "check": e["check"], "fired": n, "overridden": e["overridden"],
            "cancelled": e["cancelled"],
            "avg_return_overridden": _avg(e["over_rets"]),
            "avg_return_cancelled": _avg(e["cancel_rets"]),
            "verdict": ("too few decisions — needs %d" % MIN_N) if n < MIN_N else "judgeable",
        })
    checks.sort(key=lambda c: -c["fired"])
    return {"decisions": len(rows), "min_n": MIN_N, "checks": checks,
            "clean_buys": {"n": clean["n"], "avg_return": _avg(clean["rets"])},
            "note": "Returns are marked to the latest close from the day each decision was "
                    "made — not annualised, not risk-adjusted, and small samples mean little."}
