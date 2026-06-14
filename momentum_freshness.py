"""
Momentum tier freshness log — SQLite-persisted record of when each stock
entered its current momentum tier.

WHY:
Stocks that JUST entered the Elite tier (last 5-10 days) statistically
outperform stocks that have been in Elite for months (mature momentum is
prone to mean-reversion). Without this, every Elite stock looks the same
on the Momentum tab — fresh institutional buying gets the same display as
6-month-stretched winners.

DESIGN (mirrors stage_transitions.py):
- One row per symbol with current tier + since_date.
- Updated daily after each Momentum scan via update_all(tier_map).
- get_freshness(symbol) returns days_in_tier + bucket label.

SCHEMA:
    momentum_log(
        symbol       TEXT PRIMARY KEY,
        tier         TEXT NOT NULL,         -- "Elite" | "Strong" | "Rising" | ""
        since_date   TEXT NOT NULL,         -- ISO date when this tier began
        prev_tier    TEXT,                  -- tier they came from
        bars_in      INTEGER DEFAULT 1,     -- how many scans since transition
        updated_at   INTEGER NOT NULL
    )

BUCKETS (Tier 2F display):
    fresh    — 0-10 calendar days in current tier  → 🆕
    sweet    — 10-45 days                          → 🟢
    mature   — 45-90 days                          → 🟡
    stale    — 90+ days                            → 🟠
"""
from __future__ import annotations

import os
import sqlite3
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path

_DB_PATH = Path(os.getenv("DATA_DIR", os.path.dirname(__file__) or ".")) / "momentum_freshness.db"
_lock    = threading.Lock()


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(str(_DB_PATH), timeout=10)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS momentum_log (
            symbol      TEXT PRIMARY KEY,
            tier        TEXT NOT NULL,
            since_date  TEXT NOT NULL,
            prev_tier   TEXT,
            bars_in     INTEGER DEFAULT 1,
            updated_at  INTEGER NOT NULL
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_mtier  ON momentum_log(tier)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_msince ON momentum_log(since_date)")
    conn.commit()
    return conn


def update_tier(symbol: str, new_tier: str, today: str | None = None) -> dict:
    """
    Record today's tier for `symbol`. If different from stored tier, log
    a transition. Empty-string tier means "no tier" (stock dropped out of
    Rising+ buckets); still recorded so we can see drop-outs.

    Returns {symbol, tier, since_date, prev_tier, days_in, transitioned}.
    """
    if not symbol:
        return {"symbol": symbol, "tier": new_tier, "transitioned": False}

    new_tier = new_tier or ""
    today = today or datetime.now().strftime("%Y-%m-%d")
    with _lock:
        conn = _connect()
        try:
            row = conn.execute(
                "SELECT tier, since_date, prev_tier, bars_in FROM momentum_log WHERE symbol=?",
                (symbol,)
            ).fetchone()
            if row is None:
                # First-time insert. Treat as initialisation, NOT a transition,
                # so first-ever scan doesn't flag every Elite stock as "Fresh"
                # (same precaution as stage_transitions.py).
                conn.execute(
                    "INSERT INTO momentum_log VALUES (?, ?, ?, ?, ?, ?)",
                    (symbol, new_tier, today, None, 1, int(time.time()))
                )
                conn.commit()
                return {"symbol": symbol, "tier": new_tier, "since_date": today,
                        "prev_tier": None, "days_in": 0, "transitioned": False}
            old_tier, since_date, prev_tier, bars_in = row
            if new_tier != old_tier:
                # Tier changed — log the transition
                conn.execute(
                    "UPDATE momentum_log SET tier=?, since_date=?, prev_tier=?, "
                    "bars_in=1, updated_at=? WHERE symbol=?",
                    (new_tier, today, old_tier, int(time.time()), symbol)
                )
                conn.commit()
                return {"symbol": symbol, "tier": new_tier, "since_date": today,
                        "prev_tier": old_tier, "days_in": 0, "transitioned": True}
            # Same tier — increment bars_in counter
            conn.execute(
                "UPDATE momentum_log SET bars_in=bars_in+1, updated_at=? WHERE symbol=?",
                (int(time.time()), symbol)
            )
            conn.commit()
            try:
                days_in = (datetime.now().date()
                           - datetime.strptime(since_date, "%Y-%m-%d").date()).days
            except Exception:
                days_in = 0
            return {"symbol": symbol, "tier": new_tier, "since_date": since_date,
                    "prev_tier": prev_tier, "days_in": days_in, "transitioned": False}
        finally:
            conn.close()


def update_all(tier_map: dict[str, str], today: str | None = None) -> dict:
    """Bulk update from a scanner result. tier_map = {symbol: tier_label}.
    Symbols absent from the map keep their last-known tier (no change)."""
    updated = 0
    transitioned = 0
    transitions = []
    for sym, tier in tier_map.items():
        try:
            r = update_tier(sym, tier, today=today)
            updated += 1
            if r.get("transitioned"):
                transitioned += 1
                transitions.append({
                    "symbol": sym, "from": r.get("prev_tier"), "to": tier,
                    "date": r.get("since_date"),
                })
        except Exception:
            continue
    return {"updated": updated, "transitioned": transitioned, "transitions": transitions}


def get_freshness_map(symbols: list[str]) -> dict[str, dict]:
    """Bulk lookup — one DB round-trip for many symbols. Used by the scanner
    UI integration to attach freshness info to every result row."""
    if not symbols:
        return {}
    out: dict[str, dict] = {}
    with _lock:
        conn = _connect()
        try:
            placeholders = ",".join("?" * len(symbols))
            rows = conn.execute(
                f"SELECT symbol, tier, since_date, prev_tier, bars_in "
                f"FROM momentum_log WHERE symbol IN ({placeholders})",
                tuple(s.upper() for s in symbols),
            ).fetchall()
            today_d = datetime.now().date()
            for sym, tier, since_date, prev_tier, bars_in in rows:
                try:
                    days_in = (today_d - datetime.strptime(since_date, "%Y-%m-%d").date()).days
                except Exception:
                    days_in = 0
                bucket = _bucket_for(days_in)
                out[sym] = {
                    "tier":       tier,
                    "since_date": since_date,
                    "prev_tier":  prev_tier,
                    "bars_in":    bars_in,
                    "days_in":    days_in,
                    "bucket":     bucket,
                    "label":      _label_for(tier, days_in, bucket),
                }
            return out
        finally:
            conn.close()


# ── Internal helpers ──────────────────────────────────────────────────────────

def _bucket_for(days_in: int) -> str:
    if days_in <= 10:  return "fresh"
    if days_in <= 45:  return "sweet"
    if days_in <= 90:  return "mature"
    return "stale"


def _label_for(tier: str, days_in: int, bucket: str) -> str:
    if not tier:
        return "—"
    if bucket == "fresh":  prefix = "🆕"
    elif bucket == "sweet":  prefix = "🟢"
    elif bucket == "mature": prefix = "🟡"
    else:                    prefix = "🟠"
    return f"{prefix} {tier} {days_in}d"
