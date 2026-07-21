"""
Quality factor — an AQR "Quality-Minus-Junk" style composite, from the local
fundamentals.db (screener.in snapshot). QMJ (Asness-Frazzini-Pedersen) shows that
safe, profitable, growing, well-managed companies earn positive risk-adjusted
returns AND exhibit positive convexity — they benefit from "flight to quality" in
crises rather than crashing. That makes quality a DRAWDOWN lever, not just an alpha
lever, which is exactly what the All-Weather book wants.

Dimensions we CAN measure from fundamentals.db (coverage in parentheses):
  • Profitability — ROE                         (100%)   weight 0.40
  • Growth        — 3Y profit CAGR / TTM        (100%)   weight 0.30
  • Earnings accel— YoY quarterly profit accel  ( 99%)   weight 0.15
  • Management    — promoter holding            (100%)   weight 0.15
Leverage/safety (D/E) is INTENTIONALLY omitted — screener.in never exposes it
(0% coverage), and the QMJ "safety" dimension is already captured by the low-vol
(low-beta) leg of the defensive engine, so we don't double-count it here.

════════════════════════════════════════════════════════════════════════════════
HONEST LIMITATION (read before trusting any quality backtest):
fundamentals.db holds only the LATEST snapshot per stock, not point-in-time
history, and only for CURRENTLY-LISTED names. Applying today's quality to a
historical rebalance is therefore (a) look-ahead-biased and (b) survivorship-
biased. Quality is a slow-moving property (a high-ROE franchise was usually
high-ROE a few years ago), so the bias is moderate — but a quality backtest is an
OPTIMISTIC illustration, NOT the survivorship-free proof the price factors get.
Delisted / unknown names are assigned NEUTRAL quality (0.5) to bound the bias.
════════════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import os
import sqlite3
import pandas as pd

DB_PATH = os.path.join(os.environ.get("DATA_DIR", os.path.dirname(__file__)),
                       "fundamentals.db")

NEUTRAL = 0.5   # quality for names with no fundamentals (delisted / not on screener)

# Component weights (sum = 1.0). Profitability-led, à la QMJ.
W_PROF, W_GROWTH, W_ACCEL, W_MGMT = 0.40, 0.30, 0.15, 0.15

_cache: dict = {"map": None, "comp": None}


def _load_rows() -> list[dict]:
    if not os.path.exists(DB_PATH):
        return []
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        rows = [dict(r) for r in conn.execute(
            "SELECT symbol, roe, growth_3y_cagr, growth_ttm, eps_growth_yoy, "
            "eps_accel_yoy, promoter_holding, promoter_delta FROM fundamentals"
        ).fetchall()]
        conn.close()
        return rows
    except Exception:
        return []


def _compute() -> None:
    """Build and cache {symbol: quality 0-1} plus per-symbol components."""
    rows = _load_rows()
    if not rows:
        _cache["map"], _cache["comp"] = {}, {}
        return
    df = pd.DataFrame(rows)

    def pr(col):
        return col.rank(pct=True)

    roe = pd.to_numeric(df["roe"], errors="coerce")
    growth = pd.to_numeric(df["growth_3y_cagr"], errors="coerce")
    growth = growth.where(growth.notna(), pd.to_numeric(df["growth_ttm"], errors="coerce"))
    growth = growth.where(growth.notna(), pd.to_numeric(df["eps_growth_yoy"], errors="coerce"))
    prom = pd.to_numeric(df["promoter_holding"], errors="coerce")
    accel = df["eps_accel_yoy"].map(
        lambda v: 1.0 if v == 1 else (0.0 if v == 0 else 0.5)).astype(float).fillna(0.5)

    roe_r    = pr(roe).fillna(NEUTRAL)
    growth_r = pr(growth).fillna(NEUTRAL)
    prom_r   = pr(prom).fillna(NEUTRAL)

    q = (W_PROF * roe_r + W_GROWTH * growth_r + W_ACCEL * accel + W_MGMT * prom_r)
    df["q"] = q.clip(0.0, 1.0)

    _cache["map"] = dict(zip(df["symbol"], df["q"].astype(float)))
    _cache["comp"] = {
        r["symbol"]: {
            "roe": (None if pd.isna(r.get("roe")) else float(r["roe"])),
            "growth": (None if pd.isna(growth.iloc[i]) else round(float(growth.iloc[i]), 1)),
            "accel": (1 if r.get("eps_accel_yoy") == 1 else (0 if r.get("eps_accel_yoy") == 0 else None)),
            "promoter": (None if pd.isna(r.get("promoter_holding")) else float(r["promoter_holding"])),
            "quality": round(float(df["q"].iloc[i]), 3),
        }
        for i, r in enumerate(rows)
    }


def load_quality_map(refresh: bool = False) -> dict:
    """{symbol: quality 0-1}. Cached; call refresh=True to rebuild from the DB."""
    if refresh or _cache["map"] is None:
        _compute()
    return _cache["map"] or {}


def quality_of(symbol: str) -> float:
    """Quality 0-1 for one symbol, NEUTRAL if we have no fundamentals for it."""
    return load_quality_map().get(symbol, NEUTRAL)


def components_of(symbol: str) -> dict | None:
    """ROE / growth / accel / promoter / quality for one symbol (UI), or None."""
    if _cache["comp"] is None:
        _compute()
    return (_cache["comp"] or {}).get(symbol)


def coverage() -> dict:
    m = load_quality_map()
    return {"symbols": len(m), "db": os.path.basename(DB_PATH),
            "exists": os.path.exists(DB_PATH)}
