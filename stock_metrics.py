"""
Materialised per-stock metrics table.

Why this exists:
Each scanner currently re-computes MA50/100/200, ATR, RSI, ADX, stage, RS-rank
on the same 750 stocks every time it runs. Pre-computing all of these ONCE per
day after bhavcopy lands turns 15-second scans into 200ms SELECT-WHERE queries
against a single dataframe.

Design choices:
- Single pandas DataFrame in memory (the universe is small — 750 rows × ~20
  columns = ~120KB), persisted to a pickle file for cold-start.
- Built incrementally — same `_get_stocks()` source as every scanner so there
  is no risk of divergence between the materialised table and on-demand
  re-computation. If a scanner finds a stock in `_get_stocks()` that's not in
  the metrics table, it can still compute live.
- This module DOES NOT YET refactor every scanner. It builds the foundation;
  scanner refactors are a separate, larger change. For now, scanners can
  optionally read from `get_metrics()` to compare against their own
  computations (helps validate consistency).
- Refresh is triggered from app.py's bhavcopy_scheduler immediately after a
  new bhavcopy is fetched, so the table is always at most one trading day
  stale.

Schema (columns in the DataFrame):
    symbol            str   primary key
    last_close        float
    ma50, ma100, ma150, ma200       float
    atr14             float
    rsi14             float
    adx14             float
    stage             int   1..4 (Weinstein)
    r1m, r3m, r6m, r12m             float  (% returns)
    high52, low52     float
    pct_from_high     float
    adtv_cr           float
    bars              int   data days available
    computed_at       float unix ts (per-row, all equal at build time)
"""
from __future__ import annotations

import os
import pickle
import threading
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# Canonical helpers — single source of truth
from analysis_utils import (
    adjust_for_splits,
    atr as canonical_atr,
    stage_analysis,
)

# ── Persistence ──────────────────────────────────────────────────────────────
_DATA_DIR  = Path(os.getenv("DATA_DIR", os.path.dirname(__file__) or "."))
_PKL_PATH  = _DATA_DIR / ".stock_metrics.pkl"

# ── In-memory state ──────────────────────────────────────────────────────────
_LOCK   = threading.RLock()
_FRAME: Optional[pd.DataFrame] = None
_BUILT_AT: float = 0.0


# ── Internal: per-stock metric computation ───────────────────────────────────

def _row(symbol: str, df: pd.DataFrame) -> dict:
    """Compute one row of metrics for one stock. Returns {} on insufficient data."""
    c = df["Close"].dropna()
    n = len(c)
    if n < 50:
        return {}

    last = float(c.iloc[-1])
    out: dict = {
        "symbol":     symbol,
        "last_close": round(last, 2),
        "bars":       n,
    }

    # Moving averages
    out["ma50"]  = float(c.rolling(50).mean().iloc[-1])  if n >= 50  else None
    out["ma100"] = float(c.rolling(100).mean().iloc[-1]) if n >= 100 else None
    out["ma150"] = float(c.rolling(150).mean().iloc[-1]) if n >= 150 else None
    out["ma200"] = float(c.rolling(200).mean().iloc[-1]) if n >= 200 else None

    # ATR via canonical helper (Wilder's EWM)
    out["atr14"] = round(canonical_atr(df, period=14), 2)

    # RSI 14 (Wilder)
    try:
        delta = c.diff()
        up   = delta.clip(lower=0).ewm(alpha=1/14, adjust=False).mean()
        down = (-delta.clip(upper=0)).ewm(alpha=1/14, adjust=False).mean()
        rs   = up / down.replace(0, np.nan)
        rsi  = 100 - (100 / (1 + rs))
        out["rsi14"] = round(float(rsi.iloc[-1]), 1) if not pd.isna(rsi.iloc[-1]) else None
    except Exception:
        out["rsi14"] = None

    # ADX 14 (simplified — same formula as trending.py)
    try:
        if "High" in df.columns and "Low" in df.columns and n >= 30:
            h = df["High"].values.astype(float)
            l = df["Low"].values.astype(float)
            cl = c.values.astype(float)
            min_len = min(len(h), len(l), n)
            h, l, cl = h[-min_len:], l[-min_len:], cl[-min_len:]
            tr = np.maximum.reduce([h[1:] - l[1:],
                                    np.abs(h[1:] - cl[:-1]),
                                    np.abs(l[1:] - cl[:-1])])
            up_mv = np.maximum(h[1:] - h[:-1], 0)
            dn_mv = np.maximum(l[:-1] - l[1:], 0)
            plus_dm  = np.where(up_mv > dn_mv, up_mv, 0)
            minus_dm = np.where(dn_mv > up_mv, dn_mv, 0)
            # Wilder smoothing over 14
            def _wilder(arr, n=14):
                out = np.zeros_like(arr, dtype=float)
                out[n-1] = arr[:n].sum()
                for i in range(n, len(arr)):
                    out[i] = out[i-1] - out[i-1]/n + arr[i]
                return out
            atr_s = _wilder(tr)
            plus_di  = 100 * _wilder(plus_dm)  / np.maximum(atr_s, 1e-9)
            minus_di = 100 * _wilder(minus_dm) / np.maximum(atr_s, 1e-9)
            dx       = 100 * np.abs(plus_di - minus_di) / np.maximum(plus_di + minus_di, 1e-9)
            adx      = _wilder(dx) / 14
            out["adx14"] = round(float(adx[-1]), 1) if adx[-1] > 0 else None
        else:
            out["adx14"] = None
    except Exception:
        out["adx14"] = None

    # Stage
    try:
        out["stage"] = int(stage_analysis(c))
    except Exception:
        out["stage"] = 0

    # Returns
    out["r1m"]  = round((last / float(c.iloc[-21])  - 1) * 100, 2) if n >= 21  else None
    out["r3m"]  = round((last / float(c.iloc[-63])  - 1) * 100, 2) if n >= 63  else None
    out["r6m"]  = round((last / float(c.iloc[-126]) - 1) * 100, 2) if n >= 126 else None
    out["r12m"] = round((last / float(c.iloc[-252]) - 1) * 100, 2) if n >= 252 else None

    # 52-week window
    w52    = c.iloc[-252:] if n >= 252 else c
    out["high52"]        = round(float(w52.max()), 2)
    out["low52"]         = round(float(w52.min()), 2)
    out["pct_from_high"] = round((last / float(w52.max()) - 1) * 100, 2) if float(w52.max()) > 0 else 0.0

    # ADTV (₹ Cr) — align Close+Volume on common non-NaN index before multiply
    try:
        if "Volume" in df.columns:
            cv = df[["Close", "Volume"]].dropna()
            if len(cv) >= 20:
                out["adtv_cr"] = round(float((cv["Close"].iloc[-20:] * cv["Volume"].iloc[-20:]).mean()) / 1e7, 2)
            else:
                out["adtv_cr"] = None
        else:
            out["adtv_cr"] = None
    except Exception:
        out["adtv_cr"] = None

    return out


# ── Public API ────────────────────────────────────────────────────────────────

def refresh(progress_callback=None) -> dict:
    """
    Rebuild the metrics table from the canonical _get_stocks() universe.
    Called automatically by the bhavcopy auto-refresh handler in app.py.
    Returns {built, skipped, computed_at, source}.
    """
    global _FRAME, _BUILT_AT

    try:
        from industry_groups import _get_stocks
    except Exception as e:
        return {"error": f"_get_stocks unavailable: {e}", "built": 0}

    try:
        stocks = _get_stocks()
    except Exception as e:
        return {"error": f"_get_stocks failed: {e}", "built": 0}

    if not stocks:
        return {"error": "no stocks loaded", "built": 0}

    total = len(stocks)
    rows: list[dict] = []
    skipped = 0
    rs_rank_input: dict[str, float] = {}

    for i, (sym, df) in enumerate(stocks.items()):
        if progress_callback and i % 50 == 0:
            progress_callback(i, total, f"stock_metrics: {sym}")
        try:
            r = _row(sym, df)
            if not r:
                skipped += 1
                continue
            rows.append(r)
            if r.get("r3m") is not None:
                rs_rank_input[sym] = r["r3m"]
        except Exception:
            skipped += 1
            continue

    # Cross-sectional RS rank (percentile 1-99) based on r3m across the universe.
    # Single SQL-like cross-stock computation rather than per-scanner.
    try:
        from analysis_utils import cross_sectional_rs_rank
        rs_ranks = cross_sectional_rs_rank(rs_rank_input)
        for r in rows:
            r["rs_rank"] = rs_ranks.get(r["symbol"], 50)
    except Exception:
        for r in rows:
            r["rs_rank"] = 50

    built_at = time.time()
    for r in rows:
        r["computed_at"] = built_at

    frame = pd.DataFrame(rows).set_index("symbol")

    with _LOCK:
        _FRAME = frame
        _BUILT_AT = built_at
        try:
            with open(_PKL_PATH, "wb") as f:
                pickle.dump({"frame": frame, "built_at": built_at}, f)
        except Exception:
            pass

    return {
        "built":       len(rows),
        "skipped":     skipped,
        "computed_at": int(built_at),
        "source":      "_get_stocks (bhavcopy cache)",
    }


def get_metrics() -> Optional[pd.DataFrame]:
    """
    Return the materialised metrics DataFrame. None if not yet built.
    Callers can do e.g.:
        df = get_metrics()
        if df is not None:
            stage2_high_rs = df[(df.stage == 2) & (df.rs_rank >= 70)]
    """
    global _FRAME, _BUILT_AT
    with _LOCK:
        if _FRAME is not None:
            return _FRAME
        # Try disk cache
        if _PKL_PATH.exists():
            try:
                with open(_PKL_PATH, "rb") as f:
                    obj = pickle.load(f)
                _FRAME = obj.get("frame")
                _BUILT_AT = obj.get("built_at", 0.0)
                return _FRAME
            except Exception:
                pass
        return None


def status() -> dict:
    """Light-weight introspection — used by /api/metrics/status."""
    global _FRAME, _BUILT_AT
    with _LOCK:
        frame = _FRAME
        built_at = _BUILT_AT
    if frame is None:
        return {"built": False, "rows": 0, "built_at": 0,
                "age_minutes": None, "columns": []}
    age_min = (time.time() - built_at) / 60 if built_at else None
    return {
        "built":       True,
        "rows":        int(len(frame)),
        "built_at":    int(built_at),
        "age_minutes": round(age_min, 1) if age_min is not None else None,
        "columns":     list(frame.columns),
        "stage_counts": frame["stage"].value_counts().to_dict() if "stage" in frame.columns else {},
    }


def invalidate_cache():
    """Clear the in-memory frame (forces next get_metrics() to re-load from disk)."""
    global _FRAME, _BUILT_AT
    with _LOCK:
        _FRAME = None
        _BUILT_AT = 0.0
