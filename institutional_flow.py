"""
Institutional flow factor — derivatives open-interest positioning.

WHY THIS ONE (and not the other "institutional" sources)
--------------------------------------------------------
Audited 2026-07-25, three candidate institutional datasets:
  • bulk_deals.py   — NSE bulk.csv/block.csv is a CURRENT-DAY rolling file
                      (224 rows, all dated today). No history → a backtest would
                      be pure look-ahead. REJECTED for the ranking engine.
  • holders_data.py — NSE's shareholding-pattern API returns HTTP 404; FII/DII
                      per-stock data is simply unavailable. REJECTED.
  • fo_data.py      — daily F&O bhavcopy lives on NSE archives as DATED files
                      (BhavCopy_NSE_FO_..._{YYYYMMDD}_F_0000.csv.zip). Each file
                      is a genuine point-in-time snapshot. USABLE. ← this module

That distinction matters: unlike the quality factor (current-snapshot fundamentals
applied to the past ⇒ look-ahead-biased), this factor is **honestly backtestable**
— every value at date D is computed only from files dated ≤ D.

THE SIGNAL
----------
Open interest = the number of live derivative contracts. Its change, read together
with price direction, is the standard read on how leveraged money is positioning —
the thing retail rarely processes systematically:

    OI ↑ + price ↑ → LONG BUILDUP   — fresh money opening longs   (strongest)
    OI ↓ + price ↑ → SHORT COVERING — shorts closing, weaker fuel
    OI ↓ + price ↓ → LONG UNWINDING — longs exiting
    OI ↑ + price ↓ → SHORT BUILDUP  — fresh shorts                (weakest)

Our book is already gated to uptrends, so the useful discriminator is: *is this
uptrend backed by fresh institutional longs, or is it being shorted into?*

COVERAGE CAVEAT: only ~180-210 NSE names have listed futures, versus our ~750
universe. Symbols without F&O get NEUTRAL (0.5) — the factor tilts the F&O-eligible
(large/liquid) subset and is silent elsewhere. History starts ~Jan 2024 (NSE's
unified archive format), so it can only be validated on the recent window.
"""
from __future__ import annotations

import os
from datetime import date

import numpy as np
import pandas as pd

NEUTRAL = 0.5
LOOKBACK = 5          # sessions over which OI/price change is measured
OI_THRESH = 2.0       # % OI change to call a real buildup/unwind
PX_THRESH = 0.5       # % price change to call a direction

# conviction score per regime, mapped to 0-1 (higher = stronger institutional bid)
_SCORE = {
    "LONG_BUILDUP":  1.00,
    "SHORT_COVER":   0.65,
    "NEUTRAL":       0.50,
    "LONG_UNWIND":   0.35,
    "SHORT_BUILDUP": 0.00,
}

_cache: dict = {"oi": None, "px": None, "score": None}


def _fo_dir():
    import fo_data
    return fo_data._FO_DIR


def build_panel(refresh: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(oi, px) panels — rows = date, cols = symbol — from every cached dated
    F&O file. Loading is pure disk I/O; no network."""
    if not refresh and _cache["oi"] is not None:
        return _cache["oi"], _cache["px"]
    import pickle
    d = _fo_dir()
    oi_rows, px_rows = {}, {}
    if os.path.isdir(d):
        for fn in sorted(os.listdir(d)):
            p = os.path.join(d, fn)
            try:
                with open(p, "rb") as fh:
                    df = pickle.load(fh)
            except Exception:
                continue
            if df is None or "SYMBOL" not in getattr(df, "columns", []):
                continue
            try:
                dt = pd.Timestamp(str(df["date"].iloc[0])) if "date" in df.columns else None
                if dt is None:
                    continue
                s = df.set_index("SYMBOL")
                oi_rows[dt] = s["OPEN_INT"].astype(float)
                px_rows[dt] = s["CLOSE"].astype(float)
            except Exception:
                continue
    oi = pd.DataFrame(oi_rows).T.sort_index() if oi_rows else pd.DataFrame()
    px = pd.DataFrame(px_rows).T.sort_index() if px_rows else pd.DataFrame()
    _cache["oi"], _cache["px"] = oi, px
    return oi, px


def build_score_panel(refresh: bool = False) -> pd.DataFrame:
    """Panel of conviction scores (0-1), rows = date, cols = symbol.
    Row D uses only bars ≤ D (rolling pct_change) → point-in-time, no look-ahead."""
    if not refresh and _cache["score"] is not None:
        return _cache["score"]
    oi, px = build_panel(refresh=refresh)
    if oi.empty or px.empty:
        _cache["score"] = pd.DataFrame()
        return _cache["score"]
    oi_chg = oi.pct_change(LOOKBACK) * 100.0
    px_chg = px.pct_change(LOOKBACK) * 100.0

    score = pd.DataFrame(NEUTRAL, index=oi_chg.index, columns=oi_chg.columns, dtype=float)
    oi_up, oi_dn = oi_chg >= OI_THRESH, oi_chg <= -OI_THRESH
    px_up, px_dn = px_chg >= PX_THRESH, px_chg <= -PX_THRESH
    score = score.mask(oi_up & px_up, _SCORE["LONG_BUILDUP"])
    score = score.mask(oi_dn & px_up, _SCORE["SHORT_COVER"])
    score = score.mask(oi_dn & px_dn, _SCORE["LONG_UNWIND"])
    score = score.mask(oi_up & px_dn, _SCORE["SHORT_BUILDUP"])
    score = score.mask(oi_chg.isna() | px_chg.isna(), np.nan)
    _cache["score"] = score
    return score


def score_map_asof(on_date, max_stale_days: int = 7) -> dict:
    """{symbol: conviction 0-1} as known on `on_date` — the point-in-time lookup a
    walk-forward backtest uses. Uses the most recent row ≤ on_date; returns {} if
    that row is staler than `max_stale_days` (so a data gap can't silently freeze
    the factor at an old reading)."""
    sc = build_score_panel()
    if sc.empty:
        return {}
    ts = pd.Timestamp(on_date)
    rows = sc.loc[sc.index <= ts]
    if rows.empty:
        return {}
    last = rows.index[-1]
    if (ts - last).days > max_stale_days:
        return {}
    return rows.iloc[-1].dropna().to_dict()


def signal_label(score: float | None) -> str:
    if score is None or not np.isfinite(score):
        return "NO F&O"
    for k, v in _SCORE.items():
        if abs(score - v) < 1e-6:
            return k.replace("_", " ").title()
    return "Neutral"


def coverage() -> dict:
    sc = build_score_panel()
    if sc.empty:
        return {"days": 0, "symbols": 0, "first": None, "last": None}
    return {"days": int(len(sc)), "symbols": int(sc.shape[1]),
            "first": str(sc.index[0].date()), "last": str(sc.index[-1].date())}
