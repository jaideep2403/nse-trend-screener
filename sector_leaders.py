"""Sector Leaders — rank NSE sectoral/thematic indices by member momentum and
surface the strongest constituents inside each leading sector.

Real-money module: every public entry point is wrapped so it never raises.

Pipeline (see run_sector_leaders):
  1. Pull constituents (21 indices), official daily index levels, and the SAME
     split-adjusted OHLCV map Power Plays uses (breakout_scanner._load_all_stocks).
  2. Build a universe-wide 3-month-return map and convert it into a 1-99
     cross-sectional RS rating for every loaded symbol.
  3. Per sector, compute per-stock momentum/quality fields for each constituent
     that has data, skipping ETFs.
  4. Aggregate per sector (median r1m/r3m, breadth, live index move, PE).
  5. Rank sectors by median 3-month return (DESC); within each, keep the top_n
     constituents by RS rating.

Result is cached keyed by bhavcopy date; recompute on date change or force=True.
"""

from __future__ import annotations

import threading

import numpy as np
import pandas as pd

# ── Verified data sources ─────────────────────────────────────────────────────
import sector_indices
import breakout_scanner
import analysis_utils

try:
    from nse_stocks import is_etf
except Exception:                                    # pragma: no cover - defensive
    def is_etf(symbol: str) -> bool:
        return False

try:
    from data_fetcher import _latest_bhavcopy_date
except Exception:                                    # pragma: no cover - defensive
    def _latest_bhavcopy_date():
        return None


# ── Cache (keyed by bhavcopy date) ────────────────────────────────────────────
_CACHE: dict[str, dict] = {}
_CACHE_LOCK = threading.Lock()


# ── Helpers ───────────────────────────────────────────────────────────────────
def _ret_pct(close: pd.Series, lookback: int) -> float | None:
    """(close[-1] / close[-1-lookback] - 1) * 100, or None if too few bars."""
    if len(close) <= lookback:
        return None
    prev = float(close.iloc[-1 - lookback])
    if prev <= 0:
        return None
    return (float(close.iloc[-1]) / prev - 1.0) * 100.0


def _r3m(close: pd.Series) -> float | None:
    """3-month return = (c[-1]/c[-64]-1)*100, needs >= 64 bars."""
    return _ret_pct(close, 64)


def _universe_rs(stocks: dict[str, pd.DataFrame]) -> dict[str, int]:
    """1-99 cross-sectional RS rating keyed on universe-wide 3-month return.

    Prefers analysis_utils.cross_sectional_rs_rank; falls back to a pandas
    percentile rank if that helper is unavailable.
    """
    r3m_map: dict[str, float] = {}
    for sym, df in stocks.items():
        try:
            close = df["Close"].dropna()
        except Exception:
            continue
        r = _r3m(close)
        if r is not None and np.isfinite(r):
            r3m_map[sym] = r
    if not r3m_map:
        return {}

    fn = getattr(analysis_utils, "cross_sectional_rs_rank", None)
    if callable(fn):
        try:
            ranks = fn(r3m_map)
            if isinstance(ranks, dict) and ranks:
                return {s: int(v) for s, v in ranks.items()}
        except Exception:
            pass

    # Fallback: pandas percentile rank scaled 1-99.
    s = pd.Series(r3m_map)
    pct = s.rank(pct=True) * 99.0
    return {sym: int(max(1, min(99, round(v)))) for sym, v in pct.items()}


def _stage_label(close: pd.Series) -> str:
    """Weinstein stage label via analysis_utils, or '—' when unavailable."""
    try:
        fn = getattr(analysis_utils, "stage_analysis", None)
        if not callable(fn):
            return "—"
        s = fn(close)
        lab = getattr(analysis_utils, "stage_label", None)
        if callable(lab):
            return lab(s)
        return {1: "S1 Basing", 2: "S2 ▲", 3: "S3 Top",
                4: "S4 ▼"}.get(int(s), "—")
    except Exception:
        return "—"


# ── Core ──────────────────────────────────────────────────────────────────────
def _compute(top_n: int, as_of: str) -> dict:
    constituents = sector_indices.get_sector_constituents()          # {sector: [syms]}
    levels       = sector_indices.get_sector_levels()                # {sector: {...}}
    stocks       = breakout_scanner._load_all_stocks()               # {sym: OHLCV df}

    rs_map = _universe_rs(stocks)                                     # {sym: 1-99}

    sectors_out: list[dict] = []

    for sector, syms in (constituents or {}).items():
        rows: list[dict] = []
        for sym in (syms or []):
            if is_etf(sym):
                continue
            df = stocks.get(sym)
            if df is None:
                continue
            try:
                close = df["Close"].dropna()
                vol   = df["Volume"].dropna()
            except Exception:
                continue
            if len(close) < 64:
                continue

            price = float(close.iloc[-1])
            r1m   = _ret_pct(close, 22)
            r3m   = _r3m(close)
            if r3m is None:
                continue

            hi_window = close.iloc[-min(len(close), 252):]
            hi_max = float(hi_window.max())
            pct_from_high = ((price / hi_max - 1.0) * 100.0) if hi_max > 0 else 0.0

            ma50_window = close.iloc[-min(len(close), 50):]
            above_ma50 = price > float(ma50_window.mean())

            # ADTV (Cr) over last 20 sessions of aligned close*volume.
            cv = df[["Close", "Volume"]].dropna()
            look = min(20, len(cv))
            adtv_cr = (float((cv["Close"].iloc[-look:] * cv["Volume"].iloc[-look:]).mean()) / 1e7
                       if look else 0.0)

            rows.append({
                "symbol": sym,
                "price": round(price, 2),
                "r1m": round(r1m, 2) if r1m is not None else None,
                "r3m": round(r3m, 2),
                "pct_from_high": round(pct_from_high, 2),
                "rs_rating": int(rs_map.get(sym, 50)),
                "stage_label": _stage_label(close),
                "above_ma50": bool(above_ma50),
                "adtv_cr": round(adtv_cr, 2),
            })

        if not rows:
            continue

        r1m_vals = [r["r1m"] for r in rows if r["r1m"] is not None]
        r3m_vals = [r["r3m"] for r in rows]
        median_r1m = float(np.median(r1m_vals)) if r1m_vals else 0.0
        median_r3m = float(np.median(r3m_vals)) if r3m_vals else 0.0
        breadth_pct = 100.0 * sum(1 for r in rows if r["above_ma50"]) / len(rows)

        lv = (levels or {}).get(sector) or {}
        index_chg_pct = lv.get("chg_pct")
        pe = lv.get("pe")

        # Top constituents by RS rating DESC.
        top_sorted = sorted(rows, key=lambda r: r["rs_rating"], reverse=True)[:top_n]
        top_stocks = [{
            "symbol": r["symbol"],
            "rs": r["rs_rating"],
            "r1m": r["r1m"],
            "r3m": r["r3m"],
            "pct_from_high": r["pct_from_high"],
            "price": r["price"],
            "stage_label": r["stage_label"],
            "adtv_cr": r["adtv_cr"],
        } for r in top_sorted]

        sectors_out.append({
            "name": sector,
            "rank": None,                       # assigned after sorting
            "index_chg_pct": index_chg_pct,
            "pe": pe,
            "median_r1m": round(median_r1m, 2),
            "median_r3m": round(median_r3m, 2),
            "breadth_pct": round(breadth_pct, 1),
            "n_constituents": len(rows),
            "top_stocks": top_stocks,
        })

    # Rank sectors by median 3-month return DESC (1 = strongest).
    sectors_out.sort(key=lambda s: s["median_r3m"], reverse=True)
    for i, s in enumerate(sectors_out, start=1):
        s["rank"] = i

    return {"as_of": as_of, "sectors": sectors_out}


def run_sector_leaders(top_n: int = 5, force: bool = False) -> dict:
    """Rank sectoral indices by member momentum; return ranked sectors with their
    strongest constituents. Cached by bhavcopy date. Never raises."""
    try:
        bhav = _latest_bhavcopy_date()
        if bhav is not None:
            as_of = bhav.isoformat()
        else:
            from datetime import date as _date
            as_of = _date.today().isoformat()

        cache_key = f"{as_of}|{top_n}"
        with _CACHE_LOCK:
            if not force and cache_key in _CACHE:
                return _CACHE[cache_key]

        result = _compute(top_n=top_n, as_of=as_of)

        with _CACHE_LOCK:
            _CACHE[cache_key] = result
        return result
    except Exception as e:                              # pragma: no cover - defensive
        try:
            from datetime import date as _date
            as_of = _date.today().isoformat()
        except Exception:
            as_of = ""
        return {"as_of": as_of, "sectors": [], "error": str(e)}
