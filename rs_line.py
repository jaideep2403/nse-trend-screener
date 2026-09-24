"""
Relative-strength LINE vs the Nifty — the leadership tell that front-runs breakouts.

Our existing RS is a cross-sectional RATING (percentile of 3-month return): it says a
stock is strong TODAY. This is different and earlier: the RS *line* is price ÷ benchmark
over time. When that line makes a NEW HIGH while price itself has NOT yet — RS leading
price — it's the classic O'Neil/Minervini sign that a stock is quietly assuming
leadership before it breaks out. That "RS new high before price" flag is the single
cheapest early-monster signal we can compute from data we already have.

Benchmark = the equal-weight Nifty-50 proxy the rest of the app already uses (built
here from single-symbol history so we never pin a full deep universe in memory).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from analysis_utils import equal_weight_index, NIFTY_PROXY_SYMS

_BENCH: dict = {}          # (days) -> {"tag", "series"}


def _bhav_tag() -> str:
    try:
        from data_fetcher import _latest_bhavcopy_date
        d = _latest_bhavcopy_date()
        return d.isoformat() if d else "nodate"
    except Exception:
        return "nodate"


# ── Persistent benchmark cache ────────────────────────────────────────────────
# The equal-weight proxy is stable for a whole bhavcopy day, but `_BENCH` lived only in
# process memory — so EVERY restart threw it away and the next chart (or the startup warm)
# re-paid the full proxy-member rebuild. That rebuild was ~23s in the old glob-scan world
# and ~0.3s even now; persisting the finished series to disk, tagged by bhavcopy date,
# turns a restart into a ~1ms file read. The tag makes a new bhavcopy invalidate it for
# free, and we prune older-tag files so the store never grows.
def _bench_cache_dir():
    from data_fetcher import BHAV_DIR
    d = BHAV_DIR.parent / "bench_cache"
    try:
        d.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass
    return d


def _bench_disk_path(days: int, tag: str):
    return _bench_cache_dir() / f"bench_{int(days)}_{tag}.pkl"


def _bench_disk_load(days: int, tag: str):
    try:
        import pickle
        p = _bench_disk_path(days, tag)
        if p.exists():
            with open(p, "rb") as f:
                s = pickle.load(f)
            if isinstance(s, pd.Series) and len(s):
                return s
    except Exception:
        pass
    return None


def _bench_disk_save(days: int, tag: str, series) -> None:
    if series is None or len(series) == 0:
        return
    try:
        import pickle, tempfile, os
        d = _bench_cache_dir()
        keep = f"bench_{int(days)}_{tag}.pkl"
        for old in d.glob(f"bench_{int(days)}_*.pkl"):   # drop stale-tag snapshots
            if old.name != keep:
                try:
                    old.unlink()
                except Exception:
                    pass
        fd, tmp = tempfile.mkstemp(dir=str(d), prefix="bench.", suffix=".tmp")
        with os.fdopen(fd, "wb") as f:
            pickle.dump(series, f, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp, str(_bench_disk_path(days, tag)))
    except Exception:
        pass


def benchmark_series(days: int = 1900) -> pd.Series | None:
    """Equal-weight Nifty-50 proxy Close, date-indexed, rebased to 100. Cached per
    bhavcopy date — in memory AND on disk — built from single-symbol deep loads so it
    reuses the warm per-symbol snapshot store without rebuilding the whole universe."""
    tag = _bhav_tag()
    c = _BENCH.get(days)
    if c is not None and c["tag"] == tag:
        return c["series"]
    # Disk cache — survives restarts so a fresh process never re-pays the rebuild.
    disk = _bench_disk_load(days, tag)
    if disk is not None:
        _BENCH[days] = {"tag": tag, "series": disk}
        return disk
    try:
        import shared_universe as su
        closes = []
        for s in NIFTY_PROXY_SYMS:
            df = su.load_symbol_history(s, days)
            if df is not None and len(df) >= 60:
                closes.append(df["Close"].dropna())
        if not closes:
            return None
        combined = pd.concat(closes, axis=1).dropna(how="all")
        bench = equal_weight_index(combined)
        bench = bench if bench is not None and len(bench) >= 40 else None
    except Exception:
        bench = None
    _BENCH[days] = {"tag": tag, "series": bench}
    _bench_disk_save(days, tag, bench)
    return bench


def rs_line_for(dates: list[str], closes: list[float], lookback: int = 252) -> dict:
    """RS line aligned to the chart's own `dates`, plus the leadership flags.

    Returns:
      rs               : list[float|None] rebased to 100 at the window start (for plotting)
      new_high         : RS line at a new high over `lookback`
      high_before_price: RS at a new high while PRICE is NOT — RS leading price (the tell)
      above_zero       : Mansfield RS > 0 (outperforming its own ~1yr average)
      slope_up         : RS line rising over the last ~4 weeks
      pct_from_high    : how far the RS line sits below its `lookback` high (0 = at high)
    """
    out = {"rs": [], "new_high": False, "high_before_price": False,
           "above_zero": False, "slope_up": False, "pct_from_high": None}
    if not dates or not closes or len(dates) != len(closes):
        return out
    bench = benchmark_series()
    if bench is None or bench.empty:
        return out

    # Align the benchmark to the chart's exact dates (forward-fill across any holiday
    # the stock traded but a proxy member didn't).
    idx = pd.to_datetime(pd.Index(dates))
    b = bench.copy()
    b.index = pd.to_datetime(b.index)
    b = b[~b.index.duplicated(keep="last")].sort_index()
    b_al = b.reindex(idx, method="ffill")

    px = pd.Series(closes, index=idx, dtype=float)
    ratio = px / b_al
    ratio = ratio.replace([np.inf, -np.inf], np.nan)
    if ratio.notna().sum() < 30:
        return out

    first = ratio.dropna().iloc[0]
    rs = (ratio / first * 100.0) if first and first > 0 else ratio
    out["rs"] = [None if pd.isna(v) else round(float(v), 2) for v in rs]

    r = ratio.to_numpy(dtype=float)
    c = px.to_numpy(dtype=float)
    n = len(r)
    lb = min(lookback, n)
    r_win = r[-lb:]
    c_win = c[-lb:]
    rmax = np.nanmax(r_win)
    cmax = np.nanmax(c_win)
    r_now = r[-1]
    c_now = c[-1]
    if np.isfinite(r_now) and np.isfinite(rmax) and rmax > 0:
        out["pct_from_high"] = round((r_now / rmax - 1.0) * 100, 1)
        # "new high" = RS within 0.5% of its lookback high
        out["new_high"] = bool(r_now >= rmax * 0.995)
        price_at_high = bool(np.isfinite(c_now) and c_now >= cmax * 0.995)
        out["high_before_price"] = bool(out["new_high"] and not price_at_high)

    # Mansfield-style zero line: ratio vs its own ~1yr average.
    ma = pd.Series(r).rolling(min(252, n), min_periods=30).mean().to_numpy()
    if np.isfinite(ma[-1]) and ma[-1] > 0:
        out["above_zero"] = bool(r_now / ma[-1] - 1.0 > 0)

    if n >= 25 and np.isfinite(r[-1]) and np.isfinite(r[-21]) and r[-21] > 0:
        out["slope_up"] = bool(r[-1] > r[-21])
    return out


def rs_signals_series(px: "pd.Series", lookback: int = 252) -> dict:
    """Fast RS flags for a bulk scan: takes a date-indexed Close series, returns just the
    leadership flags (no plotting list). Reindexes the cached benchmark straight onto the
    stock's dates with ffill — O(n), no per-call rebuild."""
    out = {"new_high": False, "high_before_price": False, "above_zero": False,
           "slope_up": False, "pct_from_high": None}
    bench = benchmark_series()
    if bench is None or bench.empty or px is None or len(px) < 40:
        return out
    b = bench
    if not b.index.is_monotonic_increasing:
        b = b.sort_index()
    b_al = b.reindex(px.index, method="ffill")
    ratio = (px / b_al).replace([np.inf, -np.inf], np.nan)
    r = ratio.to_numpy(dtype=float)
    c = px.to_numpy(dtype=float)
    n = len(r)
    if np.isfinite(r).sum() < 30:
        return out
    lb = min(lookback, n)
    rmax = np.nanmax(r[-lb:]); cmax = np.nanmax(c[-lb:])
    r_now, c_now = r[-1], c[-1]
    if np.isfinite(r_now) and np.isfinite(rmax) and rmax > 0:
        out["pct_from_high"] = round((r_now / rmax - 1.0) * 100, 1)
        out["new_high"] = bool(r_now >= rmax * 0.995)
        price_high = bool(np.isfinite(c_now) and c_now >= cmax * 0.995)
        out["high_before_price"] = bool(out["new_high"] and not price_high)
    ma = pd.Series(r).rolling(min(252, n), min_periods=30).mean().to_numpy()
    if np.isfinite(ma[-1]) and ma[-1] > 0:
        out["above_zero"] = bool(r_now / ma[-1] - 1.0 > 0)
    if n >= 25 and np.isfinite(r[-1]) and np.isfinite(r[-21]) and r[-21] > 0:
        out["slope_up"] = bool(r[-1] > r[-21])
    return out
