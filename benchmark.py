"""
Benchmark module — the REAL Nifty 50 benchmark from the NIFTYBEES ETF.

Why this exists
---------------
Every "edge" claim in this app is only meaningful relative to what you'd have
earned doing nothing (buy-and-hold the index). Until now the app compared
against a synthetic equal-weight 20-stock proxy, which is NOT the cap-weighted
Nifty 50 and tracks it poorly. NIFTYBEES (Nippon India Nifty 50 ETF) trades in
the EQ series and is therefore already sitting in the bhavcopy cache, so we can
use the actual index — at zero new NSE calls — and it reflects dividend
reinvestment (the ETF accumulates/distributes index dividends).

Single source of truth for benchmark-relative (excess) return everywhere:
backtester, edge engine, alpha engine, and the live portfolio.
"""
from __future__ import annotations

import time
import pandas as pd

from data_fetcher import _weekdays_back, _download_one_day

# Primary = Nippon Nifty 50 ETF. Fallbacks are other liquid Nifty-50 ETFs that
# also live in the EQ bhavcopy, in case NIFTYBEES is missing on some old days.
BENCH_SYMBOL     = "NIFTYBEES"
FALLBACK_SYMBOLS = ["SETFNIF50", "NIFTY1"]

_cache: dict = {"series": None, "ts": 0.0, "days": 0}
CACHE_TTL = 6 * 3600   # 6 h — same cadence as the rest of the EOD pipeline


def _clean(s: pd.Series) -> pd.Series:
    s = s.dropna()
    if not isinstance(s.index, pd.DatetimeIndex):
        s.index = pd.to_datetime(s.index)
    # strip any tz so .asof() never hits naive/aware comparison errors
    if getattr(s.index, "tz", None) is not None:
        s.index = s.index.tz_localize(None)
    return s[~s.index.duplicated(keep="last")].sort_index().astype(float)


def get_benchmark(days: int = 1600) -> pd.Series | None:
    """
    Cached NIFTYBEES close series loaded directly from the bhavcopy day cache.
    `days` is calendar lookback; the on-disk cache currently spans ~6 years.
    """
    now = time.time()
    c = _cache
    if (c["series"] is not None and now - c["ts"] < CACHE_TTL
            and c["days"] >= days * 0.9):
        return c["series"]

    recs: dict[pd.Timestamp, float] = {}
    for dt in _weekdays_back(days):
        df = _download_one_day(dt)
        if df is None:
            continue
        for sym in [BENCH_SYMBOL] + FALLBACK_SYMBOLS:
            sub = df[df["Symbol"] == sym]
            if not sub.empty:
                recs[pd.Timestamp(dt)] = float(sub.iloc[0]["Close"])
                break
    if not recs:
        return None
    s = _clean(pd.Series(recs))
    c.update({"series": s, "ts": now, "days": days})
    return s


def benchmark_return(d0, d1, bench: pd.Series | None = None) -> float | None:
    """
    Benchmark % return between two dates, holiday-safe via asof().
    Returns None when either date is outside the available benchmark history.
    """
    if bench is None:
        bench = get_benchmark()
    if bench is None or len(bench) < 2:
        return None
    try:
        d0 = pd.Timestamp(d0)
        d1 = pd.Timestamp(d1)
        a = bench.asof(d0)
        b = bench.asof(d1)
        if pd.isna(a) or pd.isna(b) or a <= 0:
            return None
        return (b / a - 1.0) * 100.0
    except Exception:
        return None


def benchmark_equity(dates, base: float = 100.0,
                     bench: pd.Series | None = None) -> list[float] | None:
    """
    Normalised benchmark equity curve (base 100) sampled at the given dates —
    for plotting strategy equity against buy-and-hold Nifty on the same axis.
    """
    if bench is None:
        bench = get_benchmark()
    if bench is None or not len(dates):
        return None
    # BUG-FIX: find the first VALID benchmark value up front. The old loop
    # padded `base` for dates preceding benchmark history and then normalised
    # later points to a later `first`, skewing the curve's left edge (and any
    # strategy-vs-bench comparison anchored there).
    raw = [bench.asof(pd.Timestamp(d)) for d in dates]
    first = next((float(v) for v in raw if not pd.isna(v)), None)
    if first is None or first <= 0:
        return None
    vals = []
    for v in raw:
        if pd.isna(v):
            vals.append(vals[-1] if vals else base)
        else:
            vals.append(round(float(v) / first * base, 2))
    return vals
