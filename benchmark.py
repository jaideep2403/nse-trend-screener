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

import threading
import time
import pandas as pd

from data_fetcher import _weekdays_back, _download_one_day

# Primary = Nippon Nifty 50 ETF. Fallbacks are other liquid Nifty-50 ETFs that
# also live in the EQ bhavcopy, in case NIFTYBEES is missing on some old days.
BENCH_SYMBOL     = "NIFTYBEES"
FALLBACK_SYMBOLS = ["SETFNIF50", "NIFTY1"]

# ── Dividend asymmetry correction ────────────────────────────────────────────
# NIFTYBEES is a TOTAL-return proxy — the ETF reinvests the index's dividends — but
# stock returns come from PRICE-only bhavcopy. Comparing a price-only stock return
# to a dividend-reinvested benchmark handicaps every stock by ~the Nifty dividend
# yield (~1.2–1.4%/yr), biasing alpha DOWN (worst against high-yield names). To
# compare like with like, subtract the pro-rated dividend from the benchmark return.
# ~1.3%/yr is the Nifty 50 trailing yield; adjust here if it drifts.
NIFTY_ANNUAL_DIV_YIELD = 0.013


def dividend_drag(bars: int) -> float:
    """Pro-rated NIFTY dividend yield over `bars` trading days, as a FRACTION.
    Subtract from a NIFTYBEES total-return to get a price-comparable benchmark."""
    try:
        return NIFTY_ANNUAL_DIV_YIELD * (max(0, int(bars)) / 252.0)
    except Exception:
        return 0.0


_cache: dict = {"series": None, "ts": 0.0, "days": 0}

# Serialises the ~1600-day rebuild — see the stampede note in get_benchmark().
_COMPUTE_LOCK = threading.Lock()
CACHE_TTL = 6 * 3600   # 6 h — same cadence as the rest of the EOD pipeline


def _clean(s: pd.Series) -> pd.Series:
    s = s.dropna()
    if not isinstance(s.index, pd.DatetimeIndex):
        s.index = pd.to_datetime(s.index)
    # strip any tz so .asof() never hits naive/aware comparison errors
    if getattr(s.index, "tz", None) is not None:
        s.index = s.index.tz_localize(None)
    return s[~s.index.duplicated(keep="last")].sort_index().astype(float)


def _slice_to(s: pd.Series, days: int) -> pd.Series:
    """Trim a cached benchmark series to the caller's requested lookback so a
    longer cached series can safely serve a shorter request (see get_benchmark)."""
    try:
        wd = _weekdays_back(days)
        if not wd:
            return s
        return s[s.index >= pd.Timestamp(min(wd))]
    except Exception:
        return s


def get_benchmark(days: int = 1600) -> pd.Series | None:
    """
    Cached NIFTYBEES close series loaded directly from the bhavcopy day cache.
    `days` is calendar lookback; the on-disk cache currently spans ~6 years.
    """
    hit = _cached_for(days)
    if hit is not None:
        return hit

    # CACHE STAMPEDE FIX (2026-08-03). Rebuilding this series walks ~1600 weekday
    # bhavcopies doing a boolean filter per day, and the cache check above was
    # unguarded. On a cold process THREE callers hit it at once — the guardian
    # startup sweep, the /api/portfolio request, and result_cache's breakdown
    # annotation — so each rebuilt the whole series independently while holding the
    # GIL. Confirmed from a live SIGUSR1 dump: guardian and api_portfolio_list were
    # both inside this function's recompute loop. That is what left "My Portfolio"
    # spinning on "Refreshing…". One thread builds; the others wait and reuse it.
    with _COMPUTE_LOCK:
        hit = _cached_for(days)
        if hit is not None:
            return hit
        return _build_benchmark(days)


# ── Smallcap benchmark ──────────────────────────────────────────────────────────
# WHY: a small/micro-cap momentum + delivery-accumulation book must be judged
# against a SMALL-CAP index, not the Nifty 50. Small-caps beat large-caps over this
# sample, so benchmarking to NIFTYBEES flatters the strategy — it can "beat the
# market" while trailing a cheap smallcap index fund. HDFCSML250 (HDFC Nifty
# Smallcap 250 ETF) is the most liquid smallcap ETF in the bhavcopy (ADTV ~₹26 Cr).
# CAVEAT surfaced to callers: it only lists from 2023-02, so it can benchmark the
# recent window, NOT the full 7-yr backtest. That limitation is real and honest —
# there is no longer-history investable smallcap total-return series in the cache.
_SC_SYMBOL = "HDFCSML250"
_SC_FALLBACKS = ["MOSMALL250", "MIDSMALL"]   # thinner/shorter; last resort only
_sc_cache: dict = {"series": None, "ts": 0.0, "days": 0}


def get_smallcap_benchmark(days: int = 900) -> pd.Series | None:
    """Split-adjusted close series for the smallcap benchmark ETF, or None.

    Same construction as get_benchmark (bhavcopy walk + split adjust), for a
    smallcap ETF. History is short (~2023-02→), so callers should intersect on
    dates and report the actual covered window rather than assume full depth.
    """
    now = time.time()
    c = _sc_cache
    if (c["series"] is not None and now - c["ts"] < CACHE_TTL and c["days"] >= days * 0.9):
        return _slice_to(c["series"], days)
    with _COMPUTE_LOCK:
        c = _sc_cache
        if (c["series"] is not None and now - c["ts"] < CACHE_TTL and c["days"] >= days * 0.9):
            return _slice_to(c["series"], days)
        recs: dict[pd.Timestamp, float] = {}
        for dt in _weekdays_back(days):
            df = _download_one_day(dt)
            if df is None:
                continue
            for sym in [_SC_SYMBOL] + _SC_FALLBACKS:
                sub = df[df["Symbol"] == sym]
                if not sub.empty:
                    recs[pd.Timestamp(dt)] = float(sub.iloc[0]["Close"])
                    break
        if not recs:
            return None
        s = _clean(pd.Series(recs))
        try:
            from analysis_utils import adjust_for_splits
            s = adjust_for_splits(pd.DataFrame({"Close": s}), _SC_SYMBOL)["Close"].astype(float)
        except Exception:
            pass
        _sc_cache.update({"series": s, "ts": now, "days": days})
        return s


def _cached_for(days: int) -> pd.Series | None:
    """Return the cached series trimmed to `days`, or None if it can't serve it."""
    now = time.time()
    c = _cache
    if (c["series"] is not None and now - c["ts"] < CACHE_TTL
            and c["days"] >= days * 0.9):
        # CACHE-CORRECTNESS FIX (2026-07-26): this used to return the cached
        # series VERBATIM, so a cached LONGER series satisfied a SHORTER request.
        # That is not a harmless over-fetch: system_backtest takes its master
        # calendar from this series (`cal = bench.index`) while loading stocks
        # with its own `days`, so the two silently covered DIFFERENT periods —
        # the extra early bars had no stock data, every pick lookup missed, and
        # those rebalances sat in cash, depressing CAGR and inflating drawdown.
        # Worse, it made results depend on CALL ORDER: run a 2800-day backtest
        # first and every later 1600-day run in the same process was corrupted.
        # Slicing to the requested window keeps the cache's speed and makes the
        # answer order-independent.
        return _slice_to(c["series"], days)
    return None


def _build_benchmark(days: int) -> pd.Series | None:
    """Walk the bhavcopy day cache and rebuild the series. Callers MUST hold
    _COMPUTE_LOCK — this is the expensive path the stampede fix guards."""
    now = time.time()
    c = _cache
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
    # CRITICAL-FIX (2026-07-25): backward-adjust the ETF for splits/bonuses.
    # NIFTYBEES did a 1:10 split on 2019-12-19 (₹1292.54 → ₹130.20). Un-adjusted,
    # that reads as a −89.9% single-day crash: benchmark annualised vol showed
    # 37.5% instead of the true ~15.8%, and every MA200 / 6-month window spanning
    # the date was corrupted — which delayed the regime engine's post-COVID BULL
    # call by ~a month (it sat in cash during part of the 2020 recovery).
    # edge_engine already adjusted its own benchmark copy; this module never did,
    # so every consumer of get_benchmark() (system_backtest, regime_engine, drift
    # monitor) inherited the artifact. Same canonical helper, one source of truth.
    try:
        from analysis_utils import adjust_for_splits
        s = adjust_for_splits(pd.DataFrame({"Close": s}), BENCH_SYMBOL)["Close"].astype(float)
    except Exception:
        pass   # never let an adjustment failure take the benchmark offline
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
