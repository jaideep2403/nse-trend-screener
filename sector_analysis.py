"""
Sector Rotation Analysis
- Fetches Nifty50 (^NSEI) as benchmark via Ticker().history()
- Computes sector strength by aggregating constituent stock OHLCV data
- Reuses ohlcv_cache from screener (zero extra downloads if screener ran first)
- No FII/DII, no NSE API calls, no bulk yf.download()
"""
import time
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

from screener import MIN_BARS, ohlcv_cache
from data_fetcher import fetch_ohlcv

# ── Sector → NSE stock symbols ────────────────────────────────────────────────
SECTOR_STOCKS = {
    "IT":                ["TCS", "HCLTECH", "WIPRO", "COFORGE", "PERSISTENT"],
    "Auto":              ["MARUTI", "TVSMOTOR", "EICHERMOT", "SONACOMS", "MOTHERSON"],
    "Bank & Finance":    ["HDFCBANK", "ICICIBANK", "SBIN", "AXISBANK", "BAJFINANCE"],
    "Pharma":            ["SUNPHARMA", "DRREDDY", "DIVISLAB", "CIPLA", "TORNTPHARM"],
    "FMCG":              ["HINDUNILVR", "ITC", "NESTLEIND", "DABUR", "MARICO"],
    "Metal":             ["JSWSTEEL", "TATASTEEL", "VEDL", "HINDALCO", "SAIL"],
    "Energy":            ["RELIANCE", "ONGC", "NTPC", "POWERGRID", "TATAPOWER"],
    "Realty":            ["DLF", "GODREJPROP", "OBEROIRLTY", "PRESTIGE", "PHOENIXLTD"],
    "Infra & Cap Goods": ["LT", "ADANIPORTS", "BHEL", "CUMMINSIND", "BEML"],
    "Defence":           ["BEL", "HAL", "DATAPATTNS", "GRSE", "COCHINSHIP"],
}

# ── In-memory cache ────────────────────────────────────────────────────────────
_cache = {"data": None, "ts": 0}
CACHE_TTL = 1800  # 30 minutes


# ── Utility helpers ────────────────────────────────────────────────────────────

def _safe(v, default=0.0):
    try:
        f = float(v)
        return f if np.isfinite(f) else default
    except Exception:
        return default


def _ret(series, bars):
    # BUG-026 FIX: an N-bar return compares today's close to the close N bars
    # ago, i.e. iloc[-bars], not iloc[-bars-1].  Previously this returned an
    # (N+1)-bar return which subtly inflated/deflated the result.
    if len(series) < bars + 1:
        return 0.0
    return round((_safe(series.iloc[-1]) / _safe(series.iloc[-bars]) - 1) * 100, 2)


def _mfi(df, period=14):
    try:
        hi  = df["High"].dropna()
        lo  = df["Low"].dropna()
        cl  = df["Close"].dropna()
        vol = df["Volume"].dropna()
        idx = cl.index.intersection(hi.index).intersection(lo.index).intersection(vol.index)
        if len(idx) < period + 2:
            return 50.0
        tp   = (hi[idx] + lo[idx] + cl[idx]) / 3
        rmf  = tp * vol[idx]
        pos  = rmf.where(tp > tp.shift(1), 0.0)
        neg  = rmf.where(tp < tp.shift(1), 0.0).abs()
        ps   = pos.rolling(period).sum().iloc[-1]
        ns   = neg.rolling(period).sum().iloc[-1]
        # BUG-034 FIX: when there are zero down days the original code returned
        # MFI=100 (extreme overbought), which is misleading for thinly traded
        # series. Return the neutral midpoint 50 instead.
        if not ns or ns == 0:
            return 50.0
        return round(100 - 100 / (1 + (ps / ns)), 1)
    except Exception:
        return 50.0


def _rel_vol(df, period=20):
    try:
        vol = df["Volume"].dropna()
        if len(vol) < period + 1:
            return 1.0
        avg = float(vol.iloc[-period - 1:-1].mean())
        cur = float(vol.iloc[-1])
        return round(cur / avg, 2) if avg > 0 else 1.0
    except Exception:
        return 1.0


def _obv_trend(df, period=20):
    try:
        cl  = df["Close"].dropna()
        vol = df["Volume"].dropna()
        idx = cl.index.intersection(vol.index)
        if len(idx) < period + 2:
            return 0.0
        direction = cl[idx].diff().apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))
        obv   = (direction * vol[idx]).cumsum()
        slc   = obv.iloc[-period:]
        slope = float(np.polyfit(np.arange(len(slc)), slc.values, 1)[0])
        norm  = abs(float(slc.mean())) or 1
        return round(float(np.clip(slope / norm, -1, 1)), 4)
    except Exception:
        return 0.0


def _inflow_score(rs_1m, rs_3m, mfi, rel_v, obv, all_rs3m=None):
    """
    BUG-025 FIX: rs_3m normalization now uses dynamic bounds from the current scan's
    distribution rather than fixed ±30%. This prevents clipping sector leaders.
    Pass all_rs3m as a list of all sectors' rs_3m values for dynamic bounds.
    """
    def norm(v, lo, hi):
        return max(0.0, min(100.0, (v - lo) / (hi - lo) * 100))

    # Dynamic RS-3M bounds based on current scan distribution
    if all_rs3m and len(all_rs3m) >= 2:
        rs3m_lo = min(all_rs3m)
        rs3m_hi = max(all_rs3m)
        # Ensure minimum range of 10% to avoid division-by-zero
        if rs3m_hi - rs3m_lo < 10:
            mid = (rs3m_hi + rs3m_lo) / 2
            rs3m_lo = mid - 5
            rs3m_hi = mid + 5
    else:
        rs3m_lo, rs3m_hi = -30, 30  # fallback to original fixed bounds

    return round(
        norm(rs_1m, -15, 15) * 0.35 +
        norm(rs_3m, rs3m_lo, rs3m_hi) * 0.25 +
        norm(mfi,    20, 80) * 0.20 +
        norm(rel_v,  0.5, 2.5) * 0.10 +
        norm(obv * 100, -50, 50) * 0.10,
        1
    )


# ── Synthetic Nifty50 benchmark from equal-weighted constituent closes ─────────

# Use the canonical Nifty proxy basket from analysis_utils so every module
# uses the same benchmark composition. The old local list was a duplicate
# that would silently drift out of sync if analysis_utils.NIFTY_PROXY_SYMS
# were ever updated, giving Sector Rotation a different benchmark than RS
# calculations elsewhere.
from analysis_utils import NIFTY_PROXY_SYMS as _NIFTY50_SYMS

def fetch_nifty():
    """Build equal-weighted benchmark from cached Nifty50 closes. No network call.
    BUG-005 FIX: Falls back to bhavcopy-based data when screener's ohlcv_cache is empty
    (e.g. when sector tab is opened before screener has run).
    # TODO BUG-009: Four different Nifty proxy implementations exist across the codebase
    # (header_data.py, market_breadth.py, industry_groups.py, sector_analysis.py, portfolio.py).
    # These should be consolidated into a shared nifty_proxy.py module.
    """
    closes = []
    for sym in _NIFTY50_SYMS:
        df = ohlcv_cache.get(f"{sym}.NS")
        if df is not None and "Close" in df.columns and len(df) >= 63:
            closes.append(df["Close"].dropna())

    # BUG-FIX: fallback used `_NIFTY50_SYMS[:10]` — 10 stocks, 9 of which would
    # contribute (INFOSYS was missing). Cold-start Nifty proxy was completely
    # different from warm-cache 20-stock proxy → sector rankings flipped between
    # first call after restart and subsequent calls. Now use the full _NIFTY50_SYMS
    # list (20 stocks) so cold and warm runs produce the same benchmark.
    if not closes:
        try:
            from data_fetcher import fetch_ohlcv
            fallback_tickers = [f"{s}.NS" for s in _NIFTY50_SYMS]   # full list (20)
            got = fetch_ohlcv(fallback_tickers, min_bars=63)
            for t, df in got.items():
                if "Close" in df.columns and len(df) >= 63:
                    closes.append(df["Close"].dropna())
        except Exception:
            pass

    if not closes:
        return None
    # BUG-FIX: rebase-to-100 equal-weight (was raw price avg)
    from analysis_utils import equal_weight_index
    combined = pd.concat(closes, axis=1)
    combined = combined.dropna(how="all")
    benchmark = equal_weight_index(combined)
    return benchmark if len(benchmark) >= 20 else None


# ── Per-stock metrics ──────────────────────────────────────────────────────────

def _stock_metrics(ticker, df, nifty_close):
    try:
        cl = df["Close"].dropna()
        if len(cl) < 40:
            return None
        price   = _safe(cl.iloc[-1])
        ma50    = _safe(cl.rolling(50).mean().iloc[-1])  if len(cl) >= 50  else 0
        ma100   = _safe(cl.rolling(100).mean().iloc[-1]) if len(cl) >= 100 else 0
        ma200   = _safe(cl.rolling(200).mean().iloc[-1]) if len(cl) >= 200 else 0
        # 52-week high (252-bar window), not all-loaded-history max — keeps this
        # consistent with the 252-bar high used in breadth/edge/trending.
        ath     = _safe((cl.iloc[-252:] if len(cl) >= 252 else cl).max())
        pct_ath = round((price - ath) / ath * 100, 2) if ath else 0
        in_uptrend = (price > ma50 > ma100 > ma200) and ma200 > 0

        r1m  = _ret(cl, 21);  r3m  = _ret(cl, 63)
        r6m  = _ret(cl, 126); r12m = _ret(cl, 252)

        rel_1m = rel_3m = None
        if nifty_close is not None:
            n1m    = _ret(nifty_close, 21);  n3m = _ret(nifty_close, 63)
            rel_1m = round(r1m - n1m, 2)
            rel_3m = round(r3m - n3m, 2)

        mfi_v  = _mfi(df)
        relv_v = _rel_vol(df)
        obv_v  = _obv_trend(df)
        # Inflow score is recomputed later with dynamic bounds once we know
        # the full cross-sectional rel_3m distribution. Initialised to 0 here
        # to avoid a wasted per-stock calculation that gets immediately
        # overwritten in the loop at line ~387.
        inflow = 0

        # Compute ADTV (₹ Cr) — used as liquidity weight in sector aggregation.
        # ADTV is the most honest proxy for "how much real money trades this stock".
        vol_series = df["Volume"].dropna() if "Volume" in df.columns else None
        if vol_series is not None and len(vol_series) >= 20:
            avg_vol = float(vol_series.iloc[-20:].mean())
            adtv_cr = round(avg_vol * price / 1e7, 2) if price > 0 else 0.0
        else:
            adtv_cr = 0.0

        symbol = ticker.replace(".NS", "")
        return {
            "ticker":      ticker,
            "symbol":      symbol,
            "price":       round(price, 2),
            "ma50":        round(ma50, 2),
            "ma200":       round(ma200, 2),
            "pct_ath":     pct_ath,
            "in_uptrend":  in_uptrend,
            "r1m":         r1m,
            "r3m":         r3m,
            "r6m":         r6m,
            "r12m":        r12m,
            "rel_1m":      rel_1m,
            "rel_3m":      rel_3m,
            "mfi":         mfi_v,
            "rel_volume":  relv_v,
            "obv_trend":   obv_v,
            "inflow_score": inflow,
            "adtv_cr":      adtv_cr,
        }
    except Exception:
        return None


# ── Sector aggregation ─────────────────────────────────────────────────────────

def _aggregate_sector(stock_rows, sector_name):
    if not stock_rows:
        return None

    n = len(stock_rows)

    # BUG-FIX: previously used simple average — a 1L Cr name (HDFCBANK) counted
    # equally with a 1K Cr name (BANDHANBNK). Real sector indices weight by
    # free-float market cap. We don't have free-float here, so use ADTV
    # (turnover) as a robust market-cap proxy: it inherently weights by
    # liquidity, which is what investors actually transact in.
    def liq_weighted_avg(key, fallback_to_simple=True):
        rows = [r for r in stock_rows if r.get(key) is not None]
        if not rows:
            return 0.0
        # Use ADTV (in ₹Cr) as weight; fallback to equal if all are 0/None
        weights = [max(0.0, r.get("adtv_cr", 0.0) or 0.0) for r in rows]
        total_w = sum(weights)
        if total_w <= 0:
            if not fallback_to_simple:
                return 0.0
            # Pure equal-weight fallback
            return round(sum(r[key] for r in rows) / len(rows), 2)
        return round(sum(r[key] * w for r, w in zip(rows, weights)) / total_w, 2)

    # Keep equal-weight available for breadth-style metrics where each stock counts as 1
    def avg(key):
        vals = [r[key] for r in stock_rows if r.get(key) is not None]
        return round(sum(vals) / len(vals), 2) if vals else 0.0

    breadth    = round(sum(1 for r in stock_rows if r.get("in_uptrend")) / n * 100, 1)
    top5_sorted = sorted(stock_rows, key=lambda r: r.get("inflow_score", 0), reverse=True)

    top5 = [
        {
            "symbol":       r["symbol"],
            "price":        r["price"],
            "r1m":          r["r1m"],
            "r3m":          r["r3m"],
            "mfi":          r["mfi"],
            "inflow_score": r["inflow_score"],
            "in_uptrend":   r["in_uptrend"],
            "pct_ath":      r["pct_ath"],
        }
        for r in top5_sorted[:5]
    ]

    # Use liquidity-weighted aggregation for return / inflow metrics
    # (so a tiny stock's 200% spike doesn't drag the whole sector up)
    return {
        "sector":       sector_name,
        "r1m":          liq_weighted_avg("r1m"),
        "r3m":          liq_weighted_avg("r3m"),
        "r6m":          liq_weighted_avg("r6m"),
        "r12m":         liq_weighted_avg("r12m"),
        "mfi":          liq_weighted_avg("mfi"),
        "rel_volume":   liq_weighted_avg("rel_volume"),
        "obv_trend":    liq_weighted_avg("obv_trend"),
        "breadth":      breadth,
        "inflow_score": liq_weighted_avg("inflow_score"),
        "num_stocks":   n,
        "top5":         top5,
    }


# ── Main entry point ───────────────────────────────────────────────────────────

def run_sector_analysis():
    # Serve from cache only if it has real data
    if (_cache["data"]
            and (time.time() - _cache["ts"] < CACHE_TTL)
            and len(_cache["data"].get("sectors", [])) > 0):
        return _cache["data"]

    # 1. Nifty benchmark (single Ticker call)
    nifty_close = fetch_nifty()

    # 2. All sector tickers
    # BUG-027 FIX: hard-coded 5-stocks-per-sector lists were stale and far too
    # narrow. Pull the full sector → symbol mapping from sector_mapper which
    # is auto-derived from current Nifty500 constituents. Falls back to the
    # hard-coded SECTOR_STOCKS if the mapper is unavailable.
    sector_map: dict[str, list[str]] = {k: list(v) for k, v in SECTOR_STOCKS.items()}
    try:
        from sector_mapper import get_enriched_sector_map
        enriched = get_enriched_sector_map()   # {symbol: sector}
        dyn: dict[str, list[str]] = {}
        for sym, sect in enriched.items():
            dyn.setdefault(sect, []).append(sym)
        # Merge dyn into sector_map for matching sectors
        for sect, syms in dyn.items():
            if sect in sector_map:
                # Union dyn + hard-coded (keep order, drop dups)
                seen = set()
                merged = []
                for s in syms + sector_map[sect]:
                    if s not in seen:
                        seen.add(s); merged.append(s)
                sector_map[sect] = merged
            else:
                sector_map[sect] = syms
    except Exception:
        pass

    all_symbols = list({s for syms in sector_map.values() for s in syms})
    all_tickers = [f"{s}.NS" for s in all_symbols]

    # 3. Reuse screener OHLCV cache (zero extra downloads if screener ran first)
    ohlcv   = {t: df for t, df in ohlcv_cache.items() if t in all_tickers}
    missing = [t for t in all_tickers if t not in ohlcv]

    # 4. Fetch only what's truly missing via NSE bhavcopy (no Yahoo)
    if missing:
        got = fetch_ohlcv(missing, min_bars=40)
        ohlcv.update(got)

    # 5. Per-stock metrics
    stock_data = {}
    for ticker, df in ohlcv.items():
        if ticker not in all_tickers:
            continue
        row = _stock_metrics(ticker, df, nifty_close)
        if row:
            stock_data[ticker] = row

    # BUG-025 FIX: collect all rel_3m values across the scan for dynamic normalization
    all_rs3m = [r["rel_3m"] for r in stock_data.values()
                if r.get("rel_3m") is not None]
    # Recompute inflow scores with dynamic bounds now that full distribution is known
    for ticker, row in stock_data.items():
        row["inflow_score"] = _inflow_score(
            row.get("rel_1m") if row.get("rel_1m") is not None else row.get("r1m", 0),
            row.get("rel_3m") if row.get("rel_3m") is not None else row.get("r3m", 0),
            row.get("mfi", 50),
            row.get("rel_volume", 1.0),
            row.get("obv_trend", 0.0),
            all_rs3m=all_rs3m,
        )

    # 6. Aggregate per sector, rank by inflow score
    sector_results = []
    top5_map       = {}
    for sector, symbols in sector_map.items():
        rows = [stock_data[f"{s}.NS"] for s in symbols if f"{s}.NS" in stock_data]
        agg  = _aggregate_sector(rows, sector)
        if agg:
            top5_map[sector] = agg.pop("top5")
            sector_results.append(agg)

    sector_results.sort(key=lambda r: r["inflow_score"], reverse=True)
    for i, r in enumerate(sector_results):
        r["rank"] = i + 1

    result = {
        "sectors":     sector_results,
        "top5":        top5_map,
        "computed_at": int(time.time()),
    }

    if sector_results:
        _cache["data"] = result
        _cache["ts"]   = time.time()

    return result
