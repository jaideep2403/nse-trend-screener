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
    if len(series) < bars + 1:
        return 0.0
    return round((_safe(series.iloc[-1]) / _safe(series.iloc[-bars - 1]) - 1) * 100, 2)


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
        return round(100 - 100 / (1 + (ps / ns if ns else 1e9)), 1)
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


def _inflow_score(rs_1m, rs_3m, mfi, rel_v, obv):
    def norm(v, lo, hi):
        return max(0.0, min(100.0, (v - lo) / (hi - lo) * 100))
    return round(
        norm(rs_1m, -15, 15) * 0.35 +
        norm(rs_3m, -30, 30) * 0.25 +
        norm(mfi,    20, 80) * 0.20 +
        norm(rel_v,  0.5, 2.5) * 0.10 +
        norm(obv * 100, -50, 50) * 0.10,
        1
    )


# ── Synthetic Nifty50 benchmark from equal-weighted constituent closes ─────────

_NIFTY50_SYMS = [
    "RELIANCE", "TCS", "HDFCBANK", "ICICIBANK", "INFOSYS",
    "HINDUNILVR", "SBIN", "BHARTIARTL", "ITC", "KOTAKBANK",
    "AXISBANK", "WIPRO", "HCLTECH", "MARUTI", "BAJFINANCE",
    "TITAN", "NTPC", "POWERGRID", "NESTLEIND", "SUNPHARMA",
]

def fetch_nifty():
    """Build equal-weighted benchmark from cached Nifty50 closes. No network call."""
    closes = []
    for sym in _NIFTY50_SYMS:
        df = ohlcv_cache.get(f"{sym}.NS")
        if df is not None and "Close" in df.columns and len(df) >= 63:
            closes.append(df["Close"].dropna())
    if not closes:
        return None
    combined = pd.concat(closes, axis=1)
    combined = combined.dropna(how="all")
    benchmark = combined.mean(axis=1)
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
        ath     = _safe(cl.max())
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
        inflow = _inflow_score(
            rel_1m if rel_1m is not None else r1m,
            rel_3m if rel_3m is not None else r3m,
            mfi_v, relv_v, obv_v,
        )

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
        }
    except Exception:
        return None


# ── Sector aggregation ─────────────────────────────────────────────────────────

def _aggregate_sector(stock_rows, sector_name):
    if not stock_rows:
        return None

    n = len(stock_rows)

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

    return {
        "sector":       sector_name,
        "r1m":          avg("r1m"),
        "r3m":          avg("r3m"),
        "r6m":          avg("r6m"),
        "r12m":         avg("r12m"),
        "mfi":          avg("mfi"),
        "rel_volume":   avg("rel_volume"),
        "obv_trend":    avg("obv_trend"),
        "breadth":      breadth,
        "inflow_score": avg("inflow_score"),
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
    all_symbols = list({s for syms in SECTOR_STOCKS.values() for s in syms})
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

    # 6. Aggregate per sector, rank by inflow score
    sector_results = []
    top5_map       = {}
    for sector, symbols in SECTOR_STOCKS.items():
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
