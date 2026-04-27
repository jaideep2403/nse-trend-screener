"""
Advanced Setups Scanner
Detects: High Tight Flag · NR7 · Inside Bar · 3-Weeks-Tight · RS Line New High
Plus: Trend Template Score · Stage Analysis · Candlestick signals on every stock.

Data: NSE bhavcopy (zero Yahoo calls)
"""
import time
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
from data_fetcher import _weekdays_back, _download_one_day
from analysis_utils import (
    trend_template_score, stage_analysis, stage_label,
    is_nr7, is_inside_bar, is_3wt,
    is_high_tight_flag, rs_line_new_high,
    detect_candle_signals, atr,
    power_trend, base_count, price_vol_character,
    classify_gap, delivery_trend, composite_rank,
)

MIN_BARS     = 60
MIN_ADTV_CR  = 0.5    # Minimal liquidity guard only — universe filtered by Nifty500 membership
SCAN_WORKERS = 8
_cache = {"data": None, "ts": 0}
CACHE_TTL = 3600

_NIFTY50_SYMS = [
    "RELIANCE", "TCS", "HDFCBANK", "ICICIBANK", "INFOSYS",
    "HINDUNILVR", "SBIN", "BHARTIARTL", "ITC", "KOTAKBANK",
    "AXISBANK", "WIPRO", "HCLTECH", "MARUTI", "BAJFINANCE",
    "TITAN", "NTPC", "POWERGRID", "NESTLEIND", "SUNPHARMA",
]


# ── Data loader ───────────────────────────────────────────────────────────────

def _load_all_stocks(progress_callback=None) -> dict[str, pd.DataFrame]:
    dates  = _weekdays_back(400)
    total  = len(dates)
    frames = []
    for i, dt in enumerate(dates):
        df = _download_one_day(dt)
        if df is not None:
            frames.append(df)
        if progress_callback and i % 40 == 0:
            progress_callback(i, total, f"Loading historical data… {i}/{total} days")
    if not frames:
        return {}
    # Filter to Nifty50 ∪ NiftyNext50 ∪ Nifty500 ∪ NiftySmallcap250
    try:
        from nse_stocks import get_nifty500_symbols
        _universe = set(get_nifty500_symbols())
    except Exception:
        _universe = set()
    combined = pd.concat(frames, ignore_index=True).sort_values("Date")
    stocks = {}
    for sym, grp in combined.groupby("Symbol"):
        if _universe and sym not in _universe:
            continue
        cols = ["Open", "High", "Low", "Close", "Volume"]
        if "DelivPer" in grp.columns:
            cols.append("DelivPer")
        g = grp.set_index("Date")[cols]
        g = g[~g.index.duplicated(keep="last")].sort_index()
        if len(g) >= MIN_BARS:
            stocks[sym] = g
    return stocks


def _build_nifty(stocks: dict) -> pd.Series | None:
    closes = []
    for sym in _NIFTY50_SYMS:
        df = stocks.get(sym)
        if df is not None and len(df) >= 63:
            closes.append(df["Close"].dropna())
    if not closes:
        return None
    combined = pd.concat(closes, axis=1).dropna(how="all")
    bench = combined.mean(axis=1)
    return bench if len(bench) >= 20 else None


# ── Level computation ─────────────────────────────────────────────────────────

def _levels(df: pd.DataFrame, entry: float, base_high: float, base_low: float,
            ath: float) -> dict:
    atr14 = atr(df)
    sl = round(max(
        base_low * 0.985,
        entry - 2.5 * atr14,
        entry * 0.92,
    ), 2)
    risk = max(entry - sl, entry * 0.03)
    base_ht = max(base_high - base_low, risk)
    t1 = round(entry + base_ht, 2)
    t2 = round(entry + 2.0 * risk, 2)
    t3 = round(max(ath * 1.005, entry + 3.0 * risk), 2)
    return {
        "entry":    entry,
        "sl":       sl,
        "t1":       t1,
        "t2":       t2,
        "t3":       t3,
        "risk_pct": round((entry - sl) / entry * 100, 2),
        "rr":       round((t2 - entry) / (entry - sl), 2) if entry > sl else 0.0,
    }


# ── Per-stock analysis ────────────────────────────────────────────────────────

def _analyze(symbol: str, df: pd.DataFrame, nifty: pd.Series | None) -> dict | None:
    try:
        close = df["Close"].dropna()
        high  = df["High"].dropna()
        low   = df["Low"].dropna()
        vol   = df["Volume"].dropna()

        if len(close) < MIN_BARS:
            return None

        cur = float(close.iloc[-1])
        if cur <= 0:
            return None

        # Liquidity
        avg_vol = float(vol.rolling(20).mean().iloc[-1]) if len(vol) >= 20 else 0
        adtv_cr = round(avg_vol * cur / 1e7, 2)
        if adtv_cr < MIN_ADTV_CR:
            return None

        # ── Setup detection ───────────────────────────────────────
        setups = []
        htf_meta = {}

        # High Tight Flag
        htf_ok, htf_lvl = is_high_tight_flag(close, high)
        if htf_ok:
            setups.append("HTF")
            htf_meta = htf_lvl

        # Compression patterns
        if is_nr7(df):
            setups.append("NR7")
        if is_inside_bar(df):
            setups.append("InsideBar")
        if is_3wt(close):
            setups.append("3WT")

        # RS Line New High
        rs_nh = rs_line_new_high(close, nifty) if nifty is not None else False
        if rs_nh:
            setups.append("RSLineHigh")

        # Candlestick signals
        candles = detect_candle_signals(df)

        # Must have at least one setup OR candle signal
        if not setups and not candles:
            return None

        # ── Trend Template & Stage ────────────────────────────────
        tt_score, tt_met = trend_template_score(close, rs_rating=0)
        stage = stage_analysis(close)

        # ── New enrichment signals ────────────────────────────────
        pt_ok    = power_trend(close)
        b_cnt    = base_count(close)
        pv_char  = price_vol_character(df)
        gap_cl   = classify_gap(df)
        deliv_s  = df["DelivPer"].dropna() if "DelivPer" in df.columns else pd.Series(dtype=float)
        deliv_tr = delivery_trend(deliv_s)
        deliv_avg = round(float(deliv_s.iloc[-20:].mean()), 1) if len(deliv_s) >= 5 else None
        comp     = composite_rank(tt_score, 50, stage, deliv_tr, b_cnt, pt_ok)

        # ── Returns ───────────────────────────────────────────────
        r1m  = round((cur / float(close.iloc[-22])  - 1) * 100, 2) if len(close) > 22  else 0.0
        r3m  = round((cur / float(close.iloc[-63])  - 1) * 100, 2) if len(close) > 63  else 0.0
        r6m  = round((cur / float(close.iloc[-126]) - 1) * 100, 2) if len(close) > 126 else 0.0

        ma50  = round(float(close.rolling(50).mean().iloc[-1]),  2) if len(close) >= 50  else None
        ma200 = round(float(close.rolling(200).mean().iloc[-1]), 2) if len(close) >= 200 else None
        ath   = float(close.max())

        # ── Levels (use HTF if available, else recent pivot) ─────
        if htf_ok:
            entry    = round(htf_meta["base_high"] * 1.003, 2)
            base_hi  = htf_meta["base_high"]
            base_lo  = htf_meta["flag_lo"]
        else:
            win     = min(20, len(close) - 1)
            base_hi = float(high.iloc[-win:].max())
            base_lo = float(low.iloc[-win:].min())
            entry   = round(base_hi * 1.003, 2)

        lvl = _levels(df, entry, base_hi, base_lo, ath)

        return {
            "symbol":      symbol,
            "price":       round(cur, 2),
            "adtv_cr":     adtv_cr,
            "setups":      setups,
            "candles":     candles,
            "tt_score":    tt_score,
            "tt_met":      tt_met,
            "stage":       stage,
            "stage_label": stage_label(stage),
            "rs_rating":   50,       # updated after scan
            "r1m":         r1m,
            "r3m":         r3m,
            "r6m":         r6m,
            "above_ma50":  (cur > ma50)  if ma50  else False,
            "above_ma200": (cur > ma200) if ma200 else False,
            "pct_ath":     round((cur - ath) / ath * 100, 2),
            # HTF extras
            "htf_run_pct":  htf_meta.get("run_up_pct"),
            "htf_flag_pct": htf_meta.get("flag_pct"),
            # Levels
            **lvl,
            # Enrichment
            "power_trend":  pt_ok,
            "base_count":   b_cnt,
            "pv_char":      pv_char,
            "gap":          gap_cl,
            "deliv_trend":  deliv_tr,
            "deliv_avg":    deliv_avg,
            "comp_rank":    comp,
        }
    except Exception:
        return None


# ── Main entry ────────────────────────────────────────────────────────────────

def run_advanced_scan(progress_callback=None) -> dict:
    if (_cache["data"]
            and time.time() - _cache["ts"] < CACHE_TTL
            and _cache["data"].get("results")):
        return _cache["data"]

    stocks = _load_all_stocks(progress_callback)
    if not stocks:
        return {"results": [], "computed_at": int(time.time()), "total_scanned": 0}

    nifty = _build_nifty(stocks)

    total  = len(stocks)
    done   = [0]
    results = []

    if progress_callback:
        progress_callback(0, total, f"Scanning {total} stocks for advanced setups…")

    def _job(item):
        sym, df = item
        r = _analyze(sym, df, nifty)
        done[0] += 1
        if progress_callback and done[0] % 200 == 0:
            progress_callback(done[0], total,
                              f"Analysing {done[0]}/{total} stocks…")
        return r

    with ThreadPoolExecutor(max_workers=SCAN_WORKERS) as ex:
        for r in ex.map(_job, stocks.items()):
            if r:
                results.append(r)

    # RS Rating from r3m rank
    if results:
        r3m_s  = pd.Series([r["r3m"] for r in results])
        rs_arr = (r3m_s.rank(pct=True) * 99).round(0).astype(int).tolist()
        for r, rs in zip(results, rs_arr):
            r["rs_rating"] = int(rs)
            r["tt_score"], r["tt_met"] = trend_template_score(
                stocks[r["symbol"]]["Close"].dropna(), rs_rating=rs
            )
            # Recompute composite rank with real RS
            r["comp_rank"] = composite_rank(
                r["tt_score"], rs, r["stage"],
                r.get("deliv_trend", "Unknown"),
                r.get("base_count", 0),
                r.get("power_trend", False),
            )

    # Sort: HTF first, then stage=2, then TT score
    def _sort_key(r):
        htf  = 3 if "HTF"         in r["setups"] else 0
        rsl  = 2 if "RSLineHigh"  in r["setups"] else 0
        comp = 1 if len(r["setups"]) >= 2 else 0
        return (htf + rsl + comp, r["tt_score"], r["rs_rating"])

    results.sort(key=_sort_key, reverse=True)

    # Summary counts
    counts = {
        "htf":         sum(1 for r in results if "HTF"         in r["setups"]),
        "nr7":         sum(1 for r in results if "NR7"         in r["setups"]),
        "inside_bar":  sum(1 for r in results if "InsideBar"   in r["setups"]),
        "three_wt":    sum(1 for r in results if "3WT"         in r["setups"]),
        "rs_line":     sum(1 for r in results if "RSLineHigh"  in r["setups"]),
        "tt_8":        sum(1 for r in results if r["tt_score"] == 8),
        "tt_7plus":    sum(1 for r in results if r["tt_score"] >= 7),
        "stage2":      sum(1 for r in results if r["stage"] == 2),
        "with_candle": sum(1 for r in results if r["candles"]),
    }

    out = {
        "results":       results,
        "computed_at":   int(time.time()),
        "total_scanned": total,
        "counts":        counts,
    }
    _cache["data"] = out
    _cache["ts"]   = time.time()

    if progress_callback:
        progress_callback(total, total,
                          f"Done — {len(results)} setups · {counts['htf']} HTF · "
                          f"{counts['rs_line']} RS New High · {counts['tt_7plus']} TT≥7")
    return out
