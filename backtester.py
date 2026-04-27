"""
Historical Pattern Backtester
Scans bhavcopy history to find past instances of each pattern,
then measures forward returns at 5/10/20 trading days.
Zero extra NSE calls — purely uses existing bhavcopy cache.
"""
import time
import numpy as np
import pandas as pd
from data_fetcher import _weekdays_back, _download_one_day
from analysis_utils import (
    is_nr7, is_inside_bar, is_3wt, is_high_tight_flag,
    trend_template_score, power_trend, rs_line_new_high,
)

_cache    = {"data": None, "ts": 0}
CACHE_TTL = 7_200   # 2 hours (heavy computation)

MIN_BARS     = 80
HOLD_PERIODS = [5, 10, 20]

_NIFTY_SYMS = [
    "RELIANCE", "TCS", "HDFCBANK", "ICICIBANK", "INFOSYS",
    "HINDUNILVR", "SBIN", "BHARTIARTL", "ITC", "KOTAKBANK",
]


# ── Data loader ───────────────────────────────────────────────────────────────

def _load_stocks(progress_callback=None) -> dict[str, pd.DataFrame]:
    dates  = _weekdays_back(400)
    total  = len(dates)
    frames = []
    for i, dt in enumerate(dates):
        df = _download_one_day(dt)
        if df is not None:
            frames.append(df)
        if progress_callback and i % 60 == 0:
            progress_callback(i, total, f"Loading bhavcopy… {i}/{total} days")
    if not frames:
        return {}
    combined = pd.concat(frames, ignore_index=True).sort_values("Date")
    stocks: dict[str, pd.DataFrame] = {}
    for sym, grp in combined.groupby("Symbol"):
        cols = ["Open", "High", "Low", "Close", "Volume"]
        avail = [c for c in cols if c in grp.columns]
        g = grp.set_index("Date")[avail]
        g = g[~g.index.duplicated(keep="last")].sort_index()
        if len(g) >= MIN_BARS:
            stocks[sym] = g
    return stocks


def _build_nifty(stocks: dict) -> pd.Series | None:
    closes = [stocks[s]["Close"].dropna() for s in _NIFTY_SYMS if s in stocks and len(stocks[s]) >= 63]
    if not closes:
        return None
    combined = pd.concat(closes, axis=1).dropna(how="all")
    bench = combined.mean(axis=1)
    return bench if len(bench) >= 20 else None


# ── Pattern tester ────────────────────────────────────────────────────────────

def _stats(returns: list[float]) -> dict:
    if not returns:
        return {}
    wins   = [r for r in returns if r > 0]
    losses = [r for r in returns if r <= 0]
    return {
        "n":        len(returns),
        "win_rate": round(len(wins) / len(returns) * 100, 1),
        "avg_ret":  round(float(np.mean(returns)), 2),
        "med_ret":  round(float(np.median(returns)), 2),
        "avg_win":  round(float(np.mean(wins)),   2) if wins   else 0,
        "avg_loss": round(float(np.mean(losses)), 2) if losses else 0,
        "best":     round(float(max(returns)), 2),
        "worst":    round(float(min(returns)), 2),
    }


def _backtest_pattern(stocks: dict, nifty: pd.Series | None, pattern: str) -> dict:
    all_rets   = {h: [] for h in HOLD_PERIODS}
    sig_count  = 0
    # To avoid one stock dominating, cap signals per stock
    MAX_SIG_PER_STOCK = 20

    for sym, df in stocks.items():
        n        = len(df)
        close    = df["Close"].dropna()
        sym_sigs = 0
        # Scan each historical bar as if it were "today"
        for i in range(MIN_BARS, n - max(HOLD_PERIODS) - 1):
            if sym_sigs >= MAX_SIG_PER_STOCK:
                break
            window = df.iloc[:i + 1]
            wclose = window["Close"].dropna()
            signal = False
            try:
                if pattern == "NR7":
                    signal = is_nr7(window)
                elif pattern == "InsideBar":
                    signal = is_inside_bar(window)
                elif pattern == "3WT":
                    signal = is_3wt(wclose)
                elif pattern == "HTF":
                    ok, _ = is_high_tight_flag(wclose, window["High"].dropna())
                    signal = ok
                elif pattern == "PowerTrend":
                    signal = power_trend(wclose)
                elif pattern == "TT7Plus":
                    score, _ = trend_template_score(wclose, rs_rating=70)
                    signal = (score >= 7)
                elif pattern == "RSLineHigh" and nifty is not None:
                    signal = rs_line_new_high(wclose, nifty)
            except Exception:
                continue

            if not signal:
                continue

            entry = float(df["Close"].iloc[i])
            if entry <= 0:
                continue
            sig_count += 1
            sym_sigs  += 1
            for hold in HOLD_PERIODS:
                fut = float(df["Close"].iloc[i + hold])
                all_rets[hold].append((fut - entry) / entry * 100)

    return {
        "pattern":      pattern,
        "signal_count": sig_count,
        "stats":        {f"h{h}d": _stats(all_rets[h]) for h in HOLD_PERIODS},
    }


# ── Main entry ────────────────────────────────────────────────────────────────

def run_backtest(patterns: list[str] | None = None,
                 progress_callback=None) -> dict:
    if _cache["data"] and time.time() - _cache["ts"] < CACHE_TTL:
        return _cache["data"]

    if patterns is None:
        patterns = ["NR7", "InsideBar", "3WT", "HTF", "PowerTrend", "TT7Plus", "RSLineHigh"]

    stocks = _load_stocks(progress_callback)
    if not stocks:
        return {"results": [], "computed_at": int(time.time()), "stocks_used": 0}

    nifty  = _build_nifty(stocks)
    total  = len(patterns)
    results = []

    for i, pat in enumerate(patterns):
        if progress_callback:
            progress_callback(i, total, f"Backtesting {pat}… ({i+1}/{total})")
        r = _backtest_pattern(stocks, nifty, pat)
        results.append(r)

    out = {
        "results":      results,
        "computed_at":  int(time.time()),
        "stocks_used":  len(stocks),
        "hold_periods": HOLD_PERIODS,
    }
    _cache["data"] = out
    _cache["ts"]   = time.time()

    if progress_callback:
        progress_callback(total, total,
                          f"Backtest done — {len(patterns)} patterns tested on {len(stocks)} stocks")
    return out
