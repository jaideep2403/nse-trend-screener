"""Live-vs-backtest drift monitor — the earliest warning an edge has decayed.

Most systems discover a dead edge via the equity curve — months too late. This
compares the REALIZED forward return of the picks the app actually journaled (the
Strategy tab logs every BUY to .strategy_journal.db) against the BACKTEST
expectation for that horizon. If live is persistently far below backtest, the edge
is decaying and it says so before the drawdown does.

Realized alpha = stock forward return − NIFTYBEES forward return over the same
window (matched to the backtest's dividend-adjusted, price-comparable convention).
"""

from __future__ import annotations
import os
import sqlite3

import pandas as pd

_DIR = os.path.dirname(os.path.abspath(__file__))
JOURNAL_PATH = os.path.join(_DIR, ".strategy_journal.db")
FWD_BARS = 21            # the horizon the backtest expectation is measured over


def _forward_return(series: pd.Series, scan_date: str, bars: int):
    """Return (fwd_frac, ready) for `bars` sessions after scan_date, or (None, False)
    if the position isn't old enough yet to have a full forward window."""
    try:
        t = pd.Timestamp(scan_date)
    except Exception:
        return None, False
    idx = series.index
    # first bar strictly after the scan date = the tradeable next-open equivalent
    after = idx[idx >= t]
    if len(after) < 1:
        return None, False
    p0 = int(idx.get_loc(after[0]))
    if p0 + bars >= len(series):
        return None, False          # not enough forward bars yet — still maturing
    entry = float(series.iloc[p0])
    exitp = float(series.iloc[p0 + bars])
    if entry <= 0:
        return None, False
    return (exitp / entry - 1.0), True


def compute_drift() -> dict:
    import industry_groups as ig
    import benchmark as bm
    import validation_registry as vr

    if not os.path.exists(JOURNAL_PATH):
        return {"ready": False, "note": "No journal yet — the drift monitor starts "
                "warning once the Strategy tab has logged BUYs with a full 21-day "
                "forward window."}
    try:
        with sqlite3.connect(JOURNAL_PATH, timeout=10) as c:
            buys = c.execute(
                "SELECT scan_date, symbol FROM recommendations WHERE action='BUY'"
            ).fetchall()
    except Exception as e:
        return {"ready": False, "note": f"journal read failed: {e}"}

    stocks = ig._get_stocks()
    bench = bm.get_benchmark()
    if bench is None or len(bench) < FWD_BARS + 2:
        return {"ready": False, "note": "benchmark unavailable"}

    realized_alphas = []
    matured = 0
    maturing = 0
    for scan_date, sym in buys:
        df = stocks.get(str(sym).upper())
        if df is None or "Close" not in df:
            continue
        sret, ready = _forward_return(df["Close"], scan_date, FWD_BARS)
        if not ready:
            maturing += 1
            continue
        bret, bready = _forward_return(bench, scan_date, FWD_BARS)
        if not bready:
            maturing += 1
            continue
        # price-comparable: subtract the pro-rated dividend from the total-return bench
        bret_price = bret - bm.dividend_drag(FWD_BARS)
        realized_alphas.append((sret - bret_price) * 100.0)
        matured += 1

    expected = vr.BUCKET_STATS["SUSTAINED_BREAKOUT"]["mean_alpha"]   # % over 21 bars
    if matured < 10:
        return {"ready": False, "matured": matured, "maturing": maturing,
                "expected_alpha": expected,
                "note": f"Collecting — {matured} picks have a full 21-day window so "
                        f"far ({maturing} still maturing). Needs ~10+ for a stable read."}

    import statistics as st
    realized = round(st.mean(realized_alphas), 2)
    drift = round(realized - expected, 2)
    win = round(sum(1 for a in realized_alphas if a > 0) / len(realized_alphas) * 100, 1)
    # Verdict: is live tracking backtest, or decaying?
    if realized >= expected * 0.6:
        verdict, color = "ON TRACK", "green"
    elif realized >= 0:
        verdict, color = "SOFT — watch", "amber"
    else:
        verdict, color = "DECAYING — live alpha negative", "red"
    return {
        "ready": True, "matured": matured, "maturing": maturing,
        "realized_alpha": realized, "expected_alpha": expected, "drift": drift,
        "win_pct": win, "verdict": verdict, "color": color, "horizon_bars": FWD_BARS,
        "note": ("Realized forward alpha of journaled BUYs vs the backtest "
                 "expectation. Persistent negative drift = the edge is decaying."),
    }
