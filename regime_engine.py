"""
Regime engine — the 3-state market classifier that drives the All-Weather book.

    BULL      → run OFFENSE (strongest momentum)          full deployment
    SIDEWAYS  → run DEFENSE (Defensive Leaders, low DD)   full deployment
    BEAR      → PRESERVE CAPITAL (raise cash)             gross → 0

Why a dedicated 3-state engine (not the IBD `regime.py`)
--------------------------------------------------------
`regime.py` is the canonical IBD distribution-day / follow-through-day label used
by the Breadth and Edge tabs. It needs a *volume* series (basket volume) that the
walk-forward backtest does not have — the backtester only carries the NIFTYBEES
close array. To keep the LIVE badge and the BACKTEST switch driven by the EXACT
same rule (the codebase has been bitten before by two tabs computing regime two
different ways), this engine is **price-only**: every signal is computable from a
single close series, so `regime_series()` runs identically inside the backtest and
on today's live data. Zero look-ahead — every bar's state uses only bars ≤ itself.

The signals (all institutional, all price-computable)
-----------------------------------------------------
  • Trend         — price vs its 200-DMA, and 50-DMA above/below 200-DMA
                    (the Pacer Trendpilot / 200-DMA regime filter workhorse).
  • Abs. momentum — sign of the 6-month (126-bar) return (Antonacci dual momentum
                    absolute-momentum gate: only be "risk-on" in a live uptrend).
  • Drawdown      — decline from the trailing 6-month peak. Self-calibrating risk
                    gauge: it spikes precisely in the vol-driven crashes a fixed
                    volatility threshold would have to be hand-tuned to catch.
  • Realized vol  — 90-day annualised vol, carried for CONTEXT/UI only (no hard
                    threshold — we refuse to curve-fit a vol cutoff to history).

Whipsaw control (the #1 failure mode of regime switching)
---------------------------------------------------------
A raw state is computed each bar, but the EMITTED state only changes after the raw
state has disagreed for `CONFIRM` consecutive bars (the 5-day confirmation Pacer
uses). A transient dip below the 200-DMA that reverts inside the window does not
flip the book. Because it only looks backward, confirmation adds no look-ahead.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# ── Tunables — deliberately few, each grounded in the cited method ────────────
MA_FAST        = 50
MA_SLOW        = 200
MOM_WINDOW     = 126     # 6-month absolute-momentum window
DD_WINDOW      = 126     # trailing peak for the drawdown gauge (~6 months)
VOL_WINDOW     = 90      # realized-vol window (context only)
CONFIRM        = 5       # consecutive bars a new raw state must hold to be adopted

# Drawdown thresholds (fractions, negative). A live uptrend tolerates a shallow
# pullback; a deep decline forces BEAR regardless of the moving-average picture.
BULL_MAX_DD    = -0.10   # BULL may not run once >10% off the 6-month peak
BEAR_DD        = -0.12   # sub-200-DMA AND weak momentum with >12% DD → BEAR
BEAR_HARD_DD   = -0.16   # >16% off the peak → BEAR no matter what

# Pacer Trendpilot "extreme-valuation" guard: when the index is stretched this far
# ABOVE its 200-DMA it is euphoric/mean-reversion-prone — cap the regime at
# SIDEWAYS (lighter gross) rather than full-risk BULL. (Pacer de-risks to 50/50 at
# ±20%; >20% below is already a deep decline our BEAR logic catches.)
EXT_ABOVE_MA200 = 0.20

BULL, SIDEWAYS, BEAR, UNKNOWN = "BULL", "SIDEWAYS", "BEAR", "UNKNOWN"
_ORDER = {BEAR: 0, SIDEWAYS: 1, BULL: 2, UNKNOWN: -1}

# Which engine each state runs, and how it presents in the UI.
REGIME_PLAYBOOK = {
    BULL: {
        "engine": "defensive", "gross": "full",
        "label": "Bull — Risk-On", "color": "#22c55e",
        "action": "Full deployment in the Defensive-Momentum Leaders.",
        "why": "NIFTY above its 200-DMA with positive 6-month momentum. Be fully "
               "invested in the momentum-return engine (defensively filtered — raw "
               "breakout-momentum was tested and lost badly on NSE).",
    },
    SIDEWAYS: {
        "engine": "defensive", "gross": "full",
        "label": "Sideways / Choppy — Defense", "color": "#eab308",
        "action": "Rotate to Defensive Leaders — low-vol, smooth, high-delivery.",
        "why": "Trend is unresolved. Momentum whipsaws here; defensive leaders "
               "fall less while staying invested.",
    },
    BEAR: {
        "engine": "cash", "gross": "zero",
        "label": "Bear — Capital Preservation", "color": "#ef4444",
        "action": "Raise cash. Do not initiate new longs.",
        "why": "NIFTY below its 200-DMA in a real decline. The biggest source of "
               "long-run edge here is the drawdown you AVOID.",
    },
    UNKNOWN: {
        "engine": "cash", "gross": "zero",
        "label": "Unknown", "color": "#94a3b8",
        "action": "Insufficient history to classify.",
        "why": "Need at least 200 sessions of index history.",
    },
}


def _raw_state(price: float, ma50: float, ma200: float,
               r6m: float, dd: float) -> str:
    """Single-bar classification from already-computed point-in-time inputs."""
    if not all(np.isfinite(x) for x in (price, ma50, ma200, r6m, dd)):
        return UNKNOWN
    # Hard risk-off: a deep decline overrides everything.
    if dd <= BEAR_HARD_DD:
        return BEAR
    # BEAR: below the long trend AND (weak momentum OR a meaningful drawdown).
    if price < ma200 and (r6m <= 0.0 or dd <= BEAR_DD):
        return BEAR
    # BULL: full uptrend structure, positive momentum, only a shallow pullback.
    if price > ma200 and ma50 >= ma200 and r6m > 0.0 and dd > BULL_MAX_DD:
        # Pacer extreme-valuation guard: >20% above the 200-DMA is stretched → hold
        # the lighter SIDEWAYS gross instead of full-risk BULL.
        if ma200 > 0 and (price / ma200 - 1.0) >= EXT_ABOVE_MA200:
            return SIDEWAYS
        return BULL
    # Everything in between is choppy/unresolved.
    return SIDEWAYS


def compute_signals(close: np.ndarray) -> dict:
    """Vectorised point-in-time signal arrays over a close series (index = bars).
    Every array at bar i uses only bars ≤ i (rolling / trailing)."""
    c = pd.Series(np.asarray(close, dtype=float))
    n = len(c)
    ma50  = c.rolling(MA_FAST).mean().to_numpy()
    ma200 = c.rolling(MA_SLOW).mean().to_numpy()
    r6m   = (c / c.shift(MOM_WINDOW) - 1.0).to_numpy()
    peak  = c.rolling(DD_WINDOW, min_periods=1).max().to_numpy()
    dd    = (c.to_numpy() / peak - 1.0)
    logret = np.log(c / c.shift(1))
    vol   = (logret.rolling(VOL_WINDOW).std() * np.sqrt(252)).to_numpy()
    return {"close": c.to_numpy(), "ma50": ma50, "ma200": ma200,
            "r6m": r6m, "dd": dd, "vol": vol, "n": n}


def regime_series(close: np.ndarray, confirm: int = CONFIRM,
                  confirm_up: int | None = None,
                  confirm_down: int | None = None) -> np.ndarray:
    """Array of emitted regime labels aligned to `close`, with confirmation-window
    hysteresis. Point-in-time and look-ahead-free: bar i depends only on bars ≤ i.

    Asymmetric option ("fast out, slow in", the classic trend-following bias):
      confirm_down — bars to confirm a move to a SAFER state (BULL→SIDEWAYS→BEAR).
      confirm_up   — bars to confirm a move to a RISKIER state (BEAR→SIDEWAYS→BULL).
    Both default to `confirm` (symmetric). A small confirm_down reacts to
    deterioration quickly (cuts drawdown); a larger confirm_up avoids whipsawing
    back into risk on a dead-cat bounce."""
    cu = confirm if confirm_up is None else confirm_up
    cd = confirm if confirm_down is None else confirm_down
    sig = compute_signals(close)
    n = sig["n"]
    out = np.array([UNKNOWN] * n, dtype=object)
    emitted = UNKNOWN
    pending = None
    cnt = 0
    for i in range(n):
        raw = _raw_state(sig["close"][i], sig["ma50"][i], sig["ma200"][i],
                         sig["r6m"][i], sig["dd"][i])
        if raw == UNKNOWN:
            out[i] = emitted           # carry last known until we can classify
            continue
        if emitted == UNKNOWN:
            emitted = raw              # first real classification adopts immediately
            pending, cnt = None, 0
        elif raw == emitted:
            pending, cnt = None, 0     # still in the same state — reset the counter
        else:
            if raw == pending:
                cnt += 1
            else:
                pending, cnt = raw, 1
            # de-risking (to lower _ORDER) confirms in cd bars, re-risking in cu.
            need = cd if _ORDER.get(pending, 1) < _ORDER.get(emitted, 1) else cu
            if cnt >= need:            # new state held long enough — adopt it
                emitted, pending, cnt = pending, None, 0
        out[i] = emitted
    return out


def classify_close(close: np.ndarray, confirm: int = CONFIRM) -> dict:
    """Full regime dict for the LATEST bar of a close series (live use)."""
    close = np.asarray(close, dtype=float)
    if len(close) < MA_SLOW:
        d = dict(REGIME_PLAYBOOK[UNKNOWN]); d.update(
            {"state": UNKNOWN, "as_of": None, "bars": len(close)})
        return d
    series = regime_series(close, confirm=confirm)
    sig = compute_signals(close)
    i = len(close) - 1
    state = series[i]
    # How long the current emitted state has run (bars back to the last change).
    dur = 1
    while i - dur >= 0 and series[i - dur] == state:
        dur += 1
    d = dict(REGIME_PLAYBOOK.get(state, REGIME_PLAYBOOK[UNKNOWN]))
    d.update({
        "state": state,
        "days_in_regime": int(dur),
        "price": round(float(sig["close"][i]), 2),
        "ma200": round(float(sig["ma200"][i]), 2) if np.isfinite(sig["ma200"][i]) else None,
        "pct_vs_ma200": round(float(sig["close"][i] / sig["ma200"][i] - 1.0) * 100, 2)
                        if np.isfinite(sig["ma200"][i]) and sig["ma200"][i] > 0 else None,
        "r6m_pct": round(float(sig["r6m"][i]) * 100, 2) if np.isfinite(sig["r6m"][i]) else None,
        "drawdown_pct": round(float(sig["dd"][i]) * 100, 2) if np.isfinite(sig["dd"][i]) else None,
        "vol_pct": round(float(sig["vol"][i]) * 100, 1) if np.isfinite(sig["vol"][i]) else None,
    })
    return d


def live_regime(days: int = 900) -> dict:
    """Today's regime from the real NIFTYBEES series (single source of truth for
    the live badge). `days` calendar lookback — kept inside the on-disk cache so
    this never triggers slow network back-fills."""
    import benchmark as bm
    bench = bm.get_benchmark(days=days)
    if bench is None or len(bench) < MA_SLOW:
        d = dict(REGIME_PLAYBOOK[UNKNOWN]); d.update({"state": UNKNOWN, "as_of": None})
        return d
    out = classify_close(bench.to_numpy(dtype=float))
    out["as_of"] = str(bench.index[-1].date())
    return out
