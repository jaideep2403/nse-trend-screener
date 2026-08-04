"""Relative Rotation Graph (RRG) — turns the sector scoreboard into entry/exit events.

The sector tab could say WHICH sectors were strong; it could not say WHEN to act.
RRG is the institutional standard for exactly that. Two axes, computed against the
benchmark:

  RS-Ratio     (x) — is the sector's relative strength in an up- or downtrend?
  RS-Momentum  (y) — is that relative trend accelerating or decaying?

Crossing at 100 gives four quadrants, and sectors rotate through them CLOCKWISE:

      Improving (−RS, +Mom)  →  Leading (+RS, +Mom)
            ↑                        ↓
      Lagging  (−RS, −Mom)  ←  Weakening (+RS, −Mom)

The two transitions that matter:
  • Improving → Leading    = strength confirmed while still early   ⇒ BUY signal
  • Leading   → Weakening  = momentum rolls over while RS is STILL positive
                             ⇒ REDUCE signal, and it fires BEFORE the sector shows
                               up as weak on any returns table. That earliness is
                               the entire reason to use RRG rather than a ranking.

FORMULA HONESTY: Julius de Kempenaer's JdK RS-Ratio / RS-Momentum are proprietary.
This is the standard open replication — normalise relative strength against its own
recent mean and dispersion, then do the same to its rate of change:

    rs        = sector / benchmark
    rs_ratio  = 100 + zscore(rs,        RS_WINDOW)
    rs_mom    = 100 + zscore(rs_ratio,  MOM_WINDOW)

It reproduces the quadrant behaviour and rotation, but values will not tie out
digit-for-digit against a commercial RRG.

POINT-IN-TIME: every function returns full history arrays computed only from bars up
to each index, so a backtest can read position `i` without look-ahead. `zscore` uses
trailing windows only — never a full-sample mean.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# 63 bars ≈ one quarter of relative trend; 21 ≈ one month of momentum. These match
# the monthly rebalance cadence the engine was validated on — a shorter RS window
# makes the quadrants flicker, which is how you end up churning weekly (measured:
# weekly rotation cut CAGR 25.4% → 7.3%).
RS_WINDOW = 63
MOM_WINDOW = 21
MIN_BARS = RS_WINDOW + MOM_WINDOW + 5

LEADING, WEAKENING, LAGGING, IMPROVING = "Leading", "Weakening", "Lagging", "Improving"

QUADRANT_META = {
    LEADING:   {"colour": "green",  "stance": "Hold / add",
                "meaning": "Relative strength is rising AND still accelerating."},
    WEAKENING: {"colour": "yellow", "stance": "Reduce / stop adding",
                "meaning": "Still stronger than the index, but momentum has rolled over."},
    LAGGING:   {"colour": "red",    "stance": "Avoid",
                "meaning": "Underperforming and still getting worse."},
    IMPROVING: {"colour": "blue",   "stance": "Watchlist",
                "meaning": "Still behind the index, but momentum has turned up."},
}

# Only these two transitions are treated as actionable. The other rotations are
# information, not instructions.
BUY_TRANSITION = (IMPROVING, LEADING)
REDUCE_TRANSITION = (LEADING, WEAKENING)


def _zscore(s: pd.Series, window: int) -> pd.Series:
    """Trailing z-score. Uses only bars ≤ each index — no look-ahead."""
    mean = s.rolling(window, min_periods=window).mean()
    std = s.rolling(window, min_periods=window).std(ddof=0)
    return (s - mean) / std.replace(0.0, np.nan)


# A sector sitting exactly on an axis flips quadrant on noise. Measured on live
# data: Nifty Pharma crossed Leading→Weakening→Leading→Weakening inside six
# sessions, which would emit REDUCE, ignore, REDUCE. CONFIRM_BARS makes a new
# quadrant stick before it counts, for the same reason the engine rebalances
# monthly rather than weekly — unconfirmed flips are churn, and churn is what took
# CAGR from 25.4% to 7.3% in the rebalance sweep.
CONFIRM_BARS = 3


def quadrant_of(rs_ratio: float, rs_mom: float) -> str | None:
    if not (np.isfinite(rs_ratio) and np.isfinite(rs_mom)):
        return None
    if rs_ratio >= 100:
        return LEADING if rs_mom >= 100 else WEAKENING
    return IMPROVING if rs_mom >= 100 else LAGGING


def _confirmed(raw: pd.Series, bars: int = CONFIRM_BARS) -> pd.Series:
    """Adopt a new quadrant only after it has held for `bars` consecutive sessions.

    Look-ahead free: bar i is decided using bars ≤ i only.
    """
    out, cur, run_val, run_n = [], None, None, 0
    for v in raw:
        if v is None or (isinstance(v, float) and not np.isfinite(v)):
            out.append(cur)
            continue
        if v == run_val:
            run_n += 1
        else:
            run_val, run_n = v, 1
        if cur is None or (run_val != cur and run_n >= bars):
            cur = run_val
        out.append(cur)
    return pd.Series(out, index=raw.index)


def sector_composite(stocks: dict, members: list[str]) -> pd.Series | None:
    """Equal-weighted cumulative-return index for a sector.

    Equal weight rather than cap weight on purpose: cap weights would let two or
    three heavyweights speak for the whole sector, which is precisely how a sector
    looks healthy while most of its members are already rolling over.
    """
    cols = []
    for sym in members:
        df = stocks.get(sym)
        if df is None or len(df) < 30:
            continue
        c = df["Close"].astype(float).dropna()
        if c.empty or c.iloc[0] <= 0:
            continue
        cols.append(c.pct_change())
    if len(cols) < 3:                     # too thin to call a "sector"
        return None
    rets = pd.concat(cols, axis=1).mean(axis=1, skipna=True).fillna(0.0)
    return (1.0 + rets).cumprod() * 100.0


def compute(sector_px: pd.Series, bench_px: pd.Series) -> pd.DataFrame:
    """Full-history RS-Ratio / RS-Momentum / quadrant for one sector."""
    df = pd.concat([sector_px.rename("sec"), bench_px.rename("bench")],
                   axis=1, join="inner").dropna()
    if len(df) < MIN_BARS:
        return pd.DataFrame(columns=["rs_ratio", "rs_mom", "quadrant"])
    rs = df["sec"] / df["bench"]
    rs_ratio = 100.0 + _zscore(rs, RS_WINDOW)
    rs_mom = 100.0 + _zscore(rs_ratio, MOM_WINDOW)
    out = pd.DataFrame({"rs_ratio": rs_ratio, "rs_mom": rs_mom})
    raw_q = pd.Series([quadrant_of(a, b) for a, b in zip(out["rs_ratio"], out["rs_mom"])],
                      index=out.index)
    out["quadrant_raw"] = raw_q
    out["quadrant"] = _confirmed(raw_q)
    return out


def transitions(rrg_df: pd.DataFrame, lookback: int = 10) -> list[dict]:
    """Quadrant CHANGES in the last `lookback` bars — the actionable events.

    A transition is only emitted when the quadrant genuinely changes between two
    consecutive valid bars, so a sector sitting still produces no events.
    """
    if rrg_df.empty or "quadrant" not in rrg_df:
        return []
    q = rrg_df["quadrant"].dropna()
    if len(q) < 2:
        return []
    out = []
    tail = q.iloc[-(lookback + 1):]
    prev_val, prev_idx = tail.iloc[0], tail.index[0]
    for idx, val in tail.iloc[1:].items():
        if val != prev_val:
            kind = ("BUY" if (prev_val, val) == BUY_TRANSITION else
                    "REDUCE" if (prev_val, val) == REDUCE_TRANSITION else "INFO")
            out.append({"date": str(pd.Timestamp(idx).date()),
                        "from": prev_val, "to": val, "kind": kind,
                        "stance": QUADRANT_META.get(val, {}).get("stance")})
            prev_val = val
    return out


def run(stocks: dict, bench: pd.Series, sectors: dict[str, list[str]] | None = None,
        tail_bars: int = 8) -> dict:
    """Live RRG for every sector: position, quadrant, tail, and recent transitions."""
    if sectors is None:
        try:
            import sector_indices as _si
            sectors = _si.get_sector_constituents()
        except Exception:
            import sector_analysis as _sa
            sectors = _sa.SECTOR_STOCKS
    rows = []
    for name, members in (sectors or {}).items():
        px = sector_composite(stocks, list(members))
        if px is None:
            continue
        r = compute(px, bench)
        if r.empty or r["quadrant"].dropna().empty:
            continue
        valid = r.dropna(subset=["rs_ratio", "rs_mom"])
        if valid.empty:
            continue
        last = valid.iloc[-1]
        tail = valid.iloc[-tail_bars:]
        evs = transitions(r)
        rows.append({
            "sector": name,
            "n_members": len(members),
            "rs_ratio": round(float(last["rs_ratio"]), 2),
            "rs_mom": round(float(last["rs_mom"]), 2),
            "quadrant": last["quadrant"],
            "stance": QUADRANT_META.get(last["quadrant"], {}).get("stance"),
            "meaning": QUADRANT_META.get(last["quadrant"], {}).get("meaning"),
            "tail": [{"x": round(float(a), 2), "y": round(float(b), 2)}
                     for a, b in zip(tail["rs_ratio"], tail["rs_mom"])],
            "transitions": evs,
            # A signal is only live if its DESTINATION is where the sector still is.
            # Without this a REDUCE fired days ago would keep showing on a sector
            # that has since rotated back to Leading — a verdict contradicting the
            # state shown next to it.
            "signal": next((e["kind"] for e in reversed(evs)
                            if e["kind"] != "INFO" and e["to"] == last["quadrant"]), None),
            "as_of": str(pd.Timestamp(valid.index[-1]).date()),
        })
    # Leading first, then Improving — the two quadrants worth acting on.
    order = {LEADING: 0, IMPROVING: 1, WEAKENING: 2, LAGGING: 3}
    rows.sort(key=lambda r: (order.get(r["quadrant"], 9), -r["rs_ratio"]))
    return {
        "as_of": rows[0]["as_of"] if rows else None,
        "sectors": rows,
        "params": {"rs_window": RS_WINDOW, "mom_window": MOM_WINDOW},
        "buy_signals": [r["sector"] for r in rows if r["signal"] == "BUY"],
        "reduce_signals": [r["sector"] for r in rows if r["signal"] == "REDUCE"],
        "note": ("RS-Ratio/RS-Momentum are an open replication of the JdK method "
                 "(the originals are proprietary); quadrants and rotation behave the "
                 "same, absolute values will not match a commercial RRG."),
    }
