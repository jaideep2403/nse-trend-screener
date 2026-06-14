"""
Canonical market regime — the SINGLE source of truth for distribution-day
counting, Follow-Through-Day detection, and the regime label.

Why this module exists
----------------------
Before 2026-06-10 the Edge tab and the Market Breadth tab each had their own
D-day/FTD implementations with different price sources (synthetic proxy vs
NIFTYBEES), different volume handling (breadth's fallback counted ANY -0.5%
day as "distribution" with no volume condition at all), and different FTD
rules. Two tabs could — and did — report different regimes on the same day.
A strategy that gates deployment on the regime cannot tolerate that.

Canonical inputs (both tabs must pass the SAME series):
  price  — real NIFTYBEES close (cap-weighted Nifty 50, div-reinvested),
           from benchmark.get_benchmark() or edge_engine._BENCH.
  volume — summed volume of the canonical 20-stock NIFTY_PROXY_SYMS basket
           (build_market_volume). NIFTYBEES' own ETF volume does NOT reflect
           institutional index activity, and true index volume is not in the
           bhavcopy; the basket volume is the best available proxy and is
           used IDENTICALLY everywhere.

Canonical rules (IBD method):
  Distribution day — index closes ≤ -0.2% AND volume > prior session.
  FTD              — after a ≥5% decline, on day 4-7 of the rally attempt,
                     index closes ≥ +1.7% on volume > prior session.
  Regime           — ≥6 D-days: Correction · 4-5: Under Pressure ·
                     FTD active & ≤3: Confirmed Uptrend · else 3 = Under
                     Pressure, <3 = Confirmed Uptrend.
"""
from __future__ import annotations

import pandas as pd

from analysis_utils import NIFTY_PROXY_SYMS

# ── Canonical thresholds — change here, changes EVERYWHERE ────────────────────
DDAY_PCT_THRESHOLD  = -0.2   # close % change at or below this …
DDAY_SESSIONS       = 25     # … within the last N sessions
FTD_PCT_THRESHOLD   = 1.7    # follow-through day minimum gain %
FTD_WINDOW          = (4, 7) # valid FTD days counted from the rally trough
FTD_MIN_DECLINE_PCT = 5.0    # a rally attempt needs a real prior decline
CORRECTION_DDAYS    = 6
PRESSURE_DDAYS      = 4


def build_market_volume(stocks: dict) -> pd.Series | None:
    """Summed daily volume of the canonical 20-stock basket.
    Pass the same `stocks` dict any scanner already loaded — no extra I/O."""
    vols = [stocks[s]["Volume"].dropna() for s in NIFTY_PROXY_SYMS
            if s in stocks and len(stocks[s]) >= 60]
    if len(vols) < 5:
        return None
    v = pd.concat(vols, axis=1).dropna(how="all").sum(axis=1)
    return v if len(v) >= 30 else None


def _aligned(price: pd.Series, volume: pd.Series | None) -> pd.DataFrame | None:
    if price is None or len(price) < DDAY_SESSIONS + 5:
        return None
    df = pd.DataFrame({"p": price.astype(float)})
    if volume is not None and len(volume):
        df["v"] = volume.reindex(df.index, method="ffill")
    else:
        df["v"] = float("nan")
    return df.dropna(subset=["p"])


def count_distribution_days(price: pd.Series, volume: pd.Series | None,
                            sessions: int = DDAY_SESSIONS) -> tuple[int, list[dict]]:
    """Volume-confirmed D-day count over the last `sessions` sessions.
    Returns (count, detail rows). If volume is unavailable the volume
    condition CANNOT be checked — we return a stricter price-only count
    (≤ -0.5%) and mark detail rows with vol_chg_pct=None, never the loose
    -0.2% price-only count that over-fires."""
    df = _aligned(price, volume)
    if df is None or len(df) < sessions + 1:
        return 0, []
    sub = df.iloc[-(sessions + 1):].copy()
    sub["pct"] = sub["p"].pct_change() * 100
    have_vol = sub["v"].notna().all()
    if have_vol:
        sub["dday"] = (sub["pct"] <= DDAY_PCT_THRESHOLD) & (sub["v"] > sub["v"].shift(1))
    else:
        sub["dday"] = sub["pct"] <= -0.5
    sub = sub.iloc[1:]   # drop the seed row (NaN pct)
    detail = []
    for idx, row in sub[sub["dday"]].iterrows():
        vol_chg = None
        if have_vol:
            try:
                prev_v = float(sub["v"].shift(1).loc[idx])
                vol_chg = round((float(row["v"]) / prev_v - 1) * 100, 1) if prev_v > 0 else None
            except Exception:
                vol_chg = None
        detail.append({"date": idx.strftime("%d-%b"),
                       "pct": round(float(row["pct"]), 2),
                       "vol_chg_pct": vol_chg})
    return int(sub["dday"].sum()), detail


def detect_ftd(price: pd.Series, volume: pd.Series | None,
               lookback: int = 60) -> dict:
    """
    Canonical IBD Follow-Through Day:
      1. A peak, then a decline ≥ FTD_MIN_DECLINE_PCT to a trough (within
         `lookback` bars, trough after the peak).
      2. Rally attempt counted from the trough.
      3. FTD = a day in [trough+4, trough+7] closing ≥ +1.7% on volume >
         prior session. Without volume data, no FTD is signalled (an
         unconfirmed FTD is not an FTD).
    """
    out = {"rally_attempt": False, "days_since_trough": None,
           "trough_to_now_pct": None, "ftd_active": False,
           "ftd_day": None, "ftd_today": False, "today_chg_pct": None}
    df = _aligned(price, volume)
    if df is None or len(df) < 30:
        return out
    win = df.iloc[-lookback:]
    vals = win["p"].values
    peak_loc = int(vals.argmax())
    trough_loc = int(vals.argmin())
    if len(win) >= 2:
        out["today_chg_pct"] = round(float(win["p"].pct_change().iloc[-1]) * 100, 2)
    if trough_loc <= peak_loc:
        return out   # no decline-then-rally structure
    peak_v, trough_v = float(vals[peak_loc]), float(vals[trough_loc])
    if peak_v <= 0 or (peak_v - trough_v) / peak_v * 100 < FTD_MIN_DECLINE_PCT:
        return out
    cur = float(win["p"].iloc[-1])
    if cur <= trough_v:
        return out
    days_since = len(win) - 1 - trough_loc
    out["rally_attempt"] = True
    out["days_since_trough"] = days_since
    out["trough_to_now_pct"] = round((cur / trough_v - 1) * 100, 2)

    if win["v"].isna().all():
        return out   # cannot confirm volume — no FTD call
    lo, hi = FTD_WINDOW
    for d in range(lo, hi + 1):
        pos = trough_loc + d
        if pos >= len(win):
            break
        chg = (float(win["p"].iloc[pos]) / float(win["p"].iloc[pos - 1]) - 1) * 100
        v_now, v_prev = float(win["v"].iloc[pos]), float(win["v"].iloc[pos - 1])
        if chg >= FTD_PCT_THRESHOLD and v_now > v_prev:
            out["ftd_active"] = True
            out["ftd_day"] = win.index[pos].strftime("%d-%b-%Y")
            out["ftd_today"] = (pos == len(win) - 1)
            break
    return out


def classify_regime(dday_count: int, ftd_active: bool) -> str:
    if dday_count >= CORRECTION_DDAYS:
        return "Correction"
    if dday_count >= PRESSURE_DDAYS:
        return "Uptrend Under Pressure"
    if ftd_active and dday_count <= 3:
        return "Confirmed Uptrend"
    return "Uptrend Under Pressure" if dday_count >= 3 else "Confirmed Uptrend"


REGIME_ACTIONS = {
    "Confirmed Uptrend":      ("✅ Buy Mode", "Full deployment — buy strongest setups",        "#22c55e"),
    "Uptrend Under Pressure": ("🟡 Cautious", "Selective — only highest-conviction setups",   "#eab308"),
    "Correction":             ("🔴 Cash",     "Do not initiate new longs — preserve capital", "#ef4444"),
    "Unknown":                ("⚪ Unknown",  "Insufficient data",                              "#94a3b8"),
}


def market_regime(price: pd.Series, volume: pd.Series | None) -> dict:
    """Full canonical regime dict — both Breadth and Edge consume THIS."""
    if price is None or len(price) < DDAY_SESSIONS + 5:
        return {"regime": "Unknown", "dday_count": 0, "ftd_active": False,
                "details": [], "label": REGIME_ACTIONS["Unknown"][0],
                "advice": REGIME_ACTIONS["Unknown"][1],
                "color": REGIME_ACTIONS["Unknown"][2], "ftd": {}}
    dday_count, detail = count_distribution_days(price, volume)
    ftd = detect_ftd(price, volume)
    regime = classify_regime(dday_count, ftd["ftd_active"])
    label, advice, color = REGIME_ACTIONS[regime]
    return {
        "regime":      regime,
        "label":       label,
        "advice":      advice,
        "color":       color,
        "dday_count":  dday_count,
        "details":     detail,
        "ftd_active":  ftd["ftd_active"],
        "ftd_day":     ftd["ftd_day"],
        "ftd":         ftd,
    }
