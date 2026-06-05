"""
Edge Engine — the meta-layer that ties all scanners into a measurable edge.

Provides four core capabilities, all using ONLY local bhavcopy cache:

1. Market Regime Detector  — Distribution Day count + Follow-Through Day signal
2. Setup Quality Score     — composite 0-100 ranking across all scanners
3. Failed Breakout Detector— exit signal automation (7-8% rule, MA50 break, etc.)
4. Backtester              — walk-forward validation of any scanner output

Zero new NSE API calls. All computations from cached EOD data.
"""
import time
import math
import random
import heapq
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from data_fetcher import _weekdays_back, _download_one_day
from analysis_utils import stage_analysis, stage_label, NIFTY_PROXY_SYMS, equal_weight_index
from risk_config import (
    POSITION_SIZE_FRAC, BT_COOLDOWN_BARS,
    MAX_CONCURRENT_POSITIONS, BT_LOAD_DAYS, BT_LOOKBACK_BARS,
)
from industry_groups import INDUSTRY_GROUPS
import benchmark
from costs import round_trip_cost_pct

# ── Captured benchmark ────────────────────────────────────────────────────────
# Populated by _load_stocks(): the real NIFTYBEES close series pulled from raw
# bhavcopy BEFORE any universe filter strips it. Backtests use this — not the
# synthetic equal-weight proxy — so every "edge" claim is excess return vs the
# actual Nifty 50 (cap-weighted, dividend-reinvested).
_BENCH: pd.Series | None = None


# ── Benchmark resolver — used by every return-based calc (Phase 5) ────────────
def _best_bench(proxy_df: pd.DataFrame | None) -> pd.Series | None:
    """
    Return the single best price series to use as 'the market' for RETURN
    calculations (RS lines, alpha, regime nifty_close).

    Preference order:
      1. Real NIFTYBEES close (captured by _load_stocks)  — cap-weighted, with div
      2. Equal-weight 20-stock proxy from analysis_utils  — fallback for volume D-Day
      3. None                                              — caller must handle
    """
    if _BENCH is not None and len(_BENCH) >= 60:
        return _BENCH
    if proxy_df is not None and "Close" in getattr(proxy_df, "columns", []):
        return proxy_df["Close"]
    return None


# ── Walk-forward score validation helpers (Phase 2) ───────────────────────────
def _spearman_ic(xs: list, ys: list) -> float | None:
    """Spearman rank correlation. NaN-safe; returns None on degenerate input."""
    if not xs or not ys or len(xs) != len(ys) or len(xs) < 5:
        return None
    try:
        s = pd.Series(xs).rank()
        a = pd.Series(ys).rank()
        c = s.corr(a)
        return float(c) if not pd.isna(c) else None
    except Exception:
        return None


# ── Evidence-based component scorers ──────────────────────────────────────────
# Derived from per-feature IC analysis (analyze_score_components):
#   r12m           → IC +0.145  t-stat +2.49  STRONG  (the only single-feature alpha signal)
#   atr_pct        → IC +0.102  (lower volatility = higher forward alpha)
#   pct_from_high  → IC +0.090  (closer to 52W high = higher forward alpha)
# Best 2-component pair (r12m + atr_pct) → IC 0.159.
# Weighting 0.55 / 0.25 / 0.20 lines up with the IC ratios.

def _piece_score(value, points: list[tuple[float, float]]) -> float:
    """Piecewise-linear mapping. `points` MUST be sorted by x ascending."""
    if value is None:
        return 50.0
    try:
        if pd.isna(value):
            return 50.0
    except Exception:
        pass
    if value <= points[0][0]:
        return float(points[0][1])
    if value >= points[-1][0]:
        return float(points[-1][1])
    for i in range(len(points) - 1):
        v0, s0 = points[i]
        v1, s1 = points[i + 1]
        if v0 <= value <= v1:
            return float(s0 + (s1 - s0) * (value - v0) / (v1 - v0))
    return 50.0


# r12m (12-month return %): big gains → high score. Calibrated against the
# observed distribution of Indian-equity 12m returns over 2022-2025.
_R12M_POINTS = [(-60.0, 0), (-30.0, 15), (-10.0, 30), (0.0, 45),
                (15.0, 60), (30.0, 70), (50.0, 80), (100.0, 95), (200.0, 100)]

# atr_pct (daily ATR as % of price): lower volatility scored higher.
# Most NSE large-caps sit between 1-3%; <1% rare and pristine, >8% washy.
_ATR_POINTS = [(0.5, 100), (1.5, 90), (3.0, 70), (5.0, 50), (8.0, 25), (15.0, 0)]

# pct_from_high (% below 52W high; 0 = at high, negative = below).
# Closer to the high = stronger trend / less overhead resistance.
_PFH_POINTS = [(-60.0, 0), (-30.0, 30), (-15.0, 55),
               (-7.0, 75), (-2.0, 95), (0.0, 100)]


def _r12m_subscore(r12m): return _piece_score(r12m, _R12M_POINTS)
def _atr_subscore(atr_pct): return _piece_score(atr_pct, _ATR_POINTS)
def _pfh_subscore(pfh): return _piece_score(pfh, _PFH_POINTS)


def _compute_atr_pct(close: pd.Series, df_sub: pd.DataFrame | None) -> float | None:
    """Wilder ATR(14) / current price * 100 — None when not enough history."""
    if df_sub is None or "High" not in df_sub.columns or "Low" not in df_sub.columns:
        return None
    try:
        hi = df_sub["High"].dropna()
        lo = df_sub["Low"].dropna()
        cl = df_sub["Close"].dropna()
        if len(hi) < 15:
            return None
        cur = float(cl.iloc[-1])
        _tr = pd.concat([
            hi - lo,
            (hi - cl.shift(1)).abs(),
            (lo - cl.shift(1)).abs(),
        ], axis=1).max(axis=1)
        atr14 = float(_tr.ewm(alpha=1 / 14, adjust=False).mean().iloc[-1])
        return (atr14 / cur * 100) if cur > 0 else None
    except Exception:
        return None


def _lightweight_score(close: pd.Series, vol: pd.Series,
                       bench: pd.Series | None,
                       df_sub: pd.DataFrame | None = None) -> float | None:
    """
    Evidence-based score: weighted blend of the 3 highest-IC features.
    Same formula used in walk-forward validation, tier filtering for backtests,
    and (modulo display fields) the live ranking in compute_setup_score.

    Requires ≥ 252 bars (one year) — r12m is the dominant signal and we won't
    fake it from r6m. Stocks with shorter history are excluded from ranking.

    `df_sub` (the parent OHLCV frame) is needed for ATR. Without it we fall
    back to a neutral ATR sub-score so the function still works in legacy
    callsites that only pass close/vol.
    """
    if len(close) < 252 or len(vol) < 50:
        return None
    cur = float(close.iloc[-1])
    if cur < 30:
        return None

    r12m = (cur / float(close.iloc[-252]) - 1) * 100
    high252 = float(close.iloc[-252:].max())
    pct_from_high = (cur / high252 - 1) * 100 if high252 > 0 else 0.0
    atr_pct = _compute_atr_pct(close, df_sub)

    r12m_s = _r12m_subscore(r12m)
    atr_s  = _atr_subscore(atr_pct) if atr_pct is not None else 50.0
    pfh_s  = _pfh_subscore(pct_from_high)

    return round(0.55 * r12m_s + 0.25 * atr_s + 0.20 * pfh_s, 1)

# ── Cache ─────────────────────────────────────────────────────────────────────
_cache    = {"data": None, "ts": 0}
CACHE_TTL = 1800  # 30 min

def invalidate_cache():
    """Force-clear the edge engine cache so the next run always re-scores from scratch."""
    _cache["data"] = None
    _cache["ts"]   = 0

# ── Symbol → Group lookup ─────────────────────────────────────────────────────
_SYM_TO_GROUP: dict[str, str] = {}
for _grp, _syms in INDUSTRY_GROUPS.items():
    for _s in _syms:
        _SYM_TO_GROUP[_s] = _grp

_NIFTY_SYMS = NIFTY_PROXY_SYMS   # TIER-3: canonical 20-stock list from analysis_utils

# ── ETF exclusion ─────────────────────────────────────────────────────────────
# Symbols that END with these strings are always ETFs/funds
_ETF_END = ("ETF", "BEES", "FUND", "BENCHMARK", "NIFTY1", "IETF", "BETF")

# Symbols that CONTAIN these strings are always ETFs/funds
# "NIFTY" and "SENSEX" never appear in real equity company tickers
_ETF_HAS = (
    "NIFTY", "SENSEX",                          # index products
    "LIQUID", "IETF", "CPSE", "GOLDETF",        # fund types
    "SILVETF", "MAKEINDIA", "MAFANG",
    "MIDCAPETF", "INFRABEES", "BANKBEES",
    "JUNIORBEES", "PSUBNKBEES", "NIFTYETF",
    "SENSEXETF", "SETFNIF", "SETF", "HNGSNG",
    "CPSEETF", "BBETF", "ABSLNN50ET",
    "GSEC10", "GSEC5", "GILT5Y", "GILT10",
    "LTGILT", "LIQGRW", "DIVOPPB",
    "CONSUMB", "MANUFGB", "MID150B",
    "ELIQUID", "CASHIET",
    "GROWWG", "GROWWN",                         # Groww Gold/Nifty ETFs
)

# Exact symbol matches that are known ETFs not caught by patterns above
_ETF_EXACT = frozenset({
    # Axis ETFs
    "AXISGOLD", "AXISILVER", "AXISBPSETF", "AXISBNKETF",
    "AXISCETF", "AXISHCETF", "AXISTECETF", "AXISVALUE",
    # Aditya Birla / ABSL ETFs
    "AONEGOLD", "AONESILVER", "AONELIQUID",
    "ABSLLIQUID", "ABSL10BANK",
    # Bank/PSU factor ETFs
    "BANKBETA", "BANKADD", "BANK10ADD", "BANKPSU", "EBANKNIFTY",
    # BSL / BSE
    "BSLNIFTY", "BSLGOLDETF", "BSLSENETFG", "BSE500IETF",
    # Edelweiss ETFs (E-prefix family)
    "EGOLD", "ESILVER", "ENIFTY", "ESENSEX",
    # Alphaetf / others
    "ALPHAETF", "ALPL30IETF", "AUTOBEES", "AUTOIETF",
    # BNP Paribas
    "BBNPPGOLD", "BBNPNBETF",
    # Choice / Deccan
    "CHOICEGOLD", "DECNGOLD",
    # Commodity / sector iETFs
    "COMMOIETF", "CONSUMIETF", "EVIETF", "FINIETF", "FMCGIETF",
    "HEALTHIETF", "INFRAIETF", "ITBEES", "ITETF", "ITIETF",
    "LOWVOLIETF", "METALIETF", "MIDCAPIETF", "MIDSELIETF", "MOM30IETF",
    # Gold ETFs
    "GOLDBEES", "GOLDETF", "GOLDIETF",
    "GOLD1", "GOLD360", "GOLDADD", "GOLDBETA", "GOLDBND", "GOLDCASE",
    "HDFCGOLD", "HSBCGOLD", "IVZINGOLD", "LICMFGOLD", "MOGOLD",
    "QGOLDHALF", "SHANTIGOLD", "TATAGOLD", "UNIONGOLD",
    # Silver ETFs
    "ESILVER", "MOSILVER", "SBISILVER", "HDFCSILVER",
    "SILVER", "SILVER1", "SILVER360", "SILVERADD", "SILVERAG",
    "SILVERBETA", "SILVERBND", "SILVERCASE", "SILVERTUC",
    # Gilt / bond ETFs
    "GILT10BETA", "GILT5BETA", "EUROBOND", "GOLDBND", "SILVERBND",
    # HDFC ETFs
    "HDFCSILVER",
    # Invesco
    "IVZINGOLD", "IVZINNIFTY",
    # Mirae / Motilal
    "MOSILVER", "MOGSEC",
    "MONIFTY100", "MONIFTY500",
    # Misc
    "IDFNIFTYET", "NIFTY100EW", "NIFTYADD", "NIFTYBETA",
    "NIFTYCASE", "NIFTYQLITY", "QNIFTY",
    "SENSEXADD", "SENSEXBETA",
    "ABGSEC",
})

def _is_etf(sym: str) -> bool:
    s = sym.upper()
    return (s in _ETF_EXACT or
            any(s.endswith(x) for x in _ETF_END) or
            any(k in s for k in _ETF_HAS))


# ── Data loader ───────────────────────────────────────────────────────────────

def _load_stocks(progress_callback=None, days: int = 400,
                 survivorship_free: bool = False) -> dict[str, pd.DataFrame]:
    """
    Load per-symbol OHLCV from the bhavcopy cache.

    survivorship_free=False  → filter to the CURRENT curated universe
                               (Nifty50∪Next50∪500∪Smallcap250∪Microcap250).
                               Use for ranking/scoring — we score the present.
    survivorship_free=True   → NO universe filter; loads every symbol that
                               ever traded in the window (incl. delisted).
                               Use for backtesting — the only way to avoid
                               survivorship bias.

    Side-effect: captures the real NIFTYBEES close series into module-level
    _BENCH BEFORE filtering, so the benchmark survives even when callers
    later filter to the curated universe.
    """
    global _BENCH

    universe: set[str] = set()
    if not survivorship_free:
        try:
            from nse_stocks import get_universe_symbols
            universe = set(get_universe_symbols())
        except Exception:
            universe = set()

    dates  = _weekdays_back(days)
    total  = len(dates)
    frames = []
    bench_recs: dict[pd.Timestamp, float] = {}
    bench_syms = [benchmark.BENCH_SYMBOL] + benchmark.FALLBACK_SYMBOLS

    for i, dt in enumerate(dates):
        df = _download_one_day(dt)
        if df is not None:
            # 1) Capture NIFTYBEES BEFORE any filter touches the frame.
            for bsym in bench_syms:
                sub = df[df["Symbol"] == bsym]
                if not sub.empty:
                    bench_recs[pd.Timestamp(dt)] = float(sub.iloc[0]["Close"])
                    break
            # 2) Then optionally narrow to the curated universe to save memory.
            if universe:
                df = df[df["Symbol"].isin(universe)]
            frames.append(df)
        if progress_callback and i % 40 == 0:
            progress_callback(i, total, f"Loading bhavcopy… {i}/{total}")

    if bench_recs:
        bser = pd.Series(bench_recs)
        bser.index = pd.to_datetime(bser.index)
        if getattr(bser.index, "tz", None) is not None:
            bser.index = bser.index.tz_localize(None)
        _BENCH = bser.sort_index().astype(float)

    if not frames:
        return {}
    combined = pd.concat(frames, ignore_index=True).sort_values("Date")
    stocks: dict[str, pd.DataFrame] = {}
    for sym, grp in combined.groupby("Symbol"):
        if universe and sym not in universe:
            continue
        _keep = [c for c in ["Open", "High", "Low", "Close", "Volume", "DelivPer"]
                 if c in grp.columns]
        g = grp.set_index("Date")[_keep]
        if not isinstance(g.index, pd.DatetimeIndex):
            g.index = pd.to_datetime(g.index)
        g = g[~g.index.duplicated(keep="last")].sort_index()
        if len(g) >= 60:
            stocks[sym] = g
    return stocks


def _build_nifty_proxy(stocks: dict) -> pd.DataFrame | None:
    """Build a Nifty proxy index (price + volume) from large-cap constituents."""
    closes, vols = [], []
    for sym in _NIFTY_SYMS:
        df = stocks.get(sym)
        if df is not None and len(df) >= 100:
            closes.append(df["Close"].dropna())
            vols.append(df["Volume"].dropna())
    if len(closes) < 5:
        return None
    from analysis_utils import equal_weight_index
    px  = equal_weight_index(pd.concat(closes, axis=1).dropna(how="all"))
    vol = pd.concat(vols,   axis=1).dropna(how="all").sum(axis=1)
    out = pd.DataFrame({"Close": px, "Volume": vol}).dropna()
    return out if len(out) >= 80 else None


# ──────────────────────────────────────────────────────────────────────────────
# 1. MARKET REGIME — Distribution Day count + Follow-Through Day
# ──────────────────────────────────────────────────────────────────────────────

def detect_market_regime(nifty: pd.DataFrame) -> dict:
    """
    Distribution Day = Nifty closes ≤ -0.2% on volume HIGHER than prior day.
    5+ D-Days in last 25 sessions = market under institutional selling.

    Follow-Through Day = on day 4-7 after a recent low, Nifty closes UP ≥ 1.4%
    on volume higher than prior day. Confirms a new uptrend.

    Returns regime: "Confirmed Uptrend" | "Uptrend Under Pressure" | "Correction"
    """
    if nifty is None or len(nifty) < 30:
        return {"regime": "Unknown", "dday_count": 0, "ftd_active": False, "details": []}

    n     = nifty.tail(60).copy()
    # BUG-016 NOTE / TODO: This uses the synthetic Nifty proxy built from a
    # basket of large-caps — NOT the official Nifty50 index. The volume series
    # is therefore stock volume, which approximates institutional activity
    # but is not the index volume an IBD-style D-day count traditionally uses.
    # Known limitation: prefer official Nifty volume when available.
    n["pct_chg"]   = n["Close"].pct_change() * 100
    n["vol_chg"]   = n["Volume"].pct_change()
    n["dday"]      = (n["pct_chg"] <= -0.2) & (n["Volume"] > n["Volume"].shift(1))

    last25 = n.tail(25)
    dday_count = int(last25["dday"].sum())

    # Recent low + Follow-Through Day check
    recent_low_idx = n["Close"].tail(20).idxmin()
    days_since_low = (n.index[-1] - recent_low_idx).days if recent_low_idx else 999
    ftd_active = False
    ftd_day    = None
    if 3 <= (n.index.get_loc(recent_low_idx) if recent_low_idx in n.index else -1):
        # find FTD: day 4-7 after low with +1.4% on rising volume
        try:
            low_pos = n.index.get_loc(recent_low_idx)
            for i in range(low_pos + 4, min(low_pos + 8, len(n))):
                row = n.iloc[i]
                # BUG-019 FIX: FTD threshold changed from 1.4% to 1.7% to match market_breadth.py
                if row["pct_chg"] >= 1.7 and row["Volume"] > n.iloc[i-1]["Volume"]:
                    ftd_active = True
                    ftd_day    = n.index[i]
                    break
        except Exception:
            pass

    # Regime classification
    if dday_count >= 6:
        regime = "Correction"
    elif dday_count >= 4:
        regime = "Uptrend Under Pressure"
    elif ftd_active and dday_count <= 3:
        regime = "Confirmed Uptrend"
    else:
        regime = "Uptrend Under Pressure" if dday_count >= 3 else "Confirmed Uptrend"

    # Action recommendation
    action_map = {
        "Confirmed Uptrend":      ("✅ Buy Mode", "Full deployment — buy strongest setups",        "#22c55e"),
        "Uptrend Under Pressure": ("🟡 Cautious", "Selective — only highest-conviction setups",   "#eab308"),
        "Correction":             ("🔴 Cash",     "Do not initiate new longs — preserve capital", "#ef4444"),
        "Unknown":                ("⚪ Unknown",  "Insufficient data",                              "#94a3b8"),
    }
    label, advice, color = action_map[regime]

    # Latest 25 sessions detail
    detail = []
    for idx, row in last25.iterrows():
        if row["dday"]:
            detail.append({
                "date": idx.strftime("%d-%b"),
                "pct":  round(row["pct_chg"], 2),
                "vol_chg_pct": round(row["vol_chg"] * 100, 1),
            })

    # Phase 5: prefer real NIFTYBEES for the displayed nifty_close / nifty_chg
    # (D-Day algorithm above still uses the proxy because NIFTYBEES is an ETF
    # whose volume doesn't reflect underlying institutional Nifty activity).
    if _BENCH is not None and len(_BENCH) >= 2:
        bench_close = float(_BENCH.iloc[-1])
        bench_chg   = float(_BENCH.pct_change().iloc[-1]) * 100
        bench_src   = "NIFTYBEES"
    else:
        bench_close = float(nifty["Close"].iloc[-1])
        bench_chg   = float(nifty["Close"].pct_change().iloc[-1]) * 100
        bench_src   = "proxy"

    return {
        "regime":      regime,
        "label":       label,
        "advice":      advice,
        "color":       color,
        "dday_count":  dday_count,
        "ftd_active":  ftd_active,
        "ftd_day":     ftd_day.strftime("%d-%b-%Y") if ftd_day is not None else None,
        "days_since_low": days_since_low,
        "details":     detail,
        "nifty_close": round(bench_close, 2),
        "nifty_chg":   round(bench_chg, 2),
        "bench_source": bench_src,
    }


# ──────────────────────────────────────────────────────────────────────────────
# 2. BASE QUALITY VALIDATOR — O'Neil proper-base rules
# ──────────────────────────────────────────────────────────────────────────────

def validate_base(close: pd.Series) -> dict:
    """
    Score a base 0-100 based on O'Neil's proper-base criteria.

    Checks:
      - Base length (≥ 7 weeks = 35 trading days, ideal 7-15 weeks)
      - Base depth (≤ 33% from peak)
      - Prior uptrend before base (≥ 30%)
      - Tightness on right side
      - Volume drying up during base
    """
    if len(close) < 100:
        return {"score": 0, "valid": False, "reason": "Insufficient history"}

    # Detect base: longest recent window with range < 35%
    n = min(180, len(close) - 5)
    base_len = 0
    for days in range(n, 30, -5):
        slc = close.iloc[-days:]
        lo, hi = float(slc.min()), float(slc.max())
        if hi > lo and (hi - lo) / lo * 100 < 35:
            base_len = days
            break

    if base_len < 30:
        return {"score": 0, "valid": False, "reason": "No base detected (< 6 weeks)",
                "base_len": base_len, "depth": 0}

    base = close.iloc[-base_len:]
    base_lo, base_hi = float(base.min()), float(base.max())
    depth = (base_hi - base_lo) / base_hi * 100   # depth from peak

    # Prior uptrend (50 days before base)
    pre_base_end   = len(close) - base_len
    pre_base_start = max(0, pre_base_end - 50)
    if pre_base_end > pre_base_start + 10:
        pre = close.iloc[pre_base_start:pre_base_end]
        prior_uptrend = (float(pre.iloc[-1]) / float(pre.iloc[0]) - 1) * 100
    else:
        prior_uptrend = 0.0

    # Tightness on right side (last 10 bars vs whole base)
    right = base.iloc[-10:]
    right_range = (float(right.max()) - float(right.min())) / float(right.min()) * 100
    tightness   = max(0, 100 - right_range * 5)  # tighter = higher score

    # Score components (each 0-25)
    s_length = min(25, max(0, (base_len - 30) / 60 * 25))           # ideal 30-90
    s_depth  = 25 if depth <= 25 else max(0, 25 - (depth - 25))     # ideal ≤ 25%
    s_trend  = min(25, max(0, prior_uptrend / 30 * 25))             # need ≥ 30%
    s_tight  = min(25, tightness / 4)                                # right-side tightness

    score = round(s_length + s_depth + s_trend + s_tight, 1)
    valid = score >= 55 and depth <= 33 and base_len >= 30 and prior_uptrend >= 20

    return {
        "score":         score,
        "valid":         valid,
        "base_len":      base_len,
        "base_weeks":    round(base_len / 5, 1),
        "depth":         round(depth, 1),
        "prior_uptrend": round(prior_uptrend, 1),
        "tightness":     round(tightness, 1),
        "reason":        "" if valid else (
            "Depth > 33%" if depth > 33 else
            "Prior uptrend < 20%" if prior_uptrend < 20 else
            "Score below 55"
        ),
    }


# ──────────────────────────────────────────────────────────────────────────────
# 3. FAILED BREAKOUT / EXIT SIGNAL DETECTOR
# ──────────────────────────────────────────────────────────────────────────────

def detect_exit_signals(df: pd.DataFrame, entry_price: float | None = None,
                       entry_date: str | None = None) -> dict:
    """
    Returns active exit signals on a stock. If entry_price is given,
    also checks the 7-8% loss rule and breakeven trigger.
    """
    close = df["Close"].dropna()
    vol   = df["Volume"].dropna()
    if len(close) < 50:
        return {"signals": [], "action": "HOLD"}

    cur     = float(close.iloc[-1])
    prev    = float(close.iloc[-2])
    ma50    = float(close.rolling(50).mean().iloc[-1])
    vol_avg = float(vol.iloc[-50:-1].mean())
    vol_now = float(vol.iloc[-1])

    signals = []

    # 7-8% hard stop loss
    if entry_price is not None:
        loss_pct = (cur / entry_price - 1) * 100
        if loss_pct <= -7:
            signals.append({"sev": "HIGH", "tag": "STOP",
                            "msg": f"Down {loss_pct:.1f}% from entry — 7% stop hit. EXIT."})

    # MA50 break on heavy volume
    if cur < ma50 and prev >= ma50 and vol_now > vol_avg * 1.3:
        signals.append({"sev": "HIGH", "tag": "MA50 BREAK",
                        "msg": "Closed below 50-day MA on volume > 1.3× avg. EXIT."})

    # Distribution day on the stock itself
    day_chg = (cur / prev - 1) * 100
    if day_chg <= -2.0 and vol_now > vol_avg * 1.5:
        signals.append({"sev": "HIGH", "tag": "DISTRIBUTION",
                        "msg": f"Down {day_chg:.1f}% on volume {vol_now/vol_avg:.1f}× avg. Institutions selling."})

    # Climax run (parabolic — consider profit-taking)
    if len(close) >= 21:
        r3w = (cur / float(close.iloc[-15]) - 1) * 100
        if r3w >= 30:
            signals.append({"sev": "MEDIUM", "tag": "CLIMAX",
                            "msg": f"Up {r3w:.1f}% in 3 weeks — consider taking partial profits."})

    # Failed breakout: stock made recent 20-day high then closed below it within 3 days
    # BUG-015 FIX: a "20-day high" should be based on intraday HIGH, not just CLOSE,
    # because the breakout level traders watch is the prior 20 bars' true high.
    high = df["High"].dropna() if "High" in df.columns else close
    last20_high = float(high.tail(20).max())
    if cur < last20_high * 0.97 and float(high.tail(5).max()) >= last20_high * 0.998:
        signals.append({"sev": "MEDIUM", "tag": "FAILED BO",
                        "msg": "Made new 20-day high then reversed — breakout failed."})

    # Action
    high_count = sum(1 for s in signals if s["sev"] == "HIGH")
    if high_count >= 1:
        action = "EXIT"
    elif any(s["sev"] == "MEDIUM" for s in signals):
        action = "TRIM / TIGHTEN SL"
    else:
        action = "HOLD"

    return {"signals": signals, "action": action,
            "price": round(cur, 2), "ma50": round(ma50, 2)}


# ──────────────────────────────────────────────────────────────────────────────
# F10 helper — Historical Base Count
# ──────────────────────────────────────────────────────────────────────────────

def _count_historical_bases(close: pd.Series) -> int:
    """
    Count completed bases PRIOR to the current consolidation.
    A base = any contiguous window ≥ 30 bars where price range < 35%.
    Walks backward from the current base's starting point.
    """
    if len(close) < 100:
        return 0
    cur_base = validate_base(close)
    base_len  = cur_base.get("base_len", 0)
    if base_len < 30:
        return 0
    count = 0
    pos   = len(close) - base_len   # index of the bar just before current base
    while pos >= 60:
        sub = close.iloc[:pos]
        b   = validate_base(sub)
        if b.get("valid") and b.get("base_len", 0) >= 30:
            count += 1
            pos   -= b["base_len"]
        elif b.get("base_len", 0) >= 30:          # detected but not "valid" — still a base
            pos -= b["base_len"]
        else:
            break
    return count


# ──────────────────────────────────────────────────────────────────────────────
# 4. SETUP QUALITY SCORE — Composite 0-100 ranking
# ──────────────────────────────────────────────────────────────────────────────

def compute_setup_score(symbol: str, df: pd.DataFrame, regime: dict,
                        sector_quad: dict[str, str], fundamentals: dict | None = None,
                        nifty: pd.DataFrame | None = None) -> dict | None:
    """
    Composite Setup Quality Score combining technical, fundamental,
    market, sector, and risk:reward signals into a single 0-100 rank.

    Formula:
      Score = 0.25 × Technical
            + 0.25 × Fundamental
            + 0.20 × Market regime
            + 0.15 × Sector
            + 0.15 × Risk:Reward
    """
    if _is_etf(symbol):
        return None

    close = df["Close"].dropna()
    vol   = df["Volume"].dropna()
    if len(close) < 126:
        return None

    cur = float(close.iloc[-1])
    if cur < 30:
        return None

    # Liquidity — compute ADTV for reporting only (no gate: Nifty500 universe is already large-cap)
    # Align Close + Volume on a common non-NaN index before multiplying so
    # mis-aligned dropna() of the two columns doesn't pair wrong rows on
    # days when either column has a single missing value.
    _cv = df[["Close", "Volume"]].dropna()
    adtv_cr = float((_cv["Close"] * _cv["Volume"]).iloc[-20:].mean()) / 1e7 if len(_cv) >= 20 else 0.0

    # ── TECHNICAL (0-100) ─────────────────────────────────────────────────────
    # Returns, RS, base quality, MA alignment, volume profile
    r3m  = (cur / float(close.iloc[-63])  - 1) * 100 if len(close) >= 63  else 0
    r6m  = (cur / float(close.iloc[-126]) - 1) * 100 if len(close) >= 126 else 0
    ma50 = float(close.rolling(50).mean().iloc[-1])  if len(close) >= 50  else cur
    ma200= float(close.rolling(200).mean().iloc[-1]) if len(close) >= 200 else cur

    base = validate_base(close)
    base_score = base["score"] if base["valid"] else base["score"] * 0.5

    vol_r10 = float(vol.iloc[-10:].mean())
    vol_a50 = float(vol.iloc[-50:].mean()) if len(vol) >= 50 else float(vol.mean())
    vol_ratio = vol_r10 / vol_a50 if vol_a50 > 0 else 1.0

    ma_score    = 100 if (cur > ma50 > ma200) else (60 if cur > ma50 else 30)
    ret_score   = min(100, max(0, (r3m + 20) * 2))   # -20%→0, 30%→100
    # BUG-017 FIX: tiered volume scoring (replaces linear vol_ratio * 50 which caps at 2x)
    if vol_ratio >= 4:
        vol_score = 100
    elif vol_ratio >= 2:
        vol_score = 70 + (vol_ratio - 2) * 15
    elif vol_ratio >= 1.5:
        vol_score = 50 + (vol_ratio - 1.5) * 40
    elif vol_ratio >= 1:
        vol_score = 30 + (vol_ratio - 1) * 40
    else:
        vol_score = vol_ratio * 30
    vol_score = min(100, max(0, vol_score))
    technical   = round(0.30 * base_score + 0.25 * ma_score +
                        0.25 * ret_score + 0.20 * vol_score, 1)

    # ── FUNDAMENTAL (0-100) ───────────────────────────────────────────────────
    if fundamentals:
        eps_g   = fundamentals.get("eps_growth_yoy", 0) or 0
        sales_g = fundamentals.get("sales_growth_yoy", 0) or 0
        roe     = fundamentals.get("roe", 0) or 0
        d_e     = fundamentals.get("debt_to_equity", 999) or 999
        eps_s   = min(100, max(0, eps_g * 2))      # 25% → 50, 50% → 100
        sale_s  = min(100, max(0, sales_g * 3))    # 20% → 60, 33% → 100
        roe_s   = min(100, max(0, roe * 4))        # 15% → 60, 25% → 100
        de_s    = 100 if d_e < 0.5 else 70 if d_e < 1.0 else 40 if d_e < 2.0 else 10
        # F2: EPS acceleration bonus (quarterly acceleration adds 10 pts)
        # BUG-020 FIX: check for truthy value using int() cast to avoid silent failures
        eps_accel_val = fundamentals.get("eps_accel")
        accel_bonus   = 10 if eps_accel_val and int(eps_accel_val) >= 1 else 0
        fundamental = round(0.35 * eps_s + 0.30 * sale_s + 0.20 * roe_s + 0.15 * de_s, 1)
        fundamental = min(100, fundamental + accel_bonus)
        # F3: promoter delta
        promo_delta  = fundamentals.get("promoter_delta")
        promo_hold   = fundamentals.get("promoter_holding", 0) or 0
        has_fundamentals = True
    else:
        fundamental  = 50.0   # neutral when no data
        promo_delta  = None
        promo_hold   = None
        has_fundamentals = False

    # ── MARKET (0-100) ────────────────────────────────────────────────────────
    market_map = {
        "Confirmed Uptrend": 95, "Uptrend Under Pressure": 60,
        "Correction": 20, "Unknown": 50,
    }
    market = market_map.get(regime.get("regime", "Unknown"), 50)

    # ── SECTOR (0-100) ────────────────────────────────────────────────────────
    grp = _SYM_TO_GROUP.get(symbol, "")
    quad = sector_quad.get(grp, "")
    sector_score_map = {
        "Leading": 95, "Improving": 75, "Weakening": 40, "Lagging": 15, "": 50,
    }
    sector = sector_score_map.get(quad, 50)

    # ── F1: DELIVERY % (institutional participation signal) ───────────────────
    deliv_ser  = df["DelivPer"].dropna() if "DelivPer" in df.columns else pd.Series(dtype=float)
    deliv_pct  = float(deliv_ser.iloc[-1])   if len(deliv_ser) >= 1  else None
    deliv_avg  = float(deliv_ser.iloc[-20:].mean()) if len(deliv_ser) >= 5 else None
    # delivery score: 60%+ is institutional accumulation, 40%- is retail
    if deliv_pct is not None:
        deliv_score = min(100, max(0, deliv_pct * 1.4))   # 70% → 98, 50% → 70
    else:
        deliv_score = 50.0   # neutral when missing

    # ── F6: ATR-BASED STOP + POSITION SIZING ─────────────────────────────────
    hi_ser  = df["High"].dropna()
    lo_ser  = df["Low"].dropna()
    cl_ser  = close   # already computed
    _tr = pd.concat([
        hi_ser - lo_ser,
        (hi_ser - cl_ser.shift(1)).abs(),
        (lo_ser - cl_ser.shift(1)).abs(),
    ], axis=1).max(axis=1)
    # Wilder's ATR (EWM with alpha = 1/period). Matches analysis_utils.atr()
    # and standard trading-platform behaviour. The old rolling(14).mean()
    # was a plain SMA which over-states ATR in trending markets by 5–15%,
    # making stop distances inconsistent with every other module.
    if len(_tr) >= 14:
        atr14_val = float(_tr.ewm(alpha=1/14, adjust=False).mean().iloc[-1])
    else:
        atr14_val = float(_tr.mean())
    atr_pct   = round(atr14_val / cur * 100, 2) if cur > 0 else 0.0
    # BUG-018 FIX: use 2.0x ATR multiplier (was 1.5x) to match portfolio.py and professional practice
    atr_stop  = round(cur - 2.0 * atr14_val, 2)
    # Position sizing: risk ≤1% of ₹10L capital = ₹10,000 per trade
    risk_per_share = max(cur - atr_stop, 0.01)
    shares_1pct   = int(10_000 / risk_per_share)   # qty for 1% risk on ₹10L

    # ── F7: RS LINE TRAJECTORY ────────────────────────────────────────────────
    # Phase 5: RS line is now stock vs REAL NIFTYBEES (cap-weighted, div-reinvested)
    # — the same benchmark every alpha calc uses — instead of the 20-stock proxy.
    rs_at_52w_high = False
    rs_slope_3m    = 0.0
    rs_current     = None
    _bench_for_rs  = _best_bench(nifty)
    if _bench_for_rs is not None and len(_bench_for_rs) >= 130:
        nifty_aligned = _bench_for_rs.reindex(close.index, method="ffill").dropna()
        aligned = pd.DataFrame({"stock": close, "nifty": nifty_aligned}).dropna()
        if len(aligned) >= 100:
            rs_line   = aligned["stock"] / aligned["nifty"]
            rs_curr_f = float(rs_line.iloc[-1])
            rs_52wh   = float(rs_line.iloc[-252:].max()) if len(rs_line) >= 252 \
                        else float(rs_line.max())
            rs_3m_slc = rs_line.iloc[-66:] if len(rs_line) >= 66 else rs_line
            rs_slope_3m  = round((float(rs_3m_slc.iloc[-1]) / float(rs_3m_slc.iloc[0]) - 1) * 100, 2) \
                           if len(rs_3m_slc) > 1 else 0.0
            rs_at_52w_high = rs_curr_f >= rs_52wh * 0.98
            rs_current     = round(rs_curr_f, 6)

    # RS adds to technical score
    rs_bonus = 10 if rs_at_52w_high else (5 if rs_slope_3m > 5 else 0)
    technical = min(100, round(technical + rs_bonus, 1))

    # ── F10: HISTORICAL BASE COUNT ────────────────────────────────────────────
    historical_base_count = _count_historical_bases(close)

    # ── RISK:REWARD (0-100) ───────────────────────────────────────────────────
    # If near pivot (top of base), R:R is ideal. Far from pivot = poor R:R.
    if base["valid"] and base.get("base_len", 0) > 30:
        base_slc = close.iloc[-base["base_len"]:]
        base_hi  = float(base_slc.max())
        base_lo  = float(base_slc.min())
        pos_in_base = (cur - base_lo) / (base_hi - base_lo) if base_hi > base_lo else 0.5
        # Sweet spot: 0.7-0.95 (just below pivot or just broken out)
        if 0.7 <= pos_in_base <= 0.98:
            rr_score = 95
        elif 0.5 <= pos_in_base < 0.7:
            rr_score = 70
        elif pos_in_base > 0.98:
            rr_score = 50   # extended
        else:
            rr_score = 40   # too low in base
    else:
        rr_score = 50

    # ── EVIDENCE-BASED COMPOSITE (Phase 7 — score redesign) ──────────────────
    # Old hand-tuned blend had IC 0.038 / t-stat 0.97 / non-monotonic quintiles.
    # The per-component IC analyzer found:
    #   r12m           IC +0.145 t +2.49 STRONG   ← dominant predictor
    #   atr_pct        IC +0.102           ← lower vol = better
    #   pct_from_high  IC +0.090 t +1.70 MOD     ← closer to 52W high = better
    # Best 2-feature pair (r12m + atr_pct) → IC 0.159.
    # New composite weights mirror these IC ratios: 0.55 / 0.25 / 0.20.
    r12m_val = ((cur / float(close.iloc[-252])) - 1) * 100 if len(close) >= 253 \
               else (r6m * 2.0 if r6m else 0.0)  # graceful degrade for IPOs
    high252 = float(close.iloc[-min(252, len(close)):].max())
    pct_from_high_val = (cur / high252 - 1) * 100 if high252 > 0 else 0.0

    evidence_score = round(
        0.55 * _r12m_subscore(r12m_val)
        + 0.25 * _atr_subscore(atr_pct)
        + 0.20 * _pfh_subscore(pct_from_high_val),
        1,
    )

    # Keep the old composite under legacy_score so users can see the delta and
    # we can compare both in walk-forward validation.
    if has_fundamentals:
        legacy_score = (0.22 * technical + 0.03 * deliv_score +
                         0.25 * fundamental + 0.20 * market +
                         0.15 * sector      + 0.15 * rr_score)
    else:
        legacy_score = (0.35 * technical + 0.05 * deliv_score +
                         0.25 * market    + 0.20 * sector +
                         0.15 * rr_score)
    legacy_score = round(legacy_score, 1)

    composite = evidence_score   # PRIMARY = evidence-based

    # ── Tier ──────────────────────────────────────────────────────────────────
    if composite >= 80:
        tier = "🏆 A+"
    elif composite >= 70:
        tier = "🥇 A"
    elif composite >= 60:
        tier = "🥈 B"
    elif composite >= 50:
        tier = "🥉 C"
    else:
        tier = ""

    stg = stage_analysis(close)

    return {
        "symbol":      symbol,
        "price":       round(cur, 2),
        "score":       composite,           # evidence-based (Phase 7)
        "legacy_score": legacy_score,       # old hand-tuned for transparency
        "tier":        tier,
        "technical":   technical,
        "fundamental": fundamental,
        "market":      market,
        "sector_score":sector,
        "rr_score":    round(rr_score, 1),
        "has_fundamentals": has_fundamentals,
        "r3m":         round(r3m, 2),
        "r6m":         round(r6m, 2),
        "r12m":          round(r12m_val, 2),
        "pct_from_high": round(pct_from_high_val, 2),
        "adtv_cr":     round(adtv_cr, 1),
        "above_ma50":  bool(cur > ma50),
        "above_ma200": bool(cur > ma200),
        "vol_ratio":   round(vol_ratio, 2),
        "base_score":  base["score"],
        "base_valid":  base["valid"],
        "base_weeks":  base.get("base_weeks", 0),
        "base_depth":  base.get("depth", 0),
        "stage":       stg,
        "stage_lbl":   stage_label(stg),
        "group":       grp,
        "quadrant":    quad,
        # F2+F3 — Quarterly EPS + Promoter
        "promoter_holding": promo_hold,
        "promoter_delta":   promo_delta,
        # F1 — Delivery %
        "deliv_pct":   round(deliv_pct, 1)  if deliv_pct  is not None else None,
        "deliv_avg20": round(deliv_avg, 1)  if deliv_avg  is not None else None,
        # F6 — ATR-based risk management
        "atr14":       round(atr14_val, 2),
        "atr_pct":     atr_pct,
        "atr_stop":    atr_stop,
        "shares_1pct": shares_1pct,
        # F7 — RS Line trajectory
        "rs_at_52w_high": rs_at_52w_high,
        "rs_slope_3m":    rs_slope_3m,
        # F10 — Historical base count
        "base_count":  historical_base_count,
    }


# ──────────────────────────────────────────────────────────────────────────────
# 5. BACKTESTER — walk-forward validation
# ──────────────────────────────────────────────────────────────────────────────

def _empty_bt_result(skipped: int = 0, candidates: int = 0) -> dict:
    """Stable shape so the UI never KeyError's on a no-trade result."""
    return {
        "trades": 0, "candidates": candidates, "skipped_no_capital": skipped,
        "win_rate": 0, "avg_win": 0, "avg_loss": 0,
        "expectancy": 0, "profit_factor": 0, "avg_hold": 0,
        "best": 0, "worst": 0, "by_reason": {},
        "avg_gross": 0, "avg_cost": 0,
        "avg_alpha": 0, "pct_beat_bench": 0,
        "max_drawdown": 0,
        "final_equity": 100.0, "final_bench": None, "final_alpha": None,
        "equity_curve": [], "drawdown_curve": [], "bench_equity": [],
    }


def _downsample(seq, n: int) -> list:
    """Reduce a sequence to ~n evenly-spaced points for UI chart rendering."""
    if not seq:
        return []
    if len(seq) <= n:
        return list(seq)
    step = max(1, len(seq) // n)
    return list(seq[::step])


def _generate_candidates(stocks: dict, signal_fn,
                         hold_days: int = 20, stop_pct: float = -7.0,
                         target_pct: float = 25.0, lookback_days: int = 800,
                         max_signals: int = 5000,
                         bench: pd.Series | None = None,
                         compute_wf_score: bool = False,
                         shuffle: bool = True) -> list[dict]:
    """
    Pure candidate generation — no filtering, no sim, no stats.

    When compute_wf_score=True, attaches `wf_score` (point-in-time lightweight
    score) to every candidate so callers can cheaply derive tier-filtered
    subsets without re-iterating the bars.

    Returns a list of trade-candidate dicts ready to feed into _run_sim_and_stats.
    """
    if bench is None:
        bench = _BENCH
    candidates: list[dict] = []
    items = list(stocks.items())
    if shuffle:
        # Deterministic shuffle so the capital cap isn't biased by alphabetical
        # ordering (otherwise A* names eat every available slot).
        random.Random(42).shuffle(items)

    for symbol, df in items:
        if _is_etf(symbol) or len(df) < lookback_days + hold_days + 10:
            continue

        # Per-symbol ADTV in ₹Cr for liquidity-scaled cost lookup
        cv = df[["Close", "Volume"]].dropna()
        adtv_cr = (float((cv["Close"] * cv["Volume"]).iloc[-60:].mean()) / 1e7
                   if len(cv) >= 20 else None)

        start_idx = len(df) - lookback_days
        end_idx   = len(df) - hold_days - 1
        cooldown  = 0
        for i in range(start_idx, end_idx):
            if cooldown > 0:
                cooldown -= 1
                continue
            try:
                if not signal_fn(df, i):
                    continue
                entry_idx = i + 1
                if entry_idx >= len(df):
                    continue
                entry = float(df["Open"].iloc[entry_idx])
                if entry <= 0:
                    continue
                sl_price = entry * (1 + stop_pct / 100)
                tg_price = entry * (1 + target_pct / 100)

                exit_price = None
                exit_reason = "TIME"
                exit_idx    = min(entry_idx + hold_days, len(df) - 1)
                for j in range(entry_idx, min(entry_idx + hold_days, len(df))):
                    op = float(df["Open"].iloc[j])
                    lo = float(df["Low"].iloc[j])
                    hi = float(df["High"].iloc[j])
                    # Gap-through-stop: open BELOW stop → fill at the open
                    # (worse than stop). Old code optimistically filled at the
                    # stop price even after a -10% overnight gap.
                    if j > entry_idx and op <= sl_price:
                        exit_price = op; exit_reason = "GAP_STOP"; exit_idx = j
                        break
                    if j > entry_idx and op >= tg_price:
                        exit_price = op; exit_reason = "GAP_TARGET"; exit_idx = j
                        break
                    if lo <= sl_price:
                        exit_price = sl_price; exit_reason = "STOP"; exit_idx = j
                        break
                    if hi >= tg_price:
                        exit_price = tg_price; exit_reason = "TARGET"; exit_idx = j
                        break
                if exit_price is None:
                    exit_price = float(df["Close"].iloc[exit_idx])

                gross_ret  = (exit_price / entry - 1) * 100
                cost_pct   = round_trip_cost_pct(adtv_cr)
                net_ret    = gross_ret - cost_pct

                entry_date = df.index[entry_idx] if entry_idx < len(df.index) else None
                exit_date  = df.index[exit_idx]  if exit_idx  < len(df.index) else None

                # Excess return vs NIFTYBEES over the same dates
                bench_ret = (benchmark.benchmark_return(entry_date, exit_date, bench=bench)
                             if (entry_date is not None and exit_date is not None) else None)
                alpha = (net_ret - bench_ret) if bench_ret is not None else None

                # Walk-forward score for tier filtering (only if requested)
                wf_score = None
                if compute_wf_score:
                    try:
                        df_sub    = df.iloc[:i + 1]
                        sub_close = df_sub["Close"].dropna()
                        sub_vol   = df_sub["Volume"].dropna()
                        wf_score  = _lightweight_score(sub_close, sub_vol, bench,
                                                        df_sub=df_sub)
                    except Exception:
                        pass

                candidates.append({
                    "symbol":     symbol,
                    "entry":      round(entry, 2),
                    "exit":       round(exit_price, 2),
                    "gross_ret":  round(gross_ret, 2),
                    "cost_pct":   round(cost_pct, 3),
                    "ret_pct":    round(net_ret, 2),       # net of costs (UI compat)
                    "bench_ret":  round(bench_ret, 2) if bench_ret is not None else None,
                    "alpha":      round(alpha, 2)     if alpha     is not None else None,
                    "reason":     exit_reason,
                    "days":       exit_idx - entry_idx,
                    "entry_date": entry_date,
                    "exit_date":  exit_date,
                    "adtv_cr":    round(adtv_cr, 2) if adtv_cr is not None else None,
                    "wf_score":   round(wf_score, 1) if wf_score is not None else None,
                })
                cooldown = BT_COOLDOWN_BARS
                if len(candidates) >= max_signals:
                    break
            except Exception:
                continue
        if len(candidates) >= max_signals:
            break

    return candidates


def _apply_regime_filter(candidates: list[dict],
                          nifty: pd.DataFrame | None) -> list[dict]:
    """Drop candidates whose entry_date falls in a Correction regime (D-Day ≥ 6)."""
    if not candidates or nifty is None or len(nifty) < 50:
        return candidates
    def _regime_at(dt):
        try:
            aligned_dt = nifty.index.asof(dt)
            if aligned_dt is None or pd.isna(aligned_dt):
                return "Unknown"
            sub = nifty[nifty.index <= aligned_dt].tail(30).copy()
            if len(sub) < 10:
                return "Unknown"
            sub["pct"]  = sub["Close"].pct_change() * 100
            sub["dday"] = (sub["pct"] <= -0.2) & (sub["Volume"] > sub["Volume"].shift(1))
            ddays = int(sub.tail(25)["dday"].sum())
            return "Correction" if ddays >= 6 else \
                   "Uptrend Under Pressure" if ddays >= 4 else "Confirmed Uptrend"
        except Exception:
            return "Unknown"
    return [c for c in candidates if _regime_at(c["entry_date"]) != "Correction"]


def _run_sim_and_stats(candidates: list[dict],
                        bench: pd.Series | None = None) -> dict:
    """
    Event-driven chronological sim + full stats dict.

    Cheap to call repeatedly with different candidate subsets — that's the whole
    point of the refactor. Tier-conditional backtests pre-compute candidates
    once (with wf_score attached), then call this function with each filtered
    subset to get apples-to-apples per-tier results.
    """
    if not candidates:
        return _empty_bt_result()

    # Sort by entry date for the chronological sim
    candidates = sorted(candidates, key=lambda t: (t["entry_date"] or pd.Timestamp.min))

    active_exits: list = []
    taken: list[dict] = []
    skipped = 0
    for trade in candidates:
        entry_dt = trade["entry_date"]
        if entry_dt is None:
            continue
        while active_exits and active_exits[0] <= entry_dt:
            heapq.heappop(active_exits)
        if len(active_exits) >= MAX_CONCURRENT_POSITIONS:
            skipped += 1
            continue
        heapq.heappush(active_exits, trade["exit_date"] or entry_dt)
        taken.append(trade)

    if not taken:
        return _empty_bt_result(skipped=skipped, candidates=len(candidates))

    # Build chronological equity curve
    by_exit = sorted(taken, key=lambda t: (t["exit_date"] or t["entry_date"]))
    equity      = 100.0
    max_equity  = 100.0
    max_dd      = 0.0
    eq_dates  = [taken[0]["entry_date"]]
    eq_values = [round(equity, 2)]
    dd_values = [0.0]
    for t in by_exit:
        equity *= (1 + (t["ret_pct"] / 100) * POSITION_SIZE_FRAC)
        eq_dates.append(t["exit_date"])
        eq_values.append(round(equity, 2))
        max_equity = max(max_equity, equity)
        dd = (equity - max_equity) / max_equity * 100
        dd_values.append(round(dd, 2))
        max_dd = min(max_dd, dd)

    bench_eq = benchmark.benchmark_equity(eq_dates, base=100.0, bench=bench) or []

    # Stats
    wins   = [t for t in taken if t["ret_pct"] > 0]
    losses = [t for t in taken if t["ret_pct"] <= 0]
    win_rate = round(len(wins) / len(taken) * 100, 1) if taken else 0
    avg_win  = round(sum(t["ret_pct"] for t in wins) / len(wins), 2) if wins else 0
    avg_loss = round(sum(t["ret_pct"] for t in losses) / len(losses), 2) if losses else 0
    expectancy = round((win_rate / 100 * avg_win) + ((1 - win_rate / 100) * avg_loss), 2)
    profit_factor = round(abs(sum(t["ret_pct"] for t in wins) /
                              sum(t["ret_pct"] for t in losses)), 2) if losses else 999
    avg_hold = round(sum(t["days"] for t in taken) / len(taken), 1)

    with_alpha = [t for t in taken if t.get("alpha") is not None]
    avg_alpha  = (round(sum(t["alpha"] for t in with_alpha) / len(with_alpha), 2)
                  if with_alpha else 0.0)
    pct_beat   = (round(sum(1 for t in with_alpha if t["alpha"] > 0) / len(with_alpha) * 100, 1)
                  if with_alpha else 0.0)

    avg_gross  = round(sum(t["gross_ret"] for t in taken) / len(taken), 2)
    avg_cost   = round(sum(t["cost_pct"]  for t in taken) / len(taken), 3)

    by_reason = {}
    for t in taken:
        by_reason[t["reason"]] = by_reason.get(t["reason"], 0) + 1

    final_bench  = round(float(bench_eq[-1]), 2) if bench_eq else None
    final_equity = round(eq_values[-1], 2)
    final_alpha  = (round(final_equity - final_bench, 2)
                    if final_bench is not None else None)

    # Alpha t-stat + IS/OOS split (Phase 3)
    def _alpha_stats(trades_subset: list[dict]) -> dict | None:
        if not trades_subset:
            return None
        a = [t["alpha"] for t in trades_subset if t.get("alpha") is not None]
        if not a:
            return None
        n  = len(a)
        mu = sum(a) / n
        var = sum((x - mu) ** 2 for x in a) / max(1, n - 1)
        sd  = var ** 0.5
        t_stat = (mu / (sd / (n ** 0.5))) if sd > 0 else 0.0
        return {
            "trades":         len(trades_subset),
            "avg_alpha":      round(mu, 2),
            "std_alpha":      round(sd, 2),
            "t_stat":         round(t_stat, 2),
            "win_rate":       round(sum(1 for t in trades_subset if t["ret_pct"] > 0) / len(trades_subset) * 100, 1),
            "pct_beat_bench": round(sum(1 for x in a if x > 0) / n * 100, 1),
        }

    alpha_t_stat = _alpha_stats(taken)["t_stat"] if _alpha_stats(taken) else 0.0
    cutoff       = int(len(taken) * 0.7)
    is_stats     = _alpha_stats(taken[:cutoff])
    oos_stats    = _alpha_stats(taken[cutoff:])
    overfit_gap  = (round(is_stats["avg_alpha"] - oos_stats["avg_alpha"], 2)
                    if (is_stats and oos_stats) else None)
    if is_stats and oos_stats:
        is_a, oos_a = is_stats["avg_alpha"], oos_stats["avg_alpha"]
        if is_a > 0 and oos_a >= is_a * 0.5:
            oos_verdict = "ROBUST — OOS holds up"
        elif is_a > 0 and oos_a > 0:
            oos_verdict = "DEGRADED — OOS positive but materially weaker"
        elif is_a > 0 and oos_a <= 0:
            oos_verdict = "OVERFIT — IS positive, OOS negative"
        else:
            oos_verdict = "FAILED IN-SAMPLE — strategy isn't even fitting"
    else:
        oos_verdict = "INSUFFICIENT DATA"

    return {
        "trades":             len(taken),
        "candidates":         len(candidates),
        "skipped_no_capital": skipped,
        "win_rate":           win_rate,
        "avg_win":            avg_win,
        "avg_loss":           avg_loss,
        "expectancy":         expectancy,
        "profit_factor":      profit_factor,
        "avg_hold":           avg_hold,
        "best":               max(t["ret_pct"] for t in taken),
        "worst":              min(t["ret_pct"] for t in taken),
        "by_reason":          by_reason,
        "avg_gross":          avg_gross,
        "avg_cost":           avg_cost,
        "avg_alpha":          avg_alpha,
        "pct_beat_bench":     pct_beat,
        "max_drawdown":       round(max_dd, 2),
        "final_equity":       final_equity,
        "final_bench":        final_bench,
        "final_alpha":        final_alpha,
        "alpha_t_stat":       round(alpha_t_stat, 2),
        "is_oos": {
            "is":            is_stats,
            "oos":           oos_stats,
            "overfit_gap":   overfit_gap,
            "verdict":       oos_verdict,
        },
        "equity_curve":       _downsample(eq_values, 80),
        "drawdown_curve":     _downsample(dd_values, 80),
        "bench_equity":       _downsample(bench_eq, 80),
    }


def backtest_signal(stocks: dict, signal_fn, hold_days: int = 20,
                    stop_pct: float = -7.0, target_pct: float = 25.0,
                    lookback_days: int = 800, max_signals: int = 5000,
                    nifty: pd.DataFrame | None = None,
                    regime_filter: bool = False,
                    bench: pd.Series | None = None,
                    shuffle: bool = True,
                    min_tier_score: float | None = None) -> dict:
    """
    Public API — backward-compatible thin wrapper over the 3-step pipeline:
      _generate_candidates → optional regime/tier filters → _run_sim_and_stats.

    Pass min_tier_score=X to keep only candidates where the point-in-time
    lightweight score ≥ X (e.g. 60 = B+ tier, 70 = A tier).
    """
    if bench is None:
        bench = _BENCH
    cands = _generate_candidates(
        stocks, signal_fn,
        hold_days=hold_days, stop_pct=stop_pct, target_pct=target_pct,
        lookback_days=lookback_days, max_signals=max_signals,
        bench=bench, compute_wf_score=(min_tier_score is not None),
        shuffle=shuffle,
    )
    if min_tier_score is not None:
        cands = [c for c in cands
                 if c.get("wf_score") is not None and c["wf_score"] >= min_tier_score]
    if regime_filter:
        cands = _apply_regime_filter(cands, nifty)
    return _run_sim_and_stats(cands, bench=bench)


# ── Built-in signal functions (for backtesting) ──────────────────────────────

def _signal_breakout_20d(df: pd.DataFrame, i: int) -> bool:
    """Bar i closes above prior 20-day high on volume > 1.5× avg."""
    if i < 50:
        return False
    close_i = float(df["Close"].iloc[i])
    high20  = float(df["High"].iloc[i-20:i].max())
    vol_i   = float(df["Volume"].iloc[i])
    vol_avg = float(df["Volume"].iloc[i-50:i].mean())
    return close_i > high20 and vol_i > vol_avg * 1.5

def _signal_rs_breakout(df: pd.DataFrame, i: int) -> bool:
    """3-month return > 15% AND closes at/near recent high."""
    if i < 80:
        return False
    close_i = float(df["Close"].iloc[i])
    # BUG-FIX: was `iloc[i-66]` (66 bars), now consistent r3m = 63 bars.
    r3m = (close_i / float(df["Close"].iloc[i-63]) - 1) * 100
    # BUG-FIX: was using Close for 200-bar high (sister _signal_breakout_20d uses High).
    # Intraday highs > closes → using Close gave a structurally easier filter
    # → false RS-breakout signals.
    hi = float(df["High"].iloc[max(0, i-200):i].max())
    return r3m > 15 and close_i >= hi * 0.97

def _signal_volume_accumulation(df: pd.DataFrame, i: int) -> bool:
    """Volume > 2× avg, closed in upper 60% of day range."""
    if i < 30:
        return False
    high_i  = float(df["High"].iloc[i])
    low_i   = float(df["Low"].iloc[i])
    close_i = float(df["Close"].iloc[i])
    vol_i   = float(df["Volume"].iloc[i])
    vol_avg = float(df["Volume"].iloc[i-20:i].mean())
    if high_i <= low_i or vol_avg <= 0:
        return False
    pos = (close_i - low_i) / (high_i - low_i)
    return vol_i > vol_avg * 2 and pos >= 0.6


# ── New signals (less textbook than 20d-high breakout) ──────────────────────
# These add conditions beyond "made a new high" — pullback structure, pocket
# pivot volume test, volatility contraction — so they fire less often but
# each firing has more confluence than the vanilla breakout.

def _signal_pivot_pullback(df: pd.DataFrame, i: int) -> bool:
    """
    Stage-2 stock pulls back from a recent high to MA50, then bounces.

    Conditions:
      - cur > MA50 > MA200 (stage 2 alignment)
      - Made a meaningful high in the LAST 30 bars (within 2% of 200-bar high)
      - Currently pulled back 3-12% from that recent high
      - Within 5% of MA50 (testing support)
      - Today: closes UP > 1% on volume ≥ 1.5× 20d avg (the bounce)
    """
    if i < 200:
        return False
    cur   = float(df["Close"].iloc[i])
    ma50  = float(df["Close"].iloc[i - 49:i + 1].mean())
    ma200 = float(df["Close"].iloc[i - 199:i + 1].mean())
    if not (cur > ma50 > ma200):
        return False
    recent_high = float(df["High"].iloc[i - 30:i - 5].max())
    older_high  = float(df["High"].iloc[max(0, i - 200):i - 30].max())
    if recent_high < older_high * 0.98:           # need near-200-bar high recently
        return False
    pullback_pct = (cur / recent_high - 1) * 100  # negative when below high
    if not (-12.0 <= pullback_pct <= -3.0):
        return False
    if abs(cur / ma50 - 1) > 0.05:                # within 5% of MA50
        return False
    prev = float(df["Close"].iloc[i - 1])
    if (cur / prev - 1) < 0.01:                    # today closes up > 1%
        return False
    vol_now = float(df["Volume"].iloc[i])
    vol_avg = float(df["Volume"].iloc[i - 20:i].mean())
    return vol_avg > 0 and vol_now >= vol_avg * 1.5


def _signal_pocket_pivot(df: pd.DataFrame, i: int) -> bool:
    """
    Mike Webster's Pocket Pivot:
      - Bar closes above the prior 10-day HIGH (institutional accumulation start)
      - Today's volume is BIGGER than the largest down-day volume in the past 10
        (i.e. demand has officially overtaken supply)
      - Stage 2 alignment + within 25% of 52W high (not picking falling knives)
    """
    if i < 200:
        return False
    cur   = float(df["Close"].iloc[i])
    ma50  = float(df["Close"].iloc[i - 49:i + 1].mean())
    ma200 = float(df["Close"].iloc[i - 199:i + 1].mean())
    if not (cur > ma50 > ma200):
        return False
    high10 = float(df["High"].iloc[i - 10:i].max())
    if cur < high10:
        return False
    high252 = float(df["High"].iloc[max(0, i - 252):i].max())
    if high252 <= 0 or cur < high252 * 0.75:
        return False
    last10_close = df["Close"].iloc[i - 10:i]
    last10_vol   = df["Volume"].iloc[i - 10:i]
    chg = last10_close.pct_change()
    down_vols = last10_vol[chg < 0]
    if down_vols.empty:
        return False
    biggest_down_vol = float(down_vols.max())
    vol_now = float(df["Volume"].iloc[i])
    return vol_now > biggest_down_vol


def _signal_coiled_spring(df: pd.DataFrame, i: int) -> bool:
    """
    Minervini-style Volatility Contraction Pattern (VCP) — compressed:
      - Stage 2 alignment
      - Last 15 bars: high-low spread < 8% of mean price (tight base)
      - 50-day high made within last 20 bars (strength preceded the contraction)
      - Today: breaks above 10-day high on volume ≥ 1.5× 20d avg (spring uncoils)
    """
    if i < 200:
        return False
    cur   = float(df["Close"].iloc[i])
    ma50  = float(df["Close"].iloc[i - 49:i + 1].mean())
    ma200 = float(df["Close"].iloc[i - 199:i + 1].mean())
    if not (cur > ma50 > ma200):
        return False
    last15 = df["Close"].iloc[i - 15:i]
    rng = (float(last15.max()) - float(last15.min())) / float(last15.mean())
    if rng > 0.08:                                  # not tight enough
        return False
    # 50-day high made recently? (within last 20 bars)
    high50_window = df["High"].iloc[i - 50:i]
    if high50_window.empty:
        return False
    high50_pos_local = int(high50_window.values.argmax())
    bars_since_high  = (len(high50_window) - 1) - high50_pos_local
    if bars_since_high > 20:
        return False
    high10 = float(df["High"].iloc[i - 10:i].max())
    if cur < high10:                                # break the contraction up
        return False
    vol_now = float(df["Volume"].iloc[i])
    vol_avg = float(df["Volume"].iloc[i - 20:i].mean())
    return vol_avg > 0 and vol_now >= vol_avg * 1.5


# ──────────────────────────────────────────────────────────────────────────────
# MAIN ENTRY — full edge engine run
# ──────────────────────────────────────────────────────────────────────────────

def run_edge_engine(progress_callback=None) -> dict:
    """
    Run the full edge engine: regime detection, master ranking, backtests.
    Cached 30 min. Zero NSE API calls.
    """
    if _cache["data"] and time.time() - _cache["ts"] < CACHE_TTL:
        return _cache["data"]

    # Load fundamentals (best-effort, optional)
    fundamentals_map = {}
    try:
        from fundamentals import load_all_fundamentals
        fundamentals_map = load_all_fundamentals()
    except Exception:
        pass

    if progress_callback:
        progress_callback(0, 100,
                          f"Loading bhavcopy ({BT_LOAD_DAYS}d, survivorship-free)…")
    # SURVIVORSHIP-FREE load: every symbol that ever traded in the window
    # (incl. delisted/dropped). Also captures real NIFTYBEES into _BENCH.
    stocks_all = _load_stocks(progress_callback, days=BT_LOAD_DAYS,
                              survivorship_free=True)
    if not stocks_all:
        return {"error": "No bhavcopy data available"}

    # For RANKING / scoring we use the current curated universe — the scoreboard
    # reflects the present, so it's fine (and faster) not to score delisted names.
    try:
        from nse_stocks import get_universe_symbols
        curated = set(get_universe_symbols())
    except Exception:
        curated = set()
    stocks = ({s: df for s, df in stocks_all.items() if s in curated}
              if curated else stocks_all)

    # Build Nifty proxy (still used by regime detection + RS — Phase 5 swaps it
    # for the real NIFTYBEES series everywhere; this is just Phase 1).
    if progress_callback:
        progress_callback(20, 100, "Building Nifty proxy index…")
    nifty = _build_nifty_proxy(stocks)

    # Market regime
    if progress_callback:
        progress_callback(30, 100, "Detecting market regime (D-Day / FTD)…")
    regime = detect_market_regime(nifty)

    # Sector quadrants (RRG)
    if progress_callback:
        progress_callback(40, 100, "Computing sector RRG quadrants…")
    sector_quad = _compute_sector_quadrants(stocks, nifty)

    # Setup Quality Score for every stock
    if progress_callback:
        progress_callback(50, 100, f"Scoring {len(stocks)} stocks…")
    ranked = []
    done = 0
    total = len(stocks)
    with ThreadPoolExecutor(max_workers=8) as ex:
        futs = {
            ex.submit(compute_setup_score, sym, df, regime, sector_quad,
                      fundamentals_map.get(sym), nifty): sym
            for sym, df in stocks.items()
        }
        # Hardening: a single bad stock must NOT kill the entire scan.
        # Capture per-symbol failures so the run completes + we can diagnose.
        score_failures: dict[str, str] = {}
        for fut in as_completed(futs):
            done += 1
            if progress_callback and done % 500 == 0:
                progress_callback(50 + int(done / total * 30), 100,
                                  f"Scoring… {done}/{total}")
            try:
                r = fut.result()
                if r is not None:
                    ranked.append(r)
            except Exception as e:
                sym = futs[fut]
                score_failures[sym] = f"{type(e).__name__}: {e}"
        if score_failures:
            # Log first few to stderr — full count surfaced in the response.
            import sys
            n = len(score_failures)
            sample = list(score_failures.items())[:5]
            print(f"[run_edge_engine] {n} score failures (sample):", file=sys.stderr)
            for s, err in sample:
                print(f"  {s}: {err}", file=sys.stderr)

    ranked.sort(key=lambda x: -x["score"])

    # Backtests run on the SURVIVORSHIP-FREE universe with the REAL NIFTYBEES
    # benchmark. Numbers are now: net of costs, capital-capped, alpha vs Nifty.
    if progress_callback:
        progress_callback(85, 100,
                          f"Backtesting + tier-conditional on {len(stocks_all)} symbols…")
    backtests = {}
    SUMMARY_KEYS = ["trades", "candidates", "skipped_no_capital",
                    "win_rate", "avg_alpha", "pct_beat_bench",
                    "final_equity", "final_bench", "final_alpha",
                    "max_drawdown", "alpha_t_stat", "expectancy"]
    def _summary(bt: dict) -> dict:
        return {k: bt.get(k) for k in SUMMARY_KEYS}

    for name, fn in (("Breakout (20d high + vol)",  _signal_breakout_20d),
                     ("RS Breakout (52w + r3m)",    _signal_rs_breakout),
                     ("Volume Accumulation",        _signal_volume_accumulation),
                     # New non-textbook signals — multi-condition confluence
                     ("Pivot Pullback (MA50 bounce)", _signal_pivot_pullback),
                     ("Pocket Pivot (Webster)",       _signal_pocket_pivot),
                     ("Coiled Spring (VCP)",          _signal_coiled_spring)):
        try:
            # ONE candidate generation pass with walk-forward scores attached.
            # Every variant below (all, B+, A, regime-filtered) just re-runs the
            # cheap sim on a filtered subset — no re-walking 2700 symbols × 800 bars.
            cands = _generate_candidates(
                stocks_all, fn,
                hold_days=20, stop_pct=-7, target_pct=20,
                lookback_days=BT_LOOKBACK_BARS, max_signals=5000,
                bench=_BENCH, compute_wf_score=True,
            )

            bt_all = _run_sim_and_stats(cands, bench=_BENCH)
            bt_b   = _run_sim_and_stats(
                [c for c in cands if (c.get("wf_score") or 0) >= 60],
                bench=_BENCH,
            )
            bt_a   = _run_sim_and_stats(
                [c for c in cands if (c.get("wf_score") or 0) >= 70],
                bench=_BENCH,
            )
            bt_reg = _run_sim_and_stats(_apply_regime_filter(cands, nifty),
                                         bench=_BENCH)

            # Tier-conditional summary: does score-filtering improve alpha?
            bt_all["by_tier"] = {
                "All":              _summary(bt_all),
                "B+ (score ≥ 60)":  _summary(bt_b),
                "A  (score ≥ 70)":  _summary(bt_a),
            }
            bt_all["regime_filtered"] = bt_reg
            backtests[name] = bt_all
        except Exception as e:
            backtests[name] = {"error": str(e), "trades": 0}

    # Phase 2: walk-forward score validation (does Setup Score predict alpha?)
    if progress_callback:
        progress_callback(92, 100, "Validating Setup Score (walk-forward IC + quintiles)…")
    try:
        score_validation = validate_setup_score(stocks, _BENCH,
                                                 snapshots=12, fwd_days=20)
    except Exception as e:
        score_validation = {"error": str(e)}

    # Phase 2b: per-component IC analysis — which raw features actually predict
    # alpha? Evidence for redesigning the composite score's weights.
    if progress_callback:
        progress_callback(95, 100, "Analyzing score components (per-feature IC)…")
    try:
        score_components = analyze_score_components(stocks, _BENCH,
                                                     snapshots=8, fwd_days=20)
    except Exception as e:
        score_components = {"error": str(e)}

    # Tier counts
    tiers = {"A+": 0, "A": 0, "B": 0, "C": 0}
    for r in ranked:
        if r["score"] >= 80:   tiers["A+"] += 1
        elif r["score"] >= 70: tiers["A"]  += 1
        elif r["score"] >= 60: tiers["B"]  += 1
        elif r["score"] >= 50: tiers["C"]  += 1

    # F9 — Sector concentration risk in top-50 ranked stocks
    from collections import Counter
    top50_groups = [r["group"] for r in ranked[:50] if r.get("group")]
    sector_cnt   = Counter(top50_groups)
    total_top50  = len(top50_groups) or 1
    sector_concentration = [
        {"sector": k, "count": v, "pct": round(v / total_top50 * 100, 1)}
        for k, v in sector_cnt.most_common(10)
    ]

    out = {
        "regime":               regime,
        "ranked":               ranked[:200],   # top 200 only
        "tier_counts":          tiers,
        "total_scored":         len(ranked),
        "backtests":            backtests,
        "sector_quad":          sector_quad,
        "sector_concentration": sector_concentration,   # F9
        "computed_at":          int(time.time()),
        "fundamentals_available": bool(fundamentals_map),
        "fundamentals_count":     len(fundamentals_map),
        # Foundation tags — so the UI / users can SEE what's running
        "bench_symbol":         benchmark.BENCH_SYMBOL,
        "bench_bars":           int(len(_BENCH)) if _BENCH is not None else 0,
        "universe_total":       len(stocks_all),       # survivorship-free count
        "universe_curated":     len(stocks),           # current Nifty500-style universe
        "bt_load_days":         BT_LOAD_DAYS,
        "bt_lookback_bars":     BT_LOOKBACK_BARS,
        "max_concurrent_positions": MAX_CONCURRENT_POSITIONS,
        # Phase 2 — score validation: IC + per-quintile forward alpha
        "score_validation":     score_validation,
        # Phase 2b — per-component IC: evidence for score redesign
        "score_components":     score_components,
        # Hardening: any per-symbol scoring failures (was: 1 bad stock killed scan)
        "score_failures":       score_failures,
        "score_failure_count":  len(score_failures),
    }
    _cache["data"] = out
    _cache["ts"]   = time.time()

    if progress_callback:
        progress_callback(100, 100,
                          f"Done — {tiers['A+']} A+ · {tiers['A']} A · {tiers['B']} B candidates")
    return out


# ──────────────────────────────────────────────────────────────────────────────
# PHASE 2 — Walk-forward score validation
# Does a higher Setup Quality Score actually predict higher forward alpha?
# This is the ONLY way to know if the score is real signal or backward-fitted noise.
# ──────────────────────────────────────────────────────────────────────────────

def validate_setup_score(stocks: dict, bench: pd.Series | None,
                          snapshots: int = 12, fwd_days: int = 20) -> dict:
    """
    Walk-forward Information Coefficient (IC) + quintile bucketing.

    For each of `snapshots` historical dates (point-in-time, no look-ahead):
      1. Score every stock using ONLY data through that date (lightweight score).
      2. Look up the FORWARD fwd_days net return for each stock.
      3. Subtract the forward benchmark return → forward ALPHA.
      4. Compute Spearman rank correlation between score and forward alpha (per-snapshot IC).
    Aggregate across snapshots:
      - Mean IC + t-statistic (does the score have ANY predictive power?)
      - Quintile bucketing: avg alpha per score-quintile (top-vs-bot spread = "edge")

    A useful score: mean IC > 0.03, t-stat > 2, monotonic quintile spread.
    """
    if not stocks:
        return {"snapshots_used": 0, "error": "no stocks"}

    sample = max(stocks.values(), key=lambda d: len(d))
    if len(sample) < 400:
        return {"snapshots_used": 0, "error": "insufficient history"}

    # Window: pick the last ~600 bars, leave fwd_days buffer at the end
    end_idx   = len(sample) - fwd_days - 5
    start_idx = max(200, end_idx - 600)
    step      = max(1, (end_idx - start_idx) // snapshots)
    snap_idxs = list(range(start_idx, end_idx, step))[:snapshots]
    snap_dates = [sample.index[i] for i in snap_idxs]

    per_snapshot_ic: list[float] = []
    all_records: list[dict] = []

    for snap_dt in snap_dates:
        records = []
        for sym, df in stocks.items():
            if _is_etf(sym):
                continue
            sub = df[df.index <= snap_dt]
            if len(sub) < 130:
                continue
            close = sub["Close"].dropna()
            vol   = sub["Volume"].dropna()
            if len(close) < 252 or len(vol) < 50:
                continue
            score = _lightweight_score(close, vol, bench, df_sub=sub)
            if score is None:
                continue
            future = df[df.index > snap_dt].head(fwd_days + 1)
            if len(future) < fwd_days:
                continue
            try:
                entry_p = float(future["Open"].iloc[0])
                exit_p  = float(future["Close"].iloc[-1])
                if entry_p <= 0:
                    continue
                fwd_ret = (exit_p / entry_p - 1) * 100
            except Exception:
                continue
            bret = benchmark.benchmark_return(future.index[0], future.index[-1], bench=bench)
            if bret is None:
                continue
            records.append({"score": score, "fwd_ret": fwd_ret, "alpha": fwd_ret - bret})
        if len(records) >= 20:
            ic = _spearman_ic([r["score"] for r in records],
                              [r["alpha"] for r in records])
            if ic is not None:
                per_snapshot_ic.append(ic)
            all_records.extend(records)

    if not all_records:
        return {"snapshots_used": 0, "error": "no scoreable observations"}

    # Quintile bucketing across ALL (snapshot, stock) observations
    sorted_recs = sorted(all_records, key=lambda r: r["score"])
    n = len(sorted_recs)
    qsize = n // 5
    quintiles = []
    for q in range(5):
        s, e = q * qsize, ((q + 1) * qsize) if q < 4 else n
        bucket = sorted_recs[s:e]
        if not bucket:
            continue
        quintiles.append({
            "q":           q + 1,
            "n":           len(bucket),
            "avg_score":   round(sum(r["score"] for r in bucket) / len(bucket), 1),
            "avg_alpha":   round(sum(r["alpha"] for r in bucket) / len(bucket), 2),
            "avg_fwd_ret": round(sum(r["fwd_ret"] for r in bucket) / len(bucket), 2),
            "win_rate":    round(sum(1 for r in bucket if r["alpha"] > 0) / len(bucket) * 100, 1),
            "best_alpha":  round(max(r["alpha"] for r in bucket), 2),
            "worst_alpha": round(min(r["alpha"] for r in bucket), 2),
        })

    # IC mean + t-stat across snapshots
    if per_snapshot_ic:
        mu = sum(per_snapshot_ic) / len(per_snapshot_ic)
        if len(per_snapshot_ic) > 1:
            var = sum((x - mu) ** 2 for x in per_snapshot_ic) / (len(per_snapshot_ic) - 1)
            sd  = var ** 0.5
            ic_t = round(mu / (sd / (len(per_snapshot_ic) ** 0.5)), 2) if sd > 0 else 0.0
        else:
            ic_t = 0.0
        ic_mean = round(mu, 3)
    else:
        ic_mean = 0.0
        ic_t    = 0.0

    spread = (quintiles[-1]["avg_alpha"] - quintiles[0]["avg_alpha"]) if len(quintiles) >= 2 else None
    # Monotonicity test: are quintile alphas strictly increasing?
    monotonic = all(
        quintiles[i]["avg_alpha"] <= quintiles[i + 1]["avg_alpha"]
        for i in range(len(quintiles) - 1)
    ) if len(quintiles) >= 2 else False

    return {
        "snapshots_used":  len(per_snapshot_ic),
        "snapshots_total": len(snap_dates),
        "total_obs":       len(all_records),
        "fwd_days":        fwd_days,
        "ic_mean":         ic_mean,
        "ic_t_stat":       ic_t,
        "ic_per_snapshot": [round(x, 3) for x in per_snapshot_ic],
        "quintiles":       quintiles,
        "top_minus_bot":   round(spread, 2) if spread is not None else None,
        "monotonic":       monotonic,
        # Verdict — surfaced to the UI so users don't have to read the numbers
        "verdict":         _score_verdict(ic_mean, ic_t, spread, monotonic),
    }


def _all_features(close: pd.Series, vol: pd.Series, df_sub: pd.DataFrame | None,
                   bench: pd.Series | None) -> dict | None:
    """
    Extract every candidate predictive feature for IC analysis.
    All values are walk-forward-safe — they only use data up to close.index[-1].
    Returns None when the series is too short for reliable computation.
    """
    if len(close) < 200 or len(vol) < 50:
        return None
    cur = float(close.iloc[-1])
    if cur < 30:
        return None

    r1m  = (cur / float(close.iloc[-21])  - 1) * 100 if len(close) >= 22  else None
    r3m  = (cur / float(close.iloc[-63])  - 1) * 100
    r6m  = (cur / float(close.iloc[-126]) - 1) * 100
    r12m = (cur / float(close.iloc[-252]) - 1) * 100 if len(close) >= 253 else None

    ma50  = float(close.rolling(50).mean().iloc[-1])
    ma200 = float(close.rolling(200).mean().iloc[-1])
    ma_align  = 2 if (cur > ma50 > ma200) else 1 if cur > ma50 else 0
    dist_ma50 = (cur / ma50 - 1) * 100 if ma50 > 0 else 0.0

    high200 = float(close.iloc[-200:].max())
    pct_from_high = (cur / high200 - 1) * 100   # 0 = at high, negative = below

    vol_r10 = float(vol.iloc[-10:].mean())
    vol_a50 = float(vol.iloc[-50:].mean())
    vol_ratio = vol_r10 / vol_a50 if vol_a50 > 0 else 1.0

    atr_pct = None
    if df_sub is not None and "High" in df_sub.columns and "Low" in df_sub.columns:
        try:
            hi = df_sub["High"].dropna()
            lo = df_sub["Low"].dropna()
            cl = df_sub["Close"].dropna()
            if len(hi) >= 15:
                _tr = pd.concat([
                    hi - lo,
                    (hi - cl.shift(1)).abs(),
                    (lo - cl.shift(1)).abs(),
                ], axis=1).max(axis=1)
                atr14 = float(_tr.ewm(alpha=1/14, adjust=False).mean().iloc[-1])
                atr_pct = atr14 / cur * 100 if cur > 0 else None
        except Exception:
            pass

    rs_3m = None
    if bench is not None and len(bench) >= 100:
        try:
            baligned = bench.reindex(close.index, method="ffill").dropna()
            bsub = baligned[baligned.index <= close.index[-1]]
            if len(bsub) >= 63:
                b3m = (float(bsub.iloc[-1]) / float(bsub.iloc[-63]) - 1) * 100
                rs_3m = r3m - b3m
        except Exception:
            pass

    deliv_avg20 = None
    if df_sub is not None and "DelivPer" in df_sub.columns:
        d = df_sub["DelivPer"].dropna()
        if len(d) >= 5:
            deliv_avg20 = float(d.iloc[-20:].mean())

    return {
        "r1m":           r1m,
        "r3m":           r3m,
        "r6m":           r6m,
        "r12m":          r12m,
        "ma_align":      ma_align,
        "dist_ma50":     dist_ma50,
        "pct_from_high": pct_from_high,
        "vol_ratio":     vol_ratio,
        "atr_pct":       atr_pct,
        "rs_3m":         rs_3m,
        "deliv_avg20":   deliv_avg20,
    }


def _ic_verdict(ic: float, t_stat: float) -> str:
    """Plain-English label for a single feature's predictive power."""
    if abs(ic) >= 0.05 and abs(t_stat) >= 2.0:
        return "STRONG"
    if abs(ic) >= 0.03 and abs(t_stat) >= 1.5:
        return "MODERATE"
    if abs(ic) >= 0.02:
        return "WEAK"
    return "NONE"


def _redesign_verdict(components: list[dict], best_pair: dict | None) -> str:
    """Overall recommendation for what to do with the score."""
    strong   = [c for c in components if c["verdict"] == "STRONG"]
    moderate = [c for c in components if c["verdict"] == "MODERATE"]
    if strong:
        names = ", ".join(c["name"] for c in strong[:3])
        return f"REDESIGN: heavily weight {names}; drop the rest"
    if moderate:
        names = ", ".join(c["name"] for c in moderate[:3])
        return f"PARTIAL FIX: lean into {names}; current weights mis-allocated"
    if best_pair and abs(best_pair["ic"]) >= 0.05:
        return f"PAIR HELPS: {best_pair['f1']} + {best_pair['f2']} together → IC {best_pair['ic']}"
    return "NO SIGNAL — none of these features predict alpha here; investigate inputs"


def analyze_score_components(stocks: dict, bench: pd.Series | None,
                              snapshots: int = 8, fwd_days: int = 20) -> dict:
    """
    Per-feature Information Coefficient: which raw inputs ACTUALLY predict
    forward alpha? This is the evidence base for redesigning the composite
    Setup Quality Score — instead of hand-tuned weights, we measure each
    feature's IC against forward alpha and learn what's signal vs noise.

    Output:
      components: ranked list of features with IC, t-stat, n_obs, verdict
      best_pair: top 2-feature combo (rank-normalised + averaged)
      verdict: plain-English recommendation
    """
    if not stocks:
        return {"error": "no stocks"}
    sample = max(stocks.values(), key=lambda d: len(d))
    if len(sample) < 400:
        return {"error": "insufficient history"}

    end_idx   = len(sample) - fwd_days - 5
    start_idx = max(200, end_idx - 600)
    step      = max(1, (end_idx - start_idx) // snapshots)
    snap_idxs = list(range(start_idx, end_idx, step))[:snapshots]
    snap_dates = [sample.index[i] for i in snap_idxs]

    all_obs: list[dict] = []
    for snap_dt in snap_dates:
        for sym, df in stocks.items():
            if _is_etf(sym):
                continue
            sub = df[df.index <= snap_dt]
            if len(sub) < 250:
                continue
            close = sub["Close"].dropna()
            vol   = sub["Volume"].dropna()
            features = _all_features(close, vol, sub, bench)
            if features is None:
                continue
            future = df[df.index > snap_dt].head(fwd_days + 1)
            if len(future) < fwd_days:
                continue
            try:
                entry_p = float(future["Open"].iloc[0])
                exit_p  = float(future["Close"].iloc[-1])
                if entry_p <= 0:
                    continue
                fwd_ret = (exit_p / entry_p - 1) * 100
            except Exception:
                continue
            bret = benchmark.benchmark_return(future.index[0], future.index[-1], bench=bench)
            if bret is None:
                continue
            features["_alpha"] = fwd_ret - bret
            all_obs.append(features)

    if len(all_obs) < 100:
        return {"error": f"only {len(all_obs)} observations"}

    feature_names = [k for k in all_obs[0].keys() if k != "_alpha"]
    component_ic: list[dict] = []
    for fname in feature_names:
        paired = [(o[fname], o["_alpha"]) for o in all_obs
                  if o.get(fname) is not None and not pd.isna(o.get(fname))]
        if len(paired) < 50:
            continue
        xs = [p[0] for p in paired]
        ys = [p[1] for p in paired]
        ic = _spearman_ic(xs, ys)
        if ic is None:
            continue
        # Bootstrap t-stat by chunking
        if len(xs) >= 100:
            chunks   = 10
            csize    = len(xs) // chunks
            chunk_ics: list[float] = []
            for c in range(chunks):
                s_, e_ = c * csize, (c + 1) * csize
                ci = _spearman_ic(xs[s_:e_], ys[s_:e_])
                if ci is not None:
                    chunk_ics.append(ci)
            if len(chunk_ics) > 1:
                mu  = sum(chunk_ics) / len(chunk_ics)
                var = sum((x - mu) ** 2 for x in chunk_ics) / (len(chunk_ics) - 1)
                sd  = var ** 0.5
                t_stat = (mu / (sd / (len(chunk_ics) ** 0.5))) if sd > 0 else 0.0
            else:
                t_stat = 0.0
        else:
            t_stat = 0.0
        component_ic.append({
            "name":    fname,
            "ic":      round(ic, 3),
            "t_stat":  round(t_stat, 2),
            "n":       len(paired),
            "verdict": _ic_verdict(ic, t_stat),
        })

    component_ic.sort(key=lambda x: -abs(x["ic"]))

    # Top-6 pairwise search: rank-normalise both features, average, then IC
    top = component_ic[:6]
    best_pair = None
    for i, f1 in enumerate(top):
        for j, f2 in enumerate(top):
            if j <= i:
                continue
            paired = [(o[f1["name"]], o[f2["name"]], o["_alpha"])
                      for o in all_obs
                      if o.get(f1["name"]) is not None and o.get(f2["name"]) is not None]
            if len(paired) < 100:
                continue
            x1 = [p[0] for p in paired]
            x2 = [p[1] for p in paired]
            ys = [p[2] for p in paired]
            r1 = pd.Series(x1).rank(pct=True).values
            r2 = pd.Series(x2).rank(pct=True).values
            combo = [float(a + b) for a, b in zip(r1, r2)]
            ic = _spearman_ic(combo, ys)
            if ic is None:
                continue
            if best_pair is None or abs(ic) > abs(best_pair["ic"]):
                best_pair = {"f1": f1["name"], "f2": f2["name"],
                              "ic": round(ic, 3), "n": len(paired)}

    return {
        "n_obs":      len(all_obs),
        "snapshots":  len(snap_dates),
        "fwd_days":   fwd_days,
        "components": component_ic,
        "best_pair":  best_pair,
        "verdict":    _redesign_verdict(component_ic, best_pair),
    }


def _score_verdict(ic_mean, ic_t_stat, spread, monotonic) -> str:
    """One-line plain-English verdict on whether the score predicts alpha."""
    if spread is None:
        return "INSUFFICIENT DATA"
    if ic_mean >= 0.05 and ic_t_stat >= 2.0 and spread >= 1.0:
        return "STRONG — score reliably predicts forward alpha"
    if ic_mean >= 0.03 and ic_t_stat >= 1.5 and spread > 0:
        return "MODERATE — some predictive power, refine the score"
    if ic_mean > 0 and spread > 0:
        return "WEAK — score barely better than random"
    return "NONE — score does NOT predict alpha (consider redesign)"


def _compute_sector_quadrants(stocks: dict, nifty_df: pd.DataFrame | None) -> dict[str, str]:
    """RRG quadrant per industry group (reused from industry_groups logic)."""
    if nifty_df is None or "Close" not in nifty_df:
        return {}
    nifty_px = nifty_df["Close"]
    if not isinstance(nifty_px.index, pd.DatetimeIndex):
        nifty_px.index = pd.to_datetime(nifty_px.index)

    out = {}
    for grp, syms in INDUSTRY_GROUPS.items():
        closes = []
        for s in syms:
            if s in stocks and len(stocks[s]) >= 80:
                c = stocks[s]["Close"].dropna()
                if not isinstance(c.index, pd.DatetimeIndex):
                    c.index = pd.to_datetime(c.index)
                closes.append(c)
        if len(closes) < 2:
            continue
        try:
            grp_idx = equal_weight_index(pd.concat(closes, axis=1).dropna(how="all"))
            common  = grp_idx.index.intersection(nifty_px.index)
            if len(common) < 80:
                continue
            rs_d = grp_idx[common] / nifty_px[common]
            # BUG-013 consistency: use W-FRI (NSE week ends Friday), min_periods=20
            rs_w = rs_d.resample("W-FRI").last().dropna()
            if len(rs_w) < 12:
                continue
            rm     = rs_w.rolling(26, min_periods=20).mean()
            rratio = (rs_w / rm * 100).dropna()
            if len(rratio) < 6:
                continue
            rmom = (rratio / rratio.shift(4) * 100).dropna()
            if len(rmom) < 1:
                continue
            x = float(rratio.iloc[-1]); y = float(rmom.iloc[-1])
            out[grp] = (
                "Leading"   if x >= 100 and y >= 100 else
                "Weakening" if x >= 100 and y <  100 else
                "Improving" if x <  100 and y >= 100 else
                "Lagging"
            )
        except Exception:
            continue
    return out
