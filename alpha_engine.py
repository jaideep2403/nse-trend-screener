"""
Alpha Engine — Institutional Multi-Factor Composite Scorer
──────────────────────────────────────────────────────────
LOCAL-ONLY module. Gitignored — never pushed to GitHub.

Scores every stock 0-100 across 6 factor dimensions replicating
the actual approach used by institutional multi-factor funds:

  Factor              Max   What it measures
  ──────────────────  ────  ────────────────────────────────────────────
  1. Quality          20    ROE, D/E, EPS growth, Sales growth
  2. Momentum         20    6M/12M rank, RS line new high, 52W proximity
  3. Smart Money      25    Delivery%, Bulk deals, F&O OI, FII signal
  4. Earnings Quality 15    EPS acceleration, Promoter holding, fresh data
  5. Technical        10    Stage 2, MA stack, ADX, trend consistency
  6. Risk / Reward    10    Max drawdown, ADTV, beta-adjusted extension

  Total max: 100

Tiers:
  ≥ 80  → 🏆 BUY    (Institutional Grade — act now)
  65-79 → 💎 STRONG  (High conviction watch)
  50-64 → 👀 MONITOR (Average — needs improvement)
  < 50  → ⛔ AVOID   (Skip)

Exit Flags (smart-money exit signals):
  DELIV_DROP  — Delivery % collapsed on recent sessions (distribution)
  BULK_SELL   — Institutional sell in bulk/block deals (last 14 days)
  OI_SHORT    — F&O short buildup (OI up + price falling)
  OI_UNWIND   — Long unwinding (OI down + price falling)
  DIST_DAYS   — 4+ distribution days in last 20 sessions
  RS_WEAK     — Stock underperforming Nifty badly (6M excess < -5%)
  MA50_BREAK  — Price below MA50

Data sources (all local/cached, zero live scraping per scan):
  - NSE bhavcopy OHLCV (via industry_groups._get_stocks)
  - fundamentals.db (ROE, D/E, EPS, Sales, Promoter, EPS quarters)
  - bulk_deals.py (NSE bulk/block deals archive)
  - fo_data.py (NSE F&O bhavcopy OI signals)
  - Delivery % — loaded directly from bhavcopy per-day CSVs
"""
from __future__ import annotations

import time
import pickle
import os
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

import numpy as np
import pandas as pd

from industry_groups import _get_stocks, _build_nifty, INDUSTRY_GROUPS
from analysis_utils import stage_analysis, cross_sectional_rs_rank, sector_adjusted_rs
from fundamentals import load_all_fundamentals
from bulk_deals import run_bulk_deals
# BUG-FIX: was used at line ~947 but never imported. NameError swallowed by bare
# `except Exception` → F&O OI signal (5pts of Smart Money) was ALWAYS empty.
# "OI:LongBuild ✓" badges have never appeared in the UI since this file was written.
try:
    from fo_data import get_fo_signals
except Exception:
    def get_fo_signals():
        return {}

# Cache
CACHE_TTL = 1800   # 30 min
_cache: dict = {"data": None, "ts": 0.0}

# Industry group membership lookup
_SYM_TO_GROUP: dict[str, str] = {}
for _grp, _syms in INDUSTRY_GROUPS.items():
    for _s in _syms:
        _SYM_TO_GROUP[_s] = _grp

MIN_ADTV_CR = 1.0   # ₹1Cr daily turnover floor — institutional minimum
MIN_BARS    = 60


# ── Helpers ────────────────────────────────────────────────────────────────────

def _safe(v, default=0.0):
    """Cast numpy scalar → Python float; replace NaN/None with default."""
    if v is None:
        return default
    try:
        f = float(v)
        return default if (np.isnan(f) or np.isinf(f)) else f
    except Exception:
        return default


def _adx(df: pd.DataFrame, period: int = 14) -> float:
    """Wilder-smoothed ADX."""
    try:
        hi = df["High"].dropna()
        lo = df["Low"].dropna()
        cl = df["Close"].dropna()
        idx = hi.index.intersection(lo.index).intersection(cl.index)
        if len(idx) < period * 3:
            return 0.0
        h = hi[idx]; l = lo[idx]; c = cl[idx]
        prev_c = c.shift(1)
        tr  = pd.concat([h - l, (h - prev_c).abs(), (l - prev_c).abs()], axis=1).max(axis=1)
        up_move = h - h.shift(1)
        dn_move = l.shift(1) - l
        # BUG-FIX: Wilder's +DM/-DM rule. Previous code mutated `pdm` on line 1
        # then compared the MUTATED `pdm` against `ndm` on line 2 → zeroed BOTH
        # whenever up_move == dn_move (low-volatility tied days). Use raw moves
        # for both comparisons via separate temporaries.
        raw_pdm = up_move.clip(lower=0)
        raw_ndm = dn_move.clip(lower=0)
        pdm = raw_pdm.where((raw_pdm > raw_ndm), 0.0)
        ndm = raw_ndm.where((raw_ndm > raw_pdm), 0.0)
        # Wilder smoothing
        def _wilder(s, p):
            s = s.dropna()
            out = [s.iloc[:p].mean()]
            for v in s.iloc[p:]:
                out.append(out[-1] * (p - 1) / p + v / p)
            return pd.Series(out)
        atr14  = _wilder(tr,  period)
        pdi14  = _wilder(pdm, period) / atr14 * 100
        ndi14  = _wilder(ndm, period) / atr14 * 100
        dx     = ((pdi14 - ndi14).abs() / (pdi14 + ndi14).clip(lower=0.001)) * 100
        adx_v  = _wilder(dx.dropna(), period)
        return _safe(adx_v.iloc[-1]) if len(adx_v) else 0.0
    except Exception:
        return 0.0


def _max_drawdown(c: pd.Series, n: int = 132) -> float:
    """Worst peak-to-trough drawdown % over last n bars."""
    if len(c) < 30:
        return 100.0
    s = c.iloc[-n:]
    dd = (s - s.cummax()) / s.cummax()
    return _safe(abs(dd.min()) * 100)


def _acc_dist_days(close: pd.Series, vol: pd.Series, n: int = 20) -> tuple[int, int]:
    """Accumulation / distribution day counts over last n sessions.
    BUG-FIX: now compares each bar to its OWN trailing 20-day avg (not today's),
    so the count is unbiased even when avg volume is trending up over the window.
    """
    try:
        if len(close) < n + 20 or len(vol) < n + 20:
            return 0, 0
        avg_vol_series = vol.rolling(20).mean()
        cl = close.iloc[-n:]; v = vol.iloc[-n:]; chg = cl.diff()
        ref_avg = avg_vol_series.iloc[-n:]
        hv = v > ref_avg * 1.3
        return int(((chg > 0) & hv).sum()), int(((chg < 0) & hv).sum())
    except Exception:
        return 0, 0


def _rs_vs_nifty_6m(c: pd.Series, nifty: pd.Series) -> Optional[float]:
    """6-month excess return of stock vs Nifty composite (%)."""
    try:
        if nifty is None or len(nifty) < 126 or len(c) < 126:
            return None
        # Align on shared dates
        combined = pd.concat([c.rename("stk"), nifty.rename("nfy")], axis=1).dropna()
        if len(combined) < 126:
            return None
        s = float(combined["stk"].iloc[-1] / combined["stk"].iloc[-126] - 1) * 100
        n = float(combined["nfy"].iloc[-1] / combined["nfy"].iloc[-126] - 1) * 100
        return round(s - n, 2)
    except Exception:
        return None


def _rs_line_new_high(c: pd.Series, nifty: pd.Series) -> bool:
    """True if RS line (price / nifty) is at a new 52-week high."""
    try:
        if nifty is None or len(nifty) < 252 or len(c) < 252:
            return False
        combined = pd.concat([c.rename("s"), nifty.rename("n")], axis=1).dropna()
        if len(combined) < 252:
            return False
        rs = combined["s"] / combined["n"]
        return bool(rs.iloc[-1] >= rs.iloc[-252:-1].max() * 0.9995)
    except Exception:
        return False


_stage = stage_analysis   # TIER-3: use canonical analysis_utils implementation


def _strip_splits(df: pd.DataFrame) -> pd.DataFrame:
    """Delegate to canonical analysis_utils.adjust_for_splits which catches
    3:2 / 4:3 / 5:4 bonuses that the old `< 0.55` threshold here missed."""
    if df is None or df.empty or len(df) < 2:
        return df
    from analysis_utils import adjust_for_splits
    return adjust_for_splits(df)


# ── Delivery % loader (with batched cache) ───────────────────────────────────

# BUG-FIX: prior implementation opened each bhavcopy pkl PER SYMBOL — for ~1500
# stocks × 30 days = 45,000+ disk operations per scan; scans took 5-30 min.
# New design: load each pkl ONCE into _DELIVERY_CACHE indexed by date, then
# look up the symbol's row by date. ~30 disk reads total per scan.
_DELIVERY_CACHE: dict = {"data": None, "ts": 0.0, "n_days_loaded": 0}
_DELIV_TTL = 1800   # 30 min — matches scan cache

def _ensure_delivery_cache(n_days: int = 30) -> dict:
    """Build per-date {symbol: delivery_pct} index once per scan."""
    import time as _time
    if (_DELIVERY_CACHE["data"] is not None
            and _DELIVERY_CACHE["n_days_loaded"] >= n_days
            and _time.time() - _DELIVERY_CACHE["ts"] < _DELIV_TTL):
        return _DELIVERY_CACHE["data"]
    from data_fetcher import _weekdays_back, _bhav_cache_path
    import pickle as _pk
    by_date: dict = {}    # {date: {symbol: deliv_pct}}
    dates = _weekdays_back(n_days * 2)
    loaded = 0
    for dt in dates:
        cp = _bhav_cache_path(dt)
        if not cp.exists():
            continue
        try:
            with open(cp, "rb") as f:
                day_df = _pk.load(f)
            if "DelivPer" not in day_df.columns:
                continue
            # Build sym → deliv lookup for this date in one shot
            d = dict(zip(day_df["Symbol"].astype(str).tolist(),
                         pd.to_numeric(day_df["DelivPer"], errors="coerce").tolist()))
            by_date[dt] = {k: float(v) for k, v in d.items() if not pd.isna(v)}
            loaded += 1
            if loaded >= n_days:
                break
        except Exception:
            continue
    _DELIVERY_CACHE["data"] = by_date
    _DELIVERY_CACHE["ts"] = _time.time()
    _DELIVERY_CACHE["n_days_loaded"] = loaded
    return by_date


def _load_delivery_series(symbol: str, n_days: int = 30) -> Optional[pd.Series]:
    """Load delivery % series — uses batched cache (1 pkl read per date, not per stock)."""
    try:
        by_date = _ensure_delivery_cache(n_days)
        records: list[tuple] = []
        for dt, sym_map in by_date.items():
            v = sym_map.get(symbol)
            if v is not None:
                records.append((dt, v))
        if len(records) < 5:
            return None
        records.sort(key=lambda x: x[0])
        return pd.Series(
            [r[1] for r in records],
            index=pd.DatetimeIndex([pd.Timestamp(r[0]) for r in records])
        )
    except Exception:
        return None


# ── Bulk deal parser ──────────────────────────────────────────────────────────

# Identify institutional clients by name keywords.
# NSE bulk deals list actual entity names — MFs appear as:
# "AXIS BLUECHIP FUND", "SBI MAGNUM MIDCAP FUND", "NIPPON INDIA LARGE CAP FUND"
# FIIs appear as: "GOLDMAN SACHS (SINGAPORE) PTE", "VANGUARD EMERGING MARKETS..."
# Proprietary trading desks: "MOTILAL OSWAL SECURITIES LTD"
_INST_KEYWORDS = [
    # Primary institutional markers — these ONLY appear in fund/institution names
    " FUND", " MF ", "FII ", "FPI ", " AMC", "INSURANCE",
    "ASSET MANAGEMENT", "ASSET MGMT", "INVESTMENT TRUST",
    "MUTUAL FUND", "PROVIDENT FUND", "PENSION FUND",
    # Large Indian MF schemes (appear with scheme name)
    "NIPPON INDIA", "ADITYA BIRLA SUN LIFE", "FRANKLIN TEMPLETON",
    "MIRAE ASSET", "MOTILAL OSWAL", "WHITEOAK CAPITAL",
    "EDELWEISS ", "QUANTUM MUTUAL",
    # Large FIIs — always followed by (SINGAPORE), MAURITIUS, etc.
    "GOLDMAN SACHS", "MORGAN STANLEY", "JP MORGAN",
    "NOMURA", "VANGUARD ", "BLACKROCK ", "FIDELITY ",
    "CITIGROUP", "MERRILL LYNCH", "BARCLAYS", "SOCIETE GENERALE",
    "DEUTSCHE BANK", "CREDIT SUISSE", "MACQUARIE",
    "CLSA ", "JEFFERIES", "HSBC ", "UBS ",
    # Domestic institutions (long-form names only)
    "LIC OF INDIA", "GIC OF INDIA", "NPS TRUST",
    "NATIONAL INSURANCE", "ORIENTAL INSURANCE",
]

def _is_institutional(client_name: str) -> bool:
    cu = client_name.upper()
    return any(kw in cu for kw in _INST_KEYWORDS)


def _bulk_deal_score(symbol: str, deals: list[dict], lookback_days: int = 30) -> tuple[float, list[str]]:
    """
    Score bulk/block deals for a symbol.
    Returns (score 0-10, list of deal labels for the Why column).
    +3 for each institutional BUY deal (max 3 deals = +9, capped at 10)
    -3 for each institutional SELL deal (floor 0)
    """
    from datetime import date as _date, timedelta
    cutoff = pd.Timestamp(_date.today() - timedelta(days=lookback_days))
    sym_deals = [d for d in deals if d.get("symbol", "").upper() == symbol.upper()]
    labels: list[str] = []
    score = 0.0

    for d in sym_deals:
        # Parse date
        try:
            dt = pd.Timestamp(d.get("date", ""))
            if dt < cutoff:
                continue
        except Exception:
            continue

        client = d.get("client", "")
        side   = d.get("side", "").strip().upper()
        dtype  = d.get("deal_type", "")
        val_cr = float(d.get("value_cr", 0))
        is_inst = _is_institutional(client)

        # BUG-036 FIX: exact-match side instead of "B" in side (which would
        # match "PURCHASE" / "SUBSCRIBE" etc).
        is_buy  = side in ("BUY", "B")
        is_sell = side in ("SELL", "S")

        if is_inst:
            if is_buy:
                score = min(10.0, score + 3.0)
                labels.append(f"🟢{dtype[:2]}Buy:{client[:12]}(₹{val_cr:.0f}Cr)")
            elif is_sell:
                score = max(0.0, score - 3.0)
                labels.append(f"🔴{dtype[:2]}Sell:{client[:12]}(₹{val_cr:.0f}Cr)")
        else:
            # Non-institutional still informational
            if is_buy:
                score = min(10.0, score + 0.5)
            elif is_sell:
                score = max(0.0, score - 0.5)

    return round(score, 1), labels


# ── Factor 1: Quality ─────────────────────────────────────────────────────────

def _quality_score(funda: Optional[dict]) -> tuple[float, list[str]]:
    """
    Max 20 pts. Based purely on fundamentals.db data.
    ROE: 5pt | D/E: 5pt | EPS growth: 5pt | Sales growth: 5pt
    """
    if not funda:
        return 0.0, []
    score = 0.0
    labels: list[str] = []

    roe  = _safe(funda.get("roe"))
    de   = _safe(funda.get("debt_to_equity"), default=999.0)
    epsg = _safe(funda.get("eps_growth_yoy"))
    salg = _safe(funda.get("sales_growth_yoy"))

    # ROE (5 pts)
    if roe >= 20:
        score += 5; labels.append(f"ROE{roe:.0f}%✓")
    elif roe >= 15:
        score += 3; labels.append(f"ROE{roe:.0f}%~")
    elif roe >= 10:
        score += 1

    # D/E (5 pts)
    if de < 0.3:
        score += 5; labels.append(f"D/E{de:.2f}✓")
    elif de < 0.7:
        score += 3; labels.append(f"D/E{de:.2f}~")
    elif de < 1.2:
        score += 1

    # EPS growth (5 pts)
    if epsg >= 25:
        score += 5; labels.append(f"EPS+{epsg:.0f}%✓")
    elif epsg >= 15:
        score += 3; labels.append(f"EPS+{epsg:.0f}%~")
    elif epsg >= 5:
        score += 1

    # Sales growth (5 pts)
    if salg >= 20:
        score += 5; labels.append(f"Sales+{salg:.0f}%✓")
    elif salg >= 10:
        score += 3; labels.append(f"Sales+{salg:.0f}%~")
    elif salg >= 3:
        score += 1

    return min(20.0, score), labels


# ── Factor 2: Momentum ────────────────────────────────────────────────────────

def _momentum_score(c: pd.Series, nifty: Optional[pd.Series],
                    rank_6m: float, rank_12m: float) -> tuple[float, list[str]]:
    """
    Max 20 pts.
    6M return rank (0-1): 8pt | 12M return rank: 7pt | RS line new high: 5pt
    """
    score = 0.0
    labels: list[str] = []

    # 6M rank (8 pts): percentile among all scanned stocks
    if rank_6m >= 0.90:
        score += 8; labels.append(f"6M-Top10%✓")
    elif rank_6m >= 0.75:
        score += 6; labels.append(f"6M-Top25%")
    elif rank_6m >= 0.50:
        score += 3

    # 12M rank (7 pts)
    if rank_12m >= 0.90:
        score += 7; labels.append(f"12M-Top10%✓")
    elif rank_12m >= 0.75:
        score += 5; labels.append(f"12M-Top25%")
    elif rank_12m >= 0.50:
        score += 2

    # RS Line New High (5 pts) — the strongest pre-breakout signal
    if _rs_line_new_high(c, nifty):
        score += 5; labels.append("RSLineHigh✓")

    return min(20.0, score), labels


# ── Factor 3: Smart Money ─────────────────────────────────────────────────────

def _smart_money_score(
    symbol: str,
    c: pd.Series,
    df: pd.DataFrame,
    deliv: Optional[pd.Series],
    bulk_score: float,
    bulk_labels: list[str],
    fo_signal: dict,
    fii_signal: str,
) -> tuple[float, list[str]]:
    """
    Max 25 pts.
    Delivery% trend: 5pt | Bulk deals: up to 10pt |
    F&O OI signal: 5pt | FII market-wide: 5pt
    """
    score = 0.0
    labels: list[str] = []

    # A) Delivery % trend (5 pts)
    # BUG-029 FIX: score relative to the stock's own 30-day average delivery,
    # not absolute thresholds. This handles stocks with naturally low/high delivery %.
    if deliv is not None and len(deliv) >= 10:
        avg5d   = _safe(deliv.iloc[-5:].mean())
        avg30d  = _safe(deliv.iloc[-30:].mean()) if len(deliv) >= 30 else avg5d
        cur_del = _safe(deliv.iloc[-1])
        del_ratio = cur_del / avg30d if avg30d > 0 else 1.0
        if del_ratio >= 1.5 and cur_del >= avg5d:
            score += 5; labels.append(f"Del{cur_del:.0f}%↑✓")   # 50%+ above own baseline
        elif del_ratio >= 1.2 and cur_del >= avg5d:
            score += 3; labels.append(f"Del{cur_del:.0f}%~")
        elif del_ratio >= 1.0:
            score += 1

    # B) Bulk/Block deals (0-10 pts from bulk_score)
    score += bulk_score
    labels.extend(bulk_labels[:2])   # show at most 2 deal labels in Why

    # C) F&O OI signal (5 pts) — with OBV fallback when F&O data unavailable
    fo_sig = fo_signal.get("signal", "NEUTRAL")
    if fo_sig == "LONG_BUILDUP":
        score += 5; labels.append("OI:LongBuild✓")
    elif fo_sig == "SHORT_COVER":
        score += 3; labels.append("OI:ShortCov")
    elif fo_sig == "SHORT_BUILDUP":
        score -= 2
        labels.append("OI:ShortBuild⚠")
    elif fo_sig == "LONG_UNWIND":
        score -= 2
        labels.append("OI:LongUnwind⚠")
    else:
        # F&O data unavailable — use OBV (On-Balance Volume) as smart money proxy
        # OBV rising + price rising = volume confirming the uptrend = smart money buying
        try:
            vol_s = df["Volume"].dropna() if "Volume" in df.columns else pd.Series([], dtype=float)
            if len(c) >= 20 and len(vol_s) >= 20:
                close_chg = c.diff()
                # BUG-018 FIX: flat days (close_chg == 0) should NOT count as down days.
                # np.sign gives +1/-1/0 — multiplying volume by sign yields 0 contribution
                # on flat days instead of treating them as distribution.
                sign = np.sign(close_chg.fillna(0))
                obv_daily = vol_s * sign
                obv = obv_daily.cumsum()
                # BUG-FIX: OBV is a cumulative sum that is often NEGATIVE, so a
                # multiplicative test (obv_now > obv_then * 1.02) inverted for
                # negative values — distributing stocks could earn the +3.
                # Scale the 20-session OBV CHANGE by average volume instead:
                # ≥ 2 average days of net buying volume = clear accumulation.
                obv_chg   = float(obv.iloc[-1]) - float(obv.iloc[-20])
                avg_vol20 = float(vol_s.iloc[-20:].mean())
                if avg_vol20 > 0 and obv_chg >= avg_vol20 * 2:
                    score += 3; labels.append("OBV↑(proxy)")
                elif obv_chg > 0:
                    score += 1
        except Exception:
            pass

    # D) FII market-wide signal (5 pts)
    if "Buying" in fii_signal:
        score += 5; labels.append("FII:Buying✓")
    elif "Neutral" in fii_signal:
        score += 2

    return min(25.0, max(0.0, score)), labels


# ── Factor 4: Earnings Quality ────────────────────────────────────────────────

def _earnings_quality_score(funda: Optional[dict]) -> tuple[float, list[str]]:
    """
    Max 15 pts.
    EPS acceleration: 5pt | Promoter > 50%: 3pt |
    Promoter stable/rising: 3pt | Fresh data (<30d): 4pt
    """
    if not funda:
        return 0.0, []
    score = 0.0
    labels: list[str] = []

    # EPS acceleration (5 pts) — 3 consecutive quarters accelerating
    eps_accel = int(funda.get("eps_accel") or 0)
    if eps_accel:
        score += 5; labels.append("EPSAccel✓")

    # Promoter holding > 40% (3 pts) — Indian promoters typically 40-75%
    promo = _safe(funda.get("promoter_holding"))
    if promo >= 50:
        score += 3; labels.append(f"Promo{promo:.0f}%✓")
    elif promo >= 35:
        score += 2; labels.append(f"Promo{promo:.0f}%")
    elif promo >= 20:
        score += 1

    # Promoter stable or increasing (3 pts)
    promo_delta = _safe(funda.get("promoter_delta"))
    if promo_delta >= 0.5:
        score += 3; labels.append("Promo↑✓")
    elif promo_delta >= 0:
        score += 2; labels.append("PromoFlat")
    elif promo_delta > -1.0:
        score += 1  # slight decline, not alarming

    # Data freshness (4 pts) — fresher fundamentals are more reliable
    # BUG-030 FIX: all data within 90 days (one quarter) is equally valid.
    # Previous scoring unfairly penalized 60-90 day old data vs 30-day data
    # even though both represent the same current quarter's results.
    updated_at = funda.get("updated_at", 0) or 0
    age_days = (time.time() - float(updated_at)) / 86400
    if age_days <= 90:
        score += 4; labels.append("FreshData✓")   # current quarter — all equally valid
    elif age_days <= 180:
        score += 2                                  # one quarter stale
    else:
        score += 1                                  # two+ quarters stale

    return min(15.0, score), labels


# ── Factor 5: Technical ───────────────────────────────────────────────────────

def _technical_score(df: pd.DataFrame) -> tuple[float, list[str]]:
    """
    Max 10 pts.
    Stage 2: 4pt | MA stack (50>100>200): 3pt | ADX > 25: 3pt
    """
    score = 0.0
    labels: list[str] = []
    c = df["Close"].dropna()
    if len(c) < 50:
        return 0.0, []

    # Stage (4 pts)
    stg = _stage(c)
    if stg == 2:
        score += 4; labels.append("Stage2✓")
    elif stg == 1:
        score += 1  # basing

    # MA stack (3 pts)
    ma50  = _safe(c.rolling(50).mean().iloc[-1])
    cur   = _safe(c.iloc[-1])
    stack_pts = 0
    if cur > ma50 and ma50 > 0:
        stack_pts += 1
        if len(c) >= 100:
            ma100 = _safe(c.rolling(100).mean().iloc[-1])
            if ma50 > ma100 and ma100 > 0:
                stack_pts += 1
                if len(c) >= 200:
                    ma200 = _safe(c.rolling(200).mean().iloc[-1])
                    if ma100 > ma200 and ma200 > 0:
                        stack_pts += 1
    if stack_pts == 3:
        labels.append("MA50>100>200✓")
    elif stack_pts == 2:
        labels.append("MA50>100✓")
    score += stack_pts

    # ADX (3 pts)
    adx_v = _adx(df)
    if adx_v >= 30:
        score += 3; labels.append(f"ADX{adx_v:.0f}✓")
    elif adx_v >= 25:
        score += 2; labels.append(f"ADX{adx_v:.0f}")
    elif adx_v >= 20:
        score += 1

    return min(10.0, score), labels


# ── Factor 6: Risk / Reward ───────────────────────────────────────────────────

def _risk_score(df: pd.DataFrame, nifty: Optional[pd.Series]) -> tuple[float, list[str]]:
    """
    Max 10 pts.
    Max drawdown: 4pt | ADTV ≥₹5Cr: 3pt | Not overextended: 3pt
    """
    score = 0.0
    labels: list[str] = []
    c = df["Close"].dropna()
    v = df["Volume"].dropna() if "Volume" in df.columns else pd.Series([], dtype=float)

    if len(c) < 30:
        return 0.0, []

    # Max drawdown (4 pts) — calibrated for current market environment
    dd = _max_drawdown(c)
    if dd < 12:
        score += 4; labels.append(f"DD{dd:.0f}%✓")
    elif dd < 18:
        score += 3; labels.append(f"DD{dd:.0f}%~")
    elif dd < 25:
        score += 1

    # ADTV (3 pts)
    if len(v) >= 20:
        adtv_cr = _safe(v.rolling(20).mean().iloc[-1]) * _safe(c.iloc[-1]) / 1e7
        if adtv_cr >= 20:
            score += 3; labels.append(f"ADTV₹{adtv_cr:.0f}Cr✓")
        elif adtv_cr >= 5:
            score += 2; labels.append(f"ADTV₹{adtv_cr:.0f}Cr")
        elif adtv_cr >= 1:
            score += 1

    # Extension from MA50 (3 pts) — avoid chasing >15% extended stocks
    if len(c) >= 50:
        ma50 = _safe(c.rolling(50).mean().iloc[-1])
        ext  = ((_safe(c.iloc[-1]) - ma50) / ma50 * 100) if ma50 > 0 else 0.0
        if ext <= 5:
            score += 3; labels.append("NotExtended✓")
        elif ext <= 10:
            score += 2
        elif ext <= 15:
            score += 1
        # > 15% extended: 0 pts (risky chase)

    return min(10.0, score), labels


# ── Exit signal detector ──────────────────────────────────────────────────────

def _exit_flags(
    symbol: str,
    df: pd.DataFrame,
    deliv: Optional[pd.Series],
    fo_signal: dict,
    funda: Optional[dict],
    rs_excess_6m: Optional[float],
    deals: list[dict],
) -> list[str]:
    """Return list of active exit flag codes. Empty = no warning."""
    flags: list[str] = []
    c = df["Close"].dropna()
    v = df["Volume"].dropna() if "Volume" in df.columns else pd.Series([], dtype=float)

    # 1. Delivery % collapsed
    if deliv is not None and len(deliv) >= 10:
        avg30 = _safe(deliv.iloc[-30:].mean()) if len(deliv) >= 30 else _safe(deliv.mean())
        cur5  = _safe(deliv.iloc[-5:].mean())
        if avg30 >= 45 and cur5 < 28:
            flags.append("DELIV_DROP")

    # 2. Bulk/block institutional sell in last 14 days
    from datetime import date as _date, timedelta
    cutoff14 = pd.Timestamp(_date.today() - timedelta(days=14))
    for d in deals:
        try:
            if d.get("symbol", "").upper() != symbol.upper():
                continue
            # BUG-036 FIX: exact side match (was "S" in side which matched everything)
            if d.get("side", "").strip().upper() not in ("SELL", "S"):
                continue
            if not _is_institutional(d.get("client", "")):
                continue
            dt = pd.Timestamp(d.get("date", ""))
            if dt >= cutoff14:
                flags.append("BULK_SELL")
                break
        except Exception:
            continue

    # 3. F&O signals
    fo_sig = fo_signal.get("signal", "")
    if fo_sig == "SHORT_BUILDUP":
        flags.append("OI_SHORT")
    elif fo_sig == "LONG_UNWIND":
        flags.append("OI_UNWIND")

    # 4. Distribution days ≥ 4 in last 20 sessions
    if len(c) >= 25 and len(v) >= 25:
        _, dist = _acc_dist_days(c, v, n=20)
        if dist >= 4:
            flags.append("DIST_DAYS")

    # 5. RS underperformance vs Nifty (6M) — only flag severe underperformers
    # -10% threshold avoids triggering en masse during broad market corrections
    if rs_excess_6m is not None and rs_excess_6m < -10:
        flags.append("RS_WEAK")

    # 6. MA50 break with confirmation — must be >3% below MA50 to avoid false signals
    if len(c) >= 50:
        ma50 = _safe(c.rolling(50).mean().iloc[-1])
        if _safe(c.iloc[-1]) < ma50 * 0.97:
            flags.append("MA50_BREAK")

    return flags


# ── Per-stock analysis ────────────────────────────────────────────────────────

def _analyze_one(
    symbol: str,
    df: pd.DataFrame,
    nifty: Optional[pd.Series],
    funda_db: dict[str, dict],
    deals: list[dict],
    fo_signals: dict[str, dict],
    fii_signal: str,
    rank_6m: float,
    rank_12m: float,
) -> Optional[dict]:
    """Score one stock across all 6 factors. Returns dict or None if skip."""
    try:
        df   = _strip_splits(df)
        c    = df["Close"].dropna()
        v    = df["Volume"].dropna() if "Volume" in df.columns else pd.Series([], dtype=float)

        if len(c) < MIN_BARS:
            return None

        cur = _safe(c.iloc[-1])
        if cur <= 0:
            return None

        # Liquidity gate
        adtv_cr = 0.0
        if len(v) >= 20:
            adtv_cr = _safe(v.rolling(20).mean().iloc[-1]) * cur / 1e7
        if adtv_cr < MIN_ADTV_CR:
            return None

        funda    = funda_db.get(symbol)
        fo_sig   = fo_signals.get(symbol, {})
        deliv    = _load_delivery_series(symbol, n_days=30)

        # Bulk deal score
        b_score, b_labels = _bulk_deal_score(symbol, deals)

        # RS excess return (for exit flags + momentum factor)
        rs_6m = _rs_vs_nifty_6m(c, nifty)

        # ── Factor scoring ────────────────────────────────────────────────────
        q_score,  q_labels  = _quality_score(funda)
        m_score,  m_labels  = _momentum_score(c, nifty, rank_6m, rank_12m)
        sm_score, sm_labels = _smart_money_score(
            symbol, c, df, deliv, b_score, b_labels, fo_sig, fii_signal
        )
        eq_score, eq_labels = _earnings_quality_score(funda)
        t_score,  t_labels  = _technical_score(df)
        r_score,  r_labels  = _risk_score(df, nifty)

        total = round(q_score + m_score + sm_score + eq_score + t_score + r_score, 1)

        # Tier assigned later in run_alpha_scan() via percentile thresholds.
        # P0-4 FIX: previous fixed thresholds (55/42/30) were mis-calibrated —
        # top score was only 60 vs 55 BUY cutoff, so BUY/STRONG counts drifted
        # wildly across regimes. Percentile-based: top 5%=BUY, 5-15%=STRONG,
        # 15-40%=MONITOR, rest=AVOID. Always relative to current universe.
        tier = "AVOID"

        # Exit flags
        exits = _exit_flags(symbol, df, deliv, fo_sig, funda, rs_6m, deals)

        # Returns
        r6m  = round((cur / _safe(c.iloc[-126]) - 1) * 100, 2) if len(c) >= 126 else None
        r12m = round((cur / _safe(c.iloc[-252]) - 1) * 100, 2) if len(c) >= 252 else None
        r1m  = round((cur / _safe(c.iloc[-21])  - 1) * 100, 2) if len(c) >= 21  else None

        # Latest delivery %
        deliv_cur = round(_safe(deliv.iloc[-1]), 1) if deliv is not None and len(deliv) else None

        # F&O OI signal summary
        fo_label = fo_sig.get("signal", "—").replace("_", " ").title() if fo_sig else "—"

        why_parts = q_labels + m_labels + sm_labels + eq_labels + t_labels + r_labels
        why = " · ".join(why_parts[:8])  # top 8 reasons

        return {
            "symbol":      symbol,
            "sector":      _SYM_TO_GROUP.get(symbol, "Other"),
            "price":       round(cur, 2),
            "score":       total,
            "tier":        tier,
            # factor breakdown
            "q_score":     round(q_score,  1),
            "m_score":     round(m_score,  1),
            "sm_score":    round(sm_score, 1),
            "eq_score":    round(eq_score, 1),
            "t_score":     round(t_score,  1),
            "r_score":     round(r_score,  1),
            # returns
            "r1m":         r1m,
            "r6m":         r6m,
            "r12m":        r12m,
            "rs_6m":       rs_6m,
            # smart money detail
            "deliv_pct":   deliv_cur,
            "fo_signal":   fo_label,
            "bulk_score":  b_score,
            # fundamentals
            "roe":         round(_safe(funda.get("roe")),             1) if funda else None,
            "debt_eq":     round(_safe(funda.get("debt_to_equity")), 2) if funda else None,
            "eps_growth":  round(_safe(funda.get("eps_growth_yoy")), 1) if funda else None,
            "sales_growth":round(_safe(funda.get("sales_growth_yoy")),1) if funda else None,
            "promoter":    round(_safe(funda.get("promoter_holding")),1) if funda else None,
            "pe":          round(_safe(funda.get("pe_ratio")),        1) if funda else None,
            "eps_accel":   bool(funda.get("eps_accel")) if funda else False,
            # exit flags
            "exit_flags":  exits,
            "has_exit":    len(exits) > 0,
            # why
            "why":         why,
        }
    except Exception:
        return None


# ── FII signal helper (uses cached app.py endpoint data) ─────────────────────

_FII_CACHE: dict = {"signal": None, "ts": 0.0}
_FII_TTL = 4 * 3600   # 4 h — FII daily net only changes once per day anyway


def _get_fii_signal() -> str:
    """Read FII daily net from NSE API (same endpoint as /api/fii-dii).
    Returns 'FII Buying' / 'FII Selling' / 'Neutral'.
    Cached 4h HERE (the old docstring claimed fo_data cached it — it didn't,
    so every scan made a fresh 3-request NSE round-trip)."""
    if _FII_CACHE["signal"] is not None and time.time() - _FII_CACHE["ts"] < _FII_TTL:
        return _FII_CACHE["signal"]
    try:
        # Try the FII/DII flow endpoint
        import requests as _req
        sess = _req.Session()
        sess.headers.update({
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                          "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
            "Accept":     "application/json, text/plain, */*",
            "Referer":    "https://www.nseindia.com/",
        })
        sess.get("https://www.nseindia.com", timeout=6)
        sess.get("https://www.nseindia.com/api/marketStatus", timeout=6)
        r = sess.get("https://www.nseindia.com/api/fiidiiTradeReact", timeout=10)
        if r.status_code == 200:
            rows = r.json()
            # BUG-FIX: endpoint returns 1 row/day (not 5). Variable was misnamed
            # "fii_net_5d" but only accumulated today's net. With ₹500 Cr threshold,
            # the signal flipped on daily noise. Now: explicit 1-day net + much
            # stronger threshold (₹2000 Cr) so signal reflects a real conviction day.
            fii_net_today = 0.0
            for item in rows:
                cat = (item.get("category") or "").upper()
                if cat.startswith("FII") or cat.startswith("FPI"):
                    nv = float(str(item.get("netValue", 0)).replace(",", "") or 0)
                    fii_net_today += nv
                    break   # one FII row per day
            # Threshold: ₹2000 Cr is the magnitude that historically marks
            # actual institutional positioning vs noise (typical day is ±500 Cr).
            sig = "Neutral"
            if fii_net_today > 2000:
                sig = "FII Buying"
            elif fii_net_today < -2000:
                sig = "FII Selling"
            _FII_CACHE["signal"] = sig
            _FII_CACHE["ts"] = time.time()
            return sig
    except Exception:
        pass
    # Don't cache failures — retry on the next scan
    return "Neutral"


# ── Main entry ────────────────────────────────────────────────────────────────

def run_alpha_scan(progress_callback=None) -> dict:
    """
    Full multi-factor composite scan.
    Returns dict with 'results' list (sorted by score desc), tier counts,
    computed_at, total_scanned.
    """
    if (_cache["data"]
            and time.time() - _cache["ts"] < CACHE_TTL
            and _cache["data"].get("results")):
        return _cache["data"]

    def _prog(n, total, msg):
        if progress_callback:
            progress_callback(n, total, msg)

    _prog(0, 100, "Loading OHLCV data…")
    stocks = _get_stocks()   # {sym: df} for full universe
    if not stocks:
        return {"results": [], "computed_at": int(time.time()), "total_scanned": 0,
                "error": "No bhavcopy data available"}

    nifty = _build_nifty(stocks)
    total = len(stocks)
    _prog(5, 100, f"Loaded {total} stocks. Fetching smart-money data…")

    # ── Pre-load shared data (done once for efficiency) ───────────────────────
    funda_db = load_all_fundamentals()
    _prog(10, 100, "Fundamentals loaded. Fetching bulk/block deals…")

    try:
        bulk_data = run_bulk_deals()
        all_deals = bulk_data.get("deals", [])
    except Exception:
        all_deals = []
    _prog(15, 100, "Bulk deals loaded. Fetching F&O OI signals…")

    try:
        fo_signals = get_fo_signals()
    except Exception:
        fo_signals = {}
    _prog(20, 100, f"F&O data: {len(fo_signals)} stocks. Getting FII signal…")

    try:
        fii_signal = _get_fii_signal()
    except Exception:
        fii_signal = "Neutral"
    _prog(25, 100, f"FII signal: {fii_signal}. Computing momentum ranks…")

    # Pre-compute return ranks (needed for momentum factor)
    def _ret(sym, days):
        c = stocks[sym]["Close"].dropna()
        if len(c) <= days:
            return None
        try:
            return float(c.iloc[-1] / c.iloc[-days] - 1) * 100
        except Exception:
            return None

    ret6m_all  = {s: _ret(s, 126) for s in stocks}
    ret12m_all = {s: _ret(s, 252) for s in stocks}

    valid6m  = [v for v in ret6m_all.values()  if v is not None]
    valid12m = [v for v in ret12m_all.values() if v is not None]

    # P2-13: full-universe cross-sectional RS ranks (1-99 percentile, consistent
    # across all scanners that pass the same returns dict to this helper).
    rs_rank_6m  = cross_sectional_rs_rank(ret6m_all)
    rs_rank_12m = cross_sectional_rs_rank(ret12m_all)

    # P2-12: per-sector RS rank — sector_rs = median stock_rs of sector members.
    sector_rs_lookup: dict[str, int] = {}
    sector_to_syms: dict[str, list[str]] = {}
    for s in stocks:
        g = _SYM_TO_GROUP.get(s, "Other")
        sector_to_syms.setdefault(g, []).append(s)
    sector_median_rs: dict[str, float] = {}
    for grp, syms in sector_to_syms.items():
        rs_vals = [rs_rank_6m[s] for s in syms if s in rs_rank_6m]
        if rs_vals:
            sector_median_rs[grp] = float(np.median(rs_vals))
    # Rank sectors themselves 1-99 by their median stock RS
    sector_rs_lookup = cross_sectional_rs_rank(sector_median_rs)

    def _rank(v, series):
        if v is None or not series:
            return 0.5
        arr = sorted(series)
        idx = sum(1 for x in arr if x <= v)
        return idx / len(arr)

    _prog(30, 100, "Scanning stocks…")

    done  = [0]
    results: list[dict] = []

    def _job(item):
        sym, df = item
        r6m  = ret6m_all.get(sym)
        r12m = ret12m_all.get(sym)
        rank6  = _rank(r6m,  valid6m)
        rank12 = _rank(r12m, valid12m)
        r = _analyze_one(
            sym, df, nifty, funda_db, all_deals, fo_signals,
            fii_signal, rank6, rank12
        )
        if r is not None:
            # P2-12 + P2-13: attach full-universe RS rank + sector-adjusted RS
            stock_rs = rs_rank_6m.get(sym)
            sect     = r.get("sector", "Other")
            sect_rs  = sector_rs_lookup.get(sect)
            r["rs_rank"]       = stock_rs
            r["rs_rank_12m"]   = rs_rank_12m.get(sym)
            r["sector_rs"]     = sect_rs
            r["sector_adj_rs"] = (sector_adjusted_rs(stock_rs, sect_rs)
                                   if stock_rs is not None and sect_rs is not None
                                   else stock_rs)
        done[0] += 1
        if progress_callback and done[0] % 100 == 0:
            pct = 30 + int(done[0] / total * 60)
            progress_callback(pct, 100, f"Scanning {done[0]}/{total}…")
        return r

    with ThreadPoolExecutor(max_workers=8) as ex:
        for r in ex.map(_job, stocks.items()):
            if r is not None:
                results.append(r)

    # P2-14: persist today's stage for every scored stock so the transition
    # log can answer "did this stock JUST become Stage 2 today?" via SQLite.
    try:
        import stage_transitions as _stg
        stage_map = {r["symbol"]: int(stage_analysis(stocks[r["symbol"]]["Close"].dropna()))
                     for r in results if r.get("symbol") in stocks}
        _stg.update_all({s: v for s, v in stage_map.items() if v in (1, 2, 3, 4)})
    except Exception:
        pass

    _prog(92, 100, "Assigning percentile-based tiers…")

    # P0-4 FIX: assign tiers by percentile rank across this scan's universe.
    # Top 5% = BUY (institutional grade), 5-15% = STRONG, 15-40% = MONITOR, rest = AVOID.
    # Robust to score drift across regimes — always picks the top of *today's* distribution.
    if results:
        scores = sorted([r["score"] for r in results], reverse=True)
        n = len(scores)
        buy_cutoff     = scores[max(0, int(n * 0.05) - 1)]
        strong_cutoff  = scores[max(0, int(n * 0.15) - 1)]
        monitor_cutoff = scores[max(0, int(n * 0.40) - 1)]
        for r in results:
            s = r["score"]
            if s >= buy_cutoff:
                r["tier"] = "BUY"
            elif s >= strong_cutoff:
                r["tier"] = "STRONG"
            elif s >= monitor_cutoff:
                r["tier"] = "MONITOR"
            else:
                r["tier"] = "AVOID"

    # Sort: exit flagged last within tier, then by score desc
    tier_order = {"BUY": 0, "STRONG": 1, "MONITOR": 2, "AVOID": 3}
    results.sort(key=lambda r: (
        tier_order.get(r["tier"], 9),
        1 if r.get("has_exit") else 0,
        -r["score"]
    ))

    # Tier counts
    buy_count     = sum(1 for r in results if r["tier"] == "BUY")
    strong_count  = sum(1 for r in results if r["tier"] == "STRONG")
    monitor_count = sum(1 for r in results if r["tier"] == "MONITOR")
    avoid_count   = sum(1 for r in results if r["tier"] == "AVOID")
    exit_count    = sum(1 for r in results if r.get("has_exit"))

    out = {
        "results":       results,
        "computed_at":   int(time.time()),
        "total_scanned": total,
        "buy_count":     buy_count,
        "strong_count":  strong_count,
        "monitor_count": monitor_count,
        "avoid_count":   avoid_count,
        "exit_count":    exit_count,
        "fo_count":      len(fo_signals),
        "fii_signal":    fii_signal,
    }
    _cache["data"] = out
    _cache["ts"]   = time.time()

    # P2-10/11: enrich AFTER cache is set so Alpha's own results are
    # visible to build_consensus when it scans every scanner's _cache.
    try:
        from consensus import enrich_results, invalidate_cache as _con_inv
        _con_inv()
        enrich_results(results)
    except Exception:
        pass

    _prog(100, 100, f"Done — {buy_count} BUY(≥65) · {strong_count} STRONG(50-64) · {exit_count} exit flags")
    return out
