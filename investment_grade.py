"""
Investment-Grade Scanner — Core Portfolio Holdings
──────────────────────────────────────────────────
LOCAL-ONLY module. Gitignored — never pushed to GitHub.

Scores Nifty 500 stocks 0–12 across 8 technical + 4 fundamental criteria
to find sustained multi-month uptrenders suitable for core long-term holdings.

Technical (8 pts): Months above MA200, MA stack duration, R² smoothness,
                   Max drawdown, 6M & 12M relative return, ADTV liquidity,
                   Distance from 52W high.
Fundamental (4 pts): ROE, EPS growth, Sales growth, Debt/Equity.

Tiers: A (≥10/12) · B (8-9) · Watchlist (6-7) · Excluded (<6)
"""
from __future__ import annotations

import time
from typing import Optional

import numpy as np
import pandas as pd

from industry_groups import _get_stocks, _build_nifty, INDUSTRY_GROUPS

CACHE_TTL = 1800   # 30 min
INVEST_MIN_SCORE = 6   # below this, stock not shown

_cache = {"data": None, "ts": 0.0}


# ── Technical helpers ─────────────────────────────────────────────────────────

def _strip_split_history(df: pd.DataFrame) -> pd.DataFrame:
    """Apply canonical backward-adjustment (analysis_utils.adjust_for_splits)
    instead of stripping pre-event history. The previous strip-style approach
    threw away years of legitimate price action whenever it detected a single
    >35% drop — and used a different threshold than every other scanner, so
    the Investment Grade tab saw a DIFFERENT historical series for the same
    stock than Trending / Edge / Monster / Alpha / VVV tabs."""
    if df is None or df.empty or len(df) < 30:
        return df
    from analysis_utils import adjust_for_splits
    return adjust_for_splits(df)


def _consecutive_above_ma(c: pd.Series, ma_period: int) -> int:
    """Days price has been continuously above its `ma_period` MA, counted back from today."""
    if len(c) < ma_period + 1:
        return 0
    ma = c.rolling(ma_period).mean()
    above = (c > ma).dropna()
    days = 0
    for v in reversed(above.tolist()):
        if v:
            days += 1
        else:
            break
    return days


def _months_above_long_ma(c: pd.Series) -> tuple[int, int]:
    """
    Returns (consecutive_days_above_trend_MA, ma_period_used).
    BUG-021 NOTE: despite the name, this returns DAYS (not months). The variable name
    is legacy — callers display it as days or convert via / 22.0 for trading months.
    Strategy: pick the longest MA where we have at least 80 days of valid
    MA values to count over (otherwise the threshold is unreachable).
    For typical 212-day cache → uses MA50 (163 valid days).
    For 250+ days → uses MA100 (150+ valid).
    For 350+ days → uses MA200 (150+ valid).
    """
    n = len(c)
    # We need MA-period + meaningful-lookback worth of data.
    # 'meaningful lookback' = at least 80 days so threshold (60) is achievable.
    for period in (200, 100, 50):
        if n >= period + 80:
            return _consecutive_above_ma(c, period), period
    # Final fallback for very short history
    if n >= 55:
        return _consecutive_above_ma(c, 50), 50
    return 0, 0


def _ma_stack_duration(c: pd.Series) -> int:
    """Consecutive days the full bullish MA-stack (MA50 > MA100 > MA200) is aligned.
    BUG-024 FIX: previously only checked 50>100, advertising it as the full
    50>100>200 stack. Now requires all three MAs to be properly stacked.
    Falls back to 50>100 when ≥200 bars are not yet available."""
    if len(c) < 105:
        return 0
    ma50  = c.rolling(50).mean()
    ma100 = c.rolling(100).mean()
    if len(c) >= 205:
        ma200 = c.rolling(200).mean()
        stacked = ((ma50 > ma100) & (ma100 > ma200)).dropna()
    else:
        stacked = (ma50 > ma100).dropna()
    days = 0
    for v in reversed(stacked.tolist()):
        if v:
            days += 1
        else:
            break
    return days


def _r_squared(c: pd.Series, n: int = 132) -> Optional[float]:
    """R² of log(price) regressed on time over last n sessions (1 = perfect uptrend)."""
    if len(c) < n:
        return None
    y = np.log(c.iloc[-n:].values)
    if not np.all(np.isfinite(y)):
        return None
    x = np.arange(len(y))
    slope, intercept = np.polyfit(x, y, 1)
    pred = slope * x + intercept
    ss_res = float(((y - pred) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    if ss_tot == 0:
        return None
    return max(0.0, min(1.0, 1 - ss_res / ss_tot))


def _max_drawdown(c: pd.Series, n: int = 132) -> float:
    """Worst peak-to-trough drawdown over last n sessions, as positive %."""
    if len(c) < 30:
        return 100.0
    s = c.iloc[-n:]
    running_max = s.cummax()
    dd = (s - running_max) / running_max
    return abs(float(dd.min()) * 100)


def _adtv_cr(df: pd.DataFrame, n: int = 30) -> Optional[float]:
    """Avg daily turnover in ₹ crores over last n sessions.
    Align Close + Volume on a common non-NaN index before multiplying —
    otherwise mis-aligned dropna() would silently pair wrong rows, giving
    incorrect turnover for any stock with a single-day data gap in either
    column over the lookback window."""
    if "Close" not in df.columns or "Volume" not in df.columns:
        return None
    cv = df[["Close", "Volume"]].dropna()
    if len(cv) < n:
        return None
    turnover = (cv["Close"].iloc[-n:] * cv["Volume"].iloc[-n:]).mean()
    return round(float(turnover) / 1e7, 2)


def _pct_from_52w_high(c: pd.Series) -> Optional[float]:
    if len(c) < 30:
        return None
    hi52 = float(c.iloc[-min(252, len(c)):].max())
    cur  = float(c.iloc[-1])
    return round((cur / hi52 - 1) * 100, 2)


def _return_pct(c: pd.Series, days: int) -> Optional[float]:
    if len(c) < days + 1:
        return None
    return round((float(c.iloc[-1]) / float(c.iloc[-days]) - 1) * 100, 2)


# ── Fundamental loader ────────────────────────────────────────────────────────

def _load_fundamentals() -> dict[str, dict]:
    try:
        from fundamentals import load_all_fundamentals
        return load_all_fundamentals()
    except Exception:
        return {}


# ── Sector RS lookup for one symbol ───────────────────────────────────────────

def _sector_rs(stocks: dict, nifty: pd.Series) -> dict[str, float]:
    """Return 6M RS vs Nifty for each industry group (median of members)."""
    if nifty is None or len(nifty) < 130:
        return {}
    nifty_6m = (nifty.iloc[-1] / nifty.iloc[-126] - 1) * 100
    out = {}
    for group, syms in INDUSTRY_GROUPS.items():
        member_rs = []
        for s in syms:
            df = stocks.get(s)
            if df is None: continue
            c = df["Close"].dropna()
            if len(c) < 130: continue
            r6m = (c.iloc[-1] / c.iloc[-126] - 1) * 100
            member_rs.append(r6m - nifty_6m)
        if member_rs:
            out[group] = float(np.median(member_rs))
    return out


def _sym_to_sector() -> dict[str, str]:
    out = {}
    for group, syms in INDUSTRY_GROUPS.items():
        for s in syms:
            out[s] = group
    return out


# ── Scoring ───────────────────────────────────────────────────────────────────

def _score_technical(df: pd.DataFrame, nifty: pd.Series) -> dict:
    df = _strip_split_history(df)   # remove pre-split prices (e.g. MCX)
    c = df["Close"].dropna()
    if len(c) < 30:
        return {"score": 0, "skip": True}

    months_ma200, ma_period_used = _months_above_long_ma(c)
    stack_days   = _ma_stack_duration(c)
    r2           = _r_squared(c, 132)
    max_dd       = _max_drawdown(c, 132)
    adtv         = _adtv_cr(df, 30)
    pct_high     = _pct_from_52w_high(c)
    r6m          = _return_pct(c, 126)
    r12m         = _return_pct(c, 252)

    # Nifty benchmarks
    n6m  = _return_pct(nifty, 126) if nifty is not None else None
    n12m = _return_pct(nifty, 252) if nifty is not None else None

    score = 0
    flags = []
    # 1. Sustained uptrend: ≥60 consecutive days above the chosen trend MA
    if months_ma200 >= 60:
        score += 1
        flags.append(f"{months_ma200}d above MA{ma_period_used}")
    # 2. MA-stack (50>100>200 if data permits, else 50>100) for ≥40 days
    if stack_days >= 40:           score += 1; flags.append(f"MA stack {stack_days}d")
    # 3. R² ≥ 0.45 over 132d (smooth uptrend) — slightly relaxed from 0.50
    if r2 is not None and r2 >= 0.45: score += 1; flags.append(f"R² {r2:.2f}")
    # 4. Max DD < 20%
    if max_dd < 20:                score += 1; flags.append(f"DD only -{max_dd:.0f}%")
    # 5. 6M return > Nifty + 15%
    if r6m is not None and n6m is not None and (r6m - n6m) >= 15:
        score += 1; flags.append("6M >> Nifty")
    # 6. 12M return > Nifty + 25% (or skip if data short)
    if r12m is not None and n12m is not None and (r12m - n12m) >= 25:
        score += 1; flags.append("12M >> Nifty")
    elif r12m is None and r6m is not None and n6m is not None and (r6m - n6m) >= 40:
        # BUG-022 FIX: fallback threshold raised from 25% to 40% so it only fires
        # for exceptional 6M RS that clearly exceeds criterion 5 (6M > Nifty+15%)
        # BUG-IPO FIX: only award the 12M fallback if stock has >= 120 bars of data.
        # Stocks listed < 6 months ago always have partial 6M return that looks
        # exceptional on a percentage basis but doesn't reflect a full cycle.
        if len(c) >= 120:
            score += 1; flags.append("Strong 6M proxy")
    # 7. ADTV ≥ ₹50 Cr (institutional liquidity)
    if adtv is not None and adtv >= 50: score += 1; flags.append(f"ADTV ₹{adtv:.0f}Cr")
    # 8. Within 10% of 52W high
    if pct_high is not None and pct_high >= -10: score += 1; flags.append("Near 52W high")

    return {
        "score":         score,
        "max":           8,
        "skip":          False,
        "months_ma200":  months_ma200,
        "ma_period_used": ma_period_used,
        "stack_days":    stack_days,
        "r_squared":     round(r2, 3) if r2 is not None else None,
        "max_drawdown":  round(max_dd, 2),
        "adtv_cr":       adtv,
        "pct_from_high": pct_high,
        "ret_6m":        r6m,
        "ret_12m":       r12m,
        "ret_6m_excess": round(r6m - n6m, 2) if (r6m is not None and n6m is not None) else None,
        "ret_12m_excess":round(r12m - n12m, 2) if (r12m is not None and n12m is not None) else None,
        "flags":         flags,
    }


def _score_fundamental(f: dict | None) -> dict:
    """0-4 points from screener.in fundamentals."""
    if not f:
        return {"score": 0, "max": 4, "available": False}
    score = 0
    flags = []
    roe = f.get("roe")
    eps_g = f.get("eps_growth_yoy")
    sales_g = f.get("sales_growth_yoy")
    de = f.get("debt_to_equity")

    # ROE > 15%
    if roe is not None and roe >= 15:    score += 1; flags.append(f"ROE {roe:.0f}%")
    # EPS growth YoY > 12%
    if eps_g is not None and eps_g >= 12: score += 1; flags.append(f"EPS +{eps_g:.0f}%")
    # Sales growth YoY > 10%
    if sales_g is not None and sales_g >= 10: score += 1; flags.append(f"Sales +{sales_g:.0f}%")
    # D/E < 1.0
    if de is not None and de < 1.0:       score += 1; flags.append(f"D/E {de:.2f}")

    return {
        "score":     score,
        "max":       4,
        "available": True,
        "roe":       roe,
        "eps_growth_yoy": eps_g,
        "sales_growth_yoy": sales_g,
        "debt_to_equity": de,
        "pe":        f.get("pe_ratio"),
        "market_cap_cr": round(f["market_cap"], 0) if f.get("market_cap") else None,
        "promoter":  f.get("promoter_holding"),
        "flags":     flags,
    }


def _tier(total_score: int, fund_available: bool, tech_score: int = 0) -> tuple[str, str]:
    """
    Map total score → tier label + class.
    With fundamentals (max 12): A ≥10, B ≥8, Watchlist ≥6.
    Without fundamentals (max 8, tech-only): A ≥7, B ≥5 — labeled '(Tech)'
    so the user can see strong stocks like TDPOWERSYS/PRECWIRE that don't
    yet have screener.in coverage.
    """
    if fund_available:
        if total_score >= 10: return "A", "elite"
        if total_score >= 8:  return "B", "strong"
        if total_score >= 6:  return "Watchlist", "watch"
        return "—", "below"
    # Technical-only path (fundamentals not yet fetched for this name)
    if tech_score >= 7: return "A (Tech)",         "elite-tech"
    if tech_score >= 5: return "B (Tech)",         "strong-tech"
    if tech_score >= 4: return "Watchlist (Tech)", "watch-tech"
    return "—", "below"


# ── Public API ────────────────────────────────────────────────────────────────

def run_investment_grade_scan(progress_callback=None) -> dict:
    if (_cache["data"]
            and time.time() - _cache["ts"] < CACHE_TTL):
        return _cache["data"]

    if progress_callback:
        progress_callback(0, 100, "Loading universe…")

    stocks = _get_stocks()
    if not stocks:
        return {"error": "No stocks loaded", "stocks": []}

    nifty = _build_nifty(stocks)
    fundamentals = _load_fundamentals()
    sector_rs = _sector_rs(stocks, nifty)
    sym2sec = _sym_to_sector()

    if progress_callback:
        progress_callback(10, 100, f"Scoring {len(stocks)} stocks…")

    results = []
    total_syms = len(stocks)

    for i, (sym, df) in enumerate(stocks.items()):
        if progress_callback and i % 50 == 0:
            progress_callback(10 + int(80 * i / total_syms), 100, f"Scoring {sym}…")

        tech = _score_technical(df, nifty)
        if tech.get("skip"):
            continue
        # Hard liquidity gate — institutional accessibility required
        if tech["adtv_cr"] is None or tech["adtv_cr"] < 25:
            continue

        f_data = fundamentals.get(sym, {}) if fundamentals else {}
        funda = _score_fundamental(f_data) if f_data else {"score": 0, "max": 4, "available": False, "flags": []}

        total_score = int(tech["score"] + funda["score"])
        max_score   = int(tech["max"] + (funda["max"] if funda["available"] else 0))

        # Lower min-score threshold for technical-only stocks (max 8 vs 12)
        min_required = INVEST_MIN_SCORE if funda["available"] else 4
        if total_score < min_required:
            continue

        tier_label, tier_cls = _tier(total_score, funda["available"], tech["score"])
        sector = sym2sec.get(sym, "—")
        sec_rs = sector_rs.get(sector)

        c = df["Close"].dropna()
        results.append({
            "symbol":         sym,
            "price":          round(float(c.iloc[-1]), 2),
            "score":          total_score,
            "max_score":      max_score,
            "score_tech":     tech["score"],
            "score_funda":    funda["score"],
            "tier":           tier_label,
            "tier_cls":       tier_cls,
            "fund_available": bool(funda["available"]),
            "months_ma200":   tech["months_ma200"],
            "ma_period_used": tech["ma_period_used"],
            "months_label":   round(tech["months_ma200"] / 22.0, 1),  # ~22 trading days/month
            "stack_days":     tech["stack_days"],
            "r_squared":      tech["r_squared"],
            "max_drawdown":   tech["max_drawdown"],
            "adtv_cr":        tech["adtv_cr"],
            "pct_from_high":  tech["pct_from_high"],
            "ret_6m":         tech["ret_6m"],
            "ret_12m":        tech["ret_12m"],
            "ret_6m_excess":  tech["ret_6m_excess"],
            "ret_12m_excess": tech["ret_12m_excess"],
            "roe":            funda.get("roe"),
            "eps_growth":     funda.get("eps_growth_yoy"),
            "sales_growth":   funda.get("sales_growth_yoy"),
            "debt_eq":        funda.get("debt_to_equity"),
            "pe":             funda.get("pe"),
            "market_cap_cr":  funda.get("market_cap_cr"),
            "promoter":       funda.get("promoter"),
            "tech_flags":     tech["flags"],
            "funda_flags":    funda.get("flags", []),
            "sector":         sector,
            "sector_rs_6m":   round(sec_rs, 2) if sec_rs is not None else None,
        })

    if progress_callback:
        progress_callback(95, 100, "Sorting…")

    # Sort by score (desc), then by 12M excess return as tiebreaker
    results.sort(key=lambda r: (
        -r["score"],
        -(r["ret_12m_excess"] if r["ret_12m_excess"] is not None else r["ret_6m_excess"] or 0),
    ))

    out = {
        "stocks":         results,
        "universe_count": total_syms,
        "qualifying":     len(results),
        "tier_counts":    _tier_counts(results),
        "fund_coverage":  sum(1 for r in results if r["fund_available"]),
        "computed_at":    int(time.time()),
    }

    _cache["data"] = out
    _cache["ts"]   = time.time()

    if progress_callback:
        progress_callback(100, 100, f"Done — {len(results)} qualifying stocks")
    return out


def _tier_counts(stocks: list[dict]) -> dict[str, int]:
    """Count tiers, collapsing '(Tech)' variants into the same buckets."""
    out = {"A": 0, "B": 0, "Watchlist": 0, "A_tech": 0, "B_tech": 0}
    for s in stocks:
        t = s["tier"]
        if t == "A":              out["A"] += 1
        elif t == "B":            out["B"] += 1
        elif t == "Watchlist":    out["Watchlist"] += 1
        elif t == "A (Tech)":     out["A_tech"] += 1
        elif t == "B (Tech)":     out["B_tech"] += 1
    return out
