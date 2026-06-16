"""
Monster Growth Scanner — finds companies with accelerating EPS + revenue growth
that historically produce 3-10× moves in 1-3 years.

Scoring methodology (institutional growth investing framework):
  Factor                   Max   Rationale
  ─────────────────────── ────  ─────────────────────────────────────────────
  1. Profit growth         25    TTM profit growth > 25% = strong earnings engine
  2. Revenue growth        20    Sales growth confirms earnings not cost-cutting
  3. PEG ratio             20    Price paid relative to growth (Lynch/O'Neil)
  4. EPS acceleration      15    Consecutive quarter improvement = momentum
  5. Technical quality     10    Stage 2 + RS rank > 70 (price confirming funds)
  6. Promoter / Quality     5    ROE > 20% + promoter not reducing stake
  ──────────────────────  ────
  Total                   95    (5 pts reserved for bonus: HTF / RS-new-high)

Tiers:
  ≥ 80  → 🔥 MONSTER   (high-conviction compounding candidate)
  65-79 → 💎 STRONG     (watch closely — approaching monster quality)
  50-64 → 👀 EMERGING   (improving but not there yet)
  < 50  → ─  SKIP       (does not meet growth threshold)

Data:
  - Fundamentals: fundamentals.db (scraped from screener.in, zero live calls)
  - Price / Stage / RS: NSE bhavcopy OHLCV (local cache, zero live calls)
  - Completely offline during scan — no network requests
"""
from __future__ import annotations

import time
import threading
from typing import Optional

import numpy as np
import pandas as pd

from fundamentals import load_all_fundamentals
from data_fetcher import _weekdays_back, _download_one_day
from analysis_utils import stage_analysis, NIFTY_PROXY_SYMS, equal_weight_index
from nse_stocks import is_etf

# ── Constants ─────────────────────────────────────────────────────────────────

MIN_BARS         = 150    # need ≥150 bars for Weinstein MA150
MIN_PROFIT_PCT   = 10     # minimum TTM profit growth % to be considered
MIN_PE           = 1      # ignore negative or zero PE (loss-making)
CACHE_TTL        = 3600   # 1 hour
SCAN_WORKERS     = 8

_cache: dict = {"data": None, "ts": 0.0}
_cache_lock     = threading.Lock()

# ── Split adjustment (same algorithm as data_fetcher, portfolio, industry_groups)

def _adjust_for_splits(df):
    """Delegate to canonical analysis_utils.adjust_for_splits."""
    from analysis_utils import adjust_for_splits
    return adjust_for_splits(df)


# ── OHLCV loader ──────────────────────────────────────────────────────────────

def _load_universe_ohlcv(progress_callback=None) -> dict[str, pd.DataFrame]:
    """
    Load all Nifty Total Market 750 stocks from bhavcopy cache.
    Returns {symbol: OHLCV_df} split-adjusted, minimum MIN_BARS bars.
    Zero network calls if bhavcopy already cached.
    """
    try:
        from nse_stocks import get_universe_symbols
        universe = set(get_universe_symbols())
    except Exception:
        universe = set()

    dates  = _weekdays_back(300)
    total  = len(dates)
    frames = []

    for i, dt in enumerate(dates):
        df = _download_one_day(dt)
        if df is not None:
            frames.append(df)
        if progress_callback and i % 50 == 0:
            progress_callback(i, total, f"Loading OHLCV… {i}/{total} days")

    if not frames:
        return {}

    combined = pd.concat(frames, ignore_index=True).sort_values("Date")
    stocks   = {}

    for sym, grp in combined.groupby("Symbol"):
        if is_etf(sym): continue
        if universe and sym not in universe:
            continue
        g = grp.set_index("Date")[["Open", "High", "Low", "Close", "Volume"]]
        g = g[~g.index.duplicated(keep="last")].sort_index()
        g = _adjust_for_splits(g)
        if len(g) >= MIN_BARS:
            stocks[sym] = g

    if progress_callback:
        progress_callback(total, total, f"Loaded {len(stocks)} stocks ✓")
    return stocks


# ── Nifty proxy (return-based, splits-adjusted) ───────────────────────────────

def _build_nifty(stocks: dict) -> Optional[pd.Series]:
    """Equal-weight Nifty proxy using canonical 20-stock list from analysis_utils."""
    closes = [stocks[s]["Close"].dropna() for s in NIFTY_PROXY_SYMS
              if s in stocks and len(stocks[s]) >= 63]
    if not closes:
        return None
    combined = pd.concat(closes, axis=1).dropna(how="all")
    return equal_weight_index(combined) if len(combined) >= 20 else None


# ── Technical helpers ─────────────────────────────────────────────────────────

def _safe(v, default: float = 0.0) -> float:
    if v is None:
        return default
    try:
        f = float(v)
        return default if (np.isnan(f) or np.isinf(f)) else f
    except Exception:
        return default


_stage = stage_analysis   # TIER-3: use canonical analysis_utils implementation


def _rs_pct(c: pd.Series, nifty: Optional[pd.Series]) -> float:
    """3-month excess return vs Nifty proxy (%)."""
    if nifty is None or len(c) < 66:
        return 0.0
    try:
        combined = pd.concat([c.rename("s"), nifty.rename("n")], axis=1).dropna()
        if len(combined) < 66:
            return 0.0
        s3m = (float(combined["s"].iloc[-1]) / float(combined["s"].iloc[-63]) - 1) * 100
        n3m = (float(combined["n"].iloc[-1]) / float(combined["n"].iloc[-63]) - 1) * 100
        return round(s3m - n3m, 2)
    except Exception:
        return 0.0


def _vol_ratio(vol: pd.Series, period: int = 20) -> float:
    """Today's volume vs 20-day average."""
    if len(vol) < period + 1:
        return 1.0
    avg = float(vol.iloc[-(period + 1):-1].rolling(period, min_periods=15).mean().iloc[-1])
    return round(float(vol.iloc[-1]) / avg, 2) if avg > 0 else 1.0


def _52w_high_pct(c: pd.Series) -> float:
    """Distance from 52-week high (negative = below high)."""
    if len(c) < 10:
        return -100.0
    window = c.iloc[-252:] if len(c) >= 252 else c
    hi = float(window.max())
    cur = float(c.iloc[-1])
    return round((cur / hi - 1) * 100, 2) if hi > 0 else -100.0


# ── Monster Growth scoring ─────────────────────────────────────────────────────

def _score_stock(sym: str, df: pd.DataFrame, fund: dict,
                 nifty: Optional[pd.Series],
                 universe_r3m: dict[str, float]) -> Optional[dict]:
    """
    Score one stock across all 6 factors. Returns score dict or None if excluded.

    Exclusion criteria (hard filters before scoring):
      - Stage 4 (declining trend) → skip; monstrous moves don't start in Stage 4
      - PE ratio ≤ 0 (loss-making) → skip
      - Profit growth < MIN_PROFIT_PCT % → skip (no growth engine)
      - No fundamental data in DB → skip (unscraped stock)
    """
    try:
        c   = df["Close"].dropna()
        vol = df["Volume"].dropna()
        if len(c) < MIN_BARS:
            return None

        cur = _safe(c.iloc[-1])
        if cur <= 0:
            return None

        # ── Fundamentals ──────────────────────────────────────────────────────
        if not fund:
            return None

        # BUG-FIX: prior `profit_ttm if profit_ttm != 0.0 else ...` treated
        # legitimate 0% TTM growth (stagnation) as "missing" → fell back to 3Y CAGR
        # and silently rewarded decelerating companies with great history as MONSTER.
        # Now use raw None check so 0% means 0% (stagnant → low score).
        # Also: TTM (1-year) and 3Y CAGR (annualized) are different units; we keep
        # TTM as the primary signal and use 3Y only when TTM is truly unavailable.
        profit_ttm_raw  = fund.get("growth_ttm")
        profit_3y_raw   = fund.get("growth_3y_cagr")
        profit_yoy_raw  = fund.get("eps_growth_yoy")
        profit_ttm  = _safe(profit_ttm_raw)
        profit_3y   = _safe(profit_3y_raw)
        profit_yoy  = _safe(profit_yoy_raw)
        # Honest fallback: use TTM if it exists; else 3Y; else YoY. Track which.
        if profit_ttm_raw is not None:
            profit_gr = profit_ttm
            profit_gr_source = "TTM"
        elif profit_3y_raw is not None:
            profit_gr = profit_3y
            profit_gr_source = "3Y_CAGR"
        else:
            profit_gr = profit_yoy
            profit_gr_source = "YoY"

        # Revenue growth — same logic
        sales_ttm_raw = fund.get("sales_growth_ttm")
        sales_3y_raw  = fund.get("sales_growth_3y_cagr")
        sales_yoy_raw = fund.get("sales_growth_yoy")
        sales_ttm  = _safe(sales_ttm_raw)
        sales_3y   = _safe(sales_3y_raw)
        sales_yoy  = _safe(sales_yoy_raw)
        if sales_ttm_raw is not None:
            sales_gr = sales_ttm
        elif sales_3y_raw is not None:
            sales_gr = sales_3y
        else:
            sales_gr = sales_yoy

        pe          = _safe(fund.get("pe_ratio"))
        roe         = _safe(fund.get("roe"))
        promoter    = _safe(fund.get("promoter_holding"))
        prom_delta  = _safe(fund.get("promoter_delta"), default=None)
        eps_accel   = fund.get("eps_accel")     # QoQ acceleration (0/1/None)
        eps_accel_y = fund.get("eps_accel_yoy") # YoY acceleration (0/1/None) — new field

        # Hard filters
        if profit_gr < MIN_PROFIT_PCT:
            return None
        if pe <= MIN_PE:
            return None

        stage = _stage(c)
        if stage == 4:
            return None    # declining trend — no monstrous move starts here
        if stage == 0:
            return None    # insufficient history (<160 bars) — can't verify trend

        # ── Factor 1: Profit Growth (25 pts) ─────────────────────────────────
        # Tiered: ≥50% = 25, ≥35% = 22, ≥25% = 18, ≥15% = 12, else scaled
        if profit_gr >= 50:
            f1 = 25
        elif profit_gr >= 35:
            f1 = 22
        elif profit_gr >= 25:
            f1 = 18
        elif profit_gr >= 15:
            f1 = 12
        else:
            f1 = max(0, int(profit_gr / 15 * 12))

        # ── Factor 2: Revenue Growth (20 pts) ────────────────────────────────
        # Revenue confirms earnings quality — cost-cutting alone can't sustain
        if sales_gr >= 30:
            f2 = 20
        elif sales_gr >= 20:
            f2 = 17
        elif sales_gr >= 15:
            f2 = 14
        elif sales_gr >= 10:
            f2 = 10
        elif sales_gr >= 5:
            f2 = 6
        elif sales_gr >= 0:
            f2 = 2
        else:
            f2 = 0   # declining revenue — quality concern

        # ── Factor 3: PEG Ratio (20 pts) ─────────────────────────────────────
        # PEG = PE / TTM-growth %. Lynch: PEG ≤ 1.0 = fair; > 2.0 = expensive
        # BUG-FIX: prior code used `profit_gr` which fell back to 3Y CAGR or YoY.
        # PE is a current snapshot; the denominator MUST be a 1-year growth rate
        # for the ratio to be meaningful. Using 3Y CAGR makes decelerating stocks
        # (1Y growth 0%, 3Y CAGR 100%) look cheap (PEG = PE/100 = 0.3 → MONSTER tier).
        # Now: PEG only when we have TTM (or YoY as next-best 1-year proxy).
        if pe <= 0:
            return None

        peg = None
        f3  = 0
        # Use 1-year growth rate ONLY for PEG denominator (TTM preferred, else YoY)
        peg_growth = profit_ttm if profit_ttm_raw is not None else (
            profit_yoy if profit_yoy_raw is not None else None
        )
        if pe > 0 and peg_growth is not None and peg_growth > 0:
            raw_peg = pe / peg_growth
            if raw_peg > 0:
                # Use 3dp for very small PEGs (extreme growth like 1000%+) so we never
                # round a valid 0.003 to 0.00 (which smoke-test would flag as invalid)
                peg = round(raw_peg, 3) if raw_peg < 0.1 else round(raw_peg, 2)
            else:
                peg = None   # never store negative PEG
            if peg <= 0.5:
                f3 = 20   # deeply undervalued relative to growth
            elif peg <= 1.0:
                f3 = 18
            elif peg <= 1.5:
                f3 = 14
            elif peg <= 2.0:
                f3 = 9
            elif peg <= 3.0:
                f3 = 4
            else:
                f3 = 0    # too expensive relative to growth

        # ── Factor 4: EPS Acceleration (15 pts) ──────────────────────────────
        # Both QoQ and YoY acceleration = strongest signal
        f4 = 0
        if eps_accel == 1 and eps_accel_y == 1:
            f4 = 15    # QoQ AND YoY accelerating = strong
        elif eps_accel == 1 or eps_accel_y == 1:
            f4 = 9     # one of two = moderate
        elif eps_accel == 0 and eps_accel_y is not None:
            f4 = 2     # decelerating but we have data
        else:
            f4 = 5     # unknown (not yet scraped) — neutral

        # ── Factor 5: Technical Quality (10 pts) ─────────────────────────────
        # Stage 2 + RS rank > 70 confirms institutional accumulation
        f5 = 0
        rs_excess = _rs_pct(c, nifty)

        # Cross-sectional RS rank (percentile vs full universe)
        if universe_r3m:
            sym_r3m = (float(c.iloc[-1]) / float(c.iloc[-63]) - 1) * 100 if len(c) >= 63 else 0.0
            u_series = pd.Series(universe_r3m)
            rs_rank  = int((u_series < sym_r3m).sum() / max(len(u_series), 1) * 99)
        else:
            rs_rank = 50

        if stage == 2:
            f5 += 5
        elif stage == 1:
            f5 += 2    # basing — potential

        if rs_rank >= 80:
            f5 += 5
        elif rs_rank >= 70:
            f5 += 4
        elif rs_rank >= 60:
            f5 += 2
        elif rs_rank >= 50:
            f5 += 1

        # Bonus: near 52-week high (within 10%) = price confirming the story
        pct_hi = _52w_high_pct(c)
        bonus = 0
        if pct_hi >= -5:
            bonus += 3    # within 5% of 52W high — HTF/ATH territory
        elif pct_hi >= -10:
            bonus += 1

        # ── Factor 6: Quality / Promoter (5 pts) ─────────────────────────────
        f6 = 0
        if roe >= 25:
            f6 += 3
        elif roe >= 20:
            f6 += 2
        elif roe >= 15:
            f6 += 1

        if prom_delta is not None:
            if prom_delta > 0:
                f6 += 2    # promoters buying = skin in the game
            elif prom_delta < -2:
                f6 -= 1    # promoters reducing significantly = caution

        # ── Total Score ───────────────────────────────────────────────────────
        total = f1 + f2 + f3 + f4 + f5 + f6 + bonus
        total = max(0, min(100, total))

        # Tier label
        if total >= 80:
            tier = "MONSTER"
        elif total >= 65:
            tier = "STRONG"
        elif total >= 50:
            tier = "EMERGING"
        else:
            return None   # below threshold — not worth showing

        # Data freshness (days since last scrape)
        age_d = round((time.time() - _safe(fund.get("updated_at", 0))) / 86400, 0)

        # Volume context
        vr        = _vol_ratio(vol)
        adtv_cr   = round(float(vol.rolling(20).mean().iloc[-1] if len(vol) >= 20 else 0)
                         * cur / 1e7, 2)

        return {
            "symbol":         sym,
            "score":          total,
            "tier":           tier,
            # Fundamentals
            "profit_gr":      round(profit_gr, 1),
            "sales_gr":       round(sales_gr, 1),
            "pe":             round(pe, 1),
            "peg":            peg,
            "roe":            round(roe, 1),
            "eps_accel":      eps_accel,
            "eps_accel_yoy":  eps_accel_y,
            "promoter_pct":   round(promoter, 1),
            "promoter_delta": round(prom_delta, 2) if prom_delta is not None else None,
            "data_age_days":  int(age_d),
            # Factor breakdown
            "f1_profit":      f1,
            "f2_revenue":     f2,
            "f3_peg":         f3,
            "f4_accel":       f4,
            "f5_tech":        f5,
            "f6_quality":     f6,
            "bonus":          bonus,
            # Technical
            "price":          round(cur, 2),
            "stage":          stage,
            "rs_rank":        rs_rank,
            "rs_excess_3m":   round(rs_excess, 1),
            "pct_from_hi":    pct_hi,
            "vol_ratio":      vr,
            "adtv_cr":        adtv_cr,
        }
    except Exception:
        return None


# ── Main entry ─────────────────────────────────────────────────────────────────

def run_monster_growth_scan(progress_callback=None) -> dict:
    """
    Full Monster Growth scan across Nifty Total Market 750 universe.
    All data from local cache — zero live network calls.
    Returns dict with results list, metadata.
    """
    with _cache_lock:
        if (_cache["data"]
                and (time.time() - _cache["ts"]) < CACHE_TTL
                and _cache["data"].get("results")):
            return _cache["data"]

    if progress_callback:
        progress_callback(0, 100, "Loading OHLCV data from bhavcopy cache…")

    # 1. Load OHLCV (from cached bhavcopy files)
    stocks = _load_universe_ohlcv(progress_callback)
    if not stocks:
        return {
            "results": [], "computed_at": int(time.time()),
            "error": "No OHLCV data available — bhavcopy not yet downloaded",
            "total_scanned": 0, "fundamental_coverage": 0,
        }

    # 2. Load all fundamentals from SQLite (instant, no network)
    if progress_callback:
        progress_callback(60, 100, "Loading fundamentals from local DB…")
    all_funds = load_all_fundamentals()

    # 3. Build Nifty proxy for RS calculations
    nifty = _build_nifty(stocks)

    # 4. Pre-compute universe 3M returns for cross-sectional RS ranking
    universe_r3m: dict[str, float] = {}
    for sym, df in stocks.items():
        c = df["Close"].dropna()
        if len(c) >= 63:
            universe_r3m[sym] = (float(c.iloc[-1]) / float(c.iloc[-63]) - 1) * 100

    # 5. Score each stock
    if progress_callback:
        progress_callback(70, 100, f"Scoring {len(stocks)} stocks on 6 growth factors…")

    results = []
    for sym, df in stocks.items():
        fund = all_funds.get(sym)
        if not fund:
            continue
        r = _score_stock(sym, df, fund, nifty, universe_r3m)
        if r is not None:
            results.append(r)

    # 6. Sort by score desc, then PEG asc for equal scores
    results.sort(key=lambda x: (-x["score"], x["peg"] or 999))

    fundamental_coverage = sum(1 for s in stocks if s in all_funds)

    # P2-14: update stage transition log for every scored stock
    try:
        from stage_transitions import update_all as _stg_upd
        from analysis_utils import stage_analysis as _stg
        stg_map = {}
        for r in results:
            sym = r.get("symbol")
            if sym and sym in stocks:
                s = _stg(stocks[sym]["Close"].dropna())
                if s in (1, 2, 3, 4):
                    stg_map[sym] = s
        _stg_upd(stg_map)
    except Exception:
        pass

    out = {
        "results":               results,
        "computed_at":           int(time.time()),
        "total_scanned":         len(stocks),
        "fundamental_coverage":  fundamental_coverage,
        "monster_count":         sum(1 for r in results if r["tier"] == "MONSTER"),
        "strong_count":          sum(1 for r in results if r["tier"] == "STRONG"),
        "emerging_count":        sum(1 for r in results if r["tier"] == "EMERGING"),
    }

    with _cache_lock:
        _cache["data"] = out
        _cache["ts"]   = time.time()

    # P2-10/11: enrich AFTER cache is set so Monster's own results are
    # visible to build_consensus when it scans every scanner's _cache.
    try:
        from consensus import enrich_results, invalidate_cache as _con_inv
        _con_inv()
        enrich_results(results)   # mutates in place; out["results"] is same list
    except Exception:
        pass

    if progress_callback:
        progress_callback(100, 100,
            f"Done — {len(results)} growth stocks found ({out['monster_count']} Monster, "
            f"{out['strong_count']} Strong, {out['emerging_count']} Emerging)")
    return out


def invalidate_cache():
    """Called when new bhavcopy arrives — force rescan on next request."""
    with _cache_lock:
        _cache["data"] = None
        _cache["ts"]   = 0.0
