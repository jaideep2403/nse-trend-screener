"""
Market Breadth & Timing Dashboard
- % NSE stocks above 50-day / 200-day MA (breadth)
- New 52-week highs vs lows
- Advance / Decline ratio + Up/Down volume confirmation
- Nifty50 trend status + Stage
- Distribution Day Count on Nifty (IBD method)
- Realized volatility (India VIX proxy from Nifty 21-day stdev)
- Sector Stage-2 breadth (% of sectors in confirmed uptrend)
- Smoothed timing score (5-day MA, persisted to disk)
- Threshold backtest: past 60 days score vs forward Nifty returns

Data: NSE bhavcopy — zero Yahoo calls
"""
import json
import os
import time
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from analysis_utils import (stage_analysis, stage_label, NIFTY_PROXY_SYMS,
                            equal_weight_index, adjust_for_splits)
from data_fetcher import _weekdays_back, _download_one_day
from nse_stocks import is_etf
import regime as regime_mod
import result_cache

# ── Score history (persisted) for smoothing ────────────────────────────────────
_DATA_DIR = Path(os.getenv("DATA_DIR", os.path.dirname(__file__) or "."))
_HISTORY_PATH = _DATA_DIR / ".breadth_score_history.json"
_HISTORY_MAX_DAYS = 60

# ── Persisted breadth cache (survives restarts) ──────────────────────────────
_CACHE_PATH = _DATA_DIR / ".breadth_cache.pkl"

_NIFTY50_SYMS = NIFTY_PROXY_SYMS   # TIER-3: canonical list from analysis_utils

MIN_BARS = 60
CACHE_TTL = 1800   # 30 min (in-memory)
PERSISTED_CACHE_TTL = 6 * 3600   # 6h — disk cache valid up to 6h after computation


def _load_persisted_cache() -> dict:
    """Load breadth cache from disk if fresh enough — survives server restarts."""
    try:
        if _CACHE_PATH.exists():
            import pickle
            with open(_CACHE_PATH, "rb") as f:
                d = pickle.load(f)
            if isinstance(d, dict) and "data" in d and "ts" in d:
                age = time.time() - d["ts"]
                if age < PERSISTED_CACHE_TTL:
                    return d
    except Exception:
        pass
    return {"data": None, "ts": 0}


def _save_persisted_cache(cache: dict) -> None:
    try:
        import pickle
        _CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(_CACHE_PATH, "wb") as f:
            pickle.dump(cache, f)
    except Exception:
        pass


# Load at import — header_data picks up authoritative trend immediately on boot
_cache = _load_persisted_cache()


# ── Data loader ───────────────────────────────────────────────────────────────

def _load_all_stocks(progress_callback=None) -> dict[str, pd.DataFrame]:
    dates  = _weekdays_back(400)
    total  = len(dates)
    frames = []
    for i, dt in enumerate(dates):
        df = _download_one_day(dt)
        if df is not None:
            frames.append(df)
        if progress_callback and i % 50 == 0:
            progress_callback(i, total, f"Loading bhavcopy… {i}/{total} days")
    if not frames:
        return {}
    # BUG-FIX: was loading EVERY EQ symbol incl. illiquid penny stocks → "% above MA50"
    # got dominated by ~1500 thinly traded names; other tabs used the Nifty Total Market
    # 750 universe → cross-tab mismatch. Now consistent: breadth on the same 750-stock
    # NSE Total Market universe used by every other scanner.
    try:
        from nse_stocks import get_universe_symbols
        _universe = set(get_universe_symbols())
    except Exception:
        _universe = set()
    combined = pd.concat(frames, ignore_index=True).sort_values("Date")
    stocks = {}
    for sym, grp in combined.groupby("Symbol"):
        if is_etf(sym): continue
        if _universe and sym not in _universe:
            continue
        g = grp.set_index("Date")[["Open", "High", "Low", "Close", "Volume"]]
        g = g[~g.index.duplicated(keep="last")].sort_index()
        # BUG-FIX (2026-06-10): bhavcopy is unadjusted — without this a 1:1
        # bonus registers as a -50% "decline" (wrong A/D + down-volume), and
        # the stock then sits ~50% below its fake "52W high" for a year,
        # biasing new-highs/lows, %>MA50 and %>MA200 bearish.
        g = adjust_for_splits(g)
        if len(g) >= MIN_BARS:
            stocks[sym] = g
    return stocks


def _build_nifty(stocks: dict) -> pd.Series | None:
    # TODO BUG-009: This is one of four separate Nifty proxy implementations across
    # the codebase (market_breadth.py, header_data.py, sector_analysis.py,
    # industry_groups.py, portfolio.py). They use slightly different constituent lists
    # and scaling. Future work: consolidate into a shared nifty_proxy module.
    closes = [stocks[s]["Close"].dropna() for s in _NIFTY50_SYMS
              if s in stocks and len(stocks[s]) >= 63]
    if not closes:
        return None
    combined = pd.concat(closes, axis=1).dropna(how="all")
    bench = equal_weight_index(combined)
    return bench if len(bench) >= 20 else None


# ── Breadth metrics ───────────────────────────────────────────────────────────

def _compute_breadth(stocks: dict, smooth_days: int = 5) -> dict:
    """Compute breadth metrics across all stocks.

    BUG-FIX (2026-06-10):
    - A/D and up/down volume are ALSO aggregated per-session for the last
      `smooth_days` sessions: a single expiry-day distortion used to move the
      composite score by up to 3 of 15 points. The smoothed ratios are what
      the score consumes; today's raw values remain for display.
    - Sentinels removed: a day with zero decliners now yields ratio=None
      (capped at 10.0 when a numeric is needed), not a magic 9.9.
    """
    above50 = above200 = new_highs = new_lows = total = 0
    # per-DATE aggregates over the recent window: date -> [adv, dec, upvol, dnvol]
    daily: dict = {}

    for sym, df in stocks.items():
        cl = df["Close"].dropna()
        if len(cl) < 22:
            continue
        total += 1
        cur = float(cl.iloc[-1])

        # MA breadth
        if len(cl) >= 52:
            ma50 = float(cl.rolling(50).mean().iloc[-1])
            if cur > ma50:
                above50 += 1
        if len(cl) >= 210:
            ma200 = float(cl.rolling(200).mean().iloc[-1])
            if cur > ma200:
                above200 += 1

        # 52-week highs/lows
        lookback = min(252, len(cl))
        hi52 = float(cl.iloc[-lookback:].max())
        lo52 = float(cl.iloc[-lookback:].min())
        if cur >= hi52 * 0.995:
            new_highs += 1
        if cur <= lo52 * 1.005:
            new_lows += 1

        # Per-session A/D + up/down volume for the last `smooth_days`+1 bars
        v = df["Volume"].dropna() if "Volume" in df.columns else None
        tail = cl.iloc[-(smooth_days + 1):]
        chg = tail.diff().iloc[1:]
        for dt, c in chg.items():
            rec = daily.setdefault(dt, [0, 0, 0.0, 0.0])
            vol = 0.0
            if v is not None and dt in v.index:
                try:
                    vol = float(v.loc[dt])
                except Exception:
                    vol = 0.0
            if c > 0:
                rec[0] += 1; rec[2] += vol
            elif c < 0:
                rec[1] += 1; rec[3] += vol

    def _ratio(up, dn, cap=10.0):
        if dn <= 0:
            return cap if up > 0 else None
        return round(min(cap, up / dn), 2)

    # Today = most recent session; smoothed = mean of per-session ratios
    sessions = sorted(daily.keys())[-smooth_days:]
    if sessions:
        last = daily[sessions[-1]]
        up_today, down_today = last[0], last[1]
        up_volume, down_volume = last[2], last[3]
        adr = _ratio(up_today, down_today)
        vol_ratio = _ratio(up_volume, down_volume)
        adr_vals = [_ratio(daily[d][0], daily[d][1]) for d in sessions]
        vr_vals  = [_ratio(daily[d][2], daily[d][3]) for d in sessions]
        adr_vals = [x for x in adr_vals if x is not None]
        vr_vals  = [x for x in vr_vals if x is not None]
        adr_5d = round(float(np.mean(adr_vals)), 2) if adr_vals else None
        vol_ratio_5d = round(float(np.mean(vr_vals)), 2) if vr_vals else None
    else:
        up_today = down_today = 0
        up_volume = down_volume = 0.0
        adr = vol_ratio = adr_5d = vol_ratio_5d = None

    pct50  = round(above50  / total * 100, 1) if total else 0
    pct200 = round(above200 / total * 100, 1) if total else 0

    return {
        "total_stocks":     total,
        "pct_above_50ma":   pct50,
        "pct_above_200ma":  pct200,
        "new_highs":        new_highs,
        "new_lows":         new_lows,
        "advance":          up_today,
        "decline":          down_today,
        "adv_decl_ratio":   adr,
        "adv_decl_ratio_5d": adr_5d,                       # smoothed — feeds the score
        "up_volume":        round(up_volume / 1e6, 2),     # millions
        "down_volume":      round(down_volume / 1e6, 2),
        "up_down_vol_ratio": vol_ratio,                    # > 1 = bullish vol
        "up_down_vol_ratio_5d": vol_ratio_5d,              # smoothed — feeds the score
    }


# Distribution-day counting and FTD detection now live in regime.py — the
# canonical module shared with the Edge tab, so the two tabs can never report
# different D-day counts or regimes. The old local implementation fell back to
# a price-only count (any -0.5% day = "distribution", no volume condition) on
# the synthetic proxy — not the IBD method at all.
#
# Key time-ordering guard in regime.detect_ftd (reproduced here for auditability):
#   if trough_loc <= peak_loc:
#       return out   # no decline-then-rally structure; skip to avoid false FTDs


def _label_for_score(score: int | float) -> tuple[str, str]:
    """Map composite score (0-15 with rebalanced inputs) to label + class.

    Labels are deliberately decisive single words / short phrases so a new
    user can act on them without reading a glossary. Previous labels used
    hedge words ("Caution", "Mixed", "Correction Risk") that read more
    like analyst opinion than market state.
    """
    if score >= 12:
        return "Bull Market", "pos"      # broad uptrend with confirmation
    if score >= 9:
        return "Uptrend",     "pos"      # uptrend, some weakness underneath
    if score >= 5:
        return "Sideways",    "neutral"  # range-bound — no clear direction
    if score >= 2:
        return "Correction",  "neg"      # market under pressure
    return "Bear Market",     "neg"      # broad-based selling


def _market_timing_signal(breadth: dict, dist_days: int, nifty_stage: int,
                          sector_stage2_pct: float | None = None,
                          smoothed_score: float | None = None) -> dict:
    """
    Composite market timing score (0-13).
    Original 5 inputs → 0-10. New inputs (volume + sector breadth) add up to +3.
    """
    p50  = breadth["pct_above_50ma"]
    p200 = breadth["pct_above_200ma"]
    hl_ratio = breadth["new_highs"] / (breadth["new_highs"] + breadth["new_lows"] + 1)
    # Score consumes the 5-day SMOOTHED volume ratio — one distorted session
    # (expiry, block deals) used to swing the composite by up to 3 points.
    vol_ratio = breadth.get("up_down_vol_ratio_5d")
    if vol_ratio is None:
        vol_ratio = breadth.get("up_down_vol_ratio") or 1.0

    score = 0
    # ── Above MA50 % (0–3) — granular: 72% > 60% as it should be ──
    if   p50 >= 75: score += 3
    elif p50 >= 60: score += 2
    elif p50 >= 40: score += 1
    # ── Above MA200 % (0–2) ──
    if   p200 >= 55: score += 2
    elif p200 >= 35: score += 1
    # ── High-Low ratio (0–2) ──
    if   hl_ratio >= 0.7: score += 2
    elif hl_ratio >= 0.5: score += 1
    # ── Distribution days (0–2) ──
    if   dist_days <= 3: score += 2
    elif dist_days <= 5: score += 1
    # ── Nifty stage (0–2) ──
    if   nifty_stage == 2: score += 2
    elif nifty_stage == 1: score += 1
    # ── Volume confirmation today (0–2) ──
    if   vol_ratio >= 1.5: score += 2
    elif vol_ratio >= 1.0: score += 1
    # ── Sector Stage-2 breadth (0–2) — granular ──
    if sector_stage2_pct is not None:
        if   sector_stage2_pct >= 60: score += 2
        elif sector_stage2_pct >= 30: score += 1

    status, cls = _label_for_score(score)

    # Use smoothed score for label if available (reduces flip-flopping)
    smoothed_status, smoothed_cls = (None, None)
    if smoothed_score is not None:
        smoothed_status, smoothed_cls = _label_for_score(smoothed_score)

    # ── DEPLOYMENT GATE — evidence-based (breadth_validation.py, 166
    # walk-forward samples 2024-06 → 2026-05, fwd NIFTYBEES paths):
    #   score ≥ 12 → calmest forward paths (avg 20d DD -1.8%, only 25%
    #                of samples saw a >3% drawdown)        → risk_on
    #   score ≤ 4  → toxic tails (Bear 0-1: avg DD -4.5%, 57% saw >3% DD,
    #                avg 20d return -2.16%)                → risk_off
    #   in between → NO measurable edge either way (buckets non-monotonic,
    #                IC ~0) — gate stays neutral; stock selection decides.
    # The gate uses the SMOOTHED score so one session can't flip deployment.
    _gate_score = smoothed_score if smoothed_score is not None else score
    if _gate_score >= 12:
        gate = "risk_on"
    elif _gate_score <= 4:
        gate = "risk_off"
    else:
        gate = "neutral"

    return {
        "gate":             gate,
        "status":           status,
        "score":            score,
        "max":              15,
        "cls":              cls,
        "smoothed_score":   round(smoothed_score, 2) if smoothed_score is not None else None,
        "smoothed_status":  smoothed_status,
        "smoothed_cls":     smoothed_cls,
        "vol_confirmation": "Bullish" if vol_ratio >= 1.5 else ("Mild" if vol_ratio >= 1.0 else "Bearish"),
    }


# ── VIX proxy: realized volatility from Nifty ─────────────────────────────────

def _realized_vol(nifty: pd.Series, window: int = 21) -> dict | None:
    """
    Annualized realized volatility on Nifty (21-day rolling stdev × √252).
    TIER-3 RENAME: was _realized_vix — misleading because VIX is implied vol
    (forward-looking options pricing). This is realized vol (backward-looking
    historical stdev), a fundamentally different concept.
    """
    if nifty is None or len(nifty) < window + 1:
        return None
    rets = nifty.pct_change().dropna()
    realized = float(rets.iloc[-window:].std() * np.sqrt(252) * 100)
    rv_label = (
        "Low (Complacency)"  if realized < 12 else
        "Normal"              if realized < 18 else
        "Elevated"            if realized < 25 else
        "High (Fear)"
    )
    return {
        "realized_vol": round(realized, 2),
        "label":        rv_label,
        "window":       window,
    }


# ── Sector Stage-2 breadth ────────────────────────────────────────────────────

def _sector_stage2_breadth(stocks: dict) -> dict | None:
    """
    For each industry group, compute the median stage of its members.
    Return % of groups whose median is Stage 2 (confirmed uptrend).
    """
    try:
        from industry_groups import INDUSTRY_GROUPS
    except Exception:
        return None
    if not stocks:
        return None

    sym_to_stage = {}
    for sym, df in stocks.items():
        cl = df.get("Close")
        if cl is None or len(cl.dropna()) < 60:
            continue
        try:
            sym_to_stage[sym] = stage_analysis(cl.dropna())
        except Exception:
            continue
    if not sym_to_stage:
        return None

    total = stage2 = stage1 = 0
    sector_stages: list[tuple[str, int, int]] = []
    for group, members in INDUSTRY_GROUPS.items():
        stages = [sym_to_stage[s] for s in members if s in sym_to_stage]
        if len(stages) < 3:
            continue
        median_stage = int(np.median(stages))
        total += 1
        if median_stage == 2: stage2 += 1
        if median_stage == 1: stage1 += 1
        sector_stages.append((group, median_stage, len(stages)))

    if total == 0:
        return None
    pct = round(stage2 / total * 100, 1)
    return {
        "total_groups":     total,
        "groups_in_stage2": stage2,
        "groups_in_stage1": stage1,
        "pct_stage2":       pct,
        "label":           "Confirmed" if pct >= 60 else "Partial" if pct >= 40 else "Weak",
    }


# ── Score history / smoothing ────────────────────────────────────────────────

def _load_score_history() -> list[dict]:
    try:
        if _HISTORY_PATH.exists():
            return json.loads(_HISTORY_PATH.read_text())
    except Exception:
        pass
    return []


def _save_score_history(history: list[dict]) -> None:
    try:
        _HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
        _HISTORY_PATH.write_text(json.dumps(history[-_HISTORY_MAX_DAYS:]))
    except Exception:
        pass


def _record_today_score(score: int, trading_date: str | None = None) -> float:
    """Append the score keyed to the BHAVCOPY trading date, return 5-day MA.

    BUG-FIX (2026-06-10): was keyed to wall-clock date.today() — weekend scans
    created entries for non-trading days and holidays left gaps, so the
    "5-day MA" was not 5 trading days. Re-scans of the same trading day still
    replace (idempotent)."""
    key = trading_date or date.today().isoformat()
    history = _load_score_history()
    history = [h for h in history if h.get("date") != key]
    history.append({"date": key, "score": int(score)})
    history.sort(key=lambda h: h["date"])
    _save_score_history(history)
    last5 = [h["score"] for h in history[-5:]]
    return float(sum(last5) / len(last5)) if last5 else float(score)


# ── Backtest: historical scores vs forward Nifty returns ─────────────────────

def _historical_score_for_day(stocks: dict, day_idx: int, nifty: pd.Series) -> int | None:
    """Compute breadth score AS OF a past day (negative offset from today)."""
    p50_n = p200_n = highs = lows = adv = decl = total = 0
    for sym, df in stocks.items():
        cl = df["Close"].dropna()
        if len(cl) < 60 or abs(day_idx) > len(cl) - 22:
            continue
        slice_cl = cl.iloc[: day_idx if day_idx < 0 else None]
        if len(slice_cl) < 22:
            continue
        total += 1
        cur = float(slice_cl.iloc[-1])
        prev = float(slice_cl.iloc[-2])
        if len(slice_cl) >= 52:
            ma50 = float(slice_cl.iloc[-50:].mean())
            if cur > ma50: p50_n += 1
        if len(slice_cl) >= 210:
            ma200 = float(slice_cl.iloc[-200:].mean())
            if cur > ma200: p200_n += 1
        lookback = min(252, len(slice_cl))
        hi52 = float(slice_cl.iloc[-lookback:].max())
        lo52 = float(slice_cl.iloc[-lookback:].min())
        if cur >= hi52 * 0.995: highs += 1
        if cur <= lo52 * 1.005: lows += 1
        if cur > prev: adv += 1
        elif cur < prev: decl += 1

    if total < 50:
        return None
    p50 = p50_n / total * 100
    p200 = p200_n / total * 100
    hl_ratio = highs / (highs + lows + 1)

    # Nifty stage as of that day
    if abs(day_idx) > len(nifty) - 1:
        return None
    nifty_slice = nifty.iloc[: day_idx if day_idx < 0 else None]
    n_stage = stage_analysis(nifty_slice) if len(nifty_slice) >= 60 else 0
    # Distribution days
    pct_chg = nifty_slice.pct_change().iloc[-25:]
    dist = int((pct_chg < -0.002).sum())

    # Match the LIVE scoring scale (0-15) so backtest bucket labels are
    # directly comparable to today's score. Old version used the obsolete
    # 0-10 scale and bucket labels "Bull (8-10)" which then sat next to a
    # live score of e.g. 11 — apples vs oranges.
    s = 0
    # Above MA50% (0-3) — granular tier matches live
    if   p50 >= 75: s += 3
    elif p50 >= 60: s += 2
    elif p50 >= 40: s += 1
    # Above MA200% (0-2)
    if   p200 >= 55: s += 2
    elif p200 >= 35: s += 1
    # High-low ratio (0-2)
    if   hl_ratio >= 0.7: s += 2
    elif hl_ratio >= 0.5: s += 1
    # Distribution days (0-2)
    if   dist <= 3: s += 2
    elif dist <= 5: s += 1
    # Nifty stage (0-2)
    if   n_stage == 2: s += 2
    elif n_stage == 1: s += 1
    # Volume confirmation and sector breadth (the two new live buckets) aren't
    # cheap to compute historically, so they default to 0 here. Live can score
    # up to 15; historical caps at 11 — but the buckets re-calibrate (below)
    # against the live scale.
    return s


def _backtest_thresholds(stocks: dict, nifty: pd.Series, lookback_days: int = 60) -> dict:
    """
    For the past `lookback_days`, compute breadth score and forward
    5/10/20 day Nifty returns. Returns aggregated stats per score bucket.
    """
    if nifty is None or len(nifty) < lookback_days + 25:
        return {"error": "insufficient_data", "by_bucket": []}

    samples: list[dict] = []
    # Walk from -lookback_days to -25 (need 20d forward return + buffer)
    for d in range(-lookback_days, -20):
        score = _historical_score_for_day(stocks, d, nifty)
        if score is None:
            continue
        # Forward returns
        try:
            base = float(nifty.iloc[d])
            r5  = round((float(nifty.iloc[d + 5])  / base - 1) * 100, 2)
            r10 = round((float(nifty.iloc[d + 10]) / base - 1) * 100, 2)
            r20 = round((float(nifty.iloc[d + 20]) / base - 1) * 100, 2)
            samples.append({"score": score, "r5": r5, "r10": r10, "r20": r20})
        except Exception:
            continue

    if not samples:
        return {"error": "no_samples", "by_bucket": []}

    # Bucket on the same scale used for live scoring. Historical max is 11
    # (the two live-only inputs — volume confirmation and sector Stage-2
    # breadth — aren't computed retrospectively), and live max is 15. Labels
    # mirror _label_for_score() so the backtest row "Sideways (5-8)" reads
    # the same way as the live header "Sideways".
    def _bucket(s):
        if s >= 9: return "Bull (9+)"
        if s >= 5: return "Sideways (5-8)"
        if s >= 2: return "Correction (2-4)"
        return "Bear (0-1)"

    buckets: dict[str, list[dict]] = {}
    for s in samples:
        buckets.setdefault(_bucket(s["score"]), []).append(s)

    out = []
    for label, group in buckets.items():
        out.append({
            "bucket":      label,
            "n":           len(group),
            "avg_5d":      round(np.mean([g["r5"]  for g in group]), 2),
            "avg_10d":     round(np.mean([g["r10"] for g in group]), 2),
            "avg_20d":     round(np.mean([g["r20"] for g in group]), 2),
            "win_rate_10d": round(sum(1 for g in group if g["r10"] > 0) / len(group) * 100, 1),
        })
    # Sort buckets best-regime first. Labels must match _bucket() exactly —
    # they were updated to the 0-15 scale (Bull 9+, Caution 5-8, Correction 2-4,
    # Bear 0-1) but this sort line still referenced the obsolete 0-10 labels,
    # raising ValueError ('Bear (0-1) is not in list') and erroring the whole
    # endpoint.
    _BUCKET_ORDER = ["Bull (9+)", "Sideways (5-8)", "Correction (2-4)", "Bear (0-1)"]
    out.sort(key=lambda x: -_BUCKET_ORDER.index(x["bucket"]) if x["bucket"] in _BUCKET_ORDER else 1)
    return {"by_bucket": out, "total_samples": len(samples), "lookback_days": lookback_days}


# ── Historical breadth trend (last 10 weeks) ─────────────────────────────────

def _breadth_trend(stocks: dict) -> list[dict]:
    """
    Returns weekly % above 50-day MA for last 10 weeks.
    Used for a sparkline / trend chart.
    """
    weeks = []
    for w in range(10, 0, -1):
        above = total = 0
        for sym, df in stocks.items():
            cl = df["Close"].dropna()
            if len(cl) < 55 + w * 5:
                continue
            idx   = -(w * 5)
            cur   = float(cl.iloc[idx])
            ma50  = float(cl.iloc[:idx].rolling(50).mean().iloc[-1]) if len(cl[:idx]) >= 50 else 0
            if ma50 > 0:
                total += 1
                if cur > ma50:
                    above += 1
        pct = round(above / total * 100, 1) if total else 0
        weeks.append({"week": f"-{w}w", "pct_above_50": pct})
    return weeks


# ── Top 10 stocks hitting 52-week highs today ─────────────────────────────────

def _new_high_stocks(stocks: dict) -> list[dict]:
    results = []
    for sym, df in stocks.items():
        cl  = df["Close"].dropna()
        vol = df["Volume"].dropna()
        if len(cl) < 63:
            continue
        cur  = float(cl.iloc[-1])
        lookback = min(252, len(cl))
        hi52 = float(cl.iloc[-lookback:].max())
        if cur < hi52 * 0.995:
            continue
        avg_vol = float(vol.rolling(20).mean().iloc[-1]) if len(vol) >= 20 else 0
        adtv_cr = round(avg_vol * cur / 1e7, 2)
        if adtv_cr < 0.5:
            continue
        vr = round(float(vol.iloc[-1]) / avg_vol, 2) if avg_vol > 0 else 1.0
        r3m = round((cur / float(cl.iloc[-63]) - 1) * 100, 2) if len(cl) >= 63 else 0.0
        results.append({
            "symbol":    sym,
            "price":     round(cur, 2),
            "adtv_cr":   adtv_cr,
            "vol_ratio": vr,
            "r3m":       r3m,
            "pct_ath":   0.0,
        })
    results.sort(key=lambda x: x["vol_ratio"], reverse=True)
    # Return the FULL list (was capped at [:20], which made the header count pin at
    # a fake constant 20 on any day with ≥20 new highs). Callers truncate for the
    # scrolling tape but report len(full) as the true count.
    return results


# ── Follow-Through Day (IBD) ──────────────────────────────────────────────────

# FTD detection now lives in regime.py (canonical, volume-confirmed). The old
# local version never checked volume (a +1.7% day on FALLING volume is not a
# follow-through day) and its "ftd_in_window" flag matched any +1.7% day in
# the last 10 sessions regardless of the day-4-7 rule it claimed to enforce.


# ── Main entry ────────────────────────────────────────────────────────────────

def run_market_breadth(progress_callback=None) -> dict:
    if (_cache["data"]
            and time.time() - _cache["ts"] < CACHE_TTL):
        return _cache["data"]

    _disk = result_cache.get("breadth")
    if _disk is not None:
        _cache["data"] = _disk
        _cache["ts"] = time.time()
        return _disk

    stocks = _load_all_stocks(progress_callback)
    if not stocks:
        return {"error": "No data available"}

    if progress_callback:
        progress_callback(0, 1, "Computing market breadth…")

    # ── Benchmark: real NIFTYBEES (cap-weighted Nifty 50). The 20-stock
    # equal-weight proxy is only a fallback — it diverges from the actual
    # index and silently skewed every Nifty-based signal on this tab.
    proxy = _build_nifty(stocks)
    bench = None
    bench_source = "proxy"
    try:
        from benchmark import get_benchmark
        bench = get_benchmark(days=420)
        if bench is not None and len(bench) >= 60:
            bench_source = "NIFTYBEES"
        else:
            bench = None
    except Exception:
        bench = None
    nifty = bench if bench is not None else proxy

    # ── Canonical regime (shared with the Edge tab via regime.py):
    # volume-confirmed D-days + volume-confirmed FTD on the SAME inputs.
    market_vol = regime_mod.build_market_volume(stocks)
    regime     = regime_mod.market_regime(nifty, market_vol)
    dist_days  = regime["dday_count"]
    # UI-compatible FTD block (ftd_in_window key kept for the existing cards)
    _f = regime["ftd"]
    ftd = {
        "rally_attempt":      _f.get("rally_attempt", False),
        "days_since_trough":  _f.get("days_since_trough") or 0,
        "trough_to_now_pct":  _f.get("trough_to_now_pct") or 0.0,
        "ftd_today":          _f.get("ftd_today", False),
        "today_chg_pct":      _f.get("today_chg_pct") or 0.0,
        "ftd_in_window":      _f.get("ftd_active", False),
        "ftd_day":            _f.get("ftd_day"),
    }

    breadth     = _compute_breadth(stocks)
    nifty_stage = stage_analysis(nifty) if nifty is not None else 0
    sector_b    = _sector_stage2_breadth(stocks)
    sector_pct  = sector_b["pct_stage2"] if sector_b else None
    new_highs_all = _new_high_stocks(stocks)
    new_highs_l   = new_highs_all[:60]          # scrolling tape (display only)
    new_highs_ct  = len(new_highs_all)          # TRUE count for the header
    vix_proxy   = _realized_vol(nifty) if nifty is not None else None

    # Trading date of the data (staleness guard — score history keys to this)
    try:
        from data_fetcher import _latest_bhavcopy_date as _lbd
        _bd = _lbd()
        bhavcopy_date = str(_bd) if _bd else None
    except Exception:
        bhavcopy_date = None

    # Compute timing first WITHOUT smoothed (to get raw score), then record + smooth
    raw_timing  = _market_timing_signal(breadth, dist_days, nifty_stage, sector_pct)
    smoothed_5d = _record_today_score(raw_timing["score"], bhavcopy_date)
    timing      = _market_timing_signal(breadth, dist_days, nifty_stage, sector_pct, smoothed_5d)

    # Score-bucket evidence: prefer the offline walk-forward calibration
    # (breadth_validation.py → .breadth_calibration.json — ~2yr, full
    # 7-component point-in-time scores). The in-process 60-day mini-backtest
    # (40 samples, one regime, 0-11 partial scale) is only a fallback.
    backtest = None
    try:
        _calib_path = _DATA_DIR / ".breadth_calibration.json"
        if _calib_path.exists():
            backtest = json.loads(_calib_path.read_text())
            backtest["source"] = "walk-forward calibration"
    except Exception:
        backtest = None
    if backtest is None:
        backtest = _backtest_thresholds(stocks, nifty) if nifty is not None else {"by_bucket": []}
        backtest["source"] = "60-day mini-backtest (run breadth_validation.py for the full version)"

    # BUG-007 FIX: _breadth_trend() was dead code — now called and included in output
    breadth_trend = _breadth_trend(stocks)

    # Nifty returns
    nifty_r1m = nifty_r3m = nifty_r6m = None
    if nifty is not None and len(nifty) > 22:
        nifty_r1m = round((float(nifty.iloc[-1]) / float(nifty.iloc[-21]) - 1) * 100, 2)
    if nifty is not None and len(nifty) >= 63:
        nifty_r3m = round((float(nifty.iloc[-1]) / float(nifty.iloc[-63]) - 1) * 100, 2)
    if nifty is not None and len(nifty) >= 126:
        nifty_r6m = round((float(nifty.iloc[-1]) / float(nifty.iloc[-126]) - 1) * 100, 2)

    out = {
        "breadth":       breadth,
        "dist_days":     dist_days,
        "regime":        {k: regime[k] for k in
                          ("regime", "label", "advice", "color",
                           "dday_count", "ftd_active", "ftd_day", "details")},
        "bench_source":  bench_source,      # "NIFTYBEES" | "proxy"
        "bhavcopy_date": bhavcopy_date,     # staleness guard
        "nifty_stage":   nifty_stage,
        "nifty_stage_label": stage_label(nifty_stage),
        "nifty_r1m":     nifty_r1m,
        "nifty_r3m":     nifty_r3m,
        "nifty_r6m":     nifty_r6m,
        "timing":        timing,
        "ftd":           ftd,
        "new_highs_list": new_highs_l,      # top-60 for the tape
        "new_highs_count": new_highs_ct,    # TRUE count of 52w-high makers today
        "sector_breadth": sector_b,        # sector Stage-2 stats
        "vix_proxy":      vix_proxy,        # realized vol (NIFTYBEES-based)
        "backtest":       backtest,         # walk-forward calibration (or fallback)
        "breadth_trend":  breadth_trend,   # weekly % above 50MA sparkline
        "computed_at":   int(time.time()),
    }

    _cache["data"] = out
    _cache["ts"]   = time.time()
    _save_persisted_cache(_cache)   # persist to disk so next restart shows immediately
    result_cache.put("breadth", out)

    if progress_callback:
        progress_callback(1, 1, "Market breadth computed")
    return out
