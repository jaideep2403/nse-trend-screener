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

from analysis_utils import stage_analysis, stage_label
from data_fetcher import _weekdays_back, _download_one_day

# ── Score history (persisted) for smoothing ────────────────────────────────────
_DATA_DIR = Path(os.getenv("DATA_DIR", os.path.dirname(__file__) or "."))
_HISTORY_PATH = _DATA_DIR / ".breadth_score_history.json"
_HISTORY_MAX_DAYS = 60

# ── Persisted breadth cache (survives restarts) ──────────────────────────────
_CACHE_PATH = _DATA_DIR / ".breadth_cache.pkl"

_NIFTY50_SYMS = [
    "RELIANCE", "TCS", "HDFCBANK", "ICICIBANK", "INFOSYS",
    "HINDUNILVR", "SBIN", "BHARTIARTL", "ITC", "KOTAKBANK",
    "AXISBANK", "WIPRO", "HCLTECH", "MARUTI", "BAJFINANCE",
    "TITAN", "NTPC", "POWERGRID", "NESTLEIND", "SUNPHARMA",
    "ADANIENT", "ADANIPORTS", "LTIM", "BAJAJFINSV", "EICHERMOT",
    "TECHM", "TATASTEEL", "JSWSTEEL", "ULTRACEMCO", "GRASIM",
]

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
    combined = pd.concat(frames, ignore_index=True).sort_values("Date")
    stocks = {}
    for sym, grp in combined.groupby("Symbol"):
        g = grp.set_index("Date")[["Open", "High", "Low", "Close", "Volume"]]
        g = g[~g.index.duplicated(keep="last")].sort_index()
        if len(g) >= MIN_BARS:
            stocks[sym] = g
    return stocks


def _build_nifty(stocks: dict) -> pd.Series | None:
    closes = []
    for sym in _NIFTY50_SYMS:
        df = stocks.get(sym)
        if df is not None and len(df) >= 63:
            closes.append(df["Close"].dropna())
    if not closes:
        return None
    combined = pd.concat(closes, axis=1).dropna(how="all")
    bench = combined.mean(axis=1)
    return bench if len(bench) >= 20 else None


# ── Breadth metrics ───────────────────────────────────────────────────────────

def _compute_breadth(stocks: dict) -> dict:
    """Compute breadth metrics across all stocks (incl. volume confirmation)."""
    above50 = above200 = new_highs = new_lows = up_today = down_today = total = 0
    up_volume = 0.0
    down_volume = 0.0

    for sym, df in stocks.items():
        cl = df["Close"].dropna()
        if len(cl) < 22:
            continue
        total += 1
        cur = float(cl.iloc[-1])
        prev = float(cl.iloc[-2]) if len(cl) >= 2 else cur

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

        # Today's volume — segregate by up vs down close
        vol_today = 0.0
        if "Volume" in df.columns:
            v = df["Volume"].dropna()
            if len(v):
                try:
                    vol_today = float(v.iloc[-1])
                except Exception:
                    vol_today = 0.0

        # Advance/Decline today + volume
        if cur > prev:
            up_today += 1
            up_volume += vol_today
        elif cur < prev:
            down_today += 1
            down_volume += vol_today

    pct50  = round(above50  / total * 100, 1) if total else 0
    pct200 = round(above200 / total * 100, 1) if total else 0
    adr    = round(up_today / down_today, 2) if down_today else 9.9
    vol_ratio = round(up_volume / down_volume, 2) if down_volume else 9.9

    return {
        "total_stocks":     total,
        "pct_above_50ma":   pct50,
        "pct_above_200ma":  pct200,
        "new_highs":        new_highs,
        "new_lows":         new_lows,
        "advance":          up_today,
        "decline":          down_today,
        "adv_decl_ratio":   adr,
        "up_volume":        round(up_volume / 1e6, 2),     # millions
        "down_volume":      round(down_volume / 1e6, 2),
        "up_down_vol_ratio": vol_ratio,                     # > 1 = bullish vol
    }


def _distribution_days(nifty: pd.Series) -> int:
    """
    Count Nifty distribution days (close DOWN on higher volume than prior day)
    in the last 25 sessions. IBD counts ≥ 5 as a market under pressure.
    Note: we approximate with price alone since we don't have Nifty futures volume.
    We use the equal-weighted index change directly.
    """
    try:
        if len(nifty) < 26:
            return 0
        pct_chg  = nifty.pct_change().iloc[-25:]
        down_big = (pct_chg < -0.002).sum()   # > 0.2% drop = distribution day
        return int(down_big)
    except Exception:
        return 0


def _label_for_score(score: int | float) -> tuple[str, str]:
    """Map composite score (0-13 with new inputs) to label + class."""
    if score >= 10:
        return "Bull Market", "pos"
    if score >= 6:
        return "Uptrend (Caution)", "neutral"
    if score >= 3:
        return "Correction", "neg"
    return "Bear Market", "neg"


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
    vol_ratio = breadth.get("up_down_vol_ratio", 1.0)

    score = 0
    # ── Original 5 inputs (0–10) ──
    if p50  >= 60: score += 2
    elif p50 >= 40: score += 1
    if p200 >= 55: score += 2
    elif p200 >= 35: score += 1
    if hl_ratio >= 0.7: score += 2
    elif hl_ratio >= 0.5: score += 1
    if dist_days <= 3:   score += 2
    elif dist_days <= 5: score += 1
    if nifty_stage == 2:  score += 2
    elif nifty_stage == 1: score += 1

    # ── New: Volume confirmation (0–2) ──
    if vol_ratio >= 1.5:    score += 2  # strongly bullish vol
    elif vol_ratio >= 1.0:  score += 1  # mildly bullish vol

    # ── New: Sector Stage-2 breadth (0–1) ──
    if sector_stage2_pct is not None:
        if sector_stage2_pct >= 60: score += 1   # majority of sectors trending

    status, cls = _label_for_score(score)

    # Use smoothed score for label if available (reduces flip-flopping)
    smoothed_status, smoothed_cls = (None, None)
    if smoothed_score is not None:
        smoothed_status, smoothed_cls = _label_for_score(smoothed_score)

    return {
        "status":           status,
        "score":            score,
        "max":              13,
        "cls":              cls,
        "smoothed_score":   round(smoothed_score, 2) if smoothed_score is not None else None,
        "smoothed_status":  smoothed_status,
        "smoothed_cls":     smoothed_cls,
        "vol_confirmation": "Bullish" if vol_ratio >= 1.5 else ("Mild" if vol_ratio >= 1.0 else "Bearish"),
    }


# ── VIX proxy: realized volatility from Nifty ─────────────────────────────────

def _realized_vix(nifty: pd.Series, window: int = 21) -> dict | None:
    """
    Annualized realized volatility on Nifty as India VIX proxy.
    Real VIX is implied vol (forward-looking), this is realized (backward).
    They track each other ±20% in normal markets.
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


def _record_today_score(score: int) -> float:
    """Append today's score, return 5-day MA across history."""
    today = date.today().isoformat()
    history = _load_score_history()
    # Replace today's entry if exists, else append
    history = [h for h in history if h.get("date") != today]
    history.append({"date": today, "score": int(score)})
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

    s = 0
    if p50 >= 60: s += 2
    elif p50 >= 40: s += 1
    if p200 >= 55: s += 2
    elif p200 >= 35: s += 1
    if hl_ratio >= 0.7: s += 2
    elif hl_ratio >= 0.5: s += 1
    if dist <= 3: s += 2
    elif dist <= 5: s += 1
    if n_stage == 2: s += 2
    elif n_stage == 1: s += 1
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

    # Bucket: 0-2 / 3-4 / 5-7 / 8-10
    def _bucket(s):
        if s >= 8: return "Bull (8-10)"
        if s >= 5: return "Caution (5-7)"
        if s >= 3: return "Correction (3-4)"
        return "Bear (0-2)"

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
    out.sort(key=lambda x: -["Bull (8-10)", "Caution (5-7)", "Correction (3-4)", "Bear (0-2)"].index(x["bucket"]))
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
        r3m = round((cur / float(cl.iloc[-63]) - 1) * 100, 2) if len(cl) > 63 else 0.0
        results.append({
            "symbol":    sym,
            "price":     round(cur, 2),
            "adtv_cr":   adtv_cr,
            "vol_ratio": vr,
            "r3m":       r3m,
            "pct_ath":   0.0,
        })
    results.sort(key=lambda x: x["vol_ratio"], reverse=True)
    return results[:20]


# ── Follow-Through Day (IBD) ──────────────────────────────────────────────────

def _follow_through_day(nifty: pd.Series) -> dict:
    """
    IBD Follow-Through Day detection.
    After a market decline (≥8%), a FTD is a Nifty move ≥1.7% on day 4+ of rally.
    Signals potential new market uptrend beginning.
    """
    result = {
        "rally_attempt": False, "days_since_trough": 0,
        "trough_to_now_pct": 0.0, "ftd_today": False,
        "today_chg_pct": 0.0, "ftd_in_window": False,
    }
    try:
        if len(nifty) < 30:
            return result
        window     = nifty.iloc[-60:]
        peak_val   = float(window.max())
        trough_val = float(window.min())
        cur        = float(nifty.iloc[-1])
        if peak_val <= 0 or (peak_val - trough_val) / peak_val < 0.08:
            return result
        if cur <= trough_val * 1.01:
            return result
        trough_loc       = int(window.values.argmin())
        days_from_trough = len(window) - 1 - trough_loc
        result["rally_attempt"]      = True
        result["days_since_trough"]  = days_from_trough
        result["trough_to_now_pct"]  = round((cur - trough_val) / trough_val * 100, 2)
        if len(nifty) >= 2 and float(nifty.iloc[-2]) > 0:
            today_chg = (float(nifty.iloc[-1]) - float(nifty.iloc[-2])) / float(nifty.iloc[-2]) * 100
            result["today_chg_pct"] = round(today_chg, 2)
            result["ftd_today"]     = (days_from_trough >= 3 and today_chg >= 1.7)
        for i in range(-10, 0):
            try:
                prev = float(nifty.iloc[i - 1])
                if prev <= 0:
                    continue
                if (float(nifty.iloc[i]) - prev) / prev * 100 >= 1.7:
                    result["ftd_in_window"] = True
                    break
            except Exception:
                continue
    except Exception:
        pass
    return result


# ── Main entry ────────────────────────────────────────────────────────────────

def run_market_breadth(progress_callback=None) -> dict:
    if (_cache["data"]
            and time.time() - _cache["ts"] < CACHE_TTL):
        return _cache["data"]

    stocks = _load_all_stocks(progress_callback)
    if not stocks:
        return {"error": "No data available"}

    if progress_callback:
        progress_callback(0, 1, "Computing market breadth…")

    nifty       = _build_nifty(stocks)
    breadth     = _compute_breadth(stocks)
    dist_days   = _distribution_days(nifty) if nifty is not None else 0
    nifty_stage = stage_analysis(nifty) if nifty is not None else 0
    sector_b    = _sector_stage2_breadth(stocks)
    sector_pct  = sector_b["pct_stage2"] if sector_b else None
    new_highs_l = _new_high_stocks(stocks)
    ftd         = _follow_through_day(nifty) if nifty is not None else {}
    vix_proxy   = _realized_vix(nifty) if nifty is not None else None

    # Compute timing first WITHOUT smoothed (to get raw score), then record + smooth
    raw_timing  = _market_timing_signal(breadth, dist_days, nifty_stage, sector_pct)
    smoothed_5d = _record_today_score(raw_timing["score"])
    timing      = _market_timing_signal(breadth, dist_days, nifty_stage, sector_pct, smoothed_5d)

    # Backtest thresholds (heavy — but data is already loaded, so cheap)
    backtest = _backtest_thresholds(stocks, nifty) if nifty is not None else {"by_bucket": []}

    # Nifty returns
    nifty_r1m = nifty_r3m = nifty_r6m = None
    if nifty is not None and len(nifty) > 22:
        nifty_r1m = round((float(nifty.iloc[-1]) / float(nifty.iloc[-22]) - 1) * 100, 2)
    if nifty is not None and len(nifty) > 63:
        nifty_r3m = round((float(nifty.iloc[-1]) / float(nifty.iloc[-63]) - 1) * 100, 2)
    if nifty is not None and len(nifty) > 126:
        nifty_r6m = round((float(nifty.iloc[-1]) / float(nifty.iloc[-126]) - 1) * 100, 2)

    out = {
        "breadth":       breadth,
        "dist_days":     dist_days,
        "nifty_stage":   nifty_stage,
        "nifty_stage_label": stage_label(nifty_stage),
        "nifty_r1m":     nifty_r1m,
        "nifty_r3m":     nifty_r3m,
        "nifty_r6m":     nifty_r6m,
        "timing":        timing,
        "ftd":           ftd,
        "new_highs_list": new_highs_l,
        "sector_breadth": sector_b,        # NEW: sector Stage-2 stats
        "vix_proxy":      vix_proxy,        # NEW: realized vol as VIX proxy
        "backtest":       backtest,         # NEW: historical score buckets
        "computed_at":   int(time.time()),
    }

    _cache["data"] = out
    _cache["ts"]   = time.time()
    _save_persisted_cache(_cache)   # persist to disk so next restart shows immediately

    if progress_callback:
        progress_callback(1, 1, "Market breadth computed")
    return out
