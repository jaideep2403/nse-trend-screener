"""
Institutional Scanner — Pocket Pivots + Earnings Season Setups
Identifies stocks with institutional-grade accumulation patterns.

Pocket Pivot (Gil Morales / Chris Kacher):
  Today's up-day volume > max single down-day volume in prior 10 sessions,
  while stock is within 15% of its 50-day MA (not extended).

Earnings Season Setup (Minervini / O'Neil):
  High-volume gap-up (earnings proxy) followed by tight consolidation,
  current price near top of that base, volume drying up.

Data: NSE bhavcopy (zero Yahoo calls, zero rate limits)
"""
import time
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
from data_fetcher import _weekdays_back, _download_one_day
from nse_stocks import is_etf
import result_cache
from analysis_utils import (
    trend_template_score, stage_analysis, stage_label,
    is_nr7, is_inside_bar, is_3wt,
    rs_line_new_high, detect_candle_signals, volume_baseline,
    cross_sectional_rs_rank,
)

MIN_BARS    = 60
MIN_ADTV_CR = 0.5    # Minimal liquidity guard only — universe filtered by Nifty500 membership
SCAN_WORKERS = 8
_cache = {"data": None, "ts": 0}
CACHE_TTL = 3600


# ── Split / Bonus backward-adjustment (BUG-001) ───────────────────────────────

def _adjust_for_splits(df, symbol=None):
    """Delegate to canonical analysis_utils.adjust_for_splits."""
    from analysis_utils import adjust_for_splits
    return adjust_for_splits(df, symbol)


# Canonical Nifty proxy basket (single source of truth — was a local 20-stock
# duplicate that could drift from every other tab's RS-vs-Nifty.)
from analysis_utils import NIFTY_PROXY_SYMS as _NIFTY50_SYMS


# ── Data loader (same bhavcopy source as breakout scanner) ────────────────────

def _load_all_stocks(progress_callback=None) -> dict[str, pd.DataFrame]:
    """Return {symbol: OHLCV_df} for all NSE EQ stocks from cached bhavcopy days."""
    dates  = _weekdays_back(400)
    total  = len(dates)
    frames = []
    for i, dt in enumerate(dates):
        df = _download_one_day(dt)
        if df is not None:
            frames.append(df)
        if progress_callback and i % 40 == 0:
            progress_callback(i, total, f"Loading historical data… {i}/{total} days")
    if not frames:
        return {}
    # Filter to Nifty Total Market 750 (Nifty50 ∪ Next50 ∪ Nifty500 ∪ Smallcap250 ∪ Microcap250 ∪ TotalMarket)
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
        g = _adjust_for_splits(g, sym)
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
    # BUG-FIX: rebase-to-100 equal-weight (was raw price avg, MARUTI/BAJFINANCE dominated)
    from analysis_utils import equal_weight_index
    combined = pd.concat(closes, axis=1).dropna(how="all")
    bench = equal_weight_index(combined)
    return bench if len(bench) >= 20 else None


# ── Shared utilities ──────────────────────────────────────────────────────────

def _atr(df: pd.DataFrame, period: int = 14) -> float:
    """Wilder's ATR via canonical analysis_utils.atr."""
    from analysis_utils import atr as _canonical_atr
    return _canonical_atr(df, period=period)


def _ud_volume_ratio(close: pd.Series, vol: pd.Series, period: int = 50) -> float:
    """Up/Down Volume Ratio — ratio of cumulative volume on up days vs down days."""
    try:
        if len(close) < period + 1 or len(vol) < period + 1:
            return 1.0
        cl = close.iloc[-period:]; v = vol.iloc[-period:]
        chg = cl.diff()
        up_vol   = float(v[chg > 0].sum())
        down_vol = float(v[chg < 0].sum())
        return round(up_vol / down_vol, 2) if down_vol > 0 else 3.0
    except Exception:
        return 1.0


def _acc_dist_days(close: pd.Series, vol: pd.Series, period: int = 20):
    """
    Accumulation days: close UP on volume > 1.3× 20-day avg
    Distribution days: close DOWN on volume > 1.3× 20-day avg

    BUG-FIX: previous code compared every bar in the period to TODAY's 20-day avg.
    In a rising-volume environment, bars from 3 weeks ago looked like "low volume"
    relative to today's higher average → systematically under-counted distribution
    and over-counted accumulation. Now each bar is compared to its OWN trailing
    20-day average (the avg that prevailed AT THAT bar).
    """
    try:
        if len(close) < period + 20 or len(vol) < period + 20:
            return 0, 0
        # Rolling 20-day avg for the entire series (not just the last value)
        avg_vol_series = vol.rolling(20).mean()
        # Now slice the LAST `period` bars from both close + vol + their own avg
        cl       = close.iloc[-period:]
        v        = vol.iloc[-period:]
        ref_avg  = avg_vol_series.iloc[-period:]
        chg      = cl.diff()
        # Each bar compared to the 20-day avg that prevailed at THAT bar
        high_vol = v > ref_avg * 1.3
        acc  = int(((chg > 0) & high_vol).sum())
        dist = int(((chg < 0) & high_vol).sum())
        return acc, dist
    except Exception:
        return 0, 0


def _weekly_tightness(close: pd.Series, weeks: int = 10) -> float:
    """
    Std-dev of weekly returns (%) over last N weeks.
    Lower = tighter price action = more institutional support / accumulation.
    """
    try:
        if len(close) < weeks * 5 + 5:
            return 9.9
        wk = close.resample("W-FRI").last().dropna()
        if len(wk) < weeks + 1:
            return 9.9
        ret = wk.pct_change().dropna().iloc[-weeks:]
        return round(float(ret.std() * 100), 2)
    except Exception:
        return 9.9


# ── Pocket Pivot detection ────────────────────────────────────────────────────

def _detect_pocket_pivot(df: pd.DataFrame):
    """
    Pocket Pivot (Gil Morales / Chris Kacher):
    1. Stock at or above 50-day MA (within 15%)
    2. Today is an UP day (close > prev close)
    3. Today's volume > max down-day volume in prior 10 sessions
    4. Not more than 15% above 50-day MA (not extended)

    Returns (True, levels_dict) or (False, {})
    """
    try:
        close = df["Close"].dropna()
        high  = df["High"].dropna()
        low   = df["Low"].dropna()
        vol   = df["Volume"].dropna()

        if len(close) < 55 or len(vol) < 13:
            return False, {}

        ma50 = float(close.rolling(50).mean().iloc[-1])
        cur  = float(close.iloc[-1])
        prev = float(close.iloc[-2])

        # Must be near or above 50-day MA
        if cur < ma50 * 0.97:
            return False, {}

        # Must be an up day
        if cur <= prev:
            return False, {}

        # Prior 10 sessions' down-day volumes (Pocket Pivot rule, Morales/Kacher).
        # BUG-FIX: previous code took `iloc[-12:-1]` (11 bars ending yesterday) and
        # then `.diff()` which drops the leftmost → only 10 valid changes, but
        # `v_win` had 11 entries → off-by-one alignment. The first change was
        # also computed against the bar OUTSIDE this window (and lost to NaN).
        # New approach: take 11 bars so diff() yields 10 valid deltas, then mask
        # the corresponding 10 volume bars.
        cl_win  = close.iloc[-12:-1]   # 11 closes ending yesterday
        v_win   = vol.iloc[-11:-1]     # 10 volumes for the 10 changes diff yields
        common  = cl_win.index.intersection(v_win.index)
        # Compute diffs aligned to v_win's window
        chg_full = cl_win.diff()        # 11 entries, first is NaN
        chg = chg_full.iloc[1:]         # 10 valid deltas, indices match v_win
        # Reindex both to the same dates to be safe
        common2 = chg.index.intersection(v_win.index)
        chg_a  = chg.reindex(common2)
        v_a    = v_win.reindex(common2)
        down_mask = chg_a < 0
        dv      = v_a[down_mask].dropna()
        max_dv  = float(dv.max()) if len(dv) > 0 else 0.0

        today_vol = float(vol.iloc[-1])
        if today_vol <= max_dv:
            return False, {}

        pct_from_ma50 = (cur - ma50) / ma50 * 100
        if pct_from_ma50 > 15.0:
            return False, {}   # too extended

        # Compute levels
        entry    = round(cur * 1.002, 2)    # buy just above today's close
        atr14    = _atr(df)
        base_low = round(float(low.iloc[-12:].min()), 2)
        sl       = round(max(ma50 * 0.975, cur - 2.5 * atr14, base_low * 0.985), 2)
        sl       = round(max(sl, entry * 0.92), 2)   # floor: never risk more than 8%

        # P0 FIX — a stop MUST sit below the entry.
        # The entry gate admits price down to ma50*0.97, but the stop candidate above
        # is ma50*0.975, and max() takes the HIGHEST candidate. Solving
        # ma50*0.975 > cur*1.002 shows a permanently-open band: whenever price closes
        # between 3.00% and 2.69% under its 50-DMA, the "stop" lands ABOVE the buy
        # price. The old code then HID it — `if risk <= 0: risk = entry * 0.04`
        # invented a 4% risk, and rr / risk_pct / t1 / t2 / t3 were all priced off
        # that fiction. Real case: TEGA entry 1644.78, "stop" 1649.12, shown with
        # rr 2.5 and risk 4.0% (observed ratio 1.00264 vs 1.00263 predicted here).
        # Clamping (rather than re-gating) keeps the SAME stocks selected — this is
        # an arithmetic fix, not a selection change.
        sl       = round(min(sl, entry * 0.98), 2)   # ceiling: at least 2% of room

        risk     = entry - sl
        if risk <= 0:
            # Unreachable after the clamp. Refuse to emit a plan we cannot price
            # honestly rather than fabricate a risk number.
            return False, {}

        t1 = round(entry + risk * 1.5, 2)
        t2 = round(entry + risk * 2.5, 2)
        prior_ath = float(close.iloc[:-1].max()) if len(close) > 1 else float(close.max())
        t3 = round(max(prior_ath * 1.005, entry + risk * 3.5), 2)

        return True, {
            "entry":         entry,
            "sl":            sl,
            "t1":            t1,
            "t2":            t2,
            "t3":            t3,
            "risk_pct":      round(risk / entry * 100, 2),
            "rr":            round((t2 - entry) / risk, 2) if risk > 0 else 0.0,
            "pct_from_ma50": round(pct_from_ma50, 2),
            "vol_vs_down":   round(today_vol / max_dv, 2) if max_dv > 0 else 9.9,
        }
    except Exception:
        return False, {}


# ── Earnings Season Setup detection ──────────────────────────────────────────

def _detect_earnings_setup(df: pd.DataFrame):
    """
    Earnings Season Setup (Minervini / O'Neil):
    1. High-volume gap-up in last 10–40 sessions (gap >2.5%, volume >1.8× avg) — earnings proxy
    2. Post-gap consolidation is tight (<20% range)
    3. Gap is not filled by more than 10%
    4. Current price near top of consolidation (within 3%)
    5. Recent volume drying up vs gap-day volume

    Returns (True, levels_dict) or (False, {})
    """
    try:
        close = df["Close"].dropna()
        high  = df["High"].dropna()
        low   = df["Low"].dropna()
        vol   = df["Volume"].dropna()
        open_ = df["Open"].dropna()

        if len(close) < 35 or len(open_) < 35:
            return False, {}

        # BUG-009 FIX: length guard — need ≥40 bars before indexing iloc[-40..-9]
        if len(close) < 40 or len(open_) < 40 or len(vol) < 40:
            return False, {}

        avg_vol_s = vol.rolling(20).mean()

        # Find most recent qualifying gap-up in the window [-40, -9]
        gap_idx = None
        for i in range(-40, -9):
            try:
                avg_v = float(avg_vol_s.iloc[i])
                if avg_v <= 0:
                    continue
                if (float(open_.iloc[i]) > float(close.iloc[i - 1]) * 1.025 and
                        float(vol.iloc[i]) > avg_v * 1.8):
                    gap_idx = i   # keep overwriting → last match = most recent
            except Exception:
                continue

        if gap_idx is None:
            return False, {}

        # Post-gap price action
        post_close = close.iloc[gap_idx:]
        post_high  = high.iloc[gap_idx:]
        post_low   = low.iloc[gap_idx:]
        post_vol   = vol.iloc[gap_idx:]

        if len(post_close) < 5:
            return False, {}

        gap_close = float(close.iloc[gap_idx])
        cur       = float(close.iloc[-1])
        post_hi   = float(post_high.max())
        post_lo   = float(post_low.min())

        # Tight consolidation — range relative to gap close
        post_range_pct = (post_hi - post_lo) / gap_close * 100
        if post_range_pct > 20.0:
            return False, {}

        # Gap not filled > 10%
        if cur < gap_close * 0.90:
            return False, {}

        # Price near top of base
        if cur < post_hi * 0.97:
            return False, {}

        # Volume drying up vs gap-day volume
        gap_vol     = float(vol.iloc[gap_idx])
        recent_avg  = float(post_vol.iloc[-5:].mean()) if len(post_vol) >= 5 else gap_vol
        if recent_avg > gap_vol * 0.75:
            return False, {}

        # Levels
        entry    = round(post_hi * 1.003, 2)
        base_lo  = round(post_lo, 2)
        atr14    = _atr(df)
        sl       = round(max(post_lo * 0.985, cur - 2.5 * atr14), 2)
        sl       = round(max(sl, entry * 0.92), 2)   # floor: never risk more than 8%
        # Same invariant as the pocket-pivot path. This one is not currently
        # reachable (both candidates sit below entry by construction), but the
        # fabrication guard below was identical, so the clamp goes in here too —
        # the class of bug is what we're closing, not one instance of it.
        sl       = round(min(sl, entry * 0.98), 2)

        risk     = entry - sl
        if risk <= 0:
            return False, {}

        base_ht  = post_hi - post_lo
        prior_ath = float(close.iloc[:-1].max()) if len(close) > 1 else float(close.max())
        # Monotonic ladder t1<t2<t3 (same guard as breakout/advanced) so the
        # measured move can't sit above the 2R target.
        t1       = round(entry + base_ht, 2)
        t2       = round(max(entry + 2.0 * risk, t1 + risk), 2)
        t3       = round(max(prior_ath * 1.005, entry + 3.0 * risk, t2 + risk), 2)

        return True, {
            "entry":          entry,
            "sl":             sl,
            "t1":             t1,
            "t2":             t2,
            "t3":             t3,
            "risk_pct":       round(risk / entry * 100, 2),
            "rr":             round((t2 - entry) / risk, 2) if risk > 0 else 0.0,
            "gap_close":      round(gap_close, 2),
            "days_since_gap": len(post_close) - 1,
            "consolidation_range_pct": round(post_range_pct, 1),
        }
    except Exception:
        return False, {}


# ── Per-stock analysis ────────────────────────────────────────────────────────

def _analyze(symbol: str, df: pd.DataFrame, nifty: pd.Series | None = None) -> dict | None:
    try:
        close = df["Close"].dropna()
        vol   = df["Volume"].dropna()

        if len(close) < MIN_BARS:
            return None

        cur = float(close.iloc[-1])
        if cur <= 0:
            return None

        # Liquidity — higher floor for institutional setups (median-based, outlier-resistant)
        avg_vol = volume_baseline(vol, window=20)
        adtv_cr = round(avg_vol * cur / 1e7, 2)
        if adtv_cr < MIN_ADTV_CR:
            return None

        # Detect primary setups
        pp_ok,  pp_lvl  = _detect_pocket_pivot(df)
        ear_ok, ear_lvl = _detect_earnings_setup(df)

        if not pp_ok and not ear_ok:
            return None

        # Institutional metrics
        udr       = _ud_volume_ratio(close, vol)
        acc, dist = _acc_dist_days(close, vol)
        tightness = _weekly_tightness(close)

        # Returns
        r1m  = round((cur / float(close.iloc[-21]) - 1) * 100, 2) if len(close) >= 21  else 0.0
        r3m  = round((cur / float(close.iloc[-63]) - 1) * 100, 2) if len(close) >= 63  else 0.0
        r6m  = round((cur / float(close.iloc[-126])- 1) * 100, 2) if len(close) >= 126 else 0.0

        ma50  = round(float(close.rolling(50).mean().iloc[-1]),  2) if len(close) >= 50  else None
        ma200 = round(float(close.rolling(200).mean().iloc[-1]), 2) if len(close) >= 200 else None
        ath   = float(close.max())

        # Trend Template + Stage (RS placeholder, updated after scan)
        tt_score, tt_met = trend_template_score(close, rs_rating=0)
        stage            = stage_analysis(close)
        candles          = detect_candle_signals(df)

        # Additional compression / RS signals
        nr7_flag    = is_nr7(df)
        ib_flag     = is_inside_bar(df)
        twt_flag    = is_3wt(close)
        rs_nh_flag  = rs_line_new_high(close, nifty) if nifty is not None else False

        extra_signals = []
        if nr7_flag:   extra_signals.append("NR7")
        if ib_flag:    extra_signals.append("InsideBar")
        if twt_flag:   extra_signals.append("3WT")
        if rs_nh_flag: extra_signals.append("RSLineHigh")

        # Institutional score components (raw — RS rank added after all stocks analysed)
        udr_s    = min(30.0, max(0.0, (udr - 0.5) / 2.5 * 30.0))
        acc_s    = min(20.0, acc * 4.0)
        dist_s   = max(0.0, 20.0 - dist * 5.0)
        tight_s  = max(0.0, 10.0 - tightness * 1.5)
        r3m_s    = min(20.0, max(0.0, (r3m + 20.0) / 60.0 * 20.0))
        inst_score = round(udr_s + acc_s + dist_s + tight_s + r3m_s, 1)

        # Determine active levels (PP takes priority over Earnings if both detected)
        active_lvl = pp_lvl if pp_ok else ear_lvl

        row = {
            "symbol":       symbol,
            "price":        round(cur, 2),
            "adtv_cr":      adtv_cr,
            "setup_types":  [],
            "extra_signals": extra_signals,
            "candles":      candles,
            # institutional metrics
            "udr":          udr,
            "acc_days":     acc,
            "dist_days":    dist,
            "tightness":    tightness,
            "inst_score":   inst_score,
            "rs_rating":    50,   # placeholder; updated after full scan
            # trend quality
            "tt_score":     tt_score,
            "tt_met":       tt_met,
            "stage":        stage,
            "stage_label":  stage_label(stage),
            # returns
            "r1m":          r1m,
            "r3m":          r3m,
            "r6m":          r6m,
            # MA
            "above_ma50":   (cur > ma50)  if ma50  else False,
            "above_ma200":  (cur > ma200) if ma200 else False,
            "pct_ath":      round((cur - ath) / ath * 100, 2),
            # trading levels
            "entry":        active_lvl.get("entry"),
            "sl":           active_lvl.get("sl"),
            "t1":           active_lvl.get("t1"),
            "t2":           active_lvl.get("t2"),
            "t3":           active_lvl.get("t3"),
            "risk_pct":     active_lvl.get("risk_pct"),
            "rr":           active_lvl.get("rr"),
        }

        if pp_ok:
            row["setup_types"].append("PocketPivot")
            row["pct_from_ma50"] = pp_lvl.get("pct_from_ma50")
            row["vol_vs_down"]   = pp_lvl.get("vol_vs_down")
        else:
            row["pct_from_ma50"] = None
            row["vol_vs_down"]   = None

        if ear_ok:
            row["setup_types"].append("EarningsSetup")
            row["days_since_gap"]          = ear_lvl.get("days_since_gap")
            row["consolidation_range_pct"] = ear_lvl.get("consolidation_range_pct")
            row["gap_close"]               = ear_lvl.get("gap_close")
        else:
            row["days_since_gap"]          = None
            row["consolidation_range_pct"] = None
            row["gap_close"]               = None

        return row
    except Exception:
        return None


# ── Main entry ────────────────────────────────────────────────────────────────

def run_institutional_scan(progress_callback=None) -> dict:
    if (_cache["data"]
            and time.time() - _cache["ts"] < CACHE_TTL
            and _cache["data"].get("results")):
        return _cache["data"]

    _disk = result_cache.get_or_stale("institutional")
    if _disk is not None:
        _cache["data"] = _disk
        _cache["ts"] = time.time()
        return _disk

    stocks = _load_all_stocks(progress_callback)
    if not stocks:
        return {"results": [], "computed_at": int(time.time()), "total_scanned": 0,
                "pp_count": 0, "earnings_count": 0}

    # Build Nifty benchmark for RS Line computation
    nifty = _build_nifty(stocks)

    total  = len(stocks)
    done   = [0]
    results = []

    if progress_callback:
        progress_callback(0, total, f"Scanning {total} stocks for institutional setups…")

    def _job(item):
        sym, df = item
        r = _analyze(sym, df, nifty)
        done[0] += 1
        if progress_callback and done[0] % 200 == 0:
            progress_callback(done[0], total,
                              f"Scanning {done[0]}/{total} stocks…")
        return r

    with ThreadPoolExecutor(max_workers=SCAN_WORKERS) as ex:
        for r in ex.map(_job, stocks.items()):
            if r:
                results.append(r)

    # Compute RS Rating (1–99) by ranking 3-month return across the ENTIRE
    # loaded universe — not just the handful that passed the setup filter.
    # Ranking within `results` (Pattern A) inflated RS: the strongest of ~60
    # survivors read 99 even if it was only mid-pack market-wide.
    if results:
        univ_r3m: dict[str, float] = {}
        for _sym, _df in stocks.items():
            _cl = _df["Close"].dropna()
            if len(_cl) >= 63:
                univ_r3m[_sym] = (float(_cl.iloc[-1]) / float(_cl.iloc[-63]) - 1) * 100
        univ_rs = cross_sectional_rs_rank(univ_r3m)
        # Fallback subset rank for any symbol missing from the universe map.
        r3m_s  = pd.Series([r["r3m"] for r in results])
        sub_rs = (r3m_s.rank(pct=True) * 99).round(0).astype(int).tolist()
        for r, sub in zip(results, sub_rs):
            rs = univ_rs.get(r["symbol"], int(sub))
            r["rs_rating"] = int(rs)
            # Recompute TT with real RS rating
            r["tt_score"], r["tt_met"] = trend_template_score(
                stocks[r["symbol"]]["Close"].dropna(), rs_rating=rs
            )
            # Blend RS into inst_score (20% weight)
            r["inst_score"] = round(r["inst_score"] * 0.8 + rs * 0.2, 1)

    # Sort: RS Line New High + both setups first, then inst_score + TT
    def _sort_key(r):
        rs_hi = 3 if "RSLineHigh" in r.get("extra_signals", []) else 0
        both  = 2 if len(r["setup_types"]) >= 2 else 0
        pp    = 1 if "PocketPivot" in r["setup_types"] else 0
        return (rs_hi + both + pp, r["inst_score"], r["tt_score"])

    results.sort(key=_sort_key, reverse=True)

    pp_count  = sum(1 for r in results if "PocketPivot"   in r["setup_types"])
    ear_count = sum(1 for r in results if "EarningsSetup" in r["setup_types"])
    rs_nh_count = sum(1 for r in results if "RSLineHigh" in r.get("extra_signals", []))

    out = {
        "results":        results,
        "computed_at":    int(time.time()),
        "total_scanned":  total,
        "pp_count":       pp_count,
        "earnings_count": ear_count,
        "rs_new_high_count": rs_nh_count,
    }
    _cache["data"] = out
    _cache["ts"]   = time.time()
    result_cache.put("institutional", out)

    if progress_callback:
        progress_callback(total, total,
                          f"Done — {pp_count} PP · {ear_count} Earnings · {rs_nh_count} RS New High")
    return out
