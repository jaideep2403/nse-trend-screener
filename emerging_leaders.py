"""
Emerging Leaders Scanner — catches YOUNG, explosive market leaders at their
earliest defensible entry (the ignition / first-base breakout), MONTHS before
they qualify for the Momentum or Trending "Elite" tiers.

WHY THIS EXISTS
---------------
Our other scanners are structurally LATE on stocks like MTARTECH (+440%) and
STLTECH (+560%):
  • Momentum ranks by trailing r1m/r3m/r6m — a stock can't rank until AFTER it
    has moved, and the r6m gate blocks anything younger than ~6 months.
  • early_mover_scanner needs 200 bars AND excludes r3m > 70% — it can see
    neither the young rocket (too few bars) nor the igniting one (return capped).
  • Stage analysis needs 175 bars + a rising MA150 — blind on young stocks for
    their first ~8 months.

COVERAGE (v2): scans the FULL NSE EQ list (~2,700 symbols from bhavcopy),
not just the curated Nifty Total Market 750. The whole point of "emerging" is
to catch leaders BEFORE they are added to an index — so we cannot restrict to
an index membership list. Liquidity + anti-manipulation gates keep the expanded
universe clean (no circuit-locked microcap pumps).

EMERGENCE SCORE (0–100) = sum of confirming factors:
    RS leadership ............ 22   (cross-sectional rank of blended return)
    RS acceleration .......... 8    (short-term RS rising faster than long-term)
    Breakout / new high ...... 16   (near + recently made a fresh high)
    Follow-through ........... 16   (the breakout HELD — not a reversed spike)  ← v2
    Volume / accumulation .... 18   (surge + up-vol dominance + pocket pivots)
    Trend quality ............ 8    (above rising MA20/MA50, upper of range)
    Youth bonus .............. 8    (younger = more "emerging" / asymmetric)
    Base tightness (VCP) ..... 4    (volatility contraction before expansion)

TIERS (top tier kept tight for precision on the larger universe):
    🌋 Emerging Leader — score ≥ 72 AND rs_rank ≥ 85 AND follow_through OK
    🔥 Igniting        — score ≥ 58
    👀 Watchlist       — score ≥ 47

This is a WATCHLIST generator, not a buy list. Early-stage breakouts fizzle
often; the score concentrates the odds and surfaces names early enough to WATCH,
then buy the confirmed follow-through with a stop.
"""
from __future__ import annotations

import math
import time
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed

from data_fetcher import _weekdays_back, _download_one_day
from analysis_utils import adjust_for_splits
from nse_stocks import is_etf

# ── Symbol → sector/group lookup (reuse momentum's enriched map) ──────────────
try:
    from momentum_scanner import _SYM_TO_GROUP
except Exception:
    _SYM_TO_GROUP = {}

# Reuse the canonical ETF detector so commodity/index ETFs (GOLDBEES, SILVERETF,
# NIFTYBEES, …) never pollute an "emerging *company*" scanner. They trade in the
# EQ series so the bhavcopy filter alone does not exclude them.
try:
    from early_mover_scanner import _is_etf
except Exception:
    def _is_etf(symbol: str) -> bool:
        s = symbol.upper()
        return s.endswith(("ETF", "BEES")) or "LIQUID" in s

# Top-tier precision gate: an "Emerging Leader" must still be near a buyable base
# (not a late, parabolic chase) and its fresh high must come with real demand.
MAX_EXT_FOR_LEADER = 60.0   # max % above MA50 to still qualify as a top-tier Leader

# ── Config ────────────────────────────────────────────────────────────────────
MIN_BARS      = 40       # works on young IPOs — the whole point of this scanner
MIN_ADTV_CR   = 1.5      # avg daily turnover floor (₹Cr) — institutional-tradeable
MIN_MEDIAN_CR = 0.5      # median turnover floor — rejects "one big day" illiquidity
MAX_LOCK_DAYS = 6        # >6 circuit-locked (High==Low) days in last 30 → reject
MAX_ZEROVOL   = 3        # >3 zero-volume days in last 30 → reject (not really trading)
SCAN_WORKERS  = 8
LOOKBACK_DAYS = 400      # bhavcopy days to assemble (~282 trading bars max)

# Tier thresholds (tuned against the MTARTECH/STLTECH point-in-time validation
# on the EXPANDED universe). Top tier is RS-gated so it auto-scales with universe.
TIER_LEADER  = 72.0
TIER_IGNITE  = 58.0
TIER_WATCH   = 47.0
LEADER_RS    = 85.0      # Emerging Leader needs top-15% relative strength

YOUNG_MAX    = 180       # younger than ~9 months counts as genuinely "emerging"

_cache    = {"data": None, "ts": 0}
CACHE_TTL = 3600         # 1 hour


# ── Data loader (FULL EQ universe + liquidity pre-prune) ──────────────────────

def _load_all_stocks(progress_callback=None) -> dict[str, pd.DataFrame]:
    """Load split-adjusted OHLCV for EVERY liquid NSE EQ stock from the bhavcopy
    disk cache (bhavcopy is already EQ-series-only). NOT restricted to the Nifty
    Total Market index — that is the recall win for catching off-index rockets.
    A turnover pre-prune keeps memory/CPU bounded by dropping the illiquid tail."""
    dates  = _weekdays_back(LOOKBACK_DAYS)
    total  = len(dates)
    frames = []
    for i, dt in enumerate(dates):
        df = _download_one_day(dt)
        if df is not None:
            frames.append(df)
        if progress_callback and i % 40 == 0:
            progress_callback(i, total, f"Loading bhavcopy cache… {i}/{total} days")
    if not frames:
        return {}
    combined = pd.concat(frames, ignore_index=True).sort_values("Date")
    stocks: dict[str, pd.DataFrame] = {}
    for sym, grp in combined.groupby("Symbol"):
        if is_etf(sym): continue
        if _is_etf(sym):
            continue   # commodity/index ETFs are not "emerging companies"
        g = grp.set_index("Date")[["Open", "High", "Low", "Close", "Volume"]]
        g = g[~g.index.duplicated(keep="last")].sort_index()
        if len(g) < MIN_BARS:
            continue
        g = adjust_for_splits(g)
        # Cheap turnover pre-prune (avoids storing thousands of illiquid microcaps)
        try:
            cv = g[["Close", "Volume"]].dropna()
            look = min(20, len(cv))
            adtv = float((cv["Close"].iloc[-look:] * cv["Volume"].iloc[-look:]).mean()) / 1e7
            if adtv < MIN_ADTV_CR:
                continue
        except Exception:
            continue
        stocks[sym] = g
    return stocks


# ── Per-stock raw metrics (pre cross-sectional RS) ────────────────────────────

def _safe_ret(close: pd.Series, look: int) -> float | None:
    if len(close) <= look:
        return None
    base = float(close.iloc[-look - 1])
    return (float(close.iloc[-1]) / base - 1) * 100 if base > 0 else None


def _raw_metrics(symbol: str, df: pd.DataFrame) -> dict | None:
    """All per-stock structural metrics. Returns None if it fails the hard gates
    (liquidity, anti-manipulation, must-be-advancing). Cross-sectional RS + the
    final composite score are filled later in _rank_and_score()."""
    try:
        close = df["Close"].astype(float).dropna()
        vol   = df["Volume"].astype(float).reindex(close.index).fillna(0.0)
        high  = df["High"].astype(float).reindex(close.index).fillna(close)
        low   = df["Low"].astype(float).reindex(close.index).fillna(close)
        n = len(close)
        if n < MIN_BARS:
            return None
        cur = float(close.iloc[-1])
        if cur <= 0:
            return None

        # ── Liquidity gate (turnover, mean + median) ─────────────────────────
        cv_look = min(20, n)
        turnover = (close.iloc[-cv_look:] * vol.iloc[-cv_look:])
        adtv_cr   = float(turnover.mean())   / 1e7
        median_cr = float(turnover.median()) / 1e7
        if adtv_cr < MIN_ADTV_CR or median_cr < MIN_MEDIAN_CR:
            return None

        # ── Anti-manipulation: circuit-locked / non-trading days ─────────────
        w30 = min(30, n)
        lock_days = int((high.iloc[-w30:].values == low.iloc[-w30:].values).sum())
        zero_vol  = int((vol.iloc[-w30:].values == 0).sum())
        if lock_days > MAX_LOCK_DAYS or zero_vol > MAX_ZEROVOL:
            return None

        # ── Returns (whatever the history allows) ────────────────────────────
        r1m = _safe_ret(close, 21)
        r3m = _safe_ret(close, 63)
        r6m = _safe_ret(close, 126)
        ret_since_start = (cur / float(close.iloc[0]) - 1) * 100 if float(close.iloc[0]) > 0 else 0.0

        if r3m is not None and r1m is not None:
            rs_return = 0.6 * r3m + 0.4 * r1m
        elif r1m is not None:
            rs_return = r1m
        else:
            rs_return = ret_since_start

        # ── Price position / distance from 52-WEEK high ──────────────────────
        # Use a 252-bar (52-week) window, not all-available history — for an
        # older stock the all-time high/low badly distorts pos_in_range,
        # pct_from_high and the hard gates. Young stocks (<252 bars) correctly
        # fall back to their full history (that IS their 52-week range).
        win = close.iloc[-252:] if n >= 252 else close
        rng_hi = float(win.max()); rng_lo = float(win.min())
        pos_in_range  = (cur - rng_lo) / (rng_hi - rng_lo) * 100 if rng_hi > rng_lo else 50.0
        pct_from_high = (cur / rng_hi - 1) * 100 if rng_hi > 0 else 0.0
        days_since_high = int(len(win) - 1 - int(np.argmax(win.values)))

        new_high_5d = False
        for back in range(0, min(5, n - 31) + 1):
            idx = n - 1 - back
            if idx >= 30 and close.iloc[idx] > float(close.iloc[idx - 30:idx].max()):
                new_high_5d = True
                break

        # ── Moving averages ──────────────────────────────────────────────────
        def _ma(wn):  return float(close.rolling(wn).mean().iloc[-1]) if n >= wn else None
        def _ma_rising(wn, back=10):
            if n < wn + back: return False
            s = close.rolling(wn).mean()
            return float(s.iloc[-1]) > float(s.iloc[-back - 1])
        ma20, ma50 = _ma(20), _ma(50)
        above_ma20 = ma20 is not None and cur > ma20
        above_ma50 = ma50 is not None and cur > ma50
        ma20_rising = _ma_rising(20)
        ma50_rising = _ma_rising(50)

        # ── Volume / accumulation footprint ──────────────────────────────────
        vol_r10 = float(vol.iloc[-10:].mean())
        vbase   = float(vol.iloc[-50:].mean()) if n >= 50 else float(vol.mean())
        vol_ratio = round(vol_r10 / vbase, 2) if vbase > 0 else 1.0
        vol_surge = vol_ratio >= 1.5

        chg = close.diff()
        wv = min(20, n - 1)
        up_v   = float(vol.iloc[-wv:][chg.iloc[-wv:] > 0].sum())
        down_v = float(vol.iloc[-wv:][chg.iloc[-wv:] < 0].sum())
        up_vol_ratio = round(up_v / down_v, 2) if down_v > 0 else 3.0

        pocket_pivots = 0
        for i in range(max(11, n - 10), n):
            if close.iloc[i] > close.iloc[i - 1]:
                prior = vol.iloc[i - 10:i][close.diff().iloc[i - 10:i] < 0]
                if len(prior) == 0 or vol.iloc[i] > float(prior.max()):
                    pocket_pivots += 1

        # ── Follow-through (v2): did the breakout HOLD, or reverse? ───────────
        # 1) held — current close near the recent 10-bar high (didn't give back)
        # 2) up-day persistence over last 10 bars
        # 3) NOT a one-bar spike — the advance is distributed, not a single candle
        recent_hi = float(high.iloc[-min(10, n):].max())
        giveback  = (recent_hi - cur) / recent_hi if recent_hi > 0 else 1.0
        held      = max(0.0, min(1.0, 1 - giveback / 0.10))         # full at high, 0 if -10%
        up10      = float((chg.iloc[-min(10, n - 1):] > 0).mean())   # fraction up days
        last5     = close.pct_change().iloc[-min(5, n - 1):]
        pos5      = float(last5[last5 > 0].sum())
        maxbar    = float(last5.max()) if len(last5) else 0.0
        broad     = max(0.0, min(1.0, 1 - (maxbar / pos5))) if pos5 > 0 else 0.0  # 1 spike → low
        follow_through = round(0.5 * held + 0.3 * up10 + 0.2 * broad, 3)

        # ── Base tightness / VCP ─────────────────────────────────────────────
        rets = close.pct_change().dropna()
        if len(rets) >= 30:
            recent_v  = float(rets.iloc[-10:].std())
            earlier_v = float(rets.iloc[-30:-10].std())
            contraction = max(0.0, min(1.0, 1 - (recent_v / earlier_v))) if earlier_v > 0 else 0.0
        else:
            contraction = 0.0

        # Extension above MA50 (informational + used to keep top tier buyable)
        ext_pct = round((cur / ma50 - 1) * 100, 1) if ma50 and ma50 > 0 else 0.0

        # ── HARD GATES — must be advancing, near highs, positive momentum ────
        if not above_ma20:        return None
        if pos_in_range < 45:     return None
        if pct_from_high < -25:   return None
        if rs_return <= 0:        return None

        return {
            "symbol":          symbol,
            "price":           round(cur, 2),
            "sector":          _SYM_TO_GROUP.get(symbol, ""),
            "adtv_cr":         round(adtv_cr, 1),
            "bars_available":  n,
            "is_young":        n < YOUNG_MAX,
            "r1m":             round(r1m, 1) if r1m is not None else None,
            "r3m":             round(r3m, 1) if r3m is not None else None,
            "r6m":             round(r6m, 1) if r6m is not None else None,
            "ret_since_start": round(ret_since_start, 1),
            "rs_return":       round(rs_return, 2),
            "pos_in_range":    round(pos_in_range, 1),
            "pct_from_high":   round(pct_from_high, 1),
            "days_since_high": days_since_high,
            "new_high_5d":     new_high_5d,
            "above_ma20":      bool(above_ma20),
            "above_ma50":      bool(above_ma50),
            "ma20_rising":     bool(ma20_rising),
            "ma50_rising":     bool(ma50_rising),
            "ext_pct":         ext_pct,
            "vol_ratio":       vol_ratio,
            "vol_surge":       bool(vol_surge),
            "up_vol_ratio":    up_vol_ratio,
            "pocket_pivots":   int(pocket_pivots),
            "follow_through":  follow_through,
            "lock_days":       lock_days,
            "contraction":     round(contraction, 2),
            "rs_rank":         50.0,
            "rs_accel":        False,
            "emergence_score": 0.0,
            "tier":            "",
            "reasons":         [],
        }
    except Exception:
        return None


# ── Cross-sectional RS + composite scoring ────────────────────────────────────

def _clamp(x, lo=0.0, hi=1.0):
    return max(lo, min(hi, x))


def _denan(rec: dict) -> dict:
    """pandas' DataFrame round-trip turns None in numeric columns into NaN
    (e.g. r3m/r6m for stocks too young to have 63/126 bars). NaN is NOT valid
    JSON — browsers reject it — so restore them to None at the scalar level
    before the record leaves this module. (Defence-in-depth alongside the
    app-level SafeJSONProvider.)"""
    for k, v in rec.items():
        if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
            rec[k] = None
    return rec


def _rank_and_score(raw: list[dict]) -> list[dict]:
    if not raw:
        return []
    df = pd.DataFrame(raw)
    df["rs_rank"] = (df["rs_return"].rank(pct=True) * 100).round(1)
    r1m_rank = df["r1m"].rank(pct=True) * 100
    base_rank = df["rs_return"].rank(pct=True) * 100
    df["rs_accel"] = (r1m_rank > base_rank + 5).fillna(False)

    out = []
    for _, row in df.iterrows():
        r = row.to_dict()
        reasons = []

        # 1) RS leadership (0-22)
        rs_pts = 0.22 * r["rs_rank"]
        if r["rs_rank"] >= 90: reasons.append(f"RS top {100 - r['rs_rank']:.0f}%")

        # 2) RS acceleration (0-8)
        accel_pts = 8.0 if r["rs_accel"] else 0.0
        if r["rs_accel"]: reasons.append("RS accelerating")

        # 3) Breakout / new high (0-16)
        near    = _clamp(1 + r["pct_from_high"] / 25.0)
        dsh     = r["days_since_high"]
        recency = 1.0 if dsh <= 3 else 0.6 if dsh <= 8 else 0.3 if dsh <= 20 else 0.0
        brk_pts = 16.0 * (0.6 * near + 0.4 * recency)
        if r["new_high_5d"]: reasons.append(f"fresh high {dsh}d ago")

        # 4) Follow-through (0-16)  ← v2 precision component
        ft = r["follow_through"]
        ft_pts = 16.0 * ft
        if ft >= 0.7: reasons.append("breakout holding")

        # 5) Volume / accumulation (0-18)
        vs = 1.0 if r["vol_surge"] else _clamp((r["vol_ratio"] - 1.0) / 0.5)
        uv = _clamp((r["up_vol_ratio"] - 1.0) / 1.0)
        pp = _clamp(r["pocket_pivots"] / 2.0)
        vol_pts = 18.0 * (0.5 * vs + 0.3 * uv + 0.2 * pp)
        if r["vol_surge"]:           reasons.append(f"vol surge {r['vol_ratio']:.1f}x")
        if r["up_vol_ratio"] >= 1.5: reasons.append(f"accumulation {r['up_vol_ratio']:.1f}x")
        if r["pocket_pivots"] >= 1:  reasons.append(f"{r['pocket_pivots']} pocket pivot(s)")

        # 6) Trend quality (0-8)
        tq = 0.0
        if r["above_ma20"] and r["ma20_rising"]: tq += 3
        if r["above_ma50"] and r["ma50_rising"]: tq += 3
        elif r["above_ma50"]:                    tq += 1.5
        if r["pos_in_range"] >= 70:              tq += 2
        trend_pts = min(8.0, tq)

        # 7) Youth bonus (0-8)
        b = r["bars_available"]
        youth_pts = 8.0 if b < 80 else 6.5 if b < 120 else 4.0 if b < 180 else 1.5 if b < 250 else 0.0
        if b < YOUNG_MAX: reasons.append(f"young ({b} bars)")

        # 8) Base tightness / VCP (0-4)
        vcp_pts = 4.0 * r["contraction"]
        if r["contraction"] >= 0.4: reasons.append("tight base")

        score = rs_pts + accel_pts + brk_pts + ft_pts + vol_pts + trend_pts + youth_pts + vcp_pts
        r["emergence_score"] = round(score, 1)
        r["score_parts"] = {
            "rs": round(rs_pts, 1), "accel": round(accel_pts, 1),
            "breakout": round(brk_pts, 1), "follow_through": round(ft_pts, 1),
            "volume": round(vol_pts, 1), "trend": round(trend_pts, 1),
            "youth": round(youth_pts, 1), "vcp": round(vcp_pts, 1),
        }

        # Tier — the top "Emerging Leader" tier is gated for precision:
        #   • leadership relative strength (rs_rank ≥ 85)
        #   • the breakout is HOLDING (follow_through ≥ 0.55)
        #   • still near a buyable base, not a parabolic chase (ext ≤ 60% over MA50)
        #   • the fresh high came with REAL demand (volume surge / pocket pivot /
        #     up-vol dominance) — not a thin drift to new highs
        demand_ok = (r["vol_surge"] or r["pocket_pivots"] >= 1 or r["up_vol_ratio"] >= 1.5)
        not_extended = (r["ext_pct"] <= MAX_EXT_FOR_LEADER)
        if (score >= TIER_LEADER and r["rs_rank"] >= LEADER_RS and ft >= 0.55
                and not_extended and demand_ok):
            r["tier"] = "Emerging Leader"
        elif score >= TIER_IGNITE:
            r["tier"] = "Igniting"
        elif score >= TIER_WATCH:
            r["tier"] = "Watchlist"
        else:
            r["tier"] = ""
        if r["ext_pct"] > MAX_EXT_FOR_LEADER:
            reasons.append(f"extended +{r['ext_pct']:.0f}% > MA50")
        r["reasons"] = reasons[:5]
        out.append(_denan(r))

    out.sort(key=lambda x: -x["emergence_score"])
    return out


# ── Main entry ────────────────────────────────────────────────────────────────

def run_emerging_leaders_scan(progress_callback=None, _stocks=None,
                              as_of=None) -> dict:
    """Scan for emerging leaders. as_of truncates all series for point-in-time
    validation; _stocks reuses a pre-loaded universe (used by validation)."""
    use_cache = (_stocks is None and as_of is None)
    if use_cache and _cache["data"] and time.time() - _cache["ts"] < CACHE_TTL:
        return _cache["data"]

    stocks = _stocks if _stocks is not None else _load_all_stocks(progress_callback)
    if not stocks:
        return {"stocks": [], "computed_at": int(time.time()),
                "total_scanned": 0, "total_qualified": 0, "tier_counts": {}}

    if as_of is not None:
        cutoff = pd.Timestamp(as_of)
        trunc = {}
        for sym, df in stocks.items():
            t = df[df.index <= cutoff]
            if len(t) >= MIN_BARS:
                trunc[sym] = t
        stocks = trunc

    total = len(stocks)
    if progress_callback:
        progress_callback(0, total, f"Scoring {total} stocks for emergence…")

    raw: list[dict] = []
    done = 0
    with ThreadPoolExecutor(max_workers=SCAN_WORKERS) as ex:
        futs = {ex.submit(_raw_metrics, s, d): s for s, d in stocks.items()}
        for fut in as_completed(futs):
            done += 1
            if progress_callback and done % 400 == 0:
                progress_callback(done, total, f"Scoring… {done}/{total}")
            r = fut.result()
            if r is not None:
                raw.append(r)

    scored = _rank_and_score(raw)
    qualified = [s for s in scored if s["tier"]]
    tier_counts = {}
    for s in qualified:
        tier_counts[s["tier"]] = tier_counts.get(s["tier"], 0) + 1

    out = {
        "stocks":          qualified,
        "computed_at":     int(time.time()),
        "total_scanned":   total,
        "total_qualified": len(qualified),
        "total_passed_gates": len(raw),
        "tier_counts":     tier_counts,
        "thresholds":      {"leader": TIER_LEADER, "igniting": TIER_IGNITE,
                            "watchlist": TIER_WATCH, "leader_rs": LEADER_RS},
        "universe":        "All liquid EQ (ADTV ≥ ₹%.1fCr)" % MIN_ADTV_CR,
        "as_of":           str(as_of) if as_of is not None else None,
    }
    if use_cache:
        _cache["data"] = out
        _cache["ts"]   = time.time()
    return out


def invalidate_cache():
    _cache["data"] = None
    _cache["ts"]   = 0
