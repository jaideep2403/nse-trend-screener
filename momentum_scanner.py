"""
Momentum Scanner — ranks all ~2500 NSE EQ stocks by composite price momentum.
Zero new NSE API calls — uses only cached bhavcopy OHLCV data.

Score formula (percentile-ranked 0–100 across all qualifying stocks):
  Score = 0.40×r1m_rank + 0.30×r3m_rank + 0.20×r6m_rank + 0.10×vol_rank

Tiers:
  🔥 Elite  — score ≥ 85 AND RS ≥ 80
  ⚡ Strong  — score ≥ 70
  📈 Rising  — score ≥ 55
"""
import time
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from data_fetcher import _weekdays_back, _download_one_day
from analysis_utils import stage_analysis, stage_label
from industry_groups import INDUSTRY_GROUPS

# ── Symbol → Group lookup ─────────────────────────────────────────────────────
# Use sector_mapper's enriched map (covers all ~750 stocks via the NSE
# TotalMarket auto-mapper) instead of just the hand-curated INDUSTRY_GROUPS.
# Previously the latter alone left ~230 stocks (31%) with an empty `group`
# field — they show up in the universe but Sector filter on this tab couldn't
# group them. INDUSTRY_GROUPS still wins for stocks present in both maps,
# preserving the hand-curated fine-grained sub-sectors.
def _build_sym_to_group() -> dict[str, str]:
    m: dict[str, str] = {}
    for _grp, _syms in INDUSTRY_GROUPS.items():
        for _s in _syms:
            m[_s] = _grp
    try:
        from sector_mapper import get_enriched_sector_map
        for _s, _sec in get_enriched_sector_map().items():
            m.setdefault(_s, _sec)
    except Exception:
        pass   # sector_mapper is LOCAL-ONLY; absence is OK
    return m

_SYM_TO_GROUP: dict[str, str] = _build_sym_to_group()

# ── Config ────────────────────────────────────────────────────────────────────
MIN_BARS    = 130    # need at least 6M of history (≈130 trading days)
# TIER-3: MIN_PRICE removed — ADTV filter is the real liquidity gate.
# A ₹25 stock with ₹5Cr daily turnover (200M shares) is perfectly liquid.
MIN_ADTV_CR = 0.5    # Minimal liquidity guard only — universe filtered by Nifty500 membership
# Universe = curated 750 PLUS any LIQUID off-index stock at/above this turnover.
# Catches recent IPOs not yet in the index (e.g. AEROFLEX) that were invisible.
UNIVERSE_OFFINDEX_ADTV_CR = 2.0
SCAN_WORKERS = 8
_cache   = {"data": None, "ts": 0}
CACHE_TTL = 3600    # 1 hour


# ── Split / Bonus backward-adjustment (BUG-001) ───────────────────────────────

def _adjust_for_splits(df):
    """Delegate to canonical analysis_utils.adjust_for_splits."""
    from analysis_utils import adjust_for_splits
    return adjust_for_splits(df)


# ── Data loader ───────────────────────────────────────────────────────────────

def _load_all_stocks(progress_callback=None) -> dict[str, pd.DataFrame]:
    """Load full OHLCV history for all NSE EQ stocks from bhavcopy disk cache."""
    dates  = _weekdays_back(400)
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
    # Filter to Nifty Total Market 750 (Nifty50 ∪ Next50 ∪ Nifty500 ∪ Smallcap250 ∪ Microcap250 ∪ TotalMarket)
    try:
        from nse_stocks import get_universe_symbols
        _universe = set(get_universe_symbols())
    except Exception:
        _universe = set()
    combined = pd.concat(frames, ignore_index=True).sort_values("Date")
    stocks: dict[str, pd.DataFrame] = {}
    for sym, grp in combined.groupby("Symbol"):
        g = grp.set_index("Date")[["Open", "High", "Low", "Close", "Volume"]]
        g = g[~g.index.duplicated(keep="last")].sort_index()
        g = _adjust_for_splits(g)
        if len(g) < MIN_BARS:
            continue
        # Keep the curated 750 (all current coverage) PLUS any liquid off-index
        # stock (≥₹2Cr ADTV) so recent IPOs not yet in the index are scanned too.
        if sym not in _universe:
            cv   = g[["Close", "Volume"]].dropna()
            look = min(20, len(cv))
            adtv = float((cv["Close"].iloc[-look:] * cv["Volume"].iloc[-look:]).mean()) / 1e7 if look else 0.0
            if adtv < UNIVERSE_OFFINDEX_ADTV_CR:
                continue
        stocks[sym] = g
    return stocks


# ── Per-stock metric computation ──────────────────────────────────────────────

def _metrics(symbol: str, df: pd.DataFrame) -> dict | None:
    """Compute all momentum metrics for one stock. Returns None if filtered out.

    NEW (Tier 1B/1C):
      - vol_ann_pct       — annualised volatility (stdev of daily returns × √252)
      - sharpe_3m         — risk-adjusted momentum: r3m / vol_ann_pct
      - max_dd_6m         — biggest peak-to-trough drawdown over last 126 bars
      - r_squared         — R² of log-price linear regression (trend smoothness)
      - streak_up_weeks   — number of up-weeks out of last 12
    """
    import numpy as np
    try:
        close = df["Close"].dropna()
        vol   = df["Volume"].dropna()
        if len(close) < MIN_BARS:
            return None

        cur = float(close.iloc[-1])

        # Liquidity: average daily turnover in ₹ Cr (last 20 sessions)
        # Align Close + Volume on a common non-NaN index before multiplying
        # (mis-aligned dropna() of the two columns would silently pair wrong rows).
        cv = df[["Close", "Volume"]].dropna()
        if len(cv) < 20:
            return None
        adtv_cr = float((cv["Close"].iloc[-20:] * cv["Volume"].iloc[-20:]).mean()) / 1e7
        if adtv_cr < MIN_ADTV_CR:
            return None

        # ── Returns ──────────────────────────────────────────────────────────
        r1m = (cur / float(close.iloc[-21])  - 1) * 100 if len(close) >= 21  else None
        r3m = (cur / float(close.iloc[-63])  - 1) * 100 if len(close) >= 63  else None
        r6m = (cur / float(close.iloc[-126]) - 1) * 100 if len(close) >= 126 else None
        if r1m is None or r3m is None or r6m is None:
            return None

        # ── 52-week position ─────────────────────────────────────────────────
        window  = close.iloc[-252:] if len(close) >= 252 else close
        hi52    = float(window.max())
        lo52    = float(window.min())
        pos_52w = round((cur - lo52) / (hi52 - lo52) * 100, 1) if hi52 > lo52 else 50.0
        pct_from_high = round((cur / hi52 - 1) * 100, 2)

        # ── Volume strength (recent 10-day vs 50-day average) ────────────────
        vol_r10  = float(vol.iloc[-10:].mean())
        vol_a50  = float(vol.iloc[-50:].mean()) if len(vol) >= 50 else float(vol.mean())
        vol_ratio = round(vol_r10 / vol_a50, 2) if vol_a50 > 0 else 1.0

        # ── Moving averages ───────────────────────────────────────────────────
        ma50  = float(close.rolling(50).mean().iloc[-1])  if len(close) >= 50  else None
        ma200 = float(close.rolling(200).mean().iloc[-1]) if len(close) >= 200 else None

        # ── Trend strength: count of last 20 closes above MA50 ────────────────
        if ma50 is not None:
            above_count = int((close.iloc[-20:] > close.rolling(50).mean().iloc[-20:]).sum())
        else:
            above_count = 0
        above_streak = above_count  # kept for backward compatibility

        # ── Stage & trend ─────────────────────────────────────────────────────
        stg     = stage_analysis(close)
        stg_lbl = stage_label(stg)

        # ── Tier 1B: annualised volatility (6-month window) ──────────────────
        # Stdev of daily LOG returns over last 126 bars, scaled by √252.
        # Log returns avoid the asymmetry of percentage returns at high vol.
        try:
            log_ret = np.log(close.iloc[-126:] / close.iloc[-127:-1].values).dropna()
            vol_ann_pct = float(log_ret.std() * np.sqrt(252) * 100)
        except Exception:
            vol_ann_pct = None

        # ── Tier 1B: Sharpe-like quality score (return / volatility) ─────────
        # Higher is better. r3m / vol_ann_pct gives "return per unit of risk".
        # A stock with 30% return and 15% vol scores 2.0; same return with 60%
        # vol scores 0.5. Empirically validated as one of the strongest
        # momentum-quality enhancers (Asness, Moskowitz, Pedersen 2013).
        if vol_ann_pct and vol_ann_pct > 1.0:
            sharpe_3m = round(r3m / vol_ann_pct, 2)
        else:
            sharpe_3m = None

        # ── Tier 1C: max drawdown over last 6 months ─────────────────────────
        # Biggest peak-to-trough decline as a NEGATIVE percentage. A stock at
        # -10% has shallow DD = high quality momentum; -40% means the move
        # was preceded by a near-collapse = lower quality.
        try:
            s = close.iloc[-126:] if len(close) >= 126 else close
            running_max = s.cummax()
            dd_series = (s - running_max) / running_max * 100
            max_dd_6m = round(float(dd_series.min()), 1)   # most-negative value
        except Exception:
            max_dd_6m = None

        # ── Tier 1A: R² of log-price linear regression (trend smoothness) ───
        # Higher R² = stock compounded smoothly. R² of 0.5+ means a clear
        # uptrend; below 0.3 is jagged / mean-reverting.
        try:
            n = min(66, len(close))
            y = np.log(close.iloc[-n:].values)
            x = np.arange(len(y))
            if np.all(np.isfinite(y)):
                slope, intercept = np.polyfit(x, y, 1)
                pred = slope * x + intercept
                ss_res = float(((y - pred) ** 2).sum())
                ss_tot = float(((y - y.mean()) ** 2).sum())
                r_squared = round(max(0.0, min(1.0, 1 - ss_res / ss_tot)), 3) if ss_tot > 0 else None
            else:
                r_squared = None
        except Exception:
            r_squared = None

        # ── Streak: number of UP weeks out of last 12 ────────────────────────
        # Persistence signal — sustained momentum beats one-week moonshots.
        try:
            if isinstance(close.index, pd.DatetimeIndex):
                weekly = close.resample("W-FRI").last().dropna()
                if len(weekly) >= 13:
                    last_12 = weekly.iloc[-13:]   # 13 closes give 12 weekly changes
                    streak_up_weeks = int((last_12.diff().dropna() > 0).sum())
                else:
                    streak_up_weeks = None
            else:
                streak_up_weeks = None
        except Exception:
            streak_up_weeks = None

        # ── 🌋 IGNITION: young leader breaking out on volume (EARLY signal) ───
        # Trailing momentum (r1m/r3m/r6m) only ranks a stock AFTER it has moved.
        # By the time it's 🔥 Elite, 80-90% of the move is gone. Ignition catches
        # the START: a fresh multi-week high on expanding volume while the short
        # trend is rising — the footprint of institutional accumulation. This is
        # what would have surfaced MTARTECH / STLTECH months before the Elite badge.
        #   • new 50-bar closing high within the last 5 sessions
        #   • volume expanding (vol_ratio ≥ 1.5 = 10d avg vs 50d avg)
        #   • price above a RISING 20-day MA (trend intact, not a dead-cat bounce)
        # `bars_available` exposes stock age so the UI can rank youngest (= most
        # asymmetric upside) first. NOTE: the r6m gate above means nothing younger
        # than ~126 bars reaches here — the very-young IPO phase needs a dedicated
        # 50-bar scanner, which this intentionally does not replace.
        bars_available    = len(close)
        ignition_days_ago = None
        vol_surge         = bool(vol_ratio >= 1.5)
        try:
            ma20_series = close.rolling(20).mean()
            ma20_now    = float(ma20_series.iloc[-1])
            ma20_prev   = float(ma20_series.iloc[-11]) if len(close) >= 31 else ma20_now
            above_rising_ma20 = cur > ma20_now and ma20_now > ma20_prev
            # Most-recent new-50-bar-high breakout within the last 10 sessions
            for back in range(0, min(10, len(close) - 51) + 1):
                idx = len(close) - 1 - back
                if idx < 50:
                    break
                if close.iloc[idx] > float(close.iloc[idx-50:idx].max()):
                    ignition_days_ago = back
                    break
        except Exception:
            above_rising_ma20 = False
        igniting = bool(
            ignition_days_ago is not None
            and ignition_days_ago <= 5
            and vol_surge
            and above_rising_ma20
        )

        return {
            "symbol":         symbol,
            "price":          round(cur, 2),
            "adtv_cr":        round(adtv_cr, 1),
            "r1m":            round(r1m, 2),
            "r3m":            round(r3m, 2),
            "r6m":            round(r6m, 2),
            "vol_ratio":      vol_ratio,
            "pos_52w":        pos_52w,
            "pct_from_high":  pct_from_high,
            "above_ma50":     bool(ma50  and cur > ma50),
            "above_ma200":    bool(ma200 and cur > ma200),
            "above_streak":   above_streak,
            "above_count":    above_count,
            "stage":          stg,
            "stage_lbl":      stg_lbl,
            "group":          _SYM_TO_GROUP.get(symbol, ""),
            # ── NEW: Tier 1A/1B/1C quality metrics ───────────────────────────
            "vol_ann_pct":    round(vol_ann_pct, 1) if vol_ann_pct is not None else None,
            "sharpe_3m":      sharpe_3m,
            "max_dd_6m":      max_dd_6m,
            "r_squared":      r_squared,
            "streak_up":      streak_up_weeks,
            # ── 🌋 Ignition (early-breakout) signal ──────────────────────────
            "igniting":          igniting,
            "ignition_days_ago": ignition_days_ago,
            "vol_surge":         vol_surge,
            "bars_available":    bars_available,
            # Filled after ranking:
            "score":          0.0,
            "quality_score":  0.0,
            "rs_rating":      50,
            "tier":           "",
        }
    except Exception:
        return None


# ── Main entry ────────────────────────────────────────────────────────────────

def run_momentum_scan(progress_callback=None) -> dict:
    """
    Full momentum scan. Returns cached result if < 1 hour old.
    Uses only bhavcopy disk cache — zero NSE API calls.
    """
    if _cache["data"] and time.time() - _cache["ts"] < CACHE_TTL:
        return _cache["data"]

    # ── Load all stocks ───────────────────────────────────────────────────────
    stocks = _load_all_stocks(progress_callback)
    if not stocks:
        return {"stocks": [], "computed_at": int(time.time()),
                "total_scanned": 0, "total_qualified": 0}

    total_stocks = len(stocks)
    if progress_callback:
        progress_callback(0, total_stocks, f"Scoring {total_stocks} stocks…")

    # ── Compute metrics in parallel ───────────────────────────────────────────
    raw: list[dict] = []
    done = 0
    with ThreadPoolExecutor(max_workers=SCAN_WORKERS) as ex:
        futs = {ex.submit(_metrics, sym, df): sym for sym, df in stocks.items()}
        for fut in as_completed(futs):
            done += 1
            if progress_callback and done % 300 == 0:
                progress_callback(done, total_stocks,
                                  f"Scoring stocks… {done}/{total_stocks}")
            r = fut.result()
            if r is not None:
                raw.append(r)

    if not raw:
        return {"stocks": [], "computed_at": int(time.time()),
                "total_scanned": total_stocks, "total_qualified": 0}

    # ── Percentile-rank each metric ───────────────────────────────────────────
    df = pd.DataFrame(raw)

    def _prank(col: str) -> pd.Series:
        """Forward percentile rank (higher value → higher rank, 0-100)."""
        return df[col].rank(pct=True) * 100

    def _prank_inv(col: str) -> pd.Series:
        """Inverted percentile rank (SMALLER value → higher rank, 0-100).
        Used for max_dd_6m where -10% is BETTER than -40%."""
        return (1 - df[col].rank(pct=True)) * 100

    r1m_rank = _prank("r1m")
    r3m_rank = _prank("r3m")
    r6m_rank = _prank("r6m")
    vol_rank = _prank("vol_ratio")
    r2_rank  = _prank("r_squared").fillna(50.0)            # smoother = higher rank
    dd_rank  = _prank_inv("max_dd_6m").fillna(50.0)        # smaller DD = higher rank

    # RS Rating = percentile rank of 3M return (matches IBD convention)
    df["rs_rating"] = r3m_rank.round(0).astype(int).clip(1, 99)

    # ── Tier 1A: REWEIGHTED composite (was 40/30/20/10, now 15/30/25/10/10/10).
    # Cuts the 1M noise dominance (40 → 15%), boosts longer-horizon weight
    # (3M → 30, 6M → 25), and introduces two quality dimensions:
    #   • R² (trend smoothness)  — 10%
    #   • Drawdown control       — 10%  (smaller max DD = higher contribution)
    df["score"] = (
        0.15 * r1m_rank +
        0.30 * r3m_rank +
        0.25 * r6m_rank +
        0.10 * vol_rank +
        0.10 * r2_rank +
        0.10 * dd_rank
    ).round(1)

    # ── Tier 1B: Quality (Sharpe-like) score — separate from raw momentum.
    # Stocks with same return but lower vol get a HIGHER quality score.
    # Displayed alongside Momentum Score so user sees both axes.
    if "sharpe_3m" in df.columns:
        sharpe_clean = df["sharpe_3m"].fillna(df["sharpe_3m"].median() if df["sharpe_3m"].notna().any() else 0)
        df["quality_score"] = (sharpe_clean.rank(pct=True) * 100).round(1)
    else:
        df["quality_score"] = 50.0

    # ── Tier 1C: DRAWDOWN GATE on Elite tier ─────────────────────────────
    # Elite tier requires max_dd_6m > -25%. Stocks rallying from deep
    # drawdowns rarely sustain — demote them to Strong even if their raw
    # momentum percentile is in the top 15%.
    MAX_DD_FOR_ELITE = -25.0

    def _tier(row) -> str:
        s, rs = row["score"], row["rs_rating"]
        dd = row.get("max_dd_6m")
        # Elite: top momentum + top RS + decent drawdown control
        if s >= 85 and rs >= 80 and (dd is None or dd >= MAX_DD_FOR_ELITE):
            return "Elite"
        # Top momentum + RS but big drawdown: demote to Strong
        if s >= 85 and rs >= 80:
            return "Strong"   # demoted from Elite due to DD
        if s >= 70:
            return "Strong"
        if s >= 55:
            return "Rising"
        return ""

    df["tier"] = df.apply(_tier, axis=1)

    # ── Sort and output ───────────────────────────────────────────────────────
    df = df.sort_values("score", ascending=False).reset_index(drop=True)

    # ── Tier 2F: update momentum_freshness log + attach per-row freshness ────
    # After tier assignment, push the {symbol: tier} map into the SQLite log
    # so we know WHEN each stock entered its current tier. Then read back the
    # per-symbol days_in / bucket and attach to each row for the UI.
    stocks_list = df.to_dict(orient="records")
    try:
        from momentum_freshness import update_all as _fresh_update, get_freshness_map
        # Update log with today's tier assignments
        tier_map = {r["symbol"]: r.get("tier") or "" for r in stocks_list}
        upd = _fresh_update(tier_map)
        if progress_callback:
            progress_callback(total_stocks, total_stocks,
                              f"Freshness log: {upd.get('updated', 0)} updates, "
                              f"{upd.get('transitioned', 0)} tier transitions")
        # Bulk fetch freshness data for every result row
        freshness = get_freshness_map([r["symbol"] for r in stocks_list])
        for r in stocks_list:
            f = freshness.get(r["symbol"], {})
            r["freshness_days_in"] = f.get("days_in")
            r["freshness_bucket"]  = f.get("bucket")
            r["freshness_label"]   = f.get("label")
            r["freshness_since"]   = f.get("since_date")
            r["freshness_prev"]    = f.get("prev_tier")
            # Convenience flag: "Fresh Elite" = entered Elite in last 10 days
            r["fresh_elite"] = bool(
                r.get("tier") == "Elite"
                and f.get("days_in") is not None
                and f.get("days_in") <= 10
                and f.get("prev_tier") is not None
                and f.get("prev_tier") != "Elite"
            )
    except Exception as _e:
        # Freshness module not available — set safe defaults
        for r in stocks_list:
            r["freshness_days_in"] = None
            r["freshness_bucket"]  = None
            r["freshness_label"]   = None
            r["fresh_elite"]       = False

    out = {
        "stocks":          stocks_list,
        "computed_at":     int(time.time()),
        "total_scanned":   total_stocks,
        "total_qualified": len(raw),
        # Tier 1A: expose weights so UI can show them on hover
        "score_weights":   {"r1m": 0.15, "r3m": 0.30, "r6m": 0.25,
                            "vol_ratio": 0.10, "r_squared": 0.10, "drawdown": 0.10},
        # Tier 1C: expose drawdown threshold for UI labelling
        "max_dd_for_elite": MAX_DD_FOR_ELITE,
    }
    _cache["data"] = out
    _cache["ts"]   = time.time()

    if progress_callback:
        progress_callback(total_stocks, total_stocks,
                          f"Done — {len(raw)} stocks scored · {df[df['tier']=='Elite'].shape[0]} Elite")
    return out
