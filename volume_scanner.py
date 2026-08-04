"""
Unusual Volume Scanner — detects smart money accumulation via EOD volume spikes.
Zero new NSE API calls — bhavcopy only.

Patterns detected:
  🔴 Accumulation  — RVOL ≥ 2.0 + closed in upper 60% of day range + price held/gained
  🚀 Breakout      — RVOL ≥ 2.5 + price ≥ 1% gain + within 5% of 52W high
  🟡 Stealth       — RVOL 1.5–2.5 + tight candle (body < 1.5%) + price quietly held
  📈 Streak        — 3+ consecutive days with above-avg volume
  🔵 VDU           — Volume Dry-Up: vol collapsed after a prior spike (coil before breakout)
  ⚠️  Distribution  — RVOL ≥ 2.0 + closed in lower 35% + price fell

Alert levels:
  High   — Accumulation or Breakout with RVOL ≥ 2.5
  Medium — Stealth, Streak, VDU
  Watch  — RVOL ≥ 1.5 with mixed/unclear signals
"""
import time
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from data_fetcher import _weekdays_back, _download_one_day
from nse_stocks import is_etf
import result_cache
from analysis_utils import stage_analysis, stage_label
from industry_groups import INDUSTRY_GROUPS

# ── Symbol → group lookup ─────────────────────────────────────────────────────
_SYM_TO_GROUP: dict[str, str] = {}
for _grp, _syms in INDUSTRY_GROUPS.items():
    for _s in _syms:
        _SYM_TO_GROUP[_s] = _grp

# ── Config ────────────────────────────────────────────────────────────────────
# TIER-3: MIN_PRICE removed — ADTV filter is the real liquidity gate
MIN_ADTV_CR  = 0.5    # Minimal liquidity guard only — universe filtered by Nifty500 membership
MIN_RVOL     = 1.5    # minimum relative volume to include
SCAN_WORKERS = 8
_cache       = {"data": None, "ts": 0}
CACHE_TTL    = 1800   # 30 min — volume data ages faster than momentum


# ── Split / Bonus backward-adjustment (BUG-001) ───────────────────────────────

def _adjust_for_splits(df, symbol=None):
    """Delegate to canonical analysis_utils.adjust_for_splits."""
    from analysis_utils import adjust_for_splits
    return adjust_for_splits(df, symbol)

# ── ETF exclusion (same patterns as early_mover_scanner) ─────────────────────
_ETF_ENDSWITH = ("ETF", "BEES", "FUND", "BENCHMARK", "NIFTY1")
_ETF_CONTAINS = (
    "LIQUID", "IETF", "CPSE", "NIFTYBEES", "GOLDETF",
    "SILVETF", "MAKEINDIA", "MAFANG", "MIDCAPETF",
    "INFRABEES", "BANKBEES", "JUNIORBEES", "PSUBNKBEES",
    "NIFTYETF", "SENSEXETF", "SHARIAH", "SETFNIF", "SETF",
    "HNGSNG", "CPSEETF", "BBETF", "ABSLNN50ET",
)


def _is_etf(symbol: str) -> bool:
    s = symbol.upper()
    if any(s.endswith(sfx) for sfx in _ETF_ENDSWITH):
        return True
    if any(k in s for k in _ETF_CONTAINS):
        return True
    return False


# ── Data loader ───────────────────────────────────────────────────────────────

def _load_all_stocks(progress_callback=None) -> dict[str, pd.DataFrame]:
    """Load 120 days of bhavcopy — only need recent history for volume analysis."""
    dates  = _weekdays_back(120)
    total  = len(dates)
    frames = []
    for i, dt in enumerate(dates):
        df = _download_one_day(dt)
        if df is not None:
            frames.append(df)
        if progress_callback and i % 30 == 0:
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
        if is_etf(sym): continue
        if _universe and sym not in _universe:
            continue
        g = grp.set_index("Date")[["Open", "High", "Low", "Close", "Volume"]]
        g = g[~g.index.duplicated(keep="last")].sort_index()
        g = _adjust_for_splits(g, sym)
        if len(g) >= 25:
            stocks[sym] = g
    return stocks


# ── Pattern classifier ────────────────────────────────────────────────────────

def _classify_pattern(rvol: float, close_pct: float, day_chg: float,
                      body_pct: float, streak: int, vdu: bool,
                      pft_h: float) -> str:
    """Classify volume pattern. Priority: VDU > Breakout > Accumulation > Distribution > Stealth > Streak > Watch."""
    if vdu:
        return "VDU"
    if rvol >= 2.5 and day_chg >= 1.0 and pft_h >= -5.0:
        return "Breakout"
    if rvol >= 2.0 and close_pct >= 0.6 and day_chg >= 0:
        return "Accumulation"
    if rvol >= 2.0 and close_pct <= 0.35 and day_chg < 0:
        return "Distribution"
    if rvol >= 1.5 and close_pct >= 0.5 and body_pct < 1.5:
        return "Stealth"
    if streak >= 3:
        return "Streak"
    return "Watch"


# ── Per-stock metrics ─────────────────────────────────────────────────────────

def _metrics(symbol: str, df: pd.DataFrame) -> dict | None:
    try:
        if _is_etf(symbol):
            return None

        close = df["Close"].dropna()
        vol   = df["Volume"].dropna()
        high  = df["High"].dropna()
        low   = df["Low"].dropna()

        if len(close) < 25:
            return None

        cur = float(close.iloc[-1])

        # ADTV filter — align Close + Volume on common index before multiply.
        _cv = df[["Close", "Volume"]].dropna()
        if len(_cv) < 20:
            return None
        adtv_cr = float((_cv["Close"] * _cv["Volume"]).iloc[-20:].mean()) / 1e7
        if adtv_cr < MIN_ADTV_CR:
            return None

        # ── Volume metrics ────────────────────────────────────────────────────
        vol_today = float(vol.iloc[-1])
        # Use prior 20 bars (not including today) as baseline
        vol_avg20 = float(vol.iloc[-21:-1].mean()) if len(vol) >= 21 else float(vol.iloc[:-1].mean())
        if vol_avg20 <= 0:
            return None

        rvol = round(vol_today / vol_avg20, 2)
        if rvol < MIN_RVOL:
            return None

        # ── Price action on last bar ──────────────────────────────────────────
        day_hi  = float(high.iloc[-1])
        day_lo  = float(low.iloc[-1])
        day_rng = day_hi - day_lo
        close_pct = round((cur - day_lo) / day_rng, 2) if day_rng > 0 else 0.5

        prev_close = float(close.iloc[-2]) if len(close) >= 2 else cur
        day_chg    = round((cur / prev_close - 1) * 100, 2)

        # Candle body %: how much price moved open→close vs price
        open_p   = float(df["Open"].iloc[-1])
        body_pct = round(abs(cur - open_p) / open_p * 100, 2) if open_p > 0 else 0.0

        # ── Volume streak ─────────────────────────────────────────────────────
        # Compare each bar to its OWN trailing 20-bar baseline (the avg that
        # prevailed AT THAT bar), not a single today-anchored vol_avg20. The
        # old method counted older bars against an average that included bars
        # AFTER them — in rising-volume regimes older bars looked artificially
        # small (under-counting the streak); in falling-volume they looked
        # artificially large (over-counting).
        streak = 0
        # Pre-compute rolling 20-bar mean once for the lookback window
        vol_roll20 = vol.rolling(20).mean()
        for i in range(-1, -min(11, len(vol)), -1):
            # Need a valid trailing baseline at this bar (skip if NaN — too early)
            baseline = vol_roll20.iloc[i]
            if pd.isna(baseline) or baseline <= 0:
                break
            if float(vol.iloc[i]) > float(baseline):
                streak += 1
            else:
                break

        # ── Volume Dry-Up (VDU) ───────────────────────────────────────────────
        # Prior 3–7 bars had elevated volume; last 2 bars are very quiet
        vdu = False
        if len(vol) >= 8:
            prior_elevated = float(vol.iloc[-7:-2].mean())
            recent_quiet   = float(vol.iloc[-2:].mean())
            if prior_elevated > vol_avg20 * 1.5 and recent_quiet < vol_avg20 * 0.7:
                vdu = True

        # ── 52-week high / position ───────────────────────────────────────────
        # BUG-010 FIX: exclude current bar from the 52w lookback so the breakout
        # comparison ("near 52w high") really means current vs PRIOR 252 sessions.
        if len(close) >= 2:
            w52   = close.iloc[-min(252, len(close)):-1]
        else:
            w52   = close
        if len(w52) == 0:
            w52 = close
        hi52  = float(w52.max())
        lo52  = float(w52.min())
        pft_h = round((cur / hi52 - 1) * 100, 2) if hi52 > 0 else 0.0
        pos52 = round((cur - lo52) / (hi52 - lo52) * 100, 1) if hi52 > lo52 else 50.0

        # ── Moving averages ───────────────────────────────────────────────────
        ma50  = float(close.rolling(50).mean().iloc[-1]) if len(close) >= 50 else None
        ma200 = float(close.rolling(200).mean().iloc[-1]) if len(close) >= 200 else None

        # ── Stage ─────────────────────────────────────────────────────────────
        stg     = stage_analysis(close)
        stg_lbl = stage_label(stg)

        # ── 3M return (context) ───────────────────────────────────────────────
        r3m = round((cur / float(close.iloc[-min(66, len(close)-1)]) - 1) * 100, 2)

        # ── Pattern + Alert ───────────────────────────────────────────────────
        pattern = _classify_pattern(rvol, close_pct, day_chg, body_pct, streak, vdu, pft_h)
        if pattern in ("Accumulation", "Breakout") and rvol >= 2.5:
            alert = "High"
        elif pattern in ("Stealth", "Streak", "VDU"):
            alert = "Medium"
        else:
            alert = "Watch"

        return {
            "symbol":       symbol,
            "price":        round(cur, 2),
            "rvol":         rvol,
            "vol_today":    int(vol_today),
            "vol_avg20":    int(vol_avg20),
            "adtv_cr":      round(adtv_cr, 1),
            "close_pct":    close_pct,
            "day_chg":      day_chg,
            "body_pct":     body_pct,
            "streak":       streak,
            "vdu":          vdu,
            "pct_from_high": pft_h,
            "pos_52w":      pos52,
            "above_ma50":   bool(ma50  and cur > ma50),
            "above_ma200":  bool(ma200 and cur > ma200),
            "r3m":          r3m,
            "stage":        stg,
            "stage_lbl":    stg_lbl,
            "group":        _SYM_TO_GROUP.get(symbol, ""),
            "pattern":      pattern,
            "alert":        alert,
        }
    except Exception:
        return None


# ── Main entry ────────────────────────────────────────────────────────────────

def run_volume_scan(progress_callback=None) -> dict:
    """Full unusual-volume scan. Cached 30 min. Zero NSE API calls."""
    if _cache["data"] and time.time() - _cache["ts"] < CACHE_TTL:
        return _cache["data"]

    _disk = result_cache.get("volume")
    if _disk is not None:
        _cache["data"] = _disk
        _cache["ts"] = time.time()
        return _disk

    stocks = _load_all_stocks(progress_callback)
    if not stocks:
        return {"stocks": [], "computed_at": int(time.time()),
                "total_scanned": 0, "total_qualified": 0, "counts": {}}

    total = len(stocks)
    if progress_callback:
        progress_callback(0, total, f"Scanning {total} stocks for unusual volume…")

    raw: list[dict] = []
    done = 0
    with ThreadPoolExecutor(max_workers=SCAN_WORKERS) as ex:
        futs = {ex.submit(_metrics, sym, df): sym for sym, df in stocks.items()}
        for fut in as_completed(futs):
            done += 1
            if progress_callback and done % 500 == 0:
                progress_callback(done, total, f"Scanning… {done}/{total}")
            r = fut.result()
            if r is not None:
                raw.append(r)

    if not raw:
        return {"stocks": [], "computed_at": int(time.time()),
                "total_scanned": total, "total_qualified": 0, "counts": {}}

    # Sort: alert level first, then RVOL descending
    _alert_ord = {"High": 0, "Medium": 1, "Watch": 2}
    raw.sort(key=lambda x: (_alert_ord.get(x["alert"], 3), -x["rvol"]))

    counts = {
        "high":   sum(1 for r in raw if r["alert"] == "High"),
        "medium": sum(1 for r in raw if r["alert"] == "Medium"),
        "watch":  sum(1 for r in raw if r["alert"] == "Watch"),
        "accumulation": sum(1 for r in raw if r["pattern"] == "Accumulation"),
        "breakout":     sum(1 for r in raw if r["pattern"] == "Breakout"),
        "stealth":      sum(1 for r in raw if r["pattern"] == "Stealth"),
        "streak":       sum(1 for r in raw if r["pattern"] == "Streak"),
        "vdu":          sum(1 for r in raw if r["pattern"] == "VDU"),
    }

    out = {
        "stocks":          raw,
        "computed_at":     int(time.time()),
        "total_scanned":   total,
        "total_qualified": len(raw),
        "counts":          counts,
    }
    _cache["data"] = out
    _cache["ts"]   = time.time()
    result_cache.put("volume", out)

    if progress_callback:
        progress_callback(
            total, total,
            f"Done — {len(raw)} signals · 🔴{counts['high']} High · "
            f"🟡{counts['medium']} Medium · ⚪{counts['watch']} Watch"
        )
    return out
