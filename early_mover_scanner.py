"""
Early Mover Scanner — finds stocks with characteristics of monster-move setups.

Modelled on pre-move fingerprints of APAR Industries, Dixon, Force Motors,
Hitachi Energy, Cupid — all shared: long quiet base → volume/ADTV expansion
→ sector rotation → RS improvement BEFORE the explosive price move.

Score = 0.30 × ADTV-growth-rank      (new institutional money arriving)
      + 0.25 × base-quality-rank      (longer × tighter base = bigger coil)
      + 0.20 × volume-expansion-rank  (volume building during base)
      + 0.15 × RS-delta-rank          (RS improving from low levels)
      + 0.10 × sector-alignment-rank  (sector rotating into leadership)

Labels:
  🔥 Inflecting  — sector Improving/Leading + RS rising fast + money arriving
  💎 Hidden Gem  — long tight base (5M+) + ADTV growing quietly
  🌱 Emerging    — early signals, needs monitoring

Zero new NSE API calls — bhavcopy only.
"""
import time
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from data_fetcher import _weekdays_back, _download_one_day
from nse_stocks import is_etf
from analysis_utils import stage_analysis, stage_label
from industry_groups import INDUSTRY_GROUPS

# ── Symbol → group lookup ─────────────────────────────────────────────────────
_SYM_TO_GROUP: dict[str, str] = {}
for _grp, _syms in INDUSTRY_GROUPS.items():
    for _s in _syms:
        _SYM_TO_GROUP[_s] = _grp

# ── Config ────────────────────────────────────────────────────────────────────
MIN_BARS      = 200    # need enough history for all metrics
# TIER-3: MIN_PRICE removed — ADTV filter is the real liquidity gate
MIN_ADTV_CR   = 0.5    # Minimal liquidity guard only — universe filtered by Nifty500 membership
MAX_3M_RET    = 70.0  # exclude already-running stocks
MIN_3M_RET    = -35.0 # exclude stocks in freefall
MAX_BASE_RANGE = 55.0 # base must be reasonably tight
MIN_BASE_DAYS  = 60   # at least 3 months of consolidation

SCAN_WORKERS  = 8
_cache        = {"data": None, "ts": 0}
CACHE_TTL     = 3600


# ── Split / Bonus backward-adjustment (BUG-001) ───────────────────────────────

def _adjust_for_splits(df):
    """Delegate to canonical analysis_utils.adjust_for_splits."""
    from analysis_utils import adjust_for_splits
    return adjust_for_splits(df)


# ── ETF exclusion ─────────────────────────────────────────────────────────────
# NSE ETFs trade as EQ series but should not appear as stock picks.
# Filter by symbol suffix/keyword patterns.
_ETF_ENDSWITH = (
    "ETF", "BEES", "FUND", "BENCHMARK", "NIFTY1",
)
_ETF_CONTAINS = (
    "LIQUID", "IETF", "CPSE", "NIFTYBEES", "GOLDETF",
    "SILVETF", "MAKEINDIA", "MAFANG", "MIDCAPETF",
    "INFRABEES", "BANKBEES", "JUNIORBEES", "PSUBNKBEES",
    "NIFTYETF", "SENSEXETF", "SHARIAH", "SETFNIF", "SETF",
    "HNGSNG", "CPSEETF", "BBETF", "ABSLNN50ET",
)


def _is_etf(symbol: str) -> bool:
    """Return True if the symbol looks like an ETF — skip it."""
    s = symbol.upper()
    if any(s.endswith(sfx) for sfx in _ETF_ENDSWITH):
        return True
    if any(k in s for k in _ETF_CONTAINS):
        return True
    return False


_NIFTY_SYMS = [
    "RELIANCE", "TCS", "HDFCBANK", "ICICIBANK", "INFY",
    "HINDUNILVR", "SBIN", "BHARTIARTL", "ITC", "KOTAKBANK",
    "AXISBANK", "WIPRO", "HCLTECH", "MARUTI", "BAJFINANCE",
]


# ── Data loader ───────────────────────────────────────────────────────────────

def _load_all_stocks(progress_callback=None) -> dict[str, pd.DataFrame]:
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
        if is_etf(sym): continue
        if _universe and sym not in _universe:
            continue
        g = grp.set_index("Date")[["Open", "High", "Low", "Close", "Volume"]]
        g = g[~g.index.duplicated(keep="last")].sort_index()
        g = _adjust_for_splits(g)
        if len(g) >= MIN_BARS:
            stocks[sym] = g
    return stocks


# ── Base detection ────────────────────────────────────────────────────────────

def _detect_base(close: pd.Series) -> tuple[int, float, float]:
    """
    Find the longest recent consolidation where price range < MAX_BASE_RANGE%.
    Returns (base_len_days, base_range_pct, pos_in_base 0–1).
    pos_in_base > 0.7 = near top of base = about to break out.
    """
    n = min(250, len(close) - 5)
    best_len = 0
    for days in range(n, MIN_BASE_DAYS - 1, -5):
        slc = close.iloc[-days:]
        lo, hi = float(slc.min()), float(slc.max())
        if hi > lo and (hi - lo) / lo * 100 < MAX_BASE_RANGE:
            best_len = days
            break

    if best_len < MIN_BASE_DAYS:
        return 0, 100.0, 0.5

    base_slc  = close.iloc[-best_len:]
    base_lo   = float(base_slc.min())
    base_hi   = float(base_slc.max())
    base_rng  = round((base_hi - base_lo) / base_lo * 100, 1)
    cur       = float(close.iloc[-1])
    pos       = round((cur - base_lo) / (base_hi - base_lo), 2) if base_hi > base_lo else 0.5
    return best_len, base_rng, pos


# ── Sector RRG quadrants ──────────────────────────────────────────────────────

def _sector_quadrants(stocks: dict) -> dict[str, str]:
    """
    Compute simplified RRG quadrant for each industry group from loaded stocks.
    Returns {group_name: "Leading"|"Improving"|"Weakening"|"Lagging"}.
    """
    try:
        nifty_closes = [
            stocks[s]["Close"].dropna()
            for s in _NIFTY_SYMS if s in stocks and len(stocks[s]) >= 80
        ]
        if not nifty_closes:
            return {}
        # BUG-FIX: rebase-to-100 equal-weight (was raw price avg)
        from analysis_utils import equal_weight_index
        nifty = equal_weight_index(pd.concat(nifty_closes, axis=1).dropna(how="all"))
        if not isinstance(nifty.index, pd.DatetimeIndex):
            nifty.index = pd.to_datetime(nifty.index)

        quads: dict[str, str] = {}
        for grp, syms in INDUSTRY_GROUPS.items():
            grp_closes = []
            for s in syms:
                if s in stocks and len(stocks[s]) >= 80:
                    c = stocks[s]["Close"].dropna()
                    if not isinstance(c.index, pd.DatetimeIndex):
                        c.index = pd.to_datetime(c.index)
                    grp_closes.append(c)
            if len(grp_closes) < 2:
                continue
            # BUG-FIX: rebase-to-100 (was raw price avg, single high-priced stock skewed sector)
            grp_idx   = equal_weight_index(pd.concat(grp_closes, axis=1).dropna(how="all"))
            common    = grp_idx.index.intersection(nifty.index)
            if len(common) < 80:
                continue
            rs_daily  = grp_idx[common] / nifty[common]
            # BUG-FIX: explicit W-FRI to match NSE weekly close (W defaults to W-SUN)
            rs_weekly = rs_daily.resample("W-FRI").last().dropna()
            if len(rs_weekly) < 12:
                continue
            rm       = rs_weekly.rolling(26, min_periods=10).mean()
            rs_ratio = (rs_weekly / rm * 100).dropna()
            if len(rs_ratio) < 6:
                continue
            rs_mom   = (rs_ratio / rs_ratio.shift(4) * 100).dropna()
            if len(rs_mom) < 1:
                continue
            cr, cm    = float(rs_ratio.iloc[-1]), float(rs_mom.iloc[-1])
            quads[grp] = (
                "Leading"   if cr >= 100 and cm >= 100 else
                "Weakening" if cr >= 100 and cm <  100 else
                "Improving" if cr <  100 and cm >= 100 else
                "Lagging"
            )
        return quads
    except Exception:
        return {}


# ── Per-stock metrics ─────────────────────────────────────────────────────────

def _stock_metrics(symbol: str, df: pd.DataFrame) -> dict | None:
    try:
        # Skip ETFs immediately — they pass SERIES==EQ filter in bhavcopy
        if _is_etf(symbol):
            return None

        close = df["Close"].dropna()
        vol   = df["Volume"].dropna()
        if len(close) < MIN_BARS:
            return None

        cur = float(close.iloc[-1])

        # Price returns
        r1m = (cur / float(close.iloc[-21])  - 1) * 100 if len(close) >= 21  else 0.0
        r3m = (cur / float(close.iloc[-63])  - 1) * 100 if len(close) >= 63  else 0.0
        r6m = (cur / float(close.iloc[-126]) - 1) * 100 if len(close) >= 126 else 0.0

        # Exclude stocks already in heavy momentum or in freefall
        if r3m > MAX_3M_RET or r3m < MIN_3M_RET:
            return None

        # ADTV — recent vs historical baseline (60–180 days ago).
        # Align Close + Volume on common non-NaN index before multiplying so a
        # single mismatched dropna day doesn't silently pair wrong rows.
        _cv = df[["Close", "Volume"]].dropna()
        if len(_cv) < 20:
            return None
        turnover     = _cv["Close"] * _cv["Volume"]
        adtv_recent  = float(turnover.iloc[-20:].mean()) / 1e7
        if adtv_recent < MIN_ADTV_CR:
            return None
        adtv_hist    = float(turnover.iloc[-180:-60].mean()) / 1e7 if len(turnover) >= 180 else adtv_recent
        adtv_growth  = round(adtv_recent / adtv_hist, 2) if adtv_hist > 0.01 else 1.0

        # Volume expansion — recent 20d vs 200d baseline
        vol_r20  = float(vol.iloc[-20:].mean())
        vol_a200 = float(vol.iloc[-200:].mean()) if len(vol) >= 200 else float(vol.mean())
        vol_exp  = round(vol_r20 / vol_a200, 2) if vol_a200 > 0 else 1.0

        # Base detection
        base_len, base_rng, pos_in_base = _detect_base(close)
        if base_len < MIN_BASE_DAYS or base_rng > MAX_BASE_RANGE:
            return None
        base_months = round(base_len / 21, 1)

        # Stage
        stg     = stage_analysis(close)
        stg_lbl = stage_label(stg)
        if stg in (3, 4):          # exclude topping / downtrend
            return None

        # Moving averages
        ma50  = float(close.rolling(50).mean().iloc[-1])  if len(close) >= 50  else None
        ma200 = float(close.rolling(200).mean().iloc[-1]) if len(close) >= 200 else None

        # RS delta: today's 3M return vs the 3M return 63 bars ago.
        # Both windows must span the SAME number of intervals so the
        # rank-delta comparison is apples-to-apples.
        # r3m       = close[-1] / close[-63]  → spans 62 intervals.
        # r3m_old   = close[-64] / close[-126] → spans 62 intervals.
        if len(close) >= 126:
            r3m_old = (float(close.iloc[-64]) / float(close.iloc[-126]) - 1) * 100
        else:
            r3m_old = r3m  # insufficient history → no delta

        # 52-week high position
        w52   = close.iloc[-252:] if len(close) >= 252 else close
        hi52  = float(w52.max()); lo52 = float(w52.min())
        p52w  = round((cur - lo52) / (hi52 - lo52) * 100, 1) if hi52 > lo52 else 50.0
        pft_h = round((cur / hi52 - 1) * 100, 2)

        return {
            "symbol":         symbol,
            "price":          round(cur, 2),
            "adtv_cr":        round(adtv_recent, 1),
            "adtv_growth":    adtv_growth,
            "vol_expansion":  vol_exp,
            "r1m":            round(r1m, 2),
            "r3m":            round(r3m, 2),
            "r6m":            round(r6m, 2),
            "r3m_old":        round(r3m_old, 2),
            "base_len":       base_len,
            "base_months":    base_months,
            "base_range_pct": base_rng,
            "pos_in_base":    pos_in_base,
            "pos_52w":        p52w,
            "pct_from_high":  pft_h,
            "above_ma50":     bool(ma50  and cur > ma50),
            "above_ma200":    bool(ma200 and cur > ma200),
            "stage":          stg,
            "stage_lbl":      stg_lbl,
            "group":          _SYM_TO_GROUP.get(symbol, ""),
            # Filled after full-scan ranking:
            "score":          0.0,
            "rs_rating":      50,
            "rs_delta":       0.0,
            "sector_quad":    "",
            "label":          "",
        }
    except Exception:
        return None


# ── Main entry ────────────────────────────────────────────────────────────────

def run_early_mover_scan(progress_callback=None) -> dict:
    """
    Full early-mover scan. Cached 1 hour. Zero NSE API calls.
    """
    if _cache["data"] and time.time() - _cache["ts"] < CACHE_TTL:
        return _cache["data"]

    # ── Load stocks ───────────────────────────────────────────────────────────
    stocks = _load_all_stocks(progress_callback)
    if not stocks:
        return {"stocks": [], "computed_at": int(time.time()),
                "total_scanned": 0, "total_qualified": 0}

    total = len(stocks)

    # ── Sector quadrants ──────────────────────────────────────────────────────
    if progress_callback:
        progress_callback(0, total, "Computing sector RRG alignment…")
    quads      = _sector_quadrants(stocks)
    quad_score = {"Leading": 100, "Improving": 85, "Weakening": 45, "Lagging": 20}

    if progress_callback:
        progress_callback(0, total, f"Scanning {total} stocks for early mover signals…")

    # ── Per-stock metrics in parallel ─────────────────────────────────────────
    raw: list[dict] = []
    done = 0
    with ThreadPoolExecutor(max_workers=SCAN_WORKERS) as ex:
        futs = {ex.submit(_stock_metrics, sym, df): sym for sym, df in stocks.items()}
        for fut in as_completed(futs):
            done += 1
            if progress_callback and done % 300 == 0:
                progress_callback(done, total, f"Scanning… {done}/{total}")
            r = fut.result()
            if r is not None:
                raw.append(r)

    if not raw:
        return {"stocks": [], "computed_at": int(time.time()),
                "total_scanned": total, "total_qualified": 0}

    # ── Ranking phase ─────────────────────────────────────────────────────────
    df = pd.DataFrame(raw)

    def _prank(col: str) -> pd.Series:
        return df[col].rank(pct=True) * 100

    # RS rating: percentile rank of 3M return within this scan's universe
    df["rs_rating"] = _prank("r3m").round(0).astype(int).clip(1, 99)

    # RS delta: how much has the RS rank improved vs 90 days ago
    df["rs_delta"]  = (_prank("r3m") - _prank("r3m_old")).round(1)

    # Base quality score: longer × tighter = higher
    df["_base_q"] = df["base_months"] * (40.0 / df["base_range_pct"].clip(lower=4.0))

    # Sector alignment bonus
    df["sector_quad"]  = df["group"].map(lambda g: quads.get(g, ""))
    df["_sector_bonus"] = df["sector_quad"].map(lambda q: quad_score.get(q, 30))

    # Composite score (percentile-ranked components)
    df["score"] = (
        0.30 * _prank("adtv_growth")    +
        0.25 * _prank("_base_q")        +
        0.20 * _prank("vol_expansion")  +
        0.15 * df["rs_delta"].rank(pct=True) * 100 +
        0.10 * df["_sector_bonus"].rank(pct=True) * 100
    ).round(1)

    # ── Labels ────────────────────────────────────────────────────────────────
    def _label(row) -> str:
        sq  = row["sector_quad"]
        s   = row["score"]
        bm  = row["base_months"]
        brng = row["base_range_pct"]
        ag  = row["adtv_growth"]
        rd  = row["rs_delta"]

        # 🔥 Inflecting: sector improving/leading + RS rising + new money
        if sq in ("Improving", "Leading") and rd >= 8 and ag >= 1.8 and s >= 60:
            return "Inflecting"
        # 💎 Hidden Gem: long + tight base + ADTV quietly growing
        if bm >= 5.0 and brng <= 32 and ag >= 1.5 and s >= 50:
            return "Hidden Gem"
        # 🌱 Emerging: has early signals, needs watching
        if s >= 42:
            return "Emerging"
        return ""

    df["label"] = df.apply(_label, axis=1)

    # Key signals text (why this stock was flagged)
    def _signals(row) -> list[str]:
        tags = []
        if row["base_months"] >= 6:
            tags.append(f"{row['base_months']:.0f}M base")
        if row["adtv_growth"] >= 2.0:
            tags.append(f"ADTV ×{row['adtv_growth']:.1f}")
        if row["vol_expansion"] >= 2.0:
            tags.append(f"Vol ×{row['vol_expansion']:.1f}")
        if row["rs_delta"] >= 10:
            tags.append(f"RS↑{row['rs_delta']:.0f}")
        if row["sector_quad"] in ("Improving", "Leading"):
            tags.append(f"Sector {row['sector_quad']}")
        if row["pos_in_base"] >= 0.75:
            tags.append("Near top of base")
        return tags

    df["signals"] = df.apply(_signals, axis=1)

    # Keep only labeled stocks
    df = df[df["label"] != ""].copy()
    df = df.sort_values("score", ascending=False).reset_index(drop=True)
    df = df.drop(columns=["_base_q", "_sector_bonus"], errors="ignore")

    out = {
        "stocks":          df.to_dict(orient="records"),
        "computed_at":     int(time.time()),
        "total_scanned":   total,
        "total_qualified": len(df),
        "inflecting":      int((df["label"] == "Inflecting").sum()),
        "hidden_gems":     int((df["label"] == "Hidden Gem").sum()),
        "emerging":        int((df["label"] == "Emerging").sum()),
    }
    _cache["data"] = out
    _cache["ts"]   = time.time()

    if progress_callback:
        n = out["inflecting"]; h = out["hidden_gems"]; e = out["emerging"]
        progress_callback(total, total,
                          f"Done — {len(df)} signals · 🔥{n} Inflecting · 💎{h} Hidden Gems · 🌱{e} Emerging")
    return out
