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
_SYM_TO_GROUP: dict[str, str] = {}
for _grp, _syms in INDUSTRY_GROUPS.items():
    for _s in _syms:
        _SYM_TO_GROUP[_s] = _grp

# ── Config ────────────────────────────────────────────────────────────────────
MIN_BARS    = 130    # need at least 6M of history (≈130 trading days)
MIN_PRICE   = 30.0  # filter sub-₹30 (penny stocks)
MIN_ADTV_CR = 0.5    # Minimal liquidity guard only — universe filtered by Nifty500 membership
SCAN_WORKERS = 8
_cache   = {"data": None, "ts": 0}
CACHE_TTL = 3600    # 1 hour


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
    # Filter to Nifty50 ∪ NiftyNext50 ∪ Nifty500 ∪ NiftySmallcap250
    try:
        from nse_stocks import get_nifty500_symbols
        _universe = set(get_nifty500_symbols())
    except Exception:
        _universe = set()
    combined = pd.concat(frames, ignore_index=True).sort_values("Date")
    stocks: dict[str, pd.DataFrame] = {}
    for sym, grp in combined.groupby("Symbol"):
        if _universe and sym not in _universe:
            continue
        g = grp.set_index("Date")[["Open", "High", "Low", "Close", "Volume"]]
        g = g[~g.index.duplicated(keep="last")].sort_index()
        if len(g) >= MIN_BARS:
            stocks[sym] = g
    return stocks


# ── Per-stock metric computation ──────────────────────────────────────────────

def _metrics(symbol: str, df: pd.DataFrame) -> dict | None:
    """Compute all momentum metrics for one stock. Returns None if filtered out."""
    try:
        close = df["Close"].dropna()
        vol   = df["Volume"].dropna()
        if len(close) < MIN_BARS:
            return None

        cur = float(close.iloc[-1])
        if cur < MIN_PRICE:
            return None

        # Liquidity: average daily turnover in ₹ Cr (last 20 sessions)
        adtv_cr = float((df["Close"] * df["Volume"]).iloc[-20:].mean()) / 1e7
        if adtv_cr < MIN_ADTV_CR:
            return None

        # ── Returns ──────────────────────────────────────────────────────────
        r1m = (cur / float(close.iloc[-22])  - 1) * 100 if len(close) > 22  else None
        r3m = (cur / float(close.iloc[-66])  - 1) * 100 if len(close) > 66  else None
        r6m = (cur / float(close.iloc[-130]) - 1) * 100 if len(close) > 130 else None
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

        # ── Trend strength: consecutive closes above MA50 ─────────────────────
        if ma50 is not None:
            above_streak = int((close.iloc[-20:] > close.rolling(50).mean().iloc[-20:]).sum())
        else:
            above_streak = 0

        # ── Stage & trend ─────────────────────────────────────────────────────
        stg     = stage_analysis(close)
        stg_lbl = stage_label(stg)

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
            "stage":          stg,
            "stage_lbl":      stg_lbl,
            "group":          _SYM_TO_GROUP.get(symbol, ""),
            # Filled after ranking:
            "score":          0.0,
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
        return df[col].rank(pct=True) * 100

    r1m_rank = _prank("r1m")
    r3m_rank = _prank("r3m")
    r6m_rank = _prank("r6m")
    vol_rank = _prank("vol_ratio")

    # RS Rating = percentile rank of 3M return (matches IBD convention)
    df["rs_rating"] = r3m_rank.round(0).astype(int).clip(1, 99)

    # Composite momentum score
    df["score"] = (
        0.40 * r1m_rank +
        0.30 * r3m_rank +
        0.20 * r6m_rank +
        0.10 * vol_rank
    ).round(1)

    # ── Tier labels ───────────────────────────────────────────────────────────
    def _tier(row) -> str:
        s, rs = row["score"], row["rs_rating"]
        if s >= 85 and rs >= 80:
            return "Elite"
        if s >= 70:
            return "Strong"
        if s >= 55:
            return "Rising"
        return ""

    df["tier"] = df.apply(_tier, axis=1)

    # ── Sort and output ───────────────────────────────────────────────────────
    df = df.sort_values("score", ascending=False).reset_index(drop=True)

    out = {
        "stocks":          df.to_dict(orient="records"),
        "computed_at":     int(time.time()),
        "total_scanned":   total_stocks,
        "total_qualified": len(raw),
    }
    _cache["data"] = out
    _cache["ts"]   = time.time()

    if progress_callback:
        progress_callback(total_stocks, total_stocks,
                          f"Done — {len(raw)} stocks scored · {df[df['tier']=='Elite'].shape[0]} Elite")
    return out
