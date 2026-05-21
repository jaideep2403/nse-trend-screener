import time
import pandas as pd
import numpy as np
from nse_stocks import get_nse_tickers
from data_fetcher import fetch_ohlcv, _pkl_stats
from analysis_utils import volume_baseline
import warnings
warnings.filterwarnings("ignore")

MIN_BARS = 200   # ~10 months of trading data

# Shared OHLCV cache — populated by screener, reused by sector analysis
ohlcv_cache: dict = {}


# ── Per-stock metrics ──────────────────────────────────────────────────────────

def compute_metrics(ticker: str, df: pd.DataFrame):
    try:
        close = df["Close"].dropna()
        if len(close) < MIN_BARS:
            return None

        price   = float(close.iloc[-1])
        ma50    = float(close.rolling(50).mean().iloc[-1])
        ma100   = float(close.rolling(100).mean().iloc[-1])
        ma200   = float(close.rolling(200).mean().iloc[-1])
        ath     = float(close.max())
        pct_ath = round((price - ath) / ath * 100, 2)

        volume  = df["Volume"].dropna()
        # Median-based volume baseline (canonical) — robust to single-day
        # expiry/block-deal volume spikes that a raw 20-day mean would inflate by 2-3x.
        adtv    = volume_baseline(volume, window=20, use_median=True) if len(volume) >= 20 else 0
        adtv_cr = round(adtv * price / 1e7, 2)

        # Use None for missing-history sentinel (not 0.0) so a stock with
        # genuine 0% return is distinguishable from one missing the lookback
        # window. The IBD composite below treats None as "no signal" (NaN).
        ret_12m = (price / float(close.iloc[-252]) - 1) * 100 if len(close) >= 252 else None
        ret_6m  = (price / float(close.iloc[-126]) - 1) * 100 if len(close) >= 126 else None
        ret_3m  = (price / float(close.iloc[-63])  - 1) * 100 if len(close) >= 63  else None

        return {
            "Symbol":     ticker.replace(".NS", ""),
            "Ticker":     ticker,
            "Close":      round(price, 2),
            "ADTV_Cr":    adtv_cr,
            "MA50":       round(ma50, 2),
            "MA100":      round(ma100, 2),
            "MA200":      round(ma200, 2),
            "ATH":        round(ath, 2),
            "%_from_ATH": pct_ath,
            "_r3m":       ret_3m,
            "_r6m":       ret_6m,
            "_r12m":      ret_12m,
        }
    except Exception:
        return None


# ── IBD-style RS score ─────────────────────────────────────────────────────────

def compute_ibd_score(df_all: pd.DataFrame) -> pd.DataFrame:
    """
    IBD-style composite RS score (0-99 percentile).

    BUG-FIX: prior formula `0.4*r3m + 0.2*r6m + 0.4*r12m` triple-counted recent
    momentum because r6m INCLUDES r3m and r12m INCLUDES BOTH. A stock that
    surged 30% in the last 3 months saw that move counted in r3m, again as
    part of r6m, and a third time as part of r12m → effective weight ≈ 0.6
    on the most recent quarter, biasing top picks toward short-term runners.

    Correct IBD method: separate NON-OVERLAPPING quarter returns weighted
    40/20/20/20 with the most recent quarter highest. Each price-action
    period is counted exactly once.
    """
    df = df_all.copy()
    # Convert to numeric; preserve NaN for stocks missing the window (vs forcing 0
    # which would treat missing-data stocks the same as 0%-return stocks).
    for col in ["_r3m", "_r6m", "_r12m"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Non-overlapping quarter returns derived from cumulative windows:
    #   q1 = last 3 months                             (from _r3m)
    #   q2 = months 4-6  → (1+_r6m) / (1+_r3m)  - 1
    #   q3/q4 = months 7-12 → (1+_r12m) / (1+_r6m) - 1  (split equally as q3+q4)
    # All inputs are in percent; convert to fractions for compounding.
    r3 = df["_r3m"]  / 100.0
    r6 = df["_r6m"]  / 100.0
    r12 = df["_r12m"] / 100.0
    q1 = r3
    q2 = (1 + r6)  / (1 + r3)  - 1
    q34 = (1 + r12) / (1 + r6) - 1   # months 7-12 combined
    # Weight 40 on q1, 20 on q2, 40 on months 7-12 combined (= 20+20 for q3+q4).
    df["_composite"] = (0.40 * q1 + 0.20 * q2 + 0.40 * q34) * 100.0
    df["IBD_score"]  = (df["_composite"].rank(pct=True) * 99).round(1)
    df.drop(columns=["_composite", "_r3m", "_r6m", "_r12m"], inplace=True)
    return df


# ── Filters ────────────────────────────────────────────────────────────────────

def apply_filters(df: pd.DataFrame, params: dict):
    d = df.copy()
    funnel = [("Universe (fetched)", len(d))]

    d = d[d["Close"] > d["MA50"]];    funnel.append(("Close > MA50",  len(d)))
    d = d[d["MA50"]  > d["MA100"]];   funnel.append(("MA50 > MA100",  len(d)))
    d = d[d["MA100"] > d["MA200"]];   funnel.append(("MA100 > MA200", len(d)))

    def has(k):
        v = params.get(k)
        return v is not None and str(v).strip() != ""

    if has("min_adtv"):
        d = d[d["ADTV_Cr"] >= float(params["min_adtv"])]
        funnel.append((f"ADTV ≥ {params['min_adtv']} Cr", len(d)))
    if has("max_pct_from_ath"):
        d = d[d["%_from_ATH"] >= float(params["max_pct_from_ath"])]
        funnel.append((f"%ATH ≥ {params['max_pct_from_ath']}", len(d)))
    if has("min_ibd"):
        d = d[d["IBD_score"] >= float(params["min_ibd"])]
        funnel.append((f"IBD ≥ {params['min_ibd']}", len(d)))

    d = d.sort_values("IBD_score", ascending=False)
    return d, funnel


# ── Main screener ──────────────────────────────────────────────────────────────

def run_screener(params=None, progress_callback=None):
    if params is None:
        params = {}
    # Universe is already Nifty Total Market 750 via get_nse_tickers() — no ADTV proxy needed

    tickers = list(dict.fromkeys(get_nse_tickers()))
    total   = len(tickers)

    _, fresh = _pkl_stats()
    cache_note = f"({fresh} cached)" if fresh > 0 else "(cold start — downloading from NSE…)"

    if progress_callback:
        progress_callback(0, total, f"Starting scan {cache_note}")

    # Fetch all OHLCV via NSE bhavcopy (no Yahoo Finance, no rate limits)
    ohlcv = fetch_ohlcv(
        tickers,
        min_bars=MIN_BARS,
        progress_callback=progress_callback,
    )

    # Populate shared cache for sector analysis
    ohlcv_cache.clear()
    ohlcv_cache.update(ohlcv)

    rows = [r for t, df in ohlcv.items()
            for r in [compute_metrics(t, df)] if r]

    if progress_callback:
        progress_callback(total, total,
                          f"Computing scores for {len(rows)} stocks…")

    if not rows:
        return pd.DataFrame(), [("Universe (fetched)", 0)]

    df_all     = pd.DataFrame(rows)
    df_all     = compute_ibd_score(df_all)
    df_filtered, funnel = apply_filters(df_all, params)

    display_cols = [
        "Symbol", "Ticker", "Close", "ADTV_Cr",
        "MA50", "MA100", "MA200", "ATH", "%_from_ATH", "IBD_score"
    ]
    return df_filtered[display_cols].reset_index(drop=True), funnel
