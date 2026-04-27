import time
import pandas as pd
import numpy as np
from nse_stocks import get_nse_tickers
from data_fetcher import fetch_ohlcv, _pkl_stats
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
        adtv    = float(volume.rolling(20).mean().iloc[-1]) if len(volume) >= 20 else 0
        adtv_cr = round(adtv * price / 1e7, 2)

        ret_12m = (price / float(close.iloc[-252]) - 1) * 100 if len(close) >= 252 else 0.0
        ret_6m  = (price / float(close.iloc[-126]) - 1) * 100 if len(close) >= 126 else 0.0
        ret_3m  = (price / float(close.iloc[-63])  - 1) * 100 if len(close) >= 63  else 0.0

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
    df = df_all.copy()
    for col in ["_r3m", "_r6m", "_r12m"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
    df["_composite"] = df["_r3m"] * 0.4 + df["_r6m"] * 0.2 + df["_r12m"] * 0.4
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
    # Universe is already Nifty 500 (large-cap) via get_nse_tickers() — no ADTV proxy needed

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
