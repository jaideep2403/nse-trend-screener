import time
import yfinance as yf
import pandas as pd
import numpy as np
from nse_stocks import get_nse_tickers
import warnings
warnings.filterwarnings("ignore")

BATCH_SIZE = 30           # tickers per yf.download call
BATCH_DELAY = 1.5         # seconds between batches
MIN_BARS = 210


def fetch_batch(tickers, period="2y", retries=3):
    """Bulk-download OHLCV for a batch of tickers. Returns dict[ticker -> DataFrame]."""
    for attempt in range(retries):
        try:
            data = yf.download(
                tickers=" ".join(tickers),
                period=period,
                progress=False,
                auto_adjust=True,
                group_by="ticker",
                threads=True,
            )
            if data is None or data.empty:
                time.sleep(3 * (attempt + 1))
                continue
            result = {}
            for t in tickers:
                try:
                    if len(tickers) == 1:
                        df = data
                    else:
                        df = data[t] if t in data.columns.get_level_values(0) else None
                    if df is None or df.empty:
                        continue
                    df = df.dropna(how="all")
                    if len(df) >= MIN_BARS:
                        result[t] = df
                except Exception:
                    continue
            return result
        except Exception:
            time.sleep(3 * (attempt + 1))
    return {}


def compute_metrics(ticker, df, mkt_cap=0):
    try:
        close = df["Close"].dropna()
        if len(close) < MIN_BARS:
            return None

        price = float(close.iloc[-1])
        ma50  = float(close.rolling(50).mean().iloc[-1])
        ma100 = float(close.rolling(100).mean().iloc[-1])
        ma200 = float(close.rolling(200).mean().iloc[-1])
        ath   = float(close.max())
        pct_from_ath = round((price - ath) / ath * 100, 2)

        volume = df["Volume"].dropna()
        adtv_shares = float(volume.rolling(20).mean().iloc[-1]) if len(volume) >= 20 else 0
        adtv_cr = round(adtv_shares * price / 1e7, 2)
        # Use ADTV-based proxy for market cap (since individual yf.Ticker calls get rate-limited)
        # Approximation: real market cap is fetched lazily if needed elsewhere.
        mkt_cap_cr = round(mkt_cap / 1e7, 2) if mkt_cap else 0

        ret_12m = (price / float(close.iloc[-252]) - 1) * 100 if len(close) >= 252 else None
        ret_6m  = (price / float(close.iloc[-126]) - 1) * 100 if len(close) >= 126 else None
        ret_3m  = (price / float(close.iloc[-63])  - 1) * 100 if len(close) >= 63  else None

        return {
            "Symbol":     ticker.replace(".NS", ""),
            "Ticker":     ticker,
            "Close":      round(price, 2),
            "MktCap_Cr":  mkt_cap_cr,
            "ADTV_Cr":    adtv_cr,
            "MA50":       round(ma50, 2),
            "MA100":      round(ma100, 2),
            "MA200":      round(ma200, 2),
            "ATH":        round(ath, 2),
            "%_from_ATH": pct_from_ath,
            "Ret_3M":     round(ret_3m, 2) if ret_3m is not None else None,
            "Ret_6M":     round(ret_6m, 2) if ret_6m is not None else None,
            "Ret_12M":    round(ret_12m, 2) if ret_12m is not None else None,
        }
    except Exception:
        return None


def compute_ibd_score(df_all):
    """Rank-based RS score (1-99) using weighted multi-period returns."""
    weights = {"Ret_3M": 0.4, "Ret_6M": 0.2, "Ret_12M": 0.4}
    df_all = df_all.copy()
    for col, w in weights.items():
        df_all[col] = pd.to_numeric(df_all[col], errors="coerce").fillna(0)

    df_all["composite"] = sum(df_all[c] * w for c, w in weights.items())
    df_all["IBD_score"] = (
        df_all["composite"].rank(pct=True) * 99
    ).round(1)
    df_all.drop(columns=["composite"], inplace=True)
    return df_all


def apply_filters(df, params):
    """Apply screening filters; returns (df, funnel)."""
    d = df.copy()
    funnel = [("Universe (screened)", len(d))]

    d = d[d["Close"] > d["MA50"]];    funnel.append(("Close > MA50", len(d)))
    d = d[d["MA50"]  > d["MA100"]];   funnel.append(("MA50 > MA100", len(d)))
    d = d[d["MA100"] > d["MA200"]];   funnel.append(("MA100 > MA200", len(d)))

    def has(k):
        v = params.get(k)
        return v is not None and str(v).strip() != ""

    if has("min_mktcap"):
        d = d[d["MktCap_Cr"] >= float(params["min_mktcap"])]
        funnel.append((f"MktCap ≥ {params['min_mktcap']} Cr", len(d)))
    if has("max_mktcap"):
        d = d[d["MktCap_Cr"] <= float(params["max_mktcap"])]
        funnel.append((f"MktCap ≤ {params['max_mktcap']} Cr", len(d)))
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


def fetch_market_caps(tickers):
    """Fetch market caps in bulk via yf.Tickers — one call, reduces rate-limiting."""
    caps = {}
    try:
        t_str = " ".join(tickers)
        tk = yf.Tickers(t_str)
        for t in tickers:
            try:
                mc = tk.tickers[t].fast_info.market_cap
                caps[t] = mc if mc and mc > 0 else 0
            except Exception:
                caps[t] = 0
            time.sleep(0.15)
    except Exception:
        pass
    return caps


def run_screener(params=None, progress_callback=None):
    if params is None:
        params = {}

    tickers = get_nse_tickers()
    tickers = list(dict.fromkeys(tickers))  # dedupe
    rows = []
    total = len(tickers)

    # ---- Phase 1: Bulk-fetch OHLCV in batches ----
    ohlcv = {}
    for i in range(0, total, BATCH_SIZE):
        batch = tickers[i:i + BATCH_SIZE]
        if progress_callback:
            progress_callback(min(i + BATCH_SIZE, total), total,
                              f"Fetching batch {i//BATCH_SIZE + 1}")
        data = fetch_batch(batch)
        ohlcv.update(data)
        time.sleep(BATCH_DELAY)

    # ---- Phase 2: Compute metrics (no market cap yet) ----
    for t, df in ohlcv.items():
        row = compute_metrics(t, df, mkt_cap=0)
        if row:
            rows.append(row)

    if not rows:
        return pd.DataFrame(), [("Universe (screened)", 0)]

    df_all = pd.DataFrame(rows)
    df_all = compute_ibd_score(df_all)

    # ---- Phase 3: Fetch market caps ONLY for uptrend survivors ----
    uptrend = df_all[
        (df_all["Close"] > df_all["MA50"]) &
        (df_all["MA50"]  > df_all["MA100"]) &
        (df_all["MA100"] > df_all["MA200"])
    ]
    if progress_callback:
        progress_callback(total, total, f"Fetching market caps for {len(uptrend)} uptrend stocks")
    survivor_tickers = uptrend["Ticker"].tolist()
    caps = fetch_market_caps(survivor_tickers) if survivor_tickers else {}
    df_all["MktCap_Cr"] = df_all.apply(
        lambda r: round(caps.get(r["Ticker"], 0) / 1e7, 2), axis=1
    )

    # ---- Phase 4: Apply all filters ----
    df_filtered, funnel = apply_filters(df_all, params)

    display_cols = [
        "Symbol", "Ticker", "Close", "MktCap_Cr", "ADTV_Cr",
        "MA50", "MA100", "MA200", "ATH", "%_from_ATH", "IBD_score"
    ]
    return df_filtered[display_cols].reset_index(drop=True), funnel
