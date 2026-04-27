"""
Edge Engine — the meta-layer that ties all scanners into a measurable edge.

Provides four core capabilities, all using ONLY local bhavcopy cache:

1. Market Regime Detector  — Distribution Day count + Follow-Through Day signal
2. Setup Quality Score     — composite 0-100 ranking across all scanners
3. Failed Breakout Detector— exit signal automation (7-8% rule, MA50 break, etc.)
4. Backtester              — walk-forward validation of any scanner output

Zero new NSE API calls. All computations from cached EOD data.
"""
import time
import math
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from data_fetcher import _weekdays_back, _download_one_day
from analysis_utils import stage_analysis, stage_label
from industry_groups import INDUSTRY_GROUPS

# ── Cache ─────────────────────────────────────────────────────────────────────
_cache    = {"data": None, "ts": 0}
CACHE_TTL = 1800  # 30 min

def invalidate_cache():
    """Force-clear the edge engine cache so the next run always re-scores from scratch."""
    _cache["data"] = None
    _cache["ts"]   = 0

# ── Symbol → Group lookup ─────────────────────────────────────────────────────
_SYM_TO_GROUP: dict[str, str] = {}
for _grp, _syms in INDUSTRY_GROUPS.items():
    for _s in _syms:
        _SYM_TO_GROUP[_s] = _grp

_NIFTY_SYMS = [
    "RELIANCE", "TCS", "HDFCBANK", "ICICIBANK", "INFOSYS",
    "HINDUNILVR", "SBIN", "BHARTIARTL", "ITC", "KOTAKBANK",
    "AXISBANK", "WIPRO", "HCLTECH", "MARUTI", "BAJFINANCE",
    "TITAN", "NTPC", "POWERGRID", "NESTLEIND", "SUNPHARMA",
]

# ── ETF exclusion ─────────────────────────────────────────────────────────────
# Symbols that END with these strings are always ETFs/funds
_ETF_END = ("ETF", "BEES", "FUND", "BENCHMARK", "NIFTY1", "IETF", "BETF")

# Symbols that CONTAIN these strings are always ETFs/funds
# "NIFTY" and "SENSEX" never appear in real equity company tickers
_ETF_HAS = (
    "NIFTY", "SENSEX",                          # index products
    "LIQUID", "IETF", "CPSE", "GOLDETF",        # fund types
    "SILVETF", "MAKEINDIA", "MAFANG",
    "MIDCAPETF", "INFRABEES", "BANKBEES",
    "JUNIORBEES", "PSUBNKBEES", "NIFTYETF",
    "SENSEXETF", "SETFNIF", "SETF", "HNGSNG",
    "CPSEETF", "BBETF", "ABSLNN50ET",
    "GSEC10", "GSEC5", "GILT5Y", "GILT10",
    "LTGILT", "LIQGRW", "DIVOPPB",
    "CONSUMB", "MANUFGB", "MID150B",
    "ELIQUID", "CASHIET",
    "GROWWG", "GROWWN",                         # Groww Gold/Nifty ETFs
)

# Exact symbol matches that are known ETFs not caught by patterns above
_ETF_EXACT = frozenset({
    # Axis ETFs
    "AXISGOLD", "AXISILVER", "AXISBPSETF", "AXISBNKETF",
    "AXISCETF", "AXISHCETF", "AXISTECETF", "AXISVALUE",
    # Aditya Birla / ABSL ETFs
    "AONEGOLD", "AONESILVER", "AONELIQUID",
    "ABSLLIQUID", "ABSL10BANK",
    # Bank/PSU factor ETFs
    "BANKBETA", "BANKADD", "BANK10ADD", "BANKPSU", "EBANKNIFTY",
    # BSL / BSE
    "BSLNIFTY", "BSLGOLDETF", "BSLSENETFG", "BSE500IETF",
    # Edelweiss ETFs (E-prefix family)
    "EGOLD", "ESILVER", "ENIFTY", "ESENSEX",
    # Alphaetf / others
    "ALPHAETF", "ALPL30IETF", "AUTOBEES", "AUTOIETF",
    # BNP Paribas
    "BBNPPGOLD", "BBNPNBETF",
    # Choice / Deccan
    "CHOICEGOLD", "DECNGOLD",
    # Commodity / sector iETFs
    "COMMOIETF", "CONSUMIETF", "EVIETF", "FINIETF", "FMCGIETF",
    "HEALTHIETF", "INFRAIETF", "ITBEES", "ITETF", "ITIETF",
    "LOWVOLIETF", "METALIETF", "MIDCAPIETF", "MIDSELIETF", "MOM30IETF",
    # Gold ETFs
    "GOLDBEES", "GOLDETF", "GOLDIETF",
    "GOLD1", "GOLD360", "GOLDADD", "GOLDBETA", "GOLDBND", "GOLDCASE",
    "HDFCGOLD", "HSBCGOLD", "IVZINGOLD", "LICMFGOLD", "MOGOLD",
    "QGOLDHALF", "SHANTIGOLD", "TATAGOLD", "UNIONGOLD",
    # Silver ETFs
    "ESILVER", "MOSILVER", "SBISILVER", "HDFCSILVER",
    "SILVER", "SILVER1", "SILVER360", "SILVERADD", "SILVERAG",
    "SILVERBETA", "SILVERBND", "SILVERCASE", "SILVERTUC",
    # Gilt / bond ETFs
    "GILT10BETA", "GILT5BETA", "EUROBOND", "GOLDBND", "SILVERBND",
    # HDFC ETFs
    "HDFCSILVER",
    # Invesco
    "IVZINGOLD", "IVZINNIFTY",
    # Mirae / Motilal
    "MOSILVER", "MOGSEC",
    "MONIFTY100", "MONIFTY500",
    # Misc
    "IDFNIFTYET", "NIFTY100EW", "NIFTYADD", "NIFTYBETA",
    "NIFTYCASE", "NIFTYQLITY", "QNIFTY",
    "SENSEXADD", "SENSEXBETA",
    "ABGSEC",
})

def _is_etf(sym: str) -> bool:
    s = sym.upper()
    return (s in _ETF_EXACT or
            any(s.endswith(x) for x in _ETF_END) or
            any(k in s for k in _ETF_HAS))


# ── Data loader ───────────────────────────────────────────────────────────────

def _load_stocks(progress_callback=None, days: int = 400) -> dict[str, pd.DataFrame]:
    # Universe: Nifty50 ∪ NiftyNext50 ∪ Nifty500 ∪ NiftySmallcap250
    try:
        from nse_stocks import get_nifty500_symbols
        universe = set(get_nifty500_symbols())
    except Exception:
        universe = set()   # empty → no filter (safe fallback)

    dates  = _weekdays_back(days)
    total  = len(dates)
    frames = []
    for i, dt in enumerate(dates):
        df = _download_one_day(dt)
        if df is not None:
            # Filter to universe immediately — keeps memory small
            if universe:
                df = df[df["Symbol"].isin(universe)]
            frames.append(df)
        if progress_callback and i % 40 == 0:
            progress_callback(i, total, f"Loading bhavcopy… {i}/{total}")
    if not frames:
        return {}
    combined = pd.concat(frames, ignore_index=True).sort_values("Date")
    stocks: dict[str, pd.DataFrame] = {}
    for sym, grp in combined.groupby("Symbol"):
        if universe and sym not in universe:
            continue
        g = grp.set_index("Date")[["Open", "High", "Low", "Close", "Volume"]]
        if not isinstance(g.index, pd.DatetimeIndex):
            g.index = pd.to_datetime(g.index)
        g = g[~g.index.duplicated(keep="last")].sort_index()
        if len(g) >= 60:
            stocks[sym] = g
    return stocks


def _build_nifty_proxy(stocks: dict) -> pd.DataFrame | None:
    """Build a Nifty proxy index (price + volume) from large-cap constituents."""
    closes, vols = [], []
    for sym in _NIFTY_SYMS:
        df = stocks.get(sym)
        if df is not None and len(df) >= 100:
            closes.append(df["Close"].dropna())
            vols.append(df["Volume"].dropna())
    if len(closes) < 5:
        return None
    px  = pd.concat(closes, axis=1).dropna(how="all").mean(axis=1)
    vol = pd.concat(vols,   axis=1).dropna(how="all").mean(axis=1)
    out = pd.DataFrame({"Close": px, "Volume": vol}).dropna()
    return out if len(out) >= 80 else None


# ──────────────────────────────────────────────────────────────────────────────
# 1. MARKET REGIME — Distribution Day count + Follow-Through Day
# ──────────────────────────────────────────────────────────────────────────────

def detect_market_regime(nifty: pd.DataFrame) -> dict:
    """
    Distribution Day = Nifty closes ≤ -0.2% on volume HIGHER than prior day.
    5+ D-Days in last 25 sessions = market under institutional selling.

    Follow-Through Day = on day 4-7 after a recent low, Nifty closes UP ≥ 1.4%
    on volume higher than prior day. Confirms a new uptrend.

    Returns regime: "Confirmed Uptrend" | "Uptrend Under Pressure" | "Correction"
    """
    if nifty is None or len(nifty) < 30:
        return {"regime": "Unknown", "dday_count": 0, "ftd_active": False, "details": []}

    n     = nifty.tail(60).copy()
    n["pct_chg"]   = n["Close"].pct_change() * 100
    n["vol_chg"]   = n["Volume"].pct_change()
    n["dday"]      = (n["pct_chg"] <= -0.2) & (n["Volume"] > n["Volume"].shift(1))

    last25 = n.tail(25)
    dday_count = int(last25["dday"].sum())

    # Recent low + Follow-Through Day check
    recent_low_idx = n["Close"].tail(20).idxmin()
    days_since_low = (n.index[-1] - recent_low_idx).days if recent_low_idx else 999
    ftd_active = False
    ftd_day    = None
    if 3 <= (n.index.get_loc(recent_low_idx) if recent_low_idx in n.index else -1):
        # find FTD: day 4-7 after low with +1.4% on rising volume
        try:
            low_pos = n.index.get_loc(recent_low_idx)
            for i in range(low_pos + 4, min(low_pos + 8, len(n))):
                row = n.iloc[i]
                if row["pct_chg"] >= 1.4 and row["Volume"] > n.iloc[i-1]["Volume"]:
                    ftd_active = True
                    ftd_day    = n.index[i]
                    break
        except Exception:
            pass

    # Regime classification
    if dday_count >= 6:
        regime = "Correction"
    elif dday_count >= 4:
        regime = "Uptrend Under Pressure"
    elif ftd_active and dday_count <= 3:
        regime = "Confirmed Uptrend"
    else:
        regime = "Uptrend Under Pressure" if dday_count >= 3 else "Confirmed Uptrend"

    # Action recommendation
    action_map = {
        "Confirmed Uptrend":      ("✅ Buy Mode", "Full deployment — buy strongest setups",        "#22c55e"),
        "Uptrend Under Pressure": ("🟡 Cautious", "Selective — only highest-conviction setups",   "#eab308"),
        "Correction":             ("🔴 Cash",     "Do not initiate new longs — preserve capital", "#ef4444"),
        "Unknown":                ("⚪ Unknown",  "Insufficient data",                              "#94a3b8"),
    }
    label, advice, color = action_map[regime]

    # Latest 25 sessions detail
    detail = []
    for idx, row in last25.iterrows():
        if row["dday"]:
            detail.append({
                "date": idx.strftime("%d-%b"),
                "pct":  round(row["pct_chg"], 2),
                "vol_chg_pct": round(row["vol_chg"] * 100, 1),
            })

    return {
        "regime":      regime,
        "label":       label,
        "advice":      advice,
        "color":       color,
        "dday_count":  dday_count,
        "ftd_active":  ftd_active,
        "ftd_day":     ftd_day.strftime("%d-%b-%Y") if ftd_day is not None else None,
        "days_since_low": days_since_low,
        "details":     detail,
        "nifty_close": round(float(nifty["Close"].iloc[-1]), 2),
        "nifty_chg":   round(float(nifty["Close"].pct_change().iloc[-1]) * 100, 2),
    }


# ──────────────────────────────────────────────────────────────────────────────
# 2. BASE QUALITY VALIDATOR — O'Neil proper-base rules
# ──────────────────────────────────────────────────────────────────────────────

def validate_base(close: pd.Series) -> dict:
    """
    Score a base 0-100 based on O'Neil's proper-base criteria.

    Checks:
      - Base length (≥ 7 weeks = 35 trading days, ideal 7-15 weeks)
      - Base depth (≤ 33% from peak)
      - Prior uptrend before base (≥ 30%)
      - Tightness on right side
      - Volume drying up during base
    """
    if len(close) < 100:
        return {"score": 0, "valid": False, "reason": "Insufficient history"}

    # Detect base: longest recent window with range < 35%
    n = min(180, len(close) - 5)
    base_len = 0
    for days in range(n, 30, -5):
        slc = close.iloc[-days:]
        lo, hi = float(slc.min()), float(slc.max())
        if hi > lo and (hi - lo) / lo * 100 < 35:
            base_len = days
            break

    if base_len < 30:
        return {"score": 0, "valid": False, "reason": "No base detected (< 6 weeks)",
                "base_len": base_len, "depth": 0}

    base = close.iloc[-base_len:]
    base_lo, base_hi = float(base.min()), float(base.max())
    depth = (base_hi - base_lo) / base_hi * 100   # depth from peak

    # Prior uptrend (50 days before base)
    pre_base_end   = len(close) - base_len
    pre_base_start = max(0, pre_base_end - 50)
    if pre_base_end > pre_base_start + 10:
        pre = close.iloc[pre_base_start:pre_base_end]
        prior_uptrend = (float(pre.iloc[-1]) / float(pre.iloc[0]) - 1) * 100
    else:
        prior_uptrend = 0.0

    # Tightness on right side (last 10 bars vs whole base)
    right = base.iloc[-10:]
    right_range = (float(right.max()) - float(right.min())) / float(right.min()) * 100
    tightness   = max(0, 100 - right_range * 5)  # tighter = higher score

    # Score components (each 0-25)
    s_length = min(25, max(0, (base_len - 30) / 60 * 25))           # ideal 30-90
    s_depth  = 25 if depth <= 25 else max(0, 25 - (depth - 25))     # ideal ≤ 25%
    s_trend  = min(25, max(0, prior_uptrend / 30 * 25))             # need ≥ 30%
    s_tight  = min(25, tightness / 4)                                # right-side tightness

    score = round(s_length + s_depth + s_trend + s_tight, 1)
    valid = score >= 55 and depth <= 33 and base_len >= 30 and prior_uptrend >= 20

    return {
        "score":         score,
        "valid":         valid,
        "base_len":      base_len,
        "base_weeks":    round(base_len / 5, 1),
        "depth":         round(depth, 1),
        "prior_uptrend": round(prior_uptrend, 1),
        "tightness":     round(tightness, 1),
        "reason":        "" if valid else (
            "Depth > 33%" if depth > 33 else
            "Prior uptrend < 20%" if prior_uptrend < 20 else
            "Score below 55"
        ),
    }


# ──────────────────────────────────────────────────────────────────────────────
# 3. FAILED BREAKOUT / EXIT SIGNAL DETECTOR
# ──────────────────────────────────────────────────────────────────────────────

def detect_exit_signals(df: pd.DataFrame, entry_price: float | None = None,
                       entry_date: str | None = None) -> dict:
    """
    Returns active exit signals on a stock. If entry_price is given,
    also checks the 7-8% loss rule and breakeven trigger.
    """
    close = df["Close"].dropna()
    vol   = df["Volume"].dropna()
    if len(close) < 50:
        return {"signals": [], "action": "HOLD"}

    cur     = float(close.iloc[-1])
    prev    = float(close.iloc[-2])
    ma50    = float(close.rolling(50).mean().iloc[-1])
    vol_avg = float(vol.iloc[-50:-1].mean())
    vol_now = float(vol.iloc[-1])

    signals = []

    # 7-8% hard stop loss
    if entry_price is not None:
        loss_pct = (cur / entry_price - 1) * 100
        if loss_pct <= -7:
            signals.append({"sev": "HIGH", "tag": "STOP",
                            "msg": f"Down {loss_pct:.1f}% from entry — 7% stop hit. EXIT."})

    # MA50 break on heavy volume
    if cur < ma50 and prev >= ma50 and vol_now > vol_avg * 1.3:
        signals.append({"sev": "HIGH", "tag": "MA50 BREAK",
                        "msg": "Closed below 50-day MA on volume > 1.3× avg. EXIT."})

    # Distribution day on the stock itself
    day_chg = (cur / prev - 1) * 100
    if day_chg <= -2.0 and vol_now > vol_avg * 1.5:
        signals.append({"sev": "HIGH", "tag": "DISTRIBUTION",
                        "msg": f"Down {day_chg:.1f}% on volume {vol_now/vol_avg:.1f}× avg. Institutions selling."})

    # Climax run (parabolic — consider profit-taking)
    if len(close) >= 22:
        r3w = (cur / float(close.iloc[-15]) - 1) * 100
        if r3w >= 30:
            signals.append({"sev": "MEDIUM", "tag": "CLIMAX",
                            "msg": f"Up {r3w:.1f}% in 3 weeks — consider taking partial profits."})

    # Failed breakout: stock made recent 20-day high then closed below it within 3 days
    last20_high = float(close.tail(20).max())
    if cur < last20_high * 0.97 and float(close.tail(5).max()) >= last20_high * 0.998:
        signals.append({"sev": "MEDIUM", "tag": "FAILED BO",
                        "msg": "Made new 20-day high then reversed — breakout failed."})

    # Action
    high_count = sum(1 for s in signals if s["sev"] == "HIGH")
    if high_count >= 1:
        action = "EXIT"
    elif any(s["sev"] == "MEDIUM" for s in signals):
        action = "TRIM / TIGHTEN SL"
    else:
        action = "HOLD"

    return {"signals": signals, "action": action,
            "price": round(cur, 2), "ma50": round(ma50, 2)}


# ──────────────────────────────────────────────────────────────────────────────
# 4. SETUP QUALITY SCORE — Composite 0-100 ranking
# ──────────────────────────────────────────────────────────────────────────────

def compute_setup_score(symbol: str, df: pd.DataFrame, regime: dict,
                        sector_quad: dict[str, str], fundamentals: dict | None = None) -> dict | None:
    """
    Composite Setup Quality Score combining technical, fundamental,
    market, sector, and risk:reward signals into a single 0-100 rank.

    Formula:
      Score = 0.25 × Technical
            + 0.25 × Fundamental
            + 0.20 × Market regime
            + 0.15 × Sector
            + 0.15 × Risk:Reward
    """
    if _is_etf(symbol):
        return None

    close = df["Close"].dropna()
    vol   = df["Volume"].dropna()
    if len(close) < 130:
        return None

    cur = float(close.iloc[-1])
    if cur < 30:
        return None

    # Liquidity — compute ADTV for reporting only (no gate: Nifty500 universe is already large-cap)
    adtv_cr = float((df["Close"] * df["Volume"]).iloc[-20:].mean()) / 1e7

    # ── TECHNICAL (0-100) ─────────────────────────────────────────────────────
    # Returns, RS, base quality, MA alignment, volume profile
    r3m  = (cur / float(close.iloc[-66])  - 1) * 100 if len(close) > 66  else 0
    r6m  = (cur / float(close.iloc[-130]) - 1) * 100 if len(close) > 130 else 0
    ma50 = float(close.rolling(50).mean().iloc[-1])  if len(close) >= 50  else cur
    ma200= float(close.rolling(200).mean().iloc[-1]) if len(close) >= 200 else cur

    base = validate_base(close)
    base_score = base["score"] if base["valid"] else base["score"] * 0.5

    vol_r10 = float(vol.iloc[-10:].mean())
    vol_a50 = float(vol.iloc[-50:].mean()) if len(vol) >= 50 else float(vol.mean())
    vol_ratio = vol_r10 / vol_a50 if vol_a50 > 0 else 1.0

    ma_score    = 100 if (cur > ma50 > ma200) else (60 if cur > ma50 else 30)
    ret_score   = min(100, max(0, (r3m + 20) * 2))   # -20%→0, 30%→100
    vol_score   = min(100, vol_ratio * 50)
    technical   = round(0.30 * base_score + 0.25 * ma_score +
                        0.25 * ret_score + 0.20 * vol_score, 1)

    # ── FUNDAMENTAL (0-100) ───────────────────────────────────────────────────
    if fundamentals:
        eps_g  = fundamentals.get("eps_growth_yoy", 0) or 0
        sales_g= fundamentals.get("sales_growth_yoy", 0) or 0
        roe    = fundamentals.get("roe", 0) or 0
        d_e    = fundamentals.get("debt_to_equity", 999) or 999
        eps_s  = min(100, max(0, eps_g * 2))      # 25% → 50, 50% → 100
        sale_s = min(100, max(0, sales_g * 3))   # 20% → 60, 33% → 100
        roe_s  = min(100, max(0, roe * 4))       # 15% → 60, 25% → 100
        de_s   = 100 if d_e < 0.5 else 70 if d_e < 1.0 else 40 if d_e < 2.0 else 10
        fundamental = round(0.35 * eps_s + 0.30 * sale_s + 0.20 * roe_s + 0.15 * de_s, 1)
        has_fundamentals = True
    else:
        fundamental = 50.0   # neutral when no data
        has_fundamentals = False

    # ── MARKET (0-100) ────────────────────────────────────────────────────────
    market_map = {
        "Confirmed Uptrend": 95, "Uptrend Under Pressure": 60,
        "Correction": 20, "Unknown": 50,
    }
    market = market_map.get(regime.get("regime", "Unknown"), 50)

    # ── SECTOR (0-100) ────────────────────────────────────────────────────────
    grp = _SYM_TO_GROUP.get(symbol, "")
    quad = sector_quad.get(grp, "")
    sector_score_map = {
        "Leading": 95, "Improving": 75, "Weakening": 40, "Lagging": 15, "": 50,
    }
    sector = sector_score_map.get(quad, 50)

    # ── RISK:REWARD (0-100) ───────────────────────────────────────────────────
    # If near pivot (top of base), R:R is ideal. Far from pivot = poor R:R.
    if base["valid"] and base.get("base_len", 0) > 30:
        base_slc = close.iloc[-base["base_len"]:]
        base_hi  = float(base_slc.max())
        base_lo  = float(base_slc.min())
        pos_in_base = (cur - base_lo) / (base_hi - base_lo) if base_hi > base_lo else 0.5
        # Sweet spot: 0.7-0.95 (just below pivot or just broken out)
        if 0.7 <= pos_in_base <= 0.98:
            rr_score = 95
        elif 0.5 <= pos_in_base < 0.7:
            rr_score = 70
        elif pos_in_base > 0.98:
            rr_score = 50   # extended
        else:
            rr_score = 40   # too low in base
    else:
        rr_score = 50

    # ── COMPOSITE ─────────────────────────────────────────────────────────────
    # If no fundamentals, redistribute weight to technical + market
    if has_fundamentals:
        composite = (0.25 * technical + 0.25 * fundamental +
                     0.20 * market    + 0.15 * sector +
                     0.15 * rr_score)
    else:
        composite = (0.40 * technical + 0.25 * market +
                     0.20 * sector    + 0.15 * rr_score)

    composite = round(composite, 1)

    # ── Tier ──────────────────────────────────────────────────────────────────
    if composite >= 80:
        tier = "🏆 A+"
    elif composite >= 70:
        tier = "🥇 A"
    elif composite >= 60:
        tier = "🥈 B"
    elif composite >= 50:
        tier = "🥉 C"
    else:
        tier = ""

    stg = stage_analysis(close)

    return {
        "symbol":      symbol,
        "price":       round(cur, 2),
        "score":       composite,
        "tier":        tier,
        "technical":   technical,
        "fundamental": fundamental,
        "market":      market,
        "sector_score":sector,
        "rr_score":    round(rr_score, 1),
        "has_fundamentals": has_fundamentals,
        "r3m":         round(r3m, 2),
        "r6m":         round(r6m, 2),
        "adtv_cr":     round(adtv_cr, 1),
        "above_ma50":  bool(cur > ma50),
        "above_ma200": bool(cur > ma200),
        "vol_ratio":   round(vol_ratio, 2),
        "base_score":  base["score"],
        "base_valid":  base["valid"],
        "base_weeks":  base.get("base_weeks", 0),
        "base_depth":  base.get("depth", 0),
        "stage":       stg,
        "stage_lbl":   stage_label(stg),
        "group":       grp,
        "quadrant":    quad,
    }


# ──────────────────────────────────────────────────────────────────────────────
# 5. BACKTESTER — walk-forward validation
# ──────────────────────────────────────────────────────────────────────────────

def backtest_signal(stocks: dict, signal_fn, hold_days: int = 30,
                    stop_pct: float = -7.0, target_pct: float = 25.0,
                    lookback_days: int = 250, max_signals: int = 200) -> dict:
    """
    Walk-forward backtest of a signal function.

    signal_fn(df, idx) -> bool : True if a buy signal occurs on bar idx.
    Each signal generates a trade: enter next bar's open, exit at stop, target,
    or hold_days max.

    Returns aggregate statistics.
    """
    trades = []
    signal_count = 0
    for symbol, df in stocks.items():
        if _is_etf(symbol) or len(df) < lookback_days + hold_days + 10:
            continue
        if signal_count >= max_signals:
            break
        # Walk through history checking for signals
        start_idx = len(df) - lookback_days
        end_idx   = len(df) - hold_days - 1
        for i in range(start_idx, end_idx, 5):  # check every 5 days
            try:
                if signal_fn(df, i):
                    entry_idx = i + 1
                    if entry_idx >= len(df):
                        continue
                    entry = float(df["Open"].iloc[entry_idx])
                    if entry <= 0:
                        continue
                    sl_price = entry * (1 + stop_pct / 100)
                    tg_price = entry * (1 + target_pct / 100)

                    exit_price = None
                    exit_reason = "TIME"
                    exit_idx    = entry_idx + hold_days
                    for j in range(entry_idx, min(entry_idx + hold_days, len(df))):
                        lo = float(df["Low"].iloc[j])
                        hi = float(df["High"].iloc[j])
                        if lo <= sl_price:
                            exit_price = sl_price; exit_reason = "STOP"; exit_idx = j
                            break
                        if hi >= tg_price:
                            exit_price = tg_price; exit_reason = "TARGET"; exit_idx = j
                            break
                    if exit_price is None:
                        exit_price = float(df["Close"].iloc[min(entry_idx + hold_days, len(df) - 1)])

                    ret = (exit_price / entry - 1) * 100
                    trades.append({
                        "symbol":  symbol,
                        "entry":   round(entry, 2),
                        "exit":    round(exit_price, 2),
                        "ret_pct": round(ret, 2),
                        "reason":  exit_reason,
                        "days":    exit_idx - entry_idx,
                    })
                    signal_count += 1
                    if signal_count >= max_signals:
                        break
            except Exception:
                continue

    if not trades:
        return {"trades": 0, "win_rate": 0, "avg_win": 0, "avg_loss": 0,
                "expectancy": 0, "profit_factor": 0, "avg_hold": 0,
                "best": 0, "worst": 0, "by_reason": {}}

    wins   = [t for t in trades if t["ret_pct"] > 0]
    losses = [t for t in trades if t["ret_pct"] <= 0]
    win_rate = round(len(wins) / len(trades) * 100, 1)
    avg_win  = round(sum(t["ret_pct"] for t in wins) / len(wins), 2) if wins else 0
    avg_loss = round(sum(t["ret_pct"] for t in losses) / len(losses), 2) if losses else 0
    expectancy = round((win_rate / 100 * avg_win) + ((1 - win_rate / 100) * avg_loss), 2)
    profit_factor = round(abs(sum(t["ret_pct"] for t in wins) /
                              sum(t["ret_pct"] for t in losses)), 2) if losses else 999
    avg_hold = round(sum(t["days"] for t in trades) / len(trades), 1)
    by_reason = {}
    for t in trades:
        by_reason[t["reason"]] = by_reason.get(t["reason"], 0) + 1

    return {
        "trades":         len(trades),
        "win_rate":       win_rate,
        "avg_win":        avg_win,
        "avg_loss":       avg_loss,
        "expectancy":     expectancy,
        "profit_factor":  profit_factor,
        "avg_hold":       avg_hold,
        "best":           max(t["ret_pct"] for t in trades),
        "worst":          min(t["ret_pct"] for t in trades),
        "by_reason":      by_reason,
    }


# ── Built-in signal functions (for backtesting) ──────────────────────────────

def _signal_breakout_20d(df: pd.DataFrame, i: int) -> bool:
    """Bar i closes above prior 20-day high on volume > 1.5× avg."""
    if i < 50:
        return False
    close_i = float(df["Close"].iloc[i])
    high20  = float(df["High"].iloc[i-20:i].max())
    vol_i   = float(df["Volume"].iloc[i])
    vol_avg = float(df["Volume"].iloc[i-50:i].mean())
    return close_i > high20 and vol_i > vol_avg * 1.5

def _signal_rs_breakout(df: pd.DataFrame, i: int) -> bool:
    """3-month return > 15% AND closes at/near recent high."""
    if i < 80:
        return False
    close_i = float(df["Close"].iloc[i])
    r3m = (close_i / float(df["Close"].iloc[i-66]) - 1) * 100
    hi = float(df["Close"].iloc[max(0, i-200):i].max())
    return r3m > 15 and close_i >= hi * 0.97

def _signal_volume_accumulation(df: pd.DataFrame, i: int) -> bool:
    """Volume > 2× avg, closed in upper 60% of day range."""
    if i < 30:
        return False
    high_i  = float(df["High"].iloc[i])
    low_i   = float(df["Low"].iloc[i])
    close_i = float(df["Close"].iloc[i])
    vol_i   = float(df["Volume"].iloc[i])
    vol_avg = float(df["Volume"].iloc[i-20:i].mean())
    if high_i <= low_i or vol_avg <= 0:
        return False
    pos = (close_i - low_i) / (high_i - low_i)
    return vol_i > vol_avg * 2 and pos >= 0.6


# ──────────────────────────────────────────────────────────────────────────────
# MAIN ENTRY — full edge engine run
# ──────────────────────────────────────────────────────────────────────────────

def run_edge_engine(progress_callback=None) -> dict:
    """
    Run the full edge engine: regime detection, master ranking, backtests.
    Cached 30 min. Zero NSE API calls.
    """
    if _cache["data"] and time.time() - _cache["ts"] < CACHE_TTL:
        return _cache["data"]

    # Load fundamentals (best-effort, optional)
    fundamentals_map = {}
    try:
        from fundamentals import load_all_fundamentals
        fundamentals_map = load_all_fundamentals()
    except Exception:
        pass

    if progress_callback:
        progress_callback(0, 100, "Loading bhavcopy cache…")
    stocks = _load_stocks(progress_callback, days=400)
    if not stocks:
        return {"error": "No bhavcopy data available"}

    # Build Nifty proxy
    if progress_callback:
        progress_callback(20, 100, "Building Nifty proxy index…")
    nifty = _build_nifty_proxy(stocks)

    # Market regime
    if progress_callback:
        progress_callback(30, 100, "Detecting market regime (D-Day / FTD)…")
    regime = detect_market_regime(nifty)

    # Sector quadrants (RRG)
    if progress_callback:
        progress_callback(40, 100, "Computing sector RRG quadrants…")
    sector_quad = _compute_sector_quadrants(stocks, nifty)

    # Setup Quality Score for every stock
    if progress_callback:
        progress_callback(50, 100, f"Scoring {len(stocks)} stocks…")
    ranked = []
    done = 0
    total = len(stocks)
    with ThreadPoolExecutor(max_workers=8) as ex:
        futs = {
            ex.submit(compute_setup_score, sym, df, regime, sector_quad,
                      fundamentals_map.get(sym)): sym
            for sym, df in stocks.items()
        }
        for fut in as_completed(futs):
            done += 1
            if progress_callback and done % 500 == 0:
                progress_callback(50 + int(done / total * 30), 100,
                                  f"Scoring… {done}/{total}")
            r = fut.result()
            if r is not None:
                ranked.append(r)

    ranked.sort(key=lambda x: -x["score"])

    # Backtest each signal type
    if progress_callback:
        progress_callback(85, 100, "Running backtests on historical signals…")
    backtests = {}
    for name, fn in (("Breakout (20d high + vol)", _signal_breakout_20d),
                     ("RS Breakout (52w + r3m)",   _signal_rs_breakout),
                     ("Volume Accumulation",       _signal_volume_accumulation)):
        try:
            backtests[name] = backtest_signal(stocks, fn, hold_days=20,
                                              stop_pct=-7, target_pct=20,
                                              lookback_days=200, max_signals=300)
        except Exception as e:
            backtests[name] = {"error": str(e), "trades": 0}

    # Tier counts
    tiers = {"A+": 0, "A": 0, "B": 0, "C": 0}
    for r in ranked:
        if r["score"] >= 80:   tiers["A+"] += 1
        elif r["score"] >= 70: tiers["A"]  += 1
        elif r["score"] >= 60: tiers["B"]  += 1
        elif r["score"] >= 50: tiers["C"]  += 1

    out = {
        "regime":         regime,
        "ranked":         ranked[:200],   # top 200 only
        "tier_counts":    tiers,
        "total_scored":   len(ranked),
        "backtests":      backtests,
        "sector_quad":    sector_quad,
        "computed_at":    int(time.time()),
        "fundamentals_available": bool(fundamentals_map),
        "fundamentals_count": len(fundamentals_map),
    }
    _cache["data"] = out
    _cache["ts"]   = time.time()

    if progress_callback:
        progress_callback(100, 100,
                          f"Done — {tiers['A+']} A+ · {tiers['A']} A · {tiers['B']} B candidates")
    return out


def _compute_sector_quadrants(stocks: dict, nifty_df: pd.DataFrame | None) -> dict[str, str]:
    """RRG quadrant per industry group (reused from industry_groups logic)."""
    if nifty_df is None or "Close" not in nifty_df:
        return {}
    nifty_px = nifty_df["Close"]
    if not isinstance(nifty_px.index, pd.DatetimeIndex):
        nifty_px.index = pd.to_datetime(nifty_px.index)

    out = {}
    for grp, syms in INDUSTRY_GROUPS.items():
        closes = []
        for s in syms:
            if s in stocks and len(stocks[s]) >= 80:
                c = stocks[s]["Close"].dropna()
                if not isinstance(c.index, pd.DatetimeIndex):
                    c.index = pd.to_datetime(c.index)
                closes.append(c)
        if len(closes) < 2:
            continue
        try:
            grp_idx = pd.concat(closes, axis=1).dropna(how="all").mean(axis=1)
            common  = grp_idx.index.intersection(nifty_px.index)
            if len(common) < 80:
                continue
            rs_d = grp_idx[common] / nifty_px[common]
            rs_w = rs_d.resample("W").last().dropna()
            if len(rs_w) < 12:
                continue
            rm     = rs_w.rolling(26, min_periods=10).mean()
            rratio = (rs_w / rm * 100).dropna()
            if len(rratio) < 6:
                continue
            rmom = (rratio / rratio.shift(4) * 100).dropna()
            if len(rmom) < 1:
                continue
            x = float(rratio.iloc[-1]); y = float(rmom.iloc[-1])
            out[grp] = (
                "Leading"   if x >= 100 and y >= 100 else
                "Weakening" if x >= 100 and y <  100 else
                "Improving" if x <  100 and y >= 100 else
                "Lagging"
            )
        except Exception:
            continue
    return out
