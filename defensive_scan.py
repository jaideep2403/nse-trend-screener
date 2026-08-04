"""Defensive Leaders — a lower-drawdown, delivery-aware alternative to momentum.

Built on published, price/volume/delivery-computable edges (validated before ship
via the system yardstick, never assumed):
  • Absolute-momentum GATE (Antonacci dual momentum) — only hold live uptrends;
    the structural circuit breaker the plain momentum scan lacked.
  • Low realized VOLATILITY (low-vol anomaly) — these fall less in downturns.
  • Path SMOOTHNESS / frog-in-the-pan (Da-Gurun-Warachka) — trends built from many
    small same-direction days persist ~3× longer and reverse less than jumpy ones.
  • DELIVERY-% accumulation (NSE-specific edge) — rising delivery in a controlled
    range = quiet institutional footprint, ahead of the move.

Every feature at bar i uses ONLY bars ≤ i (rolling, no negative shift). Selection is
CROSS-SECTIONAL: gate-passers are rank-blended, so the composite is a relative
ranking at each snapshot, not an absolute score.
"""

from __future__ import annotations
import numpy as np
import pandas as pd

VOL_WINDOW   = 90     # realized-vol / smoothness formation window (~4.5 months)
DELIV_FAST   = 5      # recent delivery window
DELIV_SLOW   = 25     # baseline delivery window


def precompute(df: pd.DataFrame) -> dict | None:
    if len(df) < 260 or "Close" not in df:
        return None
    c = df["Close"].astype(float)
    o = (df["Open"].astype(float) if "Open" in df else c)
    v = (df["Volume"].astype(float) if "Volume" in df else pd.Series(0.0, index=c.index))
    dp = (df["DelivPer"].astype(float) if "DelivPer" in df else pd.Series(np.nan, index=c.index))
    n = len(c)

    logret = np.log(c / c.shift(1))
    ma200 = c.rolling(200).mean()
    r6m = c / c.shift(126) - 1.0
    r3m = c / c.shift(63) - 1.0

    # 90-day annualised realised volatility (lower = calmer).
    vol90 = logret.rolling(VOL_WINDOW).std() * np.sqrt(252)

    # Kaufman efficiency ratio over 90d = |net move| / total path length. 1 = a
    # perfectly straight trend (smooth), →0 = choppy. The price-only frog-in-the-pan.
    net_move = (c - c.shift(VOL_WINDOW)).abs()
    path_len = c.diff().abs().rolling(VOL_WINDOW).sum()
    efficiency = (net_move / path_len).replace([np.inf, -np.inf], np.nan)

    # Fraction of up-days over 90d (continuous information ↑).
    up_frac = (logret > 0).rolling(VOL_WINDOW).mean()

    # Delivery accumulation: recent delivery strength vs its own baseline, and level.
    deliv_fast = dp.rolling(DELIV_FAST).mean()
    deliv_slow = dp.rolling(DELIV_SLOW).mean()
    deliv_ratio = (deliv_fast / deliv_slow).replace([np.inf, -np.inf], np.nan)
    deliv_level = deliv_slow

    adtv = (c * v).rolling(20).mean() / 1e7   # ₹Cr

    return {
        "idx": df.index, "n": n,
        "close": c.to_numpy(dtype=float),
        "open":  o.to_numpy(dtype=float),
        "ma200": ma200.to_numpy(dtype=float),
        "r6m":   r6m.to_numpy(dtype=float),
        "r3m":   r3m.to_numpy(dtype=float),
        "vol90": vol90.to_numpy(dtype=float),
        "eff":   efficiency.to_numpy(dtype=float),
        "upfrac": up_frac.to_numpy(dtype=float),
        "deliv_ratio": deliv_ratio.to_numpy(dtype=float),
        "deliv_level": deliv_level.to_numpy(dtype=float),
        "adtv":  adtv.to_numpy(dtype=float),
    }


def passes_gate(f: dict, p: int) -> bool:
    """Absolute-momentum circuit breaker: only consider a stock in a live uptrend."""
    if p < 200 or p >= f["n"]:
        return False
    price, ma200, r6m = f["close"][p], f["ma200"][p], f["r6m"][p]
    if not (np.isfinite(price) and np.isfinite(ma200) and np.isfinite(r6m)):
        return False
    return price > ma200 and r6m > 0.0


def raw_factors(f: dict, p: int) -> dict | None:
    """The three defensive factors at bar p (point-in-time), or None if unusable."""
    vol, eff, dr, dl, uf = (f["vol90"][p], f["eff"][p], f["deliv_ratio"][p],
                            f["deliv_level"][p], f["upfrac"][p])
    if not (np.isfinite(vol) and vol > 0 and np.isfinite(eff)):
        return None
    return {"vol90": vol, "eff": eff, "deliv_ratio": dr, "deliv_level": dl,
            "upfrac": uf, "mom": f["r3m"][p]}


def rank_and_score(rows: list[dict], mom_weight: float = 0.0,
                   quality_weight: float = 0.0, flow_weight: float = 0.0,
                   promoter_weight: float = 0.0,
                   vol_weight: float | None = None) -> list[dict]:
    """CROSS-SECTIONAL composite. `rows` = [{sym,p,vol90,eff,deliv_ratio,deliv_level,upfrac}].
    Composite = mean of factor percentile ranks:
      • low-vol      (invert: calmer → higher)
      • smoothness   (efficiency ratio, higher → higher)
      • delivery     (ratio + level, higher → higher; NaN → neutral 0.5)
    Optional tilts (add to the blend when weight > 0 and the column is present):
      • mom          (`mom`)      — 3-month momentum, the RETURN engine
      • quality      (`qual_raw`) — QMJ-style fundamental quality (drawdown lever)
    Returns the same rows with `score` (0-1) and `inv_vol_weight` added."""
    if not rows:
        return []
    df = pd.DataFrame(rows)
    n = len(df)
    def pr(col, invert=False):
        s = df[col]
        r = s.rank(pct=True)
        if invert:
            r = 1.0 - r
        return r.fillna(0.5)
    low_vol   = pr("vol90", invert=True)
    smooth    = pr("eff")
    deliv     = 0.6 * pr("deliv_ratio") + 0.4 * pr("deliv_level")
    # LOW-VOL IS A TUNABLE WEIGHT (2026-07-31), not a fixed 1/3 of the base.
    # Decile test over 72 monthly cross-sections found low-vol INVERTED on this
    # universe: highest-vol decile returned +2.13%/mo vs lowest-vol +1.34%/mo,
    # monotonicity -0.75. Baked in at full weight it was cancelling a genuine
    # momentum signal (r6m monotonicity +0.99, spread +1.40%/mo) — the composite
    # beat only 44% of random portfolios while pure momentum beat 25/25.
    # Default comes from WIN_VOL_WEIGHT so callers inherit the validated value.
    _vw = WIN_VOL_WEIGHT if vol_weight is None else float(vol_weight)
    # Momentum blend: rank by momentum (for RETURN) while the defensive factors
    # (for LOW DRAWDOWN) shape the selection. mom_weight=0 → pure defensive.
    base  = _vw * low_vol + smooth + deliv
    denom = 2.0 + _vw
    if mom_weight > 0 and "mom" in df.columns:
        base = base + mom_weight * pr("mom")
        denom += mom_weight
    if quality_weight > 0 and "qual_raw" in df.columns:
        qual = pr("qual_raw")
        base = base + quality_weight * qual
        denom += quality_weight
        df["quality_rank"] = (qual * 100).round(0)
    # Institutional flow (F&O open-interest positioning). Point-in-time and
    # look-ahead-free — see institutional_flow.py. Names with no listed futures
    # carry NEUTRAL, so the tilt only differentiates the F&O-eligible subset.
    if flow_weight > 0 and "flow_raw" in df.columns:
        flow = pr("flow_raw")
        base = base + flow_weight * flow
        denom += flow_weight
        df["flow_rank"] = (flow * 100).round(0)
    # Promoter/insider accumulation (SAST Reg29 + SEBI PIT). Keyed on DISCLOSURE
    # date, so point-in-time and look-ahead-free — see promoter_flow.py.
    if promoter_weight > 0 and "prom_raw" in df.columns:
        prom = pr("prom_raw")
        base = base + promoter_weight * prom
        denom += promoter_weight
        df["promoter_rank"] = (prom * 100).round(0)
    df["score"] = base / denom
    df["low_vol_rank"] = (low_vol * 100).round(0)
    df["smooth_rank"]  = (smooth * 100).round(0)
    df["deliv_rank"]   = (deliv * 100).round(0)
    # Inverse-vol weights (volatility targeting) normalised across the picked set.
    inv = 1.0 / df["vol90"].clip(lower=1e-6)
    df["inv_vol_weight"] = inv / inv.sum()
    return df.sort_values("score", ascending=False).to_dict("records")


# Validated winning config (system yardstick, survivorship-free, OOS-checked):
# rank by momentum with the defensive factors as tilt, after dropping the highest-
# volatility tail. Beat the momentum scan on CAGR, drawdown AND Sharpe.
# Weight on the low-vol factor inside the composite. VALIDATED 2026-07-31.
#
# Set to 0.0 because low-vol is INVERTED on this universe: the decile test over 72
# monthly cross-sections found the highest-vol decile returning +2.13%/mo against the
# lowest-vol decile's +1.34%/mo (monotonicity -0.75). At full weight it was cancelling
# a genuine momentum signal (r6m monotonicity +0.99, spread +1.40%/mo).
#
# Sweep, 6.0y, top_k=30, judged against a 20-seed random control:
#     filter 0.70  vol_wt 1.0 (old) -> 16.94% / -25.35% / Sharpe 0.68   beat  35% of random
#     filter 0.70  vol_wt 0.0 (new) -> 26.68% / -19.92% / Sharpe 1.05   beat 100% of random
# Removing the weight ADDS 9.74pp CAGR, IMPROVES drawdown by 5.43pp and lifts Sharpe
# from 0.68 to 1.05.
#
# NOTE the two vol levers are different things and must not be confused:
#   WIN_VOL_FILTER (0.70) pre-drops the wildest 30% BEFORE ranking — this HELPS
#     (it is a crash guard); removing it pushed drawdown to -28.10%.
#   WIN_VOL_WEIGHT ranks calm stocks higher WITHIN the survivors — this HURT.
WIN_VOL_WEIGHT = 0.0
WIN_MOM_WEIGHT = 4.0
WIN_VOL_FILTER = 0.70   # keep the calmest 70% of gate-passers before ranking
# Quality tilt weight. Set to a MODERATE 2.0 (quality ≈ 22% of the composite; the
# survivorship-free price factors stay the dominant 78%). Justified primarily by
# the QMJ research (Asness-Frazzini-Pedersen: IR>1 across 24 countries, positive
# crisis convexity) — NOT by the quality backtest, whose gain is inflated by the
# look-ahead/survivorship bias in quality.py. In LIVE use there is no look-ahead:
# picking today's names by today's quality is QMJ done correctly.
WIN_QUAL_WEIGHT = 2.0
# Institutional-flow (F&O open-interest) tilt. VALIDATED point-in-time — unlike
# quality, this carries NO look-ahead, and the pre-2024 half of the walk-forward is
# bit-identical to the no-flow baseline (proving the factor is genuinely inert
# where it has no data, i.e. no leak). An 8-point sweep gave a smooth inverted-U
# (+0.00/+0.25/+0.35/+1.67/+1.59/−0.59/−2.76/−5.26 ΔCAGR), so 0.75 sits on a
# two-config plateau rather than an isolated spike. Kept deliberately gentle:
# only ~35% of picks have listed futures, so a heavy weight crowds out the
# non-F&O mid/small-caps that drive the book (hence the decay above ~1.25).
WIN_FLOW_WEIGHT = 0.75
# Promoter/insider accumulation (SAST Reg29 + SEBI PIT): BUILT, BACKFILLED
# (155,485 disclosures, 2015-2026), VALIDATED — and REJECTED as a ranking tilt.
# Raw sweep hurt at 5 of 6 weights. A conviction filter (open-market purchases
# only, excluding gifts/ESOP/inter-se transfers/allotments) genuinely rescued it
# at pw=0.5 (ΔCAGR −1.07 → +0.60), proving the first pass was a coding flaw, not
# just noise — but even fixed it fails the bar:
#     pw=0.5  ΔFULL +0.60  Δ2nd −0.19  Sharpe 1.59 (flat)  maxDD −10.66 (worse)
#     pw=1.0  ΔFULL −1.19  Δ2nd −1.05  Sharpe 1.48         maxDD −12.07 (worse)
#     pw=2.0  ΔFULL +0.00  Δ2nd −0.94  Sharpe 1.55         maxDD −11.83 (worse)
# NO weight improves the out-of-sample half, Sharpe never beats baseline, and
# drawdown is worse at every setting. Plausible reason: promoter buying is a
# CONTRARIAN/value signal (they buy weakness), which fights a momentum-tilted
# defensive book. The data still powers the Promoter Activity TAB, where it is
# genuinely useful as a monitoring feed. Left at 0.0 — do not ship without new
# evidence.
WIN_PROMOTER_WEIGHT = 0.0


def run_defensive_scan(stocks: dict, top_n: int = 40,
                       quality_weight: float | None = None,
                       flow_weight: float | None = None) -> dict:
    """LIVE scan — the Defensive-Momentum Leaders (the validated blend). Gate on
    absolute momentum, drop the crash-prone high-vol tail, then rank by momentum
    tilted toward low-vol / smooth / high-delivery names, optionally tilted by
    QMJ-style fundamental quality (quality_weight)."""
    qw = WIN_QUAL_WEIGHT if quality_weight is None else quality_weight
    fw = WIN_FLOW_WEIGHT if flow_weight is None else flow_weight
    qmap = {}
    if qw > 0:
        try:
            import quality as _q
            qmap = _q.load_quality_map()
        except Exception:
            qmap = {}
    fmap, _iflow = {}, None
    if fw > 0:
        try:
            import institutional_flow as _if
            _iflow = _if
            fmap = _if.score_map_asof(pd.Timestamp.today())
        except Exception:
            fmap, _iflow = {}, None
    rows = []
    last_date = None
    for sym, df in stocks.items():
        f = precompute(df)
        if f is None:
            continue
        p = f["n"] - 1
        if last_date is None:
            last_date = str(f["idx"][-1].date())
        if f["adtv"][p] < 2.0 or not passes_gate(f, p):
            continue
        rf = raw_factors(f, p)
        if rf is None:
            continue
        row = {"symbol": sym, "p": p, "price": round(float(f["close"][p]), 2),
               "adtv_cr": round(float(f["adtv"][p]), 1),
               "r3m": round(float(f["r3m"][p]) * 100, 1) if np.isfinite(f["r3m"][p]) else None,
               "r6m": round(float(f["r6m"][p]) * 100, 1) if np.isfinite(f["r6m"][p]) else None,
               **rf}
        if qw > 0:
            row["qual_raw"] = qmap.get(sym, 0.5)   # neutral if no fundamentals
        if fw > 0:
            row["flow_raw"] = fmap.get(sym, 0.5)   # neutral if no listed futures
        rows.append(row)
    if not rows:
        return {"as_of": last_date, "count": 0, "stocks": []}
    # HARD vol filter: drop the highest-volatility tail (crash-prone), same as the
    # validated backtest, then momentum-tilted composite rank.
    if len(rows) > top_n:
        vt = np.quantile([r["vol90"] for r in rows], WIN_VOL_FILTER)
        rows = [r for r in rows if r["vol90"] <= vt]
    ranked = rank_and_score(rows, mom_weight=WIN_MOM_WEIGHT, quality_weight=qw,
                            flow_weight=fw)
    qcomp = None
    if qw > 0:
        try:
            import quality as _q
            qcomp = _q
        except Exception:
            qcomp = None
    for r in ranked:
        r["vol_pct"] = round(float(r["vol90"]) * 100, 1)      # annualised vol %
        r["smoothness"] = round(float(r["eff"]), 3)
        r["deliv_level"] = round(float(r["deliv_level"]), 1) if np.isfinite(r["deliv_level"]) else None
        r["deliv_ratio"] = round(float(r["deliv_ratio"]), 2) if np.isfinite(r["deliv_ratio"]) else None
        r["score"] = round(float(r["score"]), 3)
        r["weight_pct"] = round(float(r["inv_vol_weight"]) * 100, 1)
        if qcomp is not None:
            c = qcomp.components_of(r["symbol"]) or {}
            r["roe"] = c.get("roe")
            r["earn_growth"] = c.get("growth")
            r["earn_accel"] = c.get("accel")
            r["quality"] = c.get("quality")
        if _iflow is not None:
            fv = fmap.get(r["symbol"])
            r["flow"] = round(float(fv), 2) if fv is not None else None
            r["flow_label"] = _iflow.signal_label(fv)
    keep = ["symbol", "price", "adtv_cr", "r3m", "r6m", "vol_pct", "smoothness",
            "deliv_level", "deliv_ratio", "low_vol_rank", "smooth_rank", "deliv_rank",
            "quality_rank", "roe", "earn_growth", "earn_accel", "quality",
            "flow", "flow_label", "flow_rank", "score", "weight_pct"]
    out = [{k: r.get(k) for k in keep} for r in ranked[:top_n]]
    return {"as_of": last_date, "count": len(ranked), "stocks": out}


# ── LIVE VOLATILITY TARGETING ────────────────────────────────────────────────
# The backtest's single most valuable risk control was portfolio volatility
# targeting: scaling gross exposure so the book's REALISED volatility tracks a
# target. Measured on the validated config it turned a strategy that LOST to the
# index on drawdown (-16.88% vs NIFTY -16.11%) into one that BEATS it on both
# metrics (-12.02%, +4.85pp return), in-sample and out.
#
# Until now it existed ONLY as a backtest parameter — `defensive_scan` had no
# concept of it, so the live tab handed you a stock list with no sizing guidance
# and the validated risk control was never actually applied to real money.
# This computes the same scale live, from the realised volatility of the
# equal-weighted basket the scan just produced.
LIVE_VOL_TARGET = 0.10      # annualised; the validated setting
LIVE_VOL_LOOKBACK = 60      # trading days


def suggested_exposure(stocks: dict, picks: list[dict],
                       target: float = LIVE_VOL_TARGET,
                       lookback: int = LIVE_VOL_LOOKBACK) -> dict:
    """What fraction of the book the validated rule would deploy right now.

    Capped at 1.0 — the rule never levers up. Returns `None` for scale when there
    is not enough history to measure, so the UI can say "unknown" rather than
    imply a precise number it does not have.
    """
    syms = [p.get("symbol") for p in (picks or []) if p.get("symbol")]
    rets = []
    for sym in syms:
        df = stocks.get(sym)
        if df is None or len(df) < lookback + 2:
            continue
        c = df["Close"].astype(float).dropna()
        if len(c) < lookback + 2:
            continue
        rets.append(c.pct_change().tail(lookback))
    if len(rets) < 3:
        return {"scale_pct": None, "realised_vol_pct": None,
                "target_vol_pct": round(target * 100, 1), "n_used": len(rets),
                "note": "not enough history to measure realised volatility"}
    basket = pd.concat(rets, axis=1).mean(axis=1, skipna=True).dropna()
    rv = float(basket.std()) * (252 ** 0.5)
    if rv <= 1e-9:
        return {"scale_pct": 100.0, "realised_vol_pct": 0.0,
                "target_vol_pct": round(target * 100, 1), "n_used": len(rets)}
    scale = min(1.0, target / rv)
    return {
        "scale_pct": round(scale * 100, 1),
        "realised_vol_pct": round(rv * 100, 1),
        "target_vol_pct": round(target * 100, 1),
        "n_used": len(rets),
        "note": (f"Book volatility is {rv*100:.1f}% against a {target*100:.0f}% target, "
                 f"so the validated rule would deploy {scale*100:.0f}% and hold "
                 f"{100-scale*100:.0f}% in cash. This is the mechanical rule that made "
                 f"the backtest beat the index on drawdown — it is not advice."),
    }
