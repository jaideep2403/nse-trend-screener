"""
Market-Breadth composite — walk-forward validation & calibration.

Computes the FULL 7-component breadth timing score (0-15) point-in-time for
every snapshot across ~2 years, with zero look-ahead, and measures forward
5/10/20-day NIFTYBEES returns per score bucket. Writes the results to
`.breadth_calibration.json`, which the live Market Breadth tab displays in
place of its old 60-day / 40-sample / partial-scale mini-backtest.

Unlike the old in-process backtest, ALL seven components are computed
historically (the old one skipped volume confirmation and sector breadth,
scoring 0-11 against a live 0-15 — apples vs oranges).

Honest limitations (recorded in the JSON):
  - The CURRENT curated 750-stock universe is applied to past dates
    (survivorship: breadth % levels are slightly flattered in old periods).
  - One ~2-year window. Re-run periodically; thresholds are evidence, not law.

Run:  python3 breadth_validation.py [--days 1100] [--step 3]
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

import regime as regime_mod
from analysis_utils import stage_analysis

WARMUP = 260   # bars needed before the first snapshot (252 for 52W + slack)


def _score_components(p50, p200, hl_ratio, dist_days, n_stage,
                      vol_ratio_5d, sector_pct) -> int:
    """EXACT mirror of market_breadth._market_timing_signal bucketing."""
    s = 0
    if   p50 >= 75: s += 3
    elif p50 >= 60: s += 2
    elif p50 >= 40: s += 1
    if   p200 >= 55: s += 2
    elif p200 >= 35: s += 1
    if   hl_ratio >= 0.7: s += 2
    elif hl_ratio >= 0.5: s += 1
    if   dist_days <= 3: s += 2
    elif dist_days <= 5: s += 1
    if   n_stage == 2: s += 2
    elif n_stage == 1: s += 1
    vr = vol_ratio_5d if vol_ratio_5d is not None else 1.0
    if   vr >= 1.5: s += 2
    elif vr >= 1.0: s += 1
    if sector_pct is not None:
        if   sector_pct >= 60: s += 2
        elif sector_pct >= 30: s += 1
    return s


def run_validation(days: int = 1100, step: int = 3) -> dict:
    import edge_engine as ee
    from industry_groups import INDUSTRY_GROUPS

    t0 = time.time()
    stocks = ee._load_stocks(days=days)        # split-adjusted, curated 750
    bench = ee._BENCH
    if not stocks or bench is None or len(bench) < WARMUP + 30:
        raise RuntimeError("insufficient data")
    print(f"loaded {len(stocks)} symbols, bench {len(bench)} bars "
          f"({time.time()-t0:.0f}s)", flush=True)

    # ── Wide matrices (dates × symbols) — vectorised breadth components ──
    close_w = pd.DataFrame({s: df["Close"] for s, df in stocks.items()}).sort_index()
    vol_w   = pd.DataFrame({s: df["Volume"] for s, df in stocks.items()}).reindex(close_w.index)

    ma50_w  = close_w.rolling(50).mean()
    ma200_w = close_w.rolling(200).mean()
    hi52_w  = close_w.rolling(252, min_periods=60).max()
    lo52_w  = close_w.rolling(252, min_periods=60).min()
    chg_w   = close_w.diff()

    valid      = close_w.notna()
    p50_series  = ((close_w > ma50_w) & ma50_w.notna()).sum(axis=1) / valid.sum(axis=1) * 100
    p200_series = ((close_w > ma200_w) & ma200_w.notna()).sum(axis=1) / valid.sum(axis=1) * 100
    nh_series   = (close_w >= hi52_w * 0.995).sum(axis=1)
    nl_series   = (close_w <= lo52_w * 1.005).sum(axis=1)
    upvol_series = vol_w.where(chg_w > 0, 0.0).sum(axis=1)
    dnvol_series = vol_w.where(chg_w < 0, 0.0).sum(axis=1)
    vr_daily = (upvol_series / dnvol_series.replace(0, np.nan)).clip(upper=10.0)
    vr_5d    = vr_daily.rolling(5, min_periods=3).mean()

    # Canonical D-day count per day (rolling, volume-confirmed)
    mkt_vol = regime_mod.build_market_volume(stocks)
    bpx = bench.reindex(close_w.index, method="ffill")
    bvol = mkt_vol.reindex(close_w.index, method="ffill") if mkt_vol is not None else None
    bpct = bpx.pct_change() * 100
    if bvol is not None:
        dday_flag = (bpct <= regime_mod.DDAY_PCT_THRESHOLD) & (bvol > bvol.shift(1))
    else:
        dday_flag = bpct <= -0.5
    dday_25 = dday_flag.rolling(regime_mod.DDAY_SESSIONS).sum()

    # ── Snapshots ──
    dates = close_w.index
    snap_positions = list(range(WARMUP, len(dates) - 21, step))
    print(f"{len(snap_positions)} snapshots: {dates[snap_positions[0]].date()} → "
          f"{dates[snap_positions[-1]].date()}", flush=True)

    samples = []
    for k, pos in enumerate(snap_positions):
        dt = dates[pos]
        # Nifty stage (point-in-time, real bench)
        bp = bpx.iloc[:pos + 1].dropna()
        n_stage = stage_analysis(bp) if len(bp) >= 175 else 0
        # Sector Stage-2 % (point-in-time, real stage per member median)
        sector_pct = None
        try:
            grp_total = grp_s2 = 0
            for grp, members in INDUSTRY_GROUPS.items():
                stages = []
                for s in members:
                    col = close_w.get(s)
                    if col is None:
                        continue
                    c = col.iloc[:pos + 1].dropna()
                    if len(c) >= 175:
                        stages.append(stage_analysis(c))
                if len(stages) >= 3:
                    grp_total += 1
                    if int(np.median(stages)) == 2:
                        grp_s2 += 1
            if grp_total:
                sector_pct = grp_s2 / grp_total * 100
        except Exception:
            sector_pct = None

        nh, nl = float(nh_series.iloc[pos]), float(nl_series.iloc[pos])
        hl_ratio = nh / (nh + nl + 1)
        dd = dday_25.iloc[pos]
        score = _score_components(
            float(p50_series.iloc[pos]), float(p200_series.iloc[pos]),
            hl_ratio, int(dd) if not pd.isna(dd) else 0, n_stage,
            float(vr_5d.iloc[pos]) if not pd.isna(vr_5d.iloc[pos]) else None,
            sector_pct,
        )
        # Forward NIFTYBEES returns
        base = bpx.iloc[pos]
        if pd.isna(base) or base <= 0:
            continue
        fwd = {}
        for h in (5, 10, 20):
            v = bpx.iloc[pos + h] if pos + h < len(bpx) else np.nan
            fwd[h] = (float(v) / float(base) - 1) * 100 if not pd.isna(v) else None
        if fwd[20] is None:
            continue
        # Canonical regime label at this snapshot (D-day count; FTD not
        # reconstructed historically — affects only the ≤3-dday split, both
        # branches of which are deploy-permitting)
        dd_int = int(dd) if not pd.isna(dd) else 0
        regime_label = regime_mod.classify_regime(dd_int, ftd_active=False)
        # Forward path risk: worst drawdown over the next 20 bars
        path = bpx.iloc[pos:pos + 21].dropna()
        fwd_dd = 0.0
        if len(path) > 1:
            rel = path / float(path.iloc[0]) - 1
            fwd_dd = round(float(rel.min()) * 100, 2)
        samples.append({"date": str(dt.date()), "score": score,
                        "regime": regime_label, "ddays": dd_int,
                        "fwd_dd20": fwd_dd,
                        "r5": round(fwd[5], 2), "r10": round(fwd[10], 2),
                        "r20": round(fwd[20], 2)})
        if (k + 1) % 25 == 0:
            print(f"  snapshot {k+1}/{len(snap_positions)}", flush=True)

    if len(samples) < 60:
        raise RuntimeError(f"only {len(samples)} samples")

    # ── Bucket on the LIVE thresholds (0-15 scale, same labels) ──
    def _bucket(s):
        if s >= 12: return "Bull Market (12+)"
        if s >= 9:  return "Uptrend (9-11)"
        if s >= 5:  return "Sideways (5-8)"
        if s >= 2:  return "Correction (2-4)"
        return "Bear Market (0-1)"

    def _bucket_stats(grp: list, label: str) -> dict:
        return {
            "bucket":  label,
            "n":       len(grp),
            "avg_5d":  round(float(np.mean([g["r5"] for g in grp])), 2),
            "avg_10d": round(float(np.mean([g["r10"] for g in grp])), 2),
            "avg_20d": round(float(np.mean([g["r20"] for g in grp])), 2),
            "win_rate_10d": round(sum(1 for g in grp if g["r10"] > 0) / len(grp) * 100, 1),
            # Risk — what a regime GATE is actually for:
            "avg_fwd_dd20":   round(float(np.mean([g["fwd_dd20"] for g in grp])), 2),
            "worst_fwd_dd20": round(float(min(g["fwd_dd20"] for g in grp)), 2),
            "pct_dd_gt3":     round(sum(1 for g in grp if g["fwd_dd20"] <= -3) / len(grp) * 100, 1),
        }

    buckets: dict[str, list] = {}
    for s in samples:
        buckets.setdefault(_bucket(s["score"]), []).append(s)
    order = ["Bull Market (12+)", "Uptrend (9-11)", "Sideways (5-8)",
             "Correction (2-4)", "Bear Market (0-1)"]
    by_bucket = [_bucket_stats(buckets[l], l) for l in order if buckets.get(l)]

    # ── The actual strategy gate: canonical D-day regime label ──
    rgrp: dict[str, list] = {}
    for s in samples:
        rgrp.setdefault(s["regime"], []).append(s)
    regime_order = ["Confirmed Uptrend", "Uptrend Under Pressure", "Correction"]
    by_regime = [_bucket_stats(rgrp[l], l) for l in regime_order if rgrp.get(l)]

    # Score ↔ forward-return rank correlation (does the composite work at all?)
    sc = pd.Series([s["score"] for s in samples], dtype=float)
    ic20 = float(sc.rank().corr(pd.Series([s["r20"] for s in samples]).rank()))
    ic10 = float(sc.rank().corr(pd.Series([s["r10"] for s in samples]).rank()))

    out = {
        "by_bucket":      by_bucket,
        "by_regime":      by_regime,
        "total_samples":  len(samples),
        "window":         f"{samples[0]['date']} → {samples[-1]['date']}",
        "fwd_basis":      "NIFTYBEES close-to-close",
        "score_ic_10d":   round(ic10, 3),
        "score_ic_20d":   round(ic20, 3),
        "scale":          "0-15, all 7 components computed point-in-time",
        "limitations":    "current curated universe applied historically "
                          "(mild survivorship); one ~2yr window",
        "computed_at":    int(time.time()),
    }
    return out


if __name__ == "__main__":
    days = int(sys.argv[sys.argv.index("--days") + 1]) if "--days" in sys.argv else 1100
    step = int(sys.argv[sys.argv.index("--step") + 1]) if "--step" in sys.argv else 3
    res = run_validation(days=days, step=step)
    path = Path(__file__).parent / ".breadth_calibration.json"
    path.write_text(json.dumps(res, indent=2))
    print(f"\nscore IC vs fwd10/20d: {res['score_ic_10d']} / {res['score_ic_20d']}")

    def _tbl(rows, title):
        print(f"\n== {title} ==")
        print(f"{'bucket':<24}{'n':>5}{'avg20d':>8}{'win10d':>8}"
              f"{'avgDD20':>9}{'worstDD':>9}{'DD>3%':>7}")
        for b in rows:
            print(f"{b['bucket']:<24}{b['n']:>5}{b['avg_20d']:>8}"
                  f"{b['win_rate_10d']:>7}%{b['avg_fwd_dd20']:>9}"
                  f"{b['worst_fwd_dd20']:>9}{b['pct_dd_gt3']:>6}%")

    _tbl(res["by_bucket"], "COMPOSITE SCORE BUCKETS")
    _tbl(res["by_regime"], "CANONICAL D-DAY REGIME (the strategy gate)")
    print(f"\nwritten → {path}")
