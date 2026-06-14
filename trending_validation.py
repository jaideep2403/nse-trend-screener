"""
Trending Score — walk-forward factor validation.

Measures, point-in-time (zero look-ahead), how well the current trending score
and each candidate factor predicts FORWARD 20-day benchmark-adjusted return
(alpha vs real NIFTYBEES). This is the evidence base for the v3 score weights:
nothing gets a weight that didn't earn it here.

Methodology
-----------
- Universe: curated Nifty Total Market 750 (same as the live tab), loaded
  split-adjusted via edge_engine._load_stocks (deep history).
- Snapshots: every `step` bars across the usable window, leaving a 260-bar
  factor warm-up at the start and fwd_days at the end.
- At each snapshot, every factor is computed ONLY from data <= snapshot date.
- Forward return: close(snapshot) → close(snapshot + fwd_days bars),
  minus the NIFTYBEES return over the same dates (alpha).
- IC = Spearman rank correlation(factor, forward alpha), computed PER SNAPSHOT,
  then aggregated: mean IC + t-stat across snapshots (robust to one lucky
  period). Pooled IC also reported.
- IS/OOS: snapshots split chronologically in half; a factor must hold up in
  the later half to be trusted.

Run:  python3 trending_validation.py [--days 1100] [--fwd 20] [--step 20]
"""
from __future__ import annotations

import sys
import time
import numpy as np
import pandas as pd


FWD_DAYS_DEFAULT = 20
WARMUP_BARS      = 260   # 252 for 12M factors + slack


def _spearman(xs, ys) -> float | None:
    if len(xs) != len(ys) or len(xs) < 20:
        return None
    a = pd.Series(xs, dtype=float).rank()
    b = pd.Series(ys, dtype=float).rank()
    c = a.corr(b)
    return float(c) if not pd.isna(c) else None


def _wilder_atr_pct(sub: pd.DataFrame) -> float | None:
    try:
        hi = sub["High"].dropna(); lo = sub["Low"].dropna(); cl = sub["Close"].dropna()
        if len(hi) < 15:
            return None
        tr = pd.concat([hi - lo, (hi - cl.shift(1)).abs(),
                        (lo - cl.shift(1)).abs()], axis=1).max(axis=1)
        atr14 = float(tr.ewm(alpha=1 / 14, adjust=False).mean().iloc[-1])
        cur = float(cl.iloc[-1])
        return atr14 / cur * 100 if cur > 0 else None
    except Exception:
        return None


def _factors_at(sub: pd.DataFrame, bench_pit: pd.Series) -> dict | None:
    """All candidate factors, point-in-time. sub = df sliced <= snapshot."""
    c = sub["Close"].dropna()
    if len(c) < WARMUP_BARS:
        return None
    cur = float(c.iloc[-1])
    if cur <= 0:
        return None

    def ret(k):
        return (cur / float(c.iloc[-k - 1]) - 1) * 100 if len(c) > k else None

    r1m, r3m, r6m, r12m = ret(21), ret(63), ret(126), ret(252)

    # Excess returns vs benchmark over identical windows
    b = bench_pit.reindex(c.index, method="ffill").dropna()
    rs = {}
    for name, k, rv in (("rs1m", 21, r1m), ("rs3m", 63, r3m),
                        ("rs6m", 126, r6m), ("rs12m", 252, r12m)):
        if rv is not None and len(b) > k and float(b.iloc[-k - 1]) > 0:
            bret = (float(b.iloc[-1]) / float(b.iloc[-k - 1]) - 1) * 100
            rs[name] = rv - bret
        else:
            rs[name] = None

    ma50 = float(c.iloc[-50:].mean())
    ma200 = float(c.iloc[-200:].mean()) if len(c) >= 200 else None
    ext_ma50 = (cur / ma50 - 1) * 100 if ma50 > 0 else None
    ma50_prev = float(c.iloc[-70:-20].mean()) if len(c) >= 70 else None
    ma50_slope = ((ma50 / ma50_prev - 1) * 100
                  if ma50_prev and ma50_prev > 0 else None)

    hi52 = float(c.iloc[-252:].max())
    pct_from_high = (cur / hi52 - 1) * 100 if hi52 > 0 else None

    v = sub["Volume"].dropna()
    vol_ratio = (float(v.iloc[-10:].mean()) / float(v.iloc[-50:].mean())
                 if len(v) >= 50 and float(v.iloc[-50:].mean()) > 0 else None)

    atr_pct = _wilder_atr_pct(sub)

    # ADX + R² from the live tab's own implementations (same code = same value)
    import trending as _tr
    adx = None
    try:
        h = sub["High"].dropna().values.astype(float)
        l = sub["Low"].dropna().values.astype(float)
        n = min(len(h), len(l), len(c))
        adx = _tr._adx(h[-n:], l[-n:], c.values.astype(float)[-n:])
    except Exception:
        pass
    r2 = _tr._r_squared(c)

    deliv = None
    if "DelivPer" in sub.columns:
        d = sub["DelivPer"].dropna()
        if len(d) >= 5:
            deliv = float(d.iloc[-20:].mean())

    return {
        "rs1m": rs["rs1m"], "rs3m": rs["rs3m"],
        "rs6m": rs["rs6m"], "rs12m": rs["rs12m"],
        "r3m": r3m, "r12m": r12m,
        "atr_pct": atr_pct,
        "ext_ma50": ext_ma50,
        "ma50_slope": ma50_slope,
        "pct_from_high": pct_from_high,
        "vol_ratio": vol_ratio,
        "adx": adx,
        "r_squared": r2,
        "deliv_avg20": deliv,
        "above_ma200": (1.0 if (ma200 and cur > ma200) else 0.0),
    }


def _score_v2_at(sub: pd.DataFrame, proxy_pit: pd.Series | None) -> float | None:
    """The CURRENT production score, computed point-in-time."""
    import trending as _tr
    try:
        m = _tr._score_stock(sub, proxy_pit)
        return float(m["score"]) if m else None
    except Exception:
        return None


def run_validation(days: int = 1100, fwd_days: int = FWD_DAYS_DEFAULT,
                   step: int = 20, include_v2: bool = True,
                   v3_weights: dict[str, float] | None = None) -> dict:
    """
    Returns {factors: [{name, ic_mean, ic_t, n_obs, ic_is, ic_oos}],
             snapshots, obs_total}.
    If v3_weights is given ({factor: weight}, rank-blended), the composite
    "score_v3" is also evaluated on the same observations.
    """
    import edge_engine as ee
    from analysis_utils import equal_weight_index, NIFTY_PROXY_SYMS

    t0 = time.time()
    stocks = ee._load_stocks(days=days)
    bench = ee._BENCH
    if not stocks or bench is None or len(bench) < WARMUP_BARS:
        raise RuntimeError("insufficient data: stocks or NIFTYBEES benchmark missing")
    print(f"loaded {len(stocks)} symbols, bench bars={len(bench)} "
          f"({time.time()-t0:.1f}s)", flush=True)

    # Equal-weight proxy series — only needed to reproduce the v2 score exactly
    closes = [stocks[s]["Close"].dropna() for s in NIFTY_PROXY_SYMS
              if s in stocks and len(stocks[s]) >= 100]
    proxy = (equal_weight_index(pd.concat(closes, axis=1).dropna(how="all"))
             if len(closes) >= 5 else None)

    # Snapshot dates from the longest series
    sample = max(stocks.values(), key=len)
    end = len(sample) - fwd_days - 2
    snap_positions = list(range(WARMUP_BARS, end, step))
    snap_dates = [sample.index[i] for i in snap_positions]
    print(f"{len(snap_dates)} snapshots: {snap_dates[0].date()} → "
          f"{snap_dates[-1].date()}", flush=True)

    # records[snapshot_i] = list of {factor dict + _alpha}
    per_snap: list[list[dict]] = []
    for si, snap in enumerate(snap_dates):
        rows = []
        for sym, df in stocks.items():
            sub = df[df.index <= snap]
            if len(sub) < WARMUP_BARS:
                continue
            c = sub["Close"].dropna()
            future = df[df.index > snap]["Close"].dropna()
            if len(future) < fwd_days:
                continue
            entry = float(c.iloc[-1])
            exit_ = float(future.iloc[fwd_days - 1])
            if entry <= 0:
                continue
            fwd_ret = (exit_ / entry - 1) * 100
            b0 = bench.asof(snap)
            b1 = bench.asof(future.index[fwd_days - 1])
            if pd.isna(b0) or pd.isna(b1) or b0 <= 0:
                continue
            alpha = fwd_ret - (float(b1) / float(b0) - 1) * 100

            bench_pit = bench[bench.index <= snap]
            f = _factors_at(sub, bench_pit)
            if f is None:
                continue
            if include_v2:
                proxy_pit = proxy[proxy.index <= snap] if proxy is not None else None
                f["score_v2"] = _score_v2_at(sub, proxy_pit)
            f["_alpha"] = alpha
            rows.append(f)
        per_snap.append(rows)
        print(f"  snapshot {si+1}/{len(snap_dates)} {snap.date()}: "
              f"{len(rows)} obs", flush=True)

    all_obs = [r for rows in per_snap for r in rows]
    if len(all_obs) < 200:
        raise RuntimeError(f"only {len(all_obs)} observations — not enough")

    names = [k for k in all_obs[0] if k != "_alpha"]

    def factor_ic(name, row_filter=None) -> dict | None:
        snap_ics = []
        n_obs = 0
        for rows in per_snap:
            if row_filter is not None:
                rows = [r for r in rows if row_filter(r)]
            paired = [(r[name], r["_alpha"]) for r in rows
                      if r.get(name) is not None and not pd.isna(r.get(name))]
            n_obs += len(paired)
            ic = _spearman([p[0] for p in paired], [p[1] for p in paired])
            if ic is not None:
                snap_ics.append(ic)
        if len(snap_ics) < 4:
            return None
        mu = float(np.mean(snap_ics))
        sd = float(np.std(snap_ics, ddof=1))
        t = mu / (sd / np.sqrt(len(snap_ics))) if sd > 0 else 0.0
        half = len(snap_ics) // 2
        return {"name": name, "ic_mean": round(mu, 3), "ic_t": round(t, 2),
                "n_obs": n_obs, "snapshots": len(snap_ics),
                "ic_is": round(float(np.mean(snap_ics[:half])), 3),
                "ic_oos": round(float(np.mean(snap_ics[half:])), 3)}

    results = [r for r in (factor_ic(n) for n in names) if r is not None]
    results.sort(key=lambda x: -abs(x["ic_mean"]))

    # Conditional view: IC measured ONLY among stocks the tab actually shows
    # (trending filter ≈ current score ≥ 5). This is the user's real
    # decision population — "of the trending stocks, which do I buy?"
    cond = None
    if include_v2:
        _filt = lambda r: (r.get("score_v2") or 0) >= 5
        cond = [r for r in (factor_ic(n, _filt) for n in names) if r is not None]
        cond.sort(key=lambda x: -abs(x["ic_mean"]))

    out = {"factors": results, "factors_trending_only": cond,
           "snapshots": len(snap_dates),
           "obs_total": len(all_obs), "fwd_days": fwd_days,
           "window": f"{snap_dates[0].date()} → {snap_dates[-1].date()}"}

    # Composite evaluation: per-snapshot rank-blend with given weights.
    # Negative weight = inverse factor (e.g. atr_pct: lower is better).
    def composite_ic(weights: dict[str, float], row_filter=None) -> dict | None:
        snap_ics = []
        for rows in per_snap:
            if row_filter is not None:
                rows = [r for r in rows if row_filter(r)]
            usable = [r for r in rows
                      if all(r.get(f) is not None and not pd.isna(r.get(f))
                             for f in weights)]
            if len(usable) < 30:
                continue
            blend = np.zeros(len(usable))
            for f, w in weights.items():
                vals = pd.Series([r[f] for r in usable]).rank(pct=True).values
                blend = blend + w * vals
            ic = _spearman(list(blend), [r["_alpha"] for r in usable])
            if ic is not None:
                snap_ics.append(ic)
        if len(snap_ics) < 4:
            return None
        mu = float(np.mean(snap_ics))
        sd = float(np.std(snap_ics, ddof=1))
        half = len(snap_ics) // 2
        return {"weights": weights, "ic_mean": round(mu, 3),
                "ic_t": round(mu / (sd / np.sqrt(len(snap_ics))), 2) if sd > 0 else 0.0,
                "ic_is": round(float(np.mean(snap_ics[:half])), 3),
                "ic_oos": round(float(np.mean(snap_ics[half:])), 3),
                "snapshots": len(snap_ics)}

    out["composite_ic"] = composite_ic   # callable for ad-hoc composites
    if v3_weights:
        out["composite"] = composite_ic(v3_weights)
        _filt = (lambda r: (r.get("score_v2") or 0) >= 5) if include_v2 else None
        if _filt:
            out["composite_trending_only"] = composite_ic(v3_weights, _filt)
    return out


if __name__ == "__main__":
    days = int(sys.argv[sys.argv.index("--days") + 1]) if "--days" in sys.argv else 1100
    fwd = int(sys.argv[sys.argv.index("--fwd") + 1]) if "--fwd" in sys.argv else FWD_DAYS_DEFAULT
    step = int(sys.argv[sys.argv.index("--step") + 1]) if "--step" in sys.argv else 20
    res = run_validation(days=days, fwd_days=fwd, step=step)

    def _table(rows, title):
        print(f"\n== {title} ==")
        print(f"{'factor':<16}{'IC':>8}{'t':>7}{'IC-IS':>8}{'IC-OOS':>8}{'n':>8}")
        print("-" * 56)
        for f in rows:
            print(f"{f['name']:<16}{f['ic_mean']:>8}{f['ic_t']:>7}"
                  f"{f['ic_is']:>8}{f['ic_oos']:>8}{f['n_obs']:>8}")

    _table(res["factors"], "FULL UNIVERSE")
    if res.get("factors_trending_only"):
        _table(res["factors_trending_only"],
               "TRENDING SUBSET ONLY (score_v2 ≥ 5 — the user's decision set)")
    print(f"\nwindow {res['window']} · {res['snapshots']} snapshots · "
          f"{res['obs_total']} obs · fwd {res['fwd_days']}d")
