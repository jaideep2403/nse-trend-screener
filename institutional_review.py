"""
Institutional review — the one honest number an allocator asks first.

The app already backtests survivorship-free, next-open, cost-aware, and reports
CAGR / Sharpe / max-drawdown vs a Nifty-50 buy-and-hold (system_backtest.py). Two
things were still missing before those numbers can be shown to a serious investor,
and both flatter the strategy when omitted:

  1. WRONG BENCHMARK. A small/micro-cap momentum + delivery book must be judged
     against a SMALL-CAP index. Nifty-50 is large-cap; small-caps beat it over this
     sample, so the strategy can "beat the market" while trailing a cheap smallcap
     index fund. We add the smallcap ETF benchmark (benchmark.get_smallcap_benchmark)
     for the window it actually covers (~2023-02 on — no longer investable smallcap
     total-return series exists in the cache; that limit is reported, not hidden).

  2. NO TAX. A monthly-rebalanced book turns over inside 12 months, so ~all gains
     are SHORT-TERM (20% post-2024). Ignoring that overstates what you keep. We
     apply a fiscal-year (Apr–Mar) STCG overlay with within-year loss offset.

It also reports, for #7 (multiple-comparisons honesty):
  • Probabilistic Sharpe Ratio  — P(true SR > 0) given the return moments.
  • Bonferroni haircut          — does the Sharpe survive the number of variants
                                   that were tried (garden-of-forking-paths).
  • Minimum Track Record Length — how long a record you'd NEED for the Sharpe to
                                   be significant. If MinTRL > your sample, you do
                                   not yet have significance, full stop.

Nothing here touches the live app or the demo box — it is an offline analysis that
reuses costs.py, benchmark.py and system_backtest.py. Run:  python institutional_review.py
"""
from __future__ import annotations

import math
import sys

import numpy as np
import pandas as pd

import benchmark as bm
import costs
import system_backtest as sbt

TRADING_DAYS = 252
RISK_FREE_ANNUAL = getattr(sbt, "RISK_FREE_ANNUAL", 0.06)
EULER_GAMMA = 0.5772156649


# ── normal helpers (no scipy dependency) ────────────────────────────────────────
def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _norm_ppf(p: float) -> float:
    """Inverse standard normal CDF (Acklam's rational approximation)."""
    if p <= 0.0:
        return -math.inf
    if p >= 1.0:
        return math.inf
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    plow, phigh = 0.02425, 1 - 0.02425
    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
               ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    if p > phigh:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
                ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    q = p - 0.5
    r = q * q
    return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5]) * q / \
           (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)


# ── risk metrics from a return series ───────────────────────────────────────────
def _max_dd(curve: np.ndarray) -> float:
    """Max drawdown (%) of an equity curve, as a negative number."""
    if curve.size == 0:
        return 0.0
    peak = np.maximum.accumulate(curve)
    return float((curve / peak - 1.0).min() * 100.0)


def risk_metrics(period_rets: list[float], periods_per_year: float,
                 dd_curve: np.ndarray | None = None) -> dict:
    """Sharpe / Sortino / Calmar / CAGR / vol / maxDD from period returns.

    Sharpe & Sortino are excess over the risk-free rate, annualised. maxDD uses a
    supplied finer curve (e.g. daily) when given, else the period curve.
    """
    pr = np.asarray([r for r in period_rets if r is not None], dtype=float)
    n = pr.size
    if n == 0:
        return {"n": 0, "cagr": 0.0, "vol": 0.0, "sharpe": 0.0, "sortino": 0.0,
                "max_dd": 0.0, "calmar": 0.0, "total_return": 0.0}
    curve = np.cumprod(1.0 + pr)
    yrs = n / periods_per_year
    total = float(curve[-1] - 1.0)
    cagr = (curve[-1] ** (1.0 / yrs) - 1.0) if yrs > 0 else 0.0
    rf_period = (1.0 + RISK_FREE_ANNUAL) ** (1.0 / periods_per_year) - 1.0
    excess = pr - rf_period
    sd = float(pr.std(ddof=1)) if n > 1 else 0.0
    downside = pr[pr < rf_period] - rf_period
    dsd = float(np.sqrt((downside ** 2).mean())) if downside.size else 0.0
    ppy_sqrt = math.sqrt(periods_per_year)
    sharpe = (float(excess.mean()) / sd * ppy_sqrt) if sd > 0 else 0.0
    sortino = (float(excess.mean()) / dsd * ppy_sqrt) if dsd > 0 else 0.0
    dd = _max_dd(dd_curve if dd_curve is not None and dd_curve.size else curve)
    calmar = (cagr * 100.0 / abs(dd)) if dd < 0 else 0.0
    ann_vol = sd * ppy_sqrt * 100.0
    return {"n": n, "cagr": round(cagr * 100, 2), "vol": round(ann_vol, 2),
            "sharpe": round(sharpe, 2), "sortino": round(sortino, 2),
            "max_dd": round(dd, 2), "calmar": round(calmar, 2),
            "total_return": round(total * 100, 2)}


# ── capital-gains tax overlay (fiscal-year, within-year loss offset) ─────────────
def _fiscal_year(ts: pd.Timestamp) -> int:
    """Indian FY label: Apr..Mar. FY starting Apr 2025 → 2025."""
    return ts.year if ts.month >= 4 else ts.year - 1


def apply_stcg_tax(dated_period_rets: list[tuple[pd.Timestamp, float]],
                   start_equity: float, rate: float = costs.STCG_RATE) -> dict:
    """Re-compound period returns while deducting STCG at each fiscal year-end.

    Realistic: gains realise as the monthly book turns over; losses within the SAME
    fiscal year offset gains before tax; a losing year owes nothing (and, kept
    simple, does NOT carry the loss forward — that omission makes the tax drag a
    slight UPPER bound, i.e. conservative, never flattering).
    """
    eq = start_equity
    year_start_eq = start_equity
    cur_fy = _fiscal_year(dated_period_rets[0][0]) if dated_period_rets else None
    total_tax = 0.0
    net_curve = [start_equity]
    for i, (ts, r) in enumerate(dated_period_rets):
        eq *= (1.0 + r)
        last = i == len(dated_period_rets) - 1
        fy = _fiscal_year(ts)
        if fy != cur_fy or last:
            gain = eq - year_start_eq
            tax = rate * gain if gain > 0 else 0.0
            eq -= tax
            total_tax += tax
            year_start_eq = eq
            cur_fy = fy
        net_curve.append(eq)
    gross_final = start_equity
    for _, r in dated_period_rets:
        gross_final *= (1.0 + r)
    return {"net_final": eq, "gross_final": gross_final, "total_tax": total_tax,
            "tax_drag_pct_of_gain": round(100 * total_tax / (gross_final - start_equity), 1)
            if gross_final > start_equity else 0.0,
            "net_curve": net_curve}


# ── #7 : Probabilistic Sharpe, Bonferroni haircut, MinTRL ───────────────────────
def probabilistic_sharpe(sr_ann: float, n: int, ppy: float,
                         skew: float, kurt: float, sr_benchmark: float = 0.0) -> float:
    """P(true SR > sr_benchmark). Skew/kurtosis-adjusted (Bailey & López de Prado).

    sr_ann and sr_benchmark are ANNUALISED; converted to per-period internally.
    kurt is the RAW (non-excess) kurtosis (normal = 3).
    """
    if n < 2 or ppy <= 0:
        return float("nan")
    sr = sr_ann / math.sqrt(ppy)          # per-period
    srb = sr_benchmark / math.sqrt(ppy)
    denom = math.sqrt(max(1e-12, 1.0 - skew * sr + (kurt - 1.0) / 4.0 * sr * sr))
    z = (sr - srb) * math.sqrt(n - 1) / denom
    return _norm_cdf(z)


def bonferroni_haircut(sr_ann: float, n: int, ppy: float, n_trials: int) -> dict:
    """One-sided p-value of the Sharpe, and whether it survives ×n_trials.

    t ≈ SR_ann * sqrt(years). Reports the family-wise-adjusted p and the SR you'd
    have needed to clear a Bonferroni bar at α=0.05 across n_trials.
    """
    yrs = n / ppy
    t = sr_ann * math.sqrt(yrs)
    p_one = 1.0 - _norm_cdf(t)
    p_fw = min(1.0, p_one * n_trials)
    alpha_adj = 0.05 / max(1, n_trials)
    z_needed = _norm_ppf(1.0 - alpha_adj)
    sr_needed = z_needed / math.sqrt(yrs) if yrs > 0 else float("inf")
    return {"t_stat": round(t, 2), "p_raw": p_one, "p_bonferroni": p_fw,
            "survives_bonferroni": p_fw < 0.05, "n_trials": n_trials,
            "sr_needed_ann": round(sr_needed, 2)}


def min_track_record_length(sr_ann: float, ppy: float, skew: float, kurt: float,
                            conf: float = 0.95) -> float:
    """Years of track record needed for the Sharpe to be significant at `conf`."""
    if sr_ann <= 0:
        return float("inf")
    sr = sr_ann / math.sqrt(ppy)
    z = _norm_ppf(conf)
    n = 1 + (1.0 - skew * sr + (kurt - 1.0) / 4.0 * sr * sr) * (z / sr) ** 2
    return round(n / ppy, 2)


# ── daily-window comparison vs the smallcap ETF ─────────────────────────────────
def _daily_metrics(px: pd.Series) -> dict:
    r = px.pct_change().dropna().to_numpy()
    if r.size < 2:
        return {"cagr": 0.0, "sharpe": 0.0, "sortino": 0.0, "max_dd": 0.0, "n": r.size}
    m = risk_metrics(list(r), TRADING_DAYS, dd_curve=px.to_numpy())
    return {"cagr": m["cagr"], "sharpe": m["sharpe"], "sortino": m["sortino"],
            "max_dd": m["max_dd"], "n": r.size}


def smallcap_comparison(daily_curve: list[tuple[str, float]], days: int) -> dict | None:
    """Compare the strategy to the smallcap ETF over the window they SHARE."""
    if not daily_curve:
        return None
    sc = bm.get_smallcap_benchmark(days=days)
    if sc is None or sc.empty:
        return {"error": "no smallcap benchmark series available"}
    strat = pd.Series({pd.Timestamp(d): v for d, v in daily_curve}).sort_index()
    common = strat.index.intersection(sc.index)
    if len(common) < 60:
        return {"error": f"only {len(common)} shared sessions — too short to compare"}
    s = strat.reindex(common).dropna()
    b = sc.reindex(common).dropna()
    common = s.index.intersection(b.index)
    s, b = s.reindex(common), b.reindex(common)
    sm, bmet = _daily_metrics(s), _daily_metrics(b)
    return {"window_start": str(common[0].date()), "window_end": str(common[-1].date()),
            "sessions": len(common), "strategy": sm, "smallcap": bmet,
            "cagr_delta": round(sm["cagr"] - bmet["cagr"], 2),
            "sharpe_delta": round(sm["sharpe"] - bmet["sharpe"], 2),
            "dd_delta": round(sm["max_dd"] - bmet["max_dd"], 2),
            "beats_smallcap_net": sm["cagr"] > bmet["cagr"]}


# ── driver ──────────────────────────────────────────────────────────────────────
def review_strategy(strategy: str, days: int, rebal: int, top_k: int,
                    n_trials: int) -> dict:
    # strategy is a string flag in run_system_backtest ('momentum' | 'defensive' |
    # 'all_weather' | 'blend'), NOT a boolean kwarg.
    bt = sbt.run_system_backtest(days=days, rebal=rebal, top_k=top_k, strategy=strategy)

    start_equity = 1_000_000.0
    eq = bt.get("equity_curve") or []
    if not eq:
        return {"strategy": strategy, "error": "backtest produced no equity curve"}
    dates = [pd.Timestamp(d) for d, _ in eq]
    vals = [start_equity] + [v for _, v in eq]
    period_rets = [vals[i] / vals[i - 1] - 1.0 for i in range(1, len(vals))]
    ppy = TRADING_DAYS / rebal

    daily_curve = bt.get("daily_curve") or []
    dd_daily = np.asarray([v for _, v in daily_curve], dtype=float) if daily_curve else None

    net_cost = risk_metrics(period_rets, ppy, dd_curve=dd_daily)

    # net of TAX
    tax = apply_stcg_tax(list(zip(dates, period_rets)), start_equity)
    net_tax_rets = [tax["net_curve"][i] / tax["net_curve"][i - 1] - 1.0
                    for i in range(1, len(tax["net_curve"]))]
    net_tax = risk_metrics(net_tax_rets, ppy)

    # #7 honesty stats on the net-of-cost period returns
    pr = np.asarray(period_rets, dtype=float)
    skew = float(pd.Series(pr).skew()) if pr.size > 2 else 0.0
    kurt = float(pd.Series(pr).kurt() + 3.0) if pr.size > 3 else 3.0   # raw kurtosis
    psr = probabilistic_sharpe(net_cost["sharpe"], pr.size, ppy, skew, kurt)
    bonf = bonferroni_haircut(net_cost["sharpe"], pr.size, ppy, n_trials)
    mintrl = min_track_record_length(net_cost["sharpe"], ppy, skew, kurt)

    return {
        "strategy": strategy,
        "window": {"start": bt.get("start"), "end": bt.get("as_of"),
                   "years": round(pr.size / ppy, 2), "rebalances": pr.size,
                   "trades": bt.get("trades"),
                   "avg_deployment_pct": bt.get("avg_deployment_pct")},
        "net_of_cost": net_cost,
        "net_of_tax": {**net_tax, "total_tax_rupees": round(tax["total_tax"]),
                       "tax_drag_pct_of_gain": tax["tax_drag_pct_of_gain"]},
        "vs_nifty_full": {"nifty": bt.get("benchmark"),
                          "cagr_delta": round(net_cost["cagr"] - bt["benchmark"]["cagr"], 2),
                          "note": "Nifty-50 = LARGE cap; flatters a smallcap book"},
        "vs_smallcap_window": smallcap_comparison(daily_curve, days),
        "significance_7": {"prob_sharpe_gt0": round(psr, 3),
                           "min_track_record_yrs": mintrl,
                           **bonf,
                           "skew": round(skew, 2), "kurtosis": round(kurt, 2)},
    }


def _fmt_metrics(m: dict) -> str:
    return (f"CAGR {m['cagr']:>6.2f}%  Sharpe {m['sharpe']:>5.2f}  "
            f"Sortino {m['sortino']:>5.2f}  maxDD {m['max_dd']:>7.2f}%  "
            f"Calmar {m['calmar']:>4.2f}  vol {m['vol']:>5.2f}%")


def _print(rev: dict) -> None:
    s = rev["strategy"].upper()
    print("\n" + "=" * 78)
    print(f"  {s}")
    print("=" * 78)
    if "error" in rev:
        print("  ERROR:", rev["error"]); return
    w = rev["window"]
    print(f"  window {w['start']} → {w['end']}  ({w['years']}y, {w['rebalances']} rebals, "
          f"{w['trades']} trades, avg deploy {w['avg_deployment_pct']}%)")
    print(f"  net of COST         : {_fmt_metrics(rev['net_of_cost'])}")
    nt = rev["net_of_tax"]
    print(f"  net of COST + TAX   : {_fmt_metrics(nt)}")
    print(f"                        tax paid ₹{nt['total_tax_rupees']:,} "
          f"({nt['tax_drag_pct_of_gain']}% of gross gain)")
    vn = rev["vs_nifty_full"]
    print(f"  vs Nifty-50 (full)  : Nifty CAGR {vn['nifty']['cagr']:.2f}%  "
          f"→ edge {vn['cagr_delta']:+.2f}pp   [{vn['note']}]")
    sc = rev["vs_smallcap_window"]
    if sc and "error" not in sc:
        print(f"  vs SMALLCAP 250     : window {sc['window_start']}→{sc['window_end']} "
              f"({sc['sessions']} sessions)")
        print(f"      strategy        : CAGR {sc['strategy']['cagr']:>6.2f}%  "
              f"Sharpe {sc['strategy']['sharpe']:.2f}  maxDD {sc['strategy']['max_dd']:.2f}%")
        print(f"      smallcap index  : CAGR {sc['smallcap']['cagr']:>6.2f}%  "
              f"Sharpe {sc['smallcap']['sharpe']:.2f}  maxDD {sc['smallcap']['max_dd']:.2f}%")
        verdict = "BEATS" if sc["beats_smallcap_net"] else "TRAILS"
        print(f"      → {verdict} smallcap by {sc['cagr_delta']:+.2f}pp CAGR, "
              f"{sc['sharpe_delta']:+.2f} Sharpe")
    elif sc:
        print(f"  vs SMALLCAP 250     : {sc['error']}")
    g = rev["significance_7"]
    print(f"  significance (#7)   : P(SR>0) {g['prob_sharpe_gt0']:.3f}  "
          f"t={g['t_stat']}  p_raw={g['p_raw']:.4f}  "
          f"p_bonferroni(×{g['n_trials']})={g['p_bonferroni']:.4f} "
          f"[{'survives' if g['survives_bonferroni'] else 'FAILS'}]")
    print(f"                        MinTRL {g['min_track_record_yrs']}y vs {w['years']}y actual  "
          f"(skew {g['skew']}, kurt {g['kurtosis']})")


def main() -> None:
    days = int(sys.argv[1]) if len(sys.argv) > 1 else 1600
    strategies = sys.argv[2].split(",") if len(sys.argv) > 2 else ["momentum"]
    n_trials = int(sys.argv[3]) if len(sys.argv) > 3 else 19   # Bonferroni registry size
    rebal, top_k = 21, 10
    print(f"\nINSTITUTIONAL REVIEW  (days={days}, rebal={rebal}, top_k={top_k}, "
          f"n_trials={n_trials})")
    print("net-of-cost = after costs.py round trip;  net-of-tax = + fiscal-year STCG @ "
          f"{costs.STCG_RATE*100:.0f}%")
    for strat in strategies:
        try:
            _print(review_strategy(strat, days, rebal, top_k, n_trials))
        except Exception as e:
            import traceback
            print(f"\n[{strat}] FAILED: {e}")
            traceback.print_exc()


if __name__ == "__main__":
    main()
