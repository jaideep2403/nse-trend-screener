"""
Factor decomposition (#3) — is the edge alpha, or repackaged beta?

Regresses a strategy's rebalance-period returns on investable / constructible
factors. A positive, significant INTERCEPT (alpha) after controlling for the
factors is real skill. If alpha → 0, the "edge" is factor harvesting — you are
being paid the size / momentum / illiquidity risk premia, and you inherit their
crash risk (momentum crashes, small-cap liquidity spirals) for free.

Factors, all point-in-time, aligned to the strategy's own rebalance calendar
(master calendar = benchmark dates; rebalance every `rebal` bars from bar 252):

  MKT    — Nifty (NIFTYBEES) period return minus risk-free            [investable]
  SIZE   — Smallcap-250 minus Nifty (HDFCSML250 − NIFTYBEES)          [investable]
  MOM    — universe cross-sectional 12-month winners−losers (top−bottom decile)
  ILLIQ  — universe cross-sectional illiquid−liquid (by 20-day ADTV)
  LOWVOL — universe cross-sectional low−high realized vol (60-day)

The cross-sectional factors are built from the SAME survivorship-free universe the
backtest uses; each stock's signal is measured with bars ≤ the rebalance date and
its forward return realised over the next window, so they carry no look-ahead.

SIZE needs the smallcap ETF, which only lists from 2023-02 — so run this on the
recent window (days≈1600). That is also the only window where SIZE exists and where
the smallcap benchmark comparison is valid, so it is the honest window to test.

Run:  python factor_model.py [days] [strategy] [factors]
      python factor_model.py 1600 all_weather MKT,SIZE,MOM
"""
from __future__ import annotations

import sys
import math

import numpy as np
import pandas as pd

import benchmark as bm
import system_backtest as sbt

TRADING_DAYS = 252
RISK_FREE_ANNUAL = getattr(sbt, "RISK_FREE_ANNUAL", 0.06)
DECILE = 0.10


# ── OLS with t-stats (no statsmodels dependency) ────────────────────────────────
def _ols(y: np.ndarray, X: np.ndarray) -> dict:
    """y = Xb + e, X already includes an intercept column. Returns coefs, t-stats
    (HC0 heteroskedasticity-robust — the honest choice for return regressions),
    R² and adjusted R²."""
    n, k = X.shape
    XtX = X.T @ X
    XtX_inv = np.linalg.pinv(XtX)
    beta = XtX_inv @ X.T @ y
    resid = y - X @ beta
    # HC0 (White) robust covariance: (X'X)^-1 X' diag(e^2) X (X'X)^-1
    S = (X * (resid ** 2)[:, None]).T @ X
    cov = XtX_inv @ S @ XtX_inv
    se = np.sqrt(np.maximum(np.diag(cov), 0.0))
    tstat = np.where(se > 0, beta / se, 0.0)
    ss_res = float(resid @ resid)
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    adj = 1.0 - (1.0 - r2) * (n - 1) / (n - k) if n > k else 0.0
    return {"beta": beta, "t": tstat, "r2": r2, "adj_r2": adj, "n": n, "k": k}


# ── strategy period returns, dated, on the benchmark calendar ───────────────────
def strategy_period_returns(strategy: str, days: int, rebal: int) -> pd.Series:
    bt = sbt.run_system_backtest(days=days, rebal=rebal, strategy=strategy)
    eq = bt.get("equity_curve") or []
    if not eq:
        raise RuntimeError(f"{strategy}: no equity curve")
    start_equity = 1_000_000.0
    dates = [pd.Timestamp(d) for d, _ in eq]
    vals = [start_equity] + [v for _, v in eq]
    rets = [vals[i] / vals[i - 1] - 1.0 for i in range(1, len(vals))]
    return pd.Series(rets, index=pd.DatetimeIndex(dates), name=strategy)


# ── build the factor return panel on the same rebalance calendar ────────────────
def build_factor_panel(days: int, rebal: int) -> pd.DataFrame:
    from edge_engine import _load_stocks
    stocks = _load_stocks(days=days, survivorship_free=True)
    bench = bm.get_benchmark(days=days)
    sc = bm.get_smallcap_benchmark(days=days)
    if bench is None or len(bench) < 260:
        raise RuntimeError("benchmark unavailable")
    cal = bench.index
    bench_vals = bench.to_numpy(dtype=float)
    rebal_i = list(range(252, len(cal) - rebal - 1, rebal))

    # Pre-extract arrays + own-calendar index per symbol for fast point-in-time reads.
    prep = {}
    for s, df in stocks.items():
        if len(df) < 260:
            continue
        prep[s] = {
            "idx": df.index,
            "close": df["Close"].to_numpy(dtype=float),
            "vol": df["Volume"].to_numpy(dtype=float),
            "n": len(df),
        }

    rf_period = (1.0 + RISK_FREE_ANNUAL) ** (rebal / TRADING_DAYS) - 1.0
    rows = {}
    for ri in rebal_i:
        d0, d1 = cal[ri], cal[ri + rebal]
        mkt = bench_vals[ri + rebal] / bench_vals[ri] - 1.0
        size = np.nan
        if sc is not None and not sc.empty:
            s0 = sc.reindex([d0], method="ffill").iloc[0] if d0 >= sc.index[0] else np.nan
            s1 = sc.reindex([d1], method="ffill").iloc[0] if d1 >= sc.index[0] else np.nan
            if np.isfinite(s0) and np.isfinite(s1) and s0 > 0:
                size = (s1 / s0 - 1.0) - mkt      # smallcap minus market

        mom_sig, illiq_sig, lowvol_sig, fwd = [], [], [], []
        for s, p in prep.items():
            i0 = int(p["idx"].searchsorted(d0, side="right")) - 1
            i1 = int(p["idx"].searchsorted(d1, side="right")) - 1
            if i0 < 252 or i1 <= i0 or i1 >= p["n"]:
                continue
            c = p["close"]
            c0, cprev = c[i0], c[i0 - 252]
            if not (np.isfinite(c0) and np.isfinite(cprev) and cprev > 0 and c0 > 0):
                continue
            fr = c[i1] / c0 - 1.0
            if not np.isfinite(fr):
                continue
            # signals, all point-in-time (<= i0)
            mom = c0 / cprev - 1.0
            adtv = np.nanmean(c[i0 - 20:i0] * p["vol"][i0 - 20:i0]) / 1e7  # ₹cr
            rr = np.diff(c[i0 - 60:i0]) / c[i0 - 60:i0 - 1]   # 59 diffs / 59 bases
            vol = float(np.nanstd(rr)) if rr.size > 5 else np.nan
            if not (np.isfinite(adtv) and adtv > 0 and np.isfinite(vol)):
                continue
            mom_sig.append(mom); illiq_sig.append(-math.log(adtv))
            lowvol_sig.append(-vol); fwd.append(fr)

        if len(fwd) < 40:
            continue
        fwd = np.array(fwd)

        def _ls(sig):
            sig = np.array(sig)
            order = np.argsort(sig)
            k = max(1, int(len(sig) * DECILE))
            top = fwd[order[-k:]].mean()      # highest signal
            bot = fwd[order[:k]].mean()       # lowest signal
            return float(top - bot)

        rows[d1] = {
            "MKT": mkt - rf_period,
            "SIZE": size,
            "MOM": _ls(mom_sig),
            "ILLIQ": _ls(illiq_sig),
            "LOWVOL": _ls(lowvol_sig),
        }
    return pd.DataFrame.from_dict(rows, orient="index").sort_index()


def run_factor_model(strategy: str, days: int, rebal: int, factors: list[str]) -> dict:
    y = strategy_period_returns(strategy, days, rebal)
    F = build_factor_panel(days, rebal)
    rf_period = (1.0 + RISK_FREE_ANNUAL) ** (rebal / TRADING_DAYS) - 1.0
    df = pd.concat([y.rename("strat"), F], axis=1).dropna(subset=["strat"] + factors)
    if len(df) < len(factors) + 5:
        return {"strategy": strategy, "error": f"only {len(df)} aligned periods for "
                f"{len(factors)} factors — too few"}
    yv = (df["strat"] - rf_period).to_numpy()      # strategy EXCESS return
    Xv = np.column_stack([np.ones(len(df))] + [df[f].to_numpy() for f in factors])
    res = _ols(yv, Xv)
    ppy = TRADING_DAYS / rebal
    alpha_period = res["beta"][0]
    alpha_ann = (1.0 + alpha_period) ** ppy - 1.0
    names = ["ALPHA"] + factors
    coefs = {names[i]: {"beta": round(float(res["beta"][i]), 4),
                        "t": round(float(res["t"][i]), 2)} for i in range(len(names))}
    return {"strategy": strategy, "n_periods": res["n"], "factors": factors,
            "alpha_annual_pct": round(alpha_ann * 100, 2),
            "alpha_t": round(float(res["t"][0]), 2),
            "alpha_significant": abs(float(res["t"][0])) > 2.0,
            "r2": round(res["r2"], 3), "adj_r2": round(res["adj_r2"], 3),
            "coefs": coefs}


def _print(r: dict) -> None:
    print("\n" + "=" * 74)
    print(f"  FACTOR MODEL — {r['strategy'].upper()}")
    print("=" * 74)
    if "error" in r:
        print("  ERROR:", r["error"]); return
    print(f"  {r['n_periods']} periods · factors {'+'.join(r['factors'])} · "
          f"R² {r['r2']}  adjR² {r['adj_r2']}")
    print(f"  ALPHA (annualised): {r['alpha_annual_pct']:+.2f}%   t={r['alpha_t']}   "
          f"[{'SIGNIFICANT' if r['alpha_significant'] else 'not significant (|t|<2)'}]")
    print("  loadings:")
    for name, c in r["coefs"].items():
        if name == "ALPHA":
            continue
        star = "  *" if abs(c["t"]) > 2 else ""
        print(f"      {name:<7} beta {c['beta']:+.3f}   t {c['t']:+.2f}{star}")
    if not r["alpha_significant"]:
        print("  → No significant alpha after these factors: the return is explained by")
        print("    factor exposure, not skill. You inherit those factors' crash risk.")


def main() -> None:
    days = int(sys.argv[1]) if len(sys.argv) > 1 else 1600
    strategies = sys.argv[2].split(",") if len(sys.argv) > 2 else ["all_weather"]
    factors = sys.argv[3].split(",") if len(sys.argv) > 3 else ["MKT", "SIZE", "MOM"]
    rebal = 21
    print(f"\nFACTOR DECOMPOSITION (days={days}, rebal={rebal})")
    print(f"factors: {factors}  (HC0-robust t-stats; alpha is skill after these)")
    for strat in strategies:
        try:
            _print(run_factor_model(strat, days, rebal, factors))
        except Exception as e:
            import traceback
            print(f"\n[{strat}] FAILED: {e}")
            traceback.print_exc()


if __name__ == "__main__":
    main()
