"""
Regime out-of-sample validation (#8) — is the regime engine a real signal or a fit?

The Tier-1 review left one engine standing: ALL-WEATHER, whose value comes from the
regime overlay (cash in BEAR, defense in SIDEWAYS, offense in BULL), not from stock
selection (factor_model.py showed momentum/defensive have no alpha; All-Weather is a
TIMING strategy a static factor model can't judge). So the whole thesis rests on one
question a factor model cannot answer: does the regime signal actually time markets,
or are its confirm_up/confirm_down parameters fit to this sample?

Three tests, all point-in-time (regime label at bar t-1 decides exposure for bar t):

  A. TIMING VALUE   — "hold the index unless BEAR, else cash" vs buy-and-hold, on the
                      index itself. If timing can't help the index, All-Weather's edge
                      is not coming from the regime signal.
  B. PARAM SURFACE  — sweep confirm_up × confirm_down. A robust signal is a PLATEAU
                      (neighbours agree); a fit one is a lone SPIKE at the chosen 5/5.
  C. TRAIN / TEST   — pick the best params on the first half, apply UNSEEN to the
                      second half. Edge that survives this is real; edge that only
                      shows in-sample is curve-fitting.

Run on Nifty (6y — contains the 2022 bear, the only thing a timer can add value on)
and Smallcap (3.4y — no bear, so expect little timing value, as a sanity check).

Run:  python regime_review.py
"""
from __future__ import annotations

import numpy as np

import benchmark as bm
import regime_engine as rg
import institutional_review as ir

TRADING_DAYS = 252
RISK_FREE_ANNUAL = ir.RISK_FREE_ANNUAL
RF_DAILY = (1.0 + RISK_FREE_ANNUAL) ** (1.0 / TRADING_DAYS) - 1.0


def _timed_curve(close: np.ndarray, regime: np.ndarray) -> np.ndarray:
    """Equity curve of: hold the index unless YESTERDAY'S regime is BEAR, else cash.
    Uses regime[t-1] for day t — strictly causal, no same-bar look-ahead."""
    rets = close[1:] / close[:-1] - 1.0
    eq = [1.0]
    for t in range(1, len(close)):
        invested = regime[t - 1] != rg.BEAR
        r = rets[t - 1] if invested else RF_DAILY
        eq.append(eq[-1] * (1.0 + r))
    return np.asarray(eq)


def _bh_curve(close: np.ndarray) -> np.ndarray:
    return close / close[0]


def _metrics(curve: np.ndarray) -> dict:
    r = curve[1:] / curve[:-1] - 1.0
    m = ir.risk_metrics(list(r), TRADING_DAYS, dd_curve=curve)
    return {"cagr": m["cagr"], "sharpe": m["sharpe"], "sortino": m["sortino"],
            "max_dd": m["max_dd"], "calmar": m["calmar"]}


def _timed_sharpe(close: np.ndarray, cu: int, cd: int) -> float:
    reg = rg.regime_series(close, confirm_up=cu, confirm_down=cd)
    return _metrics(_timed_curve(close, reg))["sharpe"]


def test_timing_value(name: str, close: np.ndarray, cu: int, cd: int) -> dict:
    reg = rg.regime_series(close, confirm_up=cu, confirm_down=cd)
    bh = _metrics(_bh_curve(close))
    tm = _metrics(_timed_curve(close, reg))
    bear_frac = float(np.mean(reg == rg.BEAR))
    return {"name": name, "n": len(close), "bh": bh, "timed": tm,
            "bear_frac": round(bear_frac, 3),
            "sharpe_delta": round(tm["sharpe"] - bh["sharpe"], 2),
            "cagr_delta": round(tm["cagr"] - bh["cagr"], 2),
            "dd_delta": round(tm["max_dd"] - bh["max_dd"], 2)}


def test_param_surface(close: np.ndarray, grid=(2, 3, 5, 8, 10, 15)) -> dict:
    surface = {}
    best = (-9, None)
    for cu in grid:
        for cd in grid:
            s = _timed_sharpe(close, cu, cd)
            surface[(cu, cd)] = round(s, 2)
            if s > best[0]:
                best = (s, (cu, cd))
    vals = np.array(list(surface.values()))
    return {"surface": surface, "best": best, "grid": grid,
            "mean": round(float(vals.mean()), 2), "std": round(float(vals.std()), 2),
            "at_5_5": surface.get((5, 5))}


def test_train_test(close: np.ndarray, grid=(2, 3, 5, 8, 10, 15)) -> dict:
    mid = len(close) // 2
    train, test = close[:mid], close[mid:]
    best = (-9, None)
    for cu in grid:
        for cd in grid:
            s = _timed_sharpe(train, cu, cd)
            if s > best[0]:
                best = (s, (cu, cd))
    cu, cd = best[1]
    reg_test = rg.regime_series(test, confirm_up=cu, confirm_down=cd)
    tm = _metrics(_timed_curve(test, reg_test))
    bh = _metrics(_bh_curve(test))
    # also the DEFAULT 5/5 on the test half, to compare "fit best" vs "just use 5/5"
    reg_def = rg.regime_series(test, confirm_up=5, confirm_down=5)
    tm_def = _metrics(_timed_curve(test, reg_def))
    return {"best_train_params": best[1], "best_train_sharpe": round(best[0], 2),
            "test_bh": bh, "test_timed_fit": tm, "test_timed_default_5_5": tm_def,
            "oos_sharpe_delta_fit": round(tm["sharpe"] - bh["sharpe"], 2),
            "oos_sharpe_delta_default": round(tm_def["sharpe"] - bh["sharpe"], 2)}


def _print_timing(r: dict) -> None:
    print(f"\n  [{r['name']}]  {r['n']} bars,  BEAR {r['bear_frac']*100:.0f}% of the time")
    print(f"      buy & hold : CAGR {r['bh']['cagr']:>6.2f}%  Sharpe {r['bh']['sharpe']:>5.2f}  "
          f"maxDD {r['bh']['max_dd']:>7.2f}%  Calmar {r['bh']['calmar']:.2f}")
    print(f"      regime-timed: CAGR {r['timed']['cagr']:>6.2f}%  Sharpe {r['timed']['sharpe']:>5.2f}  "
          f"maxDD {r['timed']['max_dd']:>7.2f}%  Calmar {r['timed']['calmar']:.2f}")
    print(f"      → timing adds: {r['sharpe_delta']:+.2f} Sharpe, {r['cagr_delta']:+.2f}pp CAGR, "
          f"{r['dd_delta']:+.2f}pp maxDD (less negative = better)")


def _print_surface(name: str, s: dict) -> None:
    print(f"\n  [{name}] confirm_up × confirm_down → regime-timed Sharpe")
    grid = s["grid"]
    print("        cd:  " + "  ".join(f"{cd:>5}" for cd in grid))
    for cu in grid:
        row = "  ".join(f"{s['surface'][(cu,cd)]:>5.2f}" for cd in grid)
        print(f"    cu {cu:>3}:  {row}")
    print(f"    mean {s['mean']}  std {s['std']}  best {s['best'][1]}={s['best'][0]:.2f}  "
          f"default(5,5)={s['at_5_5']}")
    spread = s["best"][0] - s["at_5_5"]
    verdict = ("PLATEAU (5/5 near best — robust)" if spread <= 0.15
               else f"5/5 is {spread:.2f} Sharpe below best — some sensitivity")
    print(f"    → {verdict}")


def _print_tt(name: str, r: dict) -> None:
    print(f"\n  [{name}] TRAIN/TEST (fit on first half, apply to second):")
    print(f"      best train params {r['best_train_params']} (train Sharpe {r['best_train_sharpe']})")
    print(f"      test buy&hold      : CAGR {r['test_bh']['cagr']:>6.2f}%  Sharpe {r['test_bh']['sharpe']:.2f}  maxDD {r['test_bh']['max_dd']:.2f}%")
    print(f"      test timed (fit)   : CAGR {r['test_timed_fit']['cagr']:>6.2f}%  Sharpe {r['test_timed_fit']['sharpe']:.2f}  maxDD {r['test_timed_fit']['max_dd']:.2f}%")
    print(f"      test timed (5/5)   : CAGR {r['test_timed_default_5_5']['cagr']:>6.2f}%  Sharpe {r['test_timed_default_5_5']['sharpe']:.2f}  maxDD {r['test_timed_default_5_5']['max_dd']:.2f}%")
    print(f"      → OOS timing edge: fit {r['oos_sharpe_delta_fit']:+.2f} Sharpe, "
          f"default-5/5 {r['oos_sharpe_delta_default']:+.2f} Sharpe")


def main() -> None:
    print("\n" + "=" * 78)
    print("  REGIME OUT-OF-SAMPLE VALIDATION (#8)")
    print("=" * 78)
    nifty = bm.get_benchmark(days=2800)
    sc = bm.get_smallcap_benchmark(days=1600)
    series = []
    if nifty is not None and len(nifty) > 300:
        series.append(("Nifty-50 (6y, has 2022 bear)", nifty.to_numpy(dtype=float)))
    if sc is not None and len(sc) > 300:
        series.append(("Smallcap-250 (3.4y, no bear)", sc.to_numpy(dtype=float)))

    print("\n--- A. TIMING VALUE (default confirm 5/5) ---")
    for name, arr in series:
        _print_timing(test_timing_value(name, arr, 5, 5))

    print("\n--- B. PARAMETER SURFACE (is 5/5 a plateau or a fit spike?) ---")
    for name, arr in series:
        _print_surface(name, test_param_surface(arr))

    print("\n--- C. TRAIN / TEST (does timing survive out of sample?) ---")
    for name, arr in series:
        _print_tt(name, test_train_test(arr))

    print("\n" + "=" * 78)


if __name__ == "__main__":
    main()
