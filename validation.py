"""
Validation harness — phase-averaging, walk-forward, and a hypothesis registry.
=============================================================================

WHY THIS EXISTS
---------------
On 2026-08-13 a portfolio test of the Strategy top-20 produced these CAGRs, all
from the SAME rule and the SAME data — the only difference being which calendar
day the rebalance happened to land on:

    rebalance      mean CAGR     min      max     spread
    monthly          10.67%     3.34%   16.62%   13.28pp
    6-weekly          4.92%    -7.32%   17.46%   24.78pp
    quarterly         3.78%   -10.80%   10.13%   20.93pp
    semi-annual      11.17%     7.17%   19.84%   12.67pp

A 13-25pp swing from date alignment alone is larger than any edge this app has
ever measured. A single-phase backtest is therefore not evidence, and one had
already produced a wrong recommendation ("cut turnover to quarterly", which
looked like +3pp/yr and was noise).

Three protections, all of which must be cheap enough that nobody skips them:

  1. PHASE AVERAGING   run every start offset, report mean AND spread.
  2. WALK-FORWARD      many rolling train/test folds, not one midpoint split.
                       A single split has one degree of freedom and hides regime
                       dependence — the confluence study looked +1.48pp in-sample
                       and -2.22pp out, which one number could not have shown.
  3. HYPOTHESIS COUNT  every variant tested is recorded. At p<0.05, 1 in 20
                       passes by luck; after ~40 variants (roughly where this app
                       now sits) you EXPECT two false positives. The registry
                       makes the denominator visible and applies a correction.

Nothing here computes a signal. It only decides whether a result is believable.
"""
from __future__ import annotations

import json
import math
import os
import time
from datetime import datetime

import numpy as np
import pandas as pd

REGISTRY_PATH = os.path.join(os.path.dirname(__file__), ".hypothesis_registry.json")


# ── 1. PHASE AVERAGING ──────────────────────────────────────────────────────
def phase_average(run_fn, period: int, n_phases: int = 6, **kw) -> dict:
    """Run `run_fn(phase=p, **kw)` across evenly spaced start offsets.

    `run_fn` must accept `phase` and return a float (the metric under test).
    Returns mean/min/max/spread/std. A spread that rivals the effect size means
    the result is alignment luck — report it, do not average it away silently.
    """
    step = max(1, period // n_phases)
    phases = list(range(0, period, step))[:n_phases]
    vals = []
    for p in phases:
        try:
            v = run_fn(phase=p, **kw)
            if v is not None and np.isfinite(v):
                vals.append(float(v))
        except Exception:
            continue
    if not vals:
        return {"n_phases": 0, "mean": None}
    a = np.array(vals, dtype=float)
    return {"n_phases": len(a), "mean": round(float(a.mean()), 4),
            "min": round(float(a.min()), 4), "max": round(float(a.max()), 4),
            "spread": round(float(a.max() - a.min()), 4),
            "std": round(float(a.std(ddof=1)), 4) if len(a) > 1 else 0.0,
            "phases": phases[:len(a)]}


def phase_verdict(res: dict, effect: float) -> str:
    """Is an effect distinguishable from alignment noise?"""
    if not res or res.get("mean") is None:
        return "NO DATA"
    if res.get("n_phases", 0) < 3:
        return "TOO FEW PHASES"
    spread = res.get("spread", 0.0)
    if abs(effect) < spread / 2:
        return f"NOISE — effect {effect:+.2f} is inside the +/-{spread/2:.2f} alignment band"
    return f"SURVIVES — effect {effect:+.2f} exceeds the {spread/2:.2f} alignment band"


# ── 2. WALK-FORWARD ─────────────────────────────────────────────────────────
def walk_forward(dates, train: int = 252, test: int = 63, step: int | None = None):
    """Yield (train_slice, test_slice) index pairs — rolling, non-overlapping tests.

    Replaces the single midpoint IS/OOS split used everywhere in this app so far.
    """
    dates = list(dates)
    step = step or test
    i = train
    while i + test <= len(dates):
        yield (slice(i - train, i), slice(i, i + test))
        i += step


def fold_report(fold_scores) -> dict:
    """Summarise per-fold results. `hit_rate` is the honest headline: a factor that
    works in 5 of 12 folds is not an edge however good the pooled mean looks."""
    a = np.array([s for s in fold_scores if s is not None and np.isfinite(s)], dtype=float)
    if a.size == 0:
        return {"folds": 0}
    pos = int((a > 0).sum())
    t = float(a.mean() / (a.std(ddof=1) / math.sqrt(a.size))) if a.size > 1 and a.std(ddof=1) > 0 else 0.0
    return {"folds": int(a.size), "mean": round(float(a.mean()), 4),
            "median": round(float(np.median(a)), 4),
            "std": round(float(a.std(ddof=1)), 4) if a.size > 1 else 0.0,
            "folds_positive": pos, "hit_rate": round(100 * pos / a.size, 1),
            "t_stat": round(t, 2)}


# ── 3. HYPOTHESIS REGISTRY ──────────────────────────────────────────────────
def _load_registry() -> list:
    try:
        with open(REGISTRY_PATH) as f:
            return json.load(f)
    except Exception:
        return []


def register(name: str, description: str, metric: float, p_value: float | None = None,
             n: int | None = None, verdict: str = "", family: str = "general") -> dict:
    """Record one tested variant. Returns the multiple-comparisons context.

    The point is the DENOMINATOR. This app has tested trend confluence, VCP
    contractions, volume dry-up, volume surge, tight range, above-MA50, quiet
    accumulation, trend gauges, 4 rebalance frequencies, concentration, hold-
    winners and trailing stops. Each looked promising alone; collectively they
    are ~40 draws, so ~2 should clear p<0.05 by chance alone.
    """
    reg = _load_registry()
    reg.append({"name": name, "description": description, "metric": metric,
                "p_value": p_value, "n": n, "verdict": verdict, "family": family,
                "at": datetime.now().isoformat(timespec="seconds")})
    try:
        tmp = REGISTRY_PATH + ".tmp"
        with open(tmp, "w") as f:
            json.dump(reg, f, indent=1)
        os.replace(tmp, REGISTRY_PATH)
    except Exception:
        pass
    return multiple_comparison_context(family=family)


def multiple_comparison_context(family: str = "general", alpha: float = 0.05) -> dict:
    """Bonferroni + Benjamini-Hochberg over everything registered in `family`."""
    reg = [r for r in _load_registry() if r.get("family") == family]
    k = len(reg)
    if k == 0:
        return {"tested": 0}
    bonf = alpha / k
    ps = sorted([r["p_value"] for r in reg if r.get("p_value") is not None])
    bh_cut = None
    for i, p in enumerate(ps, start=1):
        if p <= (i / len(ps)) * alpha:
            bh_cut = p
    return {"tested": k, "alpha": alpha,
            "bonferroni_threshold": round(bonf, 5),
            "bh_threshold": (round(bh_cut, 5) if bh_cut is not None else None),
            "expected_false_positives": round(k * alpha, 2),
            "note": (f"{k} variants tested in '{family}'; at alpha={alpha} you EXPECT "
                     f"{k*alpha:.1f} to pass by chance. A result needs p<{bonf:.5f} "
                     f"to survive Bonferroni.")}


def summary() -> dict:
    reg = _load_registry()
    fams = {}
    for r in reg:
        fams.setdefault(r.get("family", "general"), []).append(r)
    return {"total_tested": len(reg),
            "families": {k: len(v) for k, v in fams.items()},
            "registry_path": REGISTRY_PATH}
