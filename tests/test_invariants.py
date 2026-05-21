"""
Invariant tests for canonical analysis helpers in analysis_utils.

Run via:  python3 tests/test_invariants.py
Exit code 0 = pass, 1 = any test failed.

These tests guard against silent regressions in shared math used by every
scanner. If any scanner reintroduces a local copy of these formulas with
wrong values, the corresponding invariant here will fail.
"""
from __future__ import annotations

import os
import sys
import math
from pathlib import Path

import numpy as np
import pandas as pd

# Make repo importable when running from anywhere
_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))

from analysis_utils import (
    atr,
    adjust_for_splits,
    NIFTY_PROXY_SYMS,
    equal_weight_index,
    volume_baseline,
    cross_sectional_rs_rank,
    sector_adjusted_rs,
    stage_analysis,
)


# ── Test runner ───────────────────────────────────────────────────────────────

_failures: list[str] = []


def _ok(name: str):
    print(f"  ✓ {name}")


def _fail(name: str, msg: str):
    print(f"  ✗ {name}: {msg}")
    _failures.append(f"{name}: {msg}")


def assert_eq(name: str, actual, expected, *, tol: float = 1e-9):
    try:
        if isinstance(actual, float) and isinstance(expected, float):
            if math.isnan(actual) and math.isnan(expected):
                _ok(name); return
            if abs(actual - expected) <= tol:
                _ok(name); return
            _fail(name, f"actual={actual!r} expected={expected!r}")
            return
        if actual == expected:
            _ok(name)
        else:
            _fail(name, f"actual={actual!r} expected={expected!r}")
    except Exception as e:
        _fail(name, f"exception {e}")


def assert_true(name: str, condition: bool, msg: str = ""):
    if condition:
        _ok(name)
    else:
        _fail(name, msg or "condition was False")


# ── ATR (Wilder's) ────────────────────────────────────────────────────────────

def test_atr_wilders_matches_handcalc():
    """ATR of a known monotone series matches Wilder's EWM by hand."""
    # 20 bars, each rising by ₹1, range = ₹0.5
    n = 20
    idx = pd.date_range("2025-01-01", periods=n, freq="B")
    high  = pd.Series([100 + i + 0.5 for i in range(n)], index=idx)
    low   = pd.Series([100 + i - 0.5 for i in range(n)], index=idx)
    close = pd.Series([100 + i for i in range(n)], index=idx)
    df = pd.DataFrame({"High": high, "Low": low, "Close": close})

    # Reference Wilder's ATR computed independently
    tr = pd.concat([
        df["High"] - df["Low"],
        (df["High"] - df["Close"].shift(1)).abs(),
        (df["Low"]  - df["Close"].shift(1)).abs(),
    ], axis=1).max(axis=1)
    expected = float(tr.ewm(alpha=1/14, adjust=False).mean().iloc[-1])

    got = atr(df, period=14)
    assert_eq("atr_matches_wilders_ewm", got, expected, tol=1e-6)


def test_atr_differs_from_sma_in_trending_market():
    """Wilder's ATR diverges from plain SMA on a trending bar pattern."""
    n = 30
    idx = pd.date_range("2025-01-01", periods=n, freq="B")
    # Volatility expansion: range grows from 1 to 5 over the window
    rngs = [1 + i * 0.15 for i in range(n)]
    high  = pd.Series([100 + sum(rngs[:i]) + r / 2 for i, r in enumerate(rngs)], index=idx)
    low   = pd.Series([100 + sum(rngs[:i]) - r / 2 for i, r in enumerate(rngs)], index=idx)
    close = pd.Series([100 + sum(rngs[:i]) for i, _ in enumerate(rngs)], index=idx)
    df = pd.DataFrame({"High": high, "Low": low, "Close": close})
    tr = pd.concat([df["High"] - df["Low"],
                     (df["High"] - df["Close"].shift(1)).abs(),
                     (df["Low"]  - df["Close"].shift(1)).abs()], axis=1).max(axis=1)
    sma = float(tr.rolling(14).mean().iloc[-1])
    wilder = atr(df, period=14)
    # They should be different by at least 1% in a trending market
    assert_true(
        "atr_diverges_from_sma_in_trending_market",
        abs(wilder - sma) / sma > 0.01,
        f"wilder={wilder:.4f} sma={sma:.4f} diff%={abs(wilder-sma)/sma*100:.2f}",
    )


# ── Split adjustment ──────────────────────────────────────────────────────────

def test_split_adjust_3to2_bonus():
    """3:2 bonus (price drops to 2/3 of pre-event) is correctly back-adjusted."""
    n = 30
    idx = pd.date_range("2025-01-01", periods=n, freq="B")
    # Pre-event: ₹150 for 20 days; bonus drops to ₹100 (= 150 * 2/3) for last 10 days
    closes = [150.0] * 20 + [100.0 + (i * 0.5) for i in range(10)]
    df = pd.DataFrame({
        "Open":   closes,
        "High":   [c * 1.01 for c in closes],
        "Low":    [c * 0.99 for c in closes],
        "Close":  closes,
        "Volume": [1_000_000] * n,
    }, index=idx)

    out = adjust_for_splits(df)
    # Pre-event prices should be scaled by 2/3, so first bar should be 150*(2/3)=100.0
    pre_first = float(out["Close"].iloc[0])
    assert_true(
        "split_adjust_3to2_scales_prior_by_two_thirds",
        abs(pre_first - 100.0) < 0.5,
        f"pre_first={pre_first:.4f} (expected ~100.0)",
    )
    # Post-event prices must NOT be modified
    post_last = float(out["Close"].iloc[-1])
    assert_eq("split_adjust_3to2_post_event_unchanged", post_last, closes[-1], tol=1e-6)


def test_split_adjust_no_event_returns_unmodified():
    """A clean series with no >18% drop returns unchanged."""
    n = 50
    idx = pd.date_range("2025-01-01", periods=n, freq="B")
    closes = [100.0 + i * 0.5 for i in range(n)]
    df = pd.DataFrame({
        "Open": closes, "High": [c * 1.01 for c in closes],
        "Low": [c * 0.99 for c in closes], "Close": closes,
        "Volume": [1_000_000] * n,
    }, index=idx)

    out = adjust_for_splits(df)
    assert_true(
        "split_adjust_no_event_unmodified",
        bool((out["Close"].values == df["Close"].values).all()),
        "Series with no >18% drop should be returned untouched",
    )


def test_split_adjust_does_not_have_buggy_1to1_threshold():
    """The 3:2 row used to mistakenly have ratio (3/2 - 1) = 0.5 — a 1:1 bonus
    ratio. Verify the table no longer contains that bad entry by checking that
    a clean 50%-drop is detected as 1:2 split (or 1:1 bonus), not as 3:2."""
    n = 30
    idx = pd.date_range("2025-01-01", periods=n, freq="B")
    closes = [200.0] * 20 + [100.0] * 10   # exact 50% drop = 1:2 split
    df = pd.DataFrame({
        "Open": closes, "High": [c * 1.01 for c in closes],
        "Low": [c * 0.99 for c in closes], "Close": closes,
        "Volume": [1_000_000] * n,
    }, index=idx)
    out = adjust_for_splits(df)
    pre = float(out["Close"].iloc[0])
    # 1:2 split → ratio 1/2 → pre 200 becomes 100
    assert_true(
        "split_adjust_1to2_scales_prior_by_half",
        abs(pre - 100.0) < 0.5,
        f"pre={pre:.4f} (expected ~100.0 for 1:2 split)",
    )


# ── Equal-weight index (no price-level bias) ─────────────────────────────────

def test_equal_weight_index_unaffected_by_price_level():
    """A high-priced stock and a low-priced stock with the SAME % moves
    should contribute equally to the index. Raw `mean(axis=1)` would let
    the high-priced stock dominate; equal_weight_index must not."""
    n = 60
    idx = pd.date_range("2025-01-01", periods=n, freq="B")
    # Stock A starts at ₹100 and rises 0.1%/day
    a = pd.Series([100 * (1.001 ** i) for i in range(n)], index=idx)
    # Stock B starts at ₹10000 and rises 0.1%/day — same % move
    b = pd.Series([10000 * (1.001 ** i) for i in range(n)], index=idx)
    df = pd.DataFrame({"A": a, "B": b})

    proxy = equal_weight_index(df)
    # Each stock rises 0.1%/day so the proxy should rise 0.1%/day too
    expected_final = 100.0 * (1.001 ** (n - 1))   # base=100 by default
    actual_final = float(proxy.iloc[-1])
    assert_true(
        "equal_weight_index_price_invariant",
        abs(actual_final - expected_final) / expected_final < 1e-3,
        f"actual={actual_final:.4f} expected≈{expected_final:.4f}",
    )


# ── Volume baseline (median, robust to outliers) ─────────────────────────────

def test_volume_baseline_median_robust_to_outlier():
    """A single 10x volume spike must not move the median by more than 5%
    (it would move the mean by ~50%)."""
    vol = pd.Series([1_000_000] * 20)
    base_clean = volume_baseline(vol, window=20, use_median=True)
    vol_with_spike = vol.copy()
    vol_with_spike.iloc[10] = 10_000_000   # one 10x bar
    base_spike = volume_baseline(vol_with_spike, window=20, use_median=True)
    assert_true(
        "volume_baseline_median_robust_to_10x_spike",
        abs(base_spike - base_clean) / base_clean < 0.05,
        f"clean={base_clean:.0f} with_spike={base_spike:.0f}",
    )


# ── Cross-sectional RS rank ──────────────────────────────────────────────────

def test_cross_sectional_rs_rank_monotonic():
    """Stocks ranked by return should produce monotonically increasing percentiles."""
    rets = {f"S{i}": float(i) for i in range(100)}   # S0=0%, S99=99%
    ranks = cross_sectional_rs_rank(rets)
    # Worst stock (S0) should be rank ~1, best (S99) ~99
    assert_true("rs_rank_worst_is_near_1", 1 <= ranks["S0"] <= 5, f"S0={ranks['S0']}")
    assert_true("rs_rank_best_is_99", ranks["S99"] == 99, f"S99={ranks['S99']}")
    # Median stock should rank ~50
    assert_true("rs_rank_median_near_50",
                40 <= ranks["S50"] <= 60,
                f"S50={ranks['S50']}")


def test_cross_sectional_rs_rank_in_range():
    """All ranks must be in [1, 99]."""
    rets = {f"S{i}": float(i % 7) for i in range(50)}
    ranks = cross_sectional_rs_rank(rets)
    bad = [s for s, r in ranks.items() if not (1 <= r <= 99)]
    assert_true("rs_rank_all_in_1_99", not bad, f"out_of_range={bad[:3]}")


# ── Sector-adjusted RS (composite of stock-level + sector-level) ─────────────

def test_sector_adjusted_rs_leader_in_leader_sector():
    """High RS stock in a high RS sector should produce highest composite."""
    a = sector_adjusted_rs(90, 90)   # leader stock in leader sector
    b = sector_adjusted_rs(90, 10)   # leader stock in laggard sector
    c = sector_adjusted_rs(10, 90)   # laggard stock in leader sector
    assert_true("composite_leader_beats_split",
                a > b and a > c,
                f"a={a} b={b} c={c}")


def test_sector_adjusted_rs_clamped():
    """Composite always in [1, 99]."""
    cases = [(1, 1), (99, 99), (50, 50), (1, 99), (99, 1), (50, 0), (0, 50)]
    for s, g in cases:
        v = sector_adjusted_rs(s, g)
        assert_true(f"composite_clamped_{s}_{g}", 1 <= v <= 99, f"v={v}")


# ── Nifty proxy basket ────────────────────────────────────────────────────────

def test_nifty_proxy_has_required_size():
    """The canonical proxy basket must be at least 20 large-cap names so
    every scanner gets a representative benchmark."""
    assert_true(
        "nifty_proxy_basket_size_20plus",
        len(NIFTY_PROXY_SYMS) >= 20,
        f"len(NIFTY_PROXY_SYMS)={len(NIFTY_PROXY_SYMS)}",
    )


def test_nifty_proxy_has_no_duplicates():
    assert_true(
        "nifty_proxy_no_duplicate_symbols",
        len(set(NIFTY_PROXY_SYMS)) == len(NIFTY_PROXY_SYMS),
        "duplicate symbol in NIFTY_PROXY_SYMS",
    )


# ── Stage analysis ────────────────────────────────────────────────────────────

def test_stage_analysis_pure_uptrend_returns_stage_2():
    """A clean 200-day uptrend should be classified as Stage 2."""
    n = 200
    idx = pd.date_range("2024-01-01", periods=n, freq="B")
    closes = pd.Series([100 * (1.002 ** i) for i in range(n)], index=idx)
    s = stage_analysis(closes)
    assert_eq("stage_pure_uptrend_is_2", s, 2)


def test_stage_analysis_short_series_returns_0():
    """Fewer than 175 bars → return 0 (insufficient data)."""
    closes = pd.Series([100.0] * 50,
                       index=pd.date_range("2024-01-01", periods=50, freq="B"))
    s = stage_analysis(closes)
    assert_eq("stage_short_series_is_0", s, 0)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("== Invariant tests ==")
    test_atr_wilders_matches_handcalc()
    test_atr_differs_from_sma_in_trending_market()
    test_split_adjust_3to2_bonus()
    test_split_adjust_no_event_returns_unmodified()
    test_split_adjust_does_not_have_buggy_1to1_threshold()
    test_equal_weight_index_unaffected_by_price_level()
    test_volume_baseline_median_robust_to_outlier()
    test_cross_sectional_rs_rank_monotonic()
    test_cross_sectional_rs_rank_in_range()
    test_sector_adjusted_rs_leader_in_leader_sector()
    test_sector_adjusted_rs_clamped()
    test_nifty_proxy_has_required_size()
    test_nifty_proxy_has_no_duplicates()
    test_stage_analysis_pure_uptrend_returns_stage_2()
    test_stage_analysis_short_series_returns_0()

    if _failures:
        print(f"\n{len(_failures)} TEST(S) FAILED:")
        for f in _failures:
            print(f"  - {f}")
        sys.exit(1)
    print("\nAll invariant tests passed.")
    sys.exit(0)


if __name__ == "__main__":
    main()
