"""
Pre-trade check tests — the Add-form checks, their measured LEVELS, and the ledger.

Added 2026-09-18 with tiers 1-4 (book-aware checks, entry/mechanics, event risk, and
the scorecard that marks every decision to market).

Run via:  python3 tests/test_preflight_checks.py
Exit code 0 = pass, 1 = any failure.

Portfolio checks SKIP when portfolio.py is absent (gitignored, owner-only).
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))
os.environ.setdefault("ASCENT_BACKGROUND_JOBS", "0")

_failures: list[str] = []


def _ok(name):
    print(f"  ✓ {name}")


def _fail(name, msg):
    print(f"  ✗ {name}: {msg}")
    _failures.append(f"{name}: {msg}")


def assert_true(name, cond, msg=""):
    (_ok(name) if cond else _fail(name, msg or "condition was False"))


def _df(n=220, start=100.0, drift=0.004, vol=0.012, seed=5, crash_at=None, spread=0.012):
    """A steady uptrend — extended above its MA50. `spread` sets the daily High-Low
    range, which is what the Chandelier's 2.5x multiple acts on: a wide range puts the
    stop far below price (the 'wide stop' sizing flag), a narrow one keeps it close."""
    rng = np.random.default_rng(seed)
    c = start * np.exp(np.cumsum(rng.normal(drift, vol, n)))
    if crash_at is not None:
        c[crash_at:] *= 0.90
    idx = pd.bdate_range("2026-01-01", periods=n)
    d = pd.DataFrame({"Open": c, "High": c * (1 + spread), "Low": c * (1 - spread), "Close": c,
                      "Volume": np.full(n, 5e5)}, index=idx)
    if crash_at is not None:                      # make it a circuit day: closes AT the low
        d.iloc[crash_at, d.columns.get_loc("Low")] = d["Close"].iloc[crash_at]
        d.iloc[crash_at, d.columns.get_loc("High")] = d["Close"].iloc[crash_at] * 1.001
    return d


def _levels(res):
    return {c["id"]: c["level"] for c in res.get("checks", [])}


def _patch(pf, df, positions):
    pf._build_history = lambda sym, days=250: df
    pf._load_store = lambda: {"positions": positions}
    pf._sector_of = lambda sym: None              # avoid the sector-map import in tests
    pf.settings = lambda: dict(pf._SETTING_DEFAULTS)


def test_levels_match_what_was_measured(pf):
    df = _df(vol=0.02, spread=0.05)           # wide daily range ⇒ the ATR stop sits >8% away
    _patch(pf, df, [])
    res = pf.preflight_position("TESTCO", 100, float(df["Close"].iloc[-1]), df.index[-1].date().isoformat())
    assert_true("fixture puts the stop >8% below entry", (res.get("stop_dist_pct") or 0) > 8,
                f"stop {res.get('stop_dist_pct')}%")
    lv = _levels(res)
    assert_true("extended is a SIZING flag, not a warning", lv.get("extended") == "size", str(lv))
    assert_true("wide stop is a SIZING flag", lv.get("wide_stop") == "size", str(lv))
    if "index_below_ma50" in lv:
        assert_true("index filter is INFO (measured: no per-trade penalty)",
                    lv["index_below_ma50"] == "info", str(lv))
    assert_true("every check carries a level",
                all(c.get("level") in ("warn", "size", "info") for c in res["checks"]), str(lv))
    assert_true("counts split by level",
                res["warn_count"] == sum(1 for v in lv.values() if v == "warn")
                and res["size_count"] == sum(1 for v in lv.values() if v == "size"), str(res)[:200])


def test_tier1_book_awareness(pf):
    df = _df()
    px = float(df["Close"].iloc[-1])
    held = [{"id": "aaa", "symbol": "TESTCO", "qty": 10, "entry_price": px * 0.9,
             "entry_date": "2026-08-01"}]
    _patch(pf, df, held)
    res = pf.preflight_position("TESTCO", 500, px, df.index[-1].date().isoformat())
    ids = _levels(res)
    assert_true("#1 flags an existing holding", "already_held" in ids, str(ids))
    assert_true("#1 reports the new average", res["held"]["new_qty"] == 510 and res["held"]["new_avg"] > 0,
                str(res.get("held")))
    assert_true("#2 flags position size vs book", ids.get("position_size") == "warn", str(ids))
    assert_true("#2 reports % of book after", res["book"]["pct_after"] > 50, str(res.get("book")))
    assert_true("#3 capital falls back to the invested book",
                res["capital_basis"].startswith("your invested book"), str(res.get("capital_basis")))
    assert_true("#3 suggests a share count for the target risk",
                (res.get("risk") or {}).get("qty_for_target", 0) > 0, str(res.get("risk")))
    # Re-checking an existing lot must not count that lot against itself.
    res2 = pf.preflight_position("TESTCO", 10, px * 0.9, df.index[-1].date().isoformat(),
                                 exclude_position_ids={"aaa"})
    assert_true("#1 re-check excludes the position being judged",
                "already_held" not in _levels(res2), str(_levels(res2)))


def test_tier2_entry_and_mechanics(pf):
    df = _df(crash_at=-12)
    px = float(df["Close"].iloc[-1])
    _patch(pf, df, [])
    res = pf.preflight_position("TESTCO", 10, px * 1.08, df.index[-1].date().isoformat())
    ids = _levels(res)
    assert_true("#5 entry far from the last close is flagged", ids.get("entry_vs_close") == "warn", str(ids))
    assert_true("#8 a recent circuit day is flagged as a warning", ids.get("circuit") == "warn", str(ids))
    assert_true("#7 liquidity is reported", (res.get("liquidity") or {}).get("adtv_cr") is not None)
    assert_true("#9 the SL-L order is previewed",
                (res.get("sl_preview") or {}).get("trigger") is not None, str(res.get("sl_preview")))
    assert_true("#9 preview is JSON-safe (no numpy scalars)",
                all(isinstance(v, (int, float, str)) for v in (res.get("sl_preview") or {}).values()),
                str(res.get("sl_preview")))


def test_empty_book_asks_for_capital(pf):
    df = _df()
    _patch(pf, df, [])
    res = pf.preflight_position("TESTCO", 10, float(df["Close"].iloc[-1]),
                                df.index[-1].date().isoformat())
    assert_true("no book and no capital → asks for capital instead of inventing one",
                res.get("capital") is None and "no_capital" in _levels(res), str(res.get("capital")))


def test_ledger_and_scorecard(tmpdir):
    import importlib
    import preflight_log as pl
    importlib.reload(pl)
    pl.DB_PATH = Path(tmpdir) / "preflight.db"
    pl._forward_return = lambda sym, d, p: {"ret_pct": 10.0 if sym == "WINNER" else -10.0,
                                            "bars": 5, "last_close": 100.0}
    base = {"entry_date": "2026-09-18", "basis_date": "2026-09-17", "qty": 10,
            "entry_price": 100, "warn_count": 1,
            "checks": [{"id": "fading", "level": "warn", "title": "Already fading"}]}
    assert_true("bad action is refused",
                pl.log_decision({**base, "symbol": "X", "action": "nonsense"}).get("logged") is False)
    pl.log_decision({**base, "symbol": "WINNER", "action": "added_anyway"})
    pl.log_decision({**base, "symbol": "LOSER", "action": "cancelled"})
    pl.log_decision({**base, "symbol": "CLEAN", "action": "added", "warn_count": 0, "checks": []})
    sc = pl.scorecard()
    row = next((c for c in sc["checks"] if c["check"] == "fading"), None)
    assert_true("ledger records every decision", sc["decisions"] == 3, str(sc["decisions"]))
    assert_true("overrides and cancels are counted separately",
                row and row["overridden"] == 1 and row["cancelled"] == 1, str(row))
    assert_true("override outcome is marked to market", row and row["avg_return_overridden"] == 10.0, str(row))
    assert_true("cancelled trades are followed too", row and row["avg_return_cancelled"] == -10.0, str(row))
    assert_true("small samples are called out", row and row["verdict"].startswith("too few"), str(row))
    assert_true("buys with no check firing are tracked separately",
                sc["clean_buys"]["n"] == 1, str(sc["clean_buys"]))


def test_recheck_is_idempotent(pf):
    df = _df()
    ed = df.index[-1].date().isoformat()
    store = {"positions": [{"id": "zzz", "symbol": "TESTCO", "qty": 10,
                            "entry_price": float(df["Close"].iloc[-1]),
                            "entry_date": ed, "entry_checks_basis": "2026-01-02"}]}
    saved = []
    _patch(pf, df, store["positions"])
    pf._load_store = lambda: store
    pf._save_store = lambda d: saved.append(d)
    pf.invalidate_summary_cache = lambda: None
    first = pf.recheck_pending_entry_checks()
    assert_true("#6 re-checks a position judged on older data", len(first) == 1, str(first))
    assert_true("#6 stores the entry-day result",
                store["positions"][0].get("entry_checks_final", {}).get("basis_date") == ed,
                str(store["positions"][0].get("entry_checks_final")))
    second = pf.recheck_pending_entry_checks()
    assert_true("#6 does not re-run once stored", second == [], str(second))


def main():
    print("== Pre-trade check tests ==")
    try:
        import portfolio as pf
    except Exception as e:
        pf = None
        print(f"  – portfolio checks skipped (portfolio.py unavailable: {e})")
    if pf is not None:
        import copy
        orig = {k: getattr(pf, k) for k in ("_build_history", "_load_store", "_sector_of",
                                            "settings", "_save_store", "invalidate_summary_cache")}
        try:
            for t in (test_levels_match_what_was_measured, test_tier1_book_awareness,
                      test_tier2_entry_and_mechanics, test_empty_book_asks_for_capital,
                      test_recheck_is_idempotent):
                _patch(pf, _df(), [])
                t(pf)
        finally:
            for k, v in orig.items():
                setattr(pf, k, v)
    with tempfile.TemporaryDirectory() as tmp:
        test_ledger_and_scorecard(tmp)
    if _failures:
        print(f"\n{len(_failures)} TEST(S) FAILED:")
        for f in _failures:
            print(f"  - {f}")
        sys.exit(1)
    print("\nAll pre-trade check tests passed.")
    sys.exit(0)


if __name__ == "__main__":
    main()
