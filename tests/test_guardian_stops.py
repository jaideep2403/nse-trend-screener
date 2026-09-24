"""
Guardian timeliness / near-stop / history + next-session stop-order invariants.

Added 2026-09-16 after AEROFLEX, VINCOFE, CAPLIPOINT and APARINDS hit EXIT on the
2026-09-15 close with no earlier warning, and the alert only appeared at 09:15 the
next morning.

Run via:  python3 tests/test_guardian_stops.py
Exit code 0 = pass, 1 = any failure.

The portfolio checks SKIP (not fail) when portfolio.py is absent — it is gitignored,
owner-only, and does not exist on the public repo / demo box.
"""
from __future__ import annotations

import os
import sqlite3
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))
# Never start the live scheduler / bhavcopy poller / screener.in scraper from a test.
os.environ.setdefault("ASCENT_BACKGROUND_JOBS", "0")

_failures: list[str] = []


def _ok(name):
    print(f"  ✓ {name}")


def _fail(name, msg):
    print(f"  ✗ {name}: {msg}")
    _failures.append(f"{name}: {msg}")


def assert_true(name, cond, msg=""):
    (_ok(name) if cond else _fail(name, msg or "condition was False"))


def _walk(n=120, seed=0, drift=0.0005, vol=0.02, start=500.0):
    rng = np.random.default_rng(seed)
    close = start * np.exp(np.cumsum(rng.normal(drift, vol, n)))
    spread = np.abs(rng.normal(0, vol, n)) * close
    high = close + spread * rng.uniform(0.2, 1.0, n)
    low = close - spread * rng.uniform(0.2, 1.0, n)
    idx = pd.bdate_range("2026-01-01", periods=n)
    return pd.DataFrame({"Open": close, "High": high, "Low": low, "Close": close,
                         "Volume": rng.integers(1e5, 1e6, n).astype(float)}, index=idx)


# ── portfolio: the level shown tonight IS the level tested tomorrow ───────────

def test_next_session_trigger_matches_tomorrows_test(pf):
    for seed in range(8):
        df = _walk(seed=seed)
        for cut in (40, 77, 119):
            tonight = pf._next_session_trigger(df.iloc[:cut])["trigger_next"]
            tested_tomorrow = pf._chandelier_stop(df.iloc[:cut + 1])
            if abs(tonight - round(tested_tomorrow, 2)) > 0.011:
                _fail("next_session_trigger == tomorrow's tested level",
                      f"seed {seed} cut {cut}: {tonight} vs {tested_tomorrow}")
                return
    _ok("next_session_trigger == tomorrow's tested level (8 walks × 3 cuts)")


def test_exit_fires_only_on_chandelier(pf):
    """The code comment claims the swing-low SL can never fire an exit by itself
    (clamped ≥4% under the close), so the Chandelier is the only real trigger —
    which is why the broker order uses it. Prove it on many paths and entries."""
    bad = 0
    for seed in range(20):
        df = _walk(seed=seed, drift=-0.002 if seed % 2 else 0.002, vol=0.03)
        for cut in range(30, 120, 7):
            d = df.iloc[:cut]
            close = float(d["Close"].iloc[-1])
            for entry in (close * 0.8, close, close * 1.2):
                sl = pf._suggest_stop_loss(entry, d)["primary"]
                ch = pf._chandelier_stop(d)
                if ch is None:
                    continue
                if (close < sl) != (close < ch):
                    bad += 1
    assert_true("exit trigger is exactly close < Chandelier", bad == 0, f"{bad} mismatches")


def test_stop_order_shape(pf):
    nt = {"as_of": "2026-09-16", "close": 100.0, "trigger_next": 97.33,
          "trigger_today": 96.0, "cushion_pct": 2.74, "cushion_ranges": 0.9, "near_stop": True}
    so = pf._stop_order(nt, 10)
    o = so["order"]
    assert_true("order trigger floored to tick", o and abs(o["trigger"] - 97.30) < 1e-9, str(o))
    assert_true("limit sits under trigger", o and o["limit"] < o["trigger"], str(o))
    assert_true("raised change detected", so["change_dir"] == "raised", so["change_dir"])
    breached = pf._stop_order({**nt, "close": 95.0, "trigger_next": 97.33}, 10)
    assert_true("no resting sell stop above market", breached["order"] is None and breached["note"])


def test_near_stop_flag_uses_daily_ranges(pf):
    df = _walk(seed=3)
    nt = pf._next_session_trigger(df)
    expect = 0 <= nt["cushion_ranges"] < pf.NEAR_STOP_RANGES
    assert_true("near_stop == cushion_ranges < NEAR_STOP_RANGES", nt["near_stop"] == expect)


# ── guardian: history, migration, post-bhavcopy retry ─────────────────────────

def _fresh_guardian(tmpdir):
    import importlib
    import guardian
    importlib.reload(guardian)
    guardian._DB_PATH = Path(tmpdir) / "guardian.db"
    return guardian


def test_history_records_and_lag(tmpdir):
    g = _fresh_guardian(tmpdir)
    conn = g._connect()
    landed = 1_000_000
    g._record_history(conn, "2026-09-11", "ABC", "position", "ok", "ok", 100, 90, 92, 1.5, landed, landed + 60)
    g._record_history(conn, "2026-09-15", "ABC", "position", "watch", "near stop", 95, 90, 93, 0.6, landed, landed + 600)
    g._record_history(conn, "2026-09-15", "ABC", "position", "watch", "near stop", 95, 90, 93, 0.6, landed, landed + 900)
    g._record_history(conn, "2026-09-15", "ABC", "position", "exit", "stop broken", 89, 90, 93, -0.2, landed, landed + 1200)
    g._record_history(conn, "2026-09-16", "ABC", "position", "ok", "ok", 99, 90, 94, 1.2, landed, landed + 60)
    conn.commit(); conn.close()
    rows = g.get_history(days=30)
    by = {r["session"]: r for r in rows}
    assert_true("history hides steady-ok sessions", "2026-09-11" not in by, str(list(by)))
    assert_true("history keeps one row per session", sum(r["session"] == "2026-09-15" for r in rows) == 1)
    r15 = by.get("2026-09-15", {})
    assert_true("latest severity kept for the session", r15.get("severity") == "exit", str(r15))
    assert_true("severity change moves shown_at", r15.get("shown_at") == landed + 1200, str(r15))
    assert_true("prev severity is the prior session's", r15.get("prev_severity") == "ok", str(r15))
    assert_true("lag measured from bhavcopy landing", r15.get("lag_min") == 20, str(r15))
    assert_true("return to ok is recorded as a change", by.get("2026-09-16", {}).get("changed") is True)
    all_rows = g.get_history(days=30, include_ok=True)
    assert_true("include_ok returns every session", len(all_rows) == 3, str(len(all_rows)))


def test_state_table_migrates_in_place(tmpdir):
    g = _fresh_guardian(tmpdir)
    old = sqlite3.connect(str(g._DB_PATH))
    old.execute("CREATE TABLE guardian_state (symbol TEXT PRIMARY KEY, kind TEXT, bhav_date TEXT, "
                "severity TEXT, reason TEXT, price REAL, stop REAL, entry REAL, entry_window TEXT, "
                "since_date TEXT, dismissed_on TEXT, updated_at INTEGER)")
    old.execute("INSERT INTO guardian_state (symbol, severity, dismissed_on) VALUES ('X','exit',NULL)")
    old.commit(); old.close()
    alerts = g.get_active_alerts()
    cols = {r[1] for r in sqlite3.connect(str(g._DB_PATH)).execute("PRAGMA table_info(guardian_state)")}
    assert_true("old guardian.db gains new columns", {"trigger_next", "cushion_ranges", "data_date"} <= cols)
    assert_true("existing alerts survive migration", len(alerts) == 1 and alerts[0]["symbol"] == "X")


def test_post_bhavcopy_sweep_waits_for_new_data(tmpdir):
    g = _fresh_guardian(tmpdir)
    seq = iter([{"alerts": [], "data_date": "2026-09-15"},
                {"alerts": [], "data_date": "2026-09-15"},
                {"alerts": [{"symbol": "ABC"}], "data_date": "2026-09-16"}])
    calls = []
    g.run_sweep = lambda: calls.append(1) or next(seq)
    from datetime import date
    res = g.run_sweep_for_bhavcopy(date(2026, 9, 16), attempts=5, wait_sec=0)
    assert_true("retries until data reaches the new session", len(calls) == 3, str(len(calls)))
    assert_true("returns the sweep on the new data", res.get("data_date") == "2026-09-16")


# ── 2026-09-17 audit fixes #1-#8 ──────────────────────────────────────────────

def test_1_dismissal_rearms_next_session(tmpdir):
    g = _fresh_guardian(tmpdir)
    f = g.dismissal_after_sweep
    assert_true("#1 dismissed EXIT re-arms on a newer session",
                f("exit", "exit", "2026-09-11", "2026-09-15") is None)
    assert_true("#1 dismissal holds within the same session",
                f("exit", "exit", "2026-09-15", "2026-09-15") == "2026-09-15")
    assert_true("#1 escalation re-arms", f("watch", "exit", "2026-09-15", "2026-09-15") is None)
    assert_true("#1 back to ok clears the dismissal", f("exit", "ok", "2026-09-15", "2026-09-15") is None)


def _with_fake_modules(fakes: dict, fn):
    saved = {k: sys.modules.get(k) for k in fakes}
    sys.modules.update(fakes)
    try:
        return fn()
    finally:
        for k, v in saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v


def test_3_load_failure_never_erases_alerts(tmpdir):
    import types
    g = _fresh_guardian(tmpdir)
    pf = types.ModuleType("portfolio")
    def _boom():
        raise RuntimeError("analysis crashed")
    pf.list_positions = _boom
    pf._load_store = lambda: {"positions": [{"symbol": "HELD", "entry_price": 100.0}]}
    pf._build_history = lambda sym, days=250: None
    wl = types.ModuleType("watchlist"); wl.get_symbols = lambda: ["WATCHED"]
    bm = types.ModuleType("benchmark"); bm.get_benchmark = lambda days=420: None
    conn = g._connect()
    conn.execute("INSERT INTO guardian_state (symbol, kind, bhav_date, severity, reason) "
                 "VALUES ('HELD','position','2026-09-15','exit','stop broken')")
    conn.execute("INSERT INTO guardian_state (symbol, kind, bhav_date, severity, reason) "
                 "VALUES ('OTHER','position','2026-09-15','trim','weakening')")
    conn.commit(); conn.close()
    res = _with_fake_modules({"portfolio": pf, "watchlist": wl, "benchmark": bm}, g.run_sweep)
    alerts = {a["symbol"]: a for a in g.get_active_alerts()}
    assert_true("#3 sweep reports the failure", bool(res.get("error")), str(res))
    assert_true("#3 held EXIT survives a load failure (not downgraded)",
                alerts.get("HELD", {}).get("severity") == "exit", str(alerts.get("HELD")))
    assert_true("#3 its reason says it was NOT re-checked",
                "NOT RE-CHECKED" in alerts.get("HELD", {}).get("reason", ""))
    assert_true("#3 other holdings' alerts are not deleted", "OTHER" in alerts, str(list(alerts)))
    assert_true("#3 status reads error, not all-clear", g.get_status().get("state") == "error")


def test_3_never_run_is_not_all_clear(tmpdir):
    g = _fresh_guardian(tmpdir)
    assert_true("#3 no sweep yet -> state 'never'", g.get_status().get("state") == "never")


def test_2_and_6_app_wiring():
    import inspect
    try:
        import io, contextlib
        with contextlib.redirect_stdout(io.StringIO()):
            import app as A
    except Exception as e:
        print(f"  – app wiring checks skipped (app import failed: {e})")
        return
    import guardian
    calls = []
    orig = guardian.run_sweep_for_bhavcopy
    guardian.run_sweep_for_bhavcopy = lambda d: calls.append(d)
    try:
        t = A._start_guardian_after_bhavcopy("2026-09-17")
        if t is not None:
            t.join(5)
    finally:
        guardian.run_sweep_for_bhavcopy = orig
    assert_true("#2 bhavcopy hook starts the guardian sweep", calls == ["2026-09-17"], str(calls))
    src = inspect.getsource(A._bhavcopy_scheduler)
    assert_true("#2 scheduler calls the hook on a new bhavcopy", "_start_guardian_after_bhavcopy(" in src)
    assert_true("#5 scheduler refreshes the split feed before prewarm",
                src.index("_refresh_split_feed(") < src.index("_prewarm_all_scans"))
    assert_true("#5 boot refreshes the split feed before prewarm",
                "_refresh_split_feed(" in inspect.getsource(A._boot_prewarm))
    orig_status = A.fund_scheduler_status
    A.fund_scheduler_status = lambda: {"running": False, "error": "scrape disabled (SCRAPE_OFF)"}
    try:
        c = A.app.test_client()
        with c.session_transaction() as sess:
            sess["user"], sess["role"] = "jai", "admin"
        msg = c.post("/api/fundamentals/refresh").get_json() or {}
    finally:
        A.fund_scheduler_status = orig_status
    assert_true("#6 disabled scraper is reported as NOT running",
                "NOT running" in msg.get("message", "") and msg.get("status") == "not_running", str(msg))
    body = (_REPO / "app.py").read_text()
    assert_true("#6 scraper start is gated on background jobs",
                "if _BG_JOBS:\n    start_background_scheduler()" in body)


def _weekly_df(n_up=150, n_down=5, start="2026-01-05"):
    idx = pd.bdate_range(start, periods=n_up + n_down)        # starts Monday
    up = np.linspace(100, 250, n_up)
    down = np.linspace(230, 120, n_down)
    c = np.r_[up, down]
    return pd.DataFrame({"Open": c, "High": c * 1.01, "Low": c * 0.99, "Close": c,
                         "Volume": 1e6}, index=idx)


def test_4_weekly_trend_break():
    import exits
    df = _weekly_df()
    assert_true("#4 fixture ends on a Friday", df.index[-1].weekday() == 4, str(df.index[-1]))
    full = exits.evaluate_exit(df, entry_price=200.0, stop_price=1.0)["signals"]["ma_break"]
    assert_true("#4 weekly close below 20-week MA triggers", full["triggered"] and full["basis"] == "weekly_20wk", str(full))
    wed = exits.evaluate_exit(df.iloc[:-2], entry_price=200.0, stop_price=1.0)["signals"]["ma_break"]
    assert_true("#4 a mid-week dip does not trigger before the week closes",
                wed["ma100_break"] and not wed["triggered"], str(wed))
    young = exits.evaluate_exit(_weekly_df(n_up=55, n_down=5), entry_price=200.0, stop_price=1.0)["signals"]["ma_break"]
    assert_true("#4 young holding falls back to daily 50-DMA", young["basis"] == "daily_ma50_fallback", str(young))


def test_5_split_feed(tmpdir):
    import json, importlib
    import corporate_actions as ca
    importlib.reload(ca)
    ca.CACHE_PATH = str(Path(tmpdir) / "ca.json")
    Path(ca.CACHE_PATH).write_text(json.dumps({
        "built_at": "2026-07-29T13:18:06", "from_year": 2019, "to_year": 2026,
        "events": {"OLDCO": [{"ex_date": "2025-01-01", "mult": 0.5, "kind": "split", "subject": "x"}]}}))
    ca._mem.update(data=None, ts=0.0)
    recs = [{"symbol": "TDPOWERSYS", "exDate": "24-Aug-2026",
             "subject": "Face Value Split (Sub-Division) - From Rs 2/- Per Share To Re 1/- Per Share"}]
    calls = []
    ca.fetch_range = lambda a, b, session=None, log=print: (calls.append((a, b)) or recs)
    r1 = ca.refresh_recent(log=lambda m: None)
    r2 = ca.refresh_recent(log=lambda m: None)
    fetches_after_second = len(calls)
    r3 = ca.refresh_recent(log=lambda m: None, force=True)
    assert_true("#5 refresh adds the missed split", r1.get("added") == 1, str(r1))
    assert_true("#5 refresh is at most daily (no second network fetch)",
                r2.get("skipped") is True and fetches_after_second == 1, f"{r2} fetches={fetches_after_second}")
    assert_true("#5 forced refresh does not duplicate events", r3.get("added") == 0, str(r3))
    assert_true("#5 old events are kept", bool(ca.events_for("OLDCO")))
    assert_true("#5 '.NS' tickers match feed symbols", len(ca.events_for("TDPOWERSYS.NS")) == 1)


def test_7_history_cache_respects_length(pf):
    files = sorted(pf.BHAV_DIR.glob("*.pkl")) if pf.BHAV_DIR.exists() else []
    if len(files) < 300:
        print("  – #7 skipped (needs >=300 bhavcopy files on disk)")
        return
    pf._history_cache.clear()
    long_first = pf._build_history("RELIANCE", days=400)
    short_after = pf._build_history("RELIANCE", days=250)
    pf._history_cache.clear()
    short_first = pf._build_history("RELIANCE", days=250)
    long_after = pf._build_history("RELIANCE", days=400)
    ok = all(x is not None for x in (long_first, short_after, short_first, long_after))
    assert_true("#7 history length does not depend on call order",
                ok and len(long_first) == len(long_after) and len(short_first) == len(short_after)
                and len(long_first) > len(short_first),
                f"{[None if x is None else len(x) for x in (long_first, short_after, short_first, long_after)]}")


def test_8_init_stop_uses_entry_date(pf):
    df = _walk(n=120, seed=0, drift=0.001, vol=0.03)
    entry_day = df.index[80]
    entry = round(float(df["Close"].iloc[80]) / 1.04, 2)     # closed 4% above the buy
    crashed = df.copy()
    crashed.loc[crashed.index > entry_day, ["Open", "High", "Low", "Close"]] *= 0.5
    orig = pf._build_history
    pf._build_history = lambda sym, days=250: crashed
    try:
        got = pf.init_stop_at_entry("X", entry, entry_day.date().isoformat())
    finally:
        pf._build_history = orig
    want = pf._suggest_stop_loss(entry, df.loc[:entry_day], breakeven_floor=False)["primary"]
    assert_true("#8 init stop ignores bars after the entry date", got is not None and abs(got - round(want, 2)) < 0.011,
                f"{got} vs {want}")
    with_floor = pf._suggest_stop_loss(entry, df.loc[:entry_day])["primary"]
    assert_true("#8 breakeven floor would have set stop == entry (the bug)", abs(with_floor - entry) < 0.006,
                f"{with_floor} vs {entry}")
    assert_true("#8 initial stop sits below entry", got is not None and got < entry, str(got))


def main():
    print("== Guardian / stop-order tests ==")
    try:
        import portfolio as pf
    except Exception as e:           # gitignored, owner-only module
        pf = None
        print(f"  – portfolio checks skipped (portfolio.py unavailable: {e})")
    if pf is not None:
        test_next_session_trigger_matches_tomorrows_test(pf)
        test_exit_fires_only_on_chandelier(pf)
        test_stop_order_shape(pf)
        test_near_stop_flag_uses_daily_ranges(pf)
        test_7_history_cache_respects_length(pf)
        test_8_init_stop_uses_entry_date(pf)
    test_4_weekly_trend_break()
    test_2_and_6_app_wiring()
    for t in (test_history_records_and_lag, test_state_table_migrates_in_place,
              test_post_bhavcopy_sweep_waits_for_new_data, test_1_dismissal_rearms_next_session,
              test_3_load_failure_never_erases_alerts, test_3_never_run_is_not_all_clear,
              test_5_split_feed):
        with tempfile.TemporaryDirectory() as tmp:
            t(tmp)
    if _failures:
        print(f"\n{len(_failures)} TEST(S) FAILED:")
        for f in _failures:
            print(f"  - {f}")
        sys.exit(1)
    print("\nAll guardian / stop-order tests passed.")
    sys.exit(0)


if __name__ == "__main__":
    main()
