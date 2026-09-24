"""
Position Guardian — the app chases YOU, not the other way round.

WHY (the GVT&D lesson): a held position flipped to 🔴 weakening on 2026-06-24
at ₹5,052 and the exit engine said EXIT — but that intelligence only rendered
if the user happened to open the Portfolio tab. The stock bottomed 13% lower.
The Guardian closes that gap: every scan (and a post-bhavcopy pass) sweeps
all portfolio holdings + watchlist symbols, persists their alert state, and
the global header strip — visible on EVERY tab — surfaces anything that fired.

2026-09-16 — THREE GAPS CLOSED (AEROFLEX / VINCOFE / CAPLIPOINT / APARINDS all
hit EXIT on 2026-09-15; the alert appeared at 09:15 the next morning):
  1. TIMELINESS. The sweep was hooked only to a MANUAL trending scan; the
     bhavcopy prewarm calls run_trending_scan directly, so a new close never
     triggered a sweep. `run_sweep_for_bhavcopy()` now runs the moment a
     bhavcopy lands (app._bhavcopy_scheduler) and retries until every held
     symbol's data is actually on the new session.
  2. NEAR STOP. Positions whose next-session exit trigger is within one
     average daily range now raise WATCH (measured — see
     portfolio.NEAR_STOP_RANGES). Before, severity stayed 'ok' until the stop
     was already broken.
  3. HISTORY. `guardian_state` holds only the CURRENT state and `since_date`
     is overwritten on every change, so "was I warned, and when?" could not be
     answered. `guardian_history` keeps one row per (session, symbol) with the
     wall-clock time it was first shown and when that session's bhavcopy landed.

DESIGN
  • run_sweep() is the only heavy call (loads price history). It runs after
    each trending scan, after each new bhavcopy, and at startup — never from
    /api/header.
  • get_active_alerts() / get_history() are pure sqlite reads — safe for polls.
  • Severity ladder: exit > trim > watch. Portfolio symbols use the full
    ladder (exits.evaluate_exit + weakening flip + near stop); watchlist symbols
    cap at 'watch' (you can't exit what you don't hold).

SCHEMA (guardian.db)
    guardian_state(symbol PK, kind, bhav_date, severity, reason, price, stop,
                   entry, entry_window, since_date, dismissed_on, updated_at,
                   trigger_next, cushion_ranges, data_date)
    guardian_history(bhav_date, symbol, kind, severity, prev_severity, changed,
                     reason, price, stop, trigger_next, cushion_ranges,
                     bhav_landed_at, first_seen_at, sev_since_at, last_seen_at,
                     UNIQUE(bhav_date, symbol))
"""
from __future__ import annotations

import os
import sqlite3
import threading
import time
from pathlib import Path

_DB_PATH = Path(os.getenv("DATA_DIR", os.path.dirname(__file__) or ".")) / "guardian.db"
_lock = threading.Lock()

_SEV_RANK = {"ok": 0, "watch": 1, "trim": 2, "exit": 3}

# Columns added after the table first shipped — migrated in place on connect.
_STATE_MIGRATIONS = (("trigger_next", "REAL"), ("cushion_ranges", "REAL"), ("data_date", "TEXT"))


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(str(_DB_PATH), timeout=10)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS guardian_state (
            symbol       TEXT PRIMARY KEY,
            kind         TEXT,
            bhav_date    TEXT,
            severity     TEXT,
            reason       TEXT,
            price        REAL,
            stop         REAL,
            entry        REAL,
            entry_window TEXT,
            since_date   TEXT,
            dismissed_on TEXT,
            updated_at   INTEGER
        )
    """)
    have = {r[1] for r in conn.execute("PRAGMA table_info(guardian_state)")}
    for col, typ in _STATE_MIGRATIONS:
        if col not in have:
            conn.execute(f"ALTER TABLE guardian_state ADD COLUMN {col} {typ}")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS guardian_history (
            bhav_date      TEXT NOT NULL,
            symbol         TEXT NOT NULL,
            kind           TEXT,
            severity       TEXT,
            prev_severity  TEXT,
            changed        INTEGER,
            reason         TEXT,
            price          REAL,
            stop           REAL,
            trigger_next   REAL,
            cushion_ranges REAL,
            bhav_landed_at INTEGER,
            first_seen_at  INTEGER,
            sev_since_at   INTEGER,
            last_seen_at   INTEGER,
            UNIQUE (bhav_date, symbol)
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS ix_gh_date ON guardian_history (bhav_date)")
    # Last sweep outcome (2026-09-17). Without it a sweep that failed or never ran was
    # indistinguishable from "no alerts" — the banner simply showed nothing.
    conn.execute("CREATE TABLE IF NOT EXISTS guardian_meta (key TEXT PRIMARY KEY, value TEXT)")
    conn.commit()
    return conn


# ── Sweep (heavy — call from scan completion / scheduler only) ────────────────

def run_sweep() -> dict:
    """Evaluate every portfolio holding + watchlist symbol; persist states.
    Returns {checked, alerts, bhav_date, data_date}. Never raises — returns error dict."""
    try:
        return _run_sweep_inner()
    except Exception as e:
        res = {"error": f"{type(e).__name__}: {e}", "checked": 0, "alerts": []}
        _write_meta(res)
        return res


def run_sweep_for_bhavcopy(expected_date, attempts: int = 20, wait_sec: int = 90) -> dict:
    """Sweep for a bhavcopy that has JUST landed, retrying until the data is on it.

    Right after a download the shared stock snapshot can still be mid-reload (it
    deliberately serves the last COMPLETE snapshot rather than a partial one), so a
    single immediate sweep could stamp yesterday's prices with today's date. Each
    attempt checks the oldest last-bar date across held symbols; only a sweep whose
    data has reached `expected_date` counts.
    """
    exp = expected_date.isoformat() if hasattr(expected_date, "isoformat") else str(expected_date)
    res: dict = {}
    for i in range(1, attempts + 1):
        res = run_sweep()
        # Judge by the NEWEST bar across held symbols: a stale snapshot has every symbol
        # on old data, while one suspended stock must not stall the retry for 30 minutes.
        dd = res.get("data_date_max", res.get("data_date"))
        if not res.get("error") and (dd is None or dd >= exp):
            print(f"[guardian] post-bhavcopy sweep for {exp}: {len(res.get('alerts', []))} "
                  f"alert(s), data as of {dd} (attempt {i})", flush=True)
            return res
        print(f"[guardian] post-bhavcopy sweep for {exp}: data still at {dd} "
              f"{'(' + res['error'] + ')' if res.get('error') else ''} — retry {i}/{attempts}",
              flush=True)
        time.sleep(wait_sec)
    print(f"[guardian] post-bhavcopy sweep for {exp} gave up; last data_date "
          f"{res.get('data_date')}", flush=True)
    return res


def dismissal_after_sweep(prev_severity: str, severity: str,
                          dismissed_on: str | None, bhav_date: str | None) -> str | None:
    """Whether a user's dismissal still hides the alert after this sweep.

    Re-arms (returns None) on escalation, on a return to 'ok', and — since 2026-09-17 —
    on ANY newer session's data. Before, escalation was the only path back, and EXIT is
    the top rung: GVT&D's EXIT dismissed on 09-11 stayed hidden while the position fell
    a further 6.6% on 09-15. A dismissal now means "hide for today's data", no more."""
    if not dismissed_on or severity == "ok":
        return None
    if _SEV_RANK.get(severity, 0) > _SEV_RANK.get(prev_severity or "ok", 0):
        return None
    if bhav_date and dismissed_on < bhav_date:
        return None
    return dismissed_on


def _bhav_landed_at(bhav_date: str | None) -> int | None:
    try:
        from datetime import date as _d
        from data_fetcher import _bhav_cache_path
        return int(_bhav_cache_path(_d.fromisoformat(bhav_date)).stat().st_mtime)
    except Exception:
        return None


def _fresh_frame(sym: str, stocks: dict, bhav_date: str | None):
    """The symbol's history, preferring the shared snapshot but never a stale one."""
    df = stocks.get(sym)
    try:
        if df is not None and bhav_date and df.index[-1].date().isoformat() >= bhav_date:
            return df
    except Exception:
        pass
    try:
        import portfolio
        alt = portfolio._build_history(sym)
        if alt is not None and len(alt) >= 20:
            return alt
    except Exception:
        pass
    return df


def _run_sweep_inner() -> dict:
    from trending import _clean_df, _score_stock

    try:
        from data_fetcher import _latest_bhavcopy_date
        bd = _latest_bhavcopy_date()
        bhav_date = bd.isoformat() if bd else None
    except Exception:
        bhav_date = None

    # ── Collect targets: portfolio holdings (full analysis incl. the SAME
    # evaluate_exit the Portfolio tab renders) + watchlist symbols. ──
    targets: dict[str, dict] = {}
    errors: list[str] = []
    # Deleting state for "symbols no longer held" is only safe when BOTH sources
    # actually loaded. Before 2026-09-17 a failure here was swallowed, the sweep went
    # on with the watchlist alone, and the DELETE below wiped every held position's
    # EXIT/TRIM — a failure that rendered as an all-clear banner.
    sources_ok = True
    try:
        import portfolio
    except ModuleNotFoundError as e:
        portfolio = None
        if getattr(e, "name", "") != "portfolio":     # owner-only module absent = expected
            errors.append(f"portfolio import: {e}")
            sources_ok = False
    if portfolio is not None:
        try:
            # Entry checks for buys recorded before their own session printed (2026-09-18).
            portfolio.recheck_pending_entry_checks()
        except Exception as e:
            errors.append(f"entry recheck: {type(e).__name__}: {e}")
        try:
            for p in portfolio.list_positions():
                sym = (p.get("symbol") or "").upper()
                if not sym:
                    continue
                ex = p.get("exit") or {}
                so = p.get("stop_order") or {}
                targets[sym] = {
                    "kind":        "position",
                    "entry":       p.get("entry_price"),
                    "sl":          p.get("sl"),
                    "exit_action": (ex.get("action") or "HOLD").upper(),
                    "exit_reason": ex.get("reason") or "",
                    "stop_order":  so,
                    # A row that could not be analysed must never read as 'ok'.
                    "eval_error":  p.get("error") or (None if p.get("exit") else "exit engine returned nothing"),
                }
        except Exception as e:
            errors.append(f"positions: {type(e).__name__}: {e}")
            sources_ok = False
            # Fall back to the raw store so held symbols stay visible as NOT EVALUATED.
            try:
                for p in (portfolio._load_store().get("positions") or []):
                    sym = (p.get("symbol") or "").upper()
                    if sym and sym not in targets:
                        targets[sym] = {"kind": "position", "entry": p.get("entry_price"),
                                        "sl": None, "exit_action": None, "exit_reason": "",
                                        "stop_order": {}, "eval_error": f"positions failed to load: {e}"}
            except Exception:
                pass
    try:
        import watchlist
        for sym in watchlist.get_symbols():
            if sym not in targets:   # a held symbol outranks its watchlist entry
                targets[sym] = {"kind": "watchlist", "entry": None, "sl": None,
                                "exit_action": None, "exit_reason": "", "stop_order": {},
                                "eval_error": None}
    except Exception as e:
        errors.append(f"watchlist: {type(e).__name__}: {e}")
        sources_ok = False

    if not targets:
        res = {"checked": 0, "alerts": [], "bhav_date": bhav_date, "data_date": None,
               "error": "; ".join(errors) or None}
        _write_meta(res)
        return res

    # Read the shared snapshot only if it is ALREADY loaded. Right after a bhavcopy
    # lands it is empty, and calling _get_stocks() here would start a full-universe
    # reload on this thread in parallel with the prewarm's — the stampede that once
    # left My Portfolio spinning. _fresh_frame() falls back per symbol instead.
    try:
        from industry_groups import _stocks_cache
        stocks = (_stocks_cache.get("data") if _stocks_cache.get("complete") else None) or {}
    except Exception:
        stocks = {}
    try:
        from benchmark import get_benchmark
        nifty = get_benchmark(days=420)
    except Exception:
        nifty = None

    alerts = []
    data_dates = []
    stale_symbols: list[str] = []
    now = int(time.time())
    landed = _bhav_landed_at(bhav_date)
    with _lock:
        conn = _connect()
        try:
            for sym, t in targets.items():
                df = _fresh_frame(sym, stocks, bhav_date)
                ew, price, reasons, data_date = None, None, [], None
                if df is not None and len(df) >= 20:
                    try:
                        data_date = df.index[-1].date().isoformat()
                        data_dates.append(data_date)
                    except Exception:
                        pass
                    try:
                        m = _score_stock(_clean_df(df.copy()), nifty)
                    except Exception:
                        m = None
                    if m:
                        ew = m.get("entry_window")
                        price = m.get("price")
                        if ew == "weakening":
                            fade = m.get("retrace_10d")
                            reasons.append(
                                f"weakening — {fade:+.0f}% off 10d high" if fade is not None
                                else "weakening — strength fading")

                so = t.get("stop_order") or {}
                trigger_next = so.get("trigger_next")
                cushion_r = so.get("cushion_ranges")
                if df is None or len(df) < 20:
                    t["eval_error"] = t.get("eval_error") or "no price history"
                elif bhav_date and data_date and data_date < bhav_date:
                    stale_symbols.append(sym)

                prev = conn.execute(
                    "SELECT severity, since_date, dismissed_on, reason FROM guardian_state "
                    "WHERE symbol=?", (sym,)).fetchone()
                prev_sev = prev[0] if prev else "ok"

                # Severity ladder
                severity = "ok"
                if t["kind"] == "position" and t.get("eval_error") and t.get("exit_action") in (None, "HOLD"):
                    # Could not evaluate. Never let that read as 'ok' — and never
                    # DOWNGRADE a live EXIT/TRIM just because this sweep failed.
                    if prev_sev in ("exit", "trim"):
                        severity = prev_sev
                        reasons = [f"NOT RE-CHECKED — {t['eval_error']} · last known: {prev[3]}"]
                    else:
                        severity = "watch"
                        reasons.insert(0, f"NOT EVALUATED — {t['eval_error']}")
                elif t["kind"] == "position":
                    act = t.get("exit_action")
                    if act == "EXIT":
                        severity = "exit"
                        reasons.insert(0, t.get("exit_reason") or "exit signal")
                    elif act == "TRIM":
                        severity = "trim"
                        reasons.insert(0, t.get("exit_reason") or "trim signal")
                    elif ew == "weakening":
                        severity = "trim"   # held + fading = act, not just watch
                    elif so.get("near_stop"):
                        severity = "watch"
                        # The trigger level itself is rendered separately (strip, bell,
                        # history column), so the reason carries only the distance.
                        reasons.insert(0, f"near stop — {so.get('cushion_pct'):+.1f}% above the "
                                          f"exit trigger ({cushion_r:.2f}× a normal day's range)")
                elif ew == "weakening":
                    severity = "watch"      # watchlist cap

                reason = " · ".join(r for r in reasons if r) or "ok"

                since = (prev[1] if prev and prev_sev == severity and prev[1]
                         else bhav_date)
                dismissed_on = dismissal_after_sweep(
                    prev_sev, severity, prev[2] if prev else None, bhav_date)

                conn.execute(
                    "INSERT INTO guardian_state (symbol, kind, bhav_date, severity, "
                    " reason, price, stop, entry, entry_window, since_date, "
                    " dismissed_on, updated_at, trigger_next, cushion_ranges, data_date) "
                    "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?) "
                    "ON CONFLICT(symbol) DO UPDATE SET "
                    " kind=excluded.kind, bhav_date=excluded.bhav_date, "
                    " severity=excluded.severity, reason=excluded.reason, "
                    " price=excluded.price, stop=excluded.stop, entry=excluded.entry, "
                    " entry_window=excluded.entry_window, since_date=excluded.since_date, "
                    " dismissed_on=excluded.dismissed_on, updated_at=excluded.updated_at, "
                    " trigger_next=excluded.trigger_next, "
                    " cushion_ranges=excluded.cushion_ranges, data_date=excluded.data_date",
                    (sym, t["kind"], bhav_date, severity, reason, price,
                     t.get("sl"), t.get("entry"), ew, since, dismissed_on, now,
                     trigger_next, cushion_r, data_date))

                _record_history(conn, data_date or bhav_date, sym, t["kind"], severity,
                                reason, price, t.get("sl"), trigger_next, cushion_r,
                                landed if (data_date or bhav_date) == bhav_date else None,
                                now)

                if severity != "ok":
                    alerts.append({"symbol": sym, "severity": severity,
                                   "reason": reason, "kind": t["kind"]})

            # Symbols no longer held/watched drop out of the CURRENT-state table — but
            # only when both sources loaded, or a load failure would erase real alerts.
            # History is deliberately kept — that is the audit trail.
            if sources_ok:
                conn.execute(
                    "DELETE FROM guardian_state WHERE symbol NOT IN ({})".format(
                        ",".join("?" * len(targets))), tuple(targets.keys()))
            conn.commit()
        finally:
            conn.close()

    res = {"checked": len(targets), "alerts": alerts, "bhav_date": bhav_date,
           "data_date": min(data_dates) if data_dates else None,
           "data_date_max": max(data_dates) if data_dates else None,
           "stale_symbols": stale_symbols, "error": "; ".join(errors) or None}
    _write_meta(res)
    return res


def _write_meta(res: dict) -> None:
    """Persist the last sweep's outcome. Never raises."""
    try:
        import json as _json
        with _lock:
            conn = _connect()
            try:
                conn.execute("INSERT INTO guardian_meta (key, value) VALUES ('last_sweep', ?) "
                             "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                             (_json.dumps({
                                 "at": int(time.time()),
                                 "bhav_date": res.get("bhav_date"),
                                 "data_date": res.get("data_date"),
                                 "data_date_max": res.get("data_date_max"),
                                 "checked": res.get("checked", 0),
                                 "alerts": len(res.get("alerts") or []),
                                 "stale_symbols": res.get("stale_symbols") or [],
                                 "error": res.get("error"),
                             }),))
                conn.commit()
            finally:
                conn.close()
    except Exception:
        pass


def get_status() -> dict:
    """Is the alert banner trustworthy right now? Pure sqlite read + a few stat calls.

    state: ok | stale (a newer bhavcopy exists than the one last checked, or some held
    symbols were evaluated on old data) | error (the last sweep failed) | never."""
    import json as _json
    last = None
    try:
        with _lock:
            conn = _connect()
            try:
                row = conn.execute("SELECT value FROM guardian_meta WHERE key='last_sweep'").fetchone()
            finally:
                conn.close()
        last = _json.loads(row[0]) if row else None
    except Exception:
        last = None
    try:
        from data_fetcher import _latest_bhavcopy_date
        lb = _latest_bhavcopy_date()
        latest = lb.isoformat() if lb else None
    except Exception:
        latest = None
    if not last:
        return {"state": "never", "latest_bhav_date": latest}
    state = "ok"
    if last.get("error"):
        state = "error"
    elif (latest and (last.get("bhav_date") or "") < latest) or last.get("stale_symbols"):
        state = "stale"
    return {**last, "state": state, "latest_bhav_date": latest}


def _record_history(conn, session: str | None, sym, kind, severity, reason, price,
                    stop, trigger_next, cushion_r, landed, now) -> None:
    """One row per (session, symbol). first_seen_at is when this session's state was
    first written; sev_since_at moves only if the severity changes within the session."""
    if not session:
        return
    prev = conn.execute(
        "SELECT severity FROM guardian_history WHERE symbol=? AND bhav_date<? "
        "ORDER BY bhav_date DESC LIMIT 1", (sym, session)).fetchone()
    prev_sev = prev[0] if prev else None
    cur = conn.execute(
        "SELECT severity FROM guardian_history WHERE symbol=? AND bhav_date=?",
        (sym, session)).fetchone()
    changed = 1 if (prev_sev or "ok") != severity else 0
    if cur is None:
        conn.execute(
            "INSERT INTO guardian_history (bhav_date, symbol, kind, severity, prev_severity, "
            " changed, reason, price, stop, trigger_next, cushion_ranges, bhav_landed_at, "
            " first_seen_at, sev_since_at, last_seen_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (session, sym, kind, severity, prev_sev, changed, reason, price, stop,
             trigger_next, cushion_r, landed, now, now, now))
    else:
        conn.execute(
            "UPDATE guardian_history SET kind=?, severity=?, prev_severity=?, changed=?, "
            " reason=?, price=?, stop=?, trigger_next=?, cushion_ranges=?, "
            " bhav_landed_at=COALESCE(bhav_landed_at, ?), "
            " sev_since_at=CASE WHEN severity=? THEN sev_since_at ELSE ? END, "
            " last_seen_at=? WHERE symbol=? AND bhav_date=?",
            (kind, severity, prev_sev, changed, reason, price, stop, trigger_next,
             cushion_r, landed, severity, now, now, sym, session))


# ── Fast reads (safe for the /api/header poll) ────────────────────────────────

def get_active_alerts() -> list[dict]:
    """Current non-ok, non-dismissed states — a pure sqlite read."""
    with _lock:
        conn = _connect()
        try:
            rows = conn.execute(
                "SELECT symbol, kind, severity, reason, price, stop, entry, "
                "       entry_window, since_date, bhav_date, trigger_next, "
                "       cushion_ranges, data_date FROM guardian_state "
                "WHERE severity != 'ok' AND dismissed_on IS NULL "
                "ORDER BY CASE severity WHEN 'exit' THEN 0 WHEN 'trim' THEN 1 "
                "ELSE 2 END, symbol").fetchall()
        finally:
            conn.close()
    return [{
        "symbol": r[0], "kind": r[1], "severity": r[2], "reason": r[3],
        "price": r[4], "stop": r[5], "entry": r[6], "entry_window": r[7],
        "since": r[8], "bhav_date": r[9], "trigger_next": r[10],
        "cushion_ranges": r[11], "data_date": r[12],
    } for r in rows]


def get_state(symbol: str) -> dict | None:
    """Current stored state for one symbol (including 'ok' and dismissed rows).

    Used by the Add form's pre-trade checks so a buy into a holding the Guardian has
    already flagged says so (2026-09-18)."""
    sym = (symbol or "").upper().strip()
    if not sym:
        return None
    with _lock:
        conn = _connect()
        try:
            r = conn.execute(
                "SELECT symbol, kind, severity, reason, price, trigger_next, since_date, "
                "       dismissed_on, bhav_date FROM guardian_state WHERE symbol=?",
                (sym,)).fetchone()
        finally:
            conn.close()
    if not r:
        return None
    return {"symbol": r[0], "kind": r[1], "severity": r[2], "reason": r[3], "price": r[4],
            "trigger_next": r[5], "since": r[6], "dismissed_on": r[7], "bhav_date": r[8]}


def get_history(days: int = 30, symbol: str | None = None,
                include_ok: bool = False) -> list[dict]:
    """Alert history, newest session first. By default only rows that were an
    alert (non-ok) or a change back to ok — the rows worth reading."""
    days = max(1, min(int(days or 30), 400))
    sym = (symbol or "").upper().strip() or None
    with _lock:
        conn = _connect()
        try:
            sessions = [r[0] for r in conn.execute(
                "SELECT DISTINCT bhav_date FROM guardian_history "
                "ORDER BY bhav_date DESC LIMIT ?", (days,)).fetchall()]
            if not sessions:
                return []
            q = ("SELECT bhav_date, symbol, kind, severity, prev_severity, changed, reason, "
                 "       price, stop, trigger_next, cushion_ranges, bhav_landed_at, "
                 "       first_seen_at, sev_since_at FROM guardian_history "
                 "WHERE bhav_date >= ?")
            args: list = [sessions[-1]]
            if sym:
                q += " AND symbol = ?"
                args.append(sym)
            if not include_ok:
                q += " AND (severity != 'ok' OR changed = 1)"
            q += " ORDER BY bhav_date DESC, CASE severity WHEN 'exit' THEN 0 " \
                 "WHEN 'trim' THEN 1 WHEN 'watch' THEN 2 ELSE 3 END, symbol"
            rows = conn.execute(q, args).fetchall()
        finally:
            conn.close()
    out = []
    for r in rows:
        landed, shown = r[11], r[13]
        out.append({
            "session": r[0], "symbol": r[1], "kind": r[2], "severity": r[3],
            "prev_severity": r[4], "changed": bool(r[5]), "reason": r[6],
            "price": r[7], "stop": r[8], "trigger_next": r[9], "cushion_ranges": r[10],
            "bhav_landed_at": landed, "first_seen_at": r[12], "shown_at": shown,
            # Minutes from the session's bhavcopy landing to this severity first being
            # written. The 2026-09-15 EXITs would read ~800 here (landed 19:52, shown 09:15).
            "lag_min": round((shown - landed) / 60) if landed and shown else None,
        })
    return out


def dismiss(symbol: str) -> bool:
    """Silence the symbol's CURRENT alert episode (re-arms on escalation)."""
    sym = (symbol or "").upper().strip()
    if not sym:
        return False
    with _lock:
        conn = _connect()
        try:
            cur = conn.execute(
                "UPDATE guardian_state SET dismissed_on = bhav_date "
                "WHERE symbol = ? AND severity != 'ok'", (sym,))
            conn.commit()
            return cur.rowcount > 0
        finally:
            conn.close()
