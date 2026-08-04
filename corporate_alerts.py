"""Upcoming corporate actions on names you hold or are watching.

The authoritative NSE corporate-actions feed was built to fix the split adjuster —
602 split/bonus events across 459 symbols, already cached on disk and refreshed at
most daily. It was doing exactly one job. This surfaces the OTHER half for free: an
ex-date is a scheduled, known-in-advance event, and walking into one unaware is how
you end up reading a 1:2 split as a −50% crash on your own screen.

No new network calls — `corporate_actions.load()` reads the cache.

Scope note: the feed is filtered to splits and bonuses (the actions that REBASE the
price). Dividends, AGMs and buybacks are deliberately excluded upstream because they
do not rescale history — so this alerts on price-rebasing events only, and says so.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta

DEFAULT_HORIZON_DAYS = 45


def _d(s: str):
    try:
        return datetime.strptime(str(s)[:10], "%Y-%m-%d").date()
    except Exception:
        return None


def upcoming(symbols=None, horizon_days: int = DEFAULT_HORIZON_DAYS,
             today: date | None = None) -> list[dict]:
    """Price-rebasing corporate actions with an ex-date in the next `horizon_days`.

    `symbols` — restrict to these (e.g. your holdings). None = the whole feed.
    """
    try:
        import corporate_actions as ca
        events = ca.load()
    except Exception:
        return []
    today = today or date.today()
    horizon = today + timedelta(days=max(1, int(horizon_days)))
    want = {s.strip().upper() for s in symbols} if symbols else None

    out = []
    for sym, evs in (events or {}).items():
        if want is not None and sym not in want:
            continue
        for e in evs or []:
            ex = _d(e.get("ex_date"))
            if ex is None or not (today <= ex <= horizon):
                continue
            mult = e.get("mult")
            try:
                drop = (1.0 - float(mult)) * 100.0
            except Exception:
                drop = None
            out.append({
                "symbol": sym,
                "ex_date": str(ex),
                "days_away": (ex - today).days,
                "kind": e.get("kind"),
                "multiplier": mult,
                "expected_price_drop_pct": round(drop, 1) if drop is not None else None,
                "subject": e.get("subject"),
                "note": (f"{sym} goes ex on {ex} — {e.get('kind')}. The quoted price will "
                         f"fall ~{drop:.0f}% mechanically; your holding value does NOT "
                         f"change. Do not read it as a crash."
                         if drop is not None else ""),
            })
    out.sort(key=lambda x: (x["days_away"], x["symbol"]))
    return out


def recent(symbols=None, back_days: int = 10, today: date | None = None) -> list[dict]:
    """Actions that ALREADY went ex in the last `back_days`.

    This is the one that prevents panic: a name showing −50% on the screen because it
    split three days ago looks identical to a name that actually collapsed.
    """
    try:
        import corporate_actions as ca
        events = ca.load()
    except Exception:
        return []
    today = today or date.today()
    since = today - timedelta(days=max(1, int(back_days)))
    want = {s.strip().upper() for s in symbols} if symbols else None
    out = []
    for sym, evs in (events or {}).items():
        if want is not None and sym not in want:
            continue
        for e in evs or []:
            ex = _d(e.get("ex_date"))
            if ex is None or not (since <= ex <= today):
                continue
            try:
                drop = (1.0 - float(e.get("mult"))) * 100.0
            except Exception:
                drop = None
            out.append({
                "symbol": sym, "ex_date": str(ex),
                "days_ago": (today - ex).days,
                "kind": e.get("kind"), "multiplier": e.get("mult"),
                "expected_price_drop_pct": round(drop, 1) if drop is not None else None,
                "note": (f"{sym} went ex on {ex} ({e.get('kind')}). A ~{drop:.0f}% drop in "
                         f"the quoted price is MECHANICAL, not a loss."
                         if drop is not None else ""),
            })
    out.sort(key=lambda x: (x["days_ago"], x["symbol"]))
    return out


def summary(symbols=None, horizon_days: int = DEFAULT_HORIZON_DAYS) -> dict:
    up, rc = upcoming(symbols, horizon_days), recent(symbols)
    try:
        import corporate_actions as ca
        ev = ca.load()
        cov = {"symbols_in_feed": len(ev), "events_in_feed": sum(len(v) for v in ev.values())}
    except Exception:
        cov = {"symbols_in_feed": 0, "events_in_feed": 0}
    return {
        "upcoming": up, "recent": rc,
        "n_upcoming": len(up), "n_recent": len(rc),
        "horizon_days": horizon_days,
        "coverage": cov,
        "scope": ("Splits and bonuses only — the actions that REBASE the price. "
                  "Dividends, AGMs and buybacks are excluded because they do not "
                  "rescale price history."),
    }
