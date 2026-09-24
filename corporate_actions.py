"""Authoritative NSE corporate-actions feed — exact split/bonus ex-dates and ratios.

WHY THIS EXISTS (2026-07-26). `analysis_utils.adjust_for_splits()` was a price/volume
heuristic, and an audit showed it failing on major liquid names: 398/2,871 symbols in
the backtest universe still carried a fabricated −45%…−90% single-day crash, 92 of them
tradable (DIXON, EICHERMOT, ADANIPOWER, CDSL, NYKAA, RBLBANK, HDFCAMC, BEML…).

Two proven causes: a "turnover panic" guard that rejects real splits (ex-date share
volume explodes — EICHERMOT was 0.3% off a perfect 1:10 with 43× volume and was thrown
out), and a `vol_up ≥ 1.3×` requirement that blocks near-exact splits with flat volume.

Widening the heuristic is NOT the fix. Measured: loosening the clean-fraction tolerance
to 8% would wrongly capture 41 genuine crashes, and a FABRICATED split is worse than a
missed one — it rescales history into a fake breakout that gets bought. The separation
simply is not in the price/volume data: "opened at the new level" holds for 89.9% of
near-exact-ratio events but also 78.8% of far-from-clean ones.

So use the authority instead. NSE publishes every corporate action with an exact ex-date
and a ratio in the `subject` text. This module fetches, parses and caches it; the
heuristic stays only as a fallback for symbols the feed does not cover.

Politeness: one seeding request, then one request per year with a delay between. The
result is cached on disk (`.corporate_actions.json`) and refreshed at most daily, so
normal operation makes ZERO network calls.
"""

from __future__ import annotations

import json
import os
import re
import time
from datetime import date, datetime

import requests

_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_PATH = os.path.join(_DIR, ".corporate_actions.json")
CACHE_TTL = 86400          # refresh at most once a day

_BASE = "https://www.nseindia.com/api/corporates-corporateActions"
_HOME = "https://www.nseindia.com/"
_REFERER = "https://www.nseindia.com/companies-listing/corporate-filings-actions"

REQUEST_GAP = 3.0          # seconds between year requests — deliberately unhurried

_HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
                   "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"),
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": _REFERER,
}

# ── Ratio parsing ────────────────────────────────────────────────────────────
# NSE writes the action in free text, e.g.
#   "Face Value Split (Sub-Division) - From Rs 10/- Per Share To Rs 2/- Per Share"
#   "Bonus 1:1"   "Bonus issue 3:5"
# A SPLIT ratio is new_face/old_face (10→2 ⇒ price ×0.2).
# A BONUS a:b gives `a` free shares for every `b` held ⇒ price × b/(a+b).
# NOTE: NSE writes ₹1 as "Re 1/-" (singular), not "Rs 1/-". Omitting `Re` silently
# dropped every split whose NEW face value was exactly ₹1 — which is the single most
# common split target. EICHERMOT's 1:10 (Rs 10 → Re 1, ex-2020-08-24, a −90% artifact)
# was missed for exactly this reason.
_SPLIT_FV = re.compile(
    r"from\s*(?:rs\.?|re\.?|inr)?\s*([\d.]+)\s*/?-?\s*(?:per\s*share)?\s*to\s*"
    r"(?:rs\.?|re\.?|inr)?\s*([\d.]+)",
    re.I)
_BONUS = re.compile(r"bonus[^0-9]{0,20}(\d+)\s*[:/]\s*(\d+)", re.I)
_SPLIT_RATIO = re.compile(r"split[^0-9]{0,20}(\d+)\s*[:/]\s*(\d+)", re.I)


def parse_ratio(subject: str) -> tuple[float | None, str | None]:
    """Return (price_multiplier, kind) for a corporate-action subject line.

    price_multiplier is what the PRE-event price must be multiplied by to be
    comparable with post-event prices (0.5 for a 1:1 bonus / 1:2 split). Returns
    (None, None) when the text is not a split/bonus (dividends, AGMs, buybacks…).
    """
    if not subject:
        return None, None
    s = " ".join(str(subject).split())
    low = s.lower()

    # Dividends and everything else that does not rebase the price.
    if "split" not in low and "bonus" not in low and "sub-division" not in low \
            and "sub division" not in low:
        return None, None

    # NON-EQUITY bonuses do NOT rebase the equity price. TVS Motor's
    # "Scheme Of Arrangement - Bonus Ncrps 4:1" is a bonus issue of non-convertible
    # redeemable PREFERENCE shares; reading it as a 4:1 equity bonus applied ×0.2 to
    # all prior history and fabricated a +398% one-day jump. Anything that is not
    # plain equity is excluded.
    if any(w in low for w in ("ncrps", "ncd", "preference", "pref share", "debenture",
                              "warrant", "rights", "demerger", "spin")):
        return None, None

    m = _BONUS.search(s)
    if m and "bonus" in low:
        a, b = float(m.group(1)), float(m.group(2))
        if a > 0 and b > 0:
            return b / (a + b), f"bonus {int(a)}:{int(b)}"

    m = _SPLIT_FV.search(s)
    if m:
        old_fv, new_fv = float(m.group(1)), float(m.group(2))
        if old_fv > 0 and new_fv > 0 and new_fv < old_fv:
            return new_fv / old_fv, f"split FV {old_fv:g}->{new_fv:g}"

    m = _SPLIT_RATIO.search(s)
    if m:
        a, b = float(m.group(1)), float(m.group(2))
        if a > 0 and b > 0:
            # "Split 1:5" on NSE means 1 share becomes 5 ⇒ price ×1/5.
            lo, hi = min(a, b), max(a, b)
            return lo / hi, f"split {int(a)}:{int(b)}"
    return None, None


def _parse_exdate(v) -> str | None:
    for fmt in ("%d-%b-%Y", "%d-%m-%Y", "%Y-%m-%d"):
        try:
            return datetime.strptime(str(v).strip(), fmt).strftime("%Y-%m-%d")
        except Exception:
            continue
    return None


def fetch_range(from_year: int, to_year: int, session=None, log=print) -> list[dict]:
    """Fetch raw corporate actions year by year. One request per year, spaced out."""
    s = session or requests.Session()
    s.headers.update(_HEADERS)
    try:
        s.get(_HOME, timeout=15)          # seed cookies once
    except Exception:
        pass
    out: list[dict] = []
    for y in range(from_year, to_year + 1):
        url = f"{_BASE}?index=equities&from_date=01-01-{y}&to_date=31-12-{y}"
        try:
            r = s.get(url, timeout=30)
            if r.status_code != 200:
                log(f"[corp_actions] {y}: HTTP {r.status_code} — skipped")
            else:
                j = r.json()
                if isinstance(j, list):
                    out.extend(j)
                    log(f"[corp_actions] {y}: {len(j)} records")
        except Exception as e:
            log(f"[corp_actions] {y}: {type(e).__name__} — skipped")
        time.sleep(REQUEST_GAP)
        # Be a good citizen: never hammer. A missing year degrades to the
        # heuristic for that period rather than retrying in a loop.
    return out


def build(from_year: int = 2019, to_year: int | None = None, log=print) -> dict:
    """Fetch + parse + cache. Returns {symbol: [{ex_date, mult, kind, subject}, …]}."""
    to_year = to_year or date.today().year
    raw = fetch_range(from_year, to_year, log=log)
    events, kept = _parse_records(raw)
    payload = {"built_at": datetime.now().isoformat(timespec="seconds"),
               "from_year": from_year, "to_year": to_year,
               "n_raw": len(raw), "n_events": kept, "events": events}
    _write(payload)
    log(f"[corp_actions] parsed {kept} split/bonus events across {len(events)} symbols "
        f"from {len(raw)} raw records")
    return payload


def _parse_records(raw: list[dict]) -> tuple[dict, int]:
    events: dict[str, list[dict]] = {}
    kept = 0
    for rec in raw:
        sym = (rec.get("symbol") or "").strip().upper()
        subj = rec.get("subject") or ""
        mult, kind = parse_ratio(subj)
        ex = _parse_exdate(rec.get("exDate"))
        if not sym or not ex or mult is None:
            continue
        if not (0.01 <= mult < 1.0):          # sanity: must actually reduce the price
            continue
        events.setdefault(sym, []).append(
            {"ex_date": ex, "mult": round(mult, 6), "kind": kind, "subject": subj[:120]})
        kept += 1
    for sym in events:
        events[sym].sort(key=lambda e: e["ex_date"])
    return events, kept


def _write(payload: dict) -> None:
    try:
        tmp = CACHE_PATH + ".tmp"
        with open(tmp, "w") as fh:
            json.dump(payload, fh)
        os.replace(tmp, CACHE_PATH)
    except OSError:
        pass
    _mem["data"], _mem["ts"] = None, 0.0


def refresh_recent(log=print, min_age_sec: int = CACHE_TTL, force: bool = False) -> dict:
    """Pull THIS year's actions and merge them in. At most once a day.

    WHY (2026-09-17): the docstring promised "refreshed at most daily", but only
    build() existed and nothing ever called it. The cache was built once on
    2026-07-29, so every split/bonus after 2026-07-31 was invisible — TDPOWERSYS,
    GOODLUCK, INDIAGLYCO, KIRLPNU, TEMBO and TCC all carried fake 49-89% one-day
    crashes into returns, RS, 52-week highs, volatility and stops.

    Politeness: one cookie-seeding request + one request for the current year (plus
    the previous year during January, for late-December ex-dates). Existing events
    are kept; new ones are merged, never duplicated."""
    try:
        with open(CACHE_PATH) as fh:
            payload = json.load(fh)
    except Exception:
        payload = {"events": {}, "from_year": date.today().year}
    last = payload.get("refreshed_at") or payload.get("built_at")
    if not force and last:
        try:
            age = (datetime.now() - datetime.fromisoformat(last)).total_seconds()
            if age < min_age_sec:
                return {"skipped": True, "reason": f"refreshed {age/3600:.1f}h ago", "added": 0}
        except Exception:
            pass
    today = date.today()
    y0 = today.year - 1 if today.month == 1 else today.year
    raw = fetch_range(y0, today.year, log=log)
    payload["last_attempt_at"] = datetime.now().isoformat(timespec="seconds")
    if not raw:
        payload["last_error"] = "no records returned (NSE unreachable or blocked)"
        _write(payload)
        log("[corp_actions] refresh: no records returned — keeping existing events")
        return {"skipped": False, "added": 0, "error": payload["last_error"]}
    fresh, _ = _parse_records(raw)
    events = payload.get("events") or {}
    added = []
    for sym, evs in fresh.items():
        have = {(e["ex_date"], e["mult"], e["kind"]) for e in events.get(sym, [])}
        for e in evs:
            if (e["ex_date"], e["mult"], e["kind"]) not in have:
                events.setdefault(sym, []).append(e)
                added.append((sym, e["ex_date"], e["mult"], e["kind"]))
        events.get(sym, []).sort(key=lambda e: e["ex_date"])
    payload.update(events=events, refreshed_at=datetime.now().isoformat(timespec="seconds"),
                   to_year=max(int(payload.get("to_year") or today.year), today.year),
                   n_events=sum(len(v) for v in events.values()), last_error=None)
    _write(payload)
    log(f"[corp_actions] refresh: {len(raw)} records for {y0}-{today.year}, "
        f"{len(added)} new event(s)" + (f": {added[:8]}" if added else ""))
    return {"skipped": False, "added": len(added), "new_events": added}


_mem: dict = {"data": None, "ts": 0.0}


def load(refresh: bool = False) -> dict:
    """Cached {symbol: [events]}. Never triggers a network call on its own — the
    cache is built explicitly by `build()` (a scheduled/manual backfill), so a
    scanner run can never stall on NSE being slow."""
    now = time.time()
    if not refresh and _mem["data"] is not None and now - _mem["ts"] < 600:
        return _mem["data"]
    data = {}
    try:
        with open(CACHE_PATH) as fh:
            data = json.load(fh).get("events") or {}
    except Exception:
        data = {}
    _mem["data"] = data
    _mem["ts"] = now
    return data


def events_for(symbol: str) -> list[dict]:
    # Accept "TDPOWERSYS.NS" as well as "TDPOWERSYS": data_fetcher's per-ticker cache
    # passes the yfinance-style ticker, which silently matched NO feed events (2026-09-17).
    sym = (symbol or "").strip().upper()
    for suffix in (".NS", ".BO"):
        if sym.endswith(suffix):
            sym = sym[: -len(suffix)]
    return load().get(sym, [])
