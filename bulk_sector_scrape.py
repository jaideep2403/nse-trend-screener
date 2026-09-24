"""
One-time BULK sector/industry scrape for the tail stocks NSE's free index list doesn't
classify. For each symbol we fetch its screener.in page ONCE and read the human-readable
Sector + Industry breadcrumb and the company name, then map the fine industry onto our own
sector taxonomy (falling back to the raw screener industry so every name still gets a
bucket). Results are stored in a human-readable JSON (sector_details.json) and reused by
the app — so we're never blocked by re-fetching.

Designed to be polite and unblockable:
  • one request at a time, ~2–3s jittered gap (human cadence)
  • realistic browser headers + a keep-alive session
  • exponential backoff on 429 / transient errors; STOPS after sustained blocks
  • resumable — every symbol is cached, a re-run skips finished ones
"""
from __future__ import annotations

import html as _html
import json
import os
import random
import re
import time

DATA_DIR = os.environ.get("DATA_DIR", os.path.dirname(__file__))
STORE = os.path.join(DATA_DIR, "sector_details.json")
PROGRESS = os.path.join(DATA_DIR, "sector_scrape_progress.json")
_mtime = [0.0]

_UA = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/122.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.screener.in/",
    "Connection": "keep-alive",
}
_MIN_GAP, _JITTER = 1.8, 1.4        # ~1.8–3.2s between hits — human cadence
_mem: dict | None = None


def _load() -> dict:
    global _mem
    # Reload when the file changes on disk so a READER process (the web server) picks up
    # rows the separate scrape process is still writing, without a restart.
    try:
        mt = os.path.getmtime(STORE)
    except OSError:
        mt = 0.0
    if _mem is None or mt != _mtime[0]:
        try:
            with open(STORE) as f:
                _mem = json.load(f)
        except Exception:
            _mem = {} if _mem is None else _mem
        _mtime[0] = mt
    return _mem


def _save():
    tmp = STORE + ".tmp"
    with open(tmp, "w") as f:
        json.dump(_mem, f, indent=1, sort_keys=True)
    os.replace(tmp, STORE)


def _clean(s: str | None) -> str | None:
    return _html.unescape(s).strip() if s else None


def _fetch(session, sym: str) -> dict | None:
    """{name, sector, industry} from screener.in, or None on a non-200 that isn't 429."""
    r = session.get(f"https://www.screener.in/company/{sym}/", headers=_UA, timeout=15)
    if r.status_code == 429:
        raise RuntimeError("429")
    if r.status_code != 200:
        return None
    h = r.text
    name = re.search(r"<h1[^>]*>\s*([^<]+?)\s*<", h)
    vals = [(m.group(1), _clean(m.group(2)))
            for m in re.finditer(r'(Sector|Industry)"[^>]*>([^<]{2,60})<', h)]
    sector = next((v for k, v in vals if k == "Sector"), None)
    industry = next((v for k, v in vals if k == "Industry"), None)
    return {"name": _clean(name.group(1)) if name else sym,
            "sector": sector, "industry": industry}


def _map_to_ours(sector: str | None, industry: str | None) -> str | None:
    """Map the screener classification onto OUR taxonomy by keyword overlap; if nothing
    matches, keep the raw screener industry so the stock still buckets somewhere real."""
    try:
        import sector_lookup as sl
        our = sl._our_sectors()
        kw = sl._kw
    except Exception:
        return industry or sector
    best, bs = None, 0
    for src in (industry, sector):          # prefer the finer 'industry' breadcrumb
        if not src:
            continue
        ik = kw(src)
        for sec in our:
            sc = len(ik & kw(sec))
            if sc > bs:
                best, bs = sec, sc
        if best:
            break
    return best or industry or sector


def run(symbols: list[str], progress_cb=None) -> dict:
    import requests
    m = _load()
    sess = requests.Session()
    todo = [s for s in symbols if not (m.get(s) and m[s].get("checked"))]
    total = len(todo)
    done = fail = consec_fail = 0
    t0 = time.time()
    for i, sym in enumerate(todo):
        try:
            d = _fetch(sess, sym)
            if d is not None:
                d["mapped"] = _map_to_ours(d.get("sector"), d.get("industry"))
                d["checked"] = True
                m[sym] = d
                done += 1
                consec_fail = 0
            else:
                m[sym] = {"name": sym, "sector": None, "industry": None,
                          "mapped": None, "checked": True}
                done += 1
        except Exception:
            fail += 1
            consec_fail += 1
            # exponential backoff on blocks; bail out if screener is clearly cutting us off
            if consec_fail >= 8:
                _save()
                _write_progress(done, fail, total, t0, stopped="blocked")
                return {"done": done, "fail": fail, "stopped": "blocked_after_8"}
            time.sleep(min(60, 5 * consec_fail))
            continue
        if (i + 1) % 15 == 0:
            _save()
            _write_progress(done, fail, total, t0)
            if progress_cb:
                progress_cb(done, total)
        time.sleep(_MIN_GAP + random.random() * _JITTER)
    _save()
    _write_progress(done, fail, total, t0, stopped="complete")
    return {"done": done, "fail": fail, "stopped": "complete"}


def _write_progress(done, fail, total, t0, stopped=None):
    try:
        with open(PROGRESS, "w") as f:
            json.dump({"done": done, "fail": fail, "total": total,
                       "elapsed_s": round(time.time() - t0),
                       "stopped": stopped, "ts": int(time.time())}, f)
    except Exception:
        pass


def get_sector_map() -> dict:
    """{symbol: bucket} for every tail stock we've classified (mapped → raw industry)."""
    out = {}
    for sym, d in _load().items():
        if not isinstance(d, dict):
            continue
        b = d.get("mapped") or d.get("industry") or d.get("sector")
        if b:
            out[sym] = b
    return out


def targets() -> list[str]:
    """Symbols in our price universe that the NSE bulk map doesn't classify."""
    import shared_universe as su
    import sector_mapper as sm
    bulk = {k for k, v in sm.get_enriched_sector_map().items() if v}
    U = su.load_base_universe(days=200, include_stale=True)
    return sorted(s for s in U if s not in bulk)


if __name__ == "__main__":
    syms = targets()
    print(f"[bulk_sector] scraping {len(syms)} unclassified symbols…", flush=True)
    res = run(syms, progress_cb=lambda d, t: print(f"[bulk_sector] {d}/{t}", flush=True))
    print(f"[bulk_sector] finished: {res}", flush=True)
