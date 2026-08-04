"""
Promoter / insider accumulation factor — SAST Reg 29 + SEBI PIT disclosures.

WHY THIS IS THE STRONGEST "EARLY" SIGNAL AVAILABLE TO US
--------------------------------------------------------
When a promoter buys their own stock they must disclose it to the exchange within
~2 working days (SAST Reg 29 for substantial acquisitions; SEBI PIT Reg 7 for
insider/director trades). That is public, but it lands MONTHS before the same
information shows up in a quarterly shareholding pattern, and typically before
price momentum registers it. Almost no retail investor monitors the raw feed —
NSE publishes thousands of these a year — so the edge is systematic processing,
not privileged access. (Nothing here is inside information: acting on published
disclosures is entirely legal. This module reads only public NSE filings.)

════════════════════════════════════════════════════════════════════════════════
POINT-IN-TIME CORRECTNESS — the thing that makes or breaks this factor
────────────────────────────────────────────────────────────────────────────────
Every record carries BOTH a transaction date and a DISCLOSURE date, and they can
be days apart. Scoring on the transaction date would be look-ahead bias: you could
not have known about the trade until it was filed. This module therefore keys
every record on the **disclosure date** (`intimDt` / filing `timestamp`), never the
trade date. Same discipline as institutional_flow.py, and the reason both can be
honestly backtested while the quality factor cannot.
════════════════════════════════════════════════════════════════════════════════

Sources (both accept from_date/to_date, so history is backfillable):
  SAST Reg29 : /api/corporate-sast-reg29  — substantial acquisitions/disposals,
               with % of shareholding before/after (totAcqShare / totAftShare)
  SEBI PIT   : /api/corporates-pit        — insider/director/promoter trades,
               with person category and pre/post holding %
"""
from __future__ import annotations

import json
import os
import time
from datetime import date, timedelta

import numpy as np
import pandas as pd

CACHE_DIR = os.getenv("PROMOTER_DIR", os.path.join(os.path.expanduser("~"), ".ascent_cache", "nse_promoter"))
os.makedirs(CACHE_DIR, exist_ok=True)

SAST_URL = "https://www.nseindia.com/api/corporate-sast-reg29"
PIT_URL = "https://www.nseindia.com/api/corporates-pit"
REQ_DELAY = 2.0          # polite spacing between NSE calls
LOOKBACK_DAYS = 90       # trailing window over which promoter activity is scored
NEUTRAL = 0.5

_session = None
_cache: dict = {"txns": None, "score": None}


# ── HTTP ─────────────────────────────────────────────────────────────────────
def _seeded_session():
    """NSE's JSON APIs need cookies from a homepage visit first."""
    global _session
    if _session is not None:
        return _session
    import requests
    s = requests.Session()
    s.headers.update({
        "User-Agent": ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                       "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"),
        "Accept": "application/json,text/plain,*/*",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": "https://www.nseindia.com/companies-listing/corporate-filings-insider-trading",
    })
    try:
        s.get("https://www.nseindia.com", timeout=15)
        time.sleep(1)
    except Exception:
        pass
    _session = s
    return s


def _fetch(url: str, frm: date, to: date) -> list:
    s = _seeded_session()
    p = {"index": "equities", "from_date": frm.strftime("%d-%m-%Y"),
         "to_date": to.strftime("%d-%m-%Y")}
    try:
        r = s.get(url, params=p, timeout=30)
        if r.status_code != 200:
            return []
        j = r.json()
        d = j.get("data", j) if isinstance(j, dict) else j
        return d if isinstance(d, list) else []
    except Exception:
        return []


def _chunk_path(kind: str, frm: date, to: date) -> str:
    return os.path.join(CACHE_DIR, f"{kind}_{frm:%Y%m%d}_{to:%Y%m%d}.json")


def backfill(start: date, end: date | None = None, chunk_days: int = 90,
             progress=None) -> dict:
    """Backfill both feeds in `chunk_days` slices, caching each slice to disk.
    Re-running is cheap: cached slices are skipped."""
    end = end or date.today()
    got = {"sast": 0, "pit": 0, "chunks": 0, "cached": 0}
    cur = start
    while cur <= end:
        stop = min(cur + timedelta(days=chunk_days - 1), end)
        for kind, url in (("sast", SAST_URL), ("pit", PIT_URL)):
            p = _chunk_path(kind, cur, stop)
            if os.path.exists(p):
                try:
                    got[kind] += len(json.load(open(p)))
                    got["cached"] += 1
                    continue
                except Exception:
                    pass
            rows = _fetch(url, cur, stop)
            try:
                json.dump(rows, open(p, "w"))
            except Exception:
                pass
            got[kind] += len(rows)
            time.sleep(REQ_DELAY)
        got["chunks"] += 1
        if progress:
            progress(f"{cur}..{stop}  sast={got['sast']} pit={got['pit']}")
        cur = stop + timedelta(days=1)
    return got


# ── Normalisation ────────────────────────────────────────────────────────────
def _f(v) -> float | None:
    try:
        if v is None or str(v).strip() in ("", "-", "NA", "null"):
            return None
        return float(str(v).replace(",", "").strip())
    except Exception:
        return None


def _d(v) -> pd.Timestamp | None:
    if not v:
        return None
    s = str(v).split(" to ")[0].strip()
    for fmt in ("%d-%b-%Y %H:%M", "%d-%b-%Y", "%d-%B-%Y", "%d-%m-%Y", "%Y-%m-%d"):
        try:
            return pd.Timestamp(pd.to_datetime(s, format=fmt))
        except Exception:
            continue
    try:
        return pd.Timestamp(pd.to_datetime(s, dayfirst=True, errors="coerce"))
    except Exception:
        return None


def load_transactions(refresh: bool = False) -> pd.DataFrame:
    """Unified transaction table from every cached slice.
    Columns: symbol, disclosed (POINT-IN-TIME key), side, pct, pct_after,
             is_promoter, mode, person, source."""
    if not refresh and _cache["txns"] is not None:
        return _cache["txns"]
    rows = []
    for fn in sorted(os.listdir(CACHE_DIR)):
        if not fn.endswith(".json"):
            continue
        try:
            data = json.load(open(os.path.join(CACHE_DIR, fn)))
        except Exception:
            continue
        kind = "sast" if fn.startswith("sast") else "pit"
        for r in data:
            sym = (r.get("symbol") or "").strip().upper()
            if not sym:
                continue
            if kind == "sast":
                # disclosure time = when NSE published the filing
                disc = _d(r.get("timestamp") or r.get("sysTime"))
                acq = _f(r.get("totAcqShare"))
                sale = _f(r.get("totSaleShare"))
                is_buy = (r.get("acqSaleType") or "").lower().startswith("acq")
                pct = acq if is_buy else (sale if sale is not None else None)
                rows.append({
                    "symbol": sym, "disclosed": disc,
                    "side": "BUY" if is_buy else "SELL",
                    "pct": pct, "pct_after": _f(r.get("totAftShare")),
                    "is_promoter": (r.get("promoterType") or "").strip().upper() == "Y",
                    "mode": r.get("acquisitionMode"), "person": r.get("acquirerName"),
                    "source": "SAST",
                })
            else:
                disc = _d(r.get("intimDt") or r.get("date"))
                tt = (r.get("tdpTransactionType") or "").strip().lower()
                if tt.startswith("buy"):
                    side = "BUY"
                elif tt.startswith("sell") or tt.startswith("dispos"):
                    side = "SELL"
                else:
                    continue                      # pledge/encumbrance etc. — skip
                bef, aft = _f(r.get("befAcqSharesPer")), _f(r.get("afterAcqSharesPer"))
                pct = abs(aft - bef) if (bef is not None and aft is not None) else None
                cat = (r.get("personCategory") or "").lower()
                rows.append({
                    "symbol": sym, "disclosed": disc, "side": side,
                    "pct": pct, "pct_after": aft,
                    "is_promoter": ("promoter" in cat) or ("director" in cat),
                    "mode": r.get("acqMode"), "person": r.get("acqName"),
                    "source": "PIT",
                })
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df[df["disclosed"].notna()].copy()
        df["disclosed"] = pd.to_datetime(df["disclosed"]).dt.normalize()
        # SANITY CLAMP: a small number of NSE filings carry typo'd years (observed
        # 1922/1923/2036 — ~16 rows in 155k). Left unfiltered they stretch the score
        # panel across a century of empty days and can drag a stale reading forward.
        lo = pd.Timestamp("2015-01-01")
        hi = pd.Timestamp.today().normalize() + pd.Timedelta(days=7)
        df = df[(df["disclosed"] >= lo) & (df["disclosed"] <= hi)]
        df = df.sort_values("disclosed").reset_index(drop=True)
    _cache["txns"] = df
    return df


# ── Factor ───────────────────────────────────────────────────────────────────
def build_score_panel(lookback_days: int = LOOKBACK_DAYS,
                      refresh: bool = False) -> pd.DataFrame:
    """Panel of promoter-accumulation scores (rows = date, cols = symbol).

    Score at date D for symbol S = f(net promoter % bought − % sold, over
    disclosures published in (D-lookback, D]). Mapped to 0-1:
        net > +1.0%  → 1.00   (heavy accumulation)
        net > +0.1%  → 0.80
        |net| ≤ 0.1% → 0.50   (or no activity at all)
        net < −0.1%  → 0.20
        net < −1.0%  → 0.00   (heavy distribution)
    Only bars ≤ D are used, and only DISCLOSURE dates — no look-ahead."""
    if not refresh and _cache["score"] is not None:
        return _cache["score"]
    tx = load_transactions(refresh=refresh)
    if tx.empty:
        _cache["score"] = pd.DataFrame()
        return _cache["score"]
    t = tx[tx["is_promoter"] & tx["pct"].notna()].copy()
    # CONVICTION FILTER: only count trades where the insider actually put money in
    # (or took it out) on the open market. Gifts, ESOP grants, inter-se transfers
    # between promoters, preferential allotments, warrant conversions and merger
    # schemes all move shares WITHOUT expressing a view on price — ~32% of raw
    # "promoter buys" were these, which dilutes the signal with noise.
    _m = t["mode"].fillna("").str.lower()
    conviction = _m.str.contains("market purchase") | _m.str.contains("open market")
    t = t[conviction]
    if t.empty:
        _cache["score"] = pd.DataFrame()
        return _cache["score"]
    t["signed"] = np.where(t["side"] == "BUY", t["pct"], -t["pct"])
    # daily net per symbol, then a trailing rolling sum on a dense calendar
    daily = t.groupby(["disclosed", "symbol"])["signed"].sum().unstack(fill_value=0.0)
    idx = pd.date_range(daily.index.min(), daily.index.max(), freq="D")
    daily = daily.reindex(idx, fill_value=0.0)
    net = daily.rolling(f"{lookback_days}D").sum()

    score = pd.DataFrame(NEUTRAL, index=net.index, columns=net.columns, dtype=float)
    score = score.mask(net > 0.1, 0.80)
    score = score.mask(net > 1.0, 1.00)
    score = score.mask(net < -0.1, 0.20)
    score = score.mask(net < -1.0, 0.00)
    _cache["score"] = score
    return score


def score_map_asof(on_date, max_stale_days: int = 10) -> dict:
    """{symbol: score} as known on `on_date` (point-in-time). Symbols with no
    promoter disclosures in the window simply aren't present → caller uses 0.5."""
    sc = build_score_panel()
    if sc.empty:
        return {}
    ts = pd.Timestamp(on_date).normalize()
    rows = sc.loc[sc.index <= ts]
    if rows.empty:
        return {}
    if (ts - rows.index[-1]).days > max_stale_days:
        return {}
    r = rows.iloc[-1]
    return r[r != NEUTRAL].to_dict()      # only symbols with real activity


def importance(pct: float | None) -> str:
    """ValuePicker-style materiality tiers on % of shares outstanding."""
    if pct is None:
        return "—"
    a = abs(pct)
    if a >= 10: return "Very high"
    if a >= 5:  return "High"
    if a >= 2:  return "Medium"
    if a >= 0.5: return "Low"
    return "Minimal"


def recent_transactions(limit: int = 200, promoter_only: bool = False) -> list[dict]:
    """Newest disclosures first — powers the Promoter Activity tab."""
    tx = load_transactions()
    if tx.empty:
        return []
    d = tx[tx["is_promoter"]] if promoter_only else tx
    d = d.sort_values("disclosed", ascending=False).head(limit)
    out = []
    for _, r in d.iterrows():
        out.append({
            "symbol": r["symbol"],
            "date": r["disclosed"].strftime("%Y-%m-%d") if pd.notna(r["disclosed"]) else None,
            "side": r["side"],
            "pct": round(float(r["pct"]), 2) if pd.notna(r["pct"]) else None,
            "pct_after": round(float(r["pct_after"]), 2) if pd.notna(r["pct_after"]) else None,
            "importance": importance(r["pct"] if pd.notna(r["pct"]) else None),
            "is_promoter": bool(r["is_promoter"]),
            "mode": r["mode"], "person": r["person"], "source": r["source"],
        })
    return out


def coverage() -> dict:
    tx = load_transactions()
    if tx.empty:
        return {"transactions": 0, "symbols": 0, "first": None, "last": None,
                "buys": 0, "sells": 0, "promoter": 0}
    return {
        "transactions": int(len(tx)),
        "symbols": int(tx["symbol"].nunique()),
        "first": str(tx["disclosed"].min().date()),
        "last": str(tx["disclosed"].max().date()),
        "buys": int((tx["side"] == "BUY").sum()),
        "sells": int((tx["side"] == "SELL").sum()),
        "promoter": int(tx["is_promoter"].sum()),
    }
