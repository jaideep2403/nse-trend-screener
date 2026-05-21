"""
FII / DII / Promoter / MF shareholding data per stock (NSE corporate-info API).

P0-3: prior implementation never existed — `/api/holders` returned {signal:"unknown"}.
NSE's corp-info shareholding endpoint requires 2-step cookie seeding:
  1. GET https://www.nseindia.com                  (sets ak_bmsc cookie)
  2. GET https://www.nseindia.com/api/marketStatus (sets bm_sv, _abck — required)
  3. GET https://www.nseindia.com/api/corp-info?symbol=X&corpType=shareholdings_patterns

Field name variations across NSE responses (defensive parsing):
  - FII:      totFIIHldg | fii  | ForeignInstitutions
  - DII:      totDIIHldg | dii  | DomesticInstitutions
  - MF:       totMFHldg  | mf   | MutualFunds
  - Promoter: totPromoterHldg | promoter

Fallback when NSE is rate-limited: derive a "Smart Money Signal" from MFI
on local bhavcopy data (MFI > 60 = accumulation, < 40 = distribution).
"""
from __future__ import annotations

import time
import threading
from typing import Optional

import requests
import pandas as pd

# Per-symbol cache (NSE aggressively rate-limits this endpoint)
_HOLDER_CACHE: dict[str, dict] = {}
_CACHE_TTL = 6 * 3600   # 6h — quarterly shareholding rarely changes
_cache_lock = threading.Lock()

# Module-level seeded session
_session: Optional[requests.Session] = None
_session_ts: float = 0.0
_SESSION_TTL = 1800     # rebuild every 30 min


def _nse_session() -> requests.Session:
    """Build (or reuse) a session with the 2-step cookie seeding NSE requires."""
    global _session, _session_ts
    if _session is not None and (time.time() - _session_ts) < _SESSION_TTL:
        return _session
    s = requests.Session()
    s.headers.update({
        "User-Agent":      ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                            "AppleWebKit/537.36 (KHTML, like Gecko) "
                            "Chrome/124.0.0.0 Safari/537.36"),
        "Accept":          "application/json, text/plain, */*",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer":         "https://www.nseindia.com/",
    })
    try:
        s.get("https://www.nseindia.com", timeout=8)
        s.get("https://www.nseindia.com/api/marketStatus", timeout=8)
    except Exception:
        pass
    _session = s
    _session_ts = time.time()
    return s


def _pick_field(d: dict, names: list[str]) -> Optional[float]:
    """Return first non-null numeric value from `d` matching any of `names`."""
    for n in names:
        if n in d and d[n] not in (None, "", "-"):
            try:
                return float(str(d[n]).replace(",", "").strip())
            except (ValueError, TypeError):
                continue
    return None


def _fetch_holders_nse(symbol: str) -> Optional[dict]:
    """
    Hit NSE shareholding API. Returns {fii_pct, dii_pct, mf_pct, promoter_pct,
    period, qoq_fii_delta, qoq_dii_delta} or None if API fails.
    """
    try:
        sess = _nse_session()
        url  = ("https://www.nseindia.com/api/corp-info"
                f"?symbol={symbol}&corpType=shareholdings_patterns&market=equities")
        r = sess.get(url, timeout=10)
        if r.status_code != 200:
            return None
        payload = r.json()
        rows = payload.get("data") or payload.get("shareholdings_patterns") or []
        if not isinstance(rows, list) or not rows:
            return None

        # Most recent quarter first
        def _period_key(row):
            p = row.get("period", "")
            try:
                return pd.Timestamp(p)
            except Exception:
                return pd.Timestamp.min
        rows = sorted(rows, key=_period_key, reverse=True)

        latest = rows[0]
        prior  = rows[1] if len(rows) > 1 else {}

        fii_now = _pick_field(latest, ["totFIIHldg", "fii", "ForeignInstitutions"])
        dii_now = _pick_field(latest, ["totDIIHldg", "dii", "DomesticInstitutions"])
        mf_now  = _pick_field(latest, ["totMFHldg",  "mf",  "MutualFunds"])
        prom_now= _pick_field(latest, ["totPromoterHldg", "promoter", "Promoter"])

        fii_prior = _pick_field(prior, ["totFIIHldg", "fii", "ForeignInstitutions"])
        dii_prior = _pick_field(prior, ["totDIIHldg", "dii", "DomesticInstitutions"])

        if all(v is None for v in [fii_now, dii_now, mf_now, prom_now]):
            return None

        return {
            "fii_pct":        fii_now,
            "dii_pct":        dii_now,
            "mf_pct":         mf_now,
            "promoter_pct":   prom_now,
            "period":         latest.get("period") or latest.get("date"),
            "qoq_fii_delta":  (fii_now - fii_prior) if (fii_now is not None and fii_prior is not None) else None,
            "qoq_dii_delta":  (dii_now - dii_prior) if (dii_now is not None and dii_prior is not None) else None,
            "source":         "NSE",
        }
    except Exception:
        return None


def _fallback_smart_money_signal(symbol: str) -> dict:
    """
    When NSE shareholding API fails, derive a smart-money proxy from MFI on
    local bhavcopy data. MFI > 60 = accumulation (buying), MFI < 40 = distribution.
    """
    try:
        from industry_groups import _get_stocks
        stocks = _get_stocks()
        df = stocks.get(symbol)
        if df is None or df.empty or len(df) < 30:
            return {"signal": "unknown", "source": "no_data"}

        from sector_analysis import _mfi
        mfi_v = _mfi(df, period=14)

        if mfi_v >= 60:
            sig, label = "accumulation", f"Smart Money Accumulating (MFI {mfi_v:.0f})"
        elif mfi_v <= 40:
            sig, label = "distribution", f"Smart Money Distributing (MFI {mfi_v:.0f})"
        else:
            sig, label = "neutral", f"Mixed (MFI {mfi_v:.0f})"

        return {
            "signal":   sig,
            "label":    label,
            "mfi":      mfi_v,
            "source":   "MFI_proxy",
        }
    except Exception:
        return {"signal": "unknown", "source": "error"}


def fetch_holders_for_symbol(symbol: str) -> dict:
    """
    Return shareholding breakdown for `symbol`. Cached for 6 hours.
    Returns dict with keys: symbol, fii_pct, dii_pct, mf_pct, promoter_pct,
    qoq_fii_delta, qoq_dii_delta, period, source, signal, label.
    """
    symbol = (symbol or "").strip().upper()
    if not symbol:
        return {"symbol": "", "signal": "unknown", "source": "empty_symbol"}

    with _cache_lock:
        cached = _HOLDER_CACHE.get(symbol)
        if cached and (time.time() - cached.get("ts", 0)) < _CACHE_TTL:
            return cached["data"]

    nse_data = _fetch_holders_nse(symbol)
    if nse_data:
        nse_data["symbol"] = symbol
        # Pair the % data with a clean signal label
        fii_d = nse_data.get("qoq_fii_delta")
        dii_d = nse_data.get("qoq_dii_delta")
        if fii_d is not None and fii_d >= 0.3:
            nse_data["signal"] = "fii_buying"
            nse_data["label"]  = f"FII +{fii_d:.2f}% QoQ"
        elif fii_d is not None and fii_d <= -0.3:
            nse_data["signal"] = "fii_selling"
            nse_data["label"]  = f"FII {fii_d:.2f}% QoQ"
        elif dii_d is not None and dii_d >= 0.3:
            nse_data["signal"] = "dii_buying"
            nse_data["label"]  = f"DII +{dii_d:.2f}% QoQ"
        elif dii_d is not None and dii_d <= -0.3:
            # Was missing: DII selling fell through to "stable" so users saw
            # "Holdings stable" on stocks where DIIs were actually trimming
            # meaningfully (-0.3% or more QoQ).
            nse_data["signal"] = "dii_selling"
            nse_data["label"]  = f"DII {dii_d:.2f}% QoQ"
        else:
            nse_data["signal"] = "stable"
            nse_data["label"]  = "Holdings stable"
        out = nse_data
    else:
        # Fallback to MFI-based smart money signal
        out = _fallback_smart_money_signal(symbol)
        out["symbol"] = symbol

    with _cache_lock:
        _HOLDER_CACHE[symbol] = {"data": out, "ts": time.time()}
    return out


def clear_cache():
    """Manual cache clear (e.g. for testing)."""
    with _cache_lock:
        _HOLDER_CACHE.clear()
