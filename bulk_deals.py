"""
Bulk / Block Deals Monitor
Downloads NSE archives static CSV once per calendar day — completely safe.
URLs are static files, not live APIs. No session/cookie seeding needed.

Bulk:  https://archives.nseindia.com/content/equities/bulk.csv
Block: https://archives.nseindia.com/content/equities/block.csv
"""
import io
import time
import pickle
import requests
import pandas as pd
import os
from pathlib import Path
from datetime import date

CACHE_DIR = Path(os.getenv("DEALS_DIR", os.path.join(os.path.expanduser("~"), ".ascent_cache", "nse_deals")))
CACHE_DIR.mkdir(exist_ok=True)
CACHE_TTL = 86_400   # 24 h

BULK_URL  = "https://archives.nseindia.com/content/equities/bulk.csv"
BLOCK_URL = "https://archives.nseindia.com/content/equities/block.csv"

_mem = {"data": None, "ts": 0}

_HEADERS = {
    "User-Agent":      "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                       "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Accept":          "text/html,*/*",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer":         "https://www.nseindia.com/",
}


# ── Download helpers ──────────────────────────────────────────────────────────

def _fetch(url: str, label: str) -> pd.DataFrame | None:
    try:
        r = requests.get(url, headers=_HEADERS, timeout=15)
        if r.status_code != 200 or len(r.content) < 100:
            return None
        df = pd.read_csv(io.BytesIO(r.content))
        df.columns = [c.strip() for c in df.columns]
        df["_deal_type"] = label
        return df
    except Exception:
        return None


def _parse(df: pd.DataFrame) -> list[dict]:
    if df is None or df.empty:
        return []
    # Discover column names defensively
    cols = {c.strip().upper(): c for c in df.columns}
    sym_col   = next((cols[k] for k in cols if "SYMBOL" in k), None)
    date_col  = next((cols[k] for k in cols if "DATE" in k), None)
    client_col = next((cols[k] for k in cols if "CLIENT" in k or "NAME" in k), None)
    # "Buy/Sell" column contains both words — match it explicitly first,
    # then fall back to any column with BUY in name.
    side_col  = next((cols[k] for k in cols if "BUY" in k and "SELL" in k), None) or \
                next((cols[k] for k in cols if "BUY" in k), None)
    qty_col   = next((cols[k] for k in cols if "QTY" in k or "QUANT" in k), None)
    price_col = next((cols[k] for k in cols if "PRICE" in k), None)

    OLD_DATE_SENTINEL = "1900-01-01"

    records = []
    for _, row in df.iterrows():
        try:
            sym = str(row.get(sym_col, "") or "").strip()
            if not sym:
                continue
            def _num(c):
                v = row.get(c, 0) if c else 0
                return float(str(v).replace(",", "") or 0)
            qty   = _num(qty_col)
            price = _num(price_col)

            # BUG-028 FIX: parse the date defensively so a single malformed
            # row never poisons downstream filters that do pd.Timestamp(date)
            # comparisons. Failed parses get an old sentinel that gets
            # filtered out gracefully (rather than blowing up the whole row).
            raw_date = str(row.get(date_col, "") or "").strip()
            try:
                if raw_date:
                    parsed = pd.to_datetime(raw_date, errors="raise", dayfirst=True)
                    date_str = parsed.strftime("%Y-%m-%d")
                else:
                    date_str = OLD_DATE_SENTINEL
            except Exception:
                date_str = OLD_DATE_SENTINEL

            records.append({
                "symbol":    sym,
                "date":      date_str,
                "client":    str(row.get(client_col, "") or "").strip(),
                "side":      str(row.get(side_col, "") or "").strip().upper(),
                "qty":       qty,
                "price":     price,
                "value_cr":  round(qty * price / 1e7, 2),
                "deal_type": str(row.get("_deal_type", "Bulk")).strip(),
            })
        except Exception:
            continue
    return records


# ── Main entry ────────────────────────────────────────────────────────────────

def run_bulk_deals() -> dict:
    # In-memory cache
    if _mem["data"] and time.time() - _mem["ts"] < CACHE_TTL:
        return _mem["data"]

    # Disk cache (one per calendar day)
    today      = date.today().strftime("%Y%m%d")
    cache_path = CACHE_DIR / f"deals_{today}.pkl"
    if cache_path.exists():
        try:
            with open(cache_path, "rb") as f:
                out = pickle.load(f)
            _mem.update({"data": out, "ts": time.time()})
            return out
        except Exception:
            cache_path.unlink(missing_ok=True)

    # Download both files
    bulk_df  = _fetch(BULK_URL,  "Bulk")
    block_df = _fetch(BLOCK_URL, "Block")

    deals = _parse(bulk_df) + _parse(block_df)
    # Drop rows that fell through to the OLD_DATE_SENTINEL ("1900-01-01") so
    # consumers and UI never see the sentinel as an apparent deal date.
    deals = [d for d in deals if d.get("date") != "1900-01-01"]
    deals.sort(key=lambda x: x.get("value_cr", 0), reverse=True)

    # BUG-036 FIX: previously `"B" in side` matched strings like "PURCHASE"
    # (contains a B), wildly inflating buy counts. Use exact match.
    def _is_buy(s: str) -> bool:
        return s.strip().upper() in ("BUY", "B")

    def _is_sell(s: str) -> bool:
        return s.strip().upper() in ("SELL", "S")

    buy_deals  = [d for d in deals if _is_buy(d.get("side", ""))]
    sell_deals = [d for d in deals if _is_sell(d.get("side", ""))]

    out = {
        "deals":         deals,
        "total_deals":   len(deals),
        "buy_count":     len(buy_deals),
        "sell_count":    len(sell_deals),
        "total_buy_cr":  round(sum(d["value_cr"] for d in buy_deals), 2),
        "total_sell_cr": round(sum(d["value_cr"] for d in sell_deals), 2),
        "fetched_at":    int(time.time()),
        "date":          today,
        "error":         None,
    }
    try:
        with open(cache_path, "wb") as f:
            pickle.dump(out, f)
    except Exception:
        pass
    _mem.update({"data": out, "ts": time.time()})
    return out
