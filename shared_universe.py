"""
Shared base-universe loader.

Every scanner used to rebuild the SAME split-adjusted {symbol: OHLCV df} from
the bhavcopy day-files — the concat + groupby + per-symbol set_index / dedup /
split-adjust costs ~2s and was repeated 15 times (once per scanner). This builds
it ONCE per bhavcopy date, caches it in memory, and lets every scanner apply its
own (different) filter on top.

The construction here is byte-for-byte identical to what each scanner's
`_load_all_stocks` did internally (same `_weekdays_back(days)`, same
`concat().sort_values("Date")`, same `set_index(["O","H","L","C","V"])` +
`~index.duplicated(keep="last")` + `sort_index()`, same
`analysis_utils.adjust_for_splits`, same ETF drop) — so swapping a scanner over
to it does NOT change which stocks it scans or their data; it only removes the
redundant rebuild.
"""
from __future__ import annotations

import threading
import pandas as pd

from data_fetcher import _weekdays_back, _download_one_day
from analysis_utils import adjust_for_splits
from nse_stocks import is_etf

_LOCK = threading.Lock()
# Cache one base per (days) window, tagged by bhavcopy date so it auto-refreshes
# when newer data arrives.
_CACHE: dict[int, dict] = {}


def _bhav_tag() -> str:
    try:
        from data_fetcher import _latest_bhavcopy_date
        d = _latest_bhavcopy_date()
        return d.isoformat() if d else "nodate"
    except Exception:
        return "nodate"


def load_base_universe(days: int = 400, progress_callback=None) -> dict[str, pd.DataFrame]:
    """Return {symbol: split-adjusted OHLCV DataFrame} for ALL NSE EQ stocks
    (ETFs excluded), built once per bhavcopy date and cached in memory. No
    MIN_BARS / index-membership / ADTV filter is applied — each scanner applies
    its own. Identical frames to the per-scanner build."""
    tag = _bhav_tag()
    cached = _CACHE.get(days)
    if cached is not None and cached["tag"] == tag:
        return cached["data"]

    with _LOCK:
        cached = _CACHE.get(days)
        if cached is not None and cached["tag"] == tag:
            return cached["data"]

        dates = _weekdays_back(days)
        total = len(dates)
        frames = []
        for i, dt in enumerate(dates):
            df = _download_one_day(dt)
            if df is not None:
                frames.append(df)
            if progress_callback and i % 40 == 0:
                progress_callback(i, total, f"Loading bhavcopy cache… {i}/{total} days")
        if not frames:
            return {}

        combined = pd.concat(frames, ignore_index=True).sort_values("Date")
        out: dict[str, pd.DataFrame] = {}
        for sym, grp in combined.groupby("Symbol"):
            if is_etf(sym):
                continue
            g = grp.set_index("Date")[["Open", "High", "Low", "Close", "Volume"]]
            g = g[~g.index.duplicated(keep="last")].sort_index()
            g = adjust_for_splits(g, sym)
            out[sym] = g

        _CACHE[days] = {"tag": tag, "data": out}
        return out


def invalidate() -> None:
    _CACHE.clear()
