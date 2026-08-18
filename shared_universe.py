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
# A symbol with no bar in this many CALENDAR days is treated as no longer
# trading and is excluded from the universe. 10 days spans a long weekend
# plus a cluster of holidays, so a live-but-quiet stock is never dropped,
# while a suspended one leaves within two weeks.
MAX_STALE_DAYS = 10

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

        # RECENCY GATE (added 2026-08-13). A symbol that has stopped printing bars —
        # suspended, delisted, renamed, or simply never traded again — kept its last
        # frame forever and flowed into every screener, where its months-old close was
        # rendered as if it were today's. Audited on the live universe: 270 of 2,558
        # symbols (10.6%) were >7 sessions stale and 100 (3.9%) >90 days, and they WERE
        # reaching output — Weekly Breakout showed 13/111 stale rows (AUTOIND 84d,
        # CORDSCABLE 76d), Post-Breakout 25/543 (REGAAL 93d, STLTECH 92d), Accumulation
        # 2/50 (AKZOINDIA 121d). Calling a bar from three months ago "a breakout this
        # week" is the most misleading thing a screener can do, so the gate lives HERE,
        # at the one place every scanner inherits, rather than in each scanner.
        # Measured against the newest bar present ANYWHERE in the load, so it degrades
        # correctly on holidays/weekends instead of emptying the universe.
        latest = combined["Date"].max()
        cutoff = latest - pd.Timedelta(days=MAX_STALE_DAYS)

        out: dict[str, pd.DataFrame] = {}
        dropped_stale = 0
        for sym, grp in combined.groupby("Symbol"):
            if is_etf(sym):
                continue
            # DelivPer carried through (added 2026-08-13). It was being DROPPED here,
            # which is why every delivery study was capped at the ~13 months held in
            # data_fetcher's per-stock pickle cache. The raw bhavcopy day-files carry
            # DelivPer all the way back to 2019-08-23 (1,784 files, non-null on every
            # one sampled), so the cap was self-inflicted, not a data limit. Keeping
            # the column here gives the accumulation research ~7 years instead of 1.
            _cols = ["Open", "High", "Low", "Close", "Volume"]
            if "DelivPer" in grp.columns:
                _cols.append("DelivPer")
            g = grp.set_index("Date")[_cols]
            g = g[~g.index.duplicated(keep="last")].sort_index()
            if g.empty or g.index[-1] < cutoff:
                dropped_stale += 1
                continue
            g = adjust_for_splits(g, sym)
            out[sym] = g
        if dropped_stale:
            print(f"[universe] dropped {dropped_stale} stale symbols "
                  f"(no bar since {cutoff.date()}, newest in load {latest.date()})")

        _CACHE[days] = {"tag": tag, "data": out}
        return out


def invalidate() -> None:
    _CACHE.clear()
