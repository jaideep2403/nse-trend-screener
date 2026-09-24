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


def load_base_universe(days: int = 400, progress_callback=None,
                       include_stale: bool = False) -> dict[str, pd.DataFrame]:
    """Return {symbol: split-adjusted OHLCV DataFrame} for ALL NSE EQ stocks
    (ETFs excluded), built once per bhavcopy date and cached in memory. No
    MIN_BARS / index-membership / ADTV filter is applied — each scanner applies
    its own. Identical frames to the per-scanner build."""
    tag = _bhav_tag()
    _key = (days, include_stale)
    cached = _CACHE.get(_key)
    if cached is not None and cached["tag"] == tag:
        return cached["data"]

    with _LOCK:
        cached = _CACHE.get(_key)
        if cached is not None and cached["tag"] == tag:
            return cached["data"]

        # Read ONLY the bhavcopy files that exist on disk (glob) — do NOT iterate weekday
        # dates through _download_one_day, which would hit the NSE network (15s timeout
        # EACH) for every holiday/missing date whenever the negative cache is cold (e.g.
        # right after a restart). With a slow NSE archive that stacked into a ~3-minute
        # hang held under _LOCK, which blocked every caller — including each chart's peers
        # lookup — so no chart could load. New data is still fetched by the scheduler.
        from datetime import datetime as _dt2, date as _date2, timedelta as _td2
        try:
            from data_fetcher import BHAV_DIR as _BD
            _files = sorted(_BD.glob("*.pkl")) if _BD.exists() else []
        except Exception:
            _files = []
        _cut = _date2.today() - _td2(days=int(days) + 4)
        total = len(_files)
        frames = []
        for i, f in enumerate(_files):
            try:
                _d = _dt2.strptime(f.stem, "%Y%m%d").date()
            except ValueError:
                continue
            if _d < _cut:
                continue
            df = _download_one_day(_d)
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
            # include_stale keeps recently-delisted/suspended names so the CHART
            # endpoint can still draw their history — the recency gate is about not
            # SURFACING stale rows in scans, not about hiding a year of candles.
            if g.empty or (not include_stale and g.index[-1] < cutoff):
                dropped_stale += 1
                continue
            g = adjust_for_splits(g, sym)
            out[sym] = g
        if dropped_stale:
            print(f"[universe] dropped {dropped_stale} stale symbols "
                  f"(no bar since {cutoff.date()}, newest in load {latest.date()})")

        _CACHE[_key] = {"tag": tag, "data": out}
        return out


# ── Single-symbol deep history (for the CHART endpoint + RS benchmark) ────────
# THE HOT PATH. Charting ONE stock over 5Y (and the RS-line benchmark, which reads 20
# proxy members) used to scan EVERY bhavcopy day-file (~1,350 for 5Y) and filter one
# symbol out of each — 1.14s per symbol, ~23s for the benchmark. The archive is
# DAY-partitioned (one file per date, all ~2,900 symbols) but these reads are
# SYMBOL-partitioned ("give me one stock's whole history"), so 99.96% of every scan was
# thrown away.
#
# We now keep a durable per-symbol SNAPSHOT — one small pickle per symbol holding that
# name's full RAW (unadjusted) OHLCV(+DelivPer) history, date-indexed — in a persistent
# store alongside the day-files. A deep read becomes ONE ~0.3ms file load instead of a
# ~1.14s / 2.9M-row scan (~3,800x). Freshness is handled at read time: we merge only the
# tiny TAIL of day-files newer than the snapshot (normally 0-2), and rewrite ("compact")
# the snapshot once that tail grows past _COMPACT_TAIL — so the store self-maintains with
# no daily batch job, and a brand-new symbol self-populates on first read. Split
# adjustment is applied AFTER slicing to the window, so returned frames are byte-identical
# to the old glob-scan path.
_SYM_CACHE: dict = {}          # (sym, days) -> {"tag", "data"}; bounded, LRU-ish
_SYM_CACHE_MAX = 96

_COMPACT_TAIL = 10             # rewrite a snapshot once its unfolded day-file tail exceeds this
_STORE_BUILD_LOCK = threading.Lock()

import os as _os
import pickle as _pickle
from datetime import datetime as _dtc, date as _datec, timedelta as _tdc


def _sym_store_dir():
    """Persistent per-symbol snapshot store — sibling of the day-file archive so it lives
    under ~/.ascent_cache (NOT /tmp, which macOS purges)."""
    from data_fetcher import BHAV_DIR as _BD
    d = _BD.parent / "nse_symbol_hist"
    try:
        d.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass
    return d


def _sym_store_path(sym: str):
    return _sym_store_dir() / (sym.replace("/", "_") + ".pkl")


def _store_meta_path():
    return _sym_store_dir() / "_store_meta.json"


def _read_store_meta() -> dict:
    try:
        import json
        with open(_store_meta_path()) as f:
            return json.load(f)
    except Exception:
        return {}


def _write_store_meta(through: "_datec") -> None:
    try:
        import json, tempfile
        d = _sym_store_dir()
        fd, tmp = tempfile.mkstemp(dir=str(d), prefix="_store_meta.", suffix=".tmp")
        with _os.fdopen(fd, "w") as f:
            json.dump({"through": through.isoformat()}, f)
        _os.replace(tmp, str(_store_meta_path()))
    except Exception:
        pass


def _atomic_pkl(path, df) -> None:
    """Write a snapshot atomically (temp + rename) so a concurrent reader never sees a
    half-written file, and a redundant double-compact is harmless."""
    import tempfile
    tmp = None
    try:
        fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=path.stem + ".", suffix=".tmp")
        with _os.fdopen(fd, "wb") as f:
            _pickle.dump(df, f, protocol=_pickle.HIGHEST_PROTOCOL)
        _os.replace(tmp, str(path))
    except Exception:
        if tmp:
            try:
                _os.unlink(tmp)
            except Exception:
                pass


def _all_day_files() -> list:
    """(date, Path) for every bhavcopy day-file on disk, oldest first. Glob only — we
    NEVER iterate weekday dates through _download_one_day, so a cold negative-cache after
    a restart can't stall a read on a 15s NSE timeout."""
    try:
        from data_fetcher import BHAV_DIR as _BD
        files = sorted(_BD.glob("*.pkl")) if _BD.exists() else []
    except Exception:
        files = []
    try:
        from data_fetcher import _HOLIDAYS as _HOL   # market-holiday carry-forward dates
    except Exception:
        _HOL = set()
    out = []
    for f in files:
        try:
            d = _dtc.strptime(f.stem, "%Y%m%d").date()
        except ValueError:
            continue
        if d in _HOL:      # a non-trading day carried forward — never a real bar
            continue
        out.append((d, f))
    return out


def _rawify(df) -> "pd.DataFrame":
    """Day-file rows (with a Date column) → date-indexed raw OHLCV(+DelivPer), newest
    duplicate kept, sorted ascending. This is the on-disk snapshot shape."""
    cols = ["Open", "High", "Low", "Close", "Volume"]
    if "DelivPer" in df.columns:
        cols = cols + ["DelivPer"]
    g = df.set_index("Date")[cols]
    g = g[~g.index.duplicated(keep="last")].sort_index()
    return g


def _read_snapshot(path):
    if not path.exists():
        return None
    try:
        with open(path, "rb") as f:
            g = _pickle.load(f)
        if isinstance(g, pd.DataFrame) and len(g):
            return g
    except Exception:
        try:
            path.unlink()   # corrupt → drop so the next read rebuilds it
        except Exception:
            pass
    return None


def _build_symbol_raw_from_files(sym: str, files: list):
    """Full raw history for ONE symbol by scanning every day-file (~1.1s). Only used to
    first-populate a missing snapshot — every later read is the ~0.3ms snapshot path."""
    rows = []
    for d, f in files:
        df = _download_one_day(d)
        if df is None or df.empty:
            continue
        g = df[df["Symbol"] == sym]
        if not g.empty:
            rows.append(g)
    if not rows:
        return None
    return _rawify(pd.concat(rows, ignore_index=True))


def _load_symbol_raw(sym: str):
    """Full RAW (unadjusted) date-indexed history for ONE symbol: snapshot + fresh tail.
    Self-populates on a miss and self-compacts when the tail grows long."""
    path = _sym_store_path(sym)
    base = _read_snapshot(path)
    files = _all_day_files()

    if base is None:
        raw = _build_symbol_raw_from_files(sym, files)
        if raw is not None and not raw.empty:
            _atomic_pkl(path, raw)
        return raw

    # Merge only the day-files NEWER than the snapshot (normally 0-2).
    last = base.index[-1]
    last_d = (last.date() if hasattr(last, "date") else last)
    tail = [(d, f) for d, f in files if d > last_d]
    if not tail:
        return base
    extra = []
    for d, f in tail:
        df = _download_one_day(d)
        if df is None or df.empty:
            continue
        g = df[df["Symbol"] == sym]
        if not g.empty:
            extra.append(_rawify(g))
    if not extra:
        return base
    merged = pd.concat([base] + extra)
    merged = merged[~merged.index.duplicated(keep="last")].sort_index()
    if len(tail) > _COMPACT_TAIL:
        _atomic_pkl(path, merged)   # fold the tail back in so future reads stay ~0.3ms
    return merged


def build_symbol_store(progress_callback=None, force=False) -> dict:
    """One pass over ALL day-files → write one raw snapshot per symbol. This is what turns
    the 1.14s-per-symbol day-scan into a 0.3ms file read for every future deep read. Cheap
    to call repeatedly: it no-ops when the store is already current for the newest day-file
    (unless force), and otherwise folds only the new days into existing snapshots."""
    if not _STORE_BUILD_LOCK.acquire(blocking=False):
        return {"skipped": "build already running"}
    try:
        files = _all_day_files()
        if not files:
            return {"built": 0, "reason": "no day-files"}
        latest = files[-1][0]
        meta = _read_store_meta()
        d0 = _sym_store_dir()
        existing = [p for p in d0.glob("*.pkl")]
        if not force and existing and meta.get("through") == latest.isoformat():
            return {"skipped": "current", "through": latest.isoformat(), "symbols": len(existing)}

        # Incremental fold: store exists and is only a few days behind → append just the
        # new days to each affected snapshot (seconds), instead of a full rebuild.
        through = meta.get("through")
        if (not force) and existing and through:
            try:
                through_d = _datec.fromisoformat(through)
            except Exception:
                through_d = None
            new_files = [(d, f) for d, f in files if through_d and d > through_d]
            if through_d and 0 < len(new_files) <= 15:
                nf = []
                for d, f in new_files:
                    df = _download_one_day(d)
                    if df is not None and not df.empty:
                        nf.append(df)
                if nf:
                    newcomb = pd.concat(nf, ignore_index=True)
                    folded = 0
                    for sym, grp in newcomb.groupby("Symbol"):
                        p = _sym_store_path(sym)
                        prev = _read_snapshot(p)
                        add = _rawify(grp)
                        merged = pd.concat([prev, add]) if prev is not None else add
                        merged = merged[~merged.index.duplicated(keep="last")].sort_index()
                        _atomic_pkl(p, merged)
                        folded += 1
                    _write_store_meta(latest)
                    return {"folded": folded, "days": len(new_files), "through": latest.isoformat()}

        # Full (re)build — first ever build, or store too far behind.
        frames = []
        n = len(files)
        for i, (d, f) in enumerate(files):
            df = _download_one_day(d)
            if df is not None and not df.empty:
                frames.append(df)
            if progress_callback and i % 200 == 0:
                progress_callback(i, n, f"building symbol store {i}/{n} days")
        if not frames:
            return {"built": 0, "reason": "no readable day-files"}
        combined = pd.concat(frames, ignore_index=True)
        built = 0
        for sym, grp in combined.groupby("Symbol"):
            _atomic_pkl(_sym_store_path(sym), _rawify(grp))
            built += 1
        _write_store_meta(latest)
        return {"built": built, "through": latest.isoformat()}
    finally:
        _STORE_BUILD_LOCK.release()


def load_symbol_history(sym: str, days: int = 1300) -> pd.DataFrame | None:
    """Split-adjusted OHLCV(+DelivPer) for ONE symbol over `days` calendar days, or
    None if we have no bars. No recency gate — a delisted name still charts. Backed by the
    persistent per-symbol snapshot store, so a deep read is a ~0.3ms file load."""
    sym = (sym or "").upper().replace(".NS", "")
    if not sym:
        return None
    tag = _bhav_tag()
    key = (sym, days)
    cached = _SYM_CACHE.get(key)
    if cached is not None and cached["tag"] == tag:
        return cached["data"]

    raw = _load_symbol_raw(sym)
    if raw is None or raw.empty:
        _SYM_CACHE[key] = {"tag": tag, "data": None}
        return None

    # Slice to the window FIRST, THEN split-adjust — byte-identical to the old glob-scan
    # path (which only ever loaded in-window rows before adjusting). Same cutoff: days + 4
    # calendar-day pad to cover the trailing weekend/holiday.
    cutoff = pd.Timestamp(_datec.today() - _tdc(days=int(days) + 4))
    w = raw[raw.index >= cutoff]
    if w.empty:
        w = raw
    g = adjust_for_splits(w.copy(), sym)

    if len(_SYM_CACHE) >= _SYM_CACHE_MAX:      # keep memory flat — drop an old entry
        _SYM_CACHE.pop(next(iter(_SYM_CACHE)), None)
    _SYM_CACHE[key] = {"tag": tag, "data": g}
    return g


def invalidate() -> None:
    _CACHE.clear()
    _SYM_CACHE.clear()
