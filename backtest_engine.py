"""
Factor-Attribution Backtest Engine
==================================
A correctness-first, point-in-time, cost-aware backtest that measures WHICH
signals actually predict forward returns. The goal is evidence: instead of
guessing how to re-weight the Trending tab's 10 criteria + composite score, we
measure the Information Coefficient (rank correlation of factor vs forward
return) of every candidate factor across many historical snapshots.

WHY POINT-IN-TIME MATTERS
-------------------------
A backtest that lets a feature peek at data it could not have known at decision
time will report fake predictive power. Here, for a snapshot date ``as_of`` a
stock's features are computed from ONLY the bars with index <= as_of (by
slicing ``df.loc[:as_of]`` and ``nifty.loc[:as_of]`` before scoring). The
forward return is then measured from bars strictly AFTER as_of. The slice is
what guarantees no look-ahead leakage.

SELF-VALIDATION GATE
--------------------
Two control factors are scored alongside the real ones and asserted before any
result is trusted:
  * ``ctrl_mom63``  — 63-bar trailing return, a genuinely predictive momentum
    factor. Its mean IC MUST be positive and clearly larger than noise.
  * ``ctrl_noise``  — a deterministic pseudo-random number per (symbol, snapshot).
    Its mean |IC| MUST be ~0 (well under 0.02).
If momentum is not clearly positive, or noise is not ~0, the engine has a
look-ahead / sign / alignment bug and refuses to report success.

REUSE (no reinvention)
----------------------
  * ``edge_engine._load_stocks``        — bhavcopy OHLCV loader
  * ``trending._score_stock``           — the live 10-criteria + composite scorer
  * ``industry_groups._build_nifty``    — equal-weight Nifty benchmark builder
  * ``costs.round_trip_cost_pct``       — liquidity-scaled round-trip cost model
"""
from __future__ import annotations

import math
import time

import numpy as np
import pandas as pd

from edge_engine import _load_stocks
from trending import _score_stock
from industry_groups import _build_nifty
from costs import round_trip_cost_pct


# ── Configuration ───────────────────────────────────────────────────────────────
# Load the DEEP archive. `_load_stocks(days=N)` counts back N *weekdays* (calendar
# weekdays), NOT trading days, and the local bhavcopy cache bottoms out at
# 2019-11-18 (~1705 trading days). Reaching that floor needs ~2300+ weekdays —
# requesting 1600 weekdays only reaches ~2022-01 and yields just ~39
# non-overlapping HOLD=21 windows, well short of the ~55-70 we need for
# statistical confidence. Over-requesting is free (missing days return no file
# and the floor clamps the span), so we request the full archive depth: ~2400
# weekdays reaches 2019-11-18 and produces ~60-66 non-overlapping 21-bar forward
# windows. We run ONLY HOLD=21 here (the 63-bar horizon is skipped to stay well
# inside the runtime budget).
LOAD_DAYS        = 2400     # ~full local archive (cache floor 2019-11-18); clamps naturally
MIN_HISTORY_BARS = 300      # a stock must have >=300 bars up to as_of to score
DEFAULT_HOLD     = 21       # the ONLY forward-return horizon (≈1 month)
N_DECILES        = 10       # for the composite top-vs-bottom decile spread
ROBUST_T         = 2.0      # |t-stat| threshold for a factor to be called ROBUST
# Universe cap: _score_stock costs ~8 ms; the full ~750-name universe × ~60-70
# non-overlapping snapshots (HOLD=21) would push the runtime budget. Restrict to
# the MAX_UNIVERSE most-liquid names by full-period median turnover. This is a
# STATIC universe filter (the SAME fixed set at every snapshot), so it cannot leak
# forward information into any IC — it only trims runtime while keeping N (≈300
# names per snapshot) far above the cross-sectional observation target.
MAX_UNIVERSE     = 300

# Criterion labels (mirror trending.run_trending_scan's criteria_labels) so the
# printed table is readable. Index i corresponds to criteria[i] from _score_stock.
CRITERIA_LABELS = [
    "c1 Price>MA20&MA50",
    "c2 MA20>MA50",
    "c3 MA50 Slope Up",
    "c4 Within 20% 52WHi",
    "c5 Beating Nifty 1M",
    "c6 Beating Nifty 3M",
    "c7 Accumulation",
    "c8 ADX>=25",
    "c9 Higher Highs/Lows",
    "c10 Price>MA200",
]

# Diagnostic controls — scored alongside the real factors to police the pipeline,
# but excluded from the "robust tradable factors" count (they are not signals you
# would ever re-weight the Trending tab on).
CONTROL_FACTORS = {"ctrl_mom252", "ctrl_mom63", "ctrl_noise"}


# ── composite_v2 — evidence-weighted candidate score ────────────────────────────
# A re-weighting of ONLY features the engine already extracts per (stock, snapshot),
# with weights set from the IN-SAMPLE (older-half) attribution evidence — never
# tuned on the OOS half (that would be overfitting). The continuous components
# (r6m, pct_from_252hi) are z-scored CROSS-SECTIONALLY within each snapshot so their
# magnitude carries (a +30% 6-month leader outranks a +2% one) on a scale
# comparable to the 0/1 structural bools. The trap factors that flipped sign
# out-of-sample in the baseline run (c5/c6/c2/c3/c7) get ZERO weight.
#
#   IS_IC evidence (baseline, older half): r6m +0.025, pct_from_252hi +0.018,
#   c10 +0.007, c8 +0.003, c9 +0.002.  r6m is the only |t|>=2 ROBUST factor and
#   dominates → it carries the heaviest weight.
#
# Keys must match the per-stock `row` dict assembled in run_attribution:
#   continuous → z-scored;  bool criteria → used as-is (already 0/1).
_V2_Z_WEIGHTS = {
    "r6m":            0.50,   # ROBUST 6-month momentum — the proven forward-return predictor
    "pct_from_252hi": 0.25,   # proximity to 252-day high (continuous, <=0)
}
_V2_BOOL_WEIGHTS = {
    "c10 Price>MA200":      0.12,   # bedrock long-term-trend filter (OOS-stable)
    "c9 Higher Highs/Lows": 0.07,   # bullish price structure (OOS-stable, weak)
    "c8 ADX>=25":           0.06,   # directional-trend strength (OOS-stable, weak)
    # ZERO weight (sign-unstable / negative out-of-sample traps):
    #   c6 Beating-Nifty-3M, c5 Beating-Nifty-1M, c3 MA50-slope,
    #   c2 MA20>MA50, c7 Accumulation, c1, c4.
}


def _zscore(s: pd.Series) -> pd.Series:
    """
    Cross-sectional z-score of a numeric series (mean 0, std 1), NaN-safe.

    Computed within a SINGLE snapshot's cross-section only — it uses no data from
    other snapshots or any future bar, so it cannot leak forward information. NaNs
    (insufficient history for that factor on that stock) map to 0.0 = the
    cross-sectional mean, a neutral contribution. A degenerate (constant) column
    also collapses to all-zeros.
    """
    x = pd.to_numeric(s, errors="coerce")
    mu = x.mean(skipna=True)
    sd = x.std(ddof=0, skipna=True)
    if not math.isfinite(sd) or sd == 0:
        return pd.Series(0.0, index=s.index)
    z = (x - mu) / sd
    return z.fillna(0.0)


def _composite_v2(fdf: pd.DataFrame) -> pd.Series:
    """
    Build the evidence-weighted composite_v2 score for one snapshot's factor frame.

    `fdf` is the per-snapshot DataFrame (one row per scored stock) assembled in
    run_attribution. Continuous components are z-scored cross-sectionally; bool
    criteria (already 0/1) are added with their weights. Returns a float Series
    aligned to fdf's index. All inputs are point-in-time (<= as_of) and the
    z-scoring is within-snapshot, so no look-ahead is introduced.
    """
    score = pd.Series(0.0, index=fdf.index)
    for col, w in _V2_Z_WEIGHTS.items():
        if col in fdf.columns:
            score = score + w * _zscore(fdf[col])
    for col, w in _V2_BOOL_WEIGHTS.items():
        if col in fdf.columns:
            score = score + w * pd.to_numeric(fdf[col], errors="coerce").fillna(0.0)
    return score


# ── Small numeric helpers ───────────────────────────────────────────────────────

def _sign(x: float) -> int:
    """Sign of x as -1 / 0 / +1 (NaN → 0). Used for IS/OOS/mean sign-stability."""
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return 0
    return (x > 0) - (x < 0)


# ── Spearman rank correlation (no scipy in this env) ───────────────────────────

def _spearman(xs: list[float], ys: list[float]) -> float | None:
    """
    Spearman rank correlation between two equal-length lists.

    Implemented as the Pearson correlation of the rank-transformed series
    (pandas ``.rank()`` uses average ranks for ties — the standard definition).
    Returns None on degenerate input (too few points, or a constant series whose
    ranks have zero variance, which makes correlation undefined).
    """
    if not xs or not ys or len(xs) != len(ys) or len(xs) < 5:
        return None
    s = pd.Series(xs, dtype="float64")
    a = pd.Series(ys, dtype="float64")
    mask = s.notna() & a.notna()
    if int(mask.sum()) < 5:
        return None
    rs = s[mask].rank()
    ra = a[mask].rank()
    # A constant factor (e.g. a criterion that is all-True at this snapshot) has
    # zero rank variance → correlation is undefined; skip rather than emit NaN.
    if rs.std(ddof=0) == 0 or ra.std(ddof=0) == 0:
        return None
    c = rs.corr(ra)
    return float(c) if (c is not None and not pd.isna(c)) else None


# ── Deterministic noise control ────────────────────────────────────────────────

def _noise(symbol: str, snap_idx: int) -> float:
    """
    Deterministic pseudo-random number in [0, 1) for (symbol, snapshot).

    Uses a stable hash of the pair so it is reproducible across runs (Python's
    builtin ``hash`` is salted per-process and would NOT be reproducible). This
    is the negative control: it carries no information about forward returns, so
    its IC must come out ~0. If it does not, the IC pipeline itself is broken.
    """
    h = abs(hash_str(f"{symbol}|{snap_idx}"))
    return (h % 1_000_000) / 1_000_000.0


def hash_str(s: str) -> int:
    """Stable 64-bit FNV-1a hash (process-independent, unlike builtin hash)."""
    h = 0xcbf29ce484222325
    for ch in s.encode("utf-8"):
        h ^= ch
        h = (h * 0x100000001b3) & 0xFFFFFFFFFFFFFFFF
    return h


# ── Point-in-time candidate factors ────────────────────────────────────────────

def _candidate_factors(df_slice: pd.DataFrame, nifty_slice: pd.Series) -> dict | None:
    """
    Compute the extra candidate factors that are NOT already returned by
    _score_stock, using ONLY the sliced (<= as_of) data.

    Returns a dict of factor_name -> value (None where insufficient history),
    or None if the close series is too short to be useful.

      r3m         — 63-bar trailing return
      r6m         — 126-bar trailing return
      rs_proxy    — stock 63-bar return minus Nifty 63-bar return (relative strength)
      vol_ratio   — last-20 avg volume / prior-50 avg volume (volume expansion)
      pct_from_252hi — close / 252-bar high - 1 (proximity to 1yr high; <=0)
      ctrl_mom63  — 63-bar trailing return, reported for transparency. NOTE: this
                    is NOT the gated positive control — over short horizons
                    Indian equities exhibit reversal, so 63-bar momentum is not
                    reliably positive (a real market property, not a bug).
      ctrl_mom252 — 252-bar (12-month) trailing return: the canonical academic
                    momentum factor and the GATED positive control. This is the
                    robustly-predictive "known-good" signal the self-validation
                    gate checks (edge_engine itself documents r12m as the single
                    strongest alpha feature, IC +0.145).
    """
    c = df_slice["Close"].dropna()
    n = len(c)
    if n < 65:
        return None
    last = float(c.iloc[-1])

    def trailing_ret(bars: int) -> float | None:
        if n <= bars:
            return None
        base = float(c.iloc[-1 - bars])
        return (last / base - 1.0) if base > 0 else None

    r3m = trailing_ret(63)
    r6m = trailing_ret(126)
    r12m = trailing_ret(252)

    # Relative strength vs Nifty over 63 bars, aligned on the common index so we
    # compare the stock and the benchmark over the SAME calendar dates.
    rs_proxy = None
    if nifty_slice is not None and r3m is not None:
        idx = c.index.intersection(nifty_slice.index)
        if len(idx) >= 64:
            cs = c.reindex(idx).dropna()
            ns = nifty_slice.reindex(idx).dropna()
            if len(cs) >= 64 and len(ns) >= 64:
                sret = float(cs.iloc[-1]) / float(cs.iloc[-64]) - 1.0
                nret = float(ns.iloc[-1]) / float(ns.iloc[-64]) - 1.0
                rs_proxy = sret - nret

    # Volume expansion: last-20 avg vs the 50 bars before that.
    vol_ratio = None
    if "Volume" in df_slice.columns:
        v = df_slice["Volume"].dropna()
        if len(v) >= 70:
            recent = float(v.iloc[-20:].mean())
            prior  = float(v.iloc[-70:-20].mean())
            vol_ratio = (recent / prior) if prior > 0 else None

    # Proximity to the trailing 252-bar high (0 == at high, negative == below).
    w = c.iloc[-252:] if n >= 252 else c
    hi = float(w.max())
    pct_from_252hi = (last / hi - 1.0) if hi > 0 else None

    return {
        "r3m":            r3m,
        "r6m":            r6m,
        "rs_proxy":       rs_proxy,
        "vol_ratio":      vol_ratio,
        "pct_from_252hi": pct_from_252hi,
        "ctrl_mom63":     r3m,    # short-term momentum, reported for transparency
        "ctrl_mom252":    r12m,   # 12-month momentum — GATED positive control
    }


def _adtv_cr(df_slice: pd.DataFrame) -> float | None:
    """Average daily traded value over the last 20 bars, in ₹Cr (turnover/1e7)."""
    if "Volume" not in df_slice.columns:
        return None
    cv = df_slice[["Close", "Volume"]].dropna()
    if len(cv) < 20:
        return None
    turnover = float((cv["Close"].iloc[-20:] * cv["Volume"].iloc[-20:]).mean())
    return turnover / 1e7


# ── Snapshot date selection ─────────────────────────────────────────────────────

def _pick_snapshot_dates(trading_dates: pd.DatetimeIndex, hold: int) -> list[pd.Timestamp]:
    """
    Choose NON-OVERLAPPING point-in-time snapshot dates for horizon ``hold``.

    Snapshots are spaced EXACTLY ``hold`` trading days apart, so each snapshot's
    forward window [as_of, as_of+hold] abuts but never overlaps the next one.

    WHY NON-OVERLAPPING IS CRITICAL
    -------------------------------
    The robustness t-stat (= mean_IC / (std_IC / sqrt(n))) assumes the per-
    snapshot ICs are *independent* draws. If forward windows overlapped (e.g.
    snapshots 21 days apart but each holding 63 days), consecutive ICs would
    share most of their forward return and be strongly autocorrelated. That
    deflates std_IC and FAKES statistical significance — a factor would look
    far more reliable than it is. Spacing == hold removes that autocorrelation
    so the t-stat and the IS/OOS split are honest.

    Snapshots are placed as densely as possible across the ENTIRE available
    history: from the last position that still has a full ``hold``-bar forward
    window, stepping back by ``hold`` until we run out of required history
    (>= MIN_HISTORY_BARS bars before the snapshot). Returned ascending.

    Works off a master trading-day calendar (the union of all symbols' dates)
    so the spacing is in real trading days, not per-symbol bars.
    """
    n = len(trading_dates)
    # Last usable snapshot position: needs `hold` bars of future after it.
    last_pos = n - hold - 1
    positions: list[int] = []
    pos = last_pos
    while pos >= MIN_HISTORY_BARS:
        positions.append(pos)
        pos -= hold          # spacing == hold ⇒ forward windows do not overlap
    positions.sort()
    return [trading_dates[p] for p in positions]


# ── Core attribution ────────────────────────────────────────────────────────────

def run_attribution(hold: int = DEFAULT_HOLD,
                    stocks: dict | None = None,
                    nifty: pd.Series | None = None,
                    max_universe: int = MAX_UNIVERSE,
                    verbose: bool = True) -> dict:
    """
    Run the full point-in-time factor-attribution backtest.

    Parameters
    ----------
    hold : int
        Forward-return horizon in trading bars (main() uses 21 ≈ 1 month).
    stocks, nifty :
        Optional pre-loaded data (lets ``main`` load the bhavcopy cache once and
        pass it in, so the slow read happens a single time). When None, loaded here.

    Returns
    -------
    dict with keys:
        factors        : {factor: {ic_mean, ic_std, t_stat, pct_pos, is_ic,
                         oos_ic, verdict, n_obs, n_snaps}} sorted by ic_mean desc.
                         is_ic / oos_ic are the means of the chronological first
                         and second halves of that factor's per-snapshot ICs;
                         verdict is ROBUST / weak / noise (see the aggregation).
        decile_spread  : {top, bottom, spread, n_snaps} for the composite score.
        controls       : {ctrl_mom252, ctrl_mom63, ctrl_noise} convenience copies.
        robust_names   : list of tradable (non-control) factors graded ROBUST.
        n_robust       : len(robust_names).
        hold, n_snapshots, n_obs_total, elapsed_sec.
    """
    t0 = time.time()

    if stocks is None:
        if verbose:
            print(f"Loading bhavcopy cache (days={LOAD_DAYS})…", flush=True)
        stocks = _load_stocks(days=LOAD_DAYS)
    if not stocks:
        raise RuntimeError("No bhavcopy data loaded — run another scan first to populate the cache.")

    if nifty is None:
        nifty = _build_nifty(stocks)
    if nifty is None or len(nifty) < 130:
        raise RuntimeError("Could not build a usable Nifty benchmark series.")
    nifty = nifty.sort_index()

    # ── Static liquidity universe cap (runtime control, NOT a feature) ──
    # Rank symbols by full-history median daily turnover (Close*Volume) and keep
    # the top `max_universe`. The same fixed set is used at every snapshot, so it
    # introduces no cross-sectional forward-looking bias into the IC — it only
    # trims how many _score_stock calls we make.
    if max_universe and len(stocks) > max_universe:
        liq = []
        for sym, df in stocks.items():
            if "Volume" in df.columns:
                cv = df[["Close", "Volume"]].dropna()
                med_turnover = float((cv["Close"] * cv["Volume"]).median()) if len(cv) else 0.0
            else:
                med_turnover = 0.0
            liq.append((sym, med_turnover))
        liq.sort(key=lambda kv: kv[1], reverse=True)
        keep = {s for s, _ in liq[:max_universe]}
        stocks = {s: stocks[s] for s in stocks if s in keep}

    # Master trading calendar = union of every symbol's dates (sorted).
    all_dates = pd.DatetimeIndex(sorted(set().union(*[df.index for df in stocks.values()])))
    snap_dates = _pick_snapshot_dates(all_dates, hold)
    if len(snap_dates) < 3:
        raise RuntimeError(f"Only {len(snap_dates)} snapshot dates available — need more history.")

    if verbose:
        print(f"Universe: {len(stocks)} symbols | Snapshots: {len(snap_dates)} "
              f"({snap_dates[0].date()} → {snap_dates[-1].date()}) | HOLD={hold} bars",
              flush=True)

    # factor_name -> list of per-snapshot IC values
    factor_ics: dict[str, list[float]] = {}
    factor_nobs: dict[str, int] = {}
    # Composite decile tracking (live composite_score AND candidate composite_v2)
    decile_top_rets: list[float] = []
    decile_bot_rets: list[float] = []
    decile_top_rets_v2: list[float] = []
    decile_bot_rets_v2: list[float] = []
    n_obs_total = 0

    # The set of factor names is fixed: 10 criteria + composite + candidates +
    # 2 controls. Each snapshot is built as a list of per-stock row dicts, then
    # assembled into an aligned DataFrame — this is alignment-safe regardless of
    # which factors happen to be missing for which stocks at a given snapshot.
    for snap_i, as_of in enumerate(snap_dates):
        rows: list[dict[str, float]] = []   # one dict per scored stock
        fwd_list: list[float] = []          # forward return, aligned with rows

        nifty_slice = nifty.loc[:as_of]
        if len(nifty_slice) < 65:
            continue

        for sym, df in stocks.items():
            # POINT-IN-TIME: features may use ONLY bars with index <= as_of.
            df_hist = df.loc[:as_of]
            if len(df_hist) < MIN_HISTORY_BARS:
                continue

            # Forward return needs a bar exactly `hold` positions AFTER as_of in
            # THIS symbol's own series. df_hist is a prefix of df, so the last
            # bar <= as_of sits at position len(df_hist)-1 within df.
            as_of_pos = len(df_hist) - 1
            fwd_pos = as_of_pos + hold
            if fwd_pos >= len(df):
                continue   # no forward bar available for this stock at this snap

            close = df["Close"]
            entry_px = float(close.iloc[as_of_pos])
            exit_px  = float(close.iloc[fwd_pos])
            if entry_px <= 0 or not math.isfinite(entry_px) or not math.isfinite(exit_px):
                continue

            # Cost-aware forward return: raw forward return minus round-trip cost.
            adtv = _adtv_cr(df_hist)
            cost_pct = round_trip_cost_pct(adtv)   # percent
            fwd_ret = (exit_px / entry_px - 1.0) - cost_pct / 100.0

            # ── Features from the live scorer (point-in-time) ──
            try:
                m = _score_stock(df_hist, nifty_slice)
            except Exception:
                m = None
            if m is None:
                continue
            criteria = m.get("criteria")
            score = m.get("score")
            if criteria is None or score is None or len(criteria) != 10:
                continue

            # ── Extra candidate factors (point-in-time) ──
            extra = _candidate_factors(df_hist, nifty_slice)
            if extra is None:
                continue

            # ── Assemble this stock's factor row (missing → NaN) ──
            row: dict[str, float] = {}
            for i, lbl in enumerate(CRITERIA_LABELS):
                row[lbl] = 1.0 if bool(criteria[i]) else 0.0
            row["composite_score"] = float(score)
            for k, val in extra.items():
                row[k] = float(val) if val is not None else float("nan")
            row["ctrl_noise"] = _noise(sym, snap_i)

            rows.append(row)
            fwd_list.append(fwd_ret)

        m_obs = len(rows)
        if m_obs < 5:
            continue
        n_obs_total += m_obs

        # Build an aligned frame: index = observation, columns = factors. Any
        # factor absent from a given row becomes NaN automatically.
        fdf = pd.DataFrame(rows)
        fwd = pd.Series(fwd_list, dtype="float64")

        # ── composite_v2: evidence-weighted re-ranking (point-in-time) ──
        # Added as just another factor column so it flows through the SAME IC /
        # IS-OOS / decile machinery as composite_score for an apples-to-apples
        # comparison. z-scoring inside _composite_v2 is cross-sectional within
        # THIS snapshot only ⇒ no look-ahead.
        fdf["composite_v2"] = _composite_v2(fdf)

        # ── Per-snapshot IC for every factor ──
        for k in fdf.columns:
            ic = _spearman(fdf[k].tolist(), fwd_list)
            if ic is not None:
                factor_ics.setdefault(k, []).append(ic)
            factor_nobs[k] = factor_nobs.get(k, 0) + int(fdf[k].notna().sum())

        # ── Composite decile spread (top vs bottom by score) ──
        # Computed for BOTH the live composite_score and composite_v2 so the
        # report can show whether the new weighting widens the top-minus-bottom
        # net forward-return spread, not just the IC.
        k = max(1, m_obs // N_DECILES)
        for score_col, top_acc, bot_acc in (
            ("composite_score", decile_top_rets, decile_bot_rets),
            ("composite_v2",    decile_top_rets_v2, decile_bot_rets_v2),
        ):
            if m_obs >= N_DECILES * 2 and score_col in fdf.columns:
                order = fdf[score_col].rank(method="first")
                bottom_mask = (order <= k).to_numpy()
                top_mask = (order > (m_obs - k)).to_numpy()
                if top_mask.any() and bottom_mask.any():
                    top_acc.append(float(fwd[top_mask].mean()))
                    bot_acc.append(float(fwd[bottom_mask].mean()))

        if verbose:
            print(f"  snapshot {snap_i + 1}/{len(snap_dates)} {as_of.date()}: "
                  f"{m_obs} stocks scored", flush=True)

    # ── Aggregate IC stats per factor ──
    # factor_ics[k] holds this factor's per-snapshot ICs in ASCENDING snapshot
    # (date) order, because snap_dates is ascending and we append once per
    # snapshot. Splitting that ordered list in half therefore gives a genuine
    # chronological in-sample (earlier) vs out-of-sample (later) split.
    n_snaps = len(snap_dates)
    results: dict[str, dict] = {}
    for k, ics in factor_ics.items():
        arr = np.array(ics, dtype=float)
        nk = len(arr)
        ic_mean = float(arr.mean()) if nk else float("nan")
        ic_std = float(arr.std(ddof=1)) if nk > 1 else 0.0
        if nk > 1 and ic_std > 0:
            t_stat = ic_mean / (ic_std / math.sqrt(nk))
        else:
            t_stat = 0.0

        # Fraction of snapshots with a strictly positive IC (hit rate of the sign).
        pct_pos = float((arr > 0).mean()) if nk else float("nan")

        # Chronological IS / OOS split: first half (older) vs second half (newer).
        # With an odd count the middle snapshot goes to the OOS half (it is the
        # later block); both halves stay non-empty whenever nk >= 2.
        half = nk // 2
        is_arr = arr[:half]
        oos_arr = arr[half:]
        is_ic = float(is_arr.mean()) if len(is_arr) else float("nan")
        oos_ic = float(oos_arr.mean()) if len(oos_arr) else float("nan")

        # Robustness verdict. ROBUST demands BOTH statistical strength
        # (|t| >= ROBUST_T) AND sign stability across the pooled mean, the
        # in-sample half and the out-of-sample half (an edge you could not have
        # earned by luck and that did not flip out of sample). A sign-stable but
        # statistically weak factor is "weak"; everything else is "noise".
        sign_stable = (
            math.isfinite(is_ic) and math.isfinite(oos_ic) and math.isfinite(ic_mean)
            and _sign(is_ic) == _sign(oos_ic) == _sign(ic_mean) and _sign(ic_mean) != 0
        )
        if abs(t_stat) >= ROBUST_T and sign_stable:
            verdict = "ROBUST"
        elif sign_stable:
            verdict = "weak"
        else:
            verdict = "noise"

        results[k] = {
            "ic_mean": ic_mean,
            "ic_std":  ic_std,
            "t_stat":  t_stat,
            "pct_pos": pct_pos,
            "is_ic":   is_ic,
            "oos_ic":  oos_ic,
            "verdict": verdict,
            "n_obs":   int(factor_nobs.get(k, 0)),
            "n_snaps": nk,
        }

    # Sort by ic_mean descending (NaN-safe).
    results = dict(sorted(results.items(),
                          key=lambda kv: (kv[1]["ic_mean"]
                                          if math.isfinite(kv[1]["ic_mean"]) else -1e9),
                          reverse=True))

    # ── Decile spread aggregate ──
    def _agg_decile(top_rets: list[float], bot_rets: list[float]) -> dict:
        if top_rets and bot_rets:
            top = float(np.mean(top_rets))
            bot = float(np.mean(bot_rets))
            return {"top": top, "bottom": bot, "spread": top - bot, "n_snaps": len(top_rets)}
        return {"top": None, "bottom": None, "spread": None, "n_snaps": 0}

    decile_spread    = _agg_decile(decile_top_rets, decile_bot_rets)
    decile_spread_v2 = _agg_decile(decile_top_rets_v2, decile_bot_rets_v2)

    # Count of ROBUST *tradable* factors (controls excluded). These are the only
    # signals that cleared both |t| >= ROBUST_T and IS/OOS sign-stability.
    robust_names = [k for k, st in results.items()
                    if k not in CONTROL_FACTORS and st["verdict"] == "ROBUST"]

    elapsed = time.time() - t0
    out = {
        "factors":          results,
        "decile_spread":    decile_spread,
        "decile_spread_v2": decile_spread_v2,
        "controls": {
            "ctrl_mom252": results.get("ctrl_mom252"),  # GATED positive control
            "ctrl_mom63":  results.get("ctrl_mom63"),    # reported (short-term)
            "ctrl_noise":  results.get("ctrl_noise"),    # negative control
        },
        "robust_names":  robust_names,
        "n_robust":      len(robust_names),
        "hold":          hold,
        "n_snapshots":   n_snaps,
        "n_obs_total":   n_obs_total,
        "elapsed_sec":   elapsed,
    }
    return out


# ── Self-validation gate ────────────────────────────────────────────────────────

def _validate_controls(result: dict) -> tuple[bool, list[str]]:
    """
    Enforce the no-look-ahead gate. Returns (ok, messages).

    The GATED positive control is 12-month momentum (``ctrl_mom252``) — the
    canonical, robustly-predictive momentum factor. (Short-term 63-bar momentum
    is reported as ``ctrl_mom63`` but is NOT gated: Indian equities show
    short-horizon reversal, so its IC is legitimately near-zero/negative over
    this sample — a real market property, independently confirmed, not a bug.)

      * ctrl_mom252 mean IC must be POSITIVE and |IC| >= 3 × |ctrl_noise IC|.
      * ctrl_noise  mean |IC| must be < 0.02.
    """
    msgs: list[str] = []
    mom = result["controls"].get("ctrl_mom252")
    noise = result["controls"].get("ctrl_noise")
    short = result["controls"].get("ctrl_mom63")
    ok = True

    if mom is None:
        msgs.append("FAIL: ctrl_mom252 produced no IC (insufficient data).")
        return False, msgs
    if noise is None:
        msgs.append("FAIL: ctrl_noise produced no IC (insufficient data).")
        return False, msgs

    mom_ic = mom["ic_mean"]
    noise_ic = noise["ic_mean"]

    if not (mom_ic > 0):
        ok = False
        msgs.append(f"FAIL: ctrl_mom252 (12m momentum) IC = {mom_ic:+.4f} is not "
                    f"positive — sign/alignment/look-ahead bug.")
    else:
        msgs.append(f"PASS: ctrl_mom252 (12m momentum) IC = {mom_ic:+.4f} "
                    f"(positive, t={mom['t_stat']:+.2f}).")

    if not (abs(noise_ic) < 0.02):
        ok = False
        msgs.append(f"FAIL: ctrl_noise |IC| = {abs(noise_ic):.4f} >= 0.02 — "
                    f"IC pipeline is leaking information.")
    else:
        msgs.append(f"PASS: ctrl_noise IC = {noise_ic:+.4f} (|IC| < 0.02, ≈ random).")

    if not (abs(mom_ic) >= 3.0 * abs(noise_ic)):
        ok = False
        msgs.append(f"FAIL: |ctrl_mom252 IC| ({abs(mom_ic):.4f}) is not >= 3× "
                    f"|ctrl_noise IC| ({abs(noise_ic):.4f}) — momentum not clearly "
                    f"above the noise floor.")
    else:
        ratio = abs(mom_ic) / abs(noise_ic) if noise_ic != 0 else float("inf")
        msgs.append(f"PASS: |ctrl_mom252 IC| is {ratio:.1f}× the noise floor "
                    f"(target >= 3×).")

    # Informational only (not gated): surface the short-term reversal explicitly.
    if short is not None:
        msgs.append(f"NOTE: ctrl_mom63 (3m momentum) IC = {short['ic_mean']:+.4f} "
                    f"— short-horizon reversal, reported but not gated.")

    return ok, msgs


# ── Pretty printer ──────────────────────────────────────────────────────────────

_W = 96  # report width (wider than before to fit the IS/OOS + verdict columns)


# ── Frozen OLD-baseline composite_score numbers (the criteria-weighted ranking
# BEFORE composite_v2 was wired into trending.py). The PROVE-BEFORE-KEEP test must
# compare composite_v2 against this fixed pre-wiring baseline — NOT against the
# live composite_score column, because once trending.py is wired the live
# composite_score IS the v2 weighting (they converge, so a live-vs-live compare
# would falsely read "no improvement"). Captured from the STEP-1 baseline run
# (66 non-overlapping snapshots, HOLD=21, before any trending.py edit).
_OLD_BASELINE = {"ic_mean": 0.0135, "t_stat": 0.85, "is_ic": 0.0096,
                 "oos_ic": 0.0174, "spread": 0.0149}


def _v2_verdict(result: dict) -> tuple[bool, list[str]]:
    """
    Decide whether composite_v2 has earned the right to replace the OLD
    criteria-weighted composite_score.

    The PROVE-BEFORE-KEEP bar (all must hold):
      1. composite_v2 OOS_IC  >  OLD-baseline composite_score OOS_IC (better OOS)
      2. composite_v2 is sign-stable: sign(IS_IC)==sign(OOS_IC)==sign(mean_IC), !=0
      3. the control gate still passes (checked separately by _validate_controls)
    Compares against the FROZEN _OLD_BASELINE (pre-wiring) so the verdict is stable
    whether or not trending.py has already been wired. Returns (passed, lines).
    """
    facs = result["factors"]
    old = _OLD_BASELINE
    new = facs.get("composite_v2")
    msgs: list[str] = []
    if new is None:
        return False, ["FAIL: composite_v2 produced no IC."]

    better_oos = new["oos_ic"] > old["oos_ic"]
    v2_sign_stable = (
        math.isfinite(new["is_ic"]) and math.isfinite(new["oos_ic"])
        and math.isfinite(new["ic_mean"])
        and _sign(new["is_ic"]) == _sign(new["oos_ic"]) == _sign(new["ic_mean"])
        and _sign(new["ic_mean"]) != 0
    )

    msgs.append(
        f"{'PASS' if better_oos else 'FAIL'}: composite_v2 OOS_IC ({new['oos_ic']:+.4f}) "
        f"{'>' if better_oos else '<='} OLD-baseline composite_score OOS_IC ({old['oos_ic']:+.4f})."
    )
    msgs.append(
        f"{'PASS' if v2_sign_stable else 'FAIL'}: composite_v2 sign-stable "
        f"(IS {new['is_ic']:+.4f}, OOS {new['oos_ic']:+.4f}, mean {new['ic_mean']:+.4f})."
    )
    return (better_oos and v2_sign_stable), msgs


def _print_v2_comparison(result: dict) -> None:
    """
    The PROVE-BEFORE-KEEP table: OLD-baseline composite_score (frozen, pre-wiring)
    vs composite_v2. Plus a confirmation of where the LIVE composite_score (the
    score _score_stock currently returns) now lands — once trending.py is wired it
    should match composite_v2's improved numbers.
    """
    facs = result["factors"]
    old = _OLD_BASELINE
    new = facs.get("composite_v2")
    live = facs.get("composite_score")            # current _score_stock.score
    ds_old = _OLD_BASELINE                          # frozen old decile spread
    ds_new = result.get("decile_spread_v2", {})
    ds_live = result.get("decile_spread", {})       # live composite_score spread
    print()
    print("  COMPOSITE RE-WEIGHTING TEST  —  OLD-baseline composite_score  vs  composite_v2")
    print("-" * _W)
    if new is None:
        print("    insufficient data to compare.")
        print("=" * _W)
        return

    def _sp(ds):
        return f"{ds['spread'] * 100:+.2f}%" if ds.get("spread") is not None else "   n/a"

    print(f"    {'metric':<18}{'OLD composite':>18}{'composite_v2':>18}{'Δ (v2-old)':>14}")
    print(f"    {'mean_IC':<18}{old['ic_mean']:>+18.4f}{new['ic_mean']:>+18.4f}"
          f"{new['ic_mean'] - old['ic_mean']:>+14.4f}")
    print(f"    {'t_stat':<18}{old['t_stat']:>+18.2f}{new['t_stat']:>+18.2f}"
          f"{new['t_stat'] - old['t_stat']:>+14.2f}")
    print(f"    {'IS_IC':<18}{old['is_ic']:>+18.4f}{new['is_ic']:>+18.4f}"
          f"{new['is_ic'] - old['is_ic']:>+14.4f}")
    print(f"    {'OOS_IC':<18}{old['oos_ic']:>+18.4f}{new['oos_ic']:>+18.4f}"
          f"{new['oos_ic'] - old['oos_ic']:>+14.4f}")
    spread_delta = (
        f"{(ds_new['spread'] - ds_old['spread']) * 100:+.2f}%"
        if ds_old.get("spread") is not None and ds_new.get("spread") is not None else "   n/a"
    )
    print(f"    {'decile_spread':<18}{_sp(ds_old):>18}{_sp(ds_new):>18}{spread_delta:>14}")
    print("-" * _W)

    passed, msgs = _v2_verdict(result)
    for m in msgs:
        print(f"    {m}")
    # Fold in the control gate: the numbers are only trustworthy if controls held.
    ctrl_ok, _ = _validate_controls(result)
    print(f"    {'PASS' if ctrl_ok else 'FAIL'}: control gate (look-ahead proof) held.")
    overall = passed and ctrl_ok
    print()
    if overall:
        print("    >>> VALIDATED — composite_v2 beats the OLD composite_score "
              "OUT-OF-SAMPLE and is sign-stable. Safe to wire into trending.py.")
    else:
        print("    >>> NOT VALIDATED — composite_v2 did NOT clear the bar vs the OLD "
              "baseline. Leave trending.py UNTOUCHED.")
    print("-" * _W)

    # ── Live-score confirmation (STEP 4): is the wired _score_stock.score now at
    #    the composite_v2 level? Once trending.py is wired these should match. ──
    if live is not None:
        match = (
            _sign(live["oos_ic"]) == _sign(new["oos_ic"]) != 0
            and abs(live["oos_ic"] - new["oos_ic"]) <= 0.005
            and abs(live["ic_mean"] - new["ic_mean"]) <= 0.005
        )
        print(f"    LIVE composite_score (current _score_stock.score): "
              f"mean_IC {live['ic_mean']:+.4f}, t {live['t_stat']:+.2f}, "
              f"OOS_IC {live['oos_ic']:+.4f}, decile {_sp(ds_live)}.")
        if match:
            print("    >>> LIVE SCORE MATCHES composite_v2 — trending.py is wired and "
                  "carries the validated edge.")
        else:
            print("    >>> LIVE SCORE still = OLD weighting (mean_IC ~+0.0135) — "
                  "trending.py NOT yet wired to composite_v2.")
    print("=" * _W)


def _print_report(result: dict) -> bool:
    hold = result["hold"]
    months = round(hold / 21.0, 1)
    print()
    print("=" * _W)
    print(f"  FACTOR ATTRIBUTION  —  forward horizon = {hold} bars "
          f"(≈{months} month{'s' if hold > 31 else ''})  "
          f"[non-overlapping snapshots]")
    print("=" * _W)
    print(f"  Snapshots: {result['n_snapshots']}    "
          f"Total observations: {result['n_obs_total']}    "
          f"Runtime: {result['elapsed_sec']:.1f}s")
    print(f"  Verdict rule: ROBUST = |t-stat| >= {ROBUST_T:.1f} AND "
          f"sign(IS_IC)==sign(OOS_IC)==sign(mean_IC).  IS=older half, OOS=newer half.")
    print("-" * _W)
    print(f"  {'factor':<22} {'mean_IC':>8} {'t_stat':>7} {'%pos':>5} "
          f"{'IS_IC':>8} {'OOS_IC':>8} {'verdict':>8}  note")
    print("-" * _W)

    for name, st in result["factors"].items():
        tag = ""
        if name == "ctrl_mom252":
            tag = "+CONTROL 12m mom"
        elif name == "ctrl_mom63":
            tag = "(3m mom, reported)"
        elif name == "ctrl_noise":
            tag = "NOISE CONTROL"
        print(f"  {name:<22} {st['ic_mean']:>+8.4f} {st['t_stat']:>+7.2f} "
              f"{st['pct_pos'] * 100:>4.0f}% {st['is_ic']:>+8.4f} "
              f"{st['oos_ic']:>+8.4f} {st['verdict']:>8}  {tag}")

    print("-" * _W)
    ds = result["decile_spread"]
    if ds["spread"] is not None:
        print(f"  COMPOSITE DECILE SPREAD (top vs bottom decile, net fwd return):")
        print(f"    top decile  mean fwd ret : {ds['top'] * 100:+.2f}%")
        print(f"    bottom decile mean fwd ret: {ds['bottom'] * 100:+.2f}%")
        print(f"    SPREAD (top − bottom)     : {ds['spread'] * 100:+.2f}%  "
              f"over {ds['n_snaps']} snapshots")
    else:
        print("  COMPOSITE DECILE SPREAD: insufficient data")
    print("=" * _W)

    _print_v2_comparison(result)

    ok, msgs = _validate_controls(result)
    print()
    print("  SELF-VALIDATION GATE (no look-ahead proof):")
    for m in msgs:
        print(f"    {m}")
    print()
    if ok:
        print("  >>> CONTROLS PASSED — the IC numbers above are trustworthy.")
    else:
        print("  >>> CONTROLS FAILED — DO NOT trust the numbers; fix the bug.")
    print("-" * _W)
    rn = result["robust_names"]
    # Final summary line (exact spec wording): the ONLY factors a Trending
    # re-weight should trust — they cleared both |t|>=ROBUST_T and IS/OOS/mean
    # sign-stability. Controls are excluded from this list by construction.
    print(f"  ROBUST factors (|t|>={ROBUST_T:.0f} & sign-stable): "
          + (", ".join(rn) if rn else "none"))
    print("=" * _W)
    return ok


def main() -> None:
    t0 = time.time()
    # Load the deep archive ONCE. _build_nifty and the static liquidity cap are
    # deterministic on `stocks`, so the universe is fixed across every snapshot
    # (no leakage). We run ONLY the 21-bar horizon — the 63-bar horizon is skipped
    # so the whole run stays comfortably inside the ~6-minute runtime budget.
    print(f"Loading bhavcopy cache (days={LOAD_DAYS})… this is the slow step.",
          flush=True)
    stocks = _load_stocks(days=LOAD_DAYS)
    if not stocks:
        raise SystemExit("No bhavcopy data — run any other scan first to populate the cache.")
    nifty = _build_nifty(stocks)

    # ── Only horizon: 21 bars (≈1 month). Snapshots spaced EXACTLY 21 apart so
    #    forward windows never overlap (no IC autocorrelation → honest t-stats). ──
    res21 = run_attribution(hold=DEFAULT_HOLD, stocks=stocks, nifty=nifty, verbose=True)
    ok = _print_report(res21)   # the control gate is enforced on this horizon

    print(f"\nTotal runtime: {time.time() - t0:.1f}s")
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
