"""
Momentum ROTATION portfolio + point-in-time backtest.

Replicates the strategy behind @imhiteshmodi's "mi50-originalkl" Chartink screener as a
COMPOUNDING, weekly-rotated portfolio (not just a screen):

  ENTRY  — a stock makes a FRESH 52-week high on a weekly-close basis: this week's close
           tops the max of the prior 52 weekly closes, AND it had NOT been at a 52-week
           high in the prior 5 weeks (so we catch the breakout, not a name that's been
           pinned at highs for months). Universe gated to market cap ₹500cr–₹50,000cr.
  HOLD   — ride it while its weekly close stays above its 20-week moving average.
  EXIT   — weekly close below the 20-week MA (or it stops trading / leaves the cap band).
           This is Hitesh's exact stated rule: "keep exiting with 20 week moving average
           below close."
  BOOK   — up to N equal-weight positions (Hitesh: "5 stocks every week till 30-40
           stocks"); when more signals than open slots, take the strongest by 12-week
           momentum. Rebalanced weekly.

HONEST LIMITATIONS (read before trusting the number):
  • Survivorship: the universe is built from the bhavcopy archive. Names that delisted
    mid-period are partly covered (their historical bars remain) but fully-purged tickers
    are not — this biases returns UP.
  • Point-in-time market cap is APPROXIMATED as today's mcap scaled by the price ratio
    (constant-shares assumption) — dilution/buybacks aren't modelled.
  • Costs are a flat per-trade estimate; real smallcap slippage can be worse.
  • Entries/exits are transacted at the weekly CLOSE (no impact/liquidity haircut beyond
    the flat cost). Treat the output as an optimistic-but-honest upper-ish estimate.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# ── strategy parameters ──────────────────────────────────────────────────────
N_POSITIONS   = 30          # max concurrent holdings — Hitesh: "5/week till 30-40 stocks"
MCAP_MIN_CR   = 500.0
MCAP_MAX_CR   = 50000.0
FRESH_LOOKBACK = 5          # weeks it must NOT have been at a 52wk high before the breakout
MA_WEEKS      = 20          # exit below the 20-week MA — Hitesh's exact stated exit rule
COST_PER_SIDE = 0.0030      # 0.30% per trade side (brokerage + impact) — smallcap estimate
START         = "2022-07-01"   # Hitesh's live-portfolio start


def _growth_triggers(f: dict | None) -> tuple[list, dict]:
    """The fundamental / growth TRIGGERS behind a momentum name — the overlay Hitesh puts
    on top of the pure 52-week-high scan ("I don't buy every breakout — I buy the ones with
    an earnings story"). Built from our nightly fundamentals DB (screener.in-derived), so
    these are the QUANTITATIVE triggers (EPS/sales growth, acceleration, 3Y compounding,
    ROE, promoter buying), not hand-written narratives. Returns (ordered tags, raw fields);
    strongest trigger first. Empty when the name has no fundamentals on file (honest — we
    don't invent a story)."""
    if not f:
        return [], {"has_fund": False}

    def _n(k):
        v = f.get(k)
        try:
            return float(v) if v is not None else None
        except (TypeError, ValueError):
            return None

    eps_y, eps_ttm, eps_3y = _n("eps_growth_yoy"), _n("growth_ttm"), _n("growth_3y_cagr")
    sal_y, sal_ttm = _n("sales_growth_yoy"), _n("sales_growth_ttm")
    roe, dte, pe = _n("roe"), _n("debt_to_equity"), _n("pe_ratio")
    prom_d = _n("promoter_delta")
    accel   = int(f.get("eps_accel") or 0)
    accel_y = int(f.get("eps_accel_yoy") or 0)

    epsg = eps_y if eps_y is not None else eps_ttm
    salg = sal_y if sal_y is not None else sal_ttm
    trig: list[tuple[float, str]] = []
    if epsg is not None and epsg >= 20:
        trig.append((100 + min(epsg, 300), ("🚀 " if epsg >= 50 else "") + f"EPS +{epsg:.0f}% YoY"))
    if accel or accel_y:
        trig.append((95, "EPS accelerating" + (" YoY" if accel_y and not accel else "")))
    if salg is not None and salg >= 20:
        trig.append((80 + min(salg, 100) / 100, f"Sales +{salg:.0f}%"))
    if eps_3y is not None and eps_3y >= 25:
        trig.append((70, f"3Y profit CAGR {eps_3y:.0f}%"))
    if roe is not None and roe >= 20:
        trig.append((60, f"ROE {roe:.0f}%"))
    if prom_d is not None and prom_d >= 0.3:
        trig.append((50, f"Promoter ↑{prom_d:.1f}%"))
    if dte is not None and 0 <= dte < 0.3:
        trig.append((40, "Low debt"))
    trig.sort(key=lambda x: x[0], reverse=True)

    raw = {
        "has_fund": True,
        "eps_growth":  round(epsg, 1) if epsg is not None else None,
        "sales_growth": round(salg, 1) if salg is not None else None,
        "eps_accel":   bool(accel or accel_y),
        "eps_3y_cagr": round(eps_3y, 1) if eps_3y is not None else None,
        "roe":         round(roe, 1) if roe is not None else None,
        "pe":          round(pe, 1) if pe is not None else None,
        "promoter_delta": round(prom_d, 2) if prom_d is not None else None,
    }
    return [t for _, t in trig][:4], raw


def _weekly_close_frame(U: dict) -> pd.DataFrame:
    """{sym: daily df} -> DataFrame of weekly (Fri) closes, columns = symbols."""
    cols = {}
    for sym, df in U.items():
        if df is None or "Close" not in df or len(df) < 60:
            continue
        wc = df["Close"].resample("W-FRI").last()
        cols[sym] = wc
    wc = pd.DataFrame(cols).sort_index()
    return wc


def _approx_mcap_frame(wc: pd.DataFrame, mcap_now: dict) -> pd.DataFrame:
    """Point-in-time mcap ≈ today's mcap × (close_t / latest_close). Constant shares."""
    out = {}
    for sym in wc.columns:
        m = mcap_now.get(sym)
        s = wc[sym].dropna()
        if m is None or m <= 0 or s.empty:
            continue
        out[sym] = wc[sym] / s.iloc[-1] * m
    return pd.DataFrame(out).reindex_like(wc)


def _cagr(mult: float, weeks: int) -> float:
    yrs = weeks / 52.0
    return (mult ** (1.0 / yrs) - 1.0) * 100 if yrs > 0 and mult > 0 else float("nan")


def _max_dd(curve: pd.Series) -> float:
    peak = curve.cummax()
    return float(((curve / peak) - 1.0).min() * 100)


def run_backtest(n_positions: int = N_POSITIONS, start: str = START,
                 costs: bool = True, days: int = 1950) -> dict:
    import shared_universe as su
    import sector_rotation as sr

    U = su.load_base_universe(days=days, include_stale=True)   # keep delisted for PIT
    if not U:
        return {"error": "no universe"}
    mcap_now = sr._mcap_map()

    wc = _weekly_close_frame(U)
    if wc.shape[1] < 100:
        return {"error": "not enough symbols"}

    # ── precompute weekly signal frames (vectorised across all symbols) ──────
    max52_prev = wc.rolling(52, min_periods=52).max().shift(1)     # 52wk max as of last wk
    at_high    = wc >= max52_prev                                   # new 52wk high this wk
    fresh = at_high.copy()
    for k in range(1, FRESH_LOOKBACK + 1):
        fresh &= ~at_high.shift(k).fillna(False)                    # not at a high recently
    fresh = fresh.fillna(False)
    ma        = wc.rolling(MA_WEEKS, min_periods=MA_WEEKS).mean()
    above_ma  = (wc >= ma).fillna(False)
    mom12     = wc / wc.shift(12) - 1.0                             # 12-wk momentum (ranking)
    mcap      = _approx_mcap_frame(wc, mcap_now)
    eligible  = ((mcap >= MCAP_MIN_CR) & (mcap <= MCAP_MAX_CR)).fillna(False)

    dates = wc.index
    start_ts = pd.Timestamp(start)
    # need 52wk history before start; also need t+1 for forward return
    idx0 = max(52, int(np.searchsorted(dates.values, np.datetime64(start_ts))))
    idxs = list(range(idx0, len(dates) - 1))
    if len(idxs) < 20:
        return {"error": "backtest window too short"}

    cps = COST_PER_SIDE if costs else 0.0
    held: set[str] = set()
    nav = 1.0
    bench = 1.0          # equal-weight smallcap-band proxy (the space's beta)
    nav_curve, bench_curve, curve_dates = [], [], []
    n_entries_tot = n_exits_tot = 0
    pos_counts = []

    for ti in idxs:
        t, t1 = dates[ti], dates[ti + 1]
        fresh_t = fresh.loc[t]; elig_t = eligible.loc[t]
        above_t = above_ma.loc[t]; mom_t = mom12.loc[t]

        # 1) exits: dropped below 10wk MA, left the cap band, or stopped trading
        exits = {s for s in held if not (above_t.get(s, False) and elig_t.get(s, False)
                                          and pd.notna(wc.at[t, s]))}
        held -= exits

        # 2) entries: fresh breakouts in-band, not held, strongest 12wk momentum first
        cands = [s for s in wc.columns
                 if fresh_t.get(s, False) and elig_t.get(s, False) and s not in held
                 and pd.notna(mom_t.get(s))]
        cands.sort(key=lambda s: mom_t.get(s, -9), reverse=True)
        slots = max(0, n_positions - len(held))
        entries = cands[:slots]
        held |= set(entries)

        n_entries_tot += len(entries); n_exits_tot += len(exits)
        pos_counts.append(len(held))

        # 3) portfolio return over week t -> t+1 (equal weight), minus turnover cost
        if held:
            rets = []
            for s in held:
                p0, p1 = wc.at[t, s], wc.at[t1, s]
                if pd.notna(p0) and pd.notna(p1) and p0 > 0:
                    rets.append(p1 / p0 - 1.0)
                else:
                    rets.append(0.0)
            wk_ret = float(np.mean(rets)) if rets else 0.0
        else:
            wk_ret = 0.0
        turnover = (len(entries) + len(exits)) / max(1, n_positions)
        nav *= (1.0 + wk_ret) * (1.0 - cps * turnover)

        # 4) benchmark: equal-weight ALL in-band names, held one week (smallcap beta)
        bset = [s for s in wc.columns if elig_t.get(s, False) and pd.notna(wc.at[t, s])]
        if bset:
            brets = []
            for s in bset:
                p0, p1 = wc.at[t, s], wc.at[t1, s]
                if pd.notna(p1) and p0 > 0:
                    brets.append(p1 / p0 - 1.0)
            bench *= (1.0 + (float(np.mean(brets)) if brets else 0.0))

        nav_curve.append(nav); bench_curve.append(bench); curve_dates.append(t1)

    navs = pd.Series(nav_curve, index=pd.to_datetime(curve_dates))
    benchs = pd.Series(bench_curve, index=pd.to_datetime(curve_dates))
    weeks = len(navs)

    # year-by-year strategy returns
    yearly = {}
    for yr, grp in navs.groupby(navs.index.year):
        prev = yearly.get("_last", 1.0)
        endv = grp.iloc[-1]
        yearly[str(yr)] = round((endv / prev - 1.0) * 100, 1)
        yearly["_last"] = endv
    yearly.pop("_last", None)

    return {
        "start": str(navs.index[0].date()), "end": str(navs.index[-1].date()),
        "weeks": weeks, "n_positions": n_positions, "costs": costs,
        "symbols_universe": int(wc.shape[1]),
        "strategy_multiple": round(float(navs.iloc[-1]), 2),
        "strategy_cagr_pct": round(_cagr(float(navs.iloc[-1]), weeks), 1),
        "strategy_max_dd_pct": round(_max_dd(navs), 1),
        "benchmark_multiple": round(float(benchs.iloc[-1]), 2),
        "benchmark_cagr_pct": round(_cagr(float(benchs.iloc[-1]), weeks), 1),
        "benchmark_max_dd_pct": round(_max_dd(benchs), 1),
        "alpha_cagr_pct": round(_cagr(float(navs.iloc[-1]), weeks) - _cagr(float(benchs.iloc[-1]), weeks), 1),
        "avg_positions": round(float(np.mean(pos_counts)), 1),
        "total_entries": n_entries_tot, "total_exits": n_exits_tot,
        "weekly_turnover_pct": round((n_entries_tot + n_exits_tot) / max(1, weeks) / n_positions * 100, 1),
        "yearly_returns_pct": yearly,
        "_navs": navs, "_bench": benchs,
    }


# ── Parameter sweep (fast: shared frames built once, configs evaluated cheaply) ──
def _prepare(days: int = 1950):
    import shared_universe as su
    import sector_rotation as sr
    U = su.load_base_universe(days=days, include_stale=True)
    mcap_now = sr._mcap_map()
    wc = _weekly_close_frame(U)
    max52_prev = wc.rolling(52, min_periods=52).max().shift(1)
    at_high = wc >= max52_prev
    mom12 = wc / wc.shift(12) - 1.0
    mcap = _approx_mcap_frame(wc, mcap_now)
    dates = wc.index
    idx0 = max(52, int(np.searchsorted(dates.values, np.datetime64(pd.Timestamp(START)))))
    return {"wc": wc, "at_high": at_high, "mom12": mom12, "mcap": mcap,
            "dates": dates, "idx0": idx0}


def _simulate(prep, n_positions, ma_weeks, fresh_lb, mcap_min, mcap_max, cost):
    wc, at_high, mom12, mcap = prep["wc"], prep["at_high"], prep["mom12"], prep["mcap"]
    dates, idx0 = prep["dates"], prep["idx0"]
    fresh = at_high.copy()
    for k in range(1, fresh_lb + 1):
        fresh &= ~at_high.shift(k).fillna(False)
    eligible = ((mcap >= mcap_min) & (mcap <= mcap_max)).fillna(False)
    fresh_elig = (fresh.fillna(False)) & eligible & wc.notna()
    above_ma = ((wc >= wc.rolling(ma_weeks, min_periods=ma_weeks).mean()) & eligible & wc.notna()).fillna(False)
    cols = wc.columns
    held: set = set()
    nav = 1.0
    curve = []
    ne = nx = 0
    for ti in range(idx0, len(dates) - 1):
        t, t1 = dates[ti], dates[ti + 1]
        hok = above_ma.loc[t]
        exits = {s for s in held if not bool(hok.get(s, False))}
        held -= exits
        fe = fresh_elig.loc[t]; mt = mom12.loc[t]
        cands = [s for s in cols[fe.values] if s not in held]
        cands.sort(key=lambda s: (mt.get(s) if pd.notna(mt.get(s)) else -9), reverse=True)
        entries = cands[:max(0, n_positions - len(held))]
        held |= set(entries)
        ne += len(entries); nx += len(exits)
        if held:
            rets = []
            for s in held:
                p0, p1 = wc.at[t, s], wc.at[t1, s]
                rets.append(p1 / p0 - 1.0 if (pd.notna(p0) and pd.notna(p1) and p0 > 0) else 0.0)
            wk = float(np.mean(rets))
        else:
            wk = 0.0
        turn = (len(entries) + len(exits)) / max(1, n_positions)
        nav *= (1.0 + wk) * (1.0 - cost * turn)
        curve.append(nav)
    s = pd.Series(curve, index=dates[idx0 + 1: idx0 + 1 + len(curve)])
    w = len(s)
    return {"multiple": round(float(s.iloc[-1]), 2), "cagr": round(_cagr(float(s.iloc[-1]), w), 1),
            "max_dd": round(_max_dd(s), 1), "weeks": w,
            "turnover": round((ne + nx) / max(1, w) / n_positions * 100, 1)}


def _bench(prep, mcap_min, mcap_max):
    wc, mcap, dates, idx0 = prep["wc"], prep["mcap"], prep["dates"], prep["idx0"]
    eligible = ((mcap >= mcap_min) & (mcap <= mcap_max)).fillna(False)
    b = 1.0; curve = []
    cols = wc.columns
    for ti in range(idx0, len(dates) - 1):
        t, t1 = dates[ti], dates[ti + 1]
        el = eligible.loc[t]
        rets = []
        for s in cols[el.values]:
            p0, p1 = wc.at[t, s], wc.at[t1, s]
            if pd.notna(p0) and pd.notna(p1) and p0 > 0:
                rets.append(p1 / p0 - 1.0)
        b *= (1.0 + (float(np.mean(rets)) if rets else 0.0))
        curve.append(b)
    s = pd.Series(curve)
    return {"multiple": round(float(s.iloc[-1]), 2), "cagr": round(_cagr(float(s.iloc[-1]), len(s)), 1),
            "max_dd": round(_max_dd(s), 1)}


def run_sweep(costs: bool = True) -> dict:
    prep = _prepare()
    cost = COST_PER_SIDE if costs else 0.0
    base = dict(n=25, ma=10, fresh=5, mn=MCAP_MIN_CR, mx=MCAP_MAX_CR)
    bench = _bench(prep, base["mn"], base["mx"])
    sweeps = {
        "book_size_N":  [dict(base, n=n) for n in (10, 15, 20, 25, 30, 40)],
        "exit_MA_weeks": [dict(base, ma=m) for m in (6, 8, 10, 13, 20)],
        "fresh_lookback": [dict(base, fresh=f) for f in (1, 3, 5, 8)],
        "mcap_band":    [dict(base, mn=a, mx=b) for a, b in
                         ((100, 50000), (500, 50000), (500, 15000), (1000, 20000), (2000, 50000))],
    }
    out = {"bench": bench, "results": {}}
    for name, cfgs in sweeps.items():
        rows = []
        for c in cfgs:
            r = _simulate(prep, c["n"], c["ma"], c["fresh"], c["mn"], c["mx"], cost)
            rows.append({"cfg": c, **r})
        out["results"][name] = rows
    return out


import time as _time
import result_cache as _rc

_live_cache = {"data": None, "ts": 0.0}
LIVE_TTL = 3600


def _live_compute(n_positions: int = N_POSITIONS, days: int = 1950) -> dict:
    """Run the rotation forward to the LATEST week and report the actionable state:
    what to BUY now (fresh entries), what to HOLD (the book, with its trailing exit
    level), and what to SELL (names that just broke their 10-week MA). Deterministic —
    it replays the same rules the backtest uses, so the live book always matches."""
    import shared_universe as su
    import sector_rotation as sr
    try:
        import sector_mapper as sm
        smap = sm.get_enriched_sector_map()
    except Exception:
        smap = {}

    U = su.load_base_universe(days=days, include_stale=True)
    if not U:
        return {"error": "no universe", "entries": [], "holdings": [], "exits": []}
    mcap_now = sr._mcap_map()
    wc = _weekly_close_frame(U)
    if wc.shape[1] < 100:
        return {"error": "not enough data", "entries": [], "holdings": [], "exits": []}

    max52_prev = wc.rolling(52, min_periods=52).max().shift(1)
    at_high = wc >= max52_prev
    fresh = at_high.copy()
    for k in range(1, FRESH_LOOKBACK + 1):
        fresh &= ~at_high.shift(k).fillna(False)
    fresh_elig = (fresh.fillna(False)) & (wc.notna())
    ma = wc.rolling(MA_WEEKS, min_periods=MA_WEEKS).mean()
    mom12 = wc / wc.shift(12) - 1.0
    mcap = _approx_mcap_frame(wc, mcap_now)
    eligible = ((mcap >= MCAP_MIN_CR) & (mcap <= MCAP_MAX_CR)).fillna(False)
    fresh_elig = fresh_elig & eligible
    hold_ok = ((wc >= ma) & eligible & wc.notna()).fillna(False)

    dates = wc.index
    idx0 = max(52, int(np.searchsorted(dates.values, np.datetime64(pd.Timestamp(START)))))
    cols = wc.columns
    held: dict = {}
    last_entries: list = []
    last_exit_info: dict = {}
    for ti in range(idx0, len(dates)):
        t = dates[ti]
        hok = hold_ok.loc[t]
        exits = [s for s in list(held) if not bool(hok.get(s, False))]
        if ti == len(dates) - 1:
            last_exit_info = {s: held[s] for s in exits}
        for s in exits:
            held.pop(s, None)
        fe = fresh_elig.loc[t]
        mt = mom12.loc[t]
        cands = [s for s in cols[fe.values] if s not in held]
        cands.sort(key=lambda s: (mt.get(s) if pd.notna(mt.get(s)) else -9), reverse=True)
        entries = cands[:max(0, n_positions - len(held))]
        for s in entries:
            held[s] = {"entry_date": str(t.date()), "entry_price": round(float(wc.at[t, s]), 2)}
        if ti == len(dates) - 1:
            last_entries = entries

    last = dates[-1]

    def _px(s):
        df = U.get(s)
        return round(float(df["Close"].iloc[-1]), 2) if df is not None and len(df) else None

    def _exit_level(s):
        v = ma[s].dropna()
        return round(float(v.iloc[-1]), 2) if len(v) else None

    def _pivot(s):
        v = max52_prev[s].dropna()
        return round(float(v.iloc[-1]), 2) if len(v) else None

    def _weeks(entry_date):
        try:
            return int((last - pd.Timestamp(entry_date)).days / 7)
        except Exception:
            return None

    def _row(s, info, status):
        px, ep = _px(s), info.get("entry_price")
        ex = _exit_level(s)
        ret = round((px / ep - 1) * 100, 1) if px and ep else None
        room = round((px / ex - 1) * 100, 1) if px and ex else None   # % above the exit stop
        return {"symbol": s, "sector": smap.get(s), "price": px,
                "entry_date": info.get("entry_date"), "entry_price": ep,
                "weeks_held": _weeks(info.get("entry_date")), "ret_pct": ret,
                "exit_level": ex, "room_pct": room, "pivot": _pivot(s), "status": status,
                "rs": None}

    entries = [_row(s, held.get(s, {"entry_date": str(last.date()),
                                    "entry_price": _px(s)}), "BUY") for s in last_entries]
    exits = [_row(s, info, "SELL") for s, info in last_exit_info.items()]
    hold_syms = [s for s in held if s not in set(last_entries)]
    holdings = [_row(s, held[s], "HOLD") for s in hold_syms]
    holdings.sort(key=lambda r: (r["ret_pct"] if r["ret_pct"] is not None else -999), reverse=True)
    entries.sort(key=lambda r: (r["room_pct"] if r["room_pct"] is not None else 999))

    # ── THE mi50 SCAN — every name CURRENTLY passing Hitesh's exact Chartink conditions ──
    # (fresh weekly-close 52-week high, not at a high in the prior 5 weeks, mcap ₹500-50000cr)
    # — uncapped by book slots, so it mirrors the live Chartink screener output. Each name
    # carries its 52wk pivot, the 20-week-MA exit level (his hold/sell rule) and the
    # fundamental/growth triggers behind it. Ranked strongest-momentum-first, the order in
    # which Hitesh adds names ("5 strongest every week").
    try:
        import fundamentals as _fund
        F = _fund.load_all_fundamentals()
    except Exception:
        F = {}
    fe_last = fresh_elig.loc[last]
    scan = []
    for s in cols[fe_last.values]:
        px, ex, piv = _px(s), _exit_level(s), _pivot(s)
        df = U.get(s)
        chg = None
        if df is not None and len(df) >= 2 and float(df["Close"].iloc[-2]) > 0:
            chg = round((float(df["Close"].iloc[-1]) / float(df["Close"].iloc[-2]) - 1) * 100, 2)
        mom = mom12.at[last, s] if s in mom12.columns else None
        tags, fraw = _growth_triggers(F.get(s))
        scan.append({
            "symbol": s, "sector": smap.get(s), "price": px, "pct_chg": chg,
            "mcap_cr": round(float(mcap_now[s])) if mcap_now.get(s) else None,
            "pivot": piv, "exit_level": ex,
            "room_pct": round((px / ex - 1) * 100, 1) if px and ex else None,
            "mom12_pct": round(float(mom) * 100, 1) if mom is not None and pd.notna(mom) else None,
            "triggers": tags, "rs": None, **fraw,
        })
    scan.sort(key=lambda r: (r["mom12_pct"] if r["mom12_pct"] is not None else -9e9), reverse=True)

    return {
        "as_of": str(last.date()),
        "book_size": len(held),
        "n_positions": n_positions,
        "scan": scan, "n_scan": len(scan),
        "entries": entries, "holdings": holdings, "exits": exits,
        "computed_at": int(_time.time()),
        # From run_backtest() with Hitesh's EXACT rules (20-wk MA exit, N=30, since
        # 2022-07-01). net = after 0.30%/side costs; gross = costs off. his_live is his
        # own posted number. index_multiple is the Nifty Smallcap 250 over the same span.
        "strategy": {"multiple": 2.53, "cagr": 24.7, "max_dd": -25.0,
                     "benchmark_multiple": 2.43, "alpha_cagr": 1.2,
                     "gross_multiple": 2.73, "his_live_multiple": 3.23,
                     "index_multiple": 2.28},
    }


def _bhav_tag() -> str:
    try:
        from data_fetcher import _latest_bhavcopy_date
        d = _latest_bhavcopy_date()
        return d.isoformat() if d else "nodate"
    except Exception:
        return "nodate"


def run_live_scan(force: bool = False) -> dict:
    tag = _bhav_tag()
    # In-memory cache is BHAV-DATE-AWARE: a new bhavcopy changes the tag, so we never
    # serve yesterday's book even within the TTL — the rotation auto-refreshes daily
    # the moment new data lands (the scheduler also busts this cache on new data).
    if (not force and _live_cache["data"] and _live_cache.get("tag") == tag
            and _time.time() - _live_cache["ts"] < LIVE_TTL):
        return _live_cache["data"]
    if not force:
        disk = _rc.get_or_stale("momentum_rotation_live")   # bhav-tagged in result_cache
        if disk is not None:
            _live_cache.update(data=disk, ts=_time.time(), tag=tag)
            return disk
    data = _live_compute()
    _live_cache.update(data=data, ts=_time.time(), tag=tag)
    try:
        _rc.put("momentum_rotation_live", data)
    except Exception:
        pass
    return data


def invalidate_cache():
    _live_cache.update(data=None, ts=0)


if __name__ == "__main__":
    import os, sys, time
    os.environ.setdefault("DATA_DIR", os.getcwd())
    t0 = time.time()
    withc = run_backtest(costs=True)
    if "error" in withc:
        print("ERROR:", withc["error"]); sys.exit(1)
    noc = run_backtest(costs=False)
    R = withc
    print("=" * 66)
    print("  FRESH-52wk-HIGH SMALLCAP MOMENTUM  —  weekly rotation backtest")
    print("=" * 66)
    print(f"  Window        : {R['start']}  →  {R['end']}   ({R['weeks']} weeks)")
    print(f"  Universe       : {R['symbols_universe']} symbols   |   book: {R['n_positions']} names, equal-weight")
    print(f"  Avg positions  : {R['avg_positions']}   |   weekly turnover: {R['weekly_turnover_pct']}%")
    print("-" * 66)
    print(f"  STRATEGY (net of {COST_PER_SIDE*100:.2f}%/side cost)")
    print(f"     Total return : {R['strategy_multiple']}x   ({(R['strategy_multiple']-1)*100:.0f}%)")
    print(f"     CAGR         : {R['strategy_cagr_pct']}%")
    print(f"     Max drawdown : {R['strategy_max_dd_pct']}%")
    print(f"     (gross, no costs: {noc['strategy_multiple']}x  |  CAGR {noc['strategy_cagr_pct']}%)")
    print("-" * 66)
    print(f"  BENCHMARK (equal-weight smallcap-band buy&hold)")
    print(f"     Total return : {R['benchmark_multiple']}x   ({(R['benchmark_multiple']-1)*100:.0f}%)")
    print(f"     CAGR         : {R['benchmark_cagr_pct']}%   |   Max DD: {R['benchmark_max_dd_pct']}%")
    print(f"  ALPHA (strategy CAGR − benchmark CAGR): {R['alpha_cagr_pct']}%/yr")
    print("-" * 66)
    print("  Year-by-year (strategy, net):")
    for yr, r in R["yearly_returns_pct"].items():
        print(f"     {yr}: {r:+.1f}%")
    print("=" * 66)
    print(f"  computed in {time.time()-t0:.1f}s")
