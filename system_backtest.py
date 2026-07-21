"""Unified portfolio-level walk-forward — the number that matters.

Factor ICs and per-trade stats measure PIECES. This measures the WHOLE system as
one portfolio: rank (sustained-breakout gate + momentum) → regime ladder gating
exposure → equity-curve brake → position deployment → costs → next-open fills, and
reports CAGR / max-drawdown / Sharpe of the resulting equity curve versus a Nifty
(NIFTYBEES) buy-and-hold over the SAME dates. Build once; it is the yardstick every
future change gets measured against.

Honest scope: this is a PERIODIC-REBALANCE simulation (rebalance every `rebal`
bars, hold to the next rebalance), not an intra-period event-driven stop simulator.
It is survivorship-free (delisted names included), point-in-time (every feature at
a rebalance uses only bars ≤ that date), next-open entry fills, and cost-aware.
"""

from __future__ import annotations
import numpy as np
import pandas as pd

import risk_engine
from costs import round_trip_cost_pct

TRADING_DAYS = 252


def _regime_from_bench(bench_vals: np.ndarray, pos: int) -> str:
    """Point-in-time regime proxy from the Nifty (NIFTYBEES) structure at `pos`.
    Uses only bars ≤ pos. Maps to the same labels risk_engine.regime_exposure knows.
    Not the exact IBD distribution-day method (that needs index volume) — a trend +
    drawdown proxy that captures the exposure ladder's intent."""
    if pos < 200:
        return "Unknown"
    window = bench_vals[: pos + 1]
    price = window[-1]
    ma50 = np.nanmean(window[-50:])
    ma200 = np.nanmean(window[-200:])
    peak = np.nanmax(window[-120:])          # recent 6-month peak
    dd = (price / peak - 1.0) * 100.0 if peak > 0 else 0.0
    if price > ma50 and ma50 > ma200 and dd > -5:
        return "Confirmed Uptrend"
    if price < ma200 or dd <= -12:
        return "Downtrend" if dd <= -12 else "Correction"
    return "Uptrend Under Pressure"


def _precompute(df: pd.DataFrame) -> dict | None:
    if len(df) < 260 or "Close" not in df:
        return None
    c = df["Close"].to_numpy(dtype=float)
    o = (df["Open"].to_numpy(dtype=float) if "Open" in df else c)
    v = (df["Volume"].to_numpy(dtype=float) if "Volume" in df else np.zeros_like(c))
    n = len(c)
    ma20  = pd.Series(c).rolling(20).mean().to_numpy()
    ma50  = pd.Series(c).rolling(50).mean().to_numpy()
    ma200 = pd.Series(c).rolling(200).mean().to_numpy()
    hi252 = pd.Series(c).rolling(252, min_periods=252).max().to_numpy()
    hi10  = pd.Series(c).rolling(10).max().to_numpy()
    adtv  = (pd.Series(c * v).rolling(20).mean() / 1e7).to_numpy()   # ₹Cr
    # days since the 252-day high, per bar (argmax over the trailing 252 window)
    dsh = np.full(n, 999)
    for i in range(251, n):
        w = c[i - 251: i + 1]
        dsh[i] = 251 - int(np.argmax(w))
    return {"idx": df.index, "close": c, "open": o, "n": n,
            "ma20": ma20, "ma50": ma50, "ma200": ma200,
            "hi252": hi252, "hi10": hi10, "adtv": adtv, "dsh": dsh}


def _passes_gate_and_score(f: dict, p: int) -> float | None:
    """Sustained-breakout gate at position p (point-in-time). Returns a momentum
    score for ranking if it passes, else None. Mirrors trending.sustained_breakout."""
    if p >= f["n"] or p < 252:
        return None
    price = f["close"][p]
    hi = f["hi252"][p]
    ma20, ma50, ma200 = f["ma20"][p], f["ma50"][p], f["ma200"][p]
    if not all(np.isfinite(x) and x > 0 for x in (price, hi, ma20, ma50, ma200)):
        return None
    pct_from_high = (price / hi - 1.0) * 100.0
    retrace_10d = (price / f["hi10"][p] - 1.0) * 100.0 if f["hi10"][p] > 0 else 0.0
    sustained = (f["dsh"][p] <= 25 and pct_from_high >= -8.0
                 and price > ma20 and retrace_10d > -6.0
                 and price > ma50 > ma200)
    if not sustained:
        return None
    # rank score = 3-month relative momentum (higher = stronger)
    if p < 63:
        return None
    r3m = price / f["close"][p - 63] - 1.0
    return float(r3m)


def live_momentum_picks(stocks: dict, top_n: int = 15) -> dict:
    """LIVE offense list — the SAME sustained-breakout + 3-month-momentum ranking
    the backtest's BULL leg uses, run on today's bar. Keeping live selection
    identical to the validated backtest is the whole point (no live/backtest drift).
    `stocks` = the shared _get_stocks() universe. Returns picks sorted strongest-first."""
    rows = []
    last_date = None
    for s, df in stocks.items():
        f = _precompute(df)
        if f is None:
            continue
        p = f["n"] - 1
        if last_date is None:
            last_date = str(f["idx"][-1].date())
        if f["adtv"][p] < 2.0:
            continue
        sc = _passes_gate_and_score(f, p)
        if sc is None:
            continue
        c = f["close"]
        r6m = (c[p] / c[p - 126] - 1.0) if p >= 126 and c[p - 126] > 0 else None
        pff = (c[p] / f["hi252"][p] - 1.0) * 100 if f["hi252"][p] > 0 else None
        rows.append({"symbol": s, "price": round(float(c[p]), 2),
                     "r3m": round(float(sc) * 100, 1),
                     "r6m": round(float(r6m) * 100, 1) if r6m is not None else None,
                     "pct_from_high": round(float(pff), 1) if pff is not None else None,
                     "adtv_cr": round(float(f["adtv"][p]), 1), "_s": float(sc)})
    rows.sort(key=lambda x: x["_s"], reverse=True)
    out = rows[:top_n]
    for i, r in enumerate(out):
        r["rank"] = i + 1
        r.pop("_s", None)
    return {"as_of": last_date, "count": len(rows), "stocks": out}


def run_system_backtest(days: int = 1600, rebal: int = 21, top_k: int = 10,
                        risk_pct: float = 0.75, max_position_pct: float = 10.0,
                        start_equity: float = 1_000_000.0,
                        strategy: str = "momentum", vol_target: bool = False,
                        mom_weight: float = 0.0, vol_filter: float = 0.0,
                        aw_offense: str = "defensive", aw_brake: bool = False,
                        aw_confirm_up: int = 5, aw_confirm_down: int = 5,
                        aw_side_gross: float = 1.0, aw_quality_weight: float = 0.0,
                        progress=None) -> dict:
    """strategy: 'momentum' (sustained-breakout gate) or 'defensive' (absolute-
    momentum gate + low-vol/smoothness/delivery composite). vol_target: weight picks
    by inverse volatility instead of equal weight (only used for defensive)."""
    from edge_engine import _load_stocks
    import benchmark as bm
    strategy = (strategy or "momentum").lower()
    _all_weather = strategy == "all_weather"
    _blend = strategy == "blend"
    _defensive = strategy in ("defensive", "defensive_mom")
    _mom_weight = mom_weight if strategy != "defensive_mom" else (mom_weight or 2.0)
    need_mom = strategy in ("momentum", "all_weather", "blend")
    need_def = strategy in ("defensive", "defensive_mom", "all_weather", "blend")
    if need_def:
        import defensive_scan as ds
    if _all_weather:
        import regime_engine as rg

    stocks_raw = _load_stocks(days=days, survivorship_free=True)
    bench = bm.get_benchmark(days=days)
    if bench is None or len(bench) < 260:
        return {"error": "benchmark unavailable"}

    # Both engines' point-in-time features. All-Weather / blend need both; the
    # pure strategies build only the one they use.
    feats_mom: dict = {}
    feats_def: dict = {}
    for s, df in stocks_raw.items():
        if need_mom:
            fm = _precompute(df)
            if fm is not None:
                feats_mom[s] = fm
        if need_def:
            fd = ds.precompute(df)
            if fd is not None:
                feats_def[s] = fd
    if not (feats_mom or feats_def):
        return {"error": "no usable symbols"}

    # Master calendar = benchmark's dates (the tradable sessions).
    cal = bench.index
    bench_vals = bench.to_numpy(dtype=float)
    # Rebalance dates: every `rebal` bars, leaving room for a forward hold.
    start = 252
    rebal_i = list(range(start, len(cal) - rebal - 1, rebal))
    if len(rebal_i) < 4:
        return {"error": "not enough history for a walk-forward"}

    # date → position lookups, per engine (both share each symbol's own calendar).
    sym_pos_mom = {s: {d: i for i, d in enumerate(f["idx"])} for s, f in feats_mom.items()}
    sym_pos_def = {s: {d: i for i, d in enumerate(f["idx"])} for s, f in feats_def.items()}

    # Point-in-time regime label per calendar bar (All-Weather only, look-ahead-free).
    reg_lbl = (rg.regime_series(bench_vals, confirm_up=aw_confirm_up,
                                confirm_down=aw_confirm_down)
               if _all_weather else None)
    _WIN_MW = getattr(ds, "WIN_MOM_WEIGHT", 4.0) if need_def else 0.0
    _WIN_VF = getattr(ds, "WIN_VOL_FILTER", 0.70) if need_def else 0.0
    # Quality map (current-snapshot fundamentals). CAVEAT: applying today's quality
    # to a historical rebalance is look-ahead + survivorship biased — treat quality
    # backtests as OPTIMISTIC, not survivorship-free proof. Delisted/unknown → 0.5.
    _qmap = {}
    if aw_quality_weight > 0 and need_def:
        try:
            import quality as _ql
            _qmap = _ql.load_quality_map()
        except Exception:
            _qmap = {}

    def _select_momentum(date):
        """Offense: sustained-breakout gate ranked by 3-month momentum → picks
        as (feats, position, weight)."""
        cand = []
        for s, f in feats_mom.items():
            p = sym_pos_mom[s].get(date)
            if p is None:
                continue
            sc = _passes_gate_and_score(f, p)
            if sc is not None and f["adtv"][p] >= 2.0:
                cand.append((f, p, sc))
        cand.sort(key=lambda x: x[2], reverse=True)
        top = cand[:top_k]
        return [(f, p, 1.0 / len(top)) for f, p, _ in top] if top else []

    def _select_defensive(date, mw, vf, vt_flag, qw=0.0):
        """Defense: absolute-momentum gate + low-vol/smoothness/delivery composite,
        after dropping the high-vol tail → picks as (feats, position, weight).
        qw>0 adds a QMJ-style fundamental-quality tilt (see _qmap caveat above)."""
        rows = []
        for s, f in feats_def.items():
            p = sym_pos_def[s].get(date)
            if p is None or f["adtv"][p] < 2.0:
                continue
            if not ds.passes_gate(f, p):
                continue
            rf = ds.raw_factors(f, p)
            if rf is None:
                continue
            row = {"sym": s, "p": p, **rf}
            if qw > 0:
                row["qual_raw"] = _qmap.get(s, 0.5)   # neutral if no fundamentals
            rows.append(row)
        if vf and len(rows) > top_k * 2:
            thr = np.quantile([r["vol90"] for r in rows], vf)
            rows = [r for r in rows if r["vol90"] <= thr]
        ranked = ds.rank_and_score(rows, mom_weight=mw, quality_weight=qw)[:top_k]
        if not ranked:
            return []
        if vt_flag:
            inv = np.array([1.0 / max(r["vol90"], 1e-6) for r in ranked])
            w = inv / inv.sum()
            return [(feats_def[r["sym"]], r["p"], float(w[i])) for i, r in enumerate(ranked)]
        return [(feats_def[r["sym"]], r["p"], 1.0 / len(ranked)) for r in ranked]

    equity = start_equity
    eq_curve = []            # (date, equity)
    bench_curve = []
    hwm = start_equity
    exposures = []
    n_trades = 0
    _trace = []              # per-rebalance diagnostics

    bench_units = start_equity / bench_vals[rebal_i[0]]   # buy-and-hold units

    n_switch = 0            # All-Weather engine switches
    prev_state = None
    regime_periods: dict = {}
    for k, ri in enumerate(rebal_i):
        date = cal[ri]
        # Equity-curve brake (uses equity so far) — a guardrail under every strategy.
        brake = risk_engine.equity_brake(equity, hwm)["multiplier"]

        if _all_weather:
            # Regime DECIDES the engine and the gross: offense in BULL, defense in
            # SIDEWAYS, cash in BEAR. This replaces the exposure ladder used by the
            # other strategies — the regime engine IS the risk manager here.
            state = reg_lbl[ri]
            regime = state
            if state == "BULL":
                # Offense engine is configurable: momentum OR the (stronger, on NSE)
                # defensive book. BULL just means "risk-on, full deployment".
                picks = (_select_defensive(date, _WIN_MW, _WIN_VF, True, aw_quality_weight)
                         if aw_offense == "defensive" else _select_momentum(date))
                exp = 1.0
            elif state == "SIDEWAYS":
                # Trend unresolved → same defensive book but a lighter gross
                # (aw_side_gross): take less risk when the trend is choppy.
                picks = _select_defensive(date, _WIN_MW, _WIN_VF, True, aw_quality_weight)
                exp = aw_side_gross
            else:                                   # BEAR / UNKNOWN → raise cash
                picks = []
                exp = 0.0
            # The regime IS the risk manager here. The equity brake is OFF by default
            # for All-Weather: stacked on the regime switch it doom-loops (one
            # drawdown clamps the book to 25% and, never making a new high, it never
            # re-risks — permanently missing the recovery). Opt in with aw_brake=True.
            deploy = exp * (brake if aw_brake else 1.0)
            if prev_state is not None and state != prev_state:
                n_switch += 1
            prev_state = state
            regime_periods[state] = regime_periods.get(state, 0) + 1
        else:
            # Static strategies keep the regime→exposure ladder + brake.
            if _blend:
                picks = ([(f, p, 0.5 * w) for f, p, w in _select_momentum(date)] +
                         [(f, p, 0.5 * w) for f, p, w in
                          _select_defensive(date, _WIN_MW, _WIN_VF, True)])
            elif _defensive:
                picks = _select_defensive(date, _mom_weight, vol_filter, vol_target)
            else:
                picks = _select_momentum(date)
            regime = _regime_from_bench(bench_vals, ri)
            exp = risk_engine.regime_exposure(regime)["exposure_pct"] / 100.0
            deploy = exp * brake                    # fraction of book put to work
        exposures.append(deploy)

        # Period return of each pick: next-OPEN entry → close at the next rebalance,
        # minus round-trip cost. Weighted by the strategy's weights.
        wret = []   # (weight, net_return)
        for f, p, wt in picks:
            fill = p + 1
            exit_p = p + 1 + rebal
            if exit_p >= f["n"]:
                continue
            entry_px = f["open"][fill]
            if not np.isfinite(entry_px) or entry_px <= 0:
                entry_px = f["close"][fill]
            exit_px = f["close"][exit_p]
            if not (np.isfinite(entry_px) and np.isfinite(exit_px) and entry_px > 0):
                continue
            cost = round_trip_cost_pct(f["adtv"][p]) / 100.0
            wret.append((wt, exit_px / entry_px - 1.0 - cost))
            n_trades += 1
        rets = [r for _, r in wret]

        # Weighted basket return (weights renormalised over the picks that filled).
        if wret:
            wsum = sum(w for w, _ in wret)
            mean_pick = (sum(w * r for w, r in wret) / wsum) if wsum > 0 else 0.0
        else:
            mean_pick = 0.0
        period_ret = (deploy * mean_pick) if wret else 0.0  # cash earns 0
        equity *= (1.0 + period_ret)
        hwm = max(hwm, equity)
        _trace.append({
            "date": str(date.date()), "regime": regime,
            "exp_pct": round(exp * 100, 0), "brake": brake,
            "deploy_pct": round(deploy * 100, 1), "n_picks": len(rets),
            "mean_pick_ret_pct": round(mean_pick * 100, 2),
            "period_ret_pct": round(period_ret * 100, 2),
            "equity": round(equity, 0),
            "dd_from_hwm_pct": round((equity / hwm - 1) * 100, 1),
        })
        eq_curve.append((str(date.date()), round(equity, 2)))

        # Benchmark buy-and-hold value at the NEXT rebalance close (same horizon).
        nxt = rebal_i[k + 1] if k + 1 < len(rebal_i) else min(ri + rebal, len(cal) - 1)
        bench_curve.append((str(cal[nxt].date()), round(bench_units * bench_vals[nxt], 2)))
        if progress:
            progress(k + 1, len(rebal_i), f"Rebalance {k+1}/{len(rebal_i)}")

    def _metrics(curve_vals: list[float], n_periods: int) -> dict:
        arr = np.array(curve_vals, dtype=float)
        if len(arr) < 2 or arr[0] <= 0:
            return {"cagr": 0.0, "max_dd": 0.0, "sharpe": 0.0, "total_return": 0.0}
        total = arr[-1] / arr[0] - 1.0
        yrs = (n_periods * rebal) / TRADING_DAYS
        cagr = ((arr[-1] / arr[0]) ** (1 / yrs) - 1) * 100 if yrs > 0 else 0.0
        peak = np.maximum.accumulate(arr)
        max_dd = float(np.min(arr / peak - 1.0)) * 100
        pr = arr[1:] / arr[:-1] - 1.0
        periods_per_yr = TRADING_DAYS / rebal
        sharpe = (float(np.mean(pr)) / float(np.std(pr)) * np.sqrt(periods_per_yr)
                  if np.std(pr) > 0 else 0.0)
        return {"cagr": round(cagr, 2), "max_dd": round(max_dd, 2),
                "sharpe": round(sharpe, 2), "total_return": round(total * 100, 2)}

    sys_vals = [start_equity] + [e for _, e in eq_curve]
    bch_vals = [start_equity] + [b for _, b in bench_curve]
    sysm = _metrics(sys_vals, len(eq_curve))
    bchm = _metrics(bch_vals, len(bench_curve))

    return {
        "as_of": eq_curve[-1][0] if eq_curve else None,
        "start": eq_curve[0][0] if eq_curve else None,
        "rebalances": len(eq_curve), "trades": n_trades,
        "avg_deployment_pct": round(float(np.mean(exposures)) * 100, 1) if exposures else 0.0,
        "params": {"rebal": rebal, "top_k": top_k, "risk_pct": risk_pct, "days": days,
                   "strategy": strategy},
        "regime_stats": ({"switches": n_switch, "periods": regime_periods}
                         if _all_weather else None),
        "system": sysm,
        "benchmark": bchm,
        "edge": {
            "cagr_delta": round(sysm["cagr"] - bchm["cagr"], 2),
            "dd_delta": round(sysm["max_dd"] - bchm["max_dd"], 2),   # less negative = better
        },
        "equity_curve": eq_curve,
        "benchmark_curve": bench_curve,
        "trace": _trace,
    }
