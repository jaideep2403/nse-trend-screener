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

# ── Accuracy corrections (added 2026-07-26) ──────────────────────────────────
# Three known biases were quantified in the 2026-07-25 audit and left unfixed; all
# three are corrected here so the headline number describes a portfolio a real
# investor could actually have held.
#
#   1. CASH EARNED 0%. All-Weather sits in cash for ~19% of the sample (the whole
#      point of the regime switch), and un-deployed capital was earning nothing.
#      A real INR investor parks it in a liquid fund / T-bill. Understated the
#      strategy by ~1.2%/yr — and understated it MOST in exactly the bear periods
#      the strategy exists to survive.
#   2. NO DIVIDEND ADJUSTMENT. NIFTYBEES is a TOTAL-return proxy (the ETF
#      reinvests index dividends) but stock returns come from PRICE-only bhavcopy.
#      Comparing the two handicapped every strategy by ~the Nifty yield (~1.3%/yr).
#   3. SHARPE HAD NO RISK-FREE LEG. It was return/vol, which flatters any strategy
#      that holds cash — you got credit for the zero-volatility cash leg without
#      paying for the fact that cash has an opportunity cost.
#
# One rate drives (1) and (3): for a domestic INR investor the rate idle cash
# earns IS the risk-free rate. ~6% ≈ the RBI repo / liquid-fund band over the
# sample. It is an ASSUMPTION, not measured data — tune it here and re-baseline.
RISK_FREE_ANNUAL = 0.06


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
                        aw_flow_weight: float = 0.0, aw_promoter_weight: float = 0.0,
                        port_vol_target: float = 0.0, port_vol_lookback: int = 60,
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
    # Institutional-flow factor. UNLIKE quality, this IS point-in-time: the score
    # for a rebalance date is read from F&O files dated ≤ that date, so it carries
    # no look-ahead. History starts ~Jan 2024 (NSE unified archive), so on earlier
    # rebalances score_map_asof() returns {} and every name falls back to NEUTRAL.
    _iflow = None
    if aw_flow_weight > 0 and need_def:
        try:
            import institutional_flow as _if
            _if.build_score_panel()
            _iflow = _if
        except Exception:
            _iflow = None
    # Promoter/insider accumulation — also genuinely point-in-time (keyed on the
    # DISCLOSURE date, not the trade date), so honestly backtestable.
    _pflow = None
    if aw_promoter_weight > 0 and need_def:
        try:
            import promoter_flow as _pf
            _pf.build_score_panel()
            _pflow = _pf
        except Exception:
            _pflow = None

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

    def _select_defensive(date, mw, vf, vt_flag, qw=0.0, fw=0.0, pw=0.0):
        """Defense: absolute-momentum gate + low-vol/smoothness/delivery composite,
        after dropping the high-vol tail → picks as (feats, position, weight).
        qw>0 adds a QMJ-style fundamental-quality tilt (see _qmap caveat above)."""
        # Point-in-time institutional-flow map for THIS rebalance date only.
        fmap = _iflow.score_map_asof(date) if (fw > 0 and _iflow is not None) else {}
        pmap = _pflow.score_map_asof(date) if (pw > 0 and _pflow is not None) else {}
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
            if fw > 0:
                row["flow_raw"] = fmap.get(s, 0.5)    # neutral if no listed futures
            if pw > 0:
                row["prom_raw"] = pmap.get(s, 0.5)    # neutral if no disclosures
            rows.append(row)
        if vf and len(rows) > top_k * 2:
            thr = np.quantile([r["vol90"] for r in rows], vf)
            rows = [r for r in rows if r["vol90"] <= thr]
        ranked = ds.rank_and_score(rows, mom_weight=mw, quality_weight=qw,
                                   flow_weight=fw, promoter_weight=pw)[:top_k]
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
    # Per-rebalance cash return on the un-deployed fraction (accuracy fix #1).
    cash_period = (1.0 + RISK_FREE_ANNUAL) ** (rebal / TRADING_DAYS) - 1.0

    period_rets: list[float] = []   # realised period returns, for vol targeting

    daily_curve: list[tuple[str, float]] = []   # true bar-by-bar equity path

    def _vol_scale() -> float:
        """Portfolio volatility targeting — scale gross exposure so REALISED vol
        tracks `port_vol_target` (annualised fraction, e.g. 0.12).

        Estimated from the DAILY equity path, not from monthly period returns. The
        first version used the last 6 period returns; with only 6 points spanning
        6 months the estimate barely moved, so vol targeting had almost no effect
        here (24.29% -> 24.21% CAGR, drawdown unchanged) even though the same idea
        cut drawdown by 5.6-7.9pp in a daily-marked harness. Volatility targeting
        needs daily resolution to work at all.

        Capped at 1.0 — never lever up. Levering a strategy whose edge is still
        contested turns a modest loss into a ruinous one.
        """
        if port_vol_target <= 0:
            return 1.0
        lb = max(20, int(port_vol_lookback))
        if len(daily_curve) < lb + 2:
            return 1.0
        _v = np.array([v for _, v in daily_curve[-(lb + 1):]], dtype=float)
        _r = _v[1:] / _v[:-1] - 1.0
        _sd = float(np.std(_r))
        if _sd <= 1e-9:
            return 1.0
        ann = _sd * np.sqrt(TRADING_DAYS)
        return float(min(1.0, port_vol_target / ann))

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
                picks = (_select_defensive(date, _WIN_MW, _WIN_VF, True,
                                           aw_quality_weight, aw_flow_weight,
                                           aw_promoter_weight)
                         if aw_offense == "defensive" else _select_momentum(date))
                exp = 1.0
            elif state == "SIDEWAYS":
                # Trend unresolved → same defensive book but a lighter gross
                # (aw_side_gross): take less risk when the trend is choppy.
                picks = _select_defensive(date, _WIN_MW, _WIN_VF, True,
                                          aw_quality_weight, aw_flow_weight,
                                          aw_promoter_weight)
                exp = aw_side_gross
            else:                                   # BEAR / UNKNOWN → raise cash
                picks = []
                exp = 0.0
            # The regime IS the risk manager here. The equity brake is OFF by default
            # for All-Weather: stacked on the regime switch it doom-loops (one
            # drawdown clamps the book to 25% and, never making a new high, it never
            # re-risks — permanently missing the recovery). Opt in with aw_brake=True.
            deploy = exp * (brake if aw_brake else 1.0) * _vol_scale()
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
                # CONSISTENCY-FIX (2026-07-25): the standalone `defensive` strategy
                # must be backtested with the SAME weights the LIVE Defensive-Leaders
                # scan uses, or the tab's published numbers describe a different
                # engine than the one picking today's stocks. run_defensive_scan()
                # defaults to WIN_QUAL_WEIGHT / WIN_FLOW_WEIGHT, so mirror them here
                # (callers can still override via aw_quality_weight/aw_flow_weight).
                _dqw = aw_quality_weight if aw_quality_weight else getattr(ds, "WIN_QUAL_WEIGHT", 0.0)
                _dfw = aw_flow_weight if aw_flow_weight else getattr(ds, "WIN_FLOW_WEIGHT", 0.0)
                _dpw = aw_promoter_weight if aw_promoter_weight else getattr(ds, "WIN_PROMOTER_WEIGHT", 0.0)
                picks = _select_defensive(date, _mom_weight, vol_filter, vol_target,
                                          _dqw, _dfw, _dpw)
            else:
                picks = _select_momentum(date)
            regime = _regime_from_bench(bench_vals, ri)
            exp = risk_engine.regime_exposure(regime)["exposure_pct"] / 100.0
            deploy = exp * brake * _vol_scale()     # fraction of book put to work
        exposures.append(deploy)

        # Period return of each pick: next-OPEN entry → close at the next rebalance,
        # minus round-trip cost. Weighted by the strategy's weights.
        wret = []   # (weight, net_return)
        _pick_paths = []   # (weight, entry_px, feats, cost) for the daily path
        # The exit is the NEXT REBALANCE DATE on the master calendar.
        _next_date = cal[min(ri + rebal, len(cal) - 1)]
        for f, p, wt in picks:
            fill = p + 1
            # HOLDING-PERIOD FIX (2026-07-31): this used to be `exit_p = p + 1 + rebal`,
            # which counts 21 bars along the SYMBOL'S OWN index. For any name with
            # gaps (thin trading, halts, suspensions) 21 of its bars can span months
            # of calendar time, so the position was held far longer than one rebalance
            # while its return was booked as a single monthly period. Measured on the
            # real universe: 3.17% of (symbol, rebalance) pairs drifted >3 days, with
            # a MEDIAN drift of +76 days and a maximum of +1630. That silently
            # stretched holding periods for exactly the illiquid names and corrupted
            # both CAGR and the drawdown series. Anchor the exit to the calendar date
            # instead: the last bar the symbol actually traded on or before the next
            # rebalance.
            exit_p = int(f["idx"].searchsorted(_next_date, side="right")) - 1
            if fill >= f["n"]:
                continue          # no entry bar at all — nothing was ever bought
            if exit_p <= fill:
                continue          # symbol did not trade between fill and next rebalance
            if exit_p >= f["n"]:
                # AUDIT-FIX (2026-07-25): this used to `continue`, silently DROPPING
                # the position — which quietly re-introduced survivorship bias into a
                # survivorship-free backtest (the name vanishes exactly when it
                # delists/suspends) AND redistributed its weight to survivors.
                # Book the LAST AVAILABLE close instead. Deliberately NOT −100%:
                # measured, most series-endings here are symbol RENAMES
                # (AMARAJABAT→ARE&M, ADANIGAS→ATGL), where holders lost nothing, so
                # forcing a wipeout would manufacture losses that never happened.
                exit_p = f["n"] - 1
                if exit_p <= fill:
                    continue      # entry and exit would be the same bar
            entry_px = f["open"][fill]
            if not np.isfinite(entry_px) or entry_px <= 0:
                entry_px = f["close"][fill]
            exit_px = f["close"][exit_p]
            if not (np.isfinite(entry_px) and np.isfinite(exit_px) and entry_px > 0):
                continue
            cost = round_trip_cost_pct(f["adtv"][p]) / 100.0
            wret.append((wt, exit_px / entry_px - 1.0 - cost))
            # Keep what a DAILY path needs: this pick's weight, its entry price and
            # its own close series, so the equity curve can be marked every bar
            # instead of only at rebalances (see _daily_marks below).
            _pick_paths.append((wt, entry_px, f, cost))
            n_trades += 1
        rets = [r for _, r in wret]

        # Weighted basket return (weights renormalised over the picks that filled).
        if wret:
            wsum = sum(w for w, _ in wret)
            mean_pick = (sum(w * r for w, r in wret) / wsum) if wsum > 0 else 0.0
        else:
            mean_pick = 0.0
        # ACCURACY-FIX #1 (2026-07-26): the un-deployed fraction earns the cash rate,
        # not 0%. `invested` is 0 when nothing filled, so a BEAR period (no picks at
        # all) correctly earns the cash return on the WHOLE book rather than nothing.
        invested = deploy if wret else 0.0
        cash_leg = (1.0 - invested) * cash_period
        period_ret = invested * mean_pick + cash_leg
        # ── TRUE DAILY EQUITY PATH ──────────────────────────────────────────
        # The equity curve used to be recorded ONLY at rebalance dates, so
        # max-drawdown was month-end-to-month-end and could not see intra-month
        # pain. Measured on the same book, that reported −18.78% where a
        # daily-marked harness showed −26.47%. Every drawdown this app printed was
        # therefore understated. Mark each bar inside the holding period.
        _p_start, _p_end = ri + 1, min(ri + rebal, len(cal) - 1)
        _eq0 = equity
        for _t in range(_p_start, _p_end + 1):
            _d = cal[_t]
            _acc, _wsum = 0.0, 0.0
            for _wt, _epx, _f, _c in _pick_paths:
                _q = int(_f["idx"].searchsorted(_d, side="right")) - 1
                if _q < 0 or _q >= _f["n"]:
                    continue
                _px = _f["close"][_q]
                if np.isfinite(_px) and _epx > 0:
                    _acc += _wt * (_px / _epx - 1.0 - _c)
                    _wsum += _wt
            _mp = (_acc / _wsum) if _wsum > 0 else 0.0
            _inv = invested if _pick_paths else 0.0
            _cash_leg = (1.0 - _inv) * ((1.0 + RISK_FREE_ANNUAL) ** ((_t - ri) / TRADING_DAYS) - 1.0)
            daily_curve.append((str(_d.date()), round(_eq0 * (1.0 + _inv * _mp + _cash_leg), 2)))

        equity *= (1.0 + period_ret)
        period_rets.append(period_ret)
        hwm = max(hwm, equity)
        _trace.append({
            "date": str(date.date()), "regime": regime,
            "exp_pct": round(exp * 100, 0), "brake": brake,
            "deploy_pct": round(deploy * 100, 1), "n_picks": len(rets),
            "mean_pick_ret_pct": round(mean_pick * 100, 2),
            "cash_ret_pct": round(cash_leg * 100, 3),
            "period_ret_pct": round(period_ret * 100, 2),
            "equity": round(equity, 0),
            "dd_from_hwm_pct": round((equity / hwm - 1) * 100, 1),
        })
        eq_curve.append((str(date.date()), round(equity, 2)))

        # Benchmark buy-and-hold value at the NEXT rebalance close (same horizon).
        nxt = rebal_i[k + 1] if k + 1 < len(rebal_i) else min(ri + rebal, len(cal) - 1)
        # ACCURACY-FIX #2 (2026-07-26): NIFTYBEES reinvests index dividends but our
        # stock returns are price-only, so the raw comparison handicapped every
        # strategy by ~the Nifty yield. Strip the pro-rated dividend from the
        # benchmark to compare price-with-price. Uses benchmark.py's single
        # documented definition of the drag so there is one source of truth.
        _dfac = max(0.0, 1.0 - bm.dividend_drag(nxt - rebal_i[0]))
        bench_curve.append((str(cal[nxt].date()),
                            round(bench_units * bench_vals[nxt] * _dfac, 2)))
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
        # ACCURACY-FIX #3 (2026-07-26): a real Sharpe subtracts the risk-free rate.
        # Without it this was return/vol, which FLATTERS any strategy that parks in
        # cash — free credit for a zero-volatility leg with no opportunity cost
        # charged against it. All-Weather is exactly such a strategy, so this fix
        # cuts its reported Sharpe the most. (std of excess == std of raw here,
        # since rf_period is a constant.)
        rf_period = (1.0 + RISK_FREE_ANNUAL) ** (rebal / TRADING_DAYS) - 1.0
        sd = float(np.std(pr))
        sharpe = ((float(np.mean(pr)) - rf_period) / sd * np.sqrt(periods_per_yr)
                  if sd > 0 else 0.0)
        return {"cagr": round(cagr, 2), "max_dd": round(max_dd, 2),
                "sharpe": round(sharpe, 2), "total_return": round(total * 100, 2)}

    sys_vals = [start_equity] + [e for _, e in eq_curve]
    bch_vals = [start_equity] + [b for _, b in bench_curve]
    sysm = _metrics(sys_vals, len(eq_curve))
    # The honest drawdown, measured on every bar rather than at month-ends.
    if daily_curve:
        _dv = np.array([v for _, v in daily_curve], dtype=float)
        _peak = np.maximum.accumulate(_dv)
        sysm["max_dd_daily"] = round(float(np.min(_dv / _peak - 1.0)) * 100, 2)
        sysm["max_dd_note"] = ("max_dd is month-end sampled and UNDERSTATES real pain; "
                               "max_dd_daily is the true bar-by-bar figure.")
    bchm = _metrics(bch_vals, len(bench_curve))

    return {
        "as_of": eq_curve[-1][0] if eq_curve else None,
        "start": eq_curve[0][0] if eq_curve else None,
        "rebalances": len(eq_curve), "trades": n_trades,
        "avg_deployment_pct": round(float(np.mean(exposures)) * 100, 1) if exposures else 0.0,
        "params": {"rebal": rebal, "top_k": top_k, "risk_pct": risk_pct, "days": days,
                   "strategy": strategy},
        # Surfaced so the UI can state the assumptions instead of hiding them.
        "assumptions": {
            "cash_yield_annual_pct": round(RISK_FREE_ANNUAL * 100, 2),
            "risk_free_annual_pct": round(RISK_FREE_ANNUAL * 100, 2),
            "nifty_div_yield_annual_pct": round(bm.NIFTY_ANNUAL_DIV_YIELD * 100, 2),
            "sharpe_is_excess_over_rf": True,
            "port_vol_target": port_vol_target,
            "benchmark_is_price_comparable": True,
        },
        "regime_stats": ({"switches": n_switch, "periods": regime_periods}
                         if _all_weather else None),
        "system": sysm,
        "benchmark": bchm,
        "edge": {
            "cagr_delta": round(sysm["cagr"] - bchm["cagr"], 2),
            "dd_delta": round(sysm["max_dd"] - bchm["max_dd"], 2),   # less negative = better
        },
        "equity_curve": eq_curve,
        "daily_curve": daily_curve,   # FULL path — sampling it understated drawdown
        "benchmark_curve": bench_curve,
        "trace": _trace,
    }
