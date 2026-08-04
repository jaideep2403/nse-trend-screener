"""Does a DAILY sector-breakdown exit improve the monthly book? Prove or reject.

The ask was "sell immediately when a sector loses strength". The rebalance sweep
already showed that re-ranking the WHOLE book weekly is catastrophic (CAGR
25.4%→7.3%, drawdown −9.8%→−28.5%). But that test was SYMMETRIC — it churned
entries as well as exits. This harness tests the asymmetric version, which is a
genuinely different thing and was never measured:

    ENTRIES stay monthly (the validated cadence, untouched).
    EXITS may fire on ANY day, but only when the position's SECTOR rotates out of
    a healthy RRG quadrant. Proceeds sit in cash until the next monthly rebalance —
    the overlay can never buy anything.

That asymmetry is the whole point: it can only reduce exposure, so it should cut
drawdown. The open question is what it costs in CAGR, and whether the trade is
worth it. This harness answers that and nothing else.

POINT-IN-TIME: sector RRG quadrants come from trailing z-scores (rrg._zscore uses
`rolling`), and every daily check at bar i reads the quadrant AT bar i, then fills
at bar i+1's OPEN. No look-ahead. Costs are charged on every exit, so the overlay
has to earn its turnover exactly like any other change.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import rrg
from costs import round_trip_cost_pct

TRADING_DAYS = 252
EXIT_QUADRANTS = {rrg.WEAKENING, rrg.LAGGING}


def _stock_sector_map(sectors: dict[str, list[str]]) -> dict[str, str]:
    """First sector wins — a stock in several thematic indices is assigned once so
    one name cannot be exited twice for the same event."""
    out: dict[str, str] = {}
    for name, members in sectors.items():
        for s in members:
            out.setdefault(s, name)
    return out


def run(days: int = 2800, rebal: int = 21, top_k: int = 20,
        overlay: bool = True, quality_weight: float = 0.0,
        flow_weight: float = 0.0, exit_quadrants: set | None = None,
        vol_target: float = 0.0, vol_lookback: int = 60, stop_atr: float = 0.0,
        progress=None) -> dict:
    from edge_engine import _load_stocks
    import benchmark as bm
    import defensive_scan as ds
    import regime_engine as rg
    import system_backtest as sb
    import sector_indices as si

    exit_quadrants = exit_quadrants or EXIT_QUADRANTS
    stocks = _load_stocks(days=days, survivorship_free=True)
    bench = bm.get_benchmark(days=days)
    if bench is None or len(bench) < 300:
        return {"error": "benchmark unavailable"}

    cal = bench.index
    bench_vals = bench.to_numpy(dtype=float)
    reg_lbl = rg.regime_series(bench_vals, confirm_up=5, confirm_down=5)

    # ── Sector RRG panel, point-in-time ──────────────────────────────────────
    try:
        sectors = si.get_sector_constituents()
    except Exception:
        import sector_analysis as sa
        sectors = sa.SECTOR_STOCKS
    sec_of = _stock_sector_map(sectors)
    quad: dict[str, pd.Series] = {}
    for name, members in sectors.items():
        px = rrg.sector_composite(stocks, list(members))
        if px is None:
            continue
        r = rrg.compute(px, bench)
        if not r.empty:
            quad[name] = r["quadrant"].reindex(cal).ffill()

    # ── Per-symbol features for selection (mirrors system_backtest) ──────────
    feats = {}
    for s, df in stocks.items():
        f = ds.precompute(df)
        if f is not None:
            feats[s] = f
    sym_pos = {s: {d: i for i, d in enumerate(f["idx"])} for s, f in feats.items()}

    qmap = {}
    if quality_weight > 0:
        try:
            import quality as _q
            qmap = _q.load_quality_map()
        except Exception:
            qmap = {}
    iflow = None
    if flow_weight > 0:
        try:
            import institutional_flow as _if
            _if.build_score_panel()
            iflow = _if
        except Exception:
            iflow = None

    def select(date):
        fmap = iflow.score_map_asof(date) if iflow is not None else {}
        rows = []
        for s, f in feats.items():
            p = sym_pos[s].get(date)
            if p is None or f["adtv"][p] < 2.0 or not ds.passes_gate(f, p):
                continue
            rf = ds.raw_factors(f, p)
            if rf is None:
                continue
            row = {"sym": s, "p": p, **rf}
            if quality_weight > 0:
                row["qual_raw"] = qmap.get(s, 0.5)
            if flow_weight > 0:
                row["flow_raw"] = fmap.get(s, 0.5)
            rows.append(row)
        vf = getattr(ds, "WIN_VOL_FILTER", 0.70)
        if vf and len(rows) > top_k * 2:
            thr = np.quantile([r["vol90"] for r in rows], vf)
            rows = [r for r in rows if r["vol90"] <= thr]
        ranked = ds.rank_and_score(rows, mom_weight=getattr(ds, "WIN_MOM_WEIGHT", 4.0),
                                   quality_weight=quality_weight,
                                   flow_weight=flow_weight)[:top_k]
        return [(r["sym"], r["p"]) for r in ranked]

    start = 252
    rebal_i = list(range(start, len(cal) - 1, rebal))
    if len(rebal_i) < 4:
        return {"error": "not enough history"}

    # POSITION VALUES, not fixed weights. An earlier version carried a constant `w`
    # per name and applied it to each day's return, which silently rebalanced the
    # book back to equal weight EVERY DAY — a constant-mix strategy that harvests
    # volatility and is not what the monthly engine does. It inflated CAGR to 26.3%
    # against system_backtest's 9.0% for the same nominal config. Values must drift.
    cash = 1_000_000.0
    cash_rate = (1.0 + sb.RISK_FREE_ANNUAL) ** (1 / TRADING_DAYS) - 1.0
    book: dict[str, dict] = {}          # sym -> {"val": rupees, "cost": round-trip frac}
    curve, n_exits, n_trades = [], 0, 0
    rebal_set = set(rebal_i)

    daily_rets: list[float] = []      # realised strategy returns, for vol targeting

    def _equity():
        return cash + sum(b["val"] for b in book.values())

    def _exposure_scale() -> float:
        """Volatility targeting: scale gross exposure so the book's REALISED vol
        tracks `vol_target`. Capped at 1.0 — never lever up, because levering a
        strategy whose edge is unproven converts a modest loss into a large one.
        Barroso & Santa-Clara showed vol-managed momentum cuts crash risk sharply;
        this is that idea applied to the whole book rather than to a factor."""
        if vol_target <= 0 or len(daily_rets) < vol_lookback:
            return 1.0
        rv = float(np.std(daily_rets[-vol_lookback:])) * np.sqrt(TRADING_DAYS)
        if rv <= 1e-9:
            return 1.0
        return float(min(1.0, vol_target / rv))

    for i in range(rebal_i[0], len(cal) - 1):
        date = cal[i]

        # ── 1. DAILY MARK first, rebalance LAST ─────────────────────────────
        # Order matters. An earlier version created positions on bar i and then
        # marked them with close[i]/close[i-1] in the same iteration — booking the
        # SIGNAL DAY'S OWN return, which was already known when the pick was made.
        # That is a full day of look-ahead per rebalance, handed to stocks selected
        # precisely for their momentum, and it inflated CAGR from ~9% to ~21%.
        # Now: mark what is already held, then rebalance at the END of the bar so
        # new positions fill at the NEXT bar's open and start accruing from there.
        cash *= (1.0 + cash_rate)
        for s, b in list(book.items()):
            f = feats[s]
            p = sym_pos[s].get(date)
            if p is None:
                continue
            c1 = f["close"][p]
            if not np.isfinite(c1) or c1 <= 0:
                continue
            if not b["started"]:
                # First mark of a new position: fill OPEN → today's close.
                if b["entry_px"] > 0:
                    b["val"] *= c1 / b["entry_px"]
                    b["started"] = True
                    b["last"] = c1
                continue
            if b["last"] > 0:
                b["val"] *= c1 / b["last"]
            b["last"] = c1
        _eq_now = _equity()
        if curve:
            _prev = curve[-1][1]
            if _prev > 0:
                daily_rets.append(_eq_now / _prev - 1.0)
        curve.append((str(date.date()), _eq_now))

        # ── 1b. PER-POSITION STOP — exit a name that breaks its ATR stop ────
        if stop_atr > 0 and book:
            for s_, b in list(book.items()):
                if b.get("stop", 0) > 0 and b.get("last", 0) > 0 and b["last"] < b["stop"]:
                    bb = book.pop(s_)
                    cash += bb["val"] * (1.0 - bb["cost"] / 2.0)
                    n_exits += 1

        # ── 2. ASYMMETRIC OVERLAY: exit only, never buy ─────────────────────
        if overlay and book and i not in rebal_set:
            for s in list(book):
                sec = sec_of.get(s)
                if not sec or sec not in quad:
                    continue
                if quad[sec].get(date) in exit_quadrants:
                    b = book.pop(s)
                    cash += b["val"] * (1.0 - b["cost"] / 2.0)   # exit leg only
                    n_exits += 1

        # ── 3. MONTHLY REBALANCE, at the end of the bar ─────────────────────
        if i in rebal_set:
            for s, b in list(book.items()):
                cash += b["val"] * (1.0 - b["cost"] / 2.0)       # exit leg
            book = {}
            state = reg_lbl[i]
            if state in ("BULL", "SIDEWAYS"):
                picks = select(date)
                if picks:
                    _scale = _exposure_scale()
                    alloc = (cash * _scale) / len(picks)
                    for s, pp_ in picks:
                        f = feats[s]
                        fill = pp_ + 1
                        if fill >= f["n"]:
                            continue
                        px = f["open"][fill]
                        if not np.isfinite(px) or px <= 0:
                            px = f["close"][fill]
                        if not np.isfinite(px) or px <= 0:
                            continue
                        c = round_trip_cost_pct(f["adtv"][pp_]) / 100.0
                        # precompute() exposes vol90 (annualised), not ATR. Express the
                        # stop as `stop_atr` MONTHLY sigmas below entry so a calm stock
                        # and a wild one get stops that mean the same thing.
                        _v = float(f["vol90"][pp_]) if np.isfinite(f["vol90"][pp_]) else 0.0
                        _sig = _v / np.sqrt(TRADING_DAYS) * np.sqrt(21.0)   # ~1-month sigma
                        _stop = (float(px) * (1.0 - stop_atr * _sig)) if (stop_atr > 0 and _sig > 0) else 0.0
                        book[s] = {"val": alloc * (1.0 - c / 2.0), "cost": c,
                                   "entry_px": float(px), "started": False, "last": 0.0,
                                   "stop": _stop}
                        cash -= alloc
                        n_trades += 1
                    cash = max(0.0, cash)
            # BEAR ⇒ everything stays in cash, same as the base engine

    def metrics(vals):
        a = np.array(vals, dtype=float)
        if len(a) < 3 or a[0] <= 0:
            return {"cagr": 0.0, "max_dd": 0.0, "sharpe": 0.0}
        yrs = len(a) / TRADING_DAYS
        cagr = ((a[-1] / a[0]) ** (1 / yrs) - 1) * 100
        dd = float(np.min(a / np.maximum.accumulate(a) - 1)) * 100
        r = a[1:] / a[:-1] - 1
        rf = (1.0 + sb.RISK_FREE_ANNUAL) ** (1 / TRADING_DAYS) - 1.0
        sh = ((r.mean() - rf) / r.std() * np.sqrt(TRADING_DAYS)) if r.std() > 0 else 0.0
        return {"cagr": round(cagr, 2), "max_dd": round(dd, 2), "sharpe": round(sh, 2)}

    vals = [v for _, v in curve]
    bvals = bench_vals[rebal_i[0]:rebal_i[0] + len(vals)]
    return {
        "overlay": overlay, "vol_target": vol_target, "stop_atr": stop_atr,
        "start": curve[0][0] if curve else None,
        "as_of": curve[-1][0] if curve else None,
        "bars": len(curve), "sector_exits": n_exits, "entries": n_trades,
        "system": metrics(vals),
        "benchmark": metrics(list(bvals)),
        "equity_curve": curve[::5],
    }
