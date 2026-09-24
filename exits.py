"""
Exit framework — the missing other half of every scanner.

Every scanner in this repo finds ENTRIES; nothing tells you when to GET OUT.
`evaluate_exit()` takes one open position's OHLCV history (up to "today") and
the entry price, and returns a disciplined, fully-quantified exit assessment:
a battery of independent stop / profit-taking signals, each with its computed
level and a boolean trigger, rolled up into a single {HOLD, TRIM, EXIT} action
with a human-readable reason.

DESIGN
  • Pure / stateless — no I/O, no globals, no caching. Same inputs → same dict.
  • Reuses the canonical helpers in `analysis_utils.py` instead of re-deriving
    them, so an exit "ATR" is the same ATR every scanner already trusts:
        - atr()            → Wilder-smoothed ATR (used for the Chandelier stop)
        - stage_analysis() → Weinstein stage, surfaced as context
  • Every returned float is finite. NaN / Inf are converted to None so the dict
    is always JSON-safe (this app has been bitten by NaN-in-JSON before).

THE SIGNALS
  1. ATR trailing stop  — Chandelier exit: highest-high(since entry / last 22
                          bars) − 2.5 × ATR(22). Close below ⇒ trend gave way.
  2. Structure stop     — most recent significant swing low (lowest low of the
                          last ~15 bars). Close below ⇒ market structure broke.
  3. Initial/hard stop  — the line in the sand you set at entry (`stop_price`).
  4. Trend break        — WEEKLY close below the 20-week MA (EXIT); a daily close
                          below MA20 is a warning only. Young holdings without
                          20 completed weeks fall back to a daily MA50 close.
  5. Time stop          — held longer than `hold_days` but going nowhere
                          (return < +3%) ⇒ "dead money", capital better deployed.
  6. Profit-taking      — R-multiple TRIM at +2R / +3R, plus a parabolic /
                          over-extension flag when price runs >28% above MA50.

ACTION LOGIC
    EXIT  if any hard / trailing / structure stop fires, or a weekly close breaks
          the 20-week MA (daily MA50 for holdings younger than 20 weeks).
    TRIM  if a profit-taking / over-extension flag fires and no stop has.
    HOLD  otherwise (including the safe fallback on insufficient history).

VALIDATION (exit_backtest.py, 2026-07-06 — point-in-time, IS/OOS, cost-net,
1,151 real leader-entry trades over 2022-02→2026-07):
    • No exit rule beats BUY-&-HOLD on expectancy — in a trending tape, exiting
      early always costs return (buy-hold 9.3%/trade vs every exit lower). Exits
      are a RISK tool, not a return booster: they cut avg loss from −14.7% to
      −6 to −8% and the worst-decile from −20% to −9%. This holds IS→OOS even as
      raw returns fade OOS. So the defaults below are DELIBERATELY not tightened.
    • (2026-07-06) MA50 break as an EXIT (best-balanced single rule, PF 2.13) and
      MA20 as a WARNING only (a standalone MA20 exit whipsaws — +1.9%/trade).
      SUPERSEDED for the trend-break component on 2026-09-17 — see §4 below: with
      the live Chandelier the trend-break line moves outcomes by <0.2pp, and the
      weekly 20-week rule the owner chose measured no worse. The engine's
      EXIT roll-up is conservative by design (median hold ~16 bars) — the right
      posture for open-position defence (Guardian), where tail control > squeezing
      the last rupee. Guardian's EXIT→'exit' / TRIM→'trim' severity map is
      validated as-is. Re-run exit_backtest.py before changing any default here.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

# Canonical helpers — reuse, never reimplement.
from analysis_utils import atr, stage_analysis, stage_label

# ── Tunables (one place) ──────────────────────────────────────────────────────
MIN_BARS              = 30      # below this we can't form an honest opinion
ATR_PERIOD            = 22      # Chandelier ATR lookback
CHANDELIER_MULT       = 2.5     # highest-high − 2.5×ATR
CHANDELIER_LOOKBACK   = 22      # bars for the "highest high" if no entry anchor
STRUCT_LOOKBACK       = 15      # bars for the recent significant swing low
MA_FAST               = 20
MA_SLOW               = 50
MA_EXIT               = 100     # 100-DMA — reported for reference only (not a trigger)
MA_WEEKS              = 20      # trend-break EXIT: WEEKLY close below the 20-week MA.
                                # MA_SLOW (50) is the fallback for holdings <20 weeks old
                                # and the parabolic over-extension trim.
TIME_STOP_MIN_RET_PCT = 3.0     # below this % after hold_days ⇒ dead money
TRIM_R_1              = 2.0      # first profit-taking rung
TRIM_R_2              = 3.0      # second profit-taking rung
OVEREXT_MA50_PCT      = 28.0    # close >28% above MA50 ⇒ parabolic / over-extended


# ── Internal: scrub a float so the result dict is always JSON-safe ────────────
def _clean(x) -> float | None:
    """Return a finite python float, or None for NaN/Inf/None/garbage."""
    if x is None:
        return None
    try:
        f = float(x)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(f):
        return None
    return f


def _round(x, ndigits: int = 2) -> float | None:
    f = _clean(x)
    return None if f is None else round(f, ndigits)


def _safe_hold(current_price, reason: str) -> dict:
    """Uniform safe HOLD envelope (insufficient / unusable history)."""
    cp = _round(current_price)
    return {
        "action": "HOLD",
        "reason": reason,
        "current_price": cp,
        "current_return_pct": None,
        "r_multiple": None,
        "stage": None,
        "stage_label": None,
        "signals": {},
        "exit_triggers": [],
        "trim_triggers": [],
    }


def _bars_held(df: pd.DataFrame, entry_date) -> int | None:
    """Number of bars in `df` at or after entry_date (None if not resolvable)."""
    if entry_date is None:
        return None
    try:
        ed = pd.Timestamp(entry_date)
    except (TypeError, ValueError):
        return None
    try:
        # tz-align: our bhavcopy index is tz-naive.
        if getattr(df.index, "tz", None) is not None and ed.tz is None:
            ed = ed.tz_localize(df.index.tz)
        elif getattr(df.index, "tz", None) is None and ed.tz is not None:
            ed = ed.tz_localize(None)
    except (TypeError, ValueError):
        pass
    held = int((df.index >= ed).sum())
    return held if held > 0 else None


def evaluate_exit(
    df: pd.DataFrame,
    entry_price: float,
    *,
    entry_date=None,
    stop_price: float | None = None,
    hold_days: int | None = None,
) -> dict:
    """
    Compute disciplined exit signals for one open long position.

    Args:
        df:          position's OHLCV DataFrame, DatetimeIndex sorted ascending,
                     columns Open/High/Low/Close/Volume, up to "today" (the last
                     row is the most recent / current bar).
        entry_price: the price the position was opened at.
        entry_date:  (optional) entry timestamp. When given, the Chandelier
                     "highest high" is anchored from entry and the time stop and
                     bars-held become computable.
        stop_price:  (optional) the initial/hard stop set at entry. Enables the
                     hard-stop trigger and the R-multiple math.
        hold_days:   (optional) max sensible holding period in *bars*; with
                     entry_date, powers the "dead money" time stop.

    Returns:
        dict with current_price, current_return_pct, r_multiple, stage context,
        a `signals` sub-dict (each signal: its level + boolean trigger), the
        lists of which signals fired, and the overall `action` + `reason`.
        Every float is finite or None.
    """
    # ── Robustness gates ──────────────────────────────────────────────────────
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return _safe_hold(None, "insufficient history")

    needed = {"High", "Low", "Close"}
    if not needed.issubset(df.columns):
        return _safe_hold(None, "insufficient history (missing OHLC columns)")

    close = pd.to_numeric(df["Close"], errors="coerce").dropna()
    if len(close) < MIN_BARS:
        cp = float(close.iloc[-1]) if len(close) else None
        return _safe_hold(cp, "insufficient history")

    cur = float(close.iloc[-1])
    entry = _clean(entry_price)
    if entry is None or entry <= 0:
        return _safe_hold(cur, "invalid entry price")

    stop = _clean(stop_price)
    if stop is not None and stop <= 0:
        stop = None

    high = pd.to_numeric(df["High"], errors="coerce")
    low = pd.to_numeric(df["Low"], errors="coerce")

    # ── Core numbers ──────────────────────────────────────────────────────────
    current_return_pct = _round((cur / entry - 1.0) * 100.0)

    r_multiple = None
    if stop is not None and (entry - stop) > 0:
        r_multiple = _round((cur - entry) / (entry - stop), 2)

    # Weinstein stage as context (needs long history; returns 0 if too short).
    stage_val = int(stage_analysis(close))
    stage = stage_val if stage_val > 0 else None

    # Anchor the trailing-stop / over-extension lookback from entry when we can.
    held_bars = _bars_held(df, entry_date)

    signals: dict[str, dict] = {}
    exit_triggers: list[str] = []
    trim_triggers: list[str] = []

    # ── 1. ATR trailing stop — Chandelier exit ────────────────────────────────
    # ONE STOP, ONE TRUTH (2026-08-06). When the caller passes stop_price — the
    # Portfolio tab always does, and it IS the ratcheting Chandelier the SL column
    # displays — use THAT as the trailing level, so the number shown and the number
    # tested are the same. This block used to recompute its OWN Chandelier with
    # (a) the current bar included in highest-high (the same look-ahead that let an
    # ATH day set the very stop it then breached — see GRWRHITECH 2026-08-06), and
    # (b) a held_bars window keyed on the user-typed entry date. The result was a
    # tab showing "SL 4337.89, not hit" beside an "ATR trailing broken" exit driven
    # by a DIFFERENT, worse Chandelier. Two stops, two answers, one position.
    if stop_price is not None and stop_price > 0:
        chandelier   = float(stop_price)
        atr_val      = _clean(atr(df, ATR_PERIOD))     # kept for context display only
        highest_high = None
    else:
        # No position stop (backtest harness / watchlist): compute a Chandelier that
        # EXCLUDES the current bar so it cannot manufacture its own trigger, and use
        # a fixed lookback rather than an entry-date window.
        _prior       = df.iloc[:-1] if len(df) > 1 else df
        atr_val      = _clean(atr(_prior, ATR_PERIOD))
        _hh_window   = high.iloc[-(CHANDELIER_LOOKBACK + 1):-1] if len(high) > 1 else high.iloc[:0]
        highest_high = _clean(_hh_window.max()) if len(_hh_window) else None
        chandelier   = (highest_high - CHANDELIER_MULT * atr_val
                        if highest_high is not None and atr_val is not None and atr_val > 0
                        else None)
    chand_trig = bool(chandelier is not None and cur < chandelier)
    signals["atr_trailing_stop"] = {
        "level": _round(chandelier),
        "highest_high": _round(highest_high),
        "atr": _round(atr_val),
        "triggered": chand_trig,
    }
    if chand_trig:
        exit_triggers.append("ATR trailing stop (Chandelier) broken")

    # ── 2. Structure stop — break of the recent swing low ─────────────────────
    # The structural support the trade sits on = the LOWEST low of the last ~15
    # bars. A break of THAT is a real structural failure.
    #
    # FIX (2026-07-09): this previously used `below_entry.max()` — the HIGHEST
    # swing-low *just under* the entry, i.e. the tightest possible level. For a
    # position bought near a local high (e.g. CPPLUS near its 52-wk high), a
    # normal 1–2% pullback dips under that nearest low and false-fired
    # "Structure stop broken → EXIT" while the stock was still near its highs and
    # above its MAs. It also poisoned the Portfolio deep-analysis, which reads
    # this level and flipped the thesis to "BROKEN". The recent-swing-low
    # definition below is exactly what exit_backtest.py validated (break of the
    # prior 15-bar low → the exit that cut the loss tail).
    struct_window = low.iloc[-STRUCT_LOOKBACK:].dropna()
    swing_low = _clean(struct_window.min()) if len(struct_window) else None
    struct_level = swing_low
    struct_trig = bool(struct_level is not None and cur < struct_level)
    signals["structure_stop"] = {
        "level": _round(struct_level),
        "swing_low": _round(swing_low),
        "triggered": struct_trig,
    }
    if struct_trig:
        exit_triggers.append("Structure stop (swing low) broken")

    # ── 3. Initial / hard stop ────────────────────────────────────────────────
    hard_trig = bool(stop is not None and cur < stop)
    signals["hard_stop"] = {
        "level": _round(stop),
        "triggered": hard_trig,
    }
    # Since 2026-08-06 the trailing Chandelier (§1) uses this SAME stop_price when a
    # position stop is passed, so a breach would otherwise list both "ATR trailing
    # broken" and "Hard stop breached" for one event. Emit the hard-stop line only
    # when it is a DISTINCT level from the trailing stop.
    if hard_trig and not (chand_trig and chandelier is not None
                          and abs(float(chandelier) - float(stop)) < 1e-6):
        exit_triggers.append("Hard stop breached")

    # ── 4. Trend break — WEEKLY close below the 20-week MA (2026-09-17) ───────
    # The owner chose the 20-week MA (Momentum Rotation's rule) on 2026-09-04. The
    # 2026-09-05 implementation used a DAILY close under the 100-DMA instead — not the
    # same rule (it fires mid-week on dips the week recovers), described as "matching"
    # Momentum Rotation, and never re-validated despite the note above. Measured
    # 2026-09-17 with the LIVE engine definitions (exit_trend_break_validation.py):
    # 1,152 leader entries, 500 liquid names, 2022-05 → 2026-09, IS/OOS 60/40, net cost.
    #     trend break          exp%   avgL%   p10%  | OOS exp  OOS avgL  OOS p10
    #     chandelier only      1.36   -5.14   -8.45 |  0.49    -4.55     -7.39
    #     daily  < 50-DMA      1.41   -5.01   -8.35 |  0.57    -4.40     -7.33
    #     daily  < 100-DMA     1.37   -5.12   -8.43 |  0.51    -4.51     -7.39
    #     weekly < 20-wk MA    1.38   -5.12   -8.38 |  0.52    -4.50     -7.38
    #     (buy & hold          8.29  -13.80  -20.22 |  2.05   -13.03    -20.27)
    # The Chandelier fires first on almost every trade (avg hold ~14 bars), so the
    # trend-break line changes results by <0.2pp. Pre-registered rule: implement the
    # owner's rule unless its OOS tail is >1pp worse than the daily 100-DMA — it is not.
    # A week counts as COMPLETED once its Friday bar prints (or any earlier week), so a
    # holiday Friday confirms one session late, on the next week's first bar.
    ma20 = _clean(close.rolling(MA_FAST).mean().iloc[-1]) if len(close) >= MA_FAST else None
    ma50 = _clean(close.rolling(MA_SLOW).mean().iloc[-1]) if len(close) >= MA_SLOW else None
    ma100 = _clean(close.rolling(MA_EXIT).mean().iloc[-1]) if len(close) >= MA_EXIT else None
    ma20_break = bool(ma20 is not None and cur < ma20)
    ma50_break = bool(ma50 is not None and cur < ma50)
    ma100_break = bool(ma100 is not None and cur < ma100)

    wk_close = wk_ma = week_ended = None
    basis = None
    trend_break = False
    try:
        didx = pd.DatetimeIndex(close.index)
        per = didx.to_period("W-FRI")
        weekly = close.groupby(per).last()
        if didx[-1].weekday() < 4:                  # Mon–Thu: this week is not complete
            weekly = weekly[weekly.index < per[-1]]
        if len(weekly) >= MA_WEEKS:
            wma = weekly.rolling(MA_WEEKS).mean()
            wk_close, wk_ma = _clean(weekly.iloc[-1]), _clean(wma.iloc[-1])
            week_ended = str(weekly.index[-1].end_time.date())
            if wk_close is not None and wk_ma is not None:
                basis = "weekly_20wk"
                trend_break = wk_close < wk_ma
    except Exception:
        basis = None
    if basis is None and ma50 is not None:
        basis = "daily_ma50_fallback"               # < 20 completed weeks of history
        trend_break = ma50_break

    signals["ma_break"] = {
        "ma20": _round(ma20),
        "ma50": _round(ma50),
        "ma100": _round(ma100),
        "wk20_ma": _round(wk_ma),
        "week_close": _round(wk_close),
        "week_ended": week_ended,
        "basis": basis,
        "ma_exit": _round(wk_ma if basis == "weekly_20wk" else ma50),
        "ma20_break": ma20_break,
        "ma50_break": ma50_break,
        "ma100_break": ma100_break,                 # reference only — not a trigger
        "weekly_break": bool(trend_break and basis == "weekly_20wk"),
        "triggered": bool(trend_break),
    }
    if trend_break and basis == "weekly_20wk":
        exit_triggers.append(
            f"Weekly close ₹{wk_close:,.2f} below 20-week MA ₹{wk_ma:,.2f} (week ending {week_ended})")
    elif trend_break:
        exit_triggers.append("Closed below 50-DMA (holding has <20 weeks of history)")

    # ── 5. Time stop — dead money ─────────────────────────────────────────────
    ret_pct_raw = (cur / entry - 1.0) * 100.0
    time_trig = bool(
        hold_days is not None
        and held_bars is not None
        and held_bars > int(hold_days)
        and ret_pct_raw < TIME_STOP_MIN_RET_PCT
    )
    signals["time_stop"] = {
        "bars_held": held_bars,
        "hold_days": int(hold_days) if hold_days is not None else None,
        "return_pct": current_return_pct,
        "threshold_pct": TIME_STOP_MIN_RET_PCT,
        "triggered": time_trig,
    }
    # Time stop is "dead money" — a TRIM/recycle nudge, not a stop-loss EXIT.
    if time_trig:
        trim_triggers.append(
            f"Dead money: held {held_bars} bars (> {int(hold_days)}) for "
            f"{current_return_pct:+.1f}%"
        )

    # ── 6. Profit-taking: R-multiple rungs + parabolic over-extension ─────────
    trim_r_level = None
    if r_multiple is not None:
        if r_multiple >= TRIM_R_2:
            trim_r_level = TRIM_R_2
            trim_triggers.append(f"At +{r_multiple:.1f}R — trim (≥{TRIM_R_2:.0f}R)")
        elif r_multiple >= TRIM_R_1:
            trim_r_level = TRIM_R_1
            trim_triggers.append(f"At +{r_multiple:.1f}R — trim (≥{TRIM_R_1:.0f}R)")
    signals["profit_take_r"] = {
        "r_multiple": r_multiple,
        "trim_at": trim_r_level,
        "triggered": bool(trim_r_level is not None),
    }

    # Parabolic / over-extension: close well above MA50.
    ext_pct = None
    overext_trig = False
    if ma50 is not None and ma50 > 0:
        ext_pct = _round((cur / ma50 - 1.0) * 100.0)
        overext_trig = bool(ext_pct is not None and ext_pct > OVEREXT_MA50_PCT)
    signals["over_extension"] = {
        "pct_above_ma50": ext_pct,
        "threshold_pct": OVEREXT_MA50_PCT,
        "triggered": overext_trig,
    }
    if overext_trig:
        trim_triggers.append(
            f"Over-extended: {ext_pct:+.0f}% above MA50 (parabolic risk)"
        )

    # ── Roll-up: action + reason ──────────────────────────────────────────────
    if exit_triggers:
        action = "EXIT"
        reason = "; ".join(exit_triggers)
    elif trim_triggers:
        action = "TRIM"
        reason = "; ".join(trim_triggers)
    else:
        action = "HOLD"
        # Give a positive, informative reason for holding.
        bits = []
        if current_return_pct is not None:
            bits.append(f"{current_return_pct:+.1f}%")
        if r_multiple is not None:
            bits.append(f"{r_multiple:+.1f}R")
        if stage is not None:
            bits.append(stage_label(stage))
        reason = "No stop or profit signal — let it work" + (
            f" ({', '.join(bits)})" if bits else ""
        )

    return {
        "action": action,
        "reason": reason,
        "current_price": _round(cur),
        "current_return_pct": current_return_pct,
        "r_multiple": r_multiple,
        "stage": stage,
        "stage_label": stage_label(stage) if stage is not None else None,
        "signals": signals,
        "exit_triggers": exit_triggers,
        "trim_triggers": trim_triggers,
    }


# ── Demo: exercise the logic on a few liquid names ────────────────────────────
if __name__ == "__main__":
    from edge_engine import _load_stocks

    ENTRY_BARS_AGO = 40   # pretend we bought 40 bars back

    print("Loading bhavcopy history (days=400)…")
    stocks = _load_stocks(days=400)
    print(f"Loaded {len(stocks)} symbols.\n")
    # Derive demo symbols from the loaded universe (no hardcoded basket).
    DEMO_SYMS = list(stocks)[:7]

    def _fnum(x, suffix="", width=8):
        return (f"{x:{width}.2f}{suffix}" if isinstance(x, (int, float)) else f"{'—':>{width}}{suffix}")

    for sym in DEMO_SYMS:
        df = stocks.get(sym)
        if df is None or len(df) <= ENTRY_BARS_AGO + 5:
            print(f"── {sym}: no/short data, skipping ──\n")
            continue

        # Arbitrary but reproducible position: bought the close 40 bars ago,
        # with a hard stop 1.5×ATR below that entry, and a 30-bar patience window.
        entry_price = float(df["Close"].iloc[-(ENTRY_BARS_AGO + 1)])
        entry_date = df.index[-(ENTRY_BARS_AGO + 1)]
        atr_at_entry = atr(df.iloc[: len(df) - ENTRY_BARS_AGO], 22) or (entry_price * 0.02)
        stop_price = entry_price - 1.5 * atr_at_entry

        res = evaluate_exit(
            df,
            entry_price,
            entry_date=entry_date,
            stop_price=stop_price,
            hold_days=30,
        )

        print(f"── {sym} "
              f"(entry ₹{entry_price:.2f} on {pd.Timestamp(entry_date).date()}, "
              f"stop ₹{stop_price:.2f}) ──")
        print(f"   ACTION : {res['action']}  —  {res['reason']}")
        print(f"   price={_fnum(res['current_price'])}  "
              f"ret={_fnum(res['current_return_pct'], '%')}  "
              f"R={_fnum(res['r_multiple'], 'R')}  "
              f"stage={res['stage_label'] or '—'}")
        s = res["signals"]
        if s:
            print(f"   Chandelier : level={_fnum(s['atr_trailing_stop']['level'])} "
                  f"trig={s['atr_trailing_stop']['triggered']}")
            print(f"   Structure  : level={_fnum(s['structure_stop']['level'])} "
                  f"trig={s['structure_stop']['triggered']}")
            print(f"   Hard stop  : level={_fnum(s['hard_stop']['level'])} "
                  f"trig={s['hard_stop']['triggered']}")
            print(f"   MA break   : ma20={_fnum(s['ma_break']['ma20'])} "
                  f"ma50={_fnum(s['ma_break']['ma50'])} "
                  f"ma50_break={s['ma_break']['ma50_break']}")
            print(f"   Time stop  : bars_held={s['time_stop']['bars_held']} "
                  f"trig={s['time_stop']['triggered']}")
            print(f"   ProfitTake : R={_fnum(s['profit_take_r']['r_multiple'], 'R')} "
                  f"ext_above_ma50={_fnum(s['over_extension']['pct_above_ma50'], '%')} "
                  f"overext={s['over_extension']['triggered']}")
        print()

    # ── NaN / Inf safety self-check across the whole demo set ─────────────────
    def _has_bad(obj) -> bool:
        if isinstance(obj, float):
            return not math.isfinite(obj)
        if isinstance(obj, dict):
            return any(_has_bad(v) for v in obj.values())
        if isinstance(obj, (list, tuple)):
            return any(_has_bad(v) for v in obj)
        return False

    bad = False
    for sym in DEMO_SYMS:
        df = stocks.get(sym)
        if df is None or len(df) <= ENTRY_BARS_AGO + 5:
            continue
        entry_price = float(df["Close"].iloc[-(ENTRY_BARS_AGO + 1)])
        r = evaluate_exit(df, entry_price, entry_date=df.index[-(ENTRY_BARS_AGO + 1)],
                          stop_price=entry_price * 0.92, hold_days=30)
        if _has_bad(r):
            bad = True
            print(f"!! NaN/Inf detected in {sym} output")

    # Short-history guard self-check.
    tiny = evaluate_exit(stocks.get(DEMO_SYMS[0], pd.DataFrame()).head(10), 100.0)
    assert tiny["action"] == "HOLD" and "insufficient history" in tiny["reason"], tiny

    print(f"NaN/Inf-free across demo set : {not bad}")
    print(f"Short-history guard          : OK ({tiny['reason']!r})")
