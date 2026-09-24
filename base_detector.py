"""
Multi-base detector — finds EVERY base a stock built, the way BananaPatterns draws
them: each consolidation the stock formed, its pivot (the high that sets it off), how
deep and how long it was, whether it broke out, and how it behaved.

A "base" = a pullback-and-consolidation under a prior high (the pivot), bounded in
depth (a base, not a crash), lasting at least a couple of weeks, that then breaks out
when price closes back above the pivot. Bases are non-overlapping and chronological —
one breakout ends a base and the next leg can build the next one.

Per base we return the same measures BananaPatterns shows:
  weeks · depth% · pivot · base low · type (VCP / Blue sky / Flat / Base) ·
  breakout date · now-vs-pivot% · ATR-tightening · volume dry-up · up/down vol net &
  ratio · failed pokes (failed breakout attempts inside the base).

Pure functions on plain lists so the same numbers can be computed server-side and the
chart just draws the boxes. Nothing is asserted — an uptrend base that later failed
still shows a negative now-vs-pivot.
"""
from __future__ import annotations


def _sma(a, i, n):
    if i - n + 1 < 0:
        return None
    return sum(a[i - n + 1:i + 1]) / n


def _atr(h, l, c, s, e):
    """Average true range over [s, e] (inclusive), normalised to price (fraction)."""
    if e <= s:
        return None
    trs = []
    for i in range(s + 1, e + 1):
        tr = max(h[i] - l[i], abs(h[i] - c[i - 1]), abs(l[i] - c[i - 1]))
        trs.append(tr)
    if not trs:
        return None
    atr = sum(trs) / len(trs)
    mid = (h[e] + l[e]) / 2 or 1.0
    return atr / mid


def detect_bases(dates, o, h, l, c, v,
                 min_bars: int = 10, max_depth: float = 0.40,
                 min_pullback: float = 0.05):
    """Return a chronological list of bases. Each is a dict (see module docstring).

    dates: list[str YYYY-MM-DD]; o/h/l/c: list[float]; v: list[float|int].
    min_bars: shortest base (10 sessions ≈ 2 weeks). max_depth: deepest base allowed.
    min_pullback: how far price must fall from the pivot to count as basing (not just
    a one-day dip)."""
    n = len(c)
    if n < min_bars + 5:
        return []
    last = c[-1]

    bases = []

    def _finalise(s, e, pivot, low, broke, poke_count):
        dur = e - s
        if dur < min_bars:
            return None
        depth = (pivot - low) / pivot if pivot > 0 else 0.0
        if depth > max_depth or depth <= 0:
            return None
        # Type: Blue sky = pivot is (near) an all-time high at the time it formed.
        prior_max = max(h[:s]) if s > 0 else pivot
        blue_sky = pivot >= prior_max * 0.999
        # VCP = volatility contracts across the base (2nd-half ATR < 1st-half ATR).
        mid = s + dur // 2
        a1 = _atr(h, l, c, s, mid)
        a2 = _atr(h, l, c, mid, e)
        vcp = (a1 is not None and a2 is not None and a2 < a1 * 0.85)
        if vcp and blue_sky:
            btype = "VCP / Blue sky"
        elif vcp:
            btype = "VCP"
        elif blue_sky:
            btype = "Blue sky"
        elif depth <= 0.15:
            btype = "Flat base"
        else:
            btype = "Base"
        # tightening = base ATR vs the run-up ATR before it (smaller = tighter).
        pre = _atr(h, l, c, max(0, s - dur), s)
        base_atr = _atr(h, l, c, s, e)
        tightening = round(base_atr / pre, 2) if (pre and base_atr) else None
        # volume dry-up = avg base volume vs avg volume of the prior leg (smaller=dry).
        pv = v[max(0, s - dur):s]
        bv = v[s:e + 1]
        vol_dryup = round((sum(bv) / len(bv)) / (sum(pv) / len(pv)), 2) if (pv and bv and sum(pv)) else None
        # up/down volume inside the base
        upv = sum(v[i] for i in range(s + 1, e + 1) if c[i] > c[i - 1])
        dnv = sum(v[i] for i in range(s + 1, e + 1) if c[i] < c[i - 1])
        ud_ratio = round(upv / dnv, 1) if dnv else None
        ud_net = round((upv - dnv) / (upv + dnv), 2) if (upv + dnv) else None
        return {
            "start": s, "end": e,
            "start_date": dates[s], "end_date": dates[e],
            "breakout_date": dates[e] if broke else None,
            "weeks": max(1, round(dur / 5)),
            "depth_pct": round(depth * 100, 1),
            "pivot": round(pivot, 2),
            "low": round(low, 2),
            "type": btype,
            "broke_out": bool(broke),
            "now_vs_pivot_pct": round((last / pivot - 1) * 100, 1) if pivot > 0 else None,
            "tightening": tightening,
            "vol_dryup": vol_dryup,
            "ud_ratio": ud_ratio,
            "ud_net": ud_net,
            "failed_pokes": int(poke_count),
            "rs": None,   # filled by the caller (market-wide RS at the breakout)
        }

    running_high = h[0]
    rh_idx = 0
    base_open = False
    s = 0
    pivot = 0.0
    low = 1e18
    pokes = 0

    for i in range(1, n):
        if not base_open:
            if h[i] > running_high:
                running_high = h[i]
                rh_idx = i
            elif running_high > 0 and (running_high - c[i]) / running_high >= min_pullback:
                # price has pulled back from the high → a base is forming under it.
                base_open = True
                s = rh_idx
                pivot = running_high
                low = l[i]
                pokes = 0
        else:
            low = min(low, l[i])
            depth = (pivot - low) / pivot if pivot > 0 else 1.0
            if depth > max_depth:
                # too deep to be a base → abandon; start hunting a new high from here.
                base_open = False
                running_high = h[i]
                rh_idx = i
                continue
            # a "failed poke" = an intraday push above the pivot that closed back under.
            if h[i] > pivot and c[i] <= pivot:
                pokes += 1
            if c[i] > pivot:                       # BREAKOUT (close clears the pivot)
                b = _finalise(s, i, pivot, low, True, pokes)
                if b:
                    bases.append(b)
                base_open = False
                running_high = h[i]
                rh_idx = i

    # a currently-forming base still open at the right edge (not yet broken out)
    if base_open:
        b = _finalise(s, n - 1, pivot, low, False, pokes)
        if b:
            b["forming"] = True
            bases.append(b)

    return bases
