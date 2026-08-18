"""
Signal context — capacity per screen (#6) and cross-tab de-duplication (#5).
===========================================================================

Two honesty problems this app had, both invisible in the UI:

CAPACITY. Every screen presented its picks identically, but they do not absorb
the same money. The delivery-accumulation edge is validated ONLY in a Rs 1-25cr
ADTV band and measurably INVERTS above ~Rs 100cr, so at institutional size the
signal is worse than nothing. Measured 2026-08-13: 658 in-band names, median
ADTV Rs 6.7cr, so a 20-name book at 5% of daily volume holds about Rs 6.7cr and
the whole band about Rs 291cr. A screen that cannot say this is implying a
scalability it does not have.

CROSS-TAB CONFIRMATION. The same stock surfacing on eight tabs feels like eight
independent confirmations. It is not: the trend screens measured 0.41 mean
pairwise correlation, and stacking them was REJECTED out-of-sample (-2.22pp).
Seeing "also in 6 other screens" should read as "this is one correlated bet",
not as a stronger case.
"""
from __future__ import annotations

import numpy as np

# 5% of a stock's daily turnover is the conventional ceiling before your own
# order starts moving the price against you.
PARTICIPATION = 0.05


def capacity_for(rows: list[dict], adtv_key: str = "adtv_cr") -> dict:
    """Rupee capacity of a screen's current output, from its own ADTV values."""
    a = np.array([r.get(adtv_key) for r in rows
                  if isinstance(r.get(adtv_key), (int, float)) and r.get(adtv_key) > 0],
                 dtype=float)
    if a.size == 0:
        return {"names": 0, "capacity_cr": None}
    per_name = a * PARTICIPATION
    return {
        "names": int(a.size),
        "median_adtv_cr": round(float(np.median(a)), 2),
        "capacity_cr": round(float(per_name.sum()), 1),        # whole list
        "capacity_20_cr": round(float(np.sort(per_name)[::-1][:20].sum()), 1),
        "participation_pct": PARTICIPATION * 100,
        "note": (f"At {PARTICIPATION*100:.0f}% of daily volume this list absorbs about "
                 f"Rs {float(per_name.sum()):,.0f}cr in total. Sizing beyond that moves "
                 f"the price against you and the measured edge does not survive it."),
    }


def annotate_overlap(screens: dict[str, list[dict]]) -> dict:
    """`screens` = {screen_name: [row, ...]}. Returns per-symbol appearances.

    Mutates nothing; callers attach `also_in` to their own rows. Deliberately
    framed as a CORRELATION WARNING rather than a conviction score, because
    stacking these screens measured negative out-of-sample.
    """
    seen: dict[str, list[str]] = {}
    for name, rows in (screens or {}).items():
        for r in rows or []:
            sym = r.get("symbol")
            if sym:
                seen.setdefault(sym, [])
                if name not in seen[sym]:
                    seen[sym].append(name)
    multi = {s: v for s, v in seen.items() if len(v) > 1}
    return {
        "symbols": seen,
        "multi_screen": multi,
        "n_multi": len(multi),
        "warning": ("Appearing on several screens is NOT independent confirmation — "
                    "these screens measured 0.41 mean pairwise correlation and "
                    "stacking them was rejected out-of-sample (-2.22pp). Treat "
                    "overlap as concentration risk in one factor."),
    }
