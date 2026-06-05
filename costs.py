"""
Realistic Indian round-trip transaction-cost model (delivery / swing trades).

Single source of truth: every backtest imports round_trip_cost_pct() so the
"net" numbers reflect what you'd actually keep after costs. The old code either
ignored costs entirely (edge_engine) or used a flat 25 bps (backtester) that is
far too low for the small/micro-caps the scanners surface.

Two parts:
  1. REGULATORY / STATUTORY — fixed % of turnover, liquidity-independent.
  2. SLIPPAGE / IMPACT       — the part that actually hurts on illiquid names,
     modelled as a function of liquidity (ADTV in ₹Cr).

Numbers reflect a discount broker (free delivery) on NSE cash-segment delivery
trades, mid-2025 statutory rates. They are deliberately conservative and all
tunable from one place.
"""
from __future__ import annotations

# ── 1. Regulatory / statutory (per side unless noted), in % of turnover ─────────
BROKERAGE_PCT_PER_SIDE = 0.0      # discount broker: free equity delivery
STT_PCT_BUY            = 0.10     # securities transaction tax, delivery
STT_PCT_SELL           = 0.10
EXCH_TXN_PCT_PER_SIDE  = 0.00297  # NSE cash transaction charge (~₹297/cr)
SEBI_PCT_PER_SIDE      = 0.0001   # SEBI turnover fee (₹10/cr)
STAMP_PCT_BUY          = 0.015    # stamp duty, buy side only
GST_RATE               = 0.18     # GST on (brokerage + txn + sebi)


def _regulatory_rt_pct() -> float:
    gst_base = ((BROKERAGE_PCT_PER_SIDE * 2)
                + (EXCH_TXN_PCT_PER_SIDE * 2)
                + (SEBI_PCT_PER_SIDE * 2))
    gst = gst_base * GST_RATE
    return (STT_PCT_BUY + STT_PCT_SELL
            + STAMP_PCT_BUY
            + EXCH_TXN_PCT_PER_SIDE * 2
            + SEBI_PCT_PER_SIDE * 2
            + BROKERAGE_PCT_PER_SIDE * 2
            + gst)


REGULATORY_RT_PCT = round(_regulatory_rt_pct(), 4)   # ≈ 0.215 %


# ── 2. Slippage / market-impact (per side), in % of turnover ────────────────────
# raw = base + k / adtv_cr, clamped. Liquid large-caps ≈ a few bps a side;
# a ₹1 Cr-ADTV micro-cap pays ~80 bps a side because your order moves the book.
SLIP_BASE_PCT = 0.04     # floor even on the most liquid names
SLIP_K        = 0.80     # impact coefficient (₹Cr · %)
SLIP_MIN_PCT  = 0.03
SLIP_MAX_PCT  = 1.50     # cap one-way slippage for the most illiquid names


def slippage_pct_one_way(adtv_cr: float | None) -> float:
    """One-way slippage/impact as % of turnover, given average daily traded value (₹Cr)."""
    if adtv_cr is None or adtv_cr <= 0:
        return SLIP_MAX_PCT
    raw = SLIP_BASE_PCT + SLIP_K / float(adtv_cr)
    return float(min(SLIP_MAX_PCT, max(SLIP_MIN_PCT, raw)))


def round_trip_cost_pct(adtv_cr: float | None = None) -> float:
    """
    Total round-trip cost (% of position) to subtract from a trade's GROSS return:
        regulatory/statutory  +  2 × one-way slippage.

    Examples (round trip):
        ADTV ₹50 Cr  → ~0.33 %
        ADTV ₹10 Cr  → ~0.45 %
        ADTV ₹2  Cr  → ~1.10 %
        ADTV ₹0.5 Cr → ~3.2  %   (micro-cap — costs eat the edge)
    """
    return round(REGULATORY_RT_PCT + 2.0 * slippage_pct_one_way(adtv_cr), 4)
