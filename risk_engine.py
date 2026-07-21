"""Risk & Regime engine — the capital-protection layer.

A long-only end-of-day screener cannot out-SELECT a falling market; the only
mechanical edge that survives a downturn is cutting EXPOSURE when the regime
turns. This module turns the canonical regime (regime.py) into a rule-based
exposure ladder, and provides risk-budgeted position sizing so a single loss is
bounded by design.

Everything here is MECHANICAL and rule-based — not personalised advice. It maps
a market state to a deployment percentage and turns a stop distance into a share
count. It never says "buy X"; it answers "if you take a trade, how big, and how
much of your book should be at risk right now."

Grounding: the breadth walk-forward (breadth_validation.py) showed only the
EXTREMES discriminate — composite ≥12 has the calmest forward paths (risk-on),
≤4 has the toxic left tail (risk-off), the middle is noise. The exposure ladder
below mirrors that: lean in when the structure confirms, go to cash when
distribution dominates, stay neutral in the noisy middle.
"""

from __future__ import annotations

# ── Regime → recommended portfolio exposure ──────────────────────────────────
# Keys are the canonical regime labels from regime.classify_regime(). Values:
#   exposure_pct : fraction of capital that should be AT WORK in longs (0-100)
#   headline     : short label for the UI switch
#   note         : one-line rule rationale
#   color        : token-friendly hex
_EXPOSURE_LADDER = {
    "Confirmed Uptrend": (100, "Full Deployment",
                          "Structure confirms — deploy into the strongest setups.",
                          "#22c55e"),
    "Uptrend Under Pressure": (50, "Half Size",
                          "Distribution is creeping in — new longs at half size, "
                          "highest-conviction only.",
                          "#eab308"),
    "Correction": (25, "Defensive",
                          "Distribution dominates — minimal new longs, protect open gains.",
                          "#f97316"),
    "Downtrend": (0, "Cash",
                          "Trend is down — a long-only book cannot beat this; "
                          "raise cash and wait for a follow-through day.",
                          "#ef4444"),
    "Unknown": (50, "Neutral",
                          "Regime not yet computed — neutral default.",
                          "#94a3b8"),
}


def regime_exposure(regime: str | None) -> dict:
    """Map a canonical regime label to a recommended exposure stance."""
    pct, headline, note, color = _EXPOSURE_LADDER.get(
        regime or "Unknown", _EXPOSURE_LADDER["Unknown"])
    return {
        "regime":       regime or "Unknown",
        "exposure_pct": pct,
        "headline":     headline,
        "note":         note,
        "color":        color,
    }


# ── Risk-budgeted position sizing ────────────────────────────────────────────
def position_size(capital: float, entry: float, stop: float,
                  risk_pct_per_trade: float = 0.75,
                  regime_exposure_pct: float = 100.0,
                  max_position_pct: float = 10.0) -> dict:
    """Size a single position so the loss-to-stop is a fixed fraction of capital.

    risk_pct_per_trade  — % of capital you accept losing if the stop is hit (0.75)
    regime_exposure_pct — scales risk DOWN in weak regimes (from regime_exposure)
    max_position_pct    — hard cap on any one name's rupee weight

    Returns shares, rupee value, the actual ₹ at risk, and which cap bound.
    """
    out = {
        "valid": False, "shares": 0, "rupee_value": 0.0, "risk_amount": 0.0,
        "risk_per_share": 0.0, "pct_of_capital": 0.0, "capped_by": None,
        "message": "",
    }
    if not (capital and capital > 0):
        out["message"] = "Enter a positive capital amount."
        return out
    if not (entry and entry > 0):
        out["message"] = "Enter a positive entry price."
        return out
    if stop is None or stop <= 0:
        out["message"] = "Enter a positive stop price."
        return out
    risk_per_share = entry - stop
    if risk_per_share <= 0:
        out["message"] = "Stop must be BELOW entry for a long — no valid risk to size."
        return out

    # Regime scales how much of the per-trade risk budget is live.
    eff_risk_pct = risk_pct_per_trade * (max(0.0, min(100.0, regime_exposure_pct)) / 100.0)
    risk_budget = capital * eff_risk_pct / 100.0
    if risk_budget <= 0:
        out["message"] = ("Regime exposure is 0% (cash) — the rule says take no new "
                          "risk here.")
        out["risk_per_share"] = round(risk_per_share, 2)
        return out

    shares_by_risk = risk_budget / risk_per_share
    value_by_risk = shares_by_risk * entry

    # Single-name cap
    max_value = capital * max_position_pct / 100.0
    if value_by_risk > max_value:
        shares = int(max_value // entry)
        capped = "position-cap"
    else:
        shares = int(shares_by_risk)
        capped = "risk-budget"

    if shares <= 0:
        out["message"] = ("Stop is so wide that even one share exceeds the risk "
                          "budget — widen capital or tighten the stop.")
        out["risk_per_share"] = round(risk_per_share, 2)
        return out

    rupee_value = shares * entry
    risk_amount = shares * risk_per_share
    out.update({
        "valid":          True,
        "shares":         shares,
        "rupee_value":    round(rupee_value, 2),
        "risk_amount":    round(risk_amount, 2),
        "risk_per_share": round(risk_per_share, 2),
        "pct_of_capital": round(rupee_value / capital * 100, 1),
        "eff_risk_pct":   round(eff_risk_pct, 3),
        "capped_by":      capped,
        "message":        "",
    })
    return out


# ── Equity-curve brake ───────────────────────────────────────────────────────
# A portfolio-level circuit breaker that sits ON TOP of the regime ladder: when
# your own book is in drawdown, cut new-position risk regardless of what the market
# regime says. Breadth regimes can lag your personal equity curve — this reacts to
# YOUR drawdown directly, which is the thing that actually blows accounts up.
EQUITY_BRAKE_THRESHOLD_PCT = 8.0   # start braking when >8% off the high-water mark
EQUITY_BRAKE_HARD_PCT      = 15.0  # at ≥15% off, halve again (quarter size)


def equity_brake(equity: float, high_water_mark: float,
                 soft: float = EQUITY_BRAKE_THRESHOLD_PCT,
                 hard: float = EQUITY_BRAKE_HARD_PCT) -> dict:
    """Risk multiplier from the book's drawdown vs its high-water mark.
      • within `soft`% of the HWM      → 1.00 (full)
      • between soft and hard          → 0.50 (halve new-position risk)
      • at/below `hard`%               → 0.25 (quarter)
    """
    if not (high_water_mark and high_water_mark > 0) or equity is None:
        return {"drawdown_pct": 0.0, "multiplier": 1.0, "level": "full", "note": ""}
    dd = (equity / high_water_mark - 1.0) * 100.0   # ≤ 0 when below the peak
    dd_pct = round(-dd, 2) if dd < 0 else 0.0
    if dd_pct >= hard:
        return {"drawdown_pct": dd_pct, "multiplier": 0.25, "level": "hard",
                "note": f"Book is {dd_pct:.1f}% off its high — quarter-size new risk until it recovers."}
    if dd_pct >= soft:
        return {"drawdown_pct": dd_pct, "multiplier": 0.50, "level": "soft",
                "note": f"Book is {dd_pct:.1f}% off its high — halve new-position risk."}
    return {"drawdown_pct": dd_pct, "multiplier": 1.0, "level": "full",
            "note": "Book near its high — full risk allowed by the equity brake."}


# ── Portfolio heat ───────────────────────────────────────────────────────────
def portfolio_heat(open_risks: list[float], capital: float,
                   heat_cap_pct: float = 6.0) -> dict:
    """Sum of open risk across positions vs the heat cap.

    open_risks — ₹ at risk per open position (entry−stop × shares, ≥0).
    Over the cap ⇒ the book is over-exposed; the rule is to stop adding / trim.
    """
    total = float(sum(r for r in open_risks if r and r > 0))
    cap_amount = capital * heat_cap_pct / 100.0 if capital else 0.0
    heat_pct = (total / capital * 100.0) if capital else 0.0
    return {
        "open_positions": len([r for r in open_risks if r and r > 0]),
        "total_risk":     round(total, 2),
        "heat_pct":       round(heat_pct, 2),
        "heat_cap_pct":   heat_cap_pct,
        "over_cap":       heat_pct > heat_cap_pct,
        "headroom":       round(max(0.0, cap_amount - total), 2),
    }
