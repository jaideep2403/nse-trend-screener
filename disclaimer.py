"""ITEM 6 — ONE disclaimer. Single source of truth for the legal footer.

MEASURED PROBLEM (2026-08-03): 13 different `<p class="disclaimer">` strings were
hardcoded across the templates, in 5 substantively different variants:

    "Not financial advice. Data from NSE Bhavcopy. For educational purposes only."   x6
    "Not financial advice. For educational purposes only."                           x1
    "Not financial advice. Data from NSE Bhavcopy."                                  x1
    "Not financial advice. <tab-specific caveat>"                                    x4
    risk_tab's 3-line paragraph — the ONLY one naming a SEBI-registered adviser      x1

That last point is the reason this matters beyond tidiness. The strongest disclaimer
appeared on exactly one tab out of 27; the other 26 said only "not financial advice"
and never pointed the reader to a licensed professional. The operator is NOT a
registered investment adviser, so the SEBI line belongs on EVERY tab, not one.

DESIGN. Two parts, because the variation was not all noise:
  * LEGAL — identical everywhere, non-negotiable, defined once below.
  * NOTE  — an optional per-tab DATA caveat ("Volume is EOD only", "Momentum is a
    trailing indicator"). These are genuinely informative and tab-specific, so they
    are kept, but they are now clearly secondary to the constant legal line.
"""

from __future__ import annotations

# The constant. Identical on all 27 tabs. Change it here and it changes everywhere.
LEGAL = ("Mechanical, rule-based analytics on end-of-day NSE Bhavcopy data — "
         "educational only, not investment advice and not personalised. "
         "Consult a SEBI-registered investment adviser before acting.")

# Optional per-tab DATA caveats, kept because they state a real limitation of the
# numbers on that specific tab. Anything that was purely a restatement of LEGAL was
# dropped rather than duplicated.
NOTES = {
    "momentum":    "Momentum is a trailing indicator.",
    "early-mover": "Early-stage breakouts carry high false-positive rates — this is a "
                   "watchlist generator, not a signal. Confirm follow-through and use a stop.",
    "volspike":    "Volume is end-of-day only; intraday spikes are not captured.",
    "edge":        "The backtest is descriptive, not predictive.",
    "risk":        "The backtest is the survivorship-free forward return the gates "
                   "actually earned, over one market regime (2020-26) only.",
}


def note(tab: str) -> str:
    return NOTES.get(tab, "")


def full(tab: str = "") -> str:
    """LEGAL, plus this tab's data caveat when it has one."""
    n = note(tab)
    return f"{LEGAL} {n}".strip() if n else LEGAL
