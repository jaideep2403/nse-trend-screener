"""SINGLE SOURCE OF TRUTH for every performance number the UI displays.

WHY THIS EXISTS. Performance figures were hardcoded directly into template HTML —
`defensive_tab.html`, `all_weather_tab.html`, `index.html`, `risk_tab.html`,
`strategy_tab.html`, `vvv_tab.html`. When the engine was re-validated the code changed
and the copy did not, so the app went on asserting numbers that had been disproved:

  * "+25.4% / −9.8% / Sharpe 1.59" was still shown as the All-Weather result. That
    figure is REAL but only on a LOOSE basis — a 3.4-year window, the
    survivorship-biased quality factor, and MONTH-END sampled drawdown that hides
    roughly 8pp of intra-month pain.
  * "the 200-DMA regime overlay adds the cash-in-bear protection it lacks on its own"
    was measured FALSE: the regime switch sat in cash for 12 of 72 periods, NIFTY rose
    in 9 of them (+2.09%/mo average while out), a −25% cumulative drag. Removing it
    ADDED 3.04pp CAGR and lifted Sharpe 0.65 → 0.99.
  * "Defensive Leaders ... ranked toward low-volatility" — `WIN_VOL_WEIGHT` is now 0.0
    because low-vol is INVERTED on this universe.

Hardcoding is the root cause: there was no one place to update. Every displayed number
now comes from here, so a re-validation updates the UI in the same edit.

RULE: do not change these without re-running the validation that produced them, and
update `MEASURED_ON` + `BASIS` when you do.
"""

from __future__ import annotations

MEASURED_ON = "2026-08-02"
WINDOW = "2020-09-17 → 2026-07-21"

# Every figure below was produced under EXACTLY these conditions. Quoting any of them
# without the basis is how the last set of numbers became misleading.
BASIS = ("price-only (NO survivorship-biased quality factor) · survivorship-free "
         "universe · transaction costs charged · next-open fills · split-corrected "
         "data · TRUE full-daily drawdown (not month-end sampled) · top_k=30 · "
         "monthly rebalance")

# cagr / max_dd (DAILY) / sharpe, plus the out-of-sample second half
TABLE = [
    {"name": "Defensive Leaders (this scan, no risk overlay)",
     "cagr": 22.79, "dd": -16.88, "sharpe": 0.99,
     "oos_cagr": 13.10, "oos_dd": -14.52, "oos_sharpe": 0.51, "highlight": False},
    {"name": "Defensive Leaders + volatility target 10%  (SHIPPED)",
     "cagr": 19.26, "dd": -12.02, "sharpe": 1.00,
     "oos_cagr": 10.27, "oos_dd": -12.02, "oos_sharpe": 0.39, "highlight": True},
    {"name": "NIFTY (buy & hold)",
     "cagr": 14.40, "dd": -16.11, "sharpe": 0.66,
     "oos_cagr": 8.65, "oos_dd": -15.23, "oos_sharpe": 0.27, "highlight": False},
]

HEADLINE = ("Volatility targeting at 10% beats NIFTY on BOTH return and drawdown — "
            "+4.86pp CAGR and 4.09pp less drawdown — and the edge holds out-of-sample. "
            "Without the volatility target the drawdown (−16.88%) is marginally WORSE "
            "than NIFTY's −16.11%, so the risk overlay is doing the work.")

# Stated plainly so the UI can never imply more than was measured.
CAVEATS = [
    "One market regime only (2020-26, a small/mid-cap bull). The engine has never been "
    "tested through a deep bear market.",
    "The out-of-sample return margin is thin (+1.62pp); the drawdown improvement is the "
    "more robust half.",
    "Earlier figures such as '+25.4% / −9.8% / Sharpe 1.59' were measured on a LOOSER "
    "basis — 3.4 years, the survivorship-biased quality factor, and month-end sampled "
    "drawdown. They are not comparable to these.",
    "The regime cash-switch was measured HARMFUL on this book and is no longer part of "
    "the shipped configuration.",
]

DISCLAIMER = ("Backtested on historical data. Mechanical and rule-based; not investment "
              "advice and not personalised. Past results do not predict future returns.")


def summary() -> dict:
    return {"measured_on": MEASURED_ON, "window": WINDOW, "basis": BASIS,
            "table": TABLE, "headline": HEADLINE, "caveats": CAVEATS,
            "disclaimer": DISCLAIMER}
