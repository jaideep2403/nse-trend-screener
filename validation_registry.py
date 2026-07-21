"""Validation registry — the single source of truth for "what is proven".

This enforces prove-before-keep by making it VISIBLE: every gate/factor that
influences ranking must have a survivorship-free, out-of-sample, cost-aware
result recorded here, and anything that failed is listed as REJECTED so it can
never quietly creep back in.

Numbers below are from the SURVIVORSHIP-FREE re-validation on 2026-07-16
(sustained_breakout_validation.py etc., after the harnesses were fixed to pass
survivorship_free=True). They are forward 21-bar ALPHA vs NIFTYBEES (fwd return
minus the benchmark over the same calendar days). Regenerate by re-running the
harnesses and updating these constants — they are the ONLY place the numbers
live, so the UI can never drift from the evidence.
"""

VALIDATION_ASOF = "2026-07-16"
VALIDATION_METHOD = ("Forward 21-bar alpha vs NIFTYBEES · survivorship-free "
                     "(delisted names included) · POINT-IN-TIME liquidity selection "
                     "· dividend-adjusted benchmark · IS/OOS 60/40 split by snapshot")

# Forward-return scorecard by bucket. mean/oos in %, win in %, p10 = downside tail.
# Numbers re-run 2026-07-16 AFTER the review fixes (point-in-time turnover so the
# universe no longer embeds survival + dividend-adjusted benchmark). They are LOWER
# than the prior run — that's the honest, less-biased result. Note "just trending"
# now shows ~zero OOS edge; the sustained gate is where the durable alpha lives.
BUCKET_STATS = {
    "SUSTAINED_BREAKOUT": {
        "label": "Sustained breakout",
        "n": 2726, "mean_alpha": 1.36, "win_pct": 51.8, "tail_p10": -9.15,
        "oos_alpha": 1.26, "verdict": "KEPT",
        "note": "Best forward alpha and the only bucket with a real OUT-OF-SAMPLE edge. The validated buy gate.",
    },
    "ALL_TRENDING": {
        "label": "All trending",
        "n": 5712, "mean_alpha": 0.99, "win_pct": 50.4, "tail_p10": -9.61,
        "oos_alpha": 0.09, "verdict": "BASELINE",
        "note": "The whole trending list — OOS edge is ~zero, so trending alone is not enough.",
    },
    "TRENDING_NO_BREAKOUT": {
        "label": "Trending, no breakout",
        "n": 2096, "mean_alpha": 0.46, "win_pct": 49.0, "tail_p10": -9.73,
        "oos_alpha": 0.05, "verdict": "WEAK",
        "note": "Trending but hasn't broken out — below the baseline, no OOS edge.",
    },
    "FADED_BREAKOUT": {
        "label": "Faded breakout",
        "n": 890, "mean_alpha": 1.12, "win_pct": 49.8, "tail_p10": -11.16,
        "oos_alpha": 1.20, "verdict": "AVOID",
        "note": "Broke out then faded — sub-50% win rate AND the fattest downside tail.",
    },
}

# Which factors/gates are allowed to influence ranking, and why. REJECTED items
# are recorded so the reasoning survives and they don't silently return.
FACTOR_REGISTRY = [
    {"factor": "Delivery %", "status": "KEPT",
     "evidence": "IC +0.087 (strongest single factor), holds OOS",
     "used_in": "trending rank, conviction"},
    {"factor": "Sustained-breakout gate", "status": "KEPT",
     "evidence": "+2.13% vs +1.69% alpha, holds OOS (+1.37% vs +0.49%), survivorship-free",
     "used_in": "trending buy gate"},
    {"factor": "RS vs Nifty (3M/6M blend)", "status": "KEPT",
     "evidence": "Standard momentum factor; positive IC",
     "used_in": "trending, momentum rank"},
    {"factor": "Breadth composite (extremes only)", "status": "KEPT",
     "evidence": "Only ≥12 / ≤4 discriminate; middle is noise",
     "used_in": "regime / exposure switch"},
    {"factor": "Regime gating of ranking (C1)", "status": "REJECTED",
     "evidence": "Picks kept working across every sampled regime — gating added no alpha",
     "used_in": "display-only context, never gates the list"},
    {"factor": "Freshness re-ranking", "status": "REJECTED",
     "evidence": "IC +0.056 POSITIVE — i.e. extended names WIN; ranking fresh-first hurt",
     "used_in": "not used"},
    {"factor": "6-month return (r6m) tilt", "status": "REJECTED",
     "evidence": "No incremental OOS edge over the existing blend",
     "used_in": "not used"},
]


def bucket_for_pick(row: dict) -> str:
    """Classify a trending pick into a validated bucket so the UI can attach its
    real forward-return expectation. Mirrors the harness definitions."""
    if row.get("sustained_breakout"):
        return "SUSTAINED_BREAKOUT"
    # A recent breakout that has faded off its high / lost MA20.
    pct_from_high = row.get("pct_from_high")
    if isinstance(pct_from_high, (int, float)) and pct_from_high <= -8:
        return "FADED_BREAKOUT"
    if row.get("days_since_high") is not None and row.get("days_since_high") <= 25:
        return "FADED_BREAKOUT" if (isinstance(pct_from_high, (int, float))
                                    and pct_from_high < -6) else "TRENDING_NO_BREAKOUT"
    return "TRENDING_NO_BREAKOUT"


def expected_value(row: dict) -> dict | None:
    """The validated forward-return expectation for a single pick."""
    b = bucket_for_pick(row)
    s = BUCKET_STATS.get(b)
    if not s:
        return None
    return {
        "bucket": b, "label": s["label"], "mean_alpha": s["mean_alpha"],
        "win_pct": s["win_pct"], "tail_p10": s["tail_p10"],
        "oos_alpha": s["oos_alpha"], "verdict": s["verdict"],
    }
