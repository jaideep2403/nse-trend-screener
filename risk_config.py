"""
Centralized risk and position-sizing parameters.
TIER-3: all scanners / backtester import from here instead of defining locally,
so a single edit propagates to every tool simultaneously.
"""

# ── Position sizing ────────────────────────────────────────────────────────────
# 5% per trade is realistic when running 15-20 concurrent positions.
# At 100% utilization this gives 20 stocks × 5% = 100% deployed.
POSITION_SIZE_FRAC = 0.05

# ── Stop-loss bounds ───────────────────────────────────────────────────────────
# Maximum stop from entry before sizing collapses too small to be useful.
MAX_STOP_PCT = 0.08      # 8% hard stop — IBD 7-8% loss-cut rule

# Smallest "real" stop we'll quote — prevents division by near-zero when
# two consecutive closes are identical (thin stocks, circuit days).
MIN_RISK_FRAC = 0.005    # 0.5% of price

# ── Risk-reward minimum ────────────────────────────────────────────────────────
MIN_RR_RATIO = 2.0       # need at least 2:1 reward:risk to enter

# ── Backtester cooldown ────────────────────────────────────────────────────────
# After closing a trade on a symbol, don't re-enter for N bars.
# Prevents the backtester from re-entering the same trending stock every bar.
BT_COOLDOWN_BARS = 5
