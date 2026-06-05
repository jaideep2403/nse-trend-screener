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

# ── Portfolio concurrency (realistic capital constraint) ─────────────────────────
# A real account can only hold so many positions at once. The backtest equity
# curve enforces this: when all slots are full, new signals are SKIPPED (you
# can't deploy capital you don't have). Without this, a backtest "takes" every
# signal and massively overstates returns during signal clusters.
MAX_CONCURRENT_POSITIONS = 20

# ── Backtest history window ──────────────────────────────────────────────────────
# CALENDAR days of bhavcopy to load for the survivorship-free backtest universe.
# Trading days ≈ calendar days × 5/7, so 1500 cal ≈ 1070 weekdays. This MUST
# exceed BT_LOOKBACK_BARS + hold_days + ~10 buffer or every symbol gets filtered
# out by the per-symbol minimum-history gate inside backtest_signal.
# 1500d (~4yr) covers 2022 + 2023 + 2024 bull + 2025 correction → 4 regimes.
BT_LOAD_DAYS = 1500
# Per-symbol bars to walk forward over inside each backtest (~3.2 trading years).
BT_LOOKBACK_BARS = 800
