# Ascent Wealth Labs — NSE Analytics Platform

A self-hosted, end-of-day analytics and screening platform for the Indian equity
market (NSE). It runs entirely on **NSE bhavcopy OHLCV + delivery data** — no paid
data feeds — and turns it into a suite of rule-based screeners, a market-regime
engine, and survivorship-free backtests. Built with Flask + pandas and a
dependency-free vanilla-JS frontend.

> **Not investment advice.** Every output is **mechanical and rule-based**. Nothing
> here is a recommendation to buy or sell. Consult a SEBI-registered adviser before
> making any personal financial decision.

---

## Highlights

- **All-Weather Investing** — a regime-switching book that reads the NIFTY trend
  (200-DMA divide with a 5-session confirmation, à la Faber / Pacer Trendpilot; an
  absolute-momentum gate à la Antonacci dual momentum) and switches engines:
  full risk-on in bull markets, defensive leaders when choppy, and **cash in a
  confirmed downtrend**. Holdings are tilted toward QMJ-style fundamental quality.
- **Defensive Leaders** — a lower-drawdown, delivery-aware alternative to raw
  momentum (low realised volatility, trend smoothness, and NSE delivery-%
  accumulation).
- **20+ screeners** — momentum, breakout, multi-year breakout, emerging leaders,
  early movers, institutional edge, monster growth, and more, over a curated
  ~750-stock (Nifty Total Market) universe.
- **Market regime & breadth** — canonical IBD-style distribution-day / follow-through
  logic, advance-decline, sector Stage-2 breadth, and a realized-vol gauge.
- **Survivorship-free backtesting** — a portfolio-level walk-forward yardstick
  (point-in-time features, next-open fills, cost-net) benchmarked against
  buy-and-hold NIFTYBEES.
- **Two-role auth** — an owner (full access incl. positions) and a read-only demo
  user (every screener, none of the owner's positions).

## Tech stack

- **Python 3.13**, **Flask**, **pandas**, **NumPy**
- Vanilla JS + Jinja2 templates (no build step, works offline / behind a tunnel)
- Local caches: on-disk bhavcopy day cache + SQLite fundamentals store

## Quick start

```bash
pip install -r requirements.txt        # Flask, pandas, numpy, requests, …
python3 app.py                         # serves http://localhost:5050
```

On first run the app downloads and caches recent NSE bhavcopy files, then prewarms
the screeners. A background scheduler keeps the data current.

**Demo login:** `demo` / `demo123` — explore every screener without the owner's
positions.

### Configuration

| Environment variable | Purpose |
| --- | --- |
| `ASCENT_ADMIN_PASS` | Owner (admin) password. If unset, a random one is generated and printed once at first run. |
| `ASCENT_DEMO_PASS`  | Demo password (default `demo123`). |
| `BHAV_DIR` / `OHLCV_DIR` | Override the on-disk data cache locations. |

Credentials are hashed (werkzeug PBKDF2) into `.auth_users.json`, and the session
secret lives in `.auth_secret` — both are gitignored and never leave your machine.

## Data & privacy

- Source data is public NSE bhavcopy (EOD). The fetcher caches aggressively and
  throttles politely — it is not a real-time feed.
- No secrets, credentials, personal positions, or generated databases are committed
  (see `.gitignore`). Several proprietary modules are intentionally local-only.

## License

Released under the [MIT License](LICENSE).
