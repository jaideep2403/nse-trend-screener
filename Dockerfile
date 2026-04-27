# ─────────────────────────────────────────────────────────────────────────────
# NSE Market Dashboard — Production Dockerfile
# ─────────────────────────────────────────────────────────────────────────────
# Build:  docker build -t nse-dashboard .
# Run:    docker run -d \
#           -p 5050:5050 \
#           -v nse-data:/data \
#           --name nse-dashboard \
#           nse-dashboard
#
# Volume /data persists across restarts:
#   /data/fundamentals.db          — screener.in fundamentals cache
#   /data/.nse_universe_cache.pkl  — Nifty 500 index symbol cache (7-day TTL)
#   /data/nse_bhav_days/           — NSE bhavcopy daily CSVs (one file per day)
#   /data/nse_ohlcv_pkl/           — per-stock OHLCV DataFrames (6-hour TTL)
# ─────────────────────────────────────────────────────────────────────────────

FROM python:3.13-slim

# Non-root user for security
RUN useradd --create-home --shell /bin/bash appuser

WORKDIR /app

# ── Dependencies ──────────────────────────────────────────────────────────────
# Copy requirements first — Docker caches this layer unless requirements change
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Application code ──────────────────────────────────────────────────────────
COPY . .

# ── Persistent data directory ─────────────────────────────────────────────────
# All runtime data (DB, caches) written here — mount as a named volume.
RUN mkdir -p /data/nse_bhav_days /data/nse_ohlcv_pkl \
    && chown -R appuser:appuser /app /data

USER appuser

# ── Environment ───────────────────────────────────────────────────────────────
# DATA_DIR  → fundamentals.db + nse_universe_cache.pkl
# BHAV_DIR  → bhavcopy daily pkl files (one per trading day, never deleted)
# OHLCV_DIR → per-stock OHLCV DataFrames (rebuilt when newer bhavcopy exists)
ENV DATA_DIR=/data \
    BHAV_DIR=/data/nse_bhav_days \
    OHLCV_DIR=/data/nse_ohlcv_pkl \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

EXPOSE 5050

# ── Gunicorn ──────────────────────────────────────────────────────────────────
# Single worker — the fundamentals background scheduler is a daemon thread that
# must live in exactly one process. Multiple workers would spawn duplicate
# schedulers that fight over fundamentals.db via SQLite writes.
# 4 threads handles concurrent HTTP requests within that one worker.
# 300s timeout — momentum / edge-engine first run can take ~2 min.
CMD ["gunicorn", \
     "--workers=1", \
     "--threads=4", \
     "--timeout=300", \
     "--bind=0.0.0.0:5050", \
     "--access-logfile=-", \
     "--error-logfile=-", \
     "app:app"]
