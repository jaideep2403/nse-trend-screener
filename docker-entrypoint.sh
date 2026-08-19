#!/bin/sh
# ─────────────────────────────────────────────────────────────────────────────
# Runs as ROOT, fixes the /data volume ownership, then drops to appuser.
#
# WHY THIS EXISTS: the app runs as the unprivileged `appuser`, but a host bind
# mount (`-v /var/lib/ascent:/data`) is created root-owned. appuser then cannot
# write /data/.scan_cache, so EVERY scan result-cache write failed silently and
# every scan recomputed cold forever — the "scans are slow on the live site" bug.
# Chowning here, on every start, makes the cache always writable and persistent
# no matter how the volume was created. No manual `chown` on the box ever again.
# ─────────────────────────────────────────────────────────────────────────────
set -e

# Make the whole data volume owned by appuser (idempotent; ~instant when already
# correct). Never fail the boot over it — fall back to running anyway.
if [ -d /data ]; then
  chown -R appuser:appuser /data 2>/dev/null || true
  # Pre-create the caches so first write can't miss the dir.
  mkdir -p /data/.scan_cache /data/nse_bhav_days /data/nse_ohlcv_pkl 2>/dev/null || true
  chown -R appuser:appuser /data/.scan_cache /data/nse_bhav_days /data/nse_ohlcv_pkl 2>/dev/null || true
fi

# Drop root and exec the app (CMD) as appuser. `exec` keeps PID 1 so Docker's
# SIGTERM reaches gunicorn directly (clean, fast shutdown).
exec gosu appuser "$@"
