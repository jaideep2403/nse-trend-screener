"""
Gunicorn config for the NSE dashboard.

Why gunicorn instead of Flask's dev server:
  - Flask's built-in server is single-threaded synchronous — a second user's
    /api/*/scan POST waits for the first user's scan to finish. Under load
    we observed the process die from OOM when many scans arrived together.
  - Gunicorn forks N worker processes, each handles requests independently,
    and dead workers are auto-respawned by the master.

Production run:
    gunicorn -c gunicorn.conf.py app:app

Local dev: keep using `python3 app.py` (auto-reload + simpler).

Tuning notes:
  - Workers: 2-3 is the right number. Each worker loads ~150-300 MB resident
    when scans run (pandas + 211-day OHLCV cache). On a 2 GB EC2 instance,
    4+ workers will OOM during peak scan activity. Two workers + the scan
    semaphore (in app.py) cap concurrent heavy scans at ~2.
  - Timeout: scans can take 30-130 seconds on cold cache. Default 30s gunicorn
    timeout would kill the worker mid-scan. Raise to 300s for safety.
  - Worker class: `sync` is correct — scans are CPU-bound (pandas), not I/O
    bound, so async workers wouldn't help.
"""
import multiprocessing
import os

# ── Server socket ─────────────────────────────────────────────────────────────
bind             = os.getenv("BIND", "0.0.0.0:5050")
backlog          = 256

# ── Worker processes ──────────────────────────────────────────────────────────
# Cap at 3 even on bigger boxes — RAM (not CPU) is the bottleneck for scans.
_cpu_count   = multiprocessing.cpu_count()
workers      = int(os.getenv("WEB_CONCURRENCY", min(3, max(2, _cpu_count // 2))))
# Env-driven (2026-08-04). For the single-box EC2 deploy the recommended shape is
#   WEB_CONCURRENCY=1 WORKER_CLASS=gthread GUNICORN_THREADS=4
# ONE process, several threads — which is exactly how the app runs locally, and
# how its in-memory caches and the stampede locks added 2026-08-03 are designed
# to work. Extra worker PROCESSES each hold their own copy of those caches, so
# they duplicate every rebuild and multiply NSE polling instead of sharing.
worker_class = os.getenv("WORKER_CLASS", "sync")
threads      = int(os.getenv("GUNICORN_THREADS", 1))

# ── Timeouts ──────────────────────────────────────────────────────────────────
timeout          = int(os.getenv("WORKER_TIMEOUT", 300))   # 5 min — covers cold mbo scan
graceful_timeout = 30
keepalive        = 5

# ── Restart policy ────────────────────────────────────────────────────────────
# Restart each worker after max_requests to prevent slow memory leaks across
# many scans. Jitter avoids synchronised restarts.
max_requests        = 1000
max_requests_jitter = 200

# ── Logging ───────────────────────────────────────────────────────────────────
accesslog        = "-"       # stdout
errorlog         = "-"       # stderr
loglevel         = os.getenv("LOG_LEVEL", "info")

# ── Process naming (visible in `ps aux`) ──────────────────────────────────────
proc_name = "nse-dashboard"

# ── Pre-fork hooks ────────────────────────────────────────────────────────────
def post_fork(server, worker):
    """Each worker process — kick the bhavcopy auto-refresh + scan pre-warm
    only inside the FIRST worker to avoid N duplicate refreshes."""
    # Identify worker index from gunicorn's worker numbering
    worker_index = server.WORKERS.get(worker.pid, 0) if hasattr(server, "WORKERS") else 0
    if worker_index != 1:
        # Disable schedulers in non-primary workers
        os.environ["DISABLE_BHAVCOPY_SCHEDULER"] = "1"
        os.environ["DISABLE_FUND_SCHEDULER"]     = "1"
