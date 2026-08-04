"""Production hardening for public deployment (added 2026-08-04).

WHY THIS EXISTS. Everything in this app was written for localhost, where none of
the following mattered. Grepping app.py + auth.py before this module existed found
ZERO of: SESSION_COOKIE_SECURE, SESSION_COOKIE_HTTPONLY, SESSION_COOKIE_SAMESITE,
any CSRF defence, and any login rate limit. On 127.0.0.1 that is fine. On a public
EC2 with a Route53 name it is not:

  * the session cookie would travel without the Secure flag,
  * every POST (/api/portfolio/add, DELETE /api/portfolio/<id>, every scan trigger)
    was reachable cross-site,
  * /login had no throttle at all, and the demo password is printed on the page —
    so the form is a known, named target for credential stuffing of the ADMIN user.

DELIBERATELY NO NEW DEPENDENCIES. Flask-WTF / Flask-Limiter / Talisman are not
installed, and requirements.txt is six lines. Everything here is stdlib + Werkzeug,
so deployment stays `pip install -r requirements.txt` with nothing new to audit.

ACTIVATION. Hardening that would break local development (Secure cookies over
http://127.0.0.1, rejecting curl/test-client POSTs) keys off ASCENT_ENV:

    ASCENT_ENV=production   → full hardening   (set this in the systemd unit)
    anything else / unset   → local dev        (current behaviour preserved)
"""

from __future__ import annotations

import os
import threading
import time
from datetime import timedelta
from urllib.parse import urlparse

from flask import jsonify, request
from werkzeug.middleware.proxy_fix import ProxyFix

IS_PROD = os.getenv("ASCENT_ENV", "dev").strip().lower() == "production"

# ── Login throttle ───────────────────────────────────────────────────────────
# Per-IP, in-memory. A restart clears it, which is acceptable: the window is
# short and gunicorn workers are long-lived. Kept deliberately simple — a Redis
# backed limiter would be one more thing to run on a single-box deploy.
_LOGIN_MAX_ATTEMPTS = 8          # per window, per IP
_LOGIN_WINDOW_SECS  = 300        # 5 minutes
_LOGIN_BLOCK_SECS   = 900        # 15 min lockout once the window is exhausted

_attempts: dict[str, list[float]] = {}
_blocked: dict[str, float] = {}
_lock = threading.Lock()

# Methods that change state and therefore need CSRF protection.
_UNSAFE = {"POST", "PUT", "PATCH", "DELETE"}


def _client_ip() -> str:
    """Real client IP. Behind nginx, remote_addr is 127.0.0.1 for everyone, which
    would make the rate limiter throttle all users as one — ProxyFix (installed
    below in production) rewrites it from X-Forwarded-For."""
    return request.remote_addr or "unknown"


def _same_origin() -> bool:
    """True when the request demonstrably originates from this site.

    An attacker's cross-site form/fetch ALWAYS carries an Origin header, so
    comparing it to the request host is a complete CSRF defence for browsers.
    A missing Origin means a non-browser client (curl, the test client); that is
    rejected only in production, so local tooling and tests keep working.
    """
    origin = request.headers.get("Origin")
    if origin:
        return urlparse(origin).netloc == request.host
    referer = request.headers.get("Referer")
    if referer:
        return urlparse(referer).netloc == request.host
    return not IS_PROD


def _csrf_guard():
    if request.method not in _UNSAFE:
        return None
    if _same_origin():
        return None
    if request.path.startswith("/api/"):
        return jsonify({"error": "cross-origin request rejected"}), 403
    return "Cross-origin request rejected.", 403


def _login_throttle():
    """Throttle only the credential-checking POST — never GETs, never the app."""
    if request.method != "POST" or request.path != "/login":
        return None
    ip = _client_ip()
    now = time.time()
    with _lock:
        until = _blocked.get(ip, 0)
        if until > now:
            retry = int(until - now)
            return (f"Too many sign-in attempts. Try again in {retry // 60 + 1} min.",
                    429, {"Retry-After": str(retry)})
        hits = [t for t in _attempts.get(ip, []) if now - t < _LOGIN_WINDOW_SECS]
        hits.append(now)
        _attempts[ip] = hits
        if len(hits) > _LOGIN_MAX_ATTEMPTS:
            _blocked[ip] = now + _LOGIN_BLOCK_SECS
            _attempts.pop(ip, None)
            return (f"Too many sign-in attempts. Locked for "
                    f"{_LOGIN_BLOCK_SECS // 60} minutes.",
                    429, {"Retry-After": str(_LOGIN_BLOCK_SECS)})
        # opportunistic prune so the dicts can't grow without bound
        if len(_attempts) > 2048:
            for k in [k for k, v in _attempts.items()
                      if not v or now - v[-1] > _LOGIN_WINDOW_SECS]:
                _attempts.pop(k, None)
    return None


def note_successful_login() -> None:
    """Clear the throttle for this IP after a correct password."""
    ip = _client_ip()
    with _lock:
        _attempts.pop(ip, None)
        _blocked.pop(ip, None)


def _headers(resp):
    resp.headers.setdefault("X-Content-Type-Options", "nosniff")
    resp.headers.setdefault("X-Frame-Options", "DENY")
    resp.headers.setdefault("Referrer-Policy", "strict-origin-when-cross-origin")
    # NOT a full CSP: the dashboard is ~13 inline <script> blocks and thousands of
    # inline styles, so script-src/style-src restrictions would break the app
    # outright. These three directives are the ones that cost nothing here and
    # still close clickjacking, plugin injection and <base> hijacking.
    resp.headers.setdefault(
        "Content-Security-Policy",
        "frame-ancestors 'none'; base-uri 'self'; object-src 'none'")
    if IS_PROD and request.is_secure:
        resp.headers.setdefault("Strict-Transport-Security",
                                "max-age=31536000; includeSubDomains")
    return resp


def init_app(app) -> None:
    app.config.update(
        SESSION_COOKIE_HTTPONLY=True,
        SESSION_COOKIE_SAMESITE="Lax",
        SESSION_COOKIE_SECURE=IS_PROD,     # would break plain-http localhost
        PERMANENT_SESSION_LIFETIME=timedelta(days=7),
        MAX_CONTENT_LENGTH=2 * 1024 * 1024,
    )
    if IS_PROD:
        # nginx is the only thing in front, so trust exactly one hop.
        app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1)
    app.before_request(_csrf_guard)
    app.before_request(_login_throttle)
    app.after_request(_headers)
    print(f"[security] hardening active (production={IS_PROD})", flush=True)
