"""Lightweight session auth for Ascent Wealth Labs.

Two roles:
  • admin — full access, including personal positions (My Portfolio, Strategy,
    Guardian alerts).
  • demo  — every screener/analytics tab, but NONE of the owner's positions.

Design notes / honest security posture
--------------------------------------
- Passwords are HASHED (werkzeug PBKDF2) and live ONLY in `.auth_users.json`,
  which is gitignored — plaintext never enters the repo and never sits at rest.
- The Flask session secret persists in `.auth_secret` (gitignored) so sessions
  survive a server restart. If it's ever missing it is regenerated (which just
  logs everyone out — safe).
- This is cookie-session auth, adequate for a small analytics dashboard. On the
  public site it should be served over HTTPS (the session cookie is marked
  Secure only when the request is https). It is NOT a substitute for a real
  identity provider if this ever grows multi-tenant.
- Seed passwords can be overridden at first run via env vars
  ASCENT_ADMIN_PASS / ASCENT_DEMO_PASS. After first run, edit `.auth_users.json`
  (or delete it to re-seed).
"""

import os
import json
import secrets
from functools import wraps

from flask import session, redirect, url_for, request, jsonify
from werkzeug.security import generate_password_hash, check_password_hash

_BASE = os.path.dirname(os.path.abspath(__file__))
_SECRET_FILE = os.path.join(_BASE, ".auth_secret")
_USERS_FILE = os.path.join(_BASE, ".auth_users.json")

# Position features are owner-only. Kept as prefixes so a new /api/portfolio/*
# route can never be accidentally exposed to the demo user.
ADMIN_ONLY_API_PREFIXES = ("/api/portfolio", "/api/strategy", "/api/guardian")


# ── Secret key ───────────────────────────────────────────────────────────────
def get_secret_key() -> str:
    """Return a stable session-signing key, generating+persisting one once."""
    try:
        with open(_SECRET_FILE, "r") as fh:
            key = fh.read().strip()
            if key:
                return key
    except FileNotFoundError:
        pass
    key = secrets.token_hex(32)
    try:
        with open(_SECRET_FILE, "w") as fh:
            fh.write(key)
        os.chmod(_SECRET_FILE, 0o600)
    except OSError:
        pass  # in-memory key still works for this process
    return key


# ── User store ───────────────────────────────────────────────────────────────
def _seed_users() -> dict:
    """First-run seed. Real passwords come from env if set, else the owner's
    chosen defaults so the app works out of the box."""
    # Admin password: NEVER hard-code a real default in a public repo. Use the env
    # var, else seed a random one (printed once) so a fresh clone has no known
    # admin credential. Existing installs keep their `.auth_users.json` untouched.
    admin_pw = os.getenv("ASCENT_ADMIN_PASS")
    if not admin_pw:
        admin_pw = secrets.token_urlsafe(12)
        print(f"[auth] ASCENT_ADMIN_PASS not set — seeded admin 'jai' with a random "
              f"password: {admin_pw}  (set ASCENT_ADMIN_PASS to choose your own).",
              flush=True)
    # Demo password is intentionally public (shown on the login page).
    demo_pw = os.getenv("ASCENT_DEMO_PASS", "demo123")
    return {
        "jai":  {"hash": generate_password_hash(admin_pw), "role": "admin",
                 "display": "Jai"},
        "demo": {"hash": generate_password_hash(demo_pw),  "role": "demo",
                 "display": "Demo User"},
    }


def _load_users() -> dict:
    try:
        with open(_USERS_FILE, "r") as fh:
            data = json.load(fh)
            if isinstance(data, dict) and data:
                return data
    except (FileNotFoundError, json.JSONDecodeError):
        pass
    users = _seed_users()
    try:
        with open(_USERS_FILE, "w") as fh:
            json.dump(users, fh, indent=2)
        os.chmod(_USERS_FILE, 0o600)
    except OSError:
        pass
    return users


def verify(username: str, password: str):
    """Return the user's role on success, else None. Constant-ish time: always
    runs a hash check so a bad username isn't faster than a bad password."""
    username = (username or "").strip().lower()
    users = _load_users()
    rec = users.get(username)
    # Dummy hash so a missing user costs the same as a wrong password.
    ref = rec["hash"] if rec else ("pbkdf2:sha256:600000$x$" + "0" * 64)
    ok = check_password_hash(ref, password or "")
    if rec and ok:
        return rec.get("role", "demo")
    return None


def display_name(username: str) -> str:
    rec = _load_users().get((username or "").lower())
    return (rec or {}).get("display") or (username or "").title()


# ── Session helpers ──────────────────────────────────────────────────────────
def current_user():
    return session.get("user")


def current_role():
    return session.get("role")


def is_admin() -> bool:
    return session.get("role") == "admin"


def can_see_positions() -> bool:
    """The one gate for owner-only position features."""
    return is_admin()


def log_in(username: str, role: str):
    session.permanent = True
    session["user"] = username.lower()
    session["role"] = role


def log_out():
    session.clear()


# ── Decorators (kept for explicit per-route use; the app also enforces auth in
#    a fail-closed before_request hook). ──────────────────────────────────────
def login_required(f):
    @wraps(f)
    def wrapper(*a, **kw):
        if not current_user():
            return redirect(url_for("login", next=request.path))
        return f(*a, **kw)
    return wrapper


def admin_required(f):
    @wraps(f)
    def wrapper(*a, **kw):
        if not current_user():
            return redirect(url_for("login", next=request.path))
        if not is_admin():
            return jsonify({"error": "forbidden"}), 403
        return f(*a, **kw)
    return wrapper
