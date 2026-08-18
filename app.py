import os
import re
import threading
import time
from datetime import timedelta
import pandas as pd
from flask import (Flask, render_template, request, jsonify, make_response,
                   abort, session, redirect, url_for)
import auth
import disclaimer

# ── Live stack dump on demand ────────────────────────────────────────────────
# `kill -USR1 <pid>` writes every thread's Python stack to /tmp/nse-dashboard.err.
# Added 2026-08-03 while chasing a prewarm that stopped making progress with the
# process at ~1% CPU: without this there is no way to see WHICH call is blocked,
# and py-spy needs root on macOS. Costs nothing until the signal is sent.
try:
    import faulthandler as _fh, signal as _sig
    _fh.register(_sig.SIGUSR1, all_threads=True, chain=False)
except Exception:
    pass

# ── Concurrent scan semaphore (F10) ───────────────────────────────────────────
# Cap simultaneous heavy scans across the whole process. Each scan loads
# 200+ pkl files into pandas (~150-300 MB resident peak), so allowing 16
# concurrent scans drove the dev server to OOM. With the cap, excess scan
# threads wait their turn instead of stampeding RAM.
# Override via env: MAX_CONCURRENT_SCANS=4
MAX_CONCURRENT_SCANS = int(os.getenv("MAX_CONCURRENT_SCANS", 2))
_scan_semaphore = threading.BoundedSemaphore(MAX_CONCURRENT_SCANS)

from screener import run_screener
from sector_analysis import run_sector_analysis
from breakout_scanner import run_breakout_scan
from institutional_scanner import run_institutional_scan
from advanced_scanner import run_advanced_scan
from market_breadth import run_market_breadth
from industry_groups import run_industry_analysis, run_rrg_analysis
from momentum_scanner import run_momentum_scan
import emerging_leaders as emerging_leaders_mod
from emerging_leaders import run_emerging_leaders_scan, invalidate_cache as invalidate_emerging_cache
from early_mover_scanner import run_early_mover_scan
from volume_scanner import run_volume_scan
import edge_engine as edge_engine_mod
from edge_engine import run_edge_engine, detect_exit_signals, _load_stocks, invalidate_cache as invalidate_edge_cache
from trending import run_trending_scan
from header_data import get_market_header
from fundamentals import (cache_status as fund_cache_status,
                          scheduler_status as fund_scheduler_status,
                          start_background_scheduler)

app = Flask(__name__)

# ── Session auth ─────────────────────────────────────────────────────────────
# Two roles: admin (Jai — full access incl. positions) and demo (screeners only,
# no positions). Enforcement is FAIL-CLOSED in a before_request hook below, so a
# newly added route is protected by default rather than by remembering a
# decorator. Secret + hashed users live in gitignored files (see auth.py).
# ── Background-job election (added 2026-08-04, for the gunicorn deploy) ───────
# Under `python app.py` there is one process, so these threads are fine. Under
# gunicorn there are N worker PROCESSES and every one of them would start its own
# bhavcopy scheduler, boot prewarm, breadth prime and guardian sweep — N times the
# NSE polling and N concurrent full cache rebuilds. The stampede locks added today
# live in module memory and do NOT span processes, so they cannot prevent that.
# gunicorn.conf.py already tries to set DISABLE_BHAVCOPY_SCHEDULER in post_fork,
# but nothing ever read it (0 references before this change) — and its worker-index
# logic compares a Worker object to 1, so it would have disabled every worker.
# One explicit switch instead: run background jobs unless told not to.
_BG_JOBS = os.getenv("ASCENT_BACKGROUND_JOBS", "1").strip().lower() not in ("0", "false", "no")
if not _BG_JOBS:
    print("[startup] background jobs DISABLED for this process "
          "(ASCENT_BACKGROUND_JOBS=0)", flush=True)

app.secret_key = auth.get_secret_key()
app.permanent_session_lifetime = timedelta(days=14)

# Public-deployment hardening: secure/samesite cookies, CSRF origin check, login
# throttle, security headers. Registered BEFORE _require_login so a cross-origin
# or brute-force attempt is rejected before any auth work happens. No-ops most of
# its behaviour unless ASCENT_ENV=production — see security.py.
import security
security.init_app(app)


# ── #17 — gzip responses over the wire (added 2026-08-13) ───────────────────
# The dashboard HTML is ~722KB uncompressed and was being served raw — measured,
# no Content-Encoding header. HTML/JS/CSS compress ~6:1, so this is the single
# biggest load-time win available and it is nearly free: stdlib gzip in an
# after_request, no new dependency, no template change. Guards: only text-ish
# types, only when the client asks, only above 1KB (tiny bodies aren't worth the
# CPU), never double-encode. Runs BEFORE security._headers via registration order
# so both sets of headers land.
import gzip as _gzip

_COMPRESSIBLE = ("text/html", "text/css", "application/javascript",
                 "text/javascript", "application/json", "image/svg+xml")

@app.after_request
def _compress(resp):
    try:
        if resp.direct_passthrough:
            return resp
        if "gzip" not in (request.headers.get("Accept-Encoding") or ""):
            return resp
        if resp.headers.get("Content-Encoding"):
            return resp
        ctype = (resp.content_type or "").split(";")[0].strip()
        if ctype not in _COMPRESSIBLE:
            return resp
        data = resp.get_data()
        if len(data) < 1024:
            return resp
        packed = _gzip.compress(data, compresslevel=6)
        resp.set_data(packed)
        resp.headers["Content-Encoding"] = "gzip"
        resp.headers["Content-Length"] = str(len(packed))
        vary = resp.headers.get("Vary")
        resp.headers["Vary"] = (vary + ", Accept-Encoding") if vary else "Accept-Encoding"
    except Exception:
        pass
    return resp


# Paths reachable WITHOUT a session. Everything else requires login.
_PUBLIC_PATHS = {"/login", "/logout", "/favicon.ico", "/healthz"}


@app.before_request
def _require_login():
    p = request.path
    if p in _PUBLIC_PATHS or p.startswith("/static/"):
        return None
    if not auth.current_user():
        # APIs get a clean 401 (so fetch() can react); pages redirect to login.
        if p.startswith("/api/"):
            return jsonify({"error": "auth required"}), 401
        return redirect(url_for("login", next=p))
    # Owner-only position APIs — demo is authenticated but not authorised.
    if not auth.is_admin() and p.startswith(auth.ADMIN_ONLY_API_PREFIXES):
        return jsonify({"error": "forbidden"}), 403
    return None


def _login_pulse() -> dict:
    """Small EOD snapshot for the sign-in page hero (idea #10 — 'live data as hero').

    Deliberately server-rendered from market_breadth's IN-MEMORY cache rather than a
    new public /api endpoint: this data changes once per trading day, so polling buys
    nothing, and a new unauthenticated route would be extra attack surface on a page
    that is by definition reachable without a session. Reads only — never computes,
    never raises. Returns {} when the cache is cold, and the template degrades to a
    static hero with no gaps.
    """
    try:
        from market_breadth import _cache as _mb_cache
        d = _mb_cache.get("data") or {}
        if not d:
            return {}
        b  = d.get("breadth") or {}
        rg = d.get("regime") or {}
        highs = [h for h in (d.get("new_highs_list") or []) if h.get("symbol")][:16]
        return {
            "as_of":        d.get("bhavcopy_date"),
            "new_highs":    d.get("new_highs_count"),
            "new_lows":     b.get("new_lows"),
            "advance":      b.get("advance"),
            "decline":      b.get("decline"),
            "pct_above_50": b.get("pct_above_50ma"),
            "universe":     b.get("total_stocks"),
            # Show the SAME plain-English label the in-app header uses, not the raw
            # IBD term — otherwise the login page said "Uptrend Under Pressure" while
            # the header two clicks later said "Sideways" for the identical regime.
            # Mapping is the header's regime_map (_resolve_trend); kept in sync here.
            "regime":       {"Confirmed Uptrend": "Uptrend",
                             "Uptrend Under Pressure": "Sideways",
                             "Correction": "Downtrend",
                             "Downtrend": "Downtrend"}.get(rg.get("regime"), rg.get("regime")),
            "regime_raw":   rg.get("regime"),
            "regime_label": rg.get("label"),
            # Same fields the signed-in ticker renders (symbol/price/r3m/vol_ratio),
            # so the landing tape and the in-app tape are the same component with
            # the same data rather than two lookalikes that drift apart.
            "highs": [{"symbol": h.get("symbol"), "price": h.get("price"),
                       "r3m": h.get("r3m"), "vol_ratio": h.get("vol_ratio")}
                      for h in highs],
        }
    except Exception:
        return {}


@app.route("/login", methods=["GET", "POST"])
def login():
    next_url = request.args.get("next") or ""
    # Only allow same-site relative redirects (no open-redirect via ?next=http…).
    if not next_url.startswith("/") or next_url.startswith("//"):
        next_url = ""
    error = None
    last_user = ""
    if request.method == "POST":
        username = (request.form.get("username") or "").strip()
        password = request.form.get("password") or ""
        last_user = username
        role = auth.verify(username, password)
        if role:
            auth.log_in(username, role)
            security.note_successful_login()   # clear this IP's throttle counter
            return redirect(next_url or url_for("index"))
        error = "Incorrect username or password."
    # GET when a valid session already exists: DON'T silently pass through to the
    # app (that reads like an auth bypass). Render the login page with an explicit
    # "you're already signed in" state — the user can continue, switch accounts, or
    # sign out. Unauthenticated access is still fully blocked by _require_login.
    signed_in_as = None
    if request.method == "GET" and auth.current_user():
        signed_in_as = auth.display_name(auth.current_user())
        last_user = auth.current_user()
    return render_template("login.html", error=error, next_url=next_url,
                           last_user=last_user, signed_in_as=signed_in_as,
                           pulse=_login_pulse())


@app.route("/logout")
def logout():
    auth.log_out()
    return redirect(url_for("login"))


@app.route("/healthz")
def healthz():
    return "ok", 200

# ── Safe JSON: convert NaN/Inf → null so responses are valid JSON ─────────────
# Python's json (and Flask's default provider) emit bare `NaN`/`Infinity` tokens,
# which are NOT valid JSON — browsers' JSON.parse()/fetch().json() reject them
# with "Unexpected token 'N'". This bites any scanner that round-trips data
# through a pandas DataFrame: None values in a numeric column become float NaN
# (df.to_dict()), then serialize as `NaN`. A single global provider fixes every
# endpoint at once (momentum, emerging, trending, edge, …) instead of patching
# each scanner. Cost is one shallow recursive pass per response (negligible).
import math as _math
from flask.json.provider import DefaultJSONProvider as _DefaultJSONProvider

def _json_safe(o):
    if isinstance(o, float):
        return None if (_math.isnan(o) or _math.isinf(o)) else o
    if isinstance(o, dict):
        return {k: _json_safe(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_json_safe(v) for v in o]
    return o

class _SafeJSONProvider(_DefaultJSONProvider):
    def dumps(self, obj, **kwargs):
        return super().dumps(_json_safe(obj), **kwargs)

app.json = _SafeJSONProvider(app)

# ── LOCAL-ONLY: Personal Portfolio (gitignored) ───────────────────────────────
# Gracefully no-op if portfolio.py is missing (e.g. on EC2 deploy from GitHub).
PORTFOLIO_AVAILABLE = False
try:
    import portfolio as _pf
    PORTFOLIO_AVAILABLE = True
except Exception as _pf_e:
    print(f"[portfolio] disabled (file not present): {_pf_e}")

# ── LOCAL-ONLY: ⚔️ Strategy Engine (gitignored) ───────────────────────────────
STRATEGY_AVAILABLE = False
try:
    import strategy_engine as _strat
    STRATEGY_AVAILABLE = True
except Exception as _st_e:
    print(f"[strategy] disabled (file not present): {_st_e}")

if STRATEGY_AVAILABLE:
    import threading as _st_threading
    _strat_lock  = _st_threading.Lock()
    _strat_state = {"running": False, "result": None, "error": None,
                    "progress": 0, "total": 100, "message": ""}

    def _do_strategy_scan(force: bool):
        def _cb(done, total, msg):
            with _strat_lock:
                _strat_state.update({"progress": done, "total": total, "message": msg})
        try:
            res = _strat.run_strategy(progress_callback=_cb, force=force)
            with _strat_lock:
                _strat_state.update({"result": res, "running": False,
                                     "error": res.get("error")})
        except Exception as e:
            with _strat_lock:
                _strat_state.update({"error": str(e), "running": False})

    @app.route("/api/strategy/scan", methods=["POST"])
    def api_strategy_scan():
        force = (request.args.get("force") == "true")
        with _strat_lock:
            if _strat_state["running"]:
                return jsonify({"status": "already_running"}), 409
            _strat_state.update({"running": True, "result": None, "error": None,
                                 "progress": 0, "message": ""})
        _st_threading.Thread(target=_do_strategy_scan, args=(force,),
                             daemon=True).start()
        return jsonify({"status": "started"})

    @app.route("/api/strategy/status")
    def api_strategy_status():
        with _strat_lock:
            s = dict(_strat_state)
        pct = round(s["progress"] / s["total"] * 100, 1) if s["total"] else 0
        return jsonify({"running": s["running"], "pct": pct,
                        "message": s["message"], "result": s["result"],
                        "error": s["error"]})

    @app.route("/api/strategy/config", methods=["GET", "POST"])
    def api_strategy_config():
        try:
            if request.method == "POST":
                cfg = _strat.save_config(request.get_json(force=True) or {})
            else:
                cfg = _strat.load_config()
            return jsonify(cfg)
        except Exception as e:
            return jsonify({"error": str(e)}), 500

# ── 🏆 Sector Leaders (sector-rotation + leaders) ─────────────────────────────
SECTOR_LEADERS_AVAILABLE = False
try:
    import sector_leaders as _secld
    SECTOR_LEADERS_AVAILABLE = True
except Exception as _secld_e:
    print(f"[sector-leaders] disabled: {_secld_e}")

if SECTOR_LEADERS_AVAILABLE:
    import threading as _secld_threading
    _secld_lock  = _secld_threading.Lock()
    _secld_state = {"running": False, "result": None, "error": None,
                    "progress": 0, "total": 100, "message": ""}

    def _do_sector_leaders_scan(force: bool):
        try:
            res = _secld.run_sector_leaders(force=force)
            with _secld_lock:
                _secld_state.update({"result": res, "running": False,
                                     "progress": 100,
                                     "error": res.get("error")})
        except Exception as e:
            with _secld_lock:
                _secld_state.update({"error": str(e), "running": False})

    @app.route("/api/sector-leaders/scan", methods=["POST"])
    def api_sector_leaders_scan():
        force = (request.args.get("force") == "true")
        with _secld_lock:
            if _secld_state["running"]:
                return jsonify({"status": "already_running"}), 409
            _secld_state.update({"running": True, "result": None, "error": None,
                                 "progress": 0, "message": ""})
        _secld_threading.Thread(target=_do_sector_leaders_scan, args=(force,),
                                daemon=True).start()
        return jsonify({"status": "started"})

    @app.route("/api/sector-leaders/status")
    def api_sector_leaders_status():
        with _secld_lock:
            s = dict(_secld_state)
        pct = round(s["progress"] / s["total"] * 100, 1) if s["total"] else 0
        # POST-RESTART FIX (2026-08-02): _secld_state is in-memory and only filled by
        # the scan POST, so this tab read EMPTY after every restart even though
        # run_sector_leaders() keeps its own cache and returns in ~2s.
        result = s["result"]
        if not s["running"] and result is None:
            try:
                import sector_leaders as _sl
                result = _sl.run_sector_leaders()
            except Exception as _e:
                print(f"[sector-leaders] status fallback failed: {_e}", flush=True)
        return jsonify({"running": s["running"], "pct": pct,
                        "message": s["message"], "result": result,
                        "error": s["error"]})

if PORTFOLIO_AVAILABLE:
    @app.route("/api/portfolio", methods=["GET"])
    def api_portfolio_list():
        try:
            return jsonify(_pf.portfolio_summary())
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @app.route("/api/portfolio/add", methods=["POST"])
    def api_portfolio_add():
        try:
            data = request.get_json(force=True) or {}
            _pf.add_position(
                symbol=data.get("symbol"),
                qty=float(data.get("qty", 0)),
                entry_price=float(data.get("entry_price", 0)),
                entry_date=data.get("entry_date"),
            )
            return jsonify({"ok": True})
        except Exception as e:
            # Surface fuzzy suggestions on validation failure
            sym = (data.get("symbol") if isinstance(data, dict) else None) or ""
            try:
                v = _pf.validate_symbol(sym)
                if not v["valid"] and v.get("suggestions"):
                    return jsonify({"error": str(e),
                                    "suggestions": v["suggestions"]}), 400
            except Exception:
                pass
            return jsonify({"error": str(e)}), 400

    @app.route("/api/portfolio/validate", methods=["GET"])
    def api_portfolio_validate():
        sym = request.args.get("symbol", "")
        try:
            return jsonify(_pf.validate_symbol(sym))
        except Exception as e:
            return jsonify({"valid": False, "error": str(e),
                            "suggestions": []}), 200

    @app.route("/api/portfolio/<pos_id>", methods=["DELETE"])
    def api_portfolio_delete(pos_id):
        try:
            ok = _pf.delete_position(pos_id)
            return jsonify({"ok": ok})
        except Exception as e:
            return jsonify({"error": str(e)}), 500

# ── LOCAL-ONLY: Investment Grade Scanner (gitignored) ─────────────────────────
INVESTGRADE_AVAILABLE = False
try:
    import investment_grade as _ig
    INVESTGRADE_AVAILABLE = True
except Exception as _ig_e:
    print(f"[investgrade] disabled (file not present): {_ig_e}")

if INVESTGRADE_AVAILABLE:
    ig_state = {
        "running": False, "result": None, "error": None,
        "progress": 0, "total": 0, "message": "",
    }
    ig_lock = threading.Lock()

    def _do_ig_scan():
        def progress_cb(done, total, msg):
            with ig_lock:
                ig_state["progress"] = done
                ig_state["total"] = total
                ig_state["message"] = msg
        try:
            with _scan_semaphore:
                result = _ig.run_investment_grade_scan(progress_callback=progress_cb)
            with ig_lock:
                ig_state["result"] = result
                ig_state["running"] = False
        except Exception as e:
            with ig_lock:
                ig_state["error"] = str(e)
                ig_state["running"] = False

    @app.route("/api/investment_grade/scan", methods=["POST"])
    def api_ig_scan():
        with ig_lock:
            if ig_state["running"]:
                return jsonify({"status": "already_running"})
            ig_state.update({"running": True, "result": None, "error": None,
                             "progress": 0, "total": 0, "message": "Starting…"})
        threading.Thread(target=_do_ig_scan, daemon=True).start()
        return jsonify({"status": "started"})

    @app.route("/api/investment_grade/status")
    def api_ig_status():
        with ig_lock:
            s = dict(ig_state)
        # Fall back to module cache if no in-memory result yet
        if s["result"] is None and not s["running"]:
            try:
                if _ig._cache.get("data"):
                    s["result"] = _ig._cache["data"]
            except Exception:
                pass
        s["result"] = _annotate_breakdown_payload(s["result"])
        return jsonify(s)

# ── LOCAL-ONLY: Alpha Engine — Multi-Factor Composite Scorer (gitignored) ─────
ALPHA_AVAILABLE = False
try:
    import alpha_engine as _ae
    ALPHA_AVAILABLE = True
except Exception as _ae_e:
    print(f"[alpha_engine] disabled (file not present): {_ae_e}")

# ── Sector mapper — background refresh on startup ─────────────────────────────
try:
    from sector_mapper import refresh_sector_cache as _refresh_sectors
    if _BG_JOBS:                        # this one reaches out to NSE — leader only
        _refresh_sectors(background=True)   # builds .sector_cache.json in ~2s
        print("[sector_mapper] Auto-sector cache refresh started (background)")
except Exception as _sm_e:
    print(f"[sector_mapper] not available: {_sm_e}")

if ALPHA_AVAILABLE:
    alpha_state = {
        "running": False, "result": None, "error": None,
        "progress": 0, "total": 100, "message": "", "started_at": None,
    }
    alpha_lock = threading.Lock()

    def _do_alpha_scan():
        def progress_cb(done, total, msg):
            with alpha_lock:
                alpha_state["progress"] = done
                alpha_state["total"]    = total
                alpha_state["message"]  = msg
        try:
            result = _ae.run_alpha_scan(progress_callback=progress_cb)
            with alpha_lock:
                alpha_state["result"]  = result
                alpha_state["running"] = False
        except Exception as e:
            with alpha_lock:
                alpha_state["error"]   = str(e)
                alpha_state["running"] = False

    @app.route("/api/alpha/scan", methods=["POST"])
    def api_alpha_scan():
        with alpha_lock:
            if alpha_state["running"]:
                return jsonify({"status": "already_running"})
            alpha_state.update({
                "running": True, "result": None, "error": None,
                "progress": 0, "total": 100, "message": "Starting…",
            })
        threading.Thread(target=_do_alpha_scan, daemon=True).start()
        return jsonify({"status": "started"})

    @app.route("/api/alpha/status")
    def api_alpha_status():
        with alpha_lock:
            s = dict(alpha_state)
        if s["result"] is None and not s["running"]:
            try:
                if _ae._cache.get("data"):
                    s["result"] = _ae._cache["data"]
            except Exception:
                pass
        pct = round(s["progress"] / max(s["total"], 1) * 100, 1)
        s["result"] = _annotate_breakdown_payload(s["result"])
        return jsonify({
            "running":  s["running"],
            "pct":      pct,
            "message":  s["message"],
            "result":   s["result"],
            "error":    s["error"],
        })

# Inject flag into all templates (read by index.html to show/hide the tab)
@app.context_processor
def inject_portfolio_flag():
    # Position features are owner-only. For the demo role we force them OFF so the
    # My Portfolio + Strategy tabs never render — this is the UI half; the API half
    # is enforced fail-closed in _require_login().
    _pos = auth.can_see_positions()
    return {
        "role":                auth.current_role(),
        "user_display":        auth.display_name(auth.current_user() or ""),
        "can_see_positions":   _pos,
        "portfolio_enabled":   PORTFOLIO_AVAILABLE and _pos,
        "strategy_enabled":    STRATEGY_AVAILABLE and _pos,
        "investgrade_enabled": INVESTGRADE_AVAILABLE,
        "alpha_enabled":       ALPHA_AVAILABLE,
        "monster_enabled":        MONSTER_AVAILABLE,
        "early_growth_enabled":   EARLY_GROWTH_AVAILABLE,
        "vvv_enabled":         globals().get("VVV_AVAILABLE", False),
        "universe_label":      "Nifty Total Market",
        "universe_label_short": "NSE Total Market",
        # ITEM 6: one disclaimer constant. See disclaimer.py — the SEBI-adviser line
        # used to appear on 1 tab of 27; it now appears on all of them.
        "DISCLAIMER":          disclaimer.LEGAL,
        "disclaimer_note":     disclaimer.note,
    }

# ── Screener state ─────────────────────────────────────────────────────────────
scan_state = {
    "running": False,
    "progress": 0,
    "total": 0,
    "current_ticker": "",
    "result": None,
    "funnel": None,
    "error": None,
    "started_at": None,
}
scan_lock = threading.Lock()


def do_scan(params):
    def progress_cb(done, total, ticker):
        with scan_lock:
            scan_state["progress"] = done
            scan_state["total"]    = total
            scan_state["current_ticker"] = ticker

    try:
        with _scan_semaphore:
            df, funnel = run_screener(params=params, progress_callback=progress_cb)
        with scan_lock:
            scan_state["result"]  = df.to_dict(orient="records")
            scan_state["funnel"]  = funnel
            scan_state["running"] = False
    except Exception as e:
        with scan_lock:
            scan_state["error"]   = str(e)
            scan_state["running"] = False


# ── Sector state ───────────────────────────────────────────────────────────────
sector_state = {
    "running": False,
    "result":  None,
    "error":   None,
    "started_at": None,
}
sector_lock = threading.Lock()


def do_sector_analysis():
    try:
        with _scan_semaphore:
            result = run_sector_analysis()
        with sector_lock:
            sector_state["result"]  = result
            sector_state["running"] = False
    except Exception as e:
        with sector_lock:
            sector_state["error"]   = str(e)
            sector_state["running"] = False


# ── Routes ─────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    resp = make_response(render_template("index.html"))
    resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    resp.headers["Pragma"] = "no-cache"
    resp.headers["Expires"] = "0"
    return resp


# ── Per-tab deep-link routing ────────────────────────────────────────────────
# Each tab gets its own clean URL (e.g. /trending_stocks, /strategy). They ALL
# serve the same single-page app; the frontend reads the path and opens the
# matching tab. This is URL routing (deep-linking), NOT microservices — one app,
# one process. A single-segment path that matches a known slug serves the SPA;
# anything else 404s. /api/* and /static/* contain a slash so they never match
# this single-segment rule and are untouched.
TAB_SLUGS = {
    "screener", "portfolio", "investment_grade", "trending_stocks", "strategy",
    "multiyear_breakout", "breakout_stocks", "edge_engine", "momentum",
    "emerging_leaders", "sector_rotation", "volume_spike", "market_breadth",
    "industry_groups", "advanced_setups", "institutional_edge", "alpha_engine",
    "monster_growth", "minervini_vvv", "consensus",
    "early_movers", "early_growth", "sector_leaders", "risk_regime", "defensive_leaders",
    "all_weather", "promoter_activity",
}


@app.route("/<tab_slug>")
def index_tab(tab_slug):
    """Serve the SPA for a per-tab deep link; the frontend opens the tab.
    Unknown single-segment paths (favicon.ico, typos, …) 404."""
    # Owner-only tabs: a demo deep-link lands on the app home instead of a tab
    # that won't render for them.
    if tab_slug in {"portfolio", "strategy"} and not auth.can_see_positions():
        return redirect(url_for("index"))
    if tab_slug in TAB_SLUGS:
        return index()
    abort(404)


# ── Cross-tab stock search ───────────────────────────────────────────────────
# "Where does this stock show up?" — answers in one place instead of making the
# user open 20 tabs. Reads each scanner's CACHED result only (never recomputes:
# a search must be instant), so a tab whose scan hasn't run today simply has no
# cache — reported as `not_scanned` so the answer is honestly partial rather
# than a misleading "not found".
# (cache key, url slug, label, icon id). The icon id refers to a <symbol> in the
# template's inline SVG sprite — the same set the sidebar uses, so the search
# dropdown and the nav can never drift apart. Labels are emoji-free by design.
_SEARCH_SOURCES = [
    ("trending",      "trending_stocks",    "Trending Stocks",    "i-flame"),
    ("momentum",      "momentum",           "Momentum",           "i-trend"),
    ("breakout",      "breakout_stocks",    "Breakout Stocks",    "i-zap"),
    ("multiyear",     "multiyear_breakout", "Multi-Yr Breakout",  "i-mountain"),
    ("emerging",      "emerging_leaders",   "Emerging Leaders",   "i-sprout"),
    ("early_mover",   "early_movers",       "Early Movers",       "i-search"),
    ("advanced",      "advanced_setups",    "Advanced Setups",    "i-target"),
    ("institutional", "institutional_edge", "Institutional Edge", "i-bank"),
    ("monster",       "monster_growth",     "Monster Growth",     "i-flame"),
    ("early_growth",  "early_growth",       "Early Growth",       "i-sprout"),
    ("vvv",           "minervini_vvv",      "Minervini VVV",      "i-target"),
    ("edge",          "edge_engine",        "Edge Engine",        "i-cpu"),
    ("volume",        "volume_spike",       "Vol Spike",          "i-bars"),
]
_LIST_KEYS = ("stocks", "results", "ranked", "rows", "candidates", "picks", "leaders")


def _search_rows(result):
    if not isinstance(result, dict):
        return []
    for k in _LIST_KEYS:
        v = result.get(k)
        if isinstance(v, list) and v and isinstance(v[0], dict):
            return v
    return []


# Some scanners bake emoji into their DATA (edge_engine's tier is literally "🥇 A"),
# so the search dropdown can't be de-emoji'd by renaming labels alone. Strip them at
# render time; the scanner's own tab still shows its native tier string.
_EMOJI_RE = re.compile(
    "[\U0001F000-\U0001FAFF\U00002190-\U000021FF\U00002300-\U000027BF"
    "\U00002B00-\U00002BFF\U0000FE0F\U00002600-\U000026FF]"
)


def _strip_emoji(s: str) -> str:
    return _EMOJI_RE.sub("", str(s)).strip()


# ── Canonical market-wide RS Rating ──────────────────────────────────────────
# Each scanner computes "RS" as a percentile rank WITHIN ITS OWN scan set, so the
# same stock showed RS 96 (momentum, pool 799) / 98 (breakout, 342) / 76
# (emerging, 345) — confusing and not a true IBD RS Rating. This is the ONE honest
# market-wide RS: percentile of 3-month return across the FULL universe (1-99),
# the same value everywhere. Cached per bhavcopy so a search stays instant.
_rs_map_cache = {"tag": None, "map": {}}


def _market_rs_map() -> dict:
    import result_cache as _rc
    tag = _rc._tag() if hasattr(_rc, "_tag") else None
    if _rs_map_cache["tag"] == tag and _rs_map_cache["map"]:
        return _rs_map_cache["map"]
    try:
        import industry_groups as _ig
        stocks = _ig._get_stocks()
        r3m = {}
        for s, df in stocks.items():
            c = df["Close"]
            if len(c) > 64:
                r3m[s] = float(c.iloc[-1] / c.iloc[-64] - 1)
        if not r3m:
            return {}
        ser = pd.Series(r3m)
        ranks = (ser.rank(pct=True) * 99).round().clip(1, 99).astype(int).to_dict()
        _rs_map_cache.update({"tag": tag, "map": ranks})
        return ranks
    except Exception:
        return {}


def _search_summary(row) -> str:
    """Compact one-liner: WHY this symbol is on that tab."""
    bits = []
    if isinstance(row.get("rank"), (int, float)):
        bits.append(f"#{int(row['rank'])}")
    for k in ("score", "emergence_score", "setup_score", "composite_score"):
        v = row.get(k)
        if isinstance(v, (int, float)):
            bits.append(f"score {round(float(v), 1)}")
            break
    if row.get("tier"):
        tier = _strip_emoji(row["tier"])
        if tier:
            bits.append(tier)
    # Per-row "RS {int}" was DROPPED here: each scanner's rs_rating/rs_rank is a
    # percentile WITHIN ITS OWN scan set (different pool per tab), so printing them
    # side by side showed the same stock as RS 96 / 98 / 76 — not comparable, not a
    # true RS Rating. The ONE canonical market-wide RS now lives in the dropdown
    # header (rs_market). Only rs_composite stays here — it's a distinct, correctly
    # labelled metric (excess return vs Nifty, in points).
    v = row.get("rs_composite")
    if isinstance(v, (int, float)):
        bits.append(f"RS vs Nifty {'+' if v >= 0 else ''}{round(float(v), 1)}%")
    # Pattern / timeframe (breakout-style rows) so they're not left blank now that
    # the per-scan RS is gone.
    pats = row.get("patterns")
    if isinstance(pats, list) and pats:
        tf = row.get("timeframes")
        tf_str = ("/".join(str(t) for t in tf) + " ") if isinstance(tf, list) and tf else ""
        bits.append(f"{tf_str}{_strip_emoji(str(pats[0]))}")
    if isinstance(row.get("buyer_demand"), str) and row["buyer_demand"]:
        bits.append(f"demand {row['buyer_demand']}")
    if row.get("sustained_breakout"):
        bits.append("sustained breakout")
    if row.get("entry_window"):
        bits.append(_strip_emoji(row["entry_window"]))
    return " · ".join(b for b in bits[:4] if b)


@app.route("/api/new-highs")
def api_new_highs():
    """The day's 52-week-high makers — powers the header ticker tape.

    Reads ONLY the cached Market Breadth result (`new_highs_list`, already
    computed there) — never recomputes. This is honestly an END-OF-DAY fact:
    "stocks that closed at a 52-week high on <bhavcopy_date>" changes once per
    trading day, so a scrolling tape of it implies nothing about live ticks.
    """
    # Read market_breadth's LIVE module cache — the same source the header trend
    # pill uses. Do NOT rely on result_cache here: market_breadth keeps its own
    # 6h persisted cache, so run_market_breadth() usually short-circuits and
    # never reaches result_cache.put — the scan_cache copy can sit on an older
    # code-version tag for hours and read as permanently "pending".
    # The bhavcopy scheduler clears this module cache when new data lands, so the
    # next prewarm recompute repopulates it and the tape swaps automatically.
    cached = None
    try:
        from market_breadth import _cache as _mb_cache
        cached = _mb_cache.get("data")
    except Exception:
        cached = None
    if not cached:
        import result_cache as _rc
        cached = _rc.get("breadth")
    # No data = breadth hasn't computed for THIS bhavcopy yet (boot, or a new
    # bhavcopy just landed and the prewarm is recomputing). That is NOT the same
    # as "today genuinely had zero new highs" — the UI must not confuse the two.
    pending = not cached
    cached = cached or {}
    lst = cached.get("new_highs_list") or []
    nh_count = cached.get("new_highs_count")   # TRUE count (list is a top-N tape slice)
    as_of = cached.get("bhavcopy_date")

    # SELF-HEAL (2026-07-20): the breadth module cache can be built DURING the
    # bhavcopy rollover with an incomplete data snapshot — it then gets stamped with
    # today's date but an EMPTY new_highs_list, and nothing invalidates a same-date
    # cache for up to 6h. So the tape showed "No new 52-week highs" on a day 20+
    # stocks actually made new highs. When the cached list is empty (and we're not
    # genuinely still computing), recompute the LIGHT new-highs list from current
    # data — one cheap pass, no scoring — and trust that instead of a stale zero.
    # Cached in a module-level slot so the frequently-polled tape doesn't recompute.
    if not lst and not pending:
        try:
            import industry_groups as _ig
            from market_breadth import _new_high_stocks
            cur_stocks = _ig._get_stocks()
            cur_date = None
            for _df in cur_stocks.values():
                cur_date = str(_df.index[-1].date()); break
            slot = globals().setdefault("_nh_fallback", {"date": None, "list": []})
            if slot["date"] != cur_date:
                slot["date"] = cur_date
                slot["list"] = _new_high_stocks(cur_stocks)   # full list
            if slot["list"]:
                lst = slot["list"]
                nh_count = len(slot["list"])
                as_of = cur_date
        except Exception:
            pass

    # True count = new_highs_count when present (list is a top-N tape slice);
    # otherwise fall back to the list length (self-heal path returns the full list).
    count = nh_count if nh_count is not None else len(lst)
    return jsonify({
        "as_of": as_of,
        "count": count,
        "pending": pending,
        "stocks": [{
            "symbol":    x.get("symbol"),
            "price":     x.get("price"),
            "r3m":       x.get("r3m"),
            "vol_ratio": x.get("vol_ratio"),
        } for x in lst[:60] if x.get("symbol")],
    })


@app.route("/api/stock-search")
def api_stock_search():
    import result_cache as _rc
    q = (request.args.get("q") or "").strip().upper()
    if len(q) < 2:
        return jsonify({"query": q, "symbol": None, "hits": [],
                        "suggestions": [], "not_scanned": []})

    loaded, not_scanned = [], []
    for key, slug, label, icon in _SEARCH_SOURCES:
        cached = _rc.get(key)
        if cached is None:
            not_scanned.append(label)
            continue
        loaded.append((slug, label, icon, _search_rows(cached)))

    all_syms = set()
    for *_, rows in loaded:
        for r in rows:
            s = str(r.get("symbol") or r.get("Symbol") or "").upper()
            if s:
                all_syms.add(s)

    if q in all_syms:
        symbol = q
    else:
        pref = sorted(s for s in all_syms if s.startswith(q))
        cont = sorted(s for s in all_syms if q in s and not s.startswith(q))
        symbol = (pref or cont or [None])[0]
    suggestions = sorted(s for s in all_syms if s.startswith(q) and s != symbol)[:6]

    hits = []
    if symbol:
        for slug, label, icon, rows in loaded:
            for r in rows:
                if str(r.get("symbol") or r.get("Symbol") or "").upper() == symbol:
                    hits.append({"tab": slug, "label": label, "icon": icon,
                                 "summary": _search_summary(r)})
                    break
        # Do you already own it? Read the raw store — never the enriched
        # list_positions(), which re-runs per-position analysis (far too slow
        # for a keystroke-latency endpoint).
        if PORTFOLIO_AVAILABLE and auth.can_see_positions():
            try:
                import portfolio as _pf
                store = _pf._load_store()
                poss = store if isinstance(store, list) else (store or {}).get("positions", [])
                for p in poss:
                    if str(p.get("symbol", "")).upper() == symbol:
                        hits.insert(0, {
                            "tab": "portfolio", "label": "My Portfolio", "icon": "i-briefcase",
                            "summary": f"you hold {p.get('qty')} @ ₹{p.get('entry_price')}",
                        })
                        break
            except Exception:
                pass

    rs_market = _market_rs_map().get(symbol) if symbol else None
    return jsonify({"query": q, "symbol": symbol, "hits": hits, "rs_market": rs_market,
                    "suggestions": suggestions, "not_scanned": not_scanned})


# ── Live presence — count of UNIQUE visitors active in the last minute ───────
# A browser sends its random localStorage id here every ~30s; we keep
# {visitor_id: last_seen} in memory and return how many were seen within the
# window. Microsecond endpoint (a dict write + prune) — safe for the box.
_presence: dict[str, float] = {}
_presence_lock = threading.Lock()
PRESENCE_WINDOW = 60   # seconds a visitor counts as "online"


@app.route("/api/presence")
def api_presence():
    vid = (request.args.get("id") or "").strip()[:64]
    now = time.time()
    with _presence_lock:
        if vid:
            _presence[vid] = now
        cutoff = now - PRESENCE_WINDOW
        for _k in [k for k, t in _presence.items() if t < cutoff]:
            del _presence[_k]
        n = len(_presence)
    return jsonify({"online": n})


@app.route("/api/header")
def api_header():
    """Lightweight header: Nifty 50 level, day change %, adv/dec, market trend.
    Also carries Position Guardian alerts (pure guardian.db read — fast)."""
    try:
        h = get_market_header()
        # Guardian alerts are derived from the OWNER's held positions — never send
        # them to a demo session.
        if auth.can_see_positions():
            try:
                import guardian
                h["guardian_alerts"] = guardian.get_active_alerts()
            except Exception:
                h["guardian_alerts"] = []
        else:
            h["guardian_alerts"] = []
        return jsonify(h)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── Risk & Regime engine (capital protection + sizing + honest scorecard) ────
def canonical_regime_state() -> str | None:
    """THE single market-regime accessor. BULL / SIDEWAYS / BEAR, or None.

    Added 2026-08-13 to finish the consolidation. Two engines used to answer the
    same visible question and disagreed outright — regime_engine said BEAR
    ("Raise cash, do not initiate new longs") while market_breadth said "Uptrend
    Under Pressure" (which the header renders "Sideways"). Portfolio advice was
    reading the first and every other surface the second, so a BEAR-adj chip sat
    two inches under a "Sideways" header. Everything that DISPLAYS a regime now
    routes through here; regime_engine remains only for the backtest's validated
    entry gate, where its own definition is what was measured.
    """
    try:
        from market_breadth import _cache as _mb
        r = ((_mb.get("data") or {}).get("regime") or {}).get("regime")
        return {"Confirmed Uptrend": "BULL", "Uptrend Under Pressure": "SIDEWAYS",
                "Correction": "BEAR", "Downtrend": "BEAR"}.get(r)
    except Exception:
        return None


def _current_regime() -> str:
    """Canonical regime label from Market Breadth's cache (single source of truth)."""
    try:
        import market_breadth
        data = (market_breadth._cache or {}).get("data") or {}
        return (data.get("regime") or {}).get("regime") or "Unknown"
    except Exception:
        return "Unknown"


# Equity high-water-mark store (gitignored) — tracks the peak of the owner's book
# so the equity-curve brake can react to a personal drawdown even when the market
# regime hasn't turned.
_HWM_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".portfolio_hwm.json")


def _update_hwm(equity: float) -> float:
    """Persist and return the running high-water mark of portfolio equity."""
    import json
    hwm = 0.0
    try:
        with open(_HWM_FILE) as fh:
            hwm = float((json.load(fh) or {}).get("hwm") or 0.0)
    except (FileNotFoundError, ValueError, OSError):
        pass
    if equity and equity > hwm:
        hwm = float(equity)
        try:
            with open(_HWM_FILE, "w") as fh:
                json.dump({"hwm": hwm, "updated": time.time()}, fh)
        except OSError:
            pass
    return hwm


@app.route("/api/risk/summary")
def api_risk_summary():
    """Regime → recommended exposure (the capital-protection switch), the
    equity-curve brake, and portfolio heat if the caller owns positions."""
    import risk_engine
    regime = _current_regime()
    exposure = risk_engine.regime_exposure(regime)
    out = {"exposure": exposure}
    # Portfolio heat + equity brake are owner-only.
    if auth.can_see_positions() and PORTFOLIO_AVAILABLE:
        try:
            import portfolio as _pf
            summ = _pf.portfolio_summary()
            positions = summ.get("positions") or summ.get("holdings") or []
            open_risks, capital = [], float(summ.get("capital") or 1_000_000)
            equity = float(summ.get("total_current_value")
                           or summ.get("current_value") or 0.0)
            # Equity = deployed value + cash. If only holdings value is known, use
            # capital as the base so a fresh book with cash isn't seen as a drawdown.
            equity = max(equity, capital) if equity else capital
            for p in positions:
                entry = p.get("entry_price") or p.get("entry")
                stop  = p.get("stop") or p.get("sl")
                qty   = p.get("qty") or p.get("quantity") or 0
                if entry and stop and qty and entry > stop:
                    open_risks.append((entry - stop) * qty)
            out["heat"] = risk_engine.portfolio_heat(open_risks, capital)
            hwm = _update_hwm(equity)
            out["equity_brake"] = risk_engine.equity_brake(equity, hwm)
            out["equity_brake"]["equity"] = round(equity, 2)
            out["equity_brake"]["high_water_mark"] = round(hwm, 2)
            # Effective exposure = regime ladder × equity brake.
            out["effective_exposure_pct"] = round(
                exposure["exposure_pct"] * out["equity_brake"]["multiplier"], 1)
        except Exception:
            out["heat"] = None
    return jsonify(out)


@app.route("/api/risk/position-size", methods=["POST"])
def api_position_size():
    """Risk-budgeted position sizing, scaled by the current regime exposure."""
    import risk_engine
    d = request.get_json(silent=True) or {}
    try:
        capital = float(d.get("capital") or 0)
        entry   = float(d.get("entry") or 0)
        stop    = float(d.get("stop") or 0)
        risk_pct = float(d.get("risk_pct") or 0.75)
        max_pos  = float(d.get("max_position_pct") or 10.0)
    except (TypeError, ValueError):
        return jsonify({"valid": False, "message": "Numbers only, please."})
    # Apply the live regime exposure unless the caller overrides it.
    if d.get("regime_exposure_pct") is not None:
        reg_pct = float(d["regime_exposure_pct"])
    else:
        reg_pct = risk_engine.regime_exposure(_current_regime())["exposure_pct"]
    # Fold in the equity-curve brake (owner-only) — a personal drawdown cuts size
    # ON TOP of the regime. Demo sessions size on the regime alone.
    brake_mult = 1.0
    brake = None
    if auth.can_see_positions() and PORTFOLIO_AVAILABLE:
        try:
            import portfolio as _pf, json as _json
            summ = _pf.portfolio_summary()
            capital_book = float(summ.get("capital") or capital)
            equity = float(summ.get("total_current_value") or summ.get("current_value") or 0.0)
            equity = max(equity, capital_book) if equity else capital_book
            hwm = _update_hwm(equity)
            brake = risk_engine.equity_brake(equity, hwm)
            brake_mult = brake["multiplier"]
        except Exception:
            pass
    eff_pct = reg_pct * brake_mult
    res = risk_engine.position_size(capital, entry, stop, risk_pct, eff_pct, max_pos)
    res["regime"] = _current_regime()
    res["regime_exposure_pct"] = reg_pct
    res["equity_brake"] = brake
    res["effective_exposure_pct"] = round(eff_pct, 1)
    return jsonify(res)


# ── System-level walk-forward (the yardstick) — heavy, background + cached ────
_sysbt_state = {"running": False, "pct": 0, "msg": "", "result": None, "tag": None}
_sysbt_lock = threading.Lock()


def _run_sysbt():
    import system_backtest, result_cache as _rc
    def prog(done, total, msg):
        _sysbt_state.update({"pct": int(done / max(total, 1) * 100), "msg": msg})
    try:
        # VALIDATED CONFIG (2026-08-02). The Risk tab used to display days=1600,
        # top_k=10, strategy="momentum" — a configuration that was never validated
        # and does NOT beat the index. The numbers below are the ones that survived
        # a random-selection control, an IS/OOS split and a full-daily drawdown
        # measurement: 19.26% CAGR / -12.02% maxDD / Sharpe 1.00 vs NIFTY
        # 14.40% / -16.11% / 0.66 — beating the index on BOTH return and drawdown,
        # in both halves. Do not change these without re-running that validation.
        r = system_backtest.run_system_backtest(
            days=2800, rebal=21, top_k=30,
            strategy="defensive", mom_weight=4.0, vol_filter=0.70,
            port_vol_target=0.10,
            progress=prog)
        tag = _rc._tag() if hasattr(_rc, "_tag") else None
        _sysbt_state.update({"result": r, "tag": tag})
        try:
            _rc.put("system_backtest", r)
        except Exception:
            pass
    except Exception as e:
        _sysbt_state.update({"result": {"error": str(e)}})
    finally:
        _sysbt_state.update({"running": False, "pct": 100})


@app.route("/api/risk/system-backtest", methods=["POST"])
def api_system_backtest_run():
    with _sysbt_lock:
        if _sysbt_state["running"]:
            return jsonify({"status": "already_running"}), 409
        _sysbt_state.update({"running": True, "pct": 0, "msg": "starting…", "result": None})
    threading.Thread(target=_run_sysbt, daemon=True).start()
    return jsonify({"status": "started"})


@app.route("/api/risk/system-backtest/status")
def api_system_backtest_status():
    import result_cache as _rc
    # Serve a cached result if present and not currently recomputing.
    if not _sysbt_state["running"] and _sysbt_state["result"] is None:
        cached = None
        try:
            cached = _rc.get("system_backtest")
        except Exception:
            cached = None
        if cached:
            return jsonify({"running": False, "pct": 100, "result": cached})
    return jsonify({"running": _sysbt_state["running"], "pct": _sysbt_state["pct"],
                    "msg": _sysbt_state["msg"], "result": _sysbt_state["result"]})


@app.route("/api/risk/drift")
def api_risk_drift():
    """Live-vs-backtest drift monitor (owner-only — reads the journal)."""
    if not auth.can_see_positions():
        return jsonify({"ready": False, "note": "Owner-only."})
    try:
        import drift_monitor
        return jsonify(drift_monitor.compute_drift())
    except Exception as e:
        return jsonify({"ready": False, "note": f"drift error: {e}"})


@app.route("/api/defensive")
def api_defensive():
    """Defensive-Momentum Leaders — the validated lower-drawdown scan. Cached by
    bhavcopy tag; first call computes (a few seconds), then served instantly."""
    import result_cache as _rc
    cached = None
    try:
        cached = _rc.get("defensive")
    except Exception:
        cached = None
    if cached:
        return jsonify(cached)
    try:
        import industry_groups as _ig, defensive_scan as _ds
        _stocks = _ig._get_stocks()
        res = _ds.run_defensive_scan(_stocks, top_n=50)
        # Surface the VALIDATED risk control. Volatility targeting is what turned a
        # book that lost to NIFTY on drawdown (-16.88% vs -16.11%) into one that
        # beats it on both metrics (-12.02%). Until now it lived only in the
        # backtest, so the live tab gave picks with no sizing and the rule that
        # actually produced the edge was never applied.
        try:
            res["exposure"] = _ds.suggested_exposure(_stocks, res.get("stocks") or [])
        except Exception:
            res["exposure"] = None
        try:
            _rc.put("defensive", res)
        except Exception:
            pass
        return jsonify(res)
    except Exception as e:
        return jsonify({"error": str(e), "stocks": []}), 500


@app.route("/api/allweather")
def api_allweather():
    """All-Weather engine — current market regime + the ACTIVE engine's live picks.
    BULL → momentum offense · SIDEWAYS → Defensive Leaders · BEAR → cash (with the
    lowest-risk defensive names shown as a secondary 'if you must stay invested').
    Cached by bhavcopy tag; first call computes, then served instantly."""
    import result_cache as _rc
    cached = None
    try:
        cached = _rc.get("allweather")
    except Exception:
        cached = None
    if cached:
        return jsonify(cached)
    try:
        import regime_engine as _rg, industry_groups as _ig
        import system_backtest as _sb, defensive_scan as _ds
        reg = _rg.live_regime()
        # DISPLAY state comes from the canonical accessor so this tab cannot
        # contradict the header/portfolio; reg is still used for its own detail.
        state = canonical_regime_state() or reg.get("state")
        stocks = _ig._get_stocks()
        picks, secondary = [], None
        # The validated engine runs the Defensive-Momentum Leaders whenever risk-on
        # (BULL or SIDEWAYS) and holds CASH in a confirmed downtrend (BEAR). Raw
        # breakout-momentum offense was tested and rejected (it lost badly on NSE).
        if state in ("BULL", "SIDEWAYS"):
            picks = _ds.run_defensive_scan(stocks, top_n=20).get("stocks", [])
        else:                       # BEAR / UNKNOWN → raise cash
            secondary = _ds.run_defensive_scan(stocks, top_n=12).get("stocks", [])
        res = {"regime": reg, "state": state, "engine": reg.get("engine"),
               "picks": picks, "secondary": secondary, "as_of": reg.get("as_of")}
        try:
            _rc.put("allweather", res)
        except Exception:
            pass
        return jsonify(res)
    except Exception as e:
        return jsonify({"error": str(e), "picks": []}), 500


def _annotate_breakdown_payload(payload):
    """Attach breakdown scores to a payload that does NOT flow through
    result_cache.put(). Consensus, Investment Grade and Alpha Engine return
    straight from their own state/caches, so the universal hook in
    result_cache.put never sees them — measured, those three were the only tabs
    left at 0% coverage. Best-effort: never raises, never drops rows."""
    try:
        import result_cache as _rc
        return _rc._annotate_breakdown(payload)
    except Exception:
        return payload


@app.route("/api/corporate-alerts")
def api_corporate_alerts():
    """Upcoming + recent price-rebasing corporate actions.

    Uses the authoritative NSE feed already cached on disk for the split adjuster —
    zero new network calls. `?symbols=A,B` restricts to a watchlist; with no
    argument it covers the whole feed.
    """
    try:
        import corporate_alerts as _cal
        syms = request.args.get("symbols")
        symbols = [x.strip().upper() for x in syms.split(",")] if syms else None
        horizon = int(request.args.get("days", 45))
        return jsonify(_cal.summary(symbols, horizon_days=horizon))
    except Exception as e:
        return jsonify({"error": str(e), "upcoming": [], "recent": []}), 500


@app.route("/api/breakdown/<symbol>")
def api_breakdown(symbol):
    """The 7-signal breakdown score for one symbol, with the reasons that fired."""
    try:
        import breakdown_detector as _bd, industry_groups as _ig, benchmark as _bm
        df = _ig._get_stocks().get((symbol or "").strip().upper())
        if df is None:
            return jsonify({"error": f"{symbol} not in universe"}), 404
        d = _bd.evaluate(df, _bm.get_benchmark(days=900))
        d["symbol"] = symbol.strip().upper()
        d["calibration"] = ("Measured over 76,773 observations (6y): score 0-1 -> "
                            "+2.63% forward 21d return, score 4+ -> +0.83%. It does NOT "
                            "predict drawdown depth (-9.04% vs -9.82%, essentially flat). "
                            "Read it as trend deterioration, not a crash forecast.")
        return jsonify(d)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/promoter")
def api_promoter():
    """Promoter / insider activity — SAST Reg29 + SEBI PIT disclosures.
    Public NSE filings; each is published within ~2 working days of the trade, so
    this surfaces promoter accumulation long before quarterly shareholding does."""
    try:
        import promoter_flow as _pf
        # SELF-DIAGNOSIS (2026-08-02). This tab silently returned an empty list for
        # days: its cache lived in /tmp/nse_promoter, macOS purged /tmp, and nothing
        # noticed — api_promoter just read an empty directory forever. An empty tab
        # and a broken tab looked identical. The cache now lives in ~/.ascent_cache,
        # and if it is ever empty again the response SAYS SO instead of pretending
        # there were no disclosures.
        import os as _os
        _n_chunks = 0
        try:
            _n_chunks = len([f for f in _os.listdir(_pf.CACHE_DIR) if f.endswith(".json")])
        except Exception:
            _n_chunks = 0
        if _n_chunks == 0:
            return jsonify({
                "transactions": [], "counts": {}, "coverage": {},
                "error": ("No SAST/PIT data cached. This is a MISSING-DATA problem, not "
                          "an absence of disclosures. Run promoter_flow.backfill() to "
                          "repopulate."),
                "cache_dir": _pf.CACHE_DIR, "cached_chunks": 0,
            }), 200
        side = (request.args.get("side") or "all").upper()
        promoter_only = (request.args.get("promoter") or "1") == "1"
        limit = min(int(request.args.get("limit") or 300), 1000)
        rows = _pf.recent_transactions(limit=limit * 3, promoter_only=promoter_only)
        # de-dup: the same trade can be filed under both Reg29(1) and Reg29(2),
        # or reported through SAST and PIT — collapse on the identifying tuple.
        seen, ded = set(), []
        for r in rows:
            k = (r["symbol"], r["date"], r["side"], r["pct"], r["pct_after"])
            if k in seen:
                continue
            seen.add(k)
            ded.append(r)
        if side in ("BUY", "SELL"):
            ded = [r for r in ded if r["side"] == side]
        cov = _pf.coverage()
        buys = sum(1 for r in ded if r["side"] == "BUY")
        # Promoter rows carry a `symbol`, so they get the same breakdown score as
        # every other tab. This endpoint returns straight from the module and never
        # touches result_cache, so the universal hook does not see it.
        _annotate_breakdown_payload({"stocks": ded[:limit]})
        return jsonify({
            "coverage": cov,
            "counts": {"shown": len(ded[:limit]), "buys": buys,
                       "sells": len(ded) - buys,
                       "significant": sum(1 for r in ded
                                          if (r["pct"] or 0) >= 2.0)},
            "transactions": ded[:limit],
        })
    except Exception as e:
        return jsonify({"error": str(e), "transactions": []}), 500


@app.route("/api/fundamentals/history")
def api_fundamentals_history():
    """Point-in-time fundamentals-capture coverage — how many dated snapshots we've
    accumulated toward a bias-free quality validation."""
    try:
        import fundamentals_history as _fh
        cov = _fh.coverage()
        cov["dates"] = cov.get("dates", [])[-12:]   # last 12 for brevity
        return jsonify(cov)
    except Exception as e:
        return jsonify({"error": str(e)})


@app.route("/api/risk/scorecard")
def api_risk_scorecard():
    """Honest, survivorship-free validation scorecard: per-bucket forward alpha
    vs NIFTYBEES + the factor registry (what's proven / what was rejected)."""
    import validation_registry as vr
    return jsonify({
        "as_of":   vr.VALIDATION_ASOF,
        "method":  vr.VALIDATION_METHOD,
        "buckets": vr.BUCKET_STATS,
        "factors": vr.FACTOR_REGISTRY,
    })


@app.route("/api/scan", methods=["POST"])
def start_scan():
    with scan_lock:
        if scan_state["running"]:
            return jsonify({"status": "already_running"}), 409
        scan_state.update({
            "running": True,
            "progress": 0,
            "total": 0,
            "current_ticker": "",
            "result": None,
            "funnel": None,
            "error": None,
            "started_at": time.time(),
        })

    params = request.json or {}
    threading.Thread(target=do_scan, args=(params,), daemon=True).start()
    return jsonify({"status": "started"})


@app.route("/api/status")
def scan_status():
    with scan_lock:
        state = dict(scan_state)
    pct = round(state["progress"] / state["total"] * 100, 1) if state["total"] > 0 else 0
    return jsonify({
        "running":        state["running"],
        "progress":       state["progress"],
        "total":          state["total"],
        "pct":            pct,
        "current_ticker": state["current_ticker"],
        "result":         state["result"],
        "funnel":         state["funnel"],
        "error":          state["error"],
    })


@app.route("/api/sector/refresh", methods=["POST"])
def start_sector():
    with sector_lock:
        if sector_state["running"]:
            return jsonify({"status": "already_running"}), 409
        sector_state.update({
            "running": True,
            "result":  None,
            "error":   None,
            "started_at": time.time(),
        })

    threading.Thread(target=do_sector_analysis, daemon=True).start()
    return jsonify({"status": "started"})


@app.route("/api/sector/status")
def sector_status():
    with sector_lock:
        state = dict(sector_state)
    # POST-RESTART FIX (2026-08-02): `sector_state` is in-memory and only filled by
    # the refresh POST, but `_boot_prewarm` already computes sector analysis into
    # result_cache. After every restart the tab therefore read EMPTY until the user
    # clicked refresh, even though the data was sitting in the cache. Fall back to
    # the cache when memory is cold.
    if not state.get("running") and state.get("result") is None:
        # `run_sector_analysis()` keeps its OWN 30-minute in-process cache and is
        # already warmed by _boot_prewarm, so this returns immediately rather than
        # recomputing. (An earlier version read result_cache["sector"], which never
        # exists — sector_analysis does not write there.)
        try:
            import sector_analysis as _sa
            state["result"] = _sa.run_sector_analysis()
        except Exception as _e:
            state["error"] = str(_e)
    return jsonify({
        "running": state["running"],
        "result":  state["result"],
        "error":   state["error"],
    })


# ── Breakout scanner state ─────────────────────────────────────────────────────
breakout_state = {
    "running":    False,
    "result":     None,
    "error":      None,
    "progress":   0,
    "total":      0,
    "message":    "",
    "started_at": None,
}
breakout_lock = threading.Lock()


def do_breakout_scan():
    def progress_cb(done, total, msg):
        with breakout_lock:
            breakout_state["progress"] = done
            breakout_state["total"]    = total
            breakout_state["message"]  = msg

    try:
        with _scan_semaphore:
            result = run_breakout_scan(progress_callback=progress_cb)
        with breakout_lock:
            breakout_state["result"]  = result
            breakout_state["running"] = False
    except Exception as e:
        with breakout_lock:
            breakout_state["error"]   = str(e)
            breakout_state["running"] = False


@app.route("/api/breakout/scan", methods=["POST"])
def start_breakout_scan():
    with breakout_lock:
        if breakout_state["running"]:
            return jsonify({"status": "already_running"}), 409
        breakout_state.update({
            "running": True, "result": None, "error": None,
            "progress": 0, "total": 0, "message": "",
            "started_at": time.time(),
        })
    threading.Thread(target=do_breakout_scan, daemon=True).start()
    return jsonify({"status": "started"})


@app.route("/api/breakout/status")
def breakout_status():
    with breakout_lock:
        s = dict(breakout_state)
    pct = round(s["progress"] / s["total"] * 100, 1) if s["total"] > 0 else 0
    return jsonify({
        "running":  s["running"],
        "pct":      pct,
        "message":  s["message"],
        "result":   s["result"],
        "error":    s["error"],
    })


# ── Institutional scanner state ────────────────────────────────────────────────
inst_state = {
    "running":  False,
    "result":   None,
    "error":    None,
    "progress": 0,
    "total":    0,
    "message":  "",
    "started_at": None,
}
inst_lock = threading.Lock()


def do_institutional_scan():
    def progress_cb(done, total, msg):
        with inst_lock:
            inst_state["progress"] = done
            inst_state["total"]    = total
            inst_state["message"]  = msg

    try:
        with _scan_semaphore:
            result = run_institutional_scan(progress_callback=progress_cb)
        with inst_lock:
            inst_state["result"]  = result
            inst_state["running"] = False
    except Exception as e:
        with inst_lock:
            inst_state["error"]   = str(e)
            inst_state["running"] = False


@app.route("/api/institutional/scan", methods=["POST"])
def start_institutional_scan():
    with inst_lock:
        if inst_state["running"]:
            return jsonify({"status": "already_running"}), 409
        inst_state.update({
            "running": True, "result": None, "error": None,
            "progress": 0, "total": 0, "message": "",
            "started_at": time.time(),
        })
    threading.Thread(target=do_institutional_scan, daemon=True).start()
    return jsonify({"status": "started"})


@app.route("/api/institutional/status")
def institutional_status():
    with inst_lock:
        s = dict(inst_state)
    pct = round(s["progress"] / s["total"] * 100, 1) if s["total"] > 0 else 0
    return jsonify({
        "running": s["running"],
        "pct":     pct,
        "message": s["message"],
        "result":  s["result"],
        "error":   s["error"],
    })


# ── Advanced Scanner state ─────────────────────────────────────────────────────
adv_state = {
    "running": False, "result": None, "error": None,
    "progress": 0, "total": 0, "message": "", "started_at": None,
}
adv_lock = threading.Lock()


def do_advanced_scan():
    def progress_cb(done, total, msg):
        with adv_lock:
            adv_state["progress"] = done
            adv_state["total"]    = total
            adv_state["message"]  = msg
    try:
        with _scan_semaphore:
            result = run_advanced_scan(progress_callback=progress_cb)
        with adv_lock:
            adv_state["result"]  = result
            adv_state["running"] = False
    except Exception as e:
        with adv_lock:
            adv_state["error"]   = str(e)
            adv_state["running"] = False


@app.route("/api/advanced/scan", methods=["POST"])
def start_advanced_scan():
    with adv_lock:
        if adv_state["running"]:
            return jsonify({"status": "already_running"}), 409
        adv_state.update({
            "running": True, "result": None, "error": None,
            "progress": 0, "total": 0, "message": "", "started_at": time.time(),
        })
    threading.Thread(target=do_advanced_scan, daemon=True).start()
    return jsonify({"status": "started"})


@app.route("/api/advanced/status")
def advanced_status():
    with adv_lock:
        s = dict(adv_state)
    pct = round(s["progress"] / s["total"] * 100, 1) if s["total"] > 0 else 0
    return jsonify({"running": s["running"], "pct": pct, "message": s["message"],
                    "result": s["result"], "error": s["error"]})


# ── Market Breadth state ───────────────────────────────────────────────────────
breadth_state = {
    "running": False, "result": None, "error": None,
    "progress": 0, "total": 0, "message": "", "started_at": None,
}
breadth_lock = threading.Lock()


def do_breadth_scan():
    def progress_cb(done, total, msg):
        with breadth_lock:
            breadth_state["progress"] = done
            breadth_state["total"]    = total
            breadth_state["message"]  = msg
    try:
        with _scan_semaphore:
            result = run_market_breadth(progress_callback=progress_cb)
        with breadth_lock:
            breadth_state["result"]  = result
            breadth_state["running"] = False
    except Exception as e:
        with breadth_lock:
            breadth_state["error"]   = str(e)
            breadth_state["running"] = False


@app.route("/api/breadth/refresh", methods=["POST"])
def start_breadth():
    with breadth_lock:
        if breadth_state["running"]:
            return jsonify({"status": "already_running"}), 409
        breadth_state.update({
            "running": True, "result": None, "error": None,
            "progress": 0, "total": 0, "message": "", "started_at": time.time(),
        })
    threading.Thread(target=do_breadth_scan, daemon=True).start()
    return jsonify({"status": "started"})


@app.route("/api/breadth/status")
def breadth_status():
    with breadth_lock:
        s = dict(breadth_state)
    # If no in-memory result yet (e.g. just after restart), fall back to the
    # persisted breadth cache so the tab and header always show the same value.
    if s["result"] is None and not s["running"]:
        try:
            from market_breadth import _cache as mb_cache
            if mb_cache.get("data"):
                s["result"] = mb_cache["data"]
        except Exception:
            pass
    return jsonify({"running": s["running"], "message": s["message"],
                    "result": s["result"], "error": s["error"]})


# ── Industry Groups state ─────────────────────────────────────────────────────
industry_state = {
    "running": False, "result": None, "error": None,
    "progress": 0, "total": 0, "message": "", "started_at": None,
}
industry_lock = threading.Lock()


def do_industry_analysis():
    def progress_cb(done, total, msg):
        with industry_lock:
            industry_state["progress"] = done
            industry_state["total"]    = total
            industry_state["message"]  = msg
    try:
        with _scan_semaphore:
            result = run_industry_analysis(progress_callback=progress_cb)
        with industry_lock:
            industry_state["result"]  = result
            industry_state["running"] = False
    except Exception as e:
        with industry_lock:
            industry_state["error"]   = str(e)
            industry_state["running"] = False


@app.route("/api/industry/refresh", methods=["POST"])
def start_industry():
    with industry_lock:
        if industry_state["running"]:
            return jsonify({"status": "already_running"}), 409
        industry_state.update({
            "running": True, "result": None, "error": None,
            "progress": 0, "total": 0, "message": "", "started_at": time.time(),
        })
    threading.Thread(target=do_industry_analysis, daemon=True).start()
    return jsonify({"status": "started"})


@app.route("/api/industry/status")
def industry_status():
    with industry_lock:
        s = dict(industry_state)
    pct = round(s["progress"] / s["total"] * 100, 1) if s["total"] > 0 else 0
    return jsonify({"running": s["running"], "pct": pct, "message": s["message"],
                    "result": s["result"], "error": s["error"]})


# ── Momentum scanner state ────────────────────────────────────────────────────
momentum_state = {
    "running": False, "result": None, "error": None,
    "progress": 0, "total": 0, "message": "", "started_at": None,
}
momentum_lock = threading.Lock()


def do_momentum_scan():
    def progress_cb(done, total, msg):
        with momentum_lock:
            momentum_state["progress"] = done
            momentum_state["total"]    = total
            momentum_state["message"]  = msg
    try:
        with _scan_semaphore:
            result = run_momentum_scan(progress_callback=progress_cb)
        with momentum_lock:
            momentum_state["result"]  = result
            momentum_state["running"] = False
    except Exception as e:
        with momentum_lock:
            momentum_state["error"]   = str(e)
            momentum_state["running"] = False


@app.route("/api/momentum/scan", methods=["POST"])
def start_momentum_scan():
    with momentum_lock:
        if momentum_state["running"]:
            return jsonify({"status": "already_running"}), 409
        momentum_state.update({
            "running": True, "result": None, "error": None,
            "progress": 0, "total": 0, "message": "",
            "started_at": time.time(),
        })
    threading.Thread(target=do_momentum_scan, daemon=True).start()
    return jsonify({"status": "started"})


@app.route("/api/momentum/status")
def momentum_status():
    with momentum_lock:
        s = dict(momentum_state)
    pct = round(s["progress"] / s["total"] * 100, 1) if s["total"] > 0 else 0
    return jsonify({
        "running": s["running"],
        "pct":     pct,
        "message": s["message"],
        "result":  s["result"],
        "error":   s["error"],
    })


# ── Emerging Leaders scanner state ────────────────────────────────────────────
emerging_state = {
    "running": False, "result": None, "error": None,
    "progress": 0, "total": 0, "message": "", "started_at": None,
}
emerging_lock = threading.Lock()


def do_emerging_scan():
    def progress_cb(done, total, msg):
        with emerging_lock:
            emerging_state["progress"] = done
            emerging_state["total"]    = total
            emerging_state["message"]  = msg
    try:
        with _scan_semaphore:
            result = run_emerging_leaders_scan(progress_callback=progress_cb)
        with emerging_lock:
            emerging_state["result"]  = result
            emerging_state["running"] = False
    except Exception as e:
        with emerging_lock:
            emerging_state["error"]   = str(e)
            emerging_state["running"] = False


@app.route("/api/emerging/scan", methods=["POST"])
def start_emerging_scan():
    with emerging_lock:
        if emerging_state["running"]:
            return jsonify({"status": "already_running"}), 409
        emerging_state.update({
            "running": True, "result": None, "error": None,
            "progress": 0, "total": 0, "message": "",
            "started_at": time.time(),
        })
    threading.Thread(target=do_emerging_scan, daemon=True).start()
    return jsonify({"status": "started"})


@app.route("/api/emerging/status")
def emerging_status():
    with emerging_lock:
        s = dict(emerging_state)
    # Fall back to module cache if no in-memory result yet (e.g. pre-warmed)
    if s["result"] is None and not s["running"]:
        try:
            if emerging_leaders_mod._cache.get("data"):
                s["result"] = emerging_leaders_mod._cache["data"]
        except Exception:
            pass
    pct = round(s["progress"] / s["total"] * 100, 1) if s["total"] > 0 else 0
    return jsonify({
        "running": s["running"],
        "pct":     pct,
        "message": s["message"],
        "result":  s["result"],
        "error":   s["error"],
    })


# ── Early Mover scanner state ─────────────────────────────────────────────────
early_mover_state = {
    "running": False, "result": None, "error": None,
    "progress": 0, "total": 0, "message": "", "started_at": None,
}
early_mover_lock = threading.Lock()


def do_early_mover_scan():
    def progress_cb(done, total, msg):
        with early_mover_lock:
            early_mover_state["progress"] = done
            early_mover_state["total"]    = total
            early_mover_state["message"]  = msg
    try:
        with _scan_semaphore:
            result = run_early_mover_scan(progress_callback=progress_cb)
        with early_mover_lock:
            early_mover_state["result"]  = result
            early_mover_state["running"] = False
    except Exception as e:
        with early_mover_lock:
            early_mover_state["error"]   = str(e)
            early_mover_state["running"] = False


@app.route("/api/early-mover/scan", methods=["POST"])
def start_early_mover_scan():
    with early_mover_lock:
        if early_mover_state["running"]:
            return jsonify({"status": "already_running"}), 409
        early_mover_state.update({
            "running": True, "result": None, "error": None,
            "progress": 0, "total": 0, "message": "",
            "started_at": time.time(),
        })
    threading.Thread(target=do_early_mover_scan, daemon=True).start()
    return jsonify({"status": "started"})


@app.route("/api/early-mover/status")
def early_mover_status():
    with early_mover_lock:
        s = dict(early_mover_state)
    pct = round(s["progress"] / s["total"] * 100, 1) if s["total"] > 0 else 0
    return jsonify({
        "running": s["running"],
        "pct":     pct,
        "message": s["message"],
        "result":  s["result"],
        "error":   s["error"],
    })


# ── Volume Scanner state ──────────────────────────────────────────────────────
vol_state = {
    "running": False, "result": None, "error": None,
    "progress": 0, "total": 0, "message": "", "started_at": None,
}
vol_lock = threading.Lock()


def do_volume_scan():
    def progress_cb(done, total, msg):
        with vol_lock:
            vol_state["progress"] = done
            vol_state["total"]    = total
            vol_state["message"]  = msg
    try:
        with _scan_semaphore:
            result = run_volume_scan(progress_callback=progress_cb)
        with vol_lock:
            vol_state["result"]  = result
            vol_state["running"] = False
    except Exception as e:
        with vol_lock:
            vol_state["error"]   = str(e)
            vol_state["running"] = False


@app.route("/api/volume/scan", methods=["POST"])
def start_volume_scan():
    with vol_lock:
        if vol_state["running"]:
            return jsonify({"status": "already_running"}), 409
        vol_state.update({
            "running": True, "result": None, "error": None,
            "progress": 0, "total": 0, "message": "",
            "started_at": time.time(),
        })
    threading.Thread(target=do_volume_scan, daemon=True).start()
    return jsonify({"status": "started"})


@app.route("/api/volume/status")
def volume_status():
    with vol_lock:
        s = dict(vol_state)
    pct = round(s["progress"] / s["total"] * 100, 1) if s["total"] > 0 else 0
    return jsonify({
        "running": s["running"],
        "pct":     pct,
        "message": s["message"],
        "result":  s["result"],
        "error":   s["error"],
    })


# ── Multi-Year Breakout Scanner state ────────────────────────────────────────
from multiyear_breakout import run_multiyear_scan, run_near_breakout_scan

mbo_state = {
    "running": False, "result": None, "error": None,
    "progress": 0, "total": 0, "message": "", "started_at": None,
}
mbo_lock = threading.Lock()


def do_mbo_scan(min_base_years: int):
    def progress_cb(done, total, msg):
        with mbo_lock:
            mbo_state["progress"] = done
            mbo_state["total"]    = total
            mbo_state["message"]  = msg
    try:
        with _scan_semaphore:
            result = run_multiyear_scan(min_base_years=min_base_years,
                                    progress_callback=progress_cb)
        with mbo_lock:
            mbo_state["result"]  = result
            mbo_state["running"] = False
    except Exception as e:
        with mbo_lock:
            mbo_state["error"]   = str(e)
            mbo_state["running"] = False


@app.route("/api/mbo/scan", methods=["POST"])
def start_mbo_scan():
    min_base_years = int(request.json.get("min_base_years", 1)) if request.json else 1
    with mbo_lock:
        if mbo_state["running"]:
            return jsonify({"status": "already_running"}), 409
        mbo_state.update({
            "running": True, "result": None, "error": None,
            "progress": 0, "total": 0, "message": "",
            "started_at": time.time(),
        })
    threading.Thread(target=do_mbo_scan, args=(min_base_years,), daemon=True).start()
    return jsonify({"status": "started"})


@app.route("/api/mbo/status")
def mbo_status():
    with mbo_lock:
        s = dict(mbo_state)
    pct = round(s["progress"] / s["total"] * 100, 1) if s["total"] > 0 else 0
    return jsonify({
        "running": s["running"],
        "pct":     pct,
        "message": s["message"],
        "result":  s["result"],
        "error":   s["error"],
    })


# ── Near-Breakout Scanner state ──────────────────────────────────────────────
near_state = {
    "running": False, "result": None, "error": None,
    "progress": 0, "total": 0, "message": "", "started_at": None,
}
near_lock = threading.Lock()


def do_near_scan():
    def progress_cb(done, total, msg):
        with near_lock:
            near_state["progress"] = done
            near_state["total"]    = total
            near_state["message"]  = msg
    try:
        result = run_near_breakout_scan(progress_callback=progress_cb)
        with near_lock:
            near_state["result"]  = result
            near_state["running"] = False
    except Exception as e:
        with near_lock:
            near_state["error"]   = str(e)
            near_state["running"] = False


@app.route("/api/mbo/near-scan", methods=["POST"])
def start_near_scan():
    with near_lock:
        if near_state["running"]:
            return jsonify({"status": "already_running"}), 409
        near_state.update({
            "running": True, "result": None, "error": None,
            "progress": 0, "total": 0, "message": "",
            "started_at": time.time(),
        })
    threading.Thread(target=do_near_scan, daemon=True).start()
    return jsonify({"status": "started"})


@app.route("/api/mbo/near-status")
def near_scan_status():
    with near_lock:
        s = dict(near_state)
    pct = round(s["progress"] / s["total"] * 100, 1) if s["total"] > 0 else 0
    return jsonify({
        "running": s["running"],
        "pct":     pct,
        "message": s["message"],
        "result":  s["result"],
        "error":   s["error"],
    })


@app.route("/api/unusual-volume")
def api_unusual_volume():
    """Delivery-backed unusual volume — the smart-money section of the Vol Spike tab.

    Uses data_fetcher.fetch_ohlcv (not shared_universe) because only that loader
    keeps the DelivPer column, and delivery is the whole point: relative volume
    alone cannot separate accumulation from intraday churn. ~26s cold, 30min cache.
    """
    try:
        import unusual_volume as _uv
        return jsonify(_uv.run_unusual_volume_scan())
    except Exception as e:
        return jsonify({"results": [], "scanned": 0, "found": 0, "error": str(e)})


@app.route("/api/weekly-breakout")
def api_weekly_breakout():
    """Fresh weekly-bar breakouts — the short-horizon section of the Multi-Yr tab.

    A plain GET rather than the scan/status pair the monthly scans use: this runs
    off the daily bhavcopy cache that is already in memory (~6.5s cold, cached for
    6h afterwards), so it needs no background thread or progress polling.
    """
    try:
        import weekly_breakout as _wb
        return jsonify(_wb.run_weekly_breakout_scan())
    except Exception as e:
        return jsonify({"results": [], "scanned": 0, "found": 0, "error": str(e)})


@app.route("/api/today")
def api_today():
    """#8 — ONE view answering "what changed and what needs a decision today".

    The app has 27 tabs and 89 endpoints; nobody works across that. This collapses
    the signal surface to: the market call, what the ONE validated screen fired,
    what broke out and is still working, and what needs attention. Every block
    carries its capacity and its honest expectancy, because a list without those
    implies a scalability and a hit-rate the measurements do not support.
    """
    out = {"computed_at": time.time()}
    try:
        import signal_context as _sc
        import weekly_breakout as _wb
        import unusual_volume as _uv

        out["regime"] = {"state": canonical_regime_state(),
                         "label": _current_regime()}

        acc = (_uv.run_accumulation_scan() or {}).get("results") or []
        wbo = (_wb.run_weekly_breakout_scan() or {}).get("results") or []
        pb  = _wb.run_post_breakout_scan() or {}
        pbr = pb.get("results") or []

        working = [r for r in pbr if r.get("state") in ("CLIMBING", "EXTENDED")]
        attention = [r for r in pbr if r.get("state") in ("STALLED", "AT PIVOT")]

        out["blocks"] = [
            {"key": "accumulation", "title": "Institutional accumulation",
             "subtitle": "The only screen that survived 7-year walk-forward validation",
             "rows": acc[:12], "n": len(acc),
             "capacity": _sc.capacity_for(acc),
             "evidence": ("+1.15pp vs peers over 349,124 observations (2019-2026); "
                          "20 of 28 quarterly folds positive, t=3.37. Win rate 54.7% "
                          "vs 51.4% — a real but modest edge.")},
            {"key": "fresh_breakouts", "title": "Fresh breakouts",
             "subtitle": "Triggered in the last 2 weeks, not yet extended",
             "rows": wbo[:12], "n": len(wbo),
             "capacity": _sc.capacity_for(wbo),
             "evidence": ("Fat-tailed: the MEDIAN breakout LOSES to the market "
                          "(-3.41% excess, 43.5% win). The edge is entirely a "
                          "+29.6% top decile, so size for survival, not accuracy.")},
            {"key": "still_working", "title": "Still working",
             "subtitle": "Broke out earlier and holding above the pivot",
             "rows": working[:12], "n": len(working),
             "capacity": _sc.capacity_for(working),
             "evidence": (f"{pb.get('hold_rate', 0)}% of {pb.get('tracked', 0)} tracked "
                          "breakouts are still above their pivot. Bookkeeping, not a buy list.")},
            {"key": "attention", "title": "Needs a decision",
             "subtitle": "Stalled, or undecided at the pivot",
             "rows": attention[:12], "n": len(attention),
             "capacity": _sc.capacity_for(attention),
             "evidence": "Holding above the pivot but well off the high, or oscillating at the trigger."},
        ]

        ov = _sc.annotate_overlap({"accumulation": acc, "weekly": wbo, "post": pbr})
        out["overlap"] = {"n_multi": ov["n_multi"], "warning": ov["warning"],
                          "examples": {k: v for k, v in list(ov["multi_screen"].items())[:8]}}
        out["benchmark"] = {
            "note": ("Buy & hold, equal-weight liquid universe: 18.11% CAGR / -29.5% maxDD "
                     "over the tested window. No active configuration tested has beaten it. "
                     "Judge anything below against that.")}
        return jsonify(out)
    except Exception as e:
        return jsonify({"error": str(e), "blocks": []})


@app.route("/api/system")
def api_system():
    """The System tab — RS>90 leaders at a pivot that beat the index, with a thesis,
    actionable advice, and a live daily scoreboard. Honest expectancy is carried on
    the payload, not hidden: this is a fat-tail screen (see system_leaders)."""
    try:
        import system_leaders as _sl
        d = _sl.run_system_scan()
        rows = d.get("results") or []
        # daily tracking: log today's picks, then score every pick ever logged
        try:
            import market_breadth as _mb
            as_of = ((_mb._cache.get("data") or {}).get("bhavcopy_date")) or ""
        except Exception:
            as_of = ""
        _sl.record_picks(rows, as_of or "unknown")
        # attach first-seen date + days-in-list from the tracker to each row
        try:
            _track = _sl._load_track()
            import datetime as _dt
            for r in rows:
                rec = _track.get(r["symbol"]) or {}
                r["first_seen"] = rec.get("first_seen")
                try:
                    d0 = _dt.date.fromisoformat(rec.get("first_seen"))
                    r["days_in_list"] = (_dt.date.today() - d0).days
                except Exception:
                    r["days_in_list"] = 0
        except Exception:
            pass
        try:
            import shared_universe as _su
            # Reuse the SAME cached window the scans already built (days=500) instead
            # of a separate days=60 load — the latter rebuilt a whole universe (~1.4s)
            # on every /api/today call for nothing but the latest close. Same prices,
            # zero extra build. (Demo-latency fix 2026-08-17.)
            U = _su.load_base_universe(days=500)
            cur = {s: float(df["Close"].iloc[-1]) for s, df in U.items() if len(df)}
            d["scoreboard"] = _sl.performance(cur)
        except Exception:
            d["scoreboard"] = {"picks": [], "aggregate": {"tracked": 0}}
        segs = {}
        for r in rows:
            segs[r.get("segment", "?")] = segs.get(r.get("segment", "?"), 0) + 1
        d["expectancy"] = {
            "headline": ("REFINED screen (mcap>=Rs1000cr, best price action, RS 90-97), measured "
                         "7yr: mean +8.43% / ~3 months, 57% win, +3.11pp vs the universe — up from "
                         "+0.57pp for the raw RS>90 screen. Removing penny/small caps and keeping "
                         "only tight price action did the work."),
            "caveats": [
                f"Segment split now: {segs.get('CASH',0)} cash, {segs.get('F&O',0)} F&O · "
                f"{d.get('n_conviction',0)} HIGH-CONVICTION (F&O + established uptrend).",
                "DUD-FILTER — measured, not guessed: a multi-factor 'leader score' was tested and "
                "did NOT separate winners from duds (top-15 = the cohort). The ONLY filter that "
                "worked is HIGH CONVICTION = F&O + >60 sessions trending: fwd-6m median +20.8pp / "
                "69% beat vs the cohort +6.4pp / 58%. Use the 'High conviction' filter for that "
                "subset — fewer names, better odds, wider variance (n=134 over 7yr).",
                "This means the strongest subset is F&O, not cash — the opposite of the usual "
                "'small caps run more' belief. The cash multi-baggers are a rarer, lower-odds fat tail.",
                "No filter makes it certain: CUPID (+216%) and CSBBANK (-39%) looked identical at "
                "signal. Spread across the subset, cut losers at the stop, let winners run."]}
        return jsonify(d)
    except Exception as e:
        return jsonify({"results": [], "scanned": 0, "found": 0, "error": str(e)})


@app.route("/api/post-breakout")
def api_post_breakout():
    """What happened AFTER each recent breakout — climbing / extended / stalled / failed.

    Bookkeeping on open ideas, not a buy list. Same cache pattern as the sibling
    weekly-breakout route above.
    """
    try:
        import weekly_breakout as _wb
        return jsonify(_wb.run_post_breakout_scan())
    except Exception as e:
        return jsonify({"results": [], "counts": {}, "tracked": 0, "error": str(e)})


# ── Edge Engine state ─────────────────────────────────────────────────────────
edge_state = {
    "running": False, "result": None, "error": None,
    "progress": 0, "total": 0, "message": "", "started_at": None,
}
edge_lock = threading.Lock()


def do_edge_engine():
    def progress_cb(done, total, msg):
        with edge_lock:
            edge_state["progress"] = done
            edge_state["total"]    = total
            edge_state["message"]  = msg
    try:
        with _scan_semaphore:
            result = run_edge_engine(progress_callback=progress_cb)
        with edge_lock:
            edge_state["result"]  = result
            edge_state["running"] = False
    except Exception as e:
        with edge_lock:
            edge_state["error"]   = str(e)
            edge_state["running"] = False


@app.route("/api/edge/scan", methods=["POST"])
def start_edge_engine():
    with edge_lock:
        if edge_state["running"]:
            return jsonify({"status": "already_running"}), 409
        # Force a fresh ROUTINE run (new fundamentals). Do NOT touch the separate
        # 6h backtest cache — that's heavy and user-triggered, keep it warm.
        edge_engine_mod._cache["data"] = None
        edge_engine_mod._cache["ts"]   = 0
        edge_state.update({
            "running": True, "result": None, "error": None,
            "progress": 0, "total": 0, "message": "",
            "started_at": time.time(),
        })
    threading.Thread(target=do_edge_engine, daemon=True).start()
    return jsonify({"status": "started"})


@app.route("/api/edge/status")
def edge_status():
    with edge_lock:
        s = dict(edge_state)
    pct = round(s["progress"] / s["total"] * 100, 1) if s["total"] > 0 else 0
    return jsonify({
        "running": s["running"], "pct": pct, "message": s["message"],
        "result":  s["result"],  "error": s["error"],
    })


# ── Edge BACKTEST (heavy, on-demand) ──────────────────────────────────────────
# The survivorship-free walk-forward backtest (~3-5 min, ~700MB) is split off the
# routine scan so it never blocks/slows the box. Runs only when the user clicks
# "Run Backtest" on the Edge tab. Result cached 6h in edge_engine._bt_cache.
edge_bt_state = {
    "running": False, "result": None, "error": None,
    "progress": 0, "total": 0, "message": "", "started_at": None,
}
edge_bt_lock = threading.Lock()


def do_edge_backtest():
    def progress_cb(done, total, msg):
        with edge_bt_lock:
            edge_bt_state["progress"] = done
            edge_bt_state["total"]    = total
            edge_bt_state["message"]  = msg
    try:
        with _scan_semaphore:
            result = run_edge_engine(progress_callback=progress_cb,
                                     include_backtests=True)
        with edge_bt_lock:
            edge_bt_state["result"]  = result
            edge_bt_state["running"] = False
    except Exception as e:
        with edge_bt_lock:
            edge_bt_state["error"]   = str(e)
            edge_bt_state["running"] = False


@app.route("/api/edge/backtest", methods=["POST"])
def start_edge_backtest():
    with edge_bt_lock:
        if edge_bt_state["running"]:
            return jsonify({"status": "already_running"}), 409
        edge_bt_state.update({
            "running": True, "result": None, "error": None,
            "progress": 0, "total": 0, "message": "Starting backtest…",
            "started_at": time.time(),
        })
    threading.Thread(target=do_edge_backtest, daemon=True).start()
    return jsonify({"status": "started"})


@app.route("/api/edge/backtest/status")
def edge_backtest_status():
    with edge_bt_lock:
        s = dict(edge_bt_state)
    # Serve cached backtest result if present (survives across sessions for 6h)
    if s["result"] is None and not s["running"]:
        try:
            if edge_engine_mod._bt_cache.get("data") and \
               edge_engine_mod._bt_cache["data"].get("backtests_included"):
                s["result"] = edge_engine_mod._bt_cache["data"]
        except Exception:
            pass
    pct = round(s["progress"] / s["total"] * 100, 1) if s["total"] > 0 else 0
    return jsonify({
        "running": s["running"], "pct": pct, "message": s["message"],
        "result":  s["result"],  "error": s["error"],
    })


# ── Exit signal check (per-symbol, on demand) ────────────────────────────────
@app.route("/api/edge/exit-check")
def edge_exit_check():
    """Check if a symbol has active exit signals. Reads from edge engine cache."""
    symbol = request.args.get("symbol", "").upper()
    if not symbol:
        return jsonify({"error": "symbol required"}), 400
    try:
        # Need to re-load that symbol's data from bhavcopy cache
        stocks = _load_stocks(days=120)
        df = stocks.get(symbol)
        if df is None:
            return jsonify({"error": f"No data for {symbol}"}), 404
        entry_price = request.args.get("entry", type=float)
        result = detect_exit_signals(df, entry_price=entry_price)
        return jsonify({"symbol": symbol, **result})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── Trending Stocks Scanner state ────────────────────────────────────────────
trend_state = {
    "running": False, "result": None, "error": None,
    "progress": 0, "total": 0, "message": "", "started_at": None,
}
trend_lock = threading.Lock()


def do_trend_scan():
    def progress_cb(done, total, msg):
        with trend_lock:
            trend_state["progress"] = done
            trend_state["total"]    = total
            trend_state["message"]  = msg
    try:
        with _scan_semaphore:
            result = run_trending_scan(progress_callback=progress_cb)
        with trend_lock:
            trend_state["result"]  = result
            trend_state["running"] = False
        # Position Guardian sweep — piggyback on every trending scan while the
        # bhavcopy cache is warm. Cheap (only held/watched symbols), never fatal.
        try:
            import guardian
            sw = guardian.run_sweep()
            if sw.get("alerts"):
                print(f"[guardian] {len(sw['alerts'])} active alert(s): "
                      + ", ".join(a['symbol'] for a in sw['alerts']), flush=True)
        except Exception as ge:
            print(f"[guardian] sweep failed: {ge}", flush=True)
    except Exception as e:
        with trend_lock:
            trend_state["error"]   = str(e)
            trend_state["running"] = False


@app.route("/api/trending/scan", methods=["POST"])
def start_trend_scan():
    with trend_lock:
        if trend_state["running"]:
            return jsonify({"status": "already_running"}), 409
        trend_state.update({
            "running": True, "result": None, "error": None,
            "progress": 0, "total": 0, "message": "",
            "started_at": time.time(),
        })
    threading.Thread(target=do_trend_scan, daemon=True).start()
    return jsonify({"status": "started"})


@app.route("/api/trending/status")
def trend_scan_status():
    with trend_lock:
        s = dict(trend_state)
    pct = round(s["progress"] / s["total"] * 100, 1) if s["total"] > 0 else 0
    return jsonify({
        "running": s["running"],
        "pct":     pct,
        "message": s["message"],
        "result":  s["result"],
        "error":   s["error"],
    })


@app.route("/api/trending/scorecard")
def trending_scorecard_summary():
    """Honest forward-return scorecard of past trending picks (v3 evidence loop)."""
    try:
        from trending_scorecard import summary
        return jsonify(summary())
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── Position Guardian + Watchlist ─────────────────────────────────────────────
# The guardian sweep runs after every trending scan (see do_trend_scan) and
# once at startup; these routes are thin — reads hit guardian.db only.

@app.route("/api/guardian/alerts")
def guardian_alerts():
    try:
        import guardian
        return jsonify({"alerts": guardian.get_active_alerts()})
    except Exception as e:
        return jsonify({"alerts": [], "error": str(e)})


@app.route("/api/guardian/dismiss", methods=["POST"])
def guardian_dismiss():
    try:
        import guardian
        sym = (request.get_json(silent=True) or {}).get("symbol", "")
        return jsonify({"dismissed": guardian.dismiss(sym)})
    except Exception as e:
        return jsonify({"dismissed": False, "error": str(e)}), 500


@app.route("/api/guardian/sweep", methods=["POST"])
def guardian_manual_sweep():
    try:
        import guardian
        return jsonify(guardian.run_sweep())
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/watchlist")
def watchlist_get():
    try:
        import watchlist
        return jsonify({"symbols": watchlist.get_symbols()})
    except Exception as e:
        return jsonify({"symbols": [], "error": str(e)})


@app.route("/api/watchlist/toggle", methods=["POST"])
def watchlist_toggle():
    try:
        import watchlist, guardian
        sym = (request.get_json(silent=True) or {}).get("symbol", "")
        res = watchlist.toggle(sym)
        # Removing a symbol should clear its alert immediately; adding one
        # gets picked up on the next sweep (or trigger one in the background).
        threading.Thread(target=guardian.run_sweep, daemon=True).start()
        return jsonify(res)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


def _guardian_startup_sweep():
    """One sweep ~90s after launch (lets the bhavcopy cache warm first)."""
    time.sleep(90)
    try:
        import guardian
        guardian.run_sweep()
    except Exception as e:
        print(f"[guardian] startup sweep failed: {e}", flush=True)


if _BG_JOBS:
    threading.Thread(target=_guardian_startup_sweep, daemon=True).start()


# ── Fundamentals — background scheduler (auto-started at launch) ──────────────
# State is managed entirely inside fundamentals.py (_sched dict).
# No manual fund_state needed here anymore.


@app.route("/api/fundamentals/status")
def fundamentals_status():
    cache = fund_cache_status()
    sched = fund_scheduler_status()

    # Build human-readable ETA string
    eta_min = sched.get("eta_minutes", 0)
    if eta_min >= 60:
        eta_str = f"{int(eta_min // 60)}h {int(eta_min % 60)}m"
    elif eta_min > 0:
        eta_str = f"{int(eta_min)}m"
    else:
        eta_str = "—"

    scraped  = sched.get("scraped_count", 0)
    total    = sched.get("total", 0)
    pending  = sched.get("pending_count", 0)
    failed   = sched.get("failed_count", 0)
    pct      = round(scraped / total * 100, 1) if total > 0 else 0

    return jsonify({
        # Scheduler live state
        "scheduler_running":  sched.get("running", False),
        "current_symbol":     sched.get("current_symbol", ""),
        "last_scraped":       sched.get("last_scraped", ""),
        "last_scraped_at":    sched.get("last_scraped_at", 0),
        "scraped_count":      scraped,
        "failed_count":       failed,
        "pending_count":      pending,
        "total_symbols":      total,
        "pct_complete":       pct,
        "eta":                eta_str,
        "eta_minutes":        eta_min,
        "rate":               "~400 stocks / hour  (1 every 8–11 seconds)",
        "sched_error":        sched.get("error", ""),
        # DB cache info
        "cache":              cache,
    })


@app.route("/api/fundamentals/refresh", methods=["POST"])
def fundamentals_refresh():
    """Manual trigger — scheduler already runs automatically.
    This endpoint just confirms the scheduler is alive."""
    sched = fund_scheduler_status()
    return jsonify({
        "status":  "scheduler_running" if sched.get("running") else "starting",
        "message": ("Background scheduler is active — 10 stocks/hour automatically."
                    if sched.get("running")
                    else "Scheduler starting up…"),
    })


@app.route("/api/bhavcopy/status")
def bhavcopy_status():
    """
    Return the latest NSE bhavcopy date and the IST timestamp when it was cached.
    Used by the UI freshness bar to show live-data status to the user.
    """
    from datetime import datetime, timezone, timedelta
    from data_fetcher import _latest_bhavcopy_date, _bhav_cache_path
    import os

    IST = timezone(timedelta(hours=5, minutes=30))
    latest = _latest_bhavcopy_date()

    if latest is None:
        return jsonify({
            "latest_date":     None,
            "fetched_at_ist":  None,
            "fetched_at_unix": None,
            "status":          "no_data",
            "label":           "No bhavcopy cached yet",
        })

    cache_path = _bhav_cache_path(latest)
    mtime      = os.path.getmtime(cache_path)
    # fetched_ist = when the file was downloaded (mtime) — used for "fetched HH:MM"
    fetched_ist = datetime.fromtimestamp(mtime, tz=IST)
    today       = datetime.now(tz=IST).date()

    # Bug fix: trading_date_str comes from `latest` (the actual NSE trading date),
    # NOT from the file mtime — the file may have been downloaded the next morning.
    trading_date_str = latest.strftime("%d-%b-%Y")
    fetched_time_str = fetched_ist.strftime("%H:%M IST")

    if latest == today:
        status = "today"
        label  = f"Today's data · {trading_date_str} · fetched {fetched_time_str}"
    else:
        days_old = (today - latest).days
        status = "stale"
        label  = f"Last trading day: {trading_date_str} · fetched {fetched_time_str} · {days_old}d old"

    # Expose scheduler diagnostics so the UI / debugging can see WHY a stale
    # status persists (throttle, repeated NSE failures, session reset, etc.).
    from data_fetcher import _refresh_state
    sched_diag = {
        "last_attempt_msg":     _refresh_state.get("last_attempt_msg", ""),
        "consecutive_failures": _refresh_state.get("consecutive_failures", 0),
        "since_last_check_sec": int(time.time() - _refresh_state.get("last_checked", 0))
                                if _refresh_state.get("last_checked") else None,
        "since_last_success_sec": int(time.time() - _refresh_state.get("last_success_ts", 0))
                                if _refresh_state.get("last_success_ts") else None,
    }

    return jsonify({
        "latest_date":     str(latest),
        "fetched_at_ist":  fetched_ist.strftime("%d-%b-%Y %H:%M:%S IST"),
        "fetched_at_unix": int(mtime),
        "status":          status,       # "today" | "stale" | "no_data"
        "label":           label,
        "auto_refresh":    "every 5 min until today's data arrives, then every 20 min (stuck threshold: 4h)",
        "scheduler":       sched_diag,
    })


@app.route("/api/bhavcopy/refresh", methods=["POST"])
def bhavcopy_refresh():
    """Manual force-refresh — bypasses the auto-scheduler throttle. Used by
    the UI "Force Refresh" button so users can recover from a stuck scheduler
    without having to restart the server."""
    try:
        from data_fetcher import auto_refresh_bhavcopy
        result = auto_refresh_bhavcopy(force=True)
        # Convert date object to ISO string for JSON
        if result.get("date") is not None:
            result["date"] = str(result["date"])
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e), "downloaded": False}), 500


@app.route("/api/holders")
def holders():
    """P0-3: per-symbol FII/DII/MF/Promoter shareholding from NSE corp-info."""
    sym = (request.args.get("symbol") or "").strip().upper()
    if not sym:
        return jsonify({"error": "symbol required", "signal": "unknown"}), 400
    try:
        from holders_data import fetch_holders_for_symbol
        return jsonify(fetch_holders_for_symbol(sym))
    except Exception as e:
        return jsonify({"error": str(e), "signal": "unknown", "symbol": sym}), 500


@app.route("/api/consensus")
def consensus_top():
    """P2-10/11: cross-scan consensus — top stocks by appears-in-N-scans score."""
    try:
        from consensus import build_consensus, invalidate_cache as _inv
        # Allow forced rebuild so a fresh consensus reflects scanners that
        # just finished after the 10-min cache was last populated.
        if request.args.get("force") in ("1", "true", "yes"):
            _inv()
        d = build_consensus()
        d = _annotate_breakdown_payload(d)
        # Return top 50 only for the leaderboard view
        return jsonify({
            "top":         d["top"][:50],
            "total_symbols": d["total_symbols"],
            "computed_at": d["computed_at"],
        })
    except Exception as e:
        return jsonify({"error": str(e), "top": [], "total_symbols": 0}), 500


@app.route("/api/consensus/<symbol>")
def consensus_lookup(symbol):
    """P2-11: per-symbol cross-scan badge — how many scanners flagged this stock."""
    try:
        from consensus import appears_in
        return jsonify(appears_in(symbol))
    except Exception as e:
        return jsonify({"error": str(e), "symbol": symbol.upper(), "scan_count": 0}), 500


@app.route("/api/stage/<symbol>")
def stage_lookup(symbol):
    """P2-14: per-symbol stage info — current stage, days since transition."""
    try:
        from stage_transitions import get_stage_info
        return jsonify(get_stage_info(symbol))
    except Exception as e:
        return jsonify({"error": str(e), "symbol": symbol.upper()}), 500


@app.route("/api/metrics/status")
def metrics_status():
    """O1: introspection on the materialised stock_metrics table — when it
    was last built, how many rows, stage distribution."""
    try:
        from stock_metrics import status
        return jsonify(status())
    except Exception as e:
        return jsonify({"error": str(e), "built": False}), 500


@app.route("/api/metrics/refresh", methods=["POST"])
def metrics_refresh():
    """O1: force-rebuild the stock_metrics table. Normally triggered
    automatically by the bhavcopy auto-refresh handler."""
    try:
        from stock_metrics import refresh
        return jsonify(refresh())
    except Exception as e:
        return jsonify({"error": str(e), "built": 0}), 500


@app.route("/api/fii-dii")
def fii_dii_flow():
    """
    F4 — FII/DII market-wide daily flow from NSE.
    Two-step cookie seeding required: homepage → marketStatus → data API.
    Falls back to a cached result if the API is unreachable.
    Cached 4 hours (market updates intraday but wholesale enough for EOD use).
    """
    import requests as _req
    from datetime import datetime, timezone, timedelta

    _CACHE = fii_dii_flow.__dict__.setdefault("_cache", {"data": None, "ts": 0})
    if _CACHE["data"] and time.time() - _CACHE["ts"] < 4 * 3600:
        return jsonify(_CACHE["data"])

    def _seed_session():
        s = _req.Session()
        s.headers.update({
            "User-Agent":      "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                               "AppleWebKit/537.36 (KHTML, like Gecko) "
                               "Chrome/124.0.0.0 Safari/537.36",
            "Accept":          "application/json, text/plain, */*",
            "Accept-Language": "en-US,en;q=0.9",
            "Referer":         "https://www.nseindia.com/",
        })
        try:
            s.get("https://www.nseindia.com", timeout=8)
            s.get("https://www.nseindia.com/api/marketStatus", timeout=8)
        except Exception:
            pass
        return s

    IST = timezone(timedelta(hours=5, minutes=30))
    try:
        sess = _seed_session()
        r = sess.get(
            "https://www.nseindia.com/api/fiidiiTradeReact",
            timeout=12,
            headers={"Referer": "https://www.nseindia.com/market-data/fii-dii-activity"},
        )
        if r.status_code != 200:
            return jsonify({"error": f"NSE returned {r.status_code}", "rows": []}), 502
        raw = r.json()

        # NSE returns rows per category: {category: FII/DII, date, buyValue, sellValue, netValue}
        # Group by date, merge FII and DII rows
        from collections import defaultdict
        by_date: dict = defaultdict(dict)
        for item in raw:
            cat  = (item.get("category") or "").upper()
            dt   = item.get("date", "")
            bv   = float(str(item.get("buyValue",  0)).replace(",", "") or 0)
            sv   = float(str(item.get("sellValue", 0)).replace(",", "") or 0)
            nv   = float(str(item.get("netValue",  0)).replace(",", "") or 0)
            # NSE uses "FII/FPI" as category name
            key = "FII" if cat.startswith("FII") else "DII" if cat == "DII" else None
            if key and dt:
                by_date[dt][key] = {"buy": bv, "sell": sv, "net": nv}

        # BUG-026 FIX: Sort dates descending (newest first) before building response.
        # NSE returns rows in arbitrary order; without sorting, oldest dates appear first.
        sorted_dates = sorted(by_date.keys(), reverse=True)
        rows = []
        for dt in sorted_dates[:10]:   # last 10 trading days
            fd = by_date[dt].get("FII", {"buy": 0, "sell": 0, "net": 0})
            dd = by_date[dt].get("DII", {"buy": 0, "sell": 0, "net": 0})
            rows.append({
                "date":     dt,
                "fii_buy":  round(fd["buy"],  2),
                "fii_sell": round(fd["sell"], 2),
                "fii_net":  round(fd["net"],  2),
                "dii_buy":  round(dd["buy"],  2),
                "dii_sell": round(dd["sell"], 2),
                "dii_net":  round(dd["net"],  2),
            })

        # 5-day rolling net
        fii_5d = sum(r["fii_net"] for r in rows[:5])
        dii_5d = sum(r["dii_net"] for r in rows[:5])
        signal = ("FII Buying" if fii_5d > 500 else
                  "FII Selling" if fii_5d < -500 else "Neutral")
        out = {
            "rows":       rows,
            "fii_5d_net": round(fii_5d, 2),
            "dii_5d_net": round(dii_5d, 2),
            "signal":     signal,
            "fetched_at": datetime.now(tz=IST).strftime("%d-%b-%Y %H:%M IST"),
        }
        _CACHE["data"] = out
        _CACHE["ts"]   = time.time()
        return jsonify(out)

    except Exception as e:
        return jsonify({"error": str(e), "rows": []}), 500


@app.route("/api/sector/rrg")
def sector_rrg():
    """Compute Relative Rotation Graph data for all industry groups."""
    try:
        data = run_rrg_analysis()
        return jsonify(data)
    except Exception as e:
        return jsonify({"error": str(e), "sectors": [], "computed_at": int(time.time())}), 500


# ── Monster Growth Scanner ────────────────────────────────────────────────────

monster_state = {
    "running":  False,
    "progress": 0,
    "total":    100,
    "message":  "",
    "result":   None,
    "error":    None,
}
monster_lock = threading.Lock()

try:
    from monster_growth import run_monster_growth_scan, invalidate_cache as invalidate_monster_cache
    MONSTER_AVAILABLE = True
except ImportError as _mg_e:
    MONSTER_AVAILABLE = False
    print(f"[monster_growth] disabled: {_mg_e}")

# ── Early Growth Scanner ──────────────────────────────────────────────────────

early_growth_state = {
    "running":  False,
    "progress": 0,
    "total":    100,
    "message":  "",
    "result":   None,
    "error":    None,
}
early_growth_lock = threading.Lock()

try:
    from early_growth import run_early_growth_scan, invalidate_cache as invalidate_early_growth_cache
    EARLY_GROWTH_AVAILABLE = True
except ImportError as _eg_e:
    EARLY_GROWTH_AVAILABLE = False
    print(f"[early_growth] disabled: {_eg_e}")

if EARLY_GROWTH_AVAILABLE:
    def _do_early_growth_scan():
        def _pcb(done, total, msg):
            with early_growth_lock:
                early_growth_state["progress"] = done
                early_growth_state["total"]    = total
                early_growth_state["message"]  = msg
        try:
            with _scan_semaphore:
                result = run_early_growth_scan(progress_callback=_pcb)
            with early_growth_lock:
                early_growth_state["result"]  = result
                early_growth_state["running"] = False
        except Exception as e:
            with early_growth_lock:
                early_growth_state["error"]   = str(e)
                early_growth_state["running"] = False

    @app.route("/api/early_growth/scan", methods=["POST"])
    def api_early_growth_scan():
        with early_growth_lock:
            if early_growth_state["running"]:
                return jsonify({"status": "running"})
            early_growth_state.update({"running": True, "progress": 0,
                                       "result": None, "error": None,
                                       "message": "Starting scan…"})
        threading.Thread(target=_do_early_growth_scan, daemon=True).start()
        return jsonify({"status": "started"})

    @app.route("/api/early_growth/status")
    def api_early_growth_status():
        with early_growth_lock:
            s = dict(early_growth_state)
        pct = int(s["progress"] / max(s["total"], 1) * 100)
        return jsonify({
            "running":  s["running"],
            "pct":      pct,
            "message":  s["message"],
            "result":   s["result"],
            "error":    s["error"],
        })

if MONSTER_AVAILABLE:
    def _do_monster_scan():
        def _pcb(done, total, msg):
            with monster_lock:
                monster_state["progress"] = done
                monster_state["total"]    = total
                monster_state["message"]  = msg
        try:
            with _scan_semaphore:
                result = run_monster_growth_scan(progress_callback=_pcb)
            with monster_lock:
                monster_state["result"]  = result
                monster_state["running"] = False
        except Exception as e:
            with monster_lock:
                monster_state["error"]   = str(e)
                monster_state["running"] = False

    @app.route("/api/monster/scan", methods=["POST"])
    def api_monster_scan():
        with monster_lock:
            if monster_state["running"]:
                return jsonify({"status": "running"})
            monster_state.update({"running": True, "progress": 0,
                                  "result": None, "error": None,
                                  "message": "Starting scan…"})
        threading.Thread(target=_do_monster_scan, daemon=True).start()
        return jsonify({"status": "started"})

    @app.route("/api/monster/status")
    def api_monster_status():
        with monster_lock:
            s = dict(monster_state)
        pct = int(s["progress"] / max(s["total"], 1) * 100)
        return jsonify({
            "running":  s["running"],
            "pct":      pct,
            "message":  s["message"],
            "result":   s["result"],
            "error":    s["error"],
        })


# ── Mark Minervini VVV Scanner ────────────────────────────────────────────────

vvv_state = {
    "running":  False,
    "progress": 0,
    "total":    100,
    "message":  "",
    "result":   None,
    "error":    None,
}
vvv_lock = threading.Lock()

try:
    from minervini_vvv import run_vvv_scan, invalidate_cache as invalidate_vvv_cache
    VVV_AVAILABLE = True
except ImportError as _vvv_e:
    VVV_AVAILABLE = False
    print(f"[minervini_vvv] disabled: {_vvv_e}")

if VVV_AVAILABLE:
    def _do_vvv_scan():
        def _pcb(done, total, msg):
            with vvv_lock:
                vvv_state["progress"] = done
                vvv_state["total"]    = total
                vvv_state["message"]  = msg
        try:
            with _scan_semaphore:
                result = run_vvv_scan(progress_callback=_pcb)
            with vvv_lock:
                vvv_state["result"]  = result
                vvv_state["running"] = False
        except Exception as e:
            with vvv_lock:
                vvv_state["error"]   = str(e)
                vvv_state["running"] = False

    @app.route("/api/vvv/scan", methods=["POST"])
    def api_vvv_scan():
        with vvv_lock:
            if vvv_state["running"]:
                return jsonify({"status": "running"})
            vvv_state.update({"running": True, "progress": 0,
                              "result": None, "error": None,
                              "message": "Starting VVV scan…"})
        threading.Thread(target=_do_vvv_scan, daemon=True).start()
        return jsonify({"status": "started"})

    @app.route("/api/vvv/status")
    def api_vvv_status():
        with vvv_lock:
            s = dict(vvv_state)
        pct = int(s["progress"] / max(s["total"], 1) * 100)
        return jsonify({
            "running":  s["running"],
            "pct":      pct,
            "message":  s["message"],
            "result":   s["result"],
            "error":    s["error"],
        })


# ── Startup — runs whether launched via `python app.py` or gunicorn ───────────
# start_background_scheduler is idempotent (checks _sched["running"]), so calling
# it at module level is safe with both multi-worker gunicorn and plain python.
start_background_scheduler()

# Point-in-time fundamentals capture: weekly dated snapshots of fundamentals.db so
# the quality factor can eventually be validated with NO look-ahead bias (see
# fundamentals_history.py). Idempotent daemon; captures at most once every 7 days.
try:
    import fundamentals_history as _fh
    _fh.start_snapshot_scheduler()
except Exception:
    pass

# Auto-prime Market Breadth on startup so the header trend pill shows the
# authoritative breadth-based signal (instead of the quick adv/dec estimate)
# within ~30s of server boot, without the user needing to click the tab.
def _prime_market_breadth():
    try:
        from market_breadth import run_market_breadth
        time.sleep(2)        # let app finish booting
        run_market_breadth() # populates module-level _cache
    except Exception as e:
        print(f"[startup] Market Breadth prime failed: {e}")

if _BG_JOBS:
    threading.Thread(target=_prime_market_breadth, daemon=True).start()


# ── Bhavcopy auto-refresh scheduler ──────────────────────────────────────────
# ── Pre-warm (module-level so BOTH the bhavcopy scheduler AND server startup
# can trigger it). Runs every scanner SEQUENTIALLY so users never hit a cold
# cache. A non-blocking lock guarantees only one prewarm pass runs at a time
# (two overlapping passes would fight over the scan semaphore + RAM).
_prewarm_lock = threading.Lock()
_PREWARM_SCANS = [
    ("breadth",      "market_breadth",         "run_market_breadth"),
    ("trending",     "trending",               "run_trending_scan"),
    ("sector",       "sector_analysis",        "run_sector_analysis"),
    ("industry",     "industry_groups",        "run_industry_analysis"),
    # Added 2026-08-02: these read EMPTY after every restart otherwise.
    ("sector_leaders", "sector_leaders",       "run_sector_leaders"),
    ("edge",         "edge_engine",            "run_edge_engine"),
    ("breakout",     "breakout_scanner",       "run_breakout_scan"),
    ("momentum",     "momentum_scanner",       "run_momentum_scan"),
    ("emerging",     "emerging_leaders",       "run_emerging_leaders_scan"),
    ("volume",       "volume_scanner",         "run_volume_scan"),
    ("early_mover",  "early_mover_scanner",    "run_early_mover_scan"),
    ("advanced",     "advanced_scanner",       "run_advanced_scan"),
    ("institutional","institutional_scanner",  "run_institutional_scan"),
    ("monster",      "monster_growth",         "run_monster_growth_scan"),
    ("early_growth", "early_growth",           "run_early_growth_scan"),
    ("vvv",          "minervini_vvv",          "run_vvv_scan"),
    ("mbo",          "multiyear_breakout",     "run_multiyear_scan"),   # slowest — last
    # Added 2026-08-13 (#11). These three were NOT prewarmed, so the first visitor
    # after every restart paid the full scan on the request thread — the "Full scan
    # takes ~90 seconds" wait. They are cheap relative to mbo and share the already
    # warm base universe, so warming them here costs seconds and removes the worst
    # moment in the app.
    ("weekly_bo",    "weekly_breakout",        "run_weekly_breakout_scan"),
    ("post_bo",      "weekly_breakout",        "run_post_breakout_scan"),
    ("accumulation", "unusual_volume",         "run_accumulation_scan"),
    ("system",       "system_leaders",         "run_system_scan"),
]

def _prewarm_all_scans(trigger="startup"):
    if not _prewarm_lock.acquire(blocking=False):
        print(f"[prewarm/{trigger}] another prewarm already running — skipping", flush=True)
        return
    try:
        print(f"[prewarm/{trigger}] warming scan caches…", flush=True)
        for label, mod_name, fn_name in _PREWARM_SCANS:
            try:
                _t0 = time.time()
                _fn = getattr(__import__(mod_name), fn_name, None)
                if _fn is None:
                    continue
                _fn()   # synchronous — fills the module _cache as a side-effect
                print(f"[prewarm/{trigger}] {label}: {time.time()-_t0:.1f}s", flush=True)
            except Exception as _pe:
                print(f"[prewarm/{trigger}] {label} FAILED: {_pe}", flush=True)
        # Warm the CACHED-BUT-NOT-run_* endpoints too (allweather, promoter, today,
        # system) so the FIRST demo click is instant, not a 4-12s cold compute.
        # These compute inside their routes and cache by bhavcopy tag, so a single
        # server-side GET (synthetic admin session) fills their caches. Added
        # 2026-08-17 for the client demo — measured cold: today 11.9s, promoter 7.9s.
        try:
            _wc = app.test_client()
            with _wc.session_transaction() as _s:
                _s["user"], _s["role"] = "_prewarm", "admin"
            for _ep in ("/api/allweather", "/api/promoter", "/api/system", "/api/today"):
                try:
                    _t0 = time.time(); _wc.get(_ep)
                    print(f"[prewarm/{trigger}] warm {_ep}: {time.time()-_t0:.1f}s", flush=True)
                except Exception as _we:
                    print(f"[prewarm/{trigger}] warm {_ep} FAILED: {_we}", flush=True)
        except Exception as _we:
            print(f"[prewarm/{trigger}] endpoint warm skipped: {_we}", flush=True)
    finally:
        _prewarm_lock.release()


# Polls NSE every 20 minutes for the latest bhavcopy.
# NSE publishes after market close (~6–7 PM IST). This runs silently in the
# background so the DATA SOURCE bar always shows today's data automatically —
# no manual scan required.
def _bhavcopy_scheduler():
    from data_fetcher import auto_refresh_bhavcopy, _bhav_cache_path
    import time as _time
    from datetime import date as _date
    _time.sleep(5)   # let app fully boot first
    while True:
        try:
            result = auto_refresh_bhavcopy()
            if result.get("downloaded"):
                print(f"[bhavcopy_scheduler] ✅ New data: {result['date']} — "
                      "caches invalidated, next portfolio refresh uses today's prices.",
                      flush=True)
                # Also bust market breadth disk cache so it recomputes with fresh data
                try:
                    import market_breadth as _mb
                    _mb._cache["data"] = None
                    _mb._cache["ts"]   = 0
                except Exception:
                    pass
                # Bust Monster Growth cache — prices and technical signals stale
                try:
                    if MONSTER_AVAILABLE:
                        invalidate_monster_cache()
                except Exception:
                    pass
                # Bust Early Growth cache — stage/base analysis stale with new prices
                try:
                    if EARLY_GROWTH_AVAILABLE:
                        invalidate_early_growth_cache()
                except Exception:
                    pass
                # Bust Edge Engine cache — breakout levels, ATR stops, and setup
                # scores all derive from prices and are stale after a new bhavcopy.
                try:
                    invalidate_edge_cache()
                except Exception:
                    pass
                # Bust Minervini VVV cache — pipeline scoring depends on prices.
                try:
                    from minervini_vvv import invalidate_cache as _invalidate_vvv
                    _invalidate_vvv()
                except Exception:
                    pass
                # Bust Consensus cache — it aggregates across all scanner caches.
                try:
                    from consensus import invalidate_cache as _invalidate_con
                    _invalidate_con()
                except Exception:
                    pass
                # Re-populate stage_transitions on fresh bhavcopy so the
                # Stage Transitions tab reflects today's classifications
                # without waiting for the user to run Monster/Alpha/Early Growth.
                try:
                    from stage_transitions import populate_from_universe
                    res = populate_from_universe()
                    print(f"[bhavcopy_scheduler] stage_log refreshed: "
                          f"{res.get('updated', 0)} symbols, "
                          f"{res.get('transitioned', 0)} transitions",
                          flush=True)
                except Exception as _se:
                    print(f"[bhavcopy_scheduler] stage_log populate failed: {_se}",
                          flush=True)
                # Rebuild materialised stock_metrics table — pre-computed
                # MA/ATR/RSI/ADX/stage/rs_rank/ADTV indexed by symbol.
                try:
                    from stock_metrics import refresh as _stock_metrics_refresh
                    _smr = _stock_metrics_refresh()
                    print(f"[bhavcopy_scheduler] stock_metrics rebuilt: "
                          f"{_smr.get('built', 0)} rows, skipped={_smr.get('skipped', 0)}",
                          flush=True)
                except Exception as _me:
                    print(f"[bhavcopy_scheduler] stock_metrics refresh failed: {_me}",
                          flush=True)
                # Bust ALL remaining scanner caches that have their own _cache dict.
                # Without this, 8 scanner tabs serve yesterday's prices for up to
                # their TTL window after a new bhavcopy lands.
                for _mod_name in [
                    "alpha_engine",
                    "breakout_scanner",
                    "volume_scanner",
                    "momentum_scanner",
                    "emerging_leaders",
                    "advanced_scanner",
                    "institutional_scanner",
                    "multiyear_breakout",
                    "early_mover_scanner",
                    "trending",
                    "sector_analysis",
                    "industry_groups",
                    "investment_grade",
                ]:
                    try:
                        _mod = __import__(_mod_name)
                        if hasattr(_mod, "invalidate_cache"):
                            _mod.invalidate_cache()
                        elif hasattr(_mod, "_cache") and isinstance(_mod._cache, dict):
                            _mod._cache["data"] = None
                            _mod._cache["ts"]   = 0
                    except Exception:
                        pass

                # ── O2 — Pre-warm scans in background so users never hit a cold
                # cache (new bhavcopy just landed → caches were invalidated above).
                threading.Thread(target=_prewarm_all_scans, args=("bhavcopy",),
                                 daemon=True).start()
        except Exception as e:
            print(f"[bhavcopy_scheduler] error: {e}", flush=True)

        # Sleep adaptively: 5 min when today's bhavcopy is still missing
        # (aggressive — catches it within minutes of NSE publishing),
        # 20 min once we already have today's file.
        today_file = _bhav_cache_path(_date.today())
        sleep_secs = 300 if not today_file.exists() else 1200
        _time.sleep(sleep_secs)

if _BG_JOBS:
    threading.Thread(target=_bhavcopy_scheduler, daemon=True, name="bhavcopy-auto").start()
    print("[bhavcopy_scheduler] Started — checks NSE every 20 min automatically.")


# ── Boot pre-warm ────────────────────────────────────────────────────────────
# The scheduler above warms every scan when NEW bhavcopy data lands, but nothing
# warmed them on a plain start. So after a restart — or a code deploy, which
# invalidates every cache via the code-versioned tag — users hit cold tabs and
# the stock search honestly reports "N tabs not scanned today". This closes that
# gap: on boot, warm every scanner in the background, gated on a bhavcopy
# already existing locally so a cold machine doesn't stampede NSE (the scheduler
# owns that case). The non-blocking _prewarm_lock means this can never collide
# with a bhavcopy-triggered pass.
def _boot_prewarm():
    import time as _t
    _t.sleep(6)   # let the app finish booting and binding the port
    try:
        from data_fetcher import _latest_bhavcopy_date
        if _latest_bhavcopy_date() is None:
            print("[prewarm/startup] no bhavcopy on disk yet — leaving it to the scheduler",
                  flush=True)
            return
    except Exception as _e:
        print(f"[prewarm/startup] skipped: {_e}", flush=True)
        return
    _prewarm_all_scans("startup")


if _BG_JOBS:
    threading.Thread(target=_boot_prewarm, daemon=True, name="boot-prewarm").start()


# NOTE: There is intentionally NO startup pre-warm. Warming all scans on every
# boot exhausted the burstable EC2 instance's CPU credits and throttled
# production. Cache warming happens only via the once-a-day bhavcopy scheduler
# (off-peak), exactly as it did before — keep it that way on burstable hosts.


if __name__ == "__main__":
    print("NSE Trend Screener running at http://0.0.0.0:5050")
    app.run(host="0.0.0.0", debug=False, port=5050, use_reloader=False)
