import os
import threading
import time
import pandas as pd
from flask import Flask, render_template, request, jsonify, make_response

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
    _refresh_sectors(background=True)   # builds .sector_cache.json in ~2s, non-blocking
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
    return {
        "portfolio_enabled":   PORTFOLIO_AVAILABLE,
        "strategy_enabled":    STRATEGY_AVAILABLE,
        "investgrade_enabled": INVESTGRADE_AVAILABLE,
        "alpha_enabled":       ALPHA_AVAILABLE,
        "monster_enabled":        MONSTER_AVAILABLE,
        "early_growth_enabled":   EARLY_GROWTH_AVAILABLE,
        "vvv_enabled":         globals().get("VVV_AVAILABLE", False),
        "universe_label":      "Nifty Total Market",
        "universe_label_short": "NSE Total Market",
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


@app.route("/api/header")
def api_header():
    """Lightweight header: Nifty 50 level, day change %, adv/dec, market trend."""
    try:
        return jsonify(get_market_header())
    except Exception as e:
        return jsonify({"error": str(e)}), 500


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


@app.route("/api/stage-transitions")
def stage_transitions_recent():
    """P2-14: list stocks that recently transitioned into a target stage.

    Self-populates when the DB is empty so the tab works on first open even
    if the user hasn't run Monster Growth / Early Growth / Alpha Engine yet
    (those are the only scanners that explicitly call update_all). When the
    user clicks Refresh with force=1 we also re-populate, useful after a new
    bhavcopy lands.
    """
    try:
        from stage_transitions import recent_transitions, stats, populate_from_universe
        stage = int(request.args.get("into_stage", 2))
        days  = int(request.args.get("within_days", 10))
        force = request.args.get("force") in ("1", "true", "yes")

        st = stats()
        if st.get("total", 0) == 0 or force:
            populate_from_universe()
            st = stats()

        return jsonify({
            "stage":   stage,
            "within":  days,
            "stocks":  recent_transitions(stage, within_days=days),
            "stats":   st,
        })
    except Exception as e:
        return jsonify({"error": str(e), "stocks": []}), 500


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


@app.route("/api/position-size")
def position_size():
    """
    F6 — ATR-based position sizing calculator.
    Params: symbol, capital (default 1000000 = ₹10L), risk_pct (default 1.0)
    """
    symbol   = request.args.get("symbol", "").upper()
    capital  = request.args.get("capital",  1_000_000, type=float)
    risk_pct = request.args.get("risk_pct", 1.0,       type=float)
    if not symbol:
        return jsonify({"error": "symbol required"}), 400
    try:
        # Use fetch_ohlcv which builds per-stock pkl from bhavcopy cache (fast)
        from data_fetcher import fetch_ohlcv
        ticker = symbol if symbol.endswith(".NS") else f"{symbol}.NS"
        ohlcv_map = fetch_ohlcv([ticker], min_bars=20)
        df = ohlcv_map.get(ticker)
        if df is None or len(df) < 20:
            return jsonify({"error": f"No data for {symbol}"}), 404

        close = df["Close"].dropna()
        hi    = df["High"].dropna()
        lo    = df["Low"].dropna()
        cur   = float(close.iloc[-1])

        tr = pd.concat([
            hi - lo,
            (hi - close.shift(1)).abs(),
            (lo - close.shift(1)).abs(),
        ], axis=1).max(axis=1)
        # Wilder's ATR (canonical) — old rolling SMA over-stated ATR by 5-15%
        # in trending markets, making /api/position-size stops & sizing inconsistent
        # with every other scanner.
        atr14 = (float(tr.ewm(alpha=1/14, adjust=False).mean().iloc[-1])
                 if len(tr) >= 14 else float(tr.mean()))

        stop_1atr = round(cur - atr14, 2)
        stop_1_5atr = round(cur - 1.5 * atr14, 2)
        risk_amount  = capital * risk_pct / 100
        risk_per_sh  = max(cur - stop_1_5atr, 0.01)
        shares       = int(risk_amount / risk_per_sh)
        position_val = round(shares * cur, 2)
        pos_pct_cap  = round(position_val / capital * 100, 1) if capital > 0 else 0

        return jsonify({
            "symbol":       symbol,
            "price":        round(cur, 2),
            "atr14":        round(atr14, 2),
            "atr_pct":      round(atr14 / cur * 100, 2),
            "stop_1atr":    stop_1atr,
            "stop_1_5atr":  stop_1_5atr,
            "capital":      capital,
            "risk_pct":     risk_pct,
            "risk_amount":  round(risk_amount, 2),
            "shares":       shares,
            "position_value": position_val,
            "position_pct_of_capital": pos_pct_cap,
            "target_2r":    round(cur + 2 * (cur - stop_1_5atr), 2),
            "target_3r":    round(cur + 3 * (cur - stop_1_5atr), 2),
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


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

threading.Thread(target=_bhavcopy_scheduler, daemon=True, name="bhavcopy-auto").start()
print("[bhavcopy_scheduler] Started — checks NSE every 20 min automatically.")


# NOTE: There is intentionally NO startup pre-warm. Warming all scans on every
# boot exhausted the burstable EC2 instance's CPU credits and throttled
# production. Cache warming happens only via the once-a-day bhavcopy scheduler
# (off-peak), exactly as it did before — keep it that way on burstable hosts.


if __name__ == "__main__":
    print("NSE Trend Screener running at http://0.0.0.0:5050")
    app.run(host="0.0.0.0", debug=False, port=5050, use_reloader=False)
