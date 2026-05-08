import threading
import time
import pandas as pd
from flask import Flask, render_template, request, jsonify, make_response
from screener import run_screener
from sector_analysis import run_sector_analysis
from breakout_scanner import run_breakout_scan
from institutional_scanner import run_institutional_scan
from advanced_scanner import run_advanced_scan
from market_breadth import run_market_breadth
from industry_groups import run_industry_analysis, run_rrg_analysis
from momentum_scanner import run_momentum_scan
from early_mover_scanner import run_early_mover_scan
from volume_scanner import run_volume_scan
from edge_engine import run_edge_engine, detect_exit_signals, _load_stocks, invalidate_cache as invalidate_edge_cache
from trending import run_trending_scan
from header_data import get_market_header
from fundamentals import (cache_status as fund_cache_status,
                          scheduler_status as fund_scheduler_status,
                          start_background_scheduler)

app = Flask(__name__)

# ── LOCAL-ONLY: Personal Portfolio (gitignored) ───────────────────────────────
# Gracefully no-op if portfolio.py is missing (e.g. on EC2 deploy from GitHub).
PORTFOLIO_AVAILABLE = False
try:
    import portfolio as _pf
    PORTFOLIO_AVAILABLE = True
except Exception as _pf_e:
    print(f"[portfolio] disabled (file not present): {_pf_e}")

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
            return jsonify({"error": str(e)}), 400

    @app.route("/api/portfolio/<pos_id>", methods=["DELETE"])
    def api_portfolio_delete(pos_id):
        try:
            ok = _pf.delete_position(pos_id)
            return jsonify({"ok": ok})
        except Exception as e:
            return jsonify({"error": str(e)}), 500

# Inject flag into all templates (read by index.html to show/hide the tab)
@app.context_processor
def inject_portfolio_flag():
    return {"portfolio_enabled": PORTFOLIO_AVAILABLE}

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
        # Always force a fresh run — clear 30-min cache so new fundamentals are picked up
        invalidate_edge_cache()
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
        "rate":               "~240 stocks / hour  (1 every 15–20 seconds)",
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

    return jsonify({
        "latest_date":     str(latest),
        "fetched_at_ist":  fetched_ist.strftime("%d-%b-%Y %H:%M:%S IST"),
        "fetched_at_unix": int(mtime),
        "status":          status,       # "today" | "stale" | "no_data"
        "label":           label,
    })


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

        rows = []
        for dt in list(by_date.keys())[:10]:   # last 10 trading days
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
        atr14 = float(tr.rolling(14).mean().iloc[-1]) if len(tr) >= 14 else float(tr.mean())

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

if __name__ == "__main__":
    print("NSE Trend Screener running at http://0.0.0.0:5050")
    app.run(host="0.0.0.0", debug=False, port=5050, use_reloader=False)
