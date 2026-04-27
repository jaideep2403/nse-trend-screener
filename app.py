import threading
import time
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
from fundamentals import (cache_status as fund_cache_status,
                          scheduler_status as fund_scheduler_status,
                          start_background_scheduler)

app = Flask(__name__)

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
    dt_ist     = datetime.fromtimestamp(mtime, tz=IST)
    today      = datetime.now(tz=IST).date()

    if latest == today:
        status = "today"
        label  = f"Today's data · {dt_ist.strftime('%d-%b-%Y')} · fetched {dt_ist.strftime('%H:%M IST')}"
    else:
        days_old = (today - latest).days
        status = "stale"
        label  = f"Last trading day: {dt_ist.strftime('%d-%b-%Y')} · fetched {dt_ist.strftime('%H:%M IST')} · {days_old}d old"

    return jsonify({
        "latest_date":     str(latest),
        "fetched_at_ist":  dt_ist.strftime("%d-%b-%Y %H:%M:%S IST"),
        "fetched_at_unix": int(mtime),
        "status":          status,       # "today" | "stale" | "no_data"
        "label":           label,
    })


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

if __name__ == "__main__":
    print("NSE Trend Screener running at http://0.0.0.0:5050")
    app.run(host="0.0.0.0", debug=False, port=5050, use_reloader=False)
