import threading
import time
from flask import Flask, render_template, request, jsonify
from screener import run_screener
from sector_analysis import run_sector_analysis
from breakout_scanner import run_breakout_scan
from institutional_scanner import run_institutional_scan
from advanced_scanner import run_advanced_scan
from market_breadth import run_market_breadth
from industry_groups import run_industry_analysis

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
    return render_template("index.html")


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


if __name__ == "__main__":
    print("NSE Trend Screener running at http://0.0.0.0:5050")
    app.run(host="0.0.0.0", debug=False, port=5050, use_reloader=False)
