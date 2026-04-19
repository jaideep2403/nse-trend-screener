import json
import threading
import time
from flask import Flask, render_template, request, jsonify, Response
from screener import run_screener

app = Flask(__name__)

# Shared state for background scan
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
            scan_state["total"] = total
            scan_state["current_ticker"] = ticker

    try:
        df, funnel = run_screener(params=params, progress_callback=progress_cb)
        with scan_lock:
            scan_state["result"] = df.to_dict(orient="records")
            scan_state["funnel"] = funnel
            scan_state["running"] = False
    except Exception as e:
        with scan_lock:
            scan_state["error"] = str(e)
            scan_state["running"] = False


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
            "error": None,
            "started_at": time.time(),
        })

    params = request.json or {}
    t = threading.Thread(target=do_scan, args=(params,), daemon=True)
    t.start()
    return jsonify({"status": "started"})


@app.route("/api/status")
def scan_status():
    with scan_lock:
        state = dict(scan_state)
    pct = 0
    if state["total"] > 0:
        pct = round(state["progress"] / state["total"] * 100, 1)
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


if __name__ == "__main__":
    print("NSE Trend Screener running at http://0.0.0.0:5050")
    app.run(host="0.0.0.0", debug=False, port=5050, use_reloader=False)
