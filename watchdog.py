#!/usr/bin/env python3
"""
Hang watchdog for the Fortune X dashboard (launchd job com.local.nse-dashboard).

WHY: launchd's KeepAlive only restarts the app when the process EXITS. On 2026-09-23 the
process stayed alive but stopped answering — the Mac was swap-thrashing (8.8 of 10 GB swap,
~1.2B page swaps), a 3-second breadth scan took 48 minutes, and every page timed out.
KeepAlive saw nothing wrong because nothing had crashed.

HOW: this runs from its OWN launchd job every 60s, so it cannot be starved by the app it
watches (an in-process watcher thread would freeze with it). Each run:
  1. finds the app's pid + age via launchctl/ps; skips while it is still booting;
  2. probes GET /healthz (public, no auth, no work) with a generous timeout;
  3. after FAILS_TO_RESTART consecutive failures (~3 min unresponsive), and at most once per
     COOLDOWN, restarts it with `launchctl kickstart -k` — logging the memory/swap state at
     that moment and raising a macOS notification so the restart is never silent.
Slow-but-successful probes are logged too: they are the early warning of memory pressure.

State + log live in ~/.ascent_cache (persistent — /tmp is purged by macOS).
"""
from __future__ import annotations

import json
import os
import subprocess
import time
import urllib.request
from datetime import datetime
from pathlib import Path

LABEL = "com.local.nse-dashboard"
TARGET = f"gui/{os.getuid()}/{LABEL}"
URL = os.environ.get("WATCHDOG_URL", "http://127.0.0.1:5050/healthz")
DRY_RUN = os.environ.get("WATCHDOG_DRY_RUN") == "1"   # test hook: decide, but never restart

PROBE_TIMEOUT = 15        # s — generous: a thrashing-but-alive app still answers within this
SLOW_WARN = 5.0           # s — log a successful probe slower than this (memory-pressure hint)
FAILS_TO_RESTART = 3      # consecutive failed probes (job runs every 60s → ~3 min unresponsive)
BOOT_GRACE = 240          # s — never judge an app younger than this (boot + warm-up)
COOLDOWN = 600            # s — at most one restart per 10 min (no restart loops)

HOME = Path.home() / ".ascent_cache"
STATE = HOME / "watchdog_state.json"
LOG = HOME / "watchdog.log"
LOG_MAX_BYTES = 512 * 1024


def _log(msg: str) -> None:
    HOME.mkdir(parents=True, exist_ok=True)
    try:
        if LOG.exists() and LOG.stat().st_size > LOG_MAX_BYTES:      # keep it small
            lines = LOG.read_text().splitlines()[-400:]
            LOG.write_text("\n".join(lines) + "\n")
        with LOG.open("a") as f:
            f.write(f"{datetime.now():%Y-%m-%d %H:%M:%S}  {msg}\n")
    except Exception:
        pass


def _load() -> dict:
    try:
        return json.loads(STATE.read_text())
    except Exception:
        return {"fails": 0, "last_restart": 0.0}


def _save(st: dict) -> None:
    try:
        HOME.mkdir(parents=True, exist_ok=True)
        tmp = STATE.with_suffix(".tmp")
        tmp.write_text(json.dumps(st))
        os.replace(tmp, STATE)
    except Exception:
        pass


def _run(cmd: list[str], timeout: float = 10) -> str:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout).stdout
    except Exception:
        return ""


def _app_pid() -> int | None:
    for line in _run(["launchctl", "print", TARGET]).splitlines():
        line = line.strip()
        if line.startswith("pid ="):
            try:
                return int(line.split("=", 1)[1])
            except ValueError:
                return None
    return None


def _age_seconds(pid: int) -> int | None:
    """Parse `ps -o etime=` ([[dd-]hh:]mm:ss) into seconds."""
    et = _run(["ps", "-o", "etime=", "-p", str(pid)]).strip()
    if not et:
        return None
    days = 0
    if "-" in et:
        d, et = et.split("-", 1)
        days = int(d)
    parts = [int(p) for p in et.split(":")]
    while len(parts) < 3:
        parts.insert(0, 0)
    h, m, s = parts
    return ((days * 24 + h) * 60 + m) * 60 + s


def _probe() -> tuple[bool, float, str]:
    t0 = time.time()
    try:
        with urllib.request.urlopen(URL, timeout=PROBE_TIMEOUT) as r:
            body = r.read(64).decode("utf-8", "ignore").strip()
            ok = r.status == 200 and body.startswith("ok")
            return ok, time.time() - t0, f"HTTP {r.status}"
    except Exception as e:
        return False, time.time() - t0, type(e).__name__ + ": " + str(e)[:120]


def _memory_snapshot() -> str:
    swap = _run(["sysctl", "-n", "vm.swapusage"]).strip()
    free = ""
    for line in _run(["memory_pressure"], timeout=15).splitlines():
        if "free percentage" in line:
            free = line.split(":")[-1].strip()
    return f"swap[{swap}] mem-free {free or '?'}"


def _notify(msg: str) -> None:
    _run(["osascript", "-e",
          f'display notification "{msg}" with title "Fortune X watchdog"'], timeout=5)


def main() -> None:
    st = _load()
    st["last_check"] = time.time()            # heartbeat: proves the watchdog itself is running
    pid = _app_pid()
    if pid is None:
        # Not running — launchd's KeepAlive owns process restarts; just record it.
        _log("app process not running (KeepAlive will relaunch it)")
        st["fails"] = 0
        _save(st)
        return

    age = _age_seconds(pid)
    if age is not None and age < BOOT_GRACE:
        st["fails"] = 0                       # still booting / warming — don't judge yet
        _save(st)
        return

    ok, dt, info = _probe()
    if ok:
        if st.get("fails"):
            _log(f"recovered after {st['fails']} failed probe(s) — /healthz {dt:.1f}s (pid {pid})")
        elif dt > SLOW_WARN:
            _log(f"SLOW but alive — /healthz took {dt:.1f}s (pid {pid}) · {_memory_snapshot()}")
        st["fails"] = 0
        st["last_ok"] = time.time()
        _save(st)
        return

    st["fails"] = int(st.get("fails", 0)) + 1
    _log(f"probe FAILED {st['fails']}/{FAILS_TO_RESTART} after {dt:.1f}s: {info} (pid {pid}, up {age}s)")

    since = time.time() - float(st.get("last_restart", 0))
    if st["fails"] >= FAILS_TO_RESTART:
        if since < COOLDOWN:
            _log(f"restart suppressed — last restart {since:.0f}s ago (cooldown {COOLDOWN}s)")
        else:
            snap = _memory_snapshot()
            _log(f"{'DRY-RUN would restart' if DRY_RUN else 'RESTARTING'} hung app "
                 f"(pid {pid}, up {age}s, {st['fails']} failed probes) · {snap}")
            if not DRY_RUN:
                _run(["launchctl", "kickstart", "-k", TARGET], timeout=30)
                _notify("App was unresponsive for ~3 min — restarted automatically.")
            st["fails"] = 0
            st["last_restart"] = time.time()
    _save(st)


if __name__ == "__main__":
    main()
