"""
Watchlist — symbols the user is tracking but doesn't hold.

LOCAL-ONLY (same convention as portfolio.py): stored as JSON at
$DATA_DIR/.watchlist.json, gitignored, never pushed.

The Position Guardian sweeps watchlist symbols alongside portfolio holdings —
watched names alert on strength-fade (🔴 weakening flip) with severity capped
at 'watch', so the user hears about a breakdown BEFORE deciding to buy the dip.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

_PATH = Path(os.getenv("DATA_DIR", os.path.dirname(__file__) or ".")) / ".watchlist.json"


def _load() -> dict:
    if _PATH.exists():
        try:
            return json.loads(_PATH.read_text())
        except Exception:
            pass
    return {"symbols": {}}


def _save(data: dict) -> None:
    tmp = _PATH.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, indent=2))
    tmp.replace(_PATH)


def get_symbols() -> list[str]:
    return sorted(_load()["symbols"].keys())


def toggle(symbol: str) -> dict:
    """Add the symbol if absent, remove it if present. Returns {symbol, watched}."""
    sym = (symbol or "").upper().strip()
    if not sym:
        return {"symbol": symbol, "watched": False, "error": "empty symbol"}
    data = _load()
    if sym in data["symbols"]:
        del data["symbols"][sym]
        watched = False
    else:
        data["symbols"][sym] = {"added_at": int(time.time())}
        watched = True
    _save(data)
    return {"symbol": sym, "watched": watched}
