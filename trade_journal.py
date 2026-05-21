"""
Trade Journal — JSON file persistence at /tmp/trade_journal.json
Tracks trades: entry / exit / P&L / pattern / notes.
"""
import json
import time
import uuid
from pathlib import Path

JOURNAL_FILE = Path("/tmp/trade_journal.json")


def _load() -> list[dict]:
    if not JOURNAL_FILE.exists():
        return []
    try:
        with open(JOURNAL_FILE) as f:
            return json.load(f)
    except Exception:
        return []


def _save(trades: list[dict]):
    with open(JOURNAL_FILE, "w") as f:
        json.dump(trades, f, indent=2, default=str)


# ── Public API ────────────────────────────────────────────────────────────────

def get_trades() -> dict:
    trades = _load()
    closed = [t for t in trades if t.get("status") == "closed" and t.get("pnl") is not None]
    open_t = [t for t in trades if t.get("status") == "open"]
    wins   = [t for t in closed if float(t["pnl"]) > 0]
    losses = [t for t in closed if float(t["pnl"]) <= 0]

    total_pnl     = sum(float(t["pnl"]) for t in closed)
    win_rate      = round(len(wins) / len(closed) * 100, 1) if closed else 0
    avg_win       = round(sum(float(t["pnl"]) for t in wins)   / len(wins),   2) if wins   else 0
    avg_loss      = round(sum(float(t["pnl"]) for t in losses) / len(losses), 2) if losses else 0
    gross_wins    = sum(float(t["pnl"]) for t in wins)
    gross_losses  = abs(sum(float(t["pnl"]) for t in losses))
    profit_factor = round(gross_wins / gross_losses, 2) if gross_losses > 0 else (99.0 if wins else 0.0)

    return {
        "trades": trades,
        "stats": {
            "total_trades":  len(trades),
            "open_trades":   len(open_t),
            "closed_trades": len(closed),
            "win_rate":      win_rate,
            "total_pnl":     round(total_pnl, 2),
            "avg_win":       avg_win,
            "avg_loss":      avg_loss,
            "profit_factor": profit_factor,
        },
    }


def add_trade(data: dict) -> dict:
    trades = _load()
    trade = {
        "id":          str(uuid.uuid4())[:8],
        "symbol":      str(data.get("symbol", "")).upper().strip(),
        "pattern":     data.get("pattern", ""),
        "entry_price": float(data.get("entry_price", 0) or 0),
        "entry_date":  data.get("entry_date", ""),
        "sl":          float(data.get("sl", 0) or 0),
        "target":      float(data.get("target", 0) or 0),
        "qty":         int(data.get("qty", 0) or 0),
        # BUG-038 FIX: track LONG vs SHORT so P&L direction is correct.
        "side":        str(data.get("side", "LONG") or "LONG").upper(),
        "notes":       data.get("notes", ""),
        "status":      "open",
        "exit_price":  None,
        "exit_date":   None,
        "pnl":         None,
        "pnl_pct":     None,
        "created_at":  int(time.time()),
    }
    trades.append(trade)
    _save(trades)
    return trade


def update_trade(trade_id: str, data: dict) -> dict | None:
    trades = _load()
    for t in trades:
        if t["id"] == trade_id:
            if data.get("exit_price"):
                ep = float(data["exit_price"])
                t["exit_price"] = ep
                t["exit_date"]  = data.get("exit_date", "")
                t["status"]     = "closed"
                # BUG-038 FIX: shorts profit when price falls, so we must
                # reverse the sign of P&L for SHORT trades.
                side = str(t.get("side", "LONG") or "LONG").upper()
                if side == "SHORT":
                    pnl = (t["entry_price"] - ep) * t["qty"]
                    pct = (t["entry_price"] - ep) / t["entry_price"] * 100 if t["entry_price"] else 0
                else:
                    pnl = (ep - t["entry_price"]) * t["qty"]
                    pct = (ep - t["entry_price"]) / t["entry_price"] * 100 if t["entry_price"] else 0
                t["pnl"]     = round(pnl, 2)
                t["pnl_pct"] = round(pct, 2)
            for k in ["notes", "sl", "target", "qty"]:
                if k in data:
                    t[k] = data[k]
            _save(trades)
            return t
    return None


def delete_trade(trade_id: str) -> bool:
    trades   = _load()
    filtered = [t for t in trades if t["id"] != trade_id]
    if len(filtered) < len(trades):
        _save(filtered)
        return True
    return False
