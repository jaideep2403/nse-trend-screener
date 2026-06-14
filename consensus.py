"""
Cross-scan Consensus Engine.

P2-10/11: aggregates results from every scanner into a single per-symbol
"how many scanners flagged this stock" registry. Useful for:
  - "Final Setup Score" = weighted sum of tier rankings across scanners
  - "Appears in N scans" badge on every tab
  - Cross-tab leader board

This module is READ-ONLY relative to scanner outputs — it does not re-scan,
just queries cached results from each scanner module.

Scanners and their tier weights:
    monster_growth   : MONSTER=4, STRONG=3, EMERGING=2, SKIP=0
    alpha_engine     : BUY=4, STRONG=3, MONITOR=2, AVOID=0
    early_growth     : EARLY_MONSTER=4, WATCH=3, TRACK=2, _=0
    institutional    : presence in top results = 3
    momentum_scanner : presence = 2
    edge_engine      : top-quality breakouts = 3
"""
from __future__ import annotations

import time
from typing import Optional


_CACHE: dict = {"data": None, "ts": 0.0}
_CACHE_TTL = 600  # 10 min — refresh consensus when any underlying scan reruns


# Per-scanner tier-to-points map
_TIER_WEIGHTS = {
    # MONSTER GROWTH
    "MONSTER":       4,
    "STRONG":        3,
    "EMERGING":      2,
    # ALPHA ENGINE
    "BUY":           4,
    "MONITOR":       2,
    # EARLY GROWTH
    "EARLY_MONSTER": 4,
    "WATCH":         3,
    "TRACK":         2,
    # MINERVINI VVV
    "IDEAL":         4,   # VVV "🏆 IDEAL" (score ≥ 85)
}

# Theoretical max raw points a symbol can collect across all 7 recorded
# scanners: tiered scanners (Monster, Alpha, Early Growth, VVV) award up to
# 4 each = 16; presence-only scanners (Institutional, Edge, Trending) award
# 2 each = 6 → 22. The old hardcoded 24.0 ("4pts × 6 scanners") and the
# separate 4*6 in appears_in() were both wrong AND inconsistent with each
# other, so the leaderboard and per-symbol lookup scored on different scales.
MAX_RAW_POINTS = 22.0


def _read_cache(mod) -> dict:
    """Best-effort read of any scanner's `_cache` dict."""
    try:
        c = getattr(mod, "_cache", None)
        if c and isinstance(c, dict):
            return c.get("data") or {}
    except Exception:
        pass
    return {}


def _gather_scanner_hits() -> dict[str, dict]:
    """
    Returns {symbol: {
        scans: [{scanner, tier, score, why}, ...],
        scan_count: int,
        consensus_score: float (0-100),
        tier_label: str,
    }}.
    """
    hits: dict[str, dict] = {}

    def _record(scanner: str, symbol: str, tier: Optional[str], score: float, why: str = ""):
        rec = hits.setdefault(symbol, {"scans": [], "scan_count": 0,
                                       "raw_points": 0.0, "best_tier": ""})
        rec["scans"].append({
            "scanner": scanner,
            "tier":    tier,
            "score":   round(score, 1) if score else 0,
            "why":     (why or "")[:80],
        })
        rec["scan_count"] += 1
        # Normalise tier strings — VVV emits "🏆 IDEAL", "✅ STRONG", etc.
        # Strip non-ASCII (emoji) and lookup the alpha-only key.
        def _tier_key(t):
            if not t:
                return ""
            ascii_only = "".join(ch for ch in t if ch.isascii()).strip()
            # Take the last alpha token (e.g. "EARLY MONSTER" → "MONSTER"; "MONSTER" → "MONSTER")
            tokens = [tok for tok in ascii_only.upper().split() if tok.isalpha() or "_" in tok]
            if not tokens:
                return ascii_only.upper()
            # If the scanner uses underscored compound tiers (EARLY_MONSTER), keep it whole
            joined = "_".join(tokens) if len(tokens) > 1 and any("_" not in t for t in tokens) else tokens[-1]
            # Prefer exact match in weights; fall back to last token
            if joined in _TIER_WEIGHTS:
                return joined
            return tokens[-1]
        pts = _TIER_WEIGHTS.get(_tier_key(tier), 0)
        # If no tier (or unknown), give a smaller credit based on the raw score
        # so a top-30 stock in a tier-less scanner still counts.
        if pts == 0 and score and score >= 60:
            pts = 2
        rec["raw_points"] += pts
        # Track strongest tier seen
        if pts > _TIER_WEIGHTS.get(_tier_key(rec["best_tier"]), 0):
            rec["best_tier"] = tier or ""

    # ── Monster Growth ──
    try:
        import monster_growth
        d = _read_cache(monster_growth)
        for r in (d.get("results") or [])[:80]:
            _record("Monster Growth", r["symbol"], r.get("tier"), r.get("score", 0),
                    f"Profit {r.get('profit_gr',0):.0f}% PEG {r.get('peg',0)}")
    except Exception:
        pass

    # ── Alpha Engine ──
    try:
        import alpha_engine
        d = _read_cache(alpha_engine)
        for r in (d.get("results") or [])[:80]:
            _record("Alpha Engine", r["symbol"], r.get("tier"), r.get("score", 0),
                    r.get("why", ""))
    except Exception:
        pass

    # ── Early Growth ──
    try:
        import early_growth
        d = _read_cache(early_growth)
        for r in (d.get("results") or [])[:60]:
            _record("Early Growth", r["symbol"], r.get("tier"), r.get("score", 0),
                    f"Base {r.get('base_weeks',0)}w · accel")
    except Exception:
        pass

    # ── Institutional Scanner ──
    try:
        import institutional_scanner
        d = _read_cache(institutional_scanner)
        for r in (d.get("results") or [])[:40]:
            _record("Institutional", r["symbol"], None, r.get("score", 0),
                    r.get("setup_label", "PocketPivot / EarnSetup"))
    except Exception:
        pass

    # ── Edge Engine top setups ──
    try:
        import edge_engine
        d = _read_cache(edge_engine)
        # Edge Engine stores its top-200 list under "ranked"
        for r in (d.get("ranked") or d.get("top_setups") or d.get("results") or [])[:40]:
            sym = r.get("symbol")
            if sym:
                _record("Edge Engine", sym, None, r.get("score", 0),
                        r.get("setup_label", "Quality breakout"))
    except Exception:
        pass

    # ── Momentum / Volume / Industry / Trending: presence-only, no per-symbol cache typically ──
    try:
        import trending
        d = _read_cache(trending)
        # trending scanner stores under "stocks", not "results".
        # Its score is 0-10 (not 0-100): scale ×10 so the tier-less presence
        # credit (score ≥ 60 → 2 pts) can actually trigger — previously a 9.5/10
        # trending leader earned ZERO consensus points.
        for r in (d.get("stocks") or d.get("results") or [])[:30]:
            _record("Trending", r["symbol"], None, (r.get("score", 0) or 0) * 10,
                    "Trending")
    except Exception:
        pass

    # ── Minervini VVV ──
    try:
        import minervini_vvv
        d = _read_cache(minervini_vvv)
        for r in (d.get("results") or [])[:40]:
            _record("Minervini VVV", r["symbol"], r.get("tier"), r.get("score", 0),
                    r.get("pattern", "VVV setup"))
    except Exception:
        pass

    # Normalize: consensus_score = scaled to 0-100 based on MAX_RAW_POINTS (22).
    # In PRACTICE the best stocks in any market hit 12-16 raw points
    # (= 55-73 normalised) and the very top scores rarely cross 75; the tier
    # thresholds below are calibrated to that empirical distribution.
    for sym, rec in hits.items():
        rec["consensus_score"] = round(min(100.0, rec["raw_points"] / MAX_RAW_POINTS * 100), 1)
        rec["tier_label"] = _consensus_tier_label(rec["consensus_score"])

    return hits


def _consensus_tier_label(score: float) -> str:
    """Single source of truth for consensus tier labels — used by both
    build_consensus() (bulk leaderboard) and appears_in() (per-symbol lookup)
    so the same stock never shows different badges on different views."""
    if   score >= 50: return "CONSENSUS BUY"
    elif score >= 35: return "STRONG CONSENSUS"
    elif score >= 20: return "EMERGING"
    elif score >= 10: return "WATCH"
    else:             return "WEAK"


def build_consensus(force: bool = False) -> dict:
    """
    Build the cross-scanner consensus. Cached 10 min.
    Returns {top: [...sorted...], by_symbol: {sym: rec}, computed_at: ts}.
    """
    if (not force
        and _CACHE["data"]
        and time.time() - _CACHE["ts"] < _CACHE_TTL):
        return _CACHE["data"]

    by_symbol = _gather_scanner_hits()
    top = sorted(
        [{"symbol": s, **r} for s, r in by_symbol.items()],
        key=lambda x: (-x["consensus_score"], -x["scan_count"])
    )
    out = {
        "by_symbol":   by_symbol,
        "top":         top,
        "total_symbols": len(by_symbol),
        "computed_at": int(time.time()),
    }
    _CACHE["data"] = out
    _CACHE["ts"]   = time.time()
    return out


def appears_in(symbol: str) -> dict:
    """Quick lookup: every scanner cache that contains `symbol`, regardless of
    rank. The previous implementation relied on `build_consensus()` which
    truncates each scanner to its top-30/40/80 — so a stock at e.g. Edge
    rank #80 was reported as "not in Edge". This walks each scanner cache
    directly without the truncation.
    """
    if not symbol:
        return {"symbol": "", "scan_count": 0, "scans": []}
    sym = symbol.upper()
    scans: list[dict] = []
    best_pts = 0

    def _record(scanner: str, row: dict, tier_key_name: str = "tier",
                score_key_name: str = "score", why: str = ""):
        nonlocal best_pts
        tier = row.get(tier_key_name)
        score = row.get(score_key_name, 0) or 0
        scans.append({"scanner": scanner, "tier": tier,
                      "score": round(float(score), 1) if score else 0,
                      "why": (why or "")[:80]})
        # Reuse the same token-based tier→points lookup as build_consensus
        if not tier:
            pts = 2 if score and score >= 60 else 0
        else:
            ascii_only = "".join(ch for ch in str(tier) if ch.isascii()).strip().upper()
            tokens = [t for t in ascii_only.split() if t.isalpha() or "_" in t]
            joined = "_".join(tokens) if len(tokens) > 1 and any("_" not in t for t in tokens) else (tokens[-1] if tokens else "")
            pts = _TIER_WEIGHTS.get(joined if joined in _TIER_WEIGHTS else (tokens[-1] if tokens else ""), 0)
            if pts == 0 and score and score >= 60:
                pts = 2
        if pts > best_pts:
            best_pts = pts

    # Helper to find symbol in any list-like cache field
    def _find(mod_name: str, key_names: list[str]) -> dict | None:
        try:
            mod = __import__(mod_name)
            cache = getattr(mod, "_cache", None)
            if not cache:
                return None
            data = cache.get("data") or {}
            for k in key_names:
                items = data.get(k)
                if items is None:
                    continue
                for r in items:
                    if (r.get("symbol") or "").upper() == sym:
                        return r
        except Exception:
            pass
        return None

    r = _find("monster_growth",       ["results"])
    if r: _record("Monster Growth", r, why=f"Profit {r.get('profit_gr',0):.0f}%")
    r = _find("alpha_engine",         ["results"])
    if r: _record("Alpha Engine", r, why=r.get("why",""))
    r = _find("early_growth",         ["results"])
    if r: _record("Early Growth", r, why="Early growth")
    r = _find("institutional_scanner",["results"])
    if r: _record("Institutional", r, why=r.get("setup_label",""))
    r = _find("edge_engine",          ["ranked","top_setups","results"])
    if r: _record("Edge Engine", r, why=r.get("setup_label","Quality breakout"))
    r = _find("trending",             ["stocks","results"])
    if r:
        # trending score is 0-10 — scale to the 0-100 convention (same as
        # build_consensus) so presence credit can trigger.
        _record("Trending", {**r, "score": (r.get("score") or 0) * 10}, why="Trending")
    r = _find("minervini_vvv",        ["results"])
    if r: _record("Minervini VVV", r, why=r.get("pattern","VVV"))

    if not scans:
        return {"symbol": sym, "scan_count": 0, "scans": [],
                "consensus_score": 0, "tier_label": "—"}

    # Composite score on the same 0-100 scale build_consensus uses
    raw_pts = sum(_TIER_WEIGHTS.get(
        "".join(ch for ch in (s.get("tier") or "").upper() if ch.isascii()).strip().split()[-1]
        if (s.get("tier") and "".join(ch for ch in s["tier"].upper() if ch.isascii()).strip())
        else "", 0
    ) or (2 if (s.get("score") or 0) >= 60 else 0) for s in scans)
    consensus_score = round(min(100.0, raw_pts / MAX_RAW_POINTS * 100), 1)
    # Single source of truth — same labels as build_consensus() so per-symbol
    # lookups never show a different badge than the leaderboard view.
    tier_label = _consensus_tier_label(consensus_score)

    return {
        "symbol": sym,
        "scan_count": len(scans),
        "scans": scans,
        "consensus_score": consensus_score,
        "tier_label": tier_label,
        "raw_points": raw_pts,
    }


def invalidate_cache():
    _CACHE["data"] = None
    _CACHE["ts"]   = 0.0


def enrich_results(results: list[dict]) -> list[dict]:
    """
    Stamp every row with `consensus_score`, `consensus_tier`, `scan_count`,
    `stage_label` (e.g. "S2 fresh 3d"), `stage_fresh` flag — so every tab can
    render the same Final Setup Score column without per-scanner UI work.

    Called from each scanner's run_*_scan after results are computed but before
    they're cached / returned. Always cheap — reads SQLite + an in-memory dict.
    """
    if not results:
        return results

    # Build consensus once (cached 10 min internally)
    try:
        cons = build_consensus()
        by_sym = cons.get("by_symbol", {})
    except Exception:
        by_sym = {}

    # Stage info — bulk fetch one row at a time is fine (SQLite indexed)
    try:
        from stage_transitions import get_stage_info
    except Exception:
        get_stage_info = None

    for r in results:
        sym = r.get("symbol")
        if not sym:
            continue
        # ── Consensus stamp ──
        rec = by_sym.get(sym)
        if rec:
            r["consensus_score"] = rec.get("consensus_score", 0)
            r["consensus_tier"]  = rec.get("tier_label", "—")
            r["scan_count"]      = rec.get("scan_count", 0)
        else:
            # Not (yet) in any other scan — credit just this row
            r.setdefault("consensus_score", 0)
            r.setdefault("consensus_tier",  "—")
            r.setdefault("scan_count",      1)

        # ── Stage transition stamp ──
        if get_stage_info is not None:
            try:
                s = get_stage_info(sym)
                if s.get("stage") is not None:
                    r["stage_days_in"] = s.get("days_in", 0)
                    r["stage_fresh"]   = bool(s.get("fresh"))
                    r["stage_since"]   = s.get("since_date")
                    if s.get("fresh"):
                        r["stage_badge"] = f"S{s['stage']} fresh {s.get('days_in', 0)}d"
                    else:
                        r["stage_badge"] = f"S{s['stage']} {s.get('days_in', 0)}d"
            except Exception:
                pass

    return results
