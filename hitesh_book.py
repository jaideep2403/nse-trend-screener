"""
Hitesh Modi (@imhiteshmodi) — PUBLICLY DISCLOSED momentum book.

This is a hand-curated snapshot of what Hitesh has *publicly posted* on X — his
weekly "Investing Journey" update (Part 40, 30 Aug 2026) and daily portfolio
updates. It is NOT his complete book: he discloses his top-5 winners, his laggards
("draggers"), and his daily top movers — so this captures ~15 confirmed names of a
~30-name portfolio. Avg-cost / qty come from the broker screenshots he posted
(invested amounts were blurred; qty×avg reconstructs cost).

IMPORTANT — this is a RESEARCH / TRACKING aid, not advice:
  • Partial: only what he chose to disclose. Names he holds but didn't post are absent.
  • Point-in-time: as of the dates below. He rebalances weekly and may have exited
    since (e.g. IVALUE was flagged "exit on Monday" in his 30 Aug post).
  • Copying a disclosed book always LAGS his real trades and carries the usual
    momentum risks (sharp drawdowns, gap-downs). Do your own diligence.

To refresh: read his latest weekly update + daily posts and update DISCLOSED below.
"""
from __future__ import annotations

# Source posts (public):
SOURCES = {
    "weekly": "https://x.com/imhiteshmodi/status/2093901770716283054",   # Part 40, 30 Aug 2026
    "daily":  "https://x.com/imhiteshmodi/status/2095567023044759863",   # 3 Sept 2026
    "scanner": "https://chartink.com/screener/mi50-originalkl-scan",
}
AS_OF = "2026-08-30"          # weekly snapshot date (winners/laggards + avg/qty)
DAILY_AS_OF = "2026-09-03"    # daily movers snapshot date

# bucket: multibagger | dragger | mover
# avg   = his average cost (₹), from broker screenshot; qty from same
# gain_at_post = % gain he showed at snapshot; note = anything he flagged
DISCLOSED = [
    # ── "THE MULTIBAGGERS" (30 Aug weekly) — winners he's riding ──────────────
    {"symbol": "STLTECH",    "bucket": "multibagger", "avg": 159.54,  "qty": 401, "gain_at_post": 353.64},
    {"symbol": "BLISSGVS",   "bucket": "multibagger", "avg": 186.69,  "qty": 466, "gain_at_post": 215.00},
    {"symbol": "INDSWFTLAB", "bucket": "multibagger", "avg": 117.09,  "qty": 742, "gain_at_post": 209.94},
    {"symbol": "AEROFLEX",   "bucket": "multibagger", "avg": 238.59,  "qty": 423, "gain_at_post": 127.20},
    {"symbol": "SANSERA",    "bucket": "multibagger", "avg": 1776.81, "qty": 49,  "gain_at_post": 117.74},
    # ── "THE DRAGGERS" (30 Aug weekly) — laggards still held ──────────────────
    {"symbol": "IVALUE",     "bucket": "dragger", "avg": 311.43,  "gain_at_post": -15.02,
     "note": "He flagged EXIT (to be taken Monday) in the 30 Aug post"},
    {"symbol": "TARSONS",    "bucket": "dragger", "avg": 359.50,  "gain_at_post": -7.58},
    {"symbol": "IDEAFORGE",  "bucket": "dragger", "avg": 864.57,  "gain_at_post": -6.58},
    {"symbol": "TATVA",      "bucket": "dragger", "avg": 1720.00, "gain_at_post": -5.23},
    {"symbol": "ACE",        "bucket": "dragger", "avg": 1187.04, "gain_at_post": -3.91},
    # ── Daily "TOP MOVERS" (3 Sept) — confirmed holdings, avg not disclosed ────
    {"symbol": "RAYMOND",  "bucket": "mover", "note": "Top mover +13.6% on 3 Sep"},
    {"symbol": "DEEPINDS", "bucket": "mover", "note": "Top mover +7.4% on 3 Sep"},
    {"symbol": "KMEW",     "bucket": "mover", "note": "Top mover +7.4% on 3 Sep"},
    {"symbol": "WHEELS",   "bucket": "mover", "note": "Top mover +7.1% on 3 Sep"},
    {"symbol": "CENTUM",   "bucket": "mover", "note": "Top mover +6.2% on 3 Sep"},
]

# Names he *highlighted* as weekly favourites from the scanner but that aren't
# confirmed as held — shown separately as "watch / candidates", never as holdings.
RECENT_PICKS = ["SYNCOMF", "WINDLAS", "VERANDA", "SENCO", "JYOTICN", "RKFORGING",
                "MSTC", "RAMRATNA", "MAHASEML"]

# His own posted headline numbers (self-reported, unaudited):
HIS_NUMBERS = {
    "since": "2022-07-01",
    "portfolio_multiple": 3.23,        # +223% since Jul 2022
    "index_multiple": 2.28,            # Nifty Smallcap 250 over same span (+128%)
    "one_year_portfolio_pct": 47.0,
    "one_year_index_pct": 12.1,
    "as_of": "2026-08-30",
}


def get_book() -> dict:
    """Enrich the disclosed book with our live prices and cross-reference against
    OUR momentum scanner (is the name currently a fresh high / above its 20-wk MA?)."""
    import shared_universe as su
    try:
        import sector_mapper as sm
        smap = sm.get_enriched_sector_map()
    except Exception:
        smap = {}

    U = su.load_base_universe(days=400, include_stale=True)

    # Which of his names are LIVE in our rotation (held/entry) right now?
    our_syms: set = set()
    try:
        import momentum_rotation as mr
        live = mr.run_live_scan()
        for k in ("holdings", "entries"):
            for r in live.get(k, []) or []:
                if r.get("symbol"):
                    our_syms.add(r["symbol"])
    except Exception:
        pass

    import pandas as pd

    def _px(sym):
        df = U.get(sym)
        if df is None or not len(df):
            return None
        return round(float(df["Close"].iloc[-1]), 2)

    def _ma20w(sym):
        """20-week MA on weekly close — his exact exit yardstick."""
        df = U.get(sym)
        if df is None or len(df) < 100:
            return None
        try:
            wk = df["Close"].resample("W-FRI").last().dropna()
            if len(wk) < 20:
                return None
            return round(float(wk.rolling(20).mean().iloc[-1]), 2)
        except Exception:
            return None

    rows = []
    for h in DISCLOSED:
        sym = h["symbol"]
        px = _px(sym)
        avg = h.get("avg")
        ma20 = _ma20w(sym)
        cur_gain = round((px / avg - 1) * 100, 1) if px and avg else None
        invested = round(avg * h["qty"]) if avg and h.get("qty") else None
        rows.append({
            "symbol": sym,
            "sector": smap.get(sym),
            "bucket": h["bucket"],
            "avg": avg,
            "qty": h.get("qty"),
            "invested": invested,
            "price": px,
            "gain_at_post": h.get("gain_at_post"),
            "gain_now": cur_gain,                       # from HIS avg to our latest px
            "ma20w": ma20,
            "above_ma20w": (px >= ma20) if (px and ma20) else None,   # his hold/exit test
            "in_our_scanner": sym in our_syms,
            "note": h.get("note"),
        })

    # order: multibaggers (by gain desc), movers, draggers (by gain asc)
    order = {"multibagger": 0, "mover": 1, "dragger": 2}
    rows.sort(key=lambda r: (order.get(r["bucket"], 9),
                             -(r["gain_now"] if r["gain_now"] is not None else -999)
                             if r["bucket"] != "dragger"
                             else (r["gain_now"] if r["gain_now"] is not None else 999)))

    picks = []
    for sym in RECENT_PICKS:
        px = _px(sym)
        if px is not None:
            picks.append({"symbol": sym, "sector": smap.get(sym), "price": px,
                          "in_our_scanner": sym in our_syms})

    return {
        "as_of": AS_OF,
        "daily_as_of": DAILY_AS_OF,
        "sources": SOURCES,
        "his_numbers": HIS_NUMBERS,
        "holdings": rows,
        "recent_picks": picks,
        "n_disclosed": len(rows),
        "disclaimer": ("Publicly disclosed, partial (~15 of ~30 names), point-in-time. "
                       "Research/tracking only — not investment advice."),
    }
