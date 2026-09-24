"""
Sector Money-Flow analyser — find the sector the smart money is rotating INTO before the
move shows up in price. Mirrors the Value Picker "My Index" idea but with a transparent,
backtest-validated score and point-in-time (calendar) history from our own bhavcopy.

Money flow = traded turnover (₹ = Close × Volume). The tell is TURNOVER ACCELERATION:
recent (5-day) turnover running hot vs its own 22-day baseline means capital is rushing
in. When that happens while price is only starting to move, it's early accumulation.

Confirmed to match Value Picker's aggregation exactly (from a 20-Apr-2026 screenshot):
  • sector return / RSI / close   = simple MEAN across member stocks
  • % Amt Chg                     = (mean 5D turnover) / (mean 22D turnover) − 1
  • 5D+                            = count of members with a positive 5D return
Everything is point-in-time: pass a date and every metric is computed using ONLY bars up
to that date, so you can replay any day and see what the board looked like BEFORE the run.
"""
from __future__ import annotations

import time as _time
import numpy as np
import pandas as pd

import result_cache as _rc

MIN_MEMBERS   = 3       # a sector needs at least this many charted names to score
MIN_BARS      = 67      # need 66 sessions for the 66D return + 22D turnover baseline
MIN_TURN_CR   = 0.05    # drop essentially non-trading names (< ₹5 lakh/day)
STOCK_LIQ_FLOOR_CR = 3.0  # flat Stocks list only: min 5-day avg turnover (₹ Cr) — real money
CACHE_TTL     = 900

# Curated CROSS-CUTTING THEMES — baskets that span several screener industries, so the
# money rotating into a hot theme shows up as its own row (like Value Picker's custom
# indices). A stock keeps its normal industry row AND appears in any theme it belongs to.
# Edit these lists to add/adjust themes; they're deliberately hand-picked, not scraped.
THEMES: dict[str, list[str]] = {
    "Data Center": ["E2E", "NETWEB", "ANANTRAJ", "RAILTEL", "TATACOMM", "STLTECH",
                    "HFCL", "TEJASNET", "DLINKINDIA", "CYIENTDLM"],
    "EV (Electric Vehicles)": ["OLECTRA", "M&M", "TVSMOTOR", "EXIDEIND", "SONACOMS",
                    "UNOMINDA", "GREAVESCOT", "JBMA", "TATAPOWER", "HBLENGINE", "ARE&M",
                    "MOTHERSON", "MINDACORP"],
    "Defence": ["HAL", "BEL", "BDL", "MAZDOCK", "COCHINSHIP", "GRSE", "MTARTECH", "PARAS",
                    "DATAPATTNS", "ZENTEC", "BEML", "IDEAFORGE", "DCXINDIA", "ASTRAMICRO",
                    "SOLARINDS"],
    "Railways": ["IRCTC", "IRFC", "RVNL", "IRCON", "RAILTEL", "TITAGARH", "JWL", "TEXRAIL",
                    "RITES", "CONCOR", "BEML"],
    "Solar & Renewables": ["WAAREEENER", "PREMIERENE", "ADANIGREEN", "SUZLON", "INOXWIND",
                    "KPIGREEN", "SWSOLAR", "GENUSPOWER", "WAAREERTL", "WEBELSOLAR"],
    "Semiconductor & EMS": ["KAYNES", "SYRMA", "DIXON", "AMBER", "CGPOWER", "NETWEB",
                    "MOSCHIP", "CYIENTDLM", "AVALON", "ELIN", "PGEL"],
    "New-age Internet / Quick Commerce": ["ETERNAL", "SWIGGY", "PAYTM", "POLICYBZR",
                    "NYKAA", "DELHIVERY", "NAZARA", "HONASA", "CARTRADE", "TBOTEK",
                    "IXIGO", "ZAGGLE"],
    "Fintech": ["PAYTM", "POLICYBZR", "ANGELONE", "CDSL", "CAMS", "KFINTECH", "BSE",
                    "MCX", "ZAGGLE", "360ONE", "NUVAMA"],
    "Drones": ["IDEAFORGE", "ZENTEC", "PARAS", "DCXINDIA", "RTNINDIA"],
    "PSU": ["SBIN", "BEL", "HAL", "NTPC", "ONGC", "COALINDIA", "IOC", "BPCL", "PFC",
                    "RECLTD", "IRFC", "GAIL", "POWERGRID", "NHPC", "SJVN", "BEML", "MAZDOCK",
                    "CONCOR", "NBCC", "RITES", "IRCON", "RVNL", "NMDC", "SAIL", "BHEL",
                    "HUDCO", "MOIL", "NLCINDIA"],
    "Hotels & Tourism": ["INDHOTEL", "EIHOTEL", "CHALET", "LEMONTREE", "ITCHOTELS",
                    "TAJGVK", "ORIENTHOT", "SAMHI", "MHRIL", "WONDERLA", "EASEMYTRIP",
                    "THOMASCOOK"],
    "Agri & Fertilizers": ["PIIND", "UPL", "RALLIS", "BAYERCROP", "SUMICHEM", "DHANUKA",
                    "INSECTICID", "COROMANDEL", "CHAMBLFERT", "DEEPAKFERT", "GSFC", "GNFC",
                    "RCF", "FACT", "PARADEEP", "BHARATRAS"],
    "Water": ["WABAG", "IONEXCHANG", "EMSLIMITED", "THERMAX", "PRAJIND", "WELENT"],
    "Transformers & Power Equip": ["TARIL", "VOLTAMP", "INDOTECH", "POWERINDIA",
                    "TDPOWERSYS", "APARINDS"],
}

_cache: dict = {"key": None, "data": None, "ts": 0.0}


def _rsi(c: np.ndarray, n: int = 14) -> float:
    if len(c) < n + 1:
        return float("nan")
    d = np.diff(c[-(n + 1):])
    up = d[d > 0].sum() / n
    dn = -d[d < 0].sum() / n
    if dn == 0:
        return 100.0
    rs = up / dn
    return float(100.0 - 100.0 / (1.0 + rs))


def _ema_last(c: np.ndarray, span: int = 20) -> float:
    if len(c) < span:
        return float("nan")
    return float(pd.Series(c).ewm(span=span, adjust=False).mean().iloc[-1])


def _entry_signal(amt_chg: float, dist_ema20: float | None, rsi: float | None,
                  ret5: float) -> tuple[float, str]:
    """Rate a single stock as a BUY *right now*, given that money is entering its sector.

    The drill-down used to rank members by score5 (return × surge), which floats the name
    that has ALREADY run to the top — the worst thing to chase. This instead rewards the
    stock where the money is arriving but price is still EARLY: hugging its 20-EMA, RSI not
    yet overbought, and the 5-day move not already spent. That's the name you actually buy
    when a sector lights up green. Returns (0-100 entry score, state) where state is
    'fresh' (money in, still early), 'extended' (already ran — chasing), or 'neutral'."""
    # money-in component (0-40): how hard turnover is accelerating into the name
    money = max(0.0, min(1.0, amt_chg / 60.0)) * 40.0
    # room-to-run by distance from the 20-EMA — prime entry hugs or sits just above it
    if dist_ema20 is None:      room = 17.0
    elif dist_ema20 <= -6:      room = 12.0     # below the EMA — not confirmed / rolling over
    elif dist_ema20 <= 6:       room = 35.0     # AT the 20-EMA: the entry
    elif dist_ema20 <= 12:      room = 22.0
    elif dist_ema20 <= 20:      room = 10.0
    else:                       room = 3.0      # far extended above trend
    # RSI room (0-25): 45-63 is the sweet spot; overbought is punished
    if rsi is None:             rsi_room = 12.0
    elif rsi < 40:              rsi_room = 10.0
    elif rsi <= 63:             rsi_room = 25.0
    elif rsi <= 70:             rsi_room = 15.0
    elif rsi <= 76:             rsi_room = 7.0
    else:                       rsi_room = 2.0
    score = round(money + room + rsi_room, 1)
    extended = ((ret5 >= 18)
                or (rsi is not None and rsi >= 74)
                or (dist_ema20 is not None and dist_ema20 >= 16))
    # fresh = money in AND price CONFIRMING (at/just above the 20-EMA, not below it and not
    # yet extended). Below the EMA is a falling knife, not an entry — excluded.
    fresh = (amt_chg >= 8 and not extended
             and (dist_ema20 is None or -4 <= dist_ema20 <= 9)
             and (rsi is None or rsi <= 68))
    state = "fresh" if fresh else ("extended" if extended else "neutral")
    return score, state


def _stock_metrics(df: pd.DataFrame, sym: str, as_of: pd.Timestamp | None,
                   mcap: float | None) -> dict | None:
    c = df["Close"]
    v = df["Volume"]
    if as_of is not None:
        m = c.index <= as_of
        c = c[m]; v = v[m]
    if len(c) < MIN_BARS:
        return None
    cv = c.to_numpy(dtype=float)
    vv = v.to_numpy(dtype=float)
    turn = cv * vv / 1e7                      # ₹ Cr daily turnover
    turn5 = float(turn[-5:].mean())
    turn22 = float(turn[-22:].mean())
    if turn22 < MIN_TURN_CR:
        return None
    close = cv[-1]
    daily = (cv[-1] / cv[-2] - 1) * 100 if cv[-2] > 0 else 0.0
    ret5  = (cv[-1] / cv[-6] - 1) * 100 if cv[-6] > 0 else 0.0
    ret22 = (cv[-1] / cv[-23] - 1) * 100 if cv[-23] > 0 else 0.0
    ret66 = (cv[-1] / cv[-67] - 1) * 100 if cv[-67] > 0 else 0.0
    # return of the 22 sessions BEFORE the last 22 — lets the sector layer measure whether
    # the trend is ACCELERATING (recent 22d > prior 22d) or decelerating.
    prev22 = (cv[-23] / cv[-45] - 1) * 100 if len(cv) >= 45 and cv[-45] > 0 else 0.0
    # trend participation: is the name above its own 50-DMA (in an uptrend)?
    above_ma50 = bool(len(cv) >= 50 and cv[-1] > float(cv[-50:].mean()))
    amt_chg = (turn5 / turn22 - 1) * 100 if turn22 > 0 else 0.0
    surge = (turn5 / turn22) if turn22 > 0 else 1.0
    rsi = _rsi(cv, 14)
    ema20 = _ema_last(cv, 20)
    dist = (close / ema20 - 1) * 100 if ema20 and not np.isnan(ema20) else None
    # Flow score = return AMPLIFIED by money-flow acceleration. Positive return with money
    # rushing in scores high; a move on fading turnover does not. (Transparent — not a
    # copy of Value Picker's proprietary score.)
    score5  = ret5 * surge
    score22 = ret22 * surge
    entry_score, entry_state = _entry_signal(amt_chg, dist, rsi if not np.isnan(rsi) else None, ret5)
    # stock-level money-flow colour (same rule as the sector layer): green = money in AND
    # price already responding, yellow = money in but price hasn't moved yet (the early tell).
    _money_in = amt_chg >= 8
    flow_state = ("green" if (ret22 >= 5 or ret5 >= 3) else "yellow") if _money_in else None
    return {
        "symbol": sym, "close": round(close, 2), "daily_pct": round(daily, 2),
        "ret5_pct": round(ret5, 2), "ret22_pct": round(ret22, 2), "ret66_pct": round(ret66, 2),
        "prev22_pct": round(prev22, 2), "above_ma50": above_ma50,
        "turn5_cr": round(turn5, 2), "turn22_cr": round(turn22, 2), "amt_chg_pct": round(amt_chg, 2),
        "score5": round(score5, 2), "score22": round(score22, 2),
        "entry_score": entry_score, "entry_state": entry_state, "flow_state": flow_state,
        "rsi": round(rsi, 1) if not np.isnan(rsi) else None,
        "ema20": round(ema20, 2) if ema20 and not np.isnan(ema20) else None,
        "dist_ema20_pct": round(dist, 2) if dist is not None else None,
        "mcap_cr": round(float(mcap)) if mcap else None,
        "monster": False, "monster_score": None,   # set in compute() from the Monster Radar cache
    }


def _aggregate(sector: str, rows: list[dict]) -> dict:
    n = len(rows)
    def mean(k):
        xs = [r[k] for r in rows if r.get(k) is not None]
        return round(float(np.mean(xs)), 2) if xs else None
    turn5 = mean("turn5_cr") or 0.0
    turn22 = mean("turn22_cr") or 0.0
    amt_chg = round((turn5 / turn22 - 1) * 100, 2) if turn22 else 0.0
    ret5 = mean("ret5_pct") or 0.0
    ret22 = mean("ret22_pct") or 0.0
    score5 = mean("score5")
    # EXHAUSTION flag — backtest showed the extreme money-flow bucket (huge turnover
    # surge) has ~zero forward return: the news is out and everyone is piling in. Flag a
    # sector where money is flooding in (amt_chg very high) but price is already extended
    # or rolling over — that's late/distribution, not early accumulation.
    exhaustion = bool(amt_chg > 55 and (ret5 <= 0 or ret22 > 25))
    n_up = sum(1 for r in rows if (r.get("ret5_pct") or 0) > 0)
    rsi = mean("rsi")
    broad = (n_up / n) >= 0.5 if n else False
    # Colour state for the whole sector row — driven by the MONEY, so it lights up on any
    # date (even weak-breadth days), which is the whole point of a money-flow board:
    #   GREEN  — money has clearly flowed IN and price is already responding (up).
    #   YELLOW — money has entered but price HASN'T MOVED yet (the early, pre-move tell).
    # Exhaustion blow-offs (extreme surge + fading price) stay UNcoloured and flagged ⚠,
    # since the backtest showed those go nowhere.
    money_in = bool(amt_chg >= 8 and not exhaustion)
    if money_in:
        flow_state = "green" if (ret22 >= 5 or ret5 >= 3) else "yellow"
    else:
        flow_state = None

    # ── SECTOR TREND PHASE (the ValuePicker "catch-the-sector-early" method) ──────────
    # Identify the broader sectoral trend and whether it is STRENGTHENING or WEAKENING, so
    # you enter early and stay while the data supports it. Built from PARTICIPATION (share
    # of the sector above its 50-DMA — the trend's breadth), ACCELERATION (recent 22d vs
    # the prior 22d), momentum and money direction.
    ret66 = mean("ret66_pct") or 0.0
    prev22 = mean("prev22_pct") or 0.0
    accel = round(ret22 - prev22, 2)
    part = round(float(np.mean([1.0 if r.get("above_ma50") else 0.0 for r in rows])), 2) if rows else 0.0
    _clip = lambda x: max(0.0, min(1.0, x))
    trend_strength = int(round(100 * (0.55 * part + 0.25 * _clip((rsi or 50) / 70.0)
                                      + 0.20 * _clip((ret22 / 20.0) + 0.5))))
    if part >= 0.55 and ret66 >= 30 and (ret5 <= 0 or amt_chg < -8):
        trend_phase, trend_dir = "Mature", "warn"          # extended & starting to roll over
    elif part <= 0.38 or ret22 <= -4 or (accel <= -6 and amt_chg < 0):
        trend_phase, trend_dir = "Weakening", "down"
    elif prev22 <= 2 and ret22 >= 4 and accel >= 2 and part >= 0.45:
        trend_phase, trend_dir = "Emerging", "up"          # trend just turned up → ENTER EARLY
    elif ret22 > 0 and accel >= -1 and part >= 0.5:
        trend_phase, trend_dir = "Strengthening", "up"
    else:
        trend_phase, trend_dir = "Steady", "flat"

    n_monster = sum(1 for r in rows if r.get("monster"))
    return {
        "sector": sector, "n_stocks": n, "n_up_5d": n_up, "n_monster": n_monster,
        "score5": score5, "score22": mean("score22"), "flow_score": score5,
        "avg_close": mean("close"), "avg_rsi": rsi,
        "turn5_cr": turn5, "turn22_cr": turn22, "amt_chg_pct": amt_chg,
        "ret1_pct": mean("daily_pct"), "ret5_pct": ret5,
        "ret22_pct": ret22, "ret66_pct": ret66,
        "exhaustion": exhaustion, "flow_state": flow_state,
        # trend-phase engine
        "participation": part, "accel_pct": accel,
        "trend_strength": trend_strength, "trend_phase": trend_phase, "trend_dir": trend_dir,
    }


def _mark_leaders(rows: list, detail: dict, U: dict, as_of_ts, k: int = 3,
                  min_amt: float = 15.0, min_breadth: float = 0.45, min_n: int = 8):
    """Flag the ≤k strongest money-flow LEADERS and strip the colour from everyone else,
    so the board points at the 1-3 sectors to act on instead of a wall of yellow.

    Ranked by CONVICTION, not raw inflow: a sector where price is already confirming the
    money (green) and where the money is BUILDING over the week beats a bigger one-day
    yellow SPIKE. That keeps the price-confirmed leaders (green) at the top where they
    belong. Themes are hand-sized baskets, so they get a lower size floor."""
    n_floor = min(min_n, 5) if any(s.get("is_theme") for s in rows) else min_n
    cand = [s for s in rows
            if s.get("flow_state") and not s.get("exhaustion")
            and s.get("n_stocks", 0) >= n_floor
            and (s["n_up_5d"] / s["n_stocks"]) >= min_breadth
            and (s.get("amt_chg_pct") or 0) >= min_amt]
    for s in cand:                       # persistence for every candidate, before ranking
        s["flow_trend"] = _persistence([r["symbol"] for r in detail.get(s["sector"], [])],
                                       U, as_of_ts)

    def _conviction(s):
        c = float(s.get("amt_chg_pct") or 0)
        if s.get("flow_state") == "green":   c += 25      # price confirming the money
        if s.get("flow_trend") == "building": c += 30     # real accumulation, not a blip
        elif s.get("flow_trend") == "spike":  c -= 20
        return c

    cand.sort(key=_conviction, reverse=True)
    leaders = cand[:k]
    lead_ids = {id(s) for s in leaders}
    for s in rows:
        s["is_leader"] = id(s) in lead_ids
        if not s["is_leader"]:
            s["flow_state"] = None      # keep colour ONLY on the leaders
    return leaders


def _persistence(members: list, U: dict, as_of_ts) -> str | None:
    """Is a leader's inflow BUILDING over the week, or a one-day SPIKE? Sums the members'
    daily turnover and checks whether a single day dwarfs the rest (spike) or the money is
    spread / ramping across the last 5 sessions (building)."""
    ser = None
    for sym in members:
        df = U.get(sym)
        if df is None:
            continue
        c = df["Close"]; v = df["Volume"]
        if as_of_ts is not None:
            m = c.index <= as_of_ts
            c = c[m]; v = v[m]
        t = (c * v) / 1e7
        ser = t if ser is None else ser.add(t, fill_value=0)
    if ser is None or len(ser) < 6:
        return None
    last5 = ser.tail(5).to_numpy(dtype=float)
    if last5.sum() <= 0:
        return None
    mx = float(last5.max())
    others = np.sort(last5)[:-1]                      # the 4 non-peak days
    med_others = float(np.median(others)) if len(others) else 0.0
    spike_ratio = (mx / med_others) if med_others > 0 else 99.0
    return "spike" if spike_ratio >= 2.2 else "building"


def _monster_map() -> dict:
    """{symbol: Monster-Radar score} from the LAST computed Monster scan — cache-only, so
    the money-flow request never triggers a heavy synchronous scan. Empty until the
    scheduler has warmed the Monster cache at least once (then always available, even
    stale). The overlap is the highest-conviction tell: a name the money-flow board says
    is in a funded sector AND the Monster Radar says is a small-cap volume-build setup."""
    try:
        import result_cache
        data = result_cache.get_or_stale("monster_candidate")
        if not data:
            return {}
        return {r["symbol"]: r.get("score") for r in data.get("results", []) if r.get("symbol")}
    except Exception:
        return {}


def _pick_score(r: dict) -> float:
    """Ranking score for a BUY candidate: MONEY-CONFIRMED MOVE while still early.

    score5 = 5-day return × turnover-surge already fuses money strength (surge = 5D/22D
    turnover) with the price move underway (ret5), so it rewards a name where money is
    flooding in AND price is confirming — not a name with money pouring in but price still
    dead flat (which score5 keeps low). Scaled by how EARLY the entry still is (entry_score
    0-100, high near the 20-EMA with room on RSI), with a small Monster-Radar bonus. This
    is what lifts a real mover like NOVARTIND above bigger-but-flat inflows."""
    s5 = r.get("score5")
    es = r.get("entry_score")
    base = (float(s5) if s5 is not None else 0.0) * ((float(es) if es is not None else 0.0) / 100.0)
    return base * (1.15 if r.get("monster") else 1.0)


def _fresh_picks(members: list, k: int = 5) -> list:
    """The ≤k names to actually BUY from a funded sector, ranked by _pick_score (money +
    move + still-early), Monster-Radar overlaps lightly boosted. Falls back to the best
    non-extended names if none are cleanly 'fresh' yet."""
    fresh = [r for r in members if r.get("entry_state") == "fresh"]
    pool = fresh or [r for r in members if r.get("entry_state") != "extended"]
    pool.sort(key=_pick_score, reverse=True)
    return [{"symbol": r["symbol"], "close": r.get("close"),
             "entry_score": r.get("entry_score"), "entry_state": r.get("entry_state"),
             "amt_chg_pct": r.get("amt_chg_pct"), "dist_ema20_pct": r.get("dist_ema20_pct"),
             "rsi": r.get("rsi"), "ret5_pct": r.get("ret5_pct"), "score5": r.get("score5"),
             "pick_score": round(_pick_score(r), 2),
             "monster": bool(r.get("monster")), "monster_score": r.get("monster_score")}
            for r in pool[:k]]


def _entered_date(df, as_of_ts, thresh: float = 8.0) -> str | None:
    """The date this name ENTERED the money-flow scanner: the first day of its CURRENT
    money-in streak — i.e. walking back from the as-of bar while its % Amt Chg (5-day avg
    turnover ÷ 22-day avg turnover − 1) stays ≥ the money-in threshold. Tells you how long
    money has been flowing in (a run that started 2 days ago is fresh; one from 30 days ago
    has been accumulating a while). None if it isn't a money-in name on the as-of bar."""
    try:
        c = df["Close"]; v = df["Volume"]
        if as_of_ts is not None:
            m = c.index <= as_of_ts
            c = c[m]; v = v[m]
        if len(c) < 22:
            return None
        turn = (c.astype(float) * v.astype(float)) / 1e7
        t5 = turn.rolling(5).mean(); t22 = turn.rolling(22).mean()
        amt = (((t5 / t22) - 1.0) * 100.0).dropna()
        if amt.empty:
            return None
        vals = amt.to_numpy(); idx = amt.index
        i = len(vals) - 1
        if vals[i] < thresh:                 # not a money-in name on the as-of bar
            return None
        while i > 0 and vals[i - 1] >= thresh:
            i -= 1
        return str(idx[i].date())
    except Exception:
        return None


def compute(as_of: str | None = None) -> dict:
    import shared_universe as su
    try:
        import sector_mapper as sm
        smap = dict(sm.get_enriched_sector_map())
    except Exception:
        smap = {}
    # Fold in screener-derived classifications for the tail stocks the NSE Total Market
    # index doesn't cover (on-demand lookups + the bulk scrape). This is what lets Money
    # Flow span the WHOLE market instead of just the ~767 index names. Grows live as the
    # bulk scrape completes; NSE-index labels always win where both exist.
    for _src in ("bulk_sector_scrape", "sector_lookup"):
        try:
            _mod = __import__(_src)
            _extra = _mod.get_sector_map() if hasattr(_mod, "get_sector_map") else \
                     (_mod.cached_map() if hasattr(_mod, "cached_map") else {})
            for _k, _v in (_extra or {}).items():
                if _v and not smap.get(_k):
                    smap[_k] = _v
        except Exception:
            pass
    try:
        import sector_rotation as sr
        mcap = sr._mcap_map()
    except Exception:
        mcap = {}
    mmap = _monster_map()          # {symbol: Monster-Radar score} — cache-only, never blocks

    U = su.load_base_universe(days=400, include_stale=True)
    if not U:
        return {"as_of": as_of, "sectors": [], "detail": {}, "dates": []}

    as_of_ts = pd.Timestamp(as_of) if as_of else None

    by_sector: dict = {}
    for sym, sec in smap.items():
        if not sec or sec == "Other" or sym not in U:
            continue
        by_sector.setdefault(sec, []).append(sym)

    sectors, themes, detail = [], [], {}
    for sec, syms in by_sector.items():
        rows = []
        for sym in syms:
            m = _stock_metrics(U[sym], sym, as_of_ts, mcap.get(sym))
            if m:
                if sym in mmap:
                    m["monster"] = True
                    m["monster_score"] = mmap[sym]
                rows.append(m)
        if len(rows) < MIN_MEMBERS:
            continue
        rows.sort(key=lambda r: (r["score5"] if r["score5"] is not None else -9e9), reverse=True)
        detail[sec] = rows
        sectors.append(_aggregate(sec, rows))

    # Cross-cutting THEMES (Data Center, EV, …) — a SEPARATE, filterable sub-category, NOT
    # mixed into the sector board. Same metrics; members also live in their industry rows.
    for theme, syms in THEMES.items():
        rows = []
        for sym in syms:
            if sym in U:
                m = _stock_metrics(U[sym], sym, as_of_ts, mcap.get(sym))
                if m:
                    if sym in mmap:
                        m["monster"] = True
                        m["monster_score"] = mmap[sym]
                    rows.append(m)
        if len(rows) < MIN_MEMBERS:
            continue
        rows.sort(key=lambda r: (r["score5"] if r["score5"] is not None else -9e9), reverse=True)
        detail[theme] = rows
        agg = _aggregate(theme, rows)
        agg["is_theme"] = True
        themes.append(agg)

    # Attach the BUY-ranking score to EVERY stock in EVERY sector/theme (not just leaders),
    # so each drill-down ranks its names by money-confirmed move while still early — the
    # same signal the "Buy →" callout uses. This is what surfaces a mover like NOVARTIND to
    # the top of its sector instead of burying it by raw closeness-to-EMA.
    for _rows in detail.values():
        for r in _rows:
            r["pick_score"] = round(_pick_score(r), 3)

    # default rank: the backtest-validated Flow Score (return × turnover-surge). It was
    # monotonic in forward returns, unlike raw amt_chg which peaks in the mid-high bucket
    # and fades at the exhausted extreme. Exhaustion-flagged sectors sink to the bottom.
    _rank = lambda s: (-1 if s.get("exhaustion") else 0,
                       s["flow_score"] if s["flow_score"] is not None else -9e9)
    sectors.sort(key=_rank, reverse=True)
    themes.sort(key=_rank, reverse=True)

    # LEADERS — name the 1-3 sectors money is DECISIVELY entering, not paint 18 of them.
    # Ranked by conviction (price-confirmed 'green' + 📈 building beat a bigger 1-day
    # yellow spike). Only leaders keep their colour; everyone else goes plain.
    lead_secs = _mark_leaders(sectors, detail, U, as_of_ts)
    lead_thms = _mark_leaders(themes, detail, U, as_of_ts)

    # Close the loop: for every LEADER sector/theme, name the ≤3 stocks to actually BUY —
    # money entering + still early (fresh), Monster-Radar overlaps first. This turns "right
    # sector" into "buy these names" without opening the drill-down.
    for s in lead_secs + lead_thms:
        _rows = detail.get(s["sector"], [])
        s["fresh_picks"] = _fresh_picks(_rows)
        s["n_fresh"] = sum(1 for r in _rows if r.get("entry_state") == "fresh")
    n_monster_total = sum(1 for row in detail.values() for r in row if r.get("monster"))

    # the actual as-of date used (latest bar ≤ requested)
    used = None
    try:
        any_df = next(iter(U.values()))
        idx = any_df.index if as_of_ts is None else any_df.index[any_df.index <= as_of_ts]
        used = str(idx[-1].date()) if len(idx) else as_of
    except Exception:
        used = as_of

    # ── FLAT STOCK-LEVEL money flow across the WHOLE universe (the Stocks toggle) ─────────
    # "Which individual stocks is money flowing INTO" — every one of the ~2,300 names, not
    # just the sector means. Reuse the metrics already computed for the sector/theme rows
    # (deduped by symbol), compute the tail that belongs to no scored sector, drop names
    # whose last bar is stale, then rank by money inflow (% Amt Chg) and keep the top slice.
    board_ts = pd.Timestamp(used) if used else None
    seen: dict = {}
    for _rows in detail.values():
        for r in _rows:
            seen.setdefault(r["symbol"], r)
    stocks: list = []
    for sym, df in U.items():
        idx = df.index if as_of_ts is None else df.index[df.index <= as_of_ts]
        if len(idx) == 0 or (board_ts is not None and (board_ts - idx[-1]).days > 12):
            continue                                  # no bar yet / stale — skip
        m = seen.get(sym)
        if m is None:
            m = _stock_metrics(df, sym, as_of_ts, mcap.get(sym))
            if not m:
                continue
            if sym in mmap:
                m["monster"] = True
                m["monster_score"] = mmap[sym]
        # LIQUIDITY floor for the flat list: real recent money, not a ₹5-lakh micro-cap that
        # popped once. Gated on the 5-day (RECENT) turnover so a name that was dead and just
        # woke up still qualifies — that's exactly the early money-in signal we want.
        if (m.get("turn5_cr") or 0) < STOCK_LIQ_FLOOR_CR:
            continue
        stocks.append({
            "symbol": sym, "sector": smap.get(sym) or "Other",
            "close": m["close"], "amt_chg_pct": m["amt_chg_pct"],
            "turn5_cr": m["turn5_cr"], "turn22_cr": m["turn22_cr"],
            "ret1_pct": m["daily_pct"], "ret5_pct": m["ret5_pct"],
            "ret22_pct": m["ret22_pct"], "ret66_pct": m["ret66_pct"],
            "rsi": m["rsi"], "dist_ema20_pct": m["dist_ema20_pct"], "mcap_cr": m["mcap_cr"],
            "score5": m["score5"], "entry_score": m["entry_score"], "entry_state": m["entry_state"],
            "flow_state": m.get("flow_state"),
            "monster": bool(m.get("monster")), "monster_score": m.get("monster_score"),
        })
    # rank by money flowing IN (turnover acceleration), keep the top slice for the client
    stocks.sort(key=lambda r: (r["amt_chg_pct"] if r["amt_chg_pct"] is not None else -9e9), reverse=True)
    stocks = stocks[:300]
    # date each name ENTERED the scanner (start of its current money-in streak) — computed
    # only for the top slice we actually return, so it stays cheap.
    for r in stocks:
        _df = U.get(r["symbol"])
        r["entered"] = _entered_date(_df, as_of_ts) if _df is not None else None

    return {"as_of": used, "requested": as_of, "sectors": sectors, "themes": themes,
            "stocks": stocks, "n_universe": len(U),
            "detail": detail, "computed_at": int(_time.time()),
            "monster_overlap": {"available": bool(mmap), "n_on_radar": len(mmap),
                                "n_in_flow": n_monster_total},
            # backtest summary (2,220 date×sector obs, forward 22D sector return):
            "validation": {
                "obs": 2220,
                "note": ("Rising money flow works, but the EXTREME top is exhaustion. "
                         "Forward 22D sector return by turnover-surge quintile: fading "
                         "−1.3% · building +1.4% to +1.6% (the sweet spot) · blow-off ~0%. "
                         "So chase moderate-and-rising flow, avoid faders and climax spikes."),
                "by_flow": {"fading": -1.33, "building": 1.56, "blowoff": -0.08},
            }}


def run(as_of: str | None = None, force: bool = False) -> dict:
    tag = None
    try:
        from data_fetcher import _latest_bhavcopy_date
        d = _latest_bhavcopy_date()
        tag = d.isoformat() if d else "nodate"
    except Exception:
        tag = "nodate"
    key = (tag, as_of or "latest")
    if (not force and _cache["data"] is not None and _cache["key"] == key
            and _time.time() - _cache["ts"] < CACHE_TTL):
        return _cache["data"]
    # only the "latest" view is disk-cached (historical replays are ad-hoc)
    if not force and as_of is None:
        disk = _rc.get_or_stale("sector_money_flow")
        if disk is not None:
            _cache.update(key=key, data=disk, ts=_time.time())
            return disk
    data = compute(as_of)
    _cache.update(key=key, data=data, ts=_time.time())
    if as_of is None:
        try:
            _rc.put("sector_money_flow", data)
        except Exception:
            pass
    return data


def invalidate_cache() -> None:
    _cache.update(key=None, data=None, ts=0.0)
