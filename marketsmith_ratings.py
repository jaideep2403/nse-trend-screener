"""
MarketSmith-style rating stack (IBD / CAN SLIM lineage), computed on our data.

WHAT THIS IS — AND ISN'T
────────────────────────
MarketSmith India publishes what each rating MEASURES but explicitly does NOT
publish the weightings ("specific weighting formulas … are not disclosed").
So this is a faithful implementation of the documented CONCEPTS, not a clone of
their numbers. Ours will rank similarly but will not tie out digit-for-digit,
and anywhere I had to choose a weight I've used the classic IBD convention and
said so in the code.

THE STACK (their definitions → our implementation)
──────────────────────────────────────────────────
  Price Strength (RS)  1-99  12-month price performance vs the whole universe.
                             IBD's classic weighting is quarter-weighted with a
                             double weight on the most recent quarter:
                                 2×Q1 + Q2 + Q3 + Q4   (Q = 63-bar returns)
                             then percentile-ranked cross-sectionally.
  EPS Rating           1-99  2 most recent quarters' EPS growth + 3-5yr annual
                             growth, percentile-ranked. 50/50 blend here.
  SMR Rating           1-99  Sales growth + profit Margins + ROE, equal-weighted
                             percentile ranks. Degrades gracefully if margins
                             aren't scraped yet (uses the legs it has).
  A/D (Buyer Demand)   A-E   13-week institutional accumulation. We already have
                             a faithful Chaikin-CLV implementation in
                             breakout_scanner._buyer_demand — reused, not
                             reinvented, so the two tabs can never disagree.
  Group Rank           1-N   Industry group ranked on 6-MONTH price performance
                             (MarketSmith uses 6M over 197 groups; we use 6M over
                             our sector map).
  Composite Rating     1-99  EPS + Price Strength + Buyer Demand + SMR.
  Master Score         A-E   EPS + Price Strength + Buyer Demand + GROUP RANK
                             (note: Group Rank replaces SMR vs Composite).
  CAN SLIM checklist   P/F   Per-criterion pass/fail.

Their published "leading stock" screen, for reference:
  EPS≥80 · PriceStr≥80 · BuyerDemand A/B · SMR≥80 · Composite≥90 · top Group
  · Master A/B, with MCap≥₹2,000cr and avg turnover≥₹5cr.

HONEST NOTE: these are SCREENING/presentation ratings from a growth-momentum
(CAN SLIM) philosophy. They are NOT validated on our walk-forward yardstick and
are deliberately NOT wired into the All-Weather engine.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# Their published thresholds — used for the CAN SLIM checklist and UI colouring.
TH_EPS, TH_RS, TH_SMR, TH_COMPOSITE = 80, 80, 80, 90
TH_GROUP_TOP = 40          # "top 40 groups"
GRADES = ["A", "B", "C", "D", "E"]


# ── helpers ──────────────────────────────────────────────────────────────────
def _pctile_1_99(s: pd.Series) -> pd.Series:
    """Percentile-rank a series onto MarketSmith's 1-99 scale (99 = best)."""
    r = s.rank(pct=True, na_option="keep")
    return (r * 98 + 1).round(0)


def _grade_from_pct(r: float) -> str:
    """A-E from a 0-1 percentile (A = top 20%), MarketSmith's relative grading."""
    if r is None or not np.isfinite(r):
        return "C"
    for i, cut in enumerate((0.80, 0.60, 0.40, 0.20)):
        if r >= cut:
            return GRADES[i]
    return "E"


def _ret(c: np.ndarray, bars: int) -> float | None:
    if c is None or len(c) < bars + 1:
        return None
    prev = c[-bars - 1]
    if not np.isfinite(prev) or prev <= 0:
        return None
    return float(c[-1] / prev - 1.0)


# ── 1. Price Strength (true 12-month weighted RS) ────────────────────────────
def weighted_rs_raw(close: pd.Series) -> float | None:
    """IBD's classic quarter-weighted 12-month RS, double-weighting the most
    recent quarter:  2×Q1 + Q2 + Q3 + Q4  (each Q a 63-bar return).

    This is the fix for our old 3-month-only Price Strength: a stock that ripped
    3 months ago but has since stalled used to score the same as one still
    leading, because only the last quarter was measured.
    """
    c = close.dropna().to_numpy(dtype=float)
    if len(c) < 252:
        return None
    q1 = _ret(c, 63)                       # most recent quarter
    q2 = _ret(c, 126)
    q3 = _ret(c, 189)
    q4 = _ret(c, 252)
    if None in (q1, q2, q3, q4):
        return None
    return 2.0 * q1 + q2 + q3 + q4


def price_strength(stocks: dict) -> dict:
    raw = {}
    for sym, df in stocks.items():
        if "Close" not in df:
            continue
        v = weighted_rs_raw(df["Close"])
        if v is not None:
            raw[sym] = v
    if not raw:
        return {}
    return _pctile_1_99(pd.Series(raw)).astype(int).to_dict()


# ── 2. EPS Rating ────────────────────────────────────────────────────────────
def eps_rating(funds: dict) -> dict:
    """50% recent-quarters EPS growth + 50% 3yr CAGR, percentile-ranked 1-99.

    Recent-quarter growth uses the YoY pair we already store (q1 vs q5, q2 vs q6)
    — YoY rather than sequential, so seasonal businesses aren't penalised.
    """
    rows = []
    for sym, f in funds.items():
        q1, q2 = f.get("eps_q1"), f.get("eps_q2")
        q5, q6 = f.get("eps_q5"), f.get("eps_q6")
        yoy = []
        for a, b in ((q1, q5), (q2, q6)):
            if a is not None and b is not None and b != 0:
                yoy.append((a - b) / abs(b) * 100.0)
        recent = float(np.mean(yoy)) if yoy else None
        longt = f.get("growth_3y_cagr")
        if longt is None:
            longt = f.get("growth_ttm")
        if recent is None and longt is None:
            continue
        rows.append({"symbol": sym, "recent": recent, "longt": longt})
    if not rows:
        return {}
    df = pd.DataFrame(rows).set_index("symbol")
    # clip extremes so one turnaround (÷ tiny base) can't dominate the ranking
    r = _pctile_1_99(df["recent"].clip(-200, 500))
    l = _pctile_1_99(pd.to_numeric(df["longt"], errors="coerce").clip(-100, 300))
    blend = pd.concat([r, l], axis=1).mean(axis=1, skipna=True)
    # RE-RANK the blend. Averaging two uniform percentile ranks yields a
    # TRIANGULAR distribution bunched at the middle, so "≥80" stopped meaning
    # "top 20%" (measured: only 5.7% of stocks scored ≥80 instead of 20%).
    # MarketSmith's ratings are percentiles OF the composite measure, so the
    # blend has to be percentile-ranked again to restore the 1-99 scale.
    return _pctile_1_99(blend).dropna().astype(int).to_dict()


# ── 3. SMR Rating (Sales, Margins, ROE) ──────────────────────────────────────
def smr_rating(funds: dict) -> dict:
    """Equal-weight percentile blend of Sales growth, profit Margin and ROE.
    Margins are optional — if the scraper hasn't captured OPM yet the rating is
    computed from the legs that ARE present rather than returning nothing."""
    rows = []
    for sym, f in funds.items():
        sales = f.get("sales_growth_3y_cagr")
        if sales is None:
            sales = f.get("sales_growth_ttm")
        if sales is None:
            sales = f.get("sales_growth_yoy")
        rows.append({"symbol": sym, "sales": sales,
                     "margin": f.get("opm"), "roe": f.get("roe")})
    if not rows:
        return {}
    df = pd.DataFrame(rows).set_index("symbol")
    legs = []
    for col, lo, hi in (("sales", -100, 300), ("margin", -100, 100), ("roe", -100, 200)):
        s = pd.to_numeric(df[col], errors="coerce")
        if s.notna().sum() >= max(10, 0.1 * len(df)):     # enough coverage to rank
            legs.append(_pctile_1_99(s.clip(lo, hi)))
    if not legs:
        return {}
    # Re-rank the blend for the same reason as EPS above — averaged percentile
    # ranks are triangular, not uniform, so "≥80 = top 20%" only holds after
    # a second percentile pass.
    blend = pd.concat(legs, axis=1).mean(axis=1, skipna=True)
    return _pctile_1_99(blend).dropna().astype(int).to_dict()


# ── 4. A/D — Buyer Demand (reuses the existing faithful implementation) ──────
def ad_grades(stocks: dict) -> tuple[dict, dict]:
    """(grade A-E, raw score). Reuses breakout_scanner._buyer_demand — a 13-week
    Chaikin-CLV money-flow score — then grades it CROSS-SECTIONALLY (A = top 20%),
    which is how IBD's relative A/D actually discriminates."""
    try:
        from breakout_scanner import _buyer_demand
    except Exception:
        return {}, {}
    raw = {}
    for sym, df in stocks.items():
        try:
            if len(df) >= 20 and {"High", "Low", "Close", "Volume"} <= set(df.columns):
                raw[sym] = float(_buyer_demand(df))
        except Exception:
            continue
    if not raw:
        return {}, {}
    s = pd.Series(raw)
    pct = s.rank(pct=True)
    return {k: _grade_from_pct(v) for k, v in pct.items()}, {k: round(v, 4) for k, v in raw.items()}


# ── 5. Group Rank (6-month, per MarketSmith) ─────────────────────────────────
def group_ranks_6m(stocks: dict) -> tuple[dict, dict, int]:
    """(symbol→rank, group→rank, n_groups). Industry groups ranked on 6-MONTH
    price performance — MarketSmith's window (ours previously used 3-month)."""
    try:
        from industry_groups import INDUSTRY_GROUPS
    except Exception:
        return {}, {}, 0
    # Build the widest symbol→group map available. INDUSTRY_GROUPS alone is
    # hand-curated and left most scanned names ungrouped (Group Rank showed "—"
    # for MTARTECH, ASTRAMICRO, IDEAFORGE…); sector_mapper's auto-derived map
    # covers the rest. Hand-curated wins on conflict, as elsewhere in the app.
    groups: dict[str, list[str]] = {g: list(s) for g, s in INDUSTRY_GROUPS.items()}
    _curated = {s for syms in INDUSTRY_GROUPS.values() for s in syms}
    try:
        from sector_mapper import get_enriched_sector_map
        for sym, sec in (get_enriched_sector_map() or {}).items():
            if sym not in _curated and sec:
                groups.setdefault(sec, []).append(sym)
    except Exception:
        pass          # sector_mapper is LOCAL-ONLY; absence is fine

    sym2grp, grp_ret = {}, {}
    for grp, syms in groups.items():
        rets = []
        for s in syms:
            sym2grp[s] = grp
            df = stocks.get(s)
            if df is None or "Close" not in df:
                continue
            v = _ret(df["Close"].dropna().to_numpy(dtype=float), 126)   # 6 months
            if v is not None:
                rets.append(v)
        if len(rets) >= 2:
            grp_ret[grp] = float(np.mean(rets))
    if not grp_ret:
        return {}, {}, 0
    order = sorted(grp_ret.items(), key=lambda kv: kv[1], reverse=True)
    grp_rank = {g: i + 1 for i, (g, _) in enumerate(order)}
    return ({s: grp_rank[g] for s, g in sym2grp.items() if g in grp_rank},
            grp_rank, len(grp_rank))


# ── 6. Composite Rating & Master Score ───────────────────────────────────────
_GRADE_NUM = {"A": 95, "B": 80, "C": 55, "D": 30, "E": 10}


# A composite built from ONE leg is not a composite — it's that leg wearing a
# disguise. Without this floor a stock with no fundamentals scored Composite 95
# purely off its A/D grade and sorted to the top of the table.
_MIN_PARTS = 3


def composite_rating(eps: int | None, rs: int | None, ad: str | None,
                     smr: int | None) -> int | None:
    """EPS + Price Strength + Buyer Demand + SMR → 1-99 (equal weight; MarketSmith
    does not publish theirs). Returns None unless at least `_MIN_PARTS` of the
    four legs are present, so missing data reads as "—" instead of a fake score."""
    parts = [x for x in (eps, rs, _GRADE_NUM.get(ad or ""), smr) if x is not None]
    if len(parts) < _MIN_PARTS:
        return None
    return int(round(sum(parts) / len(parts)))


def master_score(eps: int | None, rs: int | None, ad: str | None,
                 grp_rank: int | None, n_groups: int) -> str | None:
    """EPS + Price Strength + Buyer Demand + GROUP RANK → A-E.
    (Group Rank replaces SMR here — that's the documented difference vs Composite.)"""
    parts = [x for x in (eps, rs, _GRADE_NUM.get(ad or "")) if x is not None]
    if grp_rank and n_groups:
        parts.append((1.0 - (grp_rank - 1) / max(1, n_groups - 1)) * 99)
    if len(parts) < _MIN_PARTS:
        return None
    return _grade_from_pct(sum(parts) / len(parts) / 99.0)


# ── 7. CAN SLIM checklist ────────────────────────────────────────────────────
def canslim_checklist(r: dict, n_groups: int) -> list[dict]:
    """Per-criterion Pass/Fail against MarketSmith's published thresholds."""
    eps, rs, smr = r.get("eps_rating"), r.get("price_str"), r.get("smr_rating")
    ad, comp = r.get("ad_grade"), r.get("composite_rating")
    grp = r.get("group_rank")
    items = [
        ("C — Current quarterly earnings", eps is not None and eps >= TH_EPS,
         f"EPS Rating {eps if eps is not None else '—'} (need ≥{TH_EPS})"),
        ("A — Annual earnings growth", smr is not None and smr >= TH_SMR,
         f"SMR {smr if smr is not None else '—'} (need ≥{TH_SMR})"),
        ("N — New high / new base", bool(r.get("pattern")) or bool(r.get("early_entry")),
         r.get("pattern") or "no fresh base"),
        ("S — Supply & demand (volume)", (r.get("vol_mult") or 0) >= 1.5,
         f"{r.get('vol_mult','—')}× avg volume (need ≥1.5)"),
        ("L — Leader, not laggard", rs is not None and rs >= TH_RS,
         f"Price Strength {rs if rs is not None else '—'} (need ≥{TH_RS})"),
        ("I — Institutional sponsorship", ad in ("A", "B"),
         f"Buyer Demand {ad or '—'} (need A/B)"),
        ("M — Market direction", True, "see Risk & Regime tab"),
        ("+ Top industry group", grp is not None and grp <= TH_GROUP_TOP,
         f"Group #{grp if grp is not None else '—'} of {n_groups} (need top {TH_GROUP_TOP})"),
        ("+ Composite", comp is not None and comp >= TH_COMPOSITE,
         f"Composite {comp if comp is not None else '—'} (need ≥{TH_COMPOSITE})"),
    ]
    return [{"label": l, "pass": bool(p), "detail": d} for l, p, d in items]


# ── entry point ──────────────────────────────────────────────────────────────
def enrich(results: list[dict], stocks: dict) -> dict:
    """Attach the full rating stack to each scan result, in place.
    Returns a small meta dict (group count, coverage) for the UI."""
    if not results:
        return {"n_groups": 0}
    try:
        from fundamentals import load_all_fundamentals
        funds = load_all_fundamentals() or {}
    except Exception:
        funds = {}

    rs_map = price_strength(stocks)
    eps_map = eps_rating(funds) if funds else {}
    smr_map = smr_rating(funds) if funds else {}
    ad_map, ad_raw = ad_grades(stocks)
    sym_grp, grp_rank, n_groups = group_ranks_6m(stocks)

    for r in results:
        s = r["symbol"]
        # Overwrite the old 3-month price_str with the true 12-month weighted RS.
        r["price_str"] = rs_map.get(s)
        r["eps_rating"] = eps_map.get(s)
        r["smr_rating"] = smr_map.get(s)
        r["ad_grade"] = ad_map.get(s)
        r["ad_raw"] = ad_raw.get(s)
        r["group_rank"] = sym_grp.get(s)
        r["n_groups"] = n_groups
        r["composite_rating"] = composite_rating(
            r["eps_rating"], r["price_str"], r["ad_grade"], r["smr_rating"])
        r["master_score"] = master_score(
            r["eps_rating"], r["price_str"], r["ad_grade"], r["group_rank"], n_groups)

    # Composite is an average of ranks, so it is triangular for the same reason
    # EPS/SMR were — re-rank it across the scanned set so MarketSmith's "≥90"
    # threshold once again means "top ~10%" rather than "impossibly rare".
    _c = {r["symbol"]: r["composite_rating"] for r in results
          if r.get("composite_rating") is not None}
    if len(_c) >= 10:
        _cr = _pctile_1_99(pd.Series(_c)).astype(int).to_dict()
        for r in results:
            if r["symbol"] in _cr:
                r["composite_rating"] = int(_cr[r["symbol"]])

    for r in results:
        r["canslim"] = canslim_checklist(r, n_groups)
        r["canslim_passed"] = sum(1 for c in r["canslim"] if c["pass"])
        r["canslim_total"] = len(r["canslim"])
    return {"n_groups": n_groups,
            "coverage": {"rs": len(rs_map), "eps": len(eps_map),
                         "smr": len(smr_map), "ad": len(ad_map)}}
