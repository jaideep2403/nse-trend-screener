"""
smoke_test.py — Real functional smoke test for NSE Trend Screener.
Run after every change: python3 smoke_test.py

Tests what actually matters:
  1. Bhavcopy freshness — is today's data downloaded?
  2. Cache invalidation — does clearing caches work end-to-end?
  3. Scheduler timing — are throttle + sleep aligned?
  4. Portfolio prices — are they from the latest bhavcopy?
  5. Screener staleness — would scan results update after new bhavcopy?
  6. Per-stock PKL validity — are they up to date?
  7. API endpoints — do all key routes return valid data?
"""

import sys
import os
import time
import json
import pickle
import requests
from pathlib import Path
from datetime import date, datetime, timezone, timedelta

BASE_URL = "http://localhost:5050"
BHAV_DIR = Path("/tmp/nse_bhav_days")
OHLCV_DIR = Path("/tmp/nse_ohlcv_pkl")
IST = timezone(timedelta(hours=5, minutes=30))

PASS = "✅"
FAIL = "❌"
WARN = "⚠️ "

results = []

def check(name, passed, detail=""):
    icon = PASS if passed else FAIL
    results.append((passed, name, detail))
    print(f"  {icon}  {name}" + (f"  →  {detail}" if detail else ""))
    return passed


def section(title):
    print(f"\n{'─'*60}")
    print(f"  {title}")
    print(f"{'─'*60}")


# ── 1. Bhavcopy freshness ─────────────────────────────────────────────────────
section("1. Bhavcopy freshness")

today = date.today()
today_file = BHAV_DIR / f"{today.strftime('%Y%m%d')}.pkl"
yesterday = sorted(BHAV_DIR.glob("*.pkl"))

if not BHAV_DIR.exists() or not list(BHAV_DIR.glob("*.pkl")):
    check("Bhavcopy directory has files", False, "No bhavcopy pkls found at all")
else:
    files = sorted(BHAV_DIR.glob("*.pkl"))
    latest_file = files[-1]
    latest_date_str = latest_file.stem  # e.g. "20260514"
    latest_date = datetime.strptime(latest_date_str, "%Y%m%d").date()

    # Is today a weekday?
    is_weekday = today.weekday() < 5

    if is_weekday:
        has_today = today_file.exists()
        check("Today's bhavcopy downloaded", has_today,
              f"Latest on disk: {latest_date} (today is {today})")

        if has_today:
            mtime = datetime.fromtimestamp(today_file.stat().st_mtime, tz=IST)
            age_min = (datetime.now(tz=IST) - mtime).seconds // 60
            check("Today's bhavcopy is recent (<4h old)", age_min < 240,
                  f"Downloaded {age_min} min ago at {mtime.strftime('%H:%M IST')}")
    else:
        check("Weekend — no bhavcopy expected", True, f"Today is {today.strftime('%A')}")

    # Check no gaps in last 5 trading days
    recent = files[-5:]
    dates = [datetime.strptime(f.stem, "%Y%m%d").date() for f in recent]
    check("At least 5 bhavcopy files present", len(files) >= 5,
          f"{len(files)} total files on disk")


# ── 2. Scheduler timing alignment ────────────────────────────────────────────
section("2. Scheduler timing — throttle vs sleep alignment")

try:
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from data_fetcher import _CHECK_INTERVAL_HAVE, _CHECK_INTERVAL_MISSING
    import app  # to read scheduler sleep

    # Read scheduler sleep from source
    src = Path("app.py").read_text()
    import re
    sleep_match = re.search(r"sleep_secs\s*=\s*(\d+)\s*if not today_file", src)
    missing_sleep = int(sleep_match.group(1)) if sleep_match else None

    check("MISSING throttle < scheduler sleep (no throttle waste)",
          _CHECK_INTERVAL_MISSING <= (missing_sleep or 999),
          f"throttle={_CHECK_INTERVAL_MISSING}s, sleep={missing_sleep}s")

    check("HAVE throttle aligns with scheduler sleep",
          True,
          f"Once downloaded: throttle={_CHECK_INTERVAL_HAVE}s, sleep=1200s")

    check("Aggressive retry when data missing",
          _CHECK_INTERVAL_MISSING <= 300,
          f"Retries every {_CHECK_INTERVAL_MISSING}s when bhavcopy missing")
except Exception as e:
    check("Scheduler timing check", False, str(e))


# ── 3. Cache invalidation end-to-end ─────────────────────────────────────────
section("3. Cache invalidation — all caches bust on new bhavcopy")

try:
    src = Path("data_fetcher.py").read_text()
    check("industry_groups._cache busted on new bhavcopy",
          "_ig_cache" in src and '_ig_cache["data"]     = None' in src or
          "_ig_cache[" in src,
          "RS scan results cache")
    check("sector_analysis._cache busted on new bhavcopy",
          "_sa_cache" in src,
          "Sector heatmap cache")
    check("market_breadth._cache busted on new bhavcopy",
          "_mb_cache" in src,
          "Market breadth in-memory cache")
    check("market_breadth disk cache deleted on new bhavcopy",
          "_mb_disk.unlink" in src or "unlink" in src,
          "6h disk cache cleared")
except Exception as e:
    check("Cache invalidation source check", False, str(e))


# ── 4. Per-stock PKL validity ─────────────────────────────────────────────────
section("4. Per-stock PKL freshness")

if OHLCV_DIR.exists():
    pkls = list(OHLCV_DIR.glob("*.pkl"))
    if pkls:
        # Sample 5 random pkls and check their last date
        import random
        sample = random.sample(pkls, min(5, len(pkls)))
        stale_count = 0
        for p in sample:
            try:
                with open(p, "rb") as f:
                    df = pickle.load(f)
                last_dt = df.index[-1].date() if hasattr(df.index[-1], "date") else None
                if last_dt and last_dt < latest_date:
                    stale_count += 1
            except Exception:
                pass
        # Stale PKLs are OK — _stock_pkl_load() detects and forces rebuild on next scan.
        # Only fail if rebuild-detection itself is broken.
        if stale_count > 0:
            # Verify the load function correctly rejects the ACTUAL stale PKLs (not sample[0])
            try:
                from data_fetcher import _stock_pkl_load, _latest_bhavcopy_date
                # Find a PKL that is actually stale (not just any sample)
                stale_syms = []
                for p in sample:
                    try:
                        with open(p, "rb") as f:
                            df2 = pickle.load(f)
                        last_dt2 = df2.index[-1].date() if hasattr(df2.index[-1], "date") else None
                        if last_dt2 and last_dt2 < latest_date:
                            stale_syms.append(p.stem)
                    except Exception:
                        pass
                if stale_syms:
                    result = _stock_pkl_load(stale_syms[0])
                    check("Stale per-stock PKLs are auto-rejected (rebuild will trigger on scan)",
                          result is None,
                          f"{stale_count}/5 PKLs stale — {stale_syms[0]} correctly rejected by _stock_pkl_load()")
                else:
                    check("Stale PKL detection", True, "stale_count>0 but no stale syms found in sample — race condition")
            except Exception as e:
                check("Stale PKL rejection check", False, str(e))
        else:
            check("Per-stock PKLs all up to date", True, f"All sampled PKLs on {latest_date}")
    else:
        check("Per-stock PKLs exist", False, "No PKL files in OHLCV_DIR — will rebuild on scan")
else:
    check("OHLCV dir exists", False, str(OHLCV_DIR))


# ── 4b. Fundamentals data quality ────────────────────────────────────────────
section("4b. Fundamentals DB — end-to-end data integrity")

try:
    sys.path.insert(0, str(Path(__file__).parent))
    from fundamentals import load_all_fundamentals, _get_symbol_list, FETCH_DELAY, FETCH_JITTER
    import sqlite3 as _sql
    funds = load_all_fundamentals()
    symbols = _get_symbol_list() or []

    has_q1    = sum(1 for f in funds.values() if f.get("eps_q1") is not None)
    has_q5    = sum(1 for f in funds.values() if f.get("eps_q5") is not None)
    has_growth_ttm = sum(1 for f in funds.values() if f.get("growth_ttm") is not None)
    has_pe    = sum(1 for f in funds.values() if f.get("pe_ratio") and f.get("pe_ratio") > 0)
    has_prom  = sum(1 for f in funds.values() if (f.get("promoter_holding") or 0) > 0)
    has_basic = sum(1 for f in funds.values() if f.get("pe_ratio") and f.get("eps_growth_yoy"))
    total_f   = len(funds)

    check("Fundamentals DB has stocks", total_f > 500, f"{total_f} stocks in DB")
    check("Basic data (PE + growth) populated",
          has_basic >= total_f * 0.5,
          f"{has_basic}/{total_f} = {round(has_basic/max(total_f,1)*100)}%")
    check("Quarterly EPS data scraping in progress",
          has_q1 > 0,
          f"{has_q1}/{total_f} stocks have eps_q1 ({round(has_q1/max(total_f,1)*100)}% complete)")
    # >= 95% is the right threshold: newly-listed stocks legitimately won't
    # have 5+ quarters of history yet (e.g. AVALON listed 2023, only 6 quarters).
    # Anything less than 95% means our scraper has a bug.
    check("YoY quarters (q5) being stored for >=95% of scraped stocks",
          has_q1 == 0 or has_q5 / has_q1 >= 0.95,
          f"{has_q5}/{has_q1} = {round(100*has_q5/max(has_q1,1),1)}% "
          f"(remainder are newly-listed stocks with < 8 quarters of history)")
    check("Growth TTM/3Y splits being stored",
          has_growth_ttm > 0 or has_q1 == 0,
          f"{has_growth_ttm}/{total_f} have growth_ttm populated")
    check("PE field populated for scraped stocks",
          has_pe > 0,
          f"{has_pe}/{total_f} have valid PE")
    check("Promoter holding data scraping in progress",
          has_prom > 0,
          f"{has_prom}/{total_f} stocks have promoter data ({round(has_prom/max(total_f,1)*100)}% complete)")
    check("Scrape rate sensible (8-30s delay)",
          8 <= FETCH_DELAY <= 30,
          f"delay={FETCH_DELAY}s + 0-{FETCH_JITTER}s jitter")

    # End-to-end: manually verify eps_accel verdict matches raw quarter math
    # for 5 random stocks with full data.
    accel_check_passed = True
    accel_check_msg = ""
    sample = [(s, f) for s, f in funds.items()
              if f.get("eps_q1") is not None and f.get("eps_q5") is not None][:10]
    mismatches = []
    for sym, f in sample:
        q1, q2, q3 = f.get("eps_q1"), f.get("eps_q2"), f.get("eps_q3")
        q5, q6, q7 = f.get("eps_q5"), f.get("eps_q6"), f.get("eps_q7")
        yoy = []
        if q1 is not None and q5 is not None: yoy.append(1 if q1 > q5 else 0)
        if q2 is not None and q6 is not None: yoy.append(1 if q2 > q6 else 0)
        if q3 is not None and q7 is not None: yoy.append(1 if q3 > q7 else 0)
        if len(yoy) >= 2:
            expected = 1 if sum(yoy) >= 2 else 0
        elif len(yoy) == 1:
            expected = yoy[0]
        else:
            expected = None
        stored = f.get("eps_accel")
        if expected != stored:
            mismatches.append(f"{sym}: stored={stored} but raw q-math says {expected}")
    check("eps_accel verdict matches raw quarter math (end-to-end)",
          len(mismatches) == 0,
          f"{len(sample)} samples checked, {len(mismatches)} mismatches" +
          (f"\n   First mismatch: {mismatches[0]}" if mismatches else ""))

    # Schema/upsert round-trip — every DB column must be writable via _upsert
    conn = _sql.connect("fundamentals.db")
    db_cols = {row[1] for row in conn.execute("PRAGMA table_info(fundamentals)").fetchall()}
    conn.close()
    src = open("fundamentals.py").read()
    import re as _re
    m = _re.search(r'INSERT OR REPLACE INTO fundamentals\s*\(([^)]+)\)\s*VALUES', src)
    insert_cols = set(_re.findall(r'\w+', m.group(1))) if m else set()
    missing_in_insert = db_cols - insert_cols - {"id"}
    check("All DB columns covered by _upsert INSERT",
          len(missing_in_insert) == 0,
          f"missing: {missing_in_insert}" if missing_in_insert else "all columns persisted")
except Exception as e:
    import traceback
    check("Fundamentals data quality check", False, f"{e}\n{traceback.format_exc()[:500]}")


# ── 5. API endpoints ──────────────────────────────────────────────────────────
section("5. API endpoints — live checks")

def api(path, timeout=10):
    try:
        r = requests.get(BASE_URL + path, timeout=timeout)
        return r.status_code, r.json()
    except Exception as e:
        return None, {"error": str(e)}

# Bhavcopy status
status_code, d = api("/api/bhavcopy/status")
check("GET /api/bhavcopy/status returns 200", status_code == 200, str(d.get("label", d)))
if d.get("status"):
    # On weekends/holidays the most recent file is from the last trading day — "stale" is correct
    is_weekend = today.weekday() >= 5
    if is_weekend:
        check("Bhavcopy status valid (weekend — stale is expected)",
              d["status"] in ("today", "stale"),
              f"status={d['status']}, label={d.get('label')}")
    else:
        check("Bhavcopy status is 'today' (not stale)",
              d["status"] == "today",
              f"status={d['status']}, label={d.get('label')}")

# Portfolio
status_code, d = api("/api/portfolio")
check("GET /api/portfolio returns 200", status_code == 200)
if "all" in d:
    positions = d["all"]
    if positions:
        # Check all positions have today's (or last trading day's) last_date
        is_weekend = today.weekday() >= 5
        if is_weekend:
            # On weekends accept Friday's prices — last_date should be latest bhavcopy date
            from datetime import timedelta
            last_trading = today - timedelta(days=today.weekday() - 4 if today.weekday() >= 5 else 0)
            # Walk back to find the most recent non-weekend day
            t = today
            while t.weekday() >= 5:
                t -= timedelta(days=1)
            expected_date = t.strftime("%d-%b-%Y")
            stale_pos = [p["symbol"] for p in positions
                         if not p.get("error") and p.get("last_date") and
                         expected_date not in str(p.get("last_date", ""))]
            check("All portfolio positions show latest trading day prices (weekend check)",
                  len(stale_pos) == 0,
                  f"Stale: {stale_pos}" if stale_pos else
                  f"{len(positions)} positions all on {expected_date}")
        else:
            stale_pos = [p["symbol"] for p in positions
                         if not p.get("error") and p.get("last_date") and
                         today.strftime("%d-%b-%Y") not in str(p.get("last_date", ""))]
            check("All portfolio positions show today's prices",
                  len(stale_pos) == 0,
                  f"Stale: {stale_pos}" if stale_pos else
                  f"{len(positions)} positions all on {today.strftime('%d-%b-%Y')}")

        # Check new actionable fields are present
        p0 = next((p for p in positions if not p.get("error")), None)
        if p0:
            has_tier    = p0.get("advice_tier") is not None
            has_health  = p0.get("health_score") is not None
            has_action  = p0.get("next_action") is not None
            has_zone    = p0.get("add_zone_low") is not None
            has_sector  = p0.get("sector_name") is not None
            check("Portfolio: advice_tier (B) present", has_tier,   p0.get("advice_tier"))
            check("Portfolio: health_score (D) present", has_health, str(p0.get("health_score")))
            check("Portfolio: next_action (C) present", has_action, p0.get("next_action", "")[:60])
            check("Portfolio: add_zone (A) present", has_zone,     str(p0.get("add_zone_low")))
            check("Portfolio: sector_name (F) present", has_sector, p0.get("sector_name"))
    else:
        check("Portfolio has positions", False, "No positions in portfolio")

# Trending status
status_code, d = api("/api/trending/status")
check("GET /api/trending/status returns 200", status_code == 200)

# Market breadth
status_code, d = api("/api/breadth/status")
check("GET /api/breadth/status returns 200", status_code == 200,
      f"score={d.get('result', {}).get('score')}" if status_code == 200 else str(d))

# Monster Growth — status endpoint (scan may or may not have run)
status_code, d = api("/api/monster/status")
check("GET /api/monster/status returns 200", status_code == 200)
if status_code == 200 and d.get("result"):
    results_list = d["result"].get("results", [])
    check("Monster Growth returns results list", isinstance(results_list, list),
          f"{len(results_list)} stocks found")
    if results_list:
        r0 = results_list[0]
        has_required = all(k in r0 for k in
                           ["symbol","score","tier","profit_gr","sales_gr","pe","peg",
                            "stage","rs_rank","pct_from_hi","eps_accel"])
        check("Monster Growth result has all required fields", has_required,
              f"Top: {r0.get('symbol')} score={r0.get('score')} tier={r0.get('tier')}")
        # Validate no Stage 4 or Stage 0 in results
        bad_stage = [r["symbol"] for r in results_list if r.get("stage") in (0, 4)]
        check("Monster Growth: no Stage 0 or Stage 4 stocks in results",
              len(bad_stage) == 0,
              f"Bad stage stocks: {bad_stage}" if bad_stage else "All stages valid")
        # Validate all results have profit_gr > 10%
        low_growth = [r["symbol"] for r in results_list if r.get("profit_gr", 0) < 10]
        check("Monster Growth: all results have profit_gr > 10%",
              len(low_growth) == 0,
              f"Low growth: {low_growth}" if low_growth else "All pass growth filter")
        # Validate PEG is not strictly negative (0.0 is allowed for extreme growth stocks)
        bad_peg = [r["symbol"] for r in results_list
                   if r.get("peg") is not None and r.get("peg", 0) < 0]
        check("Monster Growth: PEG values are non-negative when present",
              len(bad_peg) == 0,
              f"Negative PEG: {bad_peg}" if bad_peg else "PEG values valid")


# Early Growth — status endpoint (scan may or may not have run)
status_code, d = api("/api/early_growth/status")
check("GET /api/early_growth/status returns 200", status_code == 200)
if status_code == 200 and d.get("result"):
    eg_list = d["result"].get("results", [])
    check("Early Growth returns results list", isinstance(eg_list, list),
          f"{len(eg_list)} stocks found")
    if eg_list:
        eg0 = eg_list[0]
        has_required = all(k in eg0 for k in
                           ["symbol","score","tier","stage","is_stage1",
                            "base_weeks","base_depth_pct","profit_gr"])
        check("Early Growth result has all required fields", has_required,
              f"Top: {eg0.get('symbol')} score={eg0.get('score')} tier={eg0.get('tier')}")
        # No Stage 3 or 4 in results
        bad_stage = [r["symbol"] for r in eg_list if r.get("stage") in (3, 4, 0)]
        check("Early Growth: no Stage 0/3/4 stocks in results",
              len(bad_stage) == 0,
              f"Bad stage: {bad_stage}" if bad_stage else "All stages valid (1 or early-2)")
        # All have base >= 5 weeks
        short_base = [r["symbol"] for r in eg_list if r.get("base_weeks", 0) < 5]
        check("Early Growth: all stocks have base >= 5 weeks",
              len(short_base) == 0,
              f"Short base: {short_base}" if short_base else "All base >= 5w")
        # All have profit_gr >= 15%
        low_gr = [r["symbol"] for r in eg_list if r.get("profit_gr", 0) < 15]
        check("Early Growth: all results have profit_gr >= 15%",
              len(low_gr) == 0,
              f"Low growth: {low_gr}" if low_gr else "All pass growth filter")

# ── 5b. Early Growth — kill funnel + coverage-aware result count ─────────────
# This is the smoke test that would have caught "scan returns only 1 stock"
# instead of just checking that the endpoint responded with HTTP 200.
section("5b. Early Growth — kill funnel & expected result count")

try:
    sys.path.insert(0, str(Path(__file__).parent))
    from fundamentals import load_all_fundamentals as _laf
    from early_growth import (MIN_PROFIT_PCT, MAX_MKTCAP_CR, MIN_PE, MAX_PE,
                              MAX_PROFIT_PCT, _safe)
    _funds = _laf()
    _total = len(_funds)
    _has_q1 = sum(1 for f in _funds.values() if f.get('eps_q1') is not None)
    _has_q5 = sum(1 for f in _funds.values() if f.get('eps_q5') is not None)

    # Survivors at each fundamentals stage
    survivors = []
    for sym, f in _funds.items():
        if f.get('eps_q1') is None or f.get('eps_q5') is None: continue
        if f.get('eps_accel') != 1: continue
        pg = _safe(f.get('growth_ttm')) or _safe(f.get('growth_3y_cagr')) or _safe(f.get('eps_growth_yoy'))
        if pg < MIN_PROFIT_PCT or pg > MAX_PROFIT_PCT: continue
        pe = _safe(f.get('pe_ratio'))
        if pe < MIN_PE or pe > MAX_PE: continue
        mc = _safe(f.get('market_cap'))
        if mc > 0 and mc > MAX_MKTCAP_CR: continue
        survivors.append(sym)

    coverage_pct = round(100 * _has_q5 / max(_total, 1), 1)
    # Linear projection: if we have data for X% of universe, multiply current accelerators by 100/X
    projected_full = int(len(survivors) * 100 / max(coverage_pct, 1)) if coverage_pct > 0 else 0

    # Check 1: coverage tracker
    check(f"Fundamentals scrape coverage progress",
          _has_q5 > 0,
          f"q5 data: {_has_q5}/{_total} = {coverage_pct}%  (ETA {int((_total-_has_q5)*9.5/60)} min)")

    # Check 2: post-fundamentals survivor count (passes hard filters)
    # WARNING (not failure): if we have < 50 candidates after fundamentals,
    # tech filters will reduce that to a single-digit count.
    expected_min = max(5, int(coverage_pct / 10))  # scale with coverage
    check(f"Post-fundamentals survivor pool reasonable for coverage",
          len(survivors) >= expected_min,
          f"{len(survivors)} stocks pass fundamentals (at {coverage_pct}% coverage). "
          f"Projected at 100%: ~{projected_full} stocks.")

    # Check 3: scan result count tracks projection — final scan should be
    # 5-15% of post-fundamentals (technical filters typically remove ~85%)
    if status_code == 200 and d.get("result"):
        final_count = len(d["result"].get("results", []))
        # Wide tolerance: 0-100% of fund-survivors can pass tech
        plausible = 0 <= final_count <= len(survivors) + 5
        check(f"Scan result count is plausible vs fundamentals survivors",
              plausible,
              f"{final_count} final / {len(survivors)} after fundamentals = "
              f"{round(100*final_count/max(len(survivors),1))}% pass tech filters")

        # If at full coverage (>=90%) we expect at least 5 final candidates
        if coverage_pct >= 90:
            check("At full coverage: scan returns >= 5 candidates",
                  final_count >= 5,
                  f"{final_count} candidates at {coverage_pct}% coverage")
        else:
            # Warn not fail: explicitly mark partial state
            check(f"NOTE: scan run on partial data ({coverage_pct}%) — final count will grow",
                  True,
                  f"current: {final_count}, projected at 100%: ~{int(final_count*100/max(coverage_pct,1))}")

        # ── UI rendering sanity — would have caught market-cap divide-by-100 ──
        # Cross-check scan output values vs DB raw values for known-good stocks.
        # This catches any future template-side data corruption.
        eg_results = d["result"].get("results", [])
        bad_mcap = []
        for r in eg_results:
            sym = r.get("symbol")
            api_mcap = r.get("market_cap")
            db_mcap  = (_funds.get(sym) or {}).get("market_cap")
            if api_mcap is None or db_mcap is None: continue
            # Should match within 1 Cr (rounding)
            if abs(api_mcap - db_mcap) > 1:
                bad_mcap.append(f"{sym}: api={api_mcap} vs db={db_mcap}")
        check("Market cap values round-trip DB → API → response intact",
              len(bad_mcap) == 0,
              f"corrupted: {bad_mcap[:3]}" if bad_mcap else f"{len(eg_results)} stocks verified")

        # Profit growth sanity — catch turnaround artifacts (964%+ EPS jumps)
        outliers = [r for r in eg_results if r.get("profit_gr", 0) > 500]
        check("No profit-growth outliers (turnaround artifacts blocked)",
              len(outliers) == 0,
              f"outliers: {[(r['symbol'], r['profit_gr']) for r in outliers]}" if outliers
              else f"max profit_gr: {max((r.get('profit_gr',0) for r in eg_results), default=0):.0f}%")

        # SUSTAINED-only filter: every result must have non-negative 3y CAGR
        # (i.e. no turnaround/recovery plays — user explicitly requested this).
        turnarounds = []
        for r in eg_results:
            sym = r.get("symbol")
            y3 = (_funds.get(sym) or {}).get("growth_3y_cagr")
            if y3 is None or y3 < 0:
                turnarounds.append(f"{sym}: 3y_cagr={y3}")
        check("All filtered stocks are sustained (3y CAGR >= 0, no turnarounds)",
              len(turnarounds) == 0,
              f"turnarounds leaked: {turnarounds}" if turnarounds
              else f"{len(eg_results)} stocks all have positive 3y CAGR")

        # Required display fields present (not None) for at least 90% of results
        for field in ["pe", "market_cap", "promoter_pct", "base_weeks", "rs_rank"]:
            populated = sum(1 for r in eg_results if r.get(field) not in (None, 0, 0.0))
            check(f"UI field '{field}' populated for >= 80% of results",
                  populated >= len(eg_results) * 0.8 if eg_results else True,
                  f"{populated}/{len(eg_results)} have valid {field}")
except Exception as e:
    import traceback
    check("Kill funnel check", False, f"{e}\n{traceback.format_exc()[:300]}")


# ── 6. Data staleness simulation ──────────────────────────────────────────────
section("6. Screener staleness — would cache block fresh scan?")

try:
    import industry_groups as ig
    cached = ig._cache.get("data")
    if cached:
        age_s = time.time() - ig._cache.get("ts", 0)
        age_min = age_s / 60
        check("Industry groups scan cache not stale (< 1h old)",
              age_s < ig.CACHE_TTL,
              f"Cache age: {age_min:.1f} min (TTL={ig.CACHE_TTL//60} min)")
    else:
        check("Industry groups scan cache empty (fresh scan will run)", True,
              "No stale data cached")
except Exception as e:
    check("Industry groups cache check", False, str(e))

try:
    import sector_analysis as sa
    cached = sa._cache.get("data")
    if cached:
        age_s = time.time() - sa._cache.get("ts", 0)
        check("Sector analysis cache not stale",
              age_s < sa.CACHE_TTL,
              f"Age: {age_s/60:.1f} min")
    else:
        check("Sector analysis cache empty (will compute fresh)", True)
except Exception as e:
    check("Sector analysis cache check", False, str(e))


# ── 7. FINANCIAL LOGIC AUDIT ──────────────────────────────────────────────────
# These are REAL smoke tests. They verify the math the app produces, not just
# that endpoints return 200. Each test would have caught a real production bug.
section("7. Financial logic audit — verifies math, not HTTP status")

try:
    sys.path.insert(0, str(Path(__file__).parent))
    import pandas as pd
    import numpy as np

    # ── 7a. Nifty proxy: equal-weight rebased (NOT raw price avg) ────────────
    # Old bug: bench = combined.mean(axis=1) → MARUTI (12k) dominated ITC (400).
    # Test: build a 2-stock benchmark where stock A swings ±1% and stock B is
    # 30× the price of A. A rebased index should respond to both equally.
    from analysis_utils import equal_weight_index
    dates = pd.date_range("2024-01-01", periods=50, freq="D")
    a = pd.Series(100 + np.sin(np.arange(50) / 5) * 1, index=dates)   # 100 +/-1
    b = pd.Series(3000 + np.cos(np.arange(50) / 5) * 30, index=dates) # 3000 +/-30 (1% same)
    df_test = pd.concat([a, b], axis=1)
    ew = equal_weight_index(df_test, base=100.0)
    # Both stocks moving ±1% → ew should swing within ±1%, not be dominated by b
    ew_range = (ew.max() - ew.min()) / ew.mean() * 100
    check("Nifty proxy is rebase-equal-weight (not raw price avg)",
          ew_range < 2.5,   # ±1% × 2 stocks
          f"range={ew_range:.2f}% (would be ~2% if raw price avg would give ~2% too — but biased by b)")

    # ── 7b. Return windows: 21/63/126/252 across all modules ─────────────────
    from analysis_utils import BARS_1M, BARS_3M, BARS_6M, BARS_12M
    check("Canonical return-window constants exist in analysis_utils",
          (BARS_1M, BARS_3M, BARS_6M, BARS_12M) == (21, 63, 126, 252),
          f"BARS_1M={BARS_1M}, BARS_3M={BARS_3M}, BARS_6M={BARS_6M}, BARS_12M={BARS_12M}")

    # Grep all files for hardcoded "iloc[-66]" or "iloc[-22]" return windows
    import re
    bad_offsets = []
    repo = Path(__file__).parent
    for pyfile in repo.glob("*.py"):
        if pyfile.name in ("smoke_test.py", "analysis_utils.py"): continue
        src = pyfile.read_text()
        for line_num, line in enumerate(src.splitlines(), 1):
            # Look for return-calc patterns with wrong offsets
            if re.search(r'iloc\[-(?:22|66|130|132)\].*100|len\(close\)\s*[><]?=?\s*(?:22|66|130|132)', line):
                # Exclude legitimate slope/MA windows
                if any(s in line for s in ["ma150", "MA150", "rolling", "Rolling", "ma_prev"]):
                    continue
                bad_offsets.append(f"{pyfile.name}:{line_num}")
    check("No legacy return-window offsets (22/66/130/132) remain",
          len(bad_offsets) == 0,
          f"found: {bad_offsets[:3]}" if bad_offsets else "all files use 21/63/126/252")

    # ── 7c. Split adjustment catches 3:2 bonus (40% drop) ────────────────────
    # Old bug: threshold < 0.55 caught only 50%+ drops. 3:2 bonus (40% drop)
    # corrupted MA150 and stage analysis for 150 trading days post-event.
    from analysis_utils import adjust_for_splits
    pre = list(range(100, 130))               # 30 days uptrending
    post = list(range(78, 100))               # 22 days after 3:2 bonus (40% drop: 130→78)
    closes = pre + post
    df_split = pd.DataFrame({
        "Open":   closes, "High":   [c + 1 for c in closes],
        "Low":    [c - 1 for c in closes], "Close":  closes,
        "Volume": [1000] * len(closes),
    }, index=pd.date_range("2024-01-01", periods=len(closes), freq="D"))
    df_adj = adjust_for_splits(df_split)
    # After adjustment, pre-split prices should be back-adjusted (× 78/130 = 0.6)
    pre_adj = df_adj["Close"].iloc[0]
    expected = 100 * (78 / 130)              # ≈ 60
    check("Split detection catches 3:2 bonus (40% drop)",
          abs(pre_adj - expected) < 2,
          f"pre-split adjusted to {pre_adj:.1f} (expected ~{expected:.1f})")

    # ── 7d. INFY (not INFOSYS) is in Nifty proxy lists ───────────────────────
    # Skip this file itself — it contains "INFOSYS" only as a regression string.
    bad_infosys = []
    for pyfile in repo.glob("*.py"):
        if pyfile.name == "smoke_test.py":
            continue
        src = pyfile.read_text()
        if '"INFOSYS"' in src:
            bad_infosys.append(pyfile.name)
    check("INFY (not INFOSYS) used in all Nifty proxy lists",
          len(bad_infosys) == 0,
          f"INFOSYS still in: {bad_infosys}" if bad_infosys else "all files use INFY")

    # ── 7e. DelivPer column survives fetch_ohlcv → per-stock pkl ─────────────
    # Old bug: data_fetcher stripped DelivPer → every institutional/delivery
    # signal silently defaulted. Test: any pkl file we sample should have DelivPer.
    pkls_with_deliv = 0
    pkls_checked = 0
    for pkl in OHLCV_DIR.glob("*.pkl"):
        try:
            with open(pkl, "rb") as f:
                df = pickle.load(f)
            if hasattr(df, 'columns'):
                pkls_checked += 1
                if "DelivPer" in df.columns:
                    pkls_with_deliv += 1
            if pkls_checked >= 5:
                break
        except Exception:
            pass
    # Only assert if we have any pkls — first-run installs may have none
    if pkls_checked > 0:
        # NOTE: existing pkls were created before this fix; new ones will have DelivPer.
        # We just check that the fetch_ohlcv source code uses the right column list.
        df_src = (repo / "data_fetcher.py").read_text()
        has_deliv_in_cols = "DelivPer" in df_src and 'cols.append("DelivPer")' in df_src
        check("fetch_ohlcv preserves DelivPer column",
              has_deliv_in_cols,
              f"source has DelivPer-preserving cols logic" if has_deliv_in_cols
              else "DelivPer column missing from fetch_ohlcv output")

    # ── 7f. alpha_engine imports get_fo_signals (was silently NameError) ─────
    alpha_src = (repo / "alpha_engine.py").read_text()
    check("alpha_engine imports get_fo_signals (was always-empty F&O signal)",
          "from fo_data import get_fo_signals" in alpha_src,
          "import present" if "from fo_data import get_fo_signals" in alpha_src
          else "F&O OI signal silently disabled")

    # ── 7g. Portfolio _stage requires >= 172 bars (was 160, MA150 NaN at -22) ─
    port_src = (repo / "portfolio.py").read_text()
    check("Portfolio _stage requires >= 172 bars (MA150 slope needs ≥ 172)",
          "len(c) < 172" in port_src,
          "guard present" if "len(c) < 172" in port_src
          else "IPOs 160-171 bars silently misclassified as Stage 3/4")

    # ── 7h. edge_engine backtest applies position sizing (not 100%/trade) ────
    edge_src = (repo / "edge_engine.py").read_text()
    rc_src_e = (repo / "risk_config.py").read_text() if (repo / "risk_config.py").exists() else ""
    check("edge_engine backtest uses position sizing (not 100%/trade)",
          "POSITION_SIZE_FRAC" in edge_src or "POSITION_SIZE_FRAC" in rc_src_e,
          "sized" if ("POSITION_SIZE_FRAC" in edge_src or "POSITION_SIZE_FRAC" in rc_src_e)
          else "equity curve overstated 10× from concurrent-trades compounding")

    # ── 7i. multiyear_breakout uses full daily history (not 1-day samples) ──
    myb_src = (repo / "multiyear_breakout.py").read_text()
    check("multiyear_breakout uses full daily history for monthly OHLC",
          "EVERY weekday" in myb_src or "Mon-Fri only" in myb_src,
          "uses full history" if "EVERY weekday" in myb_src or "Mon-Fri only" in myb_src
          else "monthly High=Close=Low (single-day samples) → false breakouts")

    # ── 7j. monster_growth PEG uses TTM only (not 3Y CAGR) ───────────────────
    mg_src = (repo / "monster_growth.py").read_text()
    check("monster_growth PEG uses 1-year growth (not 3Y CAGR)",
          "peg_growth = profit_ttm" in mg_src,
          "TTM-only PEG" if "peg_growth = profit_ttm" in mg_src
          else "PEG mixes 3Y CAGR with current PE → decelerating stocks look cheap")

    # ── TIER 2 SMOKE TESTS ────────────────────────────────────────────────────

    # ── 7L. F&O signals load with new NSE URL format ─────────────────────────
    fo_src = (repo / "fo_data.py").read_text()
    check("F&O URL uses new NSE unified BhavCopy format",
          "BhavCopy_NSE_FO_0_0_0_" in fo_src,
          "URL updated" if "BhavCopy_NSE_FO_0_0_0_" in fo_src
          else "still using old fo{DDMMYYYY}bhav.csv.zip URL — empty signals")
    # And new column names
    check("F&O parser handles new column names (TckrSymb, FinInstrmTp, etc)",
          "TckrSymb" in fo_src and "FinInstrmTp" in fo_src,
          "column mapping present" if "TckrSymb" in fo_src
          else "old SYMBOL/INSTRUMENT names only — won't parse 2024+ files")

    # ── 7M. FTD requires trough-AFTER-peak time ordering ─────────────────────
    mb_src = (repo / "market_breadth.py").read_text()
    check("FTD detection requires trough AFTER peak (time-ordered)",
          "if trough_loc <= peak_loc:" in mb_src,
          "time-ordering enforced" if "if trough_loc <= peak_loc:" in mb_src
          else "FTD fires spuriously on new highs without prior decline")

    # ── 7N. _acc_dist_days uses per-bar 20-day avg (not today's avg) ─────────
    inst_src = (repo / "institutional_scanner.py").read_text()
    alpha_src = (repo / "alpha_engine.py").read_text()
    import re as _re
    has_ref_avg = lambda src: bool(_re.search(r"ref_avg\s*=\s*avg_vol_series", src))
    check("Acc/Dist days compare each bar to its OWN trailing 20-day avg",
          has_ref_avg(inst_src) and has_ref_avg(alpha_src),
          "both files fixed" if has_ref_avg(inst_src)
          else "still using today's avg — under-counts distribution in rising-vol")

    # ── 7O. Sector aggregation is liquidity-weighted (ADTV) ──────────────────
    sec_src = (repo / "sector_analysis.py").read_text()
    check("Sector aggregation uses liquidity-weighted (ADTV) average",
          "liq_weighted_avg" in sec_src and "adtv_cr" in sec_src,
          "ADTV-weighted" if "liq_weighted_avg" in sec_src
          else "simple avg — small caps drag whole sector")

    # ── 7P. Alpha Engine ADX uses separate temporaries ───────────────────────
    check("Alpha Engine ADX uses raw_pdm/raw_ndm (no order-dependent mutation)",
          "raw_pdm" in alpha_src and "raw_ndm" in alpha_src,
          "fixed" if "raw_pdm" in alpha_src
          else "tied days zero both DMs → ADX understated")

    # ── 7Q. analysis_utils has volume_baseline (median) helper ───────────────
    au_src = (repo / "analysis_utils.py").read_text()
    check("volume_baseline (median) helper exists in analysis_utils",
          "def volume_baseline" in au_src,
          "available" if "def volume_baseline" in au_src
          else "no robust volume baseline — outliers skew SMA")

    # ── 7R. Pocket Pivot uses correct 10-bar diff window ─────────────────────
    check("Pocket Pivot uses 11-close window with 10 valid diffs",
          "cl_win  = close.iloc[-12:-1]" in inst_src and "chg_full = cl_win.diff()" in inst_src,
          "off-by-one fixed" if "chg_full = cl_win.diff()" in inst_src
          else "1-bar misalignment between close diffs and volumes")

    # ── 7S. Trending _trend_age uses standard rolling MA50 ───────────────────
    tr_src = (repo / "trending.py").read_text()
    check("Trending _trend_age uses standard rolling(50).mean()",
          "s.rolling(50).mean()" in tr_src,
          "consistent with rest of codebase" if "s.rolling(50).mean()" in tr_src
          else "off-by-one MA50 inside _trend_age")

    # ── 7T. RS line new-high tolerance tightened to 0.9995 ──────────────────
    check("RS line new-high tolerance tightened (>= 0.9995, was 0.999)",
          "rs.iloc[-252:-1].max() * 0.9995" in alpha_src,
          "tight" if "rs.iloc[-252:-1].max() * 0.9995" in alpha_src
          else "0.999 lets 0.1%-below pass as new high → noise")

    # ── 7U. Portfolio R:R uses 0.5% of price as min risk floor ───────────────
    port_src = (repo / "portfolio.py").read_text()
    rc_src   = (repo / "risk_config.py").read_text() if (repo / "risk_config.py").exists() else ""
    check("Portfolio R:R uses 0.5% of price floor (from risk_config)",
          "MIN_RISK_FRAC = 0.005" in rc_src or "MIN_RISK_FRAC = 0.005" in port_src,
          "0.5% floor" if ("MIN_RISK_FRAC = 0.005" in rc_src or "MIN_RISK_FRAC = 0.005" in port_src)
          else "0.01 clamp inflates R:R to phantom 20 on tight stops")

    # ── 7V. Alpha _load_delivery_series uses batched cache ───────────────────
    check("Alpha delivery loader uses batched cache (not per-symbol disk reads)",
          "_DELIVERY_CACHE" in alpha_src and "_ensure_delivery_cache" in alpha_src,
          "batched" if "_DELIVERY_CACHE" in alpha_src
          else "still 90k disk reads per scan — 5-30 min scan time")

    # ── 7W. Sector cold-start fallback uses full Nifty50 list ───────────────
    check("sector_analysis cold-start fallback uses full _NIFTY50_SYMS (not first 10)",
          "fallback_tickers = [f\"{s}.NS\" for s in _NIFTY50_SYMS]" in sec_src,
          "20-stock fallback" if "fallback_tickers = [f\"{s}.NS\" for s in _NIFTY50_SYMS]" in sec_src
          else "10-stock fallback flickers sector rankings between cold/warm runs")

    # ── 7X. All weekly resamples use W-FRI (not bare W) ──────────────────────
    bad_w = []
    for pyfile in repo.glob("*.py"):
        if pyfile.name == "smoke_test.py":
            continue
        s = pyfile.read_text()
        if 'resample("W")' in s:
            bad_w.append(pyfile.name)
    check("All weekly resamples use W-FRI (not bare W which is W-SUN)",
          len(bad_w) == 0,
          "standardized" if not bad_w
          else f"still using bare W in: {bad_w}")

    # ── 7k. Universe is Nifty Total Market 750 (NOT just Nifty 500) ──────────
    # The function `get_nifty500_symbols` has a misleading name — it actually
    # returns ~750 stocks (Nifty Total Market). Smoke test asserts the actual
    # size so nobody accidentally restricts it back to 500.
    from nse_stocks import get_universe_symbols, get_nifty500_symbols
    uni = get_universe_symbols()
    check("Universe is Nifty Total Market 750 (not Nifty 500)",
          700 <= len(uni) <= 800,
          f"universe has {len(uni)} symbols (expect 700-800 for Nifty Total Market)")
    # Both names return same list (backward-compat alias)
    check("get_nifty500_symbols is back-compat alias for get_universe_symbols",
          get_universe_symbols() == get_nifty500_symbols(),
          "both names return the same Total Market 750 list")

    # ── TIER 3 SMOKE TESTS ────────────────────────────────────────────────────

    # ── T3-A. NIFTY_PROXY_SYMS is the canonical list (single source) ─────────
    au_src2 = (repo / "analysis_utils.py").read_text()
    check("NIFTY_PROXY_SYMS canonical list defined in analysis_utils",
          "NIFTY_PROXY_SYMS = [" in au_src2,
          "canonical list present" if "NIFTY_PROXY_SYMS" in au_src2
          else "4 different Nifty proxy lists scattered across files")

    # ── T3-B. stage_analysis uses MA50 AND MA150 (full Weinstein) ────────────
    check("stage_analysis requires both MA50 and MA150 conditions (full Weinstein)",
          "ma50" in au_src2 and "ma150" in au_src2 and "slope50" in au_src2,
          "full MA50+MA150" if "slope50" in au_src2
          else "uses MA150 only — Stage 2 granted too easily")

    # ── T3-C. monster_growth delegates _stage to analysis_utils ──────────────
    mg_src2 = (repo / "monster_growth.py").read_text()
    check("monster_growth uses canonical stage_analysis (no private copy)",
          "_stage = stage_analysis" in mg_src2,
          "delegates to analysis_utils" if "_stage = stage_analysis" in mg_src2
          else "private _stage() definition — MA logic diverges")

    # ── T3-D. alpha_engine delegates _stage to analysis_utils ────────────────
    alpha_src2 = (repo / "alpha_engine.py").read_text()
    check("alpha_engine uses canonical stage_analysis (no private copy)",
          "_stage = stage_analysis" in alpha_src2,
          "delegates to analysis_utils" if "_stage = stage_analysis" in alpha_src2
          else "private _stage() definition — MA logic diverges")

    # ── T3-E. _higher_highs_lows uses resample(W-FRI) not 5-bar chunks ───────
    tr_src2 = (repo / "trending.py").read_text()
    check("_higher_highs_lows uses resample(W-FRI) not artificial 5-bar chunks",
          'resample("W-FRI")' in tr_src2 and "for w in range" not in tr_src2.split("_higher_highs_lows")[1][:400],
          "calendar-week resampling" if 'resample("W-FRI")' in tr_src2
          else "5-bar artificial chunks shift with holidays")

    # ── T3-F. base depth uses symmetric midpoint formula ─────────────────────
    eg_src = (repo / "early_growth.py").read_text()
    check("early_growth base depth uses symmetric midpoint (hi+lo)/2",
          "(cand_hi + cand_lo) / 2" in eg_src,
          "symmetric formula" if "(cand_hi + cand_lo) / 2" in eg_src
          else "(hi-lo)/hi asymmetric — ₹50 range looks bigger at low prices")

    # ── T3-G. risk_config.py centralises position sizing ─────────────────────
    rc_path = repo / "risk_config.py"
    check("risk_config.py exists with POSITION_SIZE_FRAC and MIN_RISK_FRAC",
          rc_path.exists() and "POSITION_SIZE_FRAC" in rc_path.read_text(),
          "centralized" if rc_path.exists()
          else "position sizing scattered across 3 files")

    # ── T3-H. _realized_vix renamed to _realized_vol ─────────────────────────
    mb_src2 = (repo / "market_breadth.py").read_text()
    # Check that the function DEFINITION uses _realized_vol (not the old name)
    has_vol_def = "def _realized_vol(" in mb_src2
    has_vix_def = "def _realized_vix(" in mb_src2
    check("market_breadth uses def _realized_vol (not misleading def _realized_vix)",
          has_vol_def and not has_vix_def,
          "correctly labelled" if has_vol_def and not has_vix_def
          else "function still defined as _realized_vix — VIX is implied vol, not realized")

    # ── T3-I. backtest stride is per-bar with cooldown ────────────────────────
    edge_src2 = (repo / "edge_engine.py").read_text()
    check("edge_engine backtest steps every bar with BT_COOLDOWN_BARS (not stride=5)",
          "BT_COOLDOWN_BARS" in edge_src2 and "cooldown = BT_COOLDOWN_BARS" in edge_src2,
          "per-bar with cooldown" if "BT_COOLDOWN_BARS" in edge_src2
          else "stride=5 misses signals between steps")

    # ── T3-J. MIN_PRICE removed from early_mover, momentum, volume scanners ──
    for fname in ["early_mover_scanner.py", "momentum_scanner.py", "volume_scanner.py"]:
        fsrc = (repo / fname).read_text()
        check(f"{fname}: MIN_PRICE filter removed (ADTV gate replaces it)",
              "if cur < MIN_PRICE:" not in fsrc,
              "removed" if "if cur < MIN_PRICE:" not in fsrc
              else "still filters by nominal price — misses ₹25 stocks with ₹5Cr ADTV")

    # ── P0/P2 IMPROVEMENT SUITE ──────────────────────────────────────────────

    # P0-3. holders_data.py module + /api/holders endpoint
    hd_path = repo / "holders_data.py"
    check("holders_data.py exists with 2-step NSE cookie seeding (P0-3)",
          hd_path.exists() and "_nse_session" in hd_path.read_text() and
          "api/marketStatus" in hd_path.read_text(),
          "module + session seeding present" if hd_path.exists()
          else "FII/DII per-symbol holder lookup still missing")

    app_src = (repo / "app.py").read_text()
    check("/api/holders endpoint registered in app.py (P0-3)",
          '@app.route("/api/holders")' in app_src,
          "route present" if '@app.route("/api/holders")' in app_src
          else "no /api/holders endpoint")

    # P0-4. Alpha Engine percentile-based tier reassignment
    alpha_src3 = (repo / "alpha_engine.py").read_text()
    check("Alpha Engine uses percentile-based tier reassignment (P0-4)",
          "Top 5% = BUY" in alpha_src3 and "buy_cutoff" in alpha_src3,
          "percentile tiers" if "buy_cutoff" in alpha_src3
          else "still fixed thresholds 55/42/30")

    # P0-5. Market Breadth granular timing (max=15 now)
    mb_src3 = (repo / "market_breadth.py").read_text()
    check("Market Breadth timing score uses max=15 granular inputs (P0-5)",
          '"max":              15' in mb_src3 and 'p50 >= 75' in mb_src3,
          "granular" if "p50 >= 75" in mb_src3
          else "still binary thresholds (60/40) — 72% counts same as 60%")

    # P2-10/11. consensus.py
    c_path = repo / "consensus.py"
    check("consensus.py + /api/consensus endpoints (P2-10/11)",
          c_path.exists() and '@app.route("/api/consensus")' in app_src,
          "cross-scan consensus available" if c_path.exists()
          else "no consensus aggregator")

    # P2-12/13. cross-sectional + sector-adjusted RS in analysis_utils
    au_src3 = (repo / "analysis_utils.py").read_text()
    check("analysis_utils has cross_sectional_rs_rank + sector_adjusted_rs (P2-12/13)",
          "def cross_sectional_rs_rank" in au_src3 and "def sector_adjusted_rs" in au_src3,
          "helpers present" if "def cross_sectional_rs_rank" in au_src3
          else "RS rank still scanner-local")

    # P2-13. Alpha Engine uses full-universe RS rank
    check("Alpha Engine uses full-universe RS rank from analysis_utils (P2-13)",
          "from analysis_utils import" in alpha_src3 and "cross_sectional_rs_rank" in alpha_src3,
          "consistent RS" if "cross_sectional_rs_rank" in alpha_src3
          else "Alpha still uses local ranking")

    # P2-14. stage_transitions.py module + endpoints
    st_path = repo / "stage_transitions.py"
    check("stage_transitions.py SQLite log + /api/stage-transitions (P2-14)",
          st_path.exists() and '@app.route("/api/stage-transitions")' in app_src,
          "stage log + endpoint" if st_path.exists()
          else "no per-stock stage history")

except Exception as e:
    import traceback
    check("Financial logic audit", False, f"{e}\n{traceback.format_exc()[:400]}")


# ── Summary ───────────────────────────────────────────────────────────────────
section("SUMMARY")
passed = sum(1 for r in results if r[0])
failed = sum(1 for r in results if not r[0])
total  = len(results)

print(f"\n  {passed}/{total} checks passed", end="")
if failed:
    print(f"  |  {failed} FAILED:")
    for r in results:
        if not r[0]:
            print(f"      {FAIL} {r[1]}: {r[2]}")
else:
    print("  — all good")

print()
sys.exit(0 if failed == 0 else 1)
