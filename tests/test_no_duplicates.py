"""
Forbidden-pattern grep tests.

Prevents the same bug class from reappearing in N scanner files. If a future
edit re-introduces a local copy of a canonical helper (ATR, split-adjust,
Nifty proxy list, etc.) outside analysis_utils.py, this test fails.

Run via:  python3 tests/test_no_duplicates.py
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]

# Files where the "canonical" pattern itself lives — they're allowed to
# contain the patterns we ban elsewhere.
_CANONICAL = {
    "analysis_utils.py",            # all canonical helpers live here
    "test_invariants.py",           # tests reference the helpers
    "test_no_duplicates.py",        # this file
    "smoke_test.py",                # smoke test mentions patterns in comments
}


_failures: list[str] = []


def _iter_py_files():
    for p in _REPO.glob("*.py"):
        if p.name in _CANONICAL:
            continue
        yield p
    for p in (_REPO / "tests").glob("*.py"):
        if p.name in _CANONICAL:
            continue
        yield p


def _grep(files, pattern: str, why: str, exclude_comments: bool = True):
    rx = re.compile(pattern)
    hits = []
    for f in files:
        try:
            for lineno, line in enumerate(f.read_text().splitlines(), start=1):
                stripped = line.lstrip()
                if exclude_comments and (stripped.startswith("#") or stripped.startswith('"""')):
                    continue
                if rx.search(line):
                    hits.append(f"  {f.relative_to(_REPO)}:{lineno}  {line.strip()[:120]}")
        except Exception:
            continue
    return hits


def check(name: str, pattern: str, why: str, max_allowed: int = 0):
    hits = _grep(list(_iter_py_files()), pattern, why)
    if len(hits) <= max_allowed:
        print(f"  OK   {name}  ({len(hits)} hit(s), allowed {max_allowed})")
        return
    print(f"  FAIL {name}  ({len(hits)} hits, allowed {max_allowed}):")
    for h in hits[:20]:
        print(h)
    _failures.append(f"{name}: {len(hits)} hits (limit {max_allowed}). {why}")


def main():
    print("== Forbidden-pattern checks ==")

    # 1. ATR must use canonical analysis_utils.atr (Wilder's EWM), never SMA.
    check(
        "no_local_rolling_ATR",
        r"tr\.rolling\([^)]*\)\.mean\(\)",
        "Use analysis_utils.atr() — Wilder's EWM — instead of tr.rolling(N).mean() SMA.",
    )

    # 2. Old <0.55 split-detection threshold is banned (it misses 3:2/4:3/5:4 bonuses).
    check(
        "no_old_split_threshold",
        r"\(\s*cur\s*/\s*prev\s*\)\s*<\s*0\.55",
        "Use analysis_utils.adjust_for_splits() which catches 3:2/4:3/5:4 bonuses.",
    )

    # 3. No hardcoded Nifty proxy lists outside analysis_utils.py.
    check(
        "no_local_nifty_proxy_list",
        r"\[\s*[\"']RELIANCE[\"']\s*,\s*[\"']TCS[\"']",
        "Import NIFTY_PROXY_SYMS from analysis_utils instead of defining a local list.",
    )

    # 4. No `len(c) > 21` / `> 63` / `> 126` / `> 252` (off-by-one for N-day returns
    #    where we then index iloc[-N]). Should be `>= N`.
    check(
        "no_off_by_one_return_window_gt_21",
        r"len\([a-zA-Z_][a-zA-Z_0-9]*\)\s*>\s*21\b",
        "iloc[-21] needs len >= 21, not len > 21. Use >= for return windows.",
    )
    check(
        "no_off_by_one_return_window_gt_63",
        r"len\([a-zA-Z_][a-zA-Z_0-9]*\)\s*>\s*63\b",
        "iloc[-63] needs len >= 63, not len > 63.",
    )
    check(
        "no_off_by_one_return_window_gt_126",
        r"len\([a-zA-Z_][a-zA-Z_0-9]*\)\s*>\s*126\b",
        "iloc[-126] needs len >= 126.",
    )
    check(
        "no_off_by_one_return_window_gt_252",
        r"len\([a-zA-Z_][a-zA-Z_0-9]*\)\s*>\s*252\b",
        "iloc[-252] needs len >= 252.",
    )

    if _failures:
        print(f"\n{len(_failures)} CHECK(S) FAILED:")
        for f in _failures:
            print(f"  - {f}")
        sys.exit(1)
    print("\nAll forbidden-pattern checks passed.")
    sys.exit(0)


if __name__ == "__main__":
    main()
