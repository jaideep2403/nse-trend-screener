"""
Test runner — runs both invariant tests and forbidden-pattern checks.

Exit code 0 = all pass, 1 = any failure.

Usage:
  python3 tests/run_all.py
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def _run(script: str) -> bool:
    print(f"\n── {script} ────────────────────────────────────────────")
    p = subprocess.run(
        [sys.executable, str(HERE / script)],
        capture_output=False,
    )
    return p.returncode == 0


def main():
    ok1 = _run("test_invariants.py")
    ok2 = _run("test_no_duplicates.py")
    # Renders the real page per role and parses every inline <script> together,
    # the way the browser scopes them. Catches the cross-tab identifier collision
    # class that took the whole app down on 2026-07-26.
    ok3 = _run("test_page_js.py")

    print("\n══════════════════════════════════════════════════════════")
    if ok1 and ok2 and ok3:
        print("ALL TESTS PASSED")
        sys.exit(0)
    print("SOME TESTS FAILED")
    print(f"  invariants:     {'PASS' if ok1 else 'FAIL'}")
    print(f"  no_duplicates:  {'PASS' if ok2 else 'FAIL'}")
    print(f"  page_js:        {'PASS' if ok3 else 'FAIL'}")
    sys.exit(1)


if __name__ == "__main__":
    main()
