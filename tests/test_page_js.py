"""Whole-page JavaScript integrity checks, run per ROLE.

Why this exists (2026-07-26): the Advisor tab declared `let _advData`, which the
Advanced Setups tab had already declared inside index.html's MAIN script block.
Two top-level `let`s with the same name share one global lexical scope, so the
page threw `SyntaxError: Identifier '_advData' has already been declared` and the
main script died — taking the header, the tab routing and effectively the whole
app with it.

Three things made it slip through:
  • the new tab's script was syntax-checked IN ISOLATION, where it is valid;
  • the collision only exists on the ADMIN page, because the tab is owner-gated,
    so a demo-session browser check showed a completely clean console;
  • nothing compared identifiers ACROSS the ~14 inline <script> blocks.

So the check has to be: render the real page, for each role, concatenate every
inline block the way the browser scopes them, and parse THAT. Duplicate element
ids are checked at the same time — `getElementById` silently returns the first
match, so a collision there means one tab quietly drives another tab's controls.

Requires node on PATH; skips (does not fail) if node is unavailable.
"""

from __future__ import annotations

import collections
import os
import re
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROLES = [("admin", "jai"), ("demo", "demo")]
_SCRIPT_RE = re.compile(r'<script(?![^>]*\bsrc=)[^>]*>(.*?)</script>', re.S)
_DECL_RE = re.compile(r'^(?:let|const|var|function|class)\s+([A-Za-z_$][\w$]*)', re.M)
_ID_RE = re.compile(r'id="([\w-]+)"')


def _render(role: str, user: str) -> str:
    import app as A
    c = A.app.test_client()
    with c.session_transaction() as s:
        s["user"], s["role"] = user, role
    return c.get("/").get_data(as_text=True)


def _have_node() -> bool:
    try:
        subprocess.run(["node", "--version"], capture_output=True, timeout=10)
        return True
    except Exception:
        return False


def check_role(role: str, user: str) -> list[str]:
    failures: list[str] = []
    html = _render(role, user)
    blocks = _SCRIPT_RE.findall(html)

    # 1. duplicate top-level declarations across blocks (the real bug)
    decl: dict[str, list[int]] = collections.defaultdict(list)
    for i, b in enumerate(blocks):
        for m in _DECL_RE.finditer(b):
            decl[m.group(1)].append(i)
    for name, where in sorted(decl.items()):
        if len(where) > 1:
            failures.append(
                f"[{role}] duplicate top-level declaration '{name}' in script blocks "
                f"{where} — a repeated `let`/`const` throws and kills the page")

    # 2. duplicate element ids — getElementById returns only the first
    ids = collections.Counter(_ID_RE.findall(html))
    for eid, n in sorted(ids.items()):
        if n > 1:
            failures.append(f"[{role}] duplicate element id '{eid}' appears {n}× "
                            f"— getElementById will silently target the wrong node")

    # 3. tab panels must be shown/hidden by the .active CLASS, never inline style.
    # showTab() toggles `.active`, and `.tab-content{display:none}` /
    # `.tab-content.active{display:block}` do the rest. An inline `style="display:none"`
    # on the root beats the class rule, so the tab activates but stays invisible —
    # a completely blank panel with no error anywhere. Cost a full round-trip on
    # 2026-07-26 with the Advisor tab.
    for m in re.finditer(r'<div\b[^>]*id="(tab-[\w-]+)"[^>]*>', html):
        tag, tab_id = m.group(0), m.group(1)
        sm = re.search(r'style="([^"]*)"', tag)
        if sm and re.search(r'\bdisplay\s*:', sm.group(1)):
            failures.append(
                f"[{role}] tab root '{tab_id}' sets inline display "
                f"({sm.group(1).strip()}) — inline style beats '.tab-content.active', "
                f"so the tab will activate but render blank. Use the class only.")

    # 4. the page's JS must parse as the browser scopes it: all blocks, one scope
    if _have_node():
        js = "\n;\n".join(blocks)
        tmp = tempfile.NamedTemporaryFile("w", suffix=".js", delete=False)
        tmp.write(js)
        tmp.close()
        try:
            r = subprocess.run(["node", "--check", tmp.name],
                               capture_output=True, text=True, timeout=60)
            if r.returncode != 0:
                first = next((l for l in r.stderr.splitlines() if "Error" in l), "parse error")
                failures.append(f"[{role}] full-page JS does not parse: {first.strip()}")
        finally:
            os.unlink(tmp.name)
    else:
        print("  (node not found — skipping the parse check)")

    print(f"  {role:<6} blocks={len(blocks):<3} globals={len(decl):<4} ids={len(ids)}")
    return failures


# ─────────────────────────────────────────────────────────────────────────────
# ITEM 8 — design-system guard.
#
# Items 1/3/4/5/6/7 removed 384 hardcoded colours, 944 hardcoded font sizes, a
# duplicate card system, a duplicate empty-state system and 344 redundant table-header
# declarations. Nothing stops the next edit from re-introducing them one at a time,
# which is exactly how the drift happened the first time. These checks fail the build
# instead, and they assert the WCAG contrast that the tokens exist to guarantee.
# ─────────────────────────────────────────────────────────────────────────────

_SRC_GLOBS = ("templates/*.html", "static/*.js")

# A hex here is a DEFINITION or a deliberate identity, never a status colour.
_TOKEN_DEF = re.compile(r"^\s*--[a-z0-9-]+\s*:")
_ICON_HUE = re.compile(r"\.ic:has\(use")
_HEX = re.compile(r"#[0-9a-fA-F]{3,8}\b")
_FONT_PX = re.compile(r"font-size:\s*[0-9.]+px")
_RETIRED = ("metric-card", "metric-label", "metric-val", "empty-state")


def _sources() -> list[str]:
    import glob
    out: list[str] = []
    for g in _SRC_GLOBS:
        out += sorted(glob.glob(os.path.join(ROOT, g)))
    return out


def _relative_luminance(hex_colour: str) -> float:
    h = hex_colour.lstrip("#")
    if len(h) == 3:
        h = "".join(c * 2 for c in h)
    ch = [int(h[i:i + 2], 16) / 255 for i in (0, 2, 4)]
    ch = [c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4 for c in ch]
    return 0.2126 * ch[0] + 0.7152 * ch[1] + 0.0722 * ch[2]


def contrast(fg: str, bg: str) -> float:
    a, b = sorted((_relative_luminance(fg), _relative_luminance(bg)), reverse=True)
    return (a + 0.05) / (b + 0.05)


def _tokens(css: str, scope: str) -> dict[str, str]:
    """Resolve `--name: #hex` definitions inside a `:root`-like block."""
    m = re.search(re.escape(scope) + r"\s*\{(.*?)\n\s*\}", css, re.S)
    body = m.group(1) if m else ""
    return {k: v for k, v in re.findall(r"--([a-z0-9-]+)\s*:\s*(#[0-9a-fA-F]{3,8})", body)}


# The STATUS palette — the greens/reds/golds that must follow the theme toggle, plus
# the near-black that was being pinned onto table headers. Item 1 mapped exactly these;
# a hardcoded one here is a theme bug, so it is a hard failure.
_STATUS_HEX = {
    "#10b981", "#22c55e", "#16a34a", "#34d399", "#059669", "#047857",
    "#ef4444", "#dc2626", "#f43f5e", "#f87171", "#b91c1c",
    "#f59e0b", "#eab308", "#facc15", "#fbbf24", "#d97706", "#92400e",
    "#111", "#111111",
}

# Neutral/brand hexes (surfaces, borders, one-off brand tints) were NOT in item 1's
# scope. Failing all of them would leave the suite permanently red and therefore
# ignored. Instead they are RATCHETED: the count may fall, never rise.
_NEUTRAL_HEX_BUDGET = 196

_COMMENT = re.compile(r"/\*.*?\*/|<!--.*?-->", re.S)

# `var(--red)22` — produced by search-and-replacing the 6-digit PREFIX of an 8-digit
# hex like #ef444422. It is not a colour: at computed-value time it is guaranteed-
# invalid, so the browser DROPS the declaration and the tint silently disappears.
# This shipped 28 times and no test noticed, because the page still parsed and every
# JS check passed. Use color-mix(in srgb, var(--tok) N%, transparent) instead.
_VAR_PLUS_ALPHA = re.compile(r"var\(--[a-z0-9-]+\)[0-9a-fA-F]{2}")


def check_design_system() -> list[str]:
    failures: list[str] = []
    neutral = 0

    for path in _sources():
        rel = os.path.relpath(path, ROOT)
        # Strip comments first — an explanatory comment naming a retired class or a
        # hex is documentation, not a live declaration.
        text = _COMMENT.sub(lambda m: "\n" * m.group(0).count("\n"), open(path, encoding="utf-8").read())
        for i, line in enumerate(text.split("\n"), 1):
            # trailing `// …` comments too — a comment that mentions a retired class is
            # documentation, not a live usage
            line = re.sub(r"(?<!:)//.*$", "", line)
            if not (_TOKEN_DEF.match(line) or _ICON_HUE.search(line)):
                for h in _HEX.findall(line):
                    if h.lower() in _STATUS_HEX:
                        failures.append(
                            f"[design] hardcoded STATUS colour {h} at {rel}:{i} — use the "
                            f"theme token (var(--green/--red/--gold/…)) so it follows the "
                            f"theme toggle and keeps its WCAG contrast")
                    else:
                        neutral += 1
                for f in _FONT_PX.findall(line):
                    failures.append(f"[design] hardcoded {f.strip()} at {rel}:{i} — "
                                    f"use a --fs-* scale token")
            for bad in _VAR_PLUS_ALPHA.findall(line):
                failures.append(
                    f"[design] invalid colour '{bad}' at {rel}:{i} — a var() with a hex "
                    f"alpha suffix is not a colour; the browser drops the declaration. "
                    f"Use color-mix(in srgb, var(--tok) N%, transparent)")
            for dead in _RETIRED:
                if dead in line:
                    failures.append(f"[design] retired class '{dead}' at {rel}:{i} — "
                                    f"use .stat-card / .state instead")

    if neutral > _NEUTRAL_HEX_BUDGET:
        failures.append(f"[design] neutral hardcoded colours rose to {neutral}, above the "
                        f"ratchet of {_NEUTRAL_HEX_BUDGET} — tokenise the new ones, or "
                        f"lower the budget deliberately if you removed some")

    # WCAG AA (4.5:1) for every status token, in BOTH themes. This is the property the
    # tokens exist to provide; without it the mapping work could silently regress.
    css = open(os.path.join(ROOT, "templates/index.html"), encoding="utf-8").read()
    dark = _tokens(css, ":root")
    light = {**dark, **_tokens(css, "body.light")}      # light overrides only some
    if _tokens(css, "body.light") == {}:
        failures.append("[a11y] could not resolve the light-theme token block — the "
                        "contrast check would silently test dark twice")
    for theme, toks in (("dark", dark), ("light", light)):
        bg = toks.get("surface") or toks.get("bg")
        for name in ("green", "green2", "red", "red2", "gold", "gold2"):
            fg = toks.get(name)
            if not fg or not bg:
                continue
            ratio = contrast(fg, bg)
            print(f"  {theme:<5} --{name:<7} {fg} on {bg}  {ratio:5.2f}:1  "
                  f"{'ok' if ratio >= 4.5 else 'FAIL'}")
            if ratio < 4.5:
                failures.append(f"[a11y] --{name} ({fg}) on the {theme} surface ({bg}) is "
                                f"{ratio:.2f}:1, below the WCAG AA floor of 4.5:1")

    print(f"  neutral hardcoded colours={neutral} (budget {_NEUTRAL_HEX_BUDGET})")
    return failures


def main() -> int:
    print("== Whole-page JS integrity (per role) ==")
    failures: list[str] = []
    for role, user in ROLES:
        try:
            failures += check_role(role, user)
        except Exception as e:
            failures.append(f"[{role}] could not render page: {e}")
    print("\n== Design-system guard (tokens + WCAG AA contrast) ==")
    failures += check_design_system()
    if failures:
        print("\nFAILURES:")
        for f in failures:
            print("  ✗", f)
        return 1
    print("\nPage JS is clean for every role.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
