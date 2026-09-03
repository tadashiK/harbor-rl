"""Contract: in-repo references resolve (no dangling links).

Catches the class of bug found pre-open-source (deleted decision-matrix still
referenced, a `claude-harbor/` link, a /harbor:<cmd> typo).

Three reference syntaxes, all checked:

  1. `${CLAUDE_PLUGIN_ROOT}/<path>` — how an agent addresses plugin files at runtime;
  2. a **bare** repo-relative path like `knowledge/references/adapt-first.md`, used freely in
     prose and just as capable of going stale — two of these were found dangling only when a
     human happened to ask, having pointed at a `templates/.../algorithms/` directory that
     never existed;
  3. `/harbor:<command>`.
"""
import re

import pytest

from _pluginmeta import AGENTS, COMMANDS, REFERENCES, ROOT

DOC_FILES = COMMANDS + AGENTS + REFERENCES

_PATH_RE = re.compile(r"\$\{CLAUDE_PLUGIN_ROOT\}/([^\s`)\"'>]+)")
_CMD_RE = re.compile(r"/harbor:([a-z][a-z-]*)")
# Bare repo-relative paths — but ONLY into trees that are unambiguously the plugin's.
#
# harbor's docs address two different roots with the same bare syntax: the plugin, and the
# target benchmark repo it writes into. `scripts/` and `tests/` exist in BOTH — `scripts/
# run_random.py` and `tests/test_envs.py` are files in the user's repo, not here — so they
# cannot be checked this way. These four have no counterpart on the benchmark side.
#
# Anchored on a non-path char so the `knowledge/x` inside `${CLAUDE_PLUGIN_ROOT}/knowledge/x`
# isn't matched twice.
_BARE_RE = re.compile(
    r"(?<![\w/.$-])((?:knowledge|agents|commands|hooks)/[\w./-]+\.(?:md|py|sh|json|yaml|template))"
)


def _is_placeholder(ref):
    return any(c in ref for c in "<*{")


@pytest.mark.parametrize("path", DOC_FILES, ids=lambda p: p.relative_to(ROOT).as_posix())
def test_references_resolve(path):
    text = path.read_text(encoding="utf-8")
    missing = []

    for ref in _PATH_RE.findall(text):
        if _is_placeholder(ref):
            continue
        clean = ref.rstrip(".,:;")
        if not (ROOT / clean).exists():
            missing.append(f"${{CLAUDE_PLUGIN_ROOT}}/{clean}")

    for ref in set(_BARE_RE.findall(text)):
        if _is_placeholder(ref):
            continue
        if not (ROOT / ref).exists():
            missing.append(ref)

    for cmd in set(_CMD_RE.findall(text)):
        if not (ROOT / "commands" / f"{cmd}.md").exists():
            missing.append(f"/harbor:{cmd}")

    assert not missing, f"{path.relative_to(ROOT)}: dangling reference(s): {missing}"
