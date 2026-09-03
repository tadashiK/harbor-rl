"""Contract: the L4 tool layer stays uniform and its test coverage only improves.

harbor's "tools" are plain CLI scripts an agent calls with Bash. That is a deliberate choice
over MCP — the heavy operations are backgrounded GPU jobs whose state must survive a killed
agent, which is what files and exit codes give you. The cost of that choice is that nothing
validates a script's shape for you, so it is validated here.

Four rules (CLAUDE.md → L4):

  1. argparse, so `--help` works and the agent can discover the interface;
  2. JSON to stdout whenever an agent parses the output;
  3. exit 0 = the tool ran; non-zero = bad input, it could not run at all;
  4. emit a verdict object even when the answer is "it failed".

Rules 3-4 are the subtle pair. `score_iter.py` is the worked example: no metrics.jsonl means
exit 0 with `{"success_rate": null, "gate": "no_metrics"}` — the tool worked, the answer is
"ungradable" — while a missing design.json exits non-zero, because the CALLER passed
something broken. Backwards, and an ungradable candidate looks like a broken script.

The coverage rule is a ratchet: `UNTESTED_DEBT` may only shrink. New scripts must ship with a
test; the existing gaps do not block anything, but they cannot grow and cannot be forgotten.
"""
import re
import subprocess
import sys

import pytest

from _pluginmeta import ROOT

# `os.environ.get("CLAUDE_PLUGIN_ROOT")` / `os.environ["CLAUDE_PLUGIN_ROOT"]` — a runtime
# lookup. The bare name in a docstring is fine and stays.
_ENV_LOOKUP = re.compile(r"""environ(?:\.get)?\s*[\(\[]\s*["']CLAUDE_PLUGIN_ROOT""")
_PARENTS = re.compile(r"parents\[(\d)\]")

SCRIPTS = sorted(
    p for p in (ROOT / "scripts").rglob("*.py")
    if "__pycache__" not in p.parts and not p.name.startswith("_")
)

# Scripts that CANNOT be unit-tested in CI, with the reason. Not debt — a statement of fact.
UNTESTABLE = {
    "reward-add-log/sanity_check.py":
        "imports the repo's patched env helper and steps a live env",
    "reward-add-log/sanity_check_isaaclab.py":
        "needs isaaclab + torch + a running simulator",
    "plot/render_plot.py":
        "needs plotly/scipy and live W&B runs to plot",
}
# `smoke_uv.py` and `list_tasks.py` used to sit here. Both shell into a real .venv for the
# work itself, but the DECISIONS around it — which imports to probe, how tier verdicts
# combine, which task family a repo is, detection order — are pure and now covered. The
# venv-bound listers return None when their package is missing, which is what CI provides.

# Scripts with no unit test yet. THIS SET MAY ONLY SHRINK — delete a line when you add a test.
UNTESTED_DEBT = set()


def _rel(p):
    return p.relative_to(ROOT / "scripts").as_posix()


def _has_test(script):
    """A unit test that names this script's module."""
    stem = script.stem
    return any(stem in t.read_text(encoding="utf-8")
               for t in (ROOT / "tests" / "unit").glob("test_*.py"))


def _untested():
    return {_rel(p) for p in SCRIPTS if not _has_test(p) and _rel(p) not in UNTESTABLE}


def _depths(script):
    return {int(d) for d in _PARENTS.findall(script.read_text(encoding="utf-8"))}


_ROOT_FROM_FILE = [p for p in SCRIPTS if _depths(p)]


@pytest.mark.parametrize("script", SCRIPTS, ids=_rel)
def test_script_uses_argparse(script):
    text = script.read_text(encoding="utf-8")
    assert "argparse" in text, (
        f"{_rel(script)} has no argparse — an agent cannot discover its interface with --help"
    )


@pytest.mark.parametrize("script", SCRIPTS, ids=_rel)
def test_script_has_a_module_docstring(script):
    text = script.read_text(encoding="utf-8").lstrip()
    body = text.split("\n", 1)[1] if text.startswith("#!") else text
    assert body.lstrip().startswith(('"""', "'''")), (
        f"{_rel(script)} has no module docstring — it is the first thing a reader "
        f"(and --help) sees"
    )


@pytest.mark.parametrize(
    "script",
    [p for p in SCRIPTS if _rel(p) not in UNTESTABLE],
    ids=lambda p: _rel(p),
)
def test_help_exits_zero(script):
    r = subprocess.run([sys.executable, str(script), "--help"],
                       capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, f"{_rel(script)} --help failed:\n{r.stderr}"


def test_no_new_untested_scripts():
    """The ratchet: a script may not join the repo without a unit test."""
    new = _untested() - UNTESTED_DEBT
    assert not new, (
        f"new script(s) with no unit test: {sorted(new)}. Add a test under tests/unit/, "
        f"or — if it genuinely cannot be tested in CI — add it to UNTESTABLE with a reason."
    )


def test_debt_list_has_no_stale_entries():
    """The ratchet only tightens: once a script gets a test, its debt entry must go."""
    stale = UNTESTED_DEBT - _untested()
    assert not stale, (
        f"these now have tests — delete them from UNTESTED_DEBT: {sorted(stale)}"
    )


def test_untestable_entries_still_exist():
    """An UNTESTABLE entry for a deleted script hides that the exemption is unused."""
    missing = {k for k in UNTESTABLE if not (ROOT / "scripts" / k).is_file()}
    assert not missing, f"UNTESTABLE names scripts that no longer exist: {sorted(missing)}"


# --- plugin-root resolution --------------------------------------------------
#
# The tool layer is host-independent: a script finds its own tree from __file__, never from
# ambient environment. That is what lets these scripts run under Codex, from a bare shell, or
# in CI — none of which set CLAUDE_PLUGIN_ROOT. Two ways to lose it, both guarded here.

@pytest.mark.parametrize("script", SCRIPTS, ids=_rel)
def test_no_script_reads_the_plugin_root_from_the_environment(script):
    """`CLAUDE_PLUGIN_ROOT` is ambient state: a stale or foreign value would silently win and
    point the script at another plugin's tree. Callers needing a different root pass an
    explicit `--plugin-root`. The name may still appear in docstrings — that is how the
    markdown layer refers to paths — but not in a runtime lookup."""
    hit = _ENV_LOOKUP.search(script.read_text(encoding="utf-8"))
    assert not hit, (
        f"{_rel(script)} reads CLAUDE_PLUGIN_ROOT at runtime ({hit.group(0)!r}). Resolve the "
        f"tree from __file__ instead, or take an explicit --plugin-root."
    )


@pytest.mark.parametrize("script", _ROOT_FROM_FILE, ids=_rel)
def test_file_based_root_resolution_lands_on_the_plugin_root(script):
    """`parents[2]` is depth-coupled: correct at `scripts/<owner>/<file>.py`, silently wrong
    if a script is ever nested deeper. Verify the arithmetic against the real root."""
    for depth in sorted(_depths(script)):
        assert script.resolve().parents[depth] == ROOT, (
            f"{_rel(script)} uses parents[{depth}], which resolves to "
            f"{script.resolve().parents[depth]} — not the plugin root {ROOT}. "
            f"Its depth under scripts/ must have changed."
        )
