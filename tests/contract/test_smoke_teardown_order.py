"""Contract: a smoke records its verdict BEFORE it tears the simulator down.

`Verdict.done()` is the terminal call: it writes `"status": "pass"` into `<smoke>.verdict.json`
and prints the `S<N> OK: ...` line that IS the pass criterion. `sim_app.close()` hands control
to Kit's shutdown, which can take the process with it — so anything sequenced after the
teardown may simply never run.

The failure that causes is silent, which is why it needs a test rather than a convention:

  * every `[Cn] PASS` line still prints (checks flush eagerly, on construction and per check),
  * the process still exits 0,
  * but the verdict file is left reading `"status": "running"` with no summary,

which reads as a clean pass to anything watching stdout or the exit code. `_verdict.py`'s
docstring treats `"running"` as "the smoke died mid-flight" — an honest answer for a crash, and
a wrong one for a smoke that finished and merely never got to say so. `check_task_history.py`
is the backstop, but it only fires if the agent runs the history gate; an agent reading exit 0
and a screen of PASS lines gets a false green, and §3 (reset) and §4 (goal/termination) are
poor sections to be silently wrong about.

This drifted once already: five templates ordered it correctly while `smoke_s3`, `smoke_s3_render`
and `smoke_s4` did not, and `smoke_s1` carried a comment explaining the rule the other three were
breaking. Nothing failed when they broke it.

Scope is the terminal path. `smoke_s2` closes and exits early inside branch-local guards, each of
which calls `done()` first; what must hold is that the LAST `done()` precedes the LAST teardown.
A template that never tears down (e.g. one ending in `os._exit` after flushing) has nothing to
race and is exempt.
"""
import re

import pytest

from _pluginmeta import ROOT

SMOKES = ROOT / "knowledge" / "templates" / "task-generator" / "smokes"
TEMPLATES = sorted(SMOKES.glob("smoke_*.py.template"))

DONE = re.compile(r"\bV\.done\s*\(")
TEARDOWN = re.compile(r"\b(?:sim_app|simulation_app)\.close\s*\(")


def _code_lines(path):
    """(lineno, code) with comments stripped.

    Stripping matters: the fix ships an explanatory comment that names `sim_app.close()`, so a
    naive scan matches the prose and reports the very bug it is describing. That false positive
    is not hypothetical — it happened while auditing these files by hand.
    """
    out = []
    for i, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        code = raw.split("#", 1)[0]
        if code.strip():
            out.append((i, code))
    return out


def _last(pattern, lines):
    hits = [n for n, code in lines if pattern.search(code)]
    return hits[-1] if hits else None


def test_templates_are_discovered():
    """A rename that empties this glob would make every test below vacuously pass."""
    assert len(TEMPLATES) >= 6, f"expected the smoke template set, found {[p.name for p in TEMPLATES]}"


@pytest.mark.parametrize("path", TEMPLATES, ids=lambda p: p.name)
def test_verdict_is_recorded_before_teardown(path):
    lines = _code_lines(path)
    done, teardown = _last(DONE, lines), _last(TEARDOWN, lines)

    if done is None:
        pytest.skip(f"{path.name} records no verdict")
    if teardown is None:
        return  # nothing to race — no simulator teardown on the terminal path

    assert done < teardown, (
        f"{path.name}: V.done() at line {done} runs AFTER the teardown at line {teardown}. "
        "Kit's shutdown can kill the process there, so the verdict file is left reading "
        '"running" while every check still prints PASS and the process still exits 0. '
        "Move V.done() above env.close()/sim_app.close() (smoke_s1 is the worked example)."
    )


@pytest.mark.parametrize("path", TEMPLATES, ids=lambda p: p.name)
def test_every_teardown_is_preceded_by_a_verdict(path):
    """Branch-local exits count too — an early `return`/`exit` path that closes without a
    verdict loses the result exactly as the terminal path does."""
    lines = _code_lines(path)
    if _last(DONE, lines) is None:
        pytest.skip(f"{path.name} records no verdict")

    dones = [n for n, code in lines if DONE.search(code)]
    for n, code in lines:
        if TEARDOWN.search(code):
            assert any(d <= n for d in dones), (
                f"{path.name}: teardown at line {n} has no V.done() before it — "
                "a smoke reaching this path exits without recording its verdict."
            )
