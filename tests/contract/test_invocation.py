"""Contract: frontmatter is valid YAML, and the invocation boundary is declared.

Two guards the tolerant line parser in `_pluginmeta` cannot provide:

1. Claude Code reads command/agent frontmatter as YAML. `_pluginmeta.load_frontmatter`
   is deliberately tolerant, so a malformed block still "loads" in the rest of the
   suite. This asserts the real thing parses — if it doesn't, EVERY field is lost,
   including the gating flag below.

2. `disable-model-invocation: true` makes a command user-invocable ONLY
   (https://code.claude.com/docs/en/slash-commands — "Control who invokes a skill").
   Two consequences, both enforced here:
     - the gated set is exactly GATED (no silent additions/removals), and
     - **no other command or agent may slash-invoke a gated command** — Claude
       cannot invoke it, so such a call site is dead on arrival. Chains that need
       a gated command's behavior must Read its body and execute it (the pattern
       `task-create.md` §6 already uses for the reward-tune flow).

Only `reset-workspace` is gated: it is the one irreversible operation
(`git reset --hard` + `git clean -fdx`). Everything else is a building block that
other harbor commands legitimately invoke — gating those breaks the chains.
"""
import re

import pytest
import yaml

from _pluginmeta import AGENTS, COMMANDS, ROOT

_FM_RE = re.compile(r"^---\n(.*?)\n---\n?", re.DOTALL)

# Commands Claude must never invoke on its own. Keep in sync with CLAUDE.md L2.
GATED = {"reset-workspace"}

# help.md is the surface listing — it documents every command's signature by design.
_INVOCATION_SCAN_SKIP = {"help.md"}


@pytest.mark.parametrize("path", COMMANDS + AGENTS,
                         ids=lambda p: f"{p.parent.name}/{p.name}")
def test_frontmatter_is_valid_yaml(path):
    m = _FM_RE.match(path.read_text(encoding="utf-8"))
    assert m, f"{path.relative_to(ROOT)}: no frontmatter block"
    try:
        loaded = yaml.safe_load(m.group(1))
    except yaml.YAMLError as e:
        pytest.fail(f"{path.relative_to(ROOT)}: frontmatter is not valid YAML: "
                    f"{str(e).splitlines()[0]}\n"
                    f"  (quote the value, or use a `>-` block scalar for prose "
                    f"containing ': ' or '[')")
    assert isinstance(loaded, dict), \
        f"{path.relative_to(ROOT)}: frontmatter must parse to a mapping"


def _gated_flag(name):
    path = ROOT / "commands" / f"{name}.md"
    assert path.exists(), f"commands/{name}.md is missing"
    fm = yaml.safe_load(_FM_RE.match(path.read_text(encoding="utf-8")).group(1))
    return fm.get("disable-model-invocation")


@pytest.mark.parametrize("name", sorted(GATED))
def test_gated_commands_are_not_model_invocable(name):
    assert _gated_flag(name) is True, (
        f"commands/{name}.md is irreversible and must set "
        f"`disable-model-invocation: true` so only the user can trigger it"
    )


@pytest.mark.parametrize("path", COMMANDS, ids=lambda p: p.name)
def test_no_unexpected_gating(path):
    """A command gated without being in GATED is probably a mistake: nothing else
    may invoke it, so any existing call site silently breaks."""
    if path.stem in GATED:
        return
    assert _gated_flag(path.stem) is not True, (
        f"commands/{path.stem}.md sets disable-model-invocation but is not in GATED. "
        f"Gating makes it uninvokable by other harbor commands — add it to GATED "
        f"and rewire any call sites to Read-and-execute, or drop the flag."
    )


@pytest.mark.parametrize("name", sorted(GATED))
def test_gated_commands_are_not_slash_invoked_internally(name):
    """Claude cannot invoke a gated command, so a slash call site is dead code."""
    # /harbor:<name> followed by an argument (key=...) on the same line
    rx = re.compile(rf"/harbor:{re.escape(name)}\b[^\n`|]*?\s[a-z_]+=")
    bad = []
    for f in COMMANDS + AGENTS:
        if f.stem == name or f.name in _INVOCATION_SCAN_SKIP:
            continue
        for i, line in enumerate(f.read_text(encoding="utf-8").split("\n"), 1):
            if rx.search(line):
                bad.append(f"{f.parent.name}/{f.name}:{i}")
    assert not bad, (
        f"/harbor:{name} is gated (Claude cannot invoke it) but is slash-invoked at: "
        f"{bad}. Replace with: Read commands/{name}.md and execute its body."
    )
