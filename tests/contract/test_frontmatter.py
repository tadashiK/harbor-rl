"""Contract: every command/agent has valid frontmatter."""
import pytest

from _pluginmeta import AGENTS, COMMANDS, KNOWN_MODELS, KNOWN_TOOLS, load_frontmatter


@pytest.mark.parametrize("path", COMMANDS, ids=lambda p: p.name)
def test_command_frontmatter(path):
    fm, body = load_frontmatter(path)
    assert fm is not None, f"{path.name}: missing YAML frontmatter"
    assert fm.get("description", "").strip(), f"{path.name}: missing/empty description"
    assert body.strip(), f"{path.name}: empty body"


@pytest.mark.parametrize("path", AGENTS, ids=lambda p: p.name)
def test_agent_frontmatter(path):
    fm, body = load_frontmatter(path)
    assert fm is not None, f"{path.name}: missing frontmatter"
    assert fm.get("name") == path.stem, \
        f"{path.name}: name '{fm.get('name')}' must equal filename stem '{path.stem}'"
    assert str(fm.get("description", "")).strip(), f"{path.name}: missing description"
    tools = fm.get("tools")
    assert isinstance(tools, list) and tools, f"{path.name}: tools must be a non-empty list"
    for t in tools:
        assert t in KNOWN_TOOLS or str(t).startswith("mcp__"), \
            f"{path.name}: unknown tool '{t}'"
    assert fm.get("model") in KNOWN_MODELS, \
        f"{path.name}: unknown model '{fm.get('model')}'"
    assert body.strip(), f"{path.name}: empty body"
