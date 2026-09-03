"""Contract: dispatch depth stays at 2 (main -> orchestrator -> worker).

CLAUDE.md constraint #2. Claude Code lets a subagent spawn its own subagents (three
layers below main by default, v2.1.219+), so "subagents don't nest" is no longer
enforced by the platform — it has to be enforced here.

Exactly one agent may dispatch: `reward-tuning-agent`, whose whole purpose is to keep
implementation noise out of the context that designs the next reward. Every other agent
is a leaf. Two failure modes this catches:

  - a leaf agent quietly gaining `Agent` in its `tools`, pushing depth to 3 and putting
    the grandchild's output somewhere nobody reads;
  - the dispatcher LOSING `Agent`, which does not error — it silently degrades into
    doing the candidates' work in its own context, i.e. the exact contamination the
    split exists to remove.
"""
import pytest

from _pluginmeta import AGENTS, ROOT, load_frontmatter

DISPATCHERS = {"reward-tuning-agent"}
WORKERS = {"reward-tuning-agent": "reward-candidate-agent"}


def _tools(path):
    fm, _ = load_frontmatter(path)
    return set((fm or {}).get("tools") or [])


@pytest.mark.parametrize("path", AGENTS, ids=lambda p: p.name)
def test_only_declared_dispatchers_carry_the_agent_tool(path):
    has_agent = "Agent" in _tools(path)
    should = path.stem in DISPATCHERS
    assert has_agent == should, (
        f"{path.name}: tools has Agent={has_agent} but the depth policy says {should}. "
        f"Dispatchers are exactly {sorted(DISPATCHERS)} (CLAUDE.md constraint #2)."
    )


@pytest.mark.parametrize("parent,child", sorted(WORKERS.items()))
def test_each_dispatcher_has_its_worker_and_dispatches_it(parent, child):
    child_path = ROOT / "agents" / f"{child}.md"
    assert child_path.exists(), f"{parent} dispatches {child}, which does not exist"
    body = (ROOT / "agents" / f"{parent}.md").read_text(encoding="utf-8")
    assert child in body, f"{parent} declares no dispatch of {child}"


@pytest.mark.parametrize("parent,child", sorted(WORKERS.items()))
def test_workers_are_leaves(parent, child):
    assert "Agent" not in _tools(ROOT / "agents" / f"{child}.md"), (
        f"{child} is a worker at depth 2; giving it Agent would make depth 3"
    )
