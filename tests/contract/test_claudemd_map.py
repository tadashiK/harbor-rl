"""Contract: the CLAUDE.md architecture map stays in sync with disk.

Bidirectional: no CLAUDE.md reference points at a missing file, and every
command/agent on disk is documented.
"""
import re

import pytest

from _pluginmeta import AGENTS, COMMANDS, ROOT

CLAUDEMD = (ROOT / "CLAUDE.md").read_text(encoding="utf-8")


def test_claudemd_command_refs_exist():
    missing = sorted(
        r for r in set(re.findall(r"commands/([\w-]+)\.md", CLAUDEMD))
        if not (ROOT / "commands" / f"{r}.md").exists()
    )
    assert not missing, f"CLAUDE.md references missing command files: {missing}"


def test_claudemd_agent_refs_exist():
    missing = sorted(
        r for r in set(re.findall(r"agents/([\w-]+)\.md", CLAUDEMD))
        if not (ROOT / "agents" / f"{r}.md").exists()
    )
    assert not missing, f"CLAUDE.md references missing agent files: {missing}"


@pytest.mark.parametrize("path", COMMANDS, ids=lambda p: p.name)
def test_every_command_documented(path):
    name = path.stem
    assert f"commands/{name}.md" in CLAUDEMD or f"/harbor:{name}" in CLAUDEMD, \
        f"command '{name}' is not documented in CLAUDE.md"


@pytest.mark.parametrize("path", AGENTS, ids=lambda p: p.name)
def test_every_agent_documented(path):
    name = path.stem
    assert f"agents/{name}.md" in CLAUDEMD or f"`{name}`" in CLAUDEMD, \
        f"agent '{name}' is not documented in CLAUDE.md"


# --- ASCII tree listings -----------------------------------------------------
# The L4/L5 trees list bare filenames (`render_rl_suite.py`, `clone_task.py`, …)
# rather than paths, so the path-shaped checks above never see them. Without this,
# CLAUDE.md can name scripts/templates that no longer exist — which is exactly how
# the deleted rl-tuning-agent subsystem stayed documented after it stopped working.
_TREE_SUFFIXES = (".py", ".sh", ".template")


def _tree_filenames(text):
    """Bare filenames mentioned in CLAUDE.md.

    The lookbehind stops brace/glob notation from yielding fragments: without it
    `{run_random, render_random}.py.template` also matches as `py.template`, and
    `{ppo,sac,td3}{,.parallel}.yaml.template` as `yaml.template`.
    """
    return set(re.findall(r"(?<![\w.-])([\w-][\w.-]*(?:\.py|\.sh|\.template))\b", text))


# Named in CLAUDE.md but legitimately absent from the plugin tree: either rendered
# INTO the target benchmark repo, or a `<placeholder>`/glob stand-in.
_RENDERED_OUTPUTS = {
    "setup_uv.sh", "run_random.py", "render_random.py", "data_logger.py",
    "train.py", "eval.py", "render.py", "visualize.py", "env_wrapper.py",
    "actions.py", "actions_cfg.py", "rewards.py", "tune.py", "__init__.py",
    "env.py", "env_cfg.py", "joint_pos_env_cfg.py", "manager_based_env.py",
    "_env.py",      # from the `scripts/_<family>_env.py` glob
    "_isaaclab_env.py",  # rendered by /harbor:reward-add-log (Path B env factory)
    "launch.sh",    # rendered by /harbor:rl-sweep into the sweep dir
    "_verdict.py",  # the shared smoke recorder, rendered into <task_dir>/smokes/
}

# The archived section deliberately names files that no longer exist.
_ARCHIVED_HEADING = "### Historical / archived"


def test_claudemd_tree_filenames_exist():
    """Every plugin-shipped filename named in CLAUDE.md must exist on disk."""
    text = CLAUDEMD.split(_ARCHIVED_HEADING)[0]
    on_disk = {p.name for p in ROOT.rglob("*") if p.is_file() and ".git" not in p.parts}
    missing = sorted(
        n for n in _tree_filenames(text)
        if n.endswith(_TREE_SUFFIXES)
        and n not in on_disk
        and n not in _RENDERED_OUTPUTS
    )
    assert not missing, (
        f"CLAUDE.md names files that do not exist: {missing}\n"
        f"  (delete the listing, or add the file — if it is generated into the "
        f"target repo rather than shipped, add it to _RENDERED_OUTPUTS)"
    )
