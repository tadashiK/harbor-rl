"""Unit: scripts/rl-integration-generator/discover_rl_tasks.py — task discovery fallbacks.

Contract: **always exits 0**, even with zero tasks found, and says in `source` which route
produced the answer. rl-integration-generator branches on that, so a silent fallback to a
weaker source would scaffold configs against the wrong task list.
"""
import importlib.util
import json
import subprocess
import sys

import pytest

from _pluginmeta import ROOT


def _importable(name):
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False

SCRIPT = ROOT / "scripts" / "rl-integration-generator" / "discover_rl_tasks.py"


def _run(repo):
    r = subprocess.run([sys.executable, str(SCRIPT), "--repo", str(repo)],
                       capture_output=True, text=True)
    assert r.returncode == 0, f"must always exit 0; got {r.returncode}\n{r.stderr}"
    return json.loads(r.stdout)


def _spec(repo, tasks):
    d = repo / "harbor" / "benchmark-generator"
    d.mkdir(parents=True, exist_ok=True)
    (d / "benchmark-spec.json").write_text(json.dumps({"tasks": tasks}))


def test_spec_is_preferred_and_echoed(tmp_path):
    _spec(tmp_path, [{"id": "Demo-A-v0"}, {"id": "Demo-B-v0"}])
    out = _run(tmp_path)
    assert out["source"] == "benchmark-spec.json"
    assert out["count"] == 2 and [t["id"] for t in out["tasks"]] == ["Demo-A-v0", "Demo-B-v0"]


def test_falls_back_to_hydra_task_configs(tmp_path):
    d = tmp_path / "configs" / "env" / "task"
    d.mkdir(parents=True)
    (d / "Cartpole.yaml").write_text("name: Cartpole\n")
    (d / "Ant.yaml").write_text("name: Ant\n")
    out = _run(tmp_path)
    assert out["source"] == "configs/**/task/*.yaml"
    assert sorted(t["id"] for t in out["tasks"]) == ["Ant", "Cartpole"]


def test_empty_spec_does_not_stop_the_fallback_chain(tmp_path):
    """`tasks: []` is the same as no spec — otherwise a stub spec masks real tasks."""
    _spec(tmp_path, [])
    d = tmp_path / "configs" / "task"
    d.mkdir(parents=True)
    (d / "Cartpole.yaml").write_text("name: Cartpole\n")
    assert _run(tmp_path)["source"] == "configs/**/task/*.yaml"


def test_corrupt_spec_degrades_instead_of_crashing(tmp_path):
    d = tmp_path / "harbor" / "benchmark-generator"
    d.mkdir(parents=True)
    (d / "benchmark-spec.json").write_text("{ not json")
    out = _run(tmp_path)
    assert out["ok"] is True


@pytest.mark.skipif(
    _importable("gymnasium") or _importable("gym"),
    reason="needs a bare interpreter: with gym importable the third fallback "
           "(gym.envs.registry) returns the stock envs, so 'nothing found' is unreachable",
)
def test_nothing_found_is_still_a_clean_zero(tmp_path):
    """'Nothing found' only exists when every route is empty — and route 3 reads the
    INTERPRETER's gym registry, not the repo. Run this with a benchmark venv and it reports
    50 stock envs (Acrobot-v1, Ant-v2, ...) for a repo that contains no tasks."""
    out = _run(tmp_path)
    assert out["ok"] is True and out["count"] == 0 and out["tasks"] == []
