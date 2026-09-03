"""Unit: scripts/benchmark-generator/capture_spec.py — writes benchmark-spec.json.

That file is the contract every later stage reads: rl-integration-generator picks algorithms
off it, task-generator reads `gpu_sim` to size its smokes, reward-tune pre-flights on it. A
malformed or silently-empty spec fails much later and somewhere else, so the shape is pinned
here.
"""
import json
import subprocess
import sys

import pytest

from _pluginmeta import ROOT

SCRIPT = ROOT / "scripts" / "benchmark-generator" / "capture_spec.py"
BASE = ["--benchmark-name", "demo", "--language", "pytorch", "--gpu-sim", "true"]


def _run(repo, *args, expect_ok=True):
    r = subprocess.run([sys.executable, str(SCRIPT), "--repo", str(repo), *BASE, *args],
                       capture_output=True, text=True)
    if expect_ok:
        assert r.returncode == 0, r.stderr
    return r


def _spec(repo):
    return json.loads((repo / "harbor" / "benchmark-generator" / "benchmark-spec.json").read_text())


def test_writes_spec_with_the_fields_later_stages_read(tmp_path):
    _run(tmp_path, "--tasks", "Demo-A-v0,Demo-B-v0")
    spec = _spec(tmp_path)
    assert spec["benchmark_name"] == "demo"
    assert spec["gpu_sim"] is True
    assert spec["language"] == "pytorch"
    assert [t["id"] for t in spec["tasks"]] == ["Demo-A-v0", "Demo-B-v0"]


def test_gpu_sim_is_a_bool_not_the_string_true(tmp_path):
    """task-generator branches on `gpu_sim == true` to size NUM_ENVS; a string is truthy
    either way and would silently pick the wrong branch."""
    _run(tmp_path, "--tasks", "Demo-A-v0")
    assert _spec(tmp_path)["gpu_sim"] is True
    _run(tmp_path, "--tasks", "Demo-A-v0", "--gpu-sim", "false")
    assert _spec(tmp_path)["gpu_sim"] is False


def test_tasks_json_carries_per_task_detail(tmp_path):
    payload = json.dumps([
        {"id": "Demo-A-v0", "max_episode_steps": 500, "success_metric": "success"},
        {"id": "Demo-B-v0"},
    ])
    _run(tmp_path, "--tasks-json", payload)
    tasks = {t["id"]: t for t in _spec(tmp_path)["tasks"]}
    assert tasks["Demo-A-v0"]["max_episode_steps"] == 500
    assert set(tasks) == {"Demo-A-v0", "Demo-B-v0"}


def test_dry_run_prints_without_writing(tmp_path):
    r = _run(tmp_path, "--tasks", "Demo-A-v0", "--dry-run")
    assert json.loads(r.stdout)["benchmark_name"] == "demo"
    assert not (tmp_path / "harbor").exists(), "--dry-run wrote to disk"


def test_rerun_overwrites_rather_than_appending(tmp_path):
    _run(tmp_path, "--tasks", "Demo-A-v0,Demo-B-v0")
    _run(tmp_path, "--tasks", "Demo-C-v0")
    assert [t["id"] for t in _spec(tmp_path)["tasks"]] == ["Demo-C-v0"]


@pytest.mark.parametrize("missing", ["--benchmark-name", "--language", "--gpu-sim"])
def test_required_args_are_enforced(tmp_path, missing):
    args = [sys.executable, str(SCRIPT), "--repo", str(tmp_path), *BASE]
    i = args.index(missing)
    del args[i:i + 2]
    assert subprocess.run(args, capture_output=True).returncode != 0
