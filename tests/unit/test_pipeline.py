"""Unit: scripts/test/pipeline.py — the L3 stage engine (skip / resume / cascade)."""
import json
import subprocess
import sys

from _pluginmeta import ROOT

PIPE = ROOT / "scripts" / "test" / "pipeline.py"


def _run(*args):
    return subprocess.run([sys.executable, str(PIPE), *args], capture_output=True, text=True)


def _plan(state, head="h1", task="T"):
    r = _run("plan", "--repo-head", head, "--task", task, "--state", str(state))
    assert r.returncode == 0, r.stderr
    return json.loads(r.stdout)


def test_dr_skipped_by_default():
    stages = _run("stages").stdout.split()
    assert "dr-generator" not in stages
    assert stages[0] == "env-install" and stages[-1] == "reset"


def test_empty_state_runs_all(tmp_path):
    d = _plan(tmp_path / "s.json")
    assert d["first_to_run"] == "env-install"
    assert all(s["action"] == "run" for s in d["stages"])


def test_resume_skips_marked_pass(tmp_path):
    state = tmp_path / "s.json"
    fp0 = _plan(state)["stages"][0]["fp"]
    _run("mark", "--state", str(state), "--stage", "env-install", "--status", "pass", "--fp", fp0)
    d = _plan(state)
    assert d["stages"][0]["action"] == "skip"
    assert d["first_to_run"] == "benchmark"


def test_changing_task_invalidates(tmp_path):
    state = tmp_path / "s.json"
    fp0 = _plan(state, task="A")["stages"][0]["fp"]
    _run("mark", "--state", str(state), "--stage", "env-install", "--status", "pass", "--fp", fp0)
    assert _plan(state, task="B")["stages"][0]["action"] == "run"  # different task → cache miss


def test_fail_status_reruns(tmp_path):
    state = tmp_path / "s.json"
    fp0 = _plan(state)["stages"][0]["fp"]
    _run("mark", "--state", str(state), "--stage", "env-install", "--status", "fail", "--fp", fp0)
    assert _plan(state)["stages"][0]["action"] == "run"
