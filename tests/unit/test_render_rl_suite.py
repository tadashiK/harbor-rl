"""Unit: scripts/rl-integration-generator/render_rl_suite.py — the RL scaffold renderer.

Renders the train/eval/render scripts, the Hydra configs, and rl-suite-spec.json. Two things
downstream depends on and would fail late if wrong: every rendered .py must actually compile,
and the spec's key path must be `algorithm_source.slug` (the drift `resolve_suite.py` exists
to prevent — see test_resolve_suite.py).
"""
import json
import subprocess
import sys

import pytest

from _pluginmeta import ROOT

SCRIPT = ROOT / "scripts" / "rl-integration-generator" / "render_rl_suite.py"


@pytest.fixture
def repo(tmp_path):
    """A repo that has already been through dependency-generator — the script's prerequisite."""
    (tmp_path / ".venv" / "bin").mkdir(parents=True)
    (tmp_path / ".venv" / "bin" / "python").write_text("#!/bin/sh\n")
    d = tmp_path / "harbor" / "dependency-generator"
    d.mkdir(parents=True)
    (d / "setup_uv.sh").write_text("#!/usr/bin/env bash\n")
    return tmp_path


def _run(repo, *extra, source="custom_torch", tasks="Demo-A-v0", expect_ok=True):
    r = subprocess.run(
        [sys.executable, str(SCRIPT), "--repo", str(repo),
         "--algorithm-source", source, "--tasks", tasks, *extra],
        capture_output=True, text=True)
    if expect_ok:
        assert r.returncode == 0, r.stderr
    return r


def _spec(repo):
    return json.loads(
        (repo / "harbor" / "rl-integration-generator" / "rl-suite-spec.json").read_text())


def test_renders_a_compiling_training_tree(repo):
    _run(repo)
    slug = _spec(repo)["algorithm_source"]["slug"]
    scripts = repo / "harbor" / "scripts" / "rl" / slug
    rendered = sorted(scripts.glob("*.py"))
    assert {p.name for p in rendered} >= {"train.py", "eval.py", "render.py", "env_wrapper.py"}
    for p in rendered:
        text = p.read_text()
        assert "{{" not in text, f"unrendered placeholder left in {p.name}"
        compile(text, str(p), "exec")


def test_spec_uses_the_nested_slug_key_path(repo):
    """`algorithm_source.slug`, NOT a top-level `algorithm_slug` — resolve_suite.py and every
    caller read the nested path."""
    _run(repo)
    spec = _spec(repo)
    assert "slug" in spec["algorithm_source"]
    assert "algorithm_slug" not in spec, "flat key path reintroduced"


def test_configs_are_rendered_for_each_algorithm(repo):
    _run(repo, "--algorithms", "ppo,sac")
    cfg = repo / "harbor" / "configs" / "rl"
    names = {p.name for p in cfg.glob("*.yaml")}
    assert {"ppo.yaml", "sac.yaml"} <= names
    assert not any(n.startswith("td3") for n in names), "rendered an algorithm not requested"


def test_tasks_land_in_the_spec(repo):
    _run(repo, tasks="Demo-A-v0,Demo-B-v0")
    assert [t["id"] for t in _spec(repo)["tasks"]] == ["Demo-A-v0", "Demo-B-v0"]


def test_rerun_is_idempotent(repo):
    _run(repo)
    first = _spec(repo)["algorithm_source"]["slug"]
    _run(repo)
    assert _spec(repo)["algorithm_source"]["slug"] == first


@pytest.mark.parametrize("source", ["custom_torch", "stable_baseline3"])
def test_each_algorithm_source_renders(repo, source):
    _run(repo, source=source)
    slug = _spec(repo)["algorithm_source"]["slug"]
    assert (repo / "harbor" / "scripts" / "rl" / slug / "train.py").is_file()


def test_unknown_algorithm_source_is_refused(repo):
    r = _run(repo, source="not_a_source", expect_ok=False)
    assert r.returncode != 0


def test_preflight_refuses_a_repo_without_an_env(tmp_path):
    """Rendering into a repo dependency-generator never touched would produce a training
    tree with no interpreter to run it — fail loudly instead."""
    r = _run(tmp_path, expect_ok=False)
    assert r.returncode != 0
    assert "dependency-generator" in r.stderr
    assert not (tmp_path / "harbor" / "scripts").exists()
