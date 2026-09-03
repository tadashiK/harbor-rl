"""Unit: scripts/rl-tricks/apply_trick.py — applicability gate + patch application."""
import json
import subprocess
import sys

import pytest

from _pluginmeta import ROOT

pytest.importorskip("yaml")
import yaml  # noqa: E402

SCRIPT = ROOT / "scripts" / "rl-tricks" / "apply_trick.py"
TRICK = "value_norm_torch"  # backend custom_torch, algorithms [ppo]
TRICK_DIR = ROOT / "knowledge" / "templates" / "rl-tricks" / TRICK


def _run(repo, *args):
    return subprocess.run(
        [sys.executable, str(SCRIPT), "--repo", str(repo), "--plugin-root", str(ROOT), *args],
        capture_output=True, text=True,
    )


def _repo(tmp_path, slug="custom_torch"):
    d = tmp_path / "harbor" / "rl-integration-generator"
    d.mkdir(parents=True)
    (d / "rl-suite-spec.json").write_text(json.dumps({"algorithm_source": {"slug": slug}}))
    return tmp_path


# --- applicability gate (pure logic, exits before any file mutation) ---

def test_unknown_trick_exits_2(tmp_path):
    r = _run(_repo(tmp_path), "--trick", "no_such_trick")
    assert r.returncode == 2 and "no trick named" in r.stderr


def test_backend_mismatch_exits_2(tmp_path):
    r = _run(_repo(tmp_path, slug="custom_jax"), "--trick", TRICK)  # trick is custom_torch
    assert r.returncode == 2 and "backend" in r.stderr


def test_unsupported_algorithm_exits_2(tmp_path):
    r = _run(_repo(tmp_path), "--trick", TRICK, "--algorithm", "sac")  # trick supports [ppo]
    assert r.returncode == 2 and "does not support" in r.stderr


# --- happy path: build the target from the trick's own find-blocks, apply, re-apply ---

def test_file_and_config_patch_applies_then_noops(tmp_path):
    edits = TRICK_DIR / "edits" / "ppo.py"
    find1 = (edits / "ppo_01_unnormalize_next_value.find").read_text()
    find2 = (edits / "ppo_02_normalize_returns_values.find").read_text()
    repo = _repo(tmp_path)
    algo = repo / "harbor" / "scripts" / "rl" / "custom_torch" / "algo"
    algo.mkdir(parents=True)
    (algo / "ppo.py").write_text(f"# head\n{find1}\n# mid\n{find2}\n# tail\n")
    cfg = repo / "harbor" / "configs" / "rl"
    cfg.mkdir(parents=True)
    (cfg / "ppo.parallel.yaml").write_text("value_norm: false\nlearning_rate: 0.0003\n")

    r = _run(repo, "--trick", TRICK, "--algorithm", "ppo")
    assert r.returncode == 0, r.stderr + r.stdout
    # file patch applied: the replace block is now present
    repl1 = (edits / "ppo_01_unnormalize_next_value.replace").read_text().strip()
    assert repl1 and repl1 in (algo / "ppo.py").read_text()
    # config patch applied
    assert yaml.safe_load((cfg / "ppo.parallel.yaml").read_text())["value_norm"] is True

    # idempotent: a second apply is all-noop and still exits 0
    r2 = _run(repo, "--trick", TRICK, "--algorithm", "ppo")
    assert r2.returncode == 0
    assert "noop" in (r2.stdout + r2.stderr).lower()
