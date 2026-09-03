"""Unit: scripts/common/resolve_suite.py — the canonical rl-suite-spec reader.

Locks in the key-path bug fix: the slug lives at algorithm_source.slug and the
parallel flag at algorithm_source.parallel (NOT `algorithm_slug`, NOT top-level).
"""
import json
import subprocess
import sys

from _pluginmeta import ROOT

SCRIPT = ROOT / "scripts" / "common" / "resolve_suite.py"


def _write_spec(tmp_path, slug="custom_torch", parallel=True, scripts_dir=None):
    d = tmp_path / "harbor" / "rl-integration-generator"
    d.mkdir(parents=True)
    spec = {"algorithm_source": {"kind": "custom", "slug": slug, "parallel": parallel}}
    if scripts_dir is not None:
        spec["scripts_dir"] = scripts_dir
    (d / "rl-suite-spec.json").write_text(json.dumps(spec))
    return tmp_path


def _run(*args):
    return subprocess.run([sys.executable, str(SCRIPT), *args], capture_output=True, text=True)


def test_default_output(tmp_path):
    repo = _write_spec(tmp_path, slug="custom_torch", parallel=True)
    r = _run("--repo", str(repo), "--algo", "ppo")
    assert r.returncode == 0, r.stderr
    out = dict(line.split("=", 1) for line in r.stdout.strip().splitlines())
    assert out["SLUG"] == "custom_torch"
    assert out["PARALLEL"] == "true"
    assert out["CONFIG_NAME"] == "ppo.parallel"
    assert out["SCRIPTS_DIR"] == "harbor/scripts/rl/custom_torch"


def test_reads_canonical_keys_not_the_buggy_ones(tmp_path):
    repo = _write_spec(tmp_path, slug="custom_jax", parallel=False)
    assert _run("--repo", str(repo), "--field", "slug").stdout.strip() == "custom_jax"
    assert _run("--repo", str(repo), "--field", "parallel").stdout.strip() == "false"
    # parallel=False → config name has NO .parallel suffix
    assert _run("--repo", str(repo), "--algo", "sac", "--field", "config_name").stdout.strip() == "sac"


def test_scripts_dir_defaults_from_slug(tmp_path):
    repo = _write_spec(tmp_path, slug="myalgo", scripts_dir=None)
    assert _run("--repo", str(repo), "--field", "scripts_dir").stdout.strip() == "harbor/scripts/rl/myalgo"


def test_missing_spec_exits_1(tmp_path):
    r = _run("--repo", str(tmp_path))
    assert r.returncode == 1
    assert "rl-suite-spec.json" in r.stderr


def test_missing_slug_exits_1(tmp_path):
    d = tmp_path / "harbor" / "rl-integration-generator"
    d.mkdir(parents=True)
    (d / "rl-suite-spec.json").write_text('{"algorithm_source": {}}')
    r = _run("--repo", str(tmp_path))
    assert r.returncode == 1
    assert "slug" in r.stderr
