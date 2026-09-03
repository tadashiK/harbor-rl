"""Unit: scripts/rl-integration-generator/validate_rl_suite.py — T4 spec-schema tier."""
import json
import subprocess
import sys

from _pluginmeta import ROOT

SCRIPT = ROOT / "scripts" / "rl-integration-generator" / "validate_rl_suite.py"


def _spec(**override):
    spec = {
        "schema_version": 1, "benchmark": {}, "tasks": [{"id": "T"}],
        "algorithm_source": {"slug": "custom_torch", "parallel": True},
        "algorithms": {}, "logging": {}, "selection_metric": {}, "training_defaults": {},
        "scripts_dir": "harbor/scripts/rl/custom_torch",
    }
    spec.update(override)
    return spec


def _report(tmp_path, spec):
    d = tmp_path / "harbor" / "rl-integration-generator"
    d.mkdir(parents=True)
    (d / "rl-suite-spec.json").write_text(json.dumps(spec))
    r = subprocess.run([sys.executable, str(SCRIPT), "--repo", str(tmp_path)], capture_output=True, text=True)
    return json.loads(r.stdout)


def test_t4_valid_spec_ok(tmp_path):
    assert _report(tmp_path, _spec())["tiers"]["T4_spec_schema"]["ok"]


def test_t4_missing_required_key_fails(tmp_path):
    spec = _spec()
    del spec["algorithms"]
    t4 = _report(tmp_path, spec)["tiers"]["T4_spec_schema"]
    assert not t4["ok"]
    assert "algorithms" in t4["missing_or_errors"]
