"""Unit: scripts/rl-integration-generator/discover_algorithms.py — host probe shape."""
import json
import subprocess
import sys

from _pluginmeta import ROOT

SCRIPT = ROOT / "scripts" / "rl-integration-generator" / "discover_algorithms.py"


def test_probe_reports_each_source():
    out = json.loads(subprocess.run([sys.executable, str(SCRIPT)], capture_output=True, text=True).stdout)
    for source in ("stable-baselines3", "custom-torch", "custom-jax"):
        assert source in out, f"missing source {source}"
        assert "installed" in out[source]
        assert isinstance(out[source]["installed"], bool)
