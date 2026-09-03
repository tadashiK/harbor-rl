"""Unit: scripts/rl-tricks/list_tricks.py — enumerates the trick library."""
import json
import subprocess
import sys

import pytest

from _pluginmeta import ROOT

pytest.importorskip("yaml")
SCRIPT = ROOT / "scripts" / "rl-tricks" / "list_tricks.py"


def _run(*args):
    return subprocess.run([sys.executable, str(SCRIPT), "--plugin-root", str(ROOT), *args],
                          capture_output=True, text=True)


def test_lists_known_tricks_with_fields():
    recs = json.loads(_run("--json").stdout)
    names = {r["name"] for r in recs}
    assert {"value_norm_torch", "value_clip_torch", "obs_rms_torch"} <= names
    for r in recs:
        assert r["backend"], f"{r['name']}: missing backend"
        assert isinstance(r["algorithms"], list) and r["algorithms"], f"{r['name']}: no algorithms"


def test_algo_filter():
    recs = json.loads(_run("--algo", "ppo", "--json").stdout)
    assert recs and all("ppo" in r["algorithms"] for r in recs)
