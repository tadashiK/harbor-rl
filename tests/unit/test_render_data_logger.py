"""Unit: scripts/rl-integration-generator/render_data_logger.py — the mustache renderer.

The rendered DataLogger is what every algorithm logs through, so a conditional block that
renders the wrong way produces a training run with silently missing metrics. The renderer is
a pure text transform, so the block/scalar semantics are pinned directly.
"""
import importlib.util
import json
import subprocess
import sys

from _pluginmeta import ROOT

SRC = ROOT / "scripts" / "rl-integration-generator" / "render_data_logger.py"


def _mod():
    spec = importlib.util.spec_from_file_location("render_data_logger", SRC)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


M = _mod()

TPL = """head
{{#WANDB}}import wandb{{/WANDB}}
{{#TB}}from torch.utils.tensorboard import SummaryWriter{{/TB}}
project = "{{PROJECT}}"
tail
"""


def test_enabled_block_keeps_its_body_without_the_markers():
    out = M.render(TPL, {"WANDB": True, "TB": False}, {"PROJECT": "demo"})
    assert "import wandb" in out
    assert "{{#WANDB}}" not in out and "{{/WANDB}}" not in out


def test_disabled_block_is_removed_entirely():
    out = M.render(TPL, {"WANDB": False, "TB": True}, {"PROJECT": "demo"})
    assert "import wandb" not in out
    assert "SummaryWriter" in out


def test_scalars_are_substituted():
    assert 'project = "demo"' in M.render(TPL, {"WANDB": False, "TB": False}, {"PROJECT": "demo"})


def test_multiline_block_bodies_survive():
    tpl = "a\n{{#F}}line1\nline2\n{{/F}}b\n"
    out = M.render(tpl, {"F": True}, {})
    assert "line1\nline2" in out and out.startswith("a\n")


def test_unprocessed_markers_are_reported_not_silently_left(capsys):
    """A leftover marker means the caller forgot a flag — the rendered file would contain
    literal mustache and fail at import time, far from the cause."""
    M.render("x {{FORGOTTEN}} y", {}, {})
    assert "unprocessed mustache markers" in capsys.readouterr().err


def test_cli_writes_the_rendered_logger(tmp_path):
    out = tmp_path / "data_logger.py"
    r = subprocess.run(
        [sys.executable, str(SRC), "--backend", "both", "--features", "scalar,image",
         "--output", str(out)],
        capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    text = out.read_text()
    assert text.strip(), "rendered an empty file"
    assert "{{" not in text, f"unrendered markers survived into the output: {text[:200]}"
    compile(text, str(out), "exec")      # it must be importable Python


def test_unknown_feature_is_rejected_rather_than_ignored(tmp_path):
    """A typo'd feature must not silently render a logger missing that sink."""
    r = subprocess.run(
        [sys.executable, str(SRC), "--backend", "tb", "--features", "curves",
         "--output", str(tmp_path / "o.py")],
        capture_output=True, text=True)
    assert r.returncode != 0
    assert "unknown features" in r.stderr
    assert not (tmp_path / "o.py").exists()
