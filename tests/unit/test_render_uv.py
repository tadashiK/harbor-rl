"""Unit: scripts/dependency-generator/render_uv.py — renders setup_uv.sh.

This is the first thing that runs against a user's repo, and its output is executed as shell.
The two properties that matter: the install plan's steps land in the script in order, and a
missing/partial plan degrades to the documented heuristic instead of emitting an empty
installer that "succeeds" and leaves a venv with nothing in it.
"""
import importlib.util
import json
import subprocess
import sys

from _pluginmeta import ROOT

SRC = ROOT / "scripts" / "dependency-generator" / "render_uv.py"


def _mod():
    spec = importlib.util.spec_from_file_location("render_uv", SRC)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


M = _mod()


def _repo(tmp_path, plan=None, probe=None, files=()):
    d = tmp_path / "harbor" / "dependency-generator"
    d.mkdir(parents=True)
    if plan is not None:
        (d / "install_plan.json").write_text(json.dumps(plan))
    (d / "probe.json").write_text(json.dumps(probe or {"python_version": "3.11"}))
    for name, body in files:
        (tmp_path / name).write_text(body)
    return tmp_path


def _render(repo):
    out = M.render_setup_uv_sh(repo)
    return out.read_text()


def test_plan_steps_are_emitted_in_order(tmp_path):
    repo = _repo(tmp_path, plan={
        "primary_install_strategy": "uv-pip-editable",
        "installation_steps": [
            {"kind": "shell", "cmd": "echo first"},
            {"kind": "shell", "cmd": "echo second"},
        ],
    })
    sh = _render(repo)
    assert sh.index("echo first") < sh.index("echo second")


def test_no_plan_falls_back_to_the_documented_heuristic(tmp_path):
    """A missing plan must still install something — the fallback is uv.lock -> sync,
    else requirements / editable."""
    repo = _repo(tmp_path, plan=None, files=[("uv.lock", "")])
    assert "uv sync --frozen" in _render(repo)

    repo2 = _repo(tmp_path / "b", plan=None, files=[("requirements.txt", "numpy\n")])
    sh = _render(repo2)
    assert "requirements.txt" in sh


def test_rendered_script_is_valid_bash(tmp_path):
    repo = _repo(tmp_path, plan={"installation_steps": [{"kind": "shell", "cmd": "echo hi"}]})
    out = M.render_setup_uv_sh(repo)
    r = subprocess.run(["bash", "-n", str(out)], capture_output=True, text=True)
    assert r.returncode == 0, f"rendered setup_uv.sh is not valid bash:\n{r.stderr}"


def test_script_lands_in_the_agents_own_subdir(tmp_path):
    repo = _repo(tmp_path, plan={"installation_steps": []})
    out = M.render_setup_uv_sh(repo)
    assert out == repo / "harbor" / "dependency-generator" / "setup_uv.sh"


def test_python_version_from_probe_reaches_the_script(tmp_path):
    repo = _repo(tmp_path, plan={"installation_steps": []}, probe={"python_version": "3.10"})
    assert "3.10" in _render(repo)


def test_cli_runs_and_reports_the_path(tmp_path):
    repo = _repo(tmp_path, plan={"installation_steps": [{"kind": "shell", "cmd": "echo hi"}]})
    r = subprocess.run([sys.executable, str(SRC), str(repo)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert "setup_uv.sh" in (r.stdout + r.stderr)


def test_harbor_extras_include_every_downstream_dependency(tmp_path):
    """The extras block is the ONLY place a downstream agent's dependency gets installed.

    Each entry exists because some later stage imports it and would otherwise fail deep inside
    a subagent run: coacd/trimesh back task-generator's S1/C8 mesh pre-decomposition, imageio
    the render path, hydra/omegaconf every rl script.
    """
    repo = _repo(tmp_path, plan={"primary_install_strategy": "uv-pip-editable",
                                 "installation_steps": [{"kind": "shell", "cmd": "echo hi"}]})
    script = _render(repo)
    for pkg in ("wandb", "tensorboardX", "imageio[ffmpeg]", "matplotlib",
                "hydra-core", "omegaconf", "stable_baselines3[extra]", "coacd", "trimesh"):
        assert pkg in script, f"{pkg} missing from the harbor extras block"


def test_uv_is_bootstrapped_rather_than_hard_failing(tmp_path):
    """setup_uv.sh must install uv itself when it is missing.

    uv is the one host-side tool the whole chain rests on, and nothing upstream of this
    script installs it — so a hard failure here is the user's first experience of harbor.
    The env-file source before the check matters as much as the install: a non-login shell
    can lack ~/.local/bin on PATH while uv is already on disk, and reinstalling in that case
    would hit the network for nothing.
    """
    repo = _repo(tmp_path, plan={"installation_steps": [{"kind": "shell", "cmd": "echo hi"}]})
    script = _render(repo)
    assert "astral.sh/uv/install.sh" in script, "no uv bootstrap in the generated script"
    src_env = script.index('. "$HOME/.local/bin/env"')
    assert src_env < script.index("astral.sh/uv/install.sh"), \
        "PATH recovery must be attempted before falling back to a network install"
    assert script.index("astral.sh/uv/install.sh") < script.index("uv venv"), \
        "uv must be bootstrapped before the venv step uses it"


def test_generated_script_is_valid_bash(tmp_path):
    """The renderer emits shell that is executed unattended; a syntax error surfaces as a
    broken install rather than as a renderer bug."""
    repo = _repo(tmp_path, plan={"installation_steps": [{"kind": "shell", "cmd": "echo hi"}]})
    out = M.render_setup_uv_sh(repo)
    r = subprocess.run(["bash", "-n", str(out)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
