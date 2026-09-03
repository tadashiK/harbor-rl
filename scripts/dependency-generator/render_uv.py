#!/usr/bin/env python3
"""dependency-generator uv-backend renderer.

Reads `<repo>/harbor/dependency-generator/{probe,install_plan}.json` (written
during Steps 1+3 of the dependency-generator agent flow) and emits
`<repo>/harbor/dependency-generator/setup_uv.sh` — a self-contained bash script that:

  1. Creates `<repo>/.venv` via `uv venv --python <PY>`.
  2. Translates each `installation_steps` entry into the host-side equivalent:
       - `uv_sync`         → `uv sync <args>` (in repo root, with .venv active)
       - `uv_pip_install`  → `uv pip install <args>`
       - `apt_install`     → emit a sudo-apt comment (NOT executed; user runs it)
       - `git_clone`       → `git clone` to `<repo>/.harbor_thirdparty/<name>/`
       - `shell`           → run the cmd; rewrites `/workspace/<repo>` → `<repo>` (host)
  3. Runs `post_install_hooks` (gated by a `.harbor/.first_run` sentinel).

Usage:
    python render_uv.py <repo>
"""
from __future__ import annotations

import argparse
import json
import shlex
import sys
from pathlib import Path


def _read_json(p: Path) -> dict | None:
    if not p.is_file():
        return None
    try:
        return json.loads(p.read_text())
    except json.JSONDecodeError:
        return None


def _project_name(repo: Path) -> str:
    pyproject = repo / "pyproject.toml"
    if pyproject.is_file():
        for line in pyproject.read_text().splitlines():
            line = line.strip()
            if line.startswith("name") and "=" in line:
                _, _, val = line.partition("=")
                val = val.strip().strip('"').strip("'")
                if val:
                    return val
    return repo.name


def _python_version(probe: dict) -> str:
    return str(probe.get("python_version") or "3.10")


def _rewrite_workspace_paths(cmd: str, repo: Path, project_name: str) -> str:
    """`/workspace/<repo>` (docker convention) → `<repo>` (host absolute)."""
    repo_str = str(repo)
    return (
        cmd
        .replace(f"/workspace/{project_name}", repo_str)
        .replace(f"/workspace/{repo.name}", repo_str)
        .replace("/workspace/", str(repo.parent) + "/")
    )


def _render_step(step: dict, repo: Path, project_name: str) -> list[str]:
    """Translate one InstallStep into bash lines."""
    kind = step.get("kind", "")
    if kind == "apt_install":
        pkgs = step.get("pkgs") or []
        if not pkgs:
            return []
        joined = " ".join(shlex.quote(p) for p in pkgs)
        return [
            "# (system apt) NOT executed — uv backend cannot sudo. Run manually if imports fail:",
            f'#     sudo apt install -y {joined}',
        ]
    if kind == "uv_sync":
        args = step.get("args") or []
        return [f'uv sync {" ".join(shlex.quote(a) for a in args)}']
    if kind == "uv_pip_install":
        args = step.get("args") or []
        # Rewrite any /workspace/<repo>/path/to/x.txt → <repo>/path/to/x.txt for arg paths
        rewritten = [_rewrite_workspace_paths(a, repo, project_name) for a in args]
        return [f'uv pip install {" ".join(shlex.quote(a) for a in rewritten)}']
    if kind == "git_clone":
        url = step.get("url") or ""
        rev = step.get("rev")
        target_name = step.get("target") or url.rstrip("/").rsplit("/", 1)[-1].removesuffix(".git")
        target_dir = f".harbor_thirdparty/{target_name}"
        lines = [
            f'mkdir -p .harbor_thirdparty',
            f'if [ ! -d {shlex.quote(target_dir)} ]; then',
            f'  git clone {shlex.quote(url)} {shlex.quote(target_dir)}',
            f'fi',
        ]
        if rev:
            lines += [f'(cd {shlex.quote(target_dir)} && git checkout {shlex.quote(rev)})']
        return lines
    if kind == "shell":
        cmd = step.get("cmd", "")
        return [_rewrite_workspace_paths(cmd, repo, project_name)]
    return [f"# (unknown InstallStep kind={kind!r}; skipped)"]


def _render_post_install(plan: dict, repo: Path, project_name: str) -> list[str]:
    hooks = plan.get("post_install_hooks") or []
    if not hooks:
        return []
    out = [
        "",
        "# --- Post-install hooks (gated by .harbor/.first_run sentinel) ---",
        'SENTINEL=".venv/.harbor_first_run_done"',
        'if [ ! -f "$SENTINEL" ]; then',
    ]
    for h in hooks:
        kind = h.get("kind", "shell")
        when = h.get("when", "first_run")
        if when != "first_run":
            continue
        if kind == "shell":
            cmd = _rewrite_workspace_paths(h.get("cmd", ""), repo, project_name)
            out.append(f'  {cmd}')
    out += [
        '  touch "$SENTINEL"',
        'fi',
    ]
    return out


def render_setup_uv_sh(repo: Path) -> Path:
    harbor = repo / "harbor" / "dependency-generator"
    probe = _read_json(harbor / "probe.json") or {}
    plan = _read_json(harbor / "install_plan.json")
    project_name = _project_name(repo)
    py = _python_version(probe)

    out: list[str] = []
    out.append("#!/usr/bin/env bash")
    out.append("# Auto-generated by dependency-generator (uv backend) — DO NOT HAND-EDIT.")
    out.append("# Re-run via `/harbor:env-install-uv` to regenerate.")
    out.append(f"# Project: {project_name}    Python: {py}")
    out.append("set -euo pipefail")
    out.append("")
    out.append(f'cd "$(dirname "$0")/../.."   # cd to repo root (script lives in harbor/dependency-generator/)')
    out.append("")
    out.append("# --- Step 1: prerequisites (uv, installed if missing) ---")
    out.append("# A non-login shell can lack ~/.local/bin on PATH even when uv is installed,")
    out.append("# so source uv's env file before concluding it is absent.")
    out.append('if [ -f "$HOME/.local/bin/env" ]; then . "$HOME/.local/bin/env"; fi')
    out.append("if ! command -v uv >/dev/null 2>&1; then")
    out.append('  echo "[setup_uv] uv not found — installing to $HOME/.local/bin"')
    out.append("  curl -LsSf https://astral.sh/uv/install.sh | sh")
    out.append('  if [ -f "$HOME/.local/bin/env" ]; then . "$HOME/.local/bin/env"; fi')
    out.append("fi")
    out.append('command -v uv >/dev/null 2>&1 || { echo "[setup_uv] uv install failed — see https://docs.astral.sh/uv/" >&2; exit 1; }')
    out.append("")
    out.append("# --- Step 2: create venv (idempotent) ---")
    out.append("if [ ! -d .venv ]; then")
    out.append(f'  uv venv --python {shlex.quote(py)} .venv')
    out.append("else")
    out.append('  echo "[setup_uv] .venv already exists; reusing"')
    out.append("fi")
    out.append("")
    out.append("# Activate venv for the rest of this script.")
    out.append("# shellcheck disable=SC1091")
    out.append('source .venv/bin/activate')
    out.append("")
    out.append("# --- Step 3: install steps (translated from harbor/dependency-generator/install_plan.json) ---")

    if plan is None:
        # Legacy fallback: no plan — heuristic install
        if (repo / "uv.lock").is_file():
            out.append("uv sync --frozen --no-install-project --active")
        elif (repo / "requirements.txt").is_file():
            out.append("uv pip install -r requirements.txt")
        if (repo / "pyproject.toml").is_file() or (repo / "setup.py").is_file():
            out.append("uv pip install -e .")
    else:
        for step in plan.get("installation_steps") or []:
            for line in _render_step(step, repo, project_name):
                out.append(line)

    if plan is not None:
        out += _render_post_install(plan, repo, project_name)

    out.append("")
    out.append("# --- Step 4: harbor extras (logging + experiment tracking + RL deps) ---")
    out.append("# These are NOT in the upstream install plan — they're injected by dependency-generator")
    out.append("# so downstream subagents (benchmark-generator, rl-integration-generator) work")
    out.append("# out-of-the-box without extra build steps. Idempotent: uv pip install skips")
    out.append("# already-installed packages.")
    out.append("")
    out.append("# DataLogger backends used by rl-integration-generator (TensorBoard + W&B).")
    out.append('uv pip install wandb tensorboardX')
    out.append("# Video writer used by benchmark-generator render_random.py (and rl render.py).")
    out.append("uv pip install 'imageio[ffmpeg]'")
    out.append("# Plotting used by rl train.py end-of-training learning-curve dump.")
    out.append('uv pip install matplotlib')
    out.append("# Hydra / OmegaConf used by all rl-integration-generator scripts (train/eval/render).")
    out.append('uv pip install hydra-core omegaconf')
    out.append("# Stable-Baselines3 (extra) for the rl-integration-generator stable_baseline3 path.")
    out.append("uv pip install 'stable_baselines3[extra]'")
    out.append("# CoACD + trimesh: task-generator pre-decomposes every mesh collider (S1/C8)")
    out.append("# rather than letting the simulator cook convexDecomposition at load time,")
    out.append("# whose hull budget is unbounded and overflows contact buffers at high num_envs.")
    out.append('uv pip install coacd trimesh')
    out.append("")
    out.append('echo "[setup_uv] Done. Activate the venv with: source .venv/bin/activate"')
    out.append("")

    setup_path = harbor / "setup_uv.sh"
    setup_path.parent.mkdir(parents=True, exist_ok=True)
    setup_path.write_text("\n".join(out))
    setup_path.chmod(0o755)
    return setup_path


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("repo", type=Path, help="absolute path to target repo")
    args = p.parse_args()
    repo = args.repo.resolve()
    if not repo.is_dir():
        print(f"[error] repo not a directory: {repo}", file=sys.stderr)
        return 2
    if not (repo / "harbor" / "dependency-generator" / "probe.json").is_file():
        print(f"[error] expected harbor/dependency-generator/probe.json — run dependency-generator probe first", file=sys.stderr)
        return 2
    setup = render_setup_uv_sh(repo)
    print(f"[render_uv] wrote {setup}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
