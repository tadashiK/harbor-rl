#!/usr/bin/env python3
"""Smoke for distributional_critic_torch.

Verifies on the rendered repo:
  1. yaml flags set in EVERY parallel.yaml that has cri_class=DistributionalDoubleQ:
        distributional_critic=true, v_min/v_max/num_atoms present. Smoke
        scans td3.parallel.yaml and sac.parallel.yaml; algo files are
        checked only for the algorithms whose yaml has the trick applied.
  2. Patch markers present in:
        utils/torch_util.py — `def projection(`            (always)
        models/mlp.py       — `class DistributionalDoubleQ` (always)
        algo/ac_base.py     — `cri_kwargs`                 (always)
        algo/<algo>.py      — `_update_critic_c51` + `distributional_critic`
                                                           (only for patched algos)
  3. All touched files parse as valid Python.
  4. Numerical end-to-end of the distributional critic update logic:
        a. Build DistributionalDoubleQ + take a forward pass; verify outputs are
           valid probability distributions (softmax: per-row sum=1, all >=0).
        b. Sanity-check the projection function:
             - terminal step (done=1): projected dist becomes a delta at clamp(reward, v_min, v_max)
             - non-terminal: projected dist's expected value ≈ reward + gamma * E[next_dist]
             - mass conservation: every projected row sums to 1.

Usage:
    python smoke.py --repo <abs path>
"""
from __future__ import annotations

import argparse
import ast
import importlib.util
import sys
from pathlib import Path


def _load_yaml():
    try:
        import yaml
        return yaml
    except ImportError:
        print("[fail] PyYAML required", file=sys.stderr)
        sys.exit(2)


def _patched_algos(repo: Path):
    """Return the list of algo names whose parallel.yaml has cri_class=DistributionalDoubleQ."""
    yaml = _load_yaml()
    out = []
    for algo in ("td3", "sac"):
        cfg = repo / "harbor" / "configs" / "rl" / f"{algo}.parallel.yaml"
        if not cfg.is_file():
            continue
        data = yaml.safe_load(cfg.read_text()) or {}
        if data.get("cri_class") == "DistributionalDoubleQ":
            out.append(algo)
    return out


def _check_config(repo: Path, patched_algos) -> list[str]:
    if not patched_algos:
        return ["no parallel.yaml has cri_class=DistributionalDoubleQ — trick not applied to any algorithm"]
    yaml = _load_yaml()
    failures = []
    for algo in patched_algos:
        cfg = repo / "harbor" / "configs" / "rl" / f"{algo}.parallel.yaml"
        data = yaml.safe_load(cfg.read_text()) or {}
        if data.get("distributional_critic") is not True:
            failures.append(f"{cfg.name}: distributional_critic={data.get('distributional_critic')!r} (expected True)")
        for k in ("v_min", "v_max", "num_atoms"):
            if k not in data:
                failures.append(f"{cfg.name}: key {k!r} missing")
    return failures


def _check_markers(repo: Path, patched_algos) -> list[str]:
    targets = {
        repo / "harbor/scripts/rl/custom_torch/utils/torch_util.py": ["def projection("],
        repo / "harbor/scripts/rl/custom_torch/models/mlp.py":       ["class DistributionalDoubleQ", "z_atoms"],
        repo / "harbor/scripts/rl/custom_torch/algo/ac_base.py":     ["cri_kwargs"],
    }
    for algo in patched_algos:
        targets[repo / f"harbor/scripts/rl/custom_torch/algo/{algo}.py"] = [
            "_update_critic_c51", "distributional_critic"
        ]
    failures: list[str] = []
    for path, markers in targets.items():
        if not path.is_file():
            failures.append(f"{path} not found")
            continue
        src = path.read_text()
        for m in markers:
            if m not in src:
                failures.append(f"missing patch marker in {path.name}: {m!r}")
        try:
            ast.parse(src, filename=str(path))
        except SyntaxError as e:
            failures.append(f"{path.name} SyntaxError at line {e.lineno}: {e.msg}")
    return failures


def _check_numerics(repo: Path) -> list[str]:
    try:
        import torch
    except ImportError:
        return ["torch not importable in this venv (run smoke from <repo>/.venv)"]

    # Load the patched modules in isolation.
    def _load(name, path):
        spec = importlib.util.spec_from_file_location(name, str(path))
        mod = importlib.util.module_from_spec(spec)
        # mlp.py imports `from utils.torch_util import ...` — make that resolve.
        custom_torch = repo / "harbor/scripts/rl/custom_torch"
        sys.path.insert(0, str(custom_torch))
        try:
            spec.loader.exec_module(mod)
        finally:
            sys.path.pop(0)
        return mod

    failures: list[str] = []
    try:
        torch_util = _load("tu_smoke_distrl", repo / "harbor/scripts/rl/custom_torch/utils/torch_util.py")
        mlp = _load("mlp_smoke_distrl", repo / "harbor/scripts/rl/custom_torch/models/mlp.py")
    except Exception as e:
        return [f"importing patched modules raised {type(e).__name__}: {e}"]

    # ---- (a) DistributionalDoubleQ forward pass produces valid distributions ----
    torch.manual_seed(0)
    state_dim, act_dim = 8, 4
    v_min, v_max, num_atoms = -10.0, 10.0, 51
    critic = mlp.DistributionalDoubleQ(state_dim, act_dim, v_min=v_min, v_max=v_max, num_atoms=num_atoms)
    obs = torch.randn(32, state_dim)
    act = torch.randn(32, act_dim).clamp(-1, 1)
    d1, d2 = critic.get_q1_q2_dist(obs, act)
    if d1.shape != (32, num_atoms) or d2.shape != (32, num_atoms):
        failures.append(f"get_q1_q2_dist shapes {d1.shape}/{d2.shape} (expected (32, {num_atoms}))")
    if (d1.sum(dim=1) - 1.0).abs().max().item() > 1e-5:
        failures.append("Q1 distribution rows do not sum to 1 (softmax broken)")
    if (d2 < 0).any().item():
        failures.append("Q2 distribution has negative probabilities")
    q1_v, q2_v = critic.get_q1_q2(obs, act)
    if q1_v.shape != (32, 1) or q2_v.shape != (32, 1):
        failures.append(f"get_q1_q2 (expected) shapes {q1_v.shape}/{q2_v.shape} (expected (32, 1))")
    # Expected value must lie in [v_min, v_max].
    if not (v_min - 1e-3 <= q1_v.min().item() and q1_v.max().item() <= v_max + 1e-3):
        failures.append(f"E[Q1] outside support: [{q1_v.min().item():.2f}, {q1_v.max().item():.2f}]")

    # ---- (b1) projection: terminal step → delta at clamp(reward, v_min, v_max) ----
    z = critic.z_atoms
    B = 16
    reward = torch.linspace(-5.0, 5.0, B).unsqueeze(1)        # (B, 1)
    done = torch.ones(B, 1)                                    # all terminal
    next_dist = torch.full((B, num_atoms), 1.0 / num_atoms)    # uniform doesn't matter when done=1
    proj = torch_util.projection(
        next_dist=next_dist, reward=reward, done=done,
        gamma=0.99, v_min=v_min, v_max=v_max, num_atoms=num_atoms,
        support=z, device="cpu",
    )
    if (proj.sum(dim=1) - 1.0).abs().max().item() > 1e-5:
        failures.append("projection: terminal-step rows do not sum to 1")
    # Expected value of the projected delta should match the reward (clamped).
    proj_ev = (proj * z).sum(dim=1)
    expected_ev = reward.squeeze(1).clamp(v_min, v_max)
    err = (proj_ev - expected_ev).abs().max().item()
    if err > 1e-3:
        failures.append(f"projection terminal: E[proj] mismatches clamped reward (max err {err:.3e})")

    # ---- (b2) projection: non-terminal → E[proj] ≈ reward + gamma * E[next_dist] ----
    gamma = 0.95
    done_nt = torch.zeros(B, 1)
    # Non-uniform next_dist: peak at atom 25 (≈ value 0)
    next_dist = torch.zeros(B, num_atoms)
    next_dist[:, 25] = 0.6
    next_dist[:, 24] = 0.2
    next_dist[:, 26] = 0.2
    proj = torch_util.projection(
        next_dist=next_dist, reward=reward, done=done_nt,
        gamma=gamma, v_min=v_min, v_max=v_max, num_atoms=num_atoms,
        support=z, device="cpu",
    )
    if (proj.sum(dim=1) - 1.0).abs().max().item() > 1e-5:
        failures.append("projection: non-terminal rows do not sum to 1")
    proj_ev = (proj * z).sum(dim=1)
    next_ev = (next_dist * z).sum(dim=1)
    expected_ev = (reward.squeeze(1) + gamma * next_ev).clamp(v_min, v_max)
    err = (proj_ev - expected_ev).abs().max().item()
    if err > 0.1:           # projection rounds onto the discrete grid → ≤ 0.5*delta_z slack
        failures.append(f"projection non-terminal: E[proj] mismatches Bellman target (max err {err:.3e})")

    return failures


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--repo", required=True, type=Path)
    args = p.parse_args()
    repo = args.repo.resolve()
    if not (repo / "harbor").is_dir():
        print(f"[fail] {repo} does not look like a Harbor benchmark repo.", file=sys.stderr)
        return 2

    patched_algos = _patched_algos(repo)
    failures: list[str] = []
    failures += _check_config(repo, patched_algos)
    failures += _check_markers(repo, patched_algos)
    failures += _check_numerics(repo)
    if not failures:
        print(f"[info] distributional_critic_torch applied to: {patched_algos}")

    if failures:
        print(f"[fail] distributional_critic_torch smoke: {len(failures)} issue(s)")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("[ok] distributional_critic_torch smoke: yaml + markers + DistributionalDoubleQ forward + projection (terminal & Bellman) pass.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
