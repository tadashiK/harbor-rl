#!/usr/bin/env python3
"""Smoke for obs_rms_torch.

Verifies on the rendered repo:
  1. `obs_norm: true` is set in harbor/configs/rl/{ppo,sac,td3}.parallel.yaml
  2. `RunningMeanStd` is importable from the rendered utils.torch_util and
     converges to the data-generating distribution after a few hundred
     pushes (mean within 0.05, std within 0.05 of the targets).
  3. ac_base / ppo / sac / td3 still parse as valid Python (post-patch).

Usage:
    python smoke.py --repo <abs path to benchmark repo>

Exit 0 = pass, non-zero = fail with one short diagnostic line per issue.
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


def _check_config(repo: Path, algos: list[str]) -> list[str]:
    failures: list[str] = []
    yaml = _load_yaml()
    for algo in algos:
        cfg = repo / "harbor" / "configs" / "rl" / f"{algo}.parallel.yaml"
        if not cfg.is_file():
            failures.append(f"{cfg} not found")
            continue
        data = yaml.safe_load(cfg.read_text()) or {}
        if data.get("obs_norm") is not True:
            failures.append(f"{cfg.relative_to(repo)}: obs_norm={data.get('obs_norm')!r} (expected True)")
    return failures


def _check_syntax(repo: Path) -> list[str]:
    failures: list[str] = []
    targets = [
        "harbor/scripts/rl/custom_torch/algo/ac_base.py",
        "harbor/scripts/rl/custom_torch/algo/ppo.py",
        "harbor/scripts/rl/custom_torch/algo/sac.py",
        "harbor/scripts/rl/custom_torch/algo/td3.py",
        "harbor/scripts/rl/custom_torch/utils/torch_util.py",
    ]
    for rel in targets:
        path = repo / rel
        if not path.is_file():
            failures.append(f"{rel} not found in repo")
            continue
        try:
            ast.parse(path.read_text(), filename=str(path))
        except SyntaxError as e:
            failures.append(f"{rel}: SyntaxError at line {e.lineno}: {e.msg}")
    return failures


def _check_running_mean_std(repo: Path) -> list[str]:
    """Import RunningMeanStd from the rendered repo and exercise numerics on CPU."""
    failures: list[str] = []
    target = repo / "harbor" / "scripts" / "rl" / "custom_torch" / "utils" / "torch_util.py"
    if not target.is_file():
        return [f"{target} not found"]

    try:
        import torch
    except ImportError:
        return ["torch not importable in this venv (run smoke from <repo>/.venv)"]

    spec = importlib.util.spec_from_file_location("torch_util_smoke", str(target))
    if spec is None or spec.loader is None:
        return [f"could not load {target}"]
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except Exception as e:
        return [f"importing {target} raised {type(e).__name__}: {e}"]
    if not hasattr(mod, "RunningMeanStd"):
        return ["torch_util.py missing RunningMeanStd class"]

    rms = mod.RunningMeanStd(shape=(8,), device="cpu")
    torch.manual_seed(0)
    target_mean = 5.0
    target_std = 2.0
    for _ in range(40):
        rms.update(torch.randn(256, 8) * target_std + target_mean)
    mean_diff = (rms.mean - target_mean).abs().max().item()
    std_diff  = (rms.var.sqrt() - target_std).abs().max().item()
    if mean_diff > 0.05:
        failures.append(f"RunningMeanStd mean drift: max |mean - {target_mean}| = {mean_diff:.3e}")
    if std_diff > 0.05:
        failures.append(f"RunningMeanStd std drift: max |std - {target_std}| = {std_diff:.3e}")

    sample = torch.randn(4, 8) * target_std + target_mean
    norm = rms.normalize(sample)
    round_trip = rms.unnormalize(norm)
    if (round_trip - sample).abs().max().item() > 1e-4:
        failures.append("RunningMeanStd normalize/unnormalize round-trip > 1e-4")
    return failures


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--repo", required=True, type=Path)
    args = p.parse_args()
    repo = args.repo.resolve()
    if not (repo / "harbor").is_dir():
        print(f"[fail] {repo} does not look like a Harbor benchmark repo "
              f"(no harbor/ dir).", file=sys.stderr)
        return 2

    failures: list[str] = []
    failures += _check_config(repo, ["ppo", "sac", "td3"])
    failures += _check_syntax(repo)
    failures += _check_running_mean_std(repo)

    if failures:
        print(f"[fail] obs_rms_torch smoke: {len(failures)} issue(s)")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("[ok] obs_rms_torch smoke: config flag set; RunningMeanStd converges; algo files parse.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
