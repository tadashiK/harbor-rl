#!/usr/bin/env python3
"""Smoke for value_norm_torch.

Verifies on the rendered repo:
  1. `value_norm: true` is set in harbor/configs/rl/ppo.parallel.yaml.
  2. Both patch markers are present in harbor/scripts/rl/custom_torch/algo/ppo.py:
        - "self.value_rms.update(next_value)"  (unnormalize next_value)
        - "returns = self.value_rms.normalize(returns)"  (renormalize returns/values)
  3. ppo.py parses as valid Python.
  4. Numerical: instantiate RunningMeanStd, push N(2, 3) draws, then verify
     that `unnormalize(normalize(x)) == x` (round-trip identity, the property
     value_norm relies on for the critic↔target loop to remain consistent).

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


def _check_config(repo: Path) -> list[str]:
    yaml = _load_yaml()
    cfg = repo / "harbor" / "configs" / "rl" / "ppo.parallel.yaml"
    if not cfg.is_file():
        return [f"{cfg} not found"]
    data = yaml.safe_load(cfg.read_text()) or {}
    if data.get("value_norm") is not True:
        return [f"{cfg.relative_to(repo)}: value_norm={data.get('value_norm')!r} (expected True)"]
    return []


def _check_markers(repo: Path) -> list[str]:
    failures: list[str] = []
    ppo = repo / "harbor" / "scripts" / "rl" / "custom_torch" / "algo" / "ppo.py"
    if not ppo.is_file():
        return [f"{ppo} not found"]
    src = ppo.read_text()
    expected = [
        "self.value_rms.update(next_value)",
        "next_value = self.value_rms.unnormalize(next_value)",
        "returns = self.value_rms.normalize(returns)",
        "values = self.value_rms.normalize(values)",
    ]
    for marker in expected:
        if marker not in src:
            failures.append(f"missing patch marker in ppo.py: {marker!r}")
    try:
        ast.parse(src, filename=str(ppo))
    except SyntaxError as e:
        failures.append(f"ppo.py SyntaxError at line {e.lineno}: {e.msg}")
    return failures


def _check_numerics(repo: Path) -> list[str]:
    target = repo / "harbor" / "scripts" / "rl" / "custom_torch" / "utils" / "torch_util.py"
    if not target.is_file():
        return [f"{target} not found"]
    try:
        import torch
    except ImportError:
        return ["torch not importable in this venv (run smoke from <repo>/.venv)"]
    spec = importlib.util.spec_from_file_location("torch_util_smoke_vn", str(target))
    if spec is None or spec.loader is None:
        return [f"could not load {target}"]
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except Exception as e:
        return [f"importing {target} raised {type(e).__name__}: {e}"]

    rms = mod.RunningMeanStd(shape=(1,), device="cpu")
    torch.manual_seed(1)
    for _ in range(40):
        rms.update(torch.randn(256, 1) * 3.0 + 2.0)
    sample = torch.randn(8, 1) * 3.0 + 2.0
    norm = rms.normalize(sample)
    rt = rms.unnormalize(norm)
    err = (rt - sample).abs().max().item()
    if err > 1e-4:
        return [f"value_rms unnormalize(normalize(x)) round-trip drift: {err:.3e}"]
    if rms.var.sqrt().item() < 1.0:
        return [f"value_rms.std after 10240 N(2,3) samples = {rms.var.sqrt().item():.3f} (expected ≥ 1.0)"]
    return []


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--repo", required=True, type=Path)
    args = p.parse_args()
    repo = args.repo.resolve()
    if not (repo / "harbor").is_dir():
        print(f"[fail] {repo} does not look like a Harbor benchmark repo.", file=sys.stderr)
        return 2

    failures: list[str] = []
    failures += _check_config(repo)
    failures += _check_markers(repo)
    failures += _check_numerics(repo)

    if failures:
        print(f"[fail] value_norm_torch smoke: {len(failures)} issue(s)")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("[ok] value_norm_torch smoke: config flag + patch markers + RunningMeanStd round-trip pass.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
