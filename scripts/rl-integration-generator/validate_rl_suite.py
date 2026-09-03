"""
Smoke-test the RL suite that render_rl_suite.py just dropped into a target repo.

Validation tiers (each returns ok|fail):
  T1 — files exist           — harbor/scripts/rl/<slug>/*.py + harbor/configs/rl/*.yaml + harbor/rl-integration-generator/rl-suite-spec.json
  T2 — py_compile             — every harbor/scripts/rl/<slug>/**/*.py compiles
  T3 — yaml parse             — every harbor/configs/rl/*.yaml parses
  T4 — spec parse + schema    — harbor/rl-integration-generator/rl-suite-spec.json has the required keys

The per-impl directory `harbor/scripts/rl/<slug>/` is read from the spec's
`scripts_dir` field. Four sources currently:
  - custom_torch        → train/eval/render/env_wrapper.py + algo/ + replay/ + models/ + utils/
  - custom_jax          → same layout, but JAX/Flax internals; pickle checkpoints; vmap rollout required
  - stable_baseline3    → train/eval/render/env_wrapper.py only
  - local_implementation → train/eval/render/env_wrapper.py thin shims

This script does NOT run train.py. The "real" smoke (≤1k-step train per algo)
is the responsibility of the rl-integration-generator agent's Phase 4.

Exit code: 0 if all tiers pass; 1 otherwise. JSON report goes to stdout.
"""
from __future__ import annotations

import argparse
import json
import py_compile
import sys
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--repo", required=True, type=Path)
    p.add_argument("--algorithms", default="ppo,sac,td3",
                   help="comma-separated subset rendered for this repo")
    return p.parse_args()


def _scripts_dir(repo: Path) -> Path | None:
    """Read scripts_dir from rl-suite-spec.json; fall back to legacy scripts/rl."""
    spec = repo / "harbor" / "rl-integration-generator" / "rl-suite-spec.json"
    if not spec.exists():
        return None
    try:
        data = json.loads(spec.read_text())
    except json.JSONDecodeError:
        return None
    rel = data.get("scripts_dir") or "scripts/rl"
    return repo / rel


def t1_files(repo: Path, algos: list[str]) -> tuple[bool, list[str]]:
    sdir = _scripts_dir(repo)
    if sdir is None:
        return (False, ["harbor/rl-integration-generator/rl-suite-spec.json missing or invalid"])
    sdir_rel = sdir.relative_to(repo)
    expected = [
        f"{sdir_rel}/train.py",
        f"{sdir_rel}/eval.py",
        f"{sdir_rel}/render.py",
        f"{sdir_rel}/env_wrapper.py",
        "harbor/configs/rl/suite.yaml",
        "harbor/rl-integration-generator/rl-suite-spec.json",
    ]
    for a in algos:
        expected.append(f"harbor/configs/rl/{a}.yaml")
        expected.append(f"harbor/configs/rl/{a}.parallel.yaml")
    missing = [p for p in expected if not (repo / p).exists()]
    return (not missing, missing)


def t2_compile(repo: Path) -> tuple[bool, list[str]]:
    sdir = _scripts_dir(repo)
    if sdir is None or not sdir.exists():
        return (False, [f"scripts dir not found: {sdir}"])
    errs: list[str] = []
    for py in sdir.rglob("*.py"):
        try:
            py_compile.compile(str(py), doraise=True)
        except py_compile.PyCompileError as e:
            errs.append(f"{py.relative_to(repo)}: {e.msg.strip()}")
    return (not errs, errs)


def t3_yaml(repo: Path) -> tuple[bool, list[str]]:
    try:
        import yaml
    except ImportError:
        return (True, ["[skipped] PyYAML not installed on host"])
    errs: list[str] = []
    for yml in (repo / "harbor" / "configs" / "rl").glob("*.yaml"):
        try:
            yaml.safe_load(yml.read_text())
        except Exception as e:
            errs.append(f"{yml.relative_to(repo)}: {e}")
    return (not errs, errs)


def t4_spec(repo: Path) -> tuple[bool, list[str]]:
    spec = repo / "harbor" / "rl-integration-generator" / "rl-suite-spec.json"
    if not spec.exists():
        return (False, ["spec not found"])
    try:
        data = json.loads(spec.read_text())
    except json.JSONDecodeError as e:
        return (False, [f"json parse: {e}"])
    required = ["schema_version", "benchmark", "tasks", "algorithm_source",
                "algorithms", "logging", "selection_metric", "training_defaults"]
    missing = [k for k in required if k not in data]
    return (not missing, missing)


def main():
    args = parse_args()
    repo: Path = args.repo.resolve()
    algos = [a.strip().lower() for a in args.algorithms.split(",") if a.strip()]

    t1_ok, t1_detail = t1_files(repo, algos)
    t2_ok, t2_detail = t2_compile(repo)
    t3_ok, t3_detail = t3_yaml(repo)
    t4_ok, t4_detail = t4_spec(repo)

    report = {
        "ok": t1_ok and t2_ok and t3_ok and t4_ok,
        "tiers": {
            "T1_files": {"ok": t1_ok, "missing": t1_detail},
            "T2_py_compile": {"ok": t2_ok, "errors": t2_detail},
            "T3_yaml_parse": {"ok": t3_ok, "errors": t3_detail},
            "T4_spec_schema": {"ok": t4_ok, "missing_or_errors": t4_detail},
        },
    }
    print(json.dumps(report, indent=2))
    sys.exit(0 if report["ok"] else 1)


if __name__ == "__main__":
    main()
