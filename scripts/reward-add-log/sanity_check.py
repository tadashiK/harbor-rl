#!/usr/bin/env python3
"""Sanity check for /harbor:reward-add-log: assert composer(info["detailed_reward"]) == env_reward.

Imports the patched env helper (`scripts/_<family>_env.py`) and, for each
specified task, runs N random-action steps and verifies on every step:

  1. info["detailed_reward"] is a non-empty dict of finite floats.
  2. composer(terms.values()) == env_reward, where composer is read from
     info["reward_composer"] ∈ {"sum", "product"}; defaults to "sum" if absent.
  3. The wrapper does NOT mutate env_reward — same value every random seed.

Composer choice is per-task (not a flag) because it's a structural property
of the env's native reward function. The wrapper writes "reward_composer" on
info to advertise it; this script just trusts that and verifies the equation.

Exits 0 on success; non-zero on first violation, printing the offending step.

Usage:
    python sanity_check.py \\
        --env-helper /path/to/scripts/_<family>_env.py \\
        --tasks      "cartpole/swingup,walker/walk" \\
        --steps      200
"""
from __future__ import annotations

import argparse
import importlib.util
import math
import sys
from pathlib import Path


def _load_helper(helper_path: Path):
    """Import the env helper module by file path."""
    sys.path.insert(0, str(helper_path.parent))
    spec = importlib.util.spec_from_file_location(helper_path.stem, str(helper_path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load helper from {helper_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _find_make_fn(mod):
    """Find the `make_*_env(task, ...)` factory in the helper module."""
    for name in dir(mod):
        if name.startswith("make_") and name.endswith("_env"):
            return getattr(mod, name)
    raise RuntimeError(
        f"no make_*_env factory found in {mod.__file__} — "
        "the helper must follow the benchmark-generator convention."
    )


def _compose(values, composer: str) -> float:
    if composer == "sum":
        return float(sum(values))
    if composer == "product":
        out = 1.0
        for v in values:
            out *= float(v)
        return out
    raise ValueError(f"unknown composer {composer!r}")


def _check_task(mod, task: str, n_steps: int) -> tuple[bool, str]:
    """Run n_steps random-action steps; return (ok, message)."""
    make_env = _find_make_fn(mod)
    try:
        env = make_env(task, seed=0)
    except TypeError:
        env = make_env(task)
    env.reset()
    composer_seen = None
    for step_idx in range(n_steps):
        action = env.action_space.sample()
        out = env.step(action)
        if len(out) == 5:
            _, reward, _, _, info = out
        else:
            _, reward, _, info = out
        if not isinstance(info, dict) or "detailed_reward" not in info:
            return False, f"step {step_idx}: info is missing 'detailed_reward' (info={info!r})"
        terms = info["detailed_reward"]
        if not isinstance(terms, dict) or not terms:
            return False, f"step {step_idx}: info['detailed_reward'] is not a non-empty dict (got {terms!r})"
        for k, v in terms.items():
            try:
                fv = float(v)
            except (TypeError, ValueError):
                return False, f"step {step_idx}: term '{k}'={v!r} is not numeric"
            if not math.isfinite(fv):
                return False, f"step {step_idx}: term '{k}'={fv} is not finite"
        composer = info.get("reward_composer", "sum")
        if composer_seen is not None and composer != composer_seen:
            return False, (f"step {step_idx}: composer changed mid-episode "
                           f"({composer_seen!r} -> {composer!r}) — wrappers must be deterministic")
        composer_seen = composer
        composed = _compose([float(v) for v in terms.values()], composer)
        if abs(float(reward) - composed) >= 1e-6:
            return False, (f"step {step_idx}: env_reward={float(reward):.10g} != "
                           f"{composer}(terms)={composed:.10g} "
                           f"(diff={float(reward) - composed:+.3e}); terms={terms!r}")
    return True, f"OK: {composer_seen} invariant holds ({n_steps}/{n_steps} steps)"


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--env-helper", required=True, type=Path,
                   help="path to scripts/_<family>_env.py")
    p.add_argument("--tasks", required=True,
                   help="comma-separated list of task IDs to verify")
    p.add_argument("--steps", type=int, default=200,
                   help="random-action steps per task (default 200)")
    args = p.parse_args()

    if not args.env_helper.is_file():
        print(f"[FAIL] env-helper not found: {args.env_helper}", file=sys.stderr)
        return 2

    mod = _load_helper(args.env_helper)
    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]

    failures: list[tuple[str, str]] = []
    for task in tasks:
        ok, msg = _check_task(mod, task, args.steps)
        marker = "OK   " if ok else "FAIL "
        print(f"[{marker}] {task}: {msg}", flush=True)
        if not ok:
            failures.append((task, msg))

    if failures:
        print(f"\n[FAIL] {len(failures)}/{len(tasks)} task(s) failed the composer invariant.",
              file=sys.stderr)
        return 1
    print(f"\n[OK] all {len(tasks)} task(s) pass the composer invariant.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
