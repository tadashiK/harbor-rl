#!/usr/bin/env python3
"""Sanity check for /harbor:reward-add-log on IsaacLab benchmarks.

Runs random-action rollouts against batched torch envs and verifies, per
step, that:

  1. ``info["detailed_reward"]`` is a non-empty dict.
  2. **At least one real per-term key is present** beyond ``total``. If
     the wrapper falls through to passthrough mode (only ``total``), the
     task has no per-term decomposition source and the sanity FAILS —
     /harbor:reward-add-log was either misapplied or the env genuinely doesn't
     expose per-term info.
  3. The terms ``match`` the env reward, with the matching rule chosen
     by ``info["reward_composer"]``:
       - ``"sum"`` (manager-based path): ``Σ terms == env_reward`` (per
         env, ``torch.allclose`` with atol=1e-4 / rtol=1e-3).
       - ``"diagnostic"`` (Direct env w/ logs_rew_* path): per-term
         values are pre-scale means and DO NOT sum to env_reward; we
         instead verify ``detailed_reward["total"] == env_reward`` (same
         tolerance) which the wrapper guarantees for this path.
       - ``"product"``: reserved for dm_control product-of-tolerances —
         not currently produced by the IsaacLab wrapper but supported
         here for completeness.

The dm_control-flavored ``sanity_check.py`` next to this script assumes
scalar rewards / numpy actions and won't work for IsaacLab.

Mirrors the AppLauncher boilerplate from ``scripts/run_random.py`` so it
can be invoked the same way:

    source <repo>/.venv/bin/activate
    source <repo>/_isaac_sim/setup_conda_env.sh
    OMNI_KIT_ALLOW_ROOT=1 python -u \\
        ${CLAUDE_PLUGIN_ROOT}/scripts/reward-add-log/sanity_check_isaaclab.py \\
        --env-helper <repo>/scripts/_isaaclab_env.py \\
        --tasks      "Isaac-Open-Drawer-Franka-v0,Isaac-Factory-GearMesh-Direct-v0" \\
        --num-envs   4 --steps 200

Exits 0 on success; non-zero on first violation, printing the offending
step + diff. Closes Kit on exit.
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--env-helper", required=True, type=Path,
                    help="path to scripts/_isaaclab_env.py rendered by /harbor:reward-add-log")
parser.add_argument("--tasks", required=True,
                    help="comma-separated Isaac Lab gym IDs to verify")
parser.add_argument("--num-envs", type=int, default=4)
parser.add_argument("--steps", type=int, default=200)
parser.add_argument("--seed", type=int, default=0)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True

simulation_app = AppLauncher(args_cli).app

# Imports that depend on Kit being up.
import torch  # noqa: E402


def _load_helper(helper_path: Path):
    sys.path.insert(0, str(helper_path.parent))
    spec = importlib.util.spec_from_file_location(helper_path.stem, str(helper_path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load helper from {helper_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _check_task(mod, task: str, num_envs: int, n_steps: int, device: str,
                seed: int) -> tuple[bool, str]:
    env = mod.make_isaaclab_env(task, num_envs=num_envs, device=device)
    try:
        env.reset(seed=seed)
    except TypeError:
        env.reset()

    action_shape = env.action_space.shape
    env_device = env.unwrapped.device
    composer_seen = None

    seen_term_keys: set[str] = set()
    try:
        for step in range(n_steps):
            action = 2 * torch.rand(action_shape, device=env_device) - 1
            with torch.inference_mode():
                _, reward, _, _, info = env.step(action)

            detailed = info.get("detailed_reward")
            composer = info.get("reward_composer", "sum")
            if not isinstance(detailed, dict) or not detailed:
                return False, f"step {step}: missing/empty detailed_reward (got {detailed!r})"
            seen_term_keys.update(detailed.keys())

            # Per-term key presence: there MUST be at least one real per-term
            # key beyond `total`. Passthrough-only wrapping is a sanity FAIL
            # because the user invoked /harbor:reward-add-log expecting per-term
            # decomposition.
            non_total = [k for k in detailed if k != "total"]
            if not non_total:
                return False, (f"step {step}: detailed_reward has only "
                               f"`total` (no per-term keys) — wrapper fell "
                               f"through to passthrough mode. The env "
                               f"exposes no manager and no logs_rew_*; "
                               f"per-term decomposition isn't possible "
                               f"without code changes.")

            if composer_seen is None:
                composer_seen = composer
            elif composer != composer_seen:
                return False, (f"step {step}: composer changed mid-episode "
                               f"({composer_seen!r} -> {composer!r})")

            # Match-rule per composer.
            if composer == "diagnostic":
                # Direct env w/ logs_rew_*: terms are pre-scale means; only
                # `total` is guaranteed to match env_reward. Verify that.
                if "total" not in detailed:
                    return False, (f"step {step}: composer='diagnostic' but "
                                   f"missing `total` key in detailed_reward")
                tot = detailed["total"]
                if not isinstance(tot, torch.Tensor):
                    tot = torch.as_tensor(tot, device=reward.device,
                                          dtype=reward.dtype)
                if not torch.allclose(tot.float(), reward.float(),
                                      atol=1e-4, rtol=1e-3):
                    diff = (tot - reward).abs().max().item()
                    return False, (f"step {step}: detailed_reward['total'] "
                                   f"!= env_reward (max diff {diff:.3e}); "
                                   f"per-term terms present: {sorted(non_total)}")
                continue

            if composer == "product":
                composed = detailed[next(iter(detailed))].clone()
                for v in list(detailed.values())[1:]:
                    composed = composed * v
            else:  # "sum"
                composed = sum(v for v in detailed.values())
            if not torch.allclose(composed.float(), reward.float(),
                                  atol=1e-4, rtol=1e-3):
                diff = (composed - reward).abs().max().item()
                return False, (f"step {step}: max |{composer}(terms) - reward| "
                               f"= {diff:.3e}; reward={reward.detach().cpu().tolist()} "
                               f"composed={composed.detach().cpu().tolist()}")
    finally:
        try:
            env.close()
        except Exception:
            pass

    n_real = len([k for k in seen_term_keys if k != "total"])
    return True, (f"OK: {composer_seen} match holds for {n_steps}/{n_steps} "
                  f"steps (num_envs={num_envs}, {n_real} per-term keys: "
                  f"{sorted(k for k in seen_term_keys if k != 'total')})")


def main() -> int:
    if not args_cli.env_helper.is_file():
        print(f"[FAIL] env-helper not found: {args_cli.env_helper}", file=sys.stderr)
        return 2

    mod = _load_helper(args_cli.env_helper)
    if not hasattr(mod, "make_isaaclab_env"):
        print(f"[FAIL] {args_cli.env_helper} does not expose make_isaaclab_env",
              file=sys.stderr)
        return 2

    tasks = [t.strip() for t in args_cli.tasks.split(",") if t.strip()]
    device = args_cli.device or ("cuda" if torch.cuda.is_available() else "cpu")

    failures: list[tuple[str, str]] = []
    for task in tasks:
        ok, msg = _check_task(mod, task, args_cli.num_envs, args_cli.steps,
                              device, args_cli.seed)
        marker = "OK   " if ok else "FAIL "
        print(f"[{marker}] {task}: {msg}", flush=True)
        if not ok:
            failures.append((task, msg))

    if failures:
        print(f"\n[FAIL] {len(failures)}/{len(tasks)} task(s) failed the invariant.",
              file=sys.stderr)
        rc = 1
    else:
        print(f"\n[OK] all {len(tasks)} task(s) pass the composer-sum invariant.")
        rc = 0
    return rc


if __name__ == "__main__":
    try:
        rc = main()
    finally:
        try:
            simulation_app.close()
        except Exception:
            pass
    raise SystemExit(rc)
