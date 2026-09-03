#!/usr/bin/env python3
"""
Capture an RL benchmark spec into <repo>/harbor/benchmark-generator/benchmark-spec.json.

Captures the surface that rl-integration-generator + the RL training commands
(/harbor:rl-run, /harbor:rl-tune) need to set up training:

  - tasks list (id + max_episode_steps + reward_implemented + success_metric)
  - language: pytorch | jax
  - gpu_sim: bool (does this benchmark expose massively-parallel GPU envs?)
  - category: always "rl" (every repo reaching this script is treated
    as an RL benchmark; the field is kept for downstream consumers that still
    read it)

Inputs are mostly user-provided (the agent prompts the user during Step 4.5
and passes the answers via flags) since auto-discovering "is reward
implemented?" reliably is not possible without running each task.

Usage:
    python capture_spec.py \\
        --repo /abs/repo \\
        --benchmark-name <slug> \\
        --tasks <id1>,<id2>,...        \\
        --reward-implemented           \\   # one flag per task; ordered
        --reward-implemented           \\
        --max-episode-steps 1000,500   \\   # comma-separated, same order
        --success-metrics success,success \\
        --language pytorch             \\
        --gpu-sim true

If --tasks-json is supplied (a JSON file or inline JSON), it overrides the
flat flags above:

    --tasks-json '[{"id":"Ant","max_episode_steps":1000,"reward_implemented":true,"success_metric":"success"}]'
"""
from __future__ import annotations

import argparse
import datetime
import json
import sys
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--repo", required=True, type=Path)
    p.add_argument("--benchmark-name", required=True)
    p.add_argument("--tasks", default="",
                   help="comma-separated task IDs (use --tasks-json for full task records)")
    p.add_argument("--tasks-json", default="",
                   help="JSON array of task objects; overrides --tasks if set")
    p.add_argument("--max-episode-steps", default="",
                   help="comma-separated ints, one per task (default 1000 each)")
    p.add_argument("--success-metrics", default="",
                   help="comma-separated metric names, one per task; empty entry = null")
    p.add_argument("--reward-implemented", action="append", default=[],
                   help="repeated flag; one entry per task in order. Empty list = all true.")
    p.add_argument("--language", choices=["pytorch", "jax"], required=True)
    p.add_argument("--gpu-sim", choices=["true", "false"], required=True)
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def build_tasks(args) -> list[dict]:
    if args.tasks_json:
        raw = args.tasks_json
        if Path(raw).is_file():
            raw = Path(raw).read_text()
        return json.loads(raw)

    ids = [t.strip() for t in args.tasks.split(",") if t.strip()]
    if not ids:
        raise SystemExit("--tasks empty and --tasks-json not provided")

    mes_raw = [s.strip() for s in args.max_episode_steps.split(",") if s.strip()]
    mes = [int(s) for s in mes_raw] if mes_raw else [1000] * len(ids)
    sm_raw = args.success_metrics.split(",") if args.success_metrics else []
    sm = [s.strip() or None for s in sm_raw] if sm_raw else [None] * len(ids)
    ri = args.reward_implemented if args.reward_implemented else [True] * len(ids)
    # Coerce string flags to bool — accept the literal "false" / "0" as False.
    ri = [str(x).lower() not in ("false", "0", "no") for x in ri]

    if not (len(ids) == len(mes) == len(sm) == len(ri)):
        raise SystemExit(
            f"Lengths disagree: tasks={len(ids)} max_ep_steps={len(mes)} "
            f"success_metrics={len(sm)} reward_implemented={len(ri)}"
        )

    return [
        {"id": i, "max_episode_steps": m, "success_metric": s, "reward_implemented": r,
         "reward_metric": "episode_return"}
        for i, m, s, r in zip(ids, mes, sm, ri)
    ]


def main() -> int:
    args = parse_args()
    repo = args.repo.resolve()
    if not repo.is_dir():
        print(f"[error] repo path is not a directory: {repo}", file=sys.stderr)
        return 2

    record = {
        "schema_version": 1,
        "captured_at": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "benchmark_name": args.benchmark_name,
        "category": "rl",
        "language": args.language,
        "gpu_sim": args.gpu_sim == "true",
        "tasks": build_tasks(args),
    }

    if args.dry_run:
        print(json.dumps(record, indent=2))
        return 0

    out = repo / "harbor" / "benchmark-generator" / "benchmark-spec.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    # Merge with existing spec if any (preserve unrelated fields like obs_spec).
    existing = {}
    if out.exists():
        try:
            existing = json.loads(out.read_text())
        except json.JSONDecodeError:
            existing = {}
    existing.update(record)
    out.write_text(json.dumps(existing, indent=2) + "\n")
    print(f"[ok] wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
