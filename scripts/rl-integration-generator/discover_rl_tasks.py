"""
Discover or validate RL task IDs for a benchmark repo.

Reads <repo>/harbor/benchmark-generator/benchmark-spec.json (written by benchmark-generator).
If the spec already has tasks[], echo them as JSON. Otherwise, try a few
auto-discovery heuristics:

1. <repo>/configs/**/task/*.yaml         — Hydra-style task configs (IsaacGymEnvs / Isaac Lab)
2. gym.envs.registry                     — registered gymnasium envs imported via the benchmark module
3. <repo>/<benchmark>/__init__.py:TASKS  — module-level TASKS list

Always exits 0 — even with zero tasks discovered. The caller (rl-integration-
generator) prompts the user when discovery is empty.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--repo", required=True, type=Path)
    return p.parse_args()


def from_spec(repo: Path) -> list[dict]:
    spec = repo / "harbor" / "benchmark-generator" / "benchmark-spec.json"
    if not spec.exists():
        return []
    try:
        data = json.loads(spec.read_text())
    except json.JSONDecodeError:
        return []
    return data.get("tasks", []) or []


def from_hydra_task_dir(repo: Path) -> list[dict]:
    out: list[dict] = []
    for task_yaml in repo.glob("configs/**/task/*.yaml"):
        out.append({
            "id": task_yaml.stem,
            "make": "hydra",
            "max_episode_steps": None,
            "success_metric": None,
            "reward_metric": "episode_return",
            "_source": str(task_yaml.relative_to(repo)),
        })
    return out


def from_gym_registry() -> list[dict]:
    try:
        import gymnasium as gym
    except ImportError:
        try:
            import gym
        except ImportError:
            return []
    return [
        {
            "id": env_id,
            "make": "gymnasium.make",
            "max_episode_steps": None,
            "success_metric": None,
            "reward_metric": "episode_return",
        }
        for env_id in sorted(gym.envs.registry.keys())[:50]  # cap noise
    ]


def main():
    args = parse_args()
    repo: Path = args.repo.resolve()

    tasks = from_spec(repo)
    source = "benchmark-spec.json"
    if not tasks:
        tasks = from_hydra_task_dir(repo)
        source = "configs/**/task/*.yaml"
    if not tasks:
        tasks = from_gym_registry()
        source = "gym.envs.registry"

    print(json.dumps({
        "ok": True,
        "source": source,
        "count": len(tasks),
        "tasks": tasks,
    }, indent=2))


if __name__ == "__main__":
    main()
