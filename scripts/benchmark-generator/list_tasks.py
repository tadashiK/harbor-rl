#!/usr/bin/env python3
"""
Dump the registered-task universe for a benchmark repo as JSON, by sniffing
which task-listing API is available in the repo's .venv. Used by
`benchmark-generator` to fill the {{TASK_COUNT_TOTAL}} / {{TASK_TABLE_ROWS}}
fields in `task_overview.md.template`.

Recognized families (auto-detect via importability):
  - **isaaclab**: `<repo>/_isaac_sim/` symlink or `isaaclab` importable.
                  Lists `gymnasium.envs.registry` filtered by prefix `Isaac-`.
                  Does NOT instantiate envs (no AppLauncher needed).
  - **bidexhands**: `bidexhands` importable AND `<repo>/bidexhands/cfg/` exists
                    with one `<TaskName>.yaml` per task. Lists every yaml whose
                    stem is also imported by `bidexhands.utils.parse_task`.
                    No env instantiation needed (no Isaac Gym sim cost).
  - **dm_control**: `dm_control` importable. Uses `dm_control.suite.ALL_TASKS`.
  - **gymnasium / gym**: any other repo with gymnasium installed. Lists every
                         registered ID (no filter).
  - **fallback**: nothing recognized → reads `harbor/benchmark-generator/benchmark-spec.json` and
                  reports only the tasks captured there (so the agent still
                  has *something* to template against).

Output (stdout JSON):
    {
      "family": "isaaclab" | "dm_control" | "gymnasium" | "spec_only",
      "listing_function": "<human-readable string the template embeds>",
      "id_prefix": "Isaac-" | "" | ...,
      "tasks": [
        {"id": "...", "entry_point": "...", "max_episode_steps": int|null}
      ]
    }

Run inside the repo's venv:
    <repo>/.venv/bin/python <plugin>/scripts/benchmark-generator/list_tasks.py \\
        --repo <abs path>
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _spec_only_fallback(repo: Path) -> dict:
    spec = repo / "harbor" / "benchmark-generator" / "benchmark-spec.json"
    if not spec.is_file():
        return {"family": "spec_only", "listing_function": "(none)", "id_prefix": "",
                "tasks": []}
    data = json.loads(spec.read_text())
    return {
        "family": "spec_only",
        "listing_function": "harbor/benchmark-generator/benchmark-spec.json (no live env registry detected)",
        "id_prefix": "",
        "tasks": [{"id": t["id"],
                   "entry_point": t.get("make", "gymnasium.make"),
                   "max_episode_steps": t.get("max_episode_steps")}
                  for t in data.get("tasks", [])],
    }


def _list_isaaclab() -> dict | None:
    """IsaacLab listing requires AppLauncher to be running first — pxr (USD)
    is only resolvable after Kit boots. We launch headlessly with no
    cameras (cheap experience file), import isaaclab_tasks (which registers
    every Isaac-* gym id), then read gym.envs.registry. ~15-25s warmup."""
    try:
        from isaaclab.app import AppLauncher
    except ImportError:
        return None
    import argparse
    parser = argparse.ArgumentParser(add_help=False)
    try:
        AppLauncher.add_app_launcher_args(parser)
        args, _ = parser.parse_known_args([])
        args.headless = True
        args.enable_cameras = False
        AppLauncher(args)  # blocks until Kit is ready
    except Exception:
        return None
    try:
        import gymnasium as gym
        import isaaclab_tasks  # noqa: F401  — registers Isaac-* gym ids
    except ImportError:
        return None
    tasks = []
    for env_id, spec in gym.envs.registry.items():
        if not env_id.startswith("Isaac-"):
            continue
        tasks.append({
            "id": env_id,
            "entry_point": str(spec.entry_point),
            "max_episode_steps": getattr(spec, "max_episode_steps", None),
        })
    return {
        "family": "isaaclab",
        "listing_function": "gymnasium.envs.registry, prefix='Isaac-' (after `AppLauncher(headless=True)` + `import isaaclab_tasks`)",
        "id_prefix": "Isaac-",
        "tasks": tasks,
    }


def _list_bidexhands(repo: Path) -> dict | None:
    """Bi-DexHands (PKU-MARL DexterousHands) — Isaac Gym Preview 4 based.

    Tasks are not registered with gymnasium; the canonical list lives as
    one ``<TaskName>.yaml`` per task in ``<repo>/bidexhands/cfg/``. We
    cross-check against the imports in ``bidexhands.utils.parse_task`` so
    we only emit task IDs that the factory `bi.make(task, "ppo")` can
    actually instantiate (skipping meta tasks which require special
    `task_type`).
    """
    cfg_dir = repo / "bidexhands" / "cfg"
    parse_task_py = repo / "bidexhands" / "utils" / "parse_task.py"
    if not (cfg_dir.is_dir() and parse_task_py.is_file()):
        return None
    try:
        text = parse_task_py.read_text()
    except Exception:
        return None
    # Imported task class names from parse_task.py — these are exactly the
    # tasks `bi.make` can build via `eval(args.task)(...)`.
    import re as _re
    imported = set(_re.findall(r"from bidexhands\.tasks\.[\w.]+ import (\w+)", text))
    # Drop meta variants — they require args.task_type == "Meta" and are not
    # exposed via `bi.make(...)` for non-meta algos.
    imported = {t for t in imported if "Meta" not in t}
    tasks = []
    for yml in sorted(cfg_dir.glob("*.yaml")):
        name = yml.stem
        if name not in imported:
            continue
        # Parse episodeLength via lightweight regex (avoid importing yaml here).
        ep_len = None
        try:
            for line in yml.read_text().splitlines():
                s = line.strip()
                if s.startswith("episodeLength:"):
                    ep_len = int(s.split(":", 1)[1].strip())
                    break
        except Exception:
            pass
        tasks.append({
            "id": name,
            "entry_point": f"bidexhands.tasks.{name}:{name}",
            "max_episode_steps": ep_len,
        })
    if not tasks:
        return None
    return {
        "family": "bidexhands",
        "listing_function": "bidexhands/cfg/*.yaml ∩ imports in bidexhands/utils/parse_task.py",
        "id_prefix": "",
        "tasks": tasks,
    }


def _list_dm_control() -> dict | None:
    try:
        from dm_control import suite
    except ImportError:
        return None
    tasks = []
    for domain, task in suite.ALL_TASKS:
        tasks.append({
            "id": f"{domain}/{task}",
            "entry_point": "dm_control.suite.load",
            "max_episode_steps": None,
        })
    return {
        "family": "dm_control",
        "listing_function": "dm_control.suite.ALL_TASKS",
        "id_prefix": "",
        "tasks": tasks,
    }


def _list_gymnasium() -> dict | None:
    try:
        import gymnasium as gym
    except ImportError:
        return None
    tasks = []
    for env_id, spec in gym.envs.registry.items():
        tasks.append({
            "id": env_id,
            "entry_point": str(spec.entry_point),
            "max_episode_steps": getattr(spec, "max_episode_steps", None),
        })
    return {
        "family": "gymnasium",
        "listing_function": "gymnasium.envs.registry (no prefix filter)",
        "id_prefix": "",
        "tasks": tasks,
    }


def _detect(repo: Path) -> dict:
    if (repo / "_isaac_sim").exists():
        result = _list_isaaclab()
        if result is not None:
            return result
    # bidexhands check is path-anchored, so it's tried before generic
    # gymnasium fallback (Bi-DexHands envs aren't gym-registered).
    bdx = _list_bidexhands(repo)
    if bdx is not None and bdx["tasks"]:
        return bdx
    for fn in (_list_isaaclab, _list_dm_control, _list_gymnasium):
        r = fn()
        if r is not None and r["tasks"]:
            return r
    return _spec_only_fallback(repo)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--repo", required=True, type=Path)
    p.add_argument("--output", type=Path, default=None,
                   help="write JSON to this path instead of stdout (recommended for IsaacLab — "
                        "Kit logs to stdout and pollutes the JSON output otherwise)")
    args = p.parse_args()
    repo = args.repo.resolve()
    if not (repo / "harbor").is_dir():
        print(f"[fail] {repo} is not a harbor benchmark repo (no harbor/)",
              file=sys.stderr)
        return 2
    out = _detect(repo)
    payload = json.dumps(out, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload)
        print(f"[ok] wrote {len(out['tasks'])} tasks to {args.output} (family={out['family']})",
              file=sys.stderr)
    else:
        print(payload)
    return 0


if __name__ == "__main__":
    sys.exit(main())
