#!/usr/bin/env python3
"""
Render <repo>/harbor/benchmark-generator/task_overview.md from:
  - <repo>/harbor/benchmark-generator/.task_list.json   (output of list_tasks.py — full registry)
  - <repo>/harbor/benchmark-generator/benchmark-spec.json (smoke-tested subset, with obs/action info)

The template lives at:
  ${CLAUDE_PLUGIN_ROOT}/knowledge/templates/benchmark-generator/task_overview.md.template

Heuristic categorization is plugged in per-family. For families not recognized
the script falls back to a single "other" category — the agent is expected to
hand-edit the result if it wants finer grouping. This script is meant to be
the deterministic skeleton; the agent fills in the prose paragraphs
(distribution / gaps) by Edit'ing the rendered markdown after.

Usage:
    python render_task_overview.py \\
        --repo         <abs path> \\
        --task-list    <repo>/harbor/benchmark-generator/.task_list.json \\
        --spec         <repo>/harbor/benchmark-generator/benchmark-spec.json \\
        --output       <repo>/harbor/benchmark-generator/task_overview.md
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def _categorize_isaaclab(task_id: str) -> str:
    s = task_id[len("Isaac-"):] if task_id.startswith("Isaac-") else task_id
    head = s.split("-")[0]
    if "Velocity" in s or "Position" in s:
        return "locomotion"
    if "Locomanip" in s:
        return "locomanip"
    if head in {"Navigation"}:
        return "navigation"
    if head in {"Cartpole", "Cart", "Ant", "Humanoid", "Quadcopter", "Pendulum"}:
        return "classic-control"
    if head in {"Reach", "Lift", "Open", "Close", "Stack", "Allegro", "Cabinet",
                "Factory", "AutoMate", "Repose", "Insertion", "Shadow", "Franka"}:
        return "manipulation"
    return "other"


def _categorize_dmcontrol(task_id: str) -> str:
    return task_id.split("/")[0] if "/" in task_id else "other"


def _categorize_bidexhands(task_id: str) -> str:
    """Group Bi-DexHands tasks by manipulation skill family."""
    if "Catch" in task_id or task_id.endswith("HandOver") or "Over" in task_id:
        return "catch-and-handover"
    if "Door" in task_id:
        return "door"
    if "Lift" in task_id or "Grasp" in task_id or "Place" in task_id:
        return "pick-and-lift"
    if "Push" in task_id or "BlockStack" in task_id:
        return "push-and-stack"
    if "Pen" in task_id or "ReOrient" in task_id or "Switch" in task_id or "Scissors" in task_id:
        return "in-hand-manipulation"
    if "Bottle" in task_id or "Kettle" in task_id or "SwingCup" in task_id:
        return "tool-use"
    return "other"


def _categorize_generic(task_id: str) -> str:
    head = task_id.split("-")[0].split("_")[0]
    return head.lower() or "other"


def _categorize(family: str, task_id: str) -> str:
    if family == "isaaclab":
        return _categorize_isaaclab(task_id)
    if family == "dm_control":
        return _categorize_dmcontrol(task_id)
    if family == "bidexhands":
        return _categorize_bidexhands(task_id)
    return _categorize_generic(task_id)


def _reward_logger_state(repo: Path, family: str) -> tuple[str, set[str]]:
    """Inspect scripts/_<family>_env.py to determine wrapper coverage. Returns
    ``(mode, explicit_ids)`` where mode is one of:

      - ``"none"``    : no `_DetailedRewardWrapper` found. Every task → "no".
      - ``"dynamic"`` : IsaacLab-style wrapper that reads
        ``reward_manager._step_reward`` at runtime. Coverage is per-task and
        depends on the env entry_point (manager-based → "yes", Direct →
        "total only"; resolved by ``_per_task_reward_logger_token`` below).
      - ``"explicit"``: dm_control-style wrapper with a hand-coded
        `_REWARD_TERM_SPECS` dispatch table. Only the task-id prefixes in
        ``explicit_ids`` get "yes"; everything else gets "no".
    """
    scripts = repo / "scripts"
    if not scripts.is_dir():
        return ("none", set())
    for p in scripts.glob("_*_env.py"):
        try:
            text = p.read_text()
        except Exception:
            continue
        if "_DetailedRewardWrapper" not in text:
            continue
        if family == "isaaclab" or "reward_manager._step_reward" in text:
            return ("dynamic", set())
        explicit: set[str] = set()
        import re
        for m in re.finditer(r'"([A-Za-z0-9_./-]+)"\s*:\s*_[A-Za-z0-9_]+_term_specs', text):
            explicit.add(m.group(1).rstrip("/"))
        return ("explicit", explicit)
    return ("none", set())


def _entry_point_emits_logs_rew(repo: Path, entry_point: str) -> bool:
    """Static-grep the entry_point's source file for the `logs_rew_` literal.

    Direct envs that follow the Factory/Forge/Industreal convention write
    per-term scalar means onto ``self.extras[f"logs_rew_<term>"]`` inside
    a ``_log_*_metrics`` helper. The IsaacLab wrapper picks these up at
    runtime; this static check lets ``task_overview.md`` accurately mark
    the task as ``yes`` (per-term diagnostic curves available) instead of
    ``total only``.

    Heuristic — convert ``mod.path:ClassName`` to ``mod/path.py`` and
    search likely locations under ``<repo>/source/`` (IsaacLab) or the
    repo root itself. Misses are safe (returns False → falls through to
    ``total only``).
    """
    if ":" not in entry_point:
        return False
    mod_path, _ = entry_point.split(":", 1)
    rel = mod_path.replace(".", "/") + ".py"
    candidates = [
        repo / "source" / "isaaclab_tasks" / rel,
        repo / "source" / "isaaclab" / rel,
        repo / rel,
    ]
    for candidate in candidates:
        try:
            if candidate.is_file() and "logs_rew_" in candidate.read_text():
                return True
        except Exception:
            continue
    return False


def _per_task_reward_logger_token(repo: Path, mode: str, explicit_ids: set[str],
                                   task: dict) -> str:
    """Resolve the per-task token for the `Reward logger added` column.

    Tokens (strict — `/harbor:rl-run` greps these):
      - "yes"        : per-term curves available (manager-based with exact
                       decomposition, OR Direct env that writes
                       ``logs_rew_<term>`` scalars onto info — diagnostic
                       per-term means).
      - "total only" : wrapper applied but the env exposes only the
                       summed reward (Direct env, no manager, no
                       ``logs_rew_*``). ``info["detailed_reward"] =
                       {"total": env_rew}``.
      - "no"         : wrapper not applied; ``info["detailed_reward"]``
                       won't be set — `/rl-run` should dispatch
                       `/harbor:reward-add-log` before training.
    """
    if mode == "none":
        return "no"
    if mode == "explicit":
        tid = task["id"]
        return "yes" if any(tid.startswith(p) for p in explicit_ids) else "no"
    # mode == "dynamic": IsaacLab.
    ep = str(task.get("entry_point") or "")
    if "ManagerBasedRLEnv" in ep:
        return "yes"  # exact per-term decomposition via RewardManager
    if _entry_point_emits_logs_rew(repo, ep):
        return "yes"  # diagnostic per-term means via logs_rew_*
    return "total only"


def _spec_lookup(spec: dict) -> dict[str, dict]:
    """Map task_id -> spec entry. Smoke status, obs/action spaces, max_episode_steps."""
    return {t["id"]: t for t in spec.get("tasks", [])}


def _md_table_row(cells: list[str]) -> str:
    return "| " + " | ".join(c.replace("|", "\\|") for c in cells) + " |"


def _render(repo: Path, task_list: dict, spec: dict, project_name: str) -> str:
    plugin_root = Path(__file__).resolve().parents[2]   # <plugin>/scripts/<owner>/<this>.py
    tpl_path = plugin_root / "knowledge" / "templates" / "benchmark-generator" / "task_overview.md.template"
    tpl = tpl_path.read_text()

    family = task_list.get("family", "spec_only")
    listing_fn = task_list.get("listing_function", "(unknown)")
    id_prefix = task_list.get("id_prefix", "")
    tasks = task_list.get("tasks", [])
    spec_by_id = _spec_lookup(spec)
    rl_mode, rl_explicit = _reward_logger_state(repo, family)

    # Per-task rows
    rows: list[str] = []
    cat_counts: Counter[str] = Counter()
    cat_examples: dict[str, list[str]] = {}
    n_reward_impl = 0
    n_reward_logger = 0
    for t in tasks:
        tid = t["id"]
        cat = _categorize(family, tid)
        cat_counts[cat] += 1
        cat_examples.setdefault(cat, []).append(tid)
        smoked = spec_by_id.get(tid)
        if smoked is not None:
            obs = smoked.get("obs_space") or smoked.get("observation_space") or "(see benchmark-spec.json)"
            act = smoked.get("action_space") or "(see benchmark-spec.json)"
            max_steps = smoked.get("max_episode_steps", t.get("max_episode_steps")) or "(unknown)"
            smoke_status = smoked.get("smoke_status", "L1 pass")
            reward_impl = "yes" if smoked.get("reward_implemented", True) else "no"
        else:
            obs = "(unprobed)"
            act = "(unprobed)"
            max_steps = t.get("max_episode_steps") or "(unknown)"
            smoke_status = "not probed"
            reward_impl = "yes"  # default — most tasks have rewards
        if reward_impl == "yes":
            n_reward_impl += 1
        # Reward-logger column — strict tokens for /rl-run grep:
        #   "yes"        — full per-term decomposition at runtime.
        #   "total only" — wrapper applied but Direct env (no RewardManager).
        #   "no"         — no wrapper; /rl-run will dispatch /harbor:reward-add-log.
        rl = _per_task_reward_logger_token(repo, rl_mode, rl_explicit, t)
        if rl == "yes":
            n_reward_logger += 1
        # Description: cheap heuristic — split id into words.
        words = tid.replace("Isaac-", "").replace("-", " ").replace("_", " ").rstrip("0").rstrip("v").rstrip("-").strip()
        desc = words.lower().capitalize() or "(no description)"
        rows.append(_md_table_row([
            f"`{tid}`", cat, desc, reward_impl, rl,
            str(obs), str(act), str(max_steps), smoke_status,
        ]))

    # Categories table
    cat_rows: list[str] = []
    for cat, n in cat_counts.most_common():
        examples = ", ".join(f"`{e}`" for e in cat_examples[cat][:3])
        cat_rows.append(_md_table_row([cat, str(n), examples]))

    distribution_note = (
        f"{family.replace('_', ' ').title()} ships **{len(tasks)}** registered tasks across "
        f"**{len(cat_counts)}** categories. Largest: "
        + ", ".join(f"`{c}` ({n})" for c, n in cat_counts.most_common(3))
        + f". Smoke-tested: **{len(spec_by_id)}** / {len(tasks)} (see `harbor/benchmark-generator/benchmark.md` for the smoke set; the rest are unprobed)."
    )
    gaps_note = (
        "_Auto-generated by `render_task_overview.py`. Hand-edit this paragraph "
        "after generation to call out missing task families (e.g. \"no quadruped "
        "locomotion; could add Spot or Anymal-D\")._"
    )

    reward_check_method = {
        "isaaclab": "presence of `RewardManager` in the env config (manager-based) or `_get_rewards()` (Direct)",
        "dm_control": "presence of `_get_reward()` on the Task subclass",
        "bidexhands": "presence of `compute_hand_reward(...)` (or task-specific equivalent) called from `compute_reward(...)` in `bidexhands/tasks/<task>.py`",
    }.get(family, "presence of a non-trivial reward function in the env source")

    out = tpl
    out = out.replace("{{PROJECT_NAME}}", project_name)
    out = out.replace("{{TASK_COUNT_TOTAL}}", str(len(tasks)))
    out = out.replace("{{TASK_COUNT_SMOKED}}", str(len(spec_by_id)))
    out = out.replace("{{TASK_COUNT_REWARD}}", str(n_reward_impl))
    out = out.replace("{{TASK_COUNT_REWARD_LOGGER}}", str(n_reward_logger))
    out = out.replace("{{LISTING_FUNCTION}}", listing_fn)
    out = out.replace("{{ID_PREFIX}}", id_prefix or "(none)")
    out = out.replace("{{REWARD_CHECK_METHOD}}", reward_check_method)
    out = out.replace("{{CATEGORY_TABLE_ROWS}}", "\n".join(cat_rows))
    out = out.replace("{{TASK_DISTRIBUTION_NOTE}}", distribution_note)
    out = out.replace("{{GAPS_NOTE}}", gaps_note)
    out = out.replace("{{TASK_TABLE_ROWS}}", "\n".join(rows))
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--repo", required=True, type=Path)
    p.add_argument("--task-list", required=True, type=Path)
    p.add_argument("--spec", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--project-name", default=None)
    args = p.parse_args()

    repo = args.repo.resolve()
    task_list = json.loads(args.task_list.read_text())
    spec = json.loads(args.spec.read_text())
    project_name = args.project_name or spec.get("benchmark", {}).get("name", repo.name)

    md = _render(repo, task_list, spec, project_name)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(md)
    print(f"[ok] wrote {args.output} ({len(task_list.get('tasks', []))} tasks, "
          f"family={task_list.get('family')})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
