#!/usr/bin/env python3
"""
Pre-flight check for /harbor:rl-run: verify that the chosen task has the
per-term reward wrapper wired (column "reward logger added" in
``<repo>/harbor/benchmark-generator/task_overview.md``).

Exit codes:
  0 = "yes"        — full per-term decomposition; proceed with training.
  3 = "total only" — wrapper is applied but Direct env (no RewardManager);
                     proceed but warn that per-term curves won't appear in
                     W&B (only `reward/total/...`).
  1 = "no"         — wrapper not applied; caller should dispatch
                     /harbor:reward-add-log and then retry.
  2 = task not found in task_overview.md, or task_overview.md missing —
      caller should surface a clean error to the user.

Usage:
    python check_reward_logger.py --repo <abs path> --task <task-id>

The parser reads the markdown table whose header contains "Reward logger
added" and looks up the row whose first cell matches the task ID
(backticks stripped). Tokens recognized: ``yes`` / ``total only`` / ``no``
(case-insensitive).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


_HEADER_TOKEN = "reward logger added"


def _find_table(text: str) -> tuple[int, int] | None:
    """Return (header_idx, columns_count) for the per-task table, or None.

    The header is identified by the presence of the literal column name
    "reward logger added" (case-insensitive). The script does NOT assume a
    fixed column index — different families could add columns later.
    """
    lines = text.splitlines()
    for i, line in enumerate(lines):
        if not line.startswith("|"):
            continue
        cells = [c.strip().lower() for c in line.strip("|").split("|")]
        if _HEADER_TOKEN in cells:
            return i, cells.index(_HEADER_TOKEN)
    return None


def _scan_task(text: str, task_id: str, header_idx: int, rl_col: int) -> str | None:
    """Return 'yes' / 'total only' / 'no' for the task, or None if not found."""
    lines = text.splitlines()
    # Skip header + the markdown separator row that follows it.
    for line in lines[header_idx + 2:]:
        if not line.startswith("|"):
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        first = cells[0].strip("` ")
        if first == task_id:
            if rl_col >= len(cells):
                return None
            token = cells[rl_col].strip().lower()
            if token in ("yes", "no", "total only"):
                return token
            return None
    return None


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--repo", required=True, type=Path)
    p.add_argument("--task", required=True)
    args = p.parse_args()

    overview = args.repo.resolve() / "harbor" / "benchmark-generator" / "task_overview.md"
    if not overview.is_file():
        print(f"[check] task_overview.md not found at {overview}. "
              f"Run benchmark-generator first.", file=sys.stderr)
        return 2

    text = overview.read_text()
    header = _find_table(text)
    if header is None:
        print(f"[check] no 'reward logger added' column found in {overview}. "
              f"Re-render via benchmark-generator (the template may be stale).",
              file=sys.stderr)
        return 2

    header_idx, rl_col = header
    state = _scan_task(text, args.task, header_idx, rl_col)
    if state is None:
        print(f"[check] task {args.task!r} not in task_overview.md. "
              f"Either the task ID is wrong, or task_overview.md is stale "
              f"(re-run benchmark-generator).", file=sys.stderr)
        return 2
    if state == "yes":
        print(f"[check] {args.task}: reward logger present (full per-term) — proceed.")
        return 0
    if state == "total only":
        print(f"[check] {args.task}: reward logger present but Direct env — "
              f"only `reward/total/...` will be logged; per-term curves not "
              f"available without a RewardManager. Proceeding.")
        return 3
    print(f"[check] {args.task}: reward logger MISSING — caller should "
          f"dispatch /harbor:reward-add-log before training.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
