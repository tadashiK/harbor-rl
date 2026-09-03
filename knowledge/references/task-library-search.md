# Task-library search (returns ONE selected task)

Run FIRST by whoever needs a design base — `/harbor:task-create` Step 1.5, the `/harbor:reward-tune`
orchestrator (standalone), or the `reward-tuning-agent` itself when no base was passed down. It searches the cross-run **task-library** and returns the **single most-relevant** proven task
to adapt from. Every library spec is a verified, successfully-trained task (a full §1..§7 spec named
`<short-task>-<repo>.md`, filled by `/harbor:update-experience target=task-library`).

Selection only. How to build from the returned base is `adapt-first.md`.

## Step 1 — Classify the new task

From `description` + `task_id`, pick the folder under `${CLAUDE_PLUGIN_ROOT}/knowledge/experiences/task-library/`:

| Folder | When |
|---|---|
| `manipulation/` | any arm/hand manipulating objects — single- or multi-arm/bimanual (pick, place, insert, lift, stack, in-hand) |
| `locomotion/humanoid/`   | bipedal humanoid locomotion |
| `locomotion/quadrupedal/` | quadruped locomotion |

If the task straddles categories, search the closest folder first, then the sibling.

## Step 2 — Find candidates, then select one

```bash
LIB="${CLAUDE_PLUGIN_ROOT}/knowledge/experiences/task-library/<folder>"
ls "$LIB"
grep -ril "<verb|object|robot keywords>" "$LIB"
```

**Review EVERY result — never truncate this discovery search.** No `head`/`tail`/limit; the folder is
small and the best match is often alphabetically adjacent to a near-miss. Sort for readability if you
must (`| sort`), never `| head`.

Shortlist the **top 3** by verb (pick / place / lift / insert / stack …) + goal structure + object +
robot, **read each of the 3 in full** (what it actually does — scene, goal, action, reward), and
**return the single most relevant** as the `design_base`.

**Ranking priority — earlier axes dominate later ones:**

1. **Task + goal structure** (what the episode actually rewards / terminates on) — a place-into-container
   task is closest to another place-into-container task, NOT a lift task that only shares the arm.
2. **Robot / embodiment** (arm + hand DoF, dexterous hand vs parallel-jaw gripper).
3. **Simulator / benchmark family** (IsaacLab vs ManiSkill vs …).

Never let simulator match — or whether a spec is build-verified in a harbor harness — outrank a better
task + goal match. A proven place task in a different-but-portable sim beats a lift task in the same sim:
return the better task match as the `design_base` and graft its goal / reward half onto a same-sim
chassis. Break ties with a lower axis only once the higher axes are equal.

If no spec shares the verb / object / embodiment, return **none** → pure creation mode.
