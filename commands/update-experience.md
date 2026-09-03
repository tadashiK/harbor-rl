---
description: Append a learned experience into a target ledger, or file a probed task into the task-library. Use when the user types /harbor:update-experience target=<name> (experience="..." | file=<path>), or asks "record this lesson", "add this to the reward experience", "file this task into the task-library", "save this implementation spec to the library".
argument-hint: 'target=<name> (experience="<bullet>" | file=<path>)'
---

# /harbor:update-experience — Append to an Experience Ledger or the Task Library

Two jobs depending on `target`:

1. **Agent ledger** (`target` ∈ `reward-tuning-agent` | `task-generator` | `dr-generator` | `rl-tuning-agent`) — append one new **numbered bullet** to that agent's append-only experience ledger.
2. **Task library** (`target = task-library`) — file a `/harbor:probe-task` implementation spec into the correct embodiment folder under `knowledge/experiences/task-library/`, with a short accurate filename.

## Arguments

| Arg | Notes |
|---|---|
| `target` | Required. One of: `reward-tuning-agent`, `task-generator`, `dr-generator`, `rl-tuning-agent`, `task-library`. |
| `experience` | Inline bullet text (ledger targets only). A single heuristic/lesson. |
| `file` | Path to a file. For ledger targets: the file's body is appended as the entry. For `task-library`: a probe-task `*-implementation.md` to copy. |

Provide **exactly one** of `experience=` / `file=`. For `target=task-library`, `experience=` is rejected — only `file=` (an implementation spec) is accepted.

### Target → ledger file map (ledger mode)

| `target` | Ledger file |
|---|---|
| `reward-tuning-agent` | `knowledge/experiences/reward-tuning-agent/reward-experience.md` |
| `task-generator`   | `knowledge/experiences/task-generator/task-experience.md` |
| `dr-generator`     | `knowledge/experiences/dr-generator/dr-experience.md` |
| `rl-tuning-agent`  | `knowledge/experiences/rl-tuning-agent/tuning-experience.md` |

(Paths are under `${CLAUDE_PLUGIN_ROOT}`.)

---

## Mode A — Agent ledger (append a numbered bullet)

### A1 — Resolve & validate the experience text

- If `experience=` is given, that string is the entry body. If `file=` is given, read the file; its full body is the entry.
- **Length check (human bullets):** if the entry came from `experience=` (a hand-written bullet), it MUST be **≤ 5 lines** (count literal `\n`; terminal wrapping doesn't count). If longer, **refuse** and tell the user to shorten it or pass it as a `file=` instead. A `file=` body is exempt from the 5-line cap (it's a curated note) — but warn if it exceeds ~30 lines (ledgers stay skimmable).
- English-only (constraint #1). Reject non-English bullets with a one-line note.
- **High-level heuristics only:** ledger entries state the generalized lesson, NOT the run-specific story. Strip task names, iteration numbers, step counts, and percentage anecdotes before appending (e.g. "porting the base init pose unlocked grasping" — not "on <task> iter 3, return rose 6.9× at 20M steps"). If the entry as given is mostly a specific example, distill it to the heuristic and append that.

### A2 — Compute the next entry number

Read the ledger. Entries are numbered `1.`, `2.`, … at the top level. Find the highest existing number `N`; the new entry is `N+1`. If the ledger body is just the `_(no entries yet)_` / `_(no entries yet — …)_` placeholder, the new entry is `1.` and you REPLACE the placeholder line.

### A3 — Append (never renumber existing entries)

Append the new entry at the end of the numbered list:

```markdown
<N+1>. **<one-line gist>.** <the experience body>
```

If the user's text already opens with a bold gist, keep it; otherwise synthesize a short bold lead so the ledger stays scannable (mirror the existing entries' style). Do NOT touch any existing entry. Preserve the file's header prose.

### A4 — Report

> update-experience: appended entry #<N+1> to `knowledge/experiences/<target>/<file>` (<L> lines).

---

## Mode B — Task library (file a probed implementation spec)

`target=task-library` requires `file=<path>` pointing at a `/harbor:probe-task` output. `experience=` is rejected here.

### B1 — Validate it's a probe-task spec

The file MUST look like an implementation spec:
- first heading matches `# <TaskID> — Implementation Spec`
- carries the five metadata lines: `- robot:`, `- simulator:`, `- objects:`, `- bimanual:`, `- summary:`
- has `## §1` … `## §6` section anchors

If any is missing, refuse: "not an implementation spec — generate one with `/harbor:probe-task` first." Do not copy.

Every entry in the library states those five fields identically, because they are what a
search across the library matches on. A spec that omits one is filed inconsistently and
stops being findable.

### B2 — Classify embodiment (pick the destination folder)

Read the `Task summary:` paragraph + the `## §1` description (that's enough — do NOT read the whole file). Decide:

| Destination | When |
|---|---|
| `manipulation/` | any arm / hand manipulating objects — single-arm or multi-arm/bimanual (pick, place, insert, lift, stack, in-hand) |
| `locomotion/humanoid/`   | bipedal humanoid locomotion |
| `locomotion/quadrupedal/` | quadruped / 4-legged locomotion |

Decision cues, in order: (1) robot morphology named in §1 (Franka/UR/Allegro arm, dual-arm, bimanual → manipulation; Anymal/Go2/Spot → quadrupedal; H1/G1/humanoid → humanoid). If genuinely ambiguous (e.g. a mobile manipulator, or a category not covered by the three folders), **ask the user once** with `AskUserQuestion` listing the three destinations.

Base path: `${CLAUDE_PLUGIN_ROOT}/knowledge/experiences/task-library/<destination>`.

### B3 — Derive a short, accurate filename

The filename is what `task-generator` / `reward-tuning-agent` grep when searching for relevant prior tasks, so it must be **short but accurate** — describe the task, not the gym id verbatim.

- Form: `<short-task-slug>-<simulator>.md`.
- `<simulator>` is the simulator name from the spec's `- simulator:` line, without its
  parenthetical (`IsaacLab (Isaac Sim, manager-based)` → `IsaacLab`).
- `<short-task-slug>`: a concise kebab-case description of what the task DOES, ≤ ~4 words. Derive from the Task summary, not a mechanical lowercase of the TaskID. Examples (TaskID → slug):
  - `Isaac-Dex-Grasp` → `dexterous-grasp` → `dexterous-grasp-IsaacLab.md`
  - `IsaacLab-Franka-StackCube` (stacks three cubes) → `stack-three-cube` → `stack-three-cube-IsaacLab.md`
  - `IsaacLab-Lift-Box` (two arms lift a box) → `lift-box` → `lift-box-IsaacLab.md`

### B4 — Collision-safe copy

```bash
dest_dir="${CLAUDE_PLUGIN_ROOT}/knowledge/experiences/task-library/<destination>"
base="<short-task-slug>-<simulator>"
name="${base}.md"
n=2
while [ -e "${dest_dir}/${name}" ]; do name="${base}-v${n}.md"; n=$((n+1)); done
cp "<file>" "${dest_dir}/${name}"
```

So a second `dexterous-grasp-IsaacLab` becomes `dexterous-grasp-IsaacLab-v2.md`, a third `-v3.md`, etc. **Never overwrite** an existing library file. Copy the spec verbatim (do not rewrite its contents).

### B5 — Report

> update-experience: filed `<file>` → `knowledge/experiences/task-library/<destination>/<name>` (classified: <destination>).

---

## Batch (optional)

If `file=` is a directory (or a glob like `<dir>/*-implementation.md`), iterate B1–B5 per file and print one report line each. Useful for back-filling several probed tasks at once (e.g. the specs sitting in `<repo>/harbor/create-task/`).

## Hard rules

- **Append-only / never renumber** existing ledger entries; **never overwrite** existing task-library files (use the `-vN` suffix).
- **5-line cap** on hand-written (`experience=`) bullets; refuse longer ones.
- **task-library only accepts probe-task implementation specs** — refuse anything else, and refuse `experience=` for that target.
- **English-only.**
- **Copy specs verbatim** into the library — this command never edits the spec body.
