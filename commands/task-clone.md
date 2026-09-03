---
description: Clone a task into an isolated, independently-editable copy — a GENERAL primitive, not tied to any single caller. Same-repo mode registers the copy under a new suffixed gym id (suffix is caller-chosen, e.g. -rewarditer7, -abtest1); cross-benchmark mode (dest_repo=, sim2sim) is COMING SOON and refuses in this release. `op=create` dispatches the task-cloner subagent (copy source's editable surface → rewire imports → register <dest> → run clone smokes). `op=delete` removes the clone's files and confirms the source still builds. Use when the user types /harbor:task-clone op=create source=<TaskID> dest=<TaskID> | op=delete dest=<TaskID>, or asks "clone task X for isolated editing", "copy this task under a new id".
argument-hint: "op=create source=<TaskID> dest=<TaskID> [repo=<path>] [dest_repo=<path>] [surface=<sections>] [info_out=<path>] | op=delete dest=<TaskID> [repo=<path>]"
---

# /harbor:task-clone — Isolated Task Clone (general primitive)

Make a standalone copy of a task that can be edited **without touching the source or any sibling clone**. Same-repo clones are registered under a new gym id (`<source>` with a caller-chosen suffix inserted **before** the `-vN` token) and verified to build + roll out + inherit per-term logging.

This is a general-purpose primitive. Known callers / use cases (not exhaustive — new flows should reuse it rather than re-implement copying):

| Use case | Mode | Notes |
|---|---|---|
| A/B experiments on any section (actions, obs, reset, reward) | same-repo | suffix free-form (e.g. `-abtest1`); widen `surface=` to the sections being edited. The variant is edited while the source stays intact. |
| sim2sim / cross-benchmark migration (e.g. IsaacLab → Genesis) | cross-repo (`dest_repo=`) | **COMING SOON** — not available in this release; see "Cross-benchmark mode" below. |

Cloning is serialized in the caller (one clone at a time per repo), so registration never races.

`/harbor:reward-tune` is a caller whenever its effective pool exceeds 1: it creates one slot clone per in-flight candidate, copying the full editable surface (candidates vary §1–§5 as well as the reward). It calls `clone_task.py` directly rather than dispatching this command. A sequential tune clones nothing.

## Required arguments

| Arg | Notes |
|---|---|
| `op` | `create` or `delete`. |
| `source` | (create) Existing task id; `gym.make(<source>)` must succeed. |
| `dest` | New task id. Same-repo: must be valid for `gym.register` (word chars / `:` / `.` / `-` only — **no `#`**) and must insert its suffix before any `-vN` version token; the suffix itself is caller-chosen. Cross-repo: must follow the DEST family's naming convention. |

## Optional arguments

| Arg | Default | Effect |
|---|---|---|
| `repo` | `$(pwd)` | Source repo root. |
| `dest_repo` | (none) | **COMING SOON** — cross-benchmark (sim2sim) mode is not released; passing it is refused. Omitted ⇒ same-repo clone, which is fully supported. |
| `surface` | `reward` | Which editable surface to copy (same-repo mode): `reward` (env_cfg + reward mdp modules — the default), or a comma list widening it (e.g. `surface=reward,actions,observations`) for experiments that edit more than §6. |
| `info_out` | `<repo>/harbor/clones/<dest>.json` | Where to write the clone manifest (caller passes its own dir, e.g. `<task_dir>/iter_<NNN>/clone-info.json`). |

## Action

### op=create (same-repo — `dest_repo` omitted)

1. **Pre-flight** (in `repo`):
   ```bash
   test -x .venv/bin/python || exit 1
   # <source> must build — see *Does the task build?* in agent-conventions.md.
   ```
   Validate `<dest>` is a legal gym id and differs from `<source>`. Reject a `#` or a suffix placed after `-vN`.

2. **Dispatch the `task-cloner` subagent** (from the main thread; `task-cloner` is a leaf agent):
   ```
   Agent(task-cloner, prompt={
     repo_path: <abs>,
     source_id: <source>,
     dest_id:   <dest>,
     surface:   [<sections — default ["reward"]>],
     info_out:  <info_out abs>
   })
   ```
   The agent calls the deterministic `scripts/task-cloner/clone_task.py` to copy the source's editable surface (env_cfg + the task-local mdp package the requested surface touches), rewire the cloned cfg's imports to the copies, register `<dest>`, and write the manifest — then runs the sim-side clone smokes on top. See `agents/task-cloner.md`.

3. **Return** the agent verdict + manifest path. On `status: fail`, the agent has already removed any partial clone — surface the error and exit non-zero.

### op=create — Cross-benchmark mode (`dest_repo=` given, sim2sim) — COMING SOON

**Not available in this release.** When `dest_repo` is given, refuse before doing any work:

> Cross-benchmark (sim2sim) migration is coming soon and is not part of this release.
> Same-repo cloning works — re-run without `dest_repo=`.

Do not probe, do not reproduce, do not create anything in either repo. The design below is
what it will do; it is documentation, not an instruction to follow today:

1. **Pre-flight**: source repo passes the same checks as above; `dest_repo` must have completed its own bootstrap chain (`.venv/`, `benchmark-spec.json`, `harbor/create-task/task-implementation.md`) — if not, run the missing stages there first (same rule as `/harbor:task-create` Step 0).
2. **Probe the source**: run `/harbor:probe-task task=<source> repo=<repo>` → emits the portable per-task spec `<slug>-implementation.md` (§1..§7 design: scene, actions, reset, goal, obs, reward, DR).
3. **Reproduce in the destination**: run `/harbor:task-create name=<dest> from=<that spec> [assets=...]` inside `dest_repo`. Reproduce mode pastes each section's design (adapting expression to the dest family's idioms via its `task-implementation.md`), and §6 is still validated by the reward-tune training loop in the dest simulator — a reward proven in the source sim is NOT assumed to transfer (physics/contacts differ; that's the point of sim2sim).
4. **Manifest**: write `info_out` with `{mode: "cross-benchmark", source_id, source_repo, dest_id, dest_repo, spec_path, task_dir}` so the migration is traceable. Deletion of a cross-benchmark clone = deleting the dest task via the dest repo's normal task removal (the manifest records the dest `task_dir`).

### op=delete

Lightweight, no subagent. Route the file/registration removal through the same deterministic tool (it reads `cloned_files[]` + `cloned_dirs[]` + the registration anchor from the manifest and removes them, then asserts the source registration is intact):

```bash
python3 "${CLAUDE_PLUGIN_ROOT}/scripts/task-cloner/clone_task.py" --op delete \
    --repo "<repo>" --manifest "<info_out (or <repo>/harbor/clones/<dest>.json)>"
```

Then confirm the source still builds (the running-sim check the script leaves to the caller):

```bash
# <source_from_manifest> must still build — see *Does the task build?* in agent-conventions.md.
```

Delete is idempotent — a missing manifest or already-removed files is a no-op success (the clone is gone either way).

## Constraints

- **Never edit the source task's files.** A clone only writes NEW files + one registration entry (same-repo), or new files in the DEST repo only (cross-benchmark).
- **Registration suffix goes before `-vN`** (`Isaac-Lift-Cube-Franka-v0` → `Isaac-Lift-Cube-Franka-rewarditer7-v0`), never after. The suffix text is the caller's choice — `rewarditer<NNN>` is merely reward-tune's convention.
- **Clone creation is serialized by the caller.** This command assumes it is not invoked concurrently for the same repo (whatever orchestrator calls it guarantees this); trainings parallelize, cloning does not.
- **Clean up on failure.** A create that fails any smoke removes its partial files before returning — no orphans.
- **Cross-benchmark mode migrates design, not files.** It always routes through probe-task → task-create reproduce in the dest repo; the dest reward must re-earn its training validation in the dest simulator.

## Examples

```text
# reward-tune pooled candidate 7 (same-repo), manifest into the candidate dir
/harbor:task-clone op=create source=Isaac-Lift-Cube-Franka-v0 dest=Isaac-Lift-Cube-Franka-rewarditer7-v0 \
    info_out=harbor/create-task/isaac-lift-cube-franka-v0/iter_007/clone-info.json

# A/B experiment editing actions + reward on the copy (same-repo, wider surface)
/harbor:task-clone op=create source=Isaac-Lift-Cube-Franka-v0 dest=Isaac-Lift-Cube-Franka-abtest1-v0 \
    surface=reward,actions

# sim2sim (COMING SOON — this form is refused in the current release)
/harbor:task-clone op=create source=Isaac-Stack-Two-Cubes-Franka-v0 dest=Genesis-Stack-Two-Cubes-Franka-v0 \
    dest_repo=/home/me/code/Genesis

# Delete a same-repo clone after the candidate is scored
/harbor:task-clone op=delete dest=Isaac-Lift-Cube-Franka-rewarditer7-v0 \
    info_out=harbor/create-task/isaac-lift-cube-franka-v0/iter_007/clone-info.json
```
