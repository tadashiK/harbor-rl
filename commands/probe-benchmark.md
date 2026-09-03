---
description: Probe an already-set-up benchmark repo and author <repo>/harbor/create-task/task-implementation.md — the per-family task-authoring guide consumed by /harbor:task-create. Use when the user types /harbor:probe-benchmark [repo=<path>] [canonical_task=<id>] or asks "probe this benchmark", "author task-implementation.md", "scaffold the create-task guide for this repo".
argument-hint: "[repo=<path>] [canonical_task=<id>]"
---

# /harbor:probe-benchmark — Author `task-implementation.md`

Inspect an already-set-up benchmark repo (dependency-generator + benchmark-generator Step 3.5/3.6 have run), detect its family, pick a canonical example task, and render `<repo>/harbor/create-task/task-implementation.md` from the template — the single shared file `/harbor:task-create` reads.

`/harbor:task-create` boots three authoring agents (`task-generator` → `reward-tuning-agent` → `dr-generator`) that **do not re-scan the upstream repo** — they trust this doc, so getting it right is part of the probe contract.

## Optional arguments

| Arg | Default | Effect |
|---|---|---|
| `repo` | `$(pwd)` | Absolute or repo-relative path to the benchmark repo root. |
| `canonical_task` | auto-picked (first `tasks[]` entry with `smoke_status` starting `L1`) | Override the canonical example used to mirror §1–§7 from. |

## Pre-flight

```bash
test -f "<repo>/harbor/benchmark-generator/benchmark-spec.json"   || { echo "benchmark-spec.json missing — run benchmark-generator Step 3.5 first"; exit 1; }
test -f "<repo>/harbor/benchmark-generator/task_overview.md"      || { echo "task_overview.md missing — run benchmark-generator Step 3.6 first"; exit 1; }
test -f "<repo>/harbor/dependency-generator/probe.json"            || { echo "probe.json missing — run dependency-generator first"; exit 1; }
test -x "<repo>/.venv/bin/python"               || { echo ".venv/ missing — run /harbor:env-install-uv first"; exit 1; }
```

## Action

1. **Load the contract.** `Read ${CLAUDE_PLUGIN_ROOT}/knowledge/references/benchmark-generator/task-implementation-contract.md` — per-family evidence-gathering rules + the seven-section schema. Do not improvise around it.

2. **Pick `BENCHMARK_FAMILY`** using the contract's detection cues (`isaaclab-manager-based` / `isaaclab-direct` / `dexteroushands` / `loco-mujoco` / `dm_control` / `gymnasium-generic`). If the repo doesn't fit, ask the user via a single `AskUserQuestion` with the six options — do not silently pick one.

3. **Pick `CANONICAL_EXAMPLE_TASK_ID`**:
   - If `canonical_task=<id>` was passed, verify it appears in `benchmark-spec.json:tasks[]` with `smoke_status` starting `L1`; reject otherwise.
   - Else auto-pick from `tasks[]` where `smoke_status` starts `L1`. If no task smoke-passed, refuse to author the doc and surface that to the user (the downstream agents would have nothing to mirror).

4. **Locate the canonical example file** (the contract's per-family table tells you where to look). Open it with `Read` and confirm the section anchors exist:
   - registration entry
   - scene/cfg class
   - actions definition
   - reset hook
   - termination + (optional) command
   - observations definition
   - rewards definition
   - DR / event terms

   If any anchor is missing, pick a different smoke-passing task as the canonical example — never fall back to "skeleton-only" for §1 or §2.

5. **Render the doc** by copying `${CLAUDE_PLUGIN_ROOT}/knowledge/templates/benchmark-generator/task-implementation.md.template` and filling every `{{...}}` placeholder. Each section has five sub-blocks (Description / File pointers / Code template / Decisions / Smoke check); the contract's per-family table tells you where to source File pointers.

6. **Verify §1's smoke before saving.** The §1 smoke command must pass today against the canonical example. Run it inside the venv:

   ```bash
   cd "<repo>" || exit 1
   # Build <canonical_id> per *Does the task build?* in agent-conventions.md and keep the
   # printed observation/action spaces.
   ```

   (or the family equivalent — see contract). Paste the literal stdout into `{{REGISTER_SMOKE_EXPECTED}}`. §2..§7 smoke commands are written but NOT executed at this stage — they become the per-phase smoke for the future agents.

7. **Write the file**:
   ```bash
   mkdir -p "<repo>/harbor/create-task"
   ```
   Then `Write` the rendered content to `<repo>/harbor/create-task/task-implementation.md`.

8. Surface a one-line note: `task-implementation.md authored — /harbor:task-create can now run.`

## Hard rules

(full list in the contract)

- English-only.
- Regenerated on every re-run; hand-edits are lost.
- Code templates ≤ 60 lines each, with `<TaskName>` / `<Robot>` / `<Object>` placeholders — never paste the canonical task verbatim.
- Skeleton is fixed: seven sections in order. If a section is unsupported by the family (e.g. delta-EE-pose actions in `isaaclab-direct`), keep the header and write `<unsupported in this benchmark>` under each sub-block; do not delete the section.

## On failure

- Cannot detect family → ask user (one `AskUserQuestion`).
- No canonical example smoke-passed → emit a stub doc with `BENCHMARK_FAMILY: <unknown>` and report `task-implementation: skipped (no smoke-passing canonical example)`; future `/harbor:task-create` will refuse to run.
- §1 smoke fails on the canonical example → that means dependency-generator / benchmark-generator already had a problem; surface up rather than fabricating expected output.

## Constraints

- **Do NOT** edit `benchmark-spec.json` / `task_overview.md` / `probe.json` — read-only inputs.
- **Do NOT** invoke other generator subagents — this command authors the guide; downstream agents are dispatched by `/harbor:task-create`.
- **Do NOT** widen the seven-section schema — additions belong in the contract, not in individual probes.
