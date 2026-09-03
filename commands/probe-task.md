---
description: Probe an existing task in the current benchmark repo and emit `<task-slug>-implementation.md` — a portable per-task design-choice spec capturing scene / actions / reset / termination / observation / reward / DR. Use when the user types /harbor:probe-task task=<id> [repo=<path>] [output=<path>] or asks "document this task", "make a reproduction spec for task X", "extract the design choices of task Y".
argument-hint: "task=<id> [repo=<path>] [output=<path>]"
---

# /harbor:probe-task — Author Per-Task Reproduction Spec

Read an existing task's source (registration entry, env_cfg, `mdp/` tree, asset paths) and emit a single self-contained `<task-slug>-implementation.md` documenting every design choice. The output is a portable reproduction spec — feeding it back into `/harbor:task-create from=<path>` rebuilds the task identically on any other benchmark of the same family.

Distinct from `/harbor:probe-benchmark`, which authors the *family-level* guide (`task-implementation.md`). probe-task is *per-task* and contains the actual code / values, not placeholders.

## Execution model — run in a subagent (REQUIRED)

Probing reads a large volume of source and pastes it verbatim into the spec. **The main thread MUST NOT do this work inline** — it would bloat the main conversation context with file dumps never needed again. Instead, the main thread:

1. Resolves the args (`task`, `repo`, `output`) and runs nothing else.
2. Dispatches **one** subagent (`Agent(subagent_type="general-purpose")`) whose prompt is this command body (Pre-flight + Action + Hard rules) plus the resolved args. The subagent does all reading/extraction, writes the spec to `<output>`, and self-verifies.
3. The subagent returns **only** the one-line summary from step 8 (path + section/func/obs counts + the reproduce hint) — never the spec contents.
4. The main thread relays that one line to the user.

The heavy reads live and die in the subagent's isolated context. The steps below are written for that subagent to follow.

## Required argument

| Arg | Notes |
|---|---|
| `task` | Task ID already registered in `<repo>/harbor/benchmark-generator/benchmark-spec.json:tasks[].id` and importable via `gym.make`. |

## Optional arguments

| Arg | Default | Effect |
|---|---|---|
| `repo` | `$(pwd)` | Benchmark repo root. |
| `output` | `<repo>/harbor/create-task/<task-slug>-implementation.md` | Where to write the spec. |

## Pre-flight

```bash
test -f "<repo>/harbor/benchmark-generator/benchmark-spec.json"             || { echo "benchmark-spec.json missing — run benchmark-generator first"; exit 1; }
test -x "<repo>/.venv/bin/python"                         || { echo ".venv/ missing — run /harbor:env-install-uv first"; exit 1; }
```

Then build the task per *Does the task build?* in `agent-conventions.md`, and refuse to probe
only if the family's real constructor fails. A bare `gym.make('<task>')` is NOT a build check —
it raises `TypeError: missing 1 required positional argument: 'cfg'` for every manager-based
task, so gating on it refuses healthy ones.

## Action

1. **Resolve family** from `<repo>/harbor/create-task/task-implementation.md` (`BENCHMARK_FAMILY` header) if present, else fall back to family detection per the `task-implementation-contract.md` cues.

2. **Locate per-task source files.** The per-family table in `${CLAUDE_PLUGIN_ROOT}/knowledge/references/benchmark-generator/task-implementation-contract.md` maps `<task_id>` → source files. For `isaaclab-manager-based` (the most common): walk `source/isaaclab_tasks/isaaclab_tasks/manager_based/<category>/<task_root>/` for:
   - `config/<robot>/__init__.py` — `gym.register(id=…, entry_point=…, kwargs=…)`
   - `config/<robot>/joint_pos_env_cfg.py` (or family equivalent) — per-robot scene + action + reset wiring
   - `<task_root>_env_cfg.py` — abstract base: SceneCfg, ActionsCfg, ObservationsCfg, EventCfg (reset + DR), RewardsCfg, TerminationsCfg, EnvCfg
   - `mdp/{actions,actions_cfg,observations,rewards,terminations}.py` — task-local function defs

   For other families, follow the contract's per-family layout.

3. **Extract design choices per section** (read with `Read`; never paraphrase code — paste verbatim):

   - **§1 Registration + Scene**
     - `gym.register(id, entry_point, kwargs)` block.
     - `SceneCfg`: robot articulation cfg (USD path, prim_path, init joint pose, actuator stiffness/damping), objects (`RigidObjectCfg`, `ArticulationCfg`), all `FrameTransformerCfg` / `ContactSensorCfg` / camera defs, table + ground + lights. List resolved USD/asset paths verbatim.
   - **§2 Actions**
     - `ActionsCfg`: arm_action class + ctor params (scale, alpha, body_offset, IK controller cfg, pos/joint limits, forbidden zones); gripper_action class + commands.
   - **§3 Reset**
     - `EventCfg`: every `mode="reset"` term with its function, `params` dict, and ranges. Include `position_range`, `velocity_range`, `pose_range`, `asset_cfg` exactly.
   - **§4 Goal + Termination**
     - `TerminationsCfg`: every DoneTerm with `func`, `params`, `time_out` flag.
     - `CommandsCfg` (if not `None`): command term definitions + ranges.
   - **§5 Observation**
     - `ObservationsCfg.PolicyCfg`: every `ObsTerm` (`func`, `params`, `noise`), the `__post_init__` flags (`enable_corruption`, `concatenate_terms`), and any masking/zero-gate references inside `mdp/observations.py`. Resolved total obs dim.
   - **§6 Reward** (fully reproducible)
     - `RewardsCfg`: every `RewTerm` with `func`, `params`, `weight`.
     - Composer (sum / product).
     - **Embed the full source of every reward function from `mdp/rewards.py` (verbatim, in a code block).** Include latch buffers, helpers, gate logic — everything the function needs to run.
     - The planning-budget docstring per `knowledge/experiences/reward-tuning-agent/reward-experience.md` entry #2 (per-stage saturated per-step magnitudes) — extract from the `RewardsCfg` docstring if present; else compute from the weights and note "(retro-computed)".
   - **§7 DR**
     - `EventCfg`: every term with `mode != "reset"` (i.e. `startup` / `interval`). Function, params, ranges. If none, write `<no DR>`.

4. **Fill the metadata block.** `robot` and `objects` come from the §1 scene you just read;
   `bimanual` is true when the scene holds two arms the policy drives independently, and `summary`
   is one sentence naming what the robot must do. These five
   fields are what the task-library is searched on, so they are stated the same way for every
   entry.

5. **Render the doc** into a single self-contained markdown with this top-level structure:
   ```
   # <task_id> — Implementation Spec

   - robot: <make/model + DoF, and end-effector if any>
   - simulator: <simulator + its API shape, e.g. `IsaacLab (Isaac Sim, manager-based)`>
   - objects: <the scene's manipulable + goal objects, or `none (<terrain>)` for locomotion>
   - bimanual: <true|false>
   - summary: <one sentence: what the task asks the robot to do>

   ## §1 Registration + Scene
   ...
   ## §2 Actions
   ...
   ## §3 Reset
   ...
   ## §4 Goal + Termination
   ...
   ## §5 Observation
   ...
   ## §6 Reward
   ...
   ## §7 DR
   ...
   ```

   Within each section, include subblocks:
   - **Description** — one-paragraph plain-English summary of what this section does.
   - **Decisions resolved** — concrete values chosen (e.g. `action.scale = (0.02, 0.02, 0.02)`, `episode_length_s = 9.0`).
   - **Code** — verbatim source blocks pulled from the repo.
   - **Smoke** — the §N smoke command + expected stdout (literal copy from a passing run; for §2..§7 the agents will run these when reproducing).

6. **Write** the rendered content to `<output>` (default `<repo>/harbor/create-task/<task-slug>-implementation.md`). Compute `<task-slug>` the same way `/harbor:task-create` does:
   ```bash
   slug=$(echo "<task>" | tr '[:upper:]' '[:lower:]' | tr -c '[:alnum:]' '-' | sed 's/--*/-/g; s/^-//; s/-$//')
   ```

7. **Self-verify** the spec is reproducible:
   - Every code block must compile (`py_compile` on extracted snippets).
   - Every referenced asset path must exist on disk (warn on missing — the spec is still emitted, but the reproduction will fail downstream).
   - Every `func=<...>` in §3..§7 must resolve in the repo's mdp tree (`grep -r "def <name>" <mdp_dir>`).

8. Surface a one-line note:
   ```
   probe-task: wrote <output> (sections §1..§7, <N> reward funcs, <M> obs terms)
                Reproduce via: /harbor:task-create name=<new_task_id> from=<output>
   ```

## Hard rules

- **Verbatim code, not paraphrased.** Every reward / observation / action function in §6 / §5 / §2 is pasted as-is. The spec must be self-contained enough that a downstream agent can recreate the file without re-reading the source repo.
- **Resolve asset paths.** Where the env_cfg uses `Path(__file__).resolve().parents[N] / "..."`, resolve to the actual path (relative to the simulator package root) so the downstream agent can locate equivalent assets in the destination repo.
- **No machine-specific or absolute paths.** The spec must be self-contained: every path is relative to the simulator package root or a clearly-marked `<placeholder>`. Never emit a personal home path (`/home/...`, `/Users/...`) — a reader on another machine must be able to follow the spec verbatim.
- **English-only.**
- **Read-only.** probe-task never modifies the source repo — it only reads + writes the output file.

## On failure

- Task doesn't build (`gym.make` raises) → refuse, do not emit a partial spec.
- Family undetectable → ask user (one `AskUserQuestion` listing the six family options).
- A reward function references a helper that probe-task can't locate → emit the spec with a `WARN:` annotation in §6 noting the missing helper; the user must fix this before `/harbor:task-create from=...` can succeed.

## Constraints

- **Do NOT** modify the source repo's task files.
- **Do NOT** embed binary assets (USDs, meshes) — only paths and metadata. The reproduction agent locates equivalent assets in the destination repo.
- **Do NOT** silently fall back to a partial spec — if a section can't be probed, mark it `WARN:` so the failure is visible at reproduction time.
