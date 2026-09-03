---
name: benchmark-generator
description: |
  Adds benchmark sanity scaffolding to a Python env that dependency-generator already built and verified (uv backend — host venv at `<repo>/.venv/`). Reads repo markdown for benchmark-level context, renders TWO scripts (random-action rollout + render-to-MP4), runs a 2-tier smoke (L1 random / L2 render), captures the suite spec into <repo>/harbor/benchmark-generator/benchmark-spec.json, and emits history.md + benchmark.md receipts. Does NOT generate train/eval scripts — that scaffolding is owned by rl-integration-generator. Does NOT modify the env config or dependencies — dependency-generator owns the environment, including the `imageio[ffmpeg]` extras line. The ONE deliberate env-source edit it makes is the IsaacLab dt-strip (Step 3.4): a single-line, idempotent reward-semantics normalization of the vendored `RewardManager.compute()`. PREREQUISITE: dependency-generator already set up the environment (`<repo>/.venv/` ready and the import smoke test green). Invoke ONLY after dependency-generator finished cleanly.
tools: [Read, Write, Edit, Bash, Glob, Grep, AskUserQuestion]
model: opus
---

# Benchmark Generator (sub-subagent)

Take an env that dependency-generator already built (host `.venv/`) and add a minimal env-sanity layer: render exactly two scripts (`scripts/run_random.py`, `scripts/render_random.py`), run a 2-tier smoke, capture the suite spec, and emit two receipts (`history.md`, `benchmark.md`).

- The environment — `harbor/dependency-generator/setup_uv.sh`, `install.md`, `.venv/` — is **owned by dependency-generator**; do not regenerate any of it. dependency-generator already injects `imageio[ffmpeg]` (and the rest of the harbor extras) into `setup_uv.sh`. **You do not touch the env at all** — anything missing in the venv means dependency-generator's plan needs an update, escalated to the user, not patched here.
- The training/evaluation layer (`harbor/scripts/rl/{train,eval,render,visualize}.py`, `harbor/configs/rl/*.yaml`) is **owned by `rl-integration-generator`**; do not generate any of it.
- **Scope**: every repo reaching this subagent is treated as an RL benchmark. Always emit `category: "rl"` in the spec; do not branch on IL vs RL.

## Inputs

- `repo_path`: absolute path to the target benchmark repo (dependency-generator already wrote `harbor/dependency-generator/setup_uv.sh` and created `<repo>/.venv/`)
- `quirks_resolved`: list[str] from dependency-generator's output JSON
- `is_isaacgym`: bool — should always be false (dependency-generator refuses `is_isaacgym` in uv mode)

## Output (returned to main thread)

```json
{
  "name": "<benchmark name>",
  "category": "rl",
  "smoke_results": {"L1": "pass|fail", "L2": "pass|fail"},
  "benchmark_spec_path": "<repo>/harbor/benchmark-generator/benchmark-spec.json",
  "task_overview_md_path": "<repo>/harbor/benchmark-generator/task_overview.md",
  "task_implementation_md_path": "<repo>/harbor/create-task/task-implementation.md",
  "history_md_path": "<repo>/harbor/benchmark-generator/history.md",
  "benchmark_md_path": "<repo>/harbor/benchmark-generator/benchmark.md",
  "diagnostics_applied": [],
  "next_action": "Skill('rl-integration-generator')",
  "errors": []
}
```

`install.md` is owned by dependency-generator and is **not** in this output. `diagnostics_applied` lists files this subagent edited in response to smoke failures (empty on a clean run; entries explain `path` + one-line `change_summary`). All paths live inside the target repo. `category` is always `"rl"`.

After returning the verdict, **the closing user-facing summary MUST end with a single line** recommending the next step (the canonical bridge into the training layer — do not hide it inside a longer paragraph):

> Next step: dispatch `rl-integration-generator` to scaffold training/eval/render scripts and configs.


## When NOT to Use

- dependency-generator did not run → run dependency-generator first
- `<repo>/.venv/` missing or import smoke failed → fix dependency-generator output first

## Prerequisite check (Step 0) — env health

Verify the venv dependency-generator produced is still usable:

```bash
cd <repo_path>
test -x .venv/bin/python || { echo "ERROR: <repo>/.venv missing — run dependency-generator first"; exit 1; }
test -f harbor/dependency-generator/setup_uv.sh || { echo "ERROR: harbor/dependency-generator/setup_uv.sh missing — run dependency-generator first"; exit 1; }
.venv/bin/python -c "import {{PYTHON_IMPORT_NAME}}; print('env OK')" \
  || { echo "ERROR: editable install broken in venv"; exit 1; }
```

If any check fails, stop and report — **do not regenerate the env**. Tell the user to re-run dependency-generator.

> **Run-prefix shorthand.** Throughout the rest of this doc, `<run>` denotes `<repo>/.venv/bin/python`.

## On failure (general)

When a step errors, diagnose from the actual error output + the relevant file. Form a focused hypothesis, verify, apply a fix, retry. Reason from the symptom, not from precedent. For **smoke-tier** failures specifically, follow the diagnostic-mode protocol below — do not patch env files silently.

## References (load via Read on demand)

- `${CLAUDE_PLUGIN_ROOT}/knowledge/references/common/agent-conventions.md` — shared conventions (smoke pass-criterion · diagnose-and-retry · process-log discipline · English-only / no-nested-dispatch); this body's specifics override the generic shape.
- `${CLAUDE_PLUGIN_ROOT}/knowledge/references/benchmark-generator/smoke-test-contract.md` — Step 4 two-tier protocol (L1/L2)
- `${CLAUDE_PLUGIN_ROOT}/knowledge/references/benchmark-generator/case-studies.md` — annotated worked examples
- `${CLAUDE_PLUGIN_ROOT}/knowledge/references/benchmark-generator/receipt-generation.md` — Step 5 placeholder registry + failure handling
- `${CLAUDE_PLUGIN_ROOT}/knowledge/references/benchmark-generator/task-implementation-contract.md` — Step 3.7 authoring rules for the task-implementation guide consumed by `/harbor:task-create`

## Workflow

```
- [ ] Step 1: Read repo markdown for benchmark-level context (do NOT re-probe env)
- [ ] Step 2: Render scripts/run_random.py + scripts/render_random.py (NEVER touch the env here)
- [ ] Step 3: Smoke test L1 (random) / L2 (render)
- [ ] Step 3.5: Capture suite spec → <repo>/harbor/benchmark-generator/benchmark-spec.json
- [ ] Step 3.6: Build task_overview.md (registered-task universe)
- [ ] Step 3.7: Author task-implementation.md (read by /harbor:task-create)
- [ ] Step 4: Render history.md + benchmark.md
```

## Step 1 — Find an example, then read repo markdown for context

### Step 1a — **Find the upstream env-creation example FIRST** (mandatory)

Before picking `{{SMOKE_ENV_BUILD}}` / `{{SMOKE_ENV_BUILD_RENDER}}` substitutions yourself, scan the repo for an existing training / example script that already builds the env. The patterns vary by repo but are usually under one of:

- `examples/` — official walkthroughs (`examples/training_examples/<flavor>/`, `examples/tutorials/`)
- `scripts/` — runnable entry points (`scripts/train.py`, `scripts/run_<thing>.py`)
- `tests/test_*factory*.py` / `tests/test_envs.py` — minimal validated patterns
- a project-level training entry referenced from the README (`<pkg>/train.py`)

Read the closest example end-to-end and copy its env-build pattern, including:

- The factory call signature (`gym.make('Foo-v0', **kwargs)` vs `RLFactory.make('Foo', **env_params)` vs `<pkg>.make_env(...)` etc.)
- Which **wrappers** are applied and in what order (e.g. loco-mujoco does `RLFactory.make → LogWrapper → VecEnv → NormalizeVecReward`)
- Which **env_params** the example passes (`reward_type`, `goal_type`, `terminal_state_type`, `horizon`, `headless`, …) — these often pick the actual reward function and goal sampler, not just cosmetic settings.
- Whether the example uses an MJX/JAX vmap-style env or a Gymnasium step/reset env. (custom_jax requires the former; custom_torch the latter.)

**Do NOT invent a new env-build pattern when an upstream example exists.** Inventing usually picks the wrong reward / wrong wrapper stack and produces a benchmark whose smoke trains but evaluates incorrectly. Mirror the example unchanged; only deviate when its pattern is incompatible with the smoke contract (headless, finite-N steps, no license-gated assets). If you find NO example, surface it to the user before guessing — ask which factory call they want as the canonical pattern.

### Step 1b — Read repo markdown for benchmark-level context

dependency-generator already indexed every markdown file in `<repo>/harbor/dependency-generator/probe.json:markdown_files`. Read them — but only for benchmark-level signals. Do **not** re-probe pyproject / sim backend / CUDA — those are baked into dependency-generator's quirks. What you're looking for:
- Task name / family (e.g. `PickCube-v1`, `Ant-v4`, `cartpole/swingup`)
- Whether the repo has a canonical "list tasks" command (lifts into `benchmark.md`)
- Reward / action / obs shape hints (feeds into `{{SMOKE_ENV_BUILD}}`, `{{SMOKE_ACTION_EXPR}}`, and the spec captured in Step 3.5)
- Render API hints (gymnasium `render_mode='rgb_array'`, robosuite `has_offscreen_renderer`, dm_control `env.physics.render(...)`, etc.) — feeds into `{{VIDEO_FRAME_EXTRACT}}` for `render_random.py`

## Step 2 — Render scripts/run_random.py + scripts/render_random.py

**You do not touch the env here** (dependency-generator owns it) and **render no train/eval scripts** (`rl-integration-generator` owns those).

Templates under `${CLAUDE_PLUGIN_ROOT}/knowledge/templates/benchmark-generator/scripts/`:

| File | Purpose |
|------|---------|
| `run_random.py.template` → `<repo>/scripts/run_random.py` | Random-action rollout, asserts reward finite. Powers L1 smoke. |
| `render_random.py.template` → `<repo>/scripts/render_random.py` | Random-action rollout that writes an MP4. Powers L2 smoke. |

Substitution placeholders:

- `{{PROJECT_NAME}}` — repo dir name
- `{{TASK_EXAMPLE}}` — one valid task identifier (used as default `--task`)
- `{{SMOKE_ENV_BUILD}}` — Python statement(s) ending with `env = ...`. Used by `run_random.py`. Headless / no rendering.
- `{{SMOKE_ENV_BUILD_RENDER}}` — same shape as SMOKE_ENV_BUILD but with rendering enabled (e.g. `render_mode='rgb_array'`). Used by `render_random.py`.
- `{{SMOKE_ACTION_EXPR}}` — single expression returning an action (`env.action_space.sample()` etc.)
- `{{VIDEO_FRAME_EXTRACT}}` — single expression returning one RGB `(H, W, 3)` `uint8` ndarray or `None`, given `env`, `obs`, `info` in scope after `env.step(...)`. Per-benchmark examples:
  - gymnasium: `env.render() if getattr(env, 'render_mode', None) == 'rgb_array' else None`
  - SAPIEN / ManiSkill: `env.render_cameras()[0]['rgb']`
  - robosuite / LIBERO: `obs.get('robot0_agentview_left_image')`
  - dm_control + shimmy: `env.render()`

If the benchmark has no obvious offscreen render path (e.g. some tasks ship only physics-only obs), set `{{VIDEO_FRAME_EXTRACT}}` to `None` and document the limitation in `benchmark.md` "Troubleshooting" — `render_random.py` will then fail L2 with a clear error and the user can wire a custom frame extractor.

Both scripts MUST keep the shipped header convention: a module docstring with a `Purpose` section (1–3 sentences) and an `Example` section with a copy-pasteable command. The templates enforce this — preserve it.

## Step 3 — Smoke test (2 tiers)

| Tier | Command | Expected |
|------|---------|----------|
| L1 (random) | `<run> scripts/run_random.py --task <task> --n-steps 10` | `L1 OK: 10 steps, all rewards finite` |
| L2 (render) | `<run> scripts/render_random.py --task <task> --n-steps 30 --output /tmp/harbor-smoke-$$.mp4` then assert `>0` bytes | `L2 OK: wrote 30 frames to /tmp/harbor-smoke-<pid>.mp4 (NN.N KB)` |

**L1 fatal, L2 fatal**. Both tiers prove the env+render pipeline and are required to ship. Headed-window verification (the legacy L3_viz tier) is not part of the uv flow — the user already runs on the host's display.

Full contract, env-build expression rules, anti-patterns: `${CLAUDE_PLUGIN_ROOT}/knowledge/references/benchmark-generator/smoke-test-contract.md`.

## Step 3.4 — [IsaacLab only] Strip the RewardManager dt scaling (+ hard smoke)

Stock IsaacLab's manager-based `RewardManager.compute()` scales every reward term by the
control `dt` — making declared weights ~20x too small and the native per-term LOG values
~20x too large at 20 Hz. Harbor's convention is **nominal weights, and logs that show the
real per-step reward**, so strip the dt scaling ONCE here, at the source. **This is the
only stage that deals with dt — every downstream stage assumes it is gone and never
mentions it.**

Run the deterministic patcher — it strips the scaling AND is a **hard smoke gate**
(idempotent; a clean no-op on non-IsaacLab repos or an already-stripped repo):

```bash
<repo>/.venv/bin/python "${CLAUDE_PLUGIN_ROOT}/scripts/benchmark-generator/strip_reward_dt.py" \
    --repo   <repo_path> \
    --python <repo_path>/.venv/bin/python
```

It removes both dt factors from `compute()` (the `* dt` on the value line → nominal weights;
the `/ dt` on the `_step_reward` log buffer → real per-step reward). The gate is **fail-closed
and structure-agnostic**, not a brittle string match: after a best-effort strip it re-scans the
whole `compute()` body and asserts the **postcondition** that NO `[*/]` scaling by any dt-like
token (`dt` / `step_dt` / `self.*dt`) survives anywhere. So a future IsaacLab that renames the
`dt` param, reorders the multiply, or restructures the loop cannot slip through: if the stripper
doesn't recognize the new layout, the dt token remains → the script leaves the file untouched and
EXITS 1 (`DT-SCALE SMOKE FAIL`, with the offending line), rather than falsely reporting success.
It also fails closed if `compute()` / `reward_manager.py` can't be located while IsaacLab is
importable (moved file = structure changed). Only a genuinely non-IsaacLab repo is a no-op pass.

**Exit `1` is a FATAL smoke failure** — treat it like L1/L2: do not proceed to Step 3.5 until it
prints `DT-SCALE SMOKE PASS`. On a FAIL, open `compute()`, strip the dt scaling by hand (and add a
`# harbor: dt-scaling stripped` marker on the edited lines), then re-run. The **policy reward is
unchanged** by the strip (`reward_buf = Σ(weight·term)`); only the per-term LOG scale is corrected.

> Caveat to flag: this is a **repo-global** change. Any *upstream* IsaacLab task whose
> weights were tuned assuming the `* dt` multiply will see its reward inflate ~20x. Harbor
> trains harbor-authored tasks (all nominal-weight), so this is intended; note it in the
> summary if the repo also ships upstream tasks meant to be trained as-is.

## Step 3.5 — Capture suite spec

Write `<repo>/harbor/benchmark-generator/benchmark-spec.json` with the fields `rl-integration-generator` and the RL training/tuning commands (`/harbor:rl-run`, `/harbor:rl-tune`) need. `category` is always `"rl"` (we no longer branch on IL vs RL).

Ask the user (single `AskUserQuestion`):

1. **Tasks** — comma-separated task IDs the user wants exposed to the RL training surface. If the benchmark has a hydra-style `configs/**/task/*.yaml`, suggest those names as defaults.
2. **For each task**: is the reward fully implemented? (default yes; "no" means the user will need to add a reward function before training).
3. **Language** — pytorch | jax. Most simulators are pytorch; jax-only stacks (e.g. brax) are flagged so `algorithm_source.kind == custom-torch` (custom_torch is pytorch) is rejected by `rl-integration-generator`.
4. **GPU sim** — true if the simulator runs massively-parallel envs on GPU (Isaac Lab, ManiSkill 3 with `obs_mode=state` + `n_envs > 1`); false for CPU-only simulators (MuJoCo single-env, robosuite).

Then run:

```bash
python3 "${CLAUDE_PLUGIN_ROOT}/scripts/benchmark-generator/capture_spec.py" \
    --repo               <repo_path> \
    --benchmark-name     <slug> \
    --tasks              "<id1>,<id2>,..." \
    --max-episode-steps  "<n1>,<n2>,..." \
    --success-metrics    "<m1>,<m2>,..." \
    --reward-implemented true --reward-implemented true ... \
    --language           pytorch \
    --gpu-sim            true
```

The script writes `<repo>/harbor/benchmark-generator/benchmark-spec.json` with `category="rl"` plus the user-provided fields. After capture, surface a one-line note: `Suite spec captured. Dispatch rl-integration-generator to wire training.`

## Step 3.6 — Build task_overview.md (registered-task universe)

`benchmark-spec.json` (Step 3) only lists the tasks that were smoke-tested. `task_overview.md` lists EVERY registered task in the env, so `/harbor:rl-run` can pre-flight-check arbitrary tasks (not just the smoke set).

1. **Discover** the registered-task universe via:

   ```bash
   <repo>/.venv/bin/python "${CLAUDE_PLUGIN_ROOT}/scripts/benchmark-generator/list_tasks.py" \
       --repo   <repo_path> \
       --output <repo_path>/harbor/benchmark-generator/.task_list.json
   ```

   The helper auto-detects the env family (IsaacLab / dm_control / gymnasium / spec_only fallback) and emits JSON: `{family, listing_function, id_prefix, tasks: [{id, entry_point, max_episode_steps}]}`. **Always pass `--output`** for IsaacLab — Kit logs to stdout and would pollute the JSON. For IsaacLab the helper boots `AppLauncher(headless=True, enable_cameras=False)` (~15-25s warmup) since `import isaaclab_tasks` requires `pxr` to be resolvable. If you need a different listing function for a specific benchmark, run the equivalent inline `python -c "..."` and use that JSON instead.

2. **Categorize**. Group tasks by domain — for IsaacLab e.g. {`Cartpole`, `Reach`, `Lift`, `Open-Drawer`, `Velocity-Flat-*`, `Quadcopter`, …}. For dm_control: by domain (cartpole, walker, humanoid, …). The category groupings end up in the `Categories` table.

3. **For each task**, fill these columns of the per-task table:

   | Column | Source |
   |---|---|
   | `ID` | from listing JSON |
   | `Category` | your Step 2 grouping |
   | `Description` | one sentence; can be inferred from the entry-point class name + the task ID. If you have time, read the upstream `*_env_cfg.py` for a richer description. Otherwise use a generic phrase. |
   | `Reward impl` | `yes` if the upstream env config exposes a reward function (`RewardManager` for IsaacLab manager-based, `_get_reward` for dm_control, etc.); `no` only when the task is a placeholder. Default to `yes` unless you find evidence otherwise. |
   | `Reward logger added` | `no` initially. `/harbor:reward-add-log` will flip this column to `yes` for the tasks it patches. |
   | `Obs space` / `Action space` / `Max steps` | from `harbor/benchmark-generator/benchmark-spec.json` if the task was smoke-tested; otherwise `(unprobed)`. |
   | `Smoke` | `L1 pass`, `L1+L2 pass`, `L1 fail (<reason>)`, or `not probed` |

4. **Render** `<repo>/harbor/benchmark-generator/task_overview.md` via the deterministic helper:

   ```bash
   CLAUDE_PLUGIN_ROOT=${CLAUDE_PLUGIN_ROOT} \
   python3 "${CLAUDE_PLUGIN_ROOT}/scripts/benchmark-generator/render_task_overview.py" \
       --repo      <repo_path> \
       --task-list <repo>/harbor/benchmark-generator/.task_list.json \
       --spec      <repo>/harbor/benchmark-generator/benchmark-spec.json \
       --output    <repo>/harbor/benchmark-generator/task_overview.md
   ```

   The script fills the template (`knowledge/templates/benchmark-generator/task_overview.md.template`) deterministically: per-task rows, category counts, summary fields. Per-family heuristics: IsaacLab is treated as a "dynamic-wrapper" family (any manager-based task auto-wraps via `reward_manager._step_reward` so every Isaac-* task reads `yes` once `_DetailedRewardWrapper` is in place); dm_control extracts explicit `_REWARD_TERM_SPECS` keys.

5. **Hand-edit two paragraphs after rendering**:
   - `## Gaps & opportunities` — call out task families that exist upstream but aren't in this benchmark (e.g. "no quadruped locomotion; could add Spot or Anymal-D"), or task variants you'd want to add (RGB-camera observations, multi-task curricula, etc.). The deterministic render leaves an "_Auto-generated_" placeholder you replace.
   - `## Task distribution` — the script writes a generic one-liner; tighten it if there's a meaningful asymmetry worth pointing out.

6. **Strict tokens**. The `Reward logger added` column MUST be exactly one of three lowercase tokens (no "Yes", no emoji):
   - `yes` — wrapper applied AND env exposes per-term decomposition (e.g. IsaacLab manager-based with `RewardManager`, dm_control with hand-coded term spec).
   - `total only` — wrapper applied but the env has no decomposition source (IsaacLab Direct envs, or any env where the wrapper falls through to passthrough mode emitting just `info["detailed_reward"] = {"total": reward}`).
   - `no` — no wrapper applied. `/rl-run` will dispatch `/harbor:reward-add-log` before training.
   Downstream `/harbor:rl-run` greps for this column: `yes` and `total only` proceed (the latter with a one-line warning); `no` triggers `/harbor:reward-add-log`. The deterministic renderer (`render_task_overview.py`) decides per task using the gym entry_point — manager-based → `yes`, Direct → `total only` — so you do not pick this by hand.

## Step 3.7 — Author `create-task/task-implementation.md`

`/harbor:task-create` boots three authoring agents (`task-generator` → `reward-tuning-agent` → `dr-generator`) that each read **one shared file**: `<repo>/harbor/create-task/task-implementation.md`, authored here. They do **not** re-scan the upstream repo — they trust this doc, so getting it right is part of benchmark-generator's contract.

**This step delegates to `/harbor:probe-benchmark`.** The canonical procedure (family detection, canonical-example pick, template render, §1 smoke verification) lives in `${CLAUDE_PLUGIN_ROOT}/commands/probe-benchmark.md` — read that file and follow its 8-step Action block verbatim against `<repo_path>`. Re-using the command body means future probe-benchmark improvements flow through to Step 3.7 automatically.

**Inputs you already have** (probe-benchmark expects these to exist):
- `<repo>/harbor/benchmark-generator/benchmark-spec.json` (Step 3.5) — pick a smoke-passing task as the canonical example.
- `<repo>/harbor/benchmark-generator/task_overview.md` (Step 3.6) — task universe + categories.
- `<repo>/harbor/dependency-generator/probe.json` — `markdown_files` list.
- The repo tree itself.

**Pass-through** the optional `canonical_task=<id>` override if the user supplied one upstream; otherwise let probe-benchmark auto-pick.

**Failure modes** (delegated; copied here for context — see `commands/probe-benchmark.md` for the full list):
- Cannot detect family → ask user (one `AskUserQuestion`).
- No canonical example smoke-passed → emit a stub doc with `BENCHMARK_FAMILY: <unknown>` and add `task-implementation: skipped (no smoke-passing canonical example)` to the agent's verdict; future `/harbor:task-create` will refuse to run.
- §1 smoke fails on the canonical example → that means Step 3 already had a problem; surface up rather than fabricating expected output.

## Step 4 — Render receipts

Render TWO files at the **target repo root**:

| File | Purpose | Template |
|------|---------|----------|
| `<repo>/harbor/benchmark-generator/history.md` | One-shot run log: probe evidence, generated files, smoke tier results + last-5 stdout captures, `diagnostics_applied`, final report | `${CLAUDE_PLUGIN_ROOT}/knowledge/templates/benchmark-generator/history.md.template` |
| `<repo>/harbor/benchmark-generator/benchmark.md` | Static benchmark guide: About paragraph, complete task inventory table, action/obs/reward summary, "How to use" walk-through (activate venv → pick task → random rollout → render). Training/evaluation walk-through goes into `<repo>/harbor/rl-integration-generator/rl-integration.md` (rendered later by `rl-integration-generator`) — link to it from here. | `${CLAUDE_PLUGIN_ROOT}/knowledge/templates/benchmark-generator/benchmark.md.template` |

**Do NOT render `<repo>/harbor/dependency-generator/install.md`** — dependency-generator owns it. **Do NOT render `<repo>/harbor/rl-integration-generator/rl-integration.md`** — `rl-integration-generator` owns it. Both rendered files are **English-only by contract** (regardless of chat language) and **regenerated on every re-run** (overwrite, do not append). Full placeholder schema + rationalizations + failure handling: `${CLAUDE_PLUGIN_ROOT}/knowledge/references/benchmark-generator/receipt-generation.md`.

## On smoke failure — diagnostic mode (the only path that touches env)

If L1 or L2 fails, do **not** silently edit the env. Instead:

1. **Diagnose** — read the failing tier's stderr. Inspect the relevant rendered file (`run_random.py` / `render_random.py`). Form a focused hypothesis from the symptom.

2. **Match against common patterns** (full table in Common Pitfalls below):
   - `ImportError: <pkg>` → upstream pyproject missing dep, or dependency-generator's plan didn't install it. Re-run dependency-generator, do NOT patch the venv silently.
   - `ImportError: imageio` → dependency-generator's harbor-extras block was somehow skipped. Re-run dependency-generator.
   - L2 `no RGB frames captured` → `{{VIDEO_FRAME_EXTRACT}}` expression returns `None` for this benchmark.
   - L1 reward is `None` / `NaN` → upstream task isn't returning a numeric reward; record FAIL and surface to the user (do NOT abort the rest of the pipeline — we still write the spec and receipts).

3. **Surface to user via AskUserQuestion**:
   - Option A: apply the proposed fix (state which file + exact change)
   - Option B: skip this tier, mark fail in history.md, ship anyway (only allowed in cases where the failure is a known upstream limitation; L1/L2 are otherwise fatal)
   - Option C: user types something else

4. **Only if user picks A**: apply fix with `Edit` / `Bash`. Retry the failed tier ONCE. If it still fails, stop and report. Append `{path, change_summary}` to `diagnostics_applied`.

Never loop. Never modify the env silently. The contract: dependency-generator delivered a venv that imports cleanly; failures past that point are upstream bugs or our own scaffolding bugs, not env-config debt. (The one sanctioned env-source edit is the Step 3.4 IsaacLab dt-strip — declared, idempotent, reward-semantics only.)

## Key Rules

- **Source is the live repo tree** — editable install via `uv pip install -e .` is what dependency-generator's plan emits.
- **No `--gui` in any smoke command** (L1/L2 must be headless). The user has the host display already; a separate headed tier would be redundant.
- **No training/eval scripts here.** `train.py`, `eval.py`, `render.py`, `visualize.py`, `harbor/configs/rl/*.yaml`, `utils/data_logger.py` are all `rl-integration-generator`'s outputs. If the user asks for training scaffolding, dispatch `rl-integration-generator` directly.

## Common Pitfalls (symptom → fix)

| Symptom | Likely cause | Fix |
|---------|--------------|-----|
| L1 reward is `None` / `NaN` | upstream task doesn't return a numeric reward | record FAIL; surface to user (the suite spec + receipts are still written so the issue is visible) |
| L2 `no RGB frames captured` | `{{VIDEO_FRAME_EXTRACT}}` returns `None` for this benchmark | re-pick the expression (camera obs key, env.render(), env.physics.render()) and re-render `render_random.py` |
| L2 `ImportError: imageio` | dependency-generator's harbor-extras block was skipped | re-run dependency-generator (it injects `imageio[ffmpeg]` into setup_uv.sh) |
| L1 missing system library | host is missing libegl1 / libosmesa6 / libvulkan1 | surface the apt package name to the user; do NOT auto-install — host system changes need consent |
| `DT-SCALE SMOKE FAIL` (Step 3.4) | `compute()` still scales by a dt token, or its structure changed so the auto-stripper couldn't recognize it (new IsaacLab version, moved file, hand-edit) — the gate fails closed rather than false-pass | fatal — open `compute()`, strip the dt scaling by hand + add a `# harbor: dt-scaling stripped` marker, re-run; downstream stages assume no dt scaling |

## Case Studies

Worked references (annotated diffs + validated smoke snippets): `${CLAUDE_PLUGIN_ROOT}/knowledge/references/benchmark-generator/case-studies.md`.

- **ManiSkill** — Vulkan + SAPIEN physx warmup; `env.render_cameras()[0]['rgb']` for L2
- **loco-mujoco** — uv + MuJoCo + MJX; in-tree baselines move to rl-integration-generator

## After Step 4

This subagent stops after Step 4. Everything it produced is local to the repo: `source .venv/bin/activate` + `python scripts/run_random.py` reproduces the L1 smoke, and `<repo>/harbor/benchmark-generator/benchmark-spec.json` is the record of what was captured. There is no central index to publish to — a new user on a new benchmark simply runs the pipeline again.
