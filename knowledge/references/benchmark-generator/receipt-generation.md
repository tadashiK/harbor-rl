# Step 5: history.md + benchmark.md Generation

After Step 4 smoke tests complete, render **two** markdown files at the **target repo root**:

- `history.md` — one-shot record of what the skill did in this run: probe results, generated files, smoke tiers pass/fail, final report.
- `benchmark.md` — static guide to the benchmark itself: short "About" paragraph, task inventory table, action / observation / reward summary, and a numbered "How to use" walk-through limited to env sanity checks (random rollout + render). Training/evaluation walk-through lives in `<repo>/rl-integration.md`, owned by `rl-integration-generator`.

`install.md` is **owned by dependency-generator**. Do not render it here.
`rl-integration.md` is **owned by rl-integration-generator**. Do not render it here either; link to it from `benchmark.md` so users find it after they wire training.

Templates:

- `knowledge/templates/benchmark-generator/history.md.template`
- `knowledge/templates/benchmark-generator/benchmark.md.template`

Both files are **regenerated on every re-run**. Commit them to keep the build decisions in git, or `.gitignore` them if you treat them as throwaway build output.

## Why two files, not one

Split by audience — each question gets its own file:

- A reviewer checking what the skill decided reads `history.md` — probe evidence, smoke logs.
- A researcher picking a task or wiring a random rollout reads `benchmark.md` — task identifiers, action dim, reward semantics, the random+render worked example.
- A practitioner wiring training reads `rl-integration.md` (separate file, owned by `rl-integration-generator`).

**Do not merge them back.** If you catch yourself writing probe tables into `benchmark.md` or task lists into `history.md`, stop and reassign. If you catch yourself writing training commands into `benchmark.md`, stop — that belongs in `rl-integration.md`.

## When to generate

- **Always**, even if some smoke tiers failed — as long as Step 3 (env install) passed. A failed L1 / L2 is recorded in `history.md`, it does not suppress the files.
- **Skip** only when Step 3 (install) itself failed. At that point there is no env to probe; print `history.md / benchmark.md skipped: install failed` instead.
- **Overwrite** the previous copies — these are snapshots of this build, not append-only logs.

## Placeholder registry

Shared across `history.md.template` and `benchmark.md.template`. Most inherit from the agent's main placeholder registry; receipt-only additions are listed here.

| Placeholder | Source | Example |
|-------------|--------|---------|
| `{{BUILD_TIMESTAMP}}` | `date -Iseconds` at start of Step 5 | `2026-04-23T14:30:00+00:00` |
| `{{TARGET_REPO_COMMIT}}` | `git -C <repo> rev-parse --short HEAD` | `3921fed` |
| `{{GENERATED_FILES_TABLE}}` | Markdown table — every file this build created or edited, one line each | See below |
| `{{L1_RESULT}}` / `{{L2_RESULT}}` | `✅ OK` / `⚠️ FAIL: <reason>` | `✅ OK` |
| `{{SMOKE_TAIL_CAPTURES}}` | Per-tier code-fenced block with last-5 stdout lines | See below |
| `{{NVIDIA_SMI_RESULT}}` / `{{TORCH_CUDA_RESULT}}` / `{{PKG_IMPORT_RESULT}}` | `✅` or `❌ <error excerpt>` | `✅` |
| `{{FINAL_REPORT_SUMMARY}}` | 3–6 sentence summary: what works, what broke, confidence level | See below |
| `{{KNOWN_LIMITATIONS}}` | Bullet list of caveats a re-user must know about (non-obvious mount rules, python-version pins, etc.) | See below |
| `{{BENCHMARK_ABOUT}}` | 2–4 sentence plain-English summary of what the benchmark is and who uses it. Cite the upstream paper / project page when available. | `IsaacLab is a GPU-accelerated robot-learning framework built on Isaac Sim, introduced by NVIDIA. Tasks cover locomotion, manipulation, and dexterous control with massively-parallel envs ...` |
| `{{SIMULATOR_NAME}}` | Name + version of the simulator powering the benchmark | `Isaac Sim 5.1.0` |
| `{{UPSTREAM_REFERENCE}}` | Paper / project URL or citation | `Mittal et al., "Orbit", 2023 — https://isaac-orbit.github.io/` |
| `{{BENCHMARK_LICENSE}}` | License of the upstream benchmark | `Apache-2.0` |
| `{{PRIMARY_USE_CASE}}` | One phrase: what are people typically doing with this benchmark | `massively-parallel RL for locomotion and manipulation` |
| `{{TASK_COUNT}}` | Integer count of tasks documented in the table | `28` |
| `{{TASK_INVENTORY_TABLE}}` | Markdown table: task id, short description, group/difficulty. One row per task. **Do not truncate** — partial listings lose the benchmark's defining feature. | see below |
| `{{TASK_GROUPS_DESCRIPTION}}` | If the benchmark clusters tasks (e.g. by difficulty, skill family), describe the clusters in 2–5 sentences. Omit the whole block via `{{^HAS_TASK_GROUPS}}` when not applicable. | `IsaacLab tasks split into locomotion (Velocity, Position) and manipulation (Lift, Cabinet) skill groups.` |
| `{{ACTION_DIM_DISPLAY}}` | Human-readable action dim (tuple / summary for vec-env benchmarks) | `(num_envs, 12)` |
| `{{ACTION_SEMANTICS}}` | What the action vector means physically | `joint position targets, normalized [-1, 1]` |
| `{{OBS_TYPE}}` | Obs space summary | `low-dim proprio (43-D Box) + optional camera` |
| `{{CONTROL_HZ}}` | Control frequency if known, `variable` otherwise | `60 Hz (sim) / 200 Hz (env step)` |
| `{{REWARD_SIGNAL_DESCRIPTION}}` | Reward semantics — dense / sparse / weighted-sum | `dense weighted-sum: tracking_lin_vel + tracking_ang_vel + base_height + ...` |
| `{{TASK_LIST_COMMAND}}` | One-line shell or python command that prints all valid task identifiers at runtime | `python -c "import gymnasium; print([e for e in gymnasium.envs.registry.keys() if 'Isaac' in e])"` |
| `{{TASK_EXAMPLE}}` | One valid task identifier — used in the random/render example commands | `Isaac-Lift-Cube-Franka-v0` |
| `{{HAS_TASK_GROUPS}}` | Conditional flag — toggles the "Task groups" subsection | `1` or absent |

## Section templates

### `TASK_INVENTORY_TABLE`

Build a complete table — every task the benchmark exposes. No truncation, no "see upstream for the rest":

```markdown
| Task id | Description | Group |
|---------|-------------|-------|
| `Isaac-Velocity-Flat-Anymal-C-v0` | AnyMal C velocity tracking on flat terrain | locomotion |
| `Isaac-Velocity-Rough-Anymal-C-v0` | AnyMal C velocity tracking on rough terrain | locomotion |
| `Isaac-Lift-Cube-Franka-v0` | Franka picks up and lifts a cube | manipulation |
| `Isaac-Cabinet-Franka-v0` | Franka opens a cabinet drawer | manipulation |
| ... | ... | ... |
```

Discover the full task list at generation time by inspecting the env package. Typical probes:

```bash
python -c "import gymnasium; print([e for e in gymnasium.envs.registry.keys() if 'Isaac' in e])"
python <pkg>/tools/list_tasks.py
```

If the task list is datasheet-style (hundreds of entries), group them first and then show one row per group with `N tasks` in the description, then link to an appendix. For small benchmarks (< 50 tasks) show every row.

### `GENERATED_FILES_TABLE`

```markdown
| File | Role | Status |
|------|------|--------|
| `scripts/run_random.py` | Random-action rollout — env + reward sanity check | created |
| `scripts/render_random.py` | Random-action rollout that writes an MP4 — render pipeline check | created |
| `harbor/benchmark-generator/benchmark-spec.json` | RL suite spec for rl-integration-generator + the RL training/tuning commands | created |
| `harbor/benchmark-generator/history.md` | This file | created |
| `harbor/benchmark-generator/benchmark.md` | Static benchmark guide | created |
```

Prefer `created` / `edited` / `unchanged` in the status column; it makes re-runs easy to diff.

### `SMOKE_TAIL_CAPTURES`

For each tier, keep the last 5 stdout lines inside a fenced block. Order the blocks L1 → L2:

```markdown
**L1 (random)**
```text
step 7: reward=-0.0184
step 8: reward=-0.0173
step 9: reward=-0.0161
L1 OK: 10 steps, all rewards finite
```

**L2 (render)**
```text
L2 OK: wrote 30 frames to /tmp/harbor-smoke-4711.mp4 (412.7 KB)
```
```

### `FINAL_REPORT_SUMMARY`

3–6 sentences, plain prose, no markdown tables. Cover:

1. Whether L1 and L2 both passed; if either failed, which and why.
2. Whether the env is end-to-end runnable for the random/render flow.
3. One sentence pointer: training scaffolding lives in the `rl-integration-generator` subagent (dispatch directly).

Example:

> L1 + L2 pass. `scripts/run_random.py --task PickCube-v1 --n-steps 10` and `scripts/render_random.py ... --output /tmp/harbor-smoke-4711.mp4` both complete in <5 s. Residual risk: physx GPU binary is downloaded at first `enable_gpu()` call. To wire training, dispatch the `rl-integration-generator` subagent.

### `KNOWN_LIMITATIONS`

Bullet list. Include only things a re-user must know about and would not discover from reading `setup_uv.sh` / `install.md`. Typical entries: Python-version pins driven by upstream constraints, manual asset downloads, benchmarks whose native train.py is used instead of a generated script, `{{VIDEO_FRAME_EXTRACT}}` returning `None` for some sub-task.

## Rendering pipeline

Step 5 is executed by the agent, not a separate generator script. Steps:

1. Capture `date -Iseconds` at Step 5 start.
2. Collect the cumulative outputs from Steps 1–4 (smoke tiers' last-5 lines, capability probe stdout).
3. Substitute each `{{PLACEHOLDER}}` in both templates. For multi-line placeholders (`{{GENERATED_FILES_TABLE}}`, `{{SMOKE_TAIL_CAPTURES}}`), build the markdown string in Python and splice it rather than using `sed`.
4. Write the result to `<target_repo>/harbor/benchmark-generator/history.md` and `<target_repo>/harbor/benchmark-generator/benchmark.md`.
5. Echo both paths to stdout:
   `history.md generated (N KB)`
   `benchmark.md generated (N KB)`

## Failure handling

| Failing step | Effect on Step 5 |
|--------------|------------------|
| Step 3 (env install) | Skip Step 5 entirely. Print `history.md / benchmark.md skipped: install failed`. |
| Step 4 L1 | `L1_RESULT = "⚠️ FAIL: <last stdout line>"`; `L2`/`L3_viz` become `⏭️ skipped`. Still render both files — the failure itself is useful information. |
| Step 4 L2 | `L2_RESULT = "⚠️ FAIL: ..."`. Still render — most likely a `{{VIDEO_FRAME_EXTRACT}}` mismatch the user can fix; write the troubleshooting note in `benchmark.md`. |

## Rationalizations this step will face (guard against them)

These are the shortcuts future agents will attempt. Each has a counter:

| Rationalization | Reality |
|---|---|
| "I'll put everything into one file, it's simpler." | Merging was tried and failed — DOCUMENTATION.md was the kitchen sink. The split audience is the whole point. Do not merge back. |
| "I'll add a 'Train a policy' section to benchmark.md so the user has one stop." | WRONG. Training/evaluation lives in `rl-integration.md`, owned by `rl-integration-generator`. Link to it from benchmark.md; do not duplicate. |
| "`history.md` is build output, skipping it is fine." | `history.md` is the only place the probe results and smoke decisions are recorded. Skipping it makes the build un-auditable. |
| "The user ran the skill interactively so they know what happened — no need to write it down." | "They know" ≠ "future-them knows six months later when the repo has moved on". Write it down every time. |
| "Everything is in English already, I'll switch to Chinese for this user." | Output markdown is **English-only by contract**. User preferences about chat language do not apply to generated artefacts. |
| "I'll truncate the task inventory table — 80 tasks is too long." | The task inventory is the defining feature of a benchmark. Full list or the file is useless. If the list is genuinely >50, group by skill family and paginate via subsections, but never omit entries silently. |
| "The script already has a docstring, adding an `Example` block is redundant." | The `Example` block is **required** on every generated Python script in `scripts/`. A docstring that says what the file does without showing how to invoke it makes `python <file>.py --help` the de-facto documentation. |
| "Headless smoke can't render cameras, so I'll just drop `render_random.py` from the template." | Wrong amputation. `render_random.py` runs offscreen via `render_mode='rgb_array'` / `has_offscreen_renderer=True` — no X11 needed. Keep both random + render scripts. |

## Red flags — STOP before writing

- About to render `<repo>/install.md` → WRONG owner. dependency-generator owns it. Stop.
- About to render `<repo>/rl-integration.md` → WRONG owner. `rl-integration-generator` owns it. Stop.
- About to write a `Train a policy` / `Evaluate a checkpoint` / `Override hyperparameters` section into `benchmark.md` → WRONG file. Those belong in `rl-integration.md`.
- About to skip `history.md` because "smoke tests passed, we don't need a log" → WRONG. `history.md` records the probe + smoke decisions regardless of outcome.
- About to truncate `TASK_INVENTORY_TABLE` to save space → WRONG. Full list or group-and-paginate; never silent omission.
- About to generate any Python script without an `Example` block in the docstring → WRONG. Add one before writing the file.
- About to write Chinese / non-English markdown into either of the two files → WRONG. Output is English-only.

## Integration with git

The skill does not auto-add to `.gitignore`. The user chooses:

- **Commit both** (recommended): build decisions and benchmark guides stay versioned alongside the source.
- **Gitignore both**: treat them as build output. They regenerate on every re-run.

Either way, the skill ships both files on every successful run.
