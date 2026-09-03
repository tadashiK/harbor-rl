---
description: Run the harbor plugin test suite. L1 (contract) + L2 (unit) are deterministic pytest layers run by the main thread; L3 is an end-to-end agentic pipeline run in a subagent that drives the task-create chain module-by-module on an isolated, clean benchmark worktree. Default runs L1→L2→L3 in sequence (stop at first failure); `layers=` runs a subset. L3 is resumable Docker-layer style — each module is a fingerprinted stage; on re-run only the changed/failed stage and everything downstream re-runs. Maintains an append-only test history; on any failure it stops, reports the error + a suggested fix, and leaves state so the next run resumes from the unimpacted part. Use when the user types /harbor:test [layers=...] [repo=<path>] [task=<id>] or asks "run the tests", "test the plugin", "run the e2e pipeline".
argument-hint: "[layers=1,2,3] [repo=<path>] [task=<id>] [resume=true|false] [fresh=false]"
---

# /harbor:test — Plugin Test Runner

Three layers (see `tests/README.md`):
- **L1 contract** + **L2 unit** — deterministic `pytest`, no GPU/Claude. The main thread runs these directly.
- **L3 e2e pipeline** — runs the full task-create → train → reset chain on one task, module-by-module, on an isolated clean benchmark worktree. Heavy + GPU-gated → runs in a subagent.

## Arguments

| Arg | Default | Effect |
|---|---|---|
| `layers` | `1,2,3` | Subset to run, in order. e.g. `layers=1`, `layers=1,2`, `layers=3`. |
| `repo` | — | Benchmark repo (git working tree). **Required for L3.** |
| `task` | `<benchmark>-Stack-Two-Cube` | The L3 fixture is **benchmark-agnostic: a robot arm stacking two cubes**, authored by `task-create` for whatever manipulation benchmark `repo` is. `task` is just the gym id to register it under — name it per the benchmark's convention (e.g. `Isaac-Stack-Two-Cube-Franka-v0`); the fixture *design* is fixed (see Fixture). |
| `resume` | `true` | Reuse the stage cache + existing worktree; re-run only changed/failed stages onward. |
| `fresh` | `false` | Ignore the cache, recreate the worktree, run every stage. |

## Layers 1 + 2 — pytest (main thread)

```bash
pip install -r tests/requirements-test.txt >/dev/null 2>&1
pytest tests/contract -q      # if 1 in layers
pytest tests/unit -q          # if 2 in layers
```
Report the pass/fail summary. If a layer fails, print the failing test ids + stop (don't proceed to the next layer). L1/L2 need no `repo`/`task`.

## Layer 3 — e2e pipeline (subagent)

Dispatch ONE `Agent(subagent_type="general-purpose")` with this section + the resolved args. It owns setup → resumable run → teardown and returns only the final verdict + (on failure) the failing stage, error, and suggested fix.

**Fixed, benchmark-agnostic fixture:** L3 always builds the **same task — a robot arm stacking one cube on top of another (two cubes)** — via `task-create` **CREATE mode** from a fixed description: *"A robot arm picks up one cube and stacks it on top of another resting on the table; success when the top cube sits stably on the bottom one."* `task-create` authors it for whatever manipulation benchmark `repo` is (IsaacLab/Franka, ManiSkill, …) using that benchmark's default arm + cube assets — no benchmark-specific spec needed. The §6 reward stage is bounded with `success_threshold=0` (+ a small `timesteps_per_iter`) so reward-tune does **exactly one** iteration and stops: the test proves the chain runs end-to-end, not that it discovers a winning reward. A fixed task ⇒ stable fingerprints ⇒ a meaningful resumable cache. (Assumes a *manipulation* benchmark; a locomotion repo would need a different fixture.)

### Step 0 — Pre-flight
```bash
test -n "<repo>" && test -n "<task>" || { echo "L3 needs repo= and task="; exit 1; }
git -C "<repo>" rev-parse --is-inside-work-tree >/dev/null 2>&1 || { echo "<repo> not a git work tree"; exit 1; }
command -v sbatch >/dev/null  # informational; local mode otherwise
```

### Step 1 — Isolated clean worktree (your branch-but-safer)
Use a git **worktree** so the user's checkout is never touched. Name it deterministically from repo+task so resume finds it:
```bash
HEAD=$(git -C "<repo>" rev-parse HEAD)
WT="$(dirname "<repo>")/.harbor-test/<repo-slug>__<task-slug>"
[ "<fresh>" = true ] && { git -C "<repo>" worktree remove --force "$WT" 2>/dev/null; rm -rf "$WT"; }
test -d "$WT" || git -C "<repo>" worktree add "$WT" "$HEAD"
```
Then guarantee it equals the original clone (this is the "create a clean branch" step). `/harbor:reset-workspace` is user-invocable only (`disable-model-invocation: true`), so it cannot be slash-invoked from here — instead **Read `${CLAUDE_PLUGIN_ROOT}/commands/reset-workspace.md` and execute its body** against:
```text
repo="$WT"  clean_inbenchmark_tasks=true      # its own dry-run + confirm gate still applies
```
State + history live in `$WT/.harbor-test/`: `state.json` (stage cache) + `history.md` (append-only).

### Step 2 — Resumable module-by-module run
The stage order + fingerprints come from the engine (Docker-layer style — a stage re-runs iff its plugin definition files, an upstream stage, the repo HEAD, or the task changed):
```bash
PIPE="${CLAUDE_PLUGIN_ROOT}/scripts/test/pipeline.py"
STATE="$WT/.harbor-test/state.json"
python3 "$PIPE" plan --repo-head "$HEAD" --task "<task>" --state "$STATE"   # → {first_to_run, stages:[{name,fp,action}]}
```
For each stage from `first_to_run` onward (skip `action==skip` cached stages):
1. **Invoke the module** (its slash command / its agent dispatch — see the per-stage table).
2. **Run the module's assertions** — its OWN embedded smoke (which the invoke runs internally) PLUS the extra checks in the table. `[art]` checks are deterministic file/JSON/grep checks you run; `[beh]` checks rely on the module's smoke + artifact existence from the GPU run.
3. Append a `## <stage>` section to `history.md` (invoke command, smoke tail, each assertion's pass/fail, artifacts, timestamp).
4. On **PASS**: `python3 "$PIPE" mark --state "$STATE" --stage <name> --status pass --fp <fp from plan>`; continue.
5. On **FAIL** (any assertion or a module `status: fail`): **STOP.** `mark ... --status fail`, append the failure to `history.md`, and return: the failing stage, the exact error (last ~50 lines), and a **suggested fix** (which file/param to change). Do NOT run downstream stages. The next `/harbor:test` run will `plan` → skip the green stages → resume at this one.

### Per-stage modules + assertions

| # | Stage | Invoke | Assertions (smoke ＋ extra; `[art]`=deterministic, `[beh]`=GPU) |
|---|---|---|---|
| 1 | `env-install` | `/harbor:env-install-uv <repo=$WT>` (dependency-generator) | import smoke green · [art] `.venv/bin/python` exists · [art] `harbor/dependency-generator/{setup_uv.sh,probe.json,install_plan.json,install.md}` non-empty · [art] `install_plan.json` valid JSON · [art] verdict `status==pass` |
| 2 | `benchmark` | benchmark-generator (auto-dispatched by the chain, or directly) | L1 + L2 smokes · [art] `benchmark-spec.json` valid + `tasks[]` non-empty + `category=="rl"` · [art] `task_overview.md` + `create-task/task-implementation.md` exist · [art] `scripts/{run_random,render_random}.py` compile |
| 3 | `rl-integration` | rl-integration-generator | T1–T5 per algo · [art] `rl-suite-spec.json` has `algorithm_source.slug` + `.parallel` · [art] `resolve_suite.py --repo $WT --algo ppo` returns correct slug/config · [art] `harbor/scripts/rl/<slug>/*.py` + `configs/rl/*.yaml` compile · [beh] T1 `metrics.jsonl` emits canonical keys |
| 4 | `task-generator` | `/harbor:task-create name=<task> description="<stack-two-cube fixture description>"` (create; §1–§5) | S1–S6 (incl. S3-visual + S4's per-predicate C<i>/V<i>) · [art] `gym.make(<task>)` builds · [art] `task-history.md` Adaptation delta |
| 5 | `reward-add-log` | (auto, before §6) | composer sanity smoke · [art] Path A/B detect correct · [beh] `info["detailed_reward"]` present |
| 6 | `reward-tune` | §6 of task-create, bounded: `success_threshold=0 timesteps_per_iter=<small>` → exactly one iteration | §6 smoke · [beh] one iteration completes (candidate implemented → trains → scores) · [art] `design.json` has `reward.success_term` and a `task_changes.sections` list · [art] `iter_000/verdict.json` has `status=="scored"`, `gate=="ok"`, a non-null `success_rate`, and a `smokes` map containing `S1` and `S6` · [beh] `iter_000/verdict.json` carries non-empty `behavior` + populated `artifacts`, per-term keys present · [art] sequential mode: `<task_dir>/base/` exists and no `-rslot` id was registered · [beh] PROMOTE landed: the task carries the winning design and the winner's own smoke set passes against it |
| — | `dr-generator` | **SKIPPED** (not ready yet — excluded from the chain by the engine's default skip) | |
| 7 | `rl-run` | `/harbor:rl-run task=<task> algorithm=ppo` | [art] reward-logger pre-flight rc · [beh] checkpoint + `metrics.jsonl` + `resolved_config.yaml` · [beh] auto-render `render.mp4` |
| 8 | `rl-eval` | `/harbor:rl-eval checkpoint=<ckpt>` | [beh] `metrics.json` written · [art] task/algo inferred from `resolved_config.yaml` |
| 9 | `rl-render` | `/harbor:rl-render checkpoint=<ckpt>` | inference-moved + frame-diff smoke · [beh] non-empty `render.mp4` |
| 10 | `rl-tricks` | `/harbor:rl-list-tricks` + `/harbor:rl-add-trick <trick>` | [art] tricks enumerated · [art] applying a trick patches the config per its manifest (then revert) |
| 11 | `reset` | Read `commands/reset-workspace.md` + execute its body with `repo=$WT` (teardown ＋ its own test; the command is user-invocable only, so it is not slash-invoked here) | reset smoke: identical-to-clone trio · [art] `harbor/`, `.venv/`, carve-outs, caches gone |

**Skipped in headless e2e (recorded as `skipped`, never failed):** `rl-visualize` ($DISPLAY), full `rl-sweep`/`rl-tune` (multi-trial hours), `plot` (needs W&B runs), `wandb-setup` (host creds), `dr-generator` (not ready).

### Step 3 — Teardown + report
On **all stages pass**: the `reset` stage already restored `$WT`; remove the worktree (`git -C "<repo>" worktree remove --force "$WT"`) and report `L3: PASSED (N stages)`. On **failure**: leave `$WT` + `state.json` + `history.md` in place for resume; report the failing stage + error + suggested fix.

## Constraints
- **L3 never touches the user's checkout** — all work happens in the isolated worktree; the user's branch/working tree is untouched.
- **Fail-fast + resumable.** Stop at the first failing stage; a later run skips green stages and resumes there (fingerprints chain Docker-layer style — editing an upstream module re-runs it onward, a downstream module only itself onward).
- **§6 is bounded** — run reward-tune with `success_threshold=0` (+ a small `timesteps_per_iter`) so it does exactly one iteration and stops; never run an open-ended discovery tune inside the test. The test checks the §6 stage *runs* (reward authored → trains → scores), not that it masters the task.
- **English-only** history + output.

## Examples
```text
/harbor:test                                  # L1→L2→L3 (needs repo/task for L3; errors if absent)
/harbor:test layers=1,2                        # fast deterministic gate, no GPU
/harbor:test layers=3 repo=/path/to/IsaacLab                # builds the stack-two-cube fixture for IsaacLab (Franka)
/harbor:test layers=3 repo=/path/to/ManiSkill              # same fixture, ManiSkill's arm
/harbor:test layers=3 repo=/path/to/IsaacLab fresh=true    # ignore cache, full clean run
```
