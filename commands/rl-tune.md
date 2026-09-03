---
description: Grid hyperparameter tuning across tasks × algorithms. Cartesian product → one rl-tuning-agent subagent per (task, algorithm) cell, each running an open-ended tuning loop (default-config baseline → tricks → log-driven hyperparameter edits). Local mode runs cells sequentially; cluster mode dispatches all cells in parallel (each agent submits its own SLURM jobs internally). Maintains a tune-level history.md and gathers per-cell results into a final cross-cell summary. Use when the user types /harbor:rl-tune task=<list> algorithm=<list> [mode=local|cluster] or asks "tune PPO and SAC on these tasks", "grid tune".
argument-hint: "task=<id1>[,id2,...] algorithm=<a1>[,a2,...] [mode=local|cluster] [metric_weights=<json>] [stuck_threshold=N] [max_iterations=N] [smoke=true|false]"
---

# /harbor:rl-tune — Grid Hyperparameter Tuning

Cartesian product of `task × algorithm` → one `rl-tuning-agent` subagent per cell. Each subagent runs the open-ended tuning loop in `agents/rl-tuning-agent.md` (default-config baseline → tricks → log-driven hyperparameter edits → comparison plot). The orchestrator (this command body) maintains a tune-level `history.md`, gathers per-cell results, and writes a final summary.

| Mode | Trigger | Behavior |
|---|---|---|
| **local** (default) | `mode=local` (or omitted) | Dispatch one rl-tuning-agent per cell SEQUENTIALLY. Each agent runs in `mode=local` and dispatches its training trials directly via the repo's `.venv/`. |
| **cluster** | `mode=cluster` | Dispatch ALL cells in PARALLEL (single message, multiple Agent calls). Each agent runs in `mode=cluster` and submits its training trials as SLURM jobs (`sbatch` + `squeue` poll). Subagents block on their own job arrays; this orchestrator turn blocks until all subagents return. |

## Required arguments

| Arg | Notes |
|---|---|
| `task` | Comma-separated task IDs (or single id). Each must appear in `harbor/benchmark-generator/benchmark-spec.json:tasks[].id`. |
| `algorithm` | Comma-separated list (`ppo`, `sac`, `td3`). Each must appear in `harbor/rl-integration-generator/rl-suite-spec.json:algorithms[]`. |

## Optional arguments

| Arg | Default | Effect |
|---|---|---|
| `mode` | `local` | `local` (sequential subagents) or `cluster` (parallel subagents, each using SLURM internally) |
| `metric_weights` | `{"sample_efficiency":0.5,"final_return":0.5}` | JSON dict forwarded verbatim to each subagent |
| `stuck_threshold` | `3` | Consecutive non-improving iterations before a cell stops |
| `max_iterations` | `10` | Hard cap on per-cell iterations (including iter_000 baseline). Whichever of `stuck_threshold` / `max_iterations` fires first ends the cell. |
| `smoke` | `true` | Pre-dispatch fast train per (algo, task); abort the whole tune on any failure. Set `smoke=false` only when the matrix is already verified. |

**W&B is always on.** Each cell auto-derives its project name as `tuning-<benchmark>-<task>-<algorithm>` (read `benchmark_name` from `harbor/benchmark-generator/benchmark-spec.json`). Run names within a cell's project are sequential `v1`, `v2`, ... (one per tuning iteration). The orchestrator verifies `wandb login` once in Step 0 and surfaces `/harbor:wandb-setup` if missing.

## Action

### Step 0 — Parse + pre-flight

1. Split `task` and `algorithm` on commas. Compute Cartesian product → cell list. If either list is empty, error out.
2. Verify scaffolding (in `$(pwd)`):
   - `.venv/bin/python` exists → else: tell the user to run `/harbor:env-install-uv` and stop.
   - `harbor/rl-integration-generator/rl-suite-spec.json` exists → else: tell the user to dispatch `rl-integration-generator` and stop.
   - Every requested task ID is in `harbor/benchmark-generator/benchmark-spec.json:tasks[].id`.
   - Every requested algorithm is in `harbor/rl-integration-generator/rl-suite-spec.json:algorithms[]`.
3. If `mode=cluster`, additionally verify `command -v sbatch` is on PATH; if not, surface the error and stop.
4. Verify W&B credentials present (e.g. `grep -q "machine api.wandb.ai" ~/.netrc`). If missing, surface `/harbor:wandb-setup` and stop — every cell will write to W&B.

### Step 1 — Mint tune_id and tune dir

```bash
tune_id="tune_$(date -u +%Y%m%d-%H%M%S)"
tune_dir="harbor/rl_experiments/tunes/${tune_id}"
mkdir -p "${tune_dir}"
```

Each cell lives at `<tune_dir>/<wandb_project>/`, where `<wandb_project> = tuning-<benchmark_name>-<task>-<algorithm>` is derived deterministically (read `benchmark_name` from `harbor/benchmark-generator/benchmark-spec.json`). The orchestrator pre-creates each cell's folder so the subagent can drop files into a known path:

```bash
for algo in "${ALGOS[@]}"; do
  for task in "${TASKS[@]}"; do
    proj="tuning-${BENCHMARK}-${task}-${algo}"
    mkdir -p "${tune_dir}/${proj}"
  done
done
```

Render `<tune_dir>/history.md` from `${CLAUDE_PLUGIN_ROOT}/knowledge/templates/rl-tune/history.md.template` (header + initial per-cell progress table — every row starts with `status: pending`).

Write `<tune_dir>/manifest.json`:
```json
{
  "tune_id": "<tune_id>",
  "created_at": "<iso8601>",
  "repo_path": "<abs>",
  "benchmark_name": "<name>",
  "mode": "local|cluster",
  "tasks": [...],
  "algorithms": [...],
  "metric_weights": {...},
  "stuck_threshold": <N>,
  "max_iterations": <N>,
  "cells": [
    {
      "cell_id": "<algo>_<task>",
      "algorithm": "...",
      "task": "...",
      "wandb_project": "tuning-<bench>-<task>-<algo>",
      "cell_dir": "<tune_dir>/tuning-<bench>-<task>-<algo>",
      "status": "pending"
    },
    ...
  ]
}
```

### Step 2 — Pre-dispatch smoke (skipped on `smoke=false`)

For each unique (algo, task) cell, run a fast train via the `/harbor:rl-run` mirror. Smoke logs land in a temporary `<tune_dir>/_smoke/` subdir that is deleted on success:

```bash
mkdir -p "${tune_dir}/_smoke"
"<repo>/.venv/bin/python" harbor/scripts/rl/<slug>/train.py \
    --config-name=<algo>.parallel \
    task=<task> seed=0 num_envs=64 total_timesteps=5000 batch_size=256 wandb=null \
    > "${tune_dir}/_smoke/<algo>_<task>.log" 2>&1
```

Pass marker: line containing `[train] saved checkpoint`. Fail markers: any traceback / `Error:` / non-zero exit.

- All pass → print `✓ smoke OK: N/N cells passed`, delete `${tune_dir}/_smoke/`, proceed to Step 3.
- Any fail → print one block per failed cell (last 50 lines of its log), delete `${tune_dir}/`, exit non-zero. Do NOT dispatch subagents.

### Step 3a — Local dispatch (orchestrator-driven phase machine)

Local-mode `python train.py …` calls block for the duration of training, often longer than the Bash tool's 10-minute cap. So local mode uses the SAME 2-phase machine as cluster, with `sbatch + squeue wait` substituted for `bash run.sh + bg-process wait`. Subagents NEVER block on training; the orchestrator owns the wait.

Per cell, maintain `<tune_dir>/<wandb_project>/state.json` with the same schema as cluster mode (see Step 3b). Cells in local mode run **sequentially** (one cell's loop completes before the next starts) — local-mode parallelism is bounded by GPU count, not the orchestrator.

```
# Per cell, sequentially:
LOOP iter from 0 while not stopped:
  # 1. SUBMIT phase
  sibling = read_last_n_lines(<tune_dir>/_findings.jsonl, n=10)   # may be empty
  result_submit = Agent(rl-tuning-agent, prompt={
    phase=submit, mode=local, iter=state.iter, state_path=<...>,
    sibling_findings=sibling, ...all pinned overrides...
  })
  # The agent renders iter_<NNN>/overrides.yaml + iter_<NNN>/run.sh (a single-line
  # bash script that runs `.venv/bin/python train.py … <overrides>`), then returns.
  # It does NOT execute run.sh.

  # 2. RUN+WAIT — orchestrator-owned, agent-free
  Bash(run_in_background=true, command=
    "bash <tune_dir>/<proj>/iter_<NNN>/run.sh > <tune_dir>/<proj>/iter_<NNN>/run.log 2>&1
     && touch <tune_dir>/<proj>/iter_<NNN>/.done")
  # Harness fires a completion notification when the python process exits.
  # The orchestrator does NOT poll. Continue with bookkeeping or wait for the notification.

  # 3. SCORE phase
  result_score = Agent(rl-tuning-agent, prompt={
    phase=score, mode=local, iter=state.iter, state_path=<...>
  })
  # Agent reads iter_<NNN>/run.log once, extracts metrics, scores, decides.
  # On the iter that triggers stop, it ALSO finalizes tuning-history.md +
  # writes <cell_dir>/result.json + inlines `result` in the response.

  # 4. FINDINGS — orchestrator appends sibling-visible bullets
  for f in result_score.findings:
    append_jsonl(<tune_dir>/_findings.jsonl,
      {"cell": <wandb_project>, "iter": state.iter, "ts": <iso8601>, "finding": f})

  state.update(...)
  if result_score.stopped_by:
    persist(result_score.result, to=<cell_dir>/result.json)   # already written by agent; verify presence
    break
  state.iter += 1
```

There is no separate `finalize` Agent dispatch — the `score` that triggers stop also writes `result.json`. The state.json schema, the 2-phase contract, and the stop conditions (`stuck >= stuck_threshold` OR `iter >= max_iterations`) are identical to cluster mode. Differences from cluster:
- `run.sh` instead of `launch.sh` (no SBATCH directives needed)
- `iter_<NNN>/run.log` instead of `iter_<NNN>/slurm-<jobid>_0.out`
- The orchestrator's wait uses `bash run.sh` in `run_in_background=true` instead of `sbatch + while squeue …`
- Cells run sequentially (since each one consumes the local GPU)

If a cell errors out (subagent returns errors OR a phase agent fails fatally), log it in `<tune_dir>/history.md` and `manifest.json` with `status: failed`, then continue to the next cell — do NOT abort the whole tune.

### Step 3b — Cluster dispatch (orchestrator-driven phase machine)

Cluster-mode trials run for HOURS — far longer than any subagent should remain alive. The orchestrator drives a state machine per cell, dispatching the rl-tuning-agent ONLY for atomic phases (`submit` / `score` / `finalize`), each lasting seconds-to-minutes. The SLURM wait is owned by the orchestrator via background Bash. Subagents NEVER block on SLURM.

**Per cell, maintain `<tune_dir>/<wandb_project>/state.json`:**
```json
{
  "wandb_project": "tuning-loco-mujoco-UnitreeA1-sac",
  "iter": 0,                                 // current iter index (0-based)
  "stuck": 0,                                  // consecutive non-improving iters
  "max_iterations": 10,
  "stuck_threshold": 3,
  "running_best": {                          // null until iter_000 has been scored
    "iter": 0, "score": 0.42, "final_return": 312.0, "sample_eff": ..., "overrides_path": "...",
    "wall_clock_sec": 1834
  },
  "iters": [
    {"iter": 0, "status": "running|done|failed", "jobid": "12345",
     "score": 0.42, "decision": "accept|reject|failed",
     "novel_heuristic": null, "label": "baseline",
     "submitted_at": "...", "exited_at": "..."}
  ],
  "stopped_by": null,                          // "stuck_threshold" | "max_iterations" | null while running
  "pinned_overrides": ["total_timesteps=50000000", "+record_video=false"]   // forwarded into every train cmd
}
```

**State machine (per cell — cells run in parallel as separate state machines, each with its own SLURM job array and background-bash wait):**

```
LOOP:
  # 1. SUBMIT phase (with sibling-findings injection)
  sibling = read_last_n_lines(<tune_dir>/_findings.jsonl, n=10)   # may be empty
  result_submit = Agent(rl-tuning-agent, prompt={
      phase=submit, mode=cluster, iter=state.iter, state_path=<...>,
      sibling_findings=sibling, ...all pinned overrides...
  })
  state.iters[i].jobid = result_submit.jobid
  state.iters[i].status = "running"
  write state.json

  # 2. WAIT — orchestrator-owned, agent-free
  signal=<tune_dir>/<wandb_project>/iter_<NNN>/.done
  Bash(run_in_background=true, command=
    "JOBID=" + result_submit.jobid + "
     while squeue -h -j $JOBID 2>/dev/null | grep -q .; do sleep 600; done
     touch " + signal)
  # The harness notifies the orchestrator when this background process exits.
  # Per harness rules: do NOT poll, do NOT sleep, do NOT proactively check progress.

  # 3. SCORE phase (after notification)
  result_score = Agent(rl-tuning-agent, prompt={
      phase=score, mode=cluster, iter=state.iter, state_path=<...>
  })
  state.iters[i].score = result_score.score
  state.iters[i].decision = result_score.decision
  state.iters[i].status = "done" | "failed"
  if result_score.accept: state.running_best = ...; state.stuck = 0
  else: state.stuck += 1

  # 4. FINDINGS — orchestrator appends sibling-visible bullets
  for f in result_score.findings:
    append_jsonl(<tune_dir>/_findings.jsonl,
      {"cell": <wandb_project>, "iter": state.iter, "ts": <iso8601>, "finding": f})

  if result_score.stopped_by:
      # The score-phase agent ALREADY wrote <cell_dir>/result.json + finalized
      # tuning-history.md. Just persist state and break.
      state.stopped_by = result_score.stopped_by
      write state.json
      break LOOP
  state.iter += 1
  write state.json

# No finalize Agent dispatch — the score that triggered stop already wrote result.json.
```

**Parallelism across cells.** Submit-phase Agent calls for different cells SHOULD be batched in one orchestrator message (multiple `Agent(...)` calls in parallel) to start their SLURM arrays concurrently. After that, each cell's wait/score/submit cycle is independent. The orchestrator can have multiple background-bash waits in flight simultaneously and re-dispatch each cell's score-phase agent independently as each notification fires.

**Findings serialization.** Per-cell `score` calls return concurrently as their notifications land. The orchestrator's appends to `<tune_dir>/_findings.jsonl` are naturally serialized (one Agent return processed per orchestrator turn) — no locking needed. The next `submit` for any cell sees ALL findings emitted before that submit was dispatched.

**Pinning iter_000 from a pre-existing SLURM job (resume mode).** If a cell's `state.json` already has `iters[0].jobid` set on first orchestrator entry (e.g., user reused a baseline from a previous tune), skip the submit phase for iter_000 and jump straight to the wait → score path. The folder must already contain `iter_000/launch.sh + sbatch_id.txt + slurm-*.out` for the score phase to function.

### Step 4 — Aggregate per-cell results + final summary

Once all cells have returned (success or failure) — i.e., every cell's `score` phase emitted `stopped_by != null` and wrote `<cell_dir>/result.json`:

1. For each cell, read `<tune_dir>/<wandb_project>/result.json` and append/finalize the per-cell row in `<tune_dir>/history.md`.
2. Update `manifest.json:cells[*]` with final status + best score + `best_overrides_path` + `best_total_timesteps` per cell.
3. Append the **Final summary** block to `<tune_dir>/history.md`:
   - **Overall improvements** — per-cell Δ return / Δ sample-efficiency / Δ wall-clock vs baseline (one table).
   - **Tuning techniques applied (across cells)** — union of tricks kept; hyperparameter edits that paid off; edits that didn't generalize. Note any cells that hit `max_iterations` vs `stuck_threshold` (read `result.json:stopped_by`).
   - **Cross-cell takeaways** — 3–5 bullets distilled from the per-cell `tuning-history.md` summaries.
   - **Errors** — one row per failed cell.
4. Print to the user:
   ```
   tune_id : <tune_id>
   mode    : <mode>
   cells   : <N>  — done: <K>  failed: <F>
   history : <tune_dir>/history.md
   per-cell: <tune_dir>/<wandb_project>/result.json
   ```

`novel_heuristics` from each cell's `result.json` are NOT auto-promoted to `tuning-experience.md`. The user can manually review them later via `jq '.novel_heuristics' <tune_dir>/*/result.json` and update `tuning-experience.md` themselves if any are worth keeping.

The cross-algorithm comparison sweep + plot was previously Steps 5–6 of this command — they are removed. If you want a final-comparison sweep, dispatch `/harbor:rl-sweep` manually post-tune with the per-cell `best_overrides_path` values from each `result.json`, then `/harbor:plot` for the figures.

## Constraints

- **Do NOT recurse.** A subagent must NOT invoke `/harbor:rl-tune` from within itself. CLAUDE.md hard constraint #4.
- **Do NOT mutate `harbor/configs/rl/<algo>.parallel.yaml` from this command body.** Only the rl-tuning-agent (via `/harbor:rl-add-trick`) may edit it.
- **State is FILE-BASED.** Every subagent writes its own `tuning-history.md` and `result.json`; the orchestrator reads them post-hoc. This survives interruptions — re-running with the same `tune_id` (resume mode, future work) would skip cells whose `result.json` already exists.
- **Cluster-mode parallelism is per-cell, not per-iteration.** Inside one cell's tuning loop, iterations remain sequential — the next candidate depends on the running best.
- **Mode applies uniformly** to all cells. Mixed-mode tunes are not supported.
- **Smoke aborts the whole tune.** Per-cell smoke failure is a fatal signal — the matrix has a typo or a broken env wiring; better to fix it before burning hours of compute.

## Examples

```text
# 3 tasks × 3 algorithms = 9 sequential local tunes
/harbor:rl-tune task=UnitreeH1,UnitreeGo2,Cheetah algorithm=ppo,sac,td3

# Same 9 tunes in parallel on the cluster (each cell auto-creates its own W&B
# project: tuning-<benchmark>-<task>-<algo>, with run names v1, v2, ...)
/harbor:rl-tune task=UnitreeH1,UnitreeGo2,Cheetah algorithm=ppo,sac,td3 mode=cluster

# Single-cell tune (Cartesian product of one × one)
/harbor:rl-tune task=UnitreeH1 algorithm=ppo

# Skip smoke (matrix already verified earlier today)
/harbor:rl-tune task=A,B algorithm=ppo,sac smoke=false

# Custom metric weighting (favor final return over sample efficiency 70/30)
/harbor:rl-tune task=UnitreeH1 algorithm=ppo,sac metric_weights='{"sample_efficiency":0.3,"final_return":0.7}'
```
