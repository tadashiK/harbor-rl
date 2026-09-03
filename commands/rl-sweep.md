---
description: Hyperparameter / seed sweep across multiple RL trials. Takes any keys (task, algorithm, seed, learning_rate, ...) where each value may be a comma-separated list. Computes the Cartesian product, then either dispatches one sub-agent per trial (default) OR — when `cluster=...` is given — renders a SLURM launch.sh that the user submits with `sbatch`. All results land under harbor/rl_experiments/sweeps/<sweep_id>/. Use when the user types /harbor:rl-sweep ... or asks "sweep these hyperparameters", "run a grid over seeds and learning rates", "generate a SLURM launch script for this sweep".
argument-hint: "task=<id1>[,id2,...] algorithm=<ppo,sac,...> [key=v1,v2,...] [parallelism=N] [cluster=<true|path>]"
---

# /harbor:rl-sweep — Multi-Trial Hyperparameter Sweep

Cartesian-product sweep over arbitrary `key=value` arguments. Any value containing a `,` is a list (Hydra literals starting with `{`/`[`/`(` are kept whole, not split). All trials share one `sweep_id` directory.

| Mode | Trigger | Behavior |
|---|---|---|
| **local** (default) | `cluster=` not passed | One sub-agent per trial; agent invokes `/harbor:rl-run`. Sequential by default; `parallelism=N` runs N concurrently. |
| **cluster** | `cluster=true` or `cluster=<path>` | Renders a SLURM-array `launch.sh` under `<sweep_dir>/launch.sh`. No sub-agents; user submits with `sbatch`. |

## Required arguments

| Arg | Notes |
|---|---|
| `task` | Comma-separated list of task IDs (or single id). |
| `algorithm` | Comma-separated list of algorithms (`ppo`, `sac`, `td3`). |

## Optional arguments

Any other `key=value` is forwarded to `/harbor:rl-run`. Comma-separated lists (NOT inside `{...}`/`[...]`) participate in the Cartesian product.

| Arg | Effect |
|---|---|
| `seed=0,1,2` | 3 seeds per (task,algorithm) combo |
| `learning_rate=1e-4,3e-4,1e-3` | 3 LRs |
| `total_timesteps=1_000_000` | scalar — applies to every trial |
| `++env_params={horizon:1000,reward_type:LocomotionReward}` | scalar Hydra dict literal — kept whole, never split |
| `parallelism=4` | (local only) run up to 4 trials concurrently — default 1 |
| `cluster=true` | render SLURM launch.sh from default template (`knowledge/templates/rl-sweep/launch.sh.template`) |
| `cluster=/path/to/template.sh` | render launch.sh from a custom template path |
| `wandb=my-sweep-project` | every trial logs to the same W&B project |
| `smoke=true` (default) | run a fast `/harbor:rl-run` smoke per unique `(algorithm, task)` cell BEFORE dispatch — abort on any failure |
| `smoke=false` | skip the pre-dispatch smoke (use only when you've already verified the matrix) |

### Cluster-mode tunables (only read in cluster mode; ignored otherwise)

| Arg | Default | Effect |
|---|---|---|
| `time_limit` | `24:00:00` | `--time` SLURM directive |
| `gpu_type` | `a40` | passed to `--gres=gpu:<type>` |
| `proxy` | empty (no proxy) | renders `{{PROXY_DEFAULT}}` in both templates — set to your site's proxy URL if compute nodes are firewalled off outbound HTTPS (wandb, and USD asset fetches under IsaacLab) |

## Action

### Step 0 — Parse arguments

Split each `key=value`:
- If `value` starts with `{`, `[`, or `(` → scalar (do NOT split on commas — Hydra literal)
- Else if `value` contains `,` → list (split on commas, strip)
- Else → scalar (still wrapped as a 1-element list internally so the product math is uniform)

Pop these sweep-control knobs BEFORE the Cartesian product (NOT forwarded to `/harbor:rl-run`):
`parallelism`, `cluster`, `time_limit`, `gpu_type`, `proxy`, `smoke`.

### Step 1 — Cartesian product

```python
import itertools
keys, value_lists = zip(*sorted(parsed_args.items()))
trial_configs = [dict(zip(keys, combo)) for combo in itertools.product(*value_lists)]
```

`task` and `algorithm` participate in the product like any other key. So `task=A,B algorithm=ppo,sac seed=0,1` → 2 × 2 × 2 = 8 trials.

### Step 2 — Mint sweep_id and create sweep dir

```bash
sweep_id="sweep_$(date -u +%Y%m%d-%H%M%S)"
sweep_dir="harbor/rl_experiments/sweeps/${sweep_id}"
mkdir -p "${sweep_dir}/trials"
```

Write the sweep manifest skeleton (`${sweep_dir}/manifest.json`):
```json
{
  "sweep_id": "<sweep_id>",
  "created_at": "<iso8601>",
  "parsed_args": {...},
  "mode": "local|cluster",
  "trial_configs": [...],
  "trials": []
}
```

### Step 3 — Mint per-trial IDs

For each `trial_config` (in product order):
```python
trial_index = i                          # 0-based
config_hash = hashlib.sha1(json.dumps(trial_config, sort_keys=True).encode()).hexdigest()[:8]
trial_id    = f"{i:03d}_{trial_config['algorithm']}_{trial_config['task']}_{config_hash}"
trial_dir   = f"{sweep_dir}/trials/{trial_id}"
```

Each trial dir holds: `config.json` (the trial config). In **local mode** it also holds `agent_stdout.log`, `agent_stderr.log`, and a symlink `train_outputs -> ../../../../outputs/<actual_train_run>` once `/harbor:rl-run` finishes.

### Step 3.5 — Pre-dispatch smoke per (algorithm, task) cell

Skipped when `smoke=false`; enabled by default. Catches typos, missing Mjx variants, broken `env_params`, optimizer/JIT failures **before** committing to a long sweep or burning cluster quota.

Build the smoke matrix:

```python
unique_cells = sorted({(t["algorithm"], t["task"]) for t in trial_configs})
```

A 3 algos × 6 tasks × N seeds sweep is 18 smokes — each ~30–60s once JAX is warm. Sweeps with no `algorithm` or `task` axis (e.g. seed-only) collapse to one smoke per unique value.

For each `(algo, task)` cell, invoke `/harbor:rl-run`:

```bash
/harbor:rl-run \
    task=<task> algorithm=<algo> \
    seed=0 num_envs=64 total_timesteps=5000 batch_size=256 \
    wandb=null \
    <user's other scalar overrides — e.g. ++env_params={...}>
```

Forwarded scalar overrides:
- All scalars from `parsed_args` EXCEPT the sweep dimensions themselves (`task`, `algorithm`, `seed`, anything else with `len(values) > 1`).
- Hydra dict literals like `++env_params={...}` MUST be forwarded so the smoke runs against the same env config the real trial would.

Forced overrides (small enough to compile + run a couple of iters in seconds):
- `seed=0`
- `num_envs=64`
- `total_timesteps=5000`
- `batch_size=256` — keeps SAC/TD3 replay batch small enough for the warmup window
- `wandb=null`

Capture stdout/stderr to `<sweep_dir>/smoke/<algo>_<task>.log`. **Pass marker**: the line `[train] saved checkpoint to ...`. **Fail markers**: any traceback, `Error: ...`, or process exit ≠ 0.

```python
failures = []  # (algo, task, last_n_lines_of_log)
for algo, task in unique_cells:
    rc, log = run_smoke(algo, task)
    if rc != 0 or "[train] saved checkpoint" not in log:
        failures.append((algo, task, log[-2000:]))
```

**Outcome:**
- All pass → print `✓ smoke OK: <N>/<N> cells passed`, delete `<sweep_dir>/smoke/`, proceed to Step 4.
- Any fail → print one block per failed cell with the last lines of its log, delete `<sweep_dir>/` (no half-baked sweep on disk), exit non-zero. Do NOT dispatch sub-agents and do NOT render `launch.sh`.

`smoke=false` skips this — recommended only when re-rendering an already-validated matrix (e.g. flipping `total_timesteps` between two cluster runs).

### Step 4a — Dispatch (local mode, default)

For each trial, spawn a `general-purpose` sub-agent. **Run in parallel** if `parallelism > 1` — chunk the trial list, send each chunk as a single message with multiple Agent calls. Otherwise dispatch one at a time.

Per-trial agent prompt template:

> You are trial **`<trial_id>`** of sweep **`<sweep_id>`**. Your job: invoke `/harbor:rl-run` with the EXACT arguments below, capture the resulting trial directory path, and write a one-line trial-result JSON to `<trial_dir>/result.json`.
>
> Arguments to forward to `/harbor:rl-run`:
> ```
> task=<value> algorithm=<value> <key=value ...> sweep_id=<sweep_id> trial_id=<trial_id>
> ```
> (Drop `sweep_id` and `trial_id` if `/harbor:rl-run` errors on them — they're metadata only.)
>
> Capture stdout/stderr to `<trial_dir>/agent_stdout.log` / `agent_stderr.log`. After the train script exits, parse the printed trial directory line (`trial: harbor/outputs/<algo>_<task>_<ts>/`), then write:
> ```json
> {"trial_id":"<trial_id>", "trial_config":{...}, "exit_code":<int>,
>  "train_output_dir":"<absolute>", "checkpoint":"<abs path to AgentXXX_saved.pkl>",
>  "wall_time_sec":<float>}
> ```
> to `<trial_dir>/result.json`. Symlink `<trial_dir>/train_outputs -> <train_output_dir>`.
>
> Do NOT mutate other trial dirs. Do NOT modify the sweep manifest. The orchestrator will aggregate at the end.

### Step 4b — Render launch file (cluster mode)

Triggered when `cluster=` is in args. **No sub-agents are spawned.**

1. **Resolve template path:**
   - `cluster=<path>` → that path (absolute, repo-relative, or `~`-expanded). Wins over auto-detection.
   - `cluster=true` (or `cluster=default`):
     - **IsaacLab auto-detect**: read `harbor/benchmark-generator/benchmark-spec.json:benchmark.name`. If it equals `"IsaacLab"` (case-insensitive), OR if `harbor/apptainer/isaaclab.def` exists, pick `${CLAUDE_PLUGIN_ROOT}/knowledge/templates/rl-sweep/launch.sh.isaaclab.template`. The IsaacLab variant runs the trial inside an apptainer image (handles glibc 2.34+ requirement, NVIDIA Vulkan ICD injection, Kit cache writes via `--writable-tmpfs`, optional site proxy).
     - Otherwise → `${CLAUDE_PLUGIN_ROOT}/knowledge/templates/rl-sweep/launch.sh.template` (bare-metal venv flavor).
   - If the resolved template does not exist → error out with the path.

2. **Discover the trainer command** the same way `/harbor:rl-run` does:
   ```python
   import json
   spec       = json.loads(open("harbor/rl-integration-generator/rl-suite-spec.json").read())
   slug       = spec["algorithm_source"]["slug"]                 # "custom_jax" / ...
   scripts    = spec.get("scripts_dir", f"harbor/scripts/rl/{slug}")
   parallel   = bool(spec["algorithm_source"].get("parallel", False))
   ```
   The launch file targets the host venv — it sources `.venv/bin/activate` if present and otherwise errors out with a clear hint to run `/harbor:env-install-uv` first.

3. **Build per-trial command lines.** For each trial config:
   ```python
   config_name = f"{trial['algorithm']}.parallel" if parallel else trial['algorithm']
   override_tokens = []
   for k, v in trial.items():
       if k == "algorithm":
           continue                                            # baked into config-name
       # Hydra literal: keep braces verbatim, single-quote the whole token
       if isinstance(v, str) and v.lstrip().startswith(("{", "[", "(")):
           override_tokens.append(f"'{k}={v}'")
       else:
           override_tokens.append(f"{k}={v}")
   cmd = f"python {scripts}/train.py --config-name={config_name} " + " ".join(override_tokens)
   ```
   The command is enclosed in double quotes when emitted into the bash array (so the array element is one whole shell word). Inside the trial command, single quotes guard Hydra `{...}` literals from shell brace-expansion / globbing.

4. **Render template.** Replace these placeholders in the template:

   | Placeholder | Value |
   |---|---|
   | `{{N_MINUS_1}}` | `len(trials) - 1` |
   | `{{N_TRIALS}}` | `len(trials)` |
   | `{{PARSED_ARGS_SUMMARY}}` | one-line `key=v1,v2,...` summary |
   | `{{SWEEP_ID}}` | `<sweep_id>` |
   | `{{REPO_PATH}}` | `$(pwd)` — the absolute repo root baked into `REPO_ROOT` (sbatch spools the script, so $0-relative resolution is unreliable). Overridable at submit time via `HARBOR_REPO_ROOT` |
   | `{{REPO_NAME}}` | `os.path.basename($(pwd))` — (isaaclab template only) the container mount point `/repo/<name>` |
   | `{{GENERATED_AT}}` | ISO8601 UTC |
   | `{{TIME_LIMIT}}` | `time_limit` arg or `24:00:00` |
   | `{{GPU_TYPE}}` | `gpu_type` arg or `a40` |
   | `{{WANDB_API_KEY}}` | empty string (the launch file falls back to a pre-set env var) |
   | `{{PROXY_DEFAULT}}` | `proxy` arg, else empty string (no proxy). Overridable at submit time via the `HTTP(S)_PROXY` env var |
   | `{{TRIAL_IDS}}` | one-line-per-id, double-quoted (`"000_ppo_UnitreeH1_<hash>"\n  ...`) |
   | `{{TRIAL_COMMANDS}}` | one-line-per-cmd, double-quoted |

   Write to `<sweep_dir>/launch.sh`, `chmod +x`.

5. **Update manifest** with `mode: "cluster"` and `launch_script: "<sweep_dir>/launch.sh"`. Also write each trial's `<trial_dir>/config.json` so a follow-up local re-run could resume.

6. **Print summary** to the user:
   ```
   sweep_id     : <sweep_id>
   mode         : cluster
   N trials     : <N>
   launch file  : <sweep_dir>/launch.sh
   submit with  : sbatch <sweep_dir>/launch.sh
   ```
   Plus a 1-line preview of trial 0 and trial N-1 so the user can sanity-check the rendered overrides before submitting.

### Step 5 — Aggregate (local mode only)

After all sub-agents return, the orchestrator (this command body) reads `<trial_dir>/result.json` for each trial and updates `<sweep_dir>/manifest.json:trials[]` with the result records. Print a summary table:

```
sweep_id: <sweep_id>
N trials: 8 — succeeded: 7  failed: 1
top by mean_return:
  001_ppo_UnitreeH1_a3f2_e1  return=205.3  ckpt=outputs/ppo_UnitreeH1_20260504-2200/AgentPPO_saved.pkl
  ...
```

(`mean_return` requires running `/harbor:rl-eval` per trial — the sub-agent doesn't do that automatically. Add a note: "to score: `/harbor:rl-eval` each `checkpoint` listed above, then re-run this command's aggregator.")

In cluster mode, aggregation happens **after** the user has submitted and the SLURM jobs have finished — re-running this command body with the same `sweep_id` (resume mode, future work) would scan `<sweep_dir>/trials/*/result.json` written by post-job hooks. For now the user runs `/harbor:rl-eval` against each trial's checkpoint manually.

## Constraints

- **Sweep state is FILE-BASED, not in-memory.** Every trial writes its own `result.json`; the orchestrator reads them. This survives interruptions — re-running the same sweep_id resumes by skipping trials whose `result.json` already exists.
- **Do NOT recurse.** A sub-agent must NOT spawn another sub-agent (`/harbor:rl-sweep` from within a trial). CLAUDE.md hard constraint #4.
- **Do NOT trust untyped CLI input as code.** All key=value args go through `/harbor:rl-run`'s Hydra layer for validation. The sweep orchestrator does NO eval of user input.
- **GPU contention is the user's responsibility.** Default `parallelism=1`. Higher only if the user knows their GPU has the headroom (training at `n_envs=4096` typically eats most of an L4/RTX 3090 already).
- **Each trial's training cmd is identical** to a manual `/harbor:rl-run` — predictable behavior, easy to debug a failing trial in isolation.
- **Cluster mode does NOT submit jobs.** It writes the launch script and stops. The user submits with `sbatch`. This keeps the plugin agnostic to the cluster's auth/quota policies.
- **Cluster template is a starting point.** Two ship-with templates: `launch.sh.template` (bare-metal venv via `module load python` + `.venv/bin/activate`) and `launch.sh.isaaclab.template` (apptainer-wrapped, used automatically when `harbor/benchmark-generator/benchmark-spec.json:benchmark.name == "IsaacLab"` or `harbor/apptainer/isaaclab.def` exists). Both target SLURM with a40 / 24 h defaults. For other schedulers (PBS, LSF, k8s) or sites pass `cluster=<your_template>` with the same `{{...}}` placeholders.
- **No site-specific defaults are baked in.** The shipped templates assume nothing about proxies, module names, or where the repo is cloned: proxy defaults to empty (set `proxy=`), `module load python` is skipped when no module system is present, and `REPO_ROOT` is the render-time absolute repo path (override at submit time with `HARBOR_REPO_ROOT`). Site-specific values belong in the `proxy=` arg, the SLURM environment, or a custom `cluster=<template>` — not in the shipped files.
- The sweep folder is `harbor/rl_experiments/sweeps/<sweep_id>/`; per-train artifact dirs still go to `harbor/outputs/<algo>_<task>_<ts>/` (unchanged from `/harbor:rl-run`). Symlinks bridge the two.
