# Commands

Every HARBOR command, generated from the plugin source. Invoke any of them as `/harbor:<name>`.

## Environment

Set up a Python simulation repository and its virtual environment.

| Command | Arguments | Purpose |
|---|---|---|
| [`env-install-uv`](https://github.com/supersglzc/harbor-rl/blob/main/commands/env-install-uv.md) | `[path]` | Generate an isolated Python environment for a simulation repo using uv on the host (creates a `.venv/`). |

## Probing

Inspect an existing benchmark or task and emit a portable specification.

| Command | Arguments | Purpose |
|---|---|---|
| [`probe-benchmark`](https://github.com/supersglzc/harbor-rl/blob/main/commands/probe-benchmark.md) | `[repo=<path>]`<br>`[canonical_task=<id>]` | Probe an already-set-up benchmark repo and author &lt;repo&gt;/harbor/create-task/task-implementation.md — the per-family task-authoring guide consumed by /harbor:task-create. |
| [`probe-task`](https://github.com/supersglzc/harbor-rl/blob/main/commands/probe-task.md) | `task=<id>`<br>`[repo=<path>]`<br>`[output=<path>]` | Probe an existing task in the current benchmark repo and emit `<task-slug>-implementation.md` — a portable per-task design-choice spec capturing scene / actions / reset / termination / observation / reward / DR. |

## Task authoring

Create, list, and clone tasks inside a benchmark.

| Command | Arguments | Purpose |
|---|---|---|
| [`task-clone`](https://github.com/supersglzc/harbor-rl/blob/main/commands/task-clone.md) | `op=create`<br>`source=<TaskID>`<br>`dest=<TaskID>`<br>`[repo=<path>]`<br>`[dest_repo=<path>]`<br>`[surface=<sections>]`<br>`[info_out=<path>]`<br>or<br>`op=delete`<br>`dest=<TaskID>`<br>`[repo=<path>]` | Clone a task into an isolated, independently-editable copy — a GENERAL primitive, not tied to any single caller. Same-repo mode registers the copy under a new suffixed gym id (suffix is caller-chosen, e.g. -rewarditer7, -abtest1); cross-benchmark mode (dest_repo=, sim2sim) is COMING SOON and refuses in this release. `op=create` dispatches the task-cloner subagent (copy source's editable surface → rewire imports → register &lt;dest&gt; → run clone smokes). `op=delete` removes the clone's files and confirms the source still builds. |
| [`task-create`](https://github.com/supersglzc/harbor-rl/blob/main/commands/task-create.md) | `name=<TaskID>`<br>`(description="<spec>" \| from=<spec.md>)`<br>`[sections=<comma-list of 1..6>]`<br>`[assets=<path1,path2,...>]`<br>`[algorithm=<ppo\|sac\|td3>]`<br>`[timesteps_per_iter=N]`<br>`[success_threshold=0.5]` | Author a NEW task or surgically edit an EXISTING task in a benchmark repo. Pre-flight verifies the dependency-generator → benchmark-generator → rl-integration-generator chain in sequence and dispatches any missing stage first (rl-integration defaults to the custom_torch algorithm source unless the user specifies one). Then reads &lt;repo&gt;/harbor/create-task/task-implementation.md and orchestrates: task-generator (§1–§5, per-section smokes) → the /harbor:reward-tune training loop for §6 (reward-tuning-agent iterations validated by ACTUAL training until success_rate ≥ threshold) §7 DR is COMING SOON — dr-generator is never dispatched in this release. Also supports REPRODUCE mode via `from=<path>` (a per-task spec emitted by /harbor:probe-task) — rebuilds the task identically including reward / DR / observation / action code (reproduce §6 is pasted verbatim at iter 0 of the same reward-tune loop, then validated by actual training like any other reward). |
| [`task-list`](https://github.com/supersglzc/harbor-rl/blob/main/commands/task-list.md) | `[list]`<br>or<br>`<task-id>` | List or inspect tasks within a Harbor benchmark. |

## Reward engineering

Design and tune the reward, validated by actual training.

| Command | Arguments | Purpose |
|---|---|---|
| [`reward-add-log`](https://github.com/supersglzc/harbor-rl/blob/main/commands/reward-add-log.md) | — | Add per-reward-term logging to a Harbor benchmark repo without changing the env's native reward. The wrapper exposes per-term values on info["detailed_reward"] and asserts `composer(terms) == reward` every step, where composer ∈ {"sum", "product"} matches how the native reward composes its terms. |
| [`reward-tune`](https://github.com/supersglzc/harbor-rl/blob/main/commands/reward-tune.md) | `task=<id>`<br>`[algorithm=<ppo\|sac\|td3>]`<br>`[wandb=<project>]`<br>`[pool_size=N]`<br>`[gpus=N]`<br>`[success_threshold=0.5]`<br>`[timesteps_per_iter=N]`<br>`[seed=N]`<br>`[monitor_early_stop=true\|false]`<br>`[monitor_interval=240]` | Iteratively tune the §6 reward of an EXISTING task with an ASYNC fixed-pool controller. A thin orchestrator: the main agent runs pre-flight + (standalone only) picks the design base, then dispatches `reward-tuning-agent`, which designs each candidate — a bounded §1–§5 task delta plus a complete reward — and dispatches one `reward-candidate-agent` per candidate to implement both, smoke every section touched, train + render, and score per-term curves + rendered frames → success_rate. Keeps `pool_size` candidates in flight (capped by `gpus` locally), loops until `success_rate ≥ success_threshold`, then promotes the winning design onto the source task. Isolation follows the effective pool: sequential over a base snapshot when it is 1, one slot clone per candidate when it is more. |

## Training and tuning

Train, evaluate, render, sweep, and tune RL policies.

| Command | Arguments | Purpose |
|---|---|---|
| [`rl-add-log`](https://github.com/supersglzc/harbor-rl/blob/main/commands/rl-add-log.md) | — | Canonical metric-key contract for ALL rl-integration-generator algorithm implementations (custom_torch PPO/SAC/TD3, stable_baseline3, local_implementation). Defines the keys each algorithm MUST emit to metrics.jsonl + TensorBoard + W&B, plus the per-reward-term tracker pattern. |
| [`rl-add-trick`](https://github.com/supersglzc/harbor-rl/blob/main/commands/rl-add-trick.md) | `<trick_name>`<br>`[algorithm=<ppo\|sac\|td3>]`<br>`[--dry-run]` | Apply an RL training trick (e.g. obs_rms_jax, reward_norm_jax) to a chosen algorithm in the current benchmark repo. Modifies harbor/configs/rl/&lt;algo&gt;.parallel.yaml in place. |
| [`rl-eval`](https://github.com/supersglzc/harbor-rl/blob/main/commands/rl-eval.md) | `checkpoint=<path>`<br>`[task=<id>]`<br>`[n_envs=N]`<br>`[eval_total_steps=N]`<br>`[key=value ...]` | Evaluate a trained RL checkpoint. Runs harbor/scripts/rl/&lt;impl&gt;/eval.py with the given checkpoint, auto-inferring task and algorithm from the saved config. |
| [`rl-list-tricks`](https://github.com/supersglzc/harbor-rl/blob/main/commands/rl-list-tricks.md) | `(no args)` | List all available RL training tricks (in-network obs RMS, reward normalization, etc.) with descriptions, applicable algorithms, and references. |
| [`rl-render`](https://github.com/supersglzc/harbor-rl/blob/main/commands/rl-render.md) | `checkpoint=<path>`<br>`[task=<id>]`<br>`[render_max_steps=N]`<br>`[key=value ...]` | Render a trained RL checkpoint to MP4 and verify the policy actually moves (frame-difference sanity check). |
| [`rl-run`](https://github.com/supersglzc/harbor-rl/blob/main/commands/rl-run.md) | `task=<id>`<br>`algorithm=<ppo\|sac\|td3>`<br>`[key=value ...]` | Train an RL policy on the current benchmark. Wraps harbor/scripts/rl/&lt;impl&gt;/train.py with the repo's `<repo>/.venv/bin/python` and Hydra overrides. |
| [`rl-sweep`](https://github.com/supersglzc/harbor-rl/blob/main/commands/rl-sweep.md) | `task=<id1>[,id2,...]`<br>`algorithm=<ppo,sac,...>`<br>`[key=v1,v2,...]`<br>`[parallelism=N]`<br>`[cluster=<true\|path>]` | Hyperparameter / seed sweep across multiple RL trials. Takes any keys (task, algorithm, seed, learning_rate, ...) where each value may be a comma-separated list. Computes the Cartesian product, then either dispatches one sub-agent per trial (default) OR — when `cluster=...` is given — renders a SLURM launch.sh that the user submits with `sbatch`. All results land under harbor/rl_experiments/sweeps/&lt;sweep_id&gt;/. |
| [`rl-tune`](https://github.com/supersglzc/harbor-rl/blob/main/commands/rl-tune.md) | `task=<id1>[,id2,...]`<br>`algorithm=<a1>[,a2,...]`<br>`[mode=local\|cluster]`<br>`[metric_weights=<json>]`<br>`[stuck_threshold=N]`<br>`[max_iterations=N]`<br>`[smoke=true\|false]` | Grid hyperparameter tuning across tasks × algorithms. Cartesian product → one rl-tuning-agent subagent per (task, algorithm) cell, each running an open-ended tuning loop (default-config baseline → tricks → log-driven hyperparameter edits). Local mode runs cells sequentially; cluster mode dispatches all cells in parallel (each agent submits its own SLURM jobs internally). Maintains a tune-level history.md and gathers per-cell results into a final cross-cell summary. |
| [`rl-visualize`](https://github.com/supersglzc/harbor-rl/blob/main/commands/rl-visualize.md) | `checkpoint=<path>`<br>`[task=<id>]`<br>`[n_steps=N]` | Open a HEADED viewer (GLFW window) showing a trained RL policy in action. Loads a checkpoint, runs N steps on the CPU MuJoCo backend, displays each frame live. |

## Utilities

Plotting, account setup, experience ledgers, workspace reset, and tests.

| Command | Arguments | Purpose |
|---|---|---|
| [`help`](https://github.com/supersglzc/harbor-rl/blob/main/commands/help.md) | — | Show the full harbor plugin surface — slash commands, subagents, hooks. |
| [`plot`](https://github.com/supersglzc/harbor-rl/blob/main/commands/plot.md) | `spec=<path-to-yaml>`<br>`[or no args to walk through writing one]` | Plot mean ± std curves from wandb runs grouped by task × baseline. Reads a YAML spec describing one or more subplots; each subplot averages multiple seeds for the same (task, baseline). |
| [`reset-workspace`](https://github.com/supersglzc/harbor-rl/blob/main/commands/reset-workspace.md) | `repo=<path>`<br>`[clean_inbenchmark_tasks=true\|false]` | Reset a benchmark repo to its original cloned state by removing ALL harbor-plugin-generated files — the entire &lt;repo&gt;/harbor/ tree, the uv .venv/, the scripts/ carve-outs (run_random.py / render_random.py / _&lt;family&gt;_env.py), plugin caches — and (by default) reverting any task code the plugin wrote directly into the benchmark's own source tree. Runs in a subagent. DESTRUCTIVE + irreversible: shows a dry-run plan and asks to confirm first, then verifies via a git-based smoke (including hidden / ignored files) that the repo is byte-identical to its original clone before reporting success. |
| [`test`](https://github.com/supersglzc/harbor-rl/blob/main/commands/test.md) | `[layers=1,2,3]`<br>`[repo=<path>]`<br>`[task=<id>]`<br>`[resume=true\|false]`<br>`[fresh=false]` | Run the harbor plugin test suite. L1 (contract) + L2 (unit) are deterministic pytest layers run by the main thread; L3 is an end-to-end agentic pipeline run in a subagent that drives the task-create chain module-by-module on an isolated, clean benchmark worktree. Default runs L1→L2→L3 in sequence (stop at first failure); `layers=` runs a subset. L3 is resumable Docker-layer style — each module is a fingerprinted stage; on re-run only the changed/failed stage and everything downstream re-runs. Maintains an append-only test history; on any failure it stops, reports the error + a suggested fix, and leaves state so the next run resumes from the unimpacted part. |
| [`update-experience`](https://github.com/supersglzc/harbor-rl/blob/main/commands/update-experience.md) | `target=<name>`<br>`(experience="<bullet>" \| file=<path>)` | Append a learned experience into a target ledger, or file a probed task into the task-library. |
| [`wandb-setup`](https://github.com/supersglzc/harbor-rl/blob/main/commands/wandb-setup.md) | — | Show the host's current W&B login info (account name + masked API key) and offer to re-login, logout, or just inspect. |

