# Agents

Each agent owns one bounded stage of the workflow and runs in its own isolated context. Only `reward-tuning-agent` dispatches workers of its own; every other agent is a leaf, which is what keeps dispatch depth at two.

## `benchmark-generator`

Adds benchmark sanity scaffolding to a Python env that dependency-generator already built and verified (uv backend — host venv at `<repo>/.venv/`). Reads repo markdown for benchmark-level context, renders TWO scripts (random-action rollout + render-to-MP4), runs a 2-tier smoke (L1 random / L2 render), captures the suite spec into &lt;repo&gt;/harbor/benchmark-generator/benchmark-spec.json, and emits history.md + benchmark.md receipts. Does NOT generate train/eval scripts — that scaffolding is owned by rl-integration-generator. Does NOT modify the env config or dependencies — dependency-generator owns the environment, including the `imageio[ffmpeg]` extras line. The ONE deliberate env-source edit it makes is the IsaacLab dt-strip (Step 3.4): a single-line, idempotent reward-semantics normalization of the vendored `RewardManager.compute()`.

**Tools** · Read, Write, Edit, Bash, Glob, Grep, AskUserQuestion  
**Source** · [`agents/benchmark-generator.md`](https://github.com/supersglzc/harbor-rl/blob/main/agents/benchmark-generator.md)

## `dependency-generator`

ENTRY POINT for setting up a Python simulation repo via uv on the host. Probes the repo, reads README + markdown to build an InstallationPlan, renders `<repo>/harbor/dependency-generator/setup_uv.sh`, executes it (creates `<repo>/.venv/`), runs an import smoke test, and reports back to the main thread. Does NOT recursively dispatch to sub-subagents — the main thread orchestrates the next step (benchmark-generator).

**Tools** · Read, Write, Edit, Bash, Glob, Grep, AskUserQuestion  
**Source** · [`agents/dependency-generator.md`](https://github.com/supersglzc/harbor-rl/blob/main/agents/dependency-generator.md)

## `dr-generator`

COMING SOON — not released, and never dispatched by /harbor:task-create in this version. Authors §7 (domain randomization) of a task in a benchmark repo across THREE groups — (1) robot, (2) object, (3) observation noise. Two modes — **create** (replace the empty DR slot left by task-generator) and **edit** (overwrite an existing DR config). Discovers every available randomization term per group, wires the requested/default ones once-per-episode-per-env (`mode="reset"`), then smoke-checks each effective term by exact value read-back at num_envs=16. Reads task-implementation.md as a per-benchmark migration aid; relies on its own contracts (smoke template + IsaacLab DR reference). Phase A authors DR (or skips per the 3-condition gate); Phase B renders + runs the §7 smoke. Iterates up to 2× on smoke failure; ambiguity batches into a single AskUserQuestion. Writes/updates a handoff at `<task_dir>/handoff-dr-generator.md` (under `harbor/create-task/<slug>/`, next to dr-history.md).

**Tools** · Read, Write, Edit, Bash, Glob, Grep, AskUserQuestion  
**Source** · [`agents/dr-generator.md`](https://github.com/supersglzc/harbor-rl/blob/main/agents/dr-generator.md)

## `reward-candidate-agent`

Takes ONE candidate design from `reward-tuning-agent` — a bounded §1–§5 task delta plus a complete §6 reward — and carries it to a scored verdict: implement both into its own task, pass the smokes for every section it touched plus the §6 reward smoke, train + render on its assigned GPU, then score per-term curves + rendered frames into the verdict JSON. Implements only: it never designs, never re-weights to make a smoke pass, and never touches a sibling's files. Isolated by a slot clone when the pool runs more than one candidate, or working directly on the source task when it runs sequentially. Leaf agent: no Agent tool, no nested dispatch. Dispatched only by `reward-tuning-agent`.

**Tools** · Read, Write, Edit, Bash, Glob, Grep  
**Source** · [`agents/reward-candidate-agent.md`](https://github.com/supersglzc/harbor-rl/blob/main/agents/reward-candidate-agent.md)

## `reward-tuning-agent`

Designs and drives the §6 tuning loop for ONE task. Each candidate it designs is a bounded §1–§5 task delta PLUS a complete reward (B1: adapt-first, magnitude budget, concrete weights/gates/composer, in-flight-aware distinctness); it dispatches one `reward-candidate-agent` per candidate to implement both, smoke every section touched, train + render, and score. Owns DECIDE (best-so-far, convergence, refill) and PROMOTE (winning design onto the source task, re-verified by that winner's own smoke set). Keeps `pool_size` candidates in flight, capped by `gpus` locally. Isolation follows the effective pool: sequential over a `base/` snapshot when it is 1, one slot clone per candidate when it is more. Dispatched by /harbor:reward-tune (standalone) and /harbor:task-create (§6).

**Tools** · Read, Write, Edit, Bash, Glob, Grep, Agent, TaskStop  
**Source** · [`agents/reward-tuning-agent.md`](https://github.com/supersglzc/harbor-rl/blob/main/agents/reward-tuning-agent.md)

## `rl-integration-generator`

Renders the RL training/eval/render scaffold (per-impl harbor/scripts/rl/&lt;slug&gt;/{train,eval,render,env_wrapper}.py + configs + rl-suite-spec + rl-integration.md receipt) into a benchmark repo whose base env (`<repo>/.venv/`) + benchmark-spec.json already exist. Three algorithm sources: `custom_torch` (a self-contained algorithm tree shipped with the plugin — no external RL-lib pip-install), `stable_baseline3` (SB3-backed), `local_implementation` (user-provided package path or github URL — thin shims).

**Tools** · Read, Write, Edit, Bash, Glob, Grep, AskUserQuestion  
**Source** · [`agents/rl-integration-generator.md`](https://github.com/supersglzc/harbor-rl/blob/main/agents/rl-integration-generator.md)

## `rl-tuning-agent`

Open-ended hyperparameter tuning loop for ONE (algorithm, task) pair on an already-set-up RL benchmark repo. Inputs: repo_path, algorithm, task, mode (local|cluster), parent tune_dir, and an optional metric weighting (default: equal weight on sample_efficiency and final_return). Iterates: default-config baseline → tricks pass → log-driven hyperparameter edits → convergence plot. Stops when the running best is not beaten for N consecutive iterations (default 3). Per-cell state lives at &lt;tune_dir&gt;/&lt;wandb_project&gt;/.

**Tools** · Read, Write, Edit, Bash, Glob, Grep, AskUserQuestion  
**Source** · [`agents/rl-tuning-agent.md`](https://github.com/supersglzc/harbor-rl/blob/main/agents/rl-tuning-agent.md)

## `task-cloner`

Clones an existing task into an isolated, independently-editable copy registered under a new suffixed gym id (SAME-REPO mode of the general /harbor:task-clone primitive; cross-benchmark/sim2sim migration is COMING SOON and refused in this release). Copies only the task's EDITABLE surface (env_cfg + the mdp modules the requested `surface` touches — default: reward), rewires the cloned cfg's imports to the copies, mirrors the source's registration mechanism for `<dest>`, then runs the clone smokes (build + rollout + per-term-logging). Writes a manifest listing every created file so the clone can be deleted cleanly. Callers include section A/B experiments and any flow needing an isolated task variant.

**Tools** · Read, Write, Edit, Bash, Glob, Grep  
**Source** · [`agents/task-cloner.md`](https://github.com/supersglzc/harbor-rl/blob/main/agents/task-cloner.md)

## `task-generator`

Authors any subset of §1..§5 of a task in a benchmark repo (§1 register/scene · §2 actions · §3 reset · §4 goal+termination · §5 observation). Two modes — **create** (build a brand-new task with placeholder §6 reward + empty §7 DR) and **edit** (surgical re-author of one or more sections on a task that already builds). Reads task-implementation.md as a per-benchmark migration aid and one `knowledge/references/task-generator/` file per section it works on. Step 0 scaffolds task-history.md; Phase A authors each section and records the analysis behind every design choice; Phase B renders and runs that section's smokes, looping each until it passes; Phase C gates the history with check_task_history.py and emits test-checklist.md, the list of checks that actually ran. The smoke loop is bounded by progress, not by an attempt count — it escalates via AskUserQuestion after two consecutive attempts that fail to move the measured quantity, or at 10 attempts. Ambiguity batches into a single AskUserQuestion per section.

**Tools** · Read, Write, Edit, Bash, Glob, Grep, AskUserQuestion  
**Source** · [`agents/task-generator.md`](https://github.com/supersglzc/harbor-rl/blob/main/agents/task-generator.md)

