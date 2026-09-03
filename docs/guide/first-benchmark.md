# Your first benchmark

Setting up a benchmark means three stages, in order: build the environment, prove the environment actually runs, and render the training stack against it. HARBOR can run all three from one sentence, or you can drive them individually.

The order is forced by what each stage consumes. The sanity layer needs an importable package; the training stack needs the task inventory the sanity layer captures. Each stage ends in a gate, and a stage that cannot prove it worked does not advance.

## The one-sentence path

Clone a repository, open Claude Code in it, and ask:

```text
Set up the env for https://github.com/isaac-sim/IsaacLab
```

HARBOR chains the three stages and stops at the first gate that fails, leaving everything it wrote in place so you can read what happened and resume from there.

## The explicit path

### 1. Environment

```text
/harbor:env-install-uv
```

The `dependency-generator` agent probes the repository — README, `pyproject.toml`, extras, CUDA requirements — and turns what it reads into an `InstallationPlan`. That plan is written to `harbor/dependency-generator/install_plan.json`, rendered into `setup_uv.sh`, and executed to create `.venv/`.

The plan is a real artifact rather than an internal step, and that matters: `setup_uv.sh` is regenerated on every run and must never be hand-edited, so changing how the repo installs means editing the plan and re-rendering. A fix made in the plan survives; a fix made in the script is erased by the next run.

Reading the repository's own docs is what makes this work across simulators — the install sequence comes from the project's instructions rather than from a hardcoded recipe. When a repo offers nothing worth digesting, the renderer falls back to a heuristic (`uv sync --frozen` if a lockfile exists, else requirements and an editable install) and flags that it did so, so a thin plan is visible rather than silent.

After the user's plan, the renderer always appends a "harbor extras" block that idempotently installs what later stages need — `wandb`, `tensorboardX`, `imageio[ffmpeg]`, `matplotlib`, `hydra-core`, `omegaconf`, `stable_baselines3[extra]`, plus `coacd` and `trimesh` for mesh collider decomposition. Folding them in at setup time is what stops a downstream agent from failing on an import three stages later.

**Gate:** the package imports and the expected device is visible. A venv that builds but cannot see the GPU — usually a CPU-only torch build — fails here rather than at the first training run.

### 2. Sanity

The `benchmark-generator` agent renders two entry points at your repository root — `scripts/run_random.py` and `scripts/render_random.py` — and runs a two-tier smoke: a random-action rollout (L1), then a render to MP4 (L2).

**Gate:** a rollout completes with finite rewards and correct shapes, and the rendered frames differ from one another. That second check is what catches a silently frozen scene, which a scalar return will happily hide.

It also captures `harbor/benchmark-generator/benchmark-spec.json`, the task inventory everything downstream reads, plus `task_overview.md` for humans.

One deliberate edit to your source tree happens here, and only for IsaacLab: a single-line, idempotent normalization of the vendored `RewardManager.compute()` that removes the `* dt` factor. Without it, reward weights are per-second rather than per-step, which silently rescales every reward ladder by the control frequency and makes weights authored in one repo meaningless in another.

### 3. Training stack

The `rl-integration-generator` agent renders the RL tree:

```
harbor/scripts/rl/<impl>/{train,eval,render,env_wrapper}.py
harbor/configs/rl/{ppo,sac,td3}.yaml
harbor/rl-integration-generator/rl-suite-spec.json
```

Three algorithm sources are available. `custom_torch` is the default: a self-contained algorithm tree shipped with the plugin, so there is no external RL library to pip-install and fight with. `stable_baseline3` wraps SB3. `local_implementation` wires up your own package or a GitHub URL through thin shims.

**Gate:** each algorithm passes a five-tier smoke — T1 train, T2 eval, T3 render, T4 curves, T5 metrics — which produces a checkpoint, metrics, plots, and a video. The tiers run through the *production* command bodies, the same ones `/harbor:rl-run`, `/harbor:rl-eval` and `/harbor:rl-render` use, so passing the smoke means the real path works rather than a test-only path.

## Train something

```text
/harbor:rl-run task=<task-id> algorithm=ppo
```

Artifacts land in `harbor/outputs/<algo>_<task>_<timestamp>/`: a checkpoint, `metrics.jsonl`, TensorBoard logs, plotted curves, and `render.mp4`. On success the final checkpoint is rendered automatically.

To see what tasks are available:

```text
/harbor:task-list
```

This reads `benchmark-spec.json` rather than importing the simulator, so it answers instantly.

## What is now in your repository

```
<your-repo>/
├── .venv/
├── scripts/{run_random,render_random,_<family>_env}.py
└── harbor/
    ├── dependency-generator/    setup_uv.sh, probe.json, install.md
    ├── benchmark-generator/     benchmark-spec.json, benchmark.md, task_overview.md
    ├── rl-integration-generator/ rl-suite-spec.json, rl-integration.md
    ├── scripts/rl/ · configs/rl/ · utils/
    └── outputs/
```

Everything is plain Python and YAML, meant to be read and edited. Start with the three receipts — `install.md`, `benchmark.md`, `rl-integration.md` — which are written for you rather than for the next agent. See [your workspace](/guide/workspace) for the full map.

## When a stage fails

Each agent diagnoses from the actual error and retries rather than reporting upward immediately, so a transient install failure usually resolves itself. What reaches you is the case that did not.

The stage that failed leaves its `history.md` behind, which is the append-only record of what was tried: tool, command, one line of result per attempt. That is the file to read — the receipt tells you what a successful run produced, the history tells you what a failed one attempted.

Re-running a stage is safe. `setup_uv.sh` is idempotent, and every stage re-reads its inputs from disk rather than from the conversation, so resuming after a fix does not require repeating what already passed.

## Next

[See the whole workflow end to end →](/guide/end-to-end)
