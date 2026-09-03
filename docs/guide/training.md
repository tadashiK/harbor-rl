# Training and tuning

## Single trial

```text
/harbor:rl-run task=<TaskID> algorithm=ppo
```

The command resolves the algorithm implementation from `harbor/rl-integration-generator/rl-suite-spec.json` and runs `harbor/scripts/rl/<impl>/train.py` against the repository's own `.venv/bin/python`. Three implementations can sit behind that path: `custom_torch` (the default — a self-contained algorithm tree shipped with the plugin, so there is no external RL library to install and fight with), `stable_baseline3`, and `local_implementation`, which wires your own package or a GitHub URL through thin shims.

Configuration is Hydra, so any key can be overridden inline:

```text
/harbor:rl-run task=<TaskID> algorithm=sac num_envs=4096 learning_rate=3e-4 seed=7
```

Each algorithm ships two configs — `ppo.yaml` and `ppo.parallel.yaml`, and the same for SAC and TD3. The `.parallel` variant is the one used against GPU-batched simulators, where thousands of environments step at once; the plain variant suits single-environment backends. `rl-suite-spec.json` records which flavor the repository was scaffolded for, so callers do not have to guess.

Output lands in `harbor/outputs/<algo>_<task>_<timestamp>/`:

| Artifact | What it is |
|---|---|
| `checkpoint.pth` | the final policy |
| `checkpoint_best.pth` | the best-scoring policy, written on every improvement |
| `checkpoint_<steps>.pth` | periodic snapshots for post-hoc inspection and crash recovery |
| `metrics.jsonl` | one JSON object per log interval — the machine-readable record |
| `tb/` | TensorBoard events |
| `curves/` | plotted PNGs |
| `render.mp4` | rendered automatically on success |

The best/final split exists because it bit: runs routinely peak mid-training and degrade, and scoring only the last checkpoint silently grades the wrong policy. One observed tune candidate peaked at 772 and was scored at 158. Anything that renders or evaluates a run should prefer `checkpoint_best.pth` and say which one it used.

## Evaluate and watch

```text
/harbor:rl-eval checkpoint=<path>          # unbiased eval, writes metrics.json alongside
/harbor:rl-render checkpoint=<path>        # render to MP4
/harbor:rl-visualize checkpoint=<path>     # headed GLFW viewer, needs $DISPLAY
```

`rl-eval` infers task and algorithm from the config saved beside the checkpoint, so a path is usually the only argument you need.

The three answer different questions, and it is worth being deliberate about which you reach for:

**`rl-eval`** gives you the number. It runs the policy without exploration noise and writes `metrics.json` beside the checkpoint — the unbiased success rate and return, which is what you cite.

**`rl-render`** gives you the rollout as a file. It runs the policy and writes `render.mp4` to the checkpoint's directory, so the behavior becomes something you can scrub through, attach to an issue, or hand to an agent to read frame by frame. This is the one the pipeline itself uses: every training run renders on success, and reward candidates extract frames from exactly this output to describe what their policy did.

**`rl-visualize`** gives you the behavior live. It opens a headed GLFW window and steps the policy in front of you on the CPU backend, which is the only one of the three that lets you change viewing angle, follow the robot, and watch a failure develop in real time. It needs `$DISPLAY`, so it is a workstation tool — over SSH without X forwarding, or on a compute node, use `rl-render` instead.

A rough rule: `rl-eval` to know whether it worked, `rl-render` to keep evidence of what it did, `rl-visualize` to understand why.

`rl-render` carries two sanity checks, and both exist because the corresponding failure is silent:

1. **Inference produced actions** — the render script prints the action magnitude it applied, so a policy loading into a zeroed state is visible rather than assumed.
2. **Frames at different timesteps differ** — five frames are sampled across the MP4 (deliberately skipping the first ~10%, since a reset often holds a static frame) and compared by mean per-pixel L1 against a threshold. All consecutive pairs must differ, with one frozen pair tolerated for terminal-state holds.

A frozen scene and a zeroed policy both produce a perfectly valid video of nothing happening, and every scalar metric agrees that nothing is wrong. Pixels are the only signal that separates "standing still because that is optimal" from "standing still because nothing is running." See [semantic correctness](/guide/semantic-correctness) for where else this applies.

## Sweeps

A Cartesian product over any keys, one sub-agent per trial:

```text
/harbor:rl-sweep task=TaskA,TaskB algorithm=ppo,sac seed=0,1,2
```

Any key may take a comma list, not just the three above — a sweep over `learning_rate=1e-4,3e-4` works the same way. Results collect under `harbor/rl_experiments/sweeps/<sweep_id>/`, with a per-trial directory and a manifest recording each trial's config, so a trial can be re-run locally later without reconstructing its arguments.

Adding `cluster=…` renders a SLURM `launch.sh` for `sbatch` instead of running locally. That launcher is deliberately shared with reward tuning's cluster mode: it exports `HTTP(S)_PROXY` and `WANDB_API_KEY`, because most HPC sites firewall outbound HTTPS from compute nodes and the submitting shell's credentials do not necessarily survive `sbatch`. Without them W&B silently falls back to offline mode — training completes, metrics are written, and nothing reaches the dashboard.

## Hyperparameter tuning

Where a sweep enumerates a grid you specified, tuning searches open-endedly:

```text
/harbor:rl-tune task=TaskA,TaskB algorithm=ppo,sac
```

One `rl-tuning-agent` runs per `(task, algorithm)` cell, each with its own directory under `harbor/rl_experiments/tunes/<tune_id>/` and its own append-only history. The loop is: a default-config baseline, then a tricks pass, then hyperparameter edits driven by what the logs actually show — stopping when the running best is not beaten for N consecutive iterations (default 3). Local mode runs cells sequentially; cluster mode dispatches them in parallel, each agent submitting its own SLURM jobs.

Two constraints keep the results honest:

- **A tuned configuration must train in at most twice the default's wall-clock.** Otherwise a "win" is just more compute, and the comparison means nothing.
- **Convergence is required.** A run that has not converged is not a result, however good its curve looked when it stopped.

In the paper's evaluation this matched or beat hand-tuned defaults in 11 of 12 settings across IsaacLab, Bi-DexHands, and Loco-MuJoCo, including two Bi-DexHands tasks where the published SAC baseline fails entirely.

## Training tricks

```text
/harbor:rl-list-tricks
/harbor:rl-add-trick obs_rms_jax algorithm=ppo
```

Six ship today: `obs_rms_torch` and `obs_rms_jax` (in-network observation RMS normalization), `reward_norm_jax`, `value_clip_torch`, `value_norm_torch`, and `distributional_critic_torch`. Each is a directory containing a manifest declaring what it applies to, a patch set, and its own smoke test — so a trick that does not actually take effect fails at application time rather than quietly doing nothing across an entire tuning run. Application edits the algorithm's config in place.

## Plotting

```text
/harbor:plot spec=my_plot.yaml
```

Mean ± std curves from W&B runs, grouped by task × baseline into a multi-panel figure. Each panel averages the seeds belonging to one `(task, baseline)` pair, which is what makes a comparison across baselines legible rather than a tangle of individual runs.

## Metric contract

Every algorithm implementation emits the same metric keys, so tuning, scoring, and plotting read any of them without special cases:

| Scope | Keys |
|---|---|
| All algorithms | `train/episode_length`, and `reward/<term>/episodic_return_mean` per reward term when the env emits `info["detailed_reward"]` |
| PPO | `train/loss/policy`, `train/loss/value`, `train/entropy`, `train/approx_kl`, `train/clip_fraction` |
| SAC | `train/actor_loss`, `train/critic_loss`, `train/entropy`, `train/q_value`, `train/alpha` |
| TD3 | `train/actor_loss`, `train/critic_loss`, `train/q_value` |

The naming rules are non-negotiable because they are what makes the keys machine-readable: `/` namespacing rather than dots or underscores, and `train/entropy` is the **positive** entropy — SB3 emits `entropy_loss = -entropy`, so remapping requires a sign flip. `/harbor:rl-add-log` prints the full contract, and it binds any new algorithm added under `harbor/scripts/rl/`.

The per-term reward keys are what makes reward tuning legible: they are how a designer sees which rung of the ladder a policy actually climbed.

## Long runs

Training outlives an agent's attention span, so completion is defined by an **artifact, not an exit code**. GPU-sim trainers routinely finish the work — checkpoint written, output flushed — and then hang forever in simulator teardown, so a process that never exits is not evidence of failure.

Runs launch detached under `setsid` and are wrapped by `scripts/common/run_with_sentinel.sh`, which polls for either a sentinel *file* or a sentinel *substring in the log*, allows a grace period for the process to exit on its own, then tears down the process group. Its exit codes are the contract: `0` means the sentinel appeared, and `3` means the timeout elapsed without one. A trainer killed *after* its checkpoint lands is a success; a process that exits `0` having produced nothing is not.

Waits are backgrounded rather than polled, which is what keeps a multi-hour run from re-reading the agent's entire context every ten minutes — see [context optimization](/guide/context-optimization).
