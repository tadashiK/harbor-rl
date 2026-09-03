# RL Suite Spec

Schema for `<repo>/harbor/rl-integration-generator/rl-suite-spec.json` — written by `rl-integration-generator`, consumed by `rl-tuning-agent` and the RL training/tuning commands (`/harbor:rl-run`, `/harbor:rl-eval`, `/harbor:rl-sweep`, `/harbor:rl-tune`).

## Lifecycle

```
benchmark-generator  →  harbor/benchmark-generator/benchmark-spec.json   (category, tasks, language, gpu_sim, …)
rl-integration-gen   →  harbor/rl-integration-generator/rl-suite-spec.json    (algorithm source, algorithms, W&B, scoring)
rl-tuning-agent      →  harbor/rl_experiments/tunes/<tune_id>/      (per-cell tune state,
                        incl. tuning-history.md + result.json per cell)
```

## Schema (v1)

```json
{
  "schema_version": 1,
  "benchmark": {
    "name": "<benchmark name from registry or repo dir>",
    "repo_path": "/abs/path/to/<repo>",
    "category": "rl"
  },
  "tasks": [
    {
      "id": "<task id, e.g. Cartpole-v1 or Ant>",
      "make": "gymnasium.make | isaacgymenvs.make | ...",
      "max_episode_steps": 200,
      "success_metric": "success | null",
      "reward_metric": "episode_return"
    }
  ],
  "algorithm_source": {
    "kind": "custom-torch | stable-baselines3 | custom-library | custom-path",
    "package": "stable-baselines3[extra] | <import path> | <absolute path>",
    "parallel": true
  },
  "algorithms": {
    "ppo": { "class": "PPO", "config": "configs/rl/ppo.yaml" },
    "sac": { "class": "SAC", "config": "configs/rl/sac.yaml" },
    "td3": { "class": "TD3", "config": "configs/rl/td3.yaml" }
  },
  "logging": {
    "wandb": {
      "enabled": true,
      "project": "harbor-rl",
      "entity": null,
      "mode": "online | offline | disabled"
    },
    "local_dir": "harbor/rl_experiments"
  },
  "selection_metric": {
    "primary": "success_rate",
    "secondary": "eval_return_mean",
    "tertiary": "sample_efficiency",
    "stability_weight": 0.05
  },
  "training_defaults": {
    "timesteps_per_trial": 100000,
    "eval_episodes": 10,
    "n_envs_cpu": 8,
    "n_envs_gpu": 4096,
    "seed": null
  }
}
```

## Field notes

- **`benchmark.category`** — always `"rl"`. Every benchmark reaching the RL stack is treated as RL.
- **`tasks[].success_metric`** — name of the boolean key in the env's `info` dict that signals episode success. `null` for benchmarks where success is not defined (in which case scoring falls back to `eval_return_mean`).
- **`algorithm_source.kind`**
  - `custom-torch` — self-contained PyTorch implementations under `knowledge/templates/rl-integration-generator/custom_torch/` (GPU-parallel, hydra-config). Implies `parallel=true`.
  - `stable-baselines3` — `knowledge/templates/rl-integration-generator/stable_baseline3/` (CPU vec-env, gym vec). Implies `parallel=false` unless the env itself batches.
  - `custom-library` — user-provided pip-installable package; `package` is the import path (e.g. `cleanrl`).
  - `custom-path` — absolute path on disk; `rl-integration-generator` mounts it into the rl container.
- **`algorithm_source.parallel`** — drives which config family is selected by `algorithm_adapters.py`:
  - `true` → `configs/rl/<algo>.parallel.yaml` (GPU-batched parallel envs)
  - `false` → `configs/rl/<algo>.yaml` (gym vec-env, CPU)
- **`selection_metric`** — the default candidate-scoring weights `rl-tuning-agent` uses in its
  `score` phase. Each named metric is min-max normalized to `[0,1]` across the iterations seen so
  far in that cell, then weighted:
  ```
  score = Σ  weight[m] * normalized(m)          # over the metrics in `metric_weights`
  ```
  Default when unset: `{"sample_efficiency": 0.5, "final_return": 0.5}`. A caller may override it
  per tune via `/harbor:rl-tune metric_weights=<json>`, which is forwarded to the agent (see
  `agents/rl-tuning-agent.md`, `metric_weights` input). The agent computes this inline from
  `iter_<N>/run.log` — there is no separate scoring script.

## Benchmark-spec extension

`benchmark-generator` writes `harbor/benchmark-generator/benchmark-spec.json`. To support the RL training/tuning surface, the spec must include:

```json
{
  "category": "rl",
  "language": "pytorch | jax",
  "gpu_sim": true,
  "tasks": [
    { "id": "Ant", "reward_implemented": true, "max_episode_steps": 1000 }
  ]
}
```

If `tasks[]` is missing or `category != "rl"`, `rl-integration-generator` refuses to proceed.
