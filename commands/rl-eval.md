---
description: Evaluate a trained RL checkpoint. Runs harbor/scripts/rl/<impl>/eval.py with the given checkpoint, auto-inferring task and algorithm from the saved config. Use when the user types /harbor:rl-eval checkpoint=<path> [task=<id>] [n_envs=N] or asks "evaluate this checkpoint", "how good is the trained policy".
argument-hint: "checkpoint=<path> [task=<id>] [n_envs=N] [eval_total_steps=N] [key=value ...]"
---

# /harbor:rl-eval — Evaluate an RL Checkpoint

Loads a checkpoint, runs unbiased eval (steady-state aggregate over `eval_total_steps`), prints `mean_return ± std`, writes `metrics.json` next to the checkpoint.

## Required argument

| Arg | Notes |
|---|---|
| `checkpoint` | Absolute path OR repo-relative path to `<trial_dir>/AgentXXX_saved.pkl` (custom_jax) or `<trial_dir>/checkpoint.pth` (custom_torch). |

## Optional arguments

| Arg | Default | Effect |
|---|---|---|
| `task` | inferred from checkpoint's `resolved_config.yaml` | Override task — useful for cross-task eval. |
| `n_envs` | inferred (saved cfg) | Number of vmap'd envs — bigger = more stable mean. |
| `eval_total_steps` | 4000 | Per-env step budget. Total samples ≈ `n_envs × eval_total_steps`. |
| `n_episodes` | 100 | Old API, ignored when `eval_total_steps` is set. |
| `seed` | 0 | RNG seed for env reset. |
| any other `key=value` | — | Forwarded as Hydra override. |

## Action

1. **Pre-flight**:
   ```bash
   test -f harbor/rl-integration-generator/rl-suite-spec.json || { echo "no rl-suite-spec.json"; exit 1; }
   test -e <resolved_checkpoint_path>   || { echo "checkpoint not found"; exit 1; }
   ```

2. **Resolve checkpoint path** to absolute. If relative, treat as relative to `$(pwd)`.

3. **Auto-infer algorithm + task** from the checkpoint dir's `resolved_config.yaml`:
   ```python
   import yaml
   cfg = yaml.safe_load(open(Path(checkpoint).parent / "resolved_config.yaml"))
   algo = cfg["algo"]["name"] if isinstance(cfg.get("algo"), dict) else cfg.get("algo")
   inferred_task = cfg.get("task")
   ```
   If `task` was not passed by the user, use `inferred_task`.

4. **Load suite spec** via the canonical reader (same as `/harbor:rl-run`):
   ```bash
   eval "$(python3 "${CLAUDE_PLUGIN_ROOT}/scripts/common/resolve_suite.py" --algo <algo>)"
   # → SLUG, SCRIPTS_DIR, PARALLEL, CONFIG_NAME
   ```

5. `CONFIG_NAME` (resolved above) is `<algo>.parallel` when `PARALLEL=true` else `<algo>`.

6. **Build + run**:
   ```bash
   <prefix> ${SCRIPTS_DIR}/eval.py \
       --config-name=${CONFIG_NAME} \
       task=<task> \
       checkpoint=<abs_path> \
       eval_total_steps=<value> \
       n_envs=<value> \
       <user_overrides...>
   ```
   `<prefix>` resolves to `<repo>/.venv/bin/python`. Error if `.venv/` is missing.

7. **On exit**, print:
   ```
   eval result: mean_return = X ± Y, mean_length = L, n_episodes = N
   metrics:     <trial_dir>/metrics.json
   ```

## Constraints

- **Do NOT silently train** if the checkpoint is missing — error out.
- **Auto-infer task ONLY when not passed.** If the user passes a different `task=`, use that (cross-task eval).
- Output goes to `<checkpoint_dir>/metrics.json`, NOT to a new trial dir.
