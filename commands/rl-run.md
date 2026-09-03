---
description: Train an RL policy on the current benchmark. Wraps harbor/scripts/rl/<impl>/train.py with the repo's `<repo>/.venv/bin/python` and Hydra overrides. Use when the user types /harbor:rl-run task=<id> algorithm=<ppo|sac|td3> [overrides...] or asks "train PPO/SAC/TD3 on task X", "kick off training".
argument-hint: "task=<id> algorithm=<ppo|sac|td3> [key=value ...]"
---

# /harbor:rl-run — Train an RL Policy

Single-trial training entry point. Reads the rendered RL suite spec, picks the matching `train.py`, dispatches it with the requested task / algorithm / Hydra overrides.

## Required arguments

| Arg | Values | Notes |
|---|---|---|
| `task` | string | Task ID (e.g. `UnitreeH1`). Must appear in `harbor/benchmark-generator/benchmark-spec.json:tasks[].id`. |
| `algorithm` | `ppo` \| `sac` \| `td3` | Picks `harbor/configs/rl/<algo>.parallel.yaml` (or `<algo>.yaml` if the suite spec says `parallel=false`). |

## Optional arguments

Any other `key=value` token after the required two is forwarded **verbatim** as a Hydra override. Examples:

| Override | Effect |
|---|---|
| `total_timesteps=1_000_000` | shorten training |
| `num_envs=512` | smaller batch |
| `seed=7` | RNG seed (default: a fresh random seed per run, printed as `[train] seed=N` and persisted to `resolved_config.yaml`) |
| `wandb=my-project` | enable W&B (project name = value) |
| `normalize_env=true` | turn on NormalizeVecReward |
| `env_params.horizon=500` | override env horizon (Hydra dotted-path) |

## Action

1. **Pre-flight**:
   ```bash
   cd "$(pwd)"
   test -f harbor/rl-integration-generator/rl-suite-spec.json || { echo "no rl-suite-spec.json — run rl-integration-generator first"; exit 1; }
   ```

2. **Load suite spec** via the canonical reader (resolves slug / scripts dir / parallel / config name from `algorithm_source.slug` + `algorithm_source.parallel` — the single place that key path lives):
   ```bash
   eval "$(python3 "${CLAUDE_PLUGIN_ROOT}/scripts/common/resolve_suite.py" --algo <algo>)"
   # → SLUG, SCRIPTS_DIR, PARALLEL, CONFIG_NAME=<algo>[.parallel]
   ```

3. `CONFIG_NAME` (resolved above) is `<algo>.parallel` when `PARALLEL=true`, else `<algo>`.

4. **Reward-logger pre-flight**. Before launching training, check that the chosen task has the per-term reward wrapper wired (so W&B will show per-term curves, and so the user has the diagnostic signal `/harbor:reward-add-log` was meant to provide):

   ```bash
   python3 "${CLAUDE_PLUGIN_ROOT}/scripts/rl-run/check_reward_logger.py" \
       --repo "$(pwd)" --task "<task>"
   rc=$?
   ```

   - **rc=0** → reward logger present (full per-term). Proceed to step 5.
   - **rc=3** → reward logger present **but Direct env** (passthrough only — `info["detailed_reward"] = {"total": env_rew}`). Proceed to step 5, but **surface a one-line note to the user** so they're not surprised when W&B shows only `reward/total/...` and no per-term curves. The task is still trainable.
   - **rc=1** → reward logger missing for this task. Dispatch `Skill('reward-add-log')` (passing the task ID), wait for it to finish, then re-run the check. If `reward-add-log` exits non-zero or the user cancels the AskUserQuestion inside it, abort with a clear message and do NOT start training.
   - **rc=2** → `task_overview.md` missing or stale. Tell the user to run benchmark-generator and stop.

   If `harbor/benchmark-generator/task_overview.md` does not exist at all, treat as `rc=2`. Do NOT silently proceed — the file is a contract surface for `/rl-run` and its absence means benchmark-generator hasn't been run on this repo.

5. **Resolve run prefix**:
   - Require `<repo>/.venv/bin/python` to exist → prefix = `<repo>/.venv/bin/python`.
   - If missing, error out with "`.venv/` not found — run `/harbor:env-install-uv` first".

6. **Build + run the training command**:
   ```bash
   <prefix> ${SCRIPTS_DIR}/train.py \
       --config-name=${CONFIG_NAME} \
       task=<task> \
       <user_overrides...>
   ```
   Stream stdout/stderr live; do not background.

7. **On exit**, print the trial directory:
   ```
   trial: harbor/outputs/<algo>_<task>_<YYYYMMDD-HHMMSS>/
     checkpoint: <trial>/AgentXXX_saved.pkl   (or checkpoint.pth for custom_torch)
     metrics:    <trial>/metrics.jsonl
     curves:     <trial>/curves/
   ```
   Non-zero exit code from train.py → propagate.

8. **Auto-render on success (main-agent responsibility).** When this command is invoked from the main agent, the main agent MUST monitor the training process and, the moment it exits cleanly (rc=0 and a usable checkpoint exists in the trial directory), immediately dispatch `/harbor:rl-render` (or the equivalent `harbor/scripts/rl/<slug>/render.py`) on the final checkpoint to produce `<trial>/render.mp4`. No need to ask the user — render unconditionally on rc=0 and surface the MP4 path. Skip only if training errored / was aborted, or the user explicitly said "don't render this one". If training was launched as a detached background process, keep a wakeup chain alive until the final `checkpoint.pth` (or `Agent*_saved.pkl` for SB3) appears, then render. Do NOT just report "training healthy" and stop polling.

## Constraints

- **Do NOT modify the suite spec.** If the user wants a different algorithm source they need to re-run rl-integration-generator.
- **Do NOT inject defaults** for `task` or `algorithm` — both are required. If missing, print the dispatch table and stop.
- **Do NOT background** the training. The user invoked it interactively; failures should surface immediately. For sweeps that fan out across many trials, use `/harbor:rl-sweep` instead.
- Hydra overrides starting with `+` (Hydra append) and `~` (Hydra delete) are forwarded as-is — don't unquote or rewrite them.
