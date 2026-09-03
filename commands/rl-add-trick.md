---
description: Apply an RL training trick (e.g. obs_rms_jax, reward_norm_jax) to a chosen algorithm in the current benchmark repo. Modifies harbor/configs/rl/<algo>.parallel.yaml in place. Use when the user types /harbor:rl-add-trick <trick_name> [algorithm=<algo>] or asks "apply obs RMS to PPO", "enable the reward norm trick", "add this trick to my training".
argument-hint: "<trick_name> [algorithm=<ppo|sac|td3>] [--dry-run]"
---

# /harbor:rl-add-trick — Apply an RL Trick

Each trick is a self-contained code+config patch under
`${CLAUDE_PLUGIN_ROOT}/knowledge/templates/rl-tricks/<name>/`:
- `manifest.yaml` — name, description, supported algorithms.
- `patches.yaml` — `file_patches[]` (literal find/replace edits referencing
  raw `.find` / `.replace` text files in `edits/`) plus `config_patches[]`
  (yaml key=value flips on `harbor/configs/rl/<algo>.parallel.yaml`).

The default rendered scaffold is **trick-free** — applying a trick injects the
implementation (e.g. `RunningMeanStd` class, `NormalizeVecReward` wrap call)
and flips the relevant config keys. Re-running is idempotent (already-applied
edits report `[noop]`).

## Required argument

| Arg | Notes |
|---|---|
| `<trick_name>` | Name of an entry under `${CLAUDE_PLUGIN_ROOT}/knowledge/templates/rl-tricks/`. List via `/harbor:rl-list-tricks`. |

## Optional arguments

| Arg | Default | Effect |
|---|---|---|
| `algorithm=<ppo\|sac\|td3>` | applies to ALL algorithms in the trick's manifest | Restrict to one algorithm. |
| `--dry-run` | (no flag) | Print the proposed yaml change without writing. |
| `--skip-smoke` | (no flag) | Skip the trick's `smoke.py` after applying. |

## Action

1. **Pre-flight**:
   ```bash
   test -f harbor/rl-integration-generator/rl-suite-spec.json || { echo "no rl-suite-spec.json — dispatch the rl-integration-generator subagent first"; exit 1; }
   ```

2. **Resolve repo** = `$(pwd)`.

3. **Run the apply script**:
   ```bash
   python3 "${CLAUDE_PLUGIN_ROOT}/scripts/rl-tricks/apply_trick.py" \
       --trick <trick_name> \
       [--algorithm <algo>] \
       --repo "$(pwd)" \
       [--dry-run]
   ```

4. **Run the trick's smoke** (skip on `--dry-run` or `--skip-smoke`). If
   `${CLAUDE_PLUGIN_ROOT}/knowledge/templates/rl-tricks/<trick>/smoke.py` exists,
   invoke it with the repo's venv so `torch` / `PyYAML` resolve against
   the same versions the trick patched:
   ```bash
   smoke="${CLAUDE_PLUGIN_ROOT}/knowledge/templates/rl-tricks/<trick>/smoke.py"
   if [ -f "$smoke" ]; then
       "$(pwd)/.venv/bin/python" "$smoke" --repo "$(pwd)" || exit $?
   fi
   ```
   - The smoke script's stdout/stderr is printed verbatim. A non-zero exit
     means the patch did not produce the intended runtime effect; do NOT
     report success — surface the smoke's failure lines and stop.
   - Tricks with no `smoke.py` skip step 4 silently and proceed to 5.

5. **On success**, print which file(s) were patched and what to do next:
   - "Trick `<name>` applied to `<algos>` and smoke passed. Re-run training to use the new config:"
   - "  `/harbor:rl-run task=<id> algorithm=<algo>`"

6. **On failure** (unknown trick name, algo not supported by the trick, no
   matching yaml, OR smoke failed), surface the executor's / smoke's error
   message verbatim. Suggest `/harbor:rl-list-tricks` to see what's available.

## Constraints

- **Never patch the plugin templates** (`knowledge/templates/rl-integration-generator/...`)
  — only the user's `<repo>/harbor/scripts/rl/<slug>/...` and
  `<repo>/harbor/configs/rl/*.parallel.yaml`.
- **Never patch yaml files outside `harbor/configs/rl/`** for a trick.
- If `--dry-run`, every code edit prints `[dry-run] ...would replace 1 occurrence`
  and every config flip prints the `cur -> new` value, with no files written.
- Patching is **idempotent** — re-applying the same trick reports `[noop]`
  per edit (find string already consumed, post-trick string detected).
- Tricks **stack** — applying `obs_rms_jax` then `reward_norm_jax` patches both
  in order. The two are designed to be orthogonal (obs_rms_jax touches
  `models/mlp.py` + `algo/*.py` + `use_obs_rms` config; reward_norm_jax touches
  `env_wrapper.py` + `eval.py` + `normalize_env` config).
- **Un-apply is not yet supported.** Reverse order can be done manually:
  `git diff` against pre-trick state shows the exact edits to revert. A flag
  flip alone (`use_obs_rms: false`) disables the trick at runtime even with
  the implementation still wired in.

## Examples

```text
# Apply obs RMS to PPO only
/harbor:rl-add-trick obs_rms_jax algorithm=ppo

# Apply reward norm to all supported algorithms
/harbor:rl-add-trick reward_norm_jax

# Preview the change without writing
/harbor:rl-add-trick obs_rms_jax algorithm=sac --dry-run
```
