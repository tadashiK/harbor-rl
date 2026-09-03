---
description: Add per-reward-term logging to a Harbor benchmark repo without changing the env's native reward. The wrapper exposes per-term values on info["detailed_reward"] and asserts `composer(terms) == reward` every step, where composer ∈ {"sum", "product"} matches how the native reward composes its terms. Use when the user types /harbor:reward-add-log or asks "add per-term reward logging", "decompose the reward", "show me each reward component in W&B", "add reward manager to this benchmark".
---

# /harbor:reward-add-log — Per-Term Reward Logging (Composer-Aware Invariant)

Adds a per-term reward visibility wrapper to a Harbor benchmark repo. The wrapper:

1. Computes named terms `{name_i: fn_i(physics)}` per step (no weights — raw component values).
2. Asserts `composer(terms.values()) == env_reward` every step, where `composer` is **per-task** and reflects the structure of the env's native reward function:
   - `"sum"`     — IsaacLab `RewardManager` pattern (`reward = Σ weight_i * term_i`); ManagerBasedRLEnv tasks default to this.
   - `"product"` — `dm_control` pattern (`reward = ∏ tolerance_i(...)`); cartpole / walker / humanoid all multiply tolerance terms.
3. Stashes the term breakdown on `info["detailed_reward"]` so both `custom_torch` (`ac_base.update_tracker`) and `stable_baseline3` (`_DataLoggerCallback`) emit `reward/<term>/episodic_return_mean`.
4. **Does NOT modify the env's native reward.** The contract is read-only. (Earlier drafts forced `reward = Σ weight_i * term_i` for all tasks; this silently re-shaped multiplicative tasks — the policy learned to max the easy additive terms while abandoning the hard one.)

Reference: `isaaclab.managers.reward_manager.RewardManager.compute()` — that pattern is the `composer="sum"` case here.

## When to use

- The user wants per-component reward visibility in W&B (e.g. "is upright dominating, or is small_control hurting?").
- The benchmark belongs to one of the two supported families:
  - **Path A — scalar / dm_control style.** Has `scripts/_<family>_env.py` from `benchmark-generator`; reward is a Python scalar combined as `Σ w_i * f_i` or `∏ f_i(...)`.
  - **Path B — IsaacLab.** Has `_isaac_sim/` symlink at the repo root (binary install) or its tasks are all gym IDs starting with `Isaac-`; reward is a batched torch tensor produced by `isaaclab.managers.RewardManager.compute()`.
- You can identify the env's reward structure: dm_control tasks publish term lambdas in their reward source; IsaacLab manager-based tasks expose `reward_manager._step_reward` per step (the wrapper reads this dynamically — no manual term spec needed).

## When NOT to use

- The benchmark's reward is already a sum of named terms exposed via `info` — the per-term tracker in `ac_base` / SB3 callback already picks it up; no patch needed.
- Path A repos with no `scripts/_*_env.py` — run `/harbor:env-install-uv` first (sets up the env, then dispatch benchmark-generator).
- The env's reward function isn't decomposable into named terms whose sum or product equals the reward (custom recurrent / learned reward functions, etc.) — the composer invariant won't hold; tell the user and stop.
- IsaacLab Direct envs (no `RewardManager`) where the user expects per-term curves — the Path B wrapper passes through with `info["detailed_reward"] = {"total": env_reward}`; tell the user and stop, or recommend they switch to the manager-based variant.

## Benchmark family detection

Pick one path before Step 3. Run, in order:

1. **Path B (IsaacLab) if** `<repo>/_isaac_sim` exists (symlink or directory) **OR** every `tasks[i].id` in `<repo>/harbor/benchmark-generator/benchmark-spec.json` starts with `Isaac-`.
2. **Path A (scalar)** otherwise.

If you're unsure (mixed signals: `_isaac_sim/` present but tasks aren't `Isaac-*`, or vice-versa), ask the user which path before patching.

## Workflow

Execute these steps in order. Steps 3–6 branch on the family; Steps 1, 2, 7 are shared.

### Step 1 — Locate the benchmark spec

```bash
ls "$(pwd)/harbor/benchmark-generator/benchmark-spec.json"
```

If missing, stop and tell the user to run `/harbor:env-install-uv` (or `Skill('env-install-uv')` followed by `benchmark-generator`) first.

Read all task IDs:

```bash
python3 -c "import json; d=json.load(open('harbor/benchmark-generator/benchmark-spec.json')); print('\n'.join(t['id'] for t in d['tasks']))"
```

### Step 2 — Ask which tasks to instrument (toggle list)

Use `AskUserQuestion` with `multiSelect: true`, one option per task ID from Step 1. The header is `"Tasks to instrument"`. The first option must be `"All tasks"` so the user can opt-in to the whole list with one click.

If the task list is longer than ~25 entries, group into families (`cartpole/*`, `walker/*`, `humanoid/*`, …) and ask family-level first; only drill in if the user picks a family with mixed-membership intent.

### Step 3 — Locate the env helper file

**Path A — scalar / dm_control:**

```bash
ls "$(pwd)"/scripts/_*_env.py
```

Exactly one match expected (benchmark-generator convention). If zero, stop. If more than one, ask the user which to patch.

**Path B — IsaacLab:** ⚠️ **check first — `<repo>/scripts/_isaaclab_env.py` may already exist.**
`benchmark-generator` writes its L1/L2 smoke helper at that exact path, with an incompatible
signature (`make_isaaclab_env(task, num_envs, render)`, boots Kit itself, returns a smoke
adapter) and no per-term decomposition. `run_random.py` and `render_random.py` import it.
Rendering this template over it silently breaks both smokes. If the file exists, do NOT
overwrite it — check whether per-term logging is already provided by
`harbor/scripts/rl/<slug>/env_wrapper.py` (it often is), and if a helper is still needed,
render to a distinct filename. Otherwise: the helper does not exist yet (IsaacLab tasks aren't built through `scripts/_<family>_env.py` — `harbor/scripts/rl/custom_torch/env_wrapper.py::_build_isaaclab_env` builds them inline via `gym.make` + `parse_env_cfg`). Step 5 will render `<repo>/scripts/_isaaclab_env.py` from a template and add a one-line delegation in `env_wrapper.py` so the rl-integration tree picks it up.

### Step 4 — Build the term spec / composer

**Path A:** open the env's reward source (e.g. `dm_control.suite.cartpole.Balance._get_reward`) and identify:

1. The named component functions (each returning a scalar — typically in `[0, 1]`).
2. How they combine into the final reward: **sum** (`r = Σ w_i * f_i`) or **product** (`r = ∏ f_i`).

Build a per-task spec function returning `(terms, composer)`:

```python
def _<task>_term_specs(sparse: bool) -> tuple[list[tuple[str, callable]], str]:
    return [
        ("term_a", lambda phys: f_a(phys)),
        ("term_b", lambda phys: f_b(phys)),
        ...
    ], "product"   # or "sum"
```

Register under the dispatch table `_REWARD_TERM_SPECS = {"<prefix>/": _<task>_term_specs, ...}`. The wrapper does NOT weight or override the env's reward. Weights, if any, must already be embedded inside the term `fn`s for the composer equation to hold.

**Path B:** no per-task spec needed. `RewardManager` already computes per-term values and stores the real per-step reward at `manager._step_reward[:, i]` (shape `(num_envs, num_terms)`) — read it directly. Composer is always `"sum"` (`env_reward = Σ_i func_i × weight_i`). The template at `${CLAUDE_PLUGIN_ROOT}/knowledge/templates/reward-add-log/isaaclab_env_helper.py.template` reads this dynamically — drop it in unmodified.

For Direct envs (no `reward_manager`), the template falls through to passthrough mode: `info["detailed_reward"] = {"total": env_reward}`, no decomposition. Confirm with the user before patching that they understand Direct tasks won't get per-term curves.

If you can't make `composer(terms) == env_reward` hold (Path A custom decomposition or Path B with an unusual env that overrides `compute`), the decomposition is wrong — fix the term list before patching.

### Step 5 — Render the helper

**Path A:** insert (or extend) the term-spec table and the `_DetailedRewardWrapper` class into the existing `scripts/_<family>_env.py`. Reference implementation:

```
${CLAUDE_PLUGIN_ROOT}/knowledge/templates/reward-add-log/reward_terms_block.py.template
```

Copy verbatim into the helper file just below the `_NEUTRAL_CWD` block (or equivalent imports section), add the per-task `_<family>_term_specs(sparse: bool)` function with the user-confirmed term list, and wrap `make_<family>_env(...)` to return `_DetailedRewardWrapper(base, task_id)`.

Do NOT touch `harbor/scripts/rl/<impl>/env_wrapper.py` or `harbor/scripts/rl/<impl>/train.py` — the consumer side already supports `info["detailed_reward"]` (custom_torch via `ac_base.update_tracker`, stable_baseline3 via `_DataLoggerCallback._on_step`). If those files predate the per-term tracker, run `rl-integration-generator` to refresh them.

**Path B:** two surgical edits.

1. Render `<repo>/scripts/_isaaclab_env.py` verbatim from `${CLAUDE_PLUGIN_ROOT}/knowledge/templates/reward-add-log/isaaclab_env_helper.py.template`. The file is fully usable as-is; no substitutions required.
2. Patch `<repo>/harbor/scripts/rl/custom_torch/env_wrapper.py::_build_isaaclab_env` (and the `stable_baseline3` equivalent if present) to delegate to the new factory when `<repo>/scripts/_isaaclab_env.py` exists. This is the **only** allowed touch of `env_wrapper.py` — IsaacLab has no `scripts/_<family>_env.py` convention to splice into, so the env factory is the entry point. Insert at the top of `_build_isaaclab_env`:

   ```python
   helper = REPO / "scripts" / "_isaaclab_env.py"
   if helper.is_file():
       scripts_dir = str(REPO / "scripts")
       if scripts_dir not in sys.path:
           sys.path.insert(0, scripts_dir)
       import _isaaclab_env
       return _isaaclab_env.make_isaaclab_env(
           task, num_envs=num_envs, device=device, render_mode=render_mode)
   # ... existing inline gym.make path falls through ...
   ```

   Leave `train.py` / `eval.py` / `render.py` / the `IsaacLabVecAdapter` untouched — `info["detailed_reward"]` rides through the adapter unchanged because the adapter passes the dict-typed `info` to `ac_base.update_tracker` as-is.

### Step 6 — Sanity-check smoke

**Path A:**

```bash
python3 "${CLAUDE_PLUGIN_ROOT}/scripts/reward-add-log/sanity_check.py" \
    --env-helper "$(pwd)/scripts/_<family>_env.py" \
    --tasks "<comma-separated task IDs>" \
    --steps 200
```

**Path B (IsaacLab):** the Path A script assumes scalar rewards / numpy actions and won't work for batched torch envs. Use the IsaacLab variant, which boots `AppLauncher` itself and samples `2*rand-1` torch actions on the env device:

```bash
source "$(pwd)/.venv/bin/activate"
source "$(pwd)/_isaac_sim/setup_conda_env.sh"
OMNI_KIT_ALLOW_ROOT=1 python -u \
    "${CLAUDE_PLUGIN_ROOT}/scripts/reward-add-log/sanity_check_isaaclab.py" \
    --env-helper "$(pwd)/scripts/_isaaclab_env.py" \
    --tasks      "<comma-separated Isaac-* IDs>" \
    --num-envs 4 --steps 200
```

Both scripts:
- import the env helper
- run N random-action steps per task
- read `info["reward_composer"]` (defaults to `"sum"`)
- assert `composer(terms) == env_reward` (Path A: `< 1e-6`; Path B: `torch.allclose` with `atol=1e-4 rtol=1e-3` to absorb float32 batched rounding)
- print per-task `OK: <composer> invariant holds (N/N steps)` or `FAIL` with the offending step + diff

If any task FAILs, do not report success; surface the diff and ask the user whether to (a) drop that task, (b) fix the term spec (Path A — most common: terms missing or composer wrong), or (c) accept the gap if it's pure float-rounding (Path B with extreme num_envs or non-default precision — usually means tightening tolerances or running on cuda matched to the training env).

### Step 7 — Report back

1. Path of the patched env helper.
2. Per-task term lists (name + weight) so the user can sanity-check the shaping change.
3. The W&B keys they will now see: `reward/total/episodic_return_mean` plus one `reward/<name>/episodic_return_mean` per term.
4. Sample command to verify in W&B:

```bash
python harbor/scripts/rl/stable_baseline3/train.py algo=ppo task=<picked> total_timesteps=50_000 wandb=<project>
# Open the run; total should equal the sum of the per-term curves at every step.
```

## Notes

- **The wrapper is read-only by contract.** It does not alter the env's reward, only exposes per-term values for visibility. Earlier drafts that forced `reward = Σ w_i * term_i` silently re-shaped multiplicative tasks (e.g. cartpole/swingup: the policy learned to keep the cart still and apply tiny actions, maxing centered/small_control/small_velocity at ~92% while never attempting the swingup, because under the additive sum that strategy paid more than the swingup transient). If the user truly wants reward shaping, that's a separate skill — recommend writing a `RewardManager` in their training code instead, with ablation tracking.
- **Composer is structural, not tunable.** It must match how the env's native reward function combines its components. Verify by reading the source; don't guess.
- **Multi-objective tasks** (Pareto / lex-priority weighted): the single-composer model here is the wrong tool — recommend a custom `RewardManager` and exit.
- **Sparse reward tasks**: per-term decomposition is still valuable for diagnostics (which sparse subgoal fires when), but the composer should match the env's native combiner — usually `"sum"` if the sparse and dense terms are added, `"product"` if multiplied as a gate.
- **Path B (IsaacLab) reads from `RewardManager._step_reward` rather than re-evaluating term lambdas.** This is exact (no double-eval) and works for any IsaacLab `ManagerBasedRLEnv` task without per-task code. Direct envs without a manager pass through with `info["detailed_reward"] = {"total": env_reward}` — flag this to the user up-front so they aren't surprised by the lack of per-term curves.
- **Path B touches `env_wrapper.py` (one-line delegation).** This is the documented exception to the "don't touch env_wrapper.py" rule — IsaacLab has no `scripts/_<family>_env.py` convention so the env factory is the only viable splice point. Limit the change to the helper-import branch; do not modify the `IsaacLabVecAdapter` or anything else in the file.
- **Path B name collision with the Path A discovery glob.** `_make_single_env`'s `scripts/_*_env.py` auto-detect matches `_isaaclab_env.py` and would import it pre-boot (dies on `from pxr import ...`). The rendered env_wrapper templates skip `_isaaclab_env.py` in that glob; if patching an older rendered `env_wrapper.py`, add the same `if helper.name == "_isaaclab_env.py": continue` guard — the helper is only ever consumed via the gpu_sim branch (`_build_isaaclab_env`), after Kit boots.
