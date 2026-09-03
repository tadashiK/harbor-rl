# §6 smoke contract

Loaded by `reward-candidate-agent`. Template: `${CLAUDE_PLUGIN_ROOT}/knowledge/templates/reward-tuning-agent/smokes/smoke_s6.py.template`.

S6 is the structural gate a candidate reward passes before it is worth a training run. It proves the reward is **well-formed**, never that it trains the behavior — that is what the training run decides.

## What it verifies

30 random-action steps at `num_envs=128`, with the env built through `<repo>/scripts/_isaaclab_env.py:make_isaaclab_env`:

1. **Same construction path as training** — the factory is the only path that wires `info["detailed_reward"]`. A smoke that calls `gym.make` directly reports `passthrough` and proves nothing about the decomposition the tune scores on, so it must not. A reward term keying on a signal the task lacks fails here, before a training run is spent on it.
2. **Finite** — no `NaN` / `Inf` reward in any env at any step.
3. **Non-constant** — `std > 0` over the 30 steps, measured on the **mean across envs at each step** (a temporal property, not cross-env variance). Catches all-zero weights and terms that return constants.
4. **Composer match** — for `reward_composer == "sum"`, `np.allclose(sum(detailed.values()), reward, atol=1e-5)` element-wise across all 128 envs at every step. Catches per-term wiring that has silently divorced itself from the optimization signal. Skipped for `reward_composer == "diagnostic"` (Factory / Forge / Industreal expose pre-scale means that are not meant to compose).
5. **Decomposition present** — `detailed_reward` carrying only `total` is a **FAIL**, not a warning: per-term logging is wired before iter 0 and the loop scores per-term curves, so passthrough means broken wiring.

## Pass criterion

Script exits 0, no `Traceback` / `AssertionError`, final stdout line reads `S6 OK: ...`.

## Substitutions

| Slot | Notes |
|---|---|
| `{{TASK_ID}}` | the candidate's own gym id — its slot clone when the pool runs more than one candidate, the source task when it runs sequentially |
| `{{REPO}}` | absolute path to the benchmark repo; locates the env factory |

## When the contract fails

| Symptom | Likely cause | First-pass fix |
|---|---|---|
| `NaN` / `Inf` | divide-by-zero, `log(0)`, `exp` overflow | clamp the offending operand |
| `reward is constant` | weights all zero, or terms returned constants | re-check weights against the base task; verify the state the term reads is actually updating |
| `composer mismatch` | per-term values don't sum to total | wrong composer (sum vs product), or a term in the env not surfaced in `detailed_reward` |
| `only 'total'` | env not built through the instrumented factory | re-run the `/harbor:reward-add-log` flow; confirm `scripts/_isaaclab_env.py` exists and delegates |
| Build error | `mdp.<func>` symbol missing in this IsaacLab version | drop to a known function from the base task |

Fixes touch the **implementation** (idiom, imports, a missing cross-section field) — never the weights. Reweighting to pass a structural smoke hides the defect the smoke found.
