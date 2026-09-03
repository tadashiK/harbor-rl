# rl-tuning-agent — Cross-run Tuning Experience

Append-only ledger of high-level heuristics that have generalized across tunes — no run-specific examples; keep each bullet short. Read at Phase 0 of every new tune. Promote new lessons here from the per-tune `tuning-history.md` summaries when they're likely to repeat. **Each entry is numbered for stable cross-reference; never renumber existing entries — only append.**

## Hyperparameter heuristics

1. **Larger `batch_size` → more stable gradients.** Doubling typically dampens train-loss noise by ~30% with negligible final-return cost. Try this first when curves look ragged.
2. **More `updates_per_iter` (PPO `n_epochs`, SAC/TD3 `gradient_steps`) → better sample efficiency.** Doubling roughly halves env-steps-to-threshold but adds ~1.5× wall-clock per outer iter. Worth it when GPU is the bottleneck, not env stepping.
3. **Halving `num_envs`** improves sample efficiency at the cost of wall-clock; doubling does the reverse. Match to the bottleneck (env vs GPU). Constraint #1 caps the search to {0.5×, 1×, 2×}.
4. **`learning_rate` swings of 0.5× / 2×** are usually enough — finer steps rarely change the ranking.
5. **`gamma = 0.99`** is correct for almost all locomotion tasks. Don't tune it without a specific reason (e.g., very long-horizon tasks).

## Algorithm-specific

### PPO

6. `entropy_coef` collapsing → exploration died. Bump 2× if entropy < 0.01 within first 10% of training.
7. `clip_eps = 0.2` is the right default. 0.1 is for fine-tuning, 0.3 for fast early progress.
8. `value_clip_torch` trick is a cheap stabilizer when running >5 epochs per iteration.
9. `gae_lambda = 0.95` is robust; lowering biases short-horizon advantage, rarely helps.

### SAC

10. Auto-entropy (`learn_alpha = true`) is robust; only fix alpha when comparing seeds at the end.
11. `tau = 0.005` is the default. Higher (0.02) helps wall-clock but may overshoot — watch for value-loss oscillation.
12. Replay `buffer_size ≥ 1e6` for locomotion; cutting it costs final return on long-horizon tasks.

### TD3

13. `policy_delay = 2` default; raising to 4 slows learning and rarely helps.
14. `target_policy_noise = 0.2` with `target_noise_clip = 0.5` is solid; don't tune unless overfitting is visible.
15. Exploration noise (`expl_noise`) is the most impactful knob. Try 0.1 → 0.3 if learning is slow.

## Tricks that almost always help (when applicable)

16. `obs_rms_jax` / `obs_rms_torch` — observation normalization. Free win on any task with multi-scale state vectors (joint positions + velocities + IMU).
17. `reward_norm_jax` — return-scale reward normalization. Big lift on shaped-reward locomotion tasks. Do NOT use on sparse-reward tasks — it eats the signal.
18. `value_clip_torch` — PPO clipped value loss. Lets you run more epochs / larger minibatches without value collapse.

## Failure-mode signatures

19. **NaN in actor_loss within 1k steps** — learning_rate too high OR un-clipped grads. First fix: halve LR, enable grad-norm clip.
20. **eval return ≈ 0 forever** — reward not reaching the agent (env wrapper mis-wired, `/harbor:reward-add-log` shows 0). First fix: smoke-test env_wrapper outside the algo.
21. **train_loss decreasing but eval_return flat** — overfitting rollout buffer (too few env steps per update). First fix: bump `num_envs` or shrink `n_epochs`.
22. **steps/sec drops mid-run** — leak (replay buffer growing without cap) OR JIT recompile loop. First fix: check buffer size; pin shapes; print shape on every fresh trace.
23. **value_loss explodes (PPO)** — un-normalized returns + many epochs. First fix: apply `value_clip_torch` or `value_norm_torch`.
24. **entropy collapses fast (PPO)** — `entropy_coef` too low for this reward scale. First fix: raise 2×; consider `reward_norm_jax`.

## Wall-clock priorities

25. **A 10% return improvement that triples wall-clock is a worse outcome than the baseline.** Tuners who optimize purely for return without watching wall-clock waste cluster quota and produce policies the user won't run. Always log `wall_clock_sec` alongside `final_return` in `tuning-history.md`. Constraint #3 enforces this at the candidate-acceptance stage.

## Diagnosis & evaluation discipline

26. **Verify a hyperparameter is actually consumed before tuning it.** Config keys can be dead — declared, documented, and read by nothing. Grep the trainer for the key first; a silent no-op is indistinguishable from "this knob doesn't matter."
27. **Diagnose the failing checkpoint before sweeping.** Probe what the policy physically does, and rule out actuation/reach limits with an open-loop control test. A learned stall and a kinematic limit look identical in the return curve, and only one is fixable by tuning.
28. **`ent_coef` is single-peaked — too low fails as hard as too high.** #6/#24 cover collapse; below the peak the policy instead commits early to a partial strategy and never explores the full behaviour. Distinguish by DIRECTION: a decaying gated skill with entropy FALLING = too little exploration; RISING = too much. Sweep an order of magnitude both ways before concluding the knob is wrong.
29. **A learnable-`log_std` ceiling makes high `ent_coef` fail silently.** Past some value entropy pins at the cap and the policy can only sharpen by growing the actor mean, so raising it further degrades results with no distinguishing entropy signal.
30. **Time-to-first-success is a seed property; the exploration knob controls convergence AFTER breakthrough.** Budget for the slowest seed, not the mean, and don't attribute seed variance in breakthrough time to a hyperparameter.
31. **Evaluate periodic checkpoints, not just the final one.** The last checkpoint is repeatedly not the best — once episodes terminate on success, long training drifts toward minimal-effort solutions that sit just inside the success window.
32. **Auto-reset poisons post-`step()` state reads.** Vec wrappers reset terminated envs inside `step()`, so any state read after the call belongs to the NEXT episode — read from the info dict the algo itself consumes. The same trap makes a render's last frame a reset, not a result.

## Promotion guidelines

When a per-tune `tuning-history.md` ends with a takeaway like "doubling X helped on this task", promote it here ONLY if:

- it survived multi-seed verification (or matched a known reference), AND
- the mechanism is plausible (not "random search hit a lucky seed"), AND
- it's not already covered above.

Otherwise leave it in the per-tune history and revisit after another tune confirms the pattern.
