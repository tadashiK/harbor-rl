---
description: Canonical metric-key contract for ALL rl-integration-generator algorithm implementations (custom_torch PPO/SAC/TD3, stable_baseline3, local_implementation). Defines the keys each algorithm MUST emit to metrics.jsonl + TensorBoard + W&B, plus the per-reward-term tracker pattern. Use when the user types /harbor:rl-add-log, or when writing/editing any algorithm's update_net()/training-loop log_info dict, wiring SB3 callbacks, patching algo files during Diagnose+Retry, reviewing training scripts for missing metrics, or adding a new algorithm under harbor/scripts/rl/<impl>/.
---

# /harbor:rl-add-log — RL Metrics Logging Contract

Canonical metric-key schema that **every** RL training implementation under `harbor/scripts/rl/<impl>/` (custom_torch, stable_baseline3, local_implementation) MUST emit to:

1. `outputs/<run>/metrics.jsonl` — one JSON record per (step, key, value) triple
2. `outputs/<run>/tb/events.out.tfevents.*` — TensorBoard event files
3. W&B (when `wandb=<project>` is set) — same keys, online stream

The keys are part of the **plug-compatibility contract**: T4 (plot smoke) writes one PNG per metric key under `outputs/<run>/curves/`, and the rl-tuning-agent reads the same keys for hyperparameter analysis. New algorithm implementations must conform.

## Canonical key schema

| Category | Key | Algorithms | Source |
|---|---|---|---|
| **Reward (always)**           | `reward/total/episodic_return_mean`           | all | accumulated env reward over episode, averaged across recent N episodes |
| **Reward (per-term, if env emits `info["detailed_reward"]`)** | `reward/<term>/episodic_return_mean` | all | per-term cumulative episodic mean — auto-discovered from the `info` dict |
| **PPO**                       | `train/loss/policy`                           | PPO | mean PPO clipped policy loss across minibatches |
|                               | `train/loss/value`                            | PPO | mean MSE value loss |
|                               | `train/entropy`                               | PPO | mean policy entropy (positive number) |
|                               | `train/approx_kl`                             | PPO | Schulman 2020 unbiased KL estimator: `((ratio - 1) - log_ratio).mean()` |
|                               | `train/clip_fraction`                         | PPO | fraction of samples whose ratio was clipped |
| **SAC**                       | `train/actor_loss`                            | SAC | mean actor loss across update iterations |
|                               | `train/critic_loss`                           | SAC | mean (Q1-MSE + Q2-MSE) loss |
|                               | `train/entropy`                               | SAC | mean entropy estimate `-log_prob.mean()` |
|                               | `train/q_value`                               | SAC | mean of `min(Q1, Q2)` on the sampled batch |
|                               | `train/alpha`                                 | SAC | current entropy temperature (diagnostic, optional) |
| **TD3**                       | `train/actor_loss`                            | TD3 | mean actor loss (across non-skipped policy updates) |
|                               | `train/critic_loss`                           | TD3 | mean critic loss |
|                               | `train/q_value`                               | TD3 | mean of `min(Q1, Q2)` on the sampled batch |
| **Common (all algorithms)**   | `train/episode_length`                        | all | recent N-episode mean episode length |

**Naming rules** — non-negotiable:
- Use `/` to namespace (`train/loss/policy`, NOT `train.loss.policy` or `train_loss_policy`).
- `reward/<term>/episodic_return_mean` — the trailing `episodic_return_mean` makes it clear what's averaged (per-episode cumulative).
- PPO loss keys live under `train/loss/<head>` (policy / value); for SAC/TD3 they're flat (`train/actor_loss`, `train/critic_loss`) — established literature conventions, do not unify.
- `train/entropy` is the **positive** entropy. Some libraries (SB3) emit `train/entropy_loss = -entropy`; sign-flip when remapping.
- `train/q_value` is the mean of `min(Q1, Q2)` on whatever batch the update touched (typical SAC/TD3 diagnostic).

## Implementation checklist

When you author or patch an algorithm file under `harbor/scripts/rl/<impl>/algo/<name>.py` (custom_torch), write the SB3 callback (stable_baseline3), or define a local_implementation shim's expected entry, ensure:

- [ ] `update_net(...)` (or equivalent) returns a dict whose keys are exactly the schema above.
- [ ] `ac_base.update_tracker()` lazily inits `self._term_returns` + `self._term_trackers` from `info["detailed_reward"]` keys (if that field exists).
- [ ] `ac_base.reward_log_info()` always returns `{"reward/total/episodic_return_mean": ...}` plus per-term keys.
- [ ] Each algorithm's log_info dict updates with `**self.reward_log_info()` so reward keys propagate.
- [ ] `train.py`'s metric-record helper writes to `metrics.jsonl` AND calls `dl.record(key, value, step)` (DataLogger handles TB+W&B).
- [ ] `train.py` calls `_plot_curves(metrics_log, log_dir / "curves", ...)` in the `finally` block (one PNG per key).

## Reference implementations

The custom_torch tree at `knowledge/templates/rl-integration-generator/custom_torch/algo/{ac_base,ppo,sac,td3}.py.template` is the **canonical** implementation. New algorithm authors can either:

1. **Subclass `ActorCriticBase`** and call `**self.reward_log_info()` in their log_info return — gets reward keys for free.
2. **Implement from scratch** and emit the keys directly. Verify with the smoke check below.

### Snippet 1 — ac_base helper (per-reward-term tracking)

In `update_tracker(self, reward, done, info)`, after the normal return/length tracking:

```python
# Per-reward-term tracking — only when the env emits info["detailed_reward"].
# Lazy-init: discover term keys on first non-empty observation.
if isinstance(info, dict) and "detailed_reward" in info:
    detailed = info["detailed_reward"]
    if isinstance(detailed, dict) and detailed:
        if not hasattr(self, "_term_returns") or self._term_returns is None:
            n = int(self.cfg.num_envs)
            tlen = int(self.cfg.algo.get("tracker_len", 100))
            self._term_returns = {
                k: torch.zeros(n, dtype=torch.float32, device=self.device)
                for k in detailed.keys()
            }
            self._term_trackers = {k: Tracker(tlen) for k in detailed.keys()}
        for k, v in detailed.items():
            if k not in self._term_returns:
                n = int(self.cfg.num_envs)
                tlen = int(self.cfg.algo.get("tracker_len", 100))
                self._term_returns[k] = torch.zeros(n, dtype=torch.float32, device=self.device)
                self._term_trackers[k] = Tracker(tlen)
            if not isinstance(v, torch.Tensor):
                v = torch.as_tensor(v, device=self.device, dtype=torch.float32)
            self._term_returns[k] += v.float()
            if len(done_idx) > 0:
                self._term_trackers[k].update(self._term_returns[k][done_idx])
                self._term_returns[k][done_idx] = 0
```

And the public helper:

```python
def reward_log_info(self) -> dict:
    """Canonical reward metrics: total + per-term cumulative episodic-return mean."""
    out = {"reward/total/episodic_return_mean": float(self.return_tracker.mean())}
    if hasattr(self, "_term_trackers") and self._term_trackers:
        for k, t in self._term_trackers.items():
            out[f"reward/{k}/episodic_return_mean"] = float(t.mean())
    return out
```

### Snippet 2 — PPO `update_net` returns dict

After the minibatch loop accumulates `policy_losses, value_losses, entropies, kls, clip_fracs`:

```python
# Inside the minibatch loop, after the optimizer step:
with torch.no_grad():
    # Schulman 2020 unbiased low-variance KL estimator: ((ratio - 1) - log_ratio).
    approx_kl = ((ratio - 1.0) - log_ratio).mean().item()
    clip_frac = ((ratio - 1.0).abs() > clip).float().mean().item()
kls.append(approx_kl)
clip_fracs.append(clip_frac)

# After the loop:
log_info = {
    "train/loss/policy":   float(np.mean(policy_losses)),
    "train/loss/value":    float(np.mean(value_losses)),
    "train/entropy":       float(np.mean(entropies)),
    "train/approx_kl":     float(np.mean(kls)),
    "train/clip_fraction": float(np.mean(clip_fracs)),
    "train/episode_length": float(self.step_tracker.mean()),
}
log_info.update(self.reward_log_info())
return log_info
```

### Snippet 3 — SAC `update_net` returns dict

`update_critic` and `update_actor` need to return q-mean / entropy-estimate as a third value:

```python
# update_critic(...):
with torch.no_grad():
    q_mean = float(torch.min(q1, q2).mean().item())
return critic_loss.item(), gn, q_mean

# update_actor(...):
with torch.no_grad():
    ent_est = float(-log_prob.mean().item())   # tanh-Gaussian entropy ≈ -log_prob
# ... existing alpha loss ...
return actor_loss.item(), gn, ent_est

# update_net(...):
log_info = {
    "train/critic_loss": float(np.mean(critic_losses)),
    "train/actor_loss":  float(np.mean(actor_losses)),
    "train/entropy":     float(np.mean(entropies)),
    "train/q_value":     float(np.mean(q_values)),
    "train/episode_length": float(self.step_tracker.mean()),
    "train/alpha":       self.get_alpha(scalar=True),  # diagnostic, optional
}
log_info.update(self.reward_log_info())
return log_info
```

### Snippet 4 — TD3 `update_net` returns dict

`_update_critic` returns `(loss, q_mean)`:

```python
# _update_critic(...):
with torch.no_grad():
    q_mean = float(torch.min(q1, q2).mean().item())
return critic_loss.item(), q_mean

# update_net(...):
log_info = {
    "train/critic_loss": float(np.mean(critic_losses)),
    "train/actor_loss":  float(np.mean(actor_losses)) if actor_losses else 0.0,
    "train/q_value":     float(np.mean(q_values)),
    "train/episode_length": float(self.step_tracker.mean()),
}
log_info.update(self.reward_log_info())
return log_info
```

### Snippet 5 — SB3 callback name remap

SB3 emits its own keys; remap to canonical schema in the `_DataLoggerCallback`:

```python
SB3_NAME_REMAP: dict[str, tuple[str, bool]] = {
    "train/policy_loss":          ("train/loss/policy",   False),
    "train/policy_gradient_loss": ("train/loss/policy",   False),  # SB3 PPO actually emits this name
    "train/value_loss":           ("train/loss/value",    False),
    "train/entropy_loss":         ("train/entropy",       True),   # sign-flip
    "train/approx_kl":            ("train/approx_kl",     False),
    "train/clip_fraction":        ("train/clip_fraction", False),
    "train/actor_loss":           ("train/actor_loss",    False),
    "train/critic_loss":          ("train/critic_loss",   False),
    # Rollout aggregates (per the canonical metrics schema).
    "rollout/ep_rew_mean":        ("reward/total/episodic_return_mean", False),
    "rollout/ep_len_mean":        ("train/episode_length",              False),
}

def _on_rollout_end(self) -> None:
    n2v = getattr(getattr(self.model, "logger", None), "name_to_value", None)
    if not n2v:
        return
    step = int(self.num_timesteps)
    for raw_k, v in dict(n2v).items():
        try:
            f = float(v)
        except (TypeError, ValueError):
            continue
        mapped, flip = SB3_NAME_REMAP.get(raw_k, (raw_k, False))
        if flip:
            f = -f
        self._record(mapped, f, step)
    self._maybe_record_q_value(step)   # SAC/TD3 only — sample replay buffer
```

For SAC/TD3 q_value (SB3 doesn't emit this internally), sample 64 transitions from the replay buffer and compute `min(Q1, Q2).mean()` once per rollout. Silent if the model isn't off-policy:

```python
def _maybe_record_q_value(self, step: int) -> None:
    rb = getattr(self.model, "replay_buffer", None)
    if rb is None or rb.size() < 64:
        return
    try:
        data = rb.sample(64, env=getattr(self.model, "_vec_normalize_env", None))
        with torch.no_grad():
            q1, q2 = self.model.critic(data.observations, data.actions)
            q_mean = float(torch.min(q1, q2).mean().item())
        self._record("train/q_value", q_mean, step)
    except Exception:
        return
```

## Verification (post-render smoke)

After `rl-integration-generator` renders a new algorithm tree (or a Diagnose+Retry patches an algo file), run T1 train smoke and grep `metrics.jsonl` for the required keys:

```bash
TRIAL=$(ls -1td outputs/<algo>_*/ | head -1)
REQUIRED_PPO="reward/total/episodic_return_mean train/loss/policy train/loss/value train/entropy train/approx_kl train/clip_fraction train/episode_length"
REQUIRED_SAC="reward/total/episodic_return_mean train/actor_loss train/critic_loss train/entropy train/q_value train/episode_length"
REQUIRED_TD3="reward/total/episodic_return_mean train/actor_loss train/critic_loss train/q_value train/episode_length"

ACTUAL=$(cat ${TRIAL}/metrics.jsonl | python3 -c "import json,sys; print('\n'.join(sorted({json.loads(l)['key'] for l in sys.stdin})))")
for k in ${REQUIRED_PPO}; do
    if echo "${ACTUAL}" | grep -q "^${k}\$"; then
        echo "OK: ${k}"
    else
        echo "MISSING: ${k}"
    fi
done
```

If any required key is missing → enter Diagnose+Retry per the rl-integration-generator agent's Phase 4 protocol. Most common cause: the algorithm's `update_net` return dict was authored before this contract existed; patch it to match Snippet 2/3/4 above.

## When this contract changes

If a new algorithm needs a key not in this schema, add it here FIRST (with rationale), then update the templates and the rl-tuning-agent's metric reader. Don't introduce ad-hoc keys in algorithm code without updating this file — the smoke verification above and the tuner's analysis both depend on the schema being authoritative.

## Cross-references

- `knowledge/templates/rl-integration-generator/custom_torch/algo/*.py.template` — reference implementations
- `knowledge/templates/rl-integration-generator/stable_baseline3/scripts/train.py.template` — SB3 callback with remap
- `agents/rl-integration-generator.md` Phase 4 — smoke that exercises the contract
- `knowledge/references/rl-integration-generator/rl-suite-spec.md` — broader spec for `harbor/rl-integration-generator/rl-suite-spec.json`
