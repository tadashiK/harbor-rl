# IsaacLab reward composition reference

Loaded by `reward-tuning-agent` on demand: per-family rules for composing the §6 reward. If a non-IsaacLab family is in play, `task-implementation.md` documents the family-equivalent surface.

## Composer-by-family (default)

| Family | Composer | per_term_logging |
|---|---|---|
| `isaaclab-manager-based` | sum (`RewardManager` sums weighted terms) | `yes` (auto-decompose via `info["detailed_reward"]` once `_DetailedRewardWrapper` is in place) |
| `isaaclab-direct` | sum (you write the body; default to summed named tensors) | `total only` (Direct envs have no RewardManager; `/harbor:reward-add-log` falls through to passthrough) |
| `dexteroushands` | sum | `total only` |
| `loco-mujoco` | sum (selected by `reward_type` env_param) | `yes` if the upstream type exposes terms, else `total only` |
| `dm_control` | whatever your body computes | depends on the env wrapper |
| `gymnasium-generic` | whatever your body computes | depends on the env wrapper |

## isaaclab-manager-based authoring

```python
# <task>/mdp/rewards.py
import torch
from isaaclab.assets import RigidObject
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg

def block_to_goal_distance(env: ManagerBasedRLEnv,
                           command_name: str,
                           std: float,
                           object_cfg: SceneEntityCfg = SceneEntityCfg("object")) -> torch.Tensor:
    obj: RigidObject = env.scene[object_cfg.name]
    goal = env.command_manager.get_command(command_name)[..., :3]
    d    = torch.norm(obj.data.root_pos_w[..., :3] - goal, dim=-1)
    return torch.exp(-d / std)
```

```python
# <task>/<task>_env_cfg.py
from isaaclab.envs.mdp import RewTerm, action_rate_l2, joint_vel_l2
from isaaclab.utils import configclass
from .mdp import block_to_goal_distance

@configclass
class RewardsCfg:
    block_to_goal_tracking = RewTerm(
        func=block_to_goal_distance,
        weight=16.0,
        params={"command_name": "object_pose", "std": 0.3},
    )
    action_rate = RewTerm(func=action_rate_l2, weight=-1e-4)
    joint_vel   = RewTerm(func=joint_vel_l2,   weight=-1e-4)
```

Common `mdp.*` building blocks:
- `mdp.action_rate_l2`, `mdp.action_l2` — action regularizers
- `mdp.joint_vel_l2`, `mdp.joint_acc_l2`, `mdp.joint_torques_l2` — actuation regularizers
- `mdp.is_alive`, `mdp.is_terminated` — survival / failure
- `mdp.position_command_error`, `mdp.orientation_command_error` — command tracking (ManagerBasedRLEnv with CommandsCfg)
- `mdp.flat_orientation_l2` — keep an asset upright
- `mdp.lin_vel_z_l2`, `mdp.ang_vel_xy_l2` — locomotion

## Term-weight conventions

- Shaping terms: weight = 1.0
- Strong attractors (Lift / Reach style coarse + fine pair): weight 5..20 for the coarse, 5 for the fine-grained
- Regularizers: |weight| ≤ 0.1, negative for penalties
- Zero-weight diagnostic indicators (e.g. binary success flag) are fine — they show up in `info["detailed_reward"]` for logging without distorting optimization

Weights are **nominal per-step magnitudes, applied directly** — a `+200` one-shot latch reads `200`. Declare them at the value each term should pay per step; `info["detailed_reward"]` / metrics show exactly that.

## Sign convention

Positive = good. Penalties are negative weights, not negative reward functions.

## Numerical safety

Reward must be finite at every step. Common pitfalls:
- Division by zero in `1 / d` (clamp `d + 1e-3`)
- `log(0)` (clamp arg `>= 1e-8`)
- `exp(very_large)` overflow (use `torch.exp(-d / std)` with `std > 0` rather than `1/d`)

## isaaclab-direct authoring

`_get_rewards()` returns a 1D tensor `(num_envs,)`. Compose by summing named term tensors:

```python
def _get_rewards(self) -> torch.Tensor:
    self.r_track = torch.exp(-self._goal_distance / 0.3) * 16.0
    self.r_action = -1e-4 * torch.sum(self.actions ** 2, dim=-1)
    return self.r_track + self.r_action
```

`/harbor:reward-add-log` cannot decompose this without the named-tensor side effect, so per_term_logging is `total only` here.

## info["detailed_reward"] shape

When `_DetailedRewardWrapper` is in place (manager-based gets it for free):

```python
info["detailed_reward"] = {
    "block_to_goal_tracking": tensor (num_envs,),
    "action_rate":            tensor (num_envs,),
    "joint_vel":              tensor (num_envs,),
    ...
}
# composer assertion: sum(info["detailed_reward"].values()) == reward (per env, atol=1e-5)
```
