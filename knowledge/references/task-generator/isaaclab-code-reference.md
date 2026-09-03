# IsaacLab code reference (canonical)

Loaded by `task-generator` on demand. Concrete API patterns the smoke contracts and authoring rely on. If a non-IsaacLab family is in play, substitute the family-equivalent surface (`task-implementation.md` documents what to substitute).

## Boot + instantiate

```python
import argparse
from isaaclab.app import AppLauncher
parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args([])
args.headless = True
args.enable_cameras = False
sim_app = AppLauncher(args).app

import gymnasium as gym
import isaaclab_tasks  # noqa: F401  — registers all tasks
from isaaclab_tasks.utils import parse_env_cfg

cfg = parse_env_cfg("<TaskID>", device="cuda:0", num_envs=1, use_fabric=True)
env = gym.make("<TaskID>", cfg=cfg)
unw = env.unwrapped                              # ManagerBasedRLEnv
```

Closing: `env.close(); sim_app.close()`.

**Shutdown hang trap**: with Isaac Sim 5.1 + IsaacLab, `sim_app.close()` (called via the omniverse atexit hooks) HANGS indefinitely on USD stage detach. Symptom: the script writes its outputs, prints its success line, then never returns to the shell. Anywhere the plugin runs a script-and-wait against IsaacLab (smoke, render, eval), the workaround is one of:

1. **Hard-exit after the artifact is on disk** — `os.sync(); os._exit(0)` immediately after writing the MP4 / final assertion print. Bypasses atexit hooks. The plugin's `render.py.template` already does this. Use the same pattern in any new IsaacLab-targeted script that writes an artifact and exits.
2. **External watchdog** — orchestrator polls the script's log for the success marker, then `pkill -9` the python process. Less clean but works when the script source can't be modified.

The smoke templates (S1..S7) all run for ≤30 seconds and exit before reaching shutdown, so this trap doesn't bite them in practice. Long-running scripts (render, eval policy rollout) DO hit it.

**Fabric + render-buffer trap**: in some IsaacLab forks the cloner errors out with `"Failed to clone in Fabric"` when `use_fabric=True`, leading to a workaround of `parse_env_cfg(..., use_fabric=False)`. This SILENTLY breaks the offscreen render pipeline — the GPU render buffer is fed by Fabric, so with `use_fabric=False` `env.render()` returns the SAME stale frame on every call regardless of the actual articulation state. Symptom: pose data printed each step shows the env progressing, but the recorded MP4 shows a frozen scene.

**Always pass `use_fabric=True` for any path that calls `env.render()`** — smoke templates (S1..S7), render scripts, eval-policy rollouts. If the cloner error reappears, fix it at the asset / sim-cfg level (e.g. raise `physx.gpu_collision_stack_size`) rather than disabling Fabric. The plugin's smoke templates and the live `harbor/scripts/rl/<impl>/env_wrapper.py:create_render_env` already enforce this; the helper `IsaacLab/scripts/_isaaclab_env.py` (benchmark-specific, NOT a plugin file) historically hardcoded `use_fabric=False` and has been observed to break rendering — the render path bypasses it for that reason.

## Action manager

```python
term_names = list(unw.action_manager.active_terms)              # ordered names
term       = unw.action_manager._terms[term_names[0]]
target     = term.processed_actions                             # post-scaling, post-offset
raw        = term.raw_actions                                   # what you fed in
scale      = float(term.cfg.scale) if hasattr(term.cfg, "scale") else 1.0
```

`processed_actions` semantics by ActionTerm class:

| Class | `processed_actions` is |
|---|---|
| `JointPositionActionCfg` (`use_default_offset=True`) | `raw * scale + default_joint_pos` |
| `JointPositionActionCfg` (`use_default_offset=False`) | `raw * scale` |
| `JointPositionActionCfg` (`use_relative_mode=True`) | `current_joint_pos + raw * scale` |
| `DifferentialInverseKinematicsActionCfg` (abs) | absolute EE pose target |
| `DifferentialInverseKinematicsActionCfg` (rel) | SE(3) compose(initial_ee, raw * scale) |
| `BinaryJointPositionActionCfg` | `open_command_expr` if `raw > 0` else `close_command_expr` |
| `EMACumulativeRelativeJointPositionActionCfg` (custom — see below) | `clamp(EMA(q_init + Σ_{k<=t} (s·a_k + o), prev_applied), [lo, hi])` |
| `EMACumulativeDeltaPoseActionCfg` (custom — see below) | task-space analog: 6-D delta input → cumulative + anchor-on-init-pose + EMA, absolute pose target into IK |

When in doubt, read the term's `process_actions` source under `isaaclab.envs.mdp.actions`.

## Custom action term: EMA cumulative-delta joint position

A delta-from-init joint position controller with cumulative-action accumulation, EMA smoothing, and optional per-joint clamping. Useful when the policy's output is best interpreted as a small per-step velocity-like increment rather than an absolute target — common for dexterous-manipulation policies.

Per-step processing (`q_init` captured at `env.reset()`, `delta` accumulates over the episode, `α` is the EMA weight):

```
delta_t   = delta_{t-1} + s · a_t + o
base_t    = q_init + delta_t
target_t  = α · base_t + (1 - α) · target_{t-1}            # EMA against last applied target
target_t  = clamp(target_t, lower, upper)                  # if limits set
```

At t=0 (post-reset, with `target_{-1} == q_init`, `delta_{-1} == 0`):

```
target_0 = q_init + α · (s · a_0 + o)                      # before clamp
```

State carried across steps: `delta`, `prev_applied`. On `env.reset()`: `delta := 0`, `prev_applied := q_init := current_joint_pos`.

**Cfg shape** (`knowledge/templates/task-generator/action_terms/ema_delta_joint_pos_cfg.py.template`):

```python
EMACumulativeRelativeJointPositionActionCfg(
    asset_name="robot",
    joint_names=["panda_joint.*"],
    scale=0.2,                                # plugin default — override only if needed
    offset=0.0,
    use_default_offset=False,                 # offset is part of `o`; default-offset would double-count
    alpha=0.5,                                # plugin default — half-and-half EMA blend
    joint_lower_limit=[...],                  # optional, list[float], length == action_dim
    joint_upper_limit=[...],                  # optional, list[float], length == action_dim
)
```

**Wiring into a new task**: render both templates into the task's `mdp/` tree:
- `knowledge/templates/task-generator/action_terms/ema_delta_joint_pos.py.template` → `<task>/mdp/actions.py`
- `knowledge/templates/task-generator/action_terms/ema_delta_joint_pos_cfg.py.template` → `<task>/mdp/actions_cfg.py`

Update `<task>/mdp/__init__.py` to re-export the cfg, and import it in `<task>_env_cfg.py`'s `ActionsCfg`. The reference smoke for this mode is `ema_delta_joint_pos` in `smoke_s2.py.template`.

## Custom action term: EMA cumulative-delta EE pose

The task-space (Cartesian) analog of the joint-pos variant. Useful when the policy's output is best interpreted as a small per-step Cartesian increment relative to the EE pose at episode start. Subclasses `DifferentialInverseKinematicsAction` and forces the IK controller into `use_relative_mode=False` internally — the EMA class supplies the absolute pose target itself each step.

Per-step (`q_init` = EE pose captured at `env.reset()`, `s` = 6-D scale, `α` = EMA weight; raw action `a_t = (dx, dy, dz, drx, dry, drz)` is 6-D):

```
delta_t       = delta_{t-1} + s · a_t                             # cumulative 6-D delta
abs_pos_t     = q_init.pos  + delta_t[:3]
abs_quat_t    = q_init.quat ⊗ axis_angle_to_quat(delta_t[3:6])
abs_pose_t    = (abs_pos_t, abs_quat_t)                            # 7-D
target_t      = α · abs_pose_t + (1 - α) · target_{t-1}            # linear blend on 7-D
target_t.quat /= ‖target_t.quat‖                                   # re-normalize
target_t.pos  = clamp(target_t.pos, [lo, hi])                      # if limits set
```

At t=0 (post-reset, with `target_{-1} == q_init`, `delta_{-1} == 0`):

```
target_0.pos  = q_init.pos + α · (s[:3] · a_0[:3])                 # before clamp
target_0.quat = (q_init.quat ⊗ aaq(α · s[3:6] · a_0[3:6])) re-norm  # via the linear blend
```

Position is straightforward to assert; orientation is harder due to quaternion arithmetic — the smoke checks position only, mirroring the standard `delta_ee_pose` smoke convention.

State carried across steps: `delta` (6-D), `prev_applied_pose` (7-D). On `env.reset()`: `delta := 0`, `prev_applied_pose := q_init`.

**Cfg shape** (`knowledge/templates/task-generator/action_terms/ema_delta_ee_pose_cfg.py.template`):

```python
EMACumulativeDeltaPoseActionCfg(
    asset_name="robot",
    joint_names=["panda_joint.*"],
    body_name="panda_hand",
    controller=DifferentialIKControllerCfg(command_type="pose",
                                           use_relative_mode=False,   # MUST be False
                                           ik_method="dls"),
    scale=(0.02, 0.02, 0.02, 0.02, 0.02, 0.02),  # plugin default — 3 pos + 3 axis-angle rot,
                                                  #   EE-pose deltas are small (1 cm/step is already large in Cartesian)
    alpha=0.5,                                    # plugin default — half-and-half EMA blend
    pos_lower_limit=[0.2, -0.5, 0.0],             # optional, length 3
    pos_upper_limit=[0.8,  0.5, 0.6],
)
```

**Wiring into a new task**:
- `knowledge/templates/task-generator/action_terms/ema_delta_ee_pose.py.template` → `<task>/mdp/actions.py` (add to the existing actions module if both EMA variants are wanted)
- `knowledge/templates/task-generator/action_terms/ema_delta_ee_pose_cfg.py.template` → `<task>/mdp/actions_cfg.py`

Update `<task>/mdp/__init__.py` to re-export the cfg, and import it in `<task>_env_cfg.py`'s `ActionsCfg`. **Robot articulation must use the high-PD variant** (e.g. `FRANKA_PANDA_HIGH_PD_CFG`) for stable IK tracking. The reference smoke for this mode is `ema_delta_ee_pose` in `smoke_s2.py.template`.

## Scene state

```python
robot   = unw.scene["robot"]                                    # Articulation
object_ = unw.scene["object"]                                   # RigidObject

robot.data.joint_pos                                            # (N, n_joints)
robot.data.joint_vel
robot.data.default_joint_pos
robot.data.body_pos_w                                           # (N, n_bodies, 3)
robot.data.body_pose_w                                          # (N, n_bodies, 7) pos+quat
robot.data.body_names                                           # ordered list

object_.data.root_pos_w                                         # (N, 3)
object_.data.root_quat_w                                        # (N, 4)
object_.data.root_lin_vel_w
```

Force-write for termination tests (S4):
```python
import torch
new_pose = torch.tensor([[0.5, 0.0, -0.5, 1.0, 0.0, 0.0, 0.0]], device=unw.device)
object_.write_root_pose_to_sim(new_pose)                        # below floor
```

## Observation manager

```python
unw.observation_manager.active_terms                            # {group: [term_name, ...]} ordered
unw.observation_manager.group_obs_term_dim                      # {group: [shape_per_term, ...]}
unw.observation_manager.group_obs_dim                           # {group: total_dim}
```

Concatenated obs slice order = `active_terms[group]` order. First term occupies `obs[..., 0:dim_0]`, second `obs[..., dim_0:dim_0+dim_1]`, etc.

### Frame convention (plugin default)

Observation positions and EE poses default to the **robot root frame**, not the world frame. This keeps the policy invariant under base translation / rotation and matches how IsaacLab's tracking-style commands (e.g. `UniformPoseCommandCfg` with `body_name="panda_hand"`) already express targets in the robot root frame.

The pattern for a custom obs term that returns a position in robot-root frame:

```python
def cup_position_in_robot_root_frame(env, asset_cfg=SceneEntityCfg("cup_0"),
                                     robot_cfg=SceneEntityCfg("robot")) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    robot = env.scene[robot_cfg.name]
    pos_w  = asset.data.root_pos_w                     # (N, 3) world frame
    root_w = robot.data.root_pos_w                     # (N, 3) robot base in world
    return pos_w - root_w                              # (N, 3) robot root frame
```

For full pose (position + orientation) in robot root frame, use `subtract_frame_transforms` from `isaaclab.utils.math`. For EE pose, the cleanest path is the chain `body_pose_w` minus root pose, with quaternion composition.

Override the default only when the user explicitly asks for world frame, or when the task is a locomotion task where the robot's base is the moving entity (then world frame is the reference).

## Termination / Command managers

```python
unw.termination_manager.active_terms                            # ordered
unw.termination_manager._term_dones                             # {name: bool tensor} after step()

unw.command_manager.active_terms                                # ordered
unw.command_manager.get_command("object_pose")                  # (N, dim) — current goal
```

## Forcing known reset / goal values

Cleanest path = override the relevant cfg term's range to a point interval before `gym.make`:

```python
cfg.events.reset_object_position.params["pose_range"] = {"x":(K,K), "y":(K,K), "z":(K,K)}
cfg.commands.object_pose.ranges.pos_x                  = (G, G)
```

After `env.reset(seed=0)`, the value will be ≈ K / G exactly. This is also the contract `task-generator` uses when it authors §3 in DR-aware-but-noop mode (see authoring rules in the agent doc).
