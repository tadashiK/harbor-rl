# Isaac-Reach-Franka-v0 — Implementation Spec

- robot: Franka Emika Panda (7 DoF arm, no gripper action)
- simulator: IsaacLab (Isaac Sim, manager-based)
- objects: none (commanded goal pose), lab table
- bimanual: false
- summary: Move the end-effector to a commanded goal pose.

This is a manager-based reaching task: a Franka Emika Panda arm must drive its `panda_hand`
end-effector to a randomly commanded 6-DoF pose (position + orientation) in the workspace.
Control is joint-position (7 arm joints, no gripper action). The command is resampled every 4 s
and the episode times out at 12 s. Reward shapes position + orientation tracking with small
action-rate and joint-velocity penalties (curriculum ramps the penalties up after 4500 steps).

---

## §1 Registration + Scene

**Description.** Registers `Isaac-Reach-Franka-v0` as a `ManagerBasedRLEnv` whose cfg is
`FrankaReachEnvCfg` (joint-position control). The scene is a single Franka Panda on a Seattle-lab
table over a ground plane, lit by a dome light. The robot articulation is `MISSING` in the abstract
`ReachSceneCfg` and filled in by `FrankaReachEnvCfg.__post_init__` with `FRANKA_PANDA_CFG`.

**Decisions resolved.**
- env entry_point: `isaaclab.envs:ManagerBasedRLEnv`; `disable_env_checker=True`
- env_cfg_entry_point: `...config.franka.joint_pos_env_cfg:FrankaReachEnvCfg`
- num_envs (default) = 4096; env_spacing = 2.5
- decimation = 2; sim.dt = 1/60 s (control dt = 1/30 s); episode_length_s = 12.0
- robot USD: `{ISAACLAB_NUCLEUS_DIR}/Robots/FrankaEmika/panda_instanceable.usd`, prim_path `{ENV_REGEX_NS}/Robot`
- robot init joint_pos: j1=0.0, j2=-0.569, j3=0.0, j4=-2.810, j5=0.0, j6=3.037, j7=0.741, finger=0.04
- actuators (ImplicitActuatorCfg): shoulder j[1-4] effort 87, stiffness 80, damping 4; forearm j[5-7] effort 12, stiffness 80, damping 4; hand fingers effort 200, stiffness 2e3, damping 1e2
- soft_joint_pos_limit_factor = 1.0
- table USD: `{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd` at pos (0.55, 0, 0), rot (0.70711, 0, 0, 0.70711)
- ground: GroundPlaneCfg at pos (0, 0, -1.05)
- light: DomeLightCfg color (0.75,0.75,0.75) intensity 2500
- PLAY variant (`Isaac-Reach-Franka-Play-v0`): num_envs=50, env_spacing=2.5, obs corruption disabled

**Code.**
```python
# config/franka/__init__.py
gym.register(
    id="Isaac-Reach-Franka-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:FrankaReachEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:FrankaReachPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)
```

```python
# reach_env_cfg.py — ReachSceneCfg
@configclass
class ReachSceneCfg(InteractiveSceneCfg):
    """Configuration for the scene with a robotic arm."""

    # world
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -1.05)),
    )

    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd",
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.55, 0.0, 0.0), rot=(0.70711, 0.0, 0.0, 0.70711)),
    )

    # robots
    robot: ArticulationCfg = MISSING

    # lights
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=2500.0),
    )
```

```python
# reach_env_cfg.py — ReachEnvCfg.__post_init__
def __post_init__(self):
    """Post initialization."""
    # general settings
    self.decimation = 2
    self.sim.render_interval = self.decimation
    self.episode_length_s = 12.0
    self.viewer.eye = (3.5, 3.5, 3.5)
    # simulation settings
    self.sim.dt = 1.0 / 60.0
    # (teleop_devices: keyboard / gamepad / spacemouse, gripper_term=False — omitted for RL)
```

```python
# joint_pos_env_cfg.py — robot + body-name wiring (excerpt)
self.scene.robot = FRANKA_PANDA_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
```

```python
# isaaclab_assets/robots/franka.py — FRANKA_PANDA_CFG (verbatim)
FRANKA_PANDA_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAACLAB_NUCLEUS_DIR}/Robots/FrankaEmika/panda_instanceable.usd",
        activate_contact_sensors=False,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True, solver_position_iteration_count=8, solver_velocity_iteration_count=0
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={
            "panda_joint1": 0.0,
            "panda_joint2": -0.569,
            "panda_joint3": 0.0,
            "panda_joint4": -2.810,
            "panda_joint5": 0.0,
            "panda_joint6": 3.037,
            "panda_joint7": 0.741,
            "panda_finger_joint.*": 0.04,
        },
    ),
    actuators={
        "panda_shoulder": ImplicitActuatorCfg(
            joint_names_expr=["panda_joint[1-4]"],
            effort_limit_sim=87.0,
            stiffness=80.0,
            damping=4.0,
        ),
        "panda_forearm": ImplicitActuatorCfg(
            joint_names_expr=["panda_joint[5-7]"],
            effort_limit_sim=12.0,
            stiffness=80.0,
            damping=4.0,
        ),
        "panda_hand": ImplicitActuatorCfg(
            joint_names_expr=["panda_finger_joint.*"],
            effort_limit_sim=200.0,
            stiffness=2e3,
            damping=1e2,
        ),
    },
    soft_joint_pos_limit_factor=1.0,
)
```

**Smoke.**
```bash
cd <repo>
.venv/bin/python -c "
from isaaclab.app import AppLauncher
app = AppLauncher(headless=True).app
import gymnasium as gym, isaaclab_tasks  # noqa
from isaaclab_tasks.utils import parse_env_cfg
cfg = parse_env_cfg('Isaac-Reach-Franka-v0', num_envs=2)
env = gym.make('Isaac-Reach-Franka-v0', cfg=cfg)
print(env.unwrapped.observation_space, env.unwrapped.action_space)
"
```
Expected (literal): `Dict('policy': Box(-inf, inf, (2, 32), float32)) Box(-inf, inf, (2, 7), float32)`

---

## §2 Actions

**Description.** Single arm action term: joint-position control of the 7 Franka arm joints. No
gripper action (`gripper_action = None`). Actions are scaled by 0.5 and added to the default joint
positions (`use_default_offset=True`), so the policy outputs deltas around the home pose.

**Decisions resolved.**
- arm_action class: `mdp.JointPositionActionCfg`
- asset_name = "robot"; joint_names = `["panda_joint.*"]` (7 arm joints, finger joints excluded)
- scale = 0.5; use_default_offset = True
- gripper_action = None  → action dim = 7

**Code.**
```python
# reach_env_cfg.py — ActionsCfg (abstract)
@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    arm_action: ActionTerm = MISSING
    gripper_action: ActionTerm | None = None
```

```python
# joint_pos_env_cfg.py — concrete arm_action (verbatim)
self.actions.arm_action = mdp.JointPositionActionCfg(
    asset_name="robot", joint_names=["panda_joint.*"], scale=0.5, use_default_offset=True
)
```

(`JointPositionActionCfg` resolves to `isaaclab.envs.mdp.actions.actions_cfg:JointPositionActionCfg` — stock IsaacLab action term, not task-local.)

**Smoke.** Action space `Box(-inf, inf, (N, 7), float32)` (see §1 build smoke).

---

## §3 Reset

**Description.** On each episode reset, the arm's joint positions are randomized by scaling the
default joint positions by a uniform factor in [0.5, 1.5]; joint velocities are set to 0. The
command target (ee_pose) is resampled by the command manager (see §4). No object/table reset
events — table and ground are static.

**Decisions resolved.**
- reset term: `reset_robot_joints`, func `mdp.reset_joints_by_scale`, mode "reset"
- position_range = (0.5, 1.5)  (multiplicative scale on default joint pos)
- velocity_range = (0.0, 0.0)

**Code.**
```python
# reach_env_cfg.py — EventCfg (reset terms only)
@configclass
class EventCfg:
    """Configuration for events."""

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (0.5, 1.5),
            "velocity_range": (0.0, 0.0),
        },
    )
```

(`reset_joints_by_scale` resolves to `isaaclab.envs.mdp.events:reset_joints_by_scale` — stock.)

**Smoke.** Covered by the §1 build + a `env.reset()` call returning finite obs.

---

## §4 Goal + Termination

**Description.** Goal is encoded as a `UniformPoseCommandCfg` command term (`ee_pose`): a random
target pose for `panda_hand`, position sampled in a box in front of the robot and orientation with
fixed roll, fixed pitch=π (end-effector points down along -z), and free yaw. Resampled every 4 s.
The only termination is `time_out` at episode_length_s = 12.0 s — there is no success/failure
termination, so this is a continuous tracking task scored purely by reward.

**Decisions resolved.**
- command term: `ee_pose` = `mdp.UniformPoseCommandCfg`, asset_name "robot", body_name "panda_hand"
- resampling_time_range = (4.0, 4.0); debug_vis = True
- ranges: pos_x (0.35, 0.65), pos_y (-0.2, 0.2), pos_z (0.15, 0.5), roll (0.0, 0.0), pitch (π, π), yaw (-3.14, 3.14)
- termination: `time_out` = `DoneTerm(func=mdp.time_out, time_out=True)` — no other dones

**Code.**
```python
# reach_env_cfg.py — CommandsCfg
@configclass
class CommandsCfg:
    """Command terms for the MDP."""

    ee_pose = mdp.UniformPoseCommandCfg(
        asset_name="robot",
        body_name=MISSING,
        resampling_time_range=(4.0, 4.0),
        debug_vis=True,
        ranges=mdp.UniformPoseCommandCfg.Ranges(
            pos_x=(0.35, 0.65),
            pos_y=(-0.2, 0.2),
            pos_z=(0.15, 0.5),
            roll=(0.0, 0.0),
            pitch=MISSING,  # depends on end-effector axis
            yaw=(-3.14, 3.14),
        ),
    )
```

```python
# joint_pos_env_cfg.py — concrete command body + pitch (verbatim)
self.commands.ee_pose.body_name = "panda_hand"
self.commands.ee_pose.ranges.pitch = (math.pi, math.pi)
```

```python
# reach_env_cfg.py — TerminationsCfg
@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
```

(`UniformPoseCommandCfg` → `isaaclab.envs.mdp.commands.commands_cfg`; `time_out` → `isaaclab.envs.mdp.terminations` — both stock.)

**Smoke.** `env.step(zeros)` runs to time_out without NaNs; episode length ≈ 12 s / (1/30 s) ≈ 360 steps.

---

## §5 Observation

**Description.** Single policy observation group, terms concatenated in order, with corruption
(additive uniform noise) enabled for training. Terms: relative joint positions, relative joint
velocities, the current generated ee_pose command (7-vector: pos3 + quat4), and the last action.
Total per-env obs dim = 9 + 9 + 7 + 7 = 32 (confirmed by build smoke).

**Decisions resolved.**
- group: `policy`, enable_corruption=True, concatenate_terms=True
- terms (order preserved):
  - `joint_pos` = `mdp.joint_pos_rel`, noise Unoise(-0.01, 0.01)  → 9 (7 arm + 2 finger)
  - `joint_vel` = `mdp.joint_vel_rel`, noise Unoise(-0.01, 0.01)  → 9
  - `pose_command` = `mdp.generated_commands`, params command_name="ee_pose"  → 7
  - `actions` = `mdp.last_action`  → 7
- resolved total obs dim = 32
- noise disabled in PLAY variant (enable_corruption=False)

**Code.**
```python
# reach_env_cfg.py — ObservationsCfg
@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        pose_command = ObsTerm(func=mdp.generated_commands, params={"command_name": "ee_pose"})
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()
```

(`joint_pos_rel`, `joint_vel_rel`, `generated_commands`, `last_action` → `isaaclab.envs.mdp.observations` — all stock.)

**Smoke.** Build smoke prints obs `Box(-inf, inf, (N, 32), float32)`.

---

## §6 Reward

**Description.** Sum (additive) composition of five reward terms shaping end-effector pose tracking
against the `ee_pose` command plus regularization penalties:
1. position tracking (L2 distance penalty), weight -0.2
2. fine-grained position tracking (tanh kernel reward, std 0.1), weight +0.1
3. orientation tracking (shortest-path quaternion error penalty), weight -0.1
4. action rate L2 penalty, weight -0.0001 (curriculum → -0.005 after 4500 steps)
5. joint velocity L2 penalty, weight -0.0001 (curriculum → -0.001 after 4500 steps)

**Composer: sum.** (IsaacLab `RewardManager` sums all weighted RewTerms.)

**Decisions resolved (weights).**
- end_effector_position_tracking: func `position_command_error`, weight **-0.2**, body `panda_hand`
- end_effector_position_tracking_fine_grained: func `position_command_error_tanh`, std 0.1, weight **+0.1**, body `panda_hand`
- end_effector_orientation_tracking: func `orientation_command_error`, weight **-0.1**, body `panda_hand`
- action_rate: func `action_rate_l2`, weight **-0.0001** → curriculum -0.005 @ 4500 steps
- joint_vel: func `joint_vel_l2` (asset robot), weight **-0.0001** → curriculum -0.001 @ 4500 steps

**Planning budget (retro-computed, per-step nominal magnitudes).** Position penalty dominates
early (distance up to ~0.7 m → up to ~-0.14/step before dt); tanh fine-grained reward saturates to
+0.1/step as distance → 0; orientation penalty up to ~-0.1·π. Regularization terms are ~1e-4 scale
initially, rising 5–50× under curriculum once the policy is roughly tracking. No sparse success
bonus — the task is graded continuously, so there is no large terminal term to weight up.

**Code — RewardsCfg + Curriculum (verbatim from reach_env_cfg.py).**
```python
@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # task terms
    end_effector_position_tracking = RewTerm(
        func=mdp.position_command_error,
        weight=-0.2,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=MISSING), "command_name": "ee_pose"},
    )
    end_effector_position_tracking_fine_grained = RewTerm(
        func=mdp.position_command_error_tanh,
        weight=0.1,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=MISSING), "std": 0.1, "command_name": "ee_pose"},
    )
    end_effector_orientation_tracking = RewTerm(
        func=mdp.orientation_command_error,
        weight=-0.1,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=MISSING), "command_name": "ee_pose"},
    )

    # action penalty
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.0001)
    joint_vel = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-0.0001,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )


@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""

    action_rate = CurrTerm(
        func=mdp.modify_reward_weight, params={"term_name": "action_rate", "weight": -0.005, "num_steps": 4500}
    )

    joint_vel = CurrTerm(
        func=mdp.modify_reward_weight, params={"term_name": "joint_vel", "weight": -0.001, "num_steps": 4500}
    )
```

```python
# joint_pos_env_cfg.py — body-name overrides (verbatim)
self.rewards.end_effector_position_tracking.params["asset_cfg"].body_names = ["panda_hand"]
self.rewards.end_effector_position_tracking_fine_grained.params["asset_cfg"].body_names = ["panda_hand"]
self.rewards.end_effector_orientation_tracking.params["asset_cfg"].body_names = ["panda_hand"]
```

**Code — task-local reward functions (verbatim from reach/mdp/rewards.py).**
```python
from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import combine_frame_transforms, quat_error_magnitude, quat_mul

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def position_command_error(env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize tracking of the position error using L2-norm.

    The function computes the position error between the desired position (from the command) and the
    current position of the asset's body (in world frame). The position error is computed as the L2-norm
    of the difference between the desired and current positions.
    """
    # extract the asset (to enable type hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    # obtain the desired and current positions
    des_pos_b = command[:, :3]
    des_pos_w, _ = combine_frame_transforms(asset.data.root_pos_w, asset.data.root_quat_w, des_pos_b)
    curr_pos_w = asset.data.body_pos_w[:, asset_cfg.body_ids[0]]  # type: ignore
    return torch.norm(curr_pos_w - des_pos_w, dim=1)


def position_command_error_tanh(
    env: ManagerBasedRLEnv, std: float, command_name: str, asset_cfg: SceneEntityCfg
) -> torch.Tensor:
    """Reward tracking of the position using the tanh kernel.

    The function computes the position error between the desired position (from the command) and the
    current position of the asset's body (in world frame) and maps it with a tanh kernel.
    """
    # extract the asset (to enable type hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    # obtain the desired and current positions
    des_pos_b = command[:, :3]
    des_pos_w, _ = combine_frame_transforms(asset.data.root_pos_w, asset.data.root_quat_w, des_pos_b)
    curr_pos_w = asset.data.body_pos_w[:, asset_cfg.body_ids[0]]  # type: ignore
    distance = torch.norm(curr_pos_w - des_pos_w, dim=1)
    return 1 - torch.tanh(distance / std)


def orientation_command_error(env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize tracking orientation error using shortest path.

    The function computes the orientation error between the desired orientation (from the command) and the
    current orientation of the asset's body (in world frame). The orientation error is computed as the shortest
    path between the desired and current orientations.
    """
    # extract the asset (to enable type hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    # obtain the desired and current orientations
    des_quat_b = command[:, 3:7]
    des_quat_w = quat_mul(asset.data.root_quat_w, des_quat_b)
    curr_quat_w = asset.data.body_quat_w[:, asset_cfg.body_ids[0]]  # type: ignore
    return quat_error_magnitude(curr_quat_w, des_quat_w)
```

**Code — shared regularization reward functions (verbatim from isaaclab.envs.mdp.rewards).**
```python
def joint_vel_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize joint velocities on the articulation using L2 squared kernel.

    .. note::
        Only the joints configured in :attr:`asset_cfg.joint_ids` will have their joint velocities
        contribute to the term.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.joint_vel[:, asset_cfg.joint_ids]), dim=1)


def action_rate_l2(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Penalize the rate of change of the actions using L2 squared kernel."""
    return torch.sum(torch.square(env.action_manager.action - env.action_manager.prev_action), dim=1)
```

(`modify_reward_weight` → `isaaclab.envs.mdp.curriculums` — stock; not reproduced.)

**Smoke.** §6 smoke: per-step total reward finite and non-constant across a zero-action rollout;
`sum(detailed_reward.values()) == reward` each step (composer = sum).

---

## §7 DR

**Description.** No domain randomization. The `EventCfg` contains only the `mode="reset"` term
(`reset_robot_joints`, §3) — there are no `startup` or `interval` events, and no physics-material /
mass / friction / push randomization. Observation corruption (additive uniform noise on joint pos /
vel, §5) is applied but is an observation-noise feature, not an EventCfg DR term.

**Decisions resolved.** `<no DR>` — EventCfg has zero terms with `mode != "reset"`.

**Smoke.** §7 smoke not applicable (skipped is a valid success — the canonical task has no DR).

---

