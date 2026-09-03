# Isaac-Lift-Cube-Franka-v0 — Implementation Spec

- robot: Franka Emika Panda (7 DoF arm + parallel gripper)
- simulator: IsaacLab (Isaac Sim, manager-based)
- objects: DexCube, lab table, commanded goal pose
- bimanual: false
- summary: Reach a cube, lift it clear of the table, and carry it to a commanded goal pose.

This is the **joint-position-control** Franka cube-lift task. A Franka Panda must reach a cube on a table, lift it above 4 cm, then carry it to a randomly commanded 3D goal pose. Manager-based RL env (`isaaclab.envs:ManagerBasedRLEnv`). The arm is driven by absolute joint-position targets (scaled relative deltas off the default pose), the gripper by a binary open/close command.

## §1 Registration + Scene

### Description
Registers `Isaac-Lift-Cube-Franka-v0` against `ManagerBasedRLEnv` with cfg `FrankaCubeLiftEnvCfg`. The scene = Franka Panda articulation + a DexCube rigid object + a Seattle-lab table + ground plane + dome light + an `ee_frame` FrameTransformer tracking the end-effector (panda_hand + 0.1034 m z-offset to the grasp point). The abstract `ObjectTableSceneCfg` leaves `robot`/`ee_frame`/`object` as `MISSING`; the franka config fills them in `__post_init__`.

### Decisions resolved
- env_cfg_entry_point: `...lift.config.franka.joint_pos_env_cfg:FrankaCubeLiftEnvCfg`
- num_envs = 4096, env_spacing = 2.5
- decimation = 2; episode_length_s = 5.0; sim.dt = 0.01 (100 Hz); control freq = 50 Hz
- Robot = `FRANKA_PANDA_CFG`, prim `{ENV_REGEX_NS}/Robot`
  - USD: `{ISAACLAB_NUCLEUS_DIR}/Robots/FrankaEmika/panda_instanceable.usd`
  - init joint_pos: j1=0.0, j2=-0.569, j3=0.0, j4=-2.810, j5=0.0, j6=3.037, j7=0.741, finger=0.04
  - actuators: shoulder (j1-4) stiff=80 damp=4 effort=87; forearm (j5-7) stiff=80 damp=4 effort=12; hand (fingers) stiff=2e3 damp=1e2 effort=200; soft_joint_pos_limit_factor=1.0; self-collisions enabled
- Object = DexCube `RigidObjectCfg`, prim `{ENV_REGEX_NS}/Object`
  - USD: `{ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd`, scale (0.8,0.8,0.8)
  - init pos [0.5, 0, 0.055], rot [1,0,0,0]
  - rigid_props: solver_pos_iter=16, solver_vel_iter=1, max_ang_vel=1000, max_lin_vel=1000, max_depen_vel=5.0, gravity on
- Table = `{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd`, pos [0.5,0,0], rot [0.707,0,0,0.707]
- Plane at z=-1.05; DomeLight color (0.75,0.75,0.75) intensity 3000
- ee_frame: source `{ENV_REGEX_NS}/Robot/panda_link0`, target `{ENV_REGEX_NS}/Robot/panda_hand` name "end_effector", offset pos [0,0,0.1034]
- physx: bounce_threshold_velocity=0.01, gpu_found_lost_aggregate_pairs_capacity=4*1024*1024, gpu_total_aggregate_pairs_capacity=16*1024, friction_correlation_distance=0.00625

### Code

Registration (`config/franka/__init__.py`):
```python
gym.register(
    id="Isaac-Lift-Cube-Franka-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:FrankaCubeLiftEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:LiftCubePPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "sb3_cfg_entry_point": f"{agents.__name__}:sb3_ppo_cfg.yaml",
    },
    disable_env_checker=True,
)
```

Abstract scene (`lift_env_cfg.py`):
```python
@configclass
class ObjectTableSceneCfg(InteractiveSceneCfg):
    # robots: will be populated by agent env cfg
    robot: ArticulationCfg = MISSING
    # end-effector sensor: will be populated by agent env cfg
    ee_frame: FrameTransformerCfg = MISSING
    # target object: will be populated by agent env cfg
    object: RigidObjectCfg | DeformableObjectCfg = MISSING

    # Table
    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0.5, 0, 0], rot=[0.707, 0, 0, 0.707]),
        spawn=UsdFileCfg(usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd"),
    )

    # plane
    plane = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0, 0, -1.05]),
        spawn=GroundPlaneCfg(),
    )

    # lights
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )
```

EnvCfg base + `__post_init__` (`lift_env_cfg.py`):
```python
@configclass
class LiftEnvCfg(ManagerBasedRLEnvCfg):
    scene: ObjectTableSceneCfg = ObjectTableSceneCfg(num_envs=4096, env_spacing=2.5)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        self.decimation = 2
        self.episode_length_s = 5.0
        self.sim.dt = 0.01  # 100Hz
        self.sim.render_interval = self.decimation
        self.sim.physx.bounce_threshold_velocity = 0.2
        self.sim.physx.bounce_threshold_velocity = 0.01
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 1024 * 1024 * 4
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 16 * 1024
        self.sim.physx.friction_correlation_distance = 0.00625
```

Franka scene fill-in (`config/franka/joint_pos_env_cfg.py`, `__post_init__`):
```python
self.scene.robot = FRANKA_PANDA_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
...
self.commands.object_pose.body_name = "panda_hand"

self.scene.object = RigidObjectCfg(
    prim_path="{ENV_REGEX_NS}/Object",
    init_state=RigidObjectCfg.InitialStateCfg(pos=[0.5, 0, 0.055], rot=[1, 0, 0, 0]),
    spawn=UsdFileCfg(
        usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd",
        scale=(0.8, 0.8, 0.8),
        rigid_props=RigidBodyPropertiesCfg(
            solver_position_iteration_count=16,
            solver_velocity_iteration_count=1,
            max_angular_velocity=1000.0,
            max_linear_velocity=1000.0,
            max_depenetration_velocity=5.0,
            disable_gravity=False,
        ),
    ),
)

marker_cfg = FRAME_MARKER_CFG.copy()
marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
marker_cfg.prim_path = "/Visuals/FrameTransformer"
self.scene.ee_frame = FrameTransformerCfg(
    prim_path="{ENV_REGEX_NS}/Robot/panda_link0",
    debug_vis=False,
    visualizer_cfg=marker_cfg,
    target_frames=[
        FrameTransformerCfg.FrameCfg(
            prim_path="{ENV_REGEX_NS}/Robot/panda_hand",
            name="end_effector",
            offset=OffsetCfg(pos=[0.0, 0.0, 0.1034]),
        ),
    ],
)
```

FRANKA_PANDA_CFG (`isaaclab_assets/robots/franka.py`):
```python
FRANKA_PANDA_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAACLAB_NUCLEUS_DIR}/Robots/FrankaEmika/panda_instanceable.usd",
        activate_contact_sensors=False,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=False, max_depenetration_velocity=5.0),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True, solver_position_iteration_count=8, solver_velocity_iteration_count=0
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={
            "panda_joint1": 0.0, "panda_joint2": -0.569, "panda_joint3": 0.0, "panda_joint4": -2.810,
            "panda_joint5": 0.0, "panda_joint6": 3.037, "panda_joint7": 0.741, "panda_finger_joint.*": 0.04,
        },
    ),
    actuators={
        "panda_shoulder": ImplicitActuatorCfg(joint_names_expr=["panda_joint[1-4]"], effort_limit_sim=87.0, stiffness=80.0, damping=4.0),
        "panda_forearm": ImplicitActuatorCfg(joint_names_expr=["panda_joint[5-7]"], effort_limit_sim=12.0, stiffness=80.0, damping=4.0),
        "panda_hand": ImplicitActuatorCfg(joint_names_expr=["panda_finger_joint.*"], effort_limit_sim=200.0, stiffness=2e3, damping=1e2),
    },
    soft_joint_pos_limit_factor=1.0,
)
```

### Smoke
```bash
cd <repo>
.venv/bin/python - <<'PY'
from isaaclab.app import AppLauncher
app = AppLauncher(headless=True).app
import gymnasium as gym
import isaaclab_tasks  # noqa
from isaaclab_tasks.utils import parse_env_cfg
cfg = parse_env_cfg('Isaac-Lift-Cube-Franka-v0', num_envs=2)
env = gym.make('Isaac-Lift-Cube-Franka-v0', cfg=cfg)
print('SMOKE', env.observation_space, env.action_space)
env.close()
PY
```
Expected (analytic; not captured live): `SMOKE Dict('policy': Box(-inf, inf, (2, 36), float32)) Box(-inf, inf, (2, 8), float32)`.

## §2 Actions

### Description
Two action terms. The arm uses `JointPositionActionCfg` over the 7 `panda_joint.*` with scale 0.5 and `use_default_offset=True` — i.e. the policy outputs are scaled by 0.5 and added to the robot's default joint pose to form absolute position targets. The gripper uses `BinaryJointPositionActionCfg` over `panda_finger.*`: a single binary command, open → 0.04 m, close → 0.0 m. Action dim = 7 + 1 = 8.

### Decisions resolved
- arm_action: `mdp.JointPositionActionCfg(asset_name="robot", joint_names=["panda_joint.*"], scale=0.5, use_default_offset=True)`
- gripper_action: `mdp.BinaryJointPositionActionCfg(asset_name="robot", joint_names=["panda_finger.*"], open_command_expr={"panda_finger_.*": 0.04}, close_command_expr={"panda_finger_.*": 0.0})`

### Code
Abstract slots (`lift_env_cfg.py`):
```python
@configclass
class ActionsCfg:
    arm_action: mdp.JointPositionActionCfg | mdp.DifferentialInverseKinematicsActionCfg = MISSING
    gripper_action: mdp.BinaryJointPositionActionCfg = MISSING
```
Franka fill-in (`config/franka/joint_pos_env_cfg.py`):
```python
self.actions.arm_action = mdp.JointPositionActionCfg(
    asset_name="robot", joint_names=["panda_joint.*"], scale=0.5, use_default_offset=True
)
self.actions.gripper_action = mdp.BinaryJointPositionActionCfg(
    asset_name="robot",
    joint_names=["panda_finger.*"],
    open_command_expr={"panda_finger_.*": 0.04},
    close_command_expr={"panda_finger_.*": 0.0},
)
```

### Smoke
Step the env with `action_space.sample()` for a few steps; assert action shape (N, 8) and that the robot joint targets change. `JointPositionActionCfg` / `BinaryJointPositionActionCfg` are stock `isaaclab.envs.mdp` action terms (no task-local action code).

## §3 Reset

### Description
On every reset: (1) `reset_scene_to_default` snaps the whole scene (robot joints, object, etc.) back to its configured init state, then (2) `reset_object_position` perturbs only the cube's planar position uniformly in x ∈ [-0.1, 0.1] m, y ∈ [-0.25, 0.25] m about its default (z unchanged, no velocity). Both are `mode="reset"` EventTerms.

### Decisions resolved
- `reset_all`: `mdp.reset_scene_to_default`, no params
- `reset_object_position`: `mdp.reset_root_state_uniform` on object body "Object", pose_range x=(-0.1,0.1) y=(-0.25,0.25) z=(0.0,0.0), velocity_range {} (empty → zero)

### Code (`lift_env_cfg.py`)
```python
@configclass
class EventCfg:
    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")

    reset_object_position = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.1, 0.1), "y": (-0.25, 0.25), "z": (0.0, 0.0)},
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("object", body_names="Object"),
        },
    )
```
`reset_scene_to_default` and `reset_root_state_uniform` are stock `isaaclab.envs.mdp.events`.

### Smoke
Reset twice with the same seed → cube xy identical; reset with different seeds → cube xy differs within the configured ranges.

## §4 Goal + Termination

### Description
The goal is a per-env target object pose, supplied by a `UniformPoseCommandCfg` command term (`object_pose`) tied to the robot body `panda_hand`. The command samples a position box (x∈[0.4,0.6], y∈[-0.25,0.25], z∈[0.25,0.5]) with zero orientation, resampled every 5 s. There is NO success-based termination in the registered training task — only a `time_out` (episode horizon) and an `object_dropping` term that ends the episode if the cube falls below z = -0.05 m. (The `mdp/terminations.py:object_reached_goal` helper exists but is NOT wired into `TerminationsCfg` for this task.)

### Decisions resolved
- CommandsCfg.object_pose: `mdp.UniformPoseCommandCfg(asset_name="robot", body_name="panda_hand", resampling_time_range=(5.0,5.0), debug_vis=True, ranges: pos_x=(0.4,0.6) pos_y=(-0.25,0.25) pos_z=(0.25,0.5) roll=(0,0) pitch=(0,0) yaw=(0,0))`
- time_out: `mdp.time_out`, time_out=True
- object_dropping: `mdp.root_height_below_minimum`, minimum_height=-0.05, asset_cfg=object

### Code (`lift_env_cfg.py`)
```python
@configclass
class CommandsCfg:
    object_pose = mdp.UniformPoseCommandCfg(
        asset_name="robot",
        body_name=MISSING,  # set to "panda_hand" by franka cfg
        resampling_time_range=(5.0, 5.0),
        debug_vis=True,
        ranges=mdp.UniformPoseCommandCfg.Ranges(
            pos_x=(0.4, 0.6), pos_y=(-0.25, 0.25), pos_z=(0.25, 0.5), roll=(0.0, 0.0), pitch=(0.0, 0.0), yaw=(0.0, 0.0)
        ),
    )

@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    object_dropping = DoneTerm(
        func=mdp.root_height_below_minimum, params={"minimum_height": -0.05, "asset_cfg": SceneEntityCfg("object")}
    )
```
`UniformPoseCommandCfg`, `time_out`, `root_height_below_minimum` are stock `isaaclab.envs.mdp`. Task-local (unused-here) helper (`mdp/terminations.py`):
```python
def object_reached_goal(
    env: ManagerBasedRLEnv,
    command_name: str = "object_pose",
    threshold: float = 0.02,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    robot: RigidObject = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    command = env.command_manager.get_command(command_name)
    des_pos_b = command[:, :3]
    des_pos_w, _ = combine_frame_transforms(robot.data.root_pos_w, robot.data.root_quat_w, des_pos_b)
    distance = torch.norm(des_pos_w - object.data.root_pos_w[:, :3], dim=1)
    return distance < threshold
```

### Smoke
Roll out random actions to horizon; assert `time_out` fires at episode_length_s/(dt*decimation) ≈ 250 steps; drop the cube below z=-0.05 → `object_dropping` fires.

## §5 Observation

### Description
Single policy group, corruption-enabled, terms concatenated. Five terms: relative joint positions (9), relative joint velocities (9), object position in the robot root frame (3), the commanded target object pose (7 = pos 3 + quat 4 from `generated_commands`), and the last action (8). Total obs dim = 36. No per-term `noise` is configured, so although `enable_corruption=True`, observation noise is effectively a no-op at this commit.

### Decisions resolved
- joint_pos: `mdp.joint_pos_rel` (no noise)
- joint_vel: `mdp.joint_vel_rel` (no noise)
- object_position: `mdp.object_position_in_robot_root_frame` (task-local)
- target_object_position: `mdp.generated_commands(command_name="object_pose")`
- actions: `mdp.last_action`
- `__post_init__`: enable_corruption=True, concatenate_terms=True

### Code
Cfg (`lift_env_cfg.py`):
```python
@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        object_position = ObsTerm(func=mdp.object_position_in_robot_root_frame)
        target_object_position = ObsTerm(func=mdp.generated_commands, params={"command_name": "object_pose"})
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()
```
Task-local obs func (`mdp/observations.py`):
```python
def object_position_in_robot_root_frame(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """The position of the object in the robot's root frame."""
    robot: RigidObject = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    object_pos_w = object.data.root_pos_w[:, :3]
    object_pos_b, _ = subtract_frame_transforms(robot.data.root_pos_w, robot.data.root_quat_w, object_pos_w)
    return object_pos_b
```
`joint_pos_rel`, `joint_vel_rel`, `generated_commands`, `last_action` are stock `isaaclab.envs.mdp.observations`.

### Smoke
Reset, read `obs["policy"]`; assert shape (N, 36), all finite.

## §6 Reward

### Description
Composer = **sum** (IsaacLab RewardManager sums all `RewTerm` contributions, each multiplied by its `weight` and by `dt`). Six terms shape a reach → lift → goal-tracking curriculum:
1. `reaching_object` (w=1.0): tanh-kernel proximity of the EE to the cube (std=0.1).
2. `lifting_object` (w=15.0): +1 indicator while cube z > 0.04 m.
3. `object_goal_tracking` (w=16.0): tanh goal-distance, gated on cube lifted (std=0.3).
4. `object_goal_tracking_fine_grained` (w=5.0): same, tighter kernel (std=0.05).
5. `action_rate` (w=-1e-4): L2 action-rate penalty.
6. `joint_vel` (w=-1e-4): L2 joint-velocity penalty on the whole robot.
A CurriculumCfg ramps the two penalty weights to -1e-1 after 10000 steps.

### Decisions resolved (per-step saturated magnitudes, retro-computed; before the ~dt=0.02 RewardManager scaling)
- reaching_object: ∈ [0, 1] · 1.0 → max ≈ 1.0/step
- lifting_object: {0,15} → 15 when lifted
- object_goal_tracking: ∈ [0, 16] (only when lifted)
- object_goal_tracking_fine_grained: ∈ [0, 5] (only when lifted, near goal)
- action_rate: ≤ 0, small; ramps to -1e-1 weight after 10k steps
- joint_vel: ≤ 0, small; ramps to -1e-1 weight after 10k steps

### Code
RewardsCfg + CurriculumCfg (`lift_env_cfg.py`):
```python
@configclass
class RewardsCfg:
    reaching_object = RewTerm(func=mdp.object_ee_distance, params={"std": 0.1}, weight=1.0)
    lifting_object = RewTerm(func=mdp.object_is_lifted, params={"minimal_height": 0.04}, weight=15.0)
    object_goal_tracking = RewTerm(
        func=mdp.object_goal_distance,
        params={"std": 0.3, "minimal_height": 0.04, "command_name": "object_pose"},
        weight=16.0,
    )
    object_goal_tracking_fine_grained = RewTerm(
        func=mdp.object_goal_distance,
        params={"std": 0.05, "minimal_height": 0.04, "command_name": "object_pose"},
        weight=5.0,
    )
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-1e-4)
    joint_vel = RewTerm(func=mdp.joint_vel_l2, weight=-1e-4, params={"asset_cfg": SceneEntityCfg("robot")})

@configclass
class CurriculumCfg:
    action_rate = CurrTerm(
        func=mdp.modify_reward_weight, params={"term_name": "action_rate", "weight": -1e-1, "num_steps": 10000}
    )
    joint_vel = CurrTerm(
        func=mdp.modify_reward_weight, params={"term_name": "joint_vel", "weight": -1e-1, "num_steps": 10000}
    )
```
Full task-local reward source (`mdp/rewards.py`, verbatim):
```python
from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformer
from isaaclab.utils.math import combine_frame_transforms

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def object_is_lifted(
    env: ManagerBasedRLEnv, minimal_height: float, object_cfg: SceneEntityCfg = SceneEntityCfg("object")
) -> torch.Tensor:
    """Reward the agent for lifting the object above the minimal height."""
    object: RigidObject = env.scene[object_cfg.name]
    return torch.where(object.data.root_pos_w[:, 2] > minimal_height, 1.0, 0.0)


def object_ee_distance(
    env: ManagerBasedRLEnv,
    std: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Reward the agent for reaching the object using tanh-kernel."""
    # extract the used quantities (to enable type-hinting)
    object: RigidObject = env.scene[object_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    # Target object position: (num_envs, 3)
    cube_pos_w = object.data.root_pos_w
    # End-effector position: (num_envs, 3)
    ee_w = ee_frame.data.target_pos_w[..., 0, :]
    # Distance of the end-effector to the object: (num_envs,)
    object_ee_distance = torch.norm(cube_pos_w - ee_w, dim=1)

    return 1 - torch.tanh(object_ee_distance / std)


def object_goal_distance(
    env: ManagerBasedRLEnv,
    std: float,
    minimal_height: float,
    command_name: str,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Reward the agent for tracking the goal pose using tanh-kernel."""
    # extract the used quantities (to enable type-hinting)
    robot: RigidObject = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    command = env.command_manager.get_command(command_name)
    # compute the desired position in the world frame
    des_pos_b = command[:, :3]
    des_pos_w, _ = combine_frame_transforms(robot.data.root_pos_w, robot.data.root_quat_w, des_pos_b)
    # distance of the end-effector to the object: (num_envs,)
    distance = torch.norm(des_pos_w - object.data.root_pos_w, dim=1)
    # rewarded if the object is lifted above the threshold
    return (object.data.root_pos_w[:, 2] > minimal_height) * (1 - torch.tanh(distance / std))
```
`action_rate_l2`, `joint_vel_l2` are stock `isaaclab.envs.mdp.rewards`; `modify_reward_weight` is stock `isaaclab.envs.mdp.curriculums`.

### Smoke
Roll out; assert every per-term reward finite, `reaching_object` ∈ [0,1], `lifting_object` ∈ {0, 15}, total reward = sum of terms (RewardManager composer = sum).

## §7 DR

### Description
No domain randomization. `EventCfg` contains only `mode="reset"` terms (covered in §3); there are no `startup` or `interval` events. `enable_corruption=True` is set on the policy observation group, but no `ObsTerm` defines a `noise` model, so observation corruption is a no-op. The `_PLAY` variant explicitly sets `enable_corruption=False` and shrinks to 50 envs.

`<no DR>`

### Smoke
N/A (skipped is valid — no startup/interval events and no obs noise configured).

