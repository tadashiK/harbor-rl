# Isaac-Push-Block-Franka-v0 — Implementation Spec

- robot: Franka Emika Panda (gripper forced closed, used as a flat pusher)
- simulator: IsaacLab (Isaac Sim, manager-based)
- objects: block, target marker, lab table
- bimanual: false
- summary: Push a block across the table to a target marker using a closed gripper as a pusher.

This is a **non-prehensile planar push** task. A Franka Panda (gripper forced closed, used as a flat pusher) must push a small DexCube block across a table to a commanded 2-D goal position. Absolute joint-position control on the 7 arm joints; the goal is a `UniformPoseCommand` resampled every 4 s; success = block within 5 cm of the goal (logging-only). Modeled on `manipulation/lift` but planar (no lift gate, no `ee_frame` FrameTransformer).

---

## §1 Registration + Scene

**Description.** Registers `Isaac-Push-Block-Franka-v0` against the generic `ManagerBasedRLEnv` entry point with `env_cfg_entry_point = FrankaPushBlockEnvCfg`. The scene is a ground plane + SeattleLabTable + Franka Panda (PD-gain, no IK variant) + a scaled DexCube block + a dome light. The block spawns flat at the table center.

**Decisions resolved.**
- `entry_point = isaaclab.envs:ManagerBasedRLEnv`, `disable_env_checker=True`.
- `env_cfg_entry_point = joint_pos_env_cfg:FrankaPushBlockEnvCfg` (Play variant: `FrankaPushBlockEnvCfg_PLAY`).
- Robot: `FRANKA_PANDA_CFG` (standard PD-gain; NOT the HIGH_PD/IK variant), `prim_path="{ENV_REGEX_NS}/Robot"`. 9 DoF = 7 arm (`panda_joint1..7`) + 2 finger (`panda_finger_joint.*`).
  - Franka init joint pose: `j1=0.0, j2=-0.569, j3=0.0, j4=-2.810, j5=0.0, j6=3.037, j7=0.741, finger=0.04`.
  - Actuators: shoulder (`panda_joint[1-4]`) stiffness 80 / damping 4 / effort 87; forearm (`panda_joint[5-7]`) stiffness 80 / damping 4 / effort 12; hand (`panda_finger_joint.*`) stiffness 2e3 / damping 1e2 / effort 200. `soft_joint_pos_limit_factor=1.0`.
  - Robot USD: `{ISAACLAB_NUCLEUS_DIR}/Robots/FrankaEmika/panda_instanceable.usd` (nucleus-remote; resolved at runtime via carb `/persistent/isaac/asset_root/cloud`).
- Object: `RigidObjectCfg` `prim_path="{ENV_REGEX_NS}/Object"`, init `pos=[0.5, 0.0, 0.055]`, `rot=[1,0,0,0]`. USD `{ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd`, `scale=(0.8,0.8,0.8)`. Rigid props: solver_pos_iter=16, solver_vel_iter=1, max_ang_vel=1000, max_lin_vel=1000, max_depen_vel=5.0, gravity ON.
- Table: `AssetBaseCfg` `prim_path="{ENV_REGEX_NS}/Table"`, init `pos=[0.5,0.0,0.0]`, `rot=[0.707,0,0,0.707]`, USD `{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd`.
- Ground: `/World/GroundPlane` at `pos=[0,0,-1.05]`, `GroundPlaneCfg()`.
- Light: `/World/light`, `DomeLightCfg(color=(0.75,0.75,0.75), intensity=3000.0)`.
- Scene: `num_envs=4096`, `env_spacing=2.5`. EnvCfg: `decimation=2`, `sim.dt=1/60`, `episode_length_s=200/30 ≈ 6.667 s` (~200 control steps at 30 Hz), `sim.render_interval=2`, `physx.bounce_threshold_velocity=0.01`, `gpu_found_lost_aggregate_pairs_capacity=4*1024*1024`, `gpu_total_aggregate_pairs_capacity=16*1024`, `friction_correlation_distance=0.00625`.

**Code.**

`config/franka/__init__.py`:
```python
import gymnasium as gym

gym.register(
    id="Isaac-Push-Block-Franka-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:FrankaPushBlockEnvCfg",
    },
)

gym.register(
    id="Isaac-Push-Block-Franka-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:FrankaPushBlockEnvCfg_PLAY",
    },
)
```

`push_env_cfg.py` — `PushBlockSceneCfg`:
```python
@configclass
class PushBlockSceneCfg(InteractiveSceneCfg):
    """Scene: ground + table + Franka (filled by subclass) + small block + dome light."""

    # robot: filled by per-robot subclass via __post_init__.
    robot: ArticulationCfg = MISSING
    # target object: filled by per-robot subclass.
    object: RigidObjectCfg = MISSING

    # Table — same SeattleLabTable used by Reach / Lift.
    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0.5, 0.0, 0.0], rot=[0.707, 0.0, 0.0, 0.707]),
        spawn=UsdFileCfg(usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd"),
    )

    # Ground plane.
    plane = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0.0, 0.0, -1.05]),
        spawn=GroundPlaneCfg(),
    )

    # Lights.
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )
```

`push_env_cfg.py` — `PushBlockEnvCfg` (EnvCfg + post-init):
```python
@configclass
class PushBlockEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the push-block environment."""

    # Scene
    scene: PushBlockSceneCfg = PushBlockSceneCfg(num_envs=4096, env_spacing=2.5)
    # Manager dataclasses
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()

    def __post_init__(self):
        """Post-init: 30 Hz control rate * 6.667 s ≈ 200 control steps per episode."""
        self.decimation = 2
        # 200 control steps at decimation=2, sim.dt = 1/60s  →  episode_length_s = 200 / 30
        self.episode_length_s = 200.0 / 30.0
        # Simulation
        self.sim.dt = 1.0 / 60.0
        self.sim.render_interval = self.decimation
        # Physics knobs (mirror Lift — needed for stable contact between fingertip and block).
        self.sim.physx.bounce_threshold_velocity = 0.01
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 1024 * 1024 * 4
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 16 * 1024
        self.sim.physx.friction_correlation_distance = 0.00625
```

`config/franka/joint_pos_env_cfg.py` — robot + object wiring (scene-relevant parts):
```python
@configclass
class FrankaPushBlockEnvCfg(PushBlockEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        # Robot — standard PD-gain Franka (no IK, so HIGH_PD variant not needed).
        self.scene.robot = FRANKA_PANDA_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # ... (actions — see §2)

        # Goal command — bind the goal frame to panda_hand (debug-vis attaches to this body).
        self.commands.object_pose.body_name = "panda_hand"

        # Object — small DexCube block, spawned flat at table center.
        self.scene.object = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Object",
            init_state=RigidObjectCfg.InitialStateCfg(pos=[0.5, 0.0, 0.055], rot=[1.0, 0.0, 0.0, 0.0]),
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


@configclass
class FrankaPushBlockEnvCfg_PLAY(FrankaPushBlockEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        # Smaller scene for play.
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # Disable observation noise for play.
        self.observations.policy.enable_corruption = False
```

`FRANKA_PANDA_CFG` (from `isaaclab_assets/robots/franka.py`):
```python
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
            joint_names_expr=["panda_joint[1-4]"], effort_limit_sim=87.0, stiffness=80.0, damping=4.0,
        ),
        "panda_forearm": ImplicitActuatorCfg(
            joint_names_expr=["panda_joint[5-7]"], effort_limit_sim=12.0, stiffness=80.0, damping=4.0,
        ),
        "panda_hand": ImplicitActuatorCfg(
            joint_names_expr=["panda_finger_joint.*"], effort_limit_sim=200.0, stiffness=2e3, damping=1e2,
        ),
    },
    soft_joint_pos_limit_factor=1.0,
)
```

**Smoke (§1 build).**
```bash
cd "<repo>"
.venv/bin/python -c "import gymnasium as gym; import isaaclab_tasks; env = gym.make('Isaac-Push-Block-Franka-v0'); print(env.observation_space, env.action_space); env.close()"
```
Expected (analytic; not captured in this probe — `pxr` missing): `Box(-inf, inf, (N, 36)) Box(-inf, inf, (N, 8))`.

---

## §2 Actions

**Description.** Absolute joint-position control on the 7 Franka arm joints (`scale=0.5`, offset by the default home pose). The gripper retains a binary action term for canonical compatibility but both open/close expressions are forced to `0.0` — the gripper stays closed and the hand acts as a flat pusher. Action dim = 8 (7 arm + 1 gripper).

**Decisions resolved.**
- `arm_action = JointPositionActionCfg(joint_names=["panda_joint.*"], scale=0.5, use_default_offset=True)`.
- `gripper_action = BinaryJointPositionActionCfg(joint_names=["panda_finger.*"], open_command_expr={"panda_finger_.*": 0.0}, close_command_expr={"panda_finger_.*": 0.0})` — both 0.0 → gripper permanently closed.
- `ActionsCfg` base leaves both as `MISSING`; the franka subclass fills them. Action classes come from `isaaclab.envs.mdp` (library), not task-local.

**Code.**

`push_env_cfg.py` — `ActionsCfg`:
```python
@configclass
class ActionsCfg:
    """Action specs — filled by per-robot subclass."""

    arm_action: mdp.JointPositionActionCfg = MISSING
    gripper_action: mdp.BinaryJointPositionActionCfg = MISSING
```

`config/franka/joint_pos_env_cfg.py` — action wiring:
```python
        # Actions — absolute joint-position on the 7 arm joints.
        self.actions.arm_action = mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=["panda_joint.*"],
            scale=0.5,
            use_default_offset=True,
        )
        # Gripper — keep binary action term for canonical compatibility, but force
        # both expressions to 0.0 so the gripper stays closed and acts as a flat pusher.
        self.actions.gripper_action = mdp.BinaryJointPositionActionCfg(
            asset_name="robot",
            joint_names=["panda_finger.*"],
            open_command_expr={"panda_finger_.*": 0.0},
            close_command_expr={"panda_finger_.*": 0.0},
        )
```

**Smoke (§2).** Step the env with `env.action_space.sample()`; assert returned action dim == 8 and that finger joints stay near 0.04 (closed). No task-local action functions to compile.

---

## §3 Reset

**Description.** Two reset terms. Robot joints are scaled by a uniform `[0.5, 1.5]` multiple of the URDF home pose (zero velocity); the block position is jittered uniformly ±5 cm in x and y around table center (z fixed, zero velocity).

**Decisions resolved.**
- `reset_robot_joints`: `func=mdp.reset_joints_by_scale`, `position_range=(0.5, 1.5)`, `velocity_range=(0.0, 0.0)`.
- `reset_object_position`: `func=mdp.reset_root_state_uniform`, `pose_range={"x": (-0.05,0.05), "y": (-0.05,0.05), "z": (0.0,0.0)}`, `velocity_range={}`, `asset_cfg=SceneEntityCfg("object")`.
- Both functions are library `isaaclab.envs.mdp` events (not task-local).

**Code.** `push_env_cfg.py` — `EventCfg` reset terms:
```python
    # ---------------- §3 RESET TERMS ----------------

    # Robot joints: scale URDF home pose by uniform [0.5, 1.5] — Reach default.
    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (0.5, 1.5),
            "velocity_range": (0.0, 0.0),
        },
    )

    # Block position: small jitter around table center per episode.
    reset_object_position = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.05, 0.05), "y": (-0.05, 0.05), "z": (0.0, 0.0)},
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("object"),
        },
    )
```

**Smoke (§3).** Reset twice with different seeds; assert block xy differs and robot joint pos differs between resets.

---

## §4 Goal + Termination

**Description.** The goal is a `UniformPoseCommand` named `object_pose`: a 3-D target position for the block on the table surface, resampled every 4 s. Target z is pinned to the block-rest height (0.055), so the goal is effectively planar (2-D). The command frame is bound to `panda_hand` (used only for debug-vis attachment). Episode terminates on time-out (~6.667 s) or when the block drops below `z = -0.05` (falls off table).

**Decisions resolved.**
- `commands.object_pose = UniformPoseCommandCfg(asset_name="robot", body_name="panda_hand", resampling_time_range=(4.0,4.0), debug_vis=True, ranges: pos_x=(0.4,0.7), pos_y=(-0.25,0.25), pos_z=(0.055,0.055), roll/pitch/yaw all (0,0))`. `body_name` is `MISSING` in the base CommandsCfg, filled to `panda_hand` by the franka subclass.
- `terminations.time_out = DoneTerm(func=mdp.time_out, time_out=True)`.
- `terminations.object_dropping = DoneTerm(func=mdp.root_height_below_minimum, params={"minimum_height": -0.05, "asset_cfg": SceneEntityCfg("object")})`.
- `time_out`, `root_height_below_minimum`, `UniformPoseCommandCfg` all from library `isaaclab.envs.mdp`.

**Code.**

`push_env_cfg.py` — `CommandsCfg`:
```python
@configclass
class CommandsCfg:
    """Goal: a 3-D position target for the block on the table surface."""

    object_pose = mdp.UniformPoseCommandCfg(
        asset_name="robot",
        body_name=MISSING,                       # filled by subclass (panda_hand)
        resampling_time_range=(4.0, 4.0),        # resample target every 4 sim seconds
        debug_vis=True,
        ranges=mdp.UniformPoseCommandCfg.Ranges(
            pos_x=(0.4, 0.7),
            pos_y=(-0.25, 0.25),
            pos_z=(0.055, 0.055),                # planar push: target z is fixed at block-rest height
            roll=(0.0, 0.0),
            pitch=(0.0, 0.0),
            yaw=(0.0, 0.0),
        ),
    )
```

Subclass binds the body (`config/franka/joint_pos_env_cfg.py`):
```python
        # Goal command — bind the goal frame to panda_hand (debug-vis attaches to this body).
        self.commands.object_pose.body_name = "panda_hand"
```

`push_env_cfg.py` — `TerminationsCfg`:
```python
@configclass
class TerminationsCfg:
    """Time-out + block-falls-off-table failure (mirrors Lift)."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    object_dropping = DoneTerm(
        func=mdp.root_height_below_minimum,
        params={"minimum_height": -0.05, "asset_cfg": SceneEntityCfg("object")},
    )
```

**Smoke (§4).** Step ~200 steps with zero actions; assert `terminated`/`truncated` fires by episode end and that the command tensor has shape (N, 7).

---

## §5 Observation

**Description.** Single policy group, concatenated, with corruption enabled (additive uniform noise on joint pos/vel). Terms in order: relative joint position (9), relative joint velocity (9), object position in robot root frame (3), goal command (7 = pos+quat), last action (8). Total = 36.

**Decisions resolved.**
- `PolicyCfg` `__post_init__`: `enable_corruption=True`, `concatenate_terms=True` (Play variant disables corruption).
- Terms (order = concatenation order):
  - `joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(-0.01, 0.01))` → 9
  - `joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(-0.01, 0.01))` → 9
  - `object_position = ObsTerm(func=mdp.object_position_in_robot_root_frame)` → 3 (TASK-LOCAL)
  - `target_object_position = ObsTerm(func=mdp.generated_commands, params={"command_name": "object_pose"})` → 7
  - `actions = ObsTerm(func=mdp.last_action)` → 8
- Resolved total obs dim = 9+9+3+7+8 = **36**. No asymmetric critic group.

**Code.**

`push_env_cfg.py` — `ObservationsCfg`:
```python
@configclass
class ObservationsCfg:
    """Observation specs — Lift's PolicyCfg with no asymmetric critic."""

    @configclass
    class PolicyCfg(ObsGroup):
        # Observation terms — concatenation order is preserved.
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        object_position = ObsTerm(func=mdp.object_position_in_robot_root_frame)
        target_object_position = ObsTerm(func=mdp.generated_commands, params={"command_name": "object_pose"})
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()
```

Task-local observation function (`mdp/observations.py`):
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
(imports: `import torch`; `from isaaclab.assets import RigidObject`; `from isaaclab.managers import SceneEntityCfg`; `from isaaclab.utils.math import subtract_frame_transforms`.)

**Smoke (§5).** Reset and assert flat obs vector length == 36.

---

## §6 Reward

**Description.** Composer = **sum** (`RewardManager` sums `weight_i * func_i(env)` per step). Five active shaping/regularizer terms plus one logging-only success term:
1. `reaching_block` (w=1.0): tanh reaching reward pulling `panda_hand` toward the block CoM (std=0.1).
2. `block_to_goal_tracking` (w=16.0): tanh block→goal tracking, coarse kernel (std=0.3).
3. `block_to_goal_tracking_fine_grained` (w=5.0): same, sharp kernel for last cm (std=0.05).
4. `success` (w=0.0): binary block-within-5cm indicator, logging-only via `info["detailed_reward"]["success"]`.
5. `action_rate` (w=-1e-4): action-rate L2 penalty (library `action_rate_l2`).
6. `joint_vel` (w=-1e-4): joint-velocity L2 penalty (library `joint_vel_l2`).

**Planning-budget (retro-computed; no explicit docstring budget in source).** Per-step saturated maxima (nominal weights):
- `reaching_block`: weight 1.0 × max 1.0 = **1.0** when hand touches block.
- `block_to_goal_tracking`: 16.0 × max 1.0 = **16.0** when block at goal (coarse).
- `block_to_goal_tracking_fine_grained`: 5.0 × max 1.0 = **5.0** when block at goal (sharp).
- `success`: 0.0 (logging-only).
- `action_rate` / `joint_vel`: small negative regularizers (≈ -1e-4 × ‖·‖², near 0 for smooth motion).
- Saturated max per-step reward ≈ 22.0 when block sits at goal and hand is in contact.

**Decisions resolved.** See `RewardsCfg` below for exact func/params/weight per term.

**Code.**

`push_env_cfg.py` — `RewardsCfg`:
```python
@configclass
class RewardsCfg:
    """Reward terms for the push-block MDP.

    Composer is `sum` -- RewardManager sums `weight_i * func_i(env)` per step.
    The `success` term is logging-only (weight=0.0) and surfaces the user's
    "<5cm" success criterion via `info["detailed_reward"]["success"]`.
    """

    # POSITIVE shaping -- pull EE toward the block (reaching).
    reaching_block = RewTerm(
        func=mdp.object_ee_distance_body,
        params={
            "std": 0.1,
            "robot_cfg": SceneEntityCfg("robot", body_names=["panda_hand"]),
        },
        weight=1.0,
    )

    # POSITIVE shaping -- pull the block toward the target marker (coarse kernel).
    block_to_goal_tracking = RewTerm(
        func=mdp.block_to_goal_distance,
        params={"std": 0.3, "command_name": "object_pose"},
        weight=16.0,
    )

    # POSITIVE shaping -- sharper kernel for the last few centimetres.
    block_to_goal_tracking_fine_grained = RewTerm(
        func=mdp.block_to_goal_distance,
        params={"std": 0.05, "command_name": "object_pose"},
        weight=5.0,
    )

    # LOGGING-ONLY -- surfaces user's <5cm success criterion in info["detailed_reward"]["success"].
    success = RewTerm(
        func=mdp.block_at_goal,
        params={"threshold": 0.05, "command_name": "object_pose"},
        weight=0.0,
    )

    # REGULARIZER -- penalize jerky actions.
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-1e-4)

    # REGULARIZER -- penalize excessive joint velocity.
    joint_vel = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-1e-4,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
```

**Full source of all task-local reward functions** (`mdp/rewards.py`):
```python
from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import combine_frame_transforms

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def object_ee_distance_body(
    env: ManagerBasedRLEnv,
    std: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=["panda_hand"]),
) -> torch.Tensor:
    """Reward the agent for reaching the block using a tanh kernel.

    Distance is measured between the robot body referenced by `robot_cfg.body_names[0]`
    (resolved to `body_ids[0]` by SceneEntityCfg) and the object's CoM, both in world frame.
    """
    object: RigidObject = env.scene[object_cfg.name]
    robot: Articulation = env.scene[robot_cfg.name]
    cube_pos_w = object.data.root_pos_w
    ee_w = robot.data.body_pos_w[:, robot_cfg.body_ids[0]]
    distance = torch.norm(cube_pos_w - ee_w, dim=1)
    return 1 - torch.tanh(distance / std)


def block_to_goal_distance(
    env: ManagerBasedRLEnv,
    std: float,
    command_name: str,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Reward the agent for tracking the block-to-goal position using a tanh kernel.

    Mirrors `manipulation/lift/mdp/rewards.py:object_goal_distance` but DROPS the
    `minimal_height` gate -- Push is planar and the block never leaves the table.
    """
    robot: RigidObject = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    command = env.command_manager.get_command(command_name)
    des_pos_b = command[:, :3]
    des_pos_w, _ = combine_frame_transforms(robot.data.root_pos_w, robot.data.root_quat_w, des_pos_b)
    distance = torch.norm(des_pos_w - object.data.root_pos_w, dim=1)
    return 1 - torch.tanh(distance / std)


def block_at_goal(
    env: ManagerBasedRLEnv,
    threshold: float,
    command_name: str,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Binary success indicator: 1.0 when block-to-goal distance < `threshold`, else 0.0.

    Logging-only term (used at weight=0.0). Wired into `RewardsCfg.success` so
    `info["detailed_reward"]["success"]` exposes per-step success without
    contributing to the optimization signal.
    """
    robot: RigidObject = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    command = env.command_manager.get_command(command_name)
    des_pos_b = command[:, :3]
    des_pos_w, _ = combine_frame_transforms(robot.data.root_pos_w, robot.data.root_quat_w, des_pos_b)
    distance = torch.norm(des_pos_w - object.data.root_pos_w, dim=1)
    return (distance < threshold).float()
```

`action_rate_l2` and `joint_vel_l2` are library functions from `isaaclab.envs.mdp` (not task-local).

**Smoke (§6).** Step the env; assert total reward is finite, non-constant, and (with a per-term reward log) `sum(detailed_reward.values()) == reward` per step (composer = sum).

---

## §7 DR

**Description.** Two `startup`-mode domain-randomization terms (per-env-instance, fixed for env lifetime): friction randomization on the robot finger contact surfaces, and a ±20% multiplicative randomization of the block mass.

**Decisions resolved.**
- `physics_material`: `func=mdp.randomize_rigid_body_material`, `mode="startup"`, `asset_cfg=SceneEntityCfg("robot", body_names="panda_.*finger")`, `static_friction_range=(0.8,1.2)`, `dynamic_friction_range=(0.8,1.2)`, `restitution_range=(0.0,0.0)`, `num_buckets=64`.
- `block_mass`: `func=mdp.randomize_rigid_body_mass`, `mode="startup"`, `asset_cfg=SceneEntityCfg("object")`, `mass_distribution_params=(0.8,1.2)`, `operation="scale"` (multiplicative, because dex-cube nominal mass ≈ 50 g; additive would distort).
- Both functions are library `isaaclab.envs.mdp` events. No `interval`-mode DR.

**Code.** `push_env_cfg.py` — `EventCfg` startup terms:
```python
    # ---------------- §7 DR (STARTUP, MINIMAL) ----------------
    # Per-env-instance, fixed for the lifetime of the env. ...

    # Friction randomization on robot fingers. ...
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="panda_.*finger"),
            "static_friction_range": (0.8, 1.2),
            "dynamic_friction_range": (0.8, 1.2),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )

    # Block mass: ±20% multiplicative. ...
    block_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("object"),
            "mass_distribution_params": (0.8, 1.2),
            "operation": "scale",
        },
    )
```

**Smoke (§7).** Build env with DR ON vs OFF under matched seeds; assert obs trajectories diverge (different friction/mass → different contact dynamics).

---

