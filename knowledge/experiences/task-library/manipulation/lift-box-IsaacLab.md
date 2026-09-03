# IsaacLab-Lift-Box — Implementation Spec

> **SUPERSEDED by [[lift-box-IsaacLab-v2]]** (`lift-box-IsaacLab-v2.md`). This spec's §6 success
> predicate (`lift_box_success`) has NO grasp condition — training-validated reproduction
> (2026-06-06, reward-tune on Isaac-Lift-Box-Dual-Franka-v0) showed PPO exploits it via a
> grasp-less wedge-lift (success_rate 0.994 with zero finger contact), and the ungated
> `box_xy_align` term admits a tip-box-on-end optimum. v2 gates success, success_bonus and
> box_xy_align on dual finger contact (gate_0 & gate_1) and is training-validated to
> success_rate 0.89. Prefer v2 as the reproduction/adaptation base.

- robot: Two Franka FR3 arms + Franka hands (dual-arm cooperative)
- simulator: IsaacLab (Isaac Sim, manager-based)
- objects: eurobox, lab table
- bimanual: true
- summary: Two arms cooperatively grasp a eurobox and lift it off the table.

Task summary: two FR3 + Franka-hand robots stand at world `y = ±0.49` facing each other. A 0.40 × 0.30 × 0.22 m eurobox (0.5 kg) sits centered on the lab table, **rotated 90° about +Z** so its long axis runs along world Y (between the robots) and its short y-end faces (0.30 m × 0.22 m) face each robot. Two `FrameTransformerCfg` markers (`grasp_frame_0`, `grasp_frame_1`) visualize the top-center of each short face. Each robot grasps its assigned short y-end face top-down with a parallel-jaw gripper (3-D EMA xyz EE-delta + binary gripper; RPY locked at reset). Goal: lift the box COM to world env-local `(0, 0, BOX_INIT_Z + 0.25) = (0, 0, 0.36025)` with `|box.lin_vel_w| < 0.10 m/s`. Episode horizon = 10 s @ 20 Hz = 200 steps.

---

## §1 Registration + Scene

### Description

`gym.register` exposes `IsaacLab-Lift-Box` (training) and `IsaacLab-Lift-Box-Play` (eval-friendly, fewer envs, no noise). Both use `isaaclab.envs:ManagerBasedRLEnv` with the abstract `LiftBoxEnvCfg` base + `FrankaLiftBoxEnvCfg` subclass that fills in the two robot articulations, the box (RigidObjectCfg from a converted STL→USD), per-robot ee_frame transformers, two box-local grasp_frame transformer markers, and four contact sensors (left/right finger per robot). Sim timing and physx knobs mirror `manipulation/insert_drawer/` verbatim (`sim.dt=1/120`, `decimation=6 → 20 Hz`, `episode_length_s=10.0`). Both `FR3_FRANKA_HAND_CFG` and `EMACumulativeDeltaPositionActionCfg` are imported from the `insert_drawer` task (not vendored).

### Decisions resolved

| Knob | Value |
|---|---|
| Task ID (train / play) | `IsaacLab-Lift-Box` / `IsaacLab-Lift-Box-Play` |
| Robot | FR3 + Franka hand (`harbor/assets/fr3/fr3_franka_hand.usd`), 2 instances at `{ENV_REGEX_NS}/Robot_0` and `Robot_1` |
| robot_0 init pos (env-local) | `(-0.274, 0.49, 0.01)` |
| robot_1 init pos (env-local) | `(-0.274, -0.49, 0.01)` (symmetric across world y=0) |
| robot_0 init joint pose | `fr3_joint1=-0.785, joint2=-0.785, joint3=0.0, joint4=-2.655, joint5=0.0, joint6=1.87, joint7=0.0, fr3_finger_joint.*=0.04` |
| robot_1 init joint pose | `fr3_joint1=+0.785, joint2=-0.785, joint3=0.0, joint4=-2.655, joint5=0.0, joint6=1.87, joint7=-1.57, fr3_finger_joint.*=0.04` |
| Joint-pose flip rule | flip sign of joint1 (base yaw) + joint3 (forearm yaw) + joint5 (wrist yaw) + joint7 (final wrist roll). robot_0's joint7 user-overridden to 0 so gripper jaws align along world Y (closes ACROSS box's short y-face). |
| Box asset | `<downloads>/eurobox.stl` → converted via `isaaclab.sim.converters.MeshConverter` to `harbor/assets/eurobox/eurobox.usd`. Conversion script: `harbor/create-task/isaaclab-lift-box/make_eurobox_usd.py`. Recentered: `translation=-centroid` so asset origin = box geometric center. Local extents: x=±0.20, y=±0.15, z=±0.11025. |
| `BOX_INIT_Z` | `0.11025` (half z-extent — box bottom on table at world z=0) |
| Box mass | `0.5 kg` (`MassPropertiesCfg(mass=0.5)`) |
| Box init pose (env-local) | `pos=(0.0, 0.0, 0.11025)`, `rot=(0.7071068, 0.0, 0.0, 0.7071068)` (90° about +Z — long axis aligned to world Y) |
| Table | `harbor/assets/table/lab_table_instanceable_colored_rotated.usd`, kinematic, surface at z≈0, env-local pos=(0,0,0) |
| Ground plane | world z = -0.82 |
| Light | `DomeLightCfg(color=(0.75,0.75,0.75), intensity=3000.0)` at `/World/light` |
| ee_frame_0 / ee_frame_1 | per-robot `FrameTransformerCfg` from `{ENV_REGEX_NS}/Robot_<i>/fr3_link0` to `Robot_<i>/fr3_hand` with `OffsetCfg(pos=[0, 0, 0.2])` (fingertip TCP); `debug_vis=True`, marker scale `(0.1, 0.1, 0.1)` |
| grasp_frame_0 | `FrameTransformerCfg` on `{ENV_REGEX_NS}/Box` (anchor=target), `OffsetCfg(pos=[+0.20, 0.0, +0.11025])` body-local → world `(0, +0.20, 0.2205)` after the 90° Z-rotation. Marker scale `(0.08, 0.08, 0.08)`. |
| grasp_frame_1 | same with `OffsetCfg(pos=[-0.20, 0.0, +0.11025])` → world `(0, -0.20, 0.2205)`. |
| Contact sensors | `finger_left_contact_{0,1}` on `Robot_<i>/fr3_leftfinger`, `finger_right_contact_{0,1}` on `Robot_<i>/fr3_rightfinger`. All four filtered against `{ENV_REGEX_NS}/Box`. `history_length=1`, `update_period=0.0`. |
| `num_envs` (default) | 4096 (train) / 50 (PLAY) |
| `env_spacing` | 2.5 m |
| `replicate_physics` | False |
| Timing | `sim.dt=1/120`, `decimation=6` → 20 Hz, `episode_length_s=10.0` (= 200 steps) |
| physx | `bounce_threshold_velocity=0.01`, `gpu_found_lost_aggregate_pairs_capacity=4*1024*1024`, `gpu_total_aggregate_pairs_capacity=64*1024`, `friction_correlation_distance=0.00625` |

### Code (verbatim)

`source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/lift_box/__init__.py`:

```python
"""Dual-arm cooperative box-lift task (IsaacLab-Lift-Box).

Two FR3 + Franka-hand robots cooperate to lift a eurobox (40 x 30 x 22 cm,
0.5 kg) off a lab table. Scene + sim timing + actuation pattern mirror
`manipulation/insert_drawer/`.
"""

from . import mdp  # noqa: F401 — re-exports task-local helpers
```

`source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/lift_box/config/franka/__init__.py`:

```python
import gymnasium as gym

gym.register(
    id="IsaacLab-Lift-Box",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:FrankaLiftBoxEnvCfg",
    },
)

gym.register(
    id="IsaacLab-Lift-Box-Play",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:FrankaLiftBoxEnvCfg_PLAY",
    },
)
```

Abstract scene (`lift_box_env_cfg.py:LiftBoxSceneCfg`):

```python
_TABLE_USD_PATH = str(
    Path(__file__).resolve().parents[6]
    / "harbor" / "assets" / "table" / "lab_table_instanceable_colored_rotated.usd"
)


@configclass
class LiftBoxSceneCfg(InteractiveSceneCfg):
    # Robots — filled by the per-robot subclass via __post_init__.
    robot_0: ArticulationCfg = MISSING
    robot_1: ArticulationCfg = MISSING

    # End-effector frame sensors — one per robot.
    ee_frame_0: FrameTransformerCfg = MISSING
    ee_frame_1: FrameTransformerCfg = MISSING

    # The box being lifted.
    box: RigidObjectCfg = MISSING

    # Box grasp-point frame markers (debug viz only — no rl-side reader).
    # After the box's 90° Z-rotation the box's local +x maps to world +y;
    # the two markers sit at the TOP of the box's two short y-end faces:
    #   grasp_frame_0  -> local (+x_extent, 0, +z_extent) = (+0.20, 0, +0.110)
    #                   -> world (0, +0.20, BOX_INIT_Z + 0.110) — for robot_0.
    #   grasp_frame_1  -> local (-x_extent, 0, +z_extent) = (-0.20, 0, +0.110)
    #                   -> world (0, -0.20, BOX_INIT_Z + 0.110) — for robot_1.
    grasp_frame_0: FrameTransformerCfg = MISSING
    grasp_frame_1: FrameTransformerCfg = MISSING

    # Contact sensors on each robot's fingertips, filtered against the box.
    finger_left_contact_0 = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot_0/fr3_leftfinger",
        update_period=0.0, history_length=1, debug_vis=False,
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Box"],
    )
    finger_right_contact_0 = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot_0/fr3_rightfinger",
        update_period=0.0, history_length=1, debug_vis=False,
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Box"],
    )
    finger_left_contact_1 = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot_1/fr3_leftfinger",
        update_period=0.0, history_length=1, debug_vis=False,
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Box"],
    )
    finger_right_contact_1 = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot_1/fr3_rightfinger",
        update_period=0.0, history_length=1, debug_vis=False,
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Box"],
    )

    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0.0, 0.0, 0.0]),
        spawn=UsdFileCfg(usd_path=_TABLE_USD_PATH),
    )
    plane = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0.0, 0.0, -0.82]),
        spawn=GroundPlaneCfg(),
    )
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )
```

Per-robot subclass (`config/franka/joint_pos_env_cfg.py`) — robot constants + the §1 wiring inside `__post_init__`:

```python
from isaaclab_tasks.manager_based.manipulation.insert_drawer.config.franka.joint_pos_env_cfg import (
    FR3_FRANKA_HAND_CFG,
)
from isaaclab_tasks.manager_based.manipulation.insert_drawer import mdp as insert_drawer_mdp

BOX_INIT_Z: float = 0.11025
BOX_MASS: float = 0.5
_EUROBOX_USD_PATH = str(_HARBOR_ASSETS / "eurobox" / "eurobox.usd")

FRANKA_INIT_JOINT_POS_0 = {
    "fr3_joint1": -0.785, "fr3_joint2": -0.785, "fr3_joint3": 0.0,
    "fr3_joint4": -2.655, "fr3_joint5": 0.0,    "fr3_joint6": 1.87,
    "fr3_joint7": 0,      "fr3_finger_joint.*": 0.04,
}
FRANKA_INIT_JOINT_POS_1 = {
    "fr3_joint1": 0.785,  "fr3_joint2": -0.785, "fr3_joint3": 0.0,
    "fr3_joint4": -2.655, "fr3_joint5": 0.0,    "fr3_joint6": 1.87,
    "fr3_joint7": -1.57,  "fr3_finger_joint.*": 0.04,
}

self.scene.robot_0 = FR3_FRANKA_HAND_CFG.replace(
    prim_path="{ENV_REGEX_NS}/Robot_0",
    init_state=ArticulationCfg.InitialStateCfg(joint_pos=FRANKA_INIT_JOINT_POS_0, pos=(-0.274, 0.49, 0.01)),
)
self.scene.robot_1 = FR3_FRANKA_HAND_CFG.replace(
    prim_path="{ENV_REGEX_NS}/Robot_1",
    init_state=ArticulationCfg.InitialStateCfg(joint_pos=FRANKA_INIT_JOINT_POS_1, pos=(-0.274, -0.49, 0.01)),
)

# Box (90° Z-rotation so long axis runs along world Y).
self.scene.box = RigidObjectCfg(
    prim_path="{ENV_REGEX_NS}/Box",
    init_state=RigidObjectCfg.InitialStateCfg(
        pos=[0.0, 0.0, BOX_INIT_Z],
        rot=[0.7071068, 0.0, 0.0, 0.7071068],
    ),
    spawn=UsdFileCfg(
        usd_path=_EUROBOX_USD_PATH,
        mass_props=sim_utils.MassPropertiesCfg(mass=BOX_MASS),
        rigid_props=RigidBodyPropertiesCfg(
            solver_position_iteration_count=16, solver_velocity_iteration_count=1,
            max_angular_velocity=1000.0, max_linear_velocity=1000.0,
            max_depenetration_velocity=5.0, disable_gravity=False,
        ),
    ),
)

# Per-robot ee_frame + grasp_frame.
ee_marker_cfg_0 = FRAME_MARKER_CFG.copy()
ee_marker_cfg_0.markers["frame"].scale = (0.1, 0.1, 0.1)
ee_marker_cfg_0.prim_path = "/Visuals/FrameTransformer0"
self.scene.ee_frame_0 = FrameTransformerCfg(
    prim_path="{ENV_REGEX_NS}/Robot_0/fr3_link0", debug_vis=True, visualizer_cfg=ee_marker_cfg_0,
    target_frames=[FrameTransformerCfg.FrameCfg(
        prim_path="{ENV_REGEX_NS}/Robot_0/fr3_hand", name="end_effector",
        offset=OffsetCfg(pos=[0.0, 0.0, 0.2]),
    )],
)
# ee_frame_1 analogous (Robot_1 + "/Visuals/FrameTransformer1").

grasp_marker_cfg_0 = FRAME_MARKER_CFG.copy()
grasp_marker_cfg_0.markers["frame"].scale = (0.08, 0.08, 0.08)
grasp_marker_cfg_0.prim_path = "/Visuals/BoxGraspFrame0"
self.scene.grasp_frame_0 = FrameTransformerCfg(
    prim_path="{ENV_REGEX_NS}/Box", debug_vis=True, visualizer_cfg=grasp_marker_cfg_0,
    target_frames=[FrameTransformerCfg.FrameCfg(
        prim_path="{ENV_REGEX_NS}/Box", name="box_grasp_0",
        offset=OffsetCfg(pos=[0.20, 0.0, 0.11025]),
    )],
)
# grasp_frame_1 analogous (offset=[-0.20, 0.0, 0.11025], "/Visuals/BoxGraspFrame1").
```

EnvCfg `__post_init__` (sim/physx):

```python
def __post_init__(self):
    self.decimation = 6
    self.episode_length_s = 10.0
    self.sim.dt = 1 / 120
    self.sim.render_interval = self.decimation
    self.sim.physx.bounce_threshold_velocity = 0.01
    self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 1024 * 1024 * 4
    self.sim.physx.gpu_total_aggregate_pairs_capacity = 64 * 1024
    self.sim.physx.friction_correlation_distance = 0.00625


@configclass
class FrankaLiftBoxEnvCfg_PLAY(FrankaLiftBoxEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
```

### Smoke

```bash
.venv/bin/python harbor/create-task/isaaclab-lift-box/smokes/smoke_s1.py
# exit 0 (asserts action_space + observation_space non-None, max_episode_length>0, num_envs==128)
# Stdout swallowed by Omniverse log mux (expected); use exit code as truth.
```

---

## §2 Actions

### Description

Per-robot 3-D xyz EMA-smoothed EE-delta + binary gripper. Action vector concatenates `arm_action_0 (3) + gripper_action_0 (1) + arm_action_1 (3) + gripper_action_1 (1) = 8-D` total. The `EMACumulativeDeltaPositionActionCfg` class is IMPORTED from `insert_drawer/mdp` (not vendored locally). Controller is `DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls")` — absolute target pose mode where the action provides cumulative delta on top of the post-reset EE position. Per-robot workspace clamps in each robot's root frame after the `body_offset` shift; robot_0 reaches negative-y in its frame, robot_1 reaches positive-y (symmetric).

### Decisions resolved

| Knob | Value |
|---|---|
| Action class | `insert_drawer.mdp.EMACumulativeDeltaPositionActionCfg` for arm; `insert_drawer.mdp.BinaryJointPositionActionCfg` for gripper |
| Per-arm dim | 3 (xyz delta only — RPY locked at post-reset value) |
| Total action dim | 8 |
| Arm `joint_names` | `["fr3_joint.*"]` |
| Arm `body_name` | `"fr3_hand"` |
| Arm `body_offset` | `OffsetCfg(pos=(0.0, 0.0, 0.2))` |
| Controller | `DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls")` |
| `scale` | `(0.01, 0.01, 0.01)` per axis |
| `alpha` | `0.5` (EMA weight on new delta target) |
| robot_0 pos_lower_limit (root frame) | `[0.20, -0.65, 0.005]` |
| robot_0 pos_upper_limit | `[0.55, -0.20, 0.40]` |
| robot_1 pos_lower_limit | `[0.20, +0.20, 0.005]` |
| robot_1 pos_upper_limit | `[0.55, +0.65, 0.40]` |
| Gripper joints | `["fr3_finger.*"]`, open=`{"fr3_finger_.*": 0.04}`, close=`{"fr3_finger_.*": 0.0}` |

### Code (verbatim)

`lift_box_env_cfg.py:ActionsCfg` (skeleton — fields are filled by the franka subclass `__post_init__`):

```python
@configclass
class ActionsCfg:
    arm_action_0: "mdp.EMACumulativeDeltaPositionActionCfg" = MISSING
    gripper_action_0: "mdp.BinaryJointPositionActionCfg" = MISSING
    arm_action_1: "mdp.EMACumulativeDeltaPositionActionCfg" = MISSING
    gripper_action_1: "mdp.BinaryJointPositionActionCfg" = MISSING
```

`config/franka/joint_pos_env_cfg.py:FrankaLiftBoxEnvCfg.__post_init__` (action wiring):

```python
self.actions.arm_action_0 = insert_drawer_mdp.EMACumulativeDeltaPositionActionCfg(
    asset_name="robot_0",
    joint_names=["fr3_joint.*"],
    body_name="fr3_hand",
    body_offset=DifferentialInverseKinematicsActionCfg.OffsetCfg(pos=(0.0, 0.0, 0.2)),
    controller=DifferentialIKControllerCfg(
        command_type="pose", use_relative_mode=False, ik_method="dls",
    ),
    scale=(0.01, 0.01, 0.01),
    alpha=0.5,
    pos_lower_limit=[0.20, -0.65, 0.005],
    pos_upper_limit=[0.55, -0.20, 0.40],
)
self.actions.gripper_action_0 = insert_drawer_mdp.BinaryJointPositionActionCfg(
    asset_name="robot_0",
    joint_names=["fr3_finger.*"],
    open_command_expr={"fr3_finger_.*": 0.04},
    close_command_expr={"fr3_finger_.*": 0.0},
)
self.actions.arm_action_1 = insert_drawer_mdp.EMACumulativeDeltaPositionActionCfg(
    asset_name="robot_1",
    joint_names=["fr3_joint.*"],
    body_name="fr3_hand",
    body_offset=DifferentialInverseKinematicsActionCfg.OffsetCfg(pos=(0.0, 0.0, 0.2)),
    controller=DifferentialIKControllerCfg(
        command_type="pose", use_relative_mode=False, ik_method="dls",
    ),
    scale=(0.01, 0.01, 0.01),
    alpha=0.5,
    pos_lower_limit=[0.20, 0.20, 0.005],
    pos_upper_limit=[0.55, 0.65, 0.40],
)
self.actions.gripper_action_1 = insert_drawer_mdp.BinaryJointPositionActionCfg(
    asset_name="robot_1",
    joint_names=["fr3_finger.*"],
    open_command_expr={"fr3_finger_.*": 0.04},
    close_command_expr={"fr3_finger_.*": 0.0},
)
```

### Smoke

```bash
.venv/bin/python harbor/create-task/isaaclab-lift-box/smokes/smoke_s2.py
# exit 0 (asserts action_space.shape[-1] == 8, env.step(action) returns finite obs/reward)
```

---

## §3 Reset

### Description

Three reset terms — one per robot articulation (joint positions pinned to URDF home, no jitter, no init velocity) and one for the box (small ±3 cm xy jitter, z pinned to `BOX_INIT_Z`, no init velocity). All three terms explicitly bind `asset_cfg=SceneEntityCfg("robot_0"|"robot_1"|"box")` because the default `"robot"` does NOT exist in this dual-arm scene.

### Decisions resolved

| Term | Function | `position_range` / `pose_range` | `velocity_range` | asset_cfg |
|---|---|---|---|---|
| reset_robot_0_joints | `mdp.reset_joints_by_scale` | `position_range=(1.0, 1.0)` | `(0.0, 0.0)` | `SceneEntityCfg("robot_0")` |
| reset_robot_1_joints | `mdp.reset_joints_by_scale` | `position_range=(1.0, 1.0)` | `(0.0, 0.0)` | `SceneEntityCfg("robot_1")` |
| reset_box | `mdp.reset_root_state_uniform` | `pose_range={"x":(-0.03,0.03), "y":(-0.03,0.03), "z":(0.0,0.0)}` | `{}` | `SceneEntityCfg("box")` |

### Code (verbatim)

`lift_box_env_cfg.py:EventCfg`:

```python
@configclass
class EventCfg:
    """Reset terms only (§3). §7 DR is not in scope for this task-generator run."""

    reset_robot_0_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (1.0, 1.0),
            "velocity_range": (0.0, 0.0),
            "asset_cfg": SceneEntityCfg("robot_0"),
        },
    )
    reset_robot_1_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (1.0, 1.0),
            "velocity_range": (0.0, 0.0),
            "asset_cfg": SceneEntityCfg("robot_1"),
        },
    )
    reset_box = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.03, 0.03), "y": (-0.03, 0.03), "z": (0.0, 0.0)},
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("box"),
        },
    )
```

### Smoke

```bash
.venv/bin/python harbor/create-task/isaaclab-lift-box/smokes/smoke_s3.py
# exit 0 (asserts env.reset() places both robots at URDF home, box xy within ±3 cm of (0,0))
```

---

## §4 Goal + Termination

### Description

No `CommandsCfg` — the goal point is **hard-coded** in `lift_box/mdp/terminations.py` as a constant `target = (0, 0, BOX_INIT_Z + lift_height)`. Two terminations: `time_out` (after 200 steps) and `success` (box COM xy within 5 cm of target, z within 5 cm of `0.36025`, and `|lin_vel_w| < 0.10`). No failure-mode terminations (drop/tilt) — the policy is allowed to recover within the 10-s horizon.

### Decisions resolved

| Term | `func` | `params` | `time_out` flag |
|---|---|---|---|
| time_out | `mdp.time_out` | `{}` | `True` |
| success | `mdp.lift_box_success` | `{"target_xy": (0.0, 0.0), "lift_height": 0.25, "xy_pos_tol": 0.05, "z_pos_tol": 0.05, "vel_tol": 0.10, "box_cfg": SceneEntityCfg("box")}` | `False` |

`commands = None` at the EnvCfg level.

### Code (verbatim)

`lift_box_env_cfg.py:TerminationsCfg`:

```python
@configclass
class TerminationsCfg:
    """Terminations: `time_out` + `success` (box lifted to target xy/z with low
    velocity). Failure-mode terminations are intentionally not used."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    success = DoneTerm(
        func=mdp.lift_box_success,
        time_out=False,
        params={
            "target_xy": (0.0, 0.0),
            "lift_height": 0.25,
            "xy_pos_tol": 0.05,
            "z_pos_tol": 0.05,
            "vel_tol": 0.10,
            "box_cfg": SceneEntityCfg("box"),
        },
    )
```

`lift_box/mdp/terminations.py` (full source):

```python
"""Termination helpers for the lift_box task.

Goal: lift the box COM (xy) to the target xy AND z = `BOX_INIT_Z + lift_height`
AND box linear velocity is small (i.e. the lift has stabilized). The
`lift_box_success` termination fires when all three conditions hold.

The goal is HARD-CODED in this module (no CommandsCfg). `BOX_INIT_Z` matches
the constant used in `config/franka/joint_pos_env_cfg.py` for the box spawn
height (= half_z_extent of the recentered eurobox USD).
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


# Half-z extent of the eurobox after MeshConverter recentering. Must match the
# `BOX_INIT_Z` constant used in `joint_pos_env_cfg.py`.
BOX_INIT_Z: float = 0.11025


def lift_box_success(
    env: "ManagerBasedRLEnv",
    target_xy: tuple[float, float] = (0.0, 0.0),
    lift_height: float = 0.20,
    xy_pos_tol: float = 0.05,
    z_pos_tol: float = 0.05,
    vel_tol: float = 0.10,
    box_cfg: SceneEntityCfg = SceneEntityCfg("box"),
) -> torch.Tensor:
    """True when the box is centered (xy) within `xy_pos_tol` of `target_xy`,
    its z is within `z_pos_tol` of `BOX_INIT_Z + lift_height`, AND its linear
    velocity magnitude is below `vel_tol`. All quantities are in env-local
    world coordinates (root_pos_w - env_origins).
    """
    box: RigidObject = env.scene[box_cfg.name]
    box_pos_local = box.data.root_pos_w[:, :3] - env.scene.env_origins[:, :3]
    box_xy = box_pos_local[:, :2]
    box_z = box_pos_local[:, 2]

    target = torch.tensor(target_xy, device=env.device, dtype=box_xy.dtype)
    xy_err = torch.norm(box_xy - target.unsqueeze(0), dim=-1)
    z_err = torch.abs(box_z - (BOX_INIT_Z + lift_height))

    lin_vel_w = box.data.root_lin_vel_w[:, :3]
    vel_norm = torch.norm(lin_vel_w, dim=-1)

    return (xy_err < xy_pos_tol) & (z_err < z_pos_tol) & (vel_norm < vel_tol)
```

> **WARN — function default `lift_height=0.20`**: the function signature default is `0.20` but the `TerminationsCfg` and `RewardsCfg.success_bonus` both pass `lift_height=0.25` (env-local target z = `BOX_INIT_Z + 0.25 = 0.36025`). The function default is only used if `params=` omits the key — which neither cfg does. When reproducing into another repo, either align the default to `0.25` or keep the explicit `params={"lift_height": 0.25, ...}` on the consumer side.

### Smoke

```bash
.venv/bin/python harbor/create-task/isaaclab-lift-box/smokes/smoke_s4.py
.venv/bin/python harbor/create-task/isaaclab-lift-box/smokes/smoke_success.py
# Both exit 0. smoke_success writes box pose + zero velocity directly, runs one sub-step,
# then asserts termination_manager.compute() fires the success term in 128/128 envs.
# (env.step(zero_action) at decimation=6 lets free-fall velocity exceed vel_tol=0.10
# before the check — custom stepping required.)
```

---

## §5 Observation

### Description

Single `PolicyCfg` group, all terms concatenated → 33-D policy obs. Per-robot blocks (ee_pose, gripper_joint_pos) appear once per robot; the box state is shared (xyz + quat in env-local world frame); `last_action` carries the full 8-D action. `enable_corruption=True` (per-term `noise` slots default no-op; `dr-generator` may widen later). The per-robot `ee_pose_in_robot_root_frame` helper is a task-local variant of insert_drawer's that takes BOTH a `robot_cfg` and an `ee_frame_cfg` SceneEntityCfg so the same function works for either arm.

### Decisions resolved

| ObsTerm | `func` | `params` | dim |
|---|---|---|---|
| ee_pose_0 | `mdp.ee_pose_in_robot_root_frame` | `robot_cfg=SceneEntityCfg("robot_0"), ee_frame_cfg=SceneEntityCfg("ee_frame_0")` | 7 |
| ee_pose_1 | `mdp.ee_pose_in_robot_root_frame` | `robot_cfg=SceneEntityCfg("robot_1"), ee_frame_cfg=SceneEntityCfg("ee_frame_1")` | 7 |
| box_position_in_world | `mdp.box_position_in_world` | (defaults) | 3 |
| box_quat_in_world | `mdp.box_quat_in_world` | (defaults) | 4 |
| gripper_joint_pos_0 | `mdp.joint_pos` | `asset_cfg=SceneEntityCfg("robot_0", joint_names=["fr3_finger.*"])` | 2 |
| gripper_joint_pos_1 | `mdp.joint_pos` | `asset_cfg=SceneEntityCfg("robot_1", joint_names=["fr3_finger.*"])` | 2 |
| last_action | `mdp.last_action` | (defaults) | 8 |
| **Total** | | | **33** |

PolicyCfg flags: `enable_corruption = True`, `concatenate_terms = True`. No per-term noise slots populated (no-op).

### Code (verbatim)

`lift_box_env_cfg.py:ObservationsCfg`:

```python
@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        ee_pose_0 = ObsTerm(
            func=mdp.ee_pose_in_robot_root_frame,
            params={"robot_cfg": SceneEntityCfg("robot_0"), "ee_frame_cfg": SceneEntityCfg("ee_frame_0")},
        )
        ee_pose_1 = ObsTerm(
            func=mdp.ee_pose_in_robot_root_frame,
            params={"robot_cfg": SceneEntityCfg("robot_1"), "ee_frame_cfg": SceneEntityCfg("ee_frame_1")},
        )
        box_position_in_world = ObsTerm(func=mdp.box_position_in_world)
        box_quat_in_world = ObsTerm(func=mdp.box_quat_in_world)
        gripper_joint_pos_0 = ObsTerm(
            func=mdp.joint_pos,
            params={"asset_cfg": SceneEntityCfg("robot_0", joint_names=["fr3_finger.*"])},
        )
        gripper_joint_pos_1 = ObsTerm(
            func=mdp.joint_pos,
            params={"asset_cfg": SceneEntityCfg("robot_1", joint_names=["fr3_finger.*"])},
        )
        last_action = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()
```

`lift_box/mdp/observations.py` (full source — task-local helpers):

```python
"""Observation helpers for the lift_box task."""
from __future__ import annotations
from typing import TYPE_CHECKING
import torch

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformer
from isaaclab.utils.math import subtract_frame_transforms

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def ee_pose_in_robot_root_frame(
    env: "ManagerBasedRLEnv",
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot_0"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame_0"),
) -> torch.Tensor:
    """7-D end-effector pose [x, y, z, qw, qx, qy, qz] in the given robot's root frame."""
    robot: RigidObject = env.scene[robot_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    ee_pos_w = ee_frame.data.target_pos_w[..., 0, :]
    ee_quat_w = ee_frame.data.target_quat_w[..., 0, :]
    ee_pos_b, ee_quat_b = subtract_frame_transforms(
        robot.data.root_pos_w, robot.data.root_quat_w, ee_pos_w, ee_quat_w
    )
    return torch.cat([ee_pos_b, ee_quat_b], dim=-1)


def box_position_in_world(
    env: "ManagerBasedRLEnv",
    box_cfg: SceneEntityCfg = SceneEntityCfg("box"),
) -> torch.Tensor:
    """Box xyz in env-local world coordinates (root_pos_w - env_origin)."""
    box: RigidObject = env.scene[box_cfg.name]
    return box.data.root_pos_w[:, :3] - env.scene.env_origins[:, :3]


def box_quat_in_world(
    env: "ManagerBasedRLEnv",
    box_cfg: SceneEntityCfg = SceneEntityCfg("box"),
) -> torch.Tensor:
    """Box orientation quaternion (wxyz) in world frame."""
    box: RigidObject = env.scene[box_cfg.name]
    return box.data.root_quat_w[:, :4]
```

### Smoke

```bash
.venv/bin/python harbor/create-task/isaaclab-lift-box/smokes/smoke_s5.py
# exit 0 (asserts observation_space == Box(33,), per-term dims match the expected_obs.json layout)
```

---

## §6 Reward

### Description

Composer = **sum**. 7 active terms in a strictly-increasing per-stage magnitude ladder (the iter-0 set was scaled down 80× to the values shown — same ratios, same convergence behaviour). No regularizers (action_rate, joint_vel). No failure-mode penalties. The `success_bonus` is a per-env one-shot latch that fires on the FIRST step the success predicate is satisfied each episode; the latch resets at `episode_length_buf <= 1`.

Weights below are **nominal per-step magnitudes** — the declared weight is exactly what each term pays per step (and what shows up in `episodic_return_mean`).

### Decisions resolved

| RewTerm | `func` | `params` | `weight` |
|---|---|---|---:|
| ee_0_to_grasp_0 | `mdp.ee_to_grasp_distance` | `std=0.15, ee_frame_cfg=SceneEntityCfg("ee_frame_0"), grasp_frame_cfg=SceneEntityCfg("grasp_frame_0")` | **0.0125** |
| ee_1_to_grasp_1 | `mdp.ee_to_grasp_distance` | `std=0.15, ee_frame_cfg=SceneEntityCfg("ee_frame_1"), grasp_frame_cfg=SceneEntityCfg("grasp_frame_1")` | **0.0125** |
| grasp_contact_0 | `mdp.grasp_contact` | `robot_idx=0, contact_force_threshold=1e-3` | **0.025** |
| grasp_contact_1 | `mdp.grasp_contact` | `robot_idx=1, contact_force_threshold=1e-3` | **0.025** |
| lift_height | `mdp.lift_height` | `init_z=0.11025, target_lift=0.25, contact_force_threshold=1e-3` | **0.1875** |
| box_xy_align | `mdp.box_xy_align` | `std=0.15, target_xy=(0.0, 0.0), lift_threshold=0.05, init_z=0.11025` | **0.125** |
| success_bonus | `mdp.success_bonus` | `target_xy=(0.0, 0.0), lift_height=0.25, xy_pos_tol=0.05, z_pos_tol=0.05, vel_tol=0.10, init_z=0.11025` | **100.0** |

#### Per-stage saturated per-step magnitude budget (nominal weights)

| Stage | Term(s) | Per-step saturated | 200-step ep ceiling |
|---|---|---:|---:|
| Reach (×2) | ee_0_to_grasp_0, ee_1_to_grasp_1 | 0.025 | 5.0 |
| Contact (×2) | grasp_contact_0, grasp_contact_1 | 0.05 | 10.0 (when held) |
| Lift | lift_height | 0.1875 | 37.5 (dual contact) |
| Align | box_xy_align | 0.125 | 25.0 (box lifted) |
| Success | success_bonus | one-shot | +100 |

Dense ceiling ≈ 78; sparse success = 100. Sparse strictly dominates dense (ratio 100/78 ≈ 1.28) so the policy is incentivised to FINISH, not park.

### Code (verbatim)

`lift_box_env_cfg.py:RewardsCfg`:

```python
@configclass
class RewardsCfg:
    """Reward ladder (composer = sum, sign = positive=good).

    All weights scaled down 80x from the iter-0 converged set so total
    episodic return lands around ~105 instead of ~8400 — same ratios,
    same convergence behaviour, more interpretable magnitudes.
    """

    ee_0_to_grasp_0 = RewTerm(
        func=mdp.ee_to_grasp_distance,
        params={"std": 0.15, "ee_frame_cfg": SceneEntityCfg("ee_frame_0"),
                "grasp_frame_cfg": SceneEntityCfg("grasp_frame_0")},
        weight=0.0125,
    )
    ee_1_to_grasp_1 = RewTerm(
        func=mdp.ee_to_grasp_distance,
        params={"std": 0.15, "ee_frame_cfg": SceneEntityCfg("ee_frame_1"),
                "grasp_frame_cfg": SceneEntityCfg("grasp_frame_1")},
        weight=0.0125,
    )
    grasp_contact_0 = RewTerm(
        func=mdp.grasp_contact,
        params={"robot_idx": 0, "contact_force_threshold": 1e-3},
        weight=0.025,
    )
    grasp_contact_1 = RewTerm(
        func=mdp.grasp_contact,
        params={"robot_idx": 1, "contact_force_threshold": 1e-3},
        weight=0.025,
    )
    lift_height = RewTerm(
        func=mdp.lift_height,
        params={"init_z": 0.11025, "target_lift": 0.25, "contact_force_threshold": 1e-3},
        weight=0.1875,
    )
    box_xy_align = RewTerm(
        func=mdp.box_xy_align,
        params={"std": 0.15, "target_xy": (0.0, 0.0), "lift_threshold": 0.05, "init_z": 0.11025},
        weight=0.125,
    )
    success_bonus = RewTerm(
        func=mdp.success_bonus,
        params={"target_xy": (0.0, 0.0), "lift_height": 0.25, "xy_pos_tol": 0.05,
                "z_pos_tol": 0.05, "vel_tol": 0.10, "init_z": 0.11025},
        weight=100.0,
    )
```

`lift_box/mdp/rewards.py` (full source — 5 reward functions + 2 helpers + placeholder shim + per-env latch buffer registry):

```python
"""Reward functions for the dual-arm `lift_box` task."""
from __future__ import annotations
from typing import TYPE_CHECKING
import torch

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor, FrameTransformer

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

BOX_INIT_Z: float = 0.11025

# Module-level per-(env, key) latch buffers (success bonus one-shot).
_LATCH_BUFFERS: dict[tuple, torch.Tensor] = {}


def _get_latch_buffer(env, key: str) -> torch.Tensor:
    full_key = (id(env), key)
    if full_key not in _LATCH_BUFFERS:
        _LATCH_BUFFERS[full_key] = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    return _LATCH_BUFFERS[full_key]


def placeholder_zero(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """Legacy zero-reward shim."""
    return torch.zeros(env.num_envs, device=env.device)


# Phase 1 — dense EE -> grasp-point attractor (one per robot).
def ee_to_grasp_distance(
    env: "ManagerBasedRLEnv",
    std: float = 0.15,
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame_0"),
    grasp_frame_cfg: SceneEntityCfg = SceneEntityCfg("grasp_frame_0"),
) -> torch.Tensor:
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    grasp_frame: FrameTransformer = env.scene[grasp_frame_cfg.name]
    ee_w = ee_frame.data.target_pos_w[..., 0, :]
    grasp_w = grasp_frame.data.target_pos_w[..., 0, :]
    d = torch.norm(ee_w - grasp_w, dim=-1)
    return 1.0 - torch.tanh(d / max(std, 1e-6))


# Phase 2 — per-robot binary "both fingers in contact with box" gate.
def _both_fingers_in_contact(
    env: "ManagerBasedRLEnv",
    left_sensor_name: str,
    right_sensor_name: str,
    threshold: float,
) -> torch.Tensor:
    left: ContactSensor = env.scene[left_sensor_name]
    right: ContactSensor = env.scene[right_sensor_name]
    # force_matrix_w shape: (num_envs, n_bodies=1, n_filters=1, 3) per finger.
    left_f = torch.norm(left.data.force_matrix_w[:, 0, 0, :], dim=-1)
    right_f = torch.norm(right.data.force_matrix_w[:, 0, 0, :], dim=-1)
    return (left_f > threshold) & (right_f > threshold)


def grasp_contact(
    env: "ManagerBasedRLEnv",
    robot_idx: int = 0,
    contact_force_threshold: float = 1e-3,
) -> torch.Tensor:
    gate = _both_fingers_in_contact(
        env,
        left_sensor_name=f"finger_left_contact_{robot_idx}",
        right_sensor_name=f"finger_right_contact_{robot_idx}",
        threshold=contact_force_threshold,
    )
    return gate.float()


# Phase 3 — linear lift ramp gated on BOTH robots in dual contact.
def lift_height(
    env: "ManagerBasedRLEnv",
    init_z: float = BOX_INIT_Z,
    target_lift: float = 0.25,
    contact_force_threshold: float = 1e-3,
    box_cfg: SceneEntityCfg = SceneEntityCfg("box"),
) -> torch.Tensor:
    box: RigidObject = env.scene[box_cfg.name]
    box_pos_local = box.data.root_pos_w[:, :3] - env.scene.env_origins[:, :3]
    box_z = box_pos_local[:, 2]
    progress = ((box_z - init_z) / max(target_lift, 1e-6)).clamp(0.0, 1.0)
    gate_0 = _both_fingers_in_contact(env, "finger_left_contact_0", "finger_right_contact_0", contact_force_threshold)
    gate_1 = _both_fingers_in_contact(env, "finger_left_contact_1", "finger_right_contact_1", contact_force_threshold)
    dual_contact = (gate_0 & gate_1).float()
    return progress * dual_contact


# Phase 4 — tanh attractor on box xy -> target xy, gated on box lifted.
def box_xy_align(
    env: "ManagerBasedRLEnv",
    std: float = 0.15,
    target_xy: tuple[float, float] = (0.0, 0.0),
    lift_threshold: float = 0.05,
    init_z: float = BOX_INIT_Z,
    box_cfg: SceneEntityCfg = SceneEntityCfg("box"),
) -> torch.Tensor:
    box: RigidObject = env.scene[box_cfg.name]
    box_pos_local = box.data.root_pos_w[:, :3] - env.scene.env_origins[:, :3]
    box_xy = box_pos_local[:, :2]
    box_z = box_pos_local[:, 2]
    target = torch.tensor(target_xy, device=env.device, dtype=box_xy.dtype)
    d = torch.norm(box_xy - target.unsqueeze(0), dim=-1)
    base = 1.0 - torch.tanh(d / max(std, 1e-6))
    lifted = (box_z > (init_z + lift_threshold)).float()
    return lifted * base


# Phase 5 — one-shot success bonus mirroring the termination predicate.
def success_bonus(
    env: "ManagerBasedRLEnv",
    target_xy: tuple[float, float] = (0.0, 0.0),
    lift_height: float = 0.25,
    xy_pos_tol: float = 0.05,
    z_pos_tol: float = 0.05,
    vel_tol: float = 0.10,
    init_z: float = BOX_INIT_Z,
    box_cfg: SceneEntityCfg = SceneEntityCfg("box"),
) -> torch.Tensor:
    latch = _get_latch_buffer(env, "success_once")
    just_reset = env.episode_length_buf <= 1
    latch = torch.where(just_reset, torch.zeros_like(latch), latch)

    box: RigidObject = env.scene[box_cfg.name]
    box_pos_local = box.data.root_pos_w[:, :3] - env.scene.env_origins[:, :3]
    box_xy = box_pos_local[:, :2]
    box_z = box_pos_local[:, 2]
    target = torch.tensor(target_xy, device=env.device, dtype=box_xy.dtype)
    xy_err = torch.norm(box_xy - target.unsqueeze(0), dim=-1)
    z_err = torch.abs(box_z - (init_z + lift_height))
    lin_vel_w = box.data.root_lin_vel_w[:, :3]
    vel_norm = torch.norm(lin_vel_w, dim=-1)
    now_success = (xy_err < xy_pos_tol) & (z_err < z_pos_tol) & (vel_norm < vel_tol)

    fire = now_success & (~latch)
    latch = latch | now_success
    _LATCH_BUFFERS[(id(env), "success_once")] = latch
    return fire.float()
```

### Smoke

```bash
.venv/bin/python harbor/create-task/isaaclab-lift-box/smokes/smoke_s6.py
# exit 0 (asserts: 7 active terms registered, per-step reward finite + non-constant,
#         composer = sum (passthrough), reward mean within sensible bounds).
```

---

## §7 DR

`<no DR>` — `EventCfg` contains only the three reset terms above (no `mode="startup"` or `mode="interval"` randomization terms). The `dr-generator` was not run; `permit_env_edits=true` was authorized for reward-tune but the iter-0 reward fit entirely in §6, so no §7 wiring was added. To add later: `/harbor:create-task name=IsaacLab-Lift-Box description="add startup mass + friction + box pose DR" sections=7`.

---

## Reproduction

```bash
# Same source repo:
/harbor:create-task name=IsaacLab-Lift-Box-v2 from=harbor/create-task/isaaclab-lift-box-implementation.md

# Different repo: pass asset overrides if needed
/harbor:create-task name=BiArmLift from=isaaclab-lift-box-implementation.md \
  assets=path/to/dest/eurobox.usd,path/to/dest/fr3.usd,path/to/dest/table.usd
```

> probe-task: wrote `harbor/create-task/isaaclab-lift-box-implementation.md` (sections §1..§7, 5 reward funcs, 7 obs terms).
> Reproduce via: `/harbor:create-task name=<new_task_id> from=<output>`.
