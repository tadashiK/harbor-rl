# BoxLift — Implementation Spec

- robot: Bimanual UF850 arms + dual Allegro hands (44 DoF)
- simulator: IsaacLab (Isaac Sim, manager-based)
- objects: tote box (object_0), table
- bimanual: true
- summary: Two dexterous hands cooperatively grasp a tote box and lift it to a commanded pose.

> Source package name anonymized as `bimanual_suite`. This task comes from an internal
> bimanual manipulation suite rather than a public repo; the design below is otherwise verbatim.

**Task summary.** Two UF850 6-DoF arms, each capped with a 16-DoF Allegro hand
(`Robot` = right, `Robot_left` = left), cooperatively grasp a single rigid **tote/box**
(`object_0`) sitting on a fixed table and lift+reorient it to a commanded target pose. The policy
emits a 44-D action (22 joints × 2 arms) processed through an **EMA cumulative-relative
joint-position** action term (each step nudges a running joint-position target, EMA-smoothed and
clamped to per-joint limits). Success = the box stays within a position+orientation threshold of the
target for **20 consecutive steps**; the episode terminates on that consecutive-success count or on
time-out. The scene is built around a **C2 symmetry** system (left/right arm mirroring) wired in the
shared `BaseEnv`. The reward is a native weighted sum of eleven terms (dense bimanual palm
alignment, gated box goal position/orientation tracking, collision penalties, and a sparse
consecutive-success bonus); the term weights and functions are given in §6.

---

## §1 Registration + Scene

**Description.** Registered directly with gymnasium. The scene subclasses the shared
`BaseSceneCfg` (ground at z=−0.82, dome light, kinematic table) and adds both robot articulations,
the tote object (`object_0`), two per-robot contact sensors over the arm links, and two
`FrameTransformer`s (`tote_right`/`tote_left`) that publish an approach frame offset from the box.
Both robots spawn with **gravity disabled** on the rigid bodies. `replicate_physics = False` (needed
because the object uses a `MultiAssetSpawnerCfg` with `random_choice`). Sim: `dt = 1/120`,
`decimation = 6` (→ 20 Hz control), `episode_length_s = 8.3333` (~166 control steps),
`render_interval = decimation`. Physx: `gpu_max_rigid_contact_count = gpu_max_rigid_patch_count =
2**24`. Default `num_envs = 4096`, `env_spacing = 3.0`. Physics material:
static_friction=1.5, dynamic_friction=1.0, restitution=0.0. Asset roots: `bimanual_suite.LIB_PATH`
(= repo root, `Path(__file__).resolve().parent.parent`) and IsaacLab's `ISAAC_NUCLEUS_DIR`
(used only by the goal marker in the command term).

**Robot joint layout (per arm, 22 joints).** 6 arm joints `joint1..joint6`, then 16 Allegro-hand
joints ordered as index/middle/pinky finger groups `jif1..4, jmf1..4, jpf1..4` (12) + thumb
`jth1..4` (4). Action dim 44 = 22 × 2.

### Decisions resolved
| Question | Resolution |
|---|---|
| Gym id / entry point | `BoxLiftEnv-v0` → `bimanual_suite.env.tasks.BoxLift.env:BoxLiftEnv`, cfg `BoxLiftEnvCfg` |
| num_envs / spacing | 4096 / 3.0 |
| Right robot USD | `{LIB_PATH}/assets/ufactory850/uf850_allegro_right_colored.usd` |
| Left robot USD | `{LIB_PATH}/assets/ufactory850/uf850_allegro_left_colored.usd` |
| Right robot init pos | `(-0.274, -0.475, 0.01)`; Left `(-0.274, 0.475, 0.01)` (mirrored in y) |
| Robot rigid props | gravity **disabled**, max_depenetration_velocity=1000, max lin/ang vel=1000 |
| Robot articulation props | self-collisions off, solver_pos_iter=16, solver_vel_iter=1 |
| Object (`object_0`) | tote USD `{LIB_PATH}/assets/object/tote.usd`, scale=(0.8,0.6,1.0), mass=1.0 kg, dynamic rigid, contact sensors on, `MultiAssetSpawnerCfg(random_choice=True)` (single asset in list; commented-out cuboid alternatives present) |
| Object init pose | `(0.0, 0.0, 0.0)` (randomized at reset — see §3) |
| Table | `{LIB_PATH}/assets/object/table.usd`, **kinematic_enabled=True**, pos=(0,0,0), rot=(0.70710678,0,0,0.70710678) |
| Ground / light | GroundPlane at z=−0.82; DomeLight color=(0.75,0.75,0.75) intensity=2500 |
| Contact sensors | `contact_sensors_robot` on `Robot/link1|link2|link3|link4|link5`; `contact_sensors_robot_left` on the same left links; update_period=0, debug_vis=True (no explicit filter targets) |
| Frame transformers | `tote_right`/`tote_left` both source `Object_0`, target `approach_frame` offset from box (see code) |
| num_object | 1 |
| action_dim | 44 |
| Sim timing | dt=1/120, decimation=6, episode_length_s=8.3333, render_interval=6 |

**Code — registration (`bimanual_suite/env/__init__.py`):**
```python
from .tasks.BoxLift.env_cfg import BoxLiftEnvCfg
gym.register(
    id="BoxLiftEnv-v0",
    entry_point="bimanual_suite.env.tasks.BoxLift.env:BoxLiftEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": BoxLiftEnvCfg,
    },
)
```

**Code — `BoxLiftEnv` (`env.py`, full):**
```python
from __future__ import annotations

import torch
from typing import Any, ClassVar

from isaacsim.core.version import get_version
from isaaclab.envs.common import VecEnvStepReturn

from bimanual_suite.env.tasks.manager_based_env import BaseEnv
from bimanual_suite.env.tasks.BoxLift.env_cfg import BoxLiftEnvCfg


class BoxLiftEnv(BaseEnv):
    is_vector_env: ClassVar[bool] = True
    """Whether the environment is a vectorized environment."""
    metadata: ClassVar[dict[str, Any]] = {
        "render_modes": [None, "human", "rgb_array"],
        "isaac_sim_version": get_version(),
    }
    """Metadata for the environment."""

    cfg: BoxLiftEnvCfg
    """Configuration for the environment."""

    def step(self, action: torch.Tensor) -> VecEnvStepReturn:
        super().step(action)
        # Debug only
        if self.cfg.visualize_marker:
            from bimanual_suite.env.tasks.BoxLift.mdps import compute_side_points
            right_frame = compute_side_points(self.scene["object_0"].data.root_state_w, self.side_points, side="right")
            self.markers['arm_r']['ee_marker'].visualize(right_frame, self.scene["tote_right"].data.target_quat_w.reshape(-1, 4))
            left_frame = compute_side_points(self.scene["object_0"].data.root_state_w, self.side_points, side="left")
            self.markers['arm_l']['ee_marker'].visualize(left_frame, self.scene["tote_left"].data.target_quat_w.reshape(-1, 4))

        # return observations, rewards, resets and extras
        return self.obs_buf, self.reward_buf, self.reset_terminated, self.reset_time_outs, self.extras
```

**Code — `BoxLiftEnvCfg` top-level (`env_cfg.py`):**
```python
@configclass
class BoxLiftEnvCfg(BaseEnvCfg):
    name: str = "BoxLift"
    scene = BoxLiftSceneCfg(num_envs=4096, env_spacing=3.0)
    events = BoxLiftEventCfg()
    commands = BoxLiftCommandsCfg()
    observations = BoxLiftObservationsCfg()
    actions = BoxLiftActionsCfg()
    terminations = BoxLiftTerminationsCfg()
    rewards = BoxLiftRewardsCfg()
    num_object = 1
    action_dim = 44 # arm + hand
    action_scale: list = [0.05, 0.05, 0.05, 0.05, 0.05, 0.05,
                            0.03, 0.03, 0.03, 0.03,
                            0.03, 0.03, 0.03, 0.03,
                            0.03, 0.03, 0.03, 0.015,
                            0.03, 0.03, 0.03, 0.03,
                            0.05, 0.05, 0.05, 0.05, 0.05, 0.05,
                            0.03, 0.03, 0.03, 0.03,
                            0.03, 0.03, 0.03, 0.03,
                            0.03, 0.03, 0.03, 0.015,
                            0.03, 0.03, 0.03, 0.03]  # jth3 needs smaller rate

    visualize_marker: bool = False
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
```

**Code — scene (`env_cfg.py`, `BoxLiftSceneCfg`), robots / object / sensors / frames:**
```python
FRAME_MARKER_SMALL_CFG = FRAME_MARKER_CFG.copy()
FRAME_MARKER_SMALL_CFG.markers["frame"].scale = (0.10, 0.10, 0.10)

@configclass
class BoxLiftSceneCfg(BaseSceneCfg):
    # robots
    robot = ArticulationCfg(
        prim_path="/World/envs/env_.*/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{bimanual_suite.LIB_PATH}/assets/ufactory850/uf850_allegro_right_colored.usd",
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True,
                max_depenetration_velocity=1000.0,
                max_linear_velocity=1000,
                max_angular_velocity=1000,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False, solver_position_iteration_count=16, solver_velocity_iteration_count=1,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            joint_pos={
                "joint1": 0.05,
                "joint2": 0.0,
                "joint3": -0.5,
                "joint4": 1.4,
                "joint5": -1.0,
                "joint6": -3.14,
                # hand
                "jif1": 0.0, "jif2": 0.4, "jif3": 0.4, "jif4": 0.0,
                "jmf1": 0.0, "jmf2": 0.4, "jmf3": 0.4, "jmf4": 0.0,
                "jpf1": 0.0, "jpf2": 0.4, "jpf3": 0.4, "jpf4": 0.0,
                "jth1": 0.364, "jth2": 0.0, "jth3": 0.2, "jth4": 0.0,
            },
            pos=(-0.274, -0.475, 0.01),
        ),
        actuators={
            "xArm_1-6": ImplicitActuatorCfg(joint_names_expr=["joint[1-6]"], stiffness=2000.0, damping=16.0),
            "allegro_hand_1": ImplicitActuatorCfg(joint_names_expr=["j.*f1"], stiffness=325.0, damping=20.0),
            "allegro_hand_2": ImplicitActuatorCfg(joint_names_expr=["j.*f2"], stiffness=425.0, damping=25.0),
            "allegro_hand_3": ImplicitActuatorCfg(joint_names_expr=["j.*f3"], stiffness=245.0, damping=15.0),
            "allegro_hand_4": ImplicitActuatorCfg(joint_names_expr=["j.*f4"], stiffness=1050.0, damping=65.0),
            "allegro_hand_thumb_1": ImplicitActuatorCfg(joint_names_expr=["jth1"], stiffness=100.0, damping=5.0),
            "allegro_hand_thumb_2": ImplicitActuatorCfg(joint_names_expr=["jth2"], stiffness=300.0, damping=15.0),
            "allegro_hand_thumb_3": ImplicitActuatorCfg(joint_names_expr=["jth3"], stiffness=1270.0, damping=100.0),
            "allegro_hand_thumb_4": ImplicitActuatorCfg(joint_names_expr=["jth4"], stiffness=1000.0, damping=50.0),
        },
    )

    robot_left = ArticulationCfg(
        prim_path="/World/envs/env_.*/Robot_left",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{bimanual_suite.LIB_PATH}/assets/ufactory850/uf850_allegro_left_colored.usd",
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True,
                max_depenetration_velocity=1000.0,
                max_linear_velocity=1000,
                max_angular_velocity=1000,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False, solver_position_iteration_count=16, solver_velocity_iteration_count=1,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            joint_pos={
                "joint1": -0.05,
                "joint2": 0.0,
                "joint3": -0.5,
                "joint4": -1.4,
                "joint5": -1.0,
                "joint6": 3.14,
                # hand
                "jif1": 0.0, "jif2": 0.4, "jif3": 0.4, "jif4": 0.0,
                "jmf1": 0.0, "jmf2": 0.4, "jmf3": 0.4, "jmf4": 0.0,
                "jpf1": 0.0, "jpf2": 0.4, "jpf3": 0.4, "jpf4": 0.0,
                "jth1": 0.364, "jth2": 0.0, "jth3": 0.2, "jth4": 0.0,
            },
            pos=(-0.274, 0.475, 0.01),
        ),
        actuators={  # identical actuator groups as `robot`
            "xArm_1-6": ImplicitActuatorCfg(joint_names_expr=["joint[1-6]"], stiffness=2000.0, damping=16.0),
            "allegro_hand_1": ImplicitActuatorCfg(joint_names_expr=["j.*f1"], stiffness=325.0, damping=20.0),
            "allegro_hand_2": ImplicitActuatorCfg(joint_names_expr=["j.*f2"], stiffness=425.0, damping=25.0),
            "allegro_hand_3": ImplicitActuatorCfg(joint_names_expr=["j.*f3"], stiffness=245.0, damping=15.0),
            "allegro_hand_4": ImplicitActuatorCfg(joint_names_expr=["j.*f4"], stiffness=1050.0, damping=65.0),
            "allegro_hand_thumb_1": ImplicitActuatorCfg(joint_names_expr=["jth1"], stiffness=100.0, damping=5.0),
            "allegro_hand_thumb_2": ImplicitActuatorCfg(joint_names_expr=["jth2"], stiffness=300.0, damping=15.0),
            "allegro_hand_thumb_3": ImplicitActuatorCfg(joint_names_expr=["jth3"], stiffness=1270.0, damping=100.0),
            "allegro_hand_thumb_4": ImplicitActuatorCfg(joint_names_expr=["jth4"], stiffness=1000.0, damping=50.0),
        },
    )

    object_0 = RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Object_0",
        spawn=sim_utils.MultiAssetSpawnerCfg(
            assets_cfg=[
                # sim_utils.CuboidCfg(size=(0.25, 0.39, 0.155)),   # commented-out alternatives
                # sim_utils.CuboidCfg(size=(0.31, 0.405, 0.215)),
                # sim_utils.CuboidCfg(size=(0.19, 0.275, 0.215)),
                # sim_utils.CuboidCfg(size=(0.18, 0.26, 0.13)),
                sim_utils.UsdFileCfg(
                    usd_path=f"{bimanual_suite.LIB_PATH}/assets/object/tote.usd",
                    scale=(0.8, 0.6, 1.0),
                ),
            ],
            random_choice=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                solver_position_iteration_count=16, solver_velocity_iteration_count=1
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
            activate_contact_sensors=True,
            collision_props=sim_utils.CollisionPropertiesCfg(),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
    )

    # sensors
    contact_sensors_robot = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/link1|link2|link3|link4|link5",  # thumb
        update_period=0.0,
        debug_vis=True,
    )
    contact_sensors_robot_left = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot_left/link1|link2|link3|link4|link5",  # thumb
        update_period=0.0,
        debug_vis=True,
    )

    tote_right = FrameTransformerCfg(
        prim_path="{ENV_REGEX_NS}/Object_0",
        debug_vis=True,
        visualizer_cfg=FRAME_MARKER_SMALL_CFG.replace(prim_path="/Visuals/ToteRightFrameTransformer"),
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/Object_0",
                name="approach_frame",
                offset=OffsetCfg(pos=(0.0, -0.2, 0.1), rot=(0.500, 0.500, 0.500, 0.500)),
            ),
        ],
    )

    tote_left = FrameTransformerCfg(
        prim_path="{ENV_REGEX_NS}/Object_0",
        debug_vis=True,
        visualizer_cfg=FRAME_MARKER_SMALL_CFG.replace(prim_path="/Visuals/ToteLeftFrameTransformer"),
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/Object_0",
                name="approach_frame",
                offset=OffsetCfg(pos=(0.0, 0.2, 0.1), rot=(-0.500, 0.500, -0.500, 0.500)),
            ),
        ],
    )
```

**Code — shared `BaseSceneCfg` + `BaseEnvCfg` (`manager_based_env_cfg.py`): ground / light / table / sim:**
```python
@configclass
class BaseSceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -0.82)),
    )
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=2500.0),
    )
    table: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Table",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{bimanual_suite.LIB_PATH}/assets/object/table.usd",
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True,
                disable_gravity=False,
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=1,
                max_depenetration_velocity=10.0,
            ),
            scale=(1.0, 1.0, 1.0),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0), rot=(0.70710678, 0, 0., 0.70710678)),
    )
    replicate_physics = False

@configclass
class BaseEnvCfg(ManagerBasedRLEnvCfg):
    ...
    sim: SimulationCfg = SimulationCfg(
        physics_material=RigidBodyMaterialCfg(
            static_friction=1.5,
            dynamic_friction=1.0,
            restitution=0.0,
            restitution_combine_mode=min,
        ),
        physx=PhysxCfg(
            gpu_max_rigid_contact_count=2**24,
            gpu_max_rigid_patch_count=2**24,
        ),
    )
    def __post_init__(self):
        self.decimation = 6
        self.episode_length_s = 8.3333
        self.viewer.eye = (3.5, 3.5, 3.5)
        self.sim.dt = 1.0 / 120.0
        self.sim.render_interval = self.decimation
```

---

## §2 Actions

**Description.** Two action terms, one per arm, both `EMACumulativeRelativeJointPositionActionCfg`
over all 22 joints (`joint_names=[".*"]`). action_dim = 44 (22 right + 22 left). **Per-step rule
(see `process_actions`):** the raw policy action is first multiplied by `env.cfg.action_scale` in
`BaseEnv.step` (per-joint scale list, arm joints ×0.05, most hand joints ×0.03, `jth3` ×0.015), then
inside the action term: `processed += del_action` (cumulative delta), `del_action = processed`,
`processed += init_joint_pos`, then EMA blend `ema = alpha*processed + (1-alpha)*prev_applied`
(`alpha = 0.2`), clamp to `[joint_lower_limit, joint_upper_limit]`, and store as `prev_applied`. So
the command is a **running/cumulative** joint-position target seeded at the reset joint pose, EMA-
smoothed, and hard-clamped to per-joint limits. Per-arm limits differ only in the index/pinky sign
convention (`JOINT_*_LIMIT` vs `JOINT_*_LIMIT_LEFT`). `scale=1.0`, `use_default_offset=False`.

### Decisions resolved
| Question | Resolution |
|---|---|
| Action term type | `EMACumulativeRelativeJointPositionAction` (subclass of IsaacLab `JointPositionAction`) |
| # terms / joints | 2 terms (`robot`, `robot_left`), each `joint_names=[".*"]` → 22 joints |
| action_dim | 44 |
| alpha (EMA) | 0.2 (both arms) |
| scale (term) | 1.0; separate per-joint `action_scale` applied in `BaseEnv.step` (see list in §1) |
| use_default_offset | False |
| Per-joint clamp | right → `JOINT_LOWER_LIMIT/JOINT_UPPER_LIMIT`; left → `..._LEFT` variants |
| Control rate | 20 Hz (dt=1/120, decimation=6) |

**Code — action config (`env_cfg.py`, `BoxLiftActionsCfg`):**
```python
@configclass
class BoxLiftActionsCfg:
    arm_hand_action = EMACumulativeRelativeJointPositionActionCfg(
        asset_name="robot",
        joint_names=[".*"],
        scale=1.0,
        use_default_offset=False,
        joint_lower_limit=JOINT_LOWER_LIMIT,
        joint_upper_limit=JOINT_UPPER_LIMIT,
        alpha=0.2
    )
    arm_hand_action_left = EMACumulativeRelativeJointPositionActionCfg(
        asset_name="robot_left",
        joint_names=[".*"],
        scale=1.0,
        use_default_offset=False,
        joint_lower_limit=JOINT_LOWER_LIMIT_LEFT,
        joint_upper_limit=JOINT_UPPER_LIMIT_LEFT,
        alpha=0.2
    )
```

**Code — per-joint limits (`manager_based_env_cfg.py`):**
```python
JOINT_LOWER_LIMIT = [-6.283, -2.304, -4.224, -6.283, -2.164, -6.283,
                    -0.05, -0.05, -0.570, 0.364,      # jif1, jmf1, jpf1, jth1
                    -0.296, -0.296, -0.296, -0.205,   # jif2, jmf2, jpf2, jth2
                    -0.274, -0.274, -0.274, -0.290,   # jif3, jmf3, jpf3, jth3
                    -0.327, -0.327, -0.327, -0.262]   # jif4, jmf4, jpf4, jth4
JOINT_UPPER_LIMIT = [6.283, 2.304, 0.061, 6.283, 2.164, 6.283,
                    0.570, 0.05, 0.05, 1.497,
                    1.710, 1.710, 1.710, 1.130,
                    1.809, 1.809, 1.809, 1.633,
                    1.718, 1.718, 1.718, 1.820]
JOINT_LOWER_LIMIT_LEFT = [-6.283, -2.304, -4.224, -6.283, -2.164, -6.283,
                    -0.570, -0.05, -0.05, 0.364,
                    -0.296, -0.296, -0.296, -0.205,
                    -0.274, -0.274, -0.274, -0.290,
                    -0.327, -0.327, -0.327, -0.262]
JOINT_UPPER_LIMIT_LEFT = [6.283, 2.304, 0.061, 6.283, 2.164, 6.283,
                    0.05, 0.05, 0.570, 1.497,
                    1.710, 1.710, 1.710, 1.130,
                    1.809, 1.809, 1.809, 1.633,
                    1.718, 1.718, 1.718, 1.820]
```

**Code — action term config class (`action_managers/actions_cfg.py`):**
```python
@configclass
class EMACumulativeRelativeJointPositionActionCfg(JointPositionActionCfg):
    class_type: type[ActionTerm] = EMACumulativeRelativeJointPositionAction
    alpha: float | dict[str, float] = 1.0
    joint_lower_limit: list[float] = None
    joint_upper_limit: list[float] = None
```

**Code — action term impl (`action_managers/actions.py`, `EMACumulativeRelativeJointPositionAction`):**
```python
class EMACumulativeRelativeJointPositionAction(JointPositionAction):
    def __init__(self, cfg, env) -> None:
        super().__init__(cfg, env)
        if isinstance(cfg.alpha, float):
            if not 0.0 <= cfg.alpha <= 1.0:
                raise ValueError(f"Moving average weight must be in the range [0, 1]. Got {cfg.alpha}.")
            self._alpha = cfg.alpha
        elif isinstance(cfg.alpha, dict):
            self._alpha = torch.ones((env.num_envs, self.action_dim), device=self.device)
            index_list, names_list, value_list = string_utils.resolve_matching_names_values(cfg.alpha, self._joint_names)
            for name, value in zip(names_list, value_list):
                if not 0.0 <= value <= 1.0:
                    raise ValueError(f"Moving average weight must be in the range [0, 1]. Got {value} for joint {name}.")
            self._alpha[:, index_list] = torch.tensor(value_list, device=self.device)
        else:
            raise ValueError(f"Unsupported moving average weight type: {type(cfg.alpha)}. Supported types are float and dict.")
        self._prev_applied_actions = torch.zeros_like(self.processed_actions)
        self.del_action = torch.zeros((self._env.num_envs, self.action_dim), device=self._env.device)
        self.init_joint_pos = self._asset.data.joint_pos[:, self._joint_ids].clone()
        self.joint_lower_limit = torch.tensor(cfg.joint_lower_limit, device=self.device) if cfg.joint_lower_limit is not None else None
        self.joint_upper_limit = torch.tensor(cfg.joint_upper_limit, device=self.device) if cfg.joint_upper_limit is not None else None

    def reset(self, env_ids=None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        super().reset(env_ids)
        self._prev_applied_actions[env_ids, :] = self._asset.data.joint_pos[env_ids][:, self._joint_ids].clone()
        self.del_action[env_ids, :] = torch.zeros((env_ids.shape[0], self.action_dim), device=self.device)
        self.init_joint_pos[env_ids, :] = self._asset.data.joint_pos[env_ids][:, self._joint_ids].clone()

    def process_actions(self, actions: torch.Tensor):
        super().process_actions(actions)
        self._processed_actions += self.del_action
        self.del_action = self._processed_actions.clone()
        self._processed_actions += self.init_joint_pos.clone()
        ema_actions = self._alpha * self._processed_actions
        ema_actions += (1.0 - self._alpha) * self._prev_applied_actions
        if self.joint_lower_limit is not None and self.joint_upper_limit is not None:
            self._processed_actions[:] = torch.clamp(ema_actions, self.joint_lower_limit, self.joint_upper_limit)
        else:
            self._processed_actions[:] = ema_actions
        self._prev_applied_actions[:] = self._processed_actions[:]
```

**Code — action pre-scaling in the step loop (`manager_based_env.py`, `BaseEnv.step`):**
```python
def step(self, action: torch.Tensor) -> VecEnvStepReturn:
    self.last_action = action.clone()
    action = action * self._scale         # per-joint action_scale
    super().step(action)
    self.extras['success'] = self.success_tracker
    self.extras['detailed_reward'] = self.detailed_reward_buf
    reset_indices = torch.where(self.episode_length_buf == 1)[0]
    if len(reset_indices) > 0:
        self._post_reset_process(reset_indices)
    return self.obs_buf, self.reward_buf, self.reset_terminated, self.reset_time_outs, self.extras
```
`self._scale` is built in `_post_init_process` from `cfg.action_scale` (or ones if None).

---

## §3 Reset / Events

**Description.** `BoxLiftEventCfg` extends `BaseEventCfg`. Inherited `reset_robot_joints` (mode=reset,
`reset_joints_by_symmetry` on the default `robot`); task adds `reset_robot_joints_left`
(`mdp.reset_joints_by_scale` at scale (1,1)/(0,0) on `robot_left`), `reset_object` (randomizes
`object_0` pose: x∈[0,0.2], y∈[−0.1,0.1], z=0, yaw∈[−0.7,0.7], no velocity randomization), and a
**startup** event `compute_box_side_transform` that computes the box AABB and stores per-env
`side_points` (left/right grasp anchors) and `side_lengths` (half-extent along y) on the env. Note
`reset_joints_by_symmetry` mirrors joint state across the C2 group when `symmetry_tracker==1`.

### Decisions resolved
| Question | Resolution |
|---|---|
| Robot reset (right) | inherited `reset_robot_joints` → `reset_joints_by_symmetry`, pos/vel range (1,1)/(0,0), asset `robot` |
| Robot reset (left) | `reset_robot_joints_left` → `mdp.reset_joints_by_scale`, (1,1)/(0,0), asset `robot_left` |
| Object reset | `reset_object` (mode=reset): pose_range x[0,0.2] y[−0.1,0.1] z[0,0] yaw[−0.7,0.7], velocity_range {}, object_id 0 |
| Startup event | `lift.compute_box_side_transform` (mode=startup): object_id 0, axis "y" → sets `env.side_points`, `env.side_lengths` |

**Code — `BoxLiftEventCfg` (`env_cfg.py`):**
```python
@configclass
class BoxLiftEventCfg(BaseEventCfg):
    reset_robot_joints_left = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={"position_range": (1.0, 1.0), "velocity_range": (0.0, 0.0), "asset_cfg": SceneEntityCfg("robot_left")},
    )
    reset_object = EventTerm(
        func=reset_object,
        mode="reset",
        params={"pose_range": {"x": [0.0, 0.2], "y": [-0.1, 0.1], "z": [0.0, 0.0], "yaw": [-0.7, 0.7]},
                "velocity_range": {}, "object_id": 0},
    )
    compute_box_side_transform = EventTerm(
        func=lift.compute_box_side_transform,
        mode="startup",
        params={"object_id": 0, "axis": "y"},
    )
```

**Code — inherited `BaseEventCfg` (`manager_based_env_cfg.py`):**
```python
@configclass
class BaseEventCfg:
    reset_robot_joints = EventTerm(
        func=reset_joints_by_symmetry,
        mode="reset",
        params={"position_range": (1.0, 1.0), "velocity_range": (0.0, 0.0)},
    )
```

**Code — `reset_object` (`reset_mdps.py`):**
```python
def reset_object(env, env_ids, pose_range, velocity_range, object_id):
    object = env.scene[f"object_{object_id}"]
    root_states = object.data.default_root_state[env_ids].clone()
    range_list = [pose_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
    ranges = torch.tensor(range_list, device=object.device)
    rand_samples = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=object.device)
    positions = root_states[:, 0:3] + env.scene.env_origins[env_ids] + rand_samples[:, 0:3]
    orientations_delta = math_utils.quat_from_euler_xyz(rand_samples[:, 3], rand_samples[:, 4], rand_samples[:, 5])
    orientations = math_utils.quat_mul(root_states[:, 3:7], orientations_delta)
    range_list = [velocity_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
    ranges = torch.tensor(range_list, device=object.device)
    rand_samples = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=object.device)
    velocities = root_states[:, 7:13] + rand_samples
    object.write_root_pose_to_sim(torch.cat([positions, orientations], dim=-1), env_ids=env_ids)
    object.write_root_velocity_to_sim(velocities, env_ids=env_ids)
```

**Code — `compute_box_side_transform` + helpers (`BoxLift/mdps.py`):**
```python
def compute_box_side_transform(env, env_ids, object_id: int = 0, axis: Literal["x","y","z"] = "x"):
    """Compute the side position of the box."""
    object: RigidObject = env.scene[f"object_{object_id}"]
    cache = bounds_utils.create_bbox_cache()
    side_lengths = []
    side_points = []
    for i in range(len(object.root_physx_view.prim_paths)):
        min_x, min_y, min_z, max_x, max_y, max_z = bounds_utils.compute_combined_aabb(cache, prim_paths=[object.root_physx_view.prim_paths[i]])
        side_point = torch.tensor([[(min_x+max_x)/2, min_y, (min_z+max_z)/2], [(min_x+max_x)/2, max_y, (min_z+max_z)/2]], device=env.device, dtype=torch.float32)
        side_point -= env.scene.env_origins[i]
        side_points.append(side_point.unsqueeze(0))
        if axis == "x":   side_lengths.append((max_x - min_x)/2)
        elif axis == "y": side_lengths.append((max_y - min_y)/2)
        elif axis == "z": side_lengths.append((max_z - min_z)/2)
    env.side_points = torch.cat(side_points, dim=0)
    env.side_lengths = torch.tensor(side_lengths, device=env.device, dtype=torch.float32).reshape(-1, 1)

def compute_side_points(object_state_w, side_points, side: Literal["left","right"] = "right"):
    object_rot_mat = math_utils.matrix_from_quat(object_state_w[:, 3:7])
    object_pos_w = object_state_w[:, :3]
    side_points_rotated = torch.bmm(object_rot_mat, side_points.transpose(1, 2)).transpose(1, 2)
    side_points_w = side_points_rotated + object_pos_w.unsqueeze(1)
    if side == "right":  return side_points_w[:, 0, :]
    elif side == "left": return side_points_w[:, 1, :]
    else: raise ValueError(f"Invalid side: {side}")
```

**Code — `reset_joints_by_symmetry` (`reset_mdps.py`):**
```python
def reset_joints_by_symmetry(env, env_ids, position_range, velocity_range, asset_cfg=SceneEntityCfg("robot")):
    asset: Articulation = env.scene[asset_cfg.name]
    joint_pos = asset.data.default_joint_pos[env_ids].clone()
    joint_vel = asset.data.default_joint_vel[env_ids].clone()
    symmetric_idx = torch.where(env.symmetry_tracker[env_ids] == 1)[0]
    env_symmetric_idx = env_ids[symmetric_idx]
    if asset_cfg.name == "robot":
        symmetric_joint_pos = env.scene["robot_left"].data.default_joint_pos[env_symmetric_idx].clone() @ env.rep_Q_js
        symmetric_joint_vel = env.scene["robot_left"].data.default_joint_vel[env_symmetric_idx].clone() @ env.rep_Q_js
    else:
        symmetric_joint_pos = env.scene["robot"].data.default_joint_pos[env_symmetric_idx].clone() @ env.rep_Q_js
        symmetric_joint_vel = env.scene["robot"].data.default_joint_vel[env_symmetric_idx].clone() @ env.rep_Q_js
    joint_pos[symmetric_idx] = symmetric_joint_pos
    joint_vel[symmetric_idx] = symmetric_joint_vel
    joint_pos *= math_utils.sample_uniform(*position_range, joint_pos.shape, joint_pos.device)
    joint_vel *= math_utils.sample_uniform(*velocity_range, joint_vel.shape, joint_vel.device)
    joint_pos_limits = asset.data.soft_joint_pos_limits[env_ids]
    joint_pos = joint_pos.clamp_(joint_pos_limits[..., 0], joint_pos_limits[..., 1])
    joint_vel_limits = asset.data.soft_joint_vel_limits[env_ids]
    joint_vel = joint_vel.clamp_(-joint_vel_limits, joint_vel_limits)
    asset.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
```

---

## §4 Goal + Termination

**Description.** One command term `target_pos` = `TargetPositionCommandCfg` (object_id 0). It samples a
**relative goal position** in the env frame (x=0.15, y=0.0, z=0.18 — deterministic since ranges are
single-valued) and a goal orientation (roll/pitch/yaw all 0 → identity delta from the object's
default root quat). `return_type="pos"` so `command_manager.get_command("target_pos")` returns the
3-D position only. No time-based resampling (`resampling_time_range=(1e6,1e6)`);
`update_goal_on_success=True` resamples the goal once reached. `success_threshold=0.05` (position),
`success_threshold_orient=0.9` (dot of z-axes, ~"60 degree"). Because `return_type=="pos"`, the
command's `_update_metrics` uses the **position-only** success rule
(`position_error < success_threshold`) for `consecutive_success`.

**Termination:** `time_out` (inherited, `mdp.time_out`, time_out=True) + `max_consecutive_success`
(`num_success=20`, command `target_pos`): episode terminates once the box has been within threshold
for 20 consecutive control steps; this also writes `env.success_tracker`. No explicit failure/drop
termination term.

### Decisions resolved
| Question | Resolution |
|---|---|
| Command term | `target_pos` = `TargetPositionCommandCfg`, object_id 0, return_type "pos" |
| Goal pose range | x[0.15,0.15], y[0,0], z[0.18,0.18] (relative, in env frame); orientation delta identity |
| success_threshold (pos) | 0.05 m |
| success_threshold_orient | 0.9 (z-axis dot; comment "60 degree") |
| update_goal_on_success | True |
| resampling | none by time (`(1e6, 1e6)`) |
| Success predicate | pos-only branch (`return_type=="pos"`): `position_error < 0.05`, counted consecutively |
| Termination — success | `max_consecutive_success(num_success=20, command="target_pos")` |
| Termination — time_out | inherited `mdp.time_out(time_out=True)`, episode_length_s=8.3333 |
| Failure termination | none |
| Marker | goal frame marker (`frame_prim.usd` from `ISAAC_NUCLEUS_DIR`), debug_vis=True |

**Code — `BoxLiftCommandsCfg` + `BoxLiftTerminationsCfg` (`env_cfg.py`):**
```python
@configclass
class BoxLiftCommandsCfg(BaseCommandsCfg):
    target_pos = TargetPositionCommandCfg(
        object_id=0,
        success_threshold=0.05,
        success_threshold_orient=0.9, # 60 degree
        pose_range={"x": [0.15, 0.15], "y": [0.0, 0.0], "z": [0.18, 0.18]},
        update_goal_on_success=True,
        debug_vis=True,
        return_type="pos",
    )

@configclass
class BoxLiftTerminationsCfg(BaseTerminationsCfg):
    max_consecutive_success = DoneTerm(
        func=max_consecutive_success, params={"num_success": 20, "command_names": "target_pos"}
    )
```
(`BaseTerminationsCfg` supplies `time_out = DoneTerm(func=mdp.time_out, time_out=True)`.)

**Code — `max_consecutive_success` (`termination_mdps.py`):**
```python
def max_consecutive_success(env, num_success: int, command_names) -> torch.Tensor:
    if isinstance(command_names, str):
        command_term = env.command_manager.get_term(command_names)
        success = command_term.metrics["consecutive_success"] >= num_success
    else:
        success = torch.ones(env.num_envs, device=env.device)
        for command_name in command_names:
            command_term = env.command_manager.get_term(command_name)
            success = torch.logical_and(success, command_term.metrics["consecutive_success"] >= num_success)
    env.success_tracker = success.float()
    return success
```

**Code — `TargetPositionCommand` metrics/resample (`command_mdps/grasp_command.py`, key methods):**
```python
def _update_metrics(self):
    from bimanual_suite.utils.isaac_utils import get_angle_from_quat
    target_axis = get_angle_from_quat(self.quat_command_w, axis="z", normalize=True)
    cur_axis = get_angle_from_quat(self.object.data.root_quat_w, axis="z", normalize=True)
    self.metrics["orientation_error"] = torch.sum(target_axis * cur_axis, dim=-1)
    self.metrics["position_error"] = torch.norm(self.object.data.root_pos_w - self.pos_command_w, dim=1)
    if self.ranges is not None and self.return_type != "pos":
        successes = (torch.logical_and(self.metrics["position_error"] < self.cfg.success_threshold,
                                       self.metrics["orientation_error"] > self.cfg.success_threshold_orient)).float()
    else:
        successes = (self.metrics["position_error"] < self.cfg.success_threshold).float()
    unsuccesses = torch.where(successes == 0.0)[0]
    self.metrics["consecutive_success"] += successes.float()
    self.metrics["consecutive_success"][unsuccesses] = 0.0

def _resample_command(self, env_ids):
    if self.ranges is not None:
        rand_samples = math_utils.sample_uniform(self.ranges[:, 0], self.ranges[:, 1], (len(env_ids), 6), device=self.device)
        if self.cfg.offset:
            if isinstance(self.cfg.object_id, str) or self.cfg.use_initial_pose:
                rand_samples[:, 0:3] += self.object.data.root_pos_w[env_ids] - self._env.scene.env_origins[env_ids]
            else:
                rand_samples[:, 0:3] += self.env.object_init_pos[self.cfg.object_id, env_ids] - self._env.scene.env_origins[env_ids]
        self.pos_command_e[env_ids] = rand_samples[:, 0:3]
        self.pos_command_w[env_ids] = self.pos_command_e[env_ids] + self._env.scene.env_origins[env_ids]
        orientations_delta = math_utils.quat_from_euler_xyz(rand_samples[:, 3], rand_samples[:, 4], rand_samples[:, 5])
        self.quat_command_w[env_ids] = math_utils.quat_mul(self.object.data.default_root_state[env_ids, 3:7], orientations_delta)
    ...

def _update_command(self):
    if self.cfg.update_goal_on_success:
        if self.ranges is not None:
            goal_resets = torch.logical_and(self.metrics["position_error"] < self.cfg.success_threshold,
                                            self.metrics["orientation_error"] > self.cfg.success_threshold_orient)
        ...
        goal_reset_ids = goal_resets.nonzero(as_tuple=False).squeeze(-1)
        self._resample(goal_reset_ids)
```
(`TargetPositionCommandCfg`: `resampling_time_range=(1e6,1e6)`, goal marker from `{ISAAC_NUCLEUS_DIR}/Props/UIElements/frame_prim.usd` scale (0.1,0.1,0.1).)

---

## §5 Observation

**Description.** Single `policy` group, `concatenate_terms=True`, **`enable_corruption=False`** (so the
per-term Gaussian/uniform noise configs are declared but NOT applied at runtime). Bimanual: separate
normalized joint-position + joint-velocity terms per arm, box pose (pos + rotation-matrix-flattened
quat), per-side box grasp anchor points and approach-frame quats, the box half-length, and last
action. Joint-pos terms use explicit per-arm limits (`JOINT_*_LIMIT` / `..._LEFT`) via
`joint_pos_limit_normalized`. **Dim = 169** (see table).

### Decisions resolved (per-term dims)
| Term | func | params | dim |
|---|---|---|---|
| joint_pos_right | `joint_pos_limit_normalized` | joints None, limits `JOINT_*_LIMIT` | 22 |
| joint_vel_right | `joint_vel` | joints None | 22 |
| joint_pos_left | `joint_pos_limit_normalized` | limits `JOINT_*_LIMIT_LEFT`, asset `robot_left` | 22 |
| joint_vel_left | `joint_vel` | asset `robot_left` | 22 |
| box_pos | `object_pos` | object_id 0 | 3 |
| box_quat | `object_quat` | object_id 0, symmetry True → R-matrix flat | 9 |
| box_pos_right | `lift.box_side` | side "right" | 3 |
| frame_quat_right | `frame_quat` | frame "tote_right", symmetry True | 9 |
| box_pos_left | `lift.box_side` | side "left" | 3 |
| frame_quat_left | `frame_quat` | frame "tote_left", symmetry True | 9 |
| box_length | `lift.box_length` | — | 1 |
| last_action | `last_action` | — | 44 |
| **Total** | | | **169** |
| Flags | `enable_corruption=False`, `concatenate_terms=True` | | |

**Code — `BoxLiftObservationsCfg` (`env_cfg.py`):**
```python
@configclass
class BoxLiftObservationsCfg(BaseObservationsCfg):
    @configclass
    class PolicyCfg(ObsGroup):
        joint_pos_right = ObsTerm(func=joint_pos_limit_normalized, params={"joints": None,
                                    "joint_lower_limit": JOINT_LOWER_LIMIT,
                                    "joint_upper_limit": JOINT_UPPER_LIMIT}, noise=Gnoise(std=0.005))
        joint_vel_right = ObsTerm(func=joint_vel, params={"joints": None},)
        joint_pos_left = ObsTerm(func=joint_pos_limit_normalized, params={"joints": None,
                                    "joint_lower_limit": JOINT_LOWER_LIMIT_LEFT,
                                    "joint_upper_limit": JOINT_UPPER_LIMIT_LEFT,
                                    "asset_cfg": SceneEntityCfg("robot_left")}, noise=Gnoise(std=0.005))
        joint_vel_left = ObsTerm(func=joint_vel, params={"joints": None, "asset_cfg": SceneEntityCfg("robot_left")},)
        box_pos = ObsTerm(func=object_pos, noise=Unoise(n_min=0.0, n_max=0.01), params={"object_id": 0})
        box_quat = ObsTerm(func=object_quat, params={"object_id": 0, "symmetry": True}, noise=Gnoise(std=0.01))
        box_pos_right = ObsTerm(func=lift.box_side, params={"side": "right"})
        frame_quat_right = ObsTerm(func=frame_quat, params={"frame_name": "tote_right", "symmetry": True})
        box_pos_left = ObsTerm(func=lift.box_side, params={"side": "left"})
        frame_quat_left = ObsTerm(func=frame_quat, params={"frame_name": "tote_left", "symmetry": True})
        box_length = ObsTerm(func=lift.box_length, noise=Unoise(n_min=0.0, n_max=0.03))
        last_action = ObsTerm(func=last_action)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()
```

**Code — obs helpers (`obs_mdps.py`, relevant) + `BoxLift/mdps.py` box terms:**
```python
def joint_pos_limit_normalized(env, asset_cfg=SceneEntityCfg("robot"), joints=None, joint_lower_limit=None, joint_upper_limit=None):
    asset: Articulation = env.scene[asset_cfg.name]
    joint_ids = asset_cfg.joint_ids if joints is None else asset.find_joints(joints)[0]
    joint_lower_limit = asset.data.soft_joint_pos_limits[:, joint_ids, 0] if joint_lower_limit is None else torch.tensor(joint_lower_limit, device=env.device)
    joint_upper_limit = asset.data.soft_joint_pos_limits[:, joint_ids, 1] if joint_upper_limit is None else torch.tensor(joint_upper_limit, device=env.device)
    assert len(joint_lower_limit) == len(joint_upper_limit)
    return math_utils.scale_transform(asset.data.joint_pos[:, joint_ids], joint_lower_limit, joint_upper_limit)

def joint_vel(env, asset_cfg=SceneEntityCfg("robot"), joints=None):
    asset: Articulation = env.scene[asset_cfg.name]
    joint_ids = asset_cfg.joint_ids if joints is None else asset.find_joints(joints)[0]
    return asset.data.joint_vel[:, joint_ids]

def object_pos(env, object_id: int = 0):
    object = env.scene[f"object_{object_id}"]
    return object.data.root_pos_w - env.scene.env_origins

def object_quat(env, make_quat_unique=False, object_id=0, symmetry=True):
    object = env.scene[f"object_{object_id}"]
    if symmetry:
        object_R = math_utils.matrix_from_quat(object.data.root_quat_w)
        return object_R.transpose(1, 2).reshape(-1, 9)
    else:
        return math_utils.quat_unique(object.data.root_quat_w) if make_quat_unique else object.data.root_quat_w

def frame_quat(env, frame_name, make_quat_unique=False, symmetry=True):
    if symmetry:
        frame_R = math_utils.matrix_from_quat(env.scene[frame_name].data.target_quat_w.reshape(-1, 4))
        return frame_R.transpose(1, 2).reshape(-1, 9)
    else:
        return math_utils.quat_unique(env.scene[frame_name].data.target_quat_w.reshape(-1, 4)) if make_quat_unique else env.scene[frame_name].data.target_quat_w.reshape(-1, 4)

def last_action(env):
    if hasattr(env, "last_action"):
        return env.last_action
    else:
        return torch.zeros((env.num_envs, env.action_dim), device=env.device)

# --- BoxLift/mdps.py ---
def box_length(env):
    if not hasattr(env, "side_lengths"):
        return torch.zeros((env.num_envs, 1), device=env.device, dtype=torch.float32)
    return env.side_lengths

def box_side(env, side: Literal["left","right"]):
    if not hasattr(env, "side_points"):
        return torch.zeros((env.num_envs, 3), device=env.device, dtype=torch.float32)
    return compute_side_points(env.scene["object_0"].data.root_state_w, env.side_points, side) - env.scene.env_origins
```

---

## §6 Reward

**Description.** `BoxLiftRewardsCfg` extends `BaseRewardsCfg` — a native weighted sum of eleven
terms with the per-step weights shown below. Reward-manager composition is additive (weighted sum)
as in IsaacLab.

The tracking rewards are **gated** by two bimanual alignment predicates:
`if_aligned_quat` (both palms' quat error to their approach frame < 0.5) AND
`if_aligned_pos` (both palms within 0.1 m of their side grasp anchor). Collision punishment fires on
any arm-link contact force > 1 N.

### Decisions resolved (term → func / params / weight)
| Term | func | params | weight |
|---|---|---|---|
| align_hand_to_pos | `lift.align_palm_to_pos` | link `palm_link`, side "right", asset `robot` | 0.3 |
| align_hand_to_quat | `align_palm_to_quat` | link `palm_link`, frame `tote_right`, asset `robot` | 0.05 |
| align_hand_to_pos_left | `lift.align_palm_to_pos` | link `palm_link`, side "left", asset `robot_left` | 0.3 |
| align_hand_to_quat_left | `align_palm_to_quat` | link `palm_link`, frame `tote_left`, asset `robot_left` | 0.05 |
| object_goal_tracking | `lift.object_goal_distance` | command `target_pos`, object_id 0 | 20.0 |
| object_goal_orient_tracking | `lift.object_goal_orient_distance` | object_id 0, command `target_pos` | 4.0 |
| punish_collision | `lift.punish_collision` | sensor `contact_sensors_robot` | -10.0 |
| punish_collision_left | `lift.punish_collision` | sensor `contact_sensors_robot_left` | -10.0 |
| success_bonus | `success_bonus` | command `target_pos`, num_success 20 | 2000.0 |
| energy (inherited) | `energy_punishment` | asset `robot`, allegro actuators | (0.0 — not set) |
| energy_left (inherited) | `energy_punishment` | asset `robot_left`, allegro actuators | (0.0 — not set) |

### Symmetric-learning reward terms

The task's shared base supports **symmetric learning** (`base.yaml` `symmetry.symmetric_envs: True`,
C2 group): reward terms come as a **right set + their `_left` counterparts** — for the two-arm
BoxLift task both are genuine (each arm has its own palm-alignment + collision terms; both are kept).
For boxLift the reward config defines **NO `_symmetry`-suffixed duplicates** (unlike some sibling
bimanual_suite tasks), so there is nothing to drop here. When adapting **other** bimanual_suite tasks, drop any
`_symmetry`-suffixed reward terms when not using symmetric learning.

**Code — `BoxLiftRewardsCfg` (`env_cfg.py`):**
```python
@configclass
class BoxLiftRewardsCfg(BaseRewardsCfg):
    align_hand_to_pos = RewTerm(func=lift.align_palm_to_pos,
        params={"link_name": ["palm_link"], "side": "right", "asset_cfg": SceneEntityCfg("robot")}, weight=0.3)
    align_hand_to_quat = RewTerm(func=align_palm_to_quat,
        params={"link_name": ["palm_link"], "frame_name": "tote_right", "asset_cfg": SceneEntityCfg("robot")}, weight=0.05)
    align_hand_to_pos_left = RewTerm(func=lift.align_palm_to_pos,
        params={"link_name": ["palm_link"], "side": "left", "asset_cfg": SceneEntityCfg("robot_left")}, weight=0.3)
    align_hand_to_quat_left = RewTerm(func=align_palm_to_quat,
        params={"link_name": ["palm_link"], "frame_name": "tote_left", "asset_cfg": SceneEntityCfg("robot_left")}, weight=0.05)
    object_goal_tracking = RewTerm(func=lift.object_goal_distance,
        params={"command_name": "target_pos", "object_id": 0}, weight=20.0)
    object_goal_orient_tracking = RewTerm(func=lift.object_goal_orient_distance,
        params={"object_id": 0, "command_name": "target_pos"}, weight=4.0)
    punish_collision = RewTerm(func=lift.punish_collision,
        params={"sensor": "contact_sensors_robot"}, weight=-10.0)
    punish_collision_left = RewTerm(func=lift.punish_collision,
        params={"sensor": "contact_sensors_robot_left"}, weight=-10.0)
    success_bonus = RewTerm(func=success_bonus,
        params={"command_names": "target_pos", "num_success": 20}, weight=2000.0)
```

**Code — inherited `BaseRewardsCfg` (`manager_based_env_cfg.py`):**
```python
@configclass
class BaseRewardsCfg:
    energy = RewTerm(func=energy_punishment, weight=0.0,
        params={"asset_cfg": SceneEntityCfg("robot"), "actuator_name": ["allegro_hand_1","allegro_hand_2","allegro_hand_3","allegro_hand_4",
            "allegro_hand_thumb_1","allegro_hand_thumb_2","allegro_hand_thumb_3","allegro_hand_thumb_4"]})
    energy_left = RewTerm(func=energy_punishment, weight=0.0,
        params={"asset_cfg": SceneEntityCfg("robot_left"), "actuator_name": ["allegro_hand_1","allegro_hand_2","allegro_hand_3","allegro_hand_4",
            "allegro_hand_thumb_1","allegro_hand_thumb_2","allegro_hand_thumb_3","allegro_hand_thumb_4"]})
```

**Code — task reward functions (`BoxLift/mdps.py`):**
```python
def align_palm_to_pos(env, link_name, asset_cfg=SceneEntityCfg("robot"), side: Literal["left","right"]="right"):
    robot: Articulation = env.scene[asset_cfg.name]
    link_idx = robot.find_bodies(link_name)[0]
    robot_state_w = robot.data.body_state_w[:, link_idx, :7].reshape(-1, 7)
    side_pos = compute_side_points(env.scene["object_0"].data.root_state_w, env.side_points, side)
    distance = torch.norm(robot_state_w[:, :3] - side_pos, dim=-1)
    return -distance

def if_aligned_quat(env, link_names=["palm_link","palm_link"], frame_names=["tote_right","tote_left"],
                    asset_cfgs=[SceneEntityCfg("robot"), SceneEntityCfg("robot_left")], error_threshold: float = 0.5):
    aligned = []
    for i in range(len(link_names)):
        robot: Articulation = env.scene[asset_cfgs[i].name]
        link_idx = robot.find_bodies(link_names[i])[0]
        robot_state_w = robot.data.body_state_w[:, link_idx, :7].reshape(-1, 7)
        ori_distance = math_utils.quat_error_magnitude(robot_state_w[:, 3:7], env.scene[frame_names[i]].data.target_quat_w.reshape(-1, 4))
        aligned.append(torch.where(ori_distance < error_threshold, 1.0, 0.0))
    return torch.logical_and(aligned[0], aligned[1])

def if_aligned_pos(env, link_names=["palm_link","palm_link"], frame_names=["right","left"],
                   asset_cfgs=[SceneEntityCfg("robot"), SceneEntityCfg("robot_left")], error_threshold: float = 0.1):
    aligned = []
    for i in range(len(link_names)):
        robot: Articulation = env.scene[asset_cfgs[i].name]
        link_idx = robot.find_bodies(link_names[i])[0]
        robot_state_w = robot.data.body_state_w[:, link_idx, :7].reshape(-1, 7)
        side_pos = compute_side_points(env.scene["object_0"].data.root_state_w, env.side_points, frame_names[i])
        distance = torch.norm(robot_state_w[:, :3] - side_pos, dim=-1)
        aligned.append(torch.where(distance < error_threshold, 1.0, 0.0))
    return torch.logical_and(aligned[0], aligned[1])

def object_goal_distance(env, command_name, object_id: int = 0):
    object: RigidObject = env.scene[f"object_{object_id}"]
    command = env.command_manager.get_command(command_name)
    des_pos_w = command[:, :3] + env.scene.env_origins
    distance = torch.norm(des_pos_w - object.data.root_pos_w[:, :3], dim=1)
    max_distance = torch.norm(des_pos_w - env.object_init_pos[object_id], dim=1)
    distance = torch.clamp(max_distance - distance, min=0.0)
    rew = distance * if_aligned_quat(env) * if_aligned_pos(env)
    return rew

def object_goal_orient_distance(env, command_name, object_id: int = 0):
    object: RigidObject = env.scene[f"object_{object_id}"]
    command_term = env.command_manager.get_term(command_name)
    distance = math_utils.quat_error_magnitude(object.data.root_quat_w, command_term.quat_command_w)
    rew = (2.5 - distance) * if_aligned_quat(env) * if_aligned_pos(env)
    return rew

def punish_collision(env, sensor, filter_force: bool = False):
    if filter_force:
        force = env.scene[sensor].data.force_matrix_w.squeeze(1)
    else:
        force = env.scene[sensor].data.net_forces_w
    is_contact = (torch.norm(force, dim=-1) > 1.0)
    is_contact = torch.any(is_contact, dim=-1)
    return is_contact.float()
```

**Code — shared reward functions (`reward_mdps.py`): `align_palm_to_quat`, `success_bonus`, `energy_punishment`:**
```python
def align_palm_to_quat(env, link_name, asset_cfg=SceneEntityCfg("robot"), frame_name: str = "approach_frame"):
    robot: Articulation = env.scene[asset_cfg.name]
    link_idx = robot.find_bodies(link_name)[0]
    robot_state_w = robot.data.body_state_w[:, link_idx, :7].reshape(-1, 7)
    ori_distance = math_utils.quat_error_magnitude(robot_state_w[:, 3:7], env.scene[frame_name].data.target_quat_w.reshape(-1, 4))
    return -ori_distance

def success_bonus(env, command_names, num_success: int = 0):
    if isinstance(command_names, str):
        command_term = env.command_manager.get_term(command_names)
        success = command_term.metrics["consecutive_success"] >= num_success
    else:
        success = torch.ones(env.num_envs, device=env.device)
        for command_name in command_names:
            command_term = env.command_manager.get_term(command_name)
            success = torch.logical_and(success, command_term.metrics["consecutive_success"] >= num_success)
    rew = success.float()
    env.success_tracker = success.float()
    return rew

def energy_punishment(env, actuator_name=None, asset_cfg=SceneEntityCfg("robot")):
    if actuator_name is None:
        energy = get_energy_consumption(env=env, robot_name=asset_cfg.name)
    else:
        energy = get_actuator_energy_consumption(env=env, robot_name=asset_cfg.name, actuator_name=actuator_name)
    return -energy
```

> **Reward shape.** Dense palm→grasp-anchor position/quat alignment (both arms), gated box goal
> position/orient tracking, collision + energy penalties, and a sparse consecutive-success bonus.

---

## §7 DR

**Description.** No domain-randomization `EventTerm`s are wired into `BoxLiftEventCfg` (the only reset
randomization is the object pose in §3). However, the shared `BaseEnv` carries a **hydra-driven DR
infrastructure** (`DomainRandomizer` + `BaseEnv.update_randomization`) that, *if* enabled by the
external `hydra_cfg.task.randomize` config, can randomize object mass, physics material
(static/dynamic friction, restitution), the first-6 (arm) action scale, energy/collision reward
weights, external force/torque, and reset pose ranges — via the functions in
`randomization_mdps.py`. None of this is activated by the `BoxLift` task config itself; it is
opt-in from the training layer. **DR: infrastructure present but no task-level DR terms → treat as
`<no DR>` for the shipped task config.**

**Code — `BaseEnv.update_randomization` (dispatch, `manager_based_env.py`):**
```python
def update_randomization(self, success_rate):
    self.domain_randomizer.update(success_rate)
    randomized_values, randomization_state, curriculum_state = self.domain_randomizer.sample()
    if "object_mass" in randomized_values:
        randomize_mass(self, randomized_values["object_mass"])
    if "static_friction" in randomized_values:
        randomize_material(self, static_friction=randomized_values["static_friction"], ...)
    if "action_scale" in randomized_values:
        self._scale[:6] = randomized_values["action_scale"]
    if "energy_penalty" in randomized_values:
        for rew_name in curriculum_state["energy_penalty"]["names"]:
            randomize_rew_weight(self, rew_name, randomized_values["energy_penalty"])
    if "collision_penalty" in randomized_values:
        for rew_name in curriculum_state["collision_penalty"]["names"]:
            randomize_rew_weight(self, rew_name, randomized_values["collision_penalty"])
    if "external_force_torque" in randomized_values:
        randomize_external_force_torque(self, force=randomized_values["external_force_torque"], torque=randomized_values["external_force_torque"])
    for rand_name in randomized_values.keys():
        if "reset_pose" in rand_name:
            randomize_reset_pose(self, curriculum_state[rand_name]["names"], randomized_values[rand_name])
    return randomization_state, curriculum_state, self.domain_randomizer.best_so_far
```
