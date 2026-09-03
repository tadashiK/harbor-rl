# Isaac-Dex-Grasp — Implementation Spec

- robot: UFactory UF850 arm + Allegro right hand (22 DoF)
- simulator: IsaacLab (Isaac Sim, manager-based)
- objects: "dog" rigid object, lab table
- bimanual: false
- summary: Grasp a small rigid object off a table and lift it to a target height.

Task summary: a single UFactory 850 + Allegro right hand (22-DoF: 6 arm + 16 hand) sits at env-local `(-0.274, -0.475, 0.01)` and must grasp + lift a small "dog" rigid object (0.11 kg, dynamic) sitting at env-local `(0.05, -0.35, 0.0)` on a lab table. Goal: drive the dog to env-local target `(0.05, -0.35, 0.30)` — same xy as spawn, +30 cm in z — within 10 cm tolerance before the 8.33-s horizon expires. Scene + actuator stack + init pose mirror the source repo `InsertDrawer`'s right-robot half byte-for-byte (USD, init pos, joint qpos, 9-group ImplicitActuatorCfg blocks). Controller is the **joint-space** EMA cumulative-relative action vendored verbatim from the source repo into `mdp/actions.py` + `mdp/actions_cfg.py` (no runtime cross-repo import). Reward composer is `sum` over 6 dense-then-sparse terms. No DR is wired (`EventCfg` only has reset terms).

---

## §1 Registration + Scene

### Description

`gym.register` exposes `Isaac-Dex-Grasp` (training) and `Isaac-Dex-Grasp-Play` (eval-friendly: 50 envs, obs corruption disabled). Both use `isaaclab.envs:ManagerBasedRLEnv` with the abstract `DexGraspEnvCfg` base + `UF850DexGraspEnvCfg` subclass that fills in the UF850 + Allegro right-hand articulation and the palm `ee_frame` FrameTransformer. The dog (RigidObjectCfg, dynamic, 0.11 kg), the lab table (RigidObjectCfg, kinematic=True — fixed-base body that participates in collision but no rigid-body dynamics), the four fingertip contact sensors (filtered against the dog), the lift-target frame marker, the ground plane, and the dome light all live on the abstract scene. The `_TARGET_MARKER_CFG` module constant is a small (1 cm scale) frame marker rendered at env-local `DOG_TARGET_LOCAL = (0.05, -0.35, 0.30)` via a `FrameTransformerCfg` anchored on the (kinematic) table. Sim timing and physx knobs mirror the source repo InsertDrawer (`sim.dt = 1/120`, `decimation = 6` → 20 Hz control, `episode_length_s = 8.3333` → 166 steps; `gpu_max_rigid_contact_count = 2**24`, `gpu_max_rigid_patch_count = 2**24` for the 16-finger hand at 4096 envs).

### Decisions resolved

| Knob | Value |
|---|---|
| Task ID (train / play) | `Isaac-Dex-Grasp` / `Isaac-Dex-Grasp-Play` |
| Family | `isaaclab-manager-based` (`ManagerBasedRLEnv`) |
| Robot | UF850 + Allegro right hand (`harbor/assets/ufactory850/uf850_allegro_right.usd`), single instance at `{ENV_REGEX_NS}/Robot` |
| Robot init pos (env-local) | `(-0.274, -0.475, 0.01)` (the source repo right-robot verbatim) |
| Robot init joint pose | arm: `joint1=0.8, joint2=0.3, joint3=-0.6, joint4=0.0, joint5=-0.8, joint6=-1.57`; hand index/middle/pinky proximal=0.0, mid=0.4, distal=0.4, tip=0.0; thumb `jth1=1.3, jth2=0.0, jth3=0.2, jth4=0.0` |
| Robot DoF | 22 = 6 arm (joint1..joint6) + 16 hand (4 fingers × 4 joints; index/middle/pinky/thumb each j*f1..j*f4) |
| Actuator groups | 9 ImplicitActuatorCfg groups, per-joint-group stiffness/damping (see §1 verbatim) |
| Dog asset | `harbor/assets/grasp/dog.usd`, dynamic (`kinematic_enabled=False`), `mass=0.11 kg`, scale `(1,1,1)`, `activate_contact_sensors=True`, `disable_gravity=False` |
| Dog spawn pose (env-local) | `(0.0, 0.0, 0.0)` at SPAWN; reset event teleports to `(0.05, -0.35, 0.0)` every episode (see §3) |
| Table asset | `harbor/assets/table/lab_table_instanceable_colored_rotated.usd`, **RigidObjectCfg with `kinematic_enabled=True`** (fixed-base — collision shapes participate in contact but no dynamics), env-local pos `(0,0,0)`, surface at z ≈ 0 |
| Contact sensors | `contact_sensors_{0,1,2,3}` on `Robot/{if5, mf5, pf5, th5}` (index, middle, pinky, thumb fingertips), filtered against `{ENV_REGEX_NS}/Dog`, `update_period=0.0`, `debug_vis=True` |
| ee_frame | `FrameTransformerCfg` from `Robot/link_base` → `Robot/palm_link` with zero offset, `debug_vis=True`, marker scale `(0.1, 0.1, 0.1)` |
| target_marker | `FrameTransformerCfg` on `{ENV_REGEX_NS}/Table` → same table prim with `OffsetCfg(pos=DOG_TARGET_LOCAL=(0.05,-0.35,0.30))`, `debug_vis=True`, custom `_TARGET_MARKER_CFG` (scale 0.01) |
| Ground plane | world z = -0.82 |
| Light | `DomeLightCfg(color=(0.75,0.75,0.75), intensity=3000.0)` at `/World/light` |
| `DOG_TARGET_LOCAL` (module const) | `(0.05, -0.35, 0.30)` — env-local lift target |
| `DOG_TARGET_TOL` (module const) | `0.10` (10 cm tolerance for success termination) |
| `num_envs` (default) | 4096 (train) / 50 (PLAY) |
| `env_spacing` | 2.5 m |
| `replicate_physics` | False |
| Timing | `sim.dt = 1/120`, `decimation = 6` → 20 Hz, `episode_length_s = 8.3333` (= 166 control steps) |
| Physx | `gpu_max_rigid_contact_count = 2**24`, `gpu_max_rigid_patch_count = 2**24`, `bounce_threshold_velocity = 0.01`, `gpu_found_lost_aggregate_pairs_capacity = 4*1024*1024`, `gpu_total_aggregate_pairs_capacity = 64*1024`, `friction_correlation_distance = 0.00625` |
| Asset paths (resolved) | `Path(__file__).resolve().parents[6] / "harbor" / "assets" / ...` (from `dex_grasp_env_cfg.py`) and `parents[8]` (from `config/uf850/joint_pos_env_cfg.py` — two levels deeper). Both resolve to `<repo>/harbor/assets/...` (verified `test -e` on all three USDs). |

### Code (verbatim)

`source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/dex_grasp/__init__.py`:

```python
"""Dex-Grasp manipulation task -- UFactory 850 + Allegro right hand.

Single-arm 22-DoF (6 arm + 16 hand) robot grasps and lifts a `dog` object off
the lab table. Scene + actuator stack + init pose mirror the source repo
`InsertDrawer` task RIGHT-ROBOT half byte-for-byte (USD, init pos, joint qpos,
stiffness/damping per joint group). The left robot and drawer from the source repo
scene are dropped.
"""

from . import mdp  # noqa: F401 -- re-exports task-local helpers
```

`source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/dex_grasp/config/uf850/__init__.py`:

```python
import gymnasium as gym

##
# Register Gym environments.
##


gym.register(
    id="Isaac-Dex-Grasp",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:UF850DexGraspEnvCfg",
    },
)

gym.register(
    id="Isaac-Dex-Grasp-Play",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:UF850DexGraspEnvCfg_PLAY",
    },
)
```

Module-level scene constants in `dex_grasp_env_cfg.py`:

```python
# Small frame marker (1 cm scale) for visualizing the lift target -- matches
# the insert_drawer convention.
_TARGET_MARKER_CFG = FRAME_MARKER_CFG.copy()
_TARGET_MARKER_CFG.markers["frame"].scale = (0.01, 0.01, 0.01)
_TARGET_MARKER_CFG.prim_path = "/Visuals/DogTargetMarker"


# Repo-relative asset paths. This file is at
#   <repo>/source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/dex_grasp/
# so `parents[6]` is the repo root, then `harbor/assets/...`.
_HARBOR_ASSETS = Path(__file__).resolve().parents[6] / "harbor" / "assets"
_TABLE_USD_PATH = str(_HARBOR_ASSETS / "table" / "lab_table_instanceable_colored_rotated.usd")
_DOG_USD_PATH = str(_HARBOR_ASSETS / "grasp" / "dog.usd")

# Lift target the dog must reach for success (env-local world coords).
# Same xy as the dog spawn (0.05, -0.35); +0.30 m in z = lift by 30 cm.
DOG_TARGET_LOCAL: tuple[float, float, float] = (0.05, -0.35, 0.30)
# Success tolerance: episode terminates when |dog_pos - target| < 10 cm.
DOG_TARGET_TOL: float = 0.10
```

Abstract scene (`dex_grasp_env_cfg.py:DexGraspSceneCfg`):

```python
@configclass
class DexGraspSceneCfg(InteractiveSceneCfg):
    """Scene: ground + lab table + UF850/Allegro robot + dog + 4 fingertip contact
    sensors (filtered against the dog) + palm ee_frame + lights."""

    # Robot: filled by the per-robot subclass via __post_init__.
    robot: ArticulationCfg = MISSING
    # Palm EE frame transformer -- filled by per-robot subclass.
    ee_frame: FrameTransformerCfg = MISSING

    # Dog: the object to grasp+lift. the source repo verbatim: kinematic_enabled=False
    # (dynamic), mass=0.11 kg, scale=(1,1,1), activate_contact_sensors=True.
    # The reset event teleports it to env-local (0.05, -0.35, 0.0) with random
    # yaw on every episode reset.
    dog: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Dog",
        spawn=UsdFileCfg(
            usd_path=_DOG_USD_PATH,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=False,
                disable_gravity=False,
                max_linear_velocity=1000,
                max_angular_velocity=1000,
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=1,
                max_depenetration_velocity=1000.0,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.11),
            activate_contact_sensors=True,
            scale=(1.0, 1.0, 1.0),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            lin_vel=(0.0, 0.0, 0.0),
            ang_vel=(0.0, 0.0, 0.0),
            pos=(0.0, 0.0, 0.0),
        ),
    )

    # Four fingertip contact sensors filtered against the dog
    # (the source repo InsertDrawer right-robot variant: if5 / mf5 / pf5 / th5).
    contact_sensors_0 = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/if5",  # index
        update_period=0.0,
        debug_vis=True,
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Dog"],
    )
    contact_sensors_1 = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/mf5",  # middle
        update_period=0.0,
        debug_vis=True,
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Dog"],
    )
    contact_sensors_2 = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/pf5",  # pinky
        update_period=0.0,
        debug_vis=True,
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Dog"],
    )
    contact_sensors_3 = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/th5",  # thumb
        update_period=0.0,
        debug_vis=True,
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Dog"],
    )

    # Lab table -- the source repo convention: RigidObjectCfg with kinematic_enabled=True
    # so physx treats it as a fixed-base body (collision shapes participate in
    # contact, but no rigid-body dynamics — contact forces from the dog or the
    # robot fingers cannot push the table). Surface at z ~ 0.
    table: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        spawn=UsdFileCfg(
            usd_path=_TABLE_USD_PATH,
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True,
                disable_gravity=False,
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=1,
                max_depenetration_velocity=10.0,
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
    )

    # Lift-target frame marker. Anchor on the (fixed) table whose root sits at
    # env_origin + (0, 0, 0); offset = DOG_TARGET_LOCAL puts the marker at the
    # env-local target position. debug_vis=True renders a 1 cm frame in viewer.
    target_marker = FrameTransformerCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        debug_vis=True,
        visualizer_cfg=_TARGET_MARKER_CFG,
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/Table",
                name="dog_target",
                offset=OffsetCfg(pos=DOG_TARGET_LOCAL),
            ),
        ],
    )

    # Ground plane -- the source repo convention: 0.82 m below the table top.
    plane = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0.0, 0.0, -0.82]),
        spawn=GroundPlaneCfg(),
    )

    # Lights.
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )
```

Per-robot subclass (`config/uf850/joint_pos_env_cfg.py`):

```python
from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import FrameTransformerCfg, OffsetCfg
from isaaclab.utils import configclass

from isaaclab_tasks.manager_based.manipulation.dex_grasp.dex_grasp_env_cfg import DexGraspEnvCfg

##
# Pre-defined configs
##
from isaaclab.markers.config import FRAME_MARKER_CFG  # isort: skip


# Repo-relative robot USD. The file lives at
#   <repo>/source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/dex_grasp/config/uf850/
# so `parents[8]` is the repo root, then `harbor/assets/ufactory850/...`.
_ROBOT_USD_PATH = str(
    Path(__file__).resolve().parents[8]
    / "harbor" / "assets" / "ufactory850" / "uf850_allegro_right.usd"
)


# the source repo right-robot init joint pose (verbatim from
# `<source-repo>/env/tasks/InsertDrawer/env_cfg.py:InsertDrawerSceneCfg.robot.init_state.joint_pos`).
ROBOT_INIT_JOINT_POS = {
    "joint1": 0.8,
    "joint2": 0.3,
    "joint3": -0.6,
    "joint4": 0.0,
    "joint5": -0.8,
    "joint6": -1.57,
    # hand -- proximal (f1) / mid (f2) / distal (f3) / tip (f4) for index/middle/pinky;
    # thumb has its own group jth1..jth4.
    "jif1": 0.0,
    "jif2": 0.4,
    "jif3": 0.4,
    "jif4": 0.0,
    "jmf1": 0.0,
    "jmf2": 0.4,
    "jmf3": 0.4,
    "jmf4": 0.0,
    "jpf1": 0.0,
    "jpf2": 0.4,
    "jpf3": 0.4,
    "jpf4": 0.0,
    "jth1": 1.3,
    "jth2": 0.0,
    "jth3": 0.2,
    "jth4": 0.0,
}


# UF850 + Allegro right hand articulation cfg -- the source repo right-robot verbatim.
UF850_ALLEGRO_RIGHT_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=_ROBOT_USD_PATH,
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=True,
            max_depenetration_velocity=1000.0,
            max_linear_velocity=1000,
            max_angular_velocity=1000,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=16,
            solver_velocity_iteration_count=1,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos=ROBOT_INIT_JOINT_POS,
        pos=(-0.274, -0.475, 0.01),
    ),
    actuators={
        "xArm_1-6": ImplicitActuatorCfg(
            joint_names_expr=["joint[1-6]"],
            stiffness=2000.0,
            damping=16.0,
        ),
        "allegro_hand_1": ImplicitActuatorCfg(
            joint_names_expr=["j.*f1"],
            stiffness=325.0,
            damping=20.0,
        ),
        "allegro_hand_2": ImplicitActuatorCfg(
            joint_names_expr=["j.*f2"],
            stiffness=425.0,
            damping=25.0,
        ),
        "allegro_hand_3": ImplicitActuatorCfg(
            joint_names_expr=["j.*f3"],
            stiffness=245.0,
            damping=15.0,
        ),
        "allegro_hand_4": ImplicitActuatorCfg(
            joint_names_expr=["j.*f4"],
            stiffness=1050.0,
            damping=65.0,
        ),
        "allegro_hand_thumb_1": ImplicitActuatorCfg(
            joint_names_expr=["jth1"],
            stiffness=100.0,
            damping=5.0,
        ),
        "allegro_hand_thumb_2": ImplicitActuatorCfg(
            joint_names_expr=["jth2"],
            stiffness=300.0,
            damping=15.0,
        ),
        "allegro_hand_thumb_3": ImplicitActuatorCfg(
            joint_names_expr=["jth3"],
            stiffness=1270.0,
            damping=100.0,
        ),
        "allegro_hand_thumb_4": ImplicitActuatorCfg(
            joint_names_expr=["jth4"],
            stiffness=1000.0,
            damping=50.0,
        ),
    },
)


@configclass
class UF850DexGraspEnvCfg(DexGraspEnvCfg):
    """UF850 + Allegro right hand specialization for Dex-Grasp."""

    def __post_init__(self):
        super().__post_init__()

        # Robot -- UF850 + Allegro right hand at the source repo's exact pose.
        self.scene.robot = UF850_ALLEGRO_RIGHT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # Palm EE frame sensor (analog of stack_cube/lift_box ee_frame). Anchored
        # on the robot's root link; tracks `palm_link` with zero offset.
        marker_cfg = FRAME_MARKER_CFG.copy()
        marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
        marker_cfg.prim_path = "/Visuals/FrameTransformer"
        self.scene.ee_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/link_base",
            debug_vis=True,
            visualizer_cfg=marker_cfg,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/palm_link",
                    name="palm",
                    offset=OffsetCfg(pos=(0.0, 0.0, 0.0)),
                ),
            ],
        )


@configclass
class UF850DexGraspEnvCfg_PLAY(UF850DexGraspEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        # Smaller scene for play.
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # Disable obs noise for play.
        self.observations.policy.enable_corruption = False
```

EnvCfg `__post_init__` (sim/physx):

```python
def __post_init__(self):
    """Timing: the source repo InsertDrawer canonical (120 Hz physics / decimation 6
    -> 20 Hz control / ~8.33 s episode)."""
    self.decimation = 6
    self.episode_length_s = 8.3333
    # Simulation -- the source repo canonical
    self.sim.dt = 1.0 / 120.0
    self.sim.render_interval = self.decimation
    # Physics knobs -- the source repo canonical (large rigid-contact / patch caps for
    # the 16-finger hand at 4096 envs).
    self.sim.physx.gpu_max_rigid_contact_count = 2 ** 24
    self.sim.physx.gpu_max_rigid_patch_count = 2 ** 24
    # Iter 0: match insert_drawer for aggregate-pair cap headroom.
    self.sim.physx.bounce_threshold_velocity = 0.01
    self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 1024 * 1024 * 4
    self.sim.physx.gpu_total_aggregate_pairs_capacity = 64 * 1024
    self.sim.physx.friction_correlation_distance = 0.00625
```

### Smoke

```bash
.venv/bin/python harbor/create-task/isaac-dex-grasp/smokes/smoke_s1.py
# exit 0 (asserts action_space.shape[-1] == 22, observation_space == Box(47,),
#         num_envs == 4096 (train) or 50 (PLAY), max_episode_length == 166)
```

---

## §2 Actions

### Description

Single action term `arm_hand_action` — 22-D EMA cumulative-relative joint position controller, vendored verbatim from `<source-repo>/env/action_managers/actions.py:EMACumulativeRelativeJointPositionAction` into this task's `mdp/actions.py` (so there is **no runtime import from `<source-repo>`**). One scalar per joint, regex `.*` captures all 22 joints in USD canonical order (6 arm `joint1..joint6` followed by 16 hand joints in the order encoded by `JOINT_LOWER_LIMIT` / `JOINT_UPPER_LIMIT` — see §2 lists). Controller is joint-space (NOT IK / NOT task-space). Per-step rule (raw `a_t`, scale `s = 0.03`, alpha `α = 0.2`, init joint pose `q_init` captured at reset, cumulative delta `del_t`, previous applied `prev_t`):

```
1. processed_t = s * a_t              # JointPositionAction (offset disabled, use_default_offset=False)
2. processed_t += del_{t-1}           # accumulate delta
3. del_t       = processed_t          # remember new cumulative delta
4. processed_t += q_init              # anchor on captured init pose
5. ema_t       = α * processed_t + (1 - α) * prev_{t-1}
6. processed_t = clamp(ema_t, JOINT_LOWER_LIMIT, JOINT_UPPER_LIMIT)
7. prev_t      = processed_t
```

At t=0 (post-reset, `prev_0 == q_init`, `del_{-1} == 0`, offset == 0): `target_0 = clamp(q_init + α * s * a_0, lower, upper)`.

### Decisions resolved

| Knob | Value |
|---|---|
| Action class | `mdp.EMACumulativeRelativeJointPositionActionCfg` (vendored locally — extends `JointPositionActionCfg`) |
| Action dim | 22 (= 6 arm + 16 hand, regex `.*`) |
| `asset_name` | `"robot"` |
| `joint_names` | `[".*"]` |
| `scale` | `0.03` (scalar) |
| `use_default_offset` | `False` |
| `alpha` (EMA weight on new target) | `0.2` (scalar; cfg supports per-joint dict too) |
| `joint_lower_limit` / `joint_upper_limit` | the source repo right-hand 22-entry lists (see verbatim block below) |
| Composer | Single action term — total action vector = `arm_hand_action` (22-D) |

### Code (verbatim)

`dex_grasp_env_cfg.py:ActionsCfg`:

```python
@configclass
class ActionsCfg:
    """Action specs -- one term, 22-D EMA cumulative-relative joint position.

    Matches the source repo `InsertDrawerActionsCfg.arm_hand_action` for the right robot.
    Joint regex `.*` picks up all 22 joints (6 arm + 16 hand) in the USD
    canonical order.
    """

    arm_hand_action = mdp.EMACumulativeRelativeJointPositionActionCfg(
        asset_name="robot",
        joint_names=[".*"],
        scale=0.03,
        use_default_offset=False,
        joint_lower_limit=JOINT_LOWER_LIMIT,
        joint_upper_limit=JOINT_UPPER_LIMIT,
        alpha=0.2,
    )
```

`mdp/actions.py` (full source — vendored from the source repo):

```python
"""EMA cumulative-relative joint position action.

Vendored verbatim from `<source-repo>/env/action_managers/actions.py:EMACumulativeRelativeJointPositionAction`.
We do NOT import from `the source repo` at runtime; the class lives here so the task has
no cross-repo dependency.

Per-step processing (raw action `a_t`, scale `s`, offset `o`, alpha `α`,
init joint pose captured at reset `q_init`):

    1. processed_t = scale * a_t + offset            # JointPositionAction
    2. processed_t = processed_t + del_{t-1}         # accumulate delta
    3. del_t       = processed_t                     # remember cumulative delta
    4. processed_t = processed_t + q_init            # anchor on init pose
    5. ema_t       = α * processed_t + (1 - α) * prev_applied_{t-1}
    6. processed_t = clamp(ema_t, joint_lower_limit, joint_upper_limit)
    7. prev_applied_t = processed_t

At t=0 (post-reset, prev_applied == q_init, del_{-1} == 0, offset == 0):
    target_0 = clamp(q_init + α * scale * a_0, lower, upper)
"""
from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

import isaaclab.utils.string as string_utils
from isaaclab.assets import Articulation
from isaaclab.envs.mdp.actions import JointPositionAction

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

    from . import actions_cfg


class EMACumulativeRelativeJointPositionAction(JointPositionAction):
    cfg: "actions_cfg.EMACumulativeRelativeJointPositionActionCfg"
    _asset: Articulation
    """The articulation asset on which the action term is applied."""

    def __init__(
        self,
        cfg: "actions_cfg.EMACumulativeRelativeJointPositionActionCfg",
        env: "ManagerBasedRLEnv",
    ) -> None:
        # initialize the action term
        super().__init__(cfg, env)

        # parse and save the moving average weight
        if isinstance(cfg.alpha, float):
            # check that the weight is in the valid range
            if not 0.0 <= cfg.alpha <= 1.0:
                raise ValueError(f"Moving average weight must be in the range [0, 1]. Got {cfg.alpha}.")
            self._alpha = cfg.alpha
        elif isinstance(cfg.alpha, dict):
            self._alpha = torch.ones((env.num_envs, self.action_dim), device=self.device)
            # resolve the dictionary config
            index_list, names_list, value_list = string_utils.resolve_matching_names_values(
                cfg.alpha, self._joint_names
            )
            # check that the weights are in the valid range
            for name, value in zip(names_list, value_list):
                if not 0.0 <= value <= 1.0:
                    raise ValueError(
                        f"Moving average weight must be in the range [0, 1]. Got {value} for joint {name}."
                    )
            self._alpha[:, index_list] = torch.tensor(value_list, device=self.device)
        else:
            raise ValueError(
                f"Unsupported moving average weight type: {type(cfg.alpha)}. Supported types are float and dict."
            )

        # initialize the previous targets
        self._prev_applied_actions = torch.zeros_like(self.processed_actions)
        # initialize the cumulative del action
        self.del_action = torch.zeros((self._env.num_envs, self.action_dim), device=self._env.device)
        self.init_joint_pos = self._asset.data.joint_pos[:, self._joint_ids].clone()
        self.joint_lower_limit = (
            torch.tensor(cfg.joint_lower_limit, device=self.device)
            if cfg.joint_lower_limit is not None
            else None
        )
        self.joint_upper_limit = (
            torch.tensor(cfg.joint_upper_limit, device=self.device)
            if cfg.joint_upper_limit is not None
            else None
        )

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        # check if specific environment ids are provided
        if env_ids is None:
            env_ids = slice(None)
        super().reset(env_ids)
        # reset history to current joint positions
        self._prev_applied_actions[env_ids, :] = self._asset.data.joint_pos[env_ids][:, self._joint_ids].clone()
        # reset the del action
        if isinstance(env_ids, slice):
            self.del_action[:] = 0.0
        else:
            self.del_action[env_ids, :] = torch.zeros((env_ids.shape[0], self.action_dim), device=self.device)
        self.init_joint_pos[env_ids, :] = self._asset.data.joint_pos[env_ids][:, self._joint_ids].clone()

    def process_actions(self, actions: torch.Tensor):
        # apply affine transformations (scale * raw + offset)
        super().process_actions(actions)
        # compute the del action
        self._processed_actions += self.del_action
        self.del_action = self._processed_actions.clone()
        # add the initial position
        self._processed_actions += self.init_joint_pos.clone()
        # set position targets as moving average
        ema_actions = self._alpha * self._processed_actions
        ema_actions += (1.0 - self._alpha) * self._prev_applied_actions
        # clamp the targets
        if self.joint_lower_limit is not None and self.joint_upper_limit is not None:
            self._processed_actions[:] = torch.clamp(
                ema_actions,
                self.joint_lower_limit,
                self.joint_upper_limit,
            )
        else:
            self._processed_actions[:] = ema_actions
        # update previous targets
        self._prev_applied_actions[:] = self._processed_actions[:]
```

`mdp/actions_cfg.py` (full source):

```python
"""Action cfg + per-joint limit constants for the dex_grasp task.

`EMACumulativeRelativeJointPositionActionCfg` is vendored verbatim from
`<source-repo>/env/action_managers/actions_cfg.py`. `JOINT_LOWER_LIMIT` and
`JOINT_UPPER_LIMIT` are the right-hand variants from
`<source-repo>/env/tasks/manager_based_env_cfg.py`. The order matches the USD joint
order:
    joint1..joint6 (arm)
    jif1, jmf1, jpf1, jth1   (proximal)
    jif2, jmf2, jpf2, jth2   (mid)
    jif3, jmf3, jpf3, jth3   (distal)
    jif4, jmf4, jpf4, jth4   (tip)
22 entries total.
"""
from isaaclab.envs.mdp.actions import JointPositionActionCfg
from isaaclab.managers.action_manager import ActionTerm
from isaaclab.utils import configclass

from .actions import EMACumulativeRelativeJointPositionAction


# the source repo right-hand joint limits (vendored verbatim from
# `<source-repo>/env/tasks/manager_based_env_cfg.py:JOINT_LOWER_LIMIT`).
JOINT_LOWER_LIMIT = [
    -6.283, -2.304, -4.224, -6.283, -2.164, -6.283,
    # jif1, jmf1, jpf1, jth1
    -0.05, -0.05, -0.570, 0.364,
    # jif2, jmf2, jpf2, jth2
    -0.296, -0.296, -0.296, -0.205,
    # jif3, jmf3, jpf3, jth3
    -0.274, -0.274, -0.274, -0.290,
    # jif4, jmf4, jpf4, jth4
    -0.327, -0.327, -0.327, -0.262,
]
JOINT_UPPER_LIMIT = [
    6.283, 2.304, 0.061, 6.283, 2.164, 6.283,
    # jif1, jmf1, jpf1, jth1
    0.570, 0.05, 0.05, 1.497,
    # jif2, jmf2, jpf2, jth2
    1.710, 1.710, 1.710, 1.130,
    # jif3, jmf3, jpf3, jth3
    1.809, 1.809, 1.809, 1.633,
    # jif4, jmf4, jpf4, jth4
    1.718, 1.718, 1.718, 1.820,
]


@configclass
class EMACumulativeRelativeJointPositionActionCfg(JointPositionActionCfg):
    """Configuration for the EMA cumulative-relative joint position action term.

    See :class:`EMACumulativeRelativeJointPositionAction` for the per-step rule.
    """

    class_type: type[ActionTerm] = EMACumulativeRelativeJointPositionAction

    alpha: float | dict[str, float] = 1.0
    """The weight for the moving average (float or dict of regex expressions). Defaults to 1.0.

    If set to 1.0, the processed action is applied directly without any moving
    average window.
    """

    joint_lower_limit: list[float] | None = None
    joint_upper_limit: list[float] | None = None
    """The lower and upper limits for the joint positions (applied to the
    post-EMA target). Both must be set together; otherwise no clamp is applied.
    """
```

### `JOINT_LOWER_LIMIT` / `JOINT_UPPER_LIMIT` (22-entry lists, verbatim)

```python
JOINT_LOWER_LIMIT = [
    -6.283, -2.304, -4.224, -6.283, -2.164, -6.283,   # joint1..joint6 (arm)
    -0.05,  -0.05,  -0.570,  0.364,                    # jif1, jmf1, jpf1, jth1   (proximal)
    -0.296, -0.296, -0.296, -0.205,                    # jif2, jmf2, jpf2, jth2   (mid)
    -0.274, -0.274, -0.274, -0.290,                    # jif3, jmf3, jpf3, jth3   (distal)
    -0.327, -0.327, -0.327, -0.262,                    # jif4, jmf4, jpf4, jth4   (tip)
]
JOINT_UPPER_LIMIT = [
     6.283,  2.304,  0.061,  6.283,  2.164,  6.283,   # joint1..joint6 (arm)
     0.570,  0.05,   0.05,   1.497,                    # jif1, jmf1, jpf1, jth1   (proximal)
     1.710,  1.710,  1.710,  1.130,                    # jif2, jmf2, jpf2, jth2   (mid)
     1.809,  1.809,  1.809,  1.633,                    # jif3, jmf3, jpf3, jth3   (distal)
     1.718,  1.718,  1.718,  1.820,                    # jif4, jmf4, jpf4, jth4   (tip)
]
```

### Smoke

```bash
.venv/bin/python harbor/create-task/isaac-dex-grasp/smokes/smoke_s2.py
# exit 0 (asserts action_space.shape[-1] == 22; env.step(zeros) returns finite obs/reward;
#         small-perturbation step changes joint targets in the EMA / cumulative direction)
```

---

## §3 Reset

### Description

Two reset terms — one for the robot (joint positions pinned to URDF home via `reset_joints_by_scale` with point-interval ranges) and one for the dog (teleport to env-local `(0.05, -0.35, 0.0)` with yaw range pinned to `(0, 0)`). The robot reset term uses the default `asset_cfg = SceneEntityCfg("robot")` (no override needed — the single robot is named `"robot"`). The dog reset binds `asset_cfg=SceneEntityCfg("dog")` explicitly. All ranges are point intervals so the §3 reset smoke can read deterministic values; `dr-generator` is expected to widen yaw to `[-pi, pi]` later.

### Decisions resolved

| Term | Function | `position_range` / `pose_range` | `velocity_range` | asset_cfg |
|---|---|---|---|---|
| reset_robot_joints | `mdp.reset_joints_by_scale` | `position_range=(1.0, 1.0)` | `(0.0, 0.0)` | (default) `SceneEntityCfg("robot")` |
| reset_dog | `mdp.reset_root_state_uniform` | `pose_range={"x":(0.05,0.05), "y":(-0.35,-0.35), "z":(0.0,0.0), "yaw":(0.0,0.0)}` | `{}` | `SceneEntityCfg("dog")` |

### Code (verbatim)

`dex_grasp_env_cfg.py:EventCfg`:

```python
@configclass
class EventCfg:
    """Reset terms only (§3). §7 DR is not in scope; dr-generator may widen later.

    All ranges are pinned to point intervals so the §3 reset smoke can read
    deterministic values. (`dr-generator` widens later.)
    """

    # Robot joints: pinned to URDF home (scale 1.0).
    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (1.0, 1.0),
            "velocity_range": (0.0, 0.0),
        },
    )

    # Dog: the source repo `reset_object_right` verbatim -- teleport to env-local
    # (0.05, -0.35, 0.0). Yaw range pinned to (0, 0) so §3 reset is
    # deterministic; dr-generator may widen to [-pi, pi] later (the source repo
    # default).
    reset_dog = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {
                "x": (0.05, 0.05),
                "y": (-0.35, -0.35),
                "z": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("dog"),
        },
    )
```

### Smoke

```bash
.venv/bin/python harbor/create-task/isaac-dex-grasp/smokes/smoke_s3.py
# exit 0 (asserts env.reset() places the robot at URDF home (22-DoF joint pose
#         within tol of ROBOT_INIT_JOINT_POS) and the dog at env-local
#         (0.05, -0.35, 0.0) with zero linear/angular velocity)
```

---

## §4 Goal + Termination

### Description

No `CommandsCfg` (`commands = None` on the EnvCfg) — the goal point is **hard-coded** in the module-level constant `DOG_TARGET_LOCAL = (0.05, -0.35, 0.30)` defined at the top of `dex_grasp_env_cfg.py`. The `_TARGET_MARKER_CFG` module constant renders a small (1 cm scale) `/Visuals/DogTargetMarker` frame at that env-local position via a `FrameTransformerCfg` anchored on the (fixed kinematic) table. Two terminations: `time_out` (after 166 control steps = 8.3333 s) and `success` (the `dog_reached_target` predicate fires when `||dog_local - DOG_TARGET_LOCAL||_2 < DOG_TARGET_TOL = 0.10 m`). No failure-mode terminations (drop / tilt / table contact) — the policy is allowed to recover within the 8.33-s horizon.

### Decisions resolved

| Term | `func` | `params` | `time_out` flag |
|---|---|---|---|
| time_out | `mdp.time_out` | `{}` | `True` |
| success | `mdp.dog_reached_target` | `{"target_local": DOG_TARGET_LOCAL, "threshold": DOG_TARGET_TOL, "dog_cfg": SceneEntityCfg("dog")}` | `False` |

`commands = None` at the EnvCfg level. The goal is hard-coded — no command manager.

### Code (verbatim)

`dex_grasp_env_cfg.py:TerminationsCfg`:

```python
@configclass
class TerminationsCfg:
    """`time_out` + `success` (dog within 10 cm of the lift target)."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    success = DoneTerm(
        func=mdp.dog_reached_target,
        time_out=False,
        params={
            "target_local": DOG_TARGET_LOCAL,
            "threshold": DOG_TARGET_TOL,
            "dog_cfg": SceneEntityCfg("dog"),
        },
    )
```

`dex_grasp/mdp/terminations.py` (full source):

```python
# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Termination helpers for the dex_grasp task."""
from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def dog_reached_target(
    env: "ManagerBasedRLEnv",
    target_local: tuple[float, float, float] = (0.05, -0.35, 0.30),
    threshold: float = 0.10,
    dog_cfg: SceneEntityCfg = SceneEntityCfg("dog"),
) -> torch.Tensor:
    """True when the dog's env-local position is within `threshold` of `target_local`.

    `target_local` is in the same env-local world frame as `dog.data.root_pos_w -
    env_origins`. The default `(0.05, -0.35, 0.30)` lifts the dog 30 cm above
    its spawn xy.
    """
    dog: RigidObject = env.scene[dog_cfg.name]
    dog_local = dog.data.root_pos_w[:, :3] - env.scene.env_origins[:, :3]
    target = torch.tensor(target_local, device=env.device, dtype=dog_local.dtype)
    err = torch.norm(dog_local - target.unsqueeze(0), dim=-1)
    return err < threshold
```

> No `CommandsCfg` — the goal is hard-coded in the `DOG_TARGET_LOCAL` module constant (set at the top of `dex_grasp_env_cfg.py`) and rendered for visualization via the `_TARGET_MARKER_CFG` (1 cm-scale frame marker) attached as `target_marker` on the scene. The same constant is the default for the reward terms (`dog_to_target`, `success_bonus`) and the termination (`dog_reached_target`).

### Smoke

```bash
.venv/bin/python harbor/create-task/isaac-dex-grasp/smokes/smoke_s4.py
# exit 0 (asserts: (a) episode times out at exactly max_episode_length=166 steps under
#         zero action; (b) writing dog.root_pos_w to env_origin + DOG_TARGET_LOCAL fires
#         the success termination in all 128 envs on the next compute() call)
```

---

## §5 Observation

### Description

Single `PolicyCfg` group; all three terms concatenated → **47-D** policy obs. `enable_corruption=True` activates per-term `noise` slots (default `Unoise(n_min=0, n_max=0)` is a no-op; `dr-generator` widens later). `concatenate_terms=True`.

`joint_pos_right_normalized` normalizes the robot's 22-D joint position to `[-1, 1]` using the explicit `JOINT_LOWER_LIMIT` / `JOINT_UPPER_LIMIT` lists imported from `mdp/actions_cfg.py` (NOT the asset's `soft_joint_pos_limits`). `dog_position_in_world` returns the dog's xyz in env-local world coords (`root_pos_w - env_origin`). `last_action` is the standard `isaaclab.envs.mdp.last_action` (22-D, matches the action dim).

### Decisions resolved

| ObsTerm | `func` | `params` | dim |
|---|---|---|---|
| joint_pos_right_normalized | `mdp.joint_pos_right_normalized` | `joint_lower_limit=JOINT_LOWER_LIMIT, joint_upper_limit=JOINT_UPPER_LIMIT` (default `robot_cfg=SceneEntityCfg("robot")`) | 22 |
| dog_position_in_world | `mdp.dog_position_in_world` | (defaults — `object_cfg=SceneEntityCfg("dog")`) | 3 |
| last_action | `mdp.last_action` | (defaults) | 22 |
| **Total** | | | **47** |

PolicyCfg flags: `enable_corruption = True`, `concatenate_terms = True`. No per-term `noise` populated (no-op).

### Code (verbatim)

`dex_grasp_env_cfg.py:ObservationsCfg`:

```python
@configclass
class ObservationsCfg:
    """Observation specs -- 47-D concatenated policy obs.

    Term order:
        joint_pos_right_normalized   (22,)
        dog_position_in_world         (3,)
        last_action                  (22,)
    Total = 22 + 3 + 22 = 47.

    `enable_corruption=True` activates the per-term noise slots; the default
    `Unoise(n_min=0, n_max=0)` is a no-op (dr-generator widens later).
    """

    @configclass
    class PolicyCfg(ObsGroup):
        joint_pos_right_normalized = ObsTerm(
            func=mdp.joint_pos_right_normalized,
            params={
                "joint_lower_limit": JOINT_LOWER_LIMIT,
                "joint_upper_limit": JOINT_UPPER_LIMIT,
            },
        )
        dog_position_in_world = ObsTerm(func=mdp.dog_position_in_world)
        last_action = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()
```

`dex_grasp/mdp/observations.py` (full source — task-local helpers):

```python
"""Observation helpers for the dex_grasp task.

PolicyCfg term order (concatenated):
    1. joint_pos_right_normalized   (22,)  -- robot joint pos, normalized to [-1, 1] using
                                              JOINT_LOWER_LIMIT / JOINT_UPPER_LIMIT
    2. dog_position_in_world         (3,)  -- dog xyz in env-local world frame
    3. last_action                  (22,)  -- mdp.last_action
Total = 22 + 3 + 22 = 47.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import scale_transform

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def joint_pos_right_normalized(
    env: "ManagerBasedRLEnv",
    joint_lower_limit: list[float] | None = None,
    joint_upper_limit: list[float] | None = None,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Robot joint positions normalized to [-1, 1] using the explicit per-joint
    `joint_lower_limit` / `joint_upper_limit` lists (vendored from the source repo).

    Falls back to the asset's `soft_joint_pos_limits` when either list is None.
    Mirrors the source repo's `joint_pos_limit_normalized` helper (right-hand variant).
    """
    asset: Articulation = env.scene[robot_cfg.name]
    joint_ids = asset_cfg_to_joint_ids(asset, robot_cfg)
    if joint_lower_limit is None:
        lower = asset.data.soft_joint_pos_limits[:, joint_ids, 0]
    else:
        lower = torch.tensor(joint_lower_limit, device=env.device)
    if joint_upper_limit is None:
        upper = asset.data.soft_joint_pos_limits[:, joint_ids, 1]
    else:
        upper = torch.tensor(joint_upper_limit, device=env.device)
    return scale_transform(asset.data.joint_pos[:, joint_ids], lower, upper)


def dog_position_in_world(
    env: "ManagerBasedRLEnv",
    object_cfg: SceneEntityCfg = SceneEntityCfg("dog"),
) -> torch.Tensor:
    """Dog xyz in env-local world coordinates (`root_pos_w - env_origin`).

    Mirrors the source repo `object_pos`. The reset event teleports the dog to env-local
    `(0.05, -0.35, 0.0)`, so this signal is referenced to that same origin.
    """
    dog: RigidObject = env.scene[object_cfg.name]
    return dog.data.root_pos_w[:, :3] - env.scene.env_origins[:, :3]


def asset_cfg_to_joint_ids(asset: Articulation, cfg: SceneEntityCfg):
    """Resolve `cfg.joint_ids` to a Python slice / list usable as a tensor index.

    Robots whose `SceneEntityCfg` was constructed with no `joint_names` filter
    surface `joint_ids` as the literal `slice(None)`, which is fine to index
    directly. With a regex filter it's a list of ints.
    """
    if cfg.joint_ids is None or cfg.joint_ids == slice(None):
        return slice(None)
    return cfg.joint_ids
```

### Smoke

```bash
.venv/bin/python harbor/create-task/isaac-dex-grasp/smokes/smoke_s5.py
# exit 0 (asserts observation_space == Box(47,), per-term dims match
#         expected_obs.json: joint_pos=22, dog_pos=3, last_action=22)
```

---

## §6 Reward

### Description

Composer = **sum**. 6 active terms in a strictly-increasing per-stage magnitude ladder: 2 dense reach attractors (palm + per-fingertip), 1 binary grasp-predicate gate, 1 lift-progress ramp gated on grasp, 1 dense dog→target attractor gated on `(grasp_contact AND lifted)`, and 1 one-shot success bonus mirroring the termination predicate. No regularizers (no `action_rate`, no `joint_vel`). No failure-mode penalties (sign convention: positive=good, no penalties iter 0).

Weights below are **nominal per-step magnitudes** — the declared weight is exactly what each term pays per step (and what shows up in `episodic_return_mean`).

**Grasp predicate** (used by `grasp_contact`, gated inside `lift_height` and `dog_to_target`): the source repo Allegro convention — `thumb AND (index OR middle OR pinky)` — both fingertips must register force ≥ `contact_force_threshold` newtons on the dog (the only filtered prim).

**Recent change** (per orchestrator brief): `dog_to_target` was gated on `grasp_contact AND lifted` (both must hold) — previously this term was gated on `lifted` alone, and PPO learned to knock the dog up with the back of the hand and dwell over the target xy. Requiring `grasp_contact` AND `lifted` forces the policy to grip-then-lift before the dense xyz attractor fires.

### Decisions resolved

| RewTerm | `func` | `params` | `weight` | gate |
|---|---|---|---:|---|
| palm_to_dog | `mdp.palm_to_dog` | `std=0.20, ee_frame_cfg=SceneEntityCfg("ee_frame"), dog_cfg=SceneEntityCfg("dog")` | **0.05** | always |
| fingertip_to_dog | `mdp.fingertip_to_dog` | `std=0.10, fingertip_links=("if5","mf5","pf5","th5"), fingertip_weights=(1.0,1.0,1.0,1.5)` | **0.05** | always (weighted mean, thumb 1.5×) |
| grasp_contact | `mdp.grasp_contact` | `contact_force_threshold=1.0` | **0.10** | binary `thumb AND any-non-thumb` |
| lift_height | `mdp.lift_height` | `init_z=0.0, target_lift=0.30, contact_force_threshold=1.0` | **0.50** | × `grasp_contact` |
| dog_to_target | `mdp.dog_to_target` | `std=0.15, target_local=DOG_TARGET_LOCAL, lift_threshold=0.05, init_z=0.0` (function defaults `contact_force_threshold=1.0`) | **0.50** | × `(grasp_contact AND dog.z > init_z + 0.05)` |
| success_bonus | `mdp.success_bonus` | `target_local=DOG_TARGET_LOCAL, threshold=DOG_TARGET_TOL` | **100.0** | one-shot latch (first frame predicate fires) |

#### Per-stage saturated per-step magnitude budget (nominal weights)

| Stage | Term | Per-step saturated | 166-step ep ceiling |
|---|---|---:|---:|
| Reach (palm) | palm_to_dog | 0.05 | 8.3 |
| Reach (fingers) | fingertip_to_dog | 0.05 | 8.3 |
| Contact | grasp_contact | 0.10 | 16.6 (when held) |
| Lift | lift_height | 0.50 | 83 (when held + dog at target z) |
| Align | dog_to_target | 0.50 | 83 (when held + lifted >5 cm + dog at target xyz) |
| Success | success_bonus | one-shot | +100 |

Dense ceiling ≈ 199 (all stages saturated for the full episode); sparse success = +100 one-shot. The dense terms only saturate at the success state (`lift_height` saturates only when `dog.z` hits `target_z`; `dog_to_target` saturates only when dog xyz hits target xyz), so the policy still has a strict reason to terminate the episode rather than dwell.

#### Composer

`sum`. The IsaacLab `RewardManager` default composer is sum-of-weighted-terms; weights are nominal per-step values.

### Code (verbatim)

`dex_grasp_env_cfg.py:RewardsCfg`:

```python
@configclass
class RewardsCfg:
    """Reward ladder (composer = sum, sign = positive=good, no penalties iter 0).

    All weights are nominal per-step magnitudes -- the declared weight is what
    each term pays per step.

    Per-stage saturated per-step magnitude budget (composer = sum):

      palm_to_dog          0.05         -> 0.05 / step
      fingertip_to_dog     0.05         -> 0.05 / step
      grasp_contact        0.10         -> 0.10 / step (only when held)
      lift_height          0.50         -> 0.50 / step (only when held)
      dog_to_target        0.50         -> 0.50 / step (only when lifted >5 cm)
      success_bonus      100.0          -> +100 one-shot at success predicate

    166-step episode (8.33 s @ 20 Hz):
      dense ceiling (sum_stages * 166) = 199
      sparse one-shot                  = 100
    """

    # Stage 1 -- dense palm -> dog attractor.
    palm_to_dog = RewTerm(
        func=mdp.palm_to_dog,
        params={
            "std": 0.20,
            "ee_frame_cfg": SceneEntityCfg("ee_frame"),
            "dog_cfg": SceneEntityCfg("dog"),
        },
        weight=0.05,
    )

    # Stage 2 -- per-fingertip distance attractor (thumb 1.5x).
    fingertip_to_dog = RewTerm(
        func=mdp.fingertip_to_dog,
        params={
            "std": 0.10,
            "fingertip_links": ("if5", "mf5", "pf5", "th5"),
            "fingertip_weights": (1.0, 1.0, 1.0, 1.5),
        },
        weight=0.05,
    )

    # Stage 3 -- Allegro grasp predicate (thumb + any of {idx, mid, pinky}).
    grasp_contact = RewTerm(
        func=mdp.grasp_contact,
        params={"contact_force_threshold": 1.0},
        weight=0.10,
    )

    # Stage 4 -- linear lift ramp on dog.z, gated on grasp_contact.
    lift_height = RewTerm(
        func=mdp.lift_height,
        params={
            "init_z": 0.0,
            "target_lift": 0.30,
            "contact_force_threshold": 1.0,
        },
        weight=0.50,
    )

    # Stage 5 -- tanh attractor on dog -> DOG_TARGET, gated on dog lifted.
    dog_to_target = RewTerm(
        func=mdp.dog_to_target,
        params={
            "std": 0.15,
            "target_local": DOG_TARGET_LOCAL,
            "lift_threshold": 0.05,
            "init_z": 0.0,
        },
        weight=0.50,
    )

    # Stage 6 -- one-shot success bonus (mirrors `dog_reached_target` termination).
    success_bonus = RewTerm(
        func=mdp.success_bonus,
        params={
            "target_local": DOG_TARGET_LOCAL,
            "threshold": DOG_TARGET_TOL,
        },
        weight=100.0,
    )
```

> Note on `dog_to_target` `params`: only `std`, `target_local`, `lift_threshold`, `init_z` are passed from the cfg. The function signature also has `contact_force_threshold=1.0` and `dog_cfg=SceneEntityCfg("dog")` defaults which the cfg does NOT override — so the grasp-gate threshold is the default 1.0 N (same as `grasp_contact` and `lift_height`).

#### `dex_grasp/mdp/rewards.py` — file header + helpers

```python
# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reward functions for the single-arm Allegro `dex_grasp` task.

Five-stage manipulation shaping ladder (composer = sum):

  1. palm_to_dog       dense palm -> dog attractor (Allegro palm via FrameTransformer)
  2. fingertip_to_dog  weighted-mean per-fingertip attractor (thumb 1.5x others)
  3. grasp_contact     binary "thumb + at least one of {idx, mid, pinky} in
                       contact with the dog" (the source repo Allegro convention)
  4. lift_height       linear ramp on dog.z, gated on grasp_contact
  5. dog_to_target     tanh attractor on ||dog - DOG_TARGET||, gated on
                       dog.z > init_z + 0.05 (dog actually lifted)
  6. success_bonus     one-shot latch the first frame the dog is within 10 cm
                       of the target (mirrors the `success` termination)

Weights below are nominal per-step magnitudes -- the declared weight is what
each term pays per step (and what the user sees at runtime).

Per-stage saturated per-step magnitude budget (composer = sum, raw weights):

    palm_to_dog          0.05         -> 0.05 / step
    fingertip_to_dog     0.05         -> 0.05 / step
    grasp_contact        0.10         -> 0.10 / step (only when held)
    lift_height          0.50         -> 0.50 / step (only when held)
    dog_to_target        0.50         -> 0.50 / step (only when lifted >5 cm)
    success_bonus      100.0          -> +100 one-shot at success predicate

166-step episode (8.33 s @ 20 Hz) ceiling check (all stages saturated):
    dense ceiling (palm + fingertip + contact + lift + dog->target)
        ~ (0.05 + 0.05 + 0.10 + 0.50 + 0.50) * 166 = ~199
    success bonus one-shot                          +100
The dense ceiling exceeds the sparse landmark, but the dense terms only
saturate at the success state (lift_height saturates only when dog.z hits
target_z; dog_to_target saturates only when dog xyz hits target xyz). The
policy still has a strict reason to terminate the episode rather than dwell.

Sign convention: positive = good (no penalties iter 0).
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor, FrameTransformer

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


# ---------------------------------------------------------------------------
# Module-level per-(env, key) latch buffers (success bonus one-shot).
# ---------------------------------------------------------------------------

_LATCH_BUFFERS: dict[tuple, torch.Tensor] = {}


def _get_latch_buffer(env, key: str) -> torch.Tensor:
    """Get-or-lazily-create a per-env boolean latch tensor of shape (num_envs,)."""
    full_key = (id(env), key)
    if full_key not in _LATCH_BUFFERS:
        _LATCH_BUFFERS[full_key] = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    return _LATCH_BUFFERS[full_key]
```

#### Stage 1 — `palm_to_dog`

```python
# ---------------------------------------------------------------------------
# Stage 1 -- dense palm -> dog attractor.
# ---------------------------------------------------------------------------


def palm_to_dog(
    env: "ManagerBasedRLEnv",
    std: float = 0.20,
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    dog_cfg: SceneEntityCfg = SceneEntityCfg("dog"),
) -> torch.Tensor:
    """`1 - tanh(||palm - dog|| / std)` -- dense palm -> dog attractor.

    `palm` is the Allegro `palm_link` exposed via the `ee_frame` FrameTransformer
    (zero offset). `dog` is the rigid object's COM. Standard reach attractor;
    wider std (0.20 m) than `fingertip_to_dog` since the palm is the
    big-motion mover and we want a gradient over the whole ~30 cm gap.
    """
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    dog: RigidObject = env.scene[dog_cfg.name]
    palm_w = ee_frame.data.target_pos_w[..., 0, :]
    dog_w = dog.data.root_pos_w[:, :3]
    d = torch.norm(palm_w - dog_w, dim=-1)
    return 1.0 - torch.tanh(d / max(std, 1e-6))
```

#### Stage 2 — `fingertip_to_dog`

```python
# ---------------------------------------------------------------------------
# Stage 2 -- per-fingertip distance attractor (Allegro thumb-1.5x convention).
# ---------------------------------------------------------------------------


def fingertip_to_dog(
    env: "ManagerBasedRLEnv",
    std: float = 0.10,
    fingertip_links: tuple[str, ...] = ("if5", "mf5", "pf5", "th5"),
    fingertip_weights: tuple[float, ...] = (1.0, 1.0, 1.0, 1.5),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    dog_cfg: SceneEntityCfg = SceneEntityCfg("dog"),
) -> torch.Tensor:
    """Weighted-mean fingertip-to-dog attractor over the 4 Allegro fingertips.

    For each fingertip `i`:
        per_finger_i = 1 - tanh(||fingertip_i - dog|| / std)
    Return `sum(w_i * per_finger_i) / sum(w_i)` -- a weighted MEAN so the
    saturation is still 1.0 per step (regardless of how many fingers / what
    the weights sum to). Thumb is 1.5x others (the source repo `object_robot_distance`
    convention for the Allegro right hand).

    Tighter std (0.10 m) than `palm_to_dog` (0.20 m): the fingertips matter
    in the close-range "wrap around the dog" regime, not the gross-motion
    early reach.
    """
    robot: Articulation = env.scene[robot_cfg.name]
    dog: RigidObject = env.scene[dog_cfg.name]
    # Resolve the 4 fingertip body indices (order: if5, mf5, pf5, th5).
    body_ids = robot.find_bodies(list(fingertip_links))[0]
    # (num_envs, 4, 3) -- positions of the 4 fingertips in world frame.
    fingertip_w = robot.data.body_pos_w[:, body_ids, :3]
    # (num_envs, 1, 3) -- dog COM, broadcasted to subtract from all fingertips.
    dog_w = dog.data.root_pos_w[:, None, :3]
    # (num_envs, 4) -- per-finger distance.
    d = torch.norm(fingertip_w - dog_w, dim=-1)
    # (num_envs, 4) -- per-finger attractor (saturates at 1.0 each).
    per_finger = 1.0 - torch.tanh(d / max(std, 1e-6))
    # Weighted mean over fingers.
    w = torch.tensor(fingertip_weights, device=env.device, dtype=per_finger.dtype)
    w_norm = w / w.sum().clamp_min(1e-6)
    return (per_finger * w_norm.unsqueeze(0)).sum(dim=-1)
```

#### Stage 3 — `_allegro_grasp_predicate` (helper) + `grasp_contact`

```python
# ---------------------------------------------------------------------------
# Stage 3 -- the source repo Allegro grasp predicate (thumb + any of {idx, mid, pinky}).
# ---------------------------------------------------------------------------


def _allegro_grasp_predicate(
    env: "ManagerBasedRLEnv",
    contact_force_threshold: float,
) -> torch.Tensor:
    """Boolean (num_envs,) -- True iff `thumb` AND at least one of {index,
    middle, pinky} fingertips are in contact with the dog above threshold.

    Mirrors the source repo `get_allegro_contact` (for 4 sensors): "thumb opposable, any
    other finger gripping" -- the right predicate for a 4-finger Allegro hand
    grasping a small object like the 0.11 kg dog.

    Sensor name order (set by `DexGraspSceneCfg`):
        contact_sensors_0 -> if5 (index)
        contact_sensors_1 -> mf5 (middle)
        contact_sensors_2 -> pf5 (pinky)
        contact_sensors_3 -> th5 (thumb)
    Each sensor's `data.force_matrix_w` has shape (num_envs, n_bodies=1,
    n_filters=1, 3) -- one body, one filter (the dog).
    """
    forces = []
    for name in ("contact_sensors_0", "contact_sensors_1",
                 "contact_sensors_2", "contact_sensors_3"):
        sensor: ContactSensor = env.scene[name]
        # (num_envs,) -- scalar contact force magnitude on the dog filter.
        f = torch.norm(sensor.data.force_matrix_w[:, 0, 0, :], dim=-1)
        forces.append(f > contact_force_threshold)
    in_contact_idx, in_contact_mid, in_contact_pf, in_contact_thumb = forces
    any_non_thumb = in_contact_idx | in_contact_mid | in_contact_pf
    return in_contact_thumb & any_non_thumb


def grasp_contact(
    env: "ManagerBasedRLEnv",
    contact_force_threshold: float = 1.0,
) -> torch.Tensor:
    """1.0 when the Allegro hand is grasping the dog (thumb opposing any
    non-thumb finger above `contact_force_threshold` newtons), else 0.0.

    Default threshold = 1.0 N matches the source repo `get_allegro_contact` exactly. The
    dog is light (0.11 kg) so this is loose enough to fire on gentle grasps
    but tight enough that brushing against the dog with one finger does not
    spuriously count.
    """
    return _allegro_grasp_predicate(env, contact_force_threshold).float()
```

#### Stage 4 — `lift_height`

```python
# ---------------------------------------------------------------------------
# Stage 4 -- linear lift ramp gated on grasp_contact.
# ---------------------------------------------------------------------------


def lift_height(
    env: "ManagerBasedRLEnv",
    init_z: float = 0.0,
    target_lift: float = 0.30,
    contact_force_threshold: float = 1.0,
    dog_cfg: SceneEntityCfg = SceneEntityCfg("dog"),
) -> torch.Tensor:
    """`clamp((dog.z - init_z) / target_lift, 0, 1) * grasp_contact`.

    Linear ramp on the dog's env-local z position above its init height,
    multiplied by the Allegro grasp predicate. Without the grasp gate the
    policy can earn "lift" reward by knocking the dog up with the back of
    the hand or by table contact-force jitter.

    `dog.z` is computed in env-local frame (subtract env origin). The reset
    event teleports the dog to env-local (0.05, -0.35, 0.0), so init_z=0.0
    is the canonical resting height. `target_lift=0.30` matches
    `DOG_TARGET_LOCAL[2] = 0.30` exactly -- saturate when dog reaches target z.
    """
    dog: RigidObject = env.scene[dog_cfg.name]
    dog_pos_local = dog.data.root_pos_w[:, :3] - env.scene.env_origins[:, :3]
    dog_z = dog_pos_local[:, 2]
    progress = ((dog_z - init_z) / max(target_lift, 1e-6)).clamp(0.0, 1.0)
    gate = _allegro_grasp_predicate(env, contact_force_threshold).float()
    return progress * gate
```

#### Stage 5 — `dog_to_target` (gated on `grasp_contact AND lifted`)

```python
# ---------------------------------------------------------------------------
# Stage 5 -- tanh attractor on dog -> DOG_TARGET, gated on dog lifted.
# ---------------------------------------------------------------------------


def dog_to_target(
    env: "ManagerBasedRLEnv",
    std: float = 0.15,
    target_local: tuple[float, float, float] = (0.05, -0.35, 0.30),
    lift_threshold: float = 0.05,
    init_z: float = 0.0,
    contact_force_threshold: float = 1.0,
    dog_cfg: SceneEntityCfg = SceneEntityCfg("dog"),
) -> torch.Tensor:
    """`(grasp_contact AND dog.z > init_z + lift_threshold) * (1 - tanh(||dog - target|| / std))`.

    Dense full-3D attractor on the dog's env-local position toward
    `target_local`, gated on BOTH (a) the Allegro grasp predicate firing
    (thumb opposing any non-thumb finger >= `contact_force_threshold` N) AND
    (b) the dog lifted at least `lift_threshold` (5 cm) above its init height.

    Why the AND-gate: a dense xyz attractor at fingertip range from init is
    big enough that PPO will park the dog near the target xy without ever
    grasping it (knocking the dog up with the back of the hand satisfies the
    z-only lift gate). Requiring grasp_contact as well forces the policy to
    actually grip-then-lift before this reward fires.

    `target_local = (0.05, -0.35, 0.30)` matches `DOG_TARGET_LOCAL` in the
    env_cfg and `dog_reached_target` termination predicate.
    """
    dog: RigidObject = env.scene[dog_cfg.name]
    dog_pos_local = dog.data.root_pos_w[:, :3] - env.scene.env_origins[:, :3]
    target = torch.tensor(target_local, device=env.device, dtype=dog_pos_local.dtype)
    d = torch.norm(dog_pos_local - target.unsqueeze(0), dim=-1)
    base = 1.0 - torch.tanh(d / max(std, 1e-6))
    lifted = (dog_pos_local[:, 2] > (init_z + lift_threshold)).float()
    held = _allegro_grasp_predicate(env, contact_force_threshold).float()
    return held * lifted * base
```

#### Stage 6 — `success_bonus` (one-shot latch)

```python
# ---------------------------------------------------------------------------
# Stage 6 -- one-shot success bonus mirroring the termination predicate.
# ---------------------------------------------------------------------------


def success_bonus(
    env: "ManagerBasedRLEnv",
    target_local: tuple[float, float, float] = (0.05, -0.35, 0.30),
    threshold: float = 0.10,
    dog_cfg: SceneEntityCfg = SceneEntityCfg("dog"),
) -> torch.Tensor:
    """+1.0 the FIRST frame the dog meets the success predicate this episode;
    0.0 thereafter. Per-env latch resets at `env.episode_length_buf <= 1`.

    The predicate EXACTLY mirrors `terminations.dog_reached_target`:
        ||dog_local - target_local|| < threshold
    so the latched signal fires the same step the success termination kills
    the episode. Effectively a one-shot +weight bonus at task success.
    """
    latch = _get_latch_buffer(env, "success_once")
    just_reset = env.episode_length_buf <= 1
    latch = torch.where(just_reset, torch.zeros_like(latch), latch)

    dog: RigidObject = env.scene[dog_cfg.name]
    dog_pos_local = dog.data.root_pos_w[:, :3] - env.scene.env_origins[:, :3]
    target = torch.tensor(target_local, device=env.device, dtype=dog_pos_local.dtype)
    err = torch.norm(dog_pos_local - target.unsqueeze(0), dim=-1)
    now_success = err < threshold

    fire = now_success & (~latch)
    latch = latch | now_success
    _LATCH_BUFFERS[(id(env), "success_once")] = latch
    return fire.float()


# ---------------------------------------------------------------------------
# Legacy shim kept so older `placeholder` weight=0.0 RewardsCfg blocks (and
# the §6 task-generator scaffolding) still import without breaking.
# ---------------------------------------------------------------------------


def placeholder_zero(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """Constant-zero per-env reward; kept as a legacy shim."""
    return torch.zeros(env.num_envs, device=env.device)
```

### Smoke

```bash
.venv/bin/python harbor/create-task/isaac-dex-grasp/smokes/smoke_s6.py
# exit 0 (asserts: 6 active reward terms registered (palm_to_dog,
#         fingertip_to_dog, grasp_contact, lift_height, dog_to_target,
#         success_bonus), per-step reward finite + non-constant, composer
#         passthrough sum equals reward_manager.compute(), reward mean
#         within sensible bounds for a random policy)
```

---

## §7 DR

`<no DR>` — `EventCfg` contains only the two reset terms (`reset_robot_joints`, `reset_dog`). There are NO `mode="startup"` or `mode="interval"` randomization terms. The `dr-generator` was not run; the source repo source has a wider yaw range `[-pi, pi]` on the dog reset which has been pinned here to `(0, 0)` for deterministic smoke. To add later: `/harbor:create-task name=Isaac-Dex-Grasp description="add startup mass + friction + dog yaw DR" sections=7`.

---

## Reproduction

```bash
# Same source repo:
/harbor:create-task name=Isaac-Dex-Grasp-v2 from=harbor/create-task/isaac-dex-grasp-implementation.md

# Different repo: pass asset overrides if needed
/harbor:create-task name=AllegroGrasp from=isaac-dex-grasp-implementation.md \
  assets=path/to/dest/uf850_allegro_right.usd,path/to/dest/dog.usd,path/to/dest/table.usd
```

> probe-task: wrote `harbor/create-task/isaac-dex-grasp-implementation.md` (sections §1..§7, 6 reward funcs, 3 obs terms, vendored 22-D EMA cumulative-relative joint action).
> Reproduce via: `/harbor:create-task name=<new_task_id> from=<output>`.
