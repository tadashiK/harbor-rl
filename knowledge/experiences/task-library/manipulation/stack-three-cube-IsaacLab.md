# IsaacLab-Franka-StackCube — Implementation Spec

- robot: Franka FR3 arm + Franka hand (single arm)
- simulator: IsaacLab (Isaac Sim, manager-based)
- objects: three DexCubes, lab table
- bimanual: false
- summary: Stack three cubes into a tower, in order.

Task summary: a single FR3 + Franka-hand robot stacks **three** identical 4.3 cm DexCubes into a tower on a lab table. The robot base sits at world `(-0.274, +0.49, 0.01)` with `joint1=-0.785` so the EE arcs over the table. Three cubes (`cube_0` / `cube_1` / `cube_2`) spawn at staggered xy positions with ±5 cm uniform jitter (no z jitter). Goal (implicit — no `CommandsCfg`): build a 3-tier tower with `cube_1` (base, on table) ← `cube_0` ← `cube_2` (top). Two intermediate latched bonuses fire on (a) cube_0 stacked on cube_1 with the EE retreated and (b) the full 3-tier tower assembled. Episode horizon = 9.0 s @ 20 Hz = 180 control steps. Action = 3-D Cartesian EE-delta (RPY locked) + 1-D binary gripper = 4-D. Observation = 19-D, gated by a stateless mux that swaps the "currently grasping" cube between `cube_0` (state A) and `cube_2` (state B) on the predicate `cube_0_on_cube_1`.

---

## §1 Registration + Scene

### Description

`gym.register` exposes `IsaacLab-Franka-StackCube` (training) and `IsaacLab-Franka-StackCube-Play` (eval-friendly, fewer envs, observation corruption disabled). Both use `isaaclab.envs:ManagerBasedRLEnv` with the abstract `StackCubeEnvCfg` base and `FrankaStackCubeEnvCfg` subclass that fills in the single FR3 articulation, three identical DexCube `RigidObjectCfg`s, a single `ee_frame` FrameTransformer (LiftCube convention: `fr3_link0` → `fr3_hand` with `OffsetCfg(pos=[0, 0, 0.2])`), and three contact sensors (`finger_left_contact`, `finger_right_contact`, `hand_contact`) all filtered against `Cube_{0,1,2}`. Sim timing follows LiftCube (`sim.dt=1/120`, `decimation=6 → 20 Hz`, `episode_length_s=9.0`). The `FR3_FRANKA_HAND_CFG` and `FRANKA_INIT_JOINT_POS` constants are **defined locally** in this task's `config/franka/joint_pos_env_cfg.py` (NOT imported from `insert_drawer` — divergence from `lift_box`).

### Decisions resolved

| Knob | Value |
|---|---|
| Task ID (train / play) | `IsaacLab-Franka-StackCube` / `IsaacLab-Franka-StackCube-Play` |
| Robot | FR3 + Franka hand, single instance at `{ENV_REGEX_NS}/Robot` |
| Robot init pos (env-local) | `(-0.274, 0.49, 0.01)` |
| Robot init joint pose | `fr3_joint1=-0.785, joint2=-0.785, joint3=0.0, joint4=-2.655, joint5=0.0, joint6=1.87, joint7=0.0, fr3_finger_joint.*=0.04` |
| Cube USD | `${ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd` |
| Cube scale | `CUBE_SIZE / 0.06 = 0.716667` → 4.3 cm edge length. **Derive, don't hard-code** — the unscaled DexCube USD is **6 cm** (BBoxCache-measured on Isaac 5.1), so the old hard-coded `0.86` produced a 5.16 cm cube and broke the tower. See §1 code. |
| Cube mass | `0.055 kg` (`MassPropertiesCfg(mass=0.055)`) |
| `CUBE_SIZE` | `0.043` m |
| `CUBE_INIT_Z` | `0.0215` m (= `CUBE_SIZE / 2.0` — cube center half a cube above table top → base on table) |
| cube_0 init pose (env-local) | `pos=(0.0, 0.0, 0.0215)`, `rot=(1.0, 0.0, 0.0, 0.0)` |
| cube_1 init pose (env-local) | `pos=(0.0, 0.0, 0.0215)`, `rot=(1.0, 0.0, 0.0, 0.0)` |
| cube_2 init pose (env-local) | `pos=(0.0, 0.0, 0.0215)`, `rot=(1.0, 0.0, 0.0, 0.0)` |
| (NOTE) | All three cubes spawn at the same XY in `init_state`; reset events (§3) push them to staggered xy targets every reset |
| Table | `harbor/assets/table/lab_table_instanceable_colored_rotated.usd`, kinematic, surface at z≈0, env-local pos=`(0,0,0)` |
| Ground plane | world z = -0.82 |
| Light | `DomeLightCfg(color=(0.75,0.75,0.75), intensity=3000.0)` at `/World/light` |
| ee_frame | `FrameTransformerCfg` rooted at `{ENV_REGEX_NS}/Robot/fr3_link0`, target `Robot/fr3_hand` with `OffsetCfg(pos=[0, 0, 0.2])` (fingertip TCP); `debug_vis=True`, marker scale `(0.1, 0.1, 0.1)` |
| Contact sensors | `finger_left_contact` (Robot/fr3_leftfinger), `finger_right_contact` (Robot/fr3_rightfinger), `hand_contact` (Robot/fr3_hand). All three have `filter_prim_paths_expr=["{ENV_REGEX_NS}/Cube_0", "{ENV_REGEX_NS}/Cube_1", "{ENV_REGEX_NS}/Cube_2"]`, `history_length=1`, `update_period=0.0`, `debug_vis=False`. Filter index order: 0=Cube_0, 1=Cube_1, 2=Cube_2. |
| `num_envs` (default) | 4096 (train) / 50 (PLAY) |
| `env_spacing` | 2.5 m |
| `replicate_physics` | False |
| Timing | `sim.dt=1/120`, `decimation=6` → 20 Hz, `episode_length_s=9.0` (= 180 steps) |
| physx | `bounce_threshold_velocity=0.01` (set twice; second wins), `gpu_found_lost_aggregate_pairs_capacity=4*1024*1024`, `gpu_total_aggregate_pairs_capacity=16*1024`, `friction_correlation_distance=0.00625` |
| Robot articulation props | `disable_gravity=True`, `max_depenetration_velocity=5.0`, `enabled_self_collisions=True`, `solver_position_iteration_count=8`, `solver_velocity_iteration_count=0`, `activate_contact_sensors=True` |
| Robot actuators (stiffness / damping) | `fr3_joint[1-4]`: stiffness=400, damping=80, effort_limit_sim=87; `fr3_joint[5-7]`: stiffness=400, damping=80, effort_limit_sim=12; `fr3_finger_joint.*`: stiffness=2e3, damping=1e2, effort_limit_sim=200 |
| Cube rigid props | `solver_position_iteration_count=16, solver_velocity_iteration_count=1, max_angular_velocity=1000, max_linear_velocity=1000, max_depenetration_velocity=5.0, disable_gravity=False` |

### Code (verbatim)

`source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/stack_cube/__init__.py`:

```python
"""Cube-stacking environments — Franka picks up cube_0 and stacks it on top of cube_1."""
```

`source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/stack_cube/config/franka/__init__.py`:

```python
import gymnasium as gym

##
# Register Gym environments.
##

##
# EMA delta EE pose (RPY locked) — default
##

gym.register(
    id="IsaacLab-Franka-StackCube",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:FrankaStackCubeEnvCfg",
    },
)

gym.register(
    id="IsaacLab-Franka-StackCube-Play",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:FrankaStackCubeEnvCfg_PLAY",
    },
)
```

Abstract scene (`stack_cube_env_cfg.py:StackCubeSceneCfg`):

```python
_TABLE_USD_PATH = str(
    Path(__file__).resolve().parents[6]
    / "harbor" / "assets" / "table" / "lab_table_instanceable_colored_rotated.usd"
)
# Resolved at this commit:
#   <IsaacLab-repo>/harbor/assets/table/lab_table_instanceable_colored_rotated.usd


@configclass
class StackCubeSceneCfg(InteractiveSceneCfg):
    """Scene: ground + table + Franka (filled by subclass) + three cubes + ee_frame + lights."""

    # robot: filled by the per-robot subclass via __post_init__.
    robot: ArticulationCfg = MISSING
    # end-effector sensor (LiftCube convention): filled by subclass via __post_init__.
    ee_frame: FrameTransformerCfg = MISSING
    # three cubes: filled by the per-robot subclass via __post_init__.
    # Tower: cube_2 (base on table) ← cube_1 ← cube_0 (top).
    cube_0: RigidObjectCfg = MISSING
    cube_1: RigidObjectCfg = MISSING
    cube_2: RigidObjectCfg = MISSING

    # Contact sensors on the two Franka fingertips, filtered against all three
    # cubes. Per the IsaacLab ContactSensor docs, filter_prim_paths_expr only
    # works reliably when prim_path resolves to a single body per env, so we
    # register one sensor per finger (left/right). Kept from the prior edit so
    # downstream code that reads these sensors continues to work; the new
    # reward set does not use them (LiftCube doesn't read contact forces).
    finger_left_contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/fr3_leftfinger",
        update_period=0.0,
        history_length=1,
        debug_vis=False,
        filter_prim_paths_expr=[
            "{ENV_REGEX_NS}/Cube_0",
            "{ENV_REGEX_NS}/Cube_1",
            "{ENV_REGEX_NS}/Cube_2",
        ],
    )
    finger_right_contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/fr3_rightfinger",
        update_period=0.0,
        history_length=1,
        debug_vis=False,
        filter_prim_paths_expr=[
            "{ENV_REGEX_NS}/Cube_0",
            "{ENV_REGEX_NS}/Cube_1",
            "{ENV_REGEX_NS}/Cube_2",
        ],
    )
    # Hand-body contact sensor (iter 8): used to assert cube_0 has NO contact
    # with the EE/wrist body in addition to the fingertips before success fires.
    hand_contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/fr3_hand",
        update_period=0.0,
        history_length=1,
        debug_vis=False,
        filter_prim_paths_expr=[
            "{ENV_REGEX_NS}/Cube_0",
            "{ENV_REGEX_NS}/Cube_1",
            "{ENV_REGEX_NS}/Cube_2",
        ],
    )

    # Table — the source repo StackCube convention (lab_table USD, instanceable + rotated +
    # colored, kinematic). pos=(0,0,0); the table surface sits at z≈0.
    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0.0, 0.0, 0.0]),
        spawn=UsdFileCfg(
            usd_path=_TABLE_USD_PATH,
        ),
    )

    # Ground plane — the source repo convention: below the table by 0.82 m.
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

> **NOTE** — the scene docstring says "cube_2 (base on table) ← cube_1 ← cube_0 (top)" but the reward / observation modules implement a different tower ordering: success requires **cube_0 on cube_1 (lower pair)** AND **cube_2 on cube_0 (upper pair)** — i.e. the tower is `cube_1` (base) → `cube_0` → `cube_2` (top). Treat the env-cfg docstring as stale; the reward / obs modules are authoritative.

Per-robot subclass (`config/franka/joint_pos_env_cfg.py` — robot constants + scene wiring inside `__post_init__`):

```python
# Cube geometry — DexCube USD scaled to a 4.3 cm edge length, 55 g mass.
#
# The unscaled DexCube USD edge is **6 cm**, NOT 5 cm. Measured with
# UsdGeom.BBoxCache on Isaac 5.1's
# `Props/Blocks/DexCube/dex_cube_instanceable.usd`:
#     min = (-0.03, -0.03, -0.03)   max = (0.03, 0.03, 0.03)
# An earlier revision of this spec assumed 5 cm and hard-coded scale 0.86,
# which yields a **5.16 cm** cube while every threshold in §4/§5/§6 assumes
# CUBE_SIZE. The 8.6 mm error compounds per stack tier and shrinks the
# `on_stack` mux window (|dz - 0.043| < 0.01) from 10 mm of slack to 1.4 mm,
# which destabilises state B and blocks the 3-tier tower entirely.
# DERIVE the scale — never hard-code it — so a different asset revision
# cannot silently reintroduce the bug.
CUBE_SIZE = 0.043            # m — target edge length
DEX_CUBE_USD_EDGE = 0.06     # m — MEASURED unscaled edge (Isaac 5.1 DexCube)
CUBE_USD_SCALE = CUBE_SIZE / DEX_CUBE_USD_EDGE  # = 0.716667
CUBE_MASS = 0.055        # kg
CUBE_INIT_Z = CUBE_SIZE / 2.0  # center half a cube above table top → base on table

# Path to the FR3 + Franka-hand USD converted from
# `<agentic>/franka_description/urdfs/fr3_franka_hand.urdf`. Repo-relative so
# the path resolves on any clone location.
_FR3_USD_PATH = str(
    Path(__file__).resolve().parents[8] / "harbor" / "assets" / "fr3" / "fr3_franka_hand.usd"
)
# Resolved at this commit:
#   <IsaacLab-repo>/harbor/assets/fr3/fr3_franka_hand.usd

# FR3 + Franka-hand robot config — built fresh (the shipped FRANKA_PANDA_*
# cfgs assume the Isaac Sim Panda USD with `panda_*` joint/link names; the
# FR3 URDF prefixes everything with `fr3_`). Tuned with HIGH_PD stiffness so
# the joint-position tracker is stable enough for absolute-IK control.
FR3_FRANKA_HAND_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=_FR3_USD_PATH,
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=True,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=0,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(),  # filled in __post_init__
    actuators={
        "fr3_shoulder": ImplicitActuatorCfg(
            joint_names_expr=["fr3_joint[1-4]"],
            effort_limit_sim=87.0,
            stiffness=400.0,
            damping=80.0,
        ),
        "fr3_forearm": ImplicitActuatorCfg(
            joint_names_expr=["fr3_joint[5-7]"],
            effort_limit_sim=12.0,
            stiffness=400.0,
            damping=80.0,
        ),
        "fr3_hand": ImplicitActuatorCfg(
            joint_names_expr=["fr3_finger_joint.*"],
            effort_limit_sim=200.0,
            stiffness=2e3,
            damping=1e2,
        ),
    },
    soft_joint_pos_limit_factor=1.0,
)


# Init joint pose mirrors the source repo `<source-repo>/env/tasks/StackCube/env_cfg.py`:
FRANKA_INIT_JOINT_POS = {
    "fr3_joint1": -0.785,
    "fr3_joint2": -0.785,
    "fr3_joint3": 0.0,
    "fr3_joint4": -2.655,
    "fr3_joint5": 0.0,
    "fr3_joint6": 1.87,
    "fr3_joint7": 0.0,
    "fr3_finger_joint.*": 0.04,
}

# Inside FrankaStackCubeEnvCfg.__post_init__:
self.scene.robot = FR3_FRANKA_HAND_CFG.replace(
    prim_path="{ENV_REGEX_NS}/Robot",
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos=FRANKA_INIT_JOINT_POS,
        pos=(-0.274, 0.49, 0.01),
    ),
)

# End-effector frame sensor — LiftCube convention. Used by
# `mdp.cube_0_ee_distance` reward via the `ee_frame` scene entity.
marker_cfg = FRAME_MARKER_CFG.copy()
marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
marker_cfg.prim_path = "/Visuals/FrameTransformer"
self.scene.ee_frame = FrameTransformerCfg(
    prim_path="{ENV_REGEX_NS}/Robot/fr3_link0",
    debug_vis=True,
    visualizer_cfg=marker_cfg,
    target_frames=[
        FrameTransformerCfg.FrameCfg(
            prim_path="{ENV_REGEX_NS}/Robot/fr3_hand",
            name="end_effector",
            offset=OffsetCfg(pos=[0.0, 0.0, 0.2]),
        ),
    ],
)

# Three DexCubes — Nucleus USD asset, scaled to 4.3 cm edge, 55 g mass.
dex_cube_usd = f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd"
cube_spawn = UsdFileCfg(
    usd_path=dex_cube_usd,
    scale=(CUBE_USD_SCALE, CUBE_USD_SCALE, CUBE_USD_SCALE),
    mass_props=sim_utils.MassPropertiesCfg(mass=CUBE_MASS),
    rigid_props=RigidBodyPropertiesCfg(
        solver_position_iteration_count=16,
        solver_velocity_iteration_count=1,
        max_angular_velocity=1000.0,
        max_linear_velocity=1000.0,
        max_depenetration_velocity=5.0,
        disable_gravity=False,
    ),
)
self.scene.cube_0 = RigidObjectCfg(
    prim_path="{ENV_REGEX_NS}/Cube_0",
    init_state=RigidObjectCfg.InitialStateCfg(pos=[0.0, 0.0, CUBE_INIT_Z], rot=[1.0, 0.0, 0.0, 0.0]),
    spawn=cube_spawn,
)
self.scene.cube_1 = RigidObjectCfg(
    prim_path="{ENV_REGEX_NS}/Cube_1",
    init_state=RigidObjectCfg.InitialStateCfg(pos=[0.0, 0.0, CUBE_INIT_Z], rot=[1.0, 0.0, 0.0, 0.0]),
    spawn=cube_spawn,
)
self.scene.cube_2 = RigidObjectCfg(
    prim_path="{ENV_REGEX_NS}/Cube_2",
    init_state=RigidObjectCfg.InitialStateCfg(pos=[0.0, 0.0, CUBE_INIT_Z], rot=[1.0, 0.0, 0.0, 0.0]),
    spawn=cube_spawn,
)
```

EnvCfg `__post_init__` (sim/physx) — from the parent `StackCubeEnvCfg`:

```python
def __post_init__(self):
    """LiftCube timing: 100 Hz physics / decimation 2 / 5 s episode = 250 control steps."""
    # NOTE — the docstring above is stale; the actual values render to
    # decimation=6, episode_length_s=9.0, sim.dt=1/120 → 20 Hz / 180 steps.
    self.decimation = 6
    self.episode_length_s = 9.0
    # Simulation — LiftCube canonical
    self.sim.dt = 1 / 120  # 100 Hz
    self.sim.render_interval = self.decimation
    # Physics knobs — LiftCube canonical
    self.sim.physx.bounce_threshold_velocity = 0.2
    self.sim.physx.bounce_threshold_velocity = 0.01  # second assignment wins
    self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 1024 * 1024 * 4
    self.sim.physx.gpu_total_aggregate_pairs_capacity = 16 * 1024
    self.sim.physx.friction_correlation_distance = 0.00625


@configclass
class FrankaStackCubeEnvCfg_PLAY(FrankaStackCubeEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        # Smaller scene for play.
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # Disable observation noise for play.
        self.observations.policy.enable_corruption = False
```

### Smoke

```bash
.venv/bin/python harbor/create-task/isaaclab-franka-stackcube/smokes/smoke_s1.py
# exit 0 (asserts action_space + observation_space non-None, max_episode_length > 0, num_envs == 128 under smoke override)
```

---

## §2 Actions

### Description

Single-arm 3-D xyz EMA-smoothed EE-delta + binary gripper. Action vector layout is `[arm_action (3), gripper_action (1)] = 4-D`. The `EMACumulativeDeltaPositionActionCfg` class is **vendored locally** under `stack_cube/mdp/` (NOT imported from `insert_drawer` — divergence from `lift_box`). Controller is `DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls")` — absolute target pose mode where the action provides cumulative delta on top of the post-reset EE pose. Per-step the action term: clamps raw action to `[-1, 1]`, scales by `(0.01, 0.01, 0.01)`, integrates into `del_action`, computes `abs_pos = init_ee_pos + del_action`, EMA-smooths against the previously-applied position target with `alpha=0.5`, clamps in robot-root frame, and forwards a 7-D pose target (xyz from the smoothed action; quat locked at the post-reset value) to the IK controller. **Note the `scale=(0.01, 0.01, 0.01)` here vs. the actions_cfg.py default `(0.02, 0.02, 0.02)`** — the franka subclass overrides the default to a tighter per-axis scale.

### Decisions resolved

| Knob | Value |
|---|---|
| Arm action class | `mdp.EMACumulativeDeltaPositionActionCfg` (locally vendored under `stack_cube/mdp/`) |
| Gripper action class | `mdp.BinaryJointPositionActionCfg` (re-exported from `isaaclab.envs.mdp` via `from .actions_cfg import *` and `from isaaclab.envs.mdp import *`) |
| Per-arm dim | 3 (xyz delta only — RPY locked at post-reset value) |
| Total action dim | 4 |
| Arm `asset_name` | `"robot"` |
| Arm `joint_names` | `["fr3_joint.*"]` |
| Arm `body_name` | `"fr3_hand"` |
| Arm `body_offset` | `OffsetCfg(pos=(0.0, 0.0, 0.2))` |
| Controller | `DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls")` |
| `scale` | `(0.01, 0.01, 0.01)` per axis |
| `alpha` | `0.5` (EMA weight on new delta target) |
| pos_lower_limit (root frame) | `[0.05, -0.52, 0.005]` |
| pos_upper_limit (root frame) | `[0.50, -0.05, 0.16]` |
| Gripper joints | `["fr3_finger.*"]`, open=`{"fr3_finger_.*": 0.04}`, close=`{"fr3_finger_.*": 0.0}` |

### Code (verbatim)

`stack_cube/mdp/actions_cfg.py` (full file — action config):

```python
"""Cfg for EMACumulativeDeltaPositionAction — fixed-RPY (position-only) variant.

Edit_mode_014 (§2): swap stock joint-position action for this position-only
EMA EE-delta term. The policy outputs a 3-D `(dx, dy, dz)` delta. The EE
orientation is FIXED at the post-reset value (no rotation channels, no
axis-angle compose).

Inherits `DifferentialInverseKinematicsActionCfg`, so `asset_name`,
`joint_names`, `body_name`, `body_offset`, `controller` all behave the same.
The `controller` field MUST have `command_type="pose"` and
`use_relative_mode=False` — the action term forces absolute IK internally
and will raise otherwise.

Extras beyond the parent:
    scale               3-element scale (sx, sy, sz). Overrides the parent's
                        float scale with a 3-tuple of per-axis position scales.
    alpha               EMA weight in [0, 1]. 1.0 = no smoothing.
    pos_lower_limit     Optional 3-element list[float]. Per-axis position
    pos_upper_limit     clamp on the post-EMA pose target. Both must be set
                        together; otherwise no clamp.
"""
from isaaclab.envs.mdp.actions import DifferentialInverseKinematicsActionCfg
from isaaclab.managers.action_manager import ActionTerm
from isaaclab.utils import configclass

from .actions import EMACumulativeDeltaPositionAction


@configclass
class EMACumulativeDeltaPositionActionCfg(DifferentialInverseKinematicsActionCfg):
    class_type: type[ActionTerm] = EMACumulativeDeltaPositionAction

    scale: tuple[float, float, float] = (0.02, 0.02, 0.02)
    """Per-axis position scale (sx, sy, sz). Default 0.02 m / unit policy output
    (plugin convention — 2 cm/step is a noticeable Cartesian move). No rotation
    channels in this variant — orientation is locked at the post-reset EE quat.
    """

    alpha: float = 0.5
    """EMA weight in [0, 1]. Default 0.5 (plugin convention — half-and-half blend
    against the previously-applied position target). 1.0 disables smoothing."""

    pos_lower_limit: list[float] | None = None
    pos_upper_limit: list[float] | None = None
    """Optional per-axis (xyz) clamp on the post-EMA position target."""
```

`stack_cube/mdp/actions.py` (full file — action term implementation):

```python
"""EMA cumulative-delta task-space (EE position) action — fixed RPY variant.

Edit_mode_014 (§2): swap stock `mdp.JointPositionActionCfg` for a custom
position-only EMA EE-delta term. This is the 3-D-position-only variant of
the plugin's `EMACumulativeDeltaPoseAction` template — the policy outputs
ONLY a 3-D position delta `(dx, dy, dz)`. The EE quaternion is FIXED at the
post-reset value (no rotation channel, no axis-angle compose).

Behavior:
    1. Accumulates the 3-D position delta over the episode:
           delta_t = delta_{t-1} + s · a_t
    2. Anchors on the EE pose captured lazily at the first
       `process_actions` after `env.reset()`:
           abs_pos_t  = init_ee_pos + delta_t
           abs_quat_t = init_ee_quat                       (RPY locked)
    3. EMA-smooths against the previously applied target on the position
       channel only (orientation stays exactly at `init_ee_quat`):
           target_t.pos  = α · abs_pos_t + (1 - α) · prev_applied_pos
           target_t.quat = init_ee_quat
    4. Forwards the absolute 7-D pose target to the IK controller (forced
       into `use_relative_mode=False`) and reuses the parent's
       `apply_actions()` for jacobian + IK numerics.

The policy interface is 3-D (`action_dim == 3`); the IK command is 7-D
(handled internally by composing `(pos, init_ee_quat)`).

State across steps: `del_action` (3-D), `_prev_applied_pos` (3-D).
On `env.reset()`: `del_action := 0`, re-anchor flag set so the NEXT
`process_actions` captures fresh `init_ee_{pos,quat}` from `body_pose_w`.
"""
from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

from isaaclab.envs.mdp.actions.task_space_actions import DifferentialInverseKinematicsAction

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

    from . import actions_cfg


class EMACumulativeDeltaPositionAction(DifferentialInverseKinematicsAction):
    cfg: "actions_cfg.EMACumulativeDeltaPositionActionCfg"

    def __init__(
        self,
        cfg: "actions_cfg.EMACumulativeDeltaPositionActionCfg",
        env: "ManagerBasedEnv",
    ) -> None:
        # Force the IK controller into absolute pose mode — we manage relative deltas ourselves.
        if cfg.controller.use_relative_mode:
            raise ValueError(
                "EMACumulativeDeltaPositionAction handles relative deltas itself; "
                "set controller.use_relative_mode=False"
            )
        if cfg.controller.command_type != "pose":
            raise ValueError(
                "EMACumulativeDeltaPositionAction requires controller.command_type='pose'; "
                f"got '{cfg.controller.command_type}'"
            )
        super().__init__(cfg, env)

        # Override the policy-input action shape to 3 (position delta only).
        # The parent allocated _raw_actions / _processed_actions / _scale at
        # the IK controller's action_dim (=7 for pose+abs). We need:
        #   raw_actions / scale : 3-D (policy interface)
        #   _processed_actions  : 7-D (IK controller command in abs pose mode)
        self._raw_actions = torch.zeros(env.num_envs, 3, device=env.device)
        self._processed_actions = torch.zeros(env.num_envs, 7, device=env.device)
        self._scale = torch.zeros((env.num_envs, 3), device=env.device)
        self._scale[:] = torch.tensor(cfg.scale, device=env.device)

        # Alpha (EMA weight)
        if not 0.0 <= cfg.alpha <= 1.0:
            raise ValueError(f"alpha must be in [0, 1]. Got {cfg.alpha}.")
        self._alpha = cfg.alpha

        # Cumulative position delta state (3-D)
        self.del_action = torch.zeros((env.num_envs, 3), device=env.device)

        # Init pose buffers — populated lazily on the next process_actions for
        # any env in `_needs_reanchor`. We CANNOT capture init pose here (or in
        # reset()) because at those call sites the articulation's body data is
        # stale — IsaacLab's ManagerBasedRLEnv runs the post-reset sim step
        # AFTER all term.reset() calls. Deferring to process_actions means we
        # read body_pose_w once it's been refreshed by the env's internal
        # post-reset step.
        self.init_ee_pos        = torch.zeros((env.num_envs, 3), device=env.device)
        self.init_ee_quat       = torch.zeros((env.num_envs, 4), device=env.device)
        self.init_ee_quat[:, 0] = 1.0                                                  # identity quat (wxyz)
        self._prev_applied_pos  = torch.zeros((env.num_envs, 3), device=env.device)
        # Mark all envs for re-anchoring on first process_actions.
        self._needs_reanchor    = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)

        # Optional position clamp (per-axis lower/upper)
        self.pos_lower_limit = (
            torch.tensor(cfg.pos_lower_limit, device=self.device)
            if cfg.pos_lower_limit is not None
            else None
        )
        self.pos_upper_limit = (
            torch.tensor(cfg.pos_upper_limit, device=self.device)
            if cfg.pos_upper_limit is not None
            else None
        )

    # Policy sees a 3-D action; IK command is 7-D (handled internally).
    @property
    def action_dim(self) -> int:
        return 3

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        super().reset(env_ids)
        # Defer init_ee_pos refresh — articulation body_pose_w is stale here.
        if env_ids is None:
            self._needs_reanchor[:] = True
            self.del_action[:]      = 0.0
        else:
            self._needs_reanchor[env_ids] = True
            self.del_action[env_ids]      = 0.0

    def process_actions(self, actions: torch.Tensor) -> None:
        # Re-anchor init pose for any env that just reset. body_pose_w is now
        # fresh (the env's post-reset sim step has run since reset()).
        if self._needs_reanchor.any():
            ee_pos_curr, ee_quat_curr = self._compute_frame_pose()
            mask = self._needs_reanchor
            self.init_ee_pos[mask]      = ee_pos_curr[mask]
            self.init_ee_quat[mask]     = ee_quat_curr[mask]
            self._prev_applied_pos[mask] = ee_pos_curr[mask]
            self._needs_reanchor[:]     = False

        # Clamp the policy's raw action to [-1, 1] before any further compute.
        # With scale=0.02 this caps the per-step Cartesian delta at 2 cm/axis,
        # regardless of how aggressive the (unbounded) policy logit happens to be.
        actions = torch.clamp(actions, -1.0, 1.0)

        # Store raw (already clamped) 3-D action and apply per-axis scale.
        self._raw_actions[:] = actions
        scaled = actions * self._scale  # (N, 3)

        # Cumulative position delta over the episode.
        self.del_action += scaled

        # Absolute position target (RPY is locked at init).
        abs_pos = self.init_ee_pos + self.del_action  # (N, 3)

        # EMA on the position channel only.
        ema_pos = self._alpha * abs_pos + (1.0 - self._alpha) * self._prev_applied_pos

        # Optional per-axis position clamp.
        if self.pos_lower_limit is not None and self.pos_upper_limit is not None:
            ema_pos = torch.clamp(ema_pos, self.pos_lower_limit, self.pos_upper_limit)

        # Store the 7-D absolute pose target: (pos, init_ee_quat).
        self._processed_actions[:, :3] = ema_pos
        self._processed_actions[:, 3:7] = self.init_ee_quat
        self._prev_applied_pos[:] = ema_pos

        # Hand off to the IK controller in absolute mode.
        ee_pos_curr, ee_quat_curr = self._compute_frame_pose()
        self._ik_controller.set_command(self._processed_actions, ee_pos_curr, ee_quat_curr)
```

`stack_cube_env_cfg.py:ActionsCfg` (skeleton — fields filled by the franka subclass):

```python
@configclass
class ActionsCfg:
    """Action specs — filled by per-robot subclass.

    * arm_action — `mdp.EMACumulativeDeltaPositionActionCfg` (3-D position-only
      EMA EE-delta action; xyz only, EE quaternion locked at the post-reset
      value). Cfg in `config/franka/joint_pos_env_cfg.py` uses
      `scale=(0.02, 0.02, 0.02)`, `alpha=0.5`, body `panda_hand` with offset
      [0, 0, 0.1034] m, IK controller in pose / abs / dls mode.
    * gripper_action — `mdp.BinaryJointPositionActionCfg` over the 2 finger
      joints (open=0.04 m / close=0.0 m).

    Total action dim = 3 (xyz EE-delta) + 1 (binary gripper) = 4.
    """

    arm_action: mdp.EMACumulativeDeltaPositionActionCfg = MISSING
    gripper_action: mdp.BinaryJointPositionActionCfg = MISSING
```

`config/franka/joint_pos_env_cfg.py:FrankaStackCubeEnvCfg.__post_init__` (action wiring):

```python
# Action — Cartesian EE-delta with locked RPY. Policy outputs 3-D xyz
# delta; IK target = fingertip TCP at body_offset=(0,0,0.2) — same
# frame as ee_frame OffsetCfg. pos_lower_limit[2]=0.005 keeps the
# fingertip from driving through the table (root-frame z; world floor
# = root_z + 0.005 ≈ 0.015 given robot base at world z=0.01).
self.actions.arm_action = mdp.EMACumulativeDeltaPositionActionCfg(
    asset_name="robot",
    joint_names=["fr3_joint.*"],
    body_name="fr3_hand",
    body_offset=DifferentialInverseKinematicsActionCfg.OffsetCfg(pos=(0.0, 0.0, 0.2)),
    controller=DifferentialIKControllerCfg(
        command_type="pose",
        use_relative_mode=False,
        ik_method="dls",
    ),
    scale=(0.01, 0.01, 0.01),
    alpha=0.5,
    # Workspace clamp in ROBOT ROOT FRAME (robot base @ world
    # (-0.274, 0.49, 0.01)). Tight bounds matching where the cubes
    # can spawn (per `reset_cube_*` events: world x ∈ [-0.20, 0.20],
    # world y ∈ [0.0, 0.4]) plus ~3 cm margin so the EE can swing
    # around a cube edge to grasp it. z-ceiling = target_z(world
    # 0.1075) + 5 cm headroom = 0.1575 world = 0.15 root. Keeps the
    # IK from being asked to drive outside the reachable workspace
    # and prevents lift-overshoot above the stacking height.
    pos_lower_limit=[ 0.05, -0.52, 0.005],
    pos_upper_limit=[ 0.50, -0.05, 0.16 ],
)
self.actions.gripper_action = mdp.BinaryJointPositionActionCfg(
    asset_name="robot",
    joint_names=["fr3_finger.*"],
    open_command_expr={"fr3_finger_.*": 0.04},
    close_command_expr={"fr3_finger_.*": 0.0},
)
```

### Smoke

```bash
.venv/bin/python harbor/create-task/isaaclab-franka-stackcube/smokes/smoke_s2.py
# exit 0 (asserts action_space.shape[-1] == 4, env.step(action) returns finite obs/reward)
```

---

## §3 Reset

### Description

Four reset terms — the single robot articulation (joint positions pinned to URDF home, no jitter, no init velocity) and one term per cube. Each cube has a different staggered XY range so they spawn at distinct table positions after reset (no z jitter). The robot reset term uses `mdp.reset_joints_by_scale` with `position_range=(1.0, 1.0)` — multiplicative on the configured init joint pose, so the home pose is exactly reproduced.

### Decisions resolved

| Term | Function | `position_range` / `pose_range` | `velocity_range` | asset_cfg |
|---|---|---|---|---|
| reset_robot_joints | `mdp.reset_joints_by_scale` | `position_range=(1.0, 1.0)` | `(0.0, 0.0)` | (default — `SceneEntityCfg("robot")`) |
| reset_cube_0 | `mdp.reset_root_state_uniform` | `{"x": (0.0, 0.1), "y": (0.15, 0.25), "z": (0.0, 0.0)}` | `{}` | `SceneEntityCfg("cube_0")` |
| reset_cube_1 | `mdp.reset_root_state_uniform` | `{"x": (0.0, 0.1), "y": (0.0, 0.1), "z": (0.0, 0.0)}` | `{}` | `SceneEntityCfg("cube_1")` |
| reset_cube_2 | `mdp.reset_root_state_uniform` | `{"x": (-0.15, -0.05), "y": (0.0, 0.1), "z": (0.0, 0.0)}` | `{}` | `SceneEntityCfg("cube_2")` |

### Code (verbatim)

`stack_cube_env_cfg.py:EventCfg`:

```python
@configclass
class EventCfg:
    """Reset terms (§3). DR terms are added by `dr-generator` in §7."""

    # Robot joints: pinned to (1.0, 1.0) by smoke checks; dr-generator may widen.
    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (1.0, 1.0),
            "velocity_range": (0.0, 0.0),
        },
    )

    # Per-cube position resets — mirror LiftCube's `reset_object_position`
    # ranges (small additive xy perturbation around each cube's `init_state.pos`).
    reset_cube_0 = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (0.0, 0.1), "y": (0.15, 0.25), "z": (0.0, 0.0)},
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("cube_0"),
        },
    )
    reset_cube_1 = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (0.0, 0.1), "y": (0.0, 0.1), "z": (0.0, 0.0)},
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("cube_1"),
        },
    )
    reset_cube_2 = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.15, -0.05), "y": (0.0, 0.1), "z": (0.0, 0.0)},
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("cube_2"),
        },
    )
```

`stack_cube/mdp/events.py` (full file — placeholder, no custom event terms; per-cube reset uses the shared `mdp.reset_root_state_uniform`):

```python
"""Custom event-manager terms for the stack_cube task.

There are no task-specific event terms. Per-cube XY/Z spawn pose comes from
the standard ``mdp.reset_root_state_uniform`` terms wired in
``stack_cube_env_cfg.EventCfg``.

This module is kept (rather than deleted) so ``mdp/__init__.py``'s
``from .events import *`` keeps working without producing import errors.
``dr-generator`` may add startup / interval DR helpers here in §7.
"""
```

### Smoke

```bash
.venv/bin/python harbor/create-task/isaaclab-franka-stackcube/smokes/smoke_s3.py
# exit 0 (asserts env.reset() places robot at URDF home and each cube within its declared xy range)
```

---

## §4 Goal + Termination

### Description

No `CommandsCfg` — `commands = None` at the EnvCfg level. The goal is **implicit**, defined entirely by the §6 latched reward bonuses:

- `cube_0_stacked_bonus_once_per_episode`: cube_0 on cube_1 (geometric + EE retreated ≥4 cm + no contact between gripper/ee and cube_0).
- `three_tier_tower_bonus_once_per_episode`: full 3-tier tower (cube_0 on cube_1 AND cube_2 on cube_0), each pair geometric + no gripper/ee contact with the relevant cube.

The only termination is `time_out` (after 180 control steps). No failure-mode terminations — by design, dropped stacks do NOT kill the episode (the docstring on `TerminationsCfg` calls this out explicitly as option B from 2026-05-17 design). A `stack_broke_penalty` reward (single-fire −200 latch) signals the failure but lets the policy try again within the horizon.

### Decisions resolved

| Term | `func` | `params` | `time_out` flag |
|---|---|---|---|
| time_out | `mdp.time_out` | `{}` | `True` |

`commands = None` at the EnvCfg level. No `success` termination — success is recorded only via the latched reward bonuses in §6.

### Code (verbatim)

`stack_cube_env_cfg.py:TerminationsCfg`:

```python
@configclass
class TerminationsCfg:
    """Time-out only — no failure terminations.

    Both `stack_broken` and `tower_broken` are intentionally NO LONGER
    terminations (option B from 2026-05-17 design discussion). The policy
    keeps the full horizon even after knocking the stack apart, so it can
    re-grasp + re-stack and try again. Per-event reward signals still exist
    via `stack_broke_penalty` (single-fire -200 latch in §6); they just
    don't kill the rollout. This frees expected-value math for risky
    stage-2 exploration — a failed attempt costs reward but not opportunity.

    Tower order (bottom-up): cube_1 (base on table) → cube_0 → cube_2 (top).
    """

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
```

`stack_cube/mdp/terminations.py` (full file — only a private helper used by §6 reward):

```python
"""Termination helpers for the stack_cube task.

The only active termination wired into `TerminationsCfg` is `time_out` from
the shared `isaaclab.envs.mdp` namespace; this module only defines the
cross-module helper `_no_contact_between_cube_and_gripper_or_ee`, used by
`mdp.rewards.three_tier_tower_bonus_once_per_episode`.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _no_contact_between_cube_and_gripper_or_ee(
    env: "ManagerBasedRLEnv",
    cube_idx: int,
    eps: float = 1e-3,
) -> torch.Tensor:
    """True per env if neither fingertip NOR the panda_hand body has contact
    force vs `cube_<cube_idx>`. `cube_idx` ∈ {0, 1, 2} matches the filter list
    declared on each contact sensor in StackCubeSceneCfg.
    """
    left = env.scene["finger_left_contact"]
    right = env.scene["finger_right_contact"]
    hand = env.scene["hand_contact"]
    left_f = torch.norm(left.data.force_matrix_w[:, 0, cube_idx, :], dim=-1)
    right_f = torch.norm(right.data.force_matrix_w[:, 0, cube_idx, :], dim=-1)
    hand_f = torch.norm(hand.data.force_matrix_w[:, 0, cube_idx, :], dim=-1)
    in_contact = (left_f > eps) | (right_f > eps) | (hand_f > eps)
    return ~in_contact
```

> **NOTE** — there is an empty `CurriculumCfg` declared at the env-cfg level:
>
> ```python
> @configclass
> class CurriculumCfg:
>     """Empty — LiftCube's 1000× regularizer ramp at step 10k was suppressing
>     the cube-release motion needed to stack. Keeping action_rate / joint_vel
>     at their initial −1e-4 weights throughout training."""
>     pass
> ```
>
> No live curriculum terms — keep the empty class so the manager doesn't complain.

### Smoke

```bash
.venv/bin/python harbor/create-task/isaaclab-franka-stackcube/smokes/smoke_s4.py
# exit 0 (asserts time_out fires at step >= max_episode_length and no other termination triggers)
# Optional: smoke_success.py / smoke_success_visualize.py write cube poses directly to test the
# success-bonus latching path (not present as a DoneTerm — fires only inside the §6 reward).
```

---

## §5 Observation

### Description

Single `PolicyCfg` group, all terms concatenated → 19-D policy obs. The obs vector composes the EE pose in robot root frame, two muxed cube positions (the "grasping cube" and the "grasping target"), the gripper joint positions, and the last action vector. A stateless per-step mux switches between `cube_0` and `cube_2` (and between `cube_1.xyz+[0,0,CUBE_SIZE]` and `cube_0.xyz+[0,0,CUBE_SIZE]` for the target) based on the predicate `_cube_0_on_cube_1_predicate`. `enable_corruption=True` (LiftCube design preserved); no per-term noise slots are populated. The 9-D `joint_pos` term mentioned in the docstring is NOT in the active layout — the §5 re-author landed on the 5-term sequence listed below.

### Decisions resolved (total obs dim 19)

| ObsTerm | `func` | `params` | dim |
|---|---|---|---:|
| ee_pose | `mdp.ee_pose_in_robot_root_frame` | (defaults; reads `ee_frame_cfg=SceneEntityCfg("ee_frame")`) | 7 |
| grasping_cube_position | `mdp.grasping_cube_position_in_robot_root_frame` | (none) | 3 |
| grasping_target_position | `mdp.grasping_target_position_in_robot_root_frame` | (none) | 3 |
| gripper_pos | `mdp.joint_pos` | `asset_cfg=SceneEntityCfg("robot", joint_names=["fr3_finger.*"])` | 2 |
| actions | `mdp.last_action` | (defaults) | 4 |
| **Total** | | | **19** |

PolicyCfg flags: `enable_corruption = True`, `concatenate_terms = True`. No per-term noise slots populated.

### Code (verbatim)

`stack_cube_env_cfg.py:ObservationsCfg`:

```python
@configclass
class ObservationsCfg:
    """Observation specs — 3-tier tower, grasping-cube state-machine mux.

    Edit_mode_012: extended to expose all three cubes plus two per-pair
    targets so the policy can drive cube_0 onto cube_1 (top of tower) AND
    cube_1 onto cube_2 (base on table). (Term order is preserved in the §5
    re-author note below.)

    Edit_mode_013 (§5 re-author): the policy obs is collapsed to a 4-term
    layout. `joint_vel` is dropped (the `joint_vel_l2` reward still reads
    `robot.data` directly, not the obs, so the regularizer is unaffected).
    All per-cube absolute positions and per-pair targets are replaced by a
    stateless per-step mux on the predicate `cube_0_on_cube_1`:

      not-yet-stacked: grasping_cube      <- cube_0
                       grasping_target    <- cube_1.xyz + [0,0,CUBE_SIZE]
      stacked:         grasping_cube      <- cube_2
                       grasping_target    <- cube_0.xyz + [0,0,CUBE_SIZE]

    New term order:

        joint_pos                  (9,)  — mdp.joint_pos_rel (all 9 robot joints)
        grasping_cube_position     (3,)  — mdp.grasping_cube_position_in_robot_root_frame
        grasping_target_position   (3,)  — mdp.grasping_target_position_in_robot_root_frame
        actions                    (4,)  — mdp.last_action (3 xyz + 1 gripper)
        Total = 9+3+3+4 = 19.

    enable_corruption=True (LiftCube design preserved).
    """

    @configclass
    class PolicyCfg(ObsGroup):
        # Concatenation order: ee_pose -> grasping_cube_position ->
        # grasping_target_position -> gripper_pos -> actions.
        # ee_pose (7) + grasping_cube_position (3) + grasping_target_position
        # (3) + gripper_pos (2) + actions (4) = 19.
        ee_pose = ObsTerm(func=mdp.ee_pose_in_robot_root_frame)
        grasping_cube_position = ObsTerm(func=mdp.grasping_cube_position_in_robot_root_frame)
        grasping_target_position = ObsTerm(func=mdp.grasping_target_position_in_robot_root_frame)
        gripper_pos = ObsTerm(
            func=mdp.joint_pos,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=["fr3_finger.*"])},
        )
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True  # LiftCube design — corruption ON
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()
```

> **NOTE** — the docstring on `ObservationsCfg` describes a "4-term layout" with a leading `joint_pos (9,)` adding to 19. The actual code lists a 5-term layout (`ee_pose (7) + grasping_cube_position (3) + grasping_target_position (3) + gripper_pos (2) + actions (4) = 19`) — same total but a different decomposition. Use the code, not the docstring.

`stack_cube/mdp/observations.py` (full file — task-local helpers):

```python
"""Observation helpers for the stack_cube task.

The policy obs is a 5-term layout
(ee_pose + grasping_cube_position + grasping_target_position + gripper_pos +
actions = 19). A stateless per-step mux switches the "currently relevant"
cube + target based on whether cube_0 is already on cube_1
(`_cube_0_on_cube_1_predicate` here mirrors the same predicate used by the
§6 reward `mdp.rewards._cube_0_on_cube_1_predicate`).
"""
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
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """7-D end-effector pose in the robot root frame: `[x, y, z, qw, qx, qy, qz]`.

    Reads world-frame EE pose from the `ee_frame` FrameTransformer (target 0;
    LiftCube convention: `panda_hand` with [0,0,0.1034] offset) and transforms
    it into the robot's root frame via `subtract_frame_transforms`.
    """
    robot: RigidObject = env.scene["robot"]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    ee_pos_w = ee_frame.data.target_pos_w[..., 0, :]
    ee_quat_w = ee_frame.data.target_quat_w[..., 0, :]
    ee_pos_b, ee_quat_b = subtract_frame_transforms(
        robot.data.root_pos_w, robot.data.root_quat_w, ee_pos_w, ee_quat_w
    )
    return torch.cat([ee_pos_b, ee_quat_b], dim=-1)


# Cube edge length — must stay in sync with the constant in rewards.py /
# terminations.py and the USD scale in config/franka/joint_pos_env_cfg.py.
CUBE_SIZE = 0.043


def _cube_pos_in_robot_root_frame(env: "ManagerBasedRLEnv", cube_key: str) -> torch.Tensor:
    """Shared helper: cube xyz in the robot root frame."""
    robot: RigidObject = env.scene["robot"]
    cube: RigidObject = env.scene[cube_key]
    cube_pos_w = cube.data.root_pos_w[:, :3]
    cube_pos_b, _ = subtract_frame_transforms(robot.data.root_pos_w, robot.data.root_quat_w, cube_pos_w)
    return cube_pos_b


def _stack_target_in_root_frame(env: "ManagerBasedRLEnv", base_cube_key: str) -> torch.Tensor:
    """Shared helper: `base_cube.pos_w + [0, 0, CUBE_SIZE]` in the robot root frame."""
    robot: RigidObject = env.scene["robot"]
    base: RigidObject = env.scene[base_cube_key]
    target_w = base.data.root_pos_w[:, :3].clone()
    target_w[:, 2] = target_w[:, 2] + CUBE_SIZE
    target_b, _ = subtract_frame_transforms(robot.data.root_pos_w, robot.data.root_quat_w, target_w)
    return target_b


def _cube_0_on_cube_1_predicate(
    env: "ManagerBasedRLEnv",
    xy_threshold: float = 0.02,
    z_threshold: float = 0.01,
) -> torch.Tensor:
    """True per env when cube_0 is stacked on cube_1 (position-only).

    Mirrors `mdp.terminations.cube_0_stacked_on_cube_1` but lives here so the
    observation mux doesn't take a cross-module dependency on the termination
    module. Same thresholds.
    """
    cube_0: RigidObject = env.scene["cube_0"]
    cube_1: RigidObject = env.scene["cube_1"]
    top = cube_0.data.root_pos_w[:, :3]
    bot = cube_1.data.root_pos_w[:, :3]
    xy_dist = torch.norm(top[:, :2] - bot[:, :2], dim=-1)
    z_gap = top[:, 2] - bot[:, 2]
    return (xy_dist < xy_threshold) & (torch.abs(z_gap - CUBE_SIZE) < z_threshold)


def grasping_cube_position_in_robot_root_frame(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """xyz of the cube the policy currently needs to grasp (robot-root frame).

    Stateless per-step mux:
        cube_0_on_cube_1 -> cube_2   (second-stage target object)
        otherwise        -> cube_0   (initial state OR after a drop reverts here)
    """
    cube_0_pos = _cube_pos_in_robot_root_frame(env, "cube_0")
    cube_2_pos = _cube_pos_in_robot_root_frame(env, "cube_2")
    use_cube_2 = _cube_0_on_cube_1_predicate(env).unsqueeze(-1)  # (N, 1)
    return torch.where(use_cube_2, cube_2_pos, cube_0_pos)


def grasping_target_position_in_robot_root_frame(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """xyz target for the current grasping cube (robot-root frame).

    Stateless per-step mux:
        cube_0_on_cube_1 -> cube_0.xyz + [0,0,CUBE_SIZE]   (cube_2 stacks on cube_0)
        otherwise        -> cube_1.xyz + [0,0,CUBE_SIZE]   (cube_0 stacks on cube_1)
    """
    target_on_cube_1 = _stack_target_in_root_frame(env, "cube_1")
    target_on_cube_0 = _stack_target_in_root_frame(env, "cube_0")
    use_target_on_cube_0 = _cube_0_on_cube_1_predicate(env).unsqueeze(-1)
    return torch.where(use_target_on_cube_0, target_on_cube_0, target_on_cube_1)
```

### Smoke

```bash
.venv/bin/python harbor/create-task/isaaclab-franka-stackcube/smokes/smoke_s5.py
# exit 0 (asserts observation_space == Box(19,), per-term dims match the expected layout 7+3+3+2+4=19)
```

---

## §6 Reward

### Description

Composer = **sum** (verified at the call site in `isaaclab.managers.reward_manager.RewardManager.compute` — line 153: `value = term_cfg.func(...) * term_cfg.weight`; accumulated via `self._reward_buf += value`). **7 reward terms, all active.**

The reward is a grasping-cube mux + sparse latched bonuses + a no-op release-shaping placeholder. The §5 obs mux and §6 reward share the same `_cube_0_on_cube_1_predicate` (with identical thresholds — xy<0.02 AND |Δz−CUBE_SIZE|<0.01) so the "currently grasped cube" flips on the same instant in obs and reward. In state B, the dense terms (`reach`, `lift`, `align`, `linear_lift_grasping_cube`) are multiplied by an integer scale + offset (30·base+1, 10·lifted+1, 50·base+1, 10·base·gate+1) to outweigh the state-A magnitudes and prevent the policy from regressing when the predicate flips.

Weights below are **nominal per-step magnitudes** — the declared weight is exactly what each term pays per step.

### Decisions resolved

| RewTerm | `func` | `params` | `weight` |
|---|---|---|---:|
| reach | `mdp.grasping_cube_ee_distance` | `std=0.1` | **0.02** |
| lift | `mdp.grasping_cube_is_lifted` | `minimal_height=0.04` | **0.1** |
| align | `mdp.grasping_cube_goal_distance` | `std=0.08, minimal_height=0.04, minimal_height_b=0.0875` | **0.32** |
| success_bonus | `mdp.cube_0_stacked_bonus_once_per_episode` | `xy_threshold=0.02, z_threshold=0.01` | **200.0** |
| stack_broke_penalty | `mdp.cube_0_stack_broken_penalty_once_per_episode` | `xy_threshold=0.02, z_threshold=0.01` | **-200.0** |
| tower_bonus | `mdp.three_tier_tower_bonus_once_per_episode` | `xy_threshold=0.02, z_threshold=0.01` | **2000.0** |
| linear_lift_grasping_cube | `mdp.linear_lift_grasping_cube` | `init_z=0.0215, target_z_a=0.06, target_z_b=0.1075, contact_force_threshold=1e-3` | **0.15** |

`align.minimal_height_b = 0.0875` is authoritative — it is `table_top + 1.5·CUBE_SIZE`, i.e.
cube_2 must clear the existing two-cube stack before `align` fires. (Independently
corroborated by both sibling ports: Genesis `ALIGN_MIN_HEIGHT_B = 0.0875` and ManiSkill
`min_h_b_align = 0.0875`.)

#### Per-stage saturated per-step magnitude budget (nominal weights)

| Stage | Term | State-A peak | State-B peak | Notes |
|---|---|---:|---:|---|
| Reach | reach (w=0.02) | 0.02·1 = 0.02 | 0.02·(30+1) = 0.62 | tanh saturates at 1; state-B has `30·base+1` |
| Lift | lift (w=0.1) | 0.1·1 = 0.1 | 0.1·(10+1) = 1.1 | binary lifted flag; state-B `10·lifted+1` |
| Align | align (w=0.32) | 0.32·1 = 0.32 | 0.32·(50+1) = 16.32 | gated on `lifted`; state-B `50·base+1` |
| Linear lift | linear_lift_grasping_cube (w=0.15) | 0.15·1 = 0.15 | 0.15·(10+1) = 1.65 | contact-gated; state-B `10·base·gate+1` |
| Subtotal dense per-step | | ~0.59 | ~19.69 | |
| Subtotal dense per-episode (180 steps) | | ≤ 106 (state A only) | up to 3540 (full state-B run) | |
| Sparse bonus 1 | success_bonus (w=200) | one-shot | one-shot | +200 first step cube_0 stacked + EE retreated + no contact |
| Sparse penalty | stack_broke_penalty (w=-200) | one-shot | one-shot | −200 first step previously-stacked cube_0 falls off |
| Sparse bonus 2 | tower_bonus (w=2000) | one-shot | one-shot | +2000 first step full tower assembled |

Full-task ceiling per episode ≈ dense state-B (~3540) + tower_bonus (2000) + success_bonus (200) − any stack_broke (~−200 worst case) = ~5540. Tower bonus dominates the sparse band by 10×; success_bonus is a partial credit; stack_broke pays a one-time tax for breaking your own stack. Policy is incentivised to (a) stack cube_0 cleanly, (b) move EE away, (c) build the upper tower with cube_2 without disturbing the lower pair.

### Composer

**sum** — confirmed by inspecting `isaaclab.managers.reward_manager.RewardManager.compute` (line 153–155):

```python
value = term_cfg.func(self._env, **term_cfg.params) * term_cfg.weight
self._reward_buf += value
```

The env_cfg `__post_init__` does NOT override `RewardManager`; default sum composer applies. Also `RewardsCfg.__post_init__` is not defined.

### RewardsCfg verbatim

```python
@configclass
class RewardsCfg:
    """Grasping-cube reward (§6 edit_mode_015): unified reach/lift/align on the
    grasping cube (mux mirrors §5 obs) + retained success_bonus and
    stack_broke_penalty + regularizers.

    Stateless per-step grasping-cube mux (same predicate as §5 obs):
        not-yet-stacked (cube_0 NOT on cube_1):
            grasping_cube   <- cube_0
            target          <- cube_1.xyz + [0, 0, CUBE_SIZE]
        stacked (cube_0 ON cube_1):
            grasping_cube   <- cube_2
            target          <- cube_0.xyz + [0, 0, CUBE_SIZE]

    Term order (7 terms, ALL ACTIVE — these are the live weights):
        reach                  — 1 - tanh(||ee - gc|| / 0.1)             w=0.02
        lift                   — gc.z > 0.04                             w=0.1
        align                  — lifted * (1 - tanh(||gc - target||/0.08)) w=0.32
        success_bonus          — cube_0 stacked latch (once-per-episode)  w=200
        stack_broke_penalty    — cube_0 stack broken latch                w=-200
        tower_bonus            — full 3-tier tower latch                  w=2000
        linear_lift_grasping_cube — contact-gated linear lift ramp        w=0.15

    There are NO action_rate / joint_vel regularizer terms, and no curriculum
    (`CurriculumCfg` is empty — LiftCube's 1000x regularizer ramp suppressed
    the cube-release motion needed to stack).

    Composer: sum.
    """

    # Single-band reach: one term, std=0.1, applied in BOTH states via the
    # grasping-cube mux (cube_0 in state A / cube_2 in state B). State B is
    # magnitude-scaled (30*base + 1.0), NOT sharpness-scaled — the std is
    # identical in both states. Matches Genesis (REACH_STD=0.1) and ManiSkill
    # (std_reach=0.1), neither of which varies std by state either.
    reach = RewTerm(
        func=mdp.grasping_cube_ee_distance,
        params={"std": 0.1},
        weight=0.02,
    )
    lift = RewTerm(
        func=mdp.grasping_cube_is_lifted,
        params={"minimal_height": 0.04},
        weight=0.1,
    )
    align = RewTerm(
        func=mdp.grasping_cube_goal_distance,
        # state A (cube_0 grasping): minimal_height = 0.04 (cube must be lifted
        # off the table to start aligning). state B (cube_2 grasping):
        # minimal_height_b = target_z = cube_1.z + 2·CUBE_SIZE = 0.1075 (cube_2
        # must be lifted ABOVE the existing two-cube stack before align fires)
        # — pairs with `linear_lift_grasping_cube` to enforce "lift first, align
        # second" and avoid dragging cube_2 horizontally through the stack.
        params={"std": 0.08, "minimal_height": 0.04, "minimal_height_b": 0.0875},
        weight=0.32,
    )

    success_bonus = RewTerm(
        func=mdp.cube_0_stacked_bonus_once_per_episode,
        params={"xy_threshold": 0.02, "z_threshold": 0.01},
        weight=200.0,
    )

    stack_broke_penalty = RewTerm(
        func=mdp.cube_0_stack_broken_penalty_once_per_episode,
        params={"xy_threshold": 0.02, "z_threshold": 0.01},
        weight=-200.0,
    )

    tower_bonus = RewTerm(
        func=mdp.three_tier_tower_bonus_once_per_episode,
        params={"xy_threshold": 0.02, "z_threshold": 0.01},
        weight=2000.0,
    )

    # Dense linear lift reward for the grasping cube, ACTIVE IN BOTH STAGES.
    #   state A (grasping cube_0): linear ramp `init_z=0.0215` → `target_z_a=0.06`
    #     (denominator 0.06 − 0.0215 = 0.0385), gated on BOTH fingers in
    #     contact with cube_0. Term weight 0.15 → state-A peak = 0.15.
    #   state B (grasping cube_2): linear ramp init_z → `target_z_b=0.1075`
    #     (above the cube_0/cube_1 stack), gated on BOTH fingers in contact
    #     with cube_2. Composed as `10·base_b·gate_b + 1.0` to match the
    #     other §6 grasping-cube terms; state-B peak ≈ 0.15·11 = 1.65.
    # Paired with `align.minimal_height_b=0.0875` so horizontal align reward
    # only fires after cube_2 has cleared the existing stack — discourages
    # dragging cube_2 sideways through cube_0/cube_1.
    linear_lift_grasping_cube = RewTerm(
        func=mdp.linear_lift_grasping_cube,
        params={
            "init_z": 0.0215,
            "target_z_a": 0.06,
            "target_z_b": 0.1075,
            "contact_force_threshold": 1e-3,
        },
        weight=0.15,
    )

    # Per-step bonus when the grasping cube hovers over the stack target
    # (xy < 2 cm, |z| < 3 cm) AND the policy outputs an "open gripper"
    # action this step. Modest weight: the success_bonus (200) and
    # tower_bonus (2000) still dominate, but this nudges the policy to
    # release at the right moment instead of squeezing forever.
```

### Full `stack_cube/mdp/rewards.py` source

Split into three labeled blocks for readability — module preamble + predicate helper + dense terms; latch infrastructure + private helpers; sparse / once-per-episode terms.

#### Block A — module preamble, predicate, dense reward terms

```python
"""Reward functions for the stack_cube task — grasping-cube mux (edit_mode_015).

Dense shaping + sparse bonuses, all wired through the §5 obs `predicate`
`_cube_0_on_cube_1_predicate` so the reward and observation switch the
"currently-relevant cube" on the same step:

  Stateless per-step grasping-cube mux:
      not-yet-stacked (cube_0 NOT on cube_1):
          grasping_cube   <- cube_0
          target          <- cube_1.xyz + [0, 0, CUBE_SIZE]
      stacked (cube_0 ON cube_1):
          grasping_cube   <- cube_2
          target          <- cube_0.xyz + [0, 0, CUBE_SIZE]

Dense shaping (mux on grasping cube):
    grasping_cube_ee_distance     — 1 - tanh(||ee_w - grasping_cube||/std)
    grasping_cube_is_lifted       — 1.0 if grasping_cube.z_w > min_h else 0.0
    grasping_cube_goal_distance   — lifted * (1 - tanh(d_to_target/std))
    linear_lift_grasping_cube     — contact-gated linear lift ramp

Sparse / once-per-episode (cube_0 stacking + full tower):
    cube_0_stacked_bonus_once_per_episode      — +1 first step cube_0 latched
    cube_0_stack_broken_penalty_once_per_episode — −1 first step the latched
                                                     stack breaks again
    three_tier_tower_bonus_once_per_episode    — +1 first step BOTH pairs hold

EE pose is read from the FrameTransformer scene entity `ee_frame` (LiftCube
convention). The Franka per-robot cfg installs this sensor pointing at
`panda_hand` with an offset of [0, 0, 0.1034] (fingertip).
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformer

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


# Cube edge length — the expected z-gap between cube_0 (top) and cube_1 (bottom)
# when stacked. Mirrors the constant in `mdp/terminations.py`.
CUBE_SIZE = 0.043


# ---------------------------------------------------------------------------
# edit_mode_014 — grasping-cube predicate (private clone of
# `mdp.observations._cube_0_on_cube_1_predicate`). Kept here to avoid the
# cross-module dependency `rewards.py -> observations.py`. Identical thresholds
# (xy<0.02 AND |Δz - CUBE_SIZE|<0.01). Used by the three grasping-cube reward
# terms so the reward and observation mux flip on the same condition.
# ---------------------------------------------------------------------------


def _cube_0_on_cube_1_predicate(
    env: "ManagerBasedRLEnv",
    xy_threshold: float = 0.02,
    z_threshold: float = 0.01,
) -> torch.Tensor:
    """True per env when cube_0 is geometrically stacked on cube_1.

    Same thresholds as `mdp.observations._cube_0_on_cube_1_predicate` and
    `mdp.terminations.cube_0_stacked_on_cube_1`. Mirrors the §5 obs mux so the
    reward and observation switch at the same instant.
    """
    cube_0: RigidObject = env.scene["cube_0"]
    cube_1: RigidObject = env.scene["cube_1"]
    top = cube_0.data.root_pos_w[:, :3]
    bot = cube_1.data.root_pos_w[:, :3]
    xy_dist = torch.norm(top[:, :2] - bot[:, :2], dim=-1)
    z_gap = top[:, 2] - bot[:, 2]
    return (xy_dist < xy_threshold) & (torch.abs(z_gap - CUBE_SIZE) < z_threshold)


def grasping_cube_ee_distance(
    env: "ManagerBasedRLEnv",
    std: float = 0.1,
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """`1 - tanh(||grasping_cube - ee|| / std)`.

    ONE std for BOTH states — the attractor has identical sharpness in state A
    and state B. The states differ only in MAGNITUDE (state B is
    `30*base + 1.0`), never in `std`. Genesis (`REACH_STD=0.1`) and ManiSkill
    (`std_reach=0.1`) do the same; do not reintroduce a per-state std.

    grasping_cube = cube_0 when `_cube_0_on_cube_1_predicate` False, cube_2 when True.
    """
    cube_0 = env.scene["cube_0"]
    cube_2 = env.scene["cube_2"]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    pos_0 = cube_0.data.root_pos_w[:, :3]
    pos_2 = cube_2.data.root_pos_w[:, :3]
    on_stack = _cube_0_on_cube_1_predicate(env)            # bool (N,)
    grasping_pos = torch.where(on_stack.unsqueeze(-1), pos_2, pos_0)
    ee_w = ee_frame.data.target_pos_w[..., 0, :]
    d = torch.norm(grasping_pos - ee_w, dim=1)
    base = 1.0 - torch.tanh(d / std)
    # State B: scale base by 30x and add the +1.0 compensation; state A unchanged.
    return torch.where(on_stack, 30.0 * base + 1.0, base)


def grasping_cube_is_lifted(
    env: "ManagerBasedRLEnv",
    minimal_height: float = 0.04,
) -> torch.Tensor:
    """`1.0 if grasping_cube.z_w > minimal_height else 0.0`.

    Same grasping-cube mux as `grasping_cube_ee_distance`.
    """
    cube_0 = env.scene["cube_0"]
    cube_2 = env.scene["cube_2"]
    on_stack = _cube_0_on_cube_1_predicate(env)
    z_0 = cube_0.data.root_pos_w[:, 2]
    z_2 = cube_2.data.root_pos_w[:, 2]
    z = torch.where(on_stack, z_2, z_0)
    lifted = torch.where(z > minimal_height, 1.0, 0.0)
    # State B: scale base by 10× and add the +1.0 compensation; state A unchanged.
    return torch.where(on_stack, 10.0 * lifted + 1.0, lifted)


def grasping_cube_goal_distance(
    env: "ManagerBasedRLEnv",
    std: float = 0.3,
    minimal_height: float = 0.04,
    minimal_height_b: float | None = None,
) -> torch.Tensor:
    """`(grasping_cube lifted) * (1 - tanh(||grasping_cube - target|| / std))`.

    target = cube_1.xyz + [0,0,CUBE_SIZE] when predicate False,
             cube_0.xyz + [0,0,CUBE_SIZE] when True.

    Per-state lift gate:
      state A (predicate False): `minimal_height` applies (default 0.04)
      state B (predicate True ): `minimal_height_b` applies — set this to the
        target z (= cube_0.z + CUBE_SIZE ≈ 0.1075 for a cube on cube_1 on
        the table) to keep `align` from firing until cube_2 is lifted above
        the existing stack. Default = `minimal_height` (state A behaviour).
    """
    cube_0 = env.scene["cube_0"]
    cube_1 = env.scene["cube_1"]
    cube_2 = env.scene["cube_2"]
    on_stack = _cube_0_on_cube_1_predicate(env)            # bool (N,)
    pos_0 = cube_0.data.root_pos_w[:, :3]
    pos_1 = cube_1.data.root_pos_w[:, :3]
    pos_2 = cube_2.data.root_pos_w[:, :3]
    grasping_pos = torch.where(on_stack.unsqueeze(-1), pos_2, pos_0)
    base_pos     = torch.where(on_stack.unsqueeze(-1), pos_0, pos_1)
    target_pos   = base_pos.clone()
    target_pos[:, 2] = target_pos[:, 2] + CUBE_SIZE
    d = torch.norm(grasping_pos - target_pos, dim=1)
    h_b = minimal_height if minimal_height_b is None else float(minimal_height_b)
    effective_min_h = torch.where(
        on_stack,
        torch.full_like(grasping_pos[:, 2], h_b),
        torch.full_like(grasping_pos[:, 2], minimal_height),
    )
    lifted = (grasping_pos[:, 2] > effective_min_h).float()
    base = lifted * (1.0 - torch.tanh(d / std))
    # State B: scale base by 50× and add the +1.0 compensation; state A unchanged.
    return torch.where(on_stack, 50.0 * base + 1.0, base)


def linear_lift_grasping_cube(
    env: "ManagerBasedRLEnv",
    init_z: float = 0.0215,
    target_z_a: float = 0.06,
    target_z_b: float = 0.1075,
    contact_force_threshold: float = 1e-3,
) -> torch.Tensor:
    """Dense linear lift reward for the grasping cube, contact-sensor-gated.

    Per-state grasping-cube mux (same predicate as §5 obs / other §6
    grasping-cube terms):
      state A (`_cube_0_on_cube_1_predicate` False) — grasping cube = cube_0;
        base_a = clamp((cube_0.z - init_z) / (target_z_a - init_z), 0, 1)
                 with default ramp init_z=0.0215 → target_z_a=0.06
                 (denominator = 0.06 − 0.0215 = 0.0385).
        Contact gate: both fingers in contact with cube_0 (filter idx 0).
      state B (`_cube_0_on_cube_1_predicate` True)  — grasping cube = cube_2;
        base_b = clamp((cube_2.z - init_z) / (target_z_b - init_z), 0, 1)
                 (target_z_b=0.1075 = cube_1.z + 2·CUBE_SIZE — above the
                 existing two-cube stack).
        Contact gate: both fingers in contact with cube_2 (filter idx 2).

    Contact gating mirrors `insert_drawer.mdp.rewards.lift_distance`: the ramp
    fires only when BOTH fingertip contact sensors report force > threshold
    against the relevant cube, so the policy can't earn lift reward by
    knocking the cube up with the body of the hand or by single-finger flicks.

    Composition (matches the +1.0 state-B offset used by other §6
    grasping-cube terms — `grasping_cube_ee_distance`, etc. — so transitioning
    into state B never lowers reward):
        torch.where(on_stack, 10 * base_b * gate_b + 1.0, base_a * gate_a)
    """
    on_stack = _cube_0_on_cube_1_predicate(env)
    cube_0 = env.scene["cube_0"]
    cube_2 = env.scene["cube_2"]
    z_0 = cube_0.data.root_pos_w[:, 2]
    z_2 = cube_2.data.root_pos_w[:, 2]
    base_a = ((z_0 - init_z) / max(target_z_a - init_z, 1e-6)).clamp(0.0, 1.0)
    base_b = ((z_2 - init_z) / max(target_z_b - init_z, 1e-6)).clamp(0.0, 1.0)

    left = env.scene["finger_left_contact"]
    right = env.scene["finger_right_contact"]
    # filter_prim_paths_expr order: [Cube_0 (idx 0), Cube_1 (idx 1), Cube_2 (idx 2)]
    left_f0 = torch.norm(left.data.force_matrix_w[:, 0, 0, :], dim=-1)
    right_f0 = torch.norm(right.data.force_matrix_w[:, 0, 0, :], dim=-1)
    gate_a = ((left_f0 > contact_force_threshold) & (right_f0 > contact_force_threshold)).float()
    left_f2 = torch.norm(left.data.force_matrix_w[:, 0, 2, :], dim=-1)
    right_f2 = torch.norm(right.data.force_matrix_w[:, 0, 2, :], dim=-1)
    gate_b = ((left_f2 > contact_force_threshold) & (right_f2 > contact_force_threshold)).float()

    return torch.where(on_stack, 10.0 * base_b * gate_b + 1.0, base_a * gate_a)
```

#### Block B — latch infrastructure + private contact helpers

```python
# iter 33 — module-level per-env latch buffers (keyed by id(env), key_str).
# Used by `cube_1_was_stacked_latched_indicator` to gate cube_0 shaping on
# 'cube_1 has been stacked on cube_2 AT LEAST ONCE in this episode'. Once the
# latch is set it stays True for the rest of the episode (giving the policy
# ~150-200 frames of cube_0 shaping signal per stage-1-success episode vs the
# ~21 frames of the strict instantaneous gate). The latch resets on episode
# reset via the episode_length_buf <= 1 check. Single train-job safety:
# /reward-tune runs one train at a time; reassigns happen each call.
_LATCH_BUFFERS: dict[tuple, torch.Tensor] = {}


def _get_latch_buffer(env, key: str) -> torch.Tensor:
    """Get or lazily-create a per-env boolean latch tensor of shape (num_envs,).

    Stored on the module-level `_LATCH_BUFFERS` dict keyed by `(id(env), key)`.
    """
    full_key = (id(env), key)
    if full_key not in _LATCH_BUFFERS:
        _LATCH_BUFFERS[full_key] = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    return _LATCH_BUFFERS[full_key]


def _no_contact_between_cube_0_and_gripper_or_ee(
    env: "ManagerBasedRLEnv",
    eps: float = 1e-3,
) -> torch.Tensor:
    """True per env if cube_0 has NO contact force vs either fingertip OR the hand body.

    Reads `force_matrix_w[:, 0, 0, :]` (filter index 0 = Cube_0) on
    finger_left_contact, finger_right_contact, and hand_contact sensors. All
    three must have force magnitude ≤ eps for this to return True.
    """
    left = env.scene["finger_left_contact"]
    right = env.scene["finger_right_contact"]
    hand = env.scene["hand_contact"]
    left_f0 = torch.norm(left.data.force_matrix_w[:, 0, 0, :], dim=-1)
    right_f0 = torch.norm(right.data.force_matrix_w[:, 0, 0, :], dim=-1)
    hand_f0 = torch.norm(hand.data.force_matrix_w[:, 0, 0, :], dim=-1)
    in_contact = (left_f0 > eps) | (right_f0 > eps) | (hand_f0 > eps)
    return ~in_contact
```

#### Block C — sparse / once-per-episode bonuses + release placeholder

```python
def cube_0_stacked_bonus_once_per_episode(
    env: "ManagerBasedRLEnv",
    xy_threshold: float = 0.02,
    z_threshold: float = 0.01,
    cube_0_cfg: SceneEntityCfg = SceneEntityCfg("cube_0"),
    cube_1_cfg: SceneEntityCfg = SceneEntityCfg("cube_1"),
) -> torch.Tensor:
    """+1.0 the FIRST step cube_0 is stacked on cube_1 with no gripper contact, else 0.0.

    Success criteria (BOTH must hold):
      - geometric: |cube_0.xy − cube_1.xy| < xy_threshold AND |Δz − CUBE_SIZE| < z_threshold
      - no contact: neither fingertip nor the panda_hand body has any contact force on cube_0

    NOTE — there is deliberately NO explicit EE-distance criterion. An earlier
    revision computed a `far_enough` (EE ≥ 4 cm from cube_0) term and then
    discarded it, while the docstring claimed three criteria; the dead code and
    the wrong docstring have both been removed. `no_contact` is the only
    "gripper has let go" signal. The ManiSkill port DOES additionally AND in
    `far_enough`, so its success predicate is strictly stricter than this one —
    success rates are not directly comparable across the two.

    Per-env latch reset when `env.episode_length_buf <= 1` and set the first
    time the combined criteria hold. Once latched the bonus stops firing —
    encourages the policy to MOVE ON to stacking cube_2 on cube_0.
    """
    latch = _get_latch_buffer(env, "cube_0_stacked_once")
    just_reset = env.episode_length_buf <= 1
    latch = torch.where(just_reset, torch.zeros_like(latch), latch)

    cube_0: RigidObject = env.scene[cube_0_cfg.name]
    cube_1: RigidObject = env.scene[cube_1_cfg.name]
    pos_0 = cube_0.data.root_pos_w[:, :3]
    pos_1 = cube_1.data.root_pos_w[:, :3]
    xy_dist = torch.norm(pos_0[:, :2] - pos_1[:, :2], dim=-1)
    z_gap = pos_0[:, 2] - pos_1[:, 2]
    geometric = (xy_dist < xy_threshold) & (torch.abs(z_gap - CUBE_SIZE) < z_threshold)
    no_contact = _no_contact_between_cube_0_and_gripper_or_ee(env)
    now_stacked = geometric & no_contact

    fire = now_stacked & (~latch)
    latch = latch | now_stacked
    _LATCH_BUFFERS[(id(env), "cube_0_stacked_once")] = latch
    return fire.float()


def three_tier_tower_bonus_once_per_episode(
    env: "ManagerBasedRLEnv",
    xy_threshold: float = 0.02,
    z_threshold: float = 0.01,
) -> torch.Tensor:
    """+1.0 the FIRST step the full 3-tier tower is assembled, else 0.0.

    Tower pairing now mirrors the 2-cube success contract
    (`cube_0_stacked_bonus_once_per_episode`):
        cube_0 on cube_1:  geometric (|Δxy|<xy_thr AND |Δz−CUBE_SIZE|<z_thr)
                           AND no contact between cube_0 and gripper/ee
        cube_2 on cube_0:  geometric (same thresholds)
                           AND no contact between cube_2 and gripper/ee

    Per-env latch resets at `env.episode_length_buf <= 1` and locks once fired
    so the bonus counts at most once per episode.
    """
    from .terminations import _no_contact_between_cube_and_gripper_or_ee
    latch = _get_latch_buffer(env, "three_tier_tower_once")
    just_reset = env.episode_length_buf <= 1
    latch = torch.where(just_reset, torch.zeros_like(latch), latch)

    cube_0 = env.scene["cube_0"]
    cube_1 = env.scene["cube_1"]
    cube_2 = env.scene["cube_2"]
    pos_0 = cube_0.data.root_pos_w[:, :3]
    pos_1 = cube_1.data.root_pos_w[:, :3]
    pos_2 = cube_2.data.root_pos_w[:, :3]
    # Upper pair: cube_0 on cube_1 — geometric AND no contact between cube_0 and gripper/ee.
    xy_01 = torch.norm(pos_0[:, :2] - pos_1[:, :2], dim=-1)
    z_01 = pos_0[:, 2] - pos_1[:, 2]
    upper_geom = (xy_01 < xy_threshold) & (torch.abs(z_01 - CUBE_SIZE) < z_threshold)
    upper = upper_geom & _no_contact_between_cube_and_gripper_or_ee(env, cube_idx=0)
    # Top pair: cube_2 on cube_0 — geometric AND no contact between cube_2 and gripper/ee.
    xy_20 = torch.norm(pos_2[:, :2] - pos_0[:, :2], dim=-1)
    z_20 = pos_2[:, 2] - pos_0[:, 2]
    top_geom = (xy_20 < xy_threshold) & (torch.abs(z_20 - CUBE_SIZE) < z_threshold)
    top = top_geom & _no_contact_between_cube_and_gripper_or_ee(env, cube_idx=2)
    tower_built = upper & top

    fire = tower_built & (~latch)
    latch = latch | tower_built
    _LATCH_BUFFERS[(id(env), "three_tier_tower_once")] = latch
    return fire.float()


def cube_0_stack_broken_penalty_once_per_episode(
    env: "ManagerBasedRLEnv",
    xy_threshold: float = 0.02,
    z_threshold: float = 0.01,
    cube_0_cfg: SceneEntityCfg = SceneEntityCfg("cube_0"),
    cube_1_cfg: SceneEntityCfg = SceneEntityCfg("cube_1"),
) -> torch.Tensor:
    """+1.0 ONCE per episode when cube_0 had been successfully stacked (success_bonus
    latch is set) AND the stack has subsequently broken (cube_0 no longer
    geometrically on cube_1). Otherwise 0.0.

    Intended to be weighted negatively in RewardsCfg (e.g. weight=-100) — the
    policy is penalized once when it disturbs a previously successful stack.

    Uses two latches:
      - `cube_0_stacked_once` (read-only here, written by `cube_0_stacked_bonus_once_per_episode`):
        records "this episode had a successful cube_0 stack at some point".
      - `stack_broke_penalty_fired` (managed here): records "this episode
        already paid the broken-stack penalty", preventing repeated firing.

    Declaration order in `RewardsCfg` matters — place this AFTER `success_bonus`
    so the `cube_0_stacked_once` latch reads the fresh value.
    """
    stacked_once = _get_latch_buffer(env, "cube_0_stacked_once")

    penalty_fired = _get_latch_buffer(env, "stack_broke_penalty_fired")
    just_reset = env.episode_length_buf <= 1
    penalty_fired = torch.where(just_reset, torch.zeros_like(penalty_fired), penalty_fired)

    cube_0: RigidObject = env.scene[cube_0_cfg.name]
    cube_1: RigidObject = env.scene[cube_1_cfg.name]
    pos_0 = cube_0.data.root_pos_w[:, :3]
    pos_1 = cube_1.data.root_pos_w[:, :3]
    xy_dist = torch.norm(pos_0[:, :2] - pos_1[:, :2], dim=-1)
    z_gap = pos_0[:, 2] - pos_1[:, 2]
    currently_stacked = (xy_dist < xy_threshold) & (torch.abs(z_gap - CUBE_SIZE) < z_threshold)

    fire = stacked_once & (~currently_stacked) & (~penalty_fired)

    penalty_fired = penalty_fired | fire
    _LATCH_BUFFERS[(id(env), "stack_broke_penalty_fired")] = penalty_fired
    return fire.float()

```

> **Note — everything in this §6 is live.** A previous revision carried three dead artifacts,
> all now removed: an inactive `release_bonus_in_drop_zone` term (`weight=0.0`, short-circuited
> by `RewardManager`), an unused `_gripper_far_from_cube_0` helper whose result was computed and
> then discarded, and an unused `std_state_b` parameter on `grasping_cube_ee_distance`.
> Reproducing this spec should yield **7 reward terms, all active**, with no unreferenced
> helpers or parameters.
>
> Behavioural consequence worth knowing: because the discarded `far_enough` was never AND-ed
> in, `success_bonus` latches on `geometric & no_contact` alone — the gripper may still be
> within 4 cm. The ManiSkill port DOES AND it in, so its predicate is strictly stricter and
> success rates are not directly comparable across the two.

### Smoke

```bash
.venv/bin/python harbor/create-task/isaaclab-franka-stackcube/smokes/smoke_s6.py
# exit 0 (asserts: active reward terms registered, per-step reward finite + non-constant,
#         composer = sum (passthrough), latch state resets on env.reset())
```

---

## §7 DR

`<no DR>` — `EventCfg` contains only the four reset terms above (`reset_robot_joints`, `reset_cube_0`, `reset_cube_1`, `reset_cube_2`). No `EventTerm` has `mode="startup"` or `mode="interval"`. The `mdp/events.py` module is intentionally empty (only a docstring) — its purpose is to keep `from .events import *` working. The `dr-generator` was not run on this task; the env_cfg `EventCfg` docstring notes "DR terms are added by `dr-generator` in §7" but none have been added.

To add later: `/harbor:create-task name=IsaacLab-Franka-StackCube description="add startup mass + friction + cube pose DR" sections=7`.

---

## Reproduction

```bash
# Same source repo:
/harbor:create-task name=IsaacLab-Franka-StackCube-v2 from=harbor/create-task/isaaclab-franka-stackcube-implementation.md

# Different repo: pass asset overrides if the local harbor/assets tree differs
/harbor:create-task name=FrankaStack3Cubes from=isaaclab-franka-stackcube-implementation.md \
  assets=path/to/dest/fr3.usd,path/to/dest/table.usd
```

> probe-task: wrote `harbor/create-task/isaaclab-franka-stackcube-implementation.md` (sections §1..§7, 7 active reward funcs, 5 obs terms / 19-D obs, 3-D position EMA action + binary gripper / 4-D action).
> Reproduce via: `/harbor:create-task name=<new_task_id> from=<output>`.
