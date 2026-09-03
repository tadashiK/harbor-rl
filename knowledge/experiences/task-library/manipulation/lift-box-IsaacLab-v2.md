# Isaac-Lift-Box-Dual-Franka-v0 — Implementation Spec

> Portable per-task design-choice spec emitted by `/harbor:probe-task`. Feed back via
> `/harbor:task-create name=<new_task_id> from=<this file>` to clone the task identically.

- robot: Two Franka FR3 arms + Franka hands (dual-arm cooperative)
- simulator: IsaacLab (Isaac Sim, manager-based)
- objects: eurobox, lab table
- bimanual: true
- summary: Two arms cooperatively grasp a eurobox and lift it off the table.

> **AppLauncher caveat**: bare-shell `gym.make` does NOT work in this repo — `pxr` is only
> importable through `isaaclab.app.AppLauncher`. Every build/smoke must first construct
> `AppLauncher(args).app` (headless) before importing `gymnasium` / `isaaclab_tasks`. The
> canonical build proof is the passing smoke
> `harbor/create-task/isaac-lift-box-dual-franka-v0/smokes/smoke_s1.py`
> run with `.venv/bin/python -u` (exit 0).

## Overview

Two FR3 + Franka-hand robots, seated at world y = ±0.49, cooperate to lift a eurobox
(40×30×22 cm, 0.5 kg) off a lab table to a target lift height of 0.25 m centered over the
table origin, with low residual velocity and a genuine dual-finger grasp by BOTH robots.
Each robot is driven by a 3-D EMA-smoothed cumulative EE-position-delta IK action (EE
orientation locked) plus a binary gripper. Scene + sim timing + actuation pattern mirror
`manipulation/insert_drawer/` (which is source-stripped in this clone; the FR3 cfg + EMA
action classes are vendored locally). The goal is hard-coded in `mdp/terminations.py` — there
is no `CommandsCfg`.

---

## Registration

`config/franka/__init__.py` (verbatim):

```python
import gymnasium as gym

gym.register(
    id="Isaac-Lift-Box-Dual-Franka-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:FrankaLiftBoxEnvCfg",
    },
)

gym.register(
    id="Isaac-Lift-Box-Dual-Franka-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:FrankaLiftBoxEnvCfg_PLAY",
    },
)
```

Package `__init__.py` files:

```python
# lift_box/__init__.py
from . import mdp  # noqa: F401 — re-exports task-local helpers
```
```python
# config/__init__.py
"""Per-robot configurations for the dual-arm lift_box environment."""
```
```python
# mdp/__init__.py
from isaaclab.envs.mdp import *  # noqa: F401, F403

from .actions import *  # noqa: F401, F403
from .actions_cfg import *  # noqa: F401, F403
from .observations import *  # noqa: F401, F403
from .rewards import *  # noqa: F401, F403
from .terminations import *  # noqa: F401, F403
```

`reset_joints_by_scale`, `reset_root_state_uniform`, `time_out`, `joint_pos`, `last_action`,
`BinaryJointPositionActionCfg` are inherited from `isaaclab.envs.mdp` via the star import (not
defined locally). All task-local funcs referenced in §3–§7 (`ee_pose_in_robot_root_frame`,
`box_position_in_world`, `box_quat_in_world`, `ee_to_grasp_distance`, `grasp_contact`,
`lift_height`, `box_xy_align`, `success_bonus`, `_both_fingers_in_contact`, `lift_box_success`)
resolve in the mdp tree — verified by grep.

---

## §1 — Register / Scene

### Description
A dual-arm cooperative-lift scene: two FR3 + Franka-hand articulations at world y = ±0.49
facing a eurobox centered on a lab table. Scene also carries per-robot EE frame transformers,
per-robot box grasp-point markers (debug-viz only — no rl-side reader), per-finger contact
sensors filtered against the box, the table, a ground plane, and a dome light. The robot
articulations, box, frame transformers, and action stack are filled by the per-robot subclass
`FrankaLiftBoxEnvCfg.__post_init__`; the abstract `LiftBoxSceneCfg` leaves them `MISSING`.

### Decisions resolved
- **Robot**: `FR3_FRANKA_HAND_CFG`, vendored verbatim from `insert_drawer` (source-stripped in
  this clone). USD `harbor/assets/fr3/fr3_franka_hand.usd`. Implicit actuators: `fr3_shoulder`
  (joints 1–4, effort 87, stiff 400, damp 80), `fr3_forearm` (joints 5–7, effort 12, stiff 400,
  damp 80), `fr3_hand` (`fr3_finger_joint.*`, effort 200, stiff 2e3, damp 1e2).
  `disable_gravity=True` on the robot (arms float; gravity disabled per insert_drawer pattern),
  `enabled_self_collisions=True`, `activate_contact_sensors=True`.
- **Two-robot placement**: both at `pos=(-0.274, ±0.49, 0.01)`; robot_0 at +y, robot_1 at −y.
  Distinct per-robot init joint poses (`FRANKA_INIT_JOINT_POS_0/1`) orient each arm toward the
  box from its side (robot_1 has `fr3_joint1=0.785`, `fr3_joint7=-1.57` mirroring robot_0).
- **Box**: eurobox USD `harbor/assets/eurobox/eurobox.usd`, mass 0.5 kg, spawned at
  `pos=[0,0,0.11025]` with a 90° Z-rotation `rot=[0.7071068,0,0,0.7071068]` so the box's long
  axis runs along world Y (between the two robots). Solver iters (16,1), gravity enabled.
  `BOX_INIT_Z = 0.11025` (= half-z-extent of the recentered eurobox; reused in reward/term).
- **EE frames** (per robot): `FrameTransformer` from `Robot_i/fr3_link0` to `Robot_i/fr3_hand`
  with `OffsetCfg(pos=[0,0,0.2])` → fingertip TCP; `debug_vis=True`, marker scale 0.1.
- **Grasp frames** (per robot, debug-viz ONLY): `FrameTransformer` from `Box` to `Box` with
  offset `[+0.20,0,0.11025]` (grasp_frame_0, robot_0 side) / `[-0.20,0,0.11025]` (grasp_frame_1)
  — top-center of each short y-end face. These markers ARE read by the reward (`ee_to_grasp_distance`).
- **Contact sensors**: 4 `ContactSensorCfg` on each robot's `fr3_leftfinger`/`fr3_rightfinger`,
  `filter_prim_paths_expr=["{ENV_REGEX_NS}/Box"]`, `history_length=1`, `update_period=0.0`.
- **Table**: `lab_table_instanceable_colored_rotated.usd` at z=0; ground plane at z=−0.82;
  dome light intensity 3000.
- **num_envs default 4096**, `env_spacing=2.5`, `replicate_physics=False`. PLAY subclass:
  num_envs=50, corruption off.

### Code

`lift_box_env_cfg.py` — `_TABLE_USD_PATH` resolution + `LiftBoxSceneCfg` (verbatim):

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

`config/franka/joint_pos_env_cfg.py` — asset paths, FR3 cfg, init poses, and the scene-filling
portion of `__post_init__` (robots / box / ee_frame / grasp_frame; verbatim):

```python
_HARBOR_ASSETS = Path(__file__).resolve().parents[8] / "harbor" / "assets"
_FR3_USD_PATH = str(_HARBOR_ASSETS / "fr3" / "fr3_franka_hand.usd")
_EUROBOX_USD_PATH = str(_HARBOR_ASSETS / "eurobox" / "eurobox.usd")

BOX_INIT_Z: float = 0.11025
BOX_MASS: float = 0.5

# Vendored verbatim from insert_drawer's FR3_FRANKA_HAND_CFG.
FR3_FRANKA_HAND_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=_FR3_USD_PATH,
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=True, max_depenetration_velocity=5.0),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True, solver_position_iteration_count=8, solver_velocity_iteration_count=0,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(),
    actuators={
        "fr3_shoulder": ImplicitActuatorCfg(joint_names_expr=["fr3_joint[1-4]"], effort_limit_sim=87.0, stiffness=400.0, damping=80.0),
        "fr3_forearm": ImplicitActuatorCfg(joint_names_expr=["fr3_joint[5-7]"], effort_limit_sim=12.0, stiffness=400.0, damping=80.0),
        "fr3_hand": ImplicitActuatorCfg(joint_names_expr=["fr3_finger_joint.*"], effort_limit_sim=200.0, stiffness=2e3, damping=1e2),
    },
    soft_joint_pos_limit_factor=1.0,
)

FRANKA_INIT_JOINT_POS_0 = {
    "fr3_joint1": -0.785, "fr3_joint2": -0.785, "fr3_joint3": 0.0,
    "fr3_joint4": -2.655, "fr3_joint5": 0.0,    "fr3_joint6": 1.87,
    "fr3_joint7": 0.0,    "fr3_finger_joint.*": 0.04,
}
FRANKA_INIT_JOINT_POS_1 = {
    "fr3_joint1": 0.785,  "fr3_joint2": -0.785, "fr3_joint3": 0.0,
    "fr3_joint4": -2.655, "fr3_joint5": 0.0,    "fr3_joint6": 1.87,
    "fr3_joint7": -1.57,  "fr3_finger_joint.*": 0.04,
}


@configclass
class FrankaLiftBoxEnvCfg(LiftBoxEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        # Robots — two FR3 + Franka-hand at world y = ±0.49.
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

        # Per-robot ee_frame (fingertip TCP).
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
        ee_marker_cfg_1 = FRAME_MARKER_CFG.copy()
        ee_marker_cfg_1.markers["frame"].scale = (0.1, 0.1, 0.1)
        ee_marker_cfg_1.prim_path = "/Visuals/FrameTransformer1"
        self.scene.ee_frame_1 = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot_1/fr3_link0", debug_vis=True, visualizer_cfg=ee_marker_cfg_1,
            target_frames=[FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/Robot_1/fr3_hand", name="end_effector",
                offset=OffsetCfg(pos=[0.0, 0.0, 0.2]),
            )],
        )

        # Box grasp-point markers (top-center of each short y-end face).
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
        grasp_marker_cfg_1 = FRAME_MARKER_CFG.copy()
        grasp_marker_cfg_1.markers["frame"].scale = (0.08, 0.08, 0.08)
        grasp_marker_cfg_1.prim_path = "/Visuals/BoxGraspFrame1"
        self.scene.grasp_frame_1 = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Box", debug_vis=True, visualizer_cfg=grasp_marker_cfg_1,
            target_frames=[FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/Box", name="box_grasp_1",
                offset=OffsetCfg(pos=[-0.20, 0.0, 0.11025]),
            )],
        )
```

PLAY subclass (verbatim):

```python
@configclass
class FrankaLiftBoxEnvCfg_PLAY(FrankaLiftBoxEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
```

Env-cfg wiring + sim/physx knobs (verbatim, in `LiftBoxEnvCfg`):

```python
@configclass
class LiftBoxEnvCfg(ManagerBasedRLEnvCfg):
    """Abstract dual-arm lift_box env cfg (robot-agnostic)."""

    scene: LiftBoxSceneCfg = LiftBoxSceneCfg(num_envs=4096, env_spacing=2.5, replicate_physics=False)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventCfg = EventCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    commands = None

    def __post_init__(self):
        self.decimation = 6
        self.episode_length_s = 10.0
        self.sim.dt = 1 / 120
        self.sim.render_interval = self.decimation
        self.sim.physx.bounce_threshold_velocity = 0.01
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 1024 * 1024 * 4
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 64 * 1024
        self.sim.physx.friction_correlation_distance = 0.00625
```

Sim timing: `decimation=6`, `sim.dt=1/120` → `step_dt = 0.05 s`; `episode_length_s=10.0` →
`max_episode_length = 200` steps.

### Smoke
`harbor/create-task/isaac-lift-box-dual-franka-v0/smokes/smoke_s1.py` — env instantiation at
num_envs=2. Asserts action_dim==8, obs policy dim==33, max_episode_length>0.
Key output: `S1 OK: env instantiated with valid action/obs/episode-length info`.
**Re-run confirmed: exit 0.**

---

## §2 — Actions

### Description
Per-robot action stack: a 3-D EMA-smoothed cumulative EE-position-delta IK action
(`EMACumulativeDeltaPositionAction`, vendored from insert_drawer) + a binary gripper. The
policy emits `(dx,dy,dz)` per arm; the action term accumulates deltas over the episode, anchors
on the EE pose captured lazily on the first post-reset `process_actions`, LOCKS the EE
orientation at that anchor quat, clamps the absolute target into a per-axis workspace box,
EMA-smooths against the previously applied position, and forwards a 7-D `(target_pos, init_quat)`
to a DLS IK controller. Global action vector is 8-D: `[arm_0(3), gripper_0(1), arm_1(3), gripper_1(1)]`.

### Decisions resolved
- **Mode**: `ema_delta_ee_pose` (task-space, position-only; EE quat locked at post-reset value).
- **scale** `(0.01,0.01,0.01)`, **alpha** `0.5` (EMA), `command_type="pose"`,
  `use_relative_mode=False`, `ik_method="dls"`, `body_offset pos=(0,0,0.2)`.
- **Per-robot workspace box** (`pos_lower_limit`/`pos_upper_limit`): robot_0
  `[0.20,-0.65,0.005]`..`[0.55,-0.20,0.40]`; robot_1 `[0.20,0.20,0.005]`..`[0.55,0.65,0.40]`
  (each arm confined to its own y-half).
- **Gripper**: `BinaryJointPositionAction` on `fr3_finger.*`, open=0.04, close=0.0.
- `forbidden_xy_half` left None (cfg field exists for the no-go-square option; unused here).

### Code

`mdp/actions_cfg.py` (verbatim):

```python
from __future__ import annotations

from isaaclab.envs.mdp.actions import DifferentialInverseKinematicsActionCfg
from isaaclab.managers.action_manager import ActionTerm
from isaaclab.utils import configclass

from .actions import EMACumulativeDeltaPositionAction


@configclass
class EMACumulativeDeltaPositionActionCfg(DifferentialInverseKinematicsActionCfg):
    class_type: type[ActionTerm] = EMACumulativeDeltaPositionAction
    scale: tuple[float, float, float] = (0.02, 0.02, 0.02)
    alpha: float = 0.5
    pos_lower_limit: list[float] | None = None
    pos_upper_limit: list[float] | None = None
    forbidden_xy_half: float | None = None
```

`mdp/actions.py` — `EMACumulativeDeltaPositionAction` (verbatim):

```python
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

    def __init__(self, cfg: "actions_cfg.EMACumulativeDeltaPositionActionCfg", env: "ManagerBasedEnv"):
        if cfg.controller.use_relative_mode:
            raise ValueError(
                "EMACumulativeDeltaPositionAction handles relative deltas itself; "
                "set controller.use_relative_mode=False"
            )
        if cfg.controller.command_type != "pose":
            raise ValueError(f"requires command_type='pose'; got '{cfg.controller.command_type}'")
        super().__init__(cfg, env)
        self._raw_actions = torch.zeros(env.num_envs, 3, device=env.device)
        self._processed_actions = torch.zeros(env.num_envs, 7, device=env.device)
        self._scale = torch.zeros((env.num_envs, 3), device=env.device)
        self._scale[:] = torch.tensor(cfg.scale, device=env.device)
        if not 0.0 <= cfg.alpha <= 1.0:
            raise ValueError(f"alpha must be in [0, 1]. Got {cfg.alpha}.")
        self._alpha = cfg.alpha
        self.del_action = torch.zeros((env.num_envs, 3), device=env.device)
        self.init_ee_pos = torch.zeros((env.num_envs, 3), device=env.device)
        self.init_ee_quat = torch.zeros((env.num_envs, 4), device=env.device)
        self.init_ee_quat[:, 0] = 1.0
        self._prev_applied_pos = torch.zeros((env.num_envs, 3), device=env.device)
        self._needs_reanchor = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
        self.pos_lower_limit = (
            torch.tensor(cfg.pos_lower_limit, device=self.device) if cfg.pos_lower_limit is not None else None
        )
        self.pos_upper_limit = (
            torch.tensor(cfg.pos_upper_limit, device=self.device) if cfg.pos_upper_limit is not None else None
        )
        self.forbidden_xy_half: float | None = getattr(cfg, "forbidden_xy_half", None)

    @property
    def action_dim(self) -> int:
        return 3

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        super().reset(env_ids)
        if env_ids is None:
            self._needs_reanchor[:] = True
            self.del_action[:] = 0.0
        else:
            self._needs_reanchor[env_ids] = True
            self.del_action[env_ids] = 0.0

    def process_actions(self, actions: torch.Tensor) -> None:
        if self._needs_reanchor.any():
            ee_pos_curr, ee_quat_curr = self._compute_frame_pose()
            mask = self._needs_reanchor
            self.init_ee_pos[mask] = ee_pos_curr[mask]
            self.init_ee_quat[mask] = ee_quat_curr[mask]
            self._prev_applied_pos[mask] = ee_pos_curr[mask]
            self._needs_reanchor[:] = False
        actions = torch.clamp(actions, -1.0, 1.0)
        self._raw_actions[:] = actions
        scaled = actions * self._scale
        self.del_action += scaled
        if self.pos_lower_limit is not None and self.pos_upper_limit is not None:
            del_lower = self.pos_lower_limit - self.init_ee_pos
            del_upper = self.pos_upper_limit - self.init_ee_pos
            self.del_action = torch.clamp(self.del_action, del_lower, del_upper)
        abs_pos = self.init_ee_pos + self.del_action
        ema_pos = self._alpha * abs_pos + (1.0 - self._alpha) * self._prev_applied_pos
        if self.pos_lower_limit is not None and self.pos_upper_limit is not None:
            ema_pos = torch.clamp(ema_pos, self.pos_lower_limit, self.pos_upper_limit)
        if self.forbidden_xy_half is not None:
            h = self.forbidden_xy_half
            inside = (ema_pos[:, 0].abs() <= h) & (ema_pos[:, 1].abs() <= h)
            if inside.any():
                # Push the target out of the |x|<=h ∧ |y|<=h square to the nearest edge.
                dx = h - ema_pos[:, 0].abs()
                dy = h - ema_pos[:, 1].abs()
                push_x = dx <= dy
                sign_x = torch.sign(ema_pos[:, 0])
                sign_x = torch.where(sign_x == 0, torch.ones_like(sign_x), sign_x)
                sign_y = torch.sign(ema_pos[:, 1])
                sign_y = torch.where(sign_y == 0, torch.ones_like(sign_y), sign_y)
                new_x = torch.where(inside & push_x, sign_x * h, ema_pos[:, 0])
                new_y = torch.where(inside & (~push_x), sign_y * h, ema_pos[:, 1])
                ema_pos = torch.stack([new_x, new_y, ema_pos[:, 2]], dim=-1)
        self._processed_actions[:, :3] = ema_pos
        self._processed_actions[:, 3:7] = self.init_ee_quat
        self._prev_applied_pos[:] = ema_pos
        ee_pos_curr, ee_quat_curr = self._compute_frame_pose()
        self._ik_controller.set_command(self._processed_actions, ee_pos_curr, ee_quat_curr)
```

`ActionsCfg` (abstract, fields MISSING) + the action-stack portion of
`FrankaLiftBoxEnvCfg.__post_init__` (verbatim):

```python
@configclass
class ActionsCfg:
    """Per-robot 3-D xyz EMA-smoothed EE-delta + binary gripper. Fields are filled
    by the franka subclass `__post_init__`."""

    arm_action_0: "mdp.EMACumulativeDeltaPositionActionCfg" = MISSING
    gripper_action_0: "mdp.BinaryJointPositionActionCfg" = MISSING
    arm_action_1: "mdp.EMACumulativeDeltaPositionActionCfg" = MISSING
    gripper_action_1: "mdp.BinaryJointPositionActionCfg" = MISSING
```

```python
        # Per-robot action stack: 3-D EMA xyz EE-delta + binary gripper.
        self.actions.arm_action_0 = mdp.EMACumulativeDeltaPositionActionCfg(
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
        self.actions.gripper_action_0 = mdp.BinaryJointPositionActionCfg(
            asset_name="robot_0",
            joint_names=["fr3_finger.*"],
            open_command_expr={"fr3_finger_.*": 0.04},
            close_command_expr={"fr3_finger_.*": 0.0},
        )
        self.actions.arm_action_1 = mdp.EMACumulativeDeltaPositionActionCfg(
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
        self.actions.gripper_action_1 = mdp.BinaryJointPositionActionCfg(
            asset_name="robot_1",
            joint_names=["fr3_finger.*"],
            open_command_expr={"fr3_finger_.*": 0.04},
            close_command_expr={"fr3_finger_.*": 0.0},
        )
```

### Smoke
- `smoke_s2.py` — drives 0.2 across all dims; asserts `arm_action_0.processed_actions[:, :3]`
  equals `clamp(init_ee_pos + alpha*scale*a, pos_lower, pos_upper)` and quat stays locked at
  `init_ee_quat`. Key: `S2 OK: mode=ema_delta_ee_pose action 0.2 -> EE pos delta ~ <d>, quat locked`.
- `smoke_s2_5.py` — actuator-tracking sanity: holds zero EE-delta for 150 steps, asserts each
  arm's achieved joint_pos tracks `joint_pos_target` within 0.05 rad. Key: `S2.5 OK: held 150
  steps, both arms track targets ...`.

---

## §3 — Reset / Events

### Description
Reset-only event manager (no DR). Both robots reset to URDF home; the box resets to a small
uniform xy jitter around the table origin at its spawn height.

### Decisions resolved
- `reset_robot_0_joints` / `reset_robot_1_joints`: `reset_joints_by_scale`,
  `position_range=(1.0,1.0)` (URDF home, deterministic), `velocity_range=(0.0,0.0)`.
- `reset_box`: `reset_root_state_uniform`, `pose_range={x:(-0.03,0.03), y:(-0.03,0.03), z:(0,0)}`,
  empty velocity_range.
- All `mode="reset"`. **No randomization (§7) terms** — see §7.

### Code

`EventCfg` (verbatim):

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
`smoke_s3.py` — pins reset ranges to point intervals, asserts after reset: box xy at env-local
origin (atol 1e-3), box z == BOX_INIT_Z (atol 2e-2), both robots' joint_pos == default_joint_pos
(atol 1e-2). Key: `S3 OK: every injected init value matches simulation state after reset`.

---

## §4 — Goal + Termination

### Description
The goal is HARD-CODED in `mdp/terminations.py` (no `CommandsCfg`). Two terminations:
`time_out` (truncation at the 200-step episode cap) and `success` (`lift_box_success` — box
centered at target xy, lifted to `BOX_INIT_Z + lift_height`, low velocity, AND a genuine
dual-finger grasp by both robots). No failure-mode terminations.

### Decisions resolved
- `time_out`: `mdp.time_out`, `time_out=True`.
- `success`: `mdp.lift_box_success`, `time_out=False`, params `target_xy=(0,0)`,
  `lift_height=0.25`, `xy_pos_tol=0.05`, `z_pos_tol=0.05`, `vel_tol=0.10`,
  `contact_force_threshold=1e-3`, `box_cfg=SceneEntityCfg("box")`.
- The success predicate's dual-grasp gate (`gate_0 & gate_1` via `_both_fingers_in_contact`,
  imported from `rewards.py`) was added during reward tuning — see §6 Tune provenance. Import
  direction is safe (rewards.py does not import terminations.py; mdp `__init__` loads `.rewards`
  before `.terminations`).

### Code

`TerminationsCfg` (verbatim):

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
            "contact_force_threshold": 1e-3,
            "box_cfg": SceneEntityCfg("box"),
        },
    )
```

`mdp/terminations.py` — `lift_box_success` (verbatim):

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

# Reuse the dual-finger contact gate authored for the reward ladder. Import
# direction is safe: rewards.py does NOT import from terminations.py, and the
# mdp package __init__ imports `.rewards` before `.terminations`, so no cycle.
from .rewards import _both_fingers_in_contact

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


# Half-z extent of the eurobox after MeshConverter recentering. Must match the
# `BOX_INIT_Z` constant used in `joint_pos_env_cfg.py`.
BOX_INIT_Z: float = 0.11025


def lift_box_success(
    env: "ManagerBasedRLEnv",
    target_xy: tuple[float, float] = (0.0, 0.0),
    lift_height: float = 0.25,
    xy_pos_tol: float = 0.05,
    z_pos_tol: float = 0.05,
    vel_tol: float = 0.10,
    contact_force_threshold: float = 1e-3,
    box_cfg: SceneEntityCfg = SceneEntityCfg("box"),
) -> torch.Tensor:
    """True when the box is centered (xy) within `xy_pos_tol` of `target_xy`,
    its z is within `z_pos_tol` of `BOX_INIT_Z + lift_height`, its linear
    velocity magnitude is below `vel_tol`, AND BOTH robots have both fingers in
    contact with the box (a genuine dual grasp). All pose/velocity quantities
    are in env-local world coordinates (root_pos_w - env_origins).

    The grasp gate closes the iter-0 exploit where the policy wedge-lifted the
    box between the wrists/palms (no finger grasp) and still triggered success.
    `success_bonus` in rewards.py mirrors the same predicate, so the +100 bonus
    is now likewise unreachable without a real dual-finger grasp.
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

    gate_0 = _both_fingers_in_contact(
        env, "finger_left_contact_0", "finger_right_contact_0", contact_force_threshold
    )
    gate_1 = _both_fingers_in_contact(
        env, "finger_left_contact_1", "finger_right_contact_1", contact_force_threshold
    )
    dual_grasp = gate_0 & gate_1

    return (xy_err < xy_pos_tol) & (z_err < z_pos_tol) & (vel_norm < vel_tol) & dual_grasp
```

### Smoke
`smoke_s4.py` — asserts no command terms; both `time_out` and `success` active; under zero
actions the `time_out` truncation fires within `cap+1` steps. Key: `S4 OK: no command terms;
time_out termination truncates at the episode cap`. (Success-firing path exercised by the
auxiliary `smoke_success.py`.)

---

## §5 — Observation

### Description
Single `policy` group, concatenated, corruption enabled (off in PLAY). 33-D total: per-robot EE
pose in robot-root frame (7+7), box position in env-local world (3), box quat in world (4),
per-robot gripper joint pos (2+2), and the last action (8).

### Decisions resolved
Term order (must match exactly): `ee_pose_0(7)`, `ee_pose_1(7)`, `box_position_in_world(3)`,
`box_quat_in_world(4)`, `gripper_joint_pos_0(2)`, `gripper_joint_pos_1(2)`, `last_action(8)` =
33-D. `ee_pose_*` from `mdp.ee_pose_in_robot_root_frame` (per-robot robot_cfg + ee_frame_cfg);
gripper pos from `mdp.joint_pos` (fr3_finger.*); `last_action` from `mdp.last_action`.

### Code

`ObservationsCfg` (verbatim):

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

`mdp/observations.py` (verbatim):

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
`smoke_s5.py` — asserts the 7-term order/shape list above (33-D total) and value-checks
`box_position_in_world` slice == `box.root_pos_w - env_origins`. Key: `S5 OK: 7 terms, order +
shapes match (33-D), box_position value check verified`. (Schema also pinned in
`smokes/expected_obs.json`.)

---

## §6 — Reward

### Description
Composer = **sum**; sign convention positive=good. Five-stage dense ladder + one-shot sparse
success bonus, gated so that all post-grasp income is conditional on a genuine dual-finger grasp
by BOTH robots:
1. `ee_0_to_grasp_0` / `ee_1_to_grasp_1` — tanh EE→grasp-point attractors (ungated breadcrumb).
2. `grasp_contact_0` / `grasp_contact_1` — per-robot binary "both fingers in contact with box".
3. `lift_height` — linear lift ramp [0,target_lift], gated on dual contact by both robots.
4. `box_xy_align` — tanh box-xy→target attractor, gated on lifted AND dual contact.
5. `success_bonus` — one-shot +100 latch mirroring the `lift_box_success` predicate (dual grasp).

Declared weights are nominal per-step magnitudes, applied directly — the declared weight is
what each term pays per step.

**This reward was validated by training**: PPO @4096 envs, 40M steps → success_rate **0.89**
(checkpoint `harbor/outputs/ppo_Isaac-Lift-Box-Dual-Franka-v0_20260606-231207`). It is the
TUNED reward — 3 tune iterations on top of the original spec; the success predicate +
`box_xy_align` are now gated on dual finger contact, `grasp_contact` weights 0.05, `lift_height`
0.25.

**Tune provenance** (factual, short): derived from the IsaacLab-Lift-Box spec. Iter-1 added the
dual-finger-grasp gate to `success_bonus` + `lift_box_success` after training exposed a
wedge-lift exploit (box pinned between wrists, no finger grasp). Iter-2 added the same
dual-contact gate to `box_xy_align` and raised `lift_height` 0.1875→0.25 after training exposed
a tip-on-end align-farming exploit (box stood on its short end to open the loose `lifted` gate).
The verbatim weight-budget reasoning is preserved in the `RewardsCfg` docstring below.

### Decisions resolved
- Per-stage NOMINAL weights: reach 0.0125 ea, grasp_contact 0.05 ea,
  lift_height 0.25, box_xy_align 0.125, success_bonus 100.0.
- Contact gate: `_both_fingers_in_contact` thresholds each finger's `force_matrix_w` norm at
  `1e-3`. `dual_contact = gate_0 & gate_1`.
- Success / latch params mirror the §4 termination exactly (target_xy (0,0), lift_height 0.25,
  xy/z tol 0.05, vel_tol 0.10, init_z 0.11025).

### Code

`RewardsCfg` (verbatim, including the full tuning-rationale docstring):

```python
@configclass
class RewardsCfg:
    """Reward ladder (composer = sum, sign = positive=good).

    All weights scaled down 80x from the iter-0 converged set so total
    episodic return lands around ~105 instead of ~8400 — same ratios,
    same convergence behaviour, more interpretable magnitudes.

    Per-stage saturated per-step magnitude budget (NOMINAL weights, applied directly):
      reach (x2)   ee_0/1_to_grasp_0/1  -> 0.0125/step ea (0.025 paired) -> 5.0 / 200-step ep   [ungated]
      contact (x2) grasp_contact_0/1    -> 0.05/step ea  (0.10 paired)   -> 20.0 / ep (held)     [ungated]
      lift         lift_height          -> 0.25/step  (dual contact)     -> 50.0 / ep            [grasp-gated]
      align        box_xy_align         -> 0.125/step (dual contact+lift)-> 25.0 / ep            [grasp-gated]
      success      success_bonus        -> one-shot latch (dual grasp)   -> +100
    Dense ceiling ~100; sparse success 100 still dominates the dense steady
    state so the policy is incentivised to FINISH, not park.

    Post-iter-2 the ONLY income reachable without a dual grasp is the ungated
    breadcrumb pair: reach (max 0.5/step effective) + contact (max 2.0/step
    effective). Everything else (lift, align, success) is multiplied by
    dual_contact, so "hover near the box" / "stand it on end" earns at most
    ~2.5/step, while "grasp -> lift -> center" climbs to 2.0 (contact) +
    5.0 (lift) + 2.5 (align) = 9.5/step + the +100 latch. Grasp+lift strictly
    dominates every non-grasp local optimum.

    Iter 1 delta vs iter-0 baseline (closing the wedge-lift exploit):
      - success_bonus AND lift_box_success now both require a genuine dual-finger
        grasp by both robots (gate_0 & gate_1 from _both_fingers_in_contact);
        the +100 / termination are no longer reachable by wedging the box
        between the wrists.
      - grasp_contact_0/1 weights doubled 0.025 -> 0.05 (declared) to help
        robot_1 discover finger contact (iter-0: grasp_contact_1 stayed at 0).
        Still well below lift/align so the term ladder ordering is unchanged.

    Iter 2 delta vs iter-1 (closing the tip-on-end align-farming exploit):
      - box_xy_align is now multiplied by dual_contact (gate_0 & gate_1), the
        same helper as lift_height/success_bonus. Iter-1 evidence: align was
        gated only on a loose `lifted` z-threshold, so standing the box on its
        short end opened the gate and farmed ~2.5/step align with no grasp and
        lift_height ~ 0. With align grasp-gated, all post-grasp income is now
        conditional on a real dual grasp.
      - lift_height weight 0.1875 -> 0.25 (declared; 3.75 -> 5.0/step effective)
        so the lift gradient clearly out-earns align (2.5/step) once grasped,
        making "lift the box" dominate "grasp-and-park". Minimal nudge; ladder
        ordering (reach < contact < align < lift < success) unchanged.
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
        weight=0.05,
    )
    grasp_contact_1 = RewTerm(
        func=mdp.grasp_contact,
        params={"robot_idx": 1, "contact_force_threshold": 1e-3},
        weight=0.05,
    )
    lift_height = RewTerm(
        func=mdp.lift_height,
        params={"init_z": 0.11025, "target_lift": 0.25, "contact_force_threshold": 1e-3},
        weight=0.25,
    )
    box_xy_align = RewTerm(
        func=mdp.box_xy_align,
        params={"std": 0.15, "target_xy": (0.0, 0.0), "lift_threshold": 0.05,
                "init_z": 0.11025, "contact_force_threshold": 1e-3},
        weight=0.125,
    )
    success_bonus = RewTerm(
        func=mdp.success_bonus,
        params={"target_xy": (0.0, 0.0), "lift_height": 0.25, "xy_pos_tol": 0.05,
                "z_pos_tol": 0.05, "vel_tol": 0.10, "contact_force_threshold": 1e-3,
                "init_z": 0.11025},
        weight=100.0,
    )
```

`mdp/rewards.py` — FULL current source (latch registry, contact gate, all reward funcs; verbatim):

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


# Phase 4 — tanh attractor on box xy -> target xy, gated on box lifted
# AND on BOTH robots holding a genuine dual-finger grasp.
def box_xy_align(
    env: "ManagerBasedRLEnv",
    std: float = 0.15,
    target_xy: tuple[float, float] = (0.0, 0.0),
    lift_threshold: float = 0.05,
    init_z: float = BOX_INIT_Z,
    contact_force_threshold: float = 1e-3,
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
    # Iter-2: gate align income on a real dual grasp. The iter-1 tip-on-end
    # exploit standing the box on its short end raised the COM enough to open
    # the loose `lifted` gate and farm align (~2.5/step) WITHOUT any finger
    # grasp. Multiplying by dual_contact removes all align income from any
    # non-grasped pose, so the only path to align reward is grasp -> lift.
    gate_0 = _both_fingers_in_contact(env, "finger_left_contact_0", "finger_right_contact_0", contact_force_threshold)
    gate_1 = _both_fingers_in_contact(env, "finger_left_contact_1", "finger_right_contact_1", contact_force_threshold)
    dual_contact = (gate_0 & gate_1).float()
    return lifted * dual_contact * base


# Phase 5 — one-shot success bonus mirroring the termination predicate.
def success_bonus(
    env: "ManagerBasedRLEnv",
    target_xy: tuple[float, float] = (0.0, 0.0),
    lift_height: float = 0.25,
    xy_pos_tol: float = 0.05,
    z_pos_tol: float = 0.05,
    vel_tol: float = 0.10,
    contact_force_threshold: float = 1e-3,
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
    # Require a genuine dual-finger grasp by BOTH robots (mirrors the
    # lift_box_success termination predicate). Closes the iter-0 wedge-lift
    # exploit where the +100 bonus latched without any finger grasp.
    gate_0 = _both_fingers_in_contact(env, "finger_left_contact_0", "finger_right_contact_0", contact_force_threshold)
    gate_1 = _both_fingers_in_contact(env, "finger_left_contact_1", "finger_right_contact_1", contact_force_threshold)
    dual_grasp = gate_0 & gate_1
    now_success = (xy_err < xy_pos_tol) & (z_err < z_pos_tol) & (vel_norm < vel_tol) & dual_grasp

    fire = now_success & (~latch)
    latch = latch | now_success
    _LATCH_BUFFERS[(id(env), "success_once")] = latch
    return fire.float()
```

### Smoke
`smoke_s6.py` — runs 30 steps × 128 envs through the instrumented IsaacLab env factory
(`scripts/_isaaclab_env.make_isaaclab_env`), asserts reward finite + non-constant + per-term
sum composer (`Σ info["detailed_reward"] == env_reward`, composer="sum") and prints per-term
episodic means. Key: `S6 OK: 30 steps × 128 envs, reward mean=<m> std=<s> composer=sum`.

---

## §7 — Domain Randomization

`<no DR>`. The `EventCfg` contains ONLY reset terms (`reset_robot_0_joints`,
`reset_robot_1_joints`, `reset_box` — all `mode="reset"`, point/near-point ranges). There are no
`mode="reset"`-randomization or `mode="interval"` terms acting on robot params, object physics,
or observation noise. (Verified by reading `EventCfg` — see §3.) Observation corruption is
enabled in the policy group config but is an `ObsGroup` flag, not a §7 randomization term.

---

## Reproduction

Clone this task identically into another benchmark:

```
/harbor:task-create name=<new_task_id> from=harbor/create-task/isaac-lift-box-dual-franka-v0-implementation.md
```

Reproduce mode pastes §6 reward verbatim + smokes only (no re-tuning); §7 is `<no DR>` (skipped).
The new task must vendor the FR3 cfg + the `EMACumulativeDeltaPositionAction` classes locally (as
here), provide the three USD assets under `<repo>/harbor/assets/`, and rely on the
benchmark-generator dt-strip (nominal weights applied directly — no `__post_init__`
cancellation loop).
```
