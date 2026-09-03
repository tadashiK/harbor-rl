# IsaacLab-Insert-Drawer — Implementation Spec

- robot: Franka FR3 arm + Franka hand (single arm)
- simulator: IsaacLab (Isaac Sim, manager-based)
- objects: DexCube, drawer cabinet, lab table
- bimanual: false
- summary: Pick a cube off the table, place it inside an open drawer, and push the drawer closed.

Task summary: Franka FR3 (single arm) starts with a drawer ALREADY OPEN (joint pos = 0.30 m). The policy must (1) pick up a small DexCube from the table, (2) lift it above the drawer rim, (3) place it inside the open drawer, (4) retract the gripper out of the drawer interior, and (5) push the drawer closed. Episode ends on either time-out (9 s @ 20 Hz = 180 steps) or task success (cube inside + drawer joint pos < 0.10 m). Composer = sum, 8 active reward terms with monotonically-increasing per-step magnitudes (`reach < lift < align < retract < close < cube_inside_latch < success_bonus`).

---

## §1 Registration + Scene

### Description

`gym.register` exposes `IsaacLab-Insert-Drawer` (training) and `IsaacLab-Insert-Drawer-Play` (eval-friendly, fewer envs, no noise). Both use `isaaclab.envs:ManagerBasedRLEnv` with the abstract `InsertDrawerEnvCfg` base + `FrankaInsertDrawerEnvCfg` subclass that fills in the robot, IK action, cube, drawer, EE-frame transformer, drawer drop-frame transformer, and drawer front-face transformer. Two finger contact sensors are declared in the base scene and filtered against `Cube_0` for the lift-distance grasp gate.

### Decisions resolved

| Knob | Value |
|---|---|
| Task ID | `IsaacLab-Insert-Drawer` |
| Robot | FR3 + Franka hand (custom USD `harbor/assets/fr3/fr3_franka_hand.usd`) |
| Robot init pos (env-local) | `(-0.274, 0.49, 0.01)` |
| Cube | DexCube (Nucleus USD, scaled 0.86 → 4.3 cm edge), 55 g mass |
| Drawer | No-handle prismatic drawer at `harbor/assets/drawer_no_handle/drawer_no_handle.usd`; pre-scaled URDF → spawn scale (1,1,1); joint `base_drawer_joint` axis +X-local, range [0, 0.3] m |
| Drawer init pose | `pos=(0, 0, 0.10)`, `rot=(0.7071, 0, 0, 0.7071)` (90° about +Z so prismatic +X-local maps to world +Y) |
| Table | `harbor/assets/table/lab_table_instanceable_colored_rotated.usd` (rotated the source repo lab table), surface at z≈0 |
| Ground plane | `z = -0.82` |
| ee_frame target | `fr3_hand` with `+Z` offset 0.2 m (fingertip TCP) |
| drawer_drop_frame target | `Drawer/drawer` body with offset `(0, 0, 0.25)` drawer-local (≈ above the open tray rim) |
| drawer_front_face_frame target | `Drawer/drawer` body with offset `(0.17, 0, 0.15)` drawer-local (front +X-local face center) |
| Contact sensors | `finger_left_contact` on `Robot/fr3_leftfinger`, `finger_right_contact` on `Robot/fr3_rightfinger`, both filtered against `{ENV_REGEX_NS}/Cube_0` |
| `num_envs` (default) | 4096 |
| `env_spacing` | 2.5 m |
| `replicate_physics` | False |
| Timing | `episode_length_s=9.0`, `sim.dt=1/120`, `decimation=6` → 20 Hz control |

### Code (verbatim)

`config/franka/__init__.py`:

```python
import gymnasium as gym

gym.register(
    id="IsaacLab-Insert-Drawer",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:FrankaInsertDrawerEnvCfg",
    },
)

gym.register(
    id="IsaacLab-Insert-Drawer-Play",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:FrankaInsertDrawerEnvCfg_PLAY",
    },
)
```

Abstract scene (`insert_drawer_env_cfg.py:InsertDrawerSceneCfg`):

```python
_TABLE_USD_PATH = str(
    Path(__file__).resolve().parents[6]
    / "harbor" / "assets" / "table" / "lab_table_instanceable_colored_rotated.usd"
)


@configclass
class InsertDrawerSceneCfg(InteractiveSceneCfg):
    robot: ArticulationCfg = MISSING
    ee_frame: FrameTransformerCfg = MISSING
    cube_0: RigidObjectCfg = MISSING
    drawer: ArticulationCfg = MISSING
    drawer_drop_frame: FrameTransformerCfg = MISSING
    drawer_front_face_frame: FrameTransformerCfg = MISSING

    finger_left_contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/fr3_leftfinger",
        update_period=0.0, history_length=1, debug_vis=False,
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Cube_0"],
    )
    finger_right_contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/fr3_rightfinger",
        update_period=0.0, history_length=1, debug_vis=False,
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Cube_0"],
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

Per-robot subclass (`config/franka/joint_pos_env_cfg.py`) — full robot + cube + drawer + ee_frame + drop_frame + front_face_frame wiring. The full file is included at the source-file pointers below (372 lines); the critical subset that defines the assets is:

```python
_HARBOR_ASSETS = Path(__file__).resolve().parents[8] / "harbor" / "assets"
_DRAWER_USD_PATH = str(_HARBOR_ASSETS / "drawer_no_handle" / "drawer_no_handle.usd")
_FR3_USD_PATH    = str(_HARBOR_ASSETS / "fr3" / "fr3_franka_hand.usd")

CUBE_SIZE = 0.043; CUBE_USD_SCALE = 0.86; CUBE_MASS = 0.055; CUBE_INIT_Z = CUBE_SIZE / 2.0

FRANKA_INIT_JOINT_POS = {
    "fr3_joint1": -0.785, "fr3_joint2": -0.785, "fr3_joint3": 0.0,
    "fr3_joint4": -2.655, "fr3_joint5": 0.0,    "fr3_joint6": 1.87,
    "fr3_joint7": 1.57,   "fr3_finger_joint.*": 0.04,
}

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
        "fr3_forearm":  ImplicitActuatorCfg(joint_names_expr=["fr3_joint[5-7]"], effort_limit_sim=12.0, stiffness=400.0, damping=80.0),
        "fr3_hand":     ImplicitActuatorCfg(joint_names_expr=["fr3_finger_joint.*"], effort_limit_sim=200.0, stiffness=2e3, damping=1e2),
    },
    soft_joint_pos_limit_factor=1.0,
)

# In FrankaInsertDrawerEnvCfg.__post_init__:
self.scene.robot = FR3_FRANKA_HAND_CFG.replace(
    prim_path="{ENV_REGEX_NS}/Robot",
    init_state=ArticulationCfg.InitialStateCfg(joint_pos=FRANKA_INIT_JOINT_POS, pos=(-0.274, 0.49, 0.01)),
)

self.scene.cube_0 = RigidObjectCfg(
    prim_path="{ENV_REGEX_NS}/Cube_0",
    init_state=RigidObjectCfg.InitialStateCfg(pos=[0.0, 0.0, CUBE_INIT_Z], rot=[1.0, 0.0, 0.0, 0.0]),
    spawn=UsdFileCfg(
        usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd",
        scale=(CUBE_USD_SCALE,)*3,
        mass_props=sim_utils.MassPropertiesCfg(mass=CUBE_MASS),
        rigid_props=RigidBodyPropertiesCfg(
            solver_position_iteration_count=16, solver_velocity_iteration_count=1,
            max_angular_velocity=1000.0, max_linear_velocity=1000.0,
            max_depenetration_velocity=5.0, disable_gravity=False,
        ),
    ),
)

self.scene.drawer = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/Drawer",
    spawn=sim_utils.UsdFileCfg(
        usd_path=_DRAWER_USD_PATH,
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=True, max_depenetration_velocity=1000.0,
            max_linear_velocity=1000, max_angular_velocity=1000,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=16, solver_velocity_iteration_count=1,
            fix_root_link=True,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={"base_drawer_joint": 0.0},
        pos=(0.0, 0.0, 0.10),
        rot=(0.7071068, 0.0, 0.0, 0.7071068),
    ),
    actuators={
        "drawer_slide": ImplicitActuatorCfg(
            joint_names_expr=["base_drawer_joint"],
            effort_limit=87.0, velocity_limit=100.0,
            stiffness=0.0, damping=1.0, friction=2.0,
        ),
    },
)

# Frame transformers — ee_frame on robot, drop_frame above drawer interior, front_face_frame on drawer front wall.
self.scene.ee_frame = FrameTransformerCfg(
    prim_path="{ENV_REGEX_NS}/Robot/fr3_link0",
    debug_vis=True, visualizer_cfg=ee_marker_cfg,
    target_frames=[FrameTransformerCfg.FrameCfg(
        prim_path="{ENV_REGEX_NS}/Robot/fr3_hand", name="end_effector",
        offset=OffsetCfg(pos=[0.0, 0.0, 0.2]),
    )],
)
self.scene.drawer_drop_frame = FrameTransformerCfg(
    prim_path="{ENV_REGEX_NS}/Drawer/base_link",
    debug_vis=True, visualizer_cfg=drop_marker_cfg,
    target_frames=[FrameTransformerCfg.FrameCfg(
        prim_path="{ENV_REGEX_NS}/Drawer/drawer", name="drawer_drop",
        offset=OffsetCfg(pos=[0.0, 0.0, 0.25]),
    )],
)
self.scene.drawer_front_face_frame = FrameTransformerCfg(
    prim_path="{ENV_REGEX_NS}/Drawer/base_link",
    debug_vis=True, visualizer_cfg=front_marker_cfg,
    target_frames=[FrameTransformerCfg.FrameCfg(
        prim_path="{ENV_REGEX_NS}/Drawer/drawer", name="drawer_front_face",
        offset=OffsetCfg(pos=[0.17, 0.0, 0.15]),
    )],
)
```

`EnvCfg.__post_init__`:

```python
def __post_init__(self):
    self.decimation = 6
    self.episode_length_s = 9.0
    self.sim.dt = 1 / 120
    self.sim.render_interval = self.decimation
    self.sim.physx.bounce_threshold_velocity = 0.01            # second write wins
    self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 1024 * 1024 * 4
    self.sim.physx.gpu_total_aggregate_pairs_capacity = 64 * 1024
    self.sim.physx.friction_correlation_distance = 0.00625
```

### Smoke

§1 build smoke (launch via the rendered train.py because Isaac Sim's AppLauncher must initialize before `gym.make`):

```bash
cd "<repo>"
.venv/bin/python -c "
from isaaclab.app import AppLauncher
import argparse
parser = argparse.ArgumentParser(); AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args(['--headless']); app = AppLauncher(args); sim_app = app.app
import isaaclab_tasks, gymnasium as gym
spec = gym.spec('IsaacLab-Insert-Drawer'); print('OK:', spec.id, spec.entry_point)
"
```

Expected: `OK: IsaacLab-Insert-Drawer isaaclab.envs:ManagerBasedRLEnv`.

---

## §2 Actions

### Description

3-DOF EE-delta IK with locked RPY for the arm + 1-DOF binary gripper. The policy outputs `(dx, dy, dz, gripper)` (action_dim = 4). Per step:

```
delta_t = clamp(delta_{t-1} + scale * action_t, pos_lower - init_ee_pos, pos_upper - init_ee_pos)
abs_pos_t = init_ee_pos + delta_t                         # init_ee_pos captured lazily on first step after reset
quat_t   = init_ee_quat                                   # RPY locked
target_t.pos = alpha * abs_pos_t + (1 - alpha) * prev_applied_pos
target_t     = clamp(target_t, pos_lower_limit, pos_upper_limit)   # defensive
```

The 7-D `(target_pos, init_quat)` is then forwarded to a DLS DifferentialIK controller in absolute-pose mode (`use_relative_mode=False`, `command_type="pose"`). `del_action` is clamped at the source so policy reversals take effect on the next step (no overshoot to burn through).

### Decisions resolved

| Knob | Value |
|---|---|
| arm_action class | `mdp.EMACumulativeDeltaPositionActionCfg` |
| arm_action.asset_name | `"robot"` |
| arm_action.joint_names | `["fr3_joint.*"]` (7 arm joints) |
| arm_action.body_name | `"fr3_hand"` |
| arm_action.body_offset (IK TCP) | `pos=(0.0, 0.0, 0.2)` (matches ee_frame offset) |
| arm_action.controller | DLS DifferentialIK, `command_type="pose"`, `use_relative_mode=False` |
| arm_action.scale | `(0.01, 0.01, 0.01)` m / unit policy output |
| arm_action.alpha | `0.5` |
| arm_action.pos_lower_limit | `[0.34, -0.8, 0.005]` (robot-root frame, after body_offset) |
| arm_action.pos_upper_limit | `[0.50, -0.05, 0.30]` |
| arm_action.forbidden_xy_half | `None` |
| gripper_action class | `mdp.BinaryJointPositionActionCfg` |
| gripper_action.joint_names | `["fr3_finger.*"]` |
| gripper_action.open_command_expr | `{"fr3_finger_.*": 0.04}` |
| gripper_action.close_command_expr | `{"fr3_finger_.*": 0.0}` |
| Total action_dim | 4 (3 xyz + 1 gripper) |

### Code (verbatim)

`mdp/actions_cfg.py`:

```python
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

`mdp/actions.py:EMACumulativeDeltaPositionAction` — full class verbatim (184 lines including process_actions with del_action clamping). The full source is at `source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/insert_drawer/mdp/actions.py:1-184`; key reproducible block:

```python
class EMACumulativeDeltaPositionAction(DifferentialInverseKinematicsAction):
    cfg: "actions_cfg.EMACumulativeDeltaPositionActionCfg"

    def __init__(self, cfg, env):
        if cfg.controller.use_relative_mode:
            raise ValueError("EMACumulativeDeltaPositionAction handles relative deltas itself; "
                             "set controller.use_relative_mode=False")
        if cfg.controller.command_type != "pose":
            raise ValueError(f"requires command_type='pose'; got '{cfg.controller.command_type}'")
        super().__init__(cfg, env)
        self._raw_actions = torch.zeros(env.num_envs, 3, device=env.device)
        self._processed_actions = torch.zeros(env.num_envs, 7, device=env.device)
        self._scale = torch.zeros((env.num_envs, 3), device=env.device); self._scale[:] = torch.tensor(cfg.scale, device=env.device)
        if not 0.0 <= cfg.alpha <= 1.0: raise ValueError(f"alpha must be in [0, 1]. Got {cfg.alpha}.")
        self._alpha = cfg.alpha
        self.del_action = torch.zeros((env.num_envs, 3), device=env.device)
        self.init_ee_pos = torch.zeros((env.num_envs, 3), device=env.device)
        self.init_ee_quat = torch.zeros((env.num_envs, 4), device=env.device); self.init_ee_quat[:, 0] = 1.0
        self._prev_applied_pos = torch.zeros((env.num_envs, 3), device=env.device)
        self._needs_reanchor = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
        self.pos_lower_limit = torch.tensor(cfg.pos_lower_limit, device=self.device) if cfg.pos_lower_limit is not None else None
        self.pos_upper_limit = torch.tensor(cfg.pos_upper_limit, device=self.device) if cfg.pos_upper_limit is not None else None
        self.forbidden_xy_half: float | None = getattr(cfg, "forbidden_xy_half", None)

    @property
    def action_dim(self) -> int: return 3

    def reset(self, env_ids=None):
        super().reset(env_ids)
        if env_ids is None: self._needs_reanchor[:] = True; self.del_action[:] = 0.0
        else: self._needs_reanchor[env_ids] = True; self.del_action[env_ids] = 0.0

    def process_actions(self, actions):
        if self._needs_reanchor.any():
            ee_pos_curr, ee_quat_curr = self._compute_frame_pose()
            mask = self._needs_reanchor
            self.init_ee_pos[mask] = ee_pos_curr[mask]; self.init_ee_quat[mask] = ee_quat_curr[mask]
            self._prev_applied_pos[mask] = ee_pos_curr[mask]; self._needs_reanchor[:] = False
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
        # forbidden_xy_half block: pushes target out of |x|<=h ∧ |y|<=h square to nearest edge (skipped here when None)
        self._processed_actions[:, :3] = ema_pos
        self._processed_actions[:, 3:7] = self.init_ee_quat
        self._prev_applied_pos[:] = ema_pos
        ee_pos_curr, ee_quat_curr = self._compute_frame_pose()
        self._ik_controller.set_command(self._processed_actions, ee_pos_curr, ee_quat_curr)
```

Action cfg in `FrankaInsertDrawerEnvCfg.__post_init__`:

```python
self.actions.arm_action = mdp.EMACumulativeDeltaPositionActionCfg(
    asset_name="robot",
    joint_names=["fr3_joint.*"],
    body_name="fr3_hand",
    body_offset=DifferentialInverseKinematicsActionCfg.OffsetCfg(pos=(0.0, 0.0, 0.2)),
    controller=DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls"),
    scale=(0.01, 0.01, 0.01), alpha=0.5,
    pos_lower_limit=[0.34, -0.8, 0.005], pos_upper_limit=[0.50, -0.05, 0.30],
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
.venv/bin/python -c "
# inside AppLauncher: build env, take 10 zero actions, assert no NaN
import torch
env = make_env('IsaacLab-Insert-Drawer')
obs, _ = env.reset()
for _ in range(10):
    a = torch.zeros((env.num_envs, 4), device=env.device)
    obs, *_ = env.step(a)
    assert torch.isfinite(obs['policy']).all()
print('action smoke OK')
"
```

---

## §3 Reset

### Description

`EventCfg` defines four `mode="reset"` terms: robot joints reset to URDF home, cube spawns at a small xy box on the table, drawer joint pinned to 0.30 m (FULLY OPEN), drawer world pose pinned to a small xy box in front of the robot. All ranges are pinned (no jitter beyond the cube's small xy box) — DR widening is left to §7.

### Decisions resolved

| Term | Function | Key params |
|---|---|---|
| reset_robot_joints | `mdp.reset_joints_by_scale` | `position_range=(1.0, 1.0)`, `velocity_range=(0.0, 0.0)` (no asset_cfg → all robot joints) |
| reset_cube_0 | `mdp.reset_root_state_uniform` | `pose_range={"x": (0.1, 0.2), "y": (0.3, 0.4), "z": (0.0, 0.0)}`, `asset_cfg=cube_0` |
| reset_drawer_joint | `mdp.reset_joints_by_offset` | `position_range=(0.30, 0.30)`, `velocity_range=(0.0, 0.0)`, `asset_cfg=SceneEntityCfg("drawer", joint_names=["base_drawer_joint"])` |
| reset_drawer_pose | `mdp.reset_root_state_uniform` | `pose_range={"x": (0.1, 0.2), "y": (-0.3, -0.3), "z": (0.0, 0.0)}`, `asset_cfg=drawer` |

Note: `reset_joints_by_scale` multiplies the joint default; `reset_joints_by_offset` adds to it. The drawer joint default is 0, so the offset term is required to pin it open at reset.

### Code (verbatim)

```python
@configclass
class EventCfg:
    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={"position_range": (1.0, 1.0), "velocity_range": (0.0, 0.0)},
    )
    reset_cube_0 = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (0.1, 0.2), "y": (0.3, 0.4), "z": (0.0, 0.0)},
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("cube_0"),
        },
    )
    reset_drawer_joint = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "position_range": (0.30, 0.30),
            "velocity_range": (0.0, 0.0),
            "asset_cfg": SceneEntityCfg("drawer", joint_names=["base_drawer_joint"]),
        },
    )
    reset_drawer_pose = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (0.1, 0.2), "y": (-0.3, -0.3), "z": (0.0, 0.0)},
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("drawer"),
        },
    )
```

### Smoke

Reset 10 times, assert robot joints, cube xyz, drawer joint pos all match the configured ranges on every reset:

```bash
.venv/bin/python -c "
env = make_env('IsaacLab-Insert-Drawer')
for _ in range(10):
    obs, _ = env.reset()
    cube_z = env.scene['cube_0'].data.root_pos_w[:, 2]
    drawer_jp = env.scene['drawer'].data.joint_pos[:, 0]
    assert (cube_z < 0.05).all()
    assert torch.allclose(drawer_jp, torch.full_like(drawer_jp, 0.30), atol=1e-3)
print('reset smoke OK')
"
```

---

## §4 Goal + Termination

### Description

Two DoneTerms: `time_out` ends the episode at `episode_length_s=9.0` (max_episode_steps=180); `success` ends the episode the first frame the cube_inside latch is active AND the drawer joint pos < 0.10 m. The success DoneTerm uses `time_out=False` so the policy collects the `success_bonus` reward (+2000) without bootstrapping past the terminal. There are NO failure-mode terminations (no cube-dropped, no out-of-workspace) — failure signals are kept on the reward side via the latch + dense gradients. No `CommandsCfg` (goal is implicit).

### Decisions resolved

| Term | Function | `time_out` | Params |
|---|---|---|---|
| time_out | `mdp.time_out` | `True` | (none) |
| success | `mdp.success_termination` | `False` | `drawer_closed_threshold=0.1` |
| commands | None | — | — |

### Code (verbatim)

```python
@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    success  = DoneTerm(func=mdp.success_termination, time_out=False,
                        params={"drawer_closed_threshold": 0.1})
```

`mdp/terminations.py:success_termination`:

```python
def success_termination(
    env, drawer_closed_threshold: float = 0.05,
    drawer_cfg: SceneEntityCfg = SceneEntityCfg("drawer"),
    joint_name: str = "base_drawer_joint",
) -> torch.Tensor:
    drawer: Articulation = env.scene[drawer_cfg.name]
    joint_idx = drawer.find_joints(joint_name)[0][0]
    joint_pos = drawer.data.joint_pos[:, joint_idx]
    drawer_closed = joint_pos < drawer_closed_threshold
    latch_active = _cube_inside_latch_active(env) > 0.5
    return latch_active & drawer_closed
```

### Smoke

Roll out a no-op policy for 200 steps, assert exactly one episode ends per env (time_out only — no success expected from zero actions), `info["dones"]` flips at step 180:

```bash
.venv/bin/python -c "
env = make_env('IsaacLab-Insert-Drawer', num_envs=4)
obs, _ = env.reset()
dones_count = 0
for t in range(190):
    obs, *_, done, _ = env.step(torch.zeros((4, 4), device=env.device))
    if done.any(): dones_count += int(done.sum())
assert dones_count == 4
print('termination smoke OK')
"
```

---

## §5 Observation

### Description

19-D concatenated policy obs in the robot root frame. Five terms: 7-D EE pose (xyz + wxyz quat), 3-D cube xyz (zero-masked once cube_inside_bonus latch fires), 3-D sliding drawer body xyz, 2-D finger joint positions, 4-D last action. `enable_corruption=True` (per-term noise is no-op by default; DR widens later). Cube position is the only obs gated by the latch — once the cube is inserted, the policy no longer needs cube state to push the drawer closed.

### Decisions resolved

| Term | Function | Shape | Params |
|---|---|---|---|
| ee_pose | `mdp.ee_pose_in_robot_root_frame` | (7,) | (default `ee_frame_cfg`) |
| cube_position | `mdp.cube_position_in_robot_root_frame` (zero-masked post-latch) | (3,) | (default `cube_cfg`) |
| drawer_body_position | `mdp.drawer_body_position_in_robot_root_frame` | (3,) | (default `drawer_cfg`, `drawer_body_name="drawer"`) |
| gripper_joint_pos | `mdp.joint_pos` (IsaacLab built-in) | (2,) | `asset_cfg=SceneEntityCfg("robot", joint_names=["fr3_finger.*"])` |
| last_action | `mdp.last_action` (IsaacLab built-in) | (4,) | (none) |
| **Total obs dim** | | **19** | |
| `enable_corruption` | `True` |
| `concatenate_terms` | `True` |

### Code (verbatim)

`ObservationsCfg`:

```python
@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        ee_pose = ObsTerm(func=mdp.ee_pose_in_robot_root_frame)
        cube_position = ObsTerm(func=mdp.cube_position_in_robot_root_frame)
        drawer_body_position = ObsTerm(func=mdp.drawer_body_position_in_robot_root_frame)
        gripper_joint_pos = ObsTerm(
            func=mdp.joint_pos,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=["fr3_finger.*"])},
        )
        last_action = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()
```

`mdp/observations.py` — full file (imports `_cube_inside_latch_active` from `.rewards`):

```python
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformer
from isaaclab.utils.math import subtract_frame_transforms
from .rewards import _cube_inside_latch_active


def ee_pose_in_robot_root_frame(env, ee_frame_cfg=SceneEntityCfg("ee_frame")):
    robot: RigidObject = env.scene["robot"]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    ee_pos_w = ee_frame.data.target_pos_w[..., 0, :]
    ee_quat_w = ee_frame.data.target_quat_w[..., 0, :]
    ee_pos_b, ee_quat_b = subtract_frame_transforms(
        robot.data.root_pos_w, robot.data.root_quat_w, ee_pos_w, ee_quat_w,
    )
    return torch.cat([ee_pos_b, ee_quat_b], dim=-1)


def cube_position_in_robot_root_frame(env, cube_cfg=SceneEntityCfg("cube_0")):
    """Cube xyz in robot root frame, zero-masked once cube_inside latch fires."""
    robot: RigidObject = env.scene["robot"]
    cube: RigidObject = env.scene[cube_cfg.name]
    cube_pos_w = cube.data.root_pos_w[:, :3]
    cube_pos_b, _ = subtract_frame_transforms(
        robot.data.root_pos_w, robot.data.root_quat_w, cube_pos_w,
    )
    return cube_pos_b * (1.0 - _cube_inside_latch_active(env)).unsqueeze(-1)


def drawer_body_position_in_robot_root_frame(
    env, drawer_cfg=SceneEntityCfg("drawer"), drawer_body_name="drawer",
):
    robot: RigidObject = env.scene["robot"]
    drawer: Articulation = env.scene[drawer_cfg.name]
    body_idx = drawer.find_bodies(drawer_body_name)[0][0]
    body_pos_w = drawer.data.body_pos_w[:, body_idx, :]
    body_pos_b, _ = subtract_frame_transforms(
        robot.data.root_pos_w, robot.data.root_quat_w, body_pos_w,
    )
    return body_pos_b
```

### Smoke

```bash
.venv/bin/python -c "
env = make_env('IsaacLab-Insert-Drawer')
obs, _ = env.reset()
flat = obs['policy']                 # (num_envs, 19)
assert flat.shape[-1] == 19
assert torch.isfinite(flat).all()
print('obs smoke OK')
"
```

---

## §6 Reward

### Description

Composer = **sum**, sign = **positive = good**. 8 active reward terms organized into 5 phases (per [reward-experience.md entry #2](../../../reward-tuning-agent/reward-experience.md): the per-step magnitudes strictly increase across stages, and dense terms ≤ steady-state budget while sparse one-shot bonuses sit an order of magnitude above the dense sum).

Critical mechanism: a **per-(env, key) latch buffer** (`_LATCH_BUFFERS["cube_inside_once"]`) is written when `cube_inside_bonus_once_per_episode` fires (cube geometrically inside drawer + EE far). Phase 2 dense terms (`reach_cube`, `is_lifted`, `lift_distance`, `align`) multiply by `(1 - latch_active)` so they ZERO OUT once the cube is inserted, freeing the policy to focus on retract + close without dense distractions. Phase 3 `ee_retract_to_front_face` and Phase 4 `close_drawer` multiply by `latch_active` (fires only after cube inserted). Phase 5 `success_bonus` fires once when latch + drawer-closed, paired with the §4 `success` DoneTerm for terminal collection.

### Decisions resolved (planned per-step budget at saturation, after weight × per-step value)

| Stage | Term | weight | per-step max | episodic (180 step) ceiling | gate |
|---|---|---|---|---|---|
| 2 reach | `reach_cube` | 0.02 | ≈ 0.02 | ≈ 3.6 | `(1 - latch)` |
| 2 lift  | `is_lifted` | 0.2 | ≈ 0.2 | ≈ 36 | `(1 - latch)` |
| 2 lift  | `lift_distance` | 0.3 | ≈ 0.3 | ≈ 54 | `(1 - latch) * both_fingers_contact` |
| 2 align | `align` | 2.0 | ≈ 2.0 | ≈ 360 | `(1 - latch) * (cube.z > 0.25)` |
| 3 latch | `cube_inside_bonus_latch` | 300.0 | +300 one-shot | +300 | one-shot |
| 3 retract | `ee_retract_to_front_face` | 2.0 | ≈ 2.0 | ≈ 200 | `latch * y-attractor` |
| 4 close | `close_drawer` | 100.0 | ≈ 100 (alpha=1.0) | ≈ 5000 | `latch * (ee.y > front_y) * closeness^alpha` |
| 5 success | `success_bonus` | 2000.0 | +2000 one-shot | +2000 | `latch * (joint_pos < 0.10)` |
| **Composer** | sum | | | | |

Ladder: dense `reach(3.6) < is_lifted(36) < lift_distance(54) < align(360) < retract(200) < close(5000)` then sparse `latch(+300)` and `success(+2000)`. Note the close_drawer ceiling is much larger than the dense sum — close is the primary driver of policy convergence; the success bonus is a terminal landmark.

### Code (verbatim)

`RewardsCfg`:

```python
@configclass
class RewardsCfg:
    reach_cube = RewTerm(func=mdp.reach_cube, params={"std": 0.1}, weight=0.02)
    is_lifted  = RewTerm(func=mdp.is_lifted, params={"minimal_height": 0.04}, weight=0.2)
    lift_distance = RewTerm(func=mdp.lift_distance, params={"init_z": 0.0215, "target_z": 0.25}, weight=0.3)
    align = RewTerm(func=mdp.align, params={"std": 0.20, "minimal_height_b": 0.25}, weight=2.0)
    ee_retract_to_front_face = RewTerm(func=mdp.ee_retract_to_front_face, params={"std": 0.05}, weight=2.0)
    cube_inside_bonus_latch = RewTerm(
        func=mdp.cube_inside_bonus_once_per_episode,
        params={"xy_threshold": 0.15, "z_rel_floor": -0.02, "z_rel_ceiling": 0.07,
                "ee_cube_no_contact_threshold": 0.10},
        weight=300.0,
    )
    close_drawer = RewTerm(func=mdp.close_drawer, params={"max_open": 0.30, "alpha": 1.0}, weight=100.0)
    success_bonus = RewTerm(func=mdp.success_bonus, params={"drawer_closed_threshold": 0.1}, weight=2000.0)
```

`mdp/rewards.py` — full file (helpers + 8 reward functions). Pasted verbatim:

```python
from __future__ import annotations
from typing import TYPE_CHECKING
import torch
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor, FrameTransformer

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


# Module-level per-(env, key) latch buffers.
_LATCH_BUFFERS: dict[tuple, torch.Tensor] = {}


def _get_latch_buffer(env, key: str) -> torch.Tensor:
    full_key = (id(env), key)
    if full_key not in _LATCH_BUFFERS:
        _LATCH_BUFFERS[full_key] = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    return _LATCH_BUFFERS[full_key]


def _cube_inside_latch_active(env) -> torch.Tensor:
    """Returns float (num_envs,): 1.0 if the cube_inside latch fired earlier this episode."""
    latch = _get_latch_buffer(env, "cube_inside_once")
    just_reset = env.episode_length_buf <= 1
    return (latch & ~just_reset).float()


# ---- Phase 2 — pick + place (zero-masked once latch fires) ----

def reach_cube(env, std=0.1, cube_cfg=SceneEntityCfg("cube_0"), ee_frame_cfg=SceneEntityCfg("ee_frame")):
    cube: RigidObject = env.scene[cube_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    cube_pos_w = cube.data.root_pos_w[:, :3]
    ee_w = ee_frame.data.target_pos_w[..., 0, :]
    d = torch.norm(cube_pos_w - ee_w, dim=1)
    return (1.0 - torch.tanh(d / std)) * (1.0 - _cube_inside_latch_active(env))


def is_lifted(env, minimal_height=0.05, cube_cfg=SceneEntityCfg("cube_0")):
    cube: RigidObject = env.scene[cube_cfg.name]
    cube_z = cube.data.root_pos_w[:, 2]
    return (cube_z > minimal_height).float() * (1.0 - _cube_inside_latch_active(env))


def lift_distance(env, init_z=0.0215, target_z=0.20, contact_force_threshold=1e-3, cube_cfg=SceneEntityCfg("cube_0")):
    """Ramp on cube.z, gated on BOTH finger contact sensors > thr."""
    cube: RigidObject = env.scene[cube_cfg.name]
    cube_z = cube.data.root_pos_w[:, 2]
    base = ((cube_z - init_z) / max(target_z - init_z, 1e-6)).clamp(0.0, 1.0)
    left: ContactSensor = env.scene["finger_left_contact"]
    right: ContactSensor = env.scene["finger_right_contact"]
    left_f  = torch.norm(left.data.force_matrix_w[:, 0, 0, :], dim=-1)
    right_f = torch.norm(right.data.force_matrix_w[:, 0, 0, :], dim=-1)
    grasp_gate = ((left_f > contact_force_threshold) & (right_f > contact_force_threshold)).float()
    return base * grasp_gate * (1.0 - _cube_inside_latch_active(env))


def align(env, std=0.15, minimal_height_b=0.15,
          cube_cfg=SceneEntityCfg("cube_0"),
          drawer_drop_frame_cfg=SceneEntityCfg("drawer_drop_frame")):
    """Cube→drop_frame attractor, gated on cube.z > minimal_height_b (obstacle clearance)."""
    cube: RigidObject = env.scene[cube_cfg.name]
    drop_frame: FrameTransformer = env.scene[drawer_drop_frame_cfg.name]
    cube_pos = cube.data.root_pos_w[:, :3]
    drop_pos = drop_frame.data.target_pos_w[..., 0, :]
    d = torch.norm(cube_pos - drop_pos, dim=-1)
    base = 1.0 - torch.tanh(d / std)
    high_enough = (cube_pos[:, 2] > minimal_height_b).float()
    return high_enough * base * (1.0 - _cube_inside_latch_active(env))


def ee_retract_to_front_face(env, std=0.15,
                             ee_frame_cfg=SceneEntityCfg("ee_frame"),
                             drawer_front_face_frame_cfg=SceneEntityCfg("drawer_front_face_frame")):
    """Y-axis-only retract attractor; fires only after latch fires."""
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    front_face: FrameTransformer = env.scene[drawer_front_face_frame_cfg.name]
    ee_y    = ee_frame.data.target_pos_w[..., 0, 1]
    front_y = front_face.data.target_pos_w[..., 0, 1]
    dy = torch.abs(ee_y - front_y)
    return (1.0 - torch.tanh(dy / std)) * _cube_inside_latch_active(env)


# ---- Phase 3 — release latch + geometric helper ----

def _cube_inside_drawer_geometric(env, xy_threshold=0.20, z_rel_floor=-0.02, z_rel_ceiling=0.07):
    cube: RigidObject = env.scene["cube_0"]; drawer: Articulation = env.scene["drawer"]
    body_idx = drawer.find_bodies("drawer")[0][0]
    cube_pos = cube.data.root_pos_w[:, :3]
    drawer_pos = drawer.data.body_pos_w[:, body_idx, :]
    rel = cube_pos - drawer_pos
    xy_in = torch.norm(rel[:, :2], dim=-1) < xy_threshold
    z_in = (rel[:, 2] > z_rel_floor) & (rel[:, 2] < z_rel_ceiling)
    return xy_in & z_in


def cube_inside_bonus_once_per_episode(env, xy_threshold=0.20, z_rel_floor=-0.02,
                                       z_rel_ceiling=0.07, ee_cube_no_contact_threshold=0.10):
    """+1.0 the FIRST frame cube_inside_geometric AND ee_far_from_cube. Per-env latch."""
    latch = _get_latch_buffer(env, "cube_inside_once")
    just_reset = env.episode_length_buf <= 1
    latch = torch.where(just_reset, torch.zeros_like(latch), latch)
    inside = _cube_inside_drawer_geometric(env, xy_threshold, z_rel_floor, z_rel_ceiling)
    cube_pos = env.scene["cube_0"].data.root_pos_w[:, :3]
    ee_pos = env.scene["ee_frame"].data.target_pos_w[..., 0, :]
    ee_far = torch.norm(ee_pos - cube_pos, dim=-1) > ee_cube_no_contact_threshold
    now_qualifying = inside & ee_far
    fire = now_qualifying & (~latch)
    latch = latch | now_qualifying
    _LATCH_BUFFERS[(id(env), "cube_inside_once")] = latch
    return fire.float()


# ---- Phase 4 — close drawer ----

def close_drawer(env, max_open=0.30, alpha=0.5,
                 drawer_cfg=SceneEntityCfg("drawer"),
                 joint_name="base_drawer_joint",
                 drawer_front_face_frame_cfg=SceneEntityCfg("drawer_front_face_frame")):
    """latch_active * (ee.y > front_face.y) * closeness ** alpha."""
    drawer: Articulation = env.scene[drawer_cfg.name]
    joint_idx = drawer.find_joints(joint_name)[0][0]
    joint_pos = drawer.data.joint_pos[:, joint_idx]
    closeness = torch.clamp((max_open - joint_pos) / max(max_open, 1e-6), min=0.0, max=1.0)
    shaped = closeness.pow(alpha)
    gate_cube = _cube_inside_latch_active(env)
    ee_frame: FrameTransformer = env.scene["ee_frame"]
    front_face: FrameTransformer = env.scene[drawer_front_face_frame_cfg.name]
    ee_y    = ee_frame.data.target_pos_w[..., 0, 1]
    front_y = front_face.data.target_pos_w[..., 0, 1]
    gate_ee_outside = (ee_y > front_y).float()
    return gate_cube * gate_ee_outside * shaped


# ---- Phase 5 — success bonus ----

def success_bonus(env, drawer_closed_threshold=0.05,
                  drawer_cfg=SceneEntityCfg("drawer"), joint_name="base_drawer_joint"):
    """latch_active * (joint_pos < drawer_closed_threshold). Paired with success DoneTerm."""
    drawer: Articulation = env.scene[drawer_cfg.name]
    joint_idx = drawer.find_joints(joint_name)[0][0]
    joint_pos = drawer.data.joint_pos[:, joint_idx]
    drawer_closed = (joint_pos < drawer_closed_threshold).float()
    return _cube_inside_latch_active(env) * drawer_closed
```

### Smoke

```bash
.venv/bin/python -c "
env = make_env('IsaacLab-Insert-Drawer')
obs, _ = env.reset()
for _ in range(5):
    obs, rew, *_ = env.step(torch.zeros((env.num_envs, 4), device=env.device))
    assert torch.isfinite(rew).all()
# Optional: verify composer (sum) by reading info['detailed_reward'] if /add-reward-log is wired.
print('reward smoke OK')
"
```

---

## §7 DR

### Description

`<no DR>` — the abstract `EventCfg` defines only `mode='reset'` terms (covered in §3); no `mode='startup'` or `mode='interval'` randomization is wired. The dr-generator agent should leave §7 empty when reproducing.

### Decisions resolved

| Knob | Value |
|---|---|
| Domain randomization terms | none |
| Startup randomization | none |
| Interval randomization | none |

### Code

```python
# No additional EventTerms beyond the four mode="reset" entries documented in §3.
```

### Smoke

```bash
# DR smoke: seed-matched obs trajectories with DR ON vs OFF must diverge.
# Skipped — no DR to test.
```

---

## Reproduce

```bash
/harbor:task-creation name=<NewTaskID> from=harbor/task-creation/isaaclab-insert-drawer-implementation.md
```
