# Isaac-Repose-Cube-Allegro-v0 — Implementation Spec

- robot: Allegro hand (16 DoF, fixed in air, no arm)
- simulator: IsaacLab (Isaac Sim, manager-based)
- objects: cube, goal-pose cube marker
- bimanual: false
- summary: Rotate a cube in-hand to match a commanded goal orientation.

This is the **in-hand cube reorientation** task: a *fixed* Allegro hand (16-DoF, gravity disabled on the hand bodies) must reorient a free-floating cube resting in its palm to a commanded goal orientation. There is no arm and no base — the hand never translates. The goal is an orientation-only command (constant position, sampled quaternion); on each success the goal is immediately resampled, so a single episode chains many consecutive reorientations. The task has rich domain randomization (friction, mass, actuator gains) and Gaussian observation noise.

---

## §1 Registration + Scene

### Description
Registers `Isaac-Repose-Cube-Allegro-v0` (and `-Play-v0`, `-NoVelObs-v0`, `-NoVelObs-Play-v0`) against `ManagerBasedRLEnv`. The env cfg `AllegroCubeEnvCfg` extends the abstract `InHandObjectEnvCfg` and swaps in `ALLEGRO_HAND_CFG` as the robot. The scene contains: the Allegro hand articulation, one rigid DexCube object floating in the palm, a distant light, and a dome light. There is no table and no ground plane — the hand is fixed in the air and the cube floats at the palm.

### Decisions resolved
- `entry_point = isaaclab.envs:ManagerBasedRLEnv`, `disable_env_checker=True`.
- `env_cfg_entry_point = ...allegro_hand.allegro_env_cfg:AllegroCubeEnvCfg`.
- `scene.num_envs = 8192`, `env_spacing = 0.6`. (`-Play-v0` overrides to 50.)
- `scene.clone_in_fabric = True` (set in `AllegroCubeEnvCfg.__post_init__`).
- Robot = `ALLEGRO_HAND_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")`. 16 joints, implicit actuators, **gravity disabled on hand**, fixed in air at pos `(0,0,0.5)`, rot `(0.257551, 0.283045, 0.683330, -0.621782)` (palm facing up). Init joint pos: all `0.0` except `thumb_joint_0 = 0.28`.
  - Actuators (`fingers`, joints `.*`): `effort_limit_sim=0.5`, `stiffness=3.0`, `damping=0.1`, `friction=0.01`.
  - `soft_joint_pos_limit_factor=1.0`.
- Object = free rigid DexCube at `pos=(0.0, -0.19, 0.56)`, `rot=(1,0,0,0)`, `density=400.0`, gravity ENABLED, 8 position solver iterations.
- Sim: `dt = 1/120`, `decimation = 4` (control @ 30 Hz), `episode_length_s = 20.0`, `render_interval = decimation`. PhysX `bounce_threshold_velocity=0.2`, `gpu_max_rigid_contact_count=2**20`, `gpu_max_rigid_patch_count=2**23`. Default scene material: static/dynamic friction = 1.0.
- Viewer eye `(2.0, 2.0, 2.0)`.

### Resolved asset paths
- Robot USD: `{ISAAC_NUCLEUS_DIR}/Robots/WonikRobotics/AllegroHand/allegro_hand_instanceable.usd` where `ISAAC_NUCLEUS_DIR = {NUCLEUS_ASSET_ROOT_DIR}/Isaac` (remote Nucleus / S3 asset root — not on local disk; resolved at runtime).
- Cube USD: `{ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd` (remote Nucleus).
- Goal marker USD: same DexCube USD (used as a visualization-only marker, see §4).

### Code

`config/allegro_hand/__init__.py` (registration):
```python
import gymnasium as gym
from . import agents

gym.register(
    id="Isaac-Repose-Cube-Allegro-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.allegro_env_cfg:AllegroCubeEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:AllegroCubePPORunnerCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)
```

`config/allegro_hand/allegro_env_cfg.py` (robot swap):
```python
from isaaclab.utils import configclass
import isaaclab_tasks.manager_based.manipulation.inhand.inhand_env_cfg as inhand_env_cfg
from isaaclab_assets import ALLEGRO_HAND_CFG  # isort: skip


@configclass
class AllegroCubeEnvCfg(inhand_env_cfg.InHandObjectEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        # switch robot to allegro hand
        self.scene.robot = ALLEGRO_HAND_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        # enable clone in fabric
        self.scene.clone_in_fabric = True
```

`isaaclab_assets/robots/allegro.py` (`ALLEGRO_HAND_CFG`):
```python
import math
import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

ALLEGRO_HAND_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_NUCLEUS_DIR}/Robots/WonikRobotics/AllegroHand/allegro_hand_instanceable.usd",
        activate_contact_sensors=False,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=True,
            retain_accelerations=False,
            enable_gyroscopic_forces=False,
            angular_damping=0.01,
            max_linear_velocity=1000.0,
            max_angular_velocity=64 / math.pi * 180.0,
            max_depenetration_velocity=1000.0,
            max_contact_impulse=1e32,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=0,
            sleep_threshold=0.005,
            stabilization_threshold=0.0005,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.5),
        rot=(0.257551, 0.283045, 0.683330, -0.621782),
        joint_pos={"^(?!thumb_joint_0).*": 0.0, "thumb_joint_0": 0.28},
    ),
    actuators={
        "fingers": ImplicitActuatorCfg(
            joint_names_expr=[".*"],
            effort_limit_sim=0.5,
            stiffness=3.0,
            damping=0.1,
            friction=0.01,
        ),
    },
    soft_joint_pos_limit_factor=1.0,
)
```

`inhand_env_cfg.py` — `InHandObjectSceneCfg` + EnvCfg assembly:
```python
@configclass
class InHandObjectSceneCfg(InteractiveSceneCfg):
    """Configuration for a scene with an object and a dexterous hand."""

    # robots
    robot: ArticulationCfg = MISSING

    # objects
    object: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/object",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=False,
                disable_gravity=False,
                enable_gyroscopic_forces=True,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=0,
                sleep_threshold=0.005,
                stabilization_threshold=0.0025,
                max_depenetration_velocity=1000.0,
            ),
            mass_props=sim_utils.MassPropertiesCfg(density=400.0),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, -0.19, 0.56), rot=(1.0, 0.0, 0.0, 0.0)),
    )

    # lights
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DistantLightCfg(color=(0.95, 0.95, 0.95), intensity=1000.0),
    )
    dome_light = AssetBaseCfg(
        prim_path="/World/domeLight",
        spawn=sim_utils.DomeLightCfg(color=(0.02, 0.02, 0.02), intensity=1000.0),
    )


@configclass
class InHandObjectEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the in hand reorientation environment."""

    scene: InHandObjectSceneCfg = InHandObjectSceneCfg(num_envs=8192, env_spacing=0.6)
    sim: SimulationCfg = SimulationCfg(
        physics_material=RigidBodyMaterialCfg(static_friction=1.0, dynamic_friction=1.0),
        physx=PhysxCfg(
            bounce_threshold_velocity=0.2,
            gpu_max_rigid_contact_count=2**20,
            gpu_max_rigid_patch_count=2**23,
        ),
    )
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()

    def __post_init__(self):
        self.decimation = 4
        self.episode_length_s = 20.0
        self.sim.dt = 1.0 / 120.0
        self.sim.render_interval = self.decimation
        self.viewer.eye = (2.0, 2.0, 2.0)
```

### Smoke
```bash
cd <repo>
# IsaacLab manager-based envs require a parsed cfg; gym.make(id) alone raises (missing cfg).
.venv/bin/python -c "
from isaaclab.app import AppLauncher
app = AppLauncher(headless=True).app
import gymnasium as gym, isaaclab_tasks
from isaaclab_tasks.utils import parse_env_cfg
cfg = parse_env_cfg('Isaac-Repose-Cube-Allegro-v0', num_envs=2)
env = gym.make('Isaac-Repose-Cube-Allegro-v0', cfg=cfg)
print(env.observation_space, env.action_space); env.close(); app.close()
"
```
Expected (analytically resolved — not captured live, Isaac boot best-effort skipped per probe budget): `Box(..., (72,), float32) Box(-1.0, 1.0, (16,), float32)` (note: rsl_rl wrappers commonly clip action_space to [-1,1]; the raw Gym space is unbounded float32 of dim 16).

---

## §2 Actions

### Description
A single action term over all 16 hand joints using **EMA joint-position-to-limits** control. The raw policy action (per joint, expected in `[-1, 1]`) is rescaled to that joint's soft position limits, then an exponential moving average is applied against the previously-applied target with weight `alpha=0.95`. This heavily smooths the joint targets (95% new / 5% previous each control step), preventing jittery finger motion. There is no separate gripper action — the whole hand is one articulation.

### Decisions resolved
- `joint_pos = mdp.EMAJointPositionToLimitsActionCfg(asset_name="robot", joint_names=[".*"], alpha=0.95, rescale_to_limits=True)`.
- `action_dim = 16` (all Allegro joints).
- `scale = 1.0` (default), `clip = None` (default), `preserve_order = False`.
- Applied action = `alpha * processed + (1-alpha) * prev_applied`, clamped to soft joint pos limits. On reset, `prev_applied` is initialized to the current joint positions.

### Code

`ActionsCfg` (`inhand_env_cfg.py`):
```python
@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    joint_pos = mdp.EMAJointPositionToLimitsActionCfg(
        asset_name="robot",
        joint_names=[".*"],
        alpha=0.95,
        rescale_to_limits=True,
    )
```

Action term cfg (`isaaclab/envs/mdp/actions/actions_cfg.py`):
```python
@configclass
class JointPositionToLimitsActionCfg(ActionTermCfg):
    class_type: type[ActionTerm] = joint_actions_to_limits.JointPositionToLimitsAction
    joint_names: list[str] = MISSING
    scale: float | dict[str, float] = 1.0
    rescale_to_limits: bool = True
    preserve_order: bool = False


@configclass
class EMAJointPositionToLimitsActionCfg(JointPositionToLimitsActionCfg):
    class_type: type[ActionTerm] = joint_actions_to_limits.EMAJointPositionToLimitsAction
    alpha: float | dict[str, float] = 1.0
```

Action term impl (`isaaclab/envs/mdp/actions/joint_actions_to_limits.py`) — the two relevant classes:
```python
class JointPositionToLimitsAction(ActionTerm):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self._joint_ids, self._joint_names = self._asset.find_joints(
            self.cfg.joint_names, preserve_order=cfg.preserve_order
        )
        self._num_joints = len(self._joint_ids)
        if self._num_joints == self._asset.num_joints and not cfg.preserve_order:
            self._joint_ids = slice(None)
        self._raw_actions = torch.zeros(self.num_envs, self.action_dim, device=self.device)
        self._processed_actions = torch.zeros_like(self.raw_actions)
        if isinstance(cfg.scale, (float, int)):
            self._scale = float(cfg.scale)
        elif isinstance(cfg.scale, dict):
            self._scale = torch.ones(self.num_envs, self.action_dim, device=self.device)
            index_list, _, value_list = string_utils.resolve_matching_names_values(
                self.cfg.scale, self._joint_names, preserve_order=cfg.preserve_order
            )
            self._scale[:, index_list] = torch.tensor(value_list, device=self.device)
        else:
            raise ValueError(f"Unsupported scale type: {type(cfg.scale)}. Supported types are float and dict.")
        if self.cfg.clip is not None:
            if isinstance(cfg.clip, dict):
                self._clip = torch.tensor([[-float("inf"), float("inf")]], device=self.device).repeat(
                    self.num_envs, self.action_dim, 1
                )
                index_list, _, value_list = string_utils.resolve_matching_names_values(
                    self.cfg.clip, self._joint_names, preserve_order=cfg.preserve_order
                )
                self._clip[:, index_list] = torch.tensor(value_list, device=self.device)
            else:
                raise ValueError(f"Unsupported clip type: {type(cfg.clip)}. Supported types are dict.")

    @property
    def action_dim(self) -> int:
        return self._num_joints

    def process_actions(self, actions: torch.Tensor):
        self._raw_actions[:] = actions
        self._processed_actions = self._raw_actions * self._scale
        if self.cfg.clip is not None:
            self._processed_actions = torch.clamp(
                self._processed_actions, min=self._clip[:, :, 0], max=self._clip[:, :, 1]
            )
        if self.cfg.rescale_to_limits:
            actions = self._processed_actions.clamp(-1.0, 1.0)
            actions = math_utils.unscale_transform(
                actions,
                self._asset.data.soft_joint_pos_limits[:, self._joint_ids, 0],
                self._asset.data.soft_joint_pos_limits[:, self._joint_ids, 1],
            )
            self._processed_actions[:] = actions[:]

    def apply_actions(self):
        self._asset.set_joint_position_target(self.processed_actions, joint_ids=self._joint_ids)

    def reset(self, env_ids=None) -> None:
        self._raw_actions[env_ids] = 0.0


class EMAJointPositionToLimitsAction(JointPositionToLimitsAction):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        if isinstance(cfg.alpha, float):
            if not 0.0 <= cfg.alpha <= 1.0:
                raise ValueError(f"Moving average weight must be in the range [0, 1]. Got {cfg.alpha}.")
            self._alpha = cfg.alpha
        elif isinstance(cfg.alpha, dict):
            self._alpha = torch.ones((env.num_envs, self.action_dim), device=self.device)
            index_list, names_list, value_list = string_utils.resolve_matching_names_values(
                cfg.alpha, self._joint_names
            )
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
        self._prev_applied_actions = torch.zeros_like(self.processed_actions)

    def reset(self, env_ids=None) -> None:
        if env_ids is None:
            super().reset(slice(None))
            self._prev_applied_actions[:] = self._asset.data.joint_pos[:, self._joint_ids]
        else:
            super().reset(env_ids)
            curr_applied_actions = self._asset.data.joint_pos[env_ids[:, None], self._joint_ids].view(len(env_ids), -1)
            self._prev_applied_actions[env_ids, :] = curr_applied_actions

    def process_actions(self, actions: torch.Tensor):
        super().process_actions(actions)
        ema_actions = self._alpha * self._processed_actions
        ema_actions += (1.0 - self._alpha) * self._prev_applied_actions
        self._processed_actions[:] = torch.clamp(
            ema_actions,
            self._asset.data.soft_joint_pos_limits[:, self._joint_ids, 0],
            self._asset.data.soft_joint_pos_limits[:, self._joint_ids, 1],
        )
        self._prev_applied_actions[:] = self._processed_actions[:]
```

### Smoke (S2)
After build, step random actions in `[-1, 1]` for ~30 steps and confirm joint targets move and stay within limits; with `alpha=0.95` the applied-target trajectory should be visibly smoother than raw. No nominal stdout — assert finite obs and no exception.

---

## §3 Reset

### Description
On every episode reset two reset events fire: the cube root pose is perturbed by a tiny uniform box around its default position (±1 cm on each axis, no orientation/velocity perturbation), and the hand joints are reset to a scaled band around their default positions (`reset_joints_within_limits_range`, scale `[0.2, 0.2]` of the soft limits offset by the default joint pose — i.e. deterministically biased toward the default pose with `use_default_offset=True`). The orientation goal is sampled at reset by the command term (see §4), not by an EventCfg term.

### Decisions resolved
- `reset_object`: `reset_root_state_uniform`, `pose_range = {x:[-0.01,0.01], y:[-0.01,0.01], z:[-0.01,0.01]}` (position only; no roll/pitch/yaw keys → orientation unchanged), `velocity_range = {}` (zeroed), `asset_cfg = object` (all bodies).
- `reset_robot_joints`: `reset_joints_within_limits_range`, `position_range = {".*": [0.2, 0.2]}`, `velocity_range = {".*": [0.0, 0.0]}`, `use_default_offset = True`, `operation = "scale"`. With `operation="scale"` + `use_default_offset=True`, each joint's sampled range is `[0.2*lo, 0.2*hi] + default_joint_pos`, then clamped to soft limits; since lo==hi==0.2 the band collapses to a single offset value, so joints reset deterministically to `0.2*limit + default` (clamped).

### Code

`EventCfg` reset terms (`inhand_env_cfg.py`):
```python
    # reset
    reset_object = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": [-0.01, 0.01], "y": [-0.01, 0.01], "z": [-0.01, 0.01]},
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("object", body_names=".*"),
        },
    )
    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_within_limits_range,
        mode="reset",
        params={
            "position_range": {".*": [0.2, 0.2]},
            "velocity_range": {".*": [0.0, 0.0]},
            "use_default_offset": True,
            "operation": "scale",
        },
    )
```

`reset_joints_within_limits_range` (task-local, `inhand/mdp/events.py`) — full source:
```python
class reset_joints_within_limits_range(ManagerTermBase):
    """Reset an articulation's joints to a random position in the given limit ranges. ..."""

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        if "position_range" not in cfg.params or "velocity_range" not in cfg.params:
            raise ValueError(
                "The term 'reset_joints_within_range' requires parameters: 'position_range' and 'velocity_range'."
                f" Received: {list(cfg.params.keys())}."
            )
        asset_cfg: SceneEntityCfg = cfg.params.get("asset_cfg", SceneEntityCfg("robot"))
        use_default_offset = cfg.params.get("use_default_offset", False)
        operation = cfg.params.get("operation", "abs")
        if operation not in ["abs", "scale"]:
            raise ValueError(
                f"For event 'reset_joints_within_limits_range', unknown operation: '{operation}'."
                " Please use 'abs' or 'scale'."
            )
        self._asset: Articulation = env.scene[asset_cfg.name]
        default_joint_pos = self._asset.data.default_joint_pos[0]
        default_joint_vel = self._asset.data.default_joint_vel[0]
        self._pos_ranges = self._asset.data.soft_joint_pos_limits[0].clone()
        pos_joint_ids = []
        for joint_name, joint_range in cfg.params["position_range"].items():
            joint_ids = self._asset.find_joints(joint_name)[0]
            pos_joint_ids.extend(joint_ids)
            if operation == "abs":
                if joint_range[0] is not None:
                    self._pos_ranges[joint_ids, 0] = joint_range[0]
                if joint_range[1] is not None:
                    self._pos_ranges[joint_ids, 1] = joint_range[1]
            elif operation == "scale":
                if joint_range[0] is not None:
                    self._pos_ranges[joint_ids, 0] *= joint_range[0]
                if joint_range[1] is not None:
                    self._pos_ranges[joint_ids, 1] *= joint_range[1]
            else:
                raise ValueError(
                    f"Unknown operation: '{operation}' for joint position ranges. Please use 'abs' or 'scale'."
                )
            if use_default_offset:
                self._pos_ranges[joint_ids] += default_joint_pos[joint_ids].unsqueeze(1)
        self._pos_joint_ids = torch.tensor(pos_joint_ids, device=self._pos_ranges.device)
        self._pos_ranges = self._pos_ranges[self._pos_joint_ids]
        self._vel_ranges = torch.stack(
            [-self._asset.data.soft_joint_vel_limits[0], self._asset.data.soft_joint_vel_limits[0]], dim=1
        )
        vel_joint_ids = []
        for joint_name, joint_range in cfg.params["velocity_range"].items():
            joint_ids = self._asset.find_joints(joint_name)[0]
            vel_joint_ids.extend(joint_ids)
            if operation == "abs":
                if joint_range[0] is not None:
                    self._vel_ranges[joint_ids, 0] = joint_range[0]
                if joint_range[1] is not None:
                    self._vel_ranges[joint_ids, 1] = joint_range[1]
            elif operation == "scale":
                if joint_range[0] is not None:
                    self._vel_ranges[joint_ids, 0] = joint_range[0] * self._vel_ranges[joint_ids, 0]
                if joint_range[1] is not None:
                    self._vel_ranges[joint_ids, 1] = joint_range[1] * self._vel_ranges[joint_ids, 1]
            else:
                raise ValueError(
                    f"Unknown operation: '{operation}' for joint velocity ranges. Please use 'abs' or 'scale'."
                )
            if use_default_offset:
                self._vel_ranges[joint_ids] += default_joint_vel[joint_ids].unsqueeze(1)
        self._vel_joint_ids = torch.tensor(vel_joint_ids, device=self._vel_ranges.device)
        self._vel_ranges = self._vel_ranges[self._vel_joint_ids]

    def __call__(
        self,
        env: ManagerBasedEnv,
        env_ids: torch.Tensor,
        position_range: dict,
        velocity_range: dict,
        use_default_offset: bool = False,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
        operation: Literal["abs", "scale"] = "abs",
    ):
        joint_pos = self._asset.data.default_joint_pos[env_ids].clone()
        joint_vel = self._asset.data.default_joint_vel[env_ids].clone()
        if len(self._pos_joint_ids) > 0:
            joint_pos_shape = (len(env_ids), len(self._pos_joint_ids))
            joint_pos[:, self._pos_joint_ids] = sample_uniform(
                self._pos_ranges[:, 0], self._pos_ranges[:, 1], joint_pos_shape, device=joint_pos.device
            )
            joint_pos_limits = self._asset.data.soft_joint_pos_limits[0, self._pos_joint_ids]
            joint_pos = joint_pos.clamp(joint_pos_limits[:, 0], joint_pos_limits[:, 1])
        if len(self._vel_joint_ids) > 0:
            joint_vel_shape = (len(env_ids), len(self._vel_joint_ids))
            joint_vel[:, self._vel_joint_ids] = sample_uniform(
                self._vel_ranges[:, 0], self._vel_ranges[:, 1], joint_vel_shape, device=joint_vel.device
            )
            joint_vel_limits = self._asset.data.soft_joint_vel_limits[0, self._vel_joint_ids]
            joint_vel = joint_vel.clamp(-joint_vel_limits, joint_vel_limits)
        self._asset.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
```

`reset_root_state_uniform` is the stock isaaclab function: `isaaclab/envs/mdp/events.py:reset_root_state_uniform`.

### Smoke (S3)
Reset twice with different seeds; cube root pos should differ by ≤1 cm per axis from default and hand joints reset near `0.2*limit + default`. Assert all reset obs finite.

---

## §4 Goal + Termination

### Description
The goal is a **3D orientation command** produced by a custom command term `InHandReOrientationCommand`. Position is held constant at the cube's default position + `init_pos_offset`; orientation is sampled uniformly via two random rotations (about x then y axis, each angle in `[-π, π]`). The goal is **resampled the instant the object reaches it** (`update_goal_on_success=True`, threshold 0.1 rad), so one episode chains consecutive reorientations. Termination: time-out at 20 s, **success-chain done** when 50 consecutive successes are reached (`max_consecutive_success`), and **failure done** when the object drifts >0.3 m from the hand (`object_away_from_robot`).

### Decisions resolved
- Command term `object_pose = InHandReOrientationCommandCfg(asset_name="object", init_pos_offset=(0,0,-0.04), update_goal_on_success=True, orientation_success_threshold=0.1, make_quat_unique=False, marker_pos_offset=(-0.2,-0.06,0.08), debug_vis=True)`.
- `resampling_time_range = (1e6, 1e6)` → no time-based resampling; goal only changes on success.
- Command shape = 7 (pos `e`-frame xyz + goal quat wxyz). Position command is constant (default root pos + offset); only orientation is sampled.
- Orientation sampling: `quat = quat_mul(quat_from_angle_axis(u0*π, X), quat_from_angle_axis(u1*π, Y))` with `u0,u1 ~ U(-1,1)`.
- Terminations:
  - `time_out = DoneTerm(time_out, time_out=True)` (truncation).
  - `max_consecutive_success = DoneTerm(num_success=50, command_name="object_pose")` (success-chain termination).
  - `object_out_of_reach = DoneTerm(object_away_from_robot, threshold=0.3)` (failure).

### Code

`CommandsCfg` + `TerminationsCfg` (`inhand_env_cfg.py`):
```python
@configclass
class CommandsCfg:
    object_pose = mdp.InHandReOrientationCommandCfg(
        asset_name="object",
        init_pos_offset=(0.0, 0.0, -0.04),
        update_goal_on_success=True,
        orientation_success_threshold=0.1,
        make_quat_unique=False,
        marker_pos_offset=(-0.2, -0.06, 0.08),
        debug_vis=True,
    )


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    max_consecutive_success = DoneTerm(
        func=mdp.max_consecutive_success, params={"num_success": 50, "command_name": "object_pose"}
    )
    object_out_of_reach = DoneTerm(func=mdp.object_away_from_robot, params={"threshold": 0.3})
```

`InHandReOrientationCommandCfg` (`inhand/mdp/commands/commands_cfg.py`):
```python
@configclass
class InHandReOrientationCommandCfg(CommandTermCfg):
    class_type: type = InHandReOrientationCommand
    resampling_time_range: tuple[float, float] = (1e6, 1e6)  # no resampling based on time
    asset_name: str = MISSING
    init_pos_offset: tuple[float, float, float] = (0.0, 0.0, 0.0)
    make_quat_unique: bool = MISSING
    orientation_success_threshold: float = MISSING
    update_goal_on_success: bool = MISSING
    marker_pos_offset: tuple[float, float, float] = (0.0, 0.0, 0.0)
    goal_pose_visualizer_cfg: VisualizationMarkersCfg = VisualizationMarkersCfg(
        prim_path="/Visuals/Command/goal_marker",
        markers={
            "goal": sim_utils.UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd",
                scale=(1.0, 1.0, 1.0),
            ),
        },
    )
```

`InHandReOrientationCommand` (`inhand/mdp/commands/orientation_command.py`) — full source:
```python
class InHandReOrientationCommand(CommandTerm):
    """Command term that generates 3D pose commands for in-hand manipulation task. ..."""

    cfg: InHandReOrientationCommandCfg

    def __init__(self, cfg: InHandReOrientationCommandCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.object: RigidObject = env.scene[cfg.asset_name]
        init_pos_offset = torch.tensor(cfg.init_pos_offset, dtype=torch.float, device=self.device)
        self.pos_command_e = self.object.data.default_root_state[:, :3] + init_pos_offset
        self.pos_command_w = self.pos_command_e + self._env.scene.env_origins
        self.quat_command_w = torch.zeros(self.num_envs, 4, device=self.device)
        self.quat_command_w[:, 0] = 1.0  # set the scalar component to 1.0
        self._X_UNIT_VEC = torch.tensor([1.0, 0, 0], device=self.device).repeat((self.num_envs, 1))
        self._Y_UNIT_VEC = torch.tensor([0, 1.0, 0], device=self.device).repeat((self.num_envs, 1))
        self._Z_UNIT_VEC = torch.tensor([0, 0, 1.0], device=self.device).repeat((self.num_envs, 1))
        self.metrics["orientation_error"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["position_error"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["consecutive_success"] = torch.zeros(self.num_envs, device=self.device)

    @property
    def command(self) -> torch.Tensor:
        """The desired goal pose in the environment frame. Shape is (num_envs, 7)."""
        return torch.cat((self.pos_command_e, self.quat_command_w), dim=-1)

    def _update_metrics(self):
        self.metrics["orientation_error"] = math_utils.quat_error_magnitude(
            self.object.data.root_quat_w, self.quat_command_w
        )
        self.metrics["position_error"] = torch.norm(self.object.data.root_pos_w - self.pos_command_w, dim=1)
        successes = self.metrics["orientation_error"] < self.cfg.orientation_success_threshold
        self.metrics["consecutive_success"] += successes.float()

    def _resample_command(self, env_ids: Sequence[int]):
        rand_floats = 2.0 * torch.rand((len(env_ids), 2), device=self.device) - 1.0
        quat = math_utils.quat_mul(
            math_utils.quat_from_angle_axis(rand_floats[:, 0] * torch.pi, self._X_UNIT_VEC[env_ids]),
            math_utils.quat_from_angle_axis(rand_floats[:, 1] * torch.pi, self._Y_UNIT_VEC[env_ids]),
        )
        self.quat_command_w[env_ids] = math_utils.quat_unique(quat) if self.cfg.make_quat_unique else quat

    def _update_command(self):
        if self.cfg.update_goal_on_success:
            goal_resets = self.metrics["orientation_error"] < self.cfg.orientation_success_threshold
            goal_reset_ids = goal_resets.nonzero(as_tuple=False).squeeze(-1)
            self._resample(goal_reset_ids)

    def _set_debug_vis_impl(self, debug_vis):
        if debug_vis:
            if not hasattr(self, "goal_pose_visualizer"):
                self.goal_pose_visualizer = VisualizationMarkers(self.cfg.goal_pose_visualizer_cfg)
            self.goal_pose_visualizer.set_visibility(True)
        else:
            if hasattr(self, "goal_pose_visualizer"):
                self.goal_pose_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        marker_pos = self.pos_command_w + torch.tensor(self.cfg.marker_pos_offset, device=self.device)
        marker_quat = self.quat_command_w
        self.goal_pose_visualizer.visualize(translations=marker_pos, orientations=marker_quat)
```

Termination funcs (task-local, `inhand/mdp/terminations.py`) — full source:
```python
def max_consecutive_success(env: ManagerBasedRLEnv, num_success: int, command_name: str) -> torch.Tensor:
    command_term = env.command_manager.get_term(command_name)
    return command_term.metrics["consecutive_success"] >= num_success


def object_away_from_goal(
    env: ManagerBasedRLEnv, threshold: float, command_name: str,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    command_term = env.command_manager.get_term(command_name)
    asset = env.scene[object_cfg.name]
    asset_pos_e = asset.data.root_pos_w - env.scene.env_origins
    goal_pos_e = command_term.command[:, :3]
    return torch.norm(asset_pos_e - goal_pos_e, p=2, dim=1) > threshold


def object_away_from_robot(
    env: ManagerBasedRLEnv, threshold: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    robot = env.scene[asset_cfg.name]
    object = env.scene[object_cfg.name]
    dist = torch.norm(robot.data.root_pos_w - object.data.root_pos_w, dim=1)
    return dist > threshold
```
`time_out` is the stock `isaaclab/envs/mdp/terminations.py:time_out`.

### Smoke (S4)
Roll out a few hundred steps; assert `dones` eventually fire (time_out at 20 s = 600 control steps; or object-away if it falls). Confirm `command_manager.get_term("object_pose").command.shape[-1] == 7` and the goal quat changes after a simulated success.

---

## §5 Observation

### Description
One policy group `KinematicObsGroupCfg` with full-kinematic state, corruption (Gaussian noise) enabled, terms concatenated. Order is fixed. Robot terms (limit-normalized joint pos, relative joint vel), object terms (world pos/quat/lin-vel/ang-vel), command terms (the 7-D goal pose and the goal-vs-object quaternion difference), and the last action. The `-NoVelObs-v0` variant drops `joint_vel`, `object_lin_vel`, `object_ang_vel`.

### Decisions resolved
- `enable_corruption = True`, `concatenate_terms = True`.
- Per-term noise (Additive Gaussian std): joint_pos 0.005, joint_vel 0.01 (scale 0.2), object_pos 0.002, object_lin_vel 0.002, object_ang_vel 0.002 (scale 0.2). `object_quat`, `goal_pose`, `goal_quat_diff`, `last_action` have **no** noise.
- Resolved total obs dim = **72** (full group): joint_pos 16 + joint_vel 16 + object_pos 3 + object_quat 4 + object_lin_vel 3 + object_ang_vel 3 + goal_pose 7 + goal_quat_diff 4 + last_action 16 = 72.
- NoVelObs dim = 72 − 16 (joint_vel) − 3 (object_lin_vel) − 3 (object_ang_vel) = 50.

### Code

`ObservationsCfg` (`inhand_env_cfg.py`):
```python
@configclass
class ObservationsCfg:
    @configclass
    class KinematicObsGroupCfg(ObsGroup):
        # -- robot terms
        joint_pos = ObsTerm(func=mdp.joint_pos_limit_normalized, noise=Gnoise(std=0.005))
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, scale=0.2, noise=Gnoise(std=0.01))
        # -- object terms
        object_pos = ObsTerm(
            func=mdp.root_pos_w, noise=Gnoise(std=0.002), params={"asset_cfg": SceneEntityCfg("object")}
        )
        object_quat = ObsTerm(
            func=mdp.root_quat_w, params={"asset_cfg": SceneEntityCfg("object"), "make_quat_unique": False}
        )
        object_lin_vel = ObsTerm(
            func=mdp.root_lin_vel_w, noise=Gnoise(std=0.002), params={"asset_cfg": SceneEntityCfg("object")}
        )
        object_ang_vel = ObsTerm(
            func=mdp.root_ang_vel_w, scale=0.2, noise=Gnoise(std=0.002),
            params={"asset_cfg": SceneEntityCfg("object")},
        )
        # -- command terms
        goal_pose = ObsTerm(func=mdp.generated_commands, params={"command_name": "object_pose"})
        goal_quat_diff = ObsTerm(
            func=mdp.goal_quat_diff,
            params={"asset_cfg": SceneEntityCfg("object"), "command_name": "object_pose", "make_quat_unique": False},
        )
        # -- action terms
        last_action = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class NoVelocityKinematicObsGroupCfg(KinematicObsGroupCfg):
        def __post_init__(self):
            super().__post_init__()
            self.joint_vel = None
            self.object_lin_vel = None
            self.object_ang_vel = None

    policy: KinematicObsGroupCfg = KinematicObsGroupCfg()
```

`goal_quat_diff` (task-local, `inhand/mdp/observations.py`) — full source:
```python
def goal_quat_diff(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, command_name: str, make_quat_unique: bool
) -> torch.Tensor:
    """Goal orientation relative to the asset's root frame. (w, x, y, z); real part always positive."""
    asset: RigidObject = env.scene[asset_cfg.name]
    command_term = env.command_manager.get_term(command_name)
    goal_quat_w = command_term.command[:, 3:7]
    asset_quat_w = asset.data.root_quat_w
    quat = math_utils.quat_mul(asset_quat_w, math_utils.quat_conjugate(goal_quat_w))
    return math_utils.quat_unique(quat) if make_quat_unique else quat
```
All other obs funcs (`joint_pos_limit_normalized`, `joint_vel_rel`, `root_pos_w`, `root_quat_w`, `root_lin_vel_w`, `root_ang_vel_w`, `generated_commands`, `last_action`) are stock `isaaclab/envs/mdp/observations.py`.

### Smoke (S5)
Build full env, assert `env.observation_space["policy"].shape == (72,)` (or the flattened obs has 72 dims); build NoVelObs variant and assert 50. Assert obs finite and that corruption changes obs across identical states when `enable_corruption=True`.

---

## §6 Reward

### Description
Composer = **sum** (IsaacLab `RewardManager` sums weighted terms; each weight is additionally multiplied by `dt`≈ control step — see plan note). Two positive task terms — a dense inverse-orientation-error tracking reward (weight +1.0) and a sparse per-step success bonus (+250.0 when within the 0.1 rad threshold) — and three small penalties on joint velocity, action magnitude, and action rate. The optional `track_pos_l2` and `object_away_penalty` terms are commented out in source (disabled).

### Decisions resolved
- Active terms (5):
  - `track_orientation_inv_l2`: weight `+1.0`, params `{object_cfg=object, rot_eps=0.1, command_name=object_pose}`.
  - `success_bonus`: weight `+250.0`, params `{object_cfg=object, command_name=object_pose}` (returns bool→{0,1}).
  - `joint_vel_l2`: weight `-2.5e-5`.
  - `action_l2`: weight `-0.0001`.
  - `action_rate_l2`: weight `-0.01`.
- Disabled (commented out in source): `track_pos_l2` (weight −10.0), `object_away_penalty` (`is_terminated_term`, weight 0.0).
- Composer = sum of `weight * term` (RewardManager).
- **Planning-budget (retro-computed, per-step nominal magnitudes):**
  - `track_orientation_inv_l2`: `1/(dtheta+0.1)`, so per-step ∈ ≈[`1/(π+0.1)`≈0.31, `1/0.1`=10.0]. At success (dtheta≈0.1) ≈ +5.0; far ≈ +0.31.
  - `success_bonus`: +250 per step while inside threshold. Dominant signal that pins the policy at the goal.
  - `joint_vel_l2` (−2.5e-5): with 16 joints and modest vel, magnitude ≈ −1e-3..−1e-2; negligible.
  - `action_l2` (−1e-4): with 16-D action of O(1), ≈ −1.6e-3; negligible.
  - `action_rate_l2` (−1e-2): main smoothness penalty; with EMA-smoothed targets, ≈ −1e-3..−1e-1.
  - Net: tracking + success_bonus dominate; penalties are tie-breakers favoring smooth, slow finger motion.

### Code

`RewardsCfg` (`inhand_env_cfg.py`) — including the commented-out (disabled) terms verbatim:
```python
@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # -- task
    # track_pos_l2 = RewTerm(
    #     func=mdp.track_pos_l2,
    #     weight=-10.0,
    #     params={"object_cfg": SceneEntityCfg("object"), "command_name": "object_pose"},
    # )
    track_orientation_inv_l2 = RewTerm(
        func=mdp.track_orientation_inv_l2,
        weight=1.0,
        params={"object_cfg": SceneEntityCfg("object"), "rot_eps": 0.1, "command_name": "object_pose"},
    )
    success_bonus = RewTerm(
        func=mdp.success_bonus,
        weight=250.0,
        params={"object_cfg": SceneEntityCfg("object"), "command_name": "object_pose"},
    )

    # -- penalties
    joint_vel_l2 = RewTerm(func=mdp.joint_vel_l2, weight=-2.5e-5)
    action_l2 = RewTerm(func=mdp.action_l2, weight=-0.0001)
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.01)

    # -- optional penalties (these are disabled by default)
    # object_away_penalty = RewTerm(
    #     func=mdp.is_terminated_term,
    #     weight=-0.0,
    #     params={"term_keys": "object_out_of_reach"},
    # )
```

Task-local reward funcs (`inhand/mdp/rewards.py`) — full source (all three; `track_pos_l2` included for the disabled term):
```python
import isaaclab.utils.math as math_utils
from isaaclab.assets import RigidObject
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg
# if TYPE_CHECKING: from .commands import InHandReOrientationCommand


def success_bonus(
    env: ManagerBasedRLEnv, command_name: str, object_cfg: SceneEntityCfg = SceneEntityCfg("object")
) -> torch.Tensor:
    """Bonus reward for successfully reaching the goal (1.0 if within threshold else 0.0)."""
    asset: RigidObject = env.scene[object_cfg.name]
    command_term = env.command_manager.get_term(command_name)
    goal_quat_w = command_term.command[:, 3:7]
    threshold = command_term.cfg.orientation_success_threshold
    dtheta = math_utils.quat_error_magnitude(asset.data.root_quat_w, goal_quat_w)
    return dtheta <= threshold


def track_pos_l2(
    env: ManagerBasedRLEnv, command_name: str, object_cfg: SceneEntityCfg = SceneEntityCfg("object")
) -> torch.Tensor:
    """Reward for tracking the object position using the L2 norm (distance to goal pos)."""
    asset: RigidObject = env.scene[object_cfg.name]
    command_term = env.command_manager.get_term(command_name)
    goal_pos_e = command_term.command[:, 0:3]
    object_pos_e = asset.data.root_pos_w - env.scene.env_origins
    return torch.norm(goal_pos_e - object_pos_e, p=2, dim=-1)


def track_orientation_inv_l2(
    env: ManagerBasedRLEnv, command_name: str,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"), rot_eps: float = 1e-3,
) -> torch.Tensor:
    """Reward = inverse of the orientation error: 1 / (dtheta + rot_eps)."""
    asset: RigidObject = env.scene[object_cfg.name]
    command_term = env.command_manager.get_term(command_name)
    goal_quat_w = command_term.command[:, 3:7]
    dtheta = math_utils.quat_error_magnitude(asset.data.root_quat_w, goal_quat_w)
    return 1.0 / (dtheta + rot_eps)
```
Penalty funcs (`joint_vel_l2`, `action_l2`, `action_rate_l2`) are stock `isaaclab/envs/mdp/rewards.py`.

### Smoke (S6)
Per the §6 smoke contract: roll out, log per-term reward via the reward manager; assert all per-term values finite, the total reward is non-constant across steps, and `sum(weighted terms) == env reward` each step (composer = "sum").

---

## §7 DR

### Description
Rich startup-mode domain randomization on both the hand and the cube: per-environment material friction, body mass scaling, and (hand only) actuator stiffness/damping gains sampled log-uniformly. No `interval`-mode DR. (Resets, §3, are `mode="reset"` and not counted here.) `-Play-v0` variants additionally set `observations.policy.enable_corruption = False`, disabling §5 obs noise — but the EventCfg DR below still applies in Play.

### Decisions resolved (all `mode="startup"`, applied once per env at scene build)
- `robot_physics_material` (`randomize_rigid_body_material`, robot `.*`): static/dynamic friction ∈ `(0.7, 1.3)`, restitution `(0.0, 0.0)`, `num_buckets=250`.
- `robot_scale_mass` (`randomize_rigid_body_mass`, robot `.*`): mass scale ∈ `(0.95, 1.05)`, `operation="scale"`.
- `robot_joint_stiffness_and_damping` (`randomize_actuator_gains`, robot joints `.*`): stiffness scale ∈ `(0.3, 3.0)`, damping scale ∈ `(0.75, 1.5)`, `operation="scale"`, `distribution="log_uniform"`.
- `object_physics_material` (`randomize_rigid_body_material`, object `.*`): friction `(0.7, 1.3)`, restitution `(0.0, 0.0)`, `num_buckets=250`.
- `object_scale_mass` (`randomize_rigid_body_mass`, object): mass scale ∈ `(0.4, 1.6)`, `operation="scale"` (wide — cube mass varies ~4×).

### Code

`EventCfg` startup terms (`inhand_env_cfg.py`):
```python
@configclass
class EventCfg:
    """Configuration for randomization."""

    # startup
    # -- robot
    robot_physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.7, 1.3),
            "dynamic_friction_range": (0.7, 1.3),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 250,
        },
    )
    robot_scale_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "mass_distribution_params": (0.95, 1.05),
            "operation": "scale",
        },
    )
    robot_joint_stiffness_and_damping = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "stiffness_distribution_params": (0.3, 3.0),  # default: 3.0
            "damping_distribution_params": (0.75, 1.5),  # default: 0.1
            "operation": "scale",
            "distribution": "log_uniform",
        },
    )

    # -- object
    object_physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("object", body_names=".*"),
            "static_friction_range": (0.7, 1.3),
            "dynamic_friction_range": (0.7, 1.3),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 250,
        },
    )
    object_scale_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("object"),
            "mass_distribution_params": (0.4, 1.6),
            "operation": "scale",
        },
    )
    # (reset terms reset_object / reset_robot_joints are documented in §3)
```
All DR funcs (`randomize_rigid_body_material`, `randomize_rigid_body_mass`, `randomize_actuator_gains`) are stock `isaaclab/envs/mdp/events.py`.

### Smoke (S7)
Per the §7 contract: build env once with DR ON and once with all startup DR terms removed; with seed-matched action sequences the per-step obs trajectories must diverge (e.g. object/joint behavior differs due to randomized mass/friction/gains). Note: since these are `startup`-mode, divergence appears across env *builds*, not across resets within one build.

---

