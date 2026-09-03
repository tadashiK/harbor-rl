# Isaac-Velocity-Flat-Anymal-C-v0 — Implementation Spec

- robot: ANYbotics ANYmal-C quadruped (12 DoF)
- simulator: IsaacLab (Isaac Sim, manager-based)
- objects: none (flat terrain)
- bimanual: false
- summary: Track a commanded base velocity while trotting on flat ground.

This is a **quadruped velocity-command tracking** task. The ANYmal-C robot must track a commanded base linear velocity (x, y) and yaw angular velocity on **flat ground**, with a rich shaped reward (exponential velocity tracking + many regularization penalties) and substantial domain randomization (startup friction/mass/CoM, reset pose/joint scale, interval velocity pushes).

The config is layered: abstract `LocomotionVelocityRoughEnvCfg` (in `velocity_env_cfg.py`) defines the full MDP for rough terrain → `AnymalCRoughEnvCfg` swaps in the ANYmal-C articulation → `AnymalCFlatEnvCfg` flattens terrain, drops the height scanner/scan obs, drops the terrain curriculum, and overrides three reward weights.

---

## §1 Registration + Scene

**Description.** Registers `Isaac-Velocity-Flat-Anymal-C-v0` against the generic `ManagerBasedRLEnv`, with `AnymalCFlatEnvCfg` as the env cfg. The scene is a flat ground plane (terrain `"plane"`, no generator), the ANYmal-C 12-DoF quadruped articulation (ANYdrive-3 LSTM actuator net), a body-wide `ContactSensor` (`track_air_time=True`), and a dome sky light. The rough-parent `height_scanner` RayCaster and `terrain_generator` are nulled out by the flat override.

**Decisions resolved.**
- entry_point = `isaaclab.envs:ManagerBasedRLEnv`, `disable_env_checker=True`
- env_cfg_entry_point = `…config.anymal_c.flat_env_cfg:AnymalCFlatEnvCfg`
- robot = `ANYMAL_C_CFG` with `prim_path="{ENV_REGEX_NS}/Robot"`
- robot USD = `{ISAACLAB_NUCLEUS_DIR}/Robots/ANYbotics/ANYmal-C/anymal_c.usd` where `ISAACLAB_NUCLEUS_DIR = {ISAAC_NUCLEUS_DIR}/IsaacLab` and `ISAAC_NUCLEUS_DIR = {NUCLEUS_ASSET_ROOT_DIR}/Isaac` (nucleus cloud asset root; not on local disk)
- init root pos = (0, 0, 0.6); init joints: `.*HAA=0.0`, `.*F_HFE=0.4`, `.*H_HFE=-0.4`, `.*F_KFE=-0.8`, `.*H_KFE=0.8`
- actuators: ANYdrive-3 LSTM net (`{ISAACLAB_NUCLEUS_DIR}/ActuatorNets/ANYbotics/anydrive_3_lstm_jit.pt`), saturation_effort=120, effort_limit=80, velocity_limit=7.5; `soft_joint_pos_limit_factor=0.95`
- terrain (flat): `terrain_type="plane"`, `terrain_generator=None`, friction static=dynamic=1.0
- `height_scanner = None`, `contact_forces` ContactSensor on `{ENV_REGEX_NS}/Robot/.*`, history_length=3, track_air_time=True, update_period = sim.dt
- scene: num_envs=4096, env_spacing=2.5
- sim: dt=0.005, decimation=4 (control dt = 0.02 s), episode_length_s=20.0

**Code.**

`config/anymal_c/__init__.py` (registration):
```python
import gymnasium as gym

from . import agents

gym.register(
    id="Isaac-Velocity-Flat-Anymal-C-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg:AnymalCFlatEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:AnymalCFlatPPORunnerCfg",
        "rsl_rl_with_symmetry_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:AnymalCFlatPPORunnerWithSymmetryCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_flat_ppo_cfg.yaml",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_flat_ppo_cfg.yaml",
    },
)
```

`config/anymal_c/flat_env_cfg.py` (flat override — scene + reward deltas):
```python
from isaaclab.utils import configclass

from .rough_env_cfg import AnymalCRoughEnvCfg


@configclass
class AnymalCFlatEnvCfg(AnymalCRoughEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # override rewards
        self.rewards.flat_orientation_l2.weight = -5.0
        self.rewards.dof_torques_l2.weight = -2.5e-5
        self.rewards.feet_air_time.weight = 0.5
        # change terrain to flat
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        # no height scan
        self.scene.height_scanner = None
        self.observations.policy.height_scan = None
        # no terrain curriculum
        self.curriculum.terrain_levels = None
```

`config/anymal_c/rough_env_cfg.py` (robot swap):
```python
from isaaclab.utils import configclass

from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import LocomotionVelocityRoughEnvCfg

from isaaclab_assets.robots.anymal import ANYMAL_C_CFG  # isort: skip


@configclass
class AnymalCRoughEnvCfg(LocomotionVelocityRoughEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # switch robot to anymal-c
        self.scene.robot = ANYMAL_C_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
```

`velocity_env_cfg.py` — `MySceneCfg` (abstract base scene; flat override later nulls `height_scanner`/`terrain_generator`):
```python
@configclass
class MySceneCfg(InteractiveSceneCfg):
    """Configuration for the terrain scene with a legged robot."""

    # ground terrain
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=ROUGH_TERRAINS_CFG,
        max_init_terrain_level=5,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path=f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/TilesMarbleSpiderWhiteBrickBondHoned.mdl",
            project_uvw=True,
            texture_scale=(0.25, 0.25),
        ),
        debug_vis=False,
    )
    # robots
    robot: ArticulationCfg = MISSING
    # sensors
    height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0]),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )
    contact_forces = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True)
    # lights
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )
```

`isaaclab_assets/robots/anymal.py` — `ANYMAL_C_CFG` + actuator net (verbatim):
```python
ANYDRIVE_3_LSTM_ACTUATOR_CFG = ActuatorNetLSTMCfg(
    joint_names_expr=[".*HAA", ".*HFE", ".*KFE"],
    network_file=f"{ISAACLAB_NUCLEUS_DIR}/ActuatorNets/ANYbotics/anydrive_3_lstm_jit.pt",
    saturation_effort=120.0,
    effort_limit=80.0,
    velocity_limit=7.5,
)

ANYMAL_C_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAACLAB_NUCLEUS_DIR}/Robots/ANYbotics/ANYmal-C/anymal_c.usd",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True, solver_position_iteration_count=4, solver_velocity_iteration_count=0
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.6),
        joint_pos={
            ".*HAA": 0.0,  # all HAA
            ".*F_HFE": 0.4,  # both front HFE
            ".*H_HFE": -0.4,  # both hind HFE
            ".*F_KFE": -0.8,  # both front KFE
            ".*H_KFE": 0.8,  # both hind KFE
        },
    ),
    actuators={"legs": ANYDRIVE_3_LSTM_ACTUATOR_CFG},
    soft_joint_pos_limit_factor=0.95,
)
```

`LocomotionVelocityRoughEnvCfg.__post_init__` (env-level timing/sim):
```python
@configclass
class LocomotionVelocityRoughEnvCfg(ManagerBasedRLEnvCfg):
    scene: MySceneCfg = MySceneCfg(num_envs=4096, env_spacing=2.5)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        self.decimation = 4
        self.episode_length_s = 20.0
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15
        if self.scene.height_scanner is not None:
            self.scene.height_scanner.update_period = self.decimation * self.sim.dt
        if self.scene.contact_forces is not None:
            self.scene.contact_forces.update_period = self.sim.dt
        if getattr(self.curriculum, "terrain_levels", None) is not None:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = True
        else:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = False
```

**Smoke (§1 build).** `cd <repo> && .venv/bin/python -c "import gymnasium as gym; env=gym.make('Isaac-Velocity-Flat-Anymal-C-v0'); print(env.observation_space, env.action_space); env.close()"`
Expected: `Box(-inf, inf, (N_envs, 48)) Box(-inf, inf, (N_envs, 12))` (per-env: obs 48, act 12). Not captured here — Isaac `pxr`/sim cannot boot in this `.venv`.

---

## §2 Actions

**Description.** A single joint-position action term over all 12 joints, scaled by 0.5 and added to the default joint pose (`use_default_offset=True`). No gripper / no task-space IK — pure joint-position control of the quadruped legs.

**Decisions resolved.**
- term: `mdp.JointPositionActionCfg`, `asset_name="robot"`, `joint_names=[".*"]` (all 12), `scale=0.5`, `use_default_offset=True`
- action dim = 12

**Code.**
```python
@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    joint_pos = mdp.JointPositionActionCfg(asset_name="robot", joint_names=[".*"], scale=0.5, use_default_offset=True)
```

**Smoke (§2).** Step the env with `action = env.action_space.sample()` for a few steps; assert joint targets = default_joint_pos + 0.5*action and the robot articulation moves. (Run when reproducing.)

---

## §3 Reset

**Description.** On each reset: (1) apply a zero external force/torque to the base (a no-op placeholder term, ranges 0.0); (2) randomize base root pose (xy ±0.5 m, yaw ±π) and base velocity (lin/ang each ±0.5); (3) scale joint positions by a random factor in [0.5, 1.5] (joint velocities unchanged). All three are `mode="reset"` `EventTerm`s.

**Decisions resolved.**
- `base_external_force_torque`: force_range=(0,0), torque_range=(0,0), body=base (effectively disabled)
- `reset_base`: pose_range x=(-0.5,0.5), y=(-0.5,0.5), yaw=(-3.14,3.14); velocity_range x/y/z/roll/pitch/yaw all (-0.5,0.5)
- `reset_robot_joints`: position_range=(0.5,1.5) (multiplicative scale of default joints), velocity_range=(0.0,0.0)

**Code.**
```python
    # reset
    base_external_force_torque = EventTerm(
        func=mdp.apply_external_force_torque,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "force_range": (0.0, 0.0),
            "torque_range": (-0.0, 0.0),
        },
    )

    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
                "z": (-0.5, 0.5),
                "roll": (-0.5, 0.5),
                "pitch": (-0.5, 0.5),
                "yaw": (-0.5, 0.5),
            },
        },
    )

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (0.5, 1.5),
            "velocity_range": (0.0, 0.0),
        },
    )
```
Reset funcs resolve to core `isaaclab.envs.mdp.events`: `apply_external_force_torque` (events.py:1010), `reset_root_state_uniform` (events.py:1074), `reset_joints_by_scale` (events.py:1238).

**Smoke (§3).** Call `env.reset()` twice with different seeds; assert root pose / joint pos differ across resets within the documented ranges. (Run when reproducing.)

---

## §4 Goal + Termination

**Description.** The goal is to track a resampled base-velocity command. `CommandsCfg.base_velocity` is a `UniformVelocityCommand` resampled every 10 s, with heading-command mode (yaw command derived from a heading error with stiffness 0.5). Ranges: lin_vel_x/y ∈ [-1, 1] m/s, ang_vel_z ∈ [-1, 1] rad/s, heading ∈ [-π, π]. 2% of envs are "standing" (zero command); all envs use heading control. Termination: episodic timeout (20 s) OR illegal contact on the base body (contact force > 1.0 N).

**Decisions resolved.**
- command: `mdp.UniformVelocityCommandCfg`, asset="robot", resampling_time_range=(10.0, 10.0), rel_standing_envs=0.02, rel_heading_envs=1.0, heading_command=True, heading_control_stiffness=0.5, debug_vis=True
- command ranges: lin_vel_x=(-1.0,1.0), lin_vel_y=(-1.0,1.0), ang_vel_z=(-1.0,1.0), heading=(-π, π)
- terminations: `time_out` (time_out=True, episode_length_s=20.0); `base_contact` via `illegal_contact` on body `base`, threshold=1.0

**Code.**
```python
@configclass
class CommandsCfg:
    """Command specifications for the MDP."""

    base_velocity = mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=0.02,
        rel_heading_envs=1.0,
        heading_command=True,
        heading_control_stiffness=0.5,
        debug_vis=True,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-1.0, 1.0), lin_vel_y=(-1.0, 1.0), ang_vel_z=(-1.0, 1.0), heading=(-math.pi, math.pi)
        ),
    )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    base_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="base"), "threshold": 1.0},
    )
```
Termination funcs resolve to core `isaaclab.envs.mdp.terminations`: `time_out` (terminations.py:31), `illegal_contact` (terminations.py:154).

**Smoke (§4).** Roll out a flat policy / zero action; assert `terminated` triggers when the base falls (base contact) and `truncated` triggers at 20 s. Confirm the velocity command resamples every 10 s and lies within ranges. (Run when reproducing.)

---

## §5 Observation

**Description.** Single `policy` observation group, concatenated, with additive uniform noise (corruption enabled at train time). On flat, the `height_scan` term is removed by the flat override. Remaining terms (order preserved): base linear velocity (3), base angular velocity (3), projected gravity (3), velocity command (3), joint position relative to default (12), joint velocity relative to default (12), last action (12). Total = 48.

**Decisions resolved.**
- group flags: `enable_corruption=True` (train; `False` in `_PLAY`), `concatenate_terms=True`
- noise (additive uniform): base_lin_vel ±0.1, base_ang_vel ±0.2, projected_gravity ±0.05, joint_pos ±0.01, joint_vel ±1.5; velocity_commands & actions noiseless
- `height_scan` term = `None` on flat (in rough it adds a 1.6×1.0 @ 0.1-res grid scan, clipped to [-1,1])
- resolved total obs dim = 48

**Code.** Abstract base `ObservationsCfg` (the flat override sets `self.observations.policy.height_scan = None`):
```python
@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, noise=Unoise(n_min=-0.1, n_max=0.1))
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            noise=Unoise(n_min=-0.05, n_max=0.05),
        )
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-1.5, n_max=1.5))
        actions = ObsTerm(func=mdp.last_action)
        height_scan = ObsTerm(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner")},
            noise=Unoise(n_min=-0.1, n_max=0.1),
            clip=(-1.0, 1.0),
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()
```
Obs funcs (`base_lin_vel`, `base_ang_vel`, `projected_gravity`, `generated_commands`, `joint_pos_rel`, `joint_vel_rel`, `last_action`, `height_scan`) all resolve from core `isaaclab.envs.mdp` (re-exported by the velocity `mdp/__init__.py`).

**Smoke (§5).** Build the env and assert `env.observation_space["policy"].shape[-1] == 48`. (Run when reproducing.)

---

## §6 Reward

**Description.** Composer = **sum** (IsaacLab `RewardManager` sums all weighted `RewTerm`s; every weight is additionally multiplied by the control step dt ≈ 0.02 internally — see note). Two positive task terms (exponential velocity tracking) plus seven regularization penalties, plus two optional penalties parked at weight 0. The flat override raises `flat_orientation_l2` to -5.0, sets `dof_torques_l2` to -2.5e-5, and `feet_air_time` to 0.5.

**Decisions resolved (flat-resolved weights).**
- `track_lin_vel_xy_exp`: weight=1.0, params std=√0.25 (=0.5), command="base_velocity"
- `track_ang_vel_z_exp`: weight=0.5, std=√0.25, command="base_velocity"
- `lin_vel_z_l2`: weight=-2.0
- `ang_vel_xy_l2`: weight=-0.05
- `dof_torques_l2`: weight=**-2.5e-5** (flat override; base was -1.0e-5)
- `dof_acc_l2`: weight=-2.5e-7
- `action_rate_l2`: weight=-0.01
- `feet_air_time`: weight=**0.5** (flat override; base 0.125), threshold=0.5, feet=`.*FOOT`, command="base_velocity"
- `undesired_contacts`: weight=-1.0, body=`.*THIGH`, threshold=1.0
- `flat_orientation_l2`: weight=**-5.0** (flat override; base 0.0)
- `dof_pos_limits` (joint_pos_limits): weight=0.0 (disabled)

**Planning budget (per-step saturated nominal magnitudes, retro-computed).** Positive ceiling: `track_lin_vel_xy_exp` ∈ [0,1] (saturates at perfect tracking → ~1.0), `track_ang_vel_z_exp` ∈ [0,0.5] (→ ~0.5), `feet_air_time` is a sparse first-contact bonus `(air_time−0.5)·first_contact·0.5` (small positive when stepping under nonzero command). Penalties grow with the squared/abs quantity and are unbounded above; the dominant ones at training start are `flat_orientation_l2` (-5.0) and `lin_vel_z_l2` (-2.0). The task is solved when the two exp-tracking terms saturate near their ceilings while penalties stay near zero.

**Code (RewardsCfg, abstract base — flat override changes 3 weights as noted above).**
```python
@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # -- task
    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_exp, weight=1.0, params={"command_name": "base_velocity", "std": math.sqrt(0.25)}
    )
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_exp, weight=0.5, params={"command_name": "base_velocity", "std": math.sqrt(0.25)}
    )
    # -- penalties
    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-2.0)
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)
    dof_torques_l2 = RewTerm(func=mdp.joint_torques_l2, weight=-1.0e-5)
    dof_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7)
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.01)
    feet_air_time = RewTerm(
        func=mdp.feet_air_time,
        weight=0.125,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*FOOT"),
            "command_name": "base_velocity",
            "threshold": 0.5,
        },
    )
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-1.0,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*THIGH"), "threshold": 1.0},
    )
    # -- optional penalties
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=0.0)
    dof_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=0.0)
```

**Reward function source (verbatim).**

`feet_air_time` — task-local, `velocity/mdp/rewards.py`:
```python
def feet_air_time(
    env: ManagerBasedRLEnv, command_name: str, sensor_cfg: SceneEntityCfg, threshold: float
) -> torch.Tensor:
    """Reward long steps taken by the feet using L2-kernel.

    This function rewards the agent for taking steps that are longer than a threshold. This helps ensure
    that the robot lifts its feet off the ground and takes steps. The reward is computed as the sum of
    the time for which the feet are in the air.

    If the commands are small (i.e. the agent is not supposed to take a step), then the reward is zero.
    """
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    first_contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    reward = torch.sum((last_air_time - threshold) * first_contact, dim=1)
    # no reward for zero command
    reward *= torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.1
    return reward
```

The remaining reward funcs are core `isaaclab.envs.mdp.rewards` (re-exported via the velocity `mdp/__init__.py`). Verbatim:
```python
def track_lin_vel_xy_exp(
    env: ManagerBasedRLEnv, std: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of linear velocity commands (xy axes) using exponential kernel."""
    asset: RigidObject = env.scene[asset_cfg.name]
    lin_vel_error = torch.sum(
        torch.square(env.command_manager.get_command(command_name)[:, :2] - asset.data.root_lin_vel_b[:, :2]),
        dim=1,
    )
    return torch.exp(-lin_vel_error / std**2)


def track_ang_vel_z_exp(
    env: ManagerBasedRLEnv, std: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of angular velocity commands (yaw) using exponential kernel."""
    asset: RigidObject = env.scene[asset_cfg.name]
    ang_vel_error = torch.square(env.command_manager.get_command(command_name)[:, 2] - asset.data.root_ang_vel_b[:, 2])
    return torch.exp(-ang_vel_error / std**2)


def lin_vel_z_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize z-axis base linear velocity using L2 squared kernel."""
    asset: RigidObject = env.scene[asset_cfg.name]
    return torch.square(asset.data.root_lin_vel_b[:, 2])


def ang_vel_xy_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize xy-axis base angular velocity using L2 squared kernel."""
    asset: RigidObject = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.root_ang_vel_b[:, :2]), dim=1)


def joint_torques_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize joint torques applied on the articulation using L2 squared kernel."""
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.applied_torque[:, asset_cfg.joint_ids]), dim=1)


def joint_acc_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize joint accelerations on the articulation using L2 squared kernel."""
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.joint_acc[:, asset_cfg.joint_ids]), dim=1)


def action_rate_l2(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Penalize the rate of change of the actions using L2 squared kernel."""
    return torch.sum(torch.square(env.action_manager.action - env.action_manager.prev_action), dim=1)


def undesired_contacts(env: ManagerBasedRLEnv, threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize undesired contacts as the number of violations that are above a threshold."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_contact_forces = contact_sensor.data.net_forces_w_history
    is_contact = torch.max(torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1)[0] > threshold
    return torch.sum(is_contact, dim=1)


def flat_orientation_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize non-flat base orientation using L2 squared kernel.

    This is computed by penalizing the xy-components of the projected gravity vector.
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.projected_gravity_b[:, :2]), dim=1)


def joint_pos_limits(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize joint positions if they cross the soft limits.

    This is computed as a sum of the absolute value of the difference between the joint position and the soft limits.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    out_of_limits = -(
        asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.soft_joint_pos_limits[:, asset_cfg.joint_ids, 0]
    ).clip(max=0.0)
    out_of_limits += (
        asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.soft_joint_pos_limits[:, asset_cfg.joint_ids, 1]
    ).clip(min=0.0)
    return torch.sum(out_of_limits, dim=1)
```

**Smoke (§6).** Roll out random actions; assert total reward is finite and non-constant across steps, and that the per-term sum equals the env reward (composer="sum"). (Run when reproducing.)

---

## §7 DR

**Description.** Substantial domain randomization. Startup (once per env at spawn): per-body rigid-body **material** randomization (friction/restitution bucketed into 64 buckets), base **mass** offset ±5 kg, base **center-of-mass** offset (x/y ±0.05, z ±0.01). Interval (every 10–15 s during the episode): **push** the robot by directly setting base xy velocity in [-0.5, 0.5]. (The `_PLAY` variant disables corruption and removes `push_robot` + `base_external_force_torque`.) The reset-mode terms are documented in §3.

**Decisions resolved.**
- startup `physics_material`: static_friction_range=(0.8,0.8), dynamic_friction_range=(0.6,0.6), restitution_range=(0.0,0.0), num_buckets=64, body=`.*`
- startup `add_base_mass`: mass_distribution_params=(-5.0,5.0), operation="add", body=base
- startup `base_com`: com_range x=(-0.05,0.05), y=(-0.05,0.05), z=(-0.01,0.01), body=base
- interval `push_robot`: interval_range_s=(10.0,15.0), velocity_range x=(-0.5,0.5), y=(-0.5,0.5)

**Code (non-reset EventCfg terms — startup + interval).**
```python
    # startup
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.8, 0.8),
            "dynamic_friction_range": (0.6, 0.6),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )

    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "mass_distribution_params": (-5.0, 5.0),
            "operation": "add",
        },
    )

    base_com = EventTerm(
        func=mdp.randomize_rigid_body_com,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "com_range": {"x": (-0.05, 0.05), "y": (-0.05, 0.05), "z": (-0.01, 0.01)},
        },
    )

    # interval
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(10.0, 15.0),
        params={"velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)}},
    )
```
DR funcs resolve to core `isaaclab.envs.mdp.events`: `randomize_rigid_body_material` (events.py:155, ManagerTermBase class), `randomize_rigid_body_mass` (events.py:286, class), `randomize_rigid_body_com` (events.py:400), `push_by_setting_velocity` (events.py:1046).

**Smoke (§7).** Compare two seed-matched rollouts with DR ON vs DR OFF (`_PLAY` disables `push_robot`); assert obs trajectories diverge. (Run when reproducing.)

---

