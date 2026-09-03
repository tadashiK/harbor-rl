# Isaac-Velocity-Rough-Anymal-C-v0 — Implementation Spec

- robot: ANYbotics ANYmal-C quadruped (12 DoF)
- simulator: IsaacLab (Isaac Sim, manager-based)
- objects: none (procedurally generated rough terrain)
- bimanual: false
- summary: Track a commanded base velocity across procedurally generated rough terrain.

> **DELTA vs flat (`Isaac-Velocity-Flat-Anymal-C-v0`):** ROUGH is the *base* config (`AnymalCRoughEnvCfg`); FLAT *subclasses* it and strips the rough additions. The three rough-specific additions are:
> 1. **height-scan ray-caster sensor** (`scene.height_scanner`) + the `height_scan` policy obs term → adds **187** obs dims (17×11 grid). Flat sets both to `None` → obs dim **48** instead of **235**.
> 2. **terrain generator** `TerrainImporterCfg(terrain_type="generator", terrain_generator=ROUGH_TERRAINS_CFG)` (6 rough sub-terrains: stairs, inverted stairs, boxes, random-rough, sloped pyramids). Flat overrides `terrain_type="plane"`, `terrain_generator=None`.
> 3. **terrain curriculum** term `CurriculumCfg.terrain_levels = CurrTerm(func=mdp.terrain_levels_vel)`. Flat sets `curriculum.terrain_levels = None`.
> Flat also retunes 3 reward weights (`flat_orientation_l2 = -5.0`, `dof_torques_l2 = -2.5e-5`, `feet_air_time = 0.5`). All other §2/§3/§4/§6/§7 settings are identical across rough and flat.

---

## §1 Registration + Scene

**Description.** A quadruped (ANYmal-C, 12 DoF) velocity-tracking locomotion env on procedurally generated rough terrain. The robot is commanded a body-frame velocity `(vx, vy, wz)` and must track it while staying upright and avoiding illegal contacts. Scene = generated rough terrain (`/World/ground`), the ANYmal-C articulation, a downward height-scan ray-caster on the base, a body-wide contact sensor, and a dome sky light. 4096 envs at 2.5 m spacing.

**Decisions resolved.**
- `entry_point = isaaclab.envs:ManagerBasedRLEnv`; `env_cfg_entry_point = ...rough_env_cfg:AnymalCRoughEnvCfg`; `disable_env_checker=True`.
- `scene.num_envs = 4096`, `env_spacing = 2.5`.
- Terrain: `terrain_type="generator"`, `terrain_generator=ROUGH_TERRAINS_CFG`, `max_init_terrain_level=5`, friction static/dynamic = 1.0/1.0, restitution 0.0, marble visual material.
- Robot: `ANYMAL_C_CFG` at `{ENV_REGEX_NS}/Robot`. USD = `{ISAACLAB_NUCLEUS_DIR}/Robots/ANYbotics/ANYmal-C/anymal_c.usd`. Init pos `(0,0,0.6)`; init joints HAA=0, F_HFE=0.4, H_HFE=-0.4, F_KFE=-0.8, H_KFE=0.8. Actuators = `ANYDRIVE_3_LSTM_ACTUATOR_CFG` (LSTM actuator net, `network_file={ISAACLAB_NUCLEUS_DIR}/ActuatorNets/ANYbotics/anydrive_3_lstm_jit.pt`, saturation 120, effort_limit 80, vel_limit 7.5). `soft_joint_pos_limit_factor=0.95`.
- Height scanner (**rough-only**): `RayCasterCfg` on `{ENV_REGEX_NS}/Robot/base`, offset z=+20.0, `ray_alignment="yaw"`, `GridPatternCfg(resolution=0.1, size=[1.6,1.0])` → 17×11 = **187** rays, raycasts `/World/ground`.
- Contact sensor: `ContactSensorCfg` on `{ENV_REGEX_NS}/Robot/.*`, `history_length=3`, `track_air_time=True`.
- Sky light: `DomeLightCfg` intensity 750, HDR `{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr`.
- Sim: `decimation=4`, `episode_length_s=20.0`, `sim.dt=0.005` (→ control dt = 4·0.005 = 0.02 s, 50 Hz). `gpu_max_rigid_patch_count = 10·2**15`. `height_scanner.update_period = decimation*dt = 0.02`; `contact_forces.update_period = dt = 0.005`.

**Asset paths (resolved, nucleus-relative).**
- Robot USD: `{ISAACLAB_NUCLEUS_DIR}/Robots/ANYbotics/ANYmal-C/anymal_c.usd`
- Actuator net: `{ISAACLAB_NUCLEUS_DIR}/ActuatorNets/ANYbotics/anydrive_3_lstm_jit.pt`
- Sky HDR: `{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr`
- Terrain material MDL: `{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/TilesMarbleSpiderWhiteBrickBondHoned.mdl`

**Code — registration (`config/anymal_c/__init__.py`).**
```python
gym.register(
    id="Isaac-Velocity-Rough-Anymal-C-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rough_env_cfg:AnymalCRoughEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:AnymalCRoughPPORunnerCfg",
        "rsl_rl_with_symmetry_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:AnymalCRoughPPORunnerWithSymmetryCfg"
        ),
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_rough_ppo_cfg.yaml",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_rough_ppo_cfg.yaml",
    },
)
```

**Code — robot override (`config/anymal_c/rough_env_cfg.py`).**
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

**Code — base scene (`velocity_env_cfg.py:MySceneCfg`).**
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

**Code — `ROUGH_TERRAINS_CFG` (`isaaclab/terrains/config/rough.py`) — the rough-terrain generator.**
```python
ROUGH_TERRAINS_CFG = TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=10,
    num_cols=20,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    sub_terrains={
        "pyramid_stairs": terrain_gen.MeshPyramidStairsTerrainCfg(
            proportion=0.2,
            step_height_range=(0.05, 0.23),
            step_width=0.3,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
        "pyramid_stairs_inv": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
            proportion=0.2,
            step_height_range=(0.05, 0.23),
            step_width=0.3,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
        "boxes": terrain_gen.MeshRandomGridTerrainCfg(
            proportion=0.2, grid_width=0.45, grid_height_range=(0.05, 0.2), platform_width=2.0
        ),
        "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.2, noise_range=(0.02, 0.10), noise_step=0.02, border_width=0.25
        ),
        "hf_pyramid_slope": terrain_gen.HfPyramidSlopedTerrainCfg(
            proportion=0.1, slope_range=(0.0, 0.4), platform_width=2.0, border_width=0.25
        ),
        "hf_pyramid_slope_inv": terrain_gen.HfInvertedPyramidSlopedTerrainCfg(
            proportion=0.1, slope_range=(0.0, 0.4), platform_width=2.0, border_width=0.25
        ),
    },
)
```

**Code — ANYmal-C articulation (`isaaclab_assets/robots/anymal.py`).**
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
            ".*HAA": 0.0,   # all HAA
            ".*F_HFE": 0.4,  # both front HFE
            ".*H_HFE": -0.4, # both hind HFE
            ".*F_KFE": -0.8, # both front KFE
            ".*H_KFE": 0.8,  # both hind KFE
        },
    ),
    actuators={"legs": ANYDRIVE_3_LSTM_ACTUATOR_CFG},
    soft_joint_pos_limit_factor=0.95,
)
```

**Smoke (§1 build).** `cd <repo> && .venv/bin/python -c "import gymnasium as gym; env=gym.make('Isaac-Velocity-Rough-Anymal-C-v0'); print(env.observation_space, env.action_space); env.close()"` — expected `Box(..., (1, 235), ...) Box(..., (1, 12), ...)`. NOT captured here (Isaac `pxr` runtime unavailable). Run on a GPU host with full Isaac Sim install to capture canonical stdout.

---

## §2 Actions

**Description.** Single action group: per-joint position targets for all 12 joints, scaled and added to the default joint offset (no IK, no gripper — quadruped).

**Decisions resolved.** `JointPositionActionCfg(asset_name="robot", joint_names=[".*"], scale=0.5, use_default_offset=True)`. Action dim = 12. Identical in rough and flat.

**Code (`velocity_env_cfg.py:ActionsCfg`).**
```python
@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    joint_pos = mdp.JointPositionActionCfg(asset_name="robot", joint_names=[".*"], scale=0.5, use_default_offset=True)
```

**Code — Commands (`velocity_env_cfg.py:CommandsCfg`).** (Velocity locomotion uses a command manager, not a goal pose; included here for completeness — referenced by §4/§5/§6.)
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
```

**Smoke (§2).** After build, step random actions and assert action vector length 12 and policy clip respected. Expected: env steps without shape error.

---

## §3 Reset

**Description.** On reset: zero external force/torque on base (no-op placeholder term), root pose randomized in a ±0.5 m xy box with full yaw randomization and small random base velocities, and joints scaled to 0.5–1.5× their default position. Identical in rough and flat (flat only nulls these in the `_PLAY` variant).

**Decisions resolved.**
- `base_external_force_torque` (`mode="reset"`): force_range=(0,0), torque_range=(-0,0) on body `base` (effectively disabled).
- `reset_base` (`mode="reset"`, func `reset_root_state_uniform`): pose_range x/y=(-0.5,0.5), yaw=(-3.14,3.14); velocity_range x/y/z/roll/pitch/yaw=(-0.5,0.5).
- `reset_robot_joints` (`mode="reset"`, func `reset_joints_by_scale`): position_range=(0.5,1.5), velocity_range=(0.0,0.0).

**Code (`velocity_env_cfg.py:EventCfg`, `mode="reset"` terms).**
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

**Smoke (§3).** Reset twice with different seeds; assert initial root xy and joint pos differ across seeds. Expected: trajectories diverge at t=0.

---

## §4 Goal + Termination

**Description.** No discrete "success" — the objective is continuous velocity tracking (see §6). Episode ends on timeout (20 s) or when the base makes illegal contact (force > 1.0 N on `base`).

**Decisions resolved.**
- `time_out` = `DoneTerm(func=mdp.time_out, time_out=True)` (episode_length_s=20.0 → 1000 control steps).
- `base_contact` = `DoneTerm(func=mdp.illegal_contact, params={sensor_cfg=ContactSensor("base"), threshold=1.0})`.
- Command (goal signal): `base_velocity` resampled every 10 s, heading-driven; ranges lin_x/lin_y/ang_z = (-1,1), heading = (-π, π). 2% of envs commanded to stand still; 100% heading-controlled.

**Code (`velocity_env_cfg.py:TerminationsCfg`).**
```python
@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    base_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="base"), "threshold": 1.0},
    )
```

**Smoke (§4).** Step until timeout horizon; assert `time_out` fires at step 1000 and episode resets. Expected: done flag set on horizon.

---

## §5 Observation

**Description.** Single policy obs group, concatenated, with corruption (additive uniform noise) enabled. Proprioception (base lin/ang vel, projected gravity, joint pos/vel, last action) + velocity command + **height-scan terrain perception** (rough-only). Total dim **235** (flat = 48; the **187**-dim `height_scan` term is the entire rough delta).

**Decisions resolved (term : dim : noise).**
- `base_lin_vel` : 3 : Unoise(±0.1)
- `base_ang_vel` : 3 : Unoise(±0.2)
- `projected_gravity` : 3 : Unoise(±0.05)
- `velocity_commands` (`generated_commands`, command_name="base_velocity") : 3 : none
- `joint_pos` (`joint_pos_rel`) : 12 : Unoise(±0.01)
- `joint_vel` (`joint_vel_rel`) : 12 : Unoise(±1.5)
- `actions` (`last_action`) : 12 : none
- `height_scan` (`mdp.height_scan`, sensor `height_scanner`, offset 0.5) : **187** : Unoise(±0.1), clip=(-1.0, 1.0) — **rough-only**
- `__post_init__`: `enable_corruption=True`, `concatenate_terms=True`.
- Resolved total: 3+3+3+3+12+12+12+187 = **235**. (Flat strips `height_scan` → **48**.)

**Code (`velocity_env_cfg.py:ObservationsCfg`).**
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

**Code — `height_scan` obs func (`isaaclab/envs/mdp/observations.py`).**
```python
def height_scan(env: ManagerBasedEnv, sensor_cfg: SceneEntityCfg, offset: float = 0.5) -> torch.Tensor:
    """Height scan from the given sensor w.r.t. the sensor's frame.

    The provided offset (Defaults to 0.5) is subtracted from the returned values.
    """
    # extract the used quantities (to enable type-hinting)
    sensor: RayCaster = env.scene.sensors[sensor_cfg.name]
    # height scan: height = sensor_height - hit_point_z - offset
    return sensor.data.pos_w[:, 2].unsqueeze(1) - sensor.data.ray_hits_w[..., 2] - offset
```

**Smoke (§5).** Build env; assert `env.observation_space.shape[-1] == 235`. Expected: `235`.

---

## §6 Reward

**Description.** Sum-composed (additive) velocity-tracking reward with regularization penalties. Two positive tracking terms (exp-kernel lin-xy and ang-z velocity tracking), one positive gait shaping term (feet air time, gated on a non-zero command), and a stack of negative penalties (vertical/roll-pitch velocity, joint torque/accel, action rate, undesired thigh contacts). Two penalties are present but disabled (weight 0) in rough and are the ones flat retunes.

**Composer: SUM** (IsaacLab `RewardManager` sums weighted terms).

**Decisions resolved (term : func : weight : params).**
- `track_lin_vel_xy_exp` : `mdp.track_lin_vel_xy_exp` : **+1.0** : command="base_velocity", std=√0.25=0.5
- `track_ang_vel_z_exp` : `mdp.track_ang_vel_z_exp` : **+0.5** : command="base_velocity", std=0.5
- `lin_vel_z_l2` : `mdp.lin_vel_z_l2` : **-2.0**
- `ang_vel_xy_l2` : `mdp.ang_vel_xy_l2` : **-0.05**
- `dof_torques_l2` : `mdp.joint_torques_l2` : **-1.0e-5**  *(flat: -2.5e-5)*
- `dof_acc_l2` : `mdp.joint_acc_l2` : **-2.5e-7**
- `action_rate_l2` : `mdp.action_rate_l2` : **-0.01**
- `feet_air_time` : `mdp.feet_air_time` : **+0.125** : sensor=contact_forces body `.*FOOT`, command="base_velocity", threshold=0.5  *(flat: +0.5)*
- `undesired_contacts` : `mdp.undesired_contacts` : **-1.0** : sensor=contact_forces body `.*THIGH`, threshold=1.0
- `flat_orientation_l2` : `mdp.flat_orientation_l2` : **0.0** (disabled in rough)  *(flat: -5.0)*
- `dof_pos_limits` : `mdp.joint_pos_limits` : **0.0** (disabled)

**Planning-budget (retro-computed, per-step saturated nominal magnitudes).** Tracking ceiling: lin-xy exp → max +1.0/step; ang-z exp → +0.5/step; feet_air_time (per foot exceeding 0.5 s threshold) → +0.125 per qualifying first-contact. Penalties scale with the squared magnitudes of vel-z / roll-pitch-vel / torque / accel / action-rate and the count of thigh contacts (weight −1.0 each). The reward is dominated by the two exp tracking terms saturating near 1.0 once the policy tracks the command; penalties keep motion smooth and posture upright. No sparse success bonus exists (continuous task).

**Code — local reward func `feet_air_time` (velocity `mdp/rewards.py`).**
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

**Code — base reward funcs used (`isaaclab/envs/mdp/rewards.py`, verbatim).**
```python
def lin_vel_z_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize z-axis base linear velocity using L2 squared kernel."""
    asset: RigidObject = env.scene[asset_cfg.name]
    return torch.square(asset.data.root_lin_vel_b[:, 2])


def ang_vel_xy_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize xy-axis base angular velocity using L2 squared kernel."""
    asset: RigidObject = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.root_ang_vel_b[:, :2]), dim=1)


def flat_orientation_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize non-flat base orientation using L2 squared kernel.

    This is computed by penalizing the xy-components of the projected gravity vector.
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.projected_gravity_b[:, :2]), dim=1)


def joint_torques_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize joint torques applied on the articulation using L2 squared kernel.

    .. note::
        Only the joints configured in :attr:`asset_cfg.joint_ids` will have their joint torques
        contribute to the term.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.applied_torque[:, asset_cfg.joint_ids]), dim=1)


def joint_acc_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize joint accelerations on the articulation using L2 squared kernel.

    .. note::
        Only the joints configured in :attr:`asset_cfg.joint_ids` will have their joint accelerations
        contribute to the term.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.joint_acc[:, asset_cfg.joint_ids]), dim=1)


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


def action_rate_l2(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Penalize the rate of change of the actions using L2 squared kernel."""
    return torch.sum(torch.square(env.action_manager.action - env.action_manager.prev_action), dim=1)


def undesired_contacts(env: ManagerBasedRLEnv, threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize undesired contacts as the number of violations that are above a threshold."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_contact_forces = contact_sensor.data.net_forces_w_history
    is_contact = torch.max(torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1)[0] > threshold
    return torch.sum(is_contact, dim=1)


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
```

**Code (`velocity_env_cfg.py:RewardsCfg`).**
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

**Smoke (§6).** Step with random actions; assert all reward terms finite, total reward non-constant, and `sum(detailed_reward.values()) == reward` per step (composer = sum). Expected: assertion passes.

---

## §7 DR (Domain Randomization + Curriculum)

**Description.** Three startup-mode randomizations (rigid-body material friction/restitution, base mass, base COM) and one interval-mode push (random base velocity kick every 10–15 s). Plus the **rough-only terrain curriculum** (`terrain_levels`) that promotes/demotes envs across terrain difficulty levels based on distance walked.

**Decisions resolved.**
- `physics_material` (`startup`, `randomize_rigid_body_material`): static_friction (0.8,0.8), dynamic_friction (0.6,0.6), restitution (0.0,0.0), num_buckets=64, body `.*`. (Note: both friction ranges are degenerate point-ranges in this base config → effectively fixed friction, randomized via bucketing.)
- `add_base_mass` (`startup`, `randomize_rigid_body_mass`): body `base`, mass_distribution_params=(-5.0,5.0), operation="add".
- `base_com` (`startup`, `randomize_rigid_body_com`): body `base`, com_range x/y=(-0.05,0.05), z=(-0.01,0.01).
- `push_robot` (`interval`, `push_by_setting_velocity`): interval_range_s=(10.0,15.0), velocity_range x/y=(-0.5,0.5).
- **Curriculum (rough-only):** `terrain_levels = CurrTerm(func=mdp.terrain_levels_vel)`. Move-up if distance walked > size[0]/2 (= 4.0 m); move-down if distance < commanded-speed·max_episode_length·0.5. In `__post_init__`, presence of this term flips `terrain_generator.curriculum = True`. Flat nulls it → `curriculum.terrain_levels = None`.

**Code (`velocity_env_cfg.py:EventCfg`, non-reset terms).**
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

**Code — Curriculum (`velocity_env_cfg.py:CurriculumCfg` + `mdp/curriculums.py:terrain_levels_vel`).**
```python
@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""

    terrain_levels = CurrTerm(func=mdp.terrain_levels_vel)
```
```python
def terrain_levels_vel(
    env: ManagerBasedRLEnv, env_ids: Sequence[int], asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Curriculum based on the distance the robot walked when commanded to move at a desired velocity.

    This term is used to increase the difficulty of the terrain when the robot walks far enough and decrease the
    difficulty when the robot walks less than half of the distance required by the commanded velocity.

    .. note::
        It is only possible to use this term with the terrain type ``generator``. For further information
        on different terrain types, check the :class:`isaaclab.terrains.TerrainImporter` class.

    Returns:
        The mean terrain level for the given environment ids.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    terrain: TerrainImporter = env.scene.terrain
    command = env.command_manager.get_command("base_velocity")
    # compute the distance the robot walked
    distance = torch.norm(asset.data.root_pos_w[env_ids, :2] - env.scene.env_origins[env_ids, :2], dim=1)
    # robots that walked far enough progress to harder terrains
    move_up = distance > terrain.cfg.terrain_generator.size[0] / 2
    # robots that walked less than half of their required distance go to simpler terrains
    move_down = distance < torch.norm(command[env_ids, :2], dim=1) * env.max_episode_length_s * 0.5
    move_down *= ~move_up
    # update terrain levels
    terrain.update_env_origins(env_ids, move_up, move_down)
    # return the mean terrain level
    return torch.mean(terrain.terrain_levels.float())
```

**Code — `__post_init__` curriculum/terrain wiring + EnvCfg (`velocity_env_cfg.py`).**
```python
@configclass
class LocomotionVelocityRoughEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the locomotion velocity-tracking environment."""

    scene: MySceneCfg = MySceneCfg(num_envs=4096, env_spacing=2.5)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        """Post initialization."""
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

**Smoke (§7).** Run two seed-matched rollouts with DR ON vs OFF (events nulled); assert obs trajectories diverge. Expected: divergence > 0.

---

## §8 Default training config (RSL-RL PPO, reference)

`config/anymal_c/agents/rsl_rl_ppo_cfg.py:AnymalCRoughPPORunnerCfg` — for reproduction parity (not part of the env spec):
- `num_steps_per_env=24`, `max_iterations=1500`, `save_interval=50`, `experiment_name="anymal_c_rough"`.
- policy: ActorCritic, hidden dims [512,256,128] (both actor & critic, ELU), `init_noise_std=1.0`, no obs normalization.
- algo: PPO, value_loss_coef=1.0, clip_param=0.2, entropy_coef=0.005, 5 epochs, 4 mini-batches, lr=1e-3 (adaptive, desired_kl=0.01), gamma=0.99, lam=0.95, max_grad_norm=1.0.
- (Flat runner: max_iterations=300, hidden dims [128,128,128].)

---

