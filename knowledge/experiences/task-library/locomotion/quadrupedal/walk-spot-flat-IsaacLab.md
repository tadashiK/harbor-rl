# Isaac-Velocity-Flat-Spot-v0 — Implementation Spec

- robot: Boston Dynamics Spot quadruped (12 DoF, remotized-PD knee)
- simulator: IsaacLab (Isaac Sim, manager-based)
- objects: none (flat terrain)
- bimanual: false
- summary: Track a commanded base velocity while walking on flat ground.

> Boston Dynamics **Spot** quadruped, flat-terrain velocity tracking, manager-based RL env (`ManagerBasedRLEnv`). Spot is notable because it ships its **own** `mdp/` subtree (`config/spot/mdp/rewards.py`, `config/spot/mdp/events.py`) with a **distinct reward set** — gait-enforcement, foot-air-time, foot-clearance, foot-slip, air-time-variance, plus a remotized-PD knee actuator — that differs from the shared Anymal-style velocity reward (`track_lin_vel_xy_exp` / `track_ang_vel_z_exp` + L2 penalties). The whole point of this spec is to capture Spot's own reward functions verbatim.

---

## §1 Registration + Scene

**Description.** Registers `Isaac-Velocity-Flat-Spot-v0` against `ManagerBasedRLEnv` with `SpotFlatEnvCfg`. The env subclasses the shared `LocomotionVelocityRoughEnvCfg` (parent scene = terrain importer + contact sensors + sky light + height-scanner) but in `__post_init__` it (a) swaps in the `SPOT_CFG` Spot articulation, (b) swaps the terrain to a flat-biased `COBBLESTONE_ROAD_CFG` generator, (c) disables the height-scanner, (d) bumps sim to 500 Hz physics / 50 Hz control. The Spot robot has 12 actuated joints (4 legs × {hip_x `_hx`, hip_y `_hy`, knee `_kn`}), a delayed-PD hip actuator group and a **remotized-PD** knee actuator group driven by a torque/transmission lookup table.

**Decisions resolved.**
- `entry_point="isaaclab.envs:ManagerBasedRLEnv"`, `disable_env_checker=True`.
- `env_cfg_entry_point = ...flat_env_cfg:SpotFlatEnvCfg`; rsl_rl cfg `SpotFlatPPORunnerCfg`; skrl cfg `skrl_flat_ppo_cfg.yaml`.
- `decimation = 10`, control rate **50 Hz**; `sim.dt = 0.002` → **500 Hz** physics; `episode_length_s = 20.0`.
- `sim.physics_material`: static=dynamic=1.0, combine modes "multiply".
- `scene.contact_forces.update_period = sim.dt`; `scene.height_scanner = None` (no height scan).
- Robot prim: `{ENV_REGEX_NS}/Robot`. Contact sensor over `{ENV_REGEX_NS}/Robot/.*`, `history_length=3`, `track_air_time=True` (required by the air-time / gait rewards).
- Terrain generator `COBBLESTONE_ROAD_CFG`: 8×8 tiles, 9 rows × 21 cols, sub-terrains 20% flat + 20% random-rough (rest implicitly flat), difficulty 0..1.
- Init root pose `pos=(0,0,0.5)`; init joint pose: left hip_x `+0.1`, right hip_x `-0.1`, front hip_y `0.9`, hind hip_y `1.1`, all knees `-1.5`.
- Hip actuators `DelayedPDActuatorCfg` (`.*_h[xy]`): effort_limit 45, stiffness 60, damping 1.5, delay 0–4 physics steps. Knee actuators `RemotizedPDActuatorCfg` (`.*_kn`): torque limit from `joint_parameter_lookup` table (no fixed effort_limit), stiffness 60, damping 1.5, delay 0–4.

**Resolved asset paths** (`ISAAC_NUCLEUS_DIR = {NUCLEUS_ASSET_ROOT_DIR}/Isaac`, server/nucleus-resolved — not a local file):
- Spot USD: `{ISAAC_NUCLEUS_DIR}/Robots/BostonDynamics/spot/spot.usd`
- Terrain visual MDL: `{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/TilesMarbleSpiderWhiteBrickBondHoned.mdl`
- Sky HDR (from parent `MySceneCfg`): `{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr`

**Code — registration** (`config/spot/__init__.py`):
```python
gym.register(
    id="Isaac-Velocity-Flat-Spot-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg:SpotFlatEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:SpotFlatPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_flat_ppo_cfg.yaml",
    },
)
```

**Code — Spot articulation cfg** (`isaaclab_assets/robots/spot.py`, `SPOT_CFG`). NOTE: the full 99-row `joint_parameter_lookup` table feeding the remotized knee actuator is in that file (`source/isaaclab_assets/isaaclab_assets/robots/spot.py:20-122`) — reproduce it verbatim from there; abbreviated here as `joint_parameter_lookup = [[-2.792900, -24.776718, 37.165077], ...]` (99 `[angle, ratio, torque]` rows):
```python
SPOT_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_NUCLEUS_DIR}/Robots/BostonDynamics/spot/spot.usd",
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
        pos=(0.0, 0.0, 0.5),
        joint_pos={
            "[fh]l_hx": 0.1,   # all left hip_x
            "[fh]r_hx": -0.1,  # all right hip_x
            "f[rl]_hy": 0.9,   # front hip_y
            "h[rl]_hy": 1.1,   # hind hip_y
            ".*_kn": -1.5,     # all knees
        },
        joint_vel={".*": 0.0},
    ),
    actuators={
        "spot_hip": DelayedPDActuatorCfg(
            joint_names_expr=[".*_h[xy]"],
            effort_limit=45.0,
            stiffness=60.0,
            damping=1.5,
            min_delay=0,  # physics time steps (min: 2.0*0=0.0ms)
            max_delay=4,  # physics time steps (max: 2.0*4=8.0ms)
        ),
        "spot_knee": RemotizedPDActuatorCfg(
            joint_names_expr=[".*_kn"],
            joint_parameter_lookup=joint_parameter_lookup,  # 99-row table, see spot.py:20-122
            effort_limit=None,  # handled by RemotizedPDActuatorCfg.data
            stiffness=60.0,
            damping=1.5,
            min_delay=0,
            max_delay=4,
        ),
    },
)
```

**Code — terrain + `__post_init__`** (`flat_env_cfg.py`):
```python
COBBLESTONE_ROAD_CFG = terrain_gen.TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=9,
    num_cols=21,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    difficulty_range=(0.0, 1.0),
    use_cache=False,
    sub_terrains={
        "flat": terrain_gen.MeshPlaneTerrainCfg(proportion=0.2),
        "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.2, noise_range=(0.02, 0.05), noise_step=0.02, border_width=0.25
        ),
    },
)

class SpotFlatEnvCfg(LocomotionVelocityRoughEnvCfg):
    observations = SpotObservationsCfg()
    actions = SpotActionsCfg()
    commands = SpotCommandsCfg()
    rewards = SpotRewardsCfg()
    terminations = SpotTerminationsCfg()
    events = SpotEventCfg()
    viewer = ViewerCfg(eye=(10.5, 10.5, 0.3), origin_type="world", env_index=0, asset_name="robot")

    def __post_init__(self):
        super().__post_init__()
        self.decimation = 10          # 50 Hz
        self.episode_length_s = 20.0
        self.sim.dt = 0.002           # 500 Hz
        self.sim.render_interval = self.decimation
        self.sim.physics_material.static_friction = 1.0
        self.sim.physics_material.dynamic_friction = 1.0
        self.sim.physics_material.friction_combine_mode = "multiply"
        self.sim.physics_material.restitution_combine_mode = "multiply"
        self.scene.contact_forces.update_period = self.sim.dt
        self.scene.robot = SPOT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.terrain = TerrainImporterCfg(
            prim_path="/World/ground",
            terrain_type="generator",
            terrain_generator=COBBLESTONE_ROAD_CFG,
            max_init_terrain_level=COBBLESTONE_ROAD_CFG.num_rows - 1,
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
            debug_vis=True,
        )
        self.scene.height_scanner = None
```

**Smoke (§1 build).** `cd <repo> && .venv/bin/python -c "import gymnasium as gym, isaaclab_tasks; e=gym.make('Isaac-Velocity-Flat-Spot-v0'); print(e.observation_space, e.action_space); e.close()"` — expected `Box(48,) Box(12,)`. Not captured at probe time (no `pxr`). Run on an Isaac-Sim-enabled host to bake the literal stdout.

---

## §2 Actions

**Description.** Single action group: a joint-position action over all 12 Spot joints, scaled by 0.2 around the default joint pose. Output of the policy (12-dim) is multiplied by `scale=0.2` and added to the default joint positions (`use_default_offset=True`), then sent to the PD actuators. NOTE: Spot's action scale (0.2) is **smaller** than the shared parent locomotion default (0.5).

**Decisions resolved.**
- `action_dim = 12` (joint_names `[".*"]` matches all 12 actuated joints).
- `scale = 0.2`, `use_default_offset = True`.

**Code** (`flat_env_cfg.py`):
```python
@configclass
class SpotActionsCfg:
    joint_pos = mdp.JointPositionActionCfg(asset_name="robot", joint_names=[".*"], scale=0.2, use_default_offset=True)
```

**Smoke (§2).** With env built, assert `env.action_space.shape == (12,)` and a zero action holds the robot near its default pose for a few steps.

---

## §3 Reset

**Description.** On each episode reset: zero external force/torque on the body (placeholder, ranges are 0), randomize root pose (xy ±0.5 m, yaw ±π) and root velocity (lin xyz, ang rpy in moderate ranges), and randomize joints **around the default** pose/velocity using Spot's own `reset_joints_around_default` (additive ranges, clipped to soft limits) — distinct from the parent's `reset_joints_by_scale` (multiplicative). Note Spot's reset root-velocity ranges are wider than the parent's.

**Decisions resolved.**
- `base_external_force_torque` (mode reset): body=`body`, force/torque ranges = 0.
- `reset_base`: pose `{x:(-0.5,0.5), y:(-0.5,0.5), yaw:(-3.14,3.14)}`; velocity `{x:(-1.5,1.5), y:(-1.0,1.0), z:(-0.5,0.5), roll:(-0.7,0.7), pitch:(-0.7,0.7), yaw:(-1.0,1.0)}`.
- `reset_robot_joints` → `spot_mdp.reset_joints_around_default`: `position_range=(-0.2,0.2)`, `velocity_range=(-2.5,2.5)` (additive around default, soft-limit-clipped).

**Code — reset EventTerms** (`flat_env_cfg.py`, the `mode="reset"` terms of `SpotEventCfg`):
```python
base_external_force_torque = EventTerm(
    func=mdp.apply_external_force_torque,
    mode="reset",
    params={
        "asset_cfg": SceneEntityCfg("robot", body_names="body"),
        "force_range": (0.0, 0.0),
        "torque_range": (-0.0, 0.0),
    },
)

reset_base = EventTerm(
    func=mdp.reset_root_state_uniform,
    mode="reset",
    params={
        "asset_cfg": SceneEntityCfg("robot"),
        "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
        "velocity_range": {
            "x": (-1.5, 1.5), "y": (-1.0, 1.0), "z": (-0.5, 0.5),
            "roll": (-0.7, 0.7), "pitch": (-0.7, 0.7), "yaw": (-1.0, 1.0),
        },
    },
)

reset_robot_joints = EventTerm(
    func=spot_mdp.reset_joints_around_default,
    mode="reset",
    params={
        "position_range": (-0.2, 0.2),
        "velocity_range": (-2.5, 2.5),
        "asset_cfg": SceneEntityCfg("robot"),
    },
)
```

**Code — Spot-specific reset func** (`config/spot/mdp/events.py`, verbatim):
```python
def reset_joints_around_default(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    position_range: tuple[float, float],
    velocity_range: tuple[float, float],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Reset the robot joints in the interval around the default position and velocity by the given ranges."""
    asset: Articulation = env.scene[asset_cfg.name]
    joint_min_pos = asset.data.default_joint_pos[env_ids] + position_range[0]
    joint_max_pos = asset.data.default_joint_pos[env_ids] + position_range[1]
    joint_min_vel = asset.data.default_joint_vel[env_ids] + velocity_range[0]
    joint_max_vel = asset.data.default_joint_vel[env_ids] + velocity_range[1]
    joint_pos_limits = asset.data.soft_joint_pos_limits[env_ids, ...]
    joint_min_pos = torch.clamp(joint_min_pos, min=joint_pos_limits[..., 0], max=joint_pos_limits[..., 1])
    joint_max_pos = torch.clamp(joint_max_pos, min=joint_pos_limits[..., 0], max=joint_pos_limits[..., 1])
    joint_vel_abs_limits = asset.data.soft_joint_vel_limits[env_ids]
    joint_min_vel = torch.clamp(joint_min_vel, min=-joint_vel_abs_limits, max=joint_vel_abs_limits)
    joint_max_vel = torch.clamp(joint_max_vel, min=-joint_vel_abs_limits, max=joint_vel_abs_limits)
    joint_pos = sample_uniform(joint_min_pos, joint_max_pos, joint_min_pos.shape, joint_min_pos.device)
    joint_vel = sample_uniform(joint_min_vel, joint_max_vel, joint_min_vel.shape, joint_min_vel.device)
    asset.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
```

**Smoke (§3).** Two resets with different seeds produce different initial joint positions/root poses.

---

## §4 Goal + Termination

**Description.** The "goal" is a sampled base-velocity command (lin_x, lin_y, ang_z) the policy must track; there is no positional target — this is a continuous velocity-tracking task. Spot's command ranges are **wider** than the parent's, `heading_command=False` (parent uses heading control), and `rel_standing_envs=0.1`. Episodes terminate on time-out, on illegal contact of the body/legs, or when the robot leaves the terrain bounds.

**Decisions resolved.**
- `CommandsCfg.base_velocity` (`UniformVelocityCommandCfg`): `resampling_time_range=(10,10)`, `rel_standing_envs=0.1`, `rel_heading_envs=0.0`, `heading_command=False`, `debug_vis=True`; ranges `lin_vel_x=(-2.0,3.0)`, `lin_vel_y=(-1.5,1.5)`, `ang_vel_z=(-2.0,2.0)`.
- Terminations: `time_out` (time_out=True); `body_contact` → `illegal_contact` on bodies `["body", ".*leg"]`, threshold 1.0; `terrain_out_of_bounds` → `distance_buffer=3.0`, `time_out=True`.

**Code — commands** (`flat_env_cfg.py`):
```python
@configclass
class SpotCommandsCfg:
    base_velocity = mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=0.1,
        rel_heading_envs=0.0,
        heading_command=False,
        debug_vis=True,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-2.0, 3.0), lin_vel_y=(-1.5, 1.5), ang_vel_z=(-2.0, 2.0)
        ),
    )
```

**Code — terminations** (`flat_env_cfg.py`):
```python
@configclass
class SpotTerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    body_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=["body", ".*leg"]), "threshold": 1.0},
    )
    terrain_out_of_bounds = DoneTerm(
        func=mdp.terrain_out_of_bounds,
        params={"asset_cfg": SceneEntityCfg("robot"), "distance_buffer": 3.0},
        time_out=True,
    )
```

**Smoke (§4).** Stepping with a held action eventually trips a body/leg contact termination or hits time-out; `terminated`/`truncated` flags fire.

---

## §5 Observation

**Description.** One policy observation group, 7 terms concatenated, total **48-dim**: base linear velocity (3) + base angular velocity (3) + projected gravity (3) + velocity command (3) + relative joint positions (12) + joint velocities (12) + last action (12). **No height scan** (Spot disables the scanner in §1). Per-term additive uniform noise is configured, but **corruption is disabled** at this layer (`enable_corruption = False`) — unlike the parent group which enables it.

**Decisions resolved.**
- `concatenate_terms = True`, `enable_corruption = False`.
- Total obs dim = 3+3+3+3+12+12+12 = **48** (12 = Spot joint count).
- Per-term noise (only applied if corruption enabled): base_lin_vel ±0.1, base_ang_vel ±0.1, projected_gravity ±0.05, joint_pos ±0.05, joint_vel ±0.5; velocity_commands and actions: no noise.

**Code** (`flat_env_cfg.py`):
```python
@configclass
class SpotObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        base_lin_vel = ObsTerm(
            func=mdp.base_lin_vel, params={"asset_cfg": SceneEntityCfg("robot")}, noise=Unoise(n_min=-0.1, n_max=0.1)
        )
        base_ang_vel = ObsTerm(
            func=mdp.base_ang_vel, params={"asset_cfg": SceneEntityCfg("robot")}, noise=Unoise(n_min=-0.1, n_max=0.1)
        )
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            params={"asset_cfg": SceneEntityCfg("robot")},
            noise=Unoise(n_min=-0.05, n_max=0.05),
        )
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel, params={"asset_cfg": SceneEntityCfg("robot")}, noise=Unoise(n_min=-0.05, n_max=0.05)
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel, params={"asset_cfg": SceneEntityCfg("robot")}, noise=Unoise(n_min=-0.5, n_max=0.5)
        )
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()
```

**Smoke (§5).** `env.observation_space["policy"].shape == (48,)` (or flat `Box(48,)`).

---

## §6 Reward

**Description.** Spot's reward is the **sum** of 5 task terms and 9 regularization penalties (14 `RewardTermCfg`s), all in `config/spot/mdp/rewards.py`. This is the distinguishing feature of Spot vs the shared velocity reward: instead of `track_lin_vel_xy_exp`/`track_ang_vel_z_exp` + simple L2 penalties, Spot adds **gait enforcement** (`GaitReward`, a `ManagerTermBase` that products sync/async foot-pair contact-timing kernels), **foot air-time shaping** (`air_time_reward`), **foot-clearance** (`foot_clearance_reward`), **foot-slip** and **air-time-variance** penalties, plus command-gated linear/angular velocity tracking with a velocity-magnitude ramp (`base_linear_velocity_reward` scales reward above `ramp_at_vel`). Several terms are **command-gated**: they only apply when `||cmd|| > 0` OR body speed exceeds `velocity_threshold` (0.5), so the policy isn't penalized for standing when commanded to stand. `joint_position_penalty` is the inverse — it is multiplied by `stand_still_scale=5.0` when standing, to hold the default pose.

**Composer: SUM** (IsaacLab `RewardManager` sums all weighted terms).

**Decisions resolved — RewardsCfg terms (func, weight, key params).**
| term | func | weight | key params |
|---|---|---|---|
| air_time | `air_time_reward` | +5.0 | mode_time=0.3, velocity_threshold=0.5, sensor=`.*_foot` |
| base_angular_velocity | `base_angular_velocity_reward` | +5.0 | std=2.0 |
| base_linear_velocity | `base_linear_velocity_reward` | +5.0 | std=1.0, ramp_rate=0.5, ramp_at_vel=1.0 |
| foot_clearance | `foot_clearance_reward` | +0.5 | std=0.05, tanh_mult=2.0, target_height=0.1, body=`.*_foot` |
| gait | `GaitReward` | +10.0 | std=0.1, max_err=0.2, velocity_threshold=0.5, synced_feet_pair_names=((fl,hr),(fr,hl)) |
| action_smoothness | `action_smoothness_penalty` | -1.0 | — |
| air_time_variance | `air_time_variance_penalty` | -1.0 | sensor=`.*_foot` |
| base_motion | `base_motion_penalty` | -2.0 | — |
| base_orientation | `base_orientation_penalty` | -3.0 | — |
| foot_slip | `foot_slip_penalty` | -0.5 | threshold=1.0, body/sensor=`.*_foot` |
| joint_acc | `joint_acceleration_penalty` | -1.0e-4 | joints=`.*_h[xy]` |
| joint_pos | `joint_position_penalty` | -0.7 | stand_still_scale=5.0, velocity_threshold=0.5, joints=`.*` |
| joint_torques | `joint_torques_penalty` | -5.0e-4 | joints=`.*` |
| joint_vel | `joint_velocity_penalty` | -1.0e-2 | joints=`.*_h[xy]` |

**Planning-budget magnitudes (retro-computed, pre-`dt` weights).** Task rewards saturate near their weight: gait → ~+10, air_time/base_lin_vel/base_ang_vel → ~+5 each, foot_clearance → ~+0.5; positive ceiling ≈ +25.5/step. Penalties are unbounded-ish norms but coefficient-suppressed: joint_acc / joint_torques / joint_vel ≈ O(1e-4..1e-2)×norm; action_smoothness / base_motion / base_orientation / foot_slip / air_time_variance are the dominant negatives at well-behaved gaits (O(0.1..few)). Net per-step return at a good trot is positive and dominated by the gait + velocity-tracking terms. (No explicit per-stage docstring in `SpotRewardsCfg`.)

**Code — RewardsCfg** (`flat_env_cfg.py`):
```python
@configclass
class SpotRewardsCfg:
    # -- task
    air_time = RewardTermCfg(
        func=spot_mdp.air_time_reward, weight=5.0,
        params={"mode_time": 0.3, "velocity_threshold": 0.5,
                "asset_cfg": SceneEntityCfg("robot"),
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot")},
    )
    base_angular_velocity = RewardTermCfg(
        func=spot_mdp.base_angular_velocity_reward, weight=5.0,
        params={"std": 2.0, "asset_cfg": SceneEntityCfg("robot")},
    )
    base_linear_velocity = RewardTermCfg(
        func=spot_mdp.base_linear_velocity_reward, weight=5.0,
        params={"std": 1.0, "ramp_rate": 0.5, "ramp_at_vel": 1.0, "asset_cfg": SceneEntityCfg("robot")},
    )
    foot_clearance = RewardTermCfg(
        func=spot_mdp.foot_clearance_reward, weight=0.5,
        params={"std": 0.05, "tanh_mult": 2.0, "target_height": 0.1,
                "asset_cfg": SceneEntityCfg("robot", body_names=".*_foot")},
    )
    gait = RewardTermCfg(
        func=spot_mdp.GaitReward, weight=10.0,
        params={"std": 0.1, "max_err": 0.2, "velocity_threshold": 0.5,
                "synced_feet_pair_names": (("fl_foot", "hr_foot"), ("fr_foot", "hl_foot")),
                "asset_cfg": SceneEntityCfg("robot"),
                "sensor_cfg": SceneEntityCfg("contact_forces")},
    )
    # -- penalties
    action_smoothness = RewardTermCfg(func=spot_mdp.action_smoothness_penalty, weight=-1.0)
    air_time_variance = RewardTermCfg(
        func=spot_mdp.air_time_variance_penalty, weight=-1.0,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot")},
    )
    base_motion = RewardTermCfg(func=spot_mdp.base_motion_penalty, weight=-2.0, params={"asset_cfg": SceneEntityCfg("robot")})
    base_orientation = RewardTermCfg(func=spot_mdp.base_orientation_penalty, weight=-3.0, params={"asset_cfg": SceneEntityCfg("robot")})
    foot_slip = RewardTermCfg(
        func=spot_mdp.foot_slip_penalty, weight=-0.5,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=".*_foot"),
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"), "threshold": 1.0},
    )
    joint_acc = RewardTermCfg(
        func=spot_mdp.joint_acceleration_penalty, weight=-1.0e-4,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*_h[xy]")},
    )
    joint_pos = RewardTermCfg(
        func=spot_mdp.joint_position_penalty, weight=-0.7,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
                "stand_still_scale": 5.0, "velocity_threshold": 0.5},
    )
    joint_torques = RewardTermCfg(
        func=spot_mdp.joint_torques_penalty, weight=-5.0e-4,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*")},
    )
    joint_vel = RewardTermCfg(
        func=spot_mdp.joint_velocity_penalty, weight=-1.0e-2,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*_h[xy]")},
    )
```

**Code — full reward function source** (`config/spot/mdp/rewards.py`, verbatim, imports: `torch`; `from isaaclab.assets import Articulation, RigidObject`; `from isaaclab.managers import ManagerTermBase, SceneEntityCfg`; `from isaaclab.sensors import ContactSensor`):
```python
##
# Task Rewards
##

def air_time_reward(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    sensor_cfg: SceneEntityCfg,
    mode_time: float,
    velocity_threshold: float,
) -> torch.Tensor:
    """Reward longer feet air and contact time."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    asset: Articulation = env.scene[asset_cfg.name]
    if contact_sensor.cfg.track_air_time is False:
        raise RuntimeError("Activate ContactSensor's track_air_time!")
    current_air_time = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids]
    current_contact_time = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids]

    t_max = torch.max(current_air_time, current_contact_time)
    t_min = torch.clip(t_max, max=mode_time)
    stance_cmd_reward = torch.clip(current_contact_time - current_air_time, -mode_time, mode_time)
    cmd = torch.norm(env.command_manager.get_command("base_velocity"), dim=1).unsqueeze(dim=1).expand(-1, 4)
    body_vel = torch.linalg.norm(asset.data.root_lin_vel_b[:, :2], dim=1).unsqueeze(dim=1).expand(-1, 4)
    reward = torch.where(
        torch.logical_or(cmd > 0.0, body_vel > velocity_threshold),
        torch.where(t_max < mode_time, t_min, 0),
        stance_cmd_reward,
    )
    return torch.sum(reward, dim=1)


def base_angular_velocity_reward(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, std: float) -> torch.Tensor:
    """Reward tracking of angular velocity commands (yaw) using abs exponential kernel."""
    asset: RigidObject = env.scene[asset_cfg.name]
    target = env.command_manager.get_command("base_velocity")[:, 2]
    ang_vel_error = torch.linalg.norm((target - asset.data.root_ang_vel_b[:, 2]).unsqueeze(1), dim=1)
    return torch.exp(-ang_vel_error / std)


def base_linear_velocity_reward(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, std: float, ramp_at_vel: float = 1.0, ramp_rate: float = 0.5
) -> torch.Tensor:
    """Reward tracking of linear velocity commands (xy axes) using abs exponential kernel."""
    asset: RigidObject = env.scene[asset_cfg.name]
    target = env.command_manager.get_command("base_velocity")[:, :2]
    lin_vel_error = torch.linalg.norm((target - asset.data.root_lin_vel_b[:, :2]), dim=1)
    vel_cmd_magnitude = torch.linalg.norm(target, dim=1)
    velocity_scaling_multiple = torch.clamp(1.0 + ramp_rate * (vel_cmd_magnitude - ramp_at_vel), min=1.0)
    return torch.exp(-lin_vel_error / std) * velocity_scaling_multiple


class GaitReward(ManagerTermBase):
    """Gait enforcing reward term for quadrupeds.

    Penalizes contact timing differences between selected foot pairs (synced_feet_pair_names)
    to bias the policy towards a desired gait (trotting/bounding/pacing). Only supports
    quadrupedal gaits with two pairs of synchronized feet.
    """

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.std: float = cfg.params["std"]
        self.max_err: float = cfg.params["max_err"]
        self.velocity_threshold: float = cfg.params["velocity_threshold"]
        self.contact_sensor: ContactSensor = env.scene.sensors[cfg.params["sensor_cfg"].name]
        self.asset: Articulation = env.scene[cfg.params["asset_cfg"].name]
        synced_feet_pair_names = cfg.params["synced_feet_pair_names"]
        if (
            len(synced_feet_pair_names) != 2
            or len(synced_feet_pair_names[0]) != 2
            or len(synced_feet_pair_names[1]) != 2
        ):
            raise ValueError("This reward only supports gaits with two pairs of synchronized feet, like trotting.")
        synced_feet_pair_0 = self.contact_sensor.find_bodies(synced_feet_pair_names[0])[0]
        synced_feet_pair_1 = self.contact_sensor.find_bodies(synced_feet_pair_names[1])[0]
        self.synced_feet_pairs = [synced_feet_pair_0, synced_feet_pair_1]

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        std: float,
        max_err: float,
        velocity_threshold: float,
        synced_feet_pair_names,
        asset_cfg: SceneEntityCfg,
        sensor_cfg: SceneEntityCfg,
    ) -> torch.Tensor:
        sync_reward_0 = self._sync_reward_func(self.synced_feet_pairs[0][0], self.synced_feet_pairs[0][1])
        sync_reward_1 = self._sync_reward_func(self.synced_feet_pairs[1][0], self.synced_feet_pairs[1][1])
        sync_reward = sync_reward_0 * sync_reward_1
        async_reward_0 = self._async_reward_func(self.synced_feet_pairs[0][0], self.synced_feet_pairs[1][0])
        async_reward_1 = self._async_reward_func(self.synced_feet_pairs[0][1], self.synced_feet_pairs[1][1])
        async_reward_2 = self._async_reward_func(self.synced_feet_pairs[0][0], self.synced_feet_pairs[1][1])
        async_reward_3 = self._async_reward_func(self.synced_feet_pairs[1][0], self.synced_feet_pairs[0][1])
        async_reward = async_reward_0 * async_reward_1 * async_reward_2 * async_reward_3
        cmd = torch.norm(env.command_manager.get_command("base_velocity"), dim=1)
        body_vel = torch.linalg.norm(self.asset.data.root_lin_vel_b[:, :2], dim=1)
        return torch.where(
            torch.logical_or(cmd > 0.0, body_vel > self.velocity_threshold), sync_reward * async_reward, 0.0
        )

    def _sync_reward_func(self, foot_0: int, foot_1: int) -> torch.Tensor:
        air_time = self.contact_sensor.data.current_air_time
        contact_time = self.contact_sensor.data.current_contact_time
        se_air = torch.clip(torch.square(air_time[:, foot_0] - air_time[:, foot_1]), max=self.max_err**2)
        se_contact = torch.clip(torch.square(contact_time[:, foot_0] - contact_time[:, foot_1]), max=self.max_err**2)
        return torch.exp(-(se_air + se_contact) / self.std)

    def _async_reward_func(self, foot_0: int, foot_1: int) -> torch.Tensor:
        air_time = self.contact_sensor.data.current_air_time
        contact_time = self.contact_sensor.data.current_contact_time
        se_act_0 = torch.clip(torch.square(air_time[:, foot_0] - contact_time[:, foot_1]), max=self.max_err**2)
        se_act_1 = torch.clip(torch.square(contact_time[:, foot_0] - air_time[:, foot_1]), max=self.max_err**2)
        return torch.exp(-(se_act_0 + se_act_1) / self.std)


def foot_clearance_reward(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, target_height: float, std: float, tanh_mult: float
) -> torch.Tensor:
    """Reward the swinging feet for clearing a specified height off the ground"""
    asset: RigidObject = env.scene[asset_cfg.name]
    foot_z_target_error = torch.square(asset.data.body_pos_w[:, asset_cfg.body_ids, 2] - target_height)
    foot_velocity_tanh = torch.tanh(tanh_mult * torch.norm(asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2], dim=2))
    reward = foot_z_target_error * foot_velocity_tanh
    return torch.exp(-torch.sum(reward, dim=1) / std)


##
# Regularization Penalties
##

def action_smoothness_penalty(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Penalize large instantaneous changes in the network action output"""
    return torch.linalg.norm((env.action_manager.action - env.action_manager.prev_action), dim=1)


def air_time_variance_penalty(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize variance in the amount of time each foot spends in the air/on the ground relative to each other"""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    if contact_sensor.cfg.track_air_time is False:
        raise RuntimeError("Activate ContactSensor's track_air_time!")
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    last_contact_time = contact_sensor.data.last_contact_time[:, sensor_cfg.body_ids]
    return torch.var(torch.clip(last_air_time, max=0.5), dim=1) + torch.var(
        torch.clip(last_contact_time, max=0.5), dim=1
    )


def base_motion_penalty(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize base vertical and roll/pitch velocity"""
    asset: RigidObject = env.scene[asset_cfg.name]
    return 0.8 * torch.square(asset.data.root_lin_vel_b[:, 2]) + 0.2 * torch.sum(
        torch.abs(asset.data.root_ang_vel_b[:, :2]), dim=1
    )


def base_orientation_penalty(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize non-flat base orientation (xy-components of projected gravity)."""
    asset: RigidObject = env.scene[asset_cfg.name]
    return torch.linalg.norm((asset.data.projected_gravity_b[:, :2]), dim=1)


def foot_slip_penalty(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, sensor_cfg: SceneEntityCfg, threshold: float
) -> torch.Tensor:
    """Penalize foot planar (xy) slip when in contact with the ground"""
    asset: RigidObject = env.scene[asset_cfg.name]
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_contact_forces = contact_sensor.data.net_forces_w_history
    is_contact = torch.max(torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1)[0] > threshold
    foot_planar_velocity = torch.linalg.norm(asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2], dim=2)
    reward = is_contact * foot_planar_velocity
    return torch.sum(reward, dim=1)


def joint_acceleration_penalty(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize joint accelerations on the articulation."""
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.linalg.norm((asset.data.joint_acc), dim=1)


def joint_position_penalty(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, stand_still_scale: float, velocity_threshold: float
) -> torch.Tensor:
    """Penalize joint position error from default on the articulation."""
    asset: Articulation = env.scene[asset_cfg.name]
    cmd = torch.linalg.norm(env.command_manager.get_command("base_velocity"), dim=1)
    body_vel = torch.linalg.norm(asset.data.root_lin_vel_b[:, :2], dim=1)
    reward = torch.linalg.norm((asset.data.joint_pos - asset.data.default_joint_pos), dim=1)
    return torch.where(torch.logical_or(cmd > 0.0, body_vel > velocity_threshold), reward, stand_still_scale * reward)


def joint_torques_penalty(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize joint torques on the articulation."""
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.linalg.norm((asset.data.applied_torque), dim=1)


def joint_velocity_penalty(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize joint velocities on the articulation."""
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.linalg.norm((asset.data.joint_vel), dim=1)
```

**Smoke (§6).** Reward is finite and non-constant over a random rollout; with per-term logging, `sum(detailed_reward.values()) == reward` each step (composer = sum).

---

## §7 DR

**Description.** Domain randomization comes from the non-reset `EventCfg` terms: two **startup** terms (per-bucket rigid-body friction/restitution material, and additive base-body mass), and one **interval** term (random velocity push every 10–15 s). Note this Spot env has **no** `base_com` startup randomization that the parent locomotion env includes; Spot's friction ranges and base-mass range also differ from the parent.

**Decisions resolved.**
- `physics_material` (startup): static_friction (0.3,1.0), dynamic_friction (0.3,0.8), restitution (0.0,0.0), num_buckets=64, body=`.*`.
- `add_base_mass` (startup): body=`body`, mass add range (-2.5, 2.5).
- `push_robot` (interval, 10–15 s): velocity push x/y (-0.5, 0.5).

**Code** (`flat_env_cfg.py`, the `mode in {startup, interval}` terms of `SpotEventCfg`):
```python
# startup
physics_material = EventTerm(
    func=mdp.randomize_rigid_body_material,
    mode="startup",
    params={
        "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
        "static_friction_range": (0.3, 1.0),
        "dynamic_friction_range": (0.3, 0.8),
        "restitution_range": (0.0, 0.0),
        "num_buckets": 64,
    },
)

add_base_mass = EventTerm(
    func=mdp.randomize_rigid_body_mass,
    mode="startup",
    params={
        "asset_cfg": SceneEntityCfg("robot", body_names="body"),
        "mass_distribution_params": (-2.5, 2.5),
        "operation": "add",
    },
)

# interval
push_robot = EventTerm(
    func=mdp.push_by_setting_velocity,
    mode="interval",
    interval_range_s=(10.0, 15.0),
    params={
        "asset_cfg": SceneEntityCfg("robot"),
        "velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)},
    },
)
```

**Smoke (§7).** Seed-matched obs trajectories with DR ON vs OFF diverge (startup material/mass + interval push perturb dynamics).

---

