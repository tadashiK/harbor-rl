# Isaac-Velocity-Rough-H1-v0 — Implementation Spec

- robot: Unitree H1 bipedal humanoid (19 DoF)
- simulator: IsaacLab (Isaac Sim, manager-based)
- objects: none (procedurally generated rough terrain)
- bimanual: false
- summary: Track a commanded base velocity across procedurally generated rough terrain.

> **Task in one line:** Unitree H1 humanoid (19 DoF) tracks a commanded base velocity (lin_vel_x, lin_vel_y, ang_vel_z) while walking over procedurally generated ROUGH terrain (stairs / boxes / slopes / random rough), using a torso-mounted height-scan ray-caster + a terrain-level curriculum.

> **DELTA vs `Isaac-Velocity-Flat-H1-v0` (the flat sibling).** Same robot, same actions, same command/termination structure. Rough adds, on top of flat:
> 1. **Terrain** — `terrain_type="generator"` with `ROUGH_TERRAINS_CFG` (flat uses `terrain_type="plane"`, `terrain_generator=None`).
> 2. **Height-scan sensor** — `height_scanner` ray-caster present (flat sets `self.scene.height_scanner = None`).
> 3. **`height_scan` obs term** — present (flat sets `self.observations.policy.height_scan = None`). This is the only obs difference: **+187 dims** → rough obs = 256, flat obs = 69.
> 4. **Terrain curriculum** — `curriculum.terrain_levels = terrain_levels_vel` active (flat sets `self.curriculum.terrain_levels = None`, which also forces `terrain_generator.curriculum = False`).
> 5. **Reward retune** — `feet_air_time.weight = 0.25, threshold = 0.4` on rough; flat overrides to `weight = 1.0, threshold = 0.6`. (All other reward weights are identical between flat and rough — they share the same `H1Rewards` class via inheritance; flat only patches `feet_air_time`.)
>
> Both configs share the **same** H1-specific reward retunes vs the generic `LocomotionVelocityRoughEnvCfg.RewardsCfg` base (see §6).

---

## §1 Registration + Scene

**Description.** Registers the rough H1 velocity-tracking env under `Isaac-Velocity-Rough-H1-v0`, entry point `isaaclab.envs:ManagerBasedRLEnv`, env cfg `H1RoughEnvCfg`. The scene is a generator-based rough terrain, the `H1_MINIMAL_CFG` articulation (19 DoF humanoid), a torso-mounted height-scan ray-caster, an all-bodies contact sensor, and a dome sky light.

**Decisions resolved.**
- `id="Isaac-Velocity-Rough-H1-v0"`, `entry_point="isaaclab.envs:ManagerBasedRLEnv"`, `disable_env_checker=True`.
- `env_cfg_entry_point = ...rough_env_cfg:H1RoughEnvCfg`; `rsl_rl_cfg_entry_point = ...agents.rsl_rl_ppo_cfg:H1RoughPPORunnerCfg`; `skrl_cfg_entry_point = "skrl_rough_ppo_cfg.yaml"`.
- Robot articulation: `H1_MINIMAL_CFG` re-parented to `{ENV_REGEX_NS}/Robot`. USD resolved: `{ISAACLAB_NUCLEUS_DIR}/Robots/Unitree/H1/h1_minimal.usd`. init pos `(0,0,1.05)`. 19 DoF: `.*_hip_yaw`, `.*_hip_roll`, `.*_hip_pitch`, `.*_knee`, `.*_ankle` (×2 L/R = 10) + `torso` (1) + `.*_shoulder_pitch/roll/yaw`, `.*_elbow` (8) = 19. Default joint_pos: hip_pitch −0.28, knee 0.79, ankle −0.52, shoulder_pitch 0.28, elbow 0.52, rest 0.0. `soft_joint_pos_limit_factor=0.9`. Implicit actuators (legs+torso k≈150–200/d=5, feet k=20/d=4, arms k=40/d=10).
- Terrain: `TerrainImporterCfg`, `terrain_type="generator"`, `terrain_generator=ROUGH_TERRAINS_CFG`, `max_init_terrain_level=5`, friction static/dynamic = 1.0/1.0, marble visual material. (`terrain_generator.curriculum` set to `True` in base `__post_init__` because `curriculum.terrain_levels` is not None.)
- Height-scan sensor: `RayCasterCfg`, prim_path re-pointed in `H1RoughEnvCfg.__post_init__` to `{ENV_REGEX_NS}/Robot/torso_link` (base default is `/Robot/base`), `offset.pos=(0,0,20.0)`, `ray_alignment="yaw"`, `GridPatternCfg(resolution=0.1, size=[1.6, 1.0])` → **187 rays (17×11)**, casts onto `/World/ground`. `update_period = decimation*dt = 0.02`.
- Contact sensor: `ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True)`, `update_period = sim.dt = 0.005`.
- Sky light: `DomeLightCfg(intensity=750, texture=...kloofendal_43d_clear_puresky_4k.hdr)`.
- Sim: `decimation=4`, `episode_length_s=20.0`, `sim.dt=0.005` (→ control dt 0.02, 50 Hz), `num_envs=4096`, `env_spacing=2.5`, `gpu_max_rigid_patch_count=10*2**15`.

**Resolved asset paths.**
- `{ISAACLAB_NUCLEUS_DIR}/Robots/Unitree/H1/h1_minimal.usd` (robot USD — Nucleus, not on local disk)
- `{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/...mdl` (terrain visual material)
- `{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr` (sky)
- `ROUGH_TERRAINS_CFG`: `isaaclab.terrains.config.rough` (see §7 for verbatim).

**Code — registration** (`config/h1/__init__.py:14-23`):
```python
gym.register(
    id="Isaac-Velocity-Rough-H1-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rough_env_cfg:H1RoughEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:H1RoughPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_rough_ppo_cfg.yaml",
    },
)
```

**Code — Scene** (`velocity_env_cfg.py:39-82`):
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

**Code — robot wiring** (`rough_env_cfg.py:77-80`, from `H1RoughEnvCfg.__post_init__`):
```python
self.scene.robot = H1_MINIMAL_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
if self.scene.height_scanner:
    self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/torso_link"
```

**Code — H1_MINIMAL_CFG** (`isaaclab_assets/robots/unitree.py:264-269`, derives from `H1_CFG:`):
```python
H1_MINIMAL_CFG = H1_CFG.copy()
H1_MINIMAL_CFG.spawn.usd_path = f"{ISAACLAB_NUCLEUS_DIR}/Robots/Unitree/H1/h1_minimal.usd"
```
`H1_CFG` (same file, abbreviated to load-bearing fields):
```python
H1_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAACLAB_NUCLEUS_DIR}/Robots/Unitree/H1/h1.usd",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False, retain_accelerations=False, linear_damping=0.0, angular_damping=0.0,
            max_linear_velocity=1000.0, max_angular_velocity=1000.0, max_depenetration_velocity=1.0),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False, solver_position_iteration_count=4, solver_velocity_iteration_count=4),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.05),
        joint_pos={
            ".*_hip_yaw": 0.0, ".*_hip_roll": 0.0, ".*_hip_pitch": -0.28, ".*_knee": 0.79, ".*_ankle": -0.52,
            "torso": 0.0, ".*_shoulder_pitch": 0.28, ".*_shoulder_roll": 0.0, ".*_shoulder_yaw": 0.0, ".*_elbow": 0.52,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[".*_hip_yaw", ".*_hip_roll", ".*_hip_pitch", ".*_knee", "torso"],
            effort_limit_sim=300,
            stiffness={".*_hip_yaw": 150.0, ".*_hip_roll": 150.0, ".*_hip_pitch": 200.0, ".*_knee": 200.0, "torso": 200.0},
            damping={".*_hip_yaw": 5.0, ".*_hip_roll": 5.0, ".*_hip_pitch": 5.0, ".*_knee": 5.0, "torso": 5.0}),
        "feet": ImplicitActuatorCfg(
            joint_names_expr=[".*_ankle"], effort_limit_sim=100,
            stiffness={".*_ankle": 20.0}, damping={".*_ankle": 4.0}),
        "arms": ImplicitActuatorCfg(
            joint_names_expr=[".*_shoulder_pitch", ".*_shoulder_roll", ".*_shoulder_yaw", ".*_elbow"],
            effort_limit_sim=300,
            stiffness={".*_shoulder_pitch": 40.0, ".*_shoulder_roll": 40.0, ".*_shoulder_yaw": 40.0, ".*_elbow": 40.0},
            damping={".*_shoulder_pitch": 10.0, ".*_shoulder_roll": 10.0, ".*_shoulder_yaw": 10.0, ".*_elbow": 10.0}),
    },
)
```

**Smoke** (§1 build — best-effort, headless):
```bash
cd <IsaacLab-repo>
.venv/bin/python -c "import gymnasium as gym; env=gym.make('Isaac-Velocity-Rough-H1-v0'); print(env.observation_space, env.action_space); env.close()"
```
Expected (when Isaac sim deps available): `Box(..., (256,) ...) Box(-inf, inf, (19,) ...)`.
Observed in this probe environment: `gymnasium.error.NameNotFound` (Isaac `pxr` unavailable → registration module never imports). Analytic dims confirmed by summing obs terms (§5).

---

## §2 Actions

**Description.** Single action term: direct joint-position targets for all 19 joints, scaled and offset from the default pose. No IK, no gripper.

**Decisions resolved.** `joint_pos = JointPositionActionCfg(asset_name="robot", joint_names=[".*"], scale=0.5, use_default_offset=True)`. Action dim = **19**. Target = default_joint_pos + 0.5 * action.

**Code** (`velocity_env_cfg.py:108-112`):
```python
@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    joint_pos = mdp.JointPositionActionCfg(asset_name="robot", joint_names=[".*"], scale=0.5, use_default_offset=True)
```
(Unchanged in H1 rough — no `__post_init__` override of actions.)

**Smoke** (§2): step the env with a zero action; assert action_space shape `(19,)` and the env steps without raising.

---

## §3 Reset

**Description.** On reset, the base pose is randomized (±0.5 m in x/y, full yaw), base velocity is fixed to zero (H1 override), joints are set exactly to default (scale (1.0, 1.0) — H1 override), and a zero external force/torque term is registered (on `torso_link` per H1 override).

**Decisions resolved (after `H1RoughEnvCfg.__post_init__` overrides).**
- `reset_base`: `pose_range={"x":(-0.5,0.5),"y":(-0.5,0.5),"yaw":(-3.14,3.14)}`, `velocity_range` all zeros (H1 overrides the base's ±0.5 velocity ranges to 0).
- `reset_robot_joints`: `reset_joints_by_scale`, `position_range=(1.0, 1.0)` (H1 override of base (0.5,1.5)), `velocity_range=(0.0,0.0)`.
- `base_external_force_torque`: `apply_external_force_torque` on `.*torso_link` (H1 override of base `body_names="base"`), `force_range=(0,0)`, `torque_range=(0,0)`.
- H1 disables base startup DR: `add_base_mass=None`, `base_com=None` (see §7).

**Code — base EventCfg reset terms** (`velocity_env_cfg.py:185-219`):
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
                "x": (-0.5, 0.5), "y": (-0.5, 0.5), "z": (-0.5, 0.5),
                "roll": (-0.5, 0.5), "pitch": (-0.5, 0.5), "yaw": (-0.5, 0.5),
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

**Code — H1 reset overrides** (`rough_env_cfg.py:83-98`):
```python
        self.events.push_robot = None
        self.events.add_base_mass = None
        self.events.reset_robot_joints.params["position_range"] = (1.0, 1.0)
        self.events.base_external_force_torque.params["asset_cfg"].body_names = [".*torso_link"]
        self.events.reset_base.params = {
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0),
                "roll": (0.0, 0.0), "pitch": (0.0, 0.0), "yaw": (0.0, 0.0),
            },
        }
        self.events.base_com = None
```

**Smoke** (§3): reset twice with different seeds; assert base xy/yaw differ across resets but joints land on default pose (position scale = 1.0).

---

## §4 Goal + Termination

**Description.** Velocity-command-following task: there is no terminal success state. The episode ends on (a) time-out at 20 s, or (b) illegal contact on the torso (the robot fell). The command is a continuously resampled base-velocity target (heading-controlled).

**Decisions resolved.**
- `time_out`: `mdp.time_out`, `time_out=True` (20 s).
- `base_contact`: `mdp.illegal_contact`, `sensor_cfg.body_names=".*torso_link"` (H1 override of base `"base"`), `threshold=1.0`.
- Commands: `base_velocity = UniformVelocityCommandCfg`, `resampling_time_range=(10,10)`, `rel_standing_envs=0.02`, `rel_heading_envs=1.0`, `heading_command=True`, `heading_control_stiffness=0.5`. H1 ranges: `lin_vel_x=(0.0,1.0)`, `lin_vel_y=(0.0,0.0)`, `ang_vel_z=(-1.0,1.0)` (base default ranges are symmetric ±1.0 on x/y plus heading=±π; H1 narrows x to forward-only and zeros y).

**Code — base TerminationsCfg** (`velocity_env_cfg.py:266-274`):
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

**Code — base CommandsCfg** (`velocity_env_cfg.py:90-105`):
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

**Code — H1 termination + command overrides** (`rough_env_cfg.py:108-113`):
```python
        # Commands
        self.commands.base_velocity.ranges.lin_vel_x = (0.0, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)

        # Terminations
        self.terminations.base_contact.params["sensor_cfg"].body_names = ".*torso_link"
```

**Smoke** (§4): roll out with a falling policy; assert `terminated` flips and `time_outs` flips at ~1000 steps (20 s / 0.02).

---

## §5 Observation

**Description.** Single policy group, terms concatenated, corruption (noise) enabled. Proprioception + command + last action + a 187-ray downward height scan of the terrain under the torso.

**Decisions resolved — term-by-term dims (rough):**
| term | func | dim | noise / clip |
|---|---|---|---|
| base_lin_vel | `base_lin_vel` | 3 | Unoise ±0.1 |
| base_ang_vel | `base_ang_vel` | 3 | Unoise ±0.2 |
| projected_gravity | `projected_gravity` | 3 | Unoise ±0.05 |
| velocity_commands | `generated_commands(base_velocity)` | 3 | — |
| joint_pos | `joint_pos_rel` | 19 | Unoise ±0.01 |
| joint_vel | `joint_vel_rel` | 19 | Unoise ±1.5 |
| actions | `last_action` | 19 | — |
| height_scan | `height_scan(height_scanner)` | **187** | Unoise ±0.1, clip (−1,1) |

**Total obs dim = 3+3+3+3+19+19+19+187 = 256.** `enable_corruption=True`, `concatenate_terms=True`.
**Flat-H1 delta:** `height_scan` removed → **69**. (Flat = 256 − 187 = 69.)

**Code — base ObservationsCfg** (`velocity_env_cfg.py:115-146`):
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
(H1 rough does not override obs. Flat-H1 sets `self.observations.policy.height_scan = None` — that is the only obs difference.)

**Smoke** (§5): reset, read obs, assert `obs["policy"].shape[-1] == 256` and finite.

---

## §6 Reward

**Description.** Sum-composed reward (IsaacLab `RewardManager` adds all weighted terms). Two positive tracking terms (xy lin-vel in yaw frame, yaw ang-vel in world frame), a biped air-time bonus, and a battery of penalties: termination, feet slide, ankle joint-limit, hip/arm/torso default-pose deviation, plus inherited base penalties (z lin-vel, xy ang-vel, dof acc, action rate, flat orientation). H1 zeroes `dof_torques_l2` and `undesired_contacts`.

**Composer: SUM.**

**Decisions resolved — final RewTerm table (after H1 overrides + inherited base, rough):**
| term | func | weight | params |
|---|---|---|---|
| termination_penalty | `mdp.is_terminated` | **−200.0** | — |
| track_lin_vel_xy_exp | `mdp.track_lin_vel_xy_yaw_frame_exp` | **1.0** | command_name=base_velocity, std=0.5 |
| track_ang_vel_z_exp | `mdp.track_ang_vel_z_world_exp` | **1.0** | command_name=base_velocity, std=0.5 |
| feet_air_time | `mdp.feet_air_time_positive_biped` | **0.25** | base_velocity, ankle_link, threshold=0.4 |
| feet_slide | `mdp.feet_slide` | **−0.25** | sensor/asset = ankle_link |
| dof_pos_limits | `mdp.joint_pos_limits` | **−1.0** | joints `.*_ankle` |
| joint_deviation_hip | `mdp.joint_deviation_l1` | **−0.2** | `.*_hip_yaw`, `.*_hip_roll` |
| joint_deviation_arms | `mdp.joint_deviation_l1` | **−0.2** | `.*_shoulder_.*`, `.*_elbow` |
| joint_deviation_torso | `mdp.joint_deviation_l1` | **−0.1** | `torso` |
| lin_vel_z_l2 | (base) — | **None** (disabled: H1 sets `lin_vel_z_l2 = None`) | — |
| ang_vel_xy_l2 | `mdp.ang_vel_xy_l2` (base) | **−0.05** | — |
| dof_torques_l2 | `mdp.joint_torques_l2` (base) | **0.0** (H1 override) | — |
| dof_acc_l2 | `mdp.joint_acc_l2` (base) | **−1.25e-7** (H1 override of base −2.5e-7) | — |
| action_rate_l2 | `mdp.action_rate_l2` (base) | **−0.005** (H1 override of base −0.01) | — |
| undesired_contacts | (base) — | **None** (H1 sets to None) | — |
| flat_orientation_l2 | `mdp.flat_orientation_l2` (base) | **−1.0** (H1 override of base 0.0) | — |
| dof_pos_limits (base default term) | `mdp.joint_pos_limits` | overridden — H1's `dof_pos_limits` replaces it (ankle-only, −1.0) | — |

> Note: H1's `H1Rewards` subclasses base `RewardsCfg`. It re-declares `dof_pos_limits`, `track_lin_vel_xy_exp`, `track_ang_vel_z_exp`, `feet_air_time` (rough: w=0.25,thr=0.4 — different funcs than base!), and adds `termination_penalty`, `feet_slide`, the three `joint_deviation_*`. Base `feet_air_time` used `mdp.feet_air_time` (L2, `.*FOOT`); H1 replaces it with `mdp.feet_air_time_positive_biped` on `.*ankle_link`. Base `track_*` use body-frame funcs; H1 replaces with yaw-frame / world-frame variants. The `lin_vel_z_l2`, `undesired_contacts` base terms are set to `None`.

**Planning-budget (retro-computed per-step saturated nominal magnitudes).**
- track_lin_vel_xy_exp: max 1.0/step at perfect tracking.
- track_ang_vel_z_exp: max 1.0/step.
- feet_air_time: clamped to threshold 0.4, w=0.25 → ≤0.1/step when commanded.
- termination_penalty: is_terminated (1.0 on a fall) × −200 → −200 one-shot at the terminating step (large dominant negative).
- penalties (deviation/slide/orientation/action_rate/acc) are small shaping terms keeping the sum dominated by the two tracking exps under nominal walking.

**Code — H1 reward class** (`rough_env_cfg.py:19-67`):
```python
@configclass
class H1Rewards(RewardsCfg):
    """Reward terms for the MDP."""

    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-200.0)
    lin_vel_z_l2 = None
    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_yaw_frame_exp,
        weight=1.0,
        params={"command_name": "base_velocity", "std": 0.5},
    )
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_world_exp, weight=1.0, params={"command_name": "base_velocity", "std": 0.5}
    )
    feet_air_time = RewTerm(
        func=mdp.feet_air_time_positive_biped,
        weight=0.25,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*ankle_link"),
            "threshold": 0.4,
        },
    )
    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-0.25,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*ankle_link"),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*ankle_link"),
        },
    )
    # Penalize ankle joint limits
    dof_pos_limits = RewTerm(
        func=mdp.joint_pos_limits, weight=-1.0, params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*_ankle")}
    )
    # Penalize deviation from default of the joints that are not essential for locomotion
    joint_deviation_hip = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.2,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_yaw", ".*_hip_roll"])},
    )
    joint_deviation_arms = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.2,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_shoulder_.*", ".*_elbow"])},
    )
    joint_deviation_torso = RewTerm(
        func=mdp.joint_deviation_l1, weight=-0.1, params={"asset_cfg": SceneEntityCfg("robot", joint_names="torso")}
    )
```

**Code — H1 reward weight overrides in `__post_init__`** (`rough_env_cfg.py:100-105`):
```python
        # Rewards
        self.rewards.undesired_contacts = None
        self.rewards.flat_orientation_l2.weight = -1.0
        self.rewards.dof_torques_l2.weight = 0.0
        self.rewards.action_rate_l2.weight = -0.005
        self.rewards.dof_acc_l2.weight = -1.25e-7
```

**Code — base RewardsCfg** (inherited; `velocity_env_cfg.py:230-263`):
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

**Code — task-local reward funcs** (`velocity/mdp/rewards.py`, VERBATIM, full file):
```python
from __future__ import annotations
from typing import TYPE_CHECKING
import torch
from isaaclab.envs import mdp
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils.math import quat_apply_inverse, yaw_quat

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def feet_air_time(env, command_name, sensor_cfg, threshold) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    first_contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    reward = torch.sum((last_air_time - threshold) * first_contact, dim=1)
    reward *= torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.1
    return reward


def feet_air_time_positive_biped(env, command_name, threshold, sensor_cfg) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    air_time = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids]
    contact_time = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids]
    in_contact = contact_time > 0.0
    in_mode_time = torch.where(in_contact, contact_time, air_time)
    single_stance = torch.sum(in_contact.int(), dim=1) == 1
    reward = torch.min(torch.where(single_stance.unsqueeze(-1), in_mode_time, 0.0), dim=1)[0]
    reward = torch.clamp(reward, max=threshold)
    reward *= torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.1
    return reward


def feet_slide(env, sensor_cfg, asset_cfg=SceneEntityCfg("robot")) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contacts = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0] > 1.0
    asset = env.scene[asset_cfg.name]
    body_vel = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2]
    reward = torch.sum(body_vel.norm(dim=-1) * contacts, dim=1)
    return reward


def track_lin_vel_xy_yaw_frame_exp(env, std, command_name, asset_cfg=SceneEntityCfg("robot")) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    vel_yaw = quat_apply_inverse(yaw_quat(asset.data.root_quat_w), asset.data.root_lin_vel_w[:, :3])
    lin_vel_error = torch.sum(
        torch.square(env.command_manager.get_command(command_name)[:, :2] - vel_yaw[:, :2]), dim=1)
    return torch.exp(-lin_vel_error / std**2)


def track_ang_vel_z_world_exp(env, command_name, std, asset_cfg=SceneEntityCfg("robot")) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    ang_vel_error = torch.square(env.command_manager.get_command(command_name)[:, 2] - asset.data.root_ang_vel_w[:, 2])
    return torch.exp(-ang_vel_error / std**2)


def stand_still_joint_deviation_l1(env, command_name, command_threshold=0.06, asset_cfg=SceneEntityCfg("robot")) -> torch.Tensor:
    command = env.command_manager.get_command(command_name)
    return mdp.joint_deviation_l1(env, asset_cfg) * (torch.norm(command[:, :2], dim=1) < command_threshold)
```

**Code — inherited shared reward funcs** (`isaaclab/envs/mdp/rewards.py`, the ones referenced by §6, VERBATIM):
```python
def is_terminated(env) -> torch.Tensor:
    """Penalize terminated episodes that don't correspond to episodic timeouts."""
    return env.termination_manager.terminated.float()


def lin_vel_z_l2(env, asset_cfg=SceneEntityCfg("robot")) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    return torch.square(asset.data.root_lin_vel_b[:, 2])


def ang_vel_xy_l2(env, asset_cfg=SceneEntityCfg("robot")) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.root_ang_vel_b[:, :2]), dim=1)


def flat_orientation_l2(env, asset_cfg=SceneEntityCfg("robot")) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.projected_gravity_b[:, :2]), dim=1)


def joint_torques_l2(env, asset_cfg=SceneEntityCfg("robot")) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.applied_torque[:, asset_cfg.joint_ids]), dim=1)


def joint_acc_l2(env, asset_cfg=SceneEntityCfg("robot")) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.joint_acc[:, asset_cfg.joint_ids]), dim=1)


def joint_deviation_l1(env, asset_cfg=SceneEntityCfg("robot")) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    angle = asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    return torch.sum(torch.abs(angle), dim=1)


def joint_pos_limits(env, asset_cfg=SceneEntityCfg("robot")) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    out_of_limits = -(
        asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.soft_joint_pos_limits[:, asset_cfg.joint_ids, 0]).clip(max=0.0)
    out_of_limits += (
        asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.soft_joint_pos_limits[:, asset_cfg.joint_ids, 1]).clip(min=0.0)
    return torch.sum(out_of_limits, dim=1)


def action_rate_l2(env) -> torch.Tensor:
    return torch.sum(torch.square(env.action_manager.action - env.action_manager.prev_action), dim=1)


def undesired_contacts(env, threshold, sensor_cfg) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_contact_forces = contact_sensor.data.net_forces_w_history
    is_contact = torch.max(torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1)[0] > threshold
    return torch.sum(is_contact, dim=1)


def track_lin_vel_xy_exp(env, std, command_name, asset_cfg=SceneEntityCfg("robot")) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    lin_vel_error = torch.sum(
        torch.square(env.command_manager.get_command(command_name)[:, :2] - asset.data.root_lin_vel_b[:, :2]), dim=1)
    return torch.exp(-lin_vel_error / std**2)


def track_ang_vel_z_exp(env, std, command_name, asset_cfg=SceneEntityCfg("robot")) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    ang_vel_error = torch.square(env.command_manager.get_command(command_name)[:, 2] - asset.data.root_ang_vel_b[:, 2])
    return torch.exp(-ang_vel_error / std**2)
```

**Smoke** (§6): roll out 64 steps with random actions; assert per-env summed reward is finite and non-constant across steps. If a per-term reward log is added, assert `sum(detailed_reward.values()) == reward` per step (composer = sum).

---

## §7 DR (Domain Randomization + Curriculum)

**Description.** Startup DR randomizes per-body friction (note: ranges are degenerate — single value, so effectively a fixed friction set, not randomized). Reset-mode events (base pose / joints) are §3, not DR. The interval `push_robot` term and the `add_base_mass` / `base_com` startup terms are **disabled** by H1. The rough-specific element is the **terrain-level curriculum** `terrain_levels_vel`, which moves each env up/down terrain difficulty based on distance walked vs commanded.

**Decisions resolved.**
- **Startup DR (active):** `physics_material` — `randomize_rigid_body_material` on all bodies, `static_friction_range=(0.8,0.8)`, `dynamic_friction_range=(0.6,0.6)`, `restitution_range=(0.0,0.0)`, `num_buckets=64`. (Degenerate ranges → fixed, not stochastic.)
- **Startup DR (disabled by H1):** `add_base_mass = None`, `base_com = None`.
- **Interval DR (disabled by H1):** `push_robot = None` (base would push every 10–15 s with ±0.5 m/s xy velocity).
- **Curriculum (rough-only):** `terrain_levels = CurrTerm(func=mdp.terrain_levels_vel)`. Active because terrain is generator-based; flat-H1 sets it to `None`.

**Code — base startup/interval EventCfg** (`velocity_env_cfg.py:149-227`, the non-reset terms):
```python
@configclass
class EventCfg:
    """Configuration for events."""

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

    # ... (reset terms are in §3) ...

    # interval
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(10.0, 15.0),
        params={"velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)}},
    )
```

**Code — H1 DR disable overrides** (`rough_env_cfg.py:82-98`, the DR-relevant lines):
```python
        # Randomization
        self.events.push_robot = None
        self.events.add_base_mass = None
        ...
        self.events.base_com = None
```

**Code — CurriculumCfg + terrain_levels_vel** (`velocity_env_cfg.py:277-281` + `velocity/mdp/curriculums.py`, VERBATIM):
```python
@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""

    terrain_levels = CurrTerm(func=mdp.terrain_levels_vel)
```
```python
def terrain_levels_vel(env, env_ids, asset_cfg=SceneEntityCfg("robot")) -> torch.Tensor:
    """Curriculum based on the distance the robot walked when commanded to move at a desired velocity."""
    asset: Articulation = env.scene[asset_cfg.name]
    terrain: TerrainImporter = env.scene.terrain
    command = env.command_manager.get_command("base_velocity")
    distance = torch.norm(asset.data.root_pos_w[env_ids, :2] - env.scene.env_origins[env_ids, :2], dim=1)
    move_up = distance > terrain.cfg.terrain_generator.size[0] / 2
    move_down = distance < torch.norm(command[env_ids, :2], dim=1) * env.max_episode_length_s * 0.5
    move_down *= ~move_up
    terrain.update_env_origins(env_ids, move_up, move_down)
    return torch.mean(terrain.terrain_levels.float())
```

**Code — ROUGH_TERRAINS_CFG** (`isaaclab/terrains/config/rough.py`, VERBATIM):
```python
import isaaclab.terrains as terrain_gen
from ..terrain_generator_cfg import TerrainGeneratorCfg

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
            proportion=0.2, step_height_range=(0.05, 0.23), step_width=0.3,
            platform_width=3.0, border_width=1.0, holes=False),
        "pyramid_stairs_inv": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
            proportion=0.2, step_height_range=(0.05, 0.23), step_width=0.3,
            platform_width=3.0, border_width=1.0, holes=False),
        "boxes": terrain_gen.MeshRandomGridTerrainCfg(
            proportion=0.2, grid_width=0.45, grid_height_range=(0.05, 0.2), platform_width=2.0),
        "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.2, noise_range=(0.02, 0.10), noise_step=0.02, border_width=0.25),
        "hf_pyramid_slope": terrain_gen.HfPyramidSlopedTerrainCfg(
            proportion=0.1, slope_range=(0.0, 0.4), platform_width=2.0, border_width=0.25),
        "hf_pyramid_slope_inv": terrain_gen.HfInvertedPyramidSlopedTerrainCfg(
            proportion=0.1, slope_range=(0.0, 0.4), platform_width=2.0, border_width=0.25),
    },
)
```

**Smoke** (§7): roll out with DR ON vs OFF (push_robot is None here, so the live DR is only the fixed friction + terrain curriculum). Because startup friction ranges are degenerate and push is disabled, the only seed-sensitive divergence comes from terrain-level placement + reset randomization. For a true DR-divergence smoke, re-enable `push_robot` and compare seed-matched obs trajectories. NOTE: as configured, H1 rough has minimal active DR — the rough difficulty comes from the terrain generator + curriculum, not from event-based randomization.

---

