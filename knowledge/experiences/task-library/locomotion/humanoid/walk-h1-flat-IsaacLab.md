# Isaac-Velocity-Flat-H1-v0 — Implementation Spec

- robot: Unitree H1 bipedal humanoid (19 DoF)
- simulator: IsaacLab (Isaac Sim, manager-based)
- objects: none (flat terrain)
- bimanual: false
- summary: Track a commanded base velocity while walking on flat ground.

This is an UPSTREAM IsaacLab manager-based **locomotion / velocity-tracking** task for the **Unitree H1 bipedal humanoid** on **flat ground**. The agent tracks a commanded base velocity (lin_x, lin_y, ang_z) via direct joint-position targets over all 19 joints.

Config layering (each subclass only overrides `__post_init__`):

```
LocomotionVelocityRoughEnvCfg   (velocity_env_cfg.py — shared base: scene, actions, commands, obs, events, rewards, terminations, curriculum)
  └─ H1RoughEnvCfg              (config/h1/rough_env_cfg.py — H1 robot articulation + H1Rewards override + H1-specific event/command/termination retuning)
       └─ H1FlatEnvCfg          (config/h1/flat_env_cfg.py — strips terrain/height-scan/curriculum, retunes feet_air_time)
```

`Isaac-Velocity-Flat-H1-v0` → `flat_env_cfg:H1FlatEnvCfg`.

---

## §1 Registration + Scene

**Description.** Registers the flat-H1 env pointing at `H1FlatEnvCfg`. The scene is the shared locomotion `MySceneCfg` (terrain importer + contact sensors + dome sky light), with the robot articulation filled in by `H1RoughEnvCfg.__post_init__` using `H1_MINIMAL_CFG`, and the terrain/height-scanner removed by `H1FlatEnvCfg.__post_init__`.

**Decisions resolved.**
- entry_point = `isaaclab.envs:ManagerBasedRLEnv`, `disable_env_checker=True`.
- env_cfg_entry_point = `...config.h1.flat_env_cfg:H1FlatEnvCfg`.
- robot = `H1_MINIMAL_CFG` (minimal-collision variant), prim_path `{ENV_REGEX_NS}/Robot`.
- Robot USD path (resolved): `${ISAACLAB_NUCLEUS_DIR}/Robots/Unitree/H1/h1_minimal.usd`. (`H1_CFG` base uses `.../H1/h1.usd`; the minimal cfg overrides only `spawn.usd_path`.)
- init height z = 1.05 m; init joint pose: hips_yaw/roll=0, hip_pitch=-0.28 (-16°), knee=0.79 (45°), ankle=-0.52 (-30°), torso=0, shoulder_pitch=0.28, shoulder_roll/yaw=0, elbow=0.52.
- 19 joints total (legs+torso 9, feet 2, arms 8) — implicit actuators, stiffness/damping per group below.
- terrain → flat plane (`terrain_type="plane"`, generator=None); `height_scanner=None`; `curriculum.terrain_levels=None`.
- scene defaults: `num_envs=4096`, `env_spacing=2.5`.
- contact_forces sensor on `{ENV_REGEX_NS}/Robot/.*`, history_length=3, track_air_time=True.
- sim: `decimation=4`, `dt=0.005` (200 Hz physics, 50 Hz control), `episode_length_s=20.0`.

**Code (registration — `config/h1/__init__.py`).**
```python
gym.register(
    id="Isaac-Velocity-Flat-H1-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg:H1FlatEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:H1FlatPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_flat_ppo_cfg.yaml",
    },
)
```

**Code (H1 articulation — `isaaclab_assets/robots/unitree.py`).**
```python
H1_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAACLAB_NUCLEUS_DIR}/Robots/Unitree/H1/h1.usd",
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
            enabled_self_collisions=False, solver_position_iteration_count=4, solver_velocity_iteration_count=4
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.05),
        joint_pos={
            ".*_hip_yaw": 0.0,
            ".*_hip_roll": 0.0,
            ".*_hip_pitch": -0.28,  # -16 degrees
            ".*_knee": 0.79,  # 45 degrees
            ".*_ankle": -0.52,  # -30 degrees
            "torso": 0.0,
            ".*_shoulder_pitch": 0.28,
            ".*_shoulder_roll": 0.0,
            ".*_shoulder_yaw": 0.0,
            ".*_elbow": 0.52,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[".*_hip_yaw", ".*_hip_roll", ".*_hip_pitch", ".*_knee", "torso"],
            effort_limit_sim=300,
            stiffness={
                ".*_hip_yaw": 150.0,
                ".*_hip_roll": 150.0,
                ".*_hip_pitch": 200.0,
                ".*_knee": 200.0,
                "torso": 200.0,
            },
            damping={
                ".*_hip_yaw": 5.0,
                ".*_hip_roll": 5.0,
                ".*_hip_pitch": 5.0,
                ".*_knee": 5.0,
                "torso": 5.0,
            },
        ),
        "feet": ImplicitActuatorCfg(
            joint_names_expr=[".*_ankle"],
            effort_limit_sim=100,
            stiffness={".*_ankle": 20.0},
            damping={".*_ankle": 4.0},
        ),
        "arms": ImplicitActuatorCfg(
            joint_names_expr=[".*_shoulder_pitch", ".*_shoulder_roll", ".*_shoulder_yaw", ".*_elbow"],
            effort_limit_sim=300,
            stiffness={
                ".*_shoulder_pitch": 40.0,
                ".*_shoulder_roll": 40.0,
                ".*_shoulder_yaw": 40.0,
                ".*_elbow": 40.0,
            },
            damping={
                ".*_shoulder_pitch": 10.0,
                ".*_shoulder_roll": 10.0,
                ".*_shoulder_yaw": 10.0,
                ".*_elbow": 10.0,
            },
        ),
    },
)

H1_MINIMAL_CFG = H1_CFG.copy()
H1_MINIMAL_CFG.spawn.usd_path = f"{ISAACLAB_NUCLEUS_DIR}/Robots/Unitree/H1/h1_minimal.usd"
```

**Code (shared scene — `velocity_env_cfg.py:MySceneCfg`).**
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

**Code (flat overrides — `config/h1/flat_env_cfg.py`).**
```python
@configclass
class H1FlatEnvCfg(H1RoughEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        # change terrain to flat
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        # no height scan
        self.scene.height_scanner = None
        self.observations.policy.height_scan = None
        # no terrain curriculum
        self.curriculum.terrain_levels = None
        self.rewards.feet_air_time.weight = 1.0
        self.rewards.feet_air_time.params["threshold"] = 0.6
```

**Code (robot wiring — `config/h1/rough_env_cfg.py:H1RoughEnvCfg.__post_init__`, scene portion).**
```python
self.scene.robot = H1_MINIMAL_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
if self.scene.height_scanner:
    self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/torso_link"
```

**Smoke.** `cd <repo> && .venv/bin/python -c "import gymnasium as gym; env=gym.make('Isaac-Velocity-Flat-H1-v0'); print(env.observation_space, env.action_space); env.close()"` — NOT captured here (no pxr; raises NameNotFound). Expected when Isaac is bootable: `Box(..., (69,)) Box(-1.0, 1.0, (19,))` approx.

---

## §2 Actions

**Description.** Single action term: direct joint-position targets for **all 19 joints** (`joint_names=[".*"]`), scaled by 0.5 and added to the default joint pose. Bipedal humanoid → no IK, no gripper. Action dim = 19.

**Decisions resolved.** `JointPositionActionCfg(asset_name="robot", joint_names=[".*"], scale=0.5, use_default_offset=True)`. Inherited unchanged from the shared base by both H1RoughEnvCfg and H1FlatEnvCfg.

Action dim derivation: 19 joints = legs/torso {hip_yaw×2, hip_roll×2, hip_pitch×2, knee×2, torso×1 = 9} + feet {ankle×2 = 2} + arms {shoulder_pitch×2, shoulder_roll×2, shoulder_yaw×2, elbow×2 = 8}.

**Code (`velocity_env_cfg.py:ActionsCfg`).**
```python
@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    joint_pos = mdp.JointPositionActionCfg(asset_name="robot", joint_names=[".*"], scale=0.5, use_default_offset=True)
```

**Smoke.** `env.action_space.shape == (19,)`; one `env.step(zeros)` runs and obs is finite. (Run when reproducing; not captured here — no pxr.)

---

## §3 Reset

**Description.** Reset events: zero base external force/torque, randomize root pose+velocity, scale joint positions. H1 retunes these vs the shared base — disables push_robot/add_base_mass/base_com, sets joint-position reset scale to a fixed 1.0 (no randomization), zeros all reset velocities, restricts root-pose randomization to ±0.5 m in x/y and ±π yaw, and retargets the base_external_force_torque asset to `.*torso_link`.

**Decisions resolved (after H1 overrides applied).**
- `reset_base` (`reset_root_state_uniform`): pose_range `{x:(-0.5,0.5), y:(-0.5,0.5), yaw:(-3.14,3.14)}`; velocity_range all zeros (x/y/z/roll/pitch/yaw = (0,0)).
- `reset_robot_joints` (`reset_joints_by_scale`): `position_range=(1.0,1.0)` (override — base default was (0.5,1.5)), `velocity_range=(0.0,0.0)`.
- `base_external_force_torque` (`apply_external_force_torque`, mode="reset"): force/torque ranges (0,0); `asset_cfg.body_names = [".*torso_link"]` (override — base was `base`).

**Code (shared base reset terms — `velocity_env_cfg.py:EventCfg`, reset block).**
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

**Code (H1 reset overrides — `config/h1/rough_env_cfg.py:H1RoughEnvCfg.__post_init__`).**
```python
self.events.reset_robot_joints.params["position_range"] = (1.0, 1.0)
self.events.base_external_force_torque.params["asset_cfg"].body_names = [".*torso_link"]
self.events.reset_base.params = {
    "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
    "velocity_range": {
        "x": (0.0, 0.0),
        "y": (0.0, 0.0),
        "z": (0.0, 0.0),
        "roll": (0.0, 0.0),
        "pitch": (0.0, 0.0),
        "yaw": (0.0, 0.0),
    },
}
```

**Smoke.** Two `env.reset(seed=k)` produce differing root poses (yaw randomized) but identical joint-position scale (=1.0). Run when reproducing.

---

## §4 Goal + Termination

**Description.** Goal = track the commanded base velocity (defined by CommandsCfg, see below). Episode ends on timeout (20 s) or illegal contact on the torso. H1 retargets the base_contact sensor from `base` to `.*torso_link`. Velocity command: heading-based uniform sampler resampled every 10 s; H1 narrows the command ranges.

**Decisions resolved.**
- `time_out` (`mdp.time_out`, `time_out=True`) — at `episode_length_s=20.0`.
- `base_contact` (`mdp.illegal_contact`, threshold=1.0): `sensor_cfg.body_names = ".*torso_link"` (H1 override; base default `base`).
- Command `base_velocity` (`UniformVelocityCommandCfg`): resampling_time_range (10,10), rel_standing_envs=0.02, rel_heading_envs=1.0, heading_command=True, heading_control_stiffness=0.5, debug_vis=True.
- Command ranges after H1 override: `lin_vel_x=(0.0,1.0)`, `lin_vel_y=(0.0,0.0)`, `ang_vel_z=(-1.0,1.0)` (base defaults were lin_x/lin_y/ang_z all (-1,1), heading (-π,π); heading range left at base (-π,π)).

**Code (shared terminations — `velocity_env_cfg.py:TerminationsCfg`).**
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

**Code (shared command — `velocity_env_cfg.py:CommandsCfg`).**
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

**Code (H1 termination + command overrides — `config/h1/rough_env_cfg.py:H1RoughEnvCfg.__post_init__`).**
```python
# Commands
self.commands.base_velocity.ranges.lin_vel_x = (0.0, 1.0)
self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)

# Terminations
self.terminations.base_contact.params["sensor_cfg"].body_names = ".*torso_link"
```

**Source funcs (verbatim — `isaaclab/envs/mdp/terminations.py`).**
```python
def time_out(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Terminate the episode when the episode length exceeds the maximum episode length."""
    return env.episode_length_buf >= env.max_episode_length


def illegal_contact(env: ManagerBasedRLEnv, threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Terminate when the contact force on the sensor exceeds the force threshold."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_contact_forces = contact_sensor.data.net_forces_w_history
    return torch.any(
        torch.max(torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1)[0] > threshold, dim=1
    )
```

**Smoke.** Rolling out with a torso-flooring push triggers `base_contact`; an idle env runs to `episode_length_s`. Run when reproducing.

---

## §5 Observation

**Description.** Single concatenated policy group, corruption (noise) enabled. For the FLAT variant `height_scan` is removed (`self.observations.policy.height_scan = None`), so the obs vector is base_lin_vel + base_ang_vel + projected_gravity + velocity_commands + joint_pos + joint_vel + last_action.

**Decisions resolved.** `enable_corruption=True`, `concatenate_terms=True`. Terms (in order) and dims:
- base_lin_vel (3), noise Unoise(-0.1,0.1)
- base_ang_vel (3), noise Unoise(-0.2,0.2)
- projected_gravity (3), noise Unoise(-0.05,0.05)
- velocity_commands (3) [command_name="base_velocity"]
- joint_pos (joint_pos_rel, 19), noise Unoise(-0.01,0.01)
- joint_vel (joint_vel_rel, 19), noise Unoise(-1.5,1.5)
- actions (last_action, 19)
- height_scan → **None** in flat variant (removed).

Total obs dim = 3+3+3+3+19+19+19 = **69**.

**Code (shared obs — `velocity_env_cfg.py:ObservationsCfg`).**
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
Flat override (drops height_scan): `self.observations.policy.height_scan = None` (in `H1FlatEnvCfg.__post_init__`).

All obs term funcs (`base_lin_vel`, `base_ang_vel`, `projected_gravity`, `generated_commands`, `joint_pos_rel`, `joint_vel_rel`, `last_action`) come from the shared `isaaclab.envs.mdp` package (imported via `from isaaclab.envs.mdp import *`) — no task-local observation funcs.

**Smoke.** `env.observation_space["policy"].shape == (69,)` and obs finite. Run when reproducing.

---

## §6 Reward

**Description.** H1 replaces the shared `RewardsCfg` with `H1Rewards(RewardsCfg)`. Composer = **sum** (RewardManager sums weighted terms each step). Tasks-vs-penalties below; **bold** = H1-specific (added or retuned vs the shared Anymal-style base).

**Decisions resolved — final term set, weight, params:**

Task tracking (H1 swaps in the yaw-frame / world-frame biped variants):
- **track_lin_vel_xy_exp** → func `track_lin_vel_xy_yaw_frame_exp`, weight 1.0, std 0.5 (base used `track_lin_vel_xy_exp`, std √0.25).
- **track_ang_vel_z_exp** → func `track_ang_vel_z_world_exp`, weight 1.0, std 0.5 (base weight 0.5, base func `track_ang_vel_z_exp`).

Biped gait shaping (H1-added, not in base):
- **feet_air_time** → func `feet_air_time_positive_biped`, weight 0.25 in rough, **retuned to 1.0 + threshold 0.6 in FLAT** (`config/h1/flat_env_cfg.py`); params command_name="base_velocity", sensor `.*ankle_link`, threshold 0.4 (rough). Base used `feet_air_time` (non-biped) on `.*FOOT`.
- **feet_slide** → func `feet_slide`, weight -0.25, sensor+asset `.*ankle_link`. (H1-added; not in base.)

Penalties:
- **termination_penalty** → `is_terminated`, weight -200.0. (H1-added; not in base.)
- lin_vel_z_l2 → **None** (H1 disables the base term).
- **flat_orientation_l2** → weight retuned -1.0 (base 0.0).
- **dof_pos_limits** → `joint_pos_limits`, weight -1.0, restricted to `.*_ankle` joints (base weight 0.0, all joints).
- **dof_torques_l2** → weight 0.0 (base -1.0e-5; H1 disables).
- **dof_acc_l2** → weight -1.25e-7 (base -2.5e-7).
- **action_rate_l2** → weight -0.005 (base -0.01).
- ang_vel_xy_l2 → inherited from base, weight -0.05.
- undesired_contacts → **None** (H1 disables; base had `.*THIGH` weight -1.0).

Joint-deviation penalties (all H1-added, biped posture regularizers):
- **joint_deviation_hip** → `joint_deviation_l1`, weight -0.2, joints `.*_hip_yaw`, `.*_hip_roll`.
- **joint_deviation_arms** → `joint_deviation_l1`, weight -0.2, joints `.*_shoulder_.*`, `.*_elbow`.
- **joint_deviation_torso** → `joint_deviation_l1`, weight -0.1, joint `torso`.

Planning budget: no per-stage docstring present in `H1Rewards`. Retro-computed dominant magnitudes (per step, nominal): tracking terms saturate near +1 each (lin + ang ≈ +2); feet_air_time up to +threshold per foot-contact (flat: ≈+0.6 weighted ×1.0); termination_penalty −200 fires once on fall (dominant catastrophic signal). Penalties (orientation, joint-deviation, ankle-limit, acc/action-rate) are small continuous shaping terms tuned to keep an upright, low-deviation, smooth gait. (retro-computed)

**Code (shared base rewards — `velocity_env_cfg.py:RewardsCfg`).**
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

**Code (H1 reward override — `config/h1/rough_env_cfg.py:H1Rewards`).**
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

**Code (H1 reward weight overrides — `config/h1/rough_env_cfg.py:H1RoughEnvCfg.__post_init__`, reward block).**
```python
# Rewards
self.rewards.undesired_contacts = None
self.rewards.flat_orientation_l2.weight = -1.0
self.rewards.dof_torques_l2.weight = 0.0
self.rewards.action_rate_l2.weight = -0.005
self.rewards.dof_acc_l2.weight = -1.25e-7
```
Plus FLAT override (`config/h1/flat_env_cfg.py`): `self.rewards.feet_air_time.weight = 1.0`, `self.rewards.feet_air_time.params["threshold"] = 0.6`.

**Reward function source (verbatim).**

Task-local (`locomotion/velocity/mdp/rewards.py`):
```python
def feet_air_time(
    env: ManagerBasedRLEnv, command_name: str, sensor_cfg: SceneEntityCfg, threshold: float
) -> torch.Tensor:
    """Reward long steps taken by the feet using L2-kernel. ... (docstring elided)"""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    first_contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    reward = torch.sum((last_air_time - threshold) * first_contact, dim=1)
    reward *= torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.1
    return reward


def feet_air_time_positive_biped(env, command_name: str, threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Reward long steps taken by the feet for bipeds. ... (docstring elided)"""
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


def feet_slide(env, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize feet sliding. ... (docstring elided)"""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contacts = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0] > 1.0
    asset = env.scene[asset_cfg.name]
    body_vel = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2]
    reward = torch.sum(body_vel.norm(dim=-1) * contacts, dim=1)
    return reward


def track_lin_vel_xy_yaw_frame_exp(
    env, std: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of linear velocity commands (xy axes) in the gravity aligned robot frame using an exponential kernel."""
    asset = env.scene[asset_cfg.name]
    vel_yaw = quat_apply_inverse(yaw_quat(asset.data.root_quat_w), asset.data.root_lin_vel_w[:, :3])
    lin_vel_error = torch.sum(
        torch.square(env.command_manager.get_command(command_name)[:, :2] - vel_yaw[:, :2]), dim=1
    )
    return torch.exp(-lin_vel_error / std**2)


def track_ang_vel_z_world_exp(
    env, command_name: str, std: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of angular velocity commands (yaw) in world frame using exponential kernel."""
    asset = env.scene[asset_cfg.name]
    ang_vel_error = torch.square(env.command_manager.get_command(command_name)[:, 2] - asset.data.root_ang_vel_w[:, 2])
    return torch.exp(-ang_vel_error / std**2)
```
(Helper imports at top of this module: `from isaaclab.utils.math import quat_apply_inverse, yaw_quat`; `from isaaclab.sensors import ContactSensor`.)

Shared (`isaaclab/envs/mdp/rewards.py`):
```python
def is_terminated(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Penalize terminated episodes that don't correspond to episodic timeouts."""
    return env.termination_manager.terminated.float()


def ang_vel_xy_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize xy-axis base angular velocity using L2 squared kernel."""
    asset: RigidObject = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.root_ang_vel_b[:, :2]), dim=1)


def flat_orientation_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize non-flat base orientation using L2 squared kernel (xy-components of projected gravity)."""
    asset: RigidObject = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.projected_gravity_b[:, :2]), dim=1)


def joint_torques_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize joint torques applied on the articulation using L2 squared kernel."""
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.applied_torque[:, asset_cfg.joint_ids]), dim=1)


def joint_acc_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize joint accelerations on the articulation using L2 squared kernel."""
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.joint_acc[:, asset_cfg.joint_ids]), dim=1)


def joint_deviation_l1(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize joint positions that deviate from the default one."""
    asset: Articulation = env.scene[asset_cfg.name]
    angle = asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    return torch.sum(torch.abs(angle), dim=1)


def joint_pos_limits(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize joint positions if they cross the soft limits."""
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
```

**Smoke (§6 contract).** Per-step `sum(detailed_reward.values()) == env_reward` (composer = sum); reward finite + non-constant across a random rollout. Run when reproducing.

---

## §7 DR

**Description.** Domain randomization = the non-reset EventCfg terms (mode `startup`/`interval`). The shared base defines `physics_material` (startup), `add_base_mass` (startup), `base_com` (startup), and `push_robot` (interval). **H1 disables push_robot, add_base_mass, and base_com**, leaving only the startup friction randomization active.

**Decisions resolved (after H1 overrides).**
- `physics_material` (`randomize_rigid_body_material`, mode="startup"): static_friction_range (0.8,0.8), dynamic_friction_range (0.6,0.6), restitution_range (0.0,0.0), num_buckets=64, asset all bodies `.*`. **ACTIVE.** (Note: ranges are degenerate point values, so this assigns fixed friction rather than truly randomizing — effectively a constant material override across buckets.)
- `add_base_mass` → **None** (H1 disables).
- `base_com` → **None** (H1 disables).
- `push_robot` (interval) → **None** (H1 disables). Flat-PLAY also nulls it (already null here).

So effective DR for H1-Flat is: only the startup `physics_material` term, with point-valued ranges → essentially no stochastic DR. A §7 ON-vs-OFF smoke that requires divergent seed-matched obs trajectories will likely be **skipped** (`<no effective DR>`), which is a valid success per the dr-generator contract.

**Code (shared base DR terms — `velocity_env_cfg.py:EventCfg`, startup + interval block).**
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

**Code (H1 DR-disable overrides — `config/h1/rough_env_cfg.py:H1RoughEnvCfg.__post_init__`).**
```python
self.events.push_robot = None
self.events.add_base_mass = None
# ...
self.events.base_com = None
```

**Smoke (§7 contract).** With the H1 overrides, DR ON vs OFF produces identical seed-matched obs trajectories → mark `skipped` (no effective stochastic DR). Valid success.

---

