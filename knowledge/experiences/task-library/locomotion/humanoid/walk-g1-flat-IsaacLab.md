# Isaac-Velocity-Flat-G1-v0 — Implementation Spec

- robot: Unitree G1 bipedal humanoid (37 DoF, hand-equipped)
- simulator: IsaacLab (Isaac Sim, manager-based)
- objects: none (flat terrain)
- bimanual: false
- summary: Track a commanded base velocity while walking on flat ground.

This is an UPSTREAM IsaacLab manager-based **bipedal locomotion** task: Unitree **G1** humanoid tracking a commanded base velocity (`lin_vel_x`, `lin_vel_y`, `ang_vel_z`) on **flat** ground. `G1FlatEnvCfg` subclasses `G1RoughEnvCfg` (config/g1/flat_env_cfg.py), which subclasses the abstract `LocomotionVelocityRoughEnvCfg` (velocity_env_cfg.py). The flat subclass swaps terrain to a plane, removes the height scanner + height-scan obs + terrain curriculum, and retunes a few reward weights / command ranges.

## How G1 differs from H1
- **Joint-deviation groupings are finer and more numerous.** G1 splits non-locomotion deviation into FIVE separate L1 terms — `joint_deviation_hip` (hip yaw/roll, w=-0.1), `joint_deviation_arms` (shoulder×3 + elbow pitch/roll, w=-0.1), `joint_deviation_fingers` (the 7 finger-joint patterns, w=-0.05), `joint_deviation_torso` (`torso_joint`, w=-0.1). H1 has no torso joint and no fingers, so it carries fewer/coarser deviation terms.
- **G1 is hand-equipped** (37 DOF incl. finger joints) vs H1's larger-but-handless body. The whole `joint_deviation_fingers` term and the finger entries in the `arms` actuator group are G1-only.
- **Smaller, lighter robot** — init base height `z=0.74` (H1 spawns higher ~1.05). Crouched init pose: `hip_pitch=-0.20, knee=0.42, ankle_pitch=-0.23, elbow_pitch=0.87`.
- **Termination / DR body is `torso_link`** (base_contact sensor + external-force asset both retargeted to `torso_link`), whereas the base cfg / H1 use `base`.
- Otherwise the task scaffold (velocity command, action, obs, episode length, dt) is the shared locomotion-velocity template H1 also uses.

---

## §1 Registration + Scene

**Description.** Registers the flat G1 velocity-tracking env on `ManagerBasedRLEnv`. The scene is the shared locomotion scene (`MySceneCfg`) with the robot set to `G1_MINIMAL_CFG`; the flat subclass replaces terrain with an infinite plane and deletes the height scanner. Sensors: a contact sensor over all robot bodies (`track_air_time=True`). One dome sky light.

**Decisions resolved.**
- `entry_point = "isaaclab.envs:ManagerBasedRLEnv"`, `env_cfg_entry_point = ...flat_env_cfg:G1FlatEnvCfg`, `disable_env_checker=True`.
- robot = `G1_MINIMAL_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")`.
- USD path (minimal): `{ISAACLAB_NUCLEUS_DIR}/Robots/Unitree/G1/g1_minimal.usd` (full `G1_CFG` uses `.../G1/g1.usd`).
- init base pos `(0.0, 0.0, 0.74)`; crouched joint init (see code).
- `soft_joint_pos_limit_factor = 0.9`; self-collisions off; pos/vel solver iters 8/4.
- terrain → `terrain_type="plane"`, `terrain_generator=None`; `height_scanner=None`; `num_envs=4096`, `env_spacing=2.5`.
- `decimation=4`, `episode_length_s=20.0`, `sim.dt=0.005` (→ control dt 0.02 s).

**Code.**

`config/g1/__init__.py` (Flat registration):
```python
gym.register(
    id="Isaac-Velocity-Flat-G1-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg:G1FlatEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1FlatPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_flat_ppo_cfg.yaml",
    },
)
```

`G1_MINIMAL_CFG` / `G1_CFG` (isaaclab_assets/robots/unitree.py:272-385):
```python
G1_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAACLAB_NUCLEUS_DIR}/Robots/Unitree/G1/g1.usd",
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
            enabled_self_collisions=False, solver_position_iteration_count=8, solver_velocity_iteration_count=4
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.74),
        joint_pos={
            ".*_hip_pitch_joint": -0.20,
            ".*_knee_joint": 0.42,
            ".*_ankle_pitch_joint": -0.23,
            ".*_elbow_pitch_joint": 0.87,
            "left_shoulder_roll_joint": 0.16,
            "left_shoulder_pitch_joint": 0.35,
            "right_shoulder_roll_joint": -0.16,
            "right_shoulder_pitch_joint": 0.35,
            "left_one_joint": 1.0,
            "right_one_joint": -1.0,
            "left_two_joint": 0.52,
            "right_two_joint": -0.52,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_hip_yaw_joint",
                ".*_hip_roll_joint",
                ".*_hip_pitch_joint",
                ".*_knee_joint",
                "torso_joint",
            ],
            effort_limit_sim=300,
            stiffness={
                ".*_hip_yaw_joint": 150.0,
                ".*_hip_roll_joint": 150.0,
                ".*_hip_pitch_joint": 200.0,
                ".*_knee_joint": 200.0,
                "torso_joint": 200.0,
            },
            damping={
                ".*_hip_yaw_joint": 5.0,
                ".*_hip_roll_joint": 5.0,
                ".*_hip_pitch_joint": 5.0,
                ".*_knee_joint": 5.0,
                "torso_joint": 5.0,
            },
            armature={
                ".*_hip_.*": 0.01,
                ".*_knee_joint": 0.01,
                "torso_joint": 0.01,
            },
        ),
        "feet": ImplicitActuatorCfg(
            effort_limit_sim=20,
            joint_names_expr=[".*_ankle_pitch_joint", ".*_ankle_roll_joint"],
            stiffness=20.0,
            damping=2.0,
            armature=0.01,
        ),
        "arms": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_shoulder_pitch_joint",
                ".*_shoulder_roll_joint",
                ".*_shoulder_yaw_joint",
                ".*_elbow_pitch_joint",
                ".*_elbow_roll_joint",
                ".*_five_joint",
                ".*_three_joint",
                ".*_six_joint",
                ".*_four_joint",
                ".*_zero_joint",
                ".*_one_joint",
                ".*_two_joint",
            ],
            effort_limit_sim=300,
            stiffness=40.0,
            damping=10.0,
            armature={
                ".*_shoulder_.*": 0.01,
                ".*_elbow_.*": 0.01,
                ".*_five_joint": 0.001,
                ".*_three_joint": 0.001,
                ".*_six_joint": 0.001,
                ".*_four_joint": 0.001,
                ".*_zero_joint": 0.001,
                ".*_one_joint": 0.001,
                ".*_two_joint": 0.001,
            },
        ),
    },
)

G1_MINIMAL_CFG = G1_CFG.copy()
G1_MINIMAL_CFG.spawn.usd_path = f"{ISAACLAB_NUCLEUS_DIR}/Robots/Unitree/G1/g1_minimal.usd"
```

Shared scene `MySceneCfg` (velocity_env_cfg.py:39-82) — flat subclass nulls `terrain.terrain_generator` and `height_scanner`:
```python
@configclass
class MySceneCfg(InteractiveSceneCfg):
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=ROUGH_TERRAINS_CFG,   # flat sets terrain_type="plane", terrain_generator=None
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
    robot: ArticulationCfg = MISSING
    height_scanner = RayCasterCfg(   # flat sets this to None
        prim_path="{ENV_REGEX_NS}/Robot/base",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0]),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )
    contact_forces = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True)
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )
```
Note: `G1RoughEnvCfg.__post_init__` sets `height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/torso_link"` for the rough variant; the flat subclass then nulls the scanner entirely.

**Resolved asset paths** (all on Nucleus, not local disk):
- `{ISAACLAB_NUCLEUS_DIR}/Robots/Unitree/G1/g1_minimal.usd`
- `{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/TilesMarbleSpiderWhiteBrickBondHoned.mdl`
- `{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr`

**Smoke (§1).**
```bash
cd <IsaacLab-repo>
.venv/bin/python -c "import gymnasium as gym; env=gym.make('Isaac-Velocity-Flat-G1-v0'); print(env.observation_space, env.action_space); env.close()"
```
Expected stdout: NOT CAPTURED — fails headless (`pxr` import unavailable; `gymnasium.error.NameNotFound: Environment 'Isaac-Velocity-Flat-G1' doesn't exist` because the IsaacLab task plugins never registered without a GPU/Isaac boot). On a GPU host with Isaac Sim this prints `Box(123,) Box(37,)` analytically (N=37 DOF).

---

## §2 Actions

**Description.** Single joint-position action over ALL joints (`[".*"]`), with per-joint scale 0.5 added to the default joint pose (PD targets fed to the implicit actuators).

**Decisions resolved.** `JointPositionActionCfg(asset_name="robot", joint_names=[".*"], scale=0.5, use_default_offset=True)`. No gripper / separate hand action — fingers are driven by the same joint-position action.

**Code** (velocity_env_cfg.py:108-112):
```python
@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    joint_pos = mdp.JointPositionActionCfg(asset_name="robot", joint_names=[".*"], scale=0.5, use_default_offset=True)
```

**Smoke (§2).** Step the env with `action_space.sample()` for ~5 steps; assert finite obs and that joint targets move. (Run inside `.venv` on a GPU host.)

---

## §3 Reset

**Description.** Reset events randomize the base root pose (xy ±0.5 m, yaw ±π; zero velocity for G1) and scale joint positions to exactly the default (G1 overrides `position_range=(1.0,1.0)` → no joint randomization). The base external-force/torque reset term is retargeted to `torso_link` with zero force ranges.

**Decisions resolved (G1 overrides applied on top of base EventCfg).**
- `reset_base`: `pose_range = {x:(-0.5,0.5), y:(-0.5,0.5), yaw:(-3.14,3.14)}`; `velocity_range` ALL zero (G1 override — base cfg used ±0.5 on every axis).
- `reset_robot_joints`: `position_range=(1.0,1.0)` (G1 override — base used (0.5,1.5)), `velocity_range=(0.0,0.0)`.
- `base_external_force_torque`: `mode="reset"`, asset body `torso_link` (G1 override of `base`), `force_range=(0.0,0.0)`, `torque_range=(0.0,0.0)`.

**Code.**

Base reset terms (velocity_env_cfg.py:186-219):
```python
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
        params={"position_range": (0.5, 1.5), "velocity_range": (0.0, 0.0)},
    )
```

G1 reset overrides (config/g1/rough_env_cfg.py:114-130, inherited by flat):
```python
        self.events.reset_robot_joints.params["position_range"] = (1.0, 1.0)
        self.events.base_external_force_torque.params["asset_cfg"].body_names = ["torso_link"]
        self.events.reset_base.params = {
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0),
                "roll": (0.0, 0.0), "pitch": (0.0, 0.0), "yaw": (0.0, 0.0),
            },
        }
```

**Smoke (§3).** Reset twice with different seeds; assert base xy/yaw differ but joint-pos equals default (since position_range is degenerate).

---

## §4 Goal + Termination

**Description.** Goal = track the commanded base velocity (no success terminal — velocity-tracking is a continuous reward objective). Episode ends on timeout (20 s) or illegal contact on `torso_link` (the robot falling / torso hitting ground).

**Decisions resolved.**
- `time_out` (DoneTerm, `time_out=True`) — episode horizon 20 s → 1000 control steps.
- `base_contact` = `illegal_contact` on sensor body `torso_link` (G1 override of `base`), `threshold=1.0`.
- No CommandsCfg-based termination. Command is a resampled `UniformVelocityCommand` (see below), NOT a goal terminal.

**Commands** (velocity_env_cfg.py:90-105; G1 flat override of ranges):
```python
@configclass
class CommandsCfg:
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
G1 **flat** command-range overrides (flat_env_cfg.py:39-41):
```python
        self.commands.base_velocity.ranges.lin_vel_x = (0.0, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.5, 0.5)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)
```
(The G1 *rough* parent set `lin_vel_y=(0.0,0.0)`; the flat subclass re-widens it to `(-0.5,0.5)`.)

**Code** — Terminations (velocity_env_cfg.py:266-274; G1 retargets sensor body):
```python
@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    base_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="base"), "threshold": 1.0},
    )
```
G1 override (rough_env_cfg.py:152): `self.terminations.base_contact.params["sensor_cfg"].body_names = "torso_link"`.

Termination funcs (isaaclab/envs/mdp/terminations.py):
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

**Smoke (§4).** Run an episode at max length; assert `time_out` fires at step == max_episode_length and that toppling triggers `base_contact`.

---

## §5 Observation

**Description.** Single `policy` group, terms concatenated, corruption (noise) enabled. Proprioceptive only on flat (NO height_scan). Order: base lin vel, base ang vel, projected gravity, velocity command, joint pos (rel default), joint vel (rel default), last action.

**Decisions resolved.**
- `enable_corruption = True`, `concatenate_terms = True`.
- Flat subclass sets `observations.policy.height_scan = None` → term dropped.
- Resolved total obs dim (analytic): `3+3+3+3 + N + N + N = 12 + 3N`; N=37 → **123**.
- Noise: base_lin_vel ±0.1, base_ang_vel ±0.2, projected_gravity ±0.05, joint_pos ±0.01, joint_vel ±1.5; velocity_commands + actions noise-free.

**Code** (velocity_env_cfg.py:115-146; flat nulls `height_scan`):
```python
@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, noise=Unoise(n_min=-0.1, n_max=0.1))
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
        projected_gravity = ObsTerm(func=mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05))
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-1.5, n_max=1.5))
        actions = ObsTerm(func=mdp.last_action)
        height_scan = ObsTerm(   # flat subclass sets this attribute to None
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner")},
            noise=Unoise(n_min=-0.1, n_max=0.1),
            clip=(-1.0, 1.0),
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()
```
All obs funcs are upstream `isaaclab.envs.mdp` builtins (`base_lin_vel`, `base_ang_vel`, `projected_gravity`, `generated_commands`, `joint_pos_rel`, `joint_vel_rel`, `last_action`) — no task-local obs.

**Smoke (§5).** Reset; assert `obs["policy"].shape[-1] == 12 + 3*num_joints` and no NaNs.

---

## §6 Reward

**Description.** Sum-composed (IsaacLab `RewardManager` sums all weighted RewTerms). Two positive tracking terms (lin-vel xy in yaw frame, ang-vel z in world frame), a biped air-time bonus, plus a stack of regularization penalties (orientation, joint torques/acc, action rate, feet slide, ankle joint-pos limits) and FOUR finely-grouped joint-deviation penalties (hip / arms / fingers / torso), and a large termination penalty.

**Composer:** SUM (IsaacLab `RewardManager`). Weights below are the *cfg weights*; effective per-step reward = `weight · dt · term`.

**Decisions resolved — final G1 flat reward table** (base `RewardsCfg` → `G1Rewards` overrides → `G1RoughEnvCfg.__post_init__` → `G1FlatEnvCfg.__post_init__`):

| Term | func | final weight (flat) | params |
|---|---|---|---|
| termination_penalty | `is_terminated` | -200.0 | — |
| track_lin_vel_xy_exp | `track_lin_vel_xy_yaw_frame_exp` | 1.0 | command=base_velocity, std=0.5 |
| track_ang_vel_z_exp | `track_ang_vel_z_world_exp` | **1.0** (flat override of 2.0) | command=base_velocity, std=0.5 |
| feet_air_time | `feet_air_time_positive_biped` | **0.75** (flat override of 0.25) | command=base_velocity, sensor=`.*_ankle_roll_link`, **threshold=0.4** (flat override) |
| feet_slide | `feet_slide` | -0.1 | sensor + asset = `.*_ankle_roll_link` |
| dof_pos_limits | `joint_pos_limits` | -1.0 | joints `.*_ankle_pitch_joint`, `.*_ankle_roll_joint` |
| joint_deviation_hip | `joint_deviation_l1` | -0.1 | joints `.*_hip_yaw_joint`, `.*_hip_roll_joint` |
| joint_deviation_arms | `joint_deviation_l1` | -0.1 | shoulder pitch/roll/yaw + elbow pitch/roll |
| joint_deviation_fingers | `joint_deviation_l1` | -0.05 | `.*_five/three/six/four/zero/one/two_joint` |
| joint_deviation_torso | `joint_deviation_l1` | -0.1 | `torso_joint` |
| lin_vel_z_l2 | `lin_vel_z_l2` | **-0.2** (flat override; rough set 0.0) | — |
| ang_vel_xy_l2 | `ang_vel_xy_l2` | -0.05 (base) | — |
| flat_orientation_l2 | `flat_orientation_l2` | -1.0 (set by G1 rough) | — |
| action_rate_l2 | `action_rate_l2` | **-0.005** (flat override of -0.01) | — |
| dof_acc_l2 | `joint_acc_l2` | **-1.0e-7** (flat override; rough -1.25e-7) | joints `.*_hip_.*`, `.*_knee_joint` |
| dof_torques_l2 | `joint_torques_l2` | **-2.0e-6** (flat override; rough -1.5e-7) | joints `.*_hip_.*`, `.*_knee_joint` (flat narrows; rough also had `.*_ankle_.*`) |
| undesired_contacts | `undesired_contacts` | **None / disabled** (G1 rough set it None) | — |

Inactive in flat: `undesired_contacts=None`, the base `dof_pos_limits` placeholder (G1 redefines its own ankle-limit term with weight -1.0).

**Planning-budget docstring:** none present in `G1Rewards`. Retro-computed dominant magnitudes (per-step, nominal): tracking terms saturate near their weight (lin ≈ 1.0, ang ≈ 1.0); feet_air_time bonus ≈ 0.75·0.4 ≈ 0.3/step; termination penalty ≈ -200 (one-shot, only on non-timeout reset). (Retro-computed — no upstream budget annotation.)

**Code — RewardsCfg base (velocity_env_cfg.py:230-263):**
```python
@configclass
class RewardsCfg:
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
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*FOOT"),
                "command_name": "base_velocity", "threshold": 0.5},
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

**Code — G1Rewards (rough_env_cfg.py:19-100), overrides/adds:**
```python
@configclass
class G1Rewards(RewardsCfg):
    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-200.0)
    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_yaw_frame_exp,
        weight=1.0, params={"command_name": "base_velocity", "std": 0.5},
    )
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_world_exp, weight=2.0, params={"command_name": "base_velocity", "std": 0.5}
    )
    feet_air_time = RewTerm(
        func=mdp.feet_air_time_positive_biped,
        weight=0.25,
        params={"command_name": "base_velocity",
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
                "threshold": 0.4},
    )
    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-0.1,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
                "asset_cfg": SceneEntityCfg("robot", body_names=".*_ankle_roll_link")},
    )
    dof_pos_limits = RewTerm(
        func=mdp.joint_pos_limits,
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_ankle_pitch_joint", ".*_ankle_roll_joint"])},
    )
    joint_deviation_hip = RewTerm(
        func=mdp.joint_deviation_l1, weight=-0.1,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_yaw_joint", ".*_hip_roll_joint"])},
    )
    joint_deviation_arms = RewTerm(
        func=mdp.joint_deviation_l1, weight=-0.1,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[
            ".*_shoulder_pitch_joint", ".*_shoulder_roll_joint", ".*_shoulder_yaw_joint",
            ".*_elbow_pitch_joint", ".*_elbow_roll_joint"])},
    )
    joint_deviation_fingers = RewTerm(
        func=mdp.joint_deviation_l1, weight=-0.05,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[
            ".*_five_joint", ".*_three_joint", ".*_six_joint", ".*_four_joint",
            ".*_zero_joint", ".*_one_joint", ".*_two_joint"])},
    )
    joint_deviation_torso = RewTerm(
        func=mdp.joint_deviation_l1, weight=-0.1,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names="torso_joint")},
    )
```

**Code — G1RoughEnvCfg.__post_init__ reward edits (rough_env_cfg.py:132-144):**
```python
        self.rewards.lin_vel_z_l2.weight = 0.0
        self.rewards.undesired_contacts = None
        self.rewards.flat_orientation_l2.weight = -1.0
        self.rewards.action_rate_l2.weight = -0.005
        self.rewards.dof_acc_l2.weight = -1.25e-7
        self.rewards.dof_acc_l2.params["asset_cfg"] = SceneEntityCfg("robot", joint_names=[".*_hip_.*", ".*_knee_joint"])
        self.rewards.dof_torques_l2.weight = -1.5e-7
        self.rewards.dof_torques_l2.params["asset_cfg"] = SceneEntityCfg(
            "robot", joint_names=[".*_hip_.*", ".*_knee_joint", ".*_ankle_.*"])
```

**Code — G1FlatEnvCfg.__post_init__ reward edits (flat_env_cfg.py:28-37):**
```python
        self.rewards.track_ang_vel_z_exp.weight = 1.0
        self.rewards.lin_vel_z_l2.weight = -0.2
        self.rewards.action_rate_l2.weight = -0.005
        self.rewards.dof_acc_l2.weight = -1.0e-7
        self.rewards.feet_air_time.weight = 0.75
        self.rewards.feet_air_time.params["threshold"] = 0.4
        self.rewards.dof_torques_l2.weight = -2.0e-6
        self.rewards.dof_torques_l2.params["asset_cfg"] = SceneEntityCfg(
            "robot", joint_names=[".*_hip_.*", ".*_knee_joint"])
```

**Code — task-local reward funcs (velocity/mdp/rewards.py), VERBATIM:**
```python
def feet_air_time(env, command_name, sensor_cfg, threshold) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    first_contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    reward = torch.sum((last_air_time - threshold) * first_contact, dim=1)
    reward *= torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.1
    return reward


def feet_air_time_positive_biped(env, command_name: str, threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
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
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contacts = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0] > 1.0
    asset = env.scene[asset_cfg.name]
    body_vel = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2]
    reward = torch.sum(body_vel.norm(dim=-1) * contacts, dim=1)
    return reward


def track_lin_vel_xy_yaw_frame_exp(env, std: float, command_name: str,
                                   asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    vel_yaw = quat_apply_inverse(yaw_quat(asset.data.root_quat_w), asset.data.root_lin_vel_w[:, :3])
    lin_vel_error = torch.sum(
        torch.square(env.command_manager.get_command(command_name)[:, :2] - vel_yaw[:, :2]), dim=1)
    return torch.exp(-lin_vel_error / std**2)


def track_ang_vel_z_world_exp(env, command_name: str, std: float,
                              asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    ang_vel_error = torch.square(env.command_manager.get_command(command_name)[:, 2] - asset.data.root_ang_vel_w[:, 2])
    return torch.exp(-ang_vel_error / std**2)
```
(Imports the file needs: `import torch`; `from isaaclab.envs import mdp`; `from isaaclab.managers import SceneEntityCfg`; `from isaaclab.sensors import ContactSensor`; `from isaaclab.utils.math import quat_apply_inverse, yaw_quat`.)

**Code — upstream builtin reward funcs used (isaaclab/envs/mdp/rewards.py), VERBATIM:**
```python
def is_terminated(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Penalize terminated episodes that don't correspond to episodic timeouts."""
    return env.termination_manager.terminated.float()

def lin_vel_z_l2(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: RigidObject = env.scene[asset_cfg.name]
    return torch.square(asset.data.root_lin_vel_b[:, 2])

def ang_vel_xy_l2(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: RigidObject = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.root_ang_vel_b[:, :2]), dim=1)

def flat_orientation_l2(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: RigidObject = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.projected_gravity_b[:, :2]), dim=1)

def joint_torques_l2(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.applied_torque[:, asset_cfg.joint_ids]), dim=1)

def joint_acc_l2(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.joint_acc[:, asset_cfg.joint_ids]), dim=1)

def joint_deviation_l1(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    angle = asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    return torch.sum(torch.abs(angle), dim=1)

def joint_pos_limits(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    out_of_limits = -(
        asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.soft_joint_pos_limits[:, asset_cfg.joint_ids, 0]
    ).clip(max=0.0)
    out_of_limits += (
        asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.soft_joint_pos_limits[:, asset_cfg.joint_ids, 1]
    ).clip(min=0.0)
    return torch.sum(out_of_limits, dim=1)

def action_rate_l2(env: ManagerBasedRLEnv) -> torch.Tensor:
    return torch.sum(torch.square(env.action_manager.action - env.action_manager.prev_action), dim=1)

# (undesired_contacts is disabled in G1, included for completeness:)
def undesired_contacts(env, threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_contact_forces = contact_sensor.data.net_forces_w_history
    is_contact = torch.max(torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1)[0] > threshold
    return torch.sum(is_contact, dim=1)
```

**Smoke (§6).** Step ~20 steps; assert total reward finite and non-constant, and that `sum(detailed_reward.values()) == env_reward` per step (composer = "sum").

---

## §7 DR

**Description.** Startup-mode domain randomization only (set once at spawn): randomize rigid-body material friction/restitution across all robot bodies. G1 explicitly DISABLES the base-mass, base-COM, and interval push-robot randomizations (sets them `None`). No interval DR active.

**Decisions resolved (after G1 overrides).**
- ACTIVE: `physics_material` (startup) — static_friction (0.8,0.8), dynamic_friction (0.6,0.6), restitution (0.0,0.0), num_buckets=64, over all robot bodies. (Degenerate ranges → effectively a fixed material; still a "startup" event term.)
- DISABLED by G1: `add_base_mass = None`, `base_com = None`, `push_robot = None`.

**Code — base EventCfg DR terms (velocity_env_cfg.py:153-227):**
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
    add_base_mass = EventTerm(            # G1: set to None
        func=mdp.randomize_rigid_body_mass, mode="startup",
        params={"asset_cfg": SceneEntityCfg("robot", body_names="base"),
                "mass_distribution_params": (-5.0, 5.0), "operation": "add"},
    )
    base_com = EventTerm(                  # G1: set to None
        func=mdp.randomize_rigid_body_com, mode="startup",
        params={"asset_cfg": SceneEntityCfg("robot", body_names="base"),
                "com_range": {"x": (-0.05, 0.05), "y": (-0.05, 0.05), "z": (-0.01, 0.01)}},
    )
    # interval
    push_robot = EventTerm(                # G1: set to None
        func=mdp.push_by_setting_velocity, mode="interval", interval_range_s=(10.0, 15.0),
        params={"velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)}},
    )
```
G1 DR disables (rough_env_cfg.py:115-130, inherited by flat):
```python
        self.events.push_robot = None
        self.events.add_base_mass = None
        self.events.base_com = None
```
(`G1FlatEnvCfg_PLAY` additionally nulls `base_external_force_torque` and `push_robot`, but those are play-only.)

**Smoke (§7).** With only the degenerate `physics_material` startup term active and all reset velocity ranges zeroed, seed-matched DR-ON vs DR-OFF trajectories will NOT diverge — so §7 is effectively `<no meaningful DR>` for this task (status: skipped is valid; the canonical example has no randomizing DR enabled).

---

