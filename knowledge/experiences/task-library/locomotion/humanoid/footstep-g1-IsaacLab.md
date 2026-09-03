# Isaac-Footstep-G1-v0 — Implementation Spec

- robot: Unitree G1 bipedal humanoid (37 DoF, hand-equipped)
- simulator: IsaacLab (Isaac Sim, manager-based)
- objects: none (flat terrain)
- bimanual: false
- summary: Follow an alternating sequence of swing-foot touchdown poses on flat ground.

> **CAVEAT:** All dims / counts below are **ANALYTIC** from reading the source (not build-verified here).
> This captures the DESIGN for reproduction / adaptation, not a runtime trace.

Task summary:
G1 (Unitree bipedal humanoid, 37-DoF with hands) must FOLLOW an alternating sequence of
swing-foot touchdown poses on flat ground. The task **reuses the entire proven `Isaac-Velocity-Flat-G1-v0`
scaffold** (`G1FlatEnvCfg` → `G1RoughEnvCfg` → `LocomotionVelocityRoughEnvCfg`): same robot, actions,
resets, contact sensor, terminations, sim timing. Only three managers are swapped in
`G1FootstepEnvCfg.__post_init__`: **command** (base-velocity → an advance-on-reach swing-foot
`FootPoseCommand`, visualized with a frame marker), **observation** (velocity command → the 6-D foot-pose
command; height-scan dropped), and **reward** (velocity-tracking → swing-foot position + yaw tracking,
a one-shot step-completion bonus, forward-progress pull, biped air-time, alive/height terms, and the G1
balance/smoothness/posture regularizers). The command LEADS the robot one reachable stride at a time: a
target is sampled ahead of the STANCE foot; it only advances (and credits a step) when the swing foot
genuinely lifts (> `lift_clearance`) then plants (xy within `reach_xy_threshold`, z below `reach_z_threshold`);
a per-step safety timeout force-advances a stalled env WITHOUT credit.

Inheritance chain:
`G1FootstepEnvCfg` → `G1FlatEnvCfg` → `G1RoughEnvCfg` → `LocomotionVelocityRoughEnvCfg(ManagerBasedRLEnvCfg)`.
Robot asset: `G1_MINIMAL_CFG` (from `isaaclab_assets.robots.unitree`).

---

## §1 Registration + Scene

**Description.** Two gym ids registered (train + play), both `ManagerBasedRLEnv`. Scene is the velocity
`MySceneCfg` with the G1 override from `G1RoughEnvCfg`, then `G1FlatEnvCfg` swaps terrain → flat plane,
drops the height scanner, and drops the terrain curriculum. Contact sensor on all bodies (air-time tracked);
dome sky light. Terminations use `torso_link` as the "base" contact body.

**Decisions resolved**

| Decision | Value |
|---|---|
| Train id / entry | `Isaac-Footstep-G1-v0` → `isaaclab.envs:ManagerBasedRLEnv`, cfg `G1FootstepEnvCfg` |
| Play id | `Isaac-Footstep-G1-Play-v0` → `G1FootstepEnvCfg_PLAY` |
| RSL-RL runner | `G1FootstepPPORunnerCfg` (1500 iters, 24 steps/env, ac dims [256,128,128], elu, lr 1e-3 adaptive) |
| Terrain | flat `plane` (rough generator + curriculum removed on flat) |
| Robot | `G1_MINIMAL_CFG` at `{ENV_REGEX_NS}/Robot`, init pos z=0.74 |
| Height scanner | **None** on flat (`self.scene.height_scanner = None`) |
| Contact sensor | `ContactSensorCfg` prim `{ENV_REGEX_NS}/Robot/.*`, history_length=3, track_air_time=True, update_period=sim.dt |
| Lights | Dome sky light (kloofendal HDR, intensity 750) |
| Sim | dt=0.005, decimation=4 → control dt=0.02 s (50 Hz); render_interval=4 |
| num_envs / spacing | 4096 / 2.5 (train); 50 / 2.5 (play) |
| episode_length_s | 20.0 |
| Foot bodies | `left_ankle_roll_link`, `right_ankle_roll_link` |
| base-contact termination body | `torso_link` |

**Code (registration) —** `config/g1/__init__.py`
```python
gym.register(
    id="Isaac-Footstep-G1-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.footstep_env_cfg:G1FootstepEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1FootstepPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Footstep-G1-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.footstep_env_cfg:G1FootstepEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1FootstepPPORunnerCfg",
    },
)
```

**Code (scene base) —** `velocity/velocity_env_cfg.py::MySceneCfg`
```python
@configclass
class MySceneCfg(InteractiveSceneCfg):
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=ROUGH_TERRAINS_CFG,
        max_init_terrain_level=5,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply", restitution_combine_mode="multiply",
            static_friction=1.0, dynamic_friction=1.0,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path=f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/TilesMarbleSpiderWhiteBrickBondHoned.mdl",
            project_uvw=True, texture_scale=(0.25, 0.25),
        ),
        debug_vis=False,
    )
    robot: ArticulationCfg = MISSING
    height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0]),
        debug_vis=False, mesh_prim_paths=["/World/ground"],
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

**Code (G1 scene override + flat) —** `velocity/config/g1/rough_env_cfg.py` & `flat_env_cfg.py`
```python
# G1RoughEnvCfg.__post_init__
self.scene.robot = G1_MINIMAL_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/torso_link"
...
self.terminations.base_contact.params["sensor_cfg"].body_names = "torso_link"

# G1FlatEnvCfg.__post_init__
self.scene.terrain.terrain_type = "plane"
self.scene.terrain.terrain_generator = None
self.scene.height_scanner = None
self.observations.policy.height_scan = None
self.curriculum.terrain_levels = None
```

**Code (robot asset) —** `isaaclab_assets/robots/unitree.py::G1_CFG` (+ `G1_MINIMAL_CFG`)
```python
G1_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAACLAB_NUCLEUS_DIR}/Robots/Unitree/G1/g1.usd",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False, retain_accelerations=False,
            linear_damping=0.0, angular_damping=0.0,
            max_linear_velocity=1000.0, max_angular_velocity=1000.0, max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False, solver_position_iteration_count=8, solver_velocity_iteration_count=4
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.74),
        joint_pos={
            ".*_hip_pitch_joint": -0.20, ".*_knee_joint": 0.42, ".*_ankle_pitch_joint": -0.23,
            ".*_elbow_pitch_joint": 0.87,
            "left_shoulder_roll_joint": 0.16, "left_shoulder_pitch_joint": 0.35,
            "right_shoulder_roll_joint": -0.16, "right_shoulder_pitch_joint": 0.35,
            "left_one_joint": 1.0, "right_one_joint": -1.0,
            "left_two_joint": 0.52, "right_two_joint": -0.52,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[".*_hip_yaw_joint", ".*_hip_roll_joint", ".*_hip_pitch_joint", ".*_knee_joint", "torso_joint"],
            effort_limit_sim=300,
            stiffness={".*_hip_yaw_joint":150.0, ".*_hip_roll_joint":150.0, ".*_hip_pitch_joint":200.0, ".*_knee_joint":200.0, "torso_joint":200.0},
            damping={".*_hip_yaw_joint":5.0, ".*_hip_roll_joint":5.0, ".*_hip_pitch_joint":5.0, ".*_knee_joint":5.0, "torso_joint":5.0},
            armature={".*_hip_.*":0.01, ".*_knee_joint":0.01, "torso_joint":0.01},
        ),
        "feet": ImplicitActuatorCfg(
            effort_limit_sim=20, joint_names_expr=[".*_ankle_pitch_joint", ".*_ankle_roll_joint"],
            stiffness=20.0, damping=2.0, armature=0.01,
        ),
        "arms": ImplicitActuatorCfg(
            joint_names_expr=[".*_shoulder_pitch_joint", ".*_shoulder_roll_joint", ".*_shoulder_yaw_joint",
                              ".*_elbow_pitch_joint", ".*_elbow_roll_joint",
                              ".*_five_joint", ".*_three_joint", ".*_six_joint", ".*_four_joint",
                              ".*_zero_joint", ".*_one_joint", ".*_two_joint"],
            effort_limit_sim=300, stiffness=40.0, damping=10.0,
            armature={".*_shoulder_.*":0.01, ".*_elbow_.*":0.01,
                      ".*_five_joint":0.001, ".*_three_joint":0.001, ".*_six_joint":0.001, ".*_four_joint":0.001,
                      ".*_zero_joint":0.001, ".*_one_joint":0.001, ".*_two_joint":0.001},
        ),
    },
)
G1_MINIMAL_CFG = G1_CFG.copy()
G1_MINIMAL_CFG.spawn.usd_path = f"{ISAACLAB_NUCLEUS_DIR}/Robots/Unitree/G1/g1_minimal.usd"
```

---

## §2 Actions

**Description.** Single joint-position action term on ALL joints (unchanged from the velocity base),
scale 0.5, applied as offset onto the default joint pose.

**Decisions resolved**

| Decision | Value |
|---|---|
| Action term | `mdp.JointPositionActionCfg` |
| Joints | `[".*"]` (all actuated G1 joints) |
| scale | 0.5 |
| use_default_offset | True |
| **action_dim** | **37** (G1 with hands: 6 hip + 2 knee + 1 torso + 4 ankle + 6 shoulder + 4 elbow + 14 hand) — ANALYTIC |

**Code —** `velocity/velocity_env_cfg.py::ActionsCfg`
```python
@configclass
class ActionsCfg:
    joint_pos = mdp.JointPositionActionCfg(asset_name="robot", joint_names=[".*"], scale=0.5, use_default_offset=True)
```

---

## §3 Reset / Events

**Description.** Reset events come entirely from the velocity base as overridden by `G1RoughEnvCfg`.
G1 disables the random-push interval, base-mass and CoM startup randomization, and applies zero external
force. Base pose is reset uniformly (x,y ±0.5 m, yaw ±π); base velocity is fixed to zero; joints reset to
exactly the default pose (`position_range=(1.0,1.0)` scale). One startup `physics_material` term exists but
its ranges are point values (not randomized) — see §7.

**Decisions resolved**

| Event (mode) | Params (G1-effective) |
|---|---|
| `reset_base` (reset) | pose `x:(-0.5,0.5) y:(-0.5,0.5) yaw:(-3.14,3.14)`; **velocity all (0,0)** |
| `reset_robot_joints` (reset) | `position_range=(1.0,1.0)` (default pose), `velocity_range=(0.0,0.0)` |
| `base_external_force_torque` (reset) | force (0,0), torque (0,0), body `torso_link`; (None in PLAY) |
| `push_robot` (interval) | **None** (disabled for G1) |
| `add_base_mass` (startup) | **None** (disabled for G1) |
| `base_com` (startup) | **None** (disabled for G1) |
| `physics_material` (startup) | static (0.8,0.8), dynamic (0.6,0.6), restitution (0,0) — point ranges, see §7 |

**Code (G1 reset overrides) —** `velocity/config/g1/rough_env_cfg.py`
```python
self.events.push_robot = None
self.events.add_base_mass = None
self.events.reset_robot_joints.params["position_range"] = (1.0, 1.0)
self.events.base_external_force_torque.params["asset_cfg"].body_names = ["torso_link"]
self.events.reset_base.params = {
    "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
    "velocity_range": {"x":(0.0,0.0),"y":(0.0,0.0),"z":(0.0,0.0),"roll":(0.0,0.0),"pitch":(0.0,0.0),"yaw":(0.0,0.0)},
}
self.events.base_com = None
```
PLAY additionally sets `events.base_external_force_torque = None` and `events.push_robot = None`.

---

## §4 Goal + Command + Termination

**Description.** The GOAL is the `FootPoseCommand` — an alternating **swing-foot touchdown pose**. One foot is
the stance (planted), the other the swing; the swing gets a target `(x,y,z,yaw)` sampled a reachable stride
**ahead of the stance foot** in the base-yaw frame (`step_length` forward, `stance_width` to the swing side,
`target_height` up). Advance-on-reach: the target advances (roles swap, new target sampled ahead of the
just-landed foot) only when the swing foot **lifts** above `lift_clearance` then **plants** (xy < `reach_xy_threshold`
AND z < `reach_z_threshold`). That reach one-shot-credits a step (`just_completed`). A safety timeout
(`max_step_duration_s`) force-advances a stalled env WITHOUT credit (`just_timed_out`, uncredited). Resets
start with the RIGHT foot planted (left swings first). Debug viz = a frame marker at the current target.
No task-tracking termination — only base/torso contact fall + time-out (inherited from velocity).

**NOTE — resolved config values differ from cfg defaults:** the env instantiates the command with
`step_length_range=(0.14,0.30)`, `stance_width=0.20`, `step_yaw_range=(-0.15,0.15)`, `target_height=0.05`,
`lift_clearance=0.08`, `reach_xy_threshold=0.08`, `reach_z_threshold=0.10`, **`max_step_duration_s=1.0`**
(cfg default is 1.2), `debug_vis=True`. `min_swing_duration_s` is left at its default 0.40 but is **UNUSED**
in `commands.py` (`_min_swing_steps` is computed in `__init__` but never referenced — the min-swing/stance
credit gate was removed; only lift+plant gates a credit).

**Decisions resolved**

| Decision | Value |
|---|---|
| Command class | `FootPoseCommand` (`mdp.FootPoseCommandCfg`), replaces `base_velocity` |
| Command dim (policy-facing) | **6**: `[x_b, y_b, z_b, yaw_err, side_sign(±1), timer_norm∈[0,1]]` |
| Target sampling | stride ahead of STANCE foot in base-yaw frame |
| step_length_range | (0.14, 0.30) m |
| stance_width | 0.20 m (lateral, swing side) |
| step_yaw_range | (-0.15, 0.15) rad |
| target_height | 0.05 m |
| lift_clearance | 0.08 m (must clear before a plant counts) |
| reach_xy_threshold | 0.08 m |
| reach_z_threshold | 0.10 m |
| max_step_duration_s | **1.0** (uncredited force-advance on timeout) |
| min_swing_duration_s | 0.40 (default; **unused in code**) |
| resampling_time_range | (1e9, 1e9) — advancing is reach/timeout-driven, not timer-driven |
| Initial stance | right foot planted (`stance_foot_idx=1`) → left swings first |
| Debug viz | `FRAME_MARKER_CFG` scale 0.15 at target pose (yaw only) |
| Public buffers for rewards | `swing_target_pos_w (N,3)`, `swing_target_yaw_w (N,)`, `swing_foot_pos_w (N,3)`, `swing_lifted (N)`, `reached_now (N)`, `just_completed (N)`, `just_timed_out (N)` |
| Metrics logged | `steps_followed`, `swing_foot_pos_error` |
| Termination: time_out | `mdp.time_out` (time_out=True), episode 20 s |
| Termination: fall | `mdp.illegal_contact` on `torso_link`, threshold 1.0 N |
| Termination: task-success | **none** (no reach-based done) |

**Code (command cfg instance) —** `config/g1/footstep_env_cfg.py::FootstepCommandsCfg`
```python
@configclass
class FootstepCommandsCfg:
    foot_pose = mdp.FootPoseCommandCfg(
        asset_name="robot",
        left_foot_body_name="left_ankle_roll_link",
        right_foot_body_name="right_ankle_roll_link",
        step_length_range=(0.14, 0.30),
        stance_width=0.20,
        step_yaw_range=(-0.15, 0.15),
        target_height=0.05,
        lift_clearance=0.08,
        reach_xy_threshold=0.08,
        reach_z_threshold=0.10,
        max_step_duration_s=1.0,
        debug_vis=True,
    )
```

**Code (command cfg class) —** `mdp/commands_cfg.py::FootPoseCommandCfg`
```python
@configclass
class FootPoseCommandCfg(CommandTermCfg):
    class_type: type = FootPoseCommand
    asset_name: str = MISSING
    left_foot_body_name: str = MISSING
    right_foot_body_name: str = MISSING
    step_length_range: tuple[float, float] = (0.14, 0.30)
    stance_width: float = 0.20
    step_yaw_range: tuple[float, float] = (-0.15, 0.15)
    target_height: float = 0.05
    lift_clearance: float = 0.08
    reach_xy_threshold: float = 0.08
    reach_z_threshold: float = 0.10
    min_swing_duration_s: float = 0.40
    max_step_duration_s: float = 1.2
    resampling_time_range: tuple[float, float] = (1.0e9, 1.0e9)
```

**Code (command generator) —** `mdp/commands.py::FootPoseCommand` (key methods, verbatim)
```python
def __init__(self, cfg, env):
    super().__init__(cfg, env)
    self.robot = env.scene[cfg.asset_name]
    left_ids = self.robot.find_bodies(cfg.left_foot_body_name)[0]
    right_ids = self.robot.find_bodies(cfg.right_foot_body_name)[0]
    ...
    self.foot_body_ids = torch.tensor([left_ids[0], right_ids[0]], device=self.device, dtype=torch.long)
    self._side_sign = torch.tensor([1.0, -1.0], device=self.device)
    self._max_step_steps = max(1, int(round(cfg.max_step_duration_s / self._env.step_dt)))
    self._min_swing_steps = max(1, int(round(cfg.min_swing_duration_s / self._env.step_dt)))  # computed, UNUSED
    self.stance_foot_idx = torch.ones(self.num_envs, device=self.device, dtype=torch.long)
    self.target_pos_w = torch.zeros(self.num_envs, 3, device=self.device)
    self.target_yaw_w = torch.zeros(self.num_envs, device=self.device)
    self.step_timer = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
    self.steps_followed = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
    self.swing_lifted = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
    self.reached_now = torch.zeros(self.num_envs, device=self.device)
    self.just_completed = torch.zeros(self.num_envs, device=self.device)
    self.just_timed_out = torch.zeros(self.num_envs, device=self.device)
    self.foot_command_b = torch.zeros(self.num_envs, 6, device=self.device)

@property
def swing_foot_idx(self): return 1 - self.stance_foot_idx
@property
def swing_foot_body_id(self): return self.foot_body_ids[self.swing_foot_idx]
@property
def swing_foot_pos_w(self): return self.robot.data.body_pos_w[self._env_ids, self.swing_foot_body_id]
@property
def swing_target_pos_w(self): return self.target_pos_w
@property
def swing_target_yaw_w(self): return self.target_yaw_w

def _resample_command(self, env_ids):  # called on episode reset
    self.stance_foot_idx[env_ids] = 1  # right planted → left swings first
    self.step_timer[env_ids] = 0
    self.steps_followed[env_ids] = 0
    self.swing_lifted[env_ids] = False
    self.reached_now[env_ids] = 0.0
    self.just_completed[env_ids] = 0.0
    self.just_timed_out[env_ids] = 0.0
    self._sample_target(torch.as_tensor(env_ids, device=self.device).flatten())

def _update_command(self):
    swing_pos = self.swing_foot_pos_w
    self.swing_lifted |= swing_pos[:, 2] > self.cfg.lift_clearance
    xy_dist = torch.norm(swing_pos[:, :2] - self.target_pos_w[:, :2], dim=-1)
    planted = (xy_dist < self.cfg.reach_xy_threshold) & (swing_pos[:, 2] < self.cfg.reach_z_threshold)
    self.step_timer += 1
    reached = planted & self.swing_lifted
    self.reached_now = reached.float()
    timeout = self.step_timer >= self._max_step_steps
    advance = reached | timeout
    self.just_completed = reached.float()
    self.steps_followed += reached.long()
    adv_ids = advance.nonzero().flatten()
    if len(adv_ids) > 0:
        self.stance_foot_idx[adv_ids] = 1 - self.stance_foot_idx[adv_ids]
        self.step_timer[adv_ids] = 0
        self.swing_lifted[adv_ids] = False
        self._sample_target(adv_ids)
    self._compute_command_vector()

def _sample_target(self, env_ids):
    n = len(env_ids)
    if n == 0: return
    stance_body = self.foot_body_ids[self.stance_foot_idx[env_ids]]
    stance_pos = self.robot.data.body_pos_w[env_ids, stance_body]
    base_yaw = self._base_yaw(env_ids)
    swing_idx = 1 - self.stance_foot_idx[env_ids]
    r = self.cfg
    step_len = torch.empty(n, device=self.device).uniform_(*r.step_length_range)
    lateral = self._side_sign[swing_idx] * r.stance_width
    dyaw = torch.empty(n, device=self.device).uniform_(*r.step_yaw_range)
    cos_y, sin_y = torch.cos(base_yaw), torch.sin(base_yaw)
    world_dx = cos_y * step_len - sin_y * lateral
    world_dy = sin_y * step_len + cos_y * lateral
    self.target_pos_w[env_ids, 0] = stance_pos[:, 0] + world_dx
    self.target_pos_w[env_ids, 1] = stance_pos[:, 1] + world_dy
    self.target_pos_w[env_ids, 2] = r.target_height
    self.target_yaw_w[env_ids] = wrap_to_pi(base_yaw + dyaw)

def _compute_command_vector(self):
    base_pos = self.robot.data.root_pos_w
    base_yaw = self._base_yaw()
    rel = self.target_pos_w - base_pos
    cos_y, sin_y = torch.cos(base_yaw), torch.sin(base_yaw)
    x_b = cos_y * rel[:, 0] + sin_y * rel[:, 1]
    y_b = -sin_y * rel[:, 0] + cos_y * rel[:, 1]
    z_b = rel[:, 2]
    yaw_err = wrap_to_pi(self.target_yaw_w - base_yaw)
    sign = torch.where(self.swing_foot_idx == 0, 1.0, -1.0)
    timer_norm = self.step_timer.float() / self._max_step_steps
    self.foot_command_b[:, 0] = x_b; self.foot_command_b[:, 1] = y_b; self.foot_command_b[:, 2] = z_b
    self.foot_command_b[:, 3] = yaw_err; self.foot_command_b[:, 4] = sign; self.foot_command_b[:, 5] = timer_norm
```
Debug viz: `FRAME_MARKER_CFG` copy, scale (0.15,0.15,0.15), prim `/Visuals/Command/foot_pose_goal`;
`_debug_vis_callback` draws marker at `target_pos_w` with yaw-only quat.

**Code (terminations, inherited) —** `velocity/velocity_env_cfg.py::TerminationsCfg`
```python
@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    base_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="base"), "threshold": 1.0},
    )
# G1 override: base_contact body_names -> "torso_link"
```

---

## §5 Observation

**Description.** Single `policy` group, corruption ON (train) / OFF (play), terms concatenated in order.
Velocity command replaced by the 6-D foot-pose command; height-scan dropped (flat).

**Decisions resolved**

| Obs term | func | dim | noise |
|---|---|---|---|
| base_lin_vel | `mdp.base_lin_vel` | 3 | Unoise(±0.1) |
| base_ang_vel | `mdp.base_ang_vel` | 3 | Unoise(±0.2) |
| projected_gravity | `mdp.projected_gravity` | 3 | Unoise(±0.05) |
| foot_pose_command | `mdp.generated_commands` (command_name="foot_pose") | 6 | — |
| joint_pos | `mdp.joint_pos_rel` | 37 | Unoise(±0.01) |
| joint_vel | `mdp.joint_vel_rel` | 37 | Unoise(±1.5) |
| actions | `mdp.last_action` | 37 | — |
| **total (concat)** | | **126** (ANALYTIC) | enable_corruption=True |

**Code —** `config/g1/footstep_env_cfg.py::FootstepObservationsCfg`
```python
@configclass
class FootstepObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, noise=Unoise(n_min=-0.1, n_max=0.1))
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
        projected_gravity = ObsTerm(func=mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05))
        foot_pose_command = ObsTerm(func=mdp.generated_commands, params={"command_name": "foot_pose"})
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-1.5, n_max=1.5))
        actions = ObsTerm(func=mdp.last_action)
        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True
    policy: PolicyCfg = PolicyCfg()
```

---

## §6 Reward

**Description.** 20 wired `RewTerm`s. Task block: swing-foot xy tracking (gated on `swing_lifted`),
swing-foot yaw tracking, forward-progress pull, one-shot step-completion bonus, biped air-time, alive,
base-height. Penalties: termination, upright, vertical/angular velocity, feet-slide, ankle limits,
smoothness (action-rate / dof-acc / dof-torques), and posture deviations (hips / arms / fingers / torso).
Weights are NOMINAL per-step (harbor strips the RewardManager × step_dt).

**NOTE:** `rewards.py` also DEFINES `foot_target_reached`, `step_timed_out`, `feet_both_airborne` — these
are **NOT wired** into `FootstepRewardsCfg` in this version (comment says step_timed_out/feet_both_airborne
were the plan; not present in the active cfg). Only the 20 terms below are active.

**Decisions resolved — active RewTerms (name → func, weight, key params)**

| # | name | func | weight | params |
|---|---|---|---|---|
| 1 | track_foot_pos | `mdp.track_swing_foot_xy_exp` | **1.0** | command_name=foot_pose, std=0.10 |
| 2 | track_foot_yaw | `mdp.track_swing_foot_yaw_exp` | **0.3** | std=0.5 |
| 3 | forward_progress | `mdp.base_forward_progress` | **1.5** | max_speed=0.6 |
| 4 | step_completed | `mdp.step_completed` | **10.0** | one-shot on credited plant |
| 5 | feet_air_time | `mdp.feet_air_time_positive_biped` | **1.0** | sensor `.*_ankle_roll_link`, threshold=0.3 |
| 6 | alive | `mdp.is_alive` | **0.5** | — |
| 7 | base_height | `mdp.base_height_l2` | **-2.0** | target_height=0.72 |
| 8 | termination_penalty | `mdp.is_terminated` | **-40.0** | — |
| 9 | upright | `mdp.flat_orientation_l2` | **-1.0** | — |
| 10 | lin_vel_z_l2 | `mdp.lin_vel_z_l2` | **-0.5** | — |
| 11 | ang_vel_xy_l2 | `mdp.ang_vel_xy_l2` | **-0.05** | — |
| 12 | feet_slide | `mdp.feet_slide` | **-0.5** | sensor+asset `.*_ankle_roll_link` |
| 13 | dof_pos_limits | `mdp.joint_pos_limits` | **-1.0** | ankle pitch/roll joints |
| 14 | action_rate_l2 | `mdp.action_rate_l2` | **-0.003** | — |
| 15 | dof_acc_l2 | `mdp.joint_acc_l2` | **-2.5e-7** | hip/knee joints |
| 16 | dof_torques_l2 | `mdp.joint_torques_l2` | **-1.0e-6** | hip/knee joints |
| 17 | joint_deviation_hip | `mdp.joint_deviation_l1` | **-0.5** | hip yaw/roll |
| 18 | joint_deviation_arms | `mdp.joint_deviation_l1` | **-0.5** | shoulder/elbow |
| 19 | joint_deviation_fingers | `mdp.joint_deviation_l1` | **-0.25** | hand joints |
| 20 | joint_deviation_torso | `mdp.joint_deviation_l1` | **-0.5** | torso_joint |

**Code (task-local reward funcs) —** `mdp/rewards.py` (verbatim; wired funcs + re-exports)
```python
from isaaclab_tasks.manager_based.locomotion.velocity.mdp.rewards import (  # re-exported
    feet_air_time_positive_biped, feet_slide,
)

def track_swing_foot_xy_exp(env, std, command_name):
    """Dense exp tracking of swing foot to its target xy, GATED on swing_lifted (airborne)."""
    command = env.command_manager.get_term(command_name)
    error = torch.sum(torch.square(command.swing_target_pos_w[:, :2] - command.swing_foot_pos_w[:, :2]), dim=1)
    return command.swing_lifted.float() * torch.exp(-error / std**2)

def base_forward_progress(env, command_name, max_speed):
    """Reward forward base translation in heading dir, clipped to [0, max_speed]."""
    command = env.command_manager.get_term(command_name)
    robot = env.scene[command.cfg.asset_name]
    base_yaw = euler_xyz_from_quat(robot.data.root_quat_w)[2]
    vel_w = robot.data.root_lin_vel_w
    forward = torch.cos(base_yaw) * vel_w[:, 0] + torch.sin(base_yaw) * vel_w[:, 1]
    return torch.clamp(forward, min=0.0, max=max_speed)

def track_swing_foot_yaw_exp(env, std, command_name):
    command = env.command_manager.get_term(command_name)
    quat = env.scene[command.cfg.asset_name].data.body_quat_w[command._env_ids, command.swing_foot_body_id]
    foot_yaw = euler_xyz_from_quat(quat)[2]
    yaw_err = wrap_to_pi(command.swing_target_yaw_w - foot_yaw)
    return torch.exp(-torch.square(yaw_err) / std**2)

def step_completed(env, command_name):
    """One-shot 1.0 on the step where the swing foot genuinely plants and the command advances."""
    return env.command_manager.get_term(command_name).just_completed

# DEFINED BUT NOT WIRED in this env cfg:
def foot_target_reached(env, command_name):  # dense 1.0 while planted on target
    return env.command_manager.get_term(command_name).reached_now
def step_timed_out(env, command_name):  # penalty 1.0 when a step expires uncredited
    return env.command_manager.get_term(command_name).just_timed_out
def feet_both_airborne(env, command_name, height):  # penalty 1.0 when both feet off ground
    command = env.command_manager.get_term(command_name)
    feet_z = env.scene[command.cfg.asset_name].data.body_pos_w[:, command.foot_body_ids, 2]
    return (feet_z > height).all(dim=1).float()
```
Other reward funcs are stdlib `isaaclab.envs.mdp` (`is_alive`, `is_terminated`, `base_height_l2`,
`flat_orientation_l2`, `lin_vel_z_l2`, `ang_vel_xy_l2`, `joint_pos_limits`, `action_rate_l2`,
`joint_acc_l2`, `joint_torques_l2`, `joint_deviation_l1`), and `feet_air_time_positive_biped`/`feet_slide`
re-exported from the velocity task.

---

## §7 Domain Randomization

**Effective DR: essentially NONE (`<no DR>`).**

The velocity base defines startup/interval randomization, but the G1 override disables all of it and the
one remaining startup term uses point ranges:

| Term | Status on G1 |
|---|---|
| `push_robot` (interval velocity push) | **None** — disabled |
| `add_base_mass` (startup mass ±5 kg) | **None** — disabled |
| `base_com` (startup CoM ±) | **None** — disabled |
| `base_external_force_torque` (reset) | force (0,0) / torque (0,0) — no-op |
| `physics_material` (startup friction) | static (0.8,0.8), dynamic (0.6,0.6), restitution (0,0) — **point ranges, not randomized** |
| observation noise | present (see §5 Unoise terms), corruption ON in train / OFF in play |

The only per-episode stochasticity is `reset_base` pose (x,y ±0.5 m, yaw ±π) and the command's own
random target sampling — no mass/friction/push/gain randomization is wired.
