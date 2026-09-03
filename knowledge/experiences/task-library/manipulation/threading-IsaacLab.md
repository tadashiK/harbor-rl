# Threading — Implementation Spec

- robot: Bimanual UF850 arms + dual Allegro hands (44 DoF)
- simulator: IsaacLab (Isaac Sim, manager-based)
- objects: cube_with_hole, drill, table
- bimanual: true
- summary: Thread a drill head through the hole of a cube held by the other hand.

> Source package name anonymized as `bimanual_suite`. This task comes from an internal
> bimanual manipulation suite rather than a public repo; the design below is otherwise verbatim.

> **Important design note (weights):** The reward is a fully authored 24-term shaping pipeline (all functions below), and the task registers to gym id `ThreadingEnv-v0`. Each term's weight is given in §6.

## Task summary

Bimanual precision-insertion ("threading"). Two UF850 arms, each ending in a 4-finger Allegro hand (`Robot` = right at y=-0.475, `Robot_left` = left at y=+0.475), operate over a table. **Object_0** is a `cube_with_hole` (scaled 0.75, mass 0.2 kg); **Object_1** is a `drill` (scale 1.0, mass 0.5 kg) whose head must be threaded *through the hole* of the cube. One hand grasps and lifts/orients the holed cube, the other grasps and orients the drill, and the two must be brought into alignment so the drill head passes through the cube's hole while both are held aloft. Success is defined by a dedicated `drill_head_frame` (offset frame on the drill, `pos=(-0.15,0,0.07)`) coming within 3 cm of the cube center while the cube is above z=0.2, held **consecutively** for ≥20 steps (`success_bonus` / `max_consecutive_success`). Actions are an **EMA cumulative-relative joint-position** command over all 44 joints (22 per arm). Gravity is disabled on both robots (position-controlled arms). The env supports symmetric-env augmentation (C2 group) via `BaseEnv` and mirrors every reward term with a `_symmetry` variant.

Key derived quantities:
- **action_dim = 44** (22 right + 22 left; per arm = 6 arm joints + 16 Allegro joints).
- **observation dim = 180** (see §5 breakdown).
- **objects manipulated = 2** (cube_with_hole, drill) + kinematic table.
- **episode_length_s = 8.3333**, decimation = 6, sim dt = 1/120 → 20 policy steps/s, ~166 policy steps/episode.

---

## §1 Registration + Scene

**Description.** Registered as `ThreadingEnv-v0`, entry point `ThreadingEnv` (subclass of shared `BaseEnv`). Scene = `ThreadingSceneCfg(num_envs=4096, env_spacing=3.0)`. Both robots spawn from colored UF850+Allegro USDs with **gravity disabled**, implicit actuators (arm joints stiff 2000/damp 16; per-finger Allegro gains vary by joint index). Two rigid objects: a holed cube (`cube_with_hole.usd`, scale 0.75, mass 0.2, dynamic) and a `drill.usd` (scale 1.0, mass 0.5, dynamic). Kinematic table from shared base (rotated 90° about Z). Ground at z=-0.82, dome light. 8 contact sensors (4 fingertips × 2 hands: `if5/mf5/pf5/th5`), all filtered against **Object_0** (the cube) only. Three `FrameTransformer`s hang off Object_1 (drill): `drill_head_frame` (the thread-tip probe, offset (-0.15,0,0.07)) and two symmetric approach frames. Sim: dt=1/120, decimation=6, render_interval=6; PhysX contact/patch counts 2^24; global material static/dynamic friction 1.5/1.0, restitution 0.

**Decisions resolved.**

| Decision | Value |
|---|---|
| gym id / entry point | `ThreadingEnv-v0` → `bimanual_suite.env.tasks.Threading.env:ThreadingEnv`, `disable_env_checker=True`, kwarg `env_cfg_entry_point=ThreadingEnvCfg` |
| num_envs / env_spacing | 4096 / 3.0 |
| right robot USD / init pos | `assets/ufactory850/uf850_allegro_right_colored.usd` / pos=(-0.274, -0.475, 0.01) |
| left robot USD / init pos | `assets/ufactory850/uf850_allegro_left_colored.usd` / pos=(-0.274, 0.475, 0.01) |
| robot gravity | disabled (both) |
| arm actuator | joints[1-6]: stiffness 2000, damping 16 |
| Allegro finger actuators | f1 325/20, f2 425/25, f3 245/15, f4 1050/65; thumb jth1 100/5, jth2 300/15, jth3 1270/100, jth4 1000/50 |
| Object_0 (cube_with_hole) | scale 0.75, mass 0.2, dynamic (kinematic_enabled=False, gravity on), contact sensors active |
| Object_1 (drill) | scale 1.0, mass 0.5, dynamic, contact sensors active |
| table | `assets/object/table.usd`, kinematic, rot=(0.7071,0,0,0.7071) (base cfg) |
| ground / light | ground plane z=-0.82; DomeLight color 0.75, intensity 2500 |
| contact sensors | 8 total, tips if5/mf5/pf5/th5 × {Robot, Robot_left}, all filtered vs **Object_0** |
| frame transformers | `drill_head_frame` offset pos=(-0.15,0,0.07); `object_approach_frame` pos=(0.05,0,0) rot=(0,0,0.7071,-0.7071); `object_approach_frame_symmetry` rot=(0,0,0.7071,0.7071) |
| sim timing | dt=1/120, decimation=6, render_interval=6, episode_length_s=8.3333 |
| physx | gpu_max_rigid_contact_count=2^24, gpu_max_rigid_patch_count=2^24; friction 1.5/1.0, restitution 0 |
| viewer eye | (-3.5, 0.0, 3.5) |
| replicate_physics | False (base scene) |

**Code (registration).**
```python
# bimanual_suite/env/__init__.py
from .tasks.Threading.env_cfg import ThreadingEnvCfg
gym.register(
    id="ThreadingEnv-v0",
    entry_point="bimanual_suite.env.tasks.Threading.env:ThreadingEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": ThreadingEnvCfg,
    },
)
```

**Code (env class — custom step / trackers).**
```python
# bimanual_suite/env/tasks/Threading/env.py
class ThreadingEnv(BaseEnv):
    is_vector_env: ClassVar[bool] = True
    metadata: ClassVar[dict[str, Any]] = {
        "render_modes": [None, "human", "rgb_array"],
        "isaac_sim_version": get_version(),
    }
    cfg: ThreadingEnvCfg

    def step(self, action: torch.Tensor) -> VecEnvStepReturn:
        super().step(action)
        # Debug only
        if self.cfg.visualize_marker:
            self.markers['arm_l']['ee_marker'].visualize(self.scene["object_0"].data.root_state_w[:, :3], self.scene["object_0"].data.root_state_w[:, 3:7])
            self.markers['arm_r']['goal_marker'].visualize(self.scene["object_1"].data.root_state_w[:, :3], self.scene["object_1"].data.root_state_w[:, 3:7])
            left_palm_idx = self.scene["robot"].find_bodies("palm_link")[0]
            left_palm = self.scene["robot"].data.body_state_w[:, left_palm_idx, :7].reshape(-1, 7)
            self.markers['arm_r']['ee_marker'].visualize(left_palm[:, :3], left_palm[:, 3:7])
        return self.obs_buf, self.reward_buf, self.reset_terminated, self.reset_time_outs, self.extras

    def _pre_init_process(self):
        super()._pre_init_process()
        self.success_tracker_step = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)

    def _post_reset_process(self, env_ids):
        super()._post_reset_process(env_ids)
        self.success_tracker_step[env_ids] = 0.0
```

**Code (ThreadingEnvCfg + scene).** (verbatim)
```python
# bimanual_suite/env/tasks/Threading/env_cfg.py
FRAME_MARKER_SMALL_CFG = FRAME_MARKER_CFG.copy()
FRAME_MARKER_SMALL_CFG.markers["frame"].scale = (0.10, 0.10, 0.10)

@configclass
class ThreadingSceneCfg(BaseSceneCfg):
    # robots
    robot = ArticulationCfg(
        prim_path="/World/envs/env_.*/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{bimanual_suite.LIB_PATH}/assets/ufactory850/uf850_allegro_right_colored.usd",
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True,
                max_depenetration_velocity=1000.0,
                max_linear_velocity=1000,
                max_angular_velocity=1000,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False, solver_position_iteration_count=16, solver_velocity_iteration_count=1,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            joint_pos={
                "joint1": 0.6, "joint2": 0.3, "joint3": -0.6, "joint4": 0.0, "joint5": -0.8, "joint6": -1.57,
                # hand
                "jif1": 0.0, "jif2": 0.4, "jif3": 0.4, "jif4": 0.0,
                "jmf1": 0.0, "jmf2": 0.4, "jmf3": 0.4, "jmf4": 0.0,
                "jpf1": 0.0, "jpf2": 0.4, "jpf3": 0.4, "jpf4": 0.0,
                "jth1": 1.3, "jth2": 0.0, "jth3": 0.2, "jth4": 0.0,
            },
            pos=(-0.274, -0.475, 0.01),
        ),
        actuators={
            "xArm_1-6": ImplicitActuatorCfg(joint_names_expr=["joint[1-6]"], stiffness=2000.0, damping=16.0),
            "allegro_hand_1": ImplicitActuatorCfg(joint_names_expr=["j.*f1"], stiffness=325.0, damping=20.0),
            "allegro_hand_2": ImplicitActuatorCfg(joint_names_expr=["j.*f2"], stiffness=425.0, damping=25.0),
            "allegro_hand_3": ImplicitActuatorCfg(joint_names_expr=["j.*f3"], stiffness=245.0, damping=15.0),
            "allegro_hand_4": ImplicitActuatorCfg(joint_names_expr=["j.*f4"], stiffness=1050.0, damping=65.0),
            "allegro_hand_thumb_1": ImplicitActuatorCfg(joint_names_expr=["jth1"], stiffness=100.0, damping=5.0),
            "allegro_hand_thumb_2": ImplicitActuatorCfg(joint_names_expr=["jth2"], stiffness=300.0, damping=15.0),
            "allegro_hand_thumb_3": ImplicitActuatorCfg(joint_names_expr=["jth3"], stiffness=1270.0, damping=100.0),
            "allegro_hand_thumb_4": ImplicitActuatorCfg(joint_names_expr=["jth4"], stiffness=1000.0, damping=50.0),
        },
    )

    robot_left = ArticulationCfg(
        prim_path="/World/envs/env_.*/Robot_left",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{bimanual_suite.LIB_PATH}/assets/ufactory850/uf850_allegro_left_colored.usd",
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True, max_depenetration_velocity=1000.0, max_linear_velocity=1000, max_angular_velocity=1000,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False, solver_position_iteration_count=16, solver_velocity_iteration_count=1,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            joint_pos={
                "joint1": 0.25, "joint2": 0.0, "joint3": -0.5, "joint4": -1.4, "joint5": -1.0, "joint6": 3.14,
                # hand
                "jif1": 0.0, "jif2": 0.4, "jif3": 0.4, "jif4": 0.0,
                "jmf1": 0.0, "jmf2": 0.4, "jmf3": 0.4, "jmf4": 0.0,
                "jpf1": 0.0, "jpf2": 0.4, "jpf3": 0.4, "jpf4": 0.0,
                "jth1": 1.3, "jth2": 0.0, "jth3": 0.2, "jth4": 0.0,
            },
            pos=(-0.274, 0.475, 0.01),
        ),
        actuators={ ...identical gains to `robot`... },  # xArm 2000/16; f1 325/20; f2 425/25; f3 245/15; f4 1050/65; thumb 100/5,300/15,1270/100,1000/50
    )

    object_0 = RigidObjectCfg(
        prim_path=f"/World/envs/env_.*/Object_0",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{bimanual_suite.LIB_PATH}/assets/object/cube_with_hole.usd",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=False, disable_gravity=False, max_linear_velocity=1000, max_angular_velocity=1000,
                solver_position_iteration_count=16, solver_velocity_iteration_count=1, max_depenetration_velocity=1000.0,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.2),  # 0.15
            activate_contact_sensors=True,
            scale=(0.75, 0.75, 0.75),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(lin_vel=(0.0,0.0,0.0), ang_vel=(0.0,0.0,0.0), pos=(0.0,0.0,0.0)),
    )

    object_1 = RigidObjectCfg(
        prim_path=f"/World/envs/env_.*/Object_1",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{bimanual_suite.LIB_PATH}/assets/object/drill.usd",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=False, disable_gravity=False, max_linear_velocity=1000, max_angular_velocity=1000,
                solver_position_iteration_count=16, solver_velocity_iteration_count=1, max_depenetration_velocity=1000.0,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.5),  # 0.85
            activate_contact_sensors=True,
            scale=(1.0, 1.0, 1.0),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(lin_vel=(0.0,0.0,0.0), ang_vel=(0.0,0.0,0.0), pos=(0.0,0.0,0.0)),
    )

    # sensors — RIGHT hand tips, all filtered vs Object_0
    contact_sensors_0 = ContactSensorCfg(prim_path="/World/envs/env_.*/Robot/if5",  update_period=0.0, debug_vis=True, filter_prim_paths_expr=["{ENV_REGEX_NS}/Object_0"])  # index
    contact_sensors_1 = ContactSensorCfg(prim_path="/World/envs/env_.*/Robot/mf5",  update_period=0.0, debug_vis=True, filter_prim_paths_expr=["{ENV_REGEX_NS}/Object_0"])  # middle
    contact_sensors_2 = ContactSensorCfg(prim_path="/World/envs/env_.*/Robot/pf5",  update_period=0.0, debug_vis=True, filter_prim_paths_expr=["{ENV_REGEX_NS}/Object_0"])  # pinky
    contact_sensors_3 = ContactSensorCfg(prim_path="/World/envs/env_.*/Robot/th5",  update_period=0.0, debug_vis=True, filter_prim_paths_expr=["{ENV_REGEX_NS}/Object_0"])  # thumb
    # LEFT hand tips, all filtered vs Object_0
    contact_sensors_0_left = ContactSensorCfg(prim_path="/World/envs/env_.*/Robot_left/if5", update_period=0.0, debug_vis=True, filter_prim_paths_expr=["{ENV_REGEX_NS}/Object_0"])
    contact_sensors_1_left = ContactSensorCfg(prim_path="/World/envs/env_.*/Robot_left/mf5", update_period=0.0, debug_vis=True, filter_prim_paths_expr=["{ENV_REGEX_NS}/Object_0"])
    contact_sensors_2_left = ContactSensorCfg(prim_path="/World/envs/env_.*/Robot_left/pf5", update_period=0.0, debug_vis=True, filter_prim_paths_expr=["{ENV_REGEX_NS}/Object_0"])
    contact_sensors_3_left = ContactSensorCfg(prim_path="/World/envs/env_.*/Robot_left/th5", update_period=0.0, debug_vis=True, filter_prim_paths_expr=["{ENV_REGEX_NS}/Object_0"])

    # frame transformers on the drill (Object_1)
    drill_head_frame = FrameTransformerCfg(               # the "thread tip" probe used for success
        prim_path="{ENV_REGEX_NS}/Object_1", debug_vis=False,
        visualizer_cfg=FRAME_MARKER_SMALL_CFG.replace(prim_path="/Visuals/ObjectApproachFrameTransformer"),
        target_frames=[FrameTransformerCfg.FrameCfg(prim_path="{ENV_REGEX_NS}/Object_1", name="approach_frame",
            offset=OffsetCfg(pos=(-0.15, 0.0, 0.07), rot=(1.0, 0.0, 0.0, 0.0)))],
    )
    object_approach_frame = FrameTransformerCfg(          # left-hand palm approach target
        prim_path="{ENV_REGEX_NS}/Object_1", debug_vis=False,
        visualizer_cfg=FRAME_MARKER_SMALL_CFG.replace(prim_path="/Visuals/ObjectApproachFrameTransformer"),
        target_frames=[FrameTransformerCfg.FrameCfg(prim_path="{ENV_REGEX_NS}/Object_1", name="approach_frame",
            offset=OffsetCfg(pos=(0.05, 0.0, 0.0), rot=(0.0, 0.0, 0.7071, -0.7071)))],
    )
    object_approach_frame_symmetry = FrameTransformerCfg( # mirror approach target
        prim_path="{ENV_REGEX_NS}/Object_1", debug_vis=False,
        visualizer_cfg=FRAME_MARKER_SMALL_CFG.replace(prim_path="/Visuals/ObjectApproachFrameTransformer"),
        target_frames=[FrameTransformerCfg.FrameCfg(prim_path="{ENV_REGEX_NS}/Object_1", name="approach_frame",
            offset=OffsetCfg(pos=(0.05, 0.0, 0.0), rot=(0.0, 0.0, 0.7071, 0.7071)))],
    )
```

**Code (shared base scene — table/ground/light + top-level cfg timing).**
```python
# bimanual_suite/env/tasks/manager_based_env_cfg.py
@configclass
class BaseSceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(prim_path="/World/ground", spawn=sim_utils.GroundPlaneCfg(),
                          init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -0.82)))
    light = AssetBaseCfg(prim_path="/World/light",
                         spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=2500.0))
    table: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Table",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{bimanual_suite.LIB_PATH}/assets/object/table.usd", activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=False,
                solver_position_iteration_count=16, solver_velocity_iteration_count=1, max_depenetration_velocity=10.0),
            scale=(1.0, 1.0, 1.0)),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0), rot=(0.70710678, 0, 0., 0.70710678)))
    replicate_physics = False

# BaseEnvCfg (sim/physx + timing)
    sim = SimulationCfg(
        physics_material=RigidBodyMaterialCfg(static_friction=1.5, dynamic_friction=1.0, restitution=0.0, restitution_combine_mode=min),
        physx=PhysxCfg(gpu_max_rigid_contact_count=2**24, gpu_max_rigid_patch_count=2**24))
    def __post_init__(self):
        self.decimation = 6
        self.episode_length_s = 8.3333
        self.sim.dt = 1.0 / 120.0
        self.sim.render_interval = self.decimation

# ThreadingEnvCfg
@configclass
class ThreadingEnvCfg(BaseEnvCfg):
    name = "Threading"
    scene = ThreadingSceneCfg(num_envs=4096, env_spacing=3.0)
    events = ThreadingEventCfg(); commands = ThreadingCommandsCfg(); observations = ThreadingObservationsCfg()
    actions = ThreadingActionsCfg(); terminations = ThreadingTerminationsCfg(); rewards = ThreadingRewardsCfg()
    num_object = 2
    action_dim = 44
    visualize_marker = False
    def __post_init__(self):
        super().__post_init__()
        self.viewer.eye = (-3.5, 0.0, 3.5)
```

---

## §2 Actions

**Description.** Two `EMACumulativeRelativeJointPositionActionCfg` terms, one per arm, each covering all joints (`joint_names=[".*"]` → 22 joints: 6 arm + 16 Allegro). **action_dim = 44.** The action term is a cumulative-relative EMA position controller: raw policy output is first multiplied by a per-DOF `action_scale` (applied in `BaseEnv.step`, NOT in the action term), then integrated (`del_action`), offset by the reset joint pose, then low-pass filtered with `alpha=0.2` and clamped to per-joint limits (`JOINT_LOWER_LIMIT`/`_LEFT`, `JOINT_UPPER_LIMIT`/`_LEFT`).

**Per-step rule** (`process_actions`, after outer `action * action_scale`):
1. `p = scale * raw_action` (affine transform from base JointPositionAction; here scale=1.0 in cfg, real scaling done via `env._scale`).
2. `p += del_action` (cumulative integral); `del_action = p.clone()`.
3. `p += init_joint_pos` (reset-time joint pose).
4. `ema = alpha * p + (1-alpha) * prev_applied`.
5. `p = clamp(ema, joint_lower_limit, joint_upper_limit)`; `prev_applied = p`.

`BaseEnv.step`: `action = action * self._scale` where `self._scale = torch.tensor(cfg.action_scale)` — a 44-vector: arm 0.05 (×6), hand 0.03 (with `jth3`=0.015), repeated for both arms.

**Decisions resolved.**

| Decision | Value |
|---|---|
| action term class | `EMACumulativeRelativeJointPositionAction` (subclass of IsaacLab `JointPositionAction`) |
| terms | `arm_hand_action` (robot) + `arm_hand_action_left` (robot_left) |
| joint_names | `[".*"]` → 22 per arm |
| action_dim | 44 |
| cfg scale | 1.0 (both); real per-DOF scaling via `env._scale` = `cfg.action_scale` |
| alpha (EMA) | 0.2 (both) |
| use_default_offset | False |
| clamp limits (right) | JOINT_LOWER_LIMIT / JOINT_UPPER_LIMIT |
| clamp limits (left) | JOINT_LOWER_LIMIT_LEFT / JOINT_UPPER_LIMIT_LEFT (thumb-side finger limits mirrored) |
| action_scale per DOF | arm ×6 = 0.05; hand = 0.03 except `jth3`=0.015; pattern repeated for both arms (44 total) |

**Code (config + scale vector).**
```python
# env_cfg.py
@configclass
class ThreadingActionsCfg:
    arm_hand_action = EMACumulativeRelativeJointPositionActionCfg(
        asset_name="robot", joint_names=[".*"], scale=1.0, use_default_offset=False,
        joint_lower_limit=JOINT_LOWER_LIMIT, joint_upper_limit=JOINT_UPPER_LIMIT, alpha=0.2)
    arm_hand_action_left = EMACumulativeRelativeJointPositionActionCfg(
        asset_name="robot_left", joint_names=[".*"], scale=1.0, use_default_offset=False,
        joint_lower_limit=JOINT_LOWER_LIMIT_LEFT, joint_upper_limit=JOINT_UPPER_LIMIT_LEFT, alpha=0.2)

# ThreadingEnvCfg.action_scale (44 values)
action_scale = [0.05,0.05,0.05,0.05,0.05,0.05,
                0.03,0.03,0.03,0.03, 0.03,0.03,0.03,0.03, 0.03,0.03,0.03,0.015, 0.03,0.03,0.03,0.03,
                0.05,0.05,0.05,0.05,0.05,0.05,
                0.03,0.03,0.03,0.03, 0.03,0.03,0.03,0.03, 0.03,0.03,0.03,0.015, 0.03,0.03,0.03,0.03]  # jth3 needs smaller rate
```

**Code (action term — verbatim).**
```python
# bimanual_suite/env/action_managers/actions.py
class EMACumulativeRelativeJointPositionAction(JointPositionAction):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        if isinstance(cfg.alpha, float):
            if not 0.0 <= cfg.alpha <= 1.0:
                raise ValueError(...)
            self._alpha = cfg.alpha
        elif isinstance(cfg.alpha, dict):
            self._alpha = torch.ones((env.num_envs, self.action_dim), device=self.device)
            index_list, names_list, value_list = string_utils.resolve_matching_names_values(cfg.alpha, self._joint_names)
            for name, value in zip(names_list, value_list): ...
            self._alpha[:, index_list] = torch.tensor(value_list, device=self.device)
        self._prev_applied_actions = torch.zeros_like(self.processed_actions)
        self.del_action = torch.zeros((self._env.num_envs, self.action_dim), device=self._env.device)
        self.init_joint_pos = self._asset.data.joint_pos[:, self._joint_ids].clone()
        self.joint_lower_limit = torch.tensor(cfg.joint_lower_limit, device=self.device) if cfg.joint_lower_limit is not None else None
        self.joint_upper_limit = torch.tensor(cfg.joint_upper_limit, device=self.device) if cfg.joint_upper_limit is not None else None

    def reset(self, env_ids=None):
        if env_ids is None: env_ids = slice(None)
        super().reset(env_ids)
        self._prev_applied_actions[env_ids, :] = self._asset.data.joint_pos[env_ids][:, self._joint_ids].clone()
        self.del_action[env_ids, :] = torch.zeros((env_ids.shape[0], self.action_dim), device=self.device)
        self.init_joint_pos[env_ids, :] = self._asset.data.joint_pos[env_ids][:, self._joint_ids].clone()

    def process_actions(self, actions):
        super().process_actions(actions)
        self._processed_actions += self.del_action
        self.del_action = self._processed_actions.clone()
        self._processed_actions += self.init_joint_pos.clone()
        ema_actions = self._alpha * self._processed_actions
        ema_actions += (1.0 - self._alpha) * self._prev_applied_actions
        if self.joint_lower_limit is not None and self.joint_upper_limit is not None:
            self._processed_actions[:] = torch.clamp(ema_actions, self.joint_lower_limit, self.joint_upper_limit)
        else:
            self._processed_actions[:] = ema_actions
        self._prev_applied_actions[:] = self._processed_actions[:]
```

Joint limits (per arm, 22 DOF; arm 6 then jif/jmf/jpf/jth × [1,2,3,4]):
```python
JOINT_LOWER_LIMIT = [-6.283,-2.304,-4.224,-6.283,-2.164,-6.283,
                     -0.05,-0.05,-0.570,0.364,  -0.296,-0.296,-0.296,-0.205,  -0.274,-0.274,-0.274,-0.290,  -0.327,-0.327,-0.327,-0.262]
JOINT_UPPER_LIMIT = [6.283,2.304,0.061,6.283,2.164,6.283,
                     0.570,0.05,0.05,1.497,  1.710,1.710,1.710,1.130,  1.809,1.809,1.809,1.633,  1.718,1.718,1.718,1.820]
JOINT_LOWER_LIMIT_LEFT = [ ...arm same...,  -0.570,-0.05,-0.05,0.364,  <rest identical to right> ]  # jif1/jpf1 swapped sign vs right
JOINT_UPPER_LIMIT_LEFT = [ ...arm same...,   0.05,0.05,0.570,1.497,  <rest identical to right> ]
```

---

## §3 Reset / Events

**Description.** `ThreadingEventCfg` (extends `BaseEventCfg`, mode `"reset"` only — no periodic DR). Inherited `reset_robot_joints` resets the **right** robot to its default joint pose (scaled by 1.0, i.e. exact default) with symmetry mirroring via `reset_joints_by_symmetry`; the task adds `reset_robot_joints_left` for the left robot. Two object-reset terms place the cube and drill at **fixed** poses each episode (ranges are degenerate [a,a] → deterministic). Cube: x=0.1, y=-0.3, z=0.05, roll=-1.57. Drill: x=0.15, y=0.2, z=0.06, yaw=1.57. (These are env-frame offsets added to the object default root state in `reset_object`.) `BaseEnv._post_reset_process` records `object_init_pos`/`object_init_orient` per object and resets the command manager; `ThreadingEnv._post_reset_process` also zeroes `success_tracker_step`.

**Decisions resolved.**

| Decision | Value |
|---|---|
| mode | all `"reset"` (per-episode) |
| right robot reset | inherited `reset_robot_joints` (func `reset_joints_by_symmetry`, pos_range (1.0,1.0), vel (0,0)) |
| left robot reset | `reset_robot_joints_left` (same func, asset `robot_left`, pos_range (1.0,1.0)) |
| cube reset (Object_0) | `reset_object` id 0, pose x[0.1,0.1] y[-0.3,-0.3] z[0.05,0.05] roll[-1.57,-1.57] |
| drill reset (Object_1) | `reset_object` id 1, pose x[0.15,0.15] y[0.2,0.2] z[0.06,0.06] yaw[1.57,1.57] |
| velocity range | {} (zero) for both objects |
| symmetric-env augmentation | via `BaseEnv._reset_idx` — 50% envs mirrored under C2 rep (`rep_Q_js`, `rep_Rd`, `rep_SO3_flat`) when `hydra_cfg.task.symmetry.symmetric_envs` |

**Code.**
```python
# env_cfg.py
@configclass
class ThreadingEventCfg(BaseEventCfg):
    reset_robot_joints_left = EventTerm(
        func=reset_joints_by_symmetry, mode="reset",
        params={"position_range": (1.0, 1.0), "velocity_range": (0.0, 0.0), "asset_cfg": SceneEntityCfg("robot_left")})
    reset_object_cube = EventTerm(
        func=reset_object, mode="reset",
        params={"pose_range": {"x":[0.1,0.1], "y":[-0.3,-0.3], "z":[0.05,0.05], "roll":[-1.57,-1.57]},
                "velocity_range": {}, "object_id": 0})
    reset_object_drill = EventTerm(
        func=reset_object, mode="reset",
        params={"pose_range": {"x":[0.15,0.15], "y":[0.2,0.2], "z":[0.06,0.06], "yaw":[1.57,1.57]},
                "velocity_range": {}, "object_id": 1})

# BaseEventCfg (inherited)
    reset_robot_joints = EventTerm(func=reset_joints_by_symmetry, mode="reset",
        params={"position_range": (1.0, 1.0), "velocity_range": (0.0, 0.0)})
```
`reset_object` (reset_mdps.py): samples uniform in pose_range (here degenerate), adds to `default_root_state[:,0:3] + env_origins`, composes quat from euler delta; writes root pose + velocity. `reset_joints_by_symmetry` mirrors the left/right default joint pose via `env.rep_Q_js` for symmetric env_ids, scales by pos_range, clamps to soft limits.

---

## §4 Goal + Termination

**Description.** Two `TargetPositionCommandCfg` goal commands drive markers + metrics (position_error, orientation_error, consecutive_success) but success is ultimately determined by the custom `success_bonus`/`max_consecutive_success` reward+termination that checks the **drill_head_frame ↔ cube distance**. `cube_target_pos` (object_0): goal x0,y-0.15,z0.3,roll-1.57, success_threshold 0.1 pos / 0.97 orient. `drill_target_pos` (object_1): x0,y0.05,z0.23,yaw1.57, thresh 0.1 / 0.97. `update_goal_on_success=False` (fixed goal). Command frames are visualized (frame_prim markers, scale 0.1).

**Termination:** `max_consecutive_success` (task mdp) fires `done=True` when `env.success_tracker_step >= 20`. `success_tracker_step` is incremented by the `success_bonus` reward function each step the drill-head is within 3 cm of the cube **and** the cube z>0.2, reset to 0 otherwise. Plus inherited `time_out` (`episode_length_s=8.3333`). No explicit failure/drop termination.

**Success predicate (threading):** `dist(drill_head_frame.target_pos_w, cube.root_pos_w) < 0.03 * (cube_z > 0.2)` held for 20 consecutive steps. (Note the multiply: threshold becomes 0.03 only while cube is lifted, else 0 → impossible, forcing a lifted alignment.)

**Decisions resolved.**

| Decision | Value |
|---|---|
| termination (task) | `threading.max_consecutive_success` num_success=20 → `done` when `success_tracker_step>=20` |
| termination (base) | `time_out` (episode_length_s=8.3333, ~166 steps) |
| success tracker source | `threading.success_bonus`: drill_head↔cube dist<0.03 AND cube z>0.2, consecutive count |
| cube command | `cube_target_pos` obj 0: x0,y-0.15,z0.3,roll-1.57; thr 0.1 / orient 0.97; no update-on-success; debug_vis |
| drill command | `drill_target_pos` obj 1: x0,y0.05,z0.23,yaw1.57; thr 0.1 / orient 0.97; debug_vis |
| command resampling | `resampling_time_range=(1e6,1e6)` → never resamples on time |
| failure termination | none (only success + time_out) |

**Code.**
```python
# env_cfg.py
@configclass
class ThreadingCommandsCfg(BaseCommandsCfg):
    cube_target_pos = TargetPositionCommandCfg(object_id=0, success_threshold=0.1, success_threshold_orient=0.97,
        pose_range={"x":[0.0,0.0], "y":[-0.15,-0.15], "z":[0.3,0.3], "roll":[-1.57,-1.57]},
        update_goal_on_success=False, debug_vis=True)
    drill_target_pos = TargetPositionCommandCfg(object_id=1, success_threshold=0.1, success_threshold_orient=0.97,
        pose_range={"x":[0.0,0.0], "y":[0.05,0.05], "z":[0.23,0.23], "yaw":[1.57,1.57]},
        update_goal_on_success=False, debug_vis=True)

@configclass
class ThreadingTerminationsCfg(BaseTerminationsCfg):
    max_consecutive_success = DoneTerm(func=threading.max_consecutive_success, params={"num_success": 20})
# BaseTerminationsCfg: time_out = DoneTerm(func=mdp.time_out, time_out=True)
```
```python
# Threading/mdps.py — success/termination logic (verbatim)
def success_bonus(env, object_id, frame_name, num_success=0):
    object = env.scene[f"object_{object_id}"]
    distance = torch.norm(env.scene[frame_name].data.target_pos_w.reshape(-1, 3) - object.data.root_pos_w[:, :3], dim=-1)
    success = distance < 0.03 * (object.data.root_pos_w[:, 2] > 0.2)
    env.success_tracker_step[success] += 1
    env.success_tracker_step[~success] = 0
    rew = env.success_tracker_step >= num_success
    return rew.float()

def max_consecutive_success(env, num_success):
    success = env.success_tracker_step >= num_success
    env.success_tracker = success.float()
    return success
```
`TargetPositionCommand._update_metrics` computes z-axis orientation_error (dot of z-axes), position_error (‖root_pos - pos_command_w‖), and consecutive_success (incremented when pos<thr AND orient>orient_thr).

---

## §5 Observation

**Description.** Single `PolicyCfg` group, concatenated, corruption **disabled** (`enable_corruption=False`, so the per-term Gnoise/Unoise are declared but NOT applied at runtime). Bimanual: full pose+joint state for both arms, then cube/drill pose, then last action. EE pose uses rotation-matrix flattening (`symmetry=True` → 3 pos + 9 rot flat = 12). Object quat likewise flattened to 9. Joint pos normalized to explicit per-arm limits; joint vel raw.

**Dim breakdown (total = 180):**

| Term | func | dim | note |
|---|---|---|---|
| ee_pose_right | ee_pose(palm_link) | 12 | 3 pos + 9 R_flat |
| joint_pos_right | joint_pos_limit_normalized(JOINT_*_LIMIT) | 22 | 6 arm + 16 hand; Gnoise std 0.005 (disabled) |
| joint_vel_right | joint_vel | 22 | |
| ee_pose_left | ee_pose(palm_link, robot_left) | 12 | |
| joint_pos_left | joint_pos_limit_normalized(*_LEFT) | 22 | Gnoise std 0.005 (disabled) |
| joint_vel_left | joint_vel(robot_left) | 22 | |
| cube_pos | object_pos(id 0) | 3 | Unoise 0–0.015 (disabled) |
| cube_quat | object_quat(id 0, symmetry) | 9 | R_flat; Gnoise 0.005 |
| drill_pos | object_pos(id 1) | 3 | Unoise 0–0.015 |
| drill_quat | object_quat(id 1, symmetry) | 9 | R_flat; Gnoise 0.005 |
| last_action | last_action | 44 | = action_dim |
| **total** | | **180** | |

**Decisions resolved.**

| Decision | Value |
|---|---|
| groups | single `policy`, concatenate_terms=True |
| corruption | `enable_corruption=False` (noise cfgs inert) |
| ee representation | pos + flattened rotation matrix (symmetry=True), 12 each |
| object representation | pos (3) + flattened R (9) per object |
| joint pos normalization | explicit JOINT_LOWER/UPPER_LIMIT (right) and *_LEFT (left) |
| obs dim | 180 |

**Code.**
```python
# env_cfg.py
@configclass
class ThreadingObservationsCfg(BaseObservationsCfg):
    @configclass
    class PolicyCfg(ObsGroup):
        ee_pose_right = ObsTerm(func=ee_pose, params={"ee_name": "palm_link"})
        joint_pos_right = ObsTerm(func=joint_pos_limit_normalized,
            params={"joints": None, "joint_lower_limit": JOINT_LOWER_LIMIT, "joint_upper_limit": JOINT_UPPER_LIMIT}, noise=Gnoise(std=0.005))
        joint_vel_right = ObsTerm(func=joint_vel, params={"joints": None})
        ee_pose_left = ObsTerm(func=ee_pose, params={"ee_name": "palm_link", "asset_cfg": SceneEntityCfg("robot_left")})
        joint_pos_left = ObsTerm(func=joint_pos_limit_normalized,
            params={"joints": None, "joint_lower_limit": JOINT_LOWER_LIMIT_LEFT, "joint_upper_limit": JOINT_UPPER_LIMIT_LEFT,
                    "asset_cfg": SceneEntityCfg("robot_left")}, noise=Gnoise(std=0.005))
        joint_vel_left = ObsTerm(func=joint_vel, params={"joints": None, "asset_cfg": SceneEntityCfg("robot_left")})
        cube_pos  = ObsTerm(func=object_pos,  params={"object_id": 0}, noise=Unoise(n_min=0.0, n_max=0.015))
        cube_quat = ObsTerm(func=object_quat, params={"object_id": 0, "symmetry": True}, noise=Gnoise(std=0.005))
        drill_pos  = ObsTerm(func=object_pos,  params={"object_id": 1}, noise=Unoise(n_min=0.0, n_max=0.015))
        drill_quat = ObsTerm(func=object_quat, params={"object_id": 1, "symmetry": True}, noise=Gnoise(std=0.005))
        last_action = ObsTerm(func=last_action)
        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True
    policy: PolicyCfg = PolicyCfg()
```

---

## §6 Reward

**Description.** 24 `RewTerm`s (weights listed below). The design is a full staged bimanual pipeline: (right hand) reach → lift → track cube goal pos+orient → success bonus; (left hand) align palm to drill approach frame (pos+quat) → reach drill → track drill goal pos+orient → success bonus; a coupling term `drill_cube_distance` rewarding drill-head↔cube proximity while both lifted; a terminal `success_bonus` (20-step consecutive threading); plus mirrored `_symmetry` variants for every reach/lift/track/align term, and energy + collision penalties per arm.

**All reward terms (name · func · key params · weight).**

| # | name | func | params | weight |
|---|---|---|---|---|
| 1 | reaching_object | object_robot_distance | weight[1,1,1,1.5], links if5/mf5/pf5/th5, obj 0 | 0.02 |
| 2 | object_lifting | lift_distance | cmd cube_target_pos, obj 0, sensors 0-3 | 2.0 |
| 3 | cube_goal_tracking | threading.object_goal_distance | cmd cube_target_pos, obj 0, sensors 0-3 | 10.0 |
| 4 | cube_goal_orient_tracking | object_goal_distance_orient | cmd cube_target_pos, obj 0, axis z, sensors 0-3, pos_thr 0.1 | 2.5 |
| 5 | cube_success_bonus | bowl.cmd_success_bonus | cmd cube_target_pos, num_success 1 | 0.0 |
| 6 | align_hand_to_pos | align_palm_to_pos | palm_link, frame object_approach_frame, robot_left | 1.0 |
| 7 | align_hand_to_quat | align_palm_to_quat | palm_link, frame object_approach_frame, robot_left | 0.25 |
| 8 | reaching_drill | object_robot_distance | weight[1,1,1,1.5], links if5/mf5/pf5/th5, obj 1, robot_left | 0.01 |
| 9 | drill_goal_tracking | threading.object_goal_distance | cmd drill_target_pos, obj 1 | 10.0 |
| 10 | drill_goal_orient_tracking | threading.drill_goal_orient_distance | cmd drill_target_pos, obj 1 | 0.4 |
| 11 | drill_success_bonus | bowl.cmd_success_bonus | cmd drill_target_pos, num_success 1 | 0.0 |
| 12 | drill_cube_distance | threading.drill_cube_distance | frame drill_head_frame, cube 0, drill 1 | 500.0 |
| 13 | success_bonus | threading.success_bonus | num_success 20, obj 0, frame drill_head_frame | 2000.0 |
| 14 | reaching_object_symmetry | object_robot_distance | obj 0, robot (mirror) | 0.02 |
| 15 | object_lifting_symmetry | lift_distance | cmd cube_target_pos, obj 0, sensors *_left | 2.0 |
| 16 | cube_goal_tracking_symmetry | threading.object_goal_distance | cmd cube_target_pos, obj 0, sensors *_left | 10.0 |
| 17 | cube_goal_orient_tracking_symmetry | object_goal_distance_orient | obj 0, axis z, sensors *_left, pos_thr 0.1 | 2.5 |
| 18 | align_hand_to_pos_symmetry | align_palm_to_pos | frame object_approach_frame_symmetry, robot | 1.0 |
| 19 | align_hand_to_quat_symmetry | align_palm_to_quat | frame object_approach_frame_symmetry, robot | 0.25 |
| 20 | reaching_drill_symmetry | object_robot_distance | obj 1, robot | 0.01 |
| 21 | energy | energy_punishment | robot, allegro actuators | 0.000001 |
| 22 | energy_left | energy_punishment | robot_left, allegro actuators | 0.000001 |
| 23 | collision_to_table | collision_penalty | sensors 0-3 | -0.000001 |
| 24 | collision_to_table_symmetry | collision_penalty | sensors *_left | -0.000001 |

Note: `cube_success_bonus` and `drill_success_bonus` carry `weight=0.0` (declared but inactive).

### Symmetric-learning reward terms (drop if not using symmetric learning)

This task trains with **symmetric learning** (`base.yaml` → `symmetry.symmetric_envs: True`, C2 group). The **BASE** terms — including the two-arm drill/cube coupling terms (`drill_cube_distance`, `success_bonus`, `drill_goal_tracking`, `drill_goal_orient_tracking`, `align_hand_to_*`, etc.) — are **genuine** and required. The `_symmetry`-suffixed terms are **DUPLICATES** of their base counterparts (identical weights) added purely for symmetric-learning augmentation — **drop them if you are not using symmetric learning.**

`_symmetry` duplicate terms: `reaching_object_symmetry`, `object_lifting_symmetry`, `cube_goal_tracking_symmetry`, `cube_goal_orient_tracking_symmetry`, `align_hand_to_pos_symmetry`, `align_hand_to_quat_symmetry`, `reaching_drill_symmetry`, `collision_to_table_symmetry`.

**Code (reward config — abridged params repeated above).**
```python
# env_cfg.py
@configclass
class ThreadingRewardsCfg(BaseRewardsCfg):
    reaching_object = RewTerm(func=object_robot_distance,
        params={"weight":[1.0,1.0,1.0,1.5], "link_name":["if5","mf5","pf5","th5"], "object_id":0}, weight=0.02)
    object_lifting = RewTerm(func=lift_distance,
        params={"command_name":"cube_target_pos","object_id":0,"sensor_names":["contact_sensors_0","contact_sensors_1","contact_sensors_2","contact_sensors_3"]}, weight=2.0)
    cube_goal_tracking = RewTerm(func=threading.object_goal_distance,
        params={"command_name":"cube_target_pos","object_id":0,"sensor_names":["contact_sensors_0","contact_sensors_1","contact_sensors_2","contact_sensors_3"]}, weight=10.0)
    cube_goal_orient_tracking = RewTerm(func=object_goal_distance_orient,
        params={"command_name":"cube_target_pos","object_id":0,"axis":"z","sensor_names":[...0-3...],"pos_success_threshold":0.1}, weight=2.5)
    cube_success_bonus = RewTerm(func=bowl.cmd_success_bonus, params={"command_names":"cube_target_pos","num_success":1}, weight=0.0)
    align_hand_to_pos = RewTerm(func=align_palm_to_pos,
        params={"link_name":["palm_link"],"frame_name":"object_approach_frame","asset_cfg":SceneEntityCfg("robot_left")}, weight=1.0)
    align_hand_to_quat = RewTerm(func=align_palm_to_quat,
        params={"link_name":["palm_link"],"frame_name":"object_approach_frame","asset_cfg":SceneEntityCfg("robot_left")}, weight=0.25)
    reaching_drill = RewTerm(func=object_robot_distance,
        params={"weight":[1.0,1.0,1.0,1.5],"link_name":["if5","mf5","pf5","th5"],"object_id":1,"asset_cfg":SceneEntityCfg("robot_left")}, weight=0.01)
    drill_goal_tracking = RewTerm(func=threading.object_goal_distance, params={"command_name":"drill_target_pos","object_id":1}, weight=10.0)
    drill_goal_orient_tracking = RewTerm(func=threading.drill_goal_orient_distance, params={"command_name":"drill_target_pos","object_id":1}, weight=0.4)
    drill_success_bonus = RewTerm(func=bowl.cmd_success_bonus, params={"command_names":"drill_target_pos","num_success":1}, weight=0.0)
    drill_cube_distance = RewTerm(func=threading.drill_cube_distance, params={"frame_name":"drill_head_frame","cube_id":0,"drill_id":1}, weight=500.0)
    success_bonus = RewTerm(func=threading.success_bonus, params={"num_success":20,"object_id":0,"frame_name":"drill_head_frame"}, weight=2000.0)
    # symmetry mirrors (14-20): same weights as their base counterparts (reaching_object_symmetry 0.02,
    #   object_lifting_symmetry 2.0, cube_goal_tracking_symmetry 10.0, cube_goal_orient_tracking_symmetry 2.5,
    #   align_hand_to_pos_symmetry 1.0, align_hand_to_quat_symmetry 0.25, reaching_drill_symmetry 0.01)
    energy = RewTerm(func=energy_punishment, weight=0.000001,
        params={"asset_cfg":SceneEntityCfg("robot"),"actuator_name":["allegro_hand_1",...,"allegro_hand_thumb_4"]})
    energy_left = RewTerm(func=energy_punishment, weight=0.000001, params={"asset_cfg":SceneEntityCfg("robot_left"),"actuator_name":[...]})
    collision_to_table = RewTerm(func=collision_penalty, params={"sensor_names":[...0-3...]}, weight=-0.000001)
    collision_to_table_symmetry = RewTerm(func=collision_penalty, params={"sensor_names":[...*_left...]}, weight=-0.000001)
```

**Code (reward function sources — verbatim).**
```python
# reward_mdps.py
def object_robot_distance(env, weight, link_name, asset_cfg=SceneEntityCfg("robot"), object_id=0):
    weight = torch.tensor(weight, device=env.device)
    object = env.scene[f"object_{object_id}"]
    object_pos_w = object.data.root_pos_w[:, None, :]
    robot = env.scene[asset_cfg.name]
    link_idx = robot.find_bodies(link_name)[0]
    link_w = robot.data.body_pos_w[:, link_idx, :3]
    object_link_distance = torch.norm(object_pos_w - link_w, dim=-1) * weight
    object_link_distance = torch.mean(object_link_distance, dim=1)
    return 1 / object_link_distance

def get_allegro_contact(env, sensor_names):
    force = get_force(env, sensor_names, if_filter=True)          # force_matrix_w (filtered) per sensor
    is_contact = (torch.norm(force, dim=-1) > 1.0)
    is_contact_index_or_middle_or_ring = reduce(torch.logical_or, [is_contact[:,0], is_contact[:,1], is_contact[:,2]])
    is_contact = reduce(torch.logical_and, [is_contact_index_or_middle_or_ring, is_contact[:,3]])  # any of if/mf/pf AND thumb
    return is_contact

def lift_distance(env, command_name, minimal_height=None, object_id=0, sensor_names=[...0-3...]):
    command = env.command_manager.get_command(command_name)
    des_pos_w = command[:, :3] + env.scene.env_origins
    if minimal_height is None: minimal_height = des_pos_w[:, 2] + 0.05
    object = env.scene[f"object_{object_id}"]
    z_distance = (object.data.root_pos_w[:,2] - env.object_init_pos[object_id][:,2]) / (minimal_height - env.object_init_pos[object_id][:,2])
    z_distance = torch.clamp(z_distance, min=0.0)
    return (object.data.root_pos_w[:,2] < minimal_height) * z_distance * get_allegro_contact(env, sensor_names)

def object_goal_distance_orient(env, command_name, object_id=0, axis="z", pos_success_threshold=None, sensor_names=[...]):
    object = env.scene[f"object_{object_id}"]
    command = env.command_manager.get_command(command_name)
    des_orient_w = command[:, 3:]
    if axis is not None:
        from bimanual_suite.utils.isaac_utils import get_angle_from_quat
        target_axis = get_angle_from_quat(des_orient_w, axis=axis, normalize=True)
        cur_axis = get_angle_from_quat(object.data.root_quat_w, axis=axis, normalize=True)
        distance = torch.clamp(torch.sum(target_axis*cur_axis, dim=-1) * (2**0.5), min=0.0)
    else:
        distance = math_utils.quat_error_magnitude(object.data.root_quat_w, des_orient_w)
        max_distance = math_utils.quat_error_magnitude(env.object_init_orient[object_id], des_orient_w)
        distance = torch.clamp(max_distance - distance, min=0.0)
    within_range = torch.ones(env.num_envs, device=env.device)
    command_term = env.command_manager.get_term(command_name)
    if pos_success_threshold is not None:
        within_range = torch.logical_and(within_range, command_term.metrics["position_error"] < pos_success_threshold)
    else:
        within_range = torch.logical_and(within_range, command_term.metrics["position_error"] < command_term.cfg.success_threshold)
    return distance * within_range.float() * get_allegro_contact(env, sensor_names)

def align_palm_to_quat(env, link_name, asset_cfg=SceneEntityCfg("robot"), frame_name="approach_frame"):
    robot = env.scene[asset_cfg.name]
    robot_state_w = robot.data.body_state_w[:, robot.find_bodies(link_name)[0], :7].reshape(-1, 7)
    return -math_utils.quat_error_magnitude(robot_state_w[:,3:7], env.scene[frame_name].data.target_quat_w.reshape(-1,4))

def align_palm_to_pos(env, link_name, asset_cfg=SceneEntityCfg("robot"), frame_name="approach_frame"):
    robot = env.scene[asset_cfg.name]
    robot_state_w = robot.data.body_state_w[:, robot.find_bodies(link_name)[0], :7].reshape(-1, 7)
    return -torch.norm(robot_state_w[:,:3] - env.scene[frame_name].data.target_pos_w.reshape(-1,3), dim=-1)

def energy_punishment(env, actuator_name=None, asset_cfg=SceneEntityCfg("robot")):
    energy = get_actuator_energy_consumption(env, asset_cfg.name, actuator_name)  # Σ|jnt_vel * applied_torque|
    return -energy

def collision_penalty(env, sensor_names=[...0-3...]):
    filtered_is_contact = []
    for s in sensor_names:
        filtered_force = env.scene[s].data.force_matrix_w.mean(dim=...) == 0.0     # NOT touching target object
        normal_force   = env.scene[s].data.net_forces_w.mean(dim=...) != 0.0        # but touching something
        filtered_is_contact.append(torch.logical_and(filtered_force, normal_force))
    return reduce(torch.logical_or, filtered_is_contact).float()
```
```python
# Threading/mdps.py — task-specific reward funcs (verbatim)
def object_goal_distance(env, command_name, object_id=0, sensor_names=[...0-3...]):
    object = env.scene[f"object_{object_id}"]
    command = env.command_manager.get_command(command_name)
    command_term = env.command_manager.get_term(command_name)
    des_pos_w = command[:, :3] + env.scene.env_origins
    distance = torch.norm(des_pos_w - object.data.root_pos_w[:, :3], dim=1)
    max_distance = torch.norm(des_pos_w - env.object_init_pos[object_id], dim=1)
    distance = torch.clamp(max_distance - distance, min=0.0)
    if object_id == 0:
        rew = distance * get_allegro_contact(env, sensor_names) * (object.data.root_pos_w[:,2] > (des_pos_w[:,2]-0.1))
    else:
        rew = distance * (command_term.metrics["orientation_error"] > 0.75)
    return rew

def drill_goal_orient_distance(env, command_name, object_id=0):
    object = env.scene[f"object_{object_id}"]
    command_term = env.command_manager.get_term(command_name)
    des_orient_w = command_term.quat_command_w
    distance = math_utils.quat_error_magnitude(object.data.root_quat_w, des_orient_w)
    return 1 / distance * (object.data.root_pos_w[:,2] > command_term.command[:,2] - 0.1)

def drill_cube_distance(env, frame_name, cube_id=0, drill_id=1):
    cube = env.scene[f"object_{cube_id}"]; drill = env.scene[f"object_{drill_id}"]
    distance = torch.norm(env.scene[frame_name].data.target_pos_w.reshape(-1,3) - cube.data.root_pos_w[:,:3], dim=-1)
    return torch.clamp(0.1 - distance, min=0.0) * (cube.data.root_pos_w[:,2] > 0.25) * (drill.data.root_pos_w[:,2] > 0.2)

# success_bonus / max_consecutive_success — see §4
```
```python
# StirBowl/mdps.py — cmd_success_bonus (imported as bowl.*)
def cmd_success_bonus(env, command_names, num_success=0):
    if isinstance(command_names, str):
        success = env.command_manager.get_term(command_names).metrics["consecutive_success"] >= num_success
    else:
        success = torch.ones(env.num_envs, device=env.device)
        for c in command_names:
            success = torch.logical_and(success, env.command_manager.get_term(c).metrics["consecutive_success"] >= num_success)
    return success.float()
```

---

## §7 DR

**`<no DR>` in the task's own EventCfg.** `ThreadingEventCfg` wires only `mode="reset"` terms (robot-joint + object placement, §3) — there are no randomization/`interval`/`startup` DR EventTerms and no observation-noise applied at runtime (`enable_corruption=False`).

Domain-randomization **infrastructure does exist** at the `BaseEnv` level but is entirely driven by an external hydra config (`env.cfg.hydra_cfg.task.randomize`) via `bimanual_suite.utils.domain_random.DomainRandomizer`, not by this task's config. `BaseEnv.update_randomization(success_rate)` can, when the hydra `randomize` block enables them, curriculum-randomize: `object_mass`, `static/dynamic_friction`+`restitution` (material), `action_scale[:6]` (arm), per-term `energy_penalty`/`collision_penalty` reward weights, `external_force_torque`, and `reset_pose` ranges (via `randomize_mass` / `randomize_material` / `randomize_rew_weight` / `randomize_external_force_torque` / `randomize_reset_pose` in `randomization_mdps.py`). Since none of this is instantiated in `ThreadingEnvCfg`/`ThreadingEventCfg`, the task ships with **DR: no**.
