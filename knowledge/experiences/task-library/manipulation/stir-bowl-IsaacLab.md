# StirBowl — Implementation Spec

- robot: Bimanual UF850 arms + dual Allegro hands (44 DoF)
- simulator: IsaacLab (Isaac Sim, manager-based)
- objects: egg-beater tool, bowl, five rigid objects (object_0..4), table
- bimanual: true
- summary: One hand holds an egg-beater and stirs the contents of a bowl.

> Source package name anonymized as `bimanual_suite`. This task comes from an internal
> bimanual manipulation suite rather than a public repo; the design below is otherwise verbatim.

**Task summary:** A bimanual dexterous stirring task. Two UF850 arms, each ending in a 16-DoF Allegro hand (`robot` = right, `robot_left` = left), sit behind a table. The **right** hand grasps an **egg-beater** tool (`object_0`) and lifts/orients it to a goal pose above the table; the **left** hand grasps and holds a **bowl** (`object_1`) steady at its goal position. Three **balls** (`object_2..4`) rest inside the bowl. The intended behavior is: keep the bowl stationary at its goal, hold the egg-beater upright at its goal pose while in fingertip contact, and *stir* — driving ball motion inside the bowl (rewarded via ball linear velocity, gated on egg-beater success AND bowl-held success). Success is the conjunction of bowl-at-goal, egg-beater-at-goal-position, and egg-beater-orientation-aligned. The action is a per-arm EMA cumulative-relative joint-position target (44-D total).

---

## §1 Registration + Scene

**Description.** Registered as `StirBowlEnv-v0` with custom env class `StirBowlEnv(BaseEnv)`. `BaseEnv` is bimanual_suite's shared `ManagerBasedRLEnv` subclass (scales raw actions by `cfg.action_scale`, tracks per-object init pose/orient, runs C2-symmetry env mirroring, wires a hydra-driven `DomainRandomizer`). Scene = two arm+hand articulations (gravity disabled on both), 5 rigid objects (egg-beater tool, bowl, 3 balls), a kinematic table, ground @ z=-0.82, a dome light, 8 fingertip↔`Object_0` contact sensors (4 per hand), and two `FrameTransformer` approach frames on the bowl (nominal + symmetry). num_envs=4096, env_spacing=3.0. sim dt=1/120, decimation=6 (→ 20 Hz control), episode_length_s=8.3333. Asset USDs are under `{bimanual_suite.LIB_PATH}/assets/…` where `LIB_PATH` = repo root (`Path(bimanual_suite/__init__.py).parent.parent`).

**Decisions resolved.**

| Decision | Value | Source |
|---|---|---|
| num_envs / env_spacing | 4096 / 3.0 | `StirBowlSceneCfg(num_envs=4096, env_spacing=3.0)` |
| sim.dt / decimation / render_interval | 1/120 / 6 / 6 | `BaseEnvCfg.__post_init__` |
| episode_length_s | 8.3333 (≈166 control steps @ 20 Hz) | `BaseEnvCfg.__post_init__` |
| Control rate | 120/6 = 20 Hz | dt·decimation |
| Right robot USD | `assets/ufactory850/uf850_allegro_right_colored.usd` | scene |
| Left robot USD | `assets/ufactory850/uf850_allegro_left_colored.usd` | scene |
| Right robot base pos | (-0.274, -0.475, 0.01) | scene |
| Left robot base pos | (-0.274, +0.475, 0.01) | scene |
| Robot gravity | disabled (both) | `disable_gravity=True` |
| object_0 (egg-beater) USD / scale / mass | `assets/object/egg_beater.usd` / (0.001,0.001,0.001) / 0.25 kg | scene |
| object_1 (bowl) USD / scale / mass | `assets/object/bowl.usd` / (0.035,0.04,0.035) / 1.0 kg | scene |
| object_2..4 (balls) | procedural `SphereCfg` r=0.03 / mass 0.1 kg each / blue material | scene |
| Objects dynamic? | yes — `kinematic_enabled=False`, gravity on | scene |
| Table | `assets/object/table.usd`, `kinematic_enabled=True`, rot=(0.70710678,0,0,0.70710678) | `BaseSceneCfg` |
| Ground | GroundPlane @ pos=(0,0,-0.82) | `BaseSceneCfg` |
| Light | DomeLight color=(0.75,0.75,0.75) intensity=2500 | `BaseSceneCfg` |
| Contact sensors | 4 right (if5/mf5/pf5/th5) + 4 left, all filtered to `Object_0` | scene |
| Approach frames | 2 FrameTransformers on `Object_1` (offset ±0.18 in y, mirrored rot) | scene |
| num_object (tracked) | 2 (only egg-beater + bowl get init-pose buffers) | `StirBowlEnvCfg.num_object` |
| K rigid objects in scene | 5 (`object_0..4`) | scene |
| replicate_physics | False | `BaseSceneCfg` |
| PhysX | gpu_max_rigid_contact_count=2²⁴, gpu_max_rigid_patch_count=2²⁴ | `BaseEnvCfg.sim` |
| Physics material | static_friction=1.5, dynamic_friction=1.0, restitution=0.0 | `BaseEnvCfg.sim` |
| Solver iters (robots+objects) | pos=16, vel=1 | scene |
| Viewer eye | (-3.5, 0.0, 3.5) | `StirBowlEnvCfg.__post_init__` |
| visualize_marker | False | `StirBowlEnvCfg` |

**Code (registration).**
```python
# bimanual_suite/env/__init__.py
from .tasks.StirBowl.env_cfg import StirBowlEnvCfg
gym.register(
    id="StirBowlEnv-v0",
    entry_point="bimanual_suite.env.tasks.StirBowl.env:StirBowlEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": StirBowlEnvCfg,
    },
)
```

**Code (custom env class — StirBowl/env.py).** No custom success/step logic beyond the base; `step()` only adds optional debug marker visualization. All success bookkeeping lives in the reward/command terms (see §4, §6).
```python
class StirBowlEnv(BaseEnv):
    is_vector_env: ClassVar[bool] = True
    metadata: ClassVar[dict[str, Any]] = {
        "render_modes": [None, "human", "rgb_array"],
        "isaac_sim_version": get_version(),
    }
    cfg: StirBowlEnvCfg

    def step(self, action: torch.Tensor) -> VecEnvStepReturn:
        super().step(action)
        # Debug only
        if self.cfg.visualize_marker:
            left_palm_idx = self.scene["robot_left"].find_bodies("palm_link")[0]
            left_palm = self.scene["robot_left"].data.body_state_w[:, left_palm_idx, :7].reshape(-1, 7)
            self.markers['arm_r']['ee_marker'].visualize(left_palm[:, :3], left_palm[:, 3:7])
            self.markers['arm_l']['goal_marker'].visualize(self.scene["object_1"].data.root_state_w[:, :3], self.scene["object_1"].data.root_state_w[:, 3:7])
        return self.obs_buf, self.reward_buf, self.reset_terminated, self.reset_time_outs, self.extras
```

**Code (BaseEnv.step — where action scaling + success/extras happen; manager_based_env.py).**
```python
    def step(self, action: torch.Tensor) -> VecEnvStepReturn:
        # update last action inferred from the policy.
        self.last_action = action.clone()
        # scale the action
        action = action * self._scale
        super().step(action)
        self.extras['success'] = self.success_tracker
        self.extras['detailed_reward'] = self.detailed_reward_buf
        # post reset process
        reset_indices = torch.where(self.episode_length_buf == 1)[0]
        if len(reset_indices) > 0:
            self._post_reset_process(reset_indices)
        return self.obs_buf, self.reward_buf, self.reset_terminated, self.reset_time_outs, self.extras
```

**Code (scene — StirBowl/env_cfg.py, StirBowlSceneCfg).** Both robots share identical actuator gains and hand joint pose; they differ only in base y-sign, `joint1`/`joint6` sign, and USD.
```python
@configclass
class StirBowlSceneCfg(BaseSceneCfg):
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
                "joint1": 0.5, "joint2": 0.3, "joint3": -0.6, "joint4": 0.0, "joint5": -0.8, "joint6": -1.57,
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
                "joint1": -0.5, "joint2": 0.3, "joint3": -0.6, "joint4": 0.0, "joint5": -0.8, "joint6": 1.57,
                "jif1": 0.0, "jif2": 0.4, "jif3": 0.4, "jif4": 0.0,
                "jmf1": 0.0, "jmf2": 0.4, "jmf3": 0.4, "jmf4": 0.0,
                "jpf1": 0.0, "jpf2": 0.4, "jpf3": 0.4, "jpf4": 0.0,
                "jth1": 1.3, "jth2": 0.0, "jth3": 0.2, "jth4": 0.0,
            },
            pos=(-0.274, 0.475, 0.01),
        ),
        actuators={  # identical gains to `robot`
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

    object_0 = RigidObjectCfg(  # EGG-BEATER (stirring tool), grasped by RIGHT hand
        prim_path=f"/World/envs/env_.*/Object_0",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{bimanual_suite.LIB_PATH}/assets/object/egg_beater.usd",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=False, disable_gravity=False, max_linear_velocity=1000, max_angular_velocity=1000,
                solver_position_iteration_count=16, solver_velocity_iteration_count=1, max_depenetration_velocity=1000.0,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.25),
            activate_contact_sensors=True,
            scale=(0.001, 0.001, 0.001),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(lin_vel=(0.0,0.0,0.0), ang_vel=(0.0,0.0,0.0), pos=(0.0,0.0,0.0)),
    )

    object_1 = RigidObjectCfg(  # BOWL, held by LEFT hand
        prim_path=f"/World/envs/env_.*/Object_1",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{bimanual_suite.LIB_PATH}/assets/object/bowl.usd",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=False, disable_gravity=False, max_linear_velocity=1000, max_angular_velocity=1000,
                solver_position_iteration_count=16, solver_velocity_iteration_count=1, max_depenetration_velocity=1000.0,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
            activate_contact_sensors=True,
            scale=(0.035, 0.04, 0.035),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(lin_vel=(0.0,0.0,0.0), ang_vel=(0.0,0.0,0.0), pos=(0.0,0.0,0.0)),
    )

    # object_2, object_3, object_4 = three procedural balls inside the bowl (identical cfgs)
    object_2 = RigidObjectCfg(
        prim_path=f"/World/envs/env_.*/Object_2",
        spawn=sim_utils.SphereCfg(
            radius=0.03,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=False, disable_gravity=False, max_linear_velocity=1000, max_angular_velocity=1000,
                solver_position_iteration_count=16, solver_velocity_iteration_count=1, max_depenetration_velocity=1000.0,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.1),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0,0.0,1.0), metallic=0.2),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(lin_vel=(0.0,0.0,0.0), ang_vel=(0.0,0.0,0.0), pos=(0.0,0.0,0.0)),
    )
    # object_3, object_4: identical to object_2 (radius 0.03, mass 0.1, blue) — see env_cfg.py

    # sensors: 4 RIGHT-hand fingertip contact sensors filtered to Object_0 (egg-beater)
    contact_sensors_0 = ContactSensorCfg(prim_path="/World/envs/env_.*/Robot/if5",  # index
        update_period=0.0, debug_vis=True, filter_prim_paths_expr=["{ENV_REGEX_NS}/Object_0"])
    contact_sensors_1 = ContactSensorCfg(prim_path="/World/envs/env_.*/Robot/mf5",  # middle
        update_period=0.0, debug_vis=True, filter_prim_paths_expr=["{ENV_REGEX_NS}/Object_0"])
    contact_sensors_2 = ContactSensorCfg(prim_path="/World/envs/env_.*/Robot/pf5",  # pinky
        update_period=0.0, debug_vis=True, filter_prim_paths_expr=["{ENV_REGEX_NS}/Object_0"])
    contact_sensors_3 = ContactSensorCfg(prim_path="/World/envs/env_.*/Robot/th5",  # thumb
        update_period=0.0, debug_vis=True, filter_prim_paths_expr=["{ENV_REGEX_NS}/Object_0"])
    # 4 LEFT-hand fingertip contact sensors — NOTE: also filtered to Object_0 (egg-beater), NOT the bowl
    contact_sensors_0_left = ContactSensorCfg(prim_path="/World/envs/env_.*/Robot_left/if5",
        update_period=0.0, debug_vis=True, filter_prim_paths_expr=["{ENV_REGEX_NS}/Object_0"])
    contact_sensors_1_left = ContactSensorCfg(prim_path="/World/envs/env_.*/Robot_left/mf5",
        update_period=0.0, debug_vis=True, filter_prim_paths_expr=["{ENV_REGEX_NS}/Object_0"])
    contact_sensors_2_left = ContactSensorCfg(prim_path="/World/envs/env_.*/Robot_left/pf5",
        update_period=0.0, debug_vis=True, filter_prim_paths_expr=["{ENV_REGEX_NS}/Object_0"])
    contact_sensors_3_left = ContactSensorCfg(prim_path="/World/envs/env_.*/Robot_left/th5",
        update_period=0.0, debug_vis=True, filter_prim_paths_expr=["{ENV_REGEX_NS}/Object_0"])

    # approach frames on the bowl (Object_1) — nominal + mirrored (used by align_palm rewards)
    object_approach_frame = FrameTransformerCfg(
        prim_path="{ENV_REGEX_NS}/Object_1", debug_vis=True,
        visualizer_cfg=FRAME_MARKER_SMALL_CFG.replace(prim_path="/Visuals/ObjectApproachFrameTransformer"),
        target_frames=[FrameTransformerCfg.FrameCfg(
            prim_path="{ENV_REGEX_NS}/Object_1", name="approach_frame",
            offset=OffsetCfg(pos=(0.0, 0.18, 0.0), rot=(0.5, -0.5, 0.5, -0.5)))],
    )
    object_approach_frame_symmetry = FrameTransformerCfg(
        prim_path="{ENV_REGEX_NS}/Object_1", debug_vis=True,
        visualizer_cfg=FRAME_MARKER_SMALL_CFG.replace(prim_path="/Visuals/ObjectApproachFrameTransformerSymmetry"),
        target_frames=[FrameTransformerCfg.FrameCfg(
            prim_path="{ENV_REGEX_NS}/Object_1", name="approach_frame",
            offset=OffsetCfg(pos=(0.0, -0.18, 0.0), rot=(0.5, 0.5, 0.5, 0.5)))],
    )
```

**Code (shared table/ground/light — manager_based_env_cfg.py, BaseSceneCfg).**
```python
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
```

---

## §2 Actions

**Description.** One `EMACumulativeRelativeJointPositionActionCfg` per arm (`robot`, `robot_left`), each covering all 22 joints (`joint_names=[".*"]` → 6 arm + 16 hand). Total **action_dim = 44**. The action term is a custom subclass of IsaacLab `JointPositionAction`: each step the raw (scaled) action is a *delta* accumulated into a running cumulative target (`del_action`), added to the reset-time joint position (`init_joint_pos`), then EMA-filtered (`alpha=0.2`) against the previous applied target, then clamped to per-joint `[JOINT_LOWER_LIMIT, JOINT_UPPER_LIMIT]`. Note two-stage scaling: `BaseEnv.step` first multiplies the policy action by `cfg.action_scale` (per-joint 0.05/0.03/0.015 rates, 44-vector), THEN the action term applies its own `scale=1.0` and affine transform. So the effective per-step joint delta = `policy_action · action_scale`.

**Decisions resolved.**

| Decision | Value | Source |
|---|---|---|
| action_dim | 44 (22 right + 22 left) | `StirBowlEnvCfg.action_dim` |
| Per-arm joints | `[".*"]` → 6 arm (`joint1..6`) + 16 hand | actions cfg |
| Action term class | `EMACumulativeRelativeJointPositionAction` | actions.py |
| scale (term-level) | 1.0 | actions cfg |
| use_default_offset | False | actions cfg |
| alpha (EMA) | 0.2 (both arms) | actions cfg |
| Per-joint limits | `JOINT_LOWER/UPPER_LIMIT` (right), `..._LEFT` (left) | manager_based_env_cfg.py |
| action_scale (env pre-mult) | 44-vec: arm 0.05×6, hand 0.03 (jth3 → 0.015) | `StirBowlEnvCfg.action_scale` |
| Per-step rule | `target ← clamp( α·(init + Σδ) + (1-α)·prev, lo, hi )`, δ = policy·action_scale | actions.py `process_actions` |

**Code (StirBowlActionsCfg — env_cfg.py).**
```python
@configclass
class StirBowlActionsCfg:
    arm_hand_action = EMACumulativeRelativeJointPositionActionCfg(
        asset_name="robot", joint_names=[".*"], scale=1.0, use_default_offset=False,
        joint_lower_limit=JOINT_LOWER_LIMIT, joint_upper_limit=JOINT_UPPER_LIMIT, alpha=0.2)
    arm_hand_action_left = EMACumulativeRelativeJointPositionActionCfg(
        asset_name="robot_left", joint_names=[".*"], scale=1.0, use_default_offset=False,
        joint_lower_limit=JOINT_LOWER_LIMIT_LEFT, joint_upper_limit=JOINT_UPPER_LIMIT_LEFT, alpha=0.2)
```

**Code (action_scale — env_cfg.py).**
```python
    action_dim = 44 # arm + hand
    action_scale: list = [0.05, 0.05, 0.05, 0.05, 0.05, 0.05,
                            0.03, 0.03, 0.03, 0.03,
                            0.03, 0.03, 0.03, 0.03,
                            0.03, 0.03, 0.03, 0.015,
                            0.03, 0.03, 0.03, 0.03,
                            0.05, 0.05, 0.05, 0.05, 0.05, 0.05,
                            0.03, 0.03, 0.03, 0.03,
                            0.03, 0.03, 0.03, 0.03,
                            0.03, 0.03, 0.03, 0.015,
                            0.03, 0.03, 0.03, 0.03]  # jth3 needs smaller rate
```

**Code (per-step rule — actions.py `process_actions`).**
```python
    def process_actions(self, actions: torch.Tensor):
        super().process_actions(actions)              # affine (scale=1, no offset)
        self._processed_actions += self.del_action    # accumulate delta
        self.del_action = self._processed_actions.clone()
        self._processed_actions += self.init_joint_pos.clone()   # add reset-time pose
        ema_actions = self._alpha * self._processed_actions
        ema_actions += (1.0 - self._alpha) * self._prev_applied_actions
        if self.joint_lower_limit is not None and self.joint_upper_limit is not None:
            self._processed_actions[:] = torch.clamp(ema_actions, self.joint_lower_limit, self.joint_upper_limit)
        else:
            self._processed_actions[:] = ema_actions
        self._prev_applied_actions[:] = self._processed_actions[:]
```
Per-joint limit vectors (`JOINT_LOWER_LIMIT` etc.) — arm order `joint1..6` then hand `j*f1, j*f2, j*f3, j*f4` (index/middle/pinky/thumb interleaved as f1/f2/f3/f4 groups). Right vs left differ only in the `jif1/jpf1` (finger-spread) sign block:
```python
JOINT_LOWER_LIMIT = [-6.283, -2.304, -4.224, -6.283, -2.164, -6.283,
                    -0.05, -0.05, -0.570, 0.364,   -0.296,-0.296,-0.296,-0.205,
                    -0.274,-0.274,-0.274,-0.290,   -0.327,-0.327,-0.327,-0.262]
JOINT_UPPER_LIMIT = [6.283, 2.304, 0.061, 6.283, 2.164, 6.283,
                    0.570, 0.05, 0.05, 1.497,      1.710,1.710,1.710,1.130,
                    1.809,1.809,1.809,1.633,       1.718,1.718,1.718,1.820]
# LEFT: jif1/jpf1 spread limits mirrored (lower jif1=-0.570, jpf1=-0.05; upper jif1=0.05, jpf1=0.570)
```

---

## §3 Reset / Events

**Description.** All events are `mode="reset"` (once per episode per env). The base `reset_robot_joints` (right robot, `reset_joints_by_symmetry`) is inherited from `BaseEventCfg`; the task adds a left-robot joint reset and five object resets (egg-beater, bowl, 3 balls). Object resets sample a pose uniformly from tight per-axis ranges and write pose+velocity to sim. The egg-beater is reset with roll=1.57 (laid on its side, in front of the right arm at y≈-0.32); the bowl and balls are clustered at y≈+0.2 in front of the left arm. Robot-joint reset scales default pose by (1.0,1.0) (i.e., exact default) with C2-symmetry mirroring handled in `reset_joints_by_symmetry`.

**Decisions resolved.**

| Event | func / mode | Range | object_id |
|---|---|---|---|
| reset_robot_joints (inherited) | `reset_joints_by_symmetry` / reset | pos scale (1.0,1.0), vel (0,0), asset `robot` | — |
| reset_robot_joints_left | `mdp.reset_joints_by_scale` / reset | pos scale (1.0,1.0), vel (0,0), asset `robot_left` | — |
| reset_object_egg_beater | `reset_object` / reset | x[0.15,0.05], y[-0.3,-0.35], z[0.01], roll[1.57,1.57] | 0 |
| reset_object_bowl | `reset_object` / reset | x[0.1,0.1], y[0.2,0.2], z[0.06,0.06] | 1 |
| reset_ball_1 | `reset_object` / reset | x[0.1], y[0.2], z[0.05] | 2 |
| reset_ball_2 | `reset_object` / reset | x[0.14], y[0.22], z[0.05] | 3 |
| reset_ball_3 | `reset_object` / reset | x[0.06], y[0.18], z[0.05] | 4 |

Note: `reset_object` treats each range as `[min,max]`; egg-beater x is given as `[0.15,0.05]` (descending) — `sample_uniform` still samples between them. Positions are relative to `default_root_state` (which is (0,0,0)+env_origins), so these are env-frame offsets.

**Code (StirBowlEventCfg — env_cfg.py).**
```python
@configclass
class StirBowlEventCfg(BaseEventCfg):
    reset_robot_joints_left = EventTerm(
        func=mdp.reset_joints_by_scale, mode="reset",
        params={"position_range": (1.0, 1.0), "velocity_range": (0.0, 0.0),
                "asset_cfg": SceneEntityCfg("robot_left")})
    reset_object_egg_beater = EventTerm(func=reset_object, mode="reset",
        params={"pose_range": {"x": [0.15, 0.05], "y": [-0.3, -0.35], "z": [0.01, 0.01], "roll": [1.57, 1.57]},
                "velocity_range": {}, "object_id": 0})
    reset_object_bowl = EventTerm(func=reset_object, mode="reset",
        params={"pose_range": {"x": [0.1, 0.1], "y": [0.2, 0.2], "z": [0.06, 0.06]}, "velocity_range": {}, "object_id": 1})
    reset_ball_1 = EventTerm(func=reset_object, mode="reset",
        params={"pose_range": {"x": [0.1, 0.1], "y": [0.2, 0.2], "z": [0.05, 0.05]}, "velocity_range": {}, "object_id": 2})
    reset_ball_2 = EventTerm(func=reset_object, mode="reset",
        params={"pose_range": {"x": [0.14, 0.14], "y": [0.22, 0.22], "z": [0.05, 0.05]}, "velocity_range": {}, "object_id": 3})
    reset_ball_3 = EventTerm(func=reset_object, mode="reset",
        params={"pose_range": {"x": [0.06, 0.06], "y": [0.18, 0.18], "z": [0.05, 0.05]}, "velocity_range": {}, "object_id": 4})
```

**Code (reset_object — reset_mdps.py).**
```python
def reset_object(env, env_ids, pose_range, velocity_range, object_id):
    object = env.scene[f"object_{object_id}"]
    root_states = object.data.default_root_state[env_ids].clone()
    range_list = [pose_range.get(key, (0.0, 0.0)) for key in ["x","y","z","roll","pitch","yaw"]]
    ranges = torch.tensor(range_list, device=object.device)
    rand_samples = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=object.device)
    positions = root_states[:, 0:3] + env.scene.env_origins[env_ids] + rand_samples[:, 0:3]
    orientations_delta = math_utils.quat_from_euler_xyz(rand_samples[:, 3], rand_samples[:, 4], rand_samples[:, 5])
    orientations = math_utils.quat_mul(root_states[:, 3:7], orientations_delta)
    range_list = [velocity_range.get(key, (0.0, 0.0)) for key in ["x","y","z","roll","pitch","yaw"]]
    ranges = torch.tensor(range_list, device=object.device)
    rand_samples = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=object.device)
    velocities = root_states[:, 7:13] + rand_samples
    object.write_root_pose_to_sim(torch.cat([positions, orientations], dim=-1), env_ids=env_ids)
    object.write_root_velocity_to_sim(velocities, env_ids=env_ids)
```

---

## §4 Goal + Termination

**Description.** Two `TargetPositionCommand` goals (custom command term): `target_pos` for the egg-beater (`object_0`) and `bowl_target_pos` for the bowl (`object_1`). Each samples a fixed goal pose (deterministic single-point ranges) once at reset; goals do NOT resample on success (`update_goal_on_success=False`) or on time. The command tracks per-env `position_error`, an "orientation_error" (= cosine alignment of the object's z-axis with the goal z-axis, via `get_angle_from_quat`), and a `consecutive_success` counter. **Termination: time_out only** (`StirBowlTerminationsCfg` inherits just `time_out` from `BaseTerminationsCfg`; there is no failure/early-done term). "Success" is not a termination — it is computed inside reward term `bowl.success_bonus`, which writes `env.success_tracker` and requires simultaneously: bowl position_error < 0.1 AND egg-beater position_error < 0.1 AND egg-beater orientation_error > 0.7.

**Decisions resolved.**

| Command | object_id | success_threshold (pos) | success_threshold_orient | pose_range (x,y,z) | update_on_success |
|---|---|---|---|---|---|
| target_pos (egg-beater) | 0 | 0.1 | 0.86 | x=-0.1, y=0.0, z=0.28 | False |
| bowl_target_pos (bowl) | 1 | 0.05 | 0.97 | x=-0.1, y=0.0, z=0.06 | False |

| Termination | value | Source |
|---|---|---|
| time_out | True (episode_length_s=8.3333) | inherited `BaseTerminationsCfg` |
| Failure terms | none | `StirBowlTerminationsCfg: pass` |
| Success predicate (env.success_tracker) | bowl pos_err<0.1 ∧ egg pos_err<0.1 ∧ egg orient_err>0.7 | `bowl.success_bonus` |
| resampling_time_range | (1e6,1e6) — never | `TargetPositionCommandCfg` |
| orientation_error metric | cos-sim of object z-axis vs goal z-axis (higher=better; success when >thr) | `TargetPositionCommand._update_metrics` |

**Code (StirBowlCommandsCfg + StirBowlTerminationsCfg — env_cfg.py).**
```python
@configclass
class StirBowlCommandsCfg(BaseCommandsCfg):
    target_pos = TargetPositionCommandCfg(
        object_id=0, success_threshold=0.1, success_threshold_orient=0.86,
        pose_range={"x": [-0.1, -0.1], "y": [0.0, 0.0], "z": [0.28, 0.28]},
        update_goal_on_success=False, debug_vis=True)
    bowl_target_pos = TargetPositionCommandCfg(
        object_id=1, success_threshold=0.05, success_threshold_orient=0.97,
        pose_range={"x": [-0.1, -0.1], "y": [0.0, 0.0], "z": [0.06, 0.06]},
        update_goal_on_success=False, debug_vis=True)

@configclass
class StirBowlTerminationsCfg(BaseTerminationsCfg):
    pass   # inherits: time_out = DoneTerm(func=mdp.time_out, time_out=True)
```

**Code (command metric/success bookkeeping — grasp_command.py `_update_metrics`).**
```python
    def _update_metrics(self):
        from bimanual_suite.utils.isaac_utils import get_angle_from_quat
        target_axis = get_angle_from_quat(self.quat_command_w, axis="z", normalize=True)
        cur_axis = get_angle_from_quat(self.object.data.root_quat_w, axis="z", normalize=True)
        self.metrics["orientation_error"] = torch.sum(target_axis * cur_axis, dim=-1)
        self.metrics["position_error"] = torch.norm(self.object.data.root_pos_w - self.pos_command_w, dim=1)
        if self.ranges is not None and self.return_type != "pos":
            successes = (torch.logical_and(self.metrics["position_error"] < self.cfg.success_threshold,
                         self.metrics["orientation_error"] > self.cfg.success_threshold_orient)).float()
        else:
            successes = (self.metrics["position_error"] < self.cfg.success_threshold).float()
        unsuccesses = torch.where(successes == 0.0)[0]
        self.metrics["consecutive_success"] += successes.float()
        self.metrics["consecutive_success"][unsuccesses] = 0.0
```

**Code (task success predicate — StirBowl/mdps.py `success_bonus`, writes env.success_tracker).**
```python
def success_bonus(env, num_success: int = 0):
    bowl_command_term = env.command_manager.get_term("bowl_target_pos")
    bowl_within_range = bowl_command_term.metrics["position_error"] < 0.1
    egg_beater_command_term = env.command_manager.get_term("target_pos")
    egg_beater_within_range = egg_beater_command_term.metrics["position_error"] < 0.1
    egg_beater_orient_within_range = egg_beater_command_term.metrics["orientation_error"] > 0.7
    rew = bowl_within_range * egg_beater_within_range * egg_beater_orient_within_range
    env.success_tracker = rew.float()
    return rew
```

---

## §5 Observation

**Description.** Single concatenated `policy` group, corruption OFF (`enable_corruption=False`). Bimanual: full pose+joint state of both arms, poses/velocities of all 5 objects, both goals, and last action. Orientations use the "symmetry" flat-rotation-matrix encoding (a quaternion is emitted as a flattened 3×3 = 9 numbers) — so each pose = 3 (pos) + 9 (R_flat) = 12. **Total = 225** (see table).

**Decisions resolved (per-term dims).**

| Term | func | params | dim |
|---|---|---|---|
| ee_pose_right | `ee_pose` | ee_name="palm_link" (symmetry=True) | 12 |
| joint_pos_right | `joint_pos_limit_normalized` | joints=None, limits=RIGHT | 22 |
| joint_vel_right | `joint_vel` | joints=None | 22 |
| ee_pose_left | `ee_pose` | palm_link, asset robot_left | 12 |
| joint_pos_left | `joint_pos_limit_normalized` | joints=None, limits=LEFT, robot_left | 22 |
| joint_vel_left | `joint_vel` | robot_left | 22 |
| egg_beater_pos | `object_pos` | object_id=0 | 3 |
| egg_beater_quat | `object_quat` | object_id=0, symmetry=True | 9 |
| bowl_pos | `object_pos` | object_id=1 | 3 |
| bowl_quat | `object_quat` | object_id=1, symmetry=True | 9 |
| bowl_lin_vel | `object_lin_vel` | object_id=1 | 3 |
| ball_1_pos / ball_1_lin_vel | `object_pos`/`object_lin_vel` | id=2 | 3 / 3 |
| ball_2_pos / ball_2_lin_vel | id=3 | 3 / 3 |
| ball_3_pos / ball_3_lin_vel | id=4 | 3 / 3 |
| goal_pos_egg_beater | `generated_commands` | "target_pos" (symmetry=True) | 12 |
| goal_pos_bowl | `generated_commands` | "bowl_target_pos" | 12 |
| last_action | `last_action` | — | 44 |
| **Total** | | | **225** |

Robot joint count = 22 (6 arm `joint1..6` + 16 Allegro hand). `enable_corruption=False`, `concatenate_terms=True`.

**Code (StirBowlObservationsCfg.PolicyCfg — env_cfg.py).**
```python
@configclass
class StirBowlObservationsCfg(BaseObservationsCfg):
    @configclass
    class PolicyCfg(ObsGroup):
        ee_pose_right = ObsTerm(func=ee_pose, params={"ee_name": "palm_link"})
        joint_pos_right = ObsTerm(func=joint_pos_limit_normalized, params={"joints": None,
            "joint_lower_limit": JOINT_LOWER_LIMIT, "joint_upper_limit": JOINT_UPPER_LIMIT})
        joint_vel_right = ObsTerm(func=joint_vel, params={"joints": None})
        ee_pose_left = ObsTerm(func=ee_pose, params={"ee_name": "palm_link", "asset_cfg": SceneEntityCfg("robot_left")})
        joint_pos_left = ObsTerm(func=joint_pos_limit_normalized, params={"joints": None,
            "joint_lower_limit": JOINT_LOWER_LIMIT_LEFT, "joint_upper_limit": JOINT_UPPER_LIMIT_LEFT,
            "asset_cfg": SceneEntityCfg("robot_left")})
        joint_vel_left = ObsTerm(func=joint_vel, params={"joints": None, "asset_cfg": SceneEntityCfg("robot_left")})
        egg_beater_pos = ObsTerm(func=object_pos, params={"object_id": 0})
        egg_beater_quat = ObsTerm(func=object_quat, params={"object_id": 0, "symmetry": True})
        bowl_pos = ObsTerm(func=object_pos, params={"object_id": 1})
        bowl_quat = ObsTerm(func=object_quat, params={"object_id": 1, "symmetry": True})
        bowl_lin_vel = ObsTerm(func=object_lin_vel, params={"object_id": 1})
        ball_1_pos = ObsTerm(func=object_pos, params={"object_id": 2})
        ball_1_lin_vel = ObsTerm(func=object_lin_vel, params={"object_id": 2})
        ball_2_pos = ObsTerm(func=object_pos, params={"object_id": 3})
        ball_2_lin_vel = ObsTerm(func=object_lin_vel, params={"object_id": 3})
        ball_3_pos = ObsTerm(func=object_pos, params={"object_id": 4})
        ball_3_lin_vel = ObsTerm(func=object_lin_vel, params={"object_id": 4})
        goal_pos_egg_beater = ObsTerm(func=generated_commands, params={"command_name": "target_pos"})
        goal_pos_bowl = ObsTerm(func=generated_commands, params={"command_name": "bowl_target_pos"})
        last_action = ObsTerm(func=last_action)
        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True
    policy: PolicyCfg = PolicyCfg()
```

---

## §6 Reward

**Description.** 19 `RewTerm`s total: 2 inherited energy penalties (`energy`, `energy_left`), 10 task terms, and 7 symmetry-mirror terms (same funcs applied to the opposite hand / mirrored approach frame). The shaping ladder: (1) `reaching_object` — inverse fingertip→egg-beater distance; (2) `object_lifting` — lift egg-beater to goal height, gated on 4-finger contact; (3) `egg_beater_goal_tracking` / `_orient_tracking` — reduce egg-beater pos/z-axis error, contact-gated; (4) `bowl_goal_tracking` (bowl.object_goal_distance) — hold bowl at goal, orient-gated; (5) `align_hand_to_pos`/`_quat` — align LEFT palm to bowl approach frame; (6) `ball_velocity` (bowl.object_vel) — reward ball speed inside the bowl, gated on egg-beater success AND bowl consecutive_success≥5 (the actual "stirring" signal); (7) `bowl_success_bonus` / `success_bonus` — sparse conjunction bonuses. Contact gating uses `get_allegro_contact` (index/middle/ring OR + thumb AND, force>1 N).

**Decisions resolved.**

| Term | func | key params | weight |
|---|---|---|---|
| energy (inherited) | `energy_punishment` | robot, allegro actuators | (0.0 — not set) |
| energy_left (inherited) | `energy_punishment` | robot_left, allegro actuators | (0.0 — not set) |
| reaching_object | `object_robot_distance` | weight [1,1,1,1.5], links [if5,mf5,pf5,th5], obj 0 | 0.01 |
| object_lifting | `lift_distance` | cmd target_pos, obj 0, right sensors | 5.0 |
| egg_beater_goal_tracking | `object_goal_distance` | cmd target_pos, obj 0, right sensors | 50.0 |
| egg_beater_goal_orient_tracking | `object_goal_distance_orient` | cmd target_pos, obj 0, axis z, right sensors | 20.0 |
| ball_velocity | `bowl.object_vel` | objs [2,3,4], right sensors | 1000.0 |
| bowl_goal_tracking | `bowl.object_goal_distance` | cmd bowl_target_pos, obj 1 | 10.0 |
| align_hand_to_pos | `align_palm_to_pos` | palm_link, frame object_approach_frame, robot_left | 2.0 |
| align_hand_to_quat | `align_palm_to_quat` | palm_link, frame object_approach_frame, robot_left | 0.5 |
| bowl_success_bonus | `bowl.cmd_success_bonus` | cmd bowl_target_pos, num_success 1 | 5.0 |
| success_bonus | `bowl.success_bonus` | num_success 5 | 100.0 |
| reaching_object_symmetry | `object_robot_distance` | …, robot_left | 0.01 |
| object_lifting_symmetry | `lift_distance` | left sensors | 5.0 |
| egg_beater_goal_tracking_symmetry | `object_goal_distance` | left sensors | 50.0 |
| egg_beater_goal_orient_tracking_symmetry | `object_goal_distance_orient` | left sensors | 20.0 |
| ball_velocity_symmetry | `bowl.object_vel` | left sensors | 1000.0 |
| align_hand_to_pos_symmetry | `align_palm_to_pos` | frame …_symmetry, robot | 2.0 |
| align_hand_to_quat_symmetry | `align_palm_to_quat` | frame …_symmetry, robot | 0.5 |

### Symmetric-learning reward terms (drop if not using symmetric learning)

This task trains with **SYMMETRIC LEARNING** (`base.yaml` → `symmetry.symmetric_envs: True`, **C2** group). The BASE terms are genuine reward terms. The `_symmetry`-suffixed terms are **DUPLICATES** of the base terms applied to the mirrored hand / mirrored approach frame, present **only** as augmentation for symmetric learning — **drop them if you are not using symmetric learning.**

`_symmetry` terms: `reaching_object_symmetry`, `object_lifting_symmetry`, `egg_beater_goal_tracking_symmetry`, `egg_beater_goal_orient_tracking_symmetry`, `ball_velocity_symmetry`, `align_hand_to_pos_symmetry`, `align_hand_to_quat_symmetry` (7 total).

**Code (StirBowlRewardsCfg — env_cfg.py).**
```python
@configclass
class StirBowlRewardsCfg(BaseRewardsCfg):
    reaching_object = RewTerm(func=object_robot_distance,
        params={"weight": [1.0, 1.0, 1.0, 1.5], "link_name": ["if5","mf5","pf5","th5"], "object_id": 0}, weight=0.01)
    object_lifting = RewTerm(func=lift_distance,
        params={"command_name": "target_pos", "object_id": 0,
                "sensor_names": ["contact_sensors_0","contact_sensors_1","contact_sensors_2","contact_sensors_3"]}, weight=5.0)
    egg_beater_goal_tracking = RewTerm(func=object_goal_distance,
        params={"command_name": "target_pos", "object_id": 0,
                "sensor_names": ["contact_sensors_0","contact_sensors_1","contact_sensors_2","contact_sensors_3"]}, weight=50.0)
    egg_beater_goal_orient_tracking = RewTerm(func=object_goal_distance_orient,
        params={"command_name": "target_pos", "object_id": 0, "axis": "z",
                "sensor_names": ["contact_sensors_0","contact_sensors_1","contact_sensors_2","contact_sensors_3"]}, weight=20.0)
    ball_velocity = RewTerm(func=bowl.object_vel,
        params={"object_id": [2, 3, 4],
                "sensor_names": ["contact_sensors_0","contact_sensors_1","contact_sensors_2","contact_sensors_3"]}, weight=1000.0)
    bowl_goal_tracking = RewTerm(func=bowl.object_goal_distance,
        params={"command_name": "bowl_target_pos", "object_id": 1}, weight=10.0)
    align_hand_to_pos = RewTerm(func=align_palm_to_pos,
        params={"link_name": ["palm_link"], "frame_name": "object_approach_frame", "asset_cfg": SceneEntityCfg("robot_left")}, weight=2.0)
    align_hand_to_quat = RewTerm(func=align_palm_to_quat,
        params={"link_name": ["palm_link"], "frame_name": "object_approach_frame", "asset_cfg": SceneEntityCfg("robot_left")}, weight=0.5)
    bowl_success_bonus = RewTerm(func=bowl.cmd_success_bonus,
        params={"command_names": "bowl_target_pos", "num_success": 1}, weight=5.0)
    success_bonus = RewTerm(func=bowl.success_bonus, params={"num_success": 5}, weight=100.0)
    # symmetry (mirrored hand / frame)
    reaching_object_symmetry = RewTerm(func=object_robot_distance,
        params={"weight": [1.0,1.0,1.0,1.5], "link_name": ["if5","mf5","pf5","th5"], "object_id": 0,
                "asset_cfg": SceneEntityCfg("robot_left")}, weight=0.01)
    object_lifting_symmetry = RewTerm(func=lift_distance,
        params={"command_name": "target_pos", "object_id": 0,
                "sensor_names": ["contact_sensors_0_left","contact_sensors_1_left","contact_sensors_2_left","contact_sensors_3_left"]}, weight=5.0)
    egg_beater_goal_tracking_symmetry = RewTerm(func=object_goal_distance,
        params={"command_name": "target_pos", "object_id": 0,
                "sensor_names": ["contact_sensors_0_left","contact_sensors_1_left","contact_sensors_2_left","contact_sensors_3_left"]}, weight=50.0)
    egg_beater_goal_orient_tracking_symmetry = RewTerm(func=object_goal_distance_orient,
        params={"command_name": "target_pos", "object_id": 0, "axis": "z",
                "sensor_names": ["contact_sensors_0_left","contact_sensors_1_left","contact_sensors_2_left","contact_sensors_3_left"]}, weight=20.0)
    ball_velocity_symmetry = RewTerm(func=bowl.object_vel,
        params={"object_id": [2,3,4],
                "sensor_names": ["contact_sensors_0_left","contact_sensors_1_left","contact_sensors_2_left","contact_sensors_3_left"]}, weight=1000.0)
    align_hand_to_pos_symmetry = RewTerm(func=align_palm_to_pos,
        params={"link_name": ["palm_link"], "frame_name": "object_approach_frame_symmetry", "asset_cfg": SceneEntityCfg("robot")}, weight=2.0)
    align_hand_to_quat_symmetry = RewTerm(func=align_palm_to_quat,
        params={"link_name": ["palm_link"], "frame_name": "object_approach_frame_symmetry", "asset_cfg": SceneEntityCfg("robot")}, weight=0.5)
```

**Code (task-local reward funcs — StirBowl/mdps.py).**
```python
def cmd_success_bonus(env, command_names, num_success: int = 0):
    if isinstance(command_names, str):
        command_term = env.command_manager.get_term(command_names)
        success = command_term.metrics["consecutive_success"] >= num_success
    else:
        success = torch.ones(env.num_envs, device=env.device)
        for command_name in command_names:
            command_term = env.command_manager.get_term(command_name)
            success = torch.logical_and(success, command_term.metrics["consecutive_success"] >= num_success)
    return success.float()

def if_egg_beater_success(env, sensor_names):
    egg_beater_command_term = env.command_manager.get_term("target_pos")
    within_range = torch.ones(env.num_envs, device=env.device)
    within_range = reduce(torch.logical_and, [within_range,
        egg_beater_command_term.metrics["position_error"] < egg_beater_command_term.cfg.success_threshold,
        egg_beater_command_term.metrics["orientation_error"] > 0.8])
    if sensor_names is not None:
        within_range = within_range * get_allegro_contact(env, sensor_names)
    return within_range.float()

def success_bonus(env, num_success: int = 0):
    bowl_command_term = env.command_manager.get_term("bowl_target_pos")
    bowl_within_range = bowl_command_term.metrics["position_error"] < 0.1
    egg_beater_command_term = env.command_manager.get_term("target_pos")
    egg_beater_within_range = egg_beater_command_term.metrics["position_error"] < 0.1
    egg_beater_orient_within_range = egg_beater_command_term.metrics["orientation_error"] > 0.7
    rew = bowl_within_range * egg_beater_within_range * egg_beater_orient_within_range
    env.success_tracker = rew.float()
    return rew

def object_vel(env, object_id=0, sensor_names=[...]):
    """Reward encouraging ball velocity (the 'stir')."""
    if isinstance(object_id, int):
        object = env.scene[f"object_{object_id}"]
        vel = torch.abs(object.data.root_lin_vel_w[:, :2])
    else:
        vel = None
        for id in object_id:
            object = env.scene[f"object_{id}"]
            vel = torch.abs(object.data.root_lin_vel_w[:, :2]) if vel is None else vel + torch.abs(object.data.root_lin_vel_w[:, :2])
    bowl_command_term = env.command_manager.get_term("bowl_target_pos")
    success = bowl_command_term.metrics["consecutive_success"] >= 5
    rew = torch.norm(vel, dim=1) * if_egg_beater_success(env, sensor_names) * success.float()
    return rew

def object_goal_distance(env, command_name, object_id=0):
    """Bowl goal tracking (orient-gated, no contact gate)."""
    object = env.scene[f"object_{object_id}"]
    command = env.command_manager.get_command(command_name)
    des_pos_w = command[:, :3] + env.scene.env_origins
    distance = torch.norm(des_pos_w - object.data.root_pos_w[:, :3], dim=1)
    max_distance = torch.norm(des_pos_w - env.object_init_pos[object_id], dim=1)
    distance = torch.clamp(max_distance - distance, min=0.0)
    within_range = torch.ones(env.num_envs, device=env.device)
    command_term = env.command_manager.get_term(command_name)
    within_range = torch.logical_and(within_range, command_term.metrics["orientation_error"] > command_term.cfg.success_threshold_orient)
    rew = distance * within_range.float()
    return rew
```

**Code (shared reward funcs referenced — reward_mdps.py, abridged to used terms).**
```python
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

def get_allegro_contact(env, sensor_names):  # contact gate (filtered force > 1N)
    force = get_force(env, sensor_names, if_filter=True)
    is_contact = (torch.norm(force, dim=-1) > 1.0)
    is_contact_index_or_middle_or_ring = reduce(torch.logical_or, [is_contact[:,0], is_contact[:,1], is_contact[:,2]])
    is_contact = reduce(torch.logical_and, [is_contact_index_or_middle_or_ring, is_contact[:,3]])
    return is_contact

def lift_distance(env, command_name, minimal_height=None, object_id=0, sensor_names=[...]):
    command = env.command_manager.get_command(command_name)
    des_pos_w = command[:, :3] + env.scene.env_origins
    if minimal_height is None:
        minimal_height = des_pos_w[:, 2] + 0.05
    object = env.scene[f"object_{object_id}"]
    z_distance = (object.data.root_pos_w[:, 2] - env.object_init_pos[object_id][:, 2]) / (minimal_height - env.object_init_pos[object_id][:, 2])
    z_distance = torch.clamp(z_distance, min=0.0)
    return (object.data.root_pos_w[:, 2] < minimal_height) * z_distance * get_allegro_contact(env, sensor_names)

def object_goal_distance(env, command_name, object_id=0, sensor_names=[...]):  # egg-beater (contact+height gated)
    object = env.scene[f"object_{object_id}"]
    command = env.command_manager.get_command(command_name)
    des_pos_w = command[:, :3] + env.scene.env_origins
    distance = torch.norm(des_pos_w - object.data.root_pos_w[:, :3], dim=1)
    max_distance = torch.norm(des_pos_w - env.object_init_pos[object_id], dim=1)
    distance = torch.clamp(max_distance - distance, min=0.0)
    return distance * get_allegro_contact(env, sensor_names) * torch.where(object.data.root_pos_w[:, 2] > (des_pos_w[:, 2] - 0.05), 1.0, 0.0)

def object_goal_distance_orient(env, command_name, object_id=0, axis="z", pos_success_threshold=None, sensor_names=[...]):
    object = env.scene[f"object_{object_id}"]
    command = env.command_manager.get_command(command_name)
    des_orient_w = command[:, 3:]
    if axis is not None:
        from bimanual_suite.utils.isaac_utils import get_angle_from_quat
        target_axis = get_angle_from_quat(des_orient_w, axis=axis, normalize=True)
        init_axis = get_angle_from_quat(env.object_init_orient[object_id], axis=axis, normalize=True)
        cur_axis = get_angle_from_quat(object.data.root_quat_w, axis=axis, normalize=True)
        distance = torch.sum(target_axis * cur_axis, dim=-1) * (2**0.5)
        distance = torch.clamp(distance, min=0.0)
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
    link_idx = robot.find_bodies(link_name)[0]
    robot_state_w = robot.data.body_state_w[:, link_idx, :7].reshape(-1, 7)
    ori_distance = math_utils.quat_error_magnitude(robot_state_w[:, 3:7], env.scene[frame_name].data.target_quat_w.reshape(-1, 4))
    return -ori_distance

def align_palm_to_pos(env, link_name, asset_cfg=SceneEntityCfg("robot"), frame_name="approach_frame"):
    robot = env.scene[asset_cfg.name]
    link_idx = robot.find_bodies(link_name)[0]
    robot_state_w = robot.data.body_state_w[:, link_idx, :7].reshape(-1, 7)
    distance = torch.norm(robot_state_w[:, :3] - env.scene[frame_name].data.target_pos_w.reshape(-1, 3), dim=-1)
    return -distance

def energy_punishment(env, actuator_name=None, asset_cfg=SceneEntityCfg("robot")):
    energy = get_actuator_energy_consumption(env, asset_cfg.name, actuator_name) if actuator_name is not None \
             else get_energy_consumption(env, asset_cfg.name)
    return -energy
```

**Code (inherited energy terms — manager_based_env_cfg.py, BaseRewardsCfg).**
```python
@configclass
class BaseRewardsCfg:
    energy = RewTerm(func=energy_punishment, weight=0.0,
        params={"asset_cfg": SceneEntityCfg("robot"), "actuator_name": ["allegro_hand_1","allegro_hand_2","allegro_hand_3","allegro_hand_4",
            "allegro_hand_thumb_1","allegro_hand_thumb_2","allegro_hand_thumb_3","allegro_hand_thumb_4"]})
    energy_left = RewTerm(func=energy_punishment, weight=0.0,
        params={"asset_cfg": SceneEntityCfg("robot_left"), "actuator_name": [... same allegro actuators ...]})
```

---

## §7 DR

**Static DR wired in the task cfg: `<no DR>`.** `StirBowlEventCfg` contains only `mode="reset"` state resets (§3) — no `reset`/`interval` randomization EventTerms (no mass/friction/gain/push randomization declared in the config).

**However**, a *runtime, hydra-gated* DR path exists in the shared base (not part of the task cfg, driven externally by `cfg.hydra_cfg.task.randomize`):
- `BaseEnv._post_init_process` constructs `self.domain_randomizer = DomainRandomizer(self.cfg.hydra_cfg.task.randomize, num_envs=...)`.
- `BaseEnv.update_randomization(success_rate)` (called by the training loop, not the env step) samples from the `DomainRandomizer` and applies, when present in the hydra config: `object_mass` (`randomize_mass`), `static/dynamic_friction`+`restitution` (`randomize_material`, 250 buckets, applied to all objects + both robots), `action_scale` (overwrites `self._scale[:6]` arm rates), `energy_penalty`/`collision_penalty` reward-weight curricula (`randomize_rew_weight`), `external_force_torque` (`randomize_external_force_torque`), and any `reset_pose*` (`randomize_reset_pose` mutates event pose_range). These are curriculum-driven by `success_rate` and only active if the corresponding keys exist in the hydra task config — none are hard-coded in `StirBowlEnvCfg`.

So: **no declarative DR in the task**; DR is opt-in via the external hydra config through the shared `DomainRandomizer` machinery.
