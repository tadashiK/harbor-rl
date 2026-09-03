# PickPlaceObject — Implementation Spec

- robot: Bimanual UF850 arms + dual Allegro hands (44 DoF)
- simulator: IsaacLab (Isaac Sim, manager-based)
- objects: two DexCubes, tote bin, table
- bimanual: true
- summary: Two hands pick two cubes off a table and drop them into a tote, in a fixed order.

> Source package name anonymized as `bimanual_suite`. This task comes from an internal
> bimanual manipulation suite rather than a public repo; the design below is otherwise verbatim.

## Task summary

Two UF850 arms, each tipped with a 16-DOF Allegro hand (`robot` = right, `robot_left` = left), must **pick two DexCube objects off a table and drop them into a tote** (a kinematic collision-box bin). The task is *ordered*: object_1 (reached by the right hand) must land in the tote first, then object_2 (left hand). The env runs a C2-symmetry augmentation (`symmetry_tracker`) that can mirror right/left roles per env at reset. Both arms are gravity-disabled and driven by an **EMA cumulative-relative joint-position action** (per-step delta accumulated onto an init pose, exponentially smoothed, clamped to per-joint limits). Success = each object has been "in the tote" for ≥3 update-ticks AND then held there for ≥5 consecutive env steps; the reward is a bank of shaping terms (reach, lift, goal-tracking, in-tote, arm-return, energy, collision) for both arms plus a `success_bonus` (weight 1.0). A custom `PickObjectEnv.step()` tracks "object on top of tote" (distance < 0.08 m to the tote-top command) and the consecutive-success counter.

Key facts:
- **Joints per arm:** 6 arm (`joint1..joint6`) + 16 hand (`jif1-4, jmf1-4, jpf1-4, jth1-4`) = **22**.
- **action_dim = 44** (22 right + 22 left), matching the 44-entry `action_scale`.
- **num_object = 3**: `object_0` = tote (kinematic), `object_1` = DexCube (dynamic, right target), `object_2` = DexCube (dynamic, left target).
- **Policy observation dim = 171** (see §5).
- **sim.dt = 1/120**, **decimation = 6** → control dt = 1/20 s (0.05 s), **episode_length_s = 8.3333** (~166 control steps).

---

## §1 Registration + Scene

**Description.** `PickObjectEnv` subclasses `BaseEnv` (`manager_based_env.py`, itself a `ManagerBasedRLEnv`). Registration is in `bimanual_suite/env/__init__.py`. The env is built from `PickObjectEnvCfg`, which composes `PickObjectSceneCfg(num_envs=4096, env_spacing=3.0)` on top of `BaseSceneCfg` (ground/light/table). The scene holds two articulations (right `Robot`, left `Robot_left`), three rigid objects (`Object_0` tote kinematic, `Object_1`/`Object_2` DexCubes dynamic), a table (kinematic), and a large bank of fingertip + link contact sensors filtered against the two cubes. Asset paths resolve through `bimanual_suite.LIB_PATH` (= repo root, `str(Path(__file__).resolve().parent.parent)`) and `ISAAC_NUCLEUS_DIR`.

**Decisions resolved**

| Question | Value |
|---|---|
| num_envs / env_spacing | 4096 / 3.0 |
| sim.dt / decimation / render_interval | 1/120 / 6 / 6 (= decimation) |
| episode_length_s / control dt | 8.3333 s / 0.05 s (~166 steps) |
| replicate_physics | False (`BaseSceneCfg`) |
| physx knobs | `gpu_max_rigid_contact_count=2**24`, `gpu_max_rigid_patch_count=2**24` |
| sim physics material | static_friction=1.5, dynamic_friction=1.0, restitution=0.0, restitution_combine_mode=min |
| Right robot USD / init pos | `{LIB_PATH}/assets/ufactory850/uf850_allegro_right_colored.usd` / (-0.274, -0.475, 0.01) |
| Left robot USD / init pos | `{LIB_PATH}/assets/ufactory850/uf850_allegro_left_colored.usd` / (-0.274, 0.475, 0.01) |
| Robot gravity | disabled (`disable_gravity=True`) both arms |
| Object_0 (tote) USD / dynamics / scale | `{LIB_PATH}/assets/object/tote_collision.usd` / kinematic_enabled=True, mass=1.0 / scale=(0.6, 0.45, 1.0) |
| Object_1 / Object_2 USD / dynamics | `{ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd` / kinematic_enabled=False, mass=0.11, scale=(1.2,1.2,1.2) |
| Table USD / dynamics | `{LIB_PATH}/assets/object/table.usd` / kinematic_enabled=True, rot=(0.70710678,0,0,0.70710678) |
| Ground init z | -0.82 |
| Light | DomeLight color (0.75,0.75,0.75) intensity 2500 |
| Fingertip contact links | `if5` (index), `mf5` (middle), `pf5` (pinky), `th5` (thumb) — per arm, filtered per cube |
| Link contact sensors | `link1..link6` per arm (no filter) |
| Marker/ee_frame | none by default (`visualize_marker=False`); debug palm markers only if enabled |

**Code — env class + custom step (`env.py`):**

```python
class PickObjectEnv(BaseEnv):
    is_vector_env: ClassVar[bool] = True
    """Whether the environment is a vectorized environment."""
    metadata: ClassVar[dict[str, Any]] = {
        "render_modes": [None, "human", "rgb_array"],
        "isaac_sim_version": get_version(),
    }
    """Metadata for the environment."""

    cfg: PickObjectEnvCfg
    """Configuration for the environment."""
    
    def step(self, action: torch.Tensor) -> VecEnvStepReturn:
        super().step(action)
        # check if the object is on top of the tote
        object_ids = [1, 2]
        tote_top_pos = self.command_manager.get_command("target_pos") + self.scene.env_origins
        for object_id in object_ids:
            object = self.scene[f"object_{object_id}"]
            distance = torch.norm(tote_top_pos - object.data.root_pos_w, dim=1)
            idx = torch.where(distance < 0.08)[0]
            self.object_on_tote_tracker[object_id][idx] += 1

        # check if all objects are in the tote and track the success steps
        success = reduce(torch.logical_and, [self.object_in_tote_tracker[1] >= 3, self.object_in_tote_tracker[2] >= 3]).bool()
        success_idx = torch.where(success)[0]
        unsuccess_idx = torch.where(~success)[0]
        self.success_step_tracker[success_idx] += 1
        self.success_step_tracker[unsuccess_idx] = 0

        # Debug only
        if self.cfg.visualize_marker:
            right_palm_idx = self.scene["robot"].find_bodies("palm_link")[0]
            right_palm = self.scene["robot"].data.body_state_w[:, right_palm_idx, :7].reshape(-1, 7)
            left_palm_idx = self.scene["robot_left"].find_bodies("palm_link")[0]
            left_palm = self.scene["robot_left"].data.body_state_w[:, left_palm_idx, :7].reshape(-1, 7)
            self.markers['arm_r']['ee_marker'].visualize(right_palm[:, :3], right_palm[:, 3:7])
            self.markers['arm_r']['goal_marker'].visualize(left_palm[:, :3], left_palm[:, 3:7])

        # return observations, rewards, resets and extras
        return self.obs_buf, self.reward_buf, self.reset_terminated, self.reset_time_outs, self.extras

    def _pre_init_process(self):
        super()._pre_init_process()
        self.success_step_tracker = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.object_on_tote_tracker = torch.zeros(self.num_object, self.num_envs, device=self.device)
        self.object_in_tote_tracker = torch.zeros(self.num_object, self.num_envs, device=self.device)

    def _post_reset_process(self, env_ids):
        super()._post_reset_process(env_ids)
        self.object_on_tote_tracker[:, env_ids] = 0.0
        self.object_in_tote_tracker[:, env_ids] = 0.0
        self.success_step_tracker[env_ids] = 0
```

**Code — `PickObjectEnvCfg` (`env_cfg.py`):**

```python
@configclass
class PickObjectEnvCfg(BaseEnvCfg):
    name: str = "PickObject"
    scene = PickObjectSceneCfg(num_envs=4096, env_spacing=3.0)
    events = PickObjectEventCfg()
    commands = PickObjectCommandsCfg()
    observations = PickObjectObservationsCfg()
    actions = PickObjectActionsCfg()
    terminations = PickObjectTerminationsCfg()
    rewards = PickObjectRewardsCfg()
    num_object = 3
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

    visualize_marker: bool = False

    def __post_init__(self):
        # post init of parent
        super().__post_init__()
```

**Code — shared base env cfg (`manager_based_env_cfg.py`): sim/timing, table, ground, light, joint limits:**

```python
JOINT_LOWER_LIMIT = [-6.283, -2.304, -4.224, -6.283, -2.164, -6.283,
                    # jif1, jmf1, jpf1, jth1
                    -0.05, -0.05, -0.570, 0.364,
                    # jif2, jmf2, jpf2, jth2
                    -0.296, -0.296, -0.296, -0.205,
                    # jif3, jmf3, jpf3, jth3
                    -0.274, -0.274, -0.274, -0.290,
                    # jif4, jmf4, jpf4, jth4
                    -0.327, -0.327, -0.327, -0.262]
JOINT_UPPER_LIMIT = [6.283, 2.304, 0.061, 6.283, 2.164, 6.283,
                    # jif1, jmf1, jpf1, jth1
                    0.570, 0.05, 0.05, 1.497,
                    # jif2, jmf2, jpf2, jth2
                    1.710, 1.710, 1.710, 1.130, 
                    # jif3, jmf3, jpf3, jth3
                    1.809, 1.809, 1.809, 1.633, 
                    # jif4, jmf4, jpf4, jth4
                    1.718, 1.718, 1.718, 1.820]
JOINT_LOWER_LIMIT_LEFT = [-6.283, -2.304, -4.224, -6.283, -2.164, -6.283,
                    # jif1, jmf1, jpf1, jth1
                    -0.570, -0.05, -0.05, 0.364,
                    # jif2, jmf2, jpf2, jth2
                    -0.296, -0.296, -0.296, -0.205,
                    # jif3, jmf3, jpf3, jth3
                    -0.274, -0.274, -0.274, -0.290,
                    # jif4, jmf4, jpf4, jth4
                    -0.327, -0.327, -0.327, -0.262]
JOINT_UPPER_LIMIT_LEFT = [6.283, 2.304, 0.061, 6.283, 2.164, 6.283,
                    # jif1, jmf1, jpf1, jth1
                    0.05, 0.05, 0.570, 1.497,
                    # jif2, jmf2, jpf2, jth2
                    1.710, 1.710, 1.710, 1.130, 
                    # jif3, jmf3, jpf3, jth3
                    1.809, 1.809, 1.809, 1.633, 
                    # jif4, jmf4, jpf4, jth4
                    1.718, 1.718, 1.718, 1.820]

@configclass
class BaseSceneCfg(InteractiveSceneCfg):
    """Configuration for the scene with a robotic arm."""

    # world
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -0.82)),
    )

    # lights
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=2500.0),
    )

    # table
    table: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Table",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{bimanual_suite.LIB_PATH}/assets/object/table.usd",
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True,
                disable_gravity=False,
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=1,
                max_depenetration_velocity=10.0,
            ),
            scale=(1.0, 1.0, 1.0),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0), rot=(0.70710678, 0, 0., 0.70710678)),
    )

    replicate_physics = False

@configclass
class BaseEnvCfg(ManagerBasedRLEnvCfg):
    ...
    sim: SimulationCfg = SimulationCfg(
        physics_material=RigidBodyMaterialCfg(
            static_friction=1.5,
            dynamic_friction=1.0,
            restitution=0.0,
            restitution_combine_mode=min,
        ),
        physx=PhysxCfg(
            gpu_max_rigid_contact_count=2**24,
            gpu_max_rigid_patch_count=2**24,
        ),
    )

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 6
        self.episode_length_s = 8.3333
        self.viewer.eye = (3.5, 3.5, 3.5)
        # simulation settings
        self.sim.dt = 1.0 / 120.0
        self.sim.render_interval = self.decimation
```

**Code — right robot `Robot` (init pose + actuators; `env_cfg.py`):**

```python
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
                "joint1": 0.5,
                "joint2": 0.3,
                "joint3": -0.6,
                "joint4": 0.0,
                "joint5": -0.8,
                "joint6": -1.57,
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
```

**Left robot `Robot_left`** is IDENTICAL except: USD `uf850_allegro_left_colored.usd`; `joint1 = -0.5`, `joint6 = 1.57` (mirrored); `pos=(-0.274, 0.475, 0.01)`. All actuator groups/gains identical to the right robot.

**Code — objects (tote + two DexCubes; `env_cfg.py`):**

```python
    object_0 = RigidObjectCfg(
        prim_path=f"/World/envs/env_.*/Object_0",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{bimanual_suite.LIB_PATH}/assets/object/tote_collision.usd",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True,
                disable_gravity=False,
                max_linear_velocity=1000,
                max_angular_velocity=1000,
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=1,
                max_depenetration_velocity=1000.0,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
            activate_contact_sensors=True,
            scale=(0.6, 0.45, 1.0),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            lin_vel=(0.0, 0.0, 0.0), ang_vel=(0.0, 0.0, 0.0), pos=(0.0, 0.0, 0.0),
        ),
    )

    object_1 = RigidObjectCfg(
        prim_path=f"/World/envs/env_.*/Object_1",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=False,
                disable_gravity=False,
                max_linear_velocity=1000,
                max_angular_velocity=1000,
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=1,
                max_depenetration_velocity=1000.0,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.11),
            activate_contact_sensors=True,
            scale=(1.2, 1.2, 1.2),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            lin_vel=(0.0, 0.0, 0.0), ang_vel=(0.0, 0.0, 0.0), pos=(0.0, 0.0, 0.0),
        ),
    )
    # object_2 is byte-identical to object_1 (same DexCube USD, mass 0.11, scale 1.2, dynamic).
```

**Code — contact sensors (`env_cfg.py`).** Right-hand fingertips filtered vs `Object_1`; left-hand fingertips filtered vs `Object_2`; plus per-arm `link1..link6` sensors; plus a full set of "symmetry" and "left_symmetry" sensors that swap which hand is filtered against which cube (used by the symmetry-augmented reward terms):

```python
    contact_sensors_0 = ContactSensorCfg(  # right index → Object_1
        prim_path="/World/envs/env_.*/Robot/if5", update_period=0.0, debug_vis=True,
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Object_1"],
    )
    contact_sensors_1 = ContactSensorCfg(  # right middle → Object_1  (prim mf5)
        prim_path="/World/envs/env_.*/Robot/mf5", update_period=0.0, debug_vis=True,
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Object_1"],
    )
    contact_sensors_2 = ContactSensorCfg(  # right pinky → Object_1  (prim pf5)
        prim_path="/World/envs/env_.*/Robot/pf5", update_period=0.0, debug_vis=True,
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Object_1"],
    )
    contact_sensors_3 = ContactSensorCfg(  # right thumb → Object_1  (prim th5)
        prim_path="/World/envs/env_.*/Robot/th5", update_period=0.0, debug_vis=True,
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Object_1"],
    )
    # contact_sensors_{0,1,2,3}_symmetry : Robot_left/{if5,mf5,pf5,th5} filtered vs Object_1
    # contact_sensors_{0,1,2,3}_left     : Robot_left/{if5,mf5,pf5,th5} filtered vs Object_2
    # contact_sensors_{0,1,2,3}_left_symmetry : Robot/{if5,mf5,pf5,th5} filtered vs Object_2
    contact_sensors_robot = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/link1|link2|link3|link4|link5|link6",
        update_period=0.0, debug_vis=True,
    )
    contact_sensors_robot_left = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot_left/link1|link2|link3|link4|link5|link6",
        update_period=0.0, debug_vis=True,
    )
```

Full sensor roster (16 filtered fingertip sensors + 2 link sensors): `contact_sensors_{0,1,2,3}` (R→Obj1), `contact_sensors_{0,1,2,3}_symmetry` (L→Obj1), `contact_sensors_{0,1,2,3}_left` (L→Obj2), `contact_sensors_{0,1,2,3}_left_symmetry` (R→Obj2), `contact_sensors_robot`, `contact_sensors_robot_left`.

---

## §2 Actions

**Description.** Two action terms — one per arm — both `EMACumulativeRelativeJointPositionActionCfg` (`actions_cfg.py` / `actions.py`), each covering all 22 joints (`joint_names=[".*"]`). Per-arm `scale=1.0`; the true per-joint scaling is applied ONCE in `BaseEnv.step()` as `action = action * self._scale`, where `self._scale = torch.tensor(cfg.action_scale)` (the 44-vector). So the effective per-step delta = `raw_action * action_scale`. The action term is *cumulative-relative*: each step the scaled action is added to a running `del_action`, offset by the init joint pose, then EMA-smoothed (`alpha=0.2`) against the previous applied target, and finally clamped to `[joint_lower_limit, joint_upper_limit]`. `use_default_offset=False`. Total action dim = **44**.

**Decisions resolved**

| Question | Value |
|---|---|
| terms | `arm_hand_action` (robot), `arm_hand_action_left` (robot_left) |
| joints per term | `[".*"]` → all 22 joints each |
| per-term scale | 1.0 (real scale via `action_scale` in `BaseEnv.step`) |
| alpha (EMA) | 0.2 (both arms) |
| use_default_offset | False |
| limits | right → `JOINT_LOWER/UPPER_LIMIT`; left → `..._LEFT` (differ only on jif1/jpf1 hand joints) |
| action_scale (44) | arm joints 0.05 ×6; hand 0.03 (jth3=0.015) — mirrored for left arm |
| total action_dim | 44 |

**Code — action cfg (`env_cfg.py`):**

```python
@configclass
class PickObjectActionsCfg:
    arm_hand_action = EMACumulativeRelativeJointPositionActionCfg(
        asset_name="robot",
        joint_names=[".*"],
        scale=1.0,
        use_default_offset=False,
        joint_lower_limit=JOINT_LOWER_LIMIT,
        joint_upper_limit=JOINT_UPPER_LIMIT,
        alpha=0.2
    )
    arm_hand_action_left = EMACumulativeRelativeJointPositionActionCfg(
        asset_name="robot_left",
        joint_names=[".*"],
        scale=1.0,
        use_default_offset=False,
        joint_lower_limit=JOINT_LOWER_LIMIT_LEFT,
        joint_upper_limit=JOINT_UPPER_LIMIT_LEFT,
        alpha=0.2
    )
```

**Code — action term cfg (`actions_cfg.py`):**

```python
@configclass
class EMACumulativeRelativeJointPositionActionCfg(JointPositionActionCfg):
    class_type: type[ActionTerm] = EMACumulativeRelativeJointPositionAction
    alpha: float | dict[str, float] = 1.0
    joint_lower_limit: list[float] = None
    joint_upper_limit: list[float] = None
```

**Code — per-step processing rule (`actions.py`):**

```python
class EMACumulativeRelativeJointPositionAction(JointPositionAction):
    def __init__(self, cfg, env) -> None:
        super().__init__(cfg, env)
        if isinstance(cfg.alpha, float):
            if not 0.0 <= cfg.alpha <= 1.0:
                raise ValueError(f"Moving average weight must be in the range [0, 1]. Got {cfg.alpha}.")
            self._alpha = cfg.alpha
        elif isinstance(cfg.alpha, dict):
            self._alpha = torch.ones((env.num_envs, self.action_dim), device=self.device)
            index_list, names_list, value_list = string_utils.resolve_matching_names_values(cfg.alpha, self._joint_names)
            for name, value in zip(names_list, value_list):
                if not 0.0 <= value <= 1.0:
                    raise ValueError(f"Moving average weight must be in the range [0, 1]. Got {value} for joint {name}.")
            self._alpha[:, index_list] = torch.tensor(value_list, device=self.device)
        else:
            raise ValueError(f"Unsupported moving average weight type: {type(cfg.alpha)}. ...")
        self._prev_applied_actions = torch.zeros_like(self.processed_actions)
        self.del_action = torch.zeros((self._env.num_envs, self.action_dim), device=self._env.device)
        self.init_joint_pos = self._asset.data.joint_pos[:, self._joint_ids].clone()
        self.joint_lower_limit = torch.tensor(cfg.joint_lower_limit, device=self.device) if cfg.joint_lower_limit is not None else None
        self.joint_upper_limit = torch.tensor(cfg.joint_upper_limit, device=self.device) if cfg.joint_upper_limit is not None else None

    def reset(self, env_ids=None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        super().reset(env_ids)
        self._prev_applied_actions[env_ids, :] = self._asset.data.joint_pos[env_ids][:, self._joint_ids].clone()
        self.del_action[env_ids, :] = torch.zeros((env_ids.shape[0], self.action_dim), device=self.device)
        self.init_joint_pos[env_ids, :] = self._asset.data.joint_pos[env_ids][:, self._joint_ids].clone()

    def process_actions(self, actions: torch.Tensor):
        super().process_actions(actions)                    # affine (scale=1, offset=0)
        self._processed_actions += self.del_action          # accumulate delta
        self.del_action = self._processed_actions.clone()
        self._processed_actions += self.init_joint_pos.clone()   # offset by init pose
        ema_actions = self._alpha * self._processed_actions      # EMA smoothing
        ema_actions += (1.0 - self._alpha) * self._prev_applied_actions
        if self.joint_lower_limit is not None and self.joint_upper_limit is not None:
            self._processed_actions[:] = torch.clamp(ema_actions, self.joint_lower_limit, self.joint_upper_limit)
        else:
            self._processed_actions[:] = ema_actions
        self._prev_applied_actions[:] = self._processed_actions[:]
```

**Code — scale application (`manager_based_env.py`, `BaseEnv.step`):**

```python
    def step(self, action: torch.Tensor) -> VecEnvStepReturn:
        self.last_action = action.clone()
        action = action * self._scale        # self._scale = torch.tensor(cfg.action_scale)  (44-vec)
        super().step(action)
        self.extras['success'] = self.success_tracker
        self.extras['detailed_reward'] = self.detailed_reward_buf
        reset_indices = torch.where(self.episode_length_buf == 1)[0]
        if len(reset_indices) > 0:
            self._post_reset_process(reset_indices)
        return self.obs_buf, self.reward_buf, self.reset_terminated, self.reset_time_outs, self.extras
```

---

## §3 Reset / Events

**Description.** `PickObjectEventCfg` extends `BaseEventCfg`. Base contributes `reset_robot_joints` (right robot, `reset_joints_by_symmetry`). The task adds a left-robot joint reset (`mdp.reset_joints_by_scale`, position_range (1,1) → default pose) and three `reset_object` terms placing tote (id 0), Object_1 (id 1, right side, y=-0.35, random yaw), Object_2 (id 2, left side, y=+0.35, random yaw). All events are `mode="reset"`. Note `reset_joints_by_symmetry` mirrors the *other* arm's default pose into this arm for the ~50% of envs flagged symmetric (`symmetry_tracker == 1`), via `env.rep_Q_js`.

**Decisions resolved**

| Event | func | mode | key params |
|---|---|---|---|
| reset_robot_joints (base) | `reset_joints_by_symmetry` | reset | position_range (1.0,1.0), velocity_range (0.0,0.0), asset "robot" |
| reset_robot_joints_left | `mdp.reset_joints_by_scale` | reset | position_range (1.0,1.0), velocity_range (0.0,0.0), asset "robot_left" |
| reset_tote (obj 0) | `reset_object` | reset | pose x=[0.1,0.1], y=0, z=0 |
| reset_object_1 (obj 1) | `reset_object` | reset | x=[0.1,0.1], y=[-0.35,-0.35], z=0, yaw=[-3.14,3.14] |
| reset_object_2 (obj 2) | `reset_object` | reset | x=[0.1,0.1], y=[0.35,0.35], z=0, yaw=[-3.14,3.14] |

**Code — `PickObjectEventCfg` (`env_cfg.py`):**

```python
@configclass
class PickObjectEventCfg(BaseEventCfg):
    reset_robot_joints_left = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={"position_range": (1.0, 1.0), "velocity_range": (0.0, 0.0),
                "asset_cfg": SceneEntityCfg("robot_left")},
    )
    reset_tote = EventTerm(
        func=reset_object, mode="reset",
        params={"pose_range": {"x": [0.1, 0.1], "y": [0.0, 0.0], "z": [0.0, 0.0]},
                "velocity_range": {}, "object_id": 0},
    )
    reset_object_1 = EventTerm(
        func=reset_object, mode="reset",
        params={"pose_range": {"x": [0.1, 0.1], "y": [-0.35, -0.35], "z": [0.0, 0.0], "yaw": [-3.14, 3.14]},
                "velocity_range": {}, "object_id": 1},
    )
    reset_object_2 = EventTerm(
        func=reset_object, mode="reset",
        params={"pose_range": {"x": [0.1, 0.1], "y": [0.35, 0.35], "z": [0.0, 0.0], "yaw": [-3.14, 3.14]},
                "velocity_range": {}, "object_id": 2},
    )
```

**Code — base event + reset funcs (`manager_based_env_cfg.py`, `reset_mdps.py`):**

```python
@configclass
class BaseEventCfg:
    reset_robot_joints = EventTerm(
        func=reset_joints_by_symmetry, mode="reset",
        params={"position_range": (1.0, 1.0), "velocity_range": (0.0, 0.0)},
    )
```

```python
def reset_joints_by_symmetry(env, env_ids, position_range, velocity_range, asset_cfg=SceneEntityCfg("robot")):
    asset = env.scene[asset_cfg.name]
    joint_pos = asset.data.default_joint_pos[env_ids].clone()
    joint_vel = asset.data.default_joint_vel[env_ids].clone()
    symmetric_idx = torch.where(env.symmetry_tracker[env_ids] == 1)[0]
    env_symmetric_idx = env_ids[symmetric_idx]
    if asset_cfg.name == "robot":
        symmetric_joint_pos = env.scene["robot_left"].data.default_joint_pos[env_symmetric_idx].clone() @ env.rep_Q_js
        symmetric_joint_vel = env.scene["robot_left"].data.default_joint_vel[env_symmetric_idx].clone() @ env.rep_Q_js
    else:
        symmetric_joint_pos = env.scene["robot"].data.default_joint_pos[env_symmetric_idx].clone() @ env.rep_Q_js
        symmetric_joint_vel = env.scene["robot"].data.default_joint_vel[env_symmetric_idx].clone() @ env.rep_Q_js
    joint_pos[symmetric_idx] = symmetric_joint_pos
    joint_vel[symmetric_idx] = symmetric_joint_vel
    joint_pos *= math_utils.sample_uniform(*position_range, joint_pos.shape, joint_pos.device)
    joint_vel *= math_utils.sample_uniform(*velocity_range, joint_vel.shape, joint_vel.device)
    joint_pos_limits = asset.data.soft_joint_pos_limits[env_ids]
    joint_pos = joint_pos.clamp_(joint_pos_limits[..., 0], joint_pos_limits[..., 1])
    joint_vel_limits = asset.data.soft_joint_vel_limits[env_ids]
    joint_vel = joint_vel.clamp_(-joint_vel_limits, joint_vel_limits)
    asset.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)

def reset_object(env, env_ids, pose_range, velocity_range, object_id):
    object = env.scene[f"object_{object_id}"]
    root_states = object.data.default_root_state[env_ids].clone()
    range_list = [pose_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
    ranges = torch.tensor(range_list, device=object.device)
    rand_samples = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=object.device)
    positions = root_states[:, 0:3] + env.scene.env_origins[env_ids] + rand_samples[:, 0:3]
    orientations_delta = math_utils.quat_from_euler_xyz(rand_samples[:, 3], rand_samples[:, 4], rand_samples[:, 5])
    orientations = math_utils.quat_mul(root_states[:, 3:7], orientations_delta)
    range_list = [velocity_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
    ranges = torch.tensor(range_list, device=object.device)
    rand_samples = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=object.device)
    velocities = root_states[:, 7:13] + rand_samples
    object.write_root_pose_to_sim(torch.cat([positions, orientations], dim=-1), env_ids=env_ids)
    object.write_root_velocity_to_sim(velocities, env_ids=env_ids)
```

The C2 reset-symmetry logic (`BaseEnv._reset_idx` / `_reset_symmetry`) mirrors object poses & commands through `rep_Rd` / `rep_SO3_flat` for the symmetric-env subset; gated by `cfg.hydra_cfg.task.symmetry.symmetric_envs`.

---

## §4 Goal + Termination

**Description.** No task-specific failure termination — `PickObjectTerminationsCfg` is empty, inheriting only `time_out` from `BaseTerminationsCfg`. The **goal** is the ordered "both cubes in the tote" predicate, computed entirely in Python trackers (NOT via the command manager's success metric):

1. Each step, `PickObjectEnv.step()` marks a cube "on the tote" if it is within 0.08 m of `target_pos` (the tote-top command) and increments `object_on_tote_tracker`.
2. The reward function `pick.if_in_tote` (§6) increments `object_in_tote_tracker[object_id]` when the fingertips have *released* the cube AND it is within `distance_threshold` of the tote AND its z < 0.2 AND it was previously on-tote. Object_2 only counts once `object_in_tote_tracker[1] >= 3` (ordering).
3. Success = `object_in_tote_tracker[1] >= 3 AND object_in_tote_tracker[2] >= 3`. While true, `success_step_tracker` increments; `pick.success_bonus(num_success=5)` sets `env.success_tracker` and fires the reward once the streak reaches 5.

Two command terms drive the goal geometry (`PickObjectCommandsCfg`): `target_pos` (z_height command = object_0/tote pose + 0.3 m up, the drop target) and `waiting_pos` (a fixed offset hold pose at (0, +0.2, +0.3) relative to the tote, used to make the left arm *wait* until object_1 is placed).

**Decisions resolved**

| Question | Value |
|---|---|
| terminations | only `time_out` (base), episode 8.3333 s |
| failure terminations | none |
| success predicate | `object_in_tote_tracker[1]>=3 AND [2]>=3`, held `success_step_tracker>=5` |
| "in tote" gate | released contact + dist<threshold + z<0.2 + was on-tote; obj2 gated on obj1 done |
| "on tote" gate | dist(cube, target_pos+origin) < 0.08 m |
| target_pos command | z_height=0.3 above tote (use_initial_pose=True), success_threshold 0.05 |
| waiting_pos command | pose_range x=0,y=0.2,z=0.3, offset=True, return_type "pos", use_initial_pose=True |
| success_bonus num_success | 5 |

**Code — commands (`env_cfg.py`):**

```python
@configclass
class PickObjectCommandsCfg(BaseCommandsCfg):
    target_pos = TargetPositionCommandCfg(
        object_id=0, success_threshold=0.05, z_height=0.3, debug_vis=True, use_initial_pose=True,
    )
    waiting_pos = TargetPositionCommandCfg(
        object_id=0, success_threshold=0.05, success_threshold_orient=1.0,
        pose_range={"x": [0.0, 0.0], "y": [0.2, 0.2], "z": [0.3, 0.3]},
        return_type="pos", debug_vis=True, offset=True, use_initial_pose=True,
    )
```

**Code — command cfg + generator (`grasp_command_cfg.py`, `grasp_command.py`):**

```python
@configclass
class TargetPositionCommandCfg(CommandTermCfg):
    class_type: type = TargetPositionCommand
    resampling_time_range: tuple[float, float] = (1e6, 1e6)  # no resampling based on time
    object_id = 0
    pose_range = None
    z_height = None
    success_threshold: float = MISSING
    success_threshold_orient: float = MISSING
    update_goal_on_success: bool = False
    visualizer_cfg: VisualizationMarkersCfg = VisualizationMarkersCfg(
        prim_path=f"/Visuals/Command/goal_marker_{object_id}",
        markers={"goal": sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/frame_prim.usd", scale=(0.1, 0.1, 0.1))},
    )
    return_type: str = None
    offset: bool = False
    use_initial_pose: bool = False
```

```python
class TargetPositionCommand(CommandTerm):
    @property
    def command(self) -> torch.Tensor:
        if self.ranges is not None:
            if self.return_type == "pos":
                return self.pos_command_e
            elif self.return_type == "quat":
                return self.quat_command_w
            else:
                return torch.cat((self.pos_command_e, self.quat_command_w), dim=-1)
        else:
            return self.pos_command_e

    def _resample_command(self, env_ids):
        if self.ranges is not None:
            rand_samples = math_utils.sample_uniform(self.ranges[:, 0], self.ranges[:, 1], (len(env_ids), 6), device=self.device)
            if self.cfg.offset:
                if isinstance(self.cfg.object_id, str) or self.cfg.use_initial_pose:
                    rand_samples[:, 0:3] += self.object.data.root_pos_w[env_ids] - self._env.scene.env_origins[env_ids]
                else:
                    rand_samples[:, 0:3] += self.env.object_init_pos[self.cfg.object_id, env_ids] - self._env.scene.env_origins[env_ids]
            self.pos_command_e[env_ids] = rand_samples[:, 0:3]
            self.pos_command_w[env_ids] = self.pos_command_e[env_ids] + self._env.scene.env_origins[env_ids]
            orientations_delta = math_utils.quat_from_euler_xyz(rand_samples[:, 3], rand_samples[:, 4], rand_samples[:, 5])
            self.quat_command_w[env_ids] = math_utils.quat_mul(self.object.data.default_root_state[env_ids, 3:7], orientations_delta)
        elif self.z_height is not None:
            if self.cfg.use_initial_pose:
                self.pos_command_w[env_ids] = self.object.data.root_pos_w[env_ids]
            else:
                self.pos_command_w[env_ids] = self.env.object_init_pos[self.cfg.object_id, env_ids]
            self.pos_command_w[env_ids, 2] += self.z_height
            self.pos_command_e[env_ids] = self.pos_command_w[env_ids] - self._env.scene.env_origins[env_ids]
```

**Code — success bonus + `generated_commands` router (`PickObject/mdps.py`):**

```python
def success_bonus(env, num_success: int = 10) -> torch.Tensor:
    success = env.success_step_tracker >= num_success
    env.success_tracker = success.float()
    return success

def generated_commands(env, object_id: int = 0):
    waiting_pos = env.command_manager.get_command("waiting_pos").clone()
    target_pos = env.command_manager.get_command("target_pos").clone()
    if not hasattr(env, "object_in_tote_tracker"):
        return torch.zeros((env.num_envs, 3), device=env.device)
    if object_id == 1:
        tote_not_in = torch.where(env.object_in_tote_tracker[2] < 3)[0]
    elif object_id == 2:
        tote_not_in = torch.where(env.object_in_tote_tracker[1] < 3)[0]
    elif object_id == 0:
        return target_pos
    target_pos[tote_not_in] = waiting_pos[tote_not_in]
    return target_pos
```

```python
@configclass
class PickObjectTerminationsCfg(BaseTerminationsCfg):
    pass
# BaseTerminationsCfg: time_out = DoneTerm(func=mdp.time_out, time_out=True)
```

---

## §5 Observation

**Description.** Single `policy` group, `enable_corruption=False`, `concatenate_terms=True`. Twelve terms in fixed order interleaving right-arm, left-arm, object, and command observations. `ee_pose` uses `symmetry=True` (default) → position (3) + row-flattened rotation matrix (9) = 12 dims. `joint_pos_limit_normalized` and `joint_vel` cover all 22 joints per arm (joints=None → all `asset_cfg.joint_ids`). `pick.generated_commands` returns a 3-vector (routed target/waiting position).

**Observation dims (bimanual):**

| Term | func | dims |
|---|---|---|
| ee_pose_right | `ee_pose(palm_link)` | 12 |
| joint_pos_right | `joint_pos_limit_normalized` (22) | 22 |
| joint_vel_right | `joint_vel` (22) | 22 |
| object_pos_1 | `object_pos(id=1)` | 3 |
| ee_pose_left | `ee_pose(palm_link, robot_left)` | 12 |
| joint_pos_left | `joint_pos_limit_normalized` (22, left) | 22 |
| joint_vel_left | `joint_vel` (22, left) | 22 |
| object_pos_2 | `object_pos(id=2)` | 3 |
| tote_pos | `object_pos(id=0)` | 3 |
| last_action | `last_action` | 44 |
| waiting_pos | `pick.generated_commands(id=2)` | 3 |
| target_pos | `pick.generated_commands(id=0)` | 3 |
| **Total** | | **171** |

**Corruption:** `joint_pos_*` carry `Gnoise(std=0.005)`; `object_pos_1`/`object_pos_2` carry `Unoise(0.0, 0.015)`. But group `enable_corruption=False`, so noise is NOT applied (IsaacLab only applies term noise when the group flag is on).

**Code — `PolicyCfg` (`env_cfg.py`):**

```python
    @configclass
    class PolicyCfg(ObsGroup):
        ee_pose_right = ObsTerm(func=ee_pose, params={"ee_name": "palm_link"})
        joint_pos_right = ObsTerm(func=joint_pos_limit_normalized, params={"joints": None,
                                    "joint_lower_limit": JOINT_LOWER_LIMIT,
                                    "joint_upper_limit": JOINT_UPPER_LIMIT,}, noise=Gnoise(std=0.005))
        joint_vel_right = ObsTerm(func=joint_vel, params={"joints": None},)
        object_pos_1 = ObsTerm(func=object_pos, params={"object_id": 1}, noise=Unoise(n_min=0.0, n_max=0.015))
        ee_pose_left = ObsTerm(func=ee_pose, params={"ee_name": "palm_link", "asset_cfg": SceneEntityCfg("robot_left")})
        joint_pos_left = ObsTerm(func=joint_pos_limit_normalized, params={"joints": None,
                                    "joint_lower_limit": JOINT_LOWER_LIMIT_LEFT,
                                    "joint_upper_limit": JOINT_UPPER_LIMIT_LEFT,
                                    "asset_cfg": SceneEntityCfg("robot_left")}, noise=Gnoise(std=0.005))
        joint_vel_left = ObsTerm(func=joint_vel, params={"joints": None, "asset_cfg": SceneEntityCfg("robot_left")},)
        object_pos_2 = ObsTerm(func=object_pos, params={"object_id": 2}, noise=Unoise(n_min=0.0, n_max=0.015))
        tote_pos = ObsTerm(func=object_pos, params={"object_id": 0})
        last_action = ObsTerm(func=last_action)
        waiting_pos = ObsTerm(func=pick.generated_commands, params={"object_id": 2})
        target_pos = ObsTerm(func=pick.generated_commands, params={"object_id": 0})

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True
```

**Code — obs functions (`obs_mdps.py`):**

```python
def joint_pos_limit_normalized(env, asset_cfg=SceneEntityCfg("robot"), joints=None, joint_lower_limit=None, joint_upper_limit=None):
    asset = env.scene[asset_cfg.name]
    if joints is None:
        joint_ids = asset_cfg.joint_ids
    else:
        joint_ids = asset.find_joints(joints)[0]
    if joint_lower_limit is None:
        joint_lower_limit = asset.data.soft_joint_pos_limits[:, joint_ids, 0]
    else:
        joint_lower_limit = torch.tensor(joint_lower_limit, device=env.device)
    if joint_upper_limit is None:
        joint_upper_limit = asset.data.soft_joint_pos_limits[:, joint_ids, 1]
    else:
        joint_upper_limit = torch.tensor(joint_upper_limit, device=env.device)
    assert len(joint_lower_limit) == len(joint_upper_limit)
    return math_utils.scale_transform(asset.data.joint_pos[:, joint_ids], joint_lower_limit, joint_upper_limit)

def joint_vel(env, asset_cfg=SceneEntityCfg("robot"), joints=None):
    asset = env.scene[asset_cfg.name]
    joint_ids = asset_cfg.joint_ids if joints is None else asset.find_joints(joints)[0]
    return asset.data.joint_vel[:, joint_ids]

def ee_pose(env, ee_name, asset_cfg=SceneEntityCfg("robot"), symmetry=True):
    robot = env.scene[asset_cfg.name]
    ee_idx = robot.find_bodies(ee_name)[0]
    ee_w = robot.data.body_state_w[:, ee_idx, :7].clone().reshape(-1, 7)
    ee_w[:, :3] = ee_w[:, :3] - env.scene.env_origins
    if symmetry:
        ee_R = math_utils.matrix_from_quat(ee_w[:, 3:7])
        ee_R_flat = ee_R.transpose(1, 2).reshape(-1, 9)
        ee_w = torch.cat([ee_w[:, :3], ee_R_flat], dim=-1)
    return ee_w

def object_pos(env, object_id: int = 0):
    object = env.scene[f"object_{object_id}"]
    return object.data.root_pos_w - env.scene.env_origins

def last_action(env):
    if hasattr(env, "last_action"):
        return env.last_action
    else:
        return torch.zeros((env.num_envs, env.action_dim), device=env.device)
```

---

## §6 Reward

**Description.** `PickObjectRewardsCfg` extends `BaseRewardsCfg`. It defines a *large* bank of shaping terms covering reach / lift / goal-tracking / in-tote / arm-return / energy / collision for both arms AND their C2-symmetric mirrors, plus a `success_bonus` (weight 1.0). Crucially, `pick.if_in_tote` is what mutates `object_in_tote_tracker` (the success driver) — it returns a reward value but its side-effect on the tracker is the load-bearing part that advances the goal predicate each step.

**Full RewTerm table** (name → func, weight, key params):

| RewTerm | func | weight | key params |
|---|---|---|---|
| reaching_object | `object_robot_distance` | 0.01 | weight=[1,1,1,1.5], links if5/mf5/pf5/th5, obj 1 |
| object_lifting | `lift_distance` | 1.0 | cmd target_pos, obj 1, sensors _0.._3 |
| object_goal_tracking | `pick.object_goal_distance` | 10.0 | cmd target_pos, obj 1, sensors _0.._3 |
| object_1_in_tote | `pick.if_in_tote` | 2000.0 | obj 1, sensors _0.._3, dist_thr 0.15 |
| reset_robot_joint_pos | `pick.robot_goal_distance` | 200.0 | obj 1, target [0.0462,-0.3045,0.4468], palm_link |
| reaching_object_left | `object_robot_distance` | 0.01 | obj 2, robot_left |
| object_lifting_left | `lift_distance` | 1.0 | cmd target_pos, obj 2, sensors _left |
| object_goal_tracking_left | `pick.object_goal_distance` | 10.0 | cmd waiting_pos, obj 2, delay=False, switch=True |
| object_goal_tracking_left_delay | `pick.object_goal_distance` | 50.0 | cmd target_pos, obj 2, delay=True, dist_thr 0.2 |
| object_2_in_tote | `pick.if_in_tote` | 2000.0 | obj 2, sensors _left, dist_thr 0.2, delay=True |
| reset_robot_joint_pos_left | `pick.robot_goal_distance` | 200.0 | obj 2, target [0.0462,0.3045,0.4468], robot_left |
| **success_bonus** | `pick.success_bonus` | **1.0** | num_success=5 |
| reaching_object_symmetry | `object_robot_distance` | 0.01 | obj 1, robot_left |
| object_lifting_symmetry | `lift_distance` | 1.0 | obj 1, sensors _symmetry |
| object_goal_tracking_symmetry | `pick.object_goal_distance` | 10.0 | obj 1, sensors _symmetry |
| object_1_in_tote_symmetry | `pick.if_in_tote` | 2000.0 | obj 1, sensors _symmetry, dist_thr 0.15, symmetry=True |
| reset_robot_joint_pos_symmetry | `pick.robot_goal_distance` | 200.0 | obj 1, target [0.0462,0.3045,0.4468], robot_left |
| reaching_object_left_symmetry | `object_robot_distance` | 0.01 | obj 2, robot (right) |
| object_lifting_left_symmetry | `lift_distance` | 1.0 | obj 2, sensors _left_symmetry |
| object_goal_tracking_left_symmetry | `pick.object_goal_distance` | 10.0 | obj 2, waiting_pos, _left_symmetry, switch=True |
| object_goal_tracking_left_delay_symmetry | `pick.object_goal_distance` | 50.0 | obj 2, target_pos, _left_symmetry, delay=True |
| object_2_in_tote_symmetry | `pick.if_in_tote` | 2000.0 | obj 2, _left_symmetry, dist_thr 0.2, delay=True, symmetry=True |
| reset_robot_joint_pos_left_symmetry | `pick.robot_goal_distance` | 200.0 | obj 2, target [0.0462,-0.3045,0.4468], robot |
| energy | `energy_punishment` | 0.000001 | robot, allegro actuators |
| energy_left | `energy_punishment` | 0.000001 | robot_left, allegro actuators |
| collision_to_table | `collision_penalty` | -0.000001 | sensors _0.._3 |
| collision_to_table_symmetry | `collision_penalty` | -0.000001 | sensors _symmetry |
| collision_to_table_left | `collision_penalty` | -0.000001 | sensors _left |
| collision_to_table_left_symmetry | `collision_penalty` | -0.000001 | sensors _left_symmetry |

### Symmetric-learning reward terms (drop if not using symmetric learning)

This task trains with **SYMMETRIC LEARNING** (`base.yaml` `symmetry.symmetric_envs: True`, C2 reflection group). The reward terms come in two layers:

- **BASE set** — the right-arm terms plus their `_left` counterparts. **Both are genuine** and required for the bimanual two-arm task (right arm handles object_1, left arm handles object_2). Keep these regardless of symmetric learning.
- **`_symmetry`-suffixed DUPLICATES** — these exist **ONLY** for the symmetric-learning augmentation (they re-filter the same R/L hand↔cube signals under the mirrored `symmetry_tracker == 1` branch). If you are **NOT** using symmetric learning, **DROP every `_symmetry`-suffixed term** and reproduce the reward with only the base right + left terms and their weights above.

**`_symmetry` terms to drop when not using symmetric learning (13):** `reaching_object_symmetry`, `object_lifting_symmetry`, `object_goal_tracking_symmetry`, `object_1_in_tote_symmetry`, `reset_robot_joint_pos_symmetry`, `reaching_object_left_symmetry`, `object_lifting_left_symmetry`, `object_goal_tracking_left_symmetry`, `object_goal_tracking_left_delay_symmetry`, `object_2_in_tote_symmetry`, `reset_robot_joint_pos_left_symmetry`, `collision_to_table_symmetry`, `collision_to_table_left_symmetry`.

**Code — `PickObjectRewardsCfg` non-symmetry block (`env_cfg.py`), verbatim:**

```python
@configclass
class PickObjectRewardsCfg(BaseRewardsCfg):
    reaching_object = RewTerm(func=object_robot_distance,
        params={"weight": [1.0, 1.0, 1.0, 1.5], "link_name": ["if5", "mf5", "pf5", "th5"], "object_id": 1}, weight=0.01)
    object_lifting = RewTerm(func=lift_distance,
        params={"command_name": "target_pos", "object_id": 1, "sensor_names": ["contact_sensors_0", "contact_sensors_1", "contact_sensors_2", "contact_sensors_3"]}, weight=1.0)
    object_goal_tracking = RewTerm(func=pick.object_goal_distance,
        params={"command_name": "target_pos", "object_id": 1, "sensor_names": ["contact_sensors_0", "contact_sensors_1", "contact_sensors_2", "contact_sensors_3"],}, weight=10.0)
    object_1_in_tote = RewTerm(func=pick.if_in_tote,
        params={"object_id": 1, "sensor_names": ["contact_sensors_0", "contact_sensors_1", "contact_sensors_2", "contact_sensors_3"], "distance_threshold": 0.15}, weight=2000.0)
    reset_robot_joint_pos = RewTerm(func=pick.robot_goal_distance,
        params={"object_id": 1, "target_pos": [0.0462, -0.3045, 0.4468], "target_link": "palm_link"}, weight=200.0)
    reaching_object_left = RewTerm(func=object_robot_distance,
        params={"weight": [1.0, 1.0, 1.0, 1.5], "link_name": ["if5", "mf5", "pf5", "th5"], "object_id": 2, "asset_cfg": SceneEntityCfg("robot_left")}, weight=0.01)
    object_lifting_left = RewTerm(func=lift_distance,
        params={"command_name": "target_pos", "object_id": 2, "sensor_names": ["contact_sensors_0_left", "contact_sensors_1_left", "contact_sensors_2_left", "contact_sensors_3_left"]}, weight=1.0)
    object_goal_tracking_left = RewTerm(func=pick.object_goal_distance,
        params={"command_name": "waiting_pos", "object_id": 2, "sensor_names": ["contact_sensors_0_left", "contact_sensors_1_left", "contact_sensors_2_left", "contact_sensors_3_left"], "delay": False, "switch": True}, weight=10.0)
    object_goal_tracking_left_delay = RewTerm(func=pick.object_goal_distance,
        params={"command_name": "target_pos", "object_id": 2, "sensor_names": ["contact_sensors_0_left", "contact_sensors_1_left", "contact_sensors_2_left", "contact_sensors_3_left"], "delay": True, "switch": False, "distance_threshold": 0.2}, weight=50.0)
    object_2_in_tote = RewTerm(func=pick.if_in_tote,
        params={"object_id": 2, "sensor_names": ["contact_sensors_0_left", "contact_sensors_1_left", "contact_sensors_2_left", "contact_sensors_3_left"], "distance_threshold": 0.2, "delay": True}, weight=2000.0)
    reset_robot_joint_pos_left = RewTerm(func=pick.robot_goal_distance,
        params={"object_id": 2, "target_pos": [0.0462, 0.3045, 0.4468], "target_link": "palm_link", "asset_cfg": SceneEntityCfg("robot_left")}, weight=200.0)
    success_bonus = RewTerm(func=pick.success_bonus, params={"num_success": 5}, weight=1.0)
    # ... + 12 "_symmetry" mirror terms that swap R/L hand↔cube filtering, each with the same weight as its base counterpart ...
    energy = RewTerm(func=energy_punishment, weight=0.000001,
        params={"asset_cfg": SceneEntityCfg("robot"), "actuator_name": ["allegro_hand_1","allegro_hand_2","allegro_hand_3","allegro_hand_4","allegro_hand_thumb_1","allegro_hand_thumb_2","allegro_hand_thumb_3","allegro_hand_thumb_4"]})
    energy_left = RewTerm(func=energy_punishment, weight=0.000001,
        params={"asset_cfg": SceneEntityCfg("robot_left"), "actuator_name": [...same allegro list...]})
    collision_to_table = RewTerm(func=collision_penalty, params={"sensor_names": ["contact_sensors_0","contact_sensors_1","contact_sensors_2","contact_sensors_3"]}, weight=-0.000001)
    collision_to_table_symmetry = RewTerm(func=collision_penalty, params={"sensor_names": ["contact_sensors_0_symmetry",...]}, weight=-0.000001)
    collision_to_table_left = RewTerm(func=collision_penalty, params={"sensor_names": ["contact_sensors_0_left",...]}, weight=-0.000001)
    collision_to_table_left_symmetry = RewTerm(func=collision_penalty, params={"sensor_names": ["contact_sensors_0_left_symmetry",...]}, weight=-0.000001)
```

**Code — task-local reward funcs (`PickObject/mdps.py`), verbatim:**

```python
def object_goal_distance(env, object_id=0, command_name="target_pos",
        sensor_names=["contact_sensors_0","contact_sensors_1","contact_sensors_2","contact_sensors_3"],
        delay=False, switch=False, distance_threshold=0.3):
    object = env.scene[f"object_{object_id}"]
    if switch:
        assert object_id == 2
        des_pos_w = generated_commands(env, object_id) + env.scene.env_origins
    else:
        command = env.command_manager.get_command(command_name)
        des_pos_w = command[:, :3] + env.scene.env_origins
    distance = torch.norm(des_pos_w - object.data.root_pos_w[:, :3], dim=1)
    distance = torch.clamp(distance_threshold - distance, min=0.0)
    rew = distance * get_allegro_contact(env, sensor_names) * (object.data.root_pos_w[:, 2] > (des_pos_w[:, 2] - 0.05))
    if delay:
        assert object_id == 2
        if object_id == 1:
            rew = rew * (env.object_in_tote_tracker[2] >= 3)
        elif object_id == 2:
            rew = rew * (env.object_in_tote_tracker[1] >= 3)
    return rew

def if_in_tote(env, object_id=0, distance_threshold=0.2, delay=False, symmetry=False,
        sensor_names=["contact_sensors_0","contact_sensors_1","contact_sensors_2","contact_sensors_3"]):
    object = env.scene[f"object_{object_id}"]
    distance = torch.norm(object.data.root_pos_w[:, :3] - env.scene["object_0"].data.root_pos_w[:, :3], dim=-1)
    in_tote = check_release(env, sensor_names) * (distance < distance_threshold) * (object.data.root_pos_w[:, 2] < 0.2) * (env.object_on_tote_tracker[object_id] > 0)
    if symmetry:
        in_tote = in_tote * (env.symmetry_tracker == 1)
    else:
        in_tote = in_tote * (env.symmetry_tracker == 0)
    if delay:
        assert object_id == 2
    if object_id == 1:
        env.object_in_tote_tracker[object_id] += in_tote
        rew = env.object_in_tote_tracker[object_id] == 3
    elif object_id == 2:
        env.object_in_tote_tracker[object_id] += in_tote * (env.object_in_tote_tracker[1] >= 3)
        rew = (env.object_in_tote_tracker[object_id] == 3) * (env.object_in_tote_tracker[1] >= 3)
    return rew.float()

def robot_goal_distance(env, target_pos, target_link, object_id=0, asset_cfg=SceneEntityCfg("robot")):
    des_pos_w = torch.tensor(target_pos, device=env.device) + env.scene.env_origins
    target_link_idx = env.scene[asset_cfg.name].find_bodies([target_link])[0]
    distance = torch.norm(des_pos_w - env.scene[asset_cfg.name].data.body_state_w[:, target_link_idx, 0:3].squeeze(1), dim=1)
    distance = torch.clamp(0.25 - distance, min=0.0)
    rew = distance * (env.object_in_tote_tracker[object_id] >= 3)
    return rew

def success_bonus(env, num_success=10):
    success = env.success_step_tracker >= num_success
    env.success_tracker = success.float()
    return success
```

**Code — shared reward funcs (`reward_mdps.py`), verbatim:**

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
    rew = 1 / object_link_distance
    return rew

def get_allegro_contact(env, sensor_names):
    force = get_force(env, sensor_names, if_filter=True)
    is_contact = (torch.norm(force, dim=-1) > 1.0)
    is_contact_index_or_middle_or_ring = reduce(torch.logical_or, [is_contact[:, 0], is_contact[:, 1], is_contact[:, 2]])
    is_contact = reduce(torch.logical_and, [is_contact_index_or_middle_or_ring, is_contact[:, 3]])
    return is_contact

def check_release(env, sensor_names):
    force = get_force(env, sensor_names, if_filter=True)
    is_contact = (torch.norm(force, dim=-1) < 1.0)
    is_release = reduce(torch.logical_and, [is_contact[:, 0], is_contact[:, 1], is_contact[:, 2], is_contact[:, 3]])
    return is_release

def energy_punishment(env, actuator_name=None, asset_cfg=SceneEntityCfg("robot")):
    if actuator_name is None:
        energy = get_energy_consumption(env=env, robot_name=asset_cfg.name)
    else:
        energy = get_actuator_energy_consumption(env=env, robot_name=asset_cfg.name, actuator_name=actuator_name)
    rew = -energy
    return rew

def collision_penalty(env, sensor_names=["contact_sensors_0","contact_sensors_1","contact_sensors_2","contact_sensors_3"]):
    filtered_is_contact = []
    for s in sensor_names:
        filtered_force = env.scene[s].data.force_matrix_w.mean(dim=tuple(range(1, env.scene[s].data.force_matrix_w.ndim))) == 0.0
        normal_force = env.scene[s].data.net_forces_w.mean(dim=tuple(range(1, env.scene[s].data.net_forces_w.ndim))) != 0.0
        filtered_is_contact.append(torch.logical_and(filtered_force, normal_force))
    is_contact = reduce(torch.logical_or, filtered_is_contact)
    return is_contact.float()

def lift_distance(env, command_name, minimal_height=None, object_id=0,
        sensor_names=["contact_sensors_0","contact_sensors_1","contact_sensors_2","contact_sensors_3"]):
    command = env.command_manager.get_command(command_name)
    des_pos_w = command[:, :3] + env.scene.env_origins
    if minimal_height is None:
        minimal_height = des_pos_w[:, 2] + 0.05
    object = env.scene[f"object_{object_id}"]
    z_distance = (object.data.root_pos_w[:, 2] - env.object_init_pos[object_id][:, 2]) / (minimal_height - env.object_init_pos[object_id][:, 2])
    z_distance = torch.clamp(z_distance, min=0.0)
    rew = (object.data.root_pos_w[:, 2] < minimal_height) * z_distance * get_allegro_contact(env, sensor_names)
    return rew
```

*(`get_force`, `get_energy_consumption`, `get_actuator_energy_consumption` helpers omitted for brevity — they compute filtered contact forces and Σ|joint_vel·applied_torque| over the named allegro actuators.)*

**Reward composition note.** IsaacLab's `RewardManager` sums `weight * term_dt_scaled` over all terms each step. Beyond its reward value, `if_in_tote` also mutates `object_in_tote_tracker` as a side effect, so it is load-bearing for the goal predicate. `success_bonus` returns a **bool** (0/1); multiplied by dt inside the manager it yields a small positive per-step reward while the ≥5-step success streak holds.

---

## §7 DR

**No DR EventTerms are wired into the task cfg.** `PickObjectEventCfg` contains only `reset` events (§3). There are no `mode="startup"`/randomization terms in the observation/event configs beyond the (disabled) obs noise in §5.

However, the shared `BaseEnv` exposes an **external, curriculum-gated randomization hook** (`update_randomization`, `manager_based_env.py`) driven by `DomainRandomizer(cfg.hydra_cfg.task.randomize)`. It is invoked from the training loop (not from the env cfg) with a running success_rate and can, when the corresponding hydra keys are set, apply: object mass (`randomize_mass`), material friction/restitution (`randomize_material`), arm action_scale (`self._scale[:6]`), reward-weight curricula for `energy_penalty` / `collision_penalty` (`randomize_rew_weight`), external force/torque (`randomize_external_force_torque`), and reset-pose ranges (`randomize_reset_pose`). None of this is active unless the external hydra `task.randomize` config enables it, so for a standalone reproduction of the registered env: **DR is off by default.**

```python
    def update_randomization(self, success_rate):
        self.domain_randomizer.update(success_rate)
        randomized_values, randomization_state, curriculum_state = self.domain_randomizer.sample()
        if "object_mass" in randomized_values:
            randomize_mass(self, randomized_values["object_mass"])
        if "static_friction" in randomized_values:
            randomize_material(self, static_friction=..., dynamic_friction=..., restitution=..., num_buckets=250)
        if "action_scale" in randomized_values:
            self._scale[:6] = randomized_values["action_scale"]
        if "energy_penalty" in randomized_values:
            for rew_name in curriculum_state["energy_penalty"]["names"]:
                randomize_rew_weight(self, rew_name, randomized_values["energy_penalty"])
        if "collision_penalty" in randomized_values:
            for rew_name in curriculum_state["collision_penalty"]["names"]:
                randomize_rew_weight(self, rew_name, randomized_values["collision_penalty"])
        if "external_force_torque" in randomized_values:
            randomize_external_force_torque(self, force=..., torque=...)
        for rand_name in randomized_values.keys():
            if "reset_pose" in rand_name:
                randomize_reset_pose(self, curriculum_state[rand_name]["names"], randomized_values[rand_name])
        return randomization_state, curriculum_state, self.domain_randomizer.best_so_far
```
