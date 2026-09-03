# Isaac-Dexsuite-Kuka-Allegro-Reorient-v0 — Implementation Spec

- robot: Kuka LBR iiwa7 arm + Allegro hand (23 DoF)
- simulator: IsaacLab (Isaac Sim, manager-based)
- objects: randomly-shaped rigid object, table
- bimanual: false
- summary: Reorient a randomly shaped object to a commanded 6-DoF goal pose.

> **Build caveat (read first).** This task is an UPSTREAM IsaacLab manager-based env registered inside `isaaclab_tasks` (NOT in `harbor/benchmark-spec.json`). `gym.make` requires `import isaaclab_tasks` first, which in turn imports `isaaclab.envs.mdp` → `isaaclab.utils.mesh` → `from pxr import Usd` and fails because the host `.venv` does not ship the Omniverse `pxr` package. The whole spec below is from a verbatim source read; the §1 build smoke and obs/action spaces were NOT executed. Resolved dims are analytic (see §2 / §5).

This is a **Kuka LBR iiwa7 arm + Allegro 16-DoF hand** (23 actuated joints total) reorienting a randomly-shaped rigid object to a commanded 6-DoF goal pose expressed in the robot base frame. Goal pose visualized by recoloring an (invisible) table prim red (failure) / green (success). Heavy domain randomization + an ADR (adaptive domain randomization) gravity/noise curriculum.

---

## §1 Registration + Scene

### Description
Registers `Isaac-Dexsuite-Kuka-Allegro-Reorient-v0` against `isaaclab.envs:ManagerBasedRLEnv` with `env_cfg_entry_point = DexsuiteKukaAllegroReorientEnvCfg`. The scene holds the MISSING robot (filled by the kuka_allegro mixin with `KUKA_ALLEGRO_CFG`), a `MultiAssetSpawnerCfg` object (one of 16 primitive shapes sampled per env), a kinematic invisible table (reused as a success-color visualizer), a ground plane, and a dome sky light. Contact sensors on the 4 finger tips (added in the mixin `__post_init__`) are filtered against the Object prim.

### Decisions resolved
- `entry_point = isaaclab.envs:ManagerBasedRLEnv`, `disable_env_checker=True`.
- `num_envs = 4096`, `env_spacing = 3`, `replicate_physics = False`.
- Object: 16-asset `MultiAssetSpawnerCfg` (cuboids / spheres / capsules / cones), `mass=0.2`, `static_friction=0.5`, solver_position_iter=16, gravity enabled on object. Init pos `(-0.55, 0.1, 0.35)`.
- Table: kinematic cuboid `size=(0.8,1.5,0.04)`, `visible=False`, init pos `(-0.55,0.0,0.235)`. Reused as success/failure color marker (see §4 / `__post_init__`).
- Robot USD: `{ISAACLAB_NUCLEUS_DIR}/Robots/KukaAllegro/kuka.usd` (remote Omniverse nucleus asset — not a local file; resolved at runtime from the `ISAACLAB_NUCLEUS_DIR` env constant). `prim_path="{ENV_REGEX_NS}/Robot"`, EE link prim `ee_link`.
- Sky light: dome `intensity=750`, `texture_file={ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr` (remote).
- 4 contact sensors named `<link>_object_s` for `link ∈ {index_link_3, middle_link_3, ring_link_3, thumb_link_3}` at `{ENV_REGEX_NS}/Robot/ee_link/<link>`, filtered against `{ENV_REGEX_NS}/Object`.
- Sim: `dt=1/120`, `decimation=2` (→ 60 Hz control / "50 Hz" per comment), `render_interval=2`, `episode_length_s=4.0`, `is_finite_horizon=True`, `gpu_max_rigid_patch_count = 4*5*2**15`, `bounce_threshold_velocity=0.01`.

### Code — robot articulation (KUKA_ALLEGRO_CFG)
`source/isaaclab_assets/isaaclab_assets/robots/kuka_allegro.py`
```python
KUKA_ALLEGRO_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAACLAB_NUCLEUS_DIR}/Robots/KukaAllegro/kuka.usd",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=True,
            retain_accelerations=True,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1000.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=32,
            solver_velocity_iteration_count=1,
            sleep_threshold=0.005,
            stabilization_threshold=0.0005,
        ),
        joint_drive_props=sim_utils.JointDrivePropertiesCfg(drive_type="force"),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.0),
        rot=(1.0, 0.0, 0.0, 0.0),
        joint_pos={
            "iiwa7_joint_(1|2|7)": 0.0,
            "iiwa7_joint_3": 0.7854,
            "iiwa7_joint_4": 1.5708,
            "iiwa7_joint_(5|6)": -1.5708,
            "(index|middle|ring)_joint_0": 0.0,
            "(index|middle|ring)_joint_1": 0.3,
            "(index|middle|ring)_joint_2": 0.3,
            "(index|middle|ring)_joint_3": 0.3,
            "thumb_joint_0": 1.5,
            "thumb_joint_1": 0.60147215,
            "thumb_joint_2": 0.33795027,
            "thumb_joint_3": 0.60845138,
        },
    ),
    actuators={
        "kuka_allegro_actuators": ImplicitActuatorCfg(
            joint_names_expr=[
                "iiwa7_joint_(1|2|3|4|5|6|7)",
                "index_joint_(0|1|2|3)",
                "middle_joint_(0|1|2|3)",
                "ring_joint_(0|1|2|3)",
                "thumb_joint_(0|1|2|3)",
            ],
            effort_limit_sim={
                "iiwa7_joint_(1|2|3|4|5|6|7)": 300.0,
                "index_joint_(0|1|2|3)": 0.5,
                "middle_joint_(0|1|2|3)": 0.5,
                "ring_joint_(0|1|2|3)": 0.5,
                "thumb_joint_(0|1|2|3)": 0.5,
            },
            stiffness={
                "iiwa7_joint_(1|2|3|4)": 300.0,
                "iiwa7_joint_5": 100.0,
                "iiwa7_joint_6": 50.0,
                "iiwa7_joint_7": 25.0,
                "index_joint_(0|1|2|3)": 3.0,
                "middle_joint_(0|1|2|3)": 3.0,
                "ring_joint_(0|1|2|3)": 3.0,
                "thumb_joint_(0|1|2|3)": 3.0,
            },
            damping={
                "iiwa7_joint_(1|2|3|4)": 45.0,
                "iiwa7_joint_5": 20.0,
                "iiwa7_joint_6": 15.0,
                "iiwa7_joint_7": 15.0,
                "index_joint_(0|1|2|3)": 0.1,
                "middle_joint_(0|1|2|3)": 0.1,
                "ring_joint_(0|1|2|3)": 0.1,
                "thumb_joint_(0|1|2|3)": 0.1,
            },
            friction={
                "iiwa7_joint_(1|2|3|4|5|6|7)": 1.0,
                "index_joint_(0|1|2|3)": 0.01,
                "middle_joint_(0|1|2|3)": 0.01,
                "ring_joint_(0|1|2|3)": 0.01,
                "thumb_joint_(0|1|2|3)": 0.01,
            },
        ),
    },
    soft_joint_pos_limit_factor=1.0,
)
```

### Code — registration
`config/kuka_allegro/__init__.py`
```python
gym.register(
    id="Isaac-Dexsuite-Kuka-Allegro-Reorient-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.dexsuite_kuka_allegro_env_cfg:DexsuiteKukaAllegroReorientEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:DexsuiteKukaAllegroPPORunnerCfg",
    },
)
```

### Code — SceneCfg (base) + kuka_allegro mixin scene wiring
`dexsuite_env_cfg.py` (SceneCfg)
```python
@configclass
class SceneCfg(InteractiveSceneCfg):
    """Dexsuite Scene for multi-objects Lifting"""

    # robot
    robot: ArticulationCfg = MISSING

    # object
    object: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Object",
        spawn=sim_utils.MultiAssetSpawnerCfg(
            assets_cfg=[
                CuboidCfg(size=(0.05, 0.1, 0.1), physics_material=RigidBodyMaterialCfg(static_friction=0.5)),
                CuboidCfg(size=(0.05, 0.05, 0.1), physics_material=RigidBodyMaterialCfg(static_friction=0.5)),
                CuboidCfg(size=(0.025, 0.1, 0.1), physics_material=RigidBodyMaterialCfg(static_friction=0.5)),
                CuboidCfg(size=(0.025, 0.05, 0.1), physics_material=RigidBodyMaterialCfg(static_friction=0.5)),
                CuboidCfg(size=(0.025, 0.025, 0.1), physics_material=RigidBodyMaterialCfg(static_friction=0.5)),
                CuboidCfg(size=(0.01, 0.1, 0.1), physics_material=RigidBodyMaterialCfg(static_friction=0.5)),
                SphereCfg(radius=0.05, physics_material=RigidBodyMaterialCfg(static_friction=0.5)),
                SphereCfg(radius=0.025, physics_material=RigidBodyMaterialCfg(static_friction=0.5)),
                CapsuleCfg(radius=0.04, height=0.025, physics_material=RigidBodyMaterialCfg(static_friction=0.5)),
                CapsuleCfg(radius=0.04, height=0.01, physics_material=RigidBodyMaterialCfg(static_friction=0.5)),
                CapsuleCfg(radius=0.04, height=0.1, physics_material=RigidBodyMaterialCfg(static_friction=0.5)),
                CapsuleCfg(radius=0.025, height=0.1, physics_material=RigidBodyMaterialCfg(static_friction=0.5)),
                CapsuleCfg(radius=0.025, height=0.2, physics_material=RigidBodyMaterialCfg(static_friction=0.5)),
                CapsuleCfg(radius=0.01, height=0.2, physics_material=RigidBodyMaterialCfg(static_friction=0.5)),
                ConeCfg(radius=0.05, height=0.1, physics_material=RigidBodyMaterialCfg(static_friction=0.5)),
                ConeCfg(radius=0.025, height=0.1, physics_material=RigidBodyMaterialCfg(static_friction=0.5)),
            ],
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=0,
                disable_gravity=False,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.2),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(-0.55, 0.1, 0.35)),
    )

    # table
    table: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/table",
        spawn=sim_utils.CuboidCfg(
            size=(0.8, 1.5, 0.04),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            # trick: we let visualizer's color to show the table with success coloring
            visible=False,
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(-0.55, 0.0, 0.235), rot=(1.0, 0.0, 0.0, 0.0)),
    )

    # plane
    plane = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        init_state=AssetBaseCfg.InitialStateCfg(),
        spawn=sim_utils.GroundPlaneCfg(),
        collision_group=-1,
    )

    # lights
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )
```

`config/kuka_allegro/dexsuite_kuka_allegro_env_cfg.py` (mixin — scene + sensors)
```python
@configclass
class KukaAllegroMixinCfg:
    rewards: KukaAllegroReorientRewardCfg = KukaAllegroReorientRewardCfg()
    actions: KukaAllegroRelJointPosActionCfg = KukaAllegroRelJointPosActionCfg()

    def __post_init__(self: dexsuite.DexsuiteReorientEnvCfg):
        super().__post_init__()
        self.commands.object_pose.body_name = "palm_link"
        self.scene.robot = KUKA_ALLEGRO_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        finger_tip_body_list = ["index_link_3", "middle_link_3", "ring_link_3", "thumb_link_3"]
        for link_name in finger_tip_body_list:
            setattr(
                self.scene,
                f"{link_name}_object_s",
                ContactSensorCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/ee_link/" + link_name,
                    filter_prim_paths_expr=["{ENV_REGEX_NS}/Object"],
                ),
            )
        self.observations.proprio.contact = ObsTerm(
            func=mdp.fingers_contact_force_b,
            params={"contact_sensor_names": [f"{link}_object_s" for link in finger_tip_body_list]},
            clip=(-20.0, 20.0),  # contact force in finger tips is under 20N normally
        )
        self.observations.proprio.hand_tips_state_b.params["body_asset_cfg"].body_names = ["palm_link", ".*_tip"]
        self.rewards.fingers_to_object.params["asset_cfg"] = SceneEntityCfg("robot", body_names=["palm_link", ".*_tip"])
```

### Smoke
```bash
cd <IsaacLab-repo>
.venv/bin/python -c "import gymnasium as gym; env = gym.make('Isaac-Dexsuite-Kuka-Allegro-Reorient-v0'); print(env.observation_space, env.action_space); env.close()"
```
Expected stdout: **NOT CAPTURED** — see build caveat. WARN: requires booting Isaac Sim (`pxr`) which is unavailable on this host. Downstream reproduction must run inside an Isaac Sim runtime.

---

## §2 Actions

### Description
Single action term: **relative joint position** control over **all** 23 actuated joints (7 arm + 16 hand), scale `0.1`. Action = scaled delta added to current joint position targets each control step (the base IsaacLab `RelativeJointPositionAction`). No separate gripper term — the hand joints are part of the same articulation and same action term.

### Decisions resolved
- `action.scale = 0.1`, `joint_names=[".*"]` → 23 joints.
- **action dim = 23** (iiwa7_joint_1..7 = 7; index/middle/ring/thumb _joint_0..3 = 16).
- Action space (analytic): `Box(-inf, inf, (23,), float32)` per the IsaacLab joint-action convention (unbounded; clamped downstream by reward `action_l2`/`action_rate_l2` and joint limits).

### Code
`config/kuka_allegro/dexsuite_kuka_allegro_env_cfg.py`
```python
@configclass
class KukaAllegroRelJointPosActionCfg:
    action = mdp.RelativeJointPositionActionCfg(asset_name="robot", joint_names=[".*"], scale=0.1)
```
Base `ActionsCfg` in `dexsuite_env_cfg.py` is empty (`pass`); the mixin supplies the real action cfg. `mdp.RelativeJointPositionActionCfg` is the stock `isaaclab.envs.mdp` action (star-imported in `mdp/__init__.py`).

### Smoke
Covered by the §1 build smoke (`env.action_space` shape == `(23,)`). NOT CAPTURED on this host.

---

## §3 Reset

### Description
Per-episode resets (`mode="reset"` EventTerms): table jittered slightly in-plane; object dropped at a uniformly-random pose (large XY/Z and full SO(3) orientation range); robot root fixed (no randomization of base); robot joints offset uniformly ±0.5 rad with the wrist joint (`iiwa7_joint_7`) given a much larger ±3 rad offset; gravity reset to the ADR-scheduled value (starts at zero — see §7 curriculum).

### Decisions resolved
- `reset_table`: pose_range x/y ∈ [-0.05,0.05], z=0; zero velocity.
- `reset_object`: pose_range x/y ∈ [-0.2,0.2], z ∈ [0.0,0.4], roll/pitch/yaw ∈ [-3.14,3.14]; zero velocity.
- `reset_root` (robot base): all zero (no base randomization).
- `reset_robot_joints`: all joints offset ∈ [-0.50, 0.50] rad, zero vel.
- `reset_robot_wrist_joint`: `iiwa7_joint_7` offset ∈ [-3, 3] rad, zero vel.
- `variable_gravity`: reset gravity to ADR-scheduled abs value (init `[0,0,0]`).

### Code
`dexsuite_env_cfg.py` (EventCfg, `mode="reset"` terms)
```python
    reset_table = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": [-0.05, 0.05], "y": [-0.05, 0.05], "z": [0.0, 0.0]},
            "velocity_range": {"x": [-0.0, 0.0], "y": [-0.0, 0.0], "z": [-0.0, 0.0]},
            "asset_cfg": SceneEntityCfg("table"),
        },
    )

    reset_object = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {
                "x": [-0.2, 0.2],
                "y": [-0.2, 0.2],
                "z": [0.0, 0.4],
                "roll": [-3.14, 3.14],
                "pitch": [-3.14, 3.14],
                "yaw": [-3.14, 3.14],
            },
            "velocity_range": {"x": [-0.0, 0.0], "y": [-0.0, 0.0], "z": [-0.0, 0.0]},
            "asset_cfg": SceneEntityCfg("object"),
        },
    )

    reset_root = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": [-0.0, 0.0], "y": [-0.0, 0.0], "yaw": [-0.0, 0.0]},
            "velocity_range": {"x": [-0.0, 0.0], "y": [-0.0, 0.0], "z": [-0.0, 0.0]},
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "position_range": [-0.50, 0.50],
            "velocity_range": [0.0, 0.0],
        },
    )

    reset_robot_wrist_joint = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names="iiwa7_joint_7"),
            "position_range": [-3, 3],
            "velocity_range": [0.0, 0.0],
        },
    )

    variable_gravity = EventTerm(
        func=mdp.randomize_physics_scene_gravity,
        mode="reset",
        params={
            "gravity_distribution_params": ([0.0, 0.0, 0.0], [0.0, 0.0, 0.0]),
            "operation": "abs",
        },
    )
```
All `reset_*` funcs are stock `isaaclab.envs.mdp` (star-imported).

### Smoke
No dedicated reset smoke; verified by §1 build + an episode reset producing finite obs. NOT CAPTURED on this host.

---

## §4 Goal + Termination

### Description
Goal is a 6-DoF object pose command (`ObjectUniformPoseCommand`) sampled uniformly in the robot base frame: position bounds and full roll/pitch (yaw fixed to 0 in the Ranges, but reset_object scrambles initial yaw). For Reorient, `position_only=False` (orientation matters) and `resampling_time_range=(10,10)` (single goal per episode). Goal pose recolors the table prim green when within tolerance (pos<0.05 m, rot<0.5 rad). Terminations: time_out (4 s), object out of a generous workspace box, and an abnormal-robot guard on joint velocity blow-ups.

### Decisions resolved
- Command `object_pose` ranges: pos_x ∈ [-0.7,-0.3], pos_y ∈ [-0.25,0.25], pos_z ∈ [0.55,0.95], roll/pitch ∈ [-3.14,3.14], yaw fixed [0,0]. Command dim = 7 `(x,y,z,qw,qx,qy,qz)`.
- `position_only = False` (Reorient cares about orientation), `resampling_time_range = (10,10)` (set in base `__post_init__`; effectively one goal per 4 s episode), `body_name="palm_link"` (set in mixin).
- `success_vis_asset_name="table"`; success marker swaps table material green (success) / red (failure).
- `episode_length_s = 4.0`, `is_finite_horizon=True`.
- Terminations: `time_out` (time_out=True); `object_out_of_bound` box x∈(-1.5,0.5), y∈(-2.0,2.0), z∈(0.0,2.0); `abnormal_robot` (joint_vel > 2× limit on any joint). `early_termination` reward (§6) penalizes the `abnormal_robot` termination only.

### Code — CommandsCfg + base/PLAY post-init
`dexsuite_env_cfg.py`
```python
@configclass
class CommandsCfg:
    object_pose = mdp.ObjectUniformPoseCommandCfg(
        asset_name="robot",
        object_name="object",
        resampling_time_range=(3.0, 5.0),
        debug_vis=False,
        ranges=mdp.ObjectUniformPoseCommandCfg.Ranges(
            pos_x=(-0.7, -0.3),
            pos_y=(-0.25, 0.25),
            pos_z=(0.55, 0.95),
            roll=(-3.14, 3.14),
            pitch=(-3.14, 3.14),
            yaw=(0.0, 0.0),
        ),
        success_vis_asset_name="table",
    )
```
```python
    def __post_init__(self):   # DexsuiteReorientEnvCfg
        self.decimation = 2  # 50 Hz
        self.commands.object_pose.resampling_time_range = (10.0, 10.0)
        self.commands.object_pose.position_only = False
        self.commands.object_pose.success_visualizer_cfg.markers["failure"] = self.scene.table.spawn.replace(
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.25, 0.15, 0.15), roughness=0.25), visible=True
        )
        self.commands.object_pose.success_visualizer_cfg.markers["success"] = self.scene.table.spawn.replace(
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.15, 0.25, 0.15), roughness=0.25), visible=True
        )
        self.episode_length_s = 4.0
        self.is_finite_horizon = True
        self.sim.dt = 1 / 120
        self.sim.render_interval = self.decimation
        self.sim.physx.bounce_threshold_velocity = 0.2
        self.sim.physx.bounce_threshold_velocity = 0.01
        self.sim.physx.gpu_max_rigid_patch_count = 4 * 5 * 2**15
        if self.curriculum is not None:
            self.curriculum.adr.params["pos_tol"] = self.rewards.success.params["pos_std"] / 2
            self.curriculum.adr.params["rot_tol"] = self.rewards.success.params["rot_std"] / 2
```

### Code — TerminationsCfg + termination funcs
`dexsuite_env_cfg.py`
```python
@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    object_out_of_bound = DoneTerm(
        func=mdp.out_of_bound,
        params={
            "in_bound_range": {"x": (-1.5, 0.5), "y": (-2.0, 2.0), "z": (0.0, 2.0)},
            "asset_cfg": SceneEntityCfg("object"),
        },
    )
    abnormal_robot = DoneTerm(func=mdp.abnormal_robot_state)
```
`mdp/terminations.py` (verbatim)
```python
def out_of_bound(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    in_bound_range: dict[str, tuple[float, float]] = {},
) -> torch.Tensor:
    object: RigidObject = env.scene[asset_cfg.name]
    range_list = [in_bound_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z"]]
    ranges = torch.tensor(range_list, device=env.device)
    object_pos_local = object.data.root_pos_w - env.scene.env_origins
    outside_bounds = ((object_pos_local < ranges[:, 0]) | (object_pos_local > ranges[:, 1])).any(dim=1)
    return outside_bounds


def abnormal_robot_state(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    robot: Articulation = env.scene[asset_cfg.name]
    return (robot.data.joint_vel.abs() > (robot.data.joint_vel_limits * 2)).any(dim=1)
```

### Code — ObjectUniformPoseCommand (command term class)
`mdp/commands/pose_commands.py` (verbatim)
```python
class ObjectUniformPoseCommand(CommandTerm):
    cfg: dex_cmd_cfgs.ObjectUniformPoseCommandCfg

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.robot: Articulation = env.scene[cfg.asset_name]
        self.object: RigidObject = env.scene[cfg.object_name]
        self.success_vis_asset: RigidObject = env.scene[cfg.success_vis_asset_name]
        self.pose_command_b = torch.zeros(self.num_envs, 7, device=self.device)
        self.pose_command_b[:, 3] = 1.0
        self.pose_command_w = torch.zeros_like(self.pose_command_b)
        self.metrics["position_error"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["orientation_error"] = torch.zeros(self.num_envs, device=self.device)
        self.success_visualizer = VisualizationMarkers(self.cfg.success_visualizer_cfg)
        self.success_visualizer.set_visibility(True)

    @property
    def command(self) -> torch.Tensor:
        return self.pose_command_b

    def _update_metrics(self):
        self.pose_command_w[:, :3], self.pose_command_w[:, 3:] = combine_frame_transforms(
            self.robot.data.root_pos_w, self.robot.data.root_quat_w,
            self.pose_command_b[:, :3], self.pose_command_b[:, 3:],
        )
        pos_error, rot_error = compute_pose_error(
            self.pose_command_w[:, :3], self.pose_command_w[:, 3:],
            self.object.data.root_state_w[:, :3], self.object.data.root_state_w[:, 3:7],
        )
        self.metrics["position_error"] = torch.norm(pos_error, dim=-1)
        self.metrics["orientation_error"] = torch.norm(rot_error, dim=-1)
        success_id = self.metrics["position_error"] < 0.05
        if not self.cfg.position_only:
            success_id &= self.metrics["orientation_error"] < 0.5
        self.success_visualizer.visualize(self.success_vis_asset.data.root_pos_w, marker_indices=success_id.int())

    def _resample_command(self, env_ids):
        r = torch.empty(len(env_ids), device=self.device)
        self.pose_command_b[env_ids, 0] = r.uniform_(*self.cfg.ranges.pos_x)
        self.pose_command_b[env_ids, 1] = r.uniform_(*self.cfg.ranges.pos_y)
        self.pose_command_b[env_ids, 2] = r.uniform_(*self.cfg.ranges.pos_z)
        euler_angles = torch.zeros_like(self.pose_command_b[env_ids, :3])
        euler_angles[:, 0].uniform_(*self.cfg.ranges.roll)
        euler_angles[:, 1].uniform_(*self.cfg.ranges.pitch)
        euler_angles[:, 2].uniform_(*self.cfg.ranges.yaw)
        quat = quat_from_euler_xyz(euler_angles[:, 0], euler_angles[:, 1], euler_angles[:, 2])
        self.pose_command_b[env_ids, 3:] = quat_unique(quat) if self.cfg.make_quat_unique else quat

    def _update_command(self):
        pass
```
(Full source incl. `_set_debug_vis_impl` / `_debug_vis_callback` at `mdp/commands/pose_commands.py:146-181`; visualization-only, omitted here for brevity.) Cfg class `ObjectUniformPoseCommandCfg` at `mdp/commands/pose_commands_cfg.py:34-93` — `position_only` default True (overridden False in base post-init), `make_quat_unique=False`.

### Smoke
No dedicated termination smoke. Verified by build + episode rollout reaching `time_out` at 4 s. NOT CAPTURED on this host.

---

## §5 Observation

### Description
Three observation groups, each concatenated and stacked over a history of 5 control steps. **policy**: object orientation in base frame, the 7-D goal command, and last action. **proprio**: full joint pos/vel, hand-tip body states (palm + 4 tips), and 4-finger contact forces in base frame. **perception**: a 64-point object surface point cloud in base frame. `enable_corruption=True` (additive uniform noise; the noise magnitudes are zero at task spec time and grow via the ADR curriculum — see §7). Per-term `clip` bounds applied.

### Decisions resolved (per-step, before ×history_length)
- **policy** (`concatenate_terms=True`, `history_length=5`):
  - `object_quat_b` → 4
  - `target_object_pose_b` (= `generated_commands(object_pose)`) → 7
  - `actions` (`last_action`) → 23
  - per-step = 34 → **×5 = 170**
- **proprio** (`concatenate_terms=True`, `history_length=5`):
  - `joint_pos` → 23
  - `joint_vel` → 23
  - `hand_tips_state_b` (`body_state_b`, bodies = ["palm_link", ".*_tip"] = palm + 4 fingertips = 5 bodies × 13) → 65
  - `contact` (`fingers_contact_force_b`, 4 sensors × 3) → 12
  - per-step = 123 → **×5 = 615**
- **perception** (`concatenate_terms=True`, `concatenate_dim=0`, `flatten_history_dim=True`, `history_length=5`):
  - `object_point_cloud` (`object_point_cloud_b`, num_points=64, flatten=True) → 192 per step → **×5 = 960**
- Total resolved obs (analytic) = **170 + 615 + 960 = 1745** across a Dict of 3 groups `{policy:(170,), proprio:(615,), perception:(960,)}`. NOTE: history stacking layout (concat vs leading time dim) is IsaacLab-version-dependent; treat the per-group per-step dims (34 / 123 / 192) as the load-bearing values and re-verify the stacked shapes against the live build.
- Clips: `hand_tips_state_b` clip (-2,2); `contact` clip (-20,20); `object_point_cloud` clip (-2,2). Base noise all `Unoise(0,0)` at spec time.

### Code — ObservationsCfg (base) + mixin overrides
`dexsuite_env_cfg.py`
```python
@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        object_quat_b = ObsTerm(func=mdp.object_quat_b, noise=Unoise(n_min=-0.0, n_max=0.0))
        target_object_pose_b = ObsTerm(func=mdp.generated_commands, params={"command_name": "object_pose"})
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True
            self.history_length = 5

    @configclass
    class ProprioObsCfg(ObsGroup):
        joint_pos = ObsTerm(func=mdp.joint_pos, noise=Unoise(n_min=-0.0, n_max=0.0))
        joint_vel = ObsTerm(func=mdp.joint_vel, noise=Unoise(n_min=-0.0, n_max=0.0))
        hand_tips_state_b = ObsTerm(
            func=mdp.body_state_b,
            noise=Unoise(n_min=-0.0, n_max=0.0),
            clip=(-2.0, 2.0),
            params={
                "body_asset_cfg": SceneEntityCfg("robot"),
                "base_asset_cfg": SceneEntityCfg("robot"),
            },
        )
        contact: ObsTerm = MISSING

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True
            self.history_length = 5

    @configclass
    class PerceptionObsCfg(ObsGroup):
        object_point_cloud = ObsTerm(
            func=mdp.object_point_cloud_b,
            noise=Unoise(n_min=-0.0, n_max=0.0),
            clip=(-2.0, 2.0),
            params={"num_points": 64, "flatten": True},
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_dim = 0
            self.concatenate_terms = True
            self.flatten_history_dim = True
            self.history_length = 5

    policy: PolicyCfg = PolicyCfg()
    proprio: ProprioObsCfg = ProprioObsCfg()
    perception: PerceptionObsCfg = PerceptionObsCfg()
```
Mixin sets `proprio.contact` (`fingers_contact_force_b`, clip (-20,20)) and overrides `hand_tips_state_b.body_names = ["palm_link", ".*_tip"]` (see §1 mixin code).

### Code — task-local observation funcs
`mdp/observations.py` (verbatim)
```python
def object_quat_b(env, robot_cfg=SceneEntityCfg("robot"), object_cfg=SceneEntityCfg("object")) -> torch.Tensor:
    robot: RigidObject = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    return quat_mul(quat_inv(robot.data.root_quat_w), object.data.root_quat_w)


def body_state_b(env, body_asset_cfg: SceneEntityCfg, base_asset_cfg: SceneEntityCfg) -> torch.Tensor:
    body_asset: Articulation = env.scene[body_asset_cfg.name]
    base_asset: Articulation = env.scene[base_asset_cfg.name]
    body_pos_w = body_asset.data.body_pos_w[:, body_asset_cfg.body_ids].view(-1, 3)
    body_quat_w = body_asset.data.body_quat_w[:, body_asset_cfg.body_ids].view(-1, 4)
    body_lin_vel_w = body_asset.data.body_lin_vel_w[:, body_asset_cfg.body_ids].view(-1, 3)
    body_ang_vel_w = body_asset.data.body_ang_vel_w[:, body_asset_cfg.body_ids].view(-1, 3)
    num_bodies = int(body_pos_w.shape[0] / env.num_envs)
    root_pos_w = base_asset.data.root_link_pos_w.unsqueeze(1).repeat_interleave(num_bodies, dim=1).view(-1, 3)
    root_quat_w = base_asset.data.root_link_quat_w.unsqueeze(1).repeat_interleave(num_bodies, dim=1).view(-1, 4)
    body_pos_b, body_quat_b = subtract_frame_transforms(root_pos_w, root_quat_w, body_pos_w, body_quat_w)
    body_lin_vel_b = quat_apply_inverse(root_quat_w, body_lin_vel_w)
    body_ang_vel_b = quat_apply_inverse(root_quat_w, body_ang_vel_w)
    out = torch.cat((body_pos_b, body_quat_b, body_lin_vel_b, body_ang_vel_b), dim=1)
    return out.view(env.num_envs, -1)


class object_point_cloud_b(ManagerTermBase):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.object_cfg = cfg.params.get("object_cfg", SceneEntityCfg("object"))
        self.ref_asset_cfg = cfg.params.get("ref_asset_cfg", SceneEntityCfg("robot"))
        num_points: int = cfg.params.get("num_points", 10)
        self.object: RigidObject = env.scene[self.object_cfg.name]
        self.ref_asset: Articulation = env.scene[self.ref_asset_cfg.name]
        if cfg.params.get("visualize", True):
            from isaaclab.markers import VisualizationMarkers
            from isaaclab.markers.config import RAY_CASTER_MARKER_CFG
            ray_cfg = RAY_CASTER_MARKER_CFG.replace(prim_path="/Visuals/ObservationPointCloud")
            ray_cfg.markers["hit"].radius = 0.0025
            self.visualizer = VisualizationMarkers(ray_cfg)
        self.points_local = sample_object_point_cloud(
            env.num_envs, num_points, self.object.cfg.prim_path, device=env.device
        )
        self.points_w = torch.zeros_like(self.points_local)

    def __call__(self, env, ref_asset_cfg=SceneEntityCfg("robot"), object_cfg=SceneEntityCfg("object"),
                 num_points: int = 10, flatten: bool = False, visualize: bool = True):
        ref_pos_w = self.ref_asset.data.root_pos_w.unsqueeze(1).repeat(1, num_points, 1)
        ref_quat_w = self.ref_asset.data.root_quat_w.unsqueeze(1).repeat(1, num_points, 1)
        object_pos_w = self.object.data.root_pos_w.unsqueeze(1).repeat(1, num_points, 1)
        object_quat_w = self.object.data.root_quat_w.unsqueeze(1).repeat(1, num_points, 1)
        self.points_w = quat_apply(object_quat_w, self.points_local) + object_pos_w
        if visualize:
            self.visualizer.visualize(translations=self.points_w.view(-1, 3))
        object_point_cloud_pos_b, _ = subtract_frame_transforms(ref_pos_w, ref_quat_w, self.points_w, None)
        return object_point_cloud_pos_b.view(env.num_envs, -1) if flatten else object_point_cloud_pos_b


def fingers_contact_force_b(env, contact_sensor_names: list[str], asset_cfg=SceneEntityCfg("robot")) -> torch.Tensor:
    force_w = [env.scene.sensors[name].data.force_matrix_w.view(env.num_envs, 3) for name in contact_sensor_names]
    force_w = torch.stack(force_w, dim=1)
    robot: Articulation = env.scene[asset_cfg.name]
    forces_b = quat_apply_inverse(robot.data.root_link_quat_w.unsqueeze(1).repeat(1, force_w.shape[1], 1), force_w)
    return forces_b
```
`object_pos_b` also exists in this file but is unused by this task. `object_point_cloud_b` depends on `sample_object_point_cloud` from `mdp/utils.py` (helper exists; reads the object USD to sample surface points — relevant only when reproducing the perception group). WARN: `object_point_cloud_b` requires the live object USD + `pxr`; reproducing the perception group outside Isaac Sim is not possible.
`object_quat_b`, `generated_commands`, `last_action`, `joint_pos`, `joint_vel` — last 4 are stock `isaaclab.envs.mdp`.

### Smoke
Covered by §1 build smoke (`env.observation_space` Dict shapes). NOT CAPTURED on this host.

---

## §6 Reward

### Description
Composer = **sum** (IsaacLab `RewardManager` sums all weighted RewTerms per step). Terms: small action / action-rate L2 penalties; a tanh reach reward (`fingers_to_object`); contact-gated position- and orientation-tracking tanh rewards (the two tracking terms are MULTIPLIED by a contact indicator inside their functions — only paid when ≥2 fingers incl. thumb touch the object); a large product-of-tanh `success` bonus (pos AND rot within std); a penalty when the abnormal-robot termination fires; plus the kuka_allegro-specific `good_finger_contact` bonus (0.5 when thumb + ≥1 other finger contact > 1 N).

### Decisions resolved — RewardsCfg (base) + kuka_allegro override
| term | func | weight | key params |
|---|---|---|---|
| action_l2 | action_l2_clamped | -0.005 | clamp(-1000,1000) |
| action_rate_l2 | action_rate_l2_clamped | -0.005 | clamp(-1000,1000) |
| fingers_to_object | object_ee_distance | 1.0 | std=0.4; asset=palm+tips (mixin) |
| position_tracking | position_command_error_tanh | 2.0 | std=0.2, gated by contacts(1.0) |
| orientation_tracking | orientation_command_error_tanh | 4.0 | std=1.5, gated by contacts(1.0) |
| success | success_reward | 10 | pos_std=0.1, rot_std=0.5 (product of tanh) |
| early_termination | is_terminated_term | -1 | term_keys="abnormal_robot" |
| good_finger_contact | contacts | 0.5 | threshold=1.0 (kuka_allegro mixin) |

Per-step saturated nominal magnitudes (retro-computed): reach ≤ 1.0; pos_track ≤ 2.0 (gated); orient_track ≤ 4.0 (gated); success ≤ 10; good_finger_contact ≤ 0.5; penalties small/negative. Success bonus is the dominant signal once contact + alignment achieved.

### Code — RewardsCfg + kuka_allegro reward override
`dexsuite_env_cfg.py`
```python
@configclass
class RewardsCfg:
    action_l2 = RewTerm(func=mdp.action_l2_clamped, weight=-0.005)
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2_clamped, weight=-0.005)
    fingers_to_object = RewTerm(func=mdp.object_ee_distance, params={"std": 0.4}, weight=1.0)
    position_tracking = RewTerm(
        func=mdp.position_command_error_tanh,
        weight=2.0,
        params={"asset_cfg": SceneEntityCfg("robot"), "std": 0.2,
                "command_name": "object_pose", "align_asset_cfg": SceneEntityCfg("object")},
    )
    orientation_tracking = RewTerm(
        func=mdp.orientation_command_error_tanh,
        weight=4.0,
        params={"asset_cfg": SceneEntityCfg("robot"), "std": 1.5,
                "command_name": "object_pose", "align_asset_cfg": SceneEntityCfg("object")},
    )
    success = RewTerm(
        func=mdp.success_reward,
        weight=10,
        params={"asset_cfg": SceneEntityCfg("robot"), "pos_std": 0.1, "rot_std": 0.5,
                "command_name": "object_pose", "align_asset_cfg": SceneEntityCfg("object")},
    )
    early_termination = RewTerm(func=mdp.is_terminated_term, weight=-1, params={"term_keys": "abnormal_robot"})
```
`config/kuka_allegro/dexsuite_kuka_allegro_env_cfg.py`
```python
@configclass
class KukaAllegroReorientRewardCfg(dexsuite.RewardsCfg):
    # bool awarding term if 2 finger tips are in contact with object, one of the contacting fingers has to be thumb.
    good_finger_contact = RewTerm(func=mdp.contacts, weight=0.5, params={"threshold": 1.0})
```

### Code — reward functions (verbatim, `mdp/rewards.py`)
```python
def action_rate_l2_clamped(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Penalize the rate of change of the actions using L2 squared kernel."""
    return torch.sum(torch.square(env.action_manager.action - env.action_manager.prev_action), dim=1).clamp(-1000, 1000)


def action_l2_clamped(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Penalize the actions using L2 squared kernel."""
    return torch.sum(torch.square(env.action_manager.action), dim=1).clamp(-1000, 1000)


def object_ee_distance(
    env: ManagerBasedRLEnv,
    std: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward reaching the object using a tanh-kernel on end-effector distance."""
    asset: RigidObject = env.scene[asset_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    asset_pos = asset.data.body_pos_w[:, asset_cfg.body_ids]
    object_pos = object.data.root_pos_w
    object_ee_distance = torch.norm(asset_pos - object_pos[:, None, :], dim=-1).max(dim=-1).values
    return 1 - torch.tanh(object_ee_distance / std)


def contacts(env: ManagerBasedRLEnv, threshold: float) -> torch.Tensor:
    """Penalize undesired contacts as the number of violations that are above a threshold."""
    thumb_contact_sensor: ContactSensor = env.scene.sensors["thumb_link_3_object_s"]
    index_contact_sensor: ContactSensor = env.scene.sensors["index_link_3_object_s"]
    middle_contact_sensor: ContactSensor = env.scene.sensors["middle_link_3_object_s"]
    ring_contact_sensor: ContactSensor = env.scene.sensors["ring_link_3_object_s"]
    thumb_contact = thumb_contact_sensor.data.force_matrix_w.view(env.num_envs, 3)
    index_contact = index_contact_sensor.data.force_matrix_w.view(env.num_envs, 3)
    middle_contact = middle_contact_sensor.data.force_matrix_w.view(env.num_envs, 3)
    ring_contact = ring_contact_sensor.data.force_matrix_w.view(env.num_envs, 3)
    thumb_contact_mag = torch.norm(thumb_contact, dim=-1)
    index_contact_mag = torch.norm(index_contact, dim=-1)
    middle_contact_mag = torch.norm(middle_contact, dim=-1)
    ring_contact_mag = torch.norm(ring_contact, dim=-1)
    good_contact_cond1 = (thumb_contact_mag > threshold) & (
        (index_contact_mag > threshold) | (middle_contact_mag > threshold) | (ring_contact_mag > threshold)
    )
    return good_contact_cond1


def success_reward(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    align_asset_cfg: SceneEntityCfg,
    pos_std: float,
    rot_std: float | None = None,
) -> torch.Tensor:
    """Reward success by comparing commanded pose to the object pose using tanh kernels on error."""
    asset: RigidObject = env.scene[asset_cfg.name]
    object: RigidObject = env.scene[align_asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    des_pos_w, des_quat_w = combine_frame_transforms(
        asset.data.root_pos_w, asset.data.root_quat_w, command[:, :3], command[:, 3:7]
    )
    pos_err, rot_err = compute_pose_error(des_pos_w, des_quat_w, object.data.root_pos_w, object.data.root_quat_w)
    pos_dist = torch.norm(pos_err, dim=1)
    if not rot_std:
        # square is not necessary but this help to keep the final value between having rot_std or not roughly the same
        return (1 - torch.tanh(pos_dist / pos_std)) ** 2
    rot_dist = torch.norm(rot_err, dim=1)
    return (1 - torch.tanh(pos_dist / pos_std)) * (1 - torch.tanh(rot_dist / rot_std))


def position_command_error_tanh(
    env: ManagerBasedRLEnv, std: float, command_name: str, asset_cfg: SceneEntityCfg, align_asset_cfg: SceneEntityCfg
) -> torch.Tensor:
    """Reward tracking of commanded position using tanh kernel, gated by contact presence."""
    asset: RigidObject = env.scene[asset_cfg.name]
    object: RigidObject = env.scene[align_asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    des_pos_b = command[:, :3]
    des_pos_w, _ = combine_frame_transforms(asset.data.root_pos_w, asset.data.root_quat_w, des_pos_b)
    distance = torch.norm(object.data.root_pos_w - des_pos_w, dim=1)
    return (1 - torch.tanh(distance / std)) * contacts(env, 1.0).float()


def orientation_command_error_tanh(
    env: ManagerBasedRLEnv, std: float, command_name: str, asset_cfg: SceneEntityCfg, align_asset_cfg: SceneEntityCfg
) -> torch.Tensor:
    """Reward tracking of commanded orientation using tanh kernel, gated by contact presence."""
    asset: RigidObject = env.scene[asset_cfg.name]
    object: RigidObject = env.scene[align_asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    des_quat_b = command[:, 3:7]
    des_quat_w = math_utils.quat_mul(asset.data.root_state_w[:, 3:7], des_quat_b)
    quat_distance = math_utils.quat_error_magnitude(object.data.root_quat_w, des_quat_w)
    return (1 - torch.tanh(quat_distance / std)) * contacts(env, 1.0).float()
```
Imports for the above (top of `mdp/rewards.py`): `torch`; `from isaaclab.assets import RigidObject`; `from isaaclab.managers import SceneEntityCfg`; `from isaaclab.sensors import ContactSensor`; `from isaaclab.utils import math as math_utils`; `from isaaclab.utils.math import combine_frame_transforms, compute_pose_error`.
`is_terminated_term` is stock `isaaclab.envs.mdp`.

WARN: the two tracking rewards and `good_finger_contact` hard-code the four contact-sensor names `{thumb,index,middle,ring}_link_3_object_s` — those sensors MUST be wired in the scene (§1 mixin) or `contacts()` raises KeyError. Any reproduction must keep the 4-sensor naming exactly.

### Smoke
S6 reward smoke (finite + non-constant + composer==sum assertion via `info["detailed_reward"]`) — NOT CAPTURED on this host (needs Isaac Sim). When reproducing, composer="sum".

---

## §7 DR

### Description
Rich startup + pre-startup + per-reset domain randomization, plus an ADR curriculum that ramps gravity from 0 → -9.81 and ramps up several observation-noise magnitudes as the policy succeeds. Pre-startup randomizes object scale; startup randomizes friction (robot + object physics material), actuator stiffness/damping, joint friction, and object mass. Per-reset `variable_gravity` (listed in §3) is driven by the ADR curriculum. The ADR difficulty scheduler promotes/demotes per-env difficulty based on pos/rot error vs tolerances.

### Decisions resolved — startup / prestartup EventTerms
- `randomize_object_scale` (mode=prestartup): scale_range (0.75, 1.5) on object.
- `robot_physics_material` (startup): static/dynamic friction ∈ [0.5,1.0], restitution 0, 250 buckets, all robot bodies.
- `object_physics_material` (startup): same ranges, on object.
- `joint_stiffness_and_damping` (startup): stiffness & damping scaled ∈ [0.5,2.0], all joints.
- `joint_friction` (startup): friction scaled ∈ [0.0,5.0], all joints.
- `object_scale_mass` (startup): mass scaled ∈ [0.2,2.0].
- (per-reset `variable_gravity` — see §3; curriculum ramps it to gravity.)

### Code — EventCfg (mode != "reset")
`dexsuite_env_cfg.py`
```python
    randomize_object_scale = EventTerm(
        func=mdp.randomize_rigid_body_scale,
        mode="prestartup",
        params={"scale_range": (0.75, 1.5), "asset_cfg": SceneEntityCfg("object")},
    )

    robot_physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": [0.5, 1.0],
            "dynamic_friction_range": [0.5, 1.0],
            "restitution_range": [0.0, 0.0],
            "num_buckets": 250,
        },
    )

    object_physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("object", body_names=".*"),
            "static_friction_range": [0.5, 1.0],
            "dynamic_friction_range": [0.5, 1.0],
            "restitution_range": [0.0, 0.0],
            "num_buckets": 250,
        },
    )

    joint_stiffness_and_damping = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "stiffness_distribution_params": [0.5, 2.0],
            "damping_distribution_params": [0.5, 2.0],
            "operation": "scale",
        },
    )

    joint_friction = EventTerm(
        func=mdp.randomize_joint_parameters,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "friction_distribution_params": [0.0, 5.0],
            "operation": "scale",
        },
    )

    object_scale_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("object"),
            "mass_distribution_params": [0.2, 2.0],
            "operation": "scale",
        },
    )
```

### Code — ADR curriculum (CurriculumCfg + scheduler)
`adr_curriculum.py` — `CurriculumCfg`: `adr` = `DifficultyScheduler(init=0,min=0,max=10)`; ten `modify_term_cfg` terms interpolate obs-noise n_min/n_max for joint_pos (±0.1), joint_vel (±0.2), hand_tips (±0.01), object_quat (±0.03), point_cloud (-0.01), and `gravity_adr` interpolating `events.variable_gravity` gravity from `((0,0,0),(0,0,0))` → `((0,0,-9.81),(0,0,-9.81))` as difficulty rises. `pos_tol`/`rot_tol` wired in base post-init = success pos_std/2 (0.05), rot_std/2 (0.25). PLAY variant pins difficulty to max.
`mdp/curriculums.py` — `initial_final_interpolate_fn` (frac<0.1 → NO_CHANGE no-op) + recursive interpolation; `DifficultyScheduler` (ManagerTermBase) promotes a per-env difficulty when pos_dist<pos_tol (& rot_dist<rot_tol if set), demotes otherwise, exposing `difficulty_frac = mean(difficulties)/max_difficulty`. Full source at `mdp/curriculums.py:22-115`.

WARN: the curriculum is non-trivial and IsaacLab-version-coupled (`mdp.modify_term_cfg`, `mdp.modify_env_param.NO_CHANGE`, `mdp.randomize_*` are stock `isaaclab.envs.mdp`). When reproducing onto another IsaacLab fork, confirm these symbols exist; otherwise DR/curriculum must be ported manually. `gravity-as-curriculum` replaces a dedicated lift reward (per the Octi note in EventCfg).

### Smoke
S7 DR smoke (DR ON vs OFF seed-matched obs trajectories diverge) — NOT CAPTURED on this host. DR is present and rich, so a reproduction's S7 should pass once Isaac Sim is available.

---

