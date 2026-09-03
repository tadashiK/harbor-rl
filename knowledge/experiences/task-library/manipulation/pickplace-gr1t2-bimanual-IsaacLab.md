# Isaac-PickPlace-GR1T2-Abs-v0 — Implementation Spec

- robot: Fourier GR1T2 bimanual humanoid (two 7-DoF arms + dexterous hands)
- simulator: IsaacLab (Isaac Sim, manager-based)
- objects: steering wheel, packing table
- bimanual: true
- summary: A bimanual humanoid picks a steering wheel off a packing table and places it.

> **CAVEAT (teleop / IL task).** This is the absolute-IK (`-Abs`) Pink-IK pipeline used for OpenXR/Manus-Vive teleoperation + robomimic BC (imitation learning). `rewards = None`, `commands = None`, `curriculum = None` in the env_cfg. **There is no shaped RL reward (§6 is absent).** Success is a binary termination term only. The reusable value of this spec is the **bimanual scene / 36-D dual-arm absolute-pose action / Dict observation / success-termination** structure (§1–§5). To use as an RL task, a §6 reward must be authored from scratch.

---

## §1 Registration + Scene

**Description.** Registers a single GR1T2 Fourier bimanual humanoid (two 7-DoF arms + two dexterous hands) standing at a packing table, with one rigid `Object` (a steering wheel) to be picked and placed. Entry point is the generic `ManagerBasedRLEnv`. The robot uses high-PD gains on the upper body for IK tracking. Gravity is disabled on the robot (`disable_gravity=True`) — typical of teleop IK pipelines so the arms hold pose without active gravity comp.

**Decisions resolved.**
- `entry_point = isaaclab.envs:ManagerBasedRLEnv`
- `env_cfg_entry_point = ...pickplace_gr1t2_env_cfg:PickPlaceGR1T2EnvCfg`
- `robomimic_bc_cfg_entry_point = agents/robomimic/bc_rnn_low_dim.json` (BC, low-dim)
- Scene: `num_envs=1`, `env_spacing=2.5`, `replicate_physics=True`
- Robot: `GR1T2_HIGH_PD_CFG`, `prim_path=/World/envs/env_.*/Robot`, init `pos=(0,0,0.93)`, `rot=(0.7071,0,0,0.7071)` (yaw 90° so robot faces +y toward the table). Init arm pose: both elbows `-1.5708`, everything else 0.
- Object: steering wheel USD, `scale=(0.75,0.75,0.75)`, init `pos=[-0.45, 0.45, 0.9996]`, `rot=[1,0,0,0]`.
- Packing table: kinematic (`kinematic_enabled=True`), init `pos=[0.0, 0.55, 0.0]`.
- Dome light intensity 3000, ground plane, no camera in the low-dim policy obs.
- `decimation=6`, `episode_length_s=20.0`, `sim.dt=1/120` (120 Hz), `sim.render_interval=2`.

**Resolved asset paths** (Nucleus-relative; `ISAAC_NUCLEUS_DIR = <NUCLEUS_ASSET_ROOT_DIR>/Isaac`, `ISAACLAB_NUCLEUS_DIR = <ISAAC_NUCLEUS_DIR>/IsaacLab`; `NUCLEUS_ASSET_ROOT_DIR` from carb setting `/persistent/isaac/asset_root/cloud`):
- Robot USD: `{ISAAC_NUCLEUS_DIR}/Robots/FourierIntelligence/GR-1/GR1T2_fourier_hand_6dof/GR1T2_fourier_hand_6dof.usd`
- Object USD: `{ISAACLAB_NUCLEUS_DIR}/Mimic/pick_place_task/pick_place_assets/steering_wheel.usd`
- Table USD: `{ISAAC_NUCLEUS_DIR}/Props/PackingTable/packing_table.usd`

> NOTE: assets live on the Nucleus cloud server, not on local disk — cannot be stat-checked here. The reproduction must point at equivalent Nucleus paths.

**Code — `ObjectTableSceneCfg` (verbatim).**
```python
@configclass
class ObjectTableSceneCfg(InteractiveSceneCfg):
    """Configuration for the GR1T2 Pick Place Base Scene."""

    # Table
    packing_table = AssetBaseCfg(
        prim_path="/World/envs/env_.*/PackingTable",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0.0, 0.55, 0.0], rot=[1.0, 0.0, 0.0, 0.0]),
        spawn=UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/PackingTable/packing_table.usd",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
        ),
    )

    object = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Object",
        init_state=RigidObjectCfg.InitialStateCfg(pos=[-0.45, 0.45, 0.9996], rot=[1, 0, 0, 0]),
        spawn=UsdFileCfg(
            usd_path=f"{ISAACLAB_NUCLEUS_DIR}/Mimic/pick_place_task/pick_place_assets/steering_wheel.usd",
            scale=(0.75, 0.75, 0.75),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
        ),
    )

    # Humanoid robot configured for pick-place manipulation tasks
    robot: ArticulationCfg = GR1T2_HIGH_PD_CFG.replace(
        prim_path="/World/envs/env_.*/Robot",
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0, 0, 0.93),
            rot=(0.7071, 0, 0, 0.7071),
            joint_pos={
                # right-arm
                "right_shoulder_pitch_joint": 0.0,
                "right_shoulder_roll_joint": 0.0,
                "right_shoulder_yaw_joint": 0.0,
                "right_elbow_pitch_joint": -1.5708,
                "right_wrist_yaw_joint": 0.0,
                "right_wrist_roll_joint": 0.0,
                "right_wrist_pitch_joint": 0.0,
                # left-arm
                "left_shoulder_pitch_joint": 0.0,
                "left_shoulder_roll_joint": 0.0,
                "left_shoulder_yaw_joint": 0.0,
                "left_elbow_pitch_joint": -1.5708,
                "left_wrist_yaw_joint": 0.0,
                "left_wrist_roll_joint": 0.0,
                "left_wrist_pitch_joint": 0.0,
                # --
                "head_.*": 0.0,
                "waist_.*": 0.0,
                ".*_hip_.*": 0.0,
                ".*_knee_.*": 0.0,
                ".*_ankle_.*": 0.0,
                "R_.*": 0.0,
                "L_.*": 0.0,
            },
            joint_vel={".*": 0.0},
        ),
    )

    # Ground plane
    ground = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        spawn=GroundPlaneCfg(),
    )

    # Lights
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )
```

**Code — robot articulation cfg (`GR1T2_HIGH_PD_CFG`, derived from `GR1T2_CFG`; verbatim from `isaaclab_assets/robots/fourier.py`).**
```python
GR1T2_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=(
            f"{ISAAC_NUCLEUS_DIR}/Robots/FourierIntelligence/GR-1/GR1T2_fourier_hand_6dof/GR1T2_fourier_hand_6dof.usd"
        ),
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=True,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True, solver_position_iteration_count=8, solver_velocity_iteration_count=4
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.95),
        joint_pos={".*": 0.0},
        joint_vel={".*": 0.0},
    ),
    actuators={
        "head": ImplicitActuatorCfg(joint_names_expr=["head_.*"], effort_limit=None, velocity_limit=None, stiffness=None, damping=None),
        "trunk": ImplicitActuatorCfg(joint_names_expr=["waist_.*"], effort_limit=None, velocity_limit=None, stiffness=None, damping=None),
        "legs": ImplicitActuatorCfg(joint_names_expr=[".*_hip_.*", ".*_knee_.*", ".*_ankle_.*"], effort_limit=None, velocity_limit=None, stiffness=None, damping=None),
        "right-arm": ImplicitActuatorCfg(joint_names_expr=["right_shoulder_.*", "right_elbow_.*", "right_wrist_.*"], effort_limit=torch.inf, velocity_limit=torch.inf, stiffness=None, damping=None, armature=0.0),
        "left-arm": ImplicitActuatorCfg(joint_names_expr=["left_shoulder_.*", "left_elbow_.*", "left_wrist_.*"], effort_limit=torch.inf, velocity_limit=torch.inf, stiffness=None, damping=None, armature=0.0),
        "right-hand": ImplicitActuatorCfg(joint_names_expr=["R_.*"], effort_limit=None, velocity_limit=None, stiffness=None, damping=None),
        "left-hand": ImplicitActuatorCfg(joint_names_expr=["L_.*"], effort_limit=None, velocity_limit=None, stiffness=None, damping=None),
    },
)

GR1T2_HIGH_PD_CFG = GR1T2_CFG.replace(
    actuators={
        "trunk": ImplicitActuatorCfg(joint_names_expr=["waist_.*"], effort_limit=None, velocity_limit=None, stiffness=4400, damping=40.0, armature=0.01),
        "right-arm": ImplicitActuatorCfg(joint_names_expr=["right_shoulder_.*", "right_elbow_.*", "right_wrist_.*"], stiffness=4400.0, damping=40.0, armature=0.01),
        "left-arm": ImplicitActuatorCfg(joint_names_expr=["left_shoulder_.*", "left_elbow_.*", "left_wrist_.*"], stiffness=4400.0, damping=40.0, armature=0.01),
        "right-hand": ImplicitActuatorCfg(joint_names_expr=["R_.*"], stiffness=None, damping=None),
        "left-hand": ImplicitActuatorCfg(joint_names_expr=["L_.*"], stiffness=None, damping=None),
    },
)
```
> Note: `GR1T2_HIGH_PD_CFG.replace(actuators=...)` drops the `head` and `legs` actuator groups present in the base cfg (only trunk/arms/hands are re-declared); the high-PD variant therefore has no explicit actuator entry for head/legs.

**Code — `gym.register` (verbatim).**
```python
gym.register(
    id="Isaac-PickPlace-GR1T2-Abs-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.pickplace_gr1t2_env_cfg:PickPlaceGR1T2EnvCfg",
        "robomimic_bc_cfg_entry_point": f"{agents.__name__}:robomimic/bc_rnn_low_dim.json",
    },
    disable_env_checker=True,
)
```

**Smoke.** `cd <repo> && .venv/bin/python -c "import gymnasium as gym; env=gym.make('Isaac-PickPlace-GR1T2-Abs-v0'); print(env.observation_space, env.action_space); env.close()"` — **not capturable in `.venv`** (requires booting the Isaac Kit app + Pink IK). Expected runtime: `action_space=Box(36,)`, `observation_space=Dict(...)` per the per-term list above.

---

## §2 Actions

**Description.** Single absolute (not delta) bimanual end-effector IK action term `upper_body_ik` using the **Pink** differential IK solver (`PinkInverseKinematicsActionCfg`). The 36-D action is `[left_hand pos(3) + quat(4), right_hand pos(3) + quat(4), 22 hand joint targets]` — i.e. two absolute 6-DoF wrist poses solved by two `FrameTask`s plus direct position targets for the 22 dexterous-hand joints. No gripper-action term; the hands are part of the IK action's `hand_joint_names`.

**Decisions resolved.**
- Action term class: `PinkInverseKinematicsActionCfg`, `asset_name="robot"`.
- `action_dim = frame_tasks(2) * pose_dim(7) + num_hand_joints(22) = 36`.
- 14 pink-controlled arm joints (7 left + 7 right shoulder/elbow/wrist).
- 22 `hand_joint_names` (proximal/intermediate/distal of both 5-finger hands).
- `target_eef_link_names = {left_wrist: left_hand_pitch_link, right_wrist: right_hand_pitch_link}`.
- Pink controller: `base_link_name="base_link"`, `num_hand_joints=22`, `fail_on_joint_limit_violation=False`, `show_ik_warnings=False`.
- Two `FrameTask`s (left/right hand_pitch_link): `position_cost=8.0`, `orientation_cost=1.0`, `lm_damping=12`, `gain=0.5`.
- `DampingTask(cost=0.5)`.
- `NullSpacePostureTask(cost=0.5, lm_damping=1, ...)` controlling both hand frames over 8 arm joints + 3 waist joints.
- `xr_enabled` read from carb setting `/app/xr/enabled`.
- `__post_init__` converts the robot USD→URDF (`ControllerUtils.convert_usd_to_urdf(..., force_conversion=True)`) into `tempfile.gettempdir()`, then sets `controller.urdf_path` / `controller.mesh_path`.
- **Requires the `pink` package** (`from pink.tasks import DampingTask, FrameTask`) and `isaaclab.controllers.pink_ik`. If `pink` is not installed the env_cfg import fails.

**Code — `ActionsCfg` (verbatim).**
```python
@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    upper_body_ik = PinkInverseKinematicsActionCfg(
        pink_controlled_joint_names=[
            "left_shoulder_pitch_joint",
            "left_shoulder_roll_joint",
            "left_shoulder_yaw_joint",
            "left_elbow_pitch_joint",
            "left_wrist_yaw_joint",
            "left_wrist_roll_joint",
            "left_wrist_pitch_joint",
            "right_shoulder_pitch_joint",
            "right_shoulder_roll_joint",
            "right_shoulder_yaw_joint",
            "right_elbow_pitch_joint",
            "right_wrist_yaw_joint",
            "right_wrist_roll_joint",
            "right_wrist_pitch_joint",
        ],
        hand_joint_names=[
            "L_index_proximal_joint",
            "L_middle_proximal_joint",
            "L_pinky_proximal_joint",
            "L_ring_proximal_joint",
            "L_thumb_proximal_yaw_joint",
            "R_index_proximal_joint",
            "R_middle_proximal_joint",
            "R_pinky_proximal_joint",
            "R_ring_proximal_joint",
            "R_thumb_proximal_yaw_joint",
            "L_index_intermediate_joint",
            "L_middle_intermediate_joint",
            "L_pinky_intermediate_joint",
            "L_ring_intermediate_joint",
            "L_thumb_proximal_pitch_joint",
            "R_index_intermediate_joint",
            "R_middle_intermediate_joint",
            "R_pinky_intermediate_joint",
            "R_ring_intermediate_joint",
            "R_thumb_proximal_pitch_joint",
            "L_thumb_distal_joint",
            "R_thumb_distal_joint",
        ],
        target_eef_link_names={
            "left_wrist": "left_hand_pitch_link",
            "right_wrist": "right_hand_pitch_link",
        },
        asset_name="robot",
        controller=PinkIKControllerCfg(
            articulation_name="robot",
            base_link_name="base_link",
            num_hand_joints=22,
            show_ik_warnings=False,
            fail_on_joint_limit_violation=False,
            variable_input_tasks=[
                FrameTask(
                    "GR1T2_fourier_hand_6dof_left_hand_pitch_link",
                    position_cost=8.0,
                    orientation_cost=1.0,
                    lm_damping=12,
                    gain=0.5,
                ),
                FrameTask(
                    "GR1T2_fourier_hand_6dof_right_hand_pitch_link",
                    position_cost=8.0,
                    orientation_cost=1.0,
                    lm_damping=12,
                    gain=0.5,
                ),
                DampingTask(
                    cost=0.5,
                ),
                NullSpacePostureTask(
                    cost=0.5,
                    lm_damping=1,
                    controlled_frames=[
                        "GR1T2_fourier_hand_6dof_left_hand_pitch_link",
                        "GR1T2_fourier_hand_6dof_right_hand_pitch_link",
                    ],
                    controlled_joints=[
                        "left_shoulder_pitch_joint",
                        "left_shoulder_roll_joint",
                        "left_shoulder_yaw_joint",
                        "left_elbow_pitch_joint",
                        "right_shoulder_pitch_joint",
                        "right_shoulder_roll_joint",
                        "right_shoulder_yaw_joint",
                        "right_elbow_pitch_joint",
                        "waist_yaw_joint",
                        "waist_pitch_joint",
                        "waist_roll_joint",
                    ],
                ),
            ],
            fixed_input_tasks=[],
            xr_enabled=bool(carb.settings.get_settings().get("/app/xr/enabled")),
        ),
    )
```

**Idle / hold action** (36-D, format `[L pos3, L quat4, R pos3, R quat4, L hand 11, R hand 11]`):
```python
idle_action = torch.tensor([
    -0.22878, 0.2536, 1.0953, 0.5, 0.5, -0.5, 0.5,   # left hand: pos3 + quat4
     0.22878, 0.2536, 1.0953, 0.5, 0.5, -0.5, 0.5,   # right hand: pos3 + quat4
     0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,     # 11 left hand-joint targets
     0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,     # 11 right hand-joint targets
])  # total 36
```

**Smoke (S2).** Step the env with `idle_action` broadcast to `(num_envs, 36)`; assert no exception and the wrist link poses stay near the commanded absolute pose. Not runnable in `.venv` (needs Isaac + Pink).

---

## §3 Reset

**Description.** On reset, the whole scene snaps back to default poses, then the object gets a tiny ±1 cm uniform x/y jitter (essentially deterministic placement — this is a teleop demo task, not a domain-randomized RL task).

**Decisions resolved.**
- `reset_all`: `mdp.reset_scene_to_default`, `mode="reset"`, no params (resets all articulations + rigid objects to their `init_state`).
- `reset_object`: `mdp.reset_root_state_uniform`, `mode="reset"`, `pose_range={x:[-0.01,0.01], y:[-0.01,0.01]}`, `velocity_range={}`, `asset_cfg=SceneEntityCfg("object")`.

**Code — `EventCfg` reset terms (verbatim).**
```python
@configclass
class EventCfg:
    """Configuration for events."""

    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")

    reset_object = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {
                "x": [-0.01, 0.01],
                "y": [-0.01, 0.01],
            },
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("object"),
        },
    )
```
> Both reset functions come from base `isaaclab.envs.mdp` (re-exported via `from isaaclab.envs.mdp import *` in the task `mdp/__init__.py`). No task-local reset code.

**Smoke (S3).** Reset twice; assert object x/y lands within `init ± 0.01`. Not runnable in `.venv`.

---

## §4 Goal + Termination

**Description.** Episode ends on (a) timeout (20 s), (b) the object dropping below 0.5 m (failure), or (c) the binary success predicate `task_done_pick_place`: the steering wheel is placed inside a target x/y box on the table, below max height, at near-zero velocity, AND the right wrist has retracted back toward the body. `CommandsCfg` is `None` — there is no commanded goal pose; the target region is hard-coded inside the success function.

**Decisions resolved.**
- `commands = None` (no command manager).
- `time_out`: `mdp.time_out`, `time_out=True` (episode_length_s=20.0 → ~20·120/6 = 400 decimated steps).
- `object_dropping`: `mdp.root_height_below_minimum`, `minimum_height=0.5`, `asset_cfg=SceneEntityCfg("object")`.
- `success`: `mdp.task_done_pick_place`, `task_link_name="right_hand_roll_link"`. Default thresholds (env-origin-relative): object `0.40<x<0.85`, `0.35<y<0.60`, `height<1.10`, right-wrist `x<0.26`, per-axis `|vel|<0.20`.

**Code — `TerminationsCfg` (verbatim).**
```python
@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    object_dropping = DoneTerm(
        func=mdp.root_height_below_minimum, params={"minimum_height": 0.5, "asset_cfg": SceneEntityCfg("object")}
    )

    success = DoneTerm(func=mdp.task_done_pick_place, params={"task_link_name": "right_hand_roll_link"})
```

**Code — `task_done_pick_place` (verbatim, task-local `mdp/terminations.py`).**
```python
def task_done_pick_place(
    env: ManagerBasedRLEnv,
    task_link_name: str = "",
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    right_wrist_max_x: float = 0.26,
    min_x: float = 0.40,
    max_x: float = 0.85,
    min_y: float = 0.35,
    max_y: float = 0.60,
    max_height: float = 1.10,
    min_vel: float = 0.20,
) -> torch.Tensor:
    if task_link_name == "":
        raise ValueError("task_link_name must be provided to task_done_pick_place")

    object: RigidObject = env.scene[object_cfg.name]

    object_x = object.data.root_pos_w[:, 0] - env.scene.env_origins[:, 0]
    object_y = object.data.root_pos_w[:, 1] - env.scene.env_origins[:, 1]
    object_height = object.data.root_pos_w[:, 2] - env.scene.env_origins[:, 2]
    object_vel = torch.abs(object.data.root_vel_w)

    robot_body_pos_w = env.scene["robot"].data.body_pos_w
    right_eef_idx = env.scene["robot"].data.body_names.index(task_link_name)
    right_wrist_x = robot_body_pos_w[:, right_eef_idx, 0] - env.scene.env_origins[:, 0]

    done = object_x < max_x
    done = torch.logical_and(done, object_x > min_x)
    done = torch.logical_and(done, object_y < max_y)
    done = torch.logical_and(done, object_y > min_y)
    done = torch.logical_and(done, object_height < max_height)
    done = torch.logical_and(done, right_wrist_x < right_wrist_max_x)
    done = torch.logical_and(done, object_vel[:, 0] < min_vel)
    done = torch.logical_and(done, object_vel[:, 1] < min_vel)
    done = torch.logical_and(done, object_vel[:, 2] < min_vel)

    return done
```
> `time_out`, `root_height_below_minimum` come from base mdp.

**Smoke (S4).** Roll out with idle action; assert `success` is initially False and the env terminates by timeout within 400 steps. Not runnable in `.venv`.

---

## §5 Observation

**Description.** One `policy` group, **non-concatenated** (`concatenate_terms=False` → returns a `Dict`), corruption disabled. Captures the full robot proprioception (joint pos, root pose, all link states, both EEF poses, hand + head joint states), the object pose, the last action, and a compact relative `object` term (object pose + each hand→object vector). This rich dict matches the robomimic low-dim BC input.

**Decisions resolved.**
- `enable_corruption=False`, `concatenate_terms=False` → observation_space is a `Dict`.
- No `noise` on any term.
- 15 obs terms (see code). `object` term = 13-D (obj_pos3 + obj_quat4 + left_eef_to_obj3 + right_eef_to_obj3).
- EEF links queried by name: `left_hand_roll_link`, `right_hand_roll_link`.
- `hand_joint_state` selects joints by regex `["R_.*","L_.*"]`; `head_joint_state` by `["head_pitch_joint","head_roll_joint","head_yaw_joint"]`.

**Code — `ObservationsCfg` (verbatim).**
```python
@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group with state values."""

        actions = ObsTerm(func=mdp.last_action)
        robot_joint_pos = ObsTerm(
            func=base_mdp.joint_pos,
            params={"asset_cfg": SceneEntityCfg("robot")},
        )
        robot_root_pos = ObsTerm(func=base_mdp.root_pos_w, params={"asset_cfg": SceneEntityCfg("robot")})
        robot_root_rot = ObsTerm(func=base_mdp.root_quat_w, params={"asset_cfg": SceneEntityCfg("robot")})
        object_pos = ObsTerm(func=base_mdp.root_pos_w, params={"asset_cfg": SceneEntityCfg("object")})
        object_rot = ObsTerm(func=base_mdp.root_quat_w, params={"asset_cfg": SceneEntityCfg("object")})
        robot_links_state = ObsTerm(func=mdp.get_all_robot_link_state)

        left_eef_pos = ObsTerm(func=mdp.get_eef_pos, params={"link_name": "left_hand_roll_link"})
        left_eef_quat = ObsTerm(func=mdp.get_eef_quat, params={"link_name": "left_hand_roll_link"})
        right_eef_pos = ObsTerm(func=mdp.get_eef_pos, params={"link_name": "right_hand_roll_link"})
        right_eef_quat = ObsTerm(func=mdp.get_eef_quat, params={"link_name": "right_hand_roll_link"})

        hand_joint_state = ObsTerm(func=mdp.get_robot_joint_state, params={"joint_names": ["R_.*", "L_.*"]})
        head_joint_state = ObsTerm(
            func=mdp.get_robot_joint_state,
            params={"joint_names": ["head_pitch_joint", "head_roll_joint", "head_yaw_joint"]},
        )

        object = ObsTerm(
            func=mdp.object_obs,
            params={"left_eef_link_name": "left_hand_roll_link", "right_eef_link_name": "right_hand_roll_link"},
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    # observation groups
    policy: PolicyCfg = PolicyCfg()
```

**Code — task-local obs helpers (verbatim, `mdp/observations.py`).**
```python
def object_obs(env, left_eef_link_name, right_eef_link_name):
    body_pos_w = env.scene["robot"].data.body_pos_w
    left_eef_idx = env.scene["robot"].data.body_names.index(left_eef_link_name)
    right_eef_idx = env.scene["robot"].data.body_names.index(right_eef_link_name)
    left_eef_pos = body_pos_w[:, left_eef_idx] - env.scene.env_origins
    right_eef_pos = body_pos_w[:, right_eef_idx] - env.scene.env_origins

    object_pos = env.scene["object"].data.root_pos_w - env.scene.env_origins
    object_quat = env.scene["object"].data.root_quat_w

    left_eef_to_object = object_pos - left_eef_pos
    right_eef_to_object = object_pos - right_eef_pos

    return torch.cat((object_pos, object_quat, left_eef_to_object, right_eef_to_object), dim=1)


def get_eef_pos(env, link_name):
    body_pos_w = env.scene["robot"].data.body_pos_w
    left_eef_idx = env.scene["robot"].data.body_names.index(link_name)
    left_eef_pos = body_pos_w[:, left_eef_idx] - env.scene.env_origins
    return left_eef_pos


def get_eef_quat(env, link_name):
    body_quat_w = env.scene["robot"].data.body_quat_w
    left_eef_idx = env.scene["robot"].data.body_names.index(link_name)
    left_eef_quat = body_quat_w[:, left_eef_idx]
    return left_eef_quat


def get_robot_joint_state(env, joint_names):
    indexes, _ = env.scene["robot"].find_joints(joint_names)
    indexes = torch.tensor(indexes, dtype=torch.long)
    robot_joint_states = env.scene["robot"].data.joint_pos[:, indexes]
    return robot_joint_states


def get_all_robot_link_state(env):
    body_pos_w = env.scene["robot"].data.body_link_state_w[:, :, :]
    all_robot_link_pos = body_pos_w
    return all_robot_link_pos
```
> `last_action`, `joint_pos`, `root_pos_w`, `root_quat_w` come from base mdp.

**Smoke (S5).** Reset, read obs dict; assert all 15 keys present and finite, and `obs["object"].shape[-1]==13`. Not runnable in `.venv`.

---

## §6 Reward

**WARN: teleop / IL task — no shaped RL reward.** `rewards = None` in `PickPlaceGR1T2EnvCfg` (the env_cfg explicitly sets `rewards = None`, `commands = None`, `curriculum = None`). There is **no `RewardsCfg`, no `RewTerm`, no `mdp/rewards.py`** in this task directory. The only task-success signal is the boolean `success` termination (§4). Composer: **N/A**.

To turn this into an RL task you must author §6 from scratch — e.g. a dense reach/lift/place shaping built on the §5 relative `object` term (`left_eef_to_object` / `right_eef_to_object`) and the §4 `task_done_pick_place` placement box, plus a large sparse success bonus on the `success` predicate. No planning-budget docstring exists to extract.

**Smoke (S6).** N/A — no reward to assert. (If §6 is authored later, run the standard finite + non-constant + composer-equality smoke.)

---

## §7 DR

**`<no DR>`.** `EventCfg` contains only `mode="reset"` terms (`reset_all`, `reset_object`); there are no `startup` or `interval` event terms. No physics-material / mass / friction / push randomization. (The ±1 cm object x/y jitter is a reset term, already covered in §3, not domain randomization.)

**Smoke (S7).** N/A — DR-ON vs DR-OFF comparison is vacuous with no DR terms.

---

