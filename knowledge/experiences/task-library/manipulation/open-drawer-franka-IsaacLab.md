# Isaac-Open-Drawer-Franka-v0 — Implementation Spec

- robot: Franka Emika Panda (7 DoF arm + parallel gripper)
- simulator: IsaacLab (Isaac Sim, manager-based)
- objects: cabinet with drawers, table
- bimanual: false
- summary: Grasp a cabinet drawer handle and pull the drawer open.

Manager-based RL task: a Franka Panda arm must reach, align with, grasp, and pull open the **top drawer** of an articulated Sektion cabinet. The manipulated object is an `ArticulationCfg` (the cabinet) whose `drawer_top_joint` is the controlled DOF; `FrameTransformerCfg`s track the gripper TCP/fingertips and the drawer handle. Reward is a 9-term sum implementing a staged approach → align → grasp → open curriculum.

---

## §1 Registration + Scene

### Description
Registers four variants of the task; the canonical `Isaac-Open-Drawer-Franka-v0` uses joint-position control (`FrankaCabinetEnvCfg` from `joint_pos_env_cfg.py`), built on the abstract `CabinetEnvCfg`. The scene holds a Franka Panda (robot, set in the franka derived cfg), an articulated Sektion cabinet (4 joints: 2 doors + 2 drawers), an `ee_frame` FrameTransformer (TCP + 2 fingertips), a `cabinet_frame` FrameTransformer (top drawer handle), a ground plane, and a dome light.

### Decisions resolved
- `entry_point = "isaaclab.envs:ManagerBasedRLEnv"`, `env_cfg_entry_point = "...joint_pos_env_cfg:FrankaCabinetEnvCfg"`, `disable_env_checker=True`.
- Scene: `num_envs=4096`, `env_spacing=2.0`.
- Robot = `FRANKA_PANDA_CFG` (USD `{ISAACLAB_NUCLEUS_DIR}/Robots/FrankaEmika/panda_instanceable.usd`), prim_path `{ENV_REGEX_NS}/Robot`. Init joint pose: j1=0, j2=-0.569, j3=0, j4=-2.810, j5=0, j6=3.037, j7=0.741, fingers=0.04. Actuators: shoulder (j1-4) stiff=80 damp=4 effort=87; forearm (j5-7) stiff=80 damp=4 effort=12; hand (fingers) stiff=2000 damp=100 effort=200. `soft_joint_pos_limit_factor=1.0`.
- Cabinet = `ArticulationCfg`, USD `{ISAAC_NUCLEUS_DIR}/Props/Sektion_Cabinet/sektion_cabinet_instanceable.usd`, prim_path `{ENV_REGEX_NS}/Cabinet`, init pos `(0.8, 0, 0.4)`, rot `(0,0,0,1)` (note: this is `(w,x,y,z)`-ordered IsaacLab quat → a 180° yaw), all 4 joints init 0. Actuators: drawers (`drawer_top_joint`,`drawer_bottom_joint`) stiff=10 damp=1 effort=87; doors (`door_left_joint`,`door_right_joint`) stiff=10 damp=2.5 effort=87.
- `ee_frame` FrameTransformer rooted at `Robot/panda_link0`, 3 target frames in ORDER: `ee_tcp`(`panda_hand`, offset pos `(0,0,0.1034)`), `tool_leftfinger`(`panda_leftfinger`, offset `(0,0,0.046)`), `tool_rightfinger`(`panda_rightfinger`, offset `(0,0,0.046)`). debug_vis=False.
- `cabinet_frame` FrameTransformer rooted at `Cabinet/sektion`, 1 target frame `drawer_handle_top` (`Cabinet/drawer_handle_top`, offset pos `(0.305, 0.0, 0.01)`, rot `(0.5, 0.5, -0.5, -0.5)` to align with EE frame). debug_vis=True.
- `FRAME_MARKER_SMALL_CFG` = `FRAME_MARKER_CFG.copy()` with marker scale `(0.10,0.10,0.10)`.
- EnvCfg `__post_init__`: `decimation=1`, `episode_length_s=8.0`, `sim.dt=1/60` (60 Hz), `render_interval=decimation`, `physx.bounce_threshold_velocity=0.01`, `friction_correlation_distance=0.00625`. Viewer eye `(-2,2,2)` lookat `(0.8,0,0.5)`.
- PLAY variant: `num_envs=50`, `env_spacing=2.5`, `observations.policy.enable_corruption=False`.

### Code

`config/franka/__init__.py` (registration):
```python
import gymnasium as gym

from . import agents

gym.register(
    id="Isaac-Open-Drawer-Franka-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:FrankaCabinetEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:CabinetPPORunnerCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
    disable_env_checker=True,
)
```

`cabinet_env_cfg.py` — `FRAME_MARKER_SMALL_CFG` + `CabinetSceneCfg`:
```python
FRAME_MARKER_SMALL_CFG = FRAME_MARKER_CFG.copy()
FRAME_MARKER_SMALL_CFG.markers["frame"].scale = (0.10, 0.10, 0.10)


@configclass
class CabinetSceneCfg(InteractiveSceneCfg):
    """Configuration for the cabinet scene with a robot and a cabinet.

    This is the abstract base implementation, the exact scene is defined in the derived classes
    which need to set the robot and end-effector frames
    """

    # robots, Will be populated by agent env cfg
    robot: ArticulationCfg = MISSING
    # End-effector, Will be populated by agent env cfg
    ee_frame: FrameTransformerCfg = MISSING

    cabinet = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Cabinet",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Sektion_Cabinet/sektion_cabinet_instanceable.usd",
            activate_contact_sensors=False,
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.8, 0, 0.4),
            rot=(0.0, 0.0, 0.0, 1.0),
            joint_pos={
                "door_left_joint": 0.0,
                "door_right_joint": 0.0,
                "drawer_bottom_joint": 0.0,
                "drawer_top_joint": 0.0,
            },
        ),
        actuators={
            "drawers": ImplicitActuatorCfg(
                joint_names_expr=["drawer_top_joint", "drawer_bottom_joint"],
                effort_limit_sim=87.0,
                stiffness=10.0,
                damping=1.0,
            ),
            "doors": ImplicitActuatorCfg(
                joint_names_expr=["door_left_joint", "door_right_joint"],
                effort_limit_sim=87.0,
                stiffness=10.0,
                damping=2.5,
            ),
        },
    )

    # Frame definitions for the cabinet.
    cabinet_frame = FrameTransformerCfg(
        prim_path="{ENV_REGEX_NS}/Cabinet/sektion",
        debug_vis=True,
        visualizer_cfg=FRAME_MARKER_SMALL_CFG.replace(prim_path="/Visuals/CabinetFrameTransformer"),
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/Cabinet/drawer_handle_top",
                name="drawer_handle_top",
                offset=OffsetCfg(
                    pos=(0.305, 0.0, 0.01),
                    rot=(0.5, 0.5, -0.5, -0.5),  # align with end-effector frame
                ),
            ),
        ],
    )

    # plane
    plane = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        init_state=AssetBaseCfg.InitialStateCfg(),
        spawn=sim_utils.GroundPlaneCfg(),
        collision_group=-1,
    )

    # lights
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )
```

`cabinet_env_cfg.py` — `CabinetEnvCfg` (env wiring + `__post_init__`):
```python
@configclass
class CabinetEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the cabinet environment."""

    # Scene settings
    scene: CabinetSceneCfg = CabinetSceneCfg(num_envs=4096, env_spacing=2.0)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 1
        self.episode_length_s = 8.0
        self.viewer.eye = (-2.0, 2.0, 2.0)
        self.viewer.lookat = (0.8, 0.0, 0.5)
        # simulation settings
        self.sim.dt = 1 / 60  # 60Hz
        self.sim.render_interval = self.decimation
        self.sim.physx.bounce_threshold_velocity = 0.2
        self.sim.physx.bounce_threshold_velocity = 0.01
        self.sim.physx.friction_correlation_distance = 0.00625
```

`config/franka/joint_pos_env_cfg.py` — `FrankaCabinetEnvCfg` (sets robot + ee_frame + reward param overrides):
```python
@configclass
class FrankaCabinetEnvCfg(CabinetEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # Set franka as robot
        self.scene.robot = FRANKA_PANDA_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # Set Actions for the specific robot type (franka)
        self.actions.arm_action = mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=["panda_joint.*"],
            scale=1.0,
            use_default_offset=True,
        )
        self.actions.gripper_action = mdp.BinaryJointPositionActionCfg(
            asset_name="robot",
            joint_names=["panda_finger.*"],
            open_command_expr={"panda_finger_.*": 0.04},
            close_command_expr={"panda_finger_.*": 0.0},
        )

        # Listens to the required transforms
        # IMPORTANT: The order of the frames in the list is important. The first frame is the tool center point (TCP)
        # the other frames are the fingers
        self.scene.ee_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/panda_link0",
            debug_vis=False,
            visualizer_cfg=FRAME_MARKER_SMALL_CFG.replace(prim_path="/Visuals/EndEffectorFrameTransformer"),
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/panda_hand",
                    name="ee_tcp",
                    offset=OffsetCfg(
                        pos=(0.0, 0.0, 0.1034),
                    ),
                ),
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/panda_leftfinger",
                    name="tool_leftfinger",
                    offset=OffsetCfg(
                        pos=(0.0, 0.0, 0.046),
                    ),
                ),
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/panda_rightfinger",
                    name="tool_rightfinger",
                    offset=OffsetCfg(
                        pos=(0.0, 0.0, 0.046),
                    ),
                ),
            ],
        )

        # override rewards
        self.rewards.approach_gripper_handle.params["offset"] = 0.04
        self.rewards.grasp_handle.params["open_joint_pos"] = 0.04
        self.rewards.grasp_handle.params["asset_cfg"].joint_names = ["panda_finger_.*"]
```

`FRANKA_PANDA_CFG` (from `isaaclab_assets/robots/franka.py`, resolved robot articulation):
```python
FRANKA_PANDA_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAACLAB_NUCLEUS_DIR}/Robots/FrankaEmika/panda_instanceable.usd",
        activate_contact_sensors=False,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True, solver_position_iteration_count=8, solver_velocity_iteration_count=0
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={
            "panda_joint1": 0.0,
            "panda_joint2": -0.569,
            "panda_joint3": 0.0,
            "panda_joint4": -2.810,
            "panda_joint5": 0.0,
            "panda_joint6": 3.037,
            "panda_joint7": 0.741,
            "panda_finger_joint.*": 0.04,
        },
    ),
    actuators={
        "panda_shoulder": ImplicitActuatorCfg(
            joint_names_expr=["panda_joint[1-4]"],
            effort_limit_sim=87.0, stiffness=80.0, damping=4.0,
        ),
        "panda_forearm": ImplicitActuatorCfg(
            joint_names_expr=["panda_joint[5-7]"],
            effort_limit_sim=12.0, stiffness=80.0, damping=4.0,
        ),
        "panda_hand": ImplicitActuatorCfg(
            joint_names_expr=["panda_finger_joint.*"],
            effort_limit_sim=200.0, stiffness=2e3, damping=1e2,
        ),
    },
    soft_joint_pos_limit_factor=1.0,
)
```

### Smoke (§1 build)
```bash
cd <repo>
.venv/bin/python -c "import gymnasium as gym; import isaaclab_tasks; env = gym.make('Isaac-Open-Drawer-Franka-v0'); print(env.observation_space, env.action_space); env.close()"
```
Expected stdout (analytically resolved — not captured this run; Isaac Sim `pxr` unavailable in venv):
```
Box(-inf, inf, (1, 31), float32) Box(-inf, inf, (1, 8), float32)
```

---

## §2 Actions

### Description
Two action terms. The arm is driven by absolute joint-position targets on the 7 Panda arm joints (with the default joint pose added as offset, scale 1.0). The gripper is a binary open/close command mapped to the 2 finger joints.

### Decisions resolved
- `arm_action = JointPositionActionCfg(asset_name="robot", joint_names=["panda_joint.*"], scale=1.0, use_default_offset=True)` → matches `panda_joint1..7` = **7 dims**.
- `gripper_action = BinaryJointPositionActionCfg(asset_name="robot", joint_names=["panda_finger.*"], open_command_expr={"panda_finger_.*": 0.04}, close_command_expr={"panda_finger_.*": 0.0})` → **1 dim** (binary).
- Total action dim = **8**.
- Base `ActionsCfg` declares both as `MISSING`; the franka cfg fills them in.

### Code
`cabinet_env_cfg.py`:
```python
@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    arm_action: mdp.JointPositionActionCfg = MISSING
    gripper_action: mdp.BinaryJointPositionActionCfg = MISSING
```
Filled in `joint_pos_env_cfg.py` (see §1 code: `self.actions.arm_action = ...`, `self.actions.gripper_action = ...`).

### Smoke
Covered by the §1 build smoke (`action_space == Box(..., (1, 8), ...)`).

---

## §3 Reset

### Description
On every episode reset, the entire scene is restored to its configured default state, then the robot's joint positions are perturbed by a small uniform offset (±0.1 rad) with zero velocity. There is no object/goal pose randomization at reset — the cabinet always starts closed at its fixed pose.

### Decisions resolved
- `reset_all`: `mdp.reset_scene_to_default`, `mode="reset"`, no params (resets robot + cabinet to their `init_state`).
- `reset_robot_joints`: `mdp.reset_joints_by_offset`, `mode="reset"`, `position_range=(-0.1, 0.1)`, `velocity_range=(0.0, 0.0)` (applies to robot — default asset).

### Code
`cabinet_env_cfg.py` (the `mode="reset"` terms of `EventCfg`):
```python
    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "position_range": (-0.1, 0.1),
            "velocity_range": (0.0, 0.0),
        },
    )
```
(`reset_scene_to_default` and `reset_joints_by_offset` are inherited from `isaaclab.envs.mdp.events`.)

### Smoke
Behavioral reset smoke (S3): step env, call `env.reset(seed=k)` twice with the same seed, assert robot joint-pos trajectories match and lie within init±0.1; cabinet drawer joints reset to 0.

---

## §4 Goal + Termination

### Description
No explicit success/failure termination — the episode ends only on timeout (after `episode_length_s = 8.0` s, i.e. 480 steps at 60 Hz with decimation 1). "Success" (drawer fully open) is expressed entirely through the reward (§6), not a termination term. No `CommandsCfg`.

### Decisions resolved
- `time_out = DoneTerm(func=mdp.time_out, time_out=True)` — the only termination.
- `CommandsCfg`: none (env has no `commands` field).
- Episode length: 8.0 s → 480 control steps (dt=1/60, decimation=1).

### Code
`cabinet_env_cfg.py`:
```python
@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
```

### Smoke
Behavioral termination smoke (S4): roll a constant/zero-action policy for >480 steps; assert `terminated` stays False and `truncated` becomes True at step 480; no early termination.

---

## §5 Observation

### Description
Single `policy` observation group, 6 concatenated terms with corruption (noise) enabled. The agent sees its own joint pos/vel (relative to default), the top drawer joint pos/vel, the EE-to-handle distance vector, and the last action. Total dim = **31**.

### Decisions resolved
- `joint_pos = mdp.joint_pos_rel` (robot, all 9 joints) → 9
- `joint_vel = mdp.joint_vel_rel` (robot, all 9 joints) → 9
- `cabinet_joint_pos = mdp.joint_pos_rel` on `SceneEntityCfg("cabinet", joint_names=["drawer_top_joint"])` → 1
- `cabinet_joint_vel = mdp.joint_vel_rel` on same → 1
- `rel_ee_drawer_distance = mdp.rel_ee_drawer_distance` → 3
- `actions = mdp.last_action` → 8
- Total = 9+9+1+1+3+8 = **31**.
- `__post_init__`: `enable_corruption=True`, `concatenate_terms=True`. No explicit `noise` set on any term (so corruption is a no-op unless terms specify noise — none do here; PLAY disables corruption anyway).

### Code
`cabinet_env_cfg.py`:
```python
@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        cabinet_joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("cabinet", joint_names=["drawer_top_joint"])},
        )
        cabinet_joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg("cabinet", joint_names=["drawer_top_joint"])},
        )
        rel_ee_drawer_distance = ObsTerm(func=mdp.rel_ee_drawer_distance)

        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()
```

Task-local obs func (`mdp/observations.py`); `joint_pos_rel`, `joint_vel_rel`, `last_action` are inherited from `isaaclab.envs.mdp`:
```python
def rel_ee_drawer_distance(env: ManagerBasedRLEnv) -> torch.Tensor:
    """The distance between the end-effector and the object."""
    ee_tf_data: FrameTransformerData = env.scene["ee_frame"].data
    cabinet_tf_data: FrameTransformerData = env.scene["cabinet_frame"].data

    return cabinet_tf_data.target_pos_w[..., 0, :] - ee_tf_data.target_pos_w[..., 0, :]
```
(The same `mdp/observations.py` also defines unused-by-this-task helpers `rel_ee_object_distance`, `fingertips_pos`, `ee_pos`, `ee_quat` — not referenced by `Isaac-Open-Drawer-Franka-v0`'s `ObservationsCfg`.)

### Smoke
Covered by §1 build smoke (`observation_space == Box(..., (1, 31), ...)`).

---

## §6 Reward

### Description
Sum-composed reward (9 terms) implementing a staged manipulation curriculum:
1. **Approach** the handle (`approach_ee_handle` w=2.0, inverse-square shaped; `align_ee_handle` w=0.5, gripper-vs-handle orientation alignment).
2. **Grasp** (`approach_gripper_handle` w=5.0, fingertip-to-handle distance only when in a graspable pose; `align_grasp_around_handle` w=0.125, bonus when left finger above / right finger below the handle; `grasp_handle` w=0.5, reward closing fingers when close to handle).
3. **Open** (`open_drawer_bonus` w=7.5, drawer-position bonus doubled when graspable; `multi_stage_open_drawer` w=1.0, three discrete open-progress stages).
4. **Penalties** (`action_rate_l2` w=-0.01; `joint_vel_l2` w=-0.0001).

Composer = **sum** (IsaacLab `RewardManager` sums all `RewTerm`s).

### Decisions resolved (RewardsCfg)
| term | func | weight | params |
|---|---|---|---|
| approach_ee_handle | `mdp.approach_ee_handle` | 2.0 | `threshold=0.2` |
| align_ee_handle | `mdp.align_ee_handle` | 0.5 | — |
| approach_gripper_handle | `mdp.approach_gripper_handle` | 5.0 | `offset=0.04` (set in franka cfg; base = MISSING) |
| align_grasp_around_handle | `mdp.align_grasp_around_handle` | 0.125 | — |
| grasp_handle | `mdp.grasp_handle` | 0.5 | `threshold=0.03`, `open_joint_pos=0.04` (franka), `asset_cfg=SceneEntityCfg("robot", joint_names=["panda_finger_.*"])` (franka) |
| open_drawer_bonus | `mdp.open_drawer_bonus` | 7.5 | `asset_cfg=SceneEntityCfg("cabinet", joint_names=["drawer_top_joint"])` |
| multi_stage_open_drawer | `mdp.multi_stage_open_drawer` | 1.0 | `asset_cfg=SceneEntityCfg("cabinet", joint_names=["drawer_top_joint"])` |
| action_rate_l2 | `mdp.action_rate_l2` | -1e-2 | — |
| joint_vel | `mdp.joint_vel_l2` | -1e-4 | — |

Planning-budget (per-step saturated magnitudes, **retro-computed** from weights, pre-`dt`-scaling; no docstring budget present in source):
- approach_ee_handle: max ≈ 2.0·2 = 4.0 (distance ≤ threshold doubles the inverse-square term which saturates near 1).
- align_ee_handle: max ≈ 0.5·1.0 = 0.5 (perfect alignment).
- approach_gripper_handle: ≈ 5.0·(2·offset) = 5.0·0.08 = 0.4 when perfectly centered + graspable.
- align_grasp_around_handle: 0.125 when graspable (boolean).
- grasp_handle: ≈ 0.5·(2·0.04) = 0.04 when fully closed near handle.
- open_drawer_bonus: 7.5·(1+1)·drawer_pos → ≈ 7.5·2·0.4 = 6.0 at full open (~0.4 m) + graspable.
- multi_stage_open_drawer: max 1.0·(0.5+1+1)=2.5 at full open + graspable.
- action_rate_l2 / joint_vel: small negative regularizers.

### Code

`cabinet_env_cfg.py` — `RewardsCfg`:
```python
@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # 1. Approach the handle
    approach_ee_handle = RewTerm(func=mdp.approach_ee_handle, weight=2.0, params={"threshold": 0.2})
    align_ee_handle = RewTerm(func=mdp.align_ee_handle, weight=0.5)

    # 2. Grasp the handle
    approach_gripper_handle = RewTerm(func=mdp.approach_gripper_handle, weight=5.0, params={"offset": MISSING})
    align_grasp_around_handle = RewTerm(func=mdp.align_grasp_around_handle, weight=0.125)
    grasp_handle = RewTerm(
        func=mdp.grasp_handle,
        weight=0.5,
        params={
            "threshold": 0.03,
            "open_joint_pos": MISSING,
            "asset_cfg": SceneEntityCfg("robot", joint_names=MISSING),
        },
    )

    # 3. Open the drawer
    open_drawer_bonus = RewTerm(
        func=mdp.open_drawer_bonus,
        weight=7.5,
        params={"asset_cfg": SceneEntityCfg("cabinet", joint_names=["drawer_top_joint"])},
    )
    multi_stage_open_drawer = RewTerm(
        func=mdp.multi_stage_open_drawer,
        weight=1.0,
        params={"asset_cfg": SceneEntityCfg("cabinet", joint_names=["drawer_top_joint"])},
    )

    # 4. Penalize actions for cosmetic reasons
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-1e-2)
    joint_vel = RewTerm(func=mdp.joint_vel_l2, weight=-0.0001)
```

`mdp/rewards.py` — FULL verbatim source of every task-local reward function:
```python
# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import matrix_from_quat

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def approach_ee_handle(env: ManagerBasedRLEnv, threshold: float) -> torch.Tensor:
    r"""Reward the robot for reaching the drawer handle using inverse-square law.

    It uses a piecewise function to reward the robot for reaching the handle.

    .. math::

        reward = \begin{cases}
            2 * (1 / (1 + distance^2))^2 & \text{if } distance \leq threshold \\
            (1 / (1 + distance^2))^2 & \text{otherwise}
        \end{cases}

    """
    ee_tcp_pos = env.scene["ee_frame"].data.target_pos_w[..., 0, :]
    handle_pos = env.scene["cabinet_frame"].data.target_pos_w[..., 0, :]

    # Compute the distance of the end-effector to the handle
    distance = torch.norm(handle_pos - ee_tcp_pos, dim=-1, p=2)

    # Reward the robot for reaching the handle
    reward = 1.0 / (1.0 + distance**2)
    reward = torch.pow(reward, 2)
    return torch.where(distance <= threshold, 2 * reward, reward)


def align_ee_handle(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Reward for aligning the end-effector with the handle.

    The reward is based on the alignment of the gripper with the handle. It is computed as follows:

    .. math::

        reward = 0.5 * (align_z^2 + align_x^2)

    where :math:`align_z` is the dot product of the z direction of the gripper and the -x direction of the handle
    and :math:`align_x` is the dot product of the x direction of the gripper and the -y direction of the handle.
    """
    ee_tcp_quat = env.scene["ee_frame"].data.target_quat_w[..., 0, :]
    handle_quat = env.scene["cabinet_frame"].data.target_quat_w[..., 0, :]

    ee_tcp_rot_mat = matrix_from_quat(ee_tcp_quat)
    handle_mat = matrix_from_quat(handle_quat)

    # get current x and y direction of the handle
    handle_x, handle_y = handle_mat[..., 0], handle_mat[..., 1]
    # get current x and z direction of the gripper
    ee_tcp_x, ee_tcp_z = ee_tcp_rot_mat[..., 0], ee_tcp_rot_mat[..., 2]

    # make sure gripper aligns with the handle
    # in this case, the z direction of the gripper should be close to the -x direction of the handle
    # and the x direction of the gripper should be close to the -y direction of the handle
    # dot product of z and x should be large
    align_z = torch.bmm(ee_tcp_z.unsqueeze(1), -handle_x.unsqueeze(-1)).squeeze(-1).squeeze(-1)
    align_x = torch.bmm(ee_tcp_x.unsqueeze(1), -handle_y.unsqueeze(-1)).squeeze(-1).squeeze(-1)
    return 0.5 * (torch.sign(align_z) * align_z**2 + torch.sign(align_x) * align_x**2)


def align_grasp_around_handle(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Bonus for correct hand orientation around the handle.

    The correct hand orientation is when the left finger is above the handle and the right finger is below the handle.
    """
    # Target object position: (num_envs, 3)
    handle_pos = env.scene["cabinet_frame"].data.target_pos_w[..., 0, :]
    # Fingertips position: (num_envs, n_fingertips, 3)
    ee_fingertips_w = env.scene["ee_frame"].data.target_pos_w[..., 1:, :]
    lfinger_pos = ee_fingertips_w[..., 0, :]
    rfinger_pos = ee_fingertips_w[..., 1, :]

    # Check if hand is in a graspable pose
    is_graspable = (rfinger_pos[:, 2] < handle_pos[:, 2]) & (lfinger_pos[:, 2] > handle_pos[:, 2])

    # bonus if left finger is above the drawer handle and right below
    return is_graspable


def approach_gripper_handle(env: ManagerBasedRLEnv, offset: float = 0.04) -> torch.Tensor:
    """Reward the robot's gripper reaching the drawer handle with the right pose.

    This function returns the distance of fingertips to the handle when the fingers are in a grasping orientation
    (i.e., the left finger is above the handle and the right finger is below the handle). Otherwise, it returns zero.
    """
    # Target object position: (num_envs, 3)
    handle_pos = env.scene["cabinet_frame"].data.target_pos_w[..., 0, :]
    # Fingertips position: (num_envs, n_fingertips, 3)
    ee_fingertips_w = env.scene["ee_frame"].data.target_pos_w[..., 1:, :]
    lfinger_pos = ee_fingertips_w[..., 0, :]
    rfinger_pos = ee_fingertips_w[..., 1, :]

    # Compute the distance of each finger from the handle
    lfinger_dist = torch.abs(lfinger_pos[:, 2] - handle_pos[:, 2])
    rfinger_dist = torch.abs(rfinger_pos[:, 2] - handle_pos[:, 2])

    # Check if hand is in a graspable pose
    is_graspable = (rfinger_pos[:, 2] < handle_pos[:, 2]) & (lfinger_pos[:, 2] > handle_pos[:, 2])

    return is_graspable * ((offset - lfinger_dist) + (offset - rfinger_dist))


def grasp_handle(
    env: ManagerBasedRLEnv, threshold: float, open_joint_pos: float, asset_cfg: SceneEntityCfg
) -> torch.Tensor:
    """Reward for closing the fingers when being close to the handle.

    The :attr:`threshold` is the distance from the handle at which the fingers should be closed.
    The :attr:`open_joint_pos` is the joint position when the fingers are open.

    Note:
        It is assumed that zero joint position corresponds to the fingers being closed.
    """
    ee_tcp_pos = env.scene["ee_frame"].data.target_pos_w[..., 0, :]
    handle_pos = env.scene["cabinet_frame"].data.target_pos_w[..., 0, :]
    gripper_joint_pos = env.scene[asset_cfg.name].data.joint_pos[:, asset_cfg.joint_ids]

    distance = torch.norm(handle_pos - ee_tcp_pos, dim=-1, p=2)
    is_close = distance <= threshold

    return is_close * torch.sum(open_joint_pos - gripper_joint_pos, dim=-1)


def open_drawer_bonus(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Bonus for opening the drawer given by the joint position of the drawer.

    The bonus is given when the drawer is open. If the grasp is around the handle, the bonus is doubled.
    """
    drawer_pos = env.scene[asset_cfg.name].data.joint_pos[:, asset_cfg.joint_ids[0]]
    is_graspable = align_grasp_around_handle(env).float()

    return (is_graspable + 1.0) * drawer_pos


def multi_stage_open_drawer(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Multi-stage bonus for opening the drawer.

    Depending on the drawer's position, the reward is given in three stages: easy, medium, and hard.
    This helps the agent to learn to open the drawer in a controlled manner.
    """
    drawer_pos = env.scene[asset_cfg.name].data.joint_pos[:, asset_cfg.joint_ids[0]]
    is_graspable = align_grasp_around_handle(env).float()

    open_easy = (drawer_pos > 0.01) * 0.5
    open_medium = (drawer_pos > 0.2) * is_graspable
    open_hard = (drawer_pos > 0.3) * is_graspable

    return open_easy + open_medium + open_hard
```

The penalty/regularizer terms `action_rate_l2`, `joint_vel_l2`, and the timeout `time_out` are inherited from `isaaclab.envs.mdp` (not task-local).

### Smoke
§6 reward smoke (S6): wrap env with per-term reward log; over a random rollout assert each term is finite and the per-step **sum** of the 9 weighted terms equals `info["detailed_reward"]`-composed total (composer = `sum`); assert reward is non-constant.

---

## §7 DR

### Description
Domain randomization is limited to **startup** physics-material randomization on the robot bodies and the drawer handle. There are no `interval`-mode events. (The two `reset`-mode events are §3, not DR.)

### Decisions resolved
- `robot_physics_material`: `mdp.randomize_rigid_body_material`, `mode="startup"`, `asset_cfg=SceneEntityCfg("robot", body_names=".*")`, `static_friction_range=(0.8,1.25)`, `dynamic_friction_range=(0.8,1.25)`, `restitution_range=(0.0,0.0)`, `num_buckets=16`.
- `cabinet_physics_material`: `mdp.randomize_rigid_body_material`, `mode="startup"`, `asset_cfg=SceneEntityCfg("cabinet", body_names="drawer_handle_top")`, `static_friction_range=(1.0,1.25)`, `dynamic_friction_range=(1.25,1.5)`, `restitution_range=(0.0,0.0)`, `num_buckets=16`.
- No interval DR. (PLAY variant only disables obs corruption, not these startup events.)

### Code
`cabinet_env_cfg.py` (the `mode="startup"` terms of `EventCfg`):
```python
    robot_physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.8, 1.25),
            "dynamic_friction_range": (0.8, 1.25),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 16,
        },
    )

    cabinet_physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("cabinet", body_names="drawer_handle_top"),
            "static_friction_range": (1.0, 1.25),
            "dynamic_friction_range": (1.25, 1.5),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 16,
        },
    )
```
(`randomize_rigid_body_material` is a `ManagerTermBase` class in `isaaclab.envs.mdp.events`.)

### Smoke
§7 DR smoke (S7): because DR is **startup-only** physics-material (no obs perturbation), seed-matched obs trajectories with DR ON vs OFF will NOT necessarily diverge in early steps — physics-material DR only manifests through contact dynamics. Mark this smoke "weak/skipped per startup-only friction DR; verify via material-buffer inspection if needed."

---

