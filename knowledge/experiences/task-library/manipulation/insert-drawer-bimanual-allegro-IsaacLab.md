# InsertDrawer — Implementation Spec

- robot: Bimanual UF850 arms + dual Allegro hands (44 DoF)
- simulator: IsaacLab (Isaac Sim, manager-based)
- objects: drawer, rigid object, table
- bimanual: true
- summary: Place an object into a drawer using two dexterous arms.

This spec captures the design of bimanual_suite `InsertDrawerEnv-v0`. This InsertDrawer is the original from which IsaacLab's `dex_grasp` / `dex_pickplace` tasks were vendored (the right-robot half: USD, init pose, actuator gain groups, EMA cumulative-relative action).

## Task summary

Two UF850 arms, each tipped with an Allegro hand (`Robot` = right, `Robot_left` = left), stand behind a table. A small rigid object (`dog.usd`, mass 0.11 kg) sits on the table in front of the right hand; a `drawer` articulation (single prismatic `base_drawer_joint`) sits to the front-left. The bimanual task is a coordinated **pick-and-insert**: the RIGHT hand grasps and lifts the object, the LEFT hand reaches the drawer handle and pulls it open, then the object is deposited **inside** the drawer, and the drawer is pushed closed with the object retained. Success is a **consecutive-success** counter: the drawer must be nearly closed (`base_drawer_joint < 0.1`) AND the object registered "in drawer" (fingertips released + object within 0.2 m of the drawer body below 1.0 m height) for **20 consecutive steps**. The environment is `C2`-symmetric — half the envs are mirror-reset each episode and a full symmetric copy of every reward/observation/sensor exists so the same policy handles the left↔right mirror. Control is a per-joint **EMA cumulative-relative joint-position** action (α = 0.2) over all 44 joints (22 per arm). Physics run at 120 Hz with decimation 6 → 20 Hz control; episodes are 8.3333 s.

---

## §1 Registration + Scene

**Description.** Registered as gym id `InsertDrawerEnv-v0`, entry point `bimanual_suite.env.tasks.InsertDrawer.env:InsertDrawerEnv`, cfg `InsertDrawerEnvCfg`. The custom `InsertDrawerEnv` subclasses `BaseEnv` (itself a subclass of IsaacLab's `ManagerBasedRLEnv`). The scene (`InsertDrawerSceneCfg`) holds **both robots**, one rigid object, a drawer articulation, a table (from `BaseSceneCfg`), ground + dome light (from `BaseSceneCfg`), and **32 contact sensors** (8 base pairs × right/left × object/drawer-handle, plus a full symmetric mirror set). Sim: `dt = 1/120`, `decimation = 6`, `episode_length_s = 8.3333`, `num_envs = 4096`, `env_spacing = 3.0`, PhysX rigid-contact/patch counts `2**24`, scene friction static 1.5 / dynamic 1.0 / restitution 0. `replicate_physics = False`. All robot/drawer bodies spawn with `disable_gravity=True`; the object has gravity enabled; the table is kinematic.

### Decisions resolved

| Decision | Value | Source |
|---|---|---|
| gym id / entry point | `InsertDrawerEnv-v0` → `...InsertDrawer.env:InsertDrawerEnv` | `env/__init__.py:36-42` |
| env class base | `InsertDrawerEnv(BaseEnv)`, `BaseEnv(ManagerBasedRLEnv)` | `env.py:8`, `manager_based_env.py:38` |
| num_envs / env_spacing | 4096 / 3.0 | `env_cfg.py:590` |
| right robot USD | `{LIB_PATH}/assets/ufactory850/uf850_allegro_right_colored.usd` | `env_cfg.py:28` |
| left robot USD | `{LIB_PATH}/assets/ufactory850/uf850_allegro_left_colored.usd` | `env_cfg.py:120` |
| right robot init pos | `(-0.274, -0.475, 0.01)` | `env_cfg.py:66` |
| left robot init pos | `(-0.274, 0.475, 0.01)` | `env_cfg.py:158` |
| right arm init joints | j1=0.8, j2=0.3, j3=-0.6, j4=0.0, j5=-0.8, j6=-1.57 | `env_cfg.py:42-47` |
| left arm init joints | j1=-0.8, j2=0.3, j3=-0.6, j4=0.0, j5=-0.8, j6=1.57 (mirror of j1,j6) | `env_cfg.py:134-139` |
| hand init joints (both) | jXf1=0, jXf2=0.4, jXf3=0.4, jXf4=0; jth1=1.3, jth2=0, jth3=0.2, jth4=0 | `env_cfg.py:48-64` |
| object USD / mass / scale | `assets/object/dog.usd` / 0.11 kg / (1,1,1); dynamic (kinematic_enabled=False, gravity on) | `env_cfg.py:211-224` |
| drawer USD / scale | `assets/object/drawer.usd` / (0.3, 0.6, 0.5); articulation, gravity disabled | `env_cfg.py:236-247` |
| drawer joint | `base_drawer_joint` (prismatic), init 0.0, stiffness 10 / damping 1 / friction 1 / effort 87 / vel 100 | `env_cfg.py:250-264` |
| table | `assets/object/table.usd`, kinematic, pos (0,0,0), rot (0.70710678,0,0,0.70710678) | `manager_based_env_cfg.py:84-99` |
| ground | GroundPlane at z = -0.82 | `manager_based_env_cfg.py:71-75` |
| light | DomeLight color (0.75,0.75,0.75) intensity 2500 | `manager_based_env_cfg.py:78-81` |
| contact sensors | 32 total: 4 fingertips (if5/mf5/pf5/th5) × {right→object, left→handle} + symmetric mirror set | `env_cfg.py:269-365` |
| ee frame bodies | `palm_link` (obs/reward), `handle_grip` (drawer handle), `drawer` body (in-drawer check) | `env_cfg.py:436,454`, `mdps.py:26,91` |
| markers | FRAME_MARKER debug only, gated on `visualize_marker=False` | `env_cfg.py:19-20,610`, `env.py:23-28` |
| sim dt / decimation / ep len | 1/120 / 6 / 8.3333 s (→ 20 Hz control) | `manager_based_env_cfg.py:206-211` |
| physx knobs | gpu_max_rigid_contact_count = gpu_max_rigid_patch_count = 2**24 | `manager_based_env_cfg.py:197-200` |
| scene friction | static 1.5 / dynamic 1.0 / restitution 0 (combine=min) | `manager_based_env_cfg.py:190-196` |
| actuator groups (per arm) | arm j1-6: k=2000 d=16; hand f1: 325/20; f2: 425/25; f3: 245/15; f4: 1050/65; thumb th1: 100/5; th2: 300/15; th3: 1270/100; th4: 1000/50 | `env_cfg.py:68-113` |
| articulation props | self_collisions off, solver pos iter 16 / vel iter 1; max lin/ang vel 1000; max depenetration 1000 | `env_cfg.py:30-38` |

### Code

Registration (`env/__init__.py:35-43`):

```python
from .tasks.InsertDrawer.env_cfg import InsertDrawerEnvCfg
gym.register(
    id="InsertDrawerEnv-v0",
    entry_point="bimanual_suite.env.tasks.InsertDrawer.env:InsertDrawerEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": InsertDrawerEnvCfg,
    },
)
```

Custom env class + step/success logic (`InsertDrawer/env.py`):

```python
import torch
from typing import Any, ClassVar

from bimanual_suite.env.tasks.manager_based_env import *
from bimanual_suite.env.tasks.InsertDrawer.env_cfg import InsertDrawerEnvCfg


class InsertDrawerEnv(BaseEnv):
    is_vector_env: ClassVar[bool] = True
    """Whether the environment is a vectorized environment."""
    metadata: ClassVar[dict[str, Any]] = {
        "render_modes": [None, "human", "rgb_array"],
        "isaac_sim_version": get_version(),
    }
    """Metadata for the environment."""

    cfg: InsertDrawerEnvCfg
    """Configuration for the environment."""
    
    def step(self, action: torch.Tensor) -> VecEnvStepReturn:
        super().step(action)
        # Debug only
        if self.cfg.visualize_marker:
            self.markers['arm_r']['ee_marker'].visualize(self.scene["drawer"].data.root_pos_w, self.scene["drawer"].data.root_quat_w)
            ee_idx = self.scene["drawer"].find_bodies("handle_grip")[0]
            self.markers['arm_r']['goal_marker'].visualize(self.scene["drawer"].data.body_state_w[:, ee_idx, :3].reshape(-1, 3), self.scene["drawer"].data.body_state_w[:, ee_idx, 3:7].reshape(-1, 4))
            # self.markers['arm_l']['ee_marker'].visualize(right_bottom, self.scene["robot"].data.root_quat_w)
            # self.markers['arm_l']['goal_marker'].visualize(right_top, self.scene["robot"].data.root_quat_w)

        # return observations, rewards, resets and extras
        return self.obs_buf, self.reward_buf, self.reset_terminated, self.reset_time_outs, self.extras

    def _pre_init_process(self):
        super()._pre_init_process()
        self.success_tracker_step = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)

    def _post_reset_process(self, env_ids):
        super()._post_reset_process(env_ids)
        self.success_tracker_step[env_ids] = 0.0
```

`BaseEnv.step()` — the action scaling + success/detailed-reward extras + post-reset hook (`manager_based_env.py:161-176`):

```python
    def step(self, action: torch.Tensor) -> VecEnvStepReturn:
        # update last action inferred from the policy. this is different from the last action used in IsaacLab action manager
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
        
        # return observations, rewards, resets and extras
        return self.obs_buf, self.reward_buf, self.reset_terminated, self.reset_time_outs, self.extras
```

Scene cfg — robots, object, drawer (`InsertDrawer/env_cfg.py:22-266`):

```python
FRAME_MARKER_SMALL_CFG = FRAME_MARKER_CFG.copy()
FRAME_MARKER_SMALL_CFG.markers["frame"].scale = (0.10, 0.10, 0.10)

@configclass
class InsertDrawerSceneCfg(BaseSceneCfg):
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
                "joint1": 0.8,
                "joint2": 0.3,
                "joint3": -0.6,
                "joint4": 0.0,
                "joint5": -0.8,
                "joint6": -1.57,
                # hand 
                "jif1": 0.0,
                "jif2": 0.4,
                "jif3": 0.4,
                "jif4": 0.0,
                "jmf1": 0.0,
                "jmf2": 0.4,
                "jmf3": 0.4,
                "jmf4": 0.0,
                "jpf1": 0.0,
                "jpf2": 0.4,
                "jpf3": 0.4,
                "jpf4": 0.0,
                "jth1": 1.3,
                "jth2": 0.0,
                "jth3": 0.2,
                "jth4": 0.0,
            },
            pos=(-0.274, -0.475, 0.01),
        ),
        actuators={
            "xArm_1-6": ImplicitActuatorCfg(
                joint_names_expr=["joint[1-6]"],
                stiffness=2000.0,
                damping=16.0,
            ),
            "allegro_hand_1": ImplicitActuatorCfg(
                joint_names_expr=["j.*f1"],
                stiffness=325.0,
                damping=20.0,
            ),
            "allegro_hand_2": ImplicitActuatorCfg(
                joint_names_expr=["j.*f2"],
                stiffness=425.0,
                damping=25.0,
            ),
            "allegro_hand_3": ImplicitActuatorCfg(
                joint_names_expr=["j.*f3"],
                stiffness=245.0,
                damping=15.0,
            ),
            "allegro_hand_4": ImplicitActuatorCfg(
                joint_names_expr=["j.*f4"],
                stiffness=1050.0,
                damping=65.0,
            ),
            "allegro_hand_thumb_1": ImplicitActuatorCfg(
                joint_names_expr=["jth1"],
                stiffness=100.0,
                damping=5.0,
            ),
            "allegro_hand_thumb_2": ImplicitActuatorCfg(
                joint_names_expr=["jth2"],
                stiffness=300.0,
                damping=15.0,
            ),
            "allegro_hand_thumb_3": ImplicitActuatorCfg(
                joint_names_expr=["jth3"],
                stiffness=1270.0,
                damping=100.0,
            ),
            "allegro_hand_thumb_4": ImplicitActuatorCfg(
                joint_names_expr=["jth4"],
                stiffness=1000.0,
                damping=50.0,
            ),
        },
    )

    robot_left = ArticulationCfg(
        prim_path="/World/envs/env_.*/Robot_left",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{bimanual_suite.LIB_PATH}/assets/ufactory850/uf850_allegro_left_colored.usd",
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
                "joint1": -0.8,
                "joint2": 0.3,
                "joint3": -0.6,
                "joint4": 0.0,
                "joint5": -0.8,
                "joint6": 1.57,
                # hand 
                "jif1": 0.0,
                "jif2": 0.4,
                "jif3": 0.4,
                "jif4": 0.0,
                "jmf1": 0.0,
                "jmf2": 0.4,
                "jmf3": 0.4,
                "jmf4": 0.0,
                "jpf1": 0.0,
                "jpf2": 0.4,
                "jpf3": 0.4,
                "jpf4": 0.0,
                "jth1": 1.3,
                "jth2": 0.0,
                "jth3": 0.2,
                "jth4": 0.0,
            },
            pos=(-0.274, 0.475, 0.01),
        ),
        actuators={
            "xArm_1-6": ImplicitActuatorCfg(
                joint_names_expr=["joint[1-6]"],
                stiffness=2000.0,
                damping=16.0,
            ),
            "allegro_hand_1": ImplicitActuatorCfg(
                joint_names_expr=["j.*f1"],
                stiffness=325.0,
                damping=20.0,
            ),
            "allegro_hand_2": ImplicitActuatorCfg(
                joint_names_expr=["j.*f2"],
                stiffness=425.0,
                damping=25.0,
            ),
            "allegro_hand_3": ImplicitActuatorCfg(
                joint_names_expr=["j.*f3"],
                stiffness=245.0,
                damping=15.0,
            ),
            "allegro_hand_4": ImplicitActuatorCfg(
                joint_names_expr=["j.*f4"],
                stiffness=1050.0,
                damping=65.0,
            ),
            "allegro_hand_thumb_1": ImplicitActuatorCfg(
                joint_names_expr=["jth1"],
                stiffness=100.0,
                damping=5.0,
            ),
            "allegro_hand_thumb_2": ImplicitActuatorCfg(
                joint_names_expr=["jth2"],
                stiffness=300.0,
                damping=15.0,
            ),
            "allegro_hand_thumb_3": ImplicitActuatorCfg(
                joint_names_expr=["jth3"],
                stiffness=1270.0,
                damping=100.0,
            ),
            "allegro_hand_thumb_4": ImplicitActuatorCfg(
                joint_names_expr=["jth4"],
                stiffness=1000.0,
                damping=50.0,
            ),
        },
    )

    object_0 = RigidObjectCfg(
        prim_path=f"/World/envs/env_.*/Object_0",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{bimanual_suite.LIB_PATH}/assets/object/dog.usd",
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
            scale=(1.0, 1.0, 1.0),
        ), # wait for initialization
        init_state=RigidObjectCfg.InitialStateCfg(
            lin_vel=(0.0, 0.0, 0.0),
            ang_vel=(0.0, 0.0, 0.0),
            pos=(0.0, 0.0, 0.0),
        ),
    )

    drawer = ArticulationCfg(
        prim_path=f"/World/envs/env_.*/Drawer",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{bimanual_suite.LIB_PATH}/assets/object/drawer.usd",
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
            scale=(0.3, 0.6, 0.5),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            joint_pos={
                "base_drawer_joint": 0.0,
            },
            pos=(0.0, 0.0, 0.0),
            rot=(0.0, 0.0, 0.0, 1.0),
        ),
        actuators={
            "joint": ImplicitActuatorCfg(
                joint_names_expr=["base_drawer_joint"],
                effort_limit=87.0,
                velocity_limit=100.0,
                stiffness=10.0,
                damping=1.0,
                friction=1.0,
            ),
        },
    )
```

Contact sensors — right/left, object/handle, plus symmetric mirror set (`InsertDrawer/env_cfg.py:268-365`):

```python
    # sensors
    contact_sensors_0 = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/if5",  # index
        update_period=0.0, 
        debug_vis=True,
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Object_0"],
    )
    contact_sensors_1 = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/mf5",  # middle
        update_period=0.0, 
        debug_vis=True,
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Object_0"],
    )
    contact_sensors_2 = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/pf5",  # pinky
        update_period=0.0, 
        debug_vis=True,
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Object_0"],
    )
    contact_sensors_3 = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/th5",  # thumb
        update_period=0.0, 
        debug_vis=True,
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Object_0"],
    )
    contact_sensors_0_left = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot_left/if5",  # index
        update_period=0.0, 
        debug_vis=True,
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Drawer/handle_grip"],
    )
    contact_sensors_1_left = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot_left/mf5",  # middle
        update_period=0.0, 
        debug_vis=True,
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Drawer/handle_grip"],
    )
    contact_sensors_2_left = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot_left/pf5",  # pinky
        update_period=0.0, 
        debug_vis=True,
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Drawer/handle_grip"],
    )
    contact_sensors_3_left = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot_left/th5",  # thumb
        update_period=0.0, 
        debug_vis=True,
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Drawer/handle_grip"],
    )
    # symmetric sensors
    contact_sensors_0_symmetry = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot_left/if5",  # index
        update_period=0.0, 
        debug_vis=True,
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Object_0"],
    )
    contact_sensors_1_symmetry = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot_left/mf5",  # middle
        update_period=0.0, 
        debug_vis=True,
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Object_0"],
    )
    contact_sensors_2_symmetry = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot_left/pf5",  # pinky
        update_period=0.0, 
        debug_vis=True,
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Object_0"],
    )
    contact_sensors_3_symmetry = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot_left/th5",  # thumb
        update_period=0.0, 
        debug_vis=True,
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Object_0"],
    )
    contact_sensors_0_left_symmetry = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/if5",  # index
        update_period=0.0, 
        debug_vis=True,
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Drawer/handle_grip"],
    )
    contact_sensors_1_left_symmetry = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/mf5",  # middle
        update_period=0.0, 
        debug_vis=True,
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Drawer/handle_grip"],
    )
    contact_sensors_2_left_symmetry = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/pf5",  # pinky
        update_period=0.0, 
        debug_vis=True,
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Drawer/handle_grip"],
    )
    contact_sensors_3_left_symmetry = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/th5",  # thumb
        update_period=0.0, 
        debug_vis=True,
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Drawer/handle_grip"],
    )
```

Shared scene base — ground, light, table, PhysX/sim timing (`manager_based_env_cfg.py:66-101, 190-212`):

```python
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
```

```python
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

Env cfg assembly (`InsertDrawer/env_cfg.py:587-615`):

```python
@configclass
class InsertDrawerEnvCfg(BaseEnvCfg):
    name: str = "InsertDrawer"
    scene = InsertDrawerSceneCfg(num_envs=4096, env_spacing=3.0)
    events = InsertDrawerEventCfg()
    commands = InsertDrawerCommandsCfg()
    observations = InsertDrawerObservationsCfg()
    actions = InsertDrawerActionsCfg()
    terminations = InsertDrawerTerminationsCfg()
    rewards = InsertDrawerRewardsCfg()
    num_object = 1
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
        self.viewer.eye = (-3.5, 0.0, 3.5)
```

---

## §2 Actions

**Description.** Two `EMACumulativeRelativeJointPositionActionCfg` terms, one per robot (`robot`, `robot_left`), each matching all joints `[".*"]` → 22 joints/arm → **action_dim = 44** (right 22 then left 22). `scale=1.0` inside the action term, `alpha=0.2`, `use_default_offset=False`. Per-arm hard clamp limits are the `JOINT_LOWER/UPPER_LIMIT` (right) and `JOINT_LOWER/UPPER_LIMIT_LEFT` (left) lists (note the left lists mirror the index/pinky abduction limits of the hand — see the `jif1/jpf1` swap). A separate per-joint `action_scale` list (44 floats, arm 0.05 / hand 0.03 / `jth3` 0.015) is applied in `BaseEnv.step()` **before** the action reaches the manager (`action = action * self._scale`), so the raw policy action is multiplied by `action_scale` then fed through the EMA cumulative-relative rule below.

**Per-step rule** (`process_actions`, applied every control step for each arm):
1. `super().process_actions(actions)` → affine `raw * scale(=1.0) + offset(=0)` into `_processed_actions`.
2. `_processed_actions += del_action` (accumulated cumulative delta).
3. `del_action = _processed_actions.clone()` (carry the running sum forward).
4. `_processed_actions += init_joint_pos` (offset by the joint pose at last reset).
5. `ema = alpha * _processed_actions + (1-alpha) * _prev_applied_actions`.
6. Clamp `ema` to `[joint_lower_limit, joint_upper_limit]` → position targets.
7. `_prev_applied_actions = _processed_actions`.

On reset, `_prev_applied_actions`, `del_action`, and `init_joint_pos` are re-seeded to the current joint positions.

### Decisions resolved

| Decision | Value | Source |
|---|---|---|
| action term class | `EMACumulativeRelativeJointPositionAction` (subclass of IsaacLab `JointPositionAction`) | `actions.py:17`, `actions_cfg.py:15` |
| terms | `arm_hand_action` (robot), `arm_hand_action_left` (robot_left) | `env_cfg.py:467-486` |
| joint_names | `[".*"]` (all 22 joints per arm) | `env_cfg.py:469,480` |
| scale (in-term) | 1.0 | `env_cfg.py:470,481` |
| use_default_offset | False | `env_cfg.py:471,482` |
| alpha (EMA) | 0.2 | `env_cfg.py:475,485` |
| action_dim | 44 (22 right + 22 left) | `env_cfg.py:598` |
| clamp limits (right) | `JOINT_LOWER_LIMIT` / `JOINT_UPPER_LIMIT` | `env_cfg.py:473-474`, `manager_based_env_cfg.py:29-46` |
| clamp limits (left) | `JOINT_LOWER_LIMIT_LEFT` / `JOINT_UPPER_LIMIT_LEFT` | `env_cfg.py:483-484`, `manager_based_env_cfg.py:47-64` |
| per-joint action_scale | applied in `BaseEnv.step` as `action * self._scale` (44-vec) | `manager_based_env.py:165`, `env_cfg.py:599-608` |

### Code

Per-joint limit constants (`manager_based_env_cfg.py:29-64`):

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
```

Action cfg (`InsertDrawer/env_cfg.py:466-486`):

```python
@configclass
class InsertDrawerActionsCfg:
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

Action cfg class (`action_managers/actions_cfg.py`):

```python
@configclass
class EMACumulativeRelativeJointPositionActionCfg(JointPositionActionCfg):
    """Configuration for the binary joint position action term.

    See :class:`JointPositionAction` for more details.
    """
    class_type: type[ActionTerm] = EMACumulativeRelativeJointPositionAction
    """Class type."""

    alpha: float | dict[str, float] = 1.0
    """The weight for the moving average (float or dict of regex expressions). Defaults to 1.0.

    If set to 1.0, the processed action is applied directly without any moving average window.
    """

    joint_lower_limit: list[float] = None
    joint_upper_limit: list[float] = None
    """The lower and upper limits for the joint positions."""
```

Action term implementation (`action_managers/actions.py:17-90`):

```python
class EMACumulativeRelativeJointPositionAction(JointPositionAction):
    cfg: actions_cfg.EMACumulativeRelativeJointPositionActionCfg
    _asset: Articulation
    """The articulation asset on which the action term is applied."""

    def __init__(self, cfg: actions_cfg.EMACumulativeRelativeJointPositionActionCfg, env: ManagerBasedRLEnv) -> None:
        # initialize the action term
        super().__init__(cfg, env)

        # parse and save the moving average weight
        if isinstance(cfg.alpha, float):
            # check that the weight is in the valid range
            if not 0.0 <= cfg.alpha <= 1.0:
                raise ValueError(f"Moving average weight must be in the range [0, 1]. Got {cfg.alpha}.")
            self._alpha = cfg.alpha
        elif isinstance(cfg.alpha, dict):
            self._alpha = torch.ones((env.num_envs, self.action_dim), device=self.device)
            # resolve the dictionary config
            index_list, names_list, value_list = string_utils.resolve_matching_names_values(
                cfg.alpha, self._joint_names
            )
            # check that the weights are in the valid range
            for name, value in zip(names_list, value_list):
                if not 0.0 <= value <= 1.0:
                    raise ValueError(
                        f"Moving average weight must be in the range [0, 1]. Got {value} for joint {name}."
                    )
            self._alpha[:, index_list] = torch.tensor(value_list, device=self.device)
        else:
            raise ValueError(
                f"Unsupported moving average weight type: {type(cfg.alpha)}. Supported types are float and dict."
            )

        # initialize the previous targets
        self._prev_applied_actions = torch.zeros_like(self.processed_actions)
        # initialize the cumulative del action
        self.del_action = torch.zeros((self._env.num_envs, self.action_dim), device=self._env.device)
        self.init_joint_pos = self._asset.data.joint_pos[:, self._joint_ids].clone()
        self.joint_lower_limit = torch.tensor(cfg.joint_lower_limit, device=self.device) if cfg.joint_lower_limit is not None else None
        self.joint_upper_limit = torch.tensor(cfg.joint_upper_limit, device=self.device) if cfg.joint_upper_limit is not None else None
    
    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        # check if specific environment ids are provided
        if env_ids is None:
            env_ids = slice(None)
        super().reset(env_ids)
        # reset history to current joint positions
        self._prev_applied_actions[env_ids, :] = self._asset.data.joint_pos[env_ids][:, self._joint_ids].clone()
        # reset the del action
        self.del_action[env_ids, :] = torch.zeros((env_ids.shape[0], self.action_dim), device=self.device)
        self.init_joint_pos[env_ids, :] = self._asset.data.joint_pos[env_ids][:, self._joint_ids].clone()

    def process_actions(self, actions: torch.Tensor):
        # apply affine transformations
        super().process_actions(actions)
        # compute the del action
        self._processed_actions += self.del_action
        self.del_action = self._processed_actions.clone()
        # add the initial position
        self._processed_actions += self.init_joint_pos.clone()
        # set position targets as moving average
        ema_actions = self._alpha * self._processed_actions
        ema_actions += (1.0 - self._alpha) * self._prev_applied_actions
        # clamp the targets
        if self.joint_lower_limit is not None and self.joint_upper_limit is not None:
            self._processed_actions[:] = torch.clamp(
                ema_actions,
                self.joint_lower_limit,
                self.joint_upper_limit,
            )
        else:
            self._processed_actions[:] = ema_actions
        # update previous targets
        self._prev_applied_actions[:] = self._processed_actions[:]
```

Action-scale application in `BaseEnv.step` (`manager_based_env.py:161-166`):

```python
    def step(self, action: torch.Tensor) -> VecEnvStepReturn:
        # update last action inferred from the policy. this is different from the last action used in IsaacLab action manager
        self.last_action = action.clone()
        # scale the action
        action = action * self._scale
        super().step(action)
```

Scale tensor init (`manager_based_env.py:229-232`):

```python
        # save the scale as tensors
        if self.cfg.action_scale is not None:
            self._scale = torch.tensor(self.cfg.action_scale, device=self.device)
        else:
            self._scale = torch.ones(self.action_space.shape, device=self.device)
```

---

## §3 Reset / Events

**Description.** Five reset `EventTerm`s (all `mode="reset"`, deterministic ranges — every range is a degenerate point interval, so this is a **fixed layout**, not a randomized one). (1) `reset_robot_joints` from `BaseEventCfg` resets the RIGHT robot to its default pose via `reset_joints_by_symmetry` (with a possible C2 mirror of the left robot's pose for the symmetric envs). (2) `reset_robot_joints_left` does the same for the LEFT robot. (3) `reset_drawer` zeros the drawer joint via IsaacLab `reset_joints_by_scale`. (4) `reset_drawer_pos` teleports the drawer to env-frame `(0.4, 0.1, 0.1)`. (5) `reset_object_right` places the object at `(0.05, -0.35, 0.0)` with a **random yaw** in `[-π, π]` (the ONLY nondeterministic reset). Note `reset_joints_by_symmetry` uses `env.rep_Q_js` and `env.symmetry_tracker` to optionally mirror-copy the opposite arm's default pose into the symmetric envs.

### Decisions resolved

| Decision | Value | Source |
|---|---|---|
| mode | all `"reset"` (once per episode reset) | `env_cfg.py:372-410`, `manager_based_env_cfg.py:133-140` |
| right robot reset | `reset_joints_by_symmetry`, pos_range (1,1), vel_range (0,0) | `manager_based_env_cfg.py:133-140` |
| left robot reset | `reset_joints_by_symmetry`, asset `robot_left`, pos (1,1) vel (0,0) | `env_cfg.py:372-380` |
| drawer joint reset | IsaacLab `mdp.reset_joints_by_scale`, pos (1,1) vel (0,0) | `env_cfg.py:382-390` |
| drawer pose reset | `reset_articulation`, x=0.4 y=0.1 z=0.1 (fixed) | `env_cfg.py:392-400` |
| object reset | `reset_object`, x=0.05 y=-0.35 z=0.0, yaw ∈ [-3.14, 3.14], object_id 0 | `env_cfg.py:402-410` |
| symmetric-env mirror | half of env_ids mirror-reset via `symmetry_tracker`/`rep_Q_js` | `manager_based_env.py:289-306`, `reset_mdps.py:34-43` |

### Code

Event cfg (`InsertDrawer/env_cfg.py:368-410`):

```python
@configclass
class InsertDrawerEventCfg(BaseEventCfg):
    """Configuration for events."""

    reset_robot_joints_left = EventTerm(
        func=reset_joints_by_symmetry,
        mode="reset",
        params={
            "position_range": (1.0, 1.0),
            "velocity_range": (0.0, 0.0),
            "asset_cfg": SceneEntityCfg("robot_left")
        },
    )

    reset_drawer = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (1.0, 1.0),
            "velocity_range": (0.0, 0.0),
            "asset_cfg": SceneEntityCfg("drawer")
        },
    )

    reset_drawer_pos = EventTerm(
        func=reset_articulation,
        mode="reset",
        params={
            "pose_range": {"x": [0.4, 0.4], "y": [0.1, 0.1], "z": [0.1, 0.1]}, 
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("drawer")
        },
    )
    
    reset_object_right = EventTerm(
        func=reset_object,
        mode="reset",
        params={
            "pose_range": {"x": [0.05, 0.05], "y": [-0.35, -0.35], "z": [0.0, 0.0], "yaw": [-3.14, 3.14]},
            "velocity_range": {},
            "object_id": 0,
        },
    )
```

Base event (right robot reset, `manager_based_env_cfg.py:130-140`):

```python
@configclass
class BaseEventCfg:
    """Configuration for events."""
    reset_robot_joints = EventTerm(
        func=reset_joints_by_symmetry,
        mode="reset",
        params={
            "position_range": (1.0, 1.0),
            "velocity_range": (0.0, 0.0),
        },
    )
```

`reset_joints_by_symmetry` (`reset_mdps.py:17-57`):

```python
def reset_joints_by_symmetry(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    position_range: tuple[float, float],
    velocity_range: tuple[float, float],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Reset the robot joints by scaling the default position and velocity by the given ranges.

    This function samples random values from the given ranges and scales the default joint positions and velocities
    by these values. The scaled values are then set into the physics simulation.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # get default joint state
    joint_pos = asset.data.default_joint_pos[env_ids].clone()
    joint_vel = asset.data.default_joint_vel[env_ids].clone()
    # symmetry
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

    # scale these values randomly
    joint_pos *= math_utils.sample_uniform(*position_range, joint_pos.shape, joint_pos.device)
    joint_vel *= math_utils.sample_uniform(*velocity_range, joint_vel.shape, joint_vel.device)

    # clamp joint pos to limits
    joint_pos_limits = asset.data.soft_joint_pos_limits[env_ids]
    joint_pos = joint_pos.clamp_(joint_pos_limits[..., 0], joint_pos_limits[..., 1])
    # clamp joint vel to limits
    joint_vel_limits = asset.data.soft_joint_vel_limits[env_ids]
    joint_vel = joint_vel.clamp_(-joint_vel_limits, joint_vel_limits)
    # set into the physics simulation
    asset.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
```

`reset_object` (`reset_mdps.py:60-84`):

```python
def reset_object(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    pose_range: dict[str, tuple[float, float]],
    velocity_range: dict[str, tuple[float, float]],
    object_id: int,
):
    object = env.scene[f"object_{object_id}"]
    root_states = object.data.default_root_state[env_ids].clone()
    # poses
    range_list = [pose_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
    ranges = torch.tensor(range_list, device=object.device)
    rand_samples = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=object.device)
    positions = root_states[:, 0:3] + env.scene.env_origins[env_ids] + rand_samples[:, 0:3]
    orientations_delta = math_utils.quat_from_euler_xyz(rand_samples[:, 3], rand_samples[:, 4], rand_samples[:, 5])
    orientations = math_utils.quat_mul(root_states[:, 3:7], orientations_delta)
    # velocities
    range_list = [velocity_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
    ranges = torch.tensor(range_list, device=object.device)
    rand_samples = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=object.device)
    velocities = root_states[:, 7:13] + rand_samples

        # set into the physics simulation
    object.write_root_pose_to_sim(torch.cat([positions, orientations], dim=-1), env_ids=env_ids)
    object.write_root_velocity_to_sim(velocities, env_ids=env_ids)
```

`reset_articulation` (`reset_mdps.py:86-110`):

```python
def reset_articulation(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    pose_range: dict[str, tuple[float, float]],
    velocity_range: dict[str, tuple[float, float]],
    asset_cfg: SceneEntityCfg,
):
    asset: Articulation = env.scene[asset_cfg.name]
    root_states = asset.data.default_root_state[env_ids].clone()
    # poses
    range_list = [pose_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
    ranges = torch.tensor(range_list, device=asset.device)
    rand_samples = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=asset.device)
    positions = root_states[:, 0:3] + env.scene.env_origins[env_ids] + rand_samples[:, 0:3]
    orientations_delta = math_utils.quat_from_euler_xyz(rand_samples[:, 3], rand_samples[:, 4], rand_samples[:, 5])
    orientations = math_utils.quat_mul(root_states[:, 3:7], orientations_delta)
    # velocities
    range_list = [velocity_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
    ranges = torch.tensor(range_list, device=asset.device)
    rand_samples = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=asset.device)
    velocities = root_states[:, 7:13] + rand_samples

    # set into the physics simulation
    asset.write_root_pose_to_sim(torch.cat([positions, orientations], dim=-1), env_ids=env_ids)
    asset.write_root_velocity_to_sim(velocities, env_ids=env_ids)
```

---

## §4 Goal + Termination

**Description.** Goal is delivered by a single command term `target_pos` (`TargetPositionCommandCfg`, `object_id="drawer"`) that samples a fixed env-frame offset `(-0.3, 0.0, 0.18)` relative to the drawer body (`offset=True`), with `success_threshold=0.05`, orientation threshold disabled (`-1`), `update_goal_on_success=True`, and a debug goal marker. (The command's own consecutive_success metric is largely unused by the actual success predicate here.)

**Success predicate** is the custom termination `drawer.max_consecutive_success(num_success=20)`: the drawer joint must be nearly closed (`base_drawer_joint < 0.1`) AND the object must be "in drawer" (`if_in_drawer`: all four right-hand fingertips released from the object AND object within 0.2 m of the drawer body AND object below 1.0 m). This must hold for **20 consecutive control steps** (`success_tracker_step`); when it does, `env.success_tracker` is set and the episode terminates. **Failure terminations:** only `time_out` (from `BaseTerminationsCfg`) at episode length 8.3333 s — there is no explicit drop/out-of-bounds failure term.

### Decisions resolved

| Decision | Value | Source |
|---|---|---|
| command term | `TargetPositionCommand`, target on `drawer` body | `env_cfg.py:416-424`, `grasp_command.py` |
| goal offset | env-frame `(-0.3, 0.0, 0.18)` relative to drawer, `offset=True` | `env_cfg.py:420-424` |
| pos success threshold | 0.05 m; orient threshold -1 (disabled) | `env_cfg.py:418-419` |
| update_goal_on_success | True | `env_cfg.py:421` |
| success termination | `drawer.max_consecutive_success`, num_success=20 | `env_cfg.py:490-492`, `mdps.py:128-137` |
| success predicate | `joint<0.1` AND `if_in_drawer` for 20 consecutive steps | `mdps.py:128-137` |
| `if_in_drawer` | fingertips released AND dist(object,drawer)<0.2 AND object_z<1.0 | `mdps.py:85-96` |
| failure terminations | `time_out` only (8.3333 s) | `manager_based_env_cfg.py:156-159, 207` |

### Code

Command cfg (`InsertDrawer/env_cfg.py:412-424`):

```python
@configclass
class InsertDrawerCommandsCfg(BaseCommandsCfg):
    """Command specifications for the MDP."""

    target_pos = TargetPositionCommandCfg(
        object_id="drawer",
        success_threshold=0.05,
        success_threshold_orient=-1, # 30 degree 
        pose_range={"x": [-0.3, -0.3], "y": [0.0, 0.0], "z": [0.18, 0.18]},
        update_goal_on_success=True,
        debug_vis=True,
        offset=True,
    )
```

Termination cfg (`InsertDrawer/env_cfg.py:488-492`):

```python
@configclass
class InsertDrawerTerminationsCfg(BaseTerminationsCfg):
    max_consecutive_success = DoneTerm(
        func=drawer.max_consecutive_success, params={"num_success": 20}
    )
```

Base termination — time_out (`manager_based_env_cfg.py:156-159`):

```python
@configclass
class BaseTerminationsCfg:
    """Termination terms for the MDP."""
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
```

Success predicate + in-drawer check (`InsertDrawer/mdps.py:85-96, 128-137`):

```python
def if_in_drawer(
    env: ManagerBasedRLEnv,
    object_id: int = 0,
    sensor_names: list = ["contact_sensors_0", "contact_sensors_1", "contact_sensors_2", "contact_sensors_3"],
) -> torch.Tensor:
    drawer: Articulation = env.scene["drawer"]
    drawer_idx = drawer.find_bodies("drawer")[0]
    drawer_pos_w = drawer.data.body_state_w[:, drawer_idx, :3].reshape(-1, 3)
    object: RigidObject = env.scene[f"object_{object_id}"]
    distance = torch.norm(object.data.root_pos_w[:, :3] - drawer_pos_w, dim=-1)
    rew = check_release(env, sensor_names) * (distance < 0.2) * (object.data.root_pos_w[:, 2] < 1.0)
    return rew
```

```python
def max_consecutive_success(env: ManagerBasedRLEnv, num_success: int) -> torch.Tensor:
    drawer: Articulation = env.scene["drawer"]
    joint_ids = drawer.find_joints("base_drawer_joint")[0]
    joint_pos = drawer.data.joint_pos[:, joint_ids].reshape(-1)
    success = (joint_pos < 0.1) * if_in_drawer(env)
    env.success_tracker_step[success] += 1
    env.success_tracker_step[~success] = 0
    success = env.success_tracker_step >= num_success
    env.success_tracker = success.float()
    return success
```

Command term (`command_mdps/grasp_command.py` — key `_update_metrics` / `_resample_command`):

```python
    def _update_metrics(self):
        # logs data
        # -- compute the orientation error
        from bimanual_suite.utils.isaac_utils import get_angle_from_quat
        target_axis = get_angle_from_quat(self.quat_command_w, axis="z", normalize=True)
        cur_axis = get_angle_from_quat(self.object.data.root_quat_w, axis="z", normalize=True)
        self.metrics["orientation_error"] = torch.sum(target_axis * cur_axis, dim=-1)
        # -- compute the position error
        self.metrics["position_error"] = torch.norm(self.object.data.root_pos_w - self.pos_command_w, dim=1)
        # -- compute the number of consecutive successes
        if self.ranges is not None and self.return_type != "pos":
            successes = (torch.logical_and(self.metrics["position_error"] < self.cfg.success_threshold, self.metrics["orientation_error"] > self.cfg.success_threshold_orient)).float()
        else:
            successes = (self.metrics["position_error"] < self.cfg.success_threshold).float()
        unsuccesses = torch.where(successes == 0.0)[0]
        self.metrics["consecutive_success"] += successes.float()
        self.metrics["consecutive_success"][unsuccesses] = 0.0

    def _resample_command(self, env_ids: Sequence[int]):
        # sample the goal position       
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
```

---

## §5 Observation

**Description.** Single concatenated `policy` group, `enable_corruption = False`, `concatenate_terms = True`. Ten terms, order preserved. All `ee_pose` terms return **12-dim** (position 3 + flattened rotation matrix 9, `symmetry=True` default). Joint terms cover all 22 joints/arm. **Total = 172** per env: ee_pose_right 12 + joint_pos_right 22 + joint_vel_right 22 + object_pos 3 + ee_pose_left 12 + joint_pos_left 22 + joint_vel_left 22 + drawer_handle_pose 12 + drawer_joint_pos 1 + last_action 44. Joint-position terms are normalized against the per-arm `JOINT_LOWER/UPPER_LIMIT` lists (drawer term against its own soft limits) and carry Gaussian obs noise (std 0.005) despite `enable_corruption=False` (corruption disabled → noise not applied). `object_pos` carries uniform noise `[0, 0.01]`.

### Decisions resolved

| Term | func | params | dim | Source |
|---|---|---|---|---|
| ee_pose_right | `ee_pose` | ee_name `palm_link` (robot) | 12 | `env_cfg.py:436` |
| joint_pos_right | `joint_pos_limit_normalized` | JOINT_LOWER/UPPER_LIMIT, Gnoise 0.005 | 22 | `env_cfg.py:437-440` |
| joint_vel_right | `joint_vel` | robot, all joints | 22 | `env_cfg.py:441` |
| object_pos | `object_pos` | object_id 0, Unoise [0,0.01] | 3 | `env_cfg.py:442-446` |
| ee_pose_left | `ee_pose` | palm_link, robot_left | 12 | `env_cfg.py:447` |
| joint_pos_left | `joint_pos_limit_normalized` | LEFT limits, robot_left, Gnoise 0.005 | 22 | `env_cfg.py:448-452` |
| joint_vel_left | `joint_vel` | robot_left | 22 | `env_cfg.py:453` |
| drawer_handle_pose | `ee_pose` | handle_grip (drawer) | 12 | `env_cfg.py:454` |
| drawer_joint_pos | `joint_pos_limit_normalized` | drawer (1 joint, soft limits) | 1 | `env_cfg.py:455` |
| last_action | `last_action` | — | 44 | `env_cfg.py:456` |
| **total** | | | **172** | sum of terms |
| flags | `enable_corruption=False`, `concatenate_terms=True` | | | `env_cfg.py:458-460` |

### Code

Observation cfg (`InsertDrawer/env_cfg.py:427-463`):

```python
@configclass
class InsertDrawerObservationsCfg(BaseObservationsCfg):
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # -- robot terms (order preserved)
        ee_pose_right = ObsTerm(func=ee_pose, params={"ee_name": "palm_link"})
        joint_pos_right = ObsTerm(func=joint_pos_limit_normalized, params={"joints": None, 
                                                                           "joint_lower_limit": JOINT_LOWER_LIMIT, 
                                                                           "joint_upper_limit": JOINT_UPPER_LIMIT,},
                                                                           noise=Gnoise(std=0.005))
        joint_vel_right = ObsTerm(func=joint_vel, params={"joints": None},)
        object_pos = ObsTerm(
            func=object_pos,
            noise=Unoise(n_min=0.0, n_max=0.01),
            params={"object_id": 0}
        )
        ee_pose_left = ObsTerm(func=ee_pose, params={"ee_name": "palm_link", "asset_cfg": SceneEntityCfg("robot_left")})
        joint_pos_left = ObsTerm(func=joint_pos_limit_normalized, params={"joints": None, 
                                                                          "joint_lower_limit": JOINT_LOWER_LIMIT_LEFT, 
                                                                          "joint_upper_limit": JOINT_UPPER_LIMIT_LEFT, 
                                                                          "asset_cfg": SceneEntityCfg("robot_left")},
                                                                          noise=Gnoise(std=0.005))
        joint_vel_left = ObsTerm(func=joint_vel, params={"joints": None, "asset_cfg": SceneEntityCfg("robot_left")},)
        drawer_handle_pose = ObsTerm(func=ee_pose, params={"ee_name": "handle_grip", "asset_cfg": SceneEntityCfg("drawer")})
        drawer_joint_pos = ObsTerm(func=joint_pos_limit_normalized, params={"joints": None, "asset_cfg": SceneEntityCfg("drawer")}, )
        last_action = ObsTerm(func=last_action)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()
```

`ee_pose` obs func (12-dim with symmetry, `obs_mdps.py:59-70`):

```python
def ee_pose(env: ManagerBasedEnv, ee_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), symmetry: bool = True) -> torch.Tensor:
    """EE pose in the environment frame."""
    # extract the used quantities (to enable type-hinting)
    robot: Articulation = env.scene[asset_cfg.name]
    ee_idx = robot.find_bodies(ee_name)[0]
    ee_w = robot.data.body_state_w[:, ee_idx, :7].clone().reshape(-1, 7)
    ee_w[:, :3] = ee_w[:, :3] - env.scene.env_origins
    if symmetry:
        ee_R = math_utils.matrix_from_quat(ee_w[:, 3:7])
        ee_R_flat = ee_R.transpose(1, 2).reshape(-1, 9)
        ee_w = torch.cat([ee_w[:, :3], ee_R_flat], dim=-1)
    return ee_w
```

`joint_pos_limit_normalized`, `joint_vel`, `object_pos`, `last_action` (`obs_mdps.py:15-98, 134-139`):

```python
def joint_pos_limit_normalized(
    env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), joints = None, joint_lower_limit = None, joint_upper_limit = None
) -> torch.Tensor:
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
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

    return math_utils.scale_transform(
        asset.data.joint_pos[:, joint_ids],
        joint_lower_limit,
        joint_upper_limit,
    )

def joint_vel(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), joints = None):
    asset: Articulation = env.scene[asset_cfg.name]
    if joints is None:
        joint_ids = asset_cfg.joint_ids
    else:
        joint_ids = asset.find_joints(joints)[0]
    return asset.data.joint_vel[:, joint_ids]

def object_pos(env: ManagerBasedEnv, object_id: int = 0) -> torch.Tensor:
    """Object root position in the environment frame."""
    object: RigidObject = env.scene[f"object_{object_id}"]
    return object.data.root_pos_w - env.scene.env_origins

def last_action(env: ManagerBasedEnv) -> torch.Tensor:
    """The last input action to the environment."""
    if hasattr(env, "last_action"):
        return env.last_action
    else:
        return torch.zeros((env.num_envs, env.action_dim), device=env.device)
```

---

## §6 Reward

**Description.** 23 `RewTerm`s total — 2 inherited from `BaseRewardsCfg` (`energy`, `energy_left`) plus 21 in `InsertDrawerRewardsCfg`. The task defines a **full right-and-left symmetric bank**: each behavioral term (`reaching_object`, `object_lifting`, `object_goal_tracking`, `object_in_drawer`, `reset_robot_joint_pos`, `reaching_handle`, `moving_drawer`, `moving_drawer_inside`, `collision_to_table`, `collision_to_drawer`) has a `..._symmetry` twin that swaps `robot`↔`robot_left` and the object/drawer contact-sensor sets. `success_bonus` returns `env.success_tracker`.

The reward is the weighted sum of all 23 `RewTerm`s; each term's weight is listed in the table below.

Behavioral term semantics (see verbatim funcs below):
- `object_robot_distance` (`reaching_object`): `1 / mean(weighted fingertip→object distance)`, fingertip weights `[1,1,1,1.5]` (thumb up-weighted).
- `lift_distance` (`object_lifting`): clamped normalized z-lift toward goal height, gated on 4-finger allegro contact.
- `object_goal_distance` (`object_goal_tracking`): progress `max_dist - dist` toward goal, gated on contact and object near goal height.
- `drawer.if_in_drawer` (`object_in_drawer`): binary in-drawer indicator (fingers released + within 0.2 m + z<1.0).
- `drawer.robot_goal_distance` (`reset_robot_joint_pos`): `1/(dist+eps)` of palm to a fixed target `[-0.1277, ∓0.3174, 1.2583]`, gated on `if_in_drawer` — pulls the hand back after insertion.
- `drawer.drawer_handle_robot_distance` (`reaching_handle`): `1/mean(handle→fingertip dist)` for index+middle.
- `drawer.drawer_move` (`moving_drawer`): drawer joint pos × handle-contact — opens the drawer (or clamps to max when object already inside).
- `drawer.drawer_move_inside` (`moving_drawer_inside`): `10·(max_joint - joint)` gated on object-inside + handle-contact — closes the drawer once loaded.
- `collision_penalty` (`collision_to_table`/`_to_drawer`): 1.0 when a non-target contact is registered.
- `energy_punishment` (`energy`/`energy_left`): `-Σ|joint_vel·applied_torque|` over hand actuators.

### Decisions resolved (every RewTerm)

| Term | func | key params | weight |
|---|---|---|---|
| energy | `energy_punishment` | robot, allegro hand actuators | 0.000001 |
| energy_left | `energy_punishment` | robot_left, allegro hand actuators | 0.000001 |
| reaching_object | `object_robot_distance` | weight [1,1,1,1.5], links if5/mf5/pf5/th5, obj 0 | 0.01 |
| object_lifting | `lift_distance` | cmd target_pos, obj 0, sensors 0-3 | 5.0 |
| object_goal_tracking | `object_goal_distance` | cmd target_pos, obj 0, sensors 0-3 | 50.0 |
| object_in_drawer | `drawer.if_in_drawer` | obj 0, sensors 0-3 | 100.0 |
| reset_robot_joint_pos | `drawer.robot_goal_distance` | target [-0.1277,-0.3174,1.2583], palm_link | 1.0 |
| reaching_handle | `drawer.drawer_handle_robot_distance` | weight [1,1], links if5/mf5, robot_left | 0.01 |
| moving_drawer | `drawer.drawer_move` | base_drawer_joint, drawer, sensor 0_left | 10 |
| moving_drawer_inside | `drawer.drawer_move_inside` | base_drawer_joint, drawer, sensor 0_left | 10 |
| success_bonus | `drawer.success_bonus` | — | 5000 |
| collision_to_table | `collision_penalty` | sensors 0-3 (right→object) | 0.000001 |
| collision_to_drawer | `collision_penalty` | sensors 0-3_left (left→handle) | 0.000001 |
| reaching_object_symmetry | `object_robot_distance` | …robot_left | 0.01 |
| object_lifting_symmetry | `lift_distance` | …_symmetry sensors | 5.0 |
| object_goal_tracking_symmetry | `object_goal_distance` | …_symmetry sensors | 50.0 |
| object_in_drawer_symmetry | `drawer.if_in_drawer` | …_symmetry sensors | 100.0 |
| reset_robot_joint_pos_symmetry | `drawer.robot_goal_distance` | target [-0.1277,+0.3174,1.2583], robot_left | 1.0 |
| reaching_handle_symmetry | `drawer.drawer_handle_robot_distance` | links if5/mf5, robot | 0.01 |
| moving_drawer_symmetry | `drawer.drawer_move` | sensor 0_left_symmetry | 10 |
| moving_drawer_inside_symmetry | `drawer.drawer_move_inside` | sensor 0_left_symmetry | 10 |
| collision_to_table_symmetry | `collision_penalty` | …_symmetry sensors | 0.000001 |
| collision_to_drawer_symmetry | `collision_penalty` | …_left_symmetry sensors | 0.000001 |

### Symmetric-learning reward terms (drop if not using symmetric learning)

This task trains with **symmetric learning** (`base.yaml` → `symmetry.symmetric_envs: True`, `C2` group). The **BASE reward set** consists of the right-arm object terms **plus** the left-arm drawer-handle terms — both halves are genuine for the bimanual insert task (right hand picks/lifts/inserts the object; left hand reaches the handle and opens/closes the drawer):

- Base (right-arm): `reaching_object`, `object_lifting`, `object_goal_tracking`, `object_in_drawer`, `reset_robot_joint_pos`, `energy`, `collision_to_table`
- Base (left-arm / drawer): `reaching_handle`, `moving_drawer`, `moving_drawer_inside`, `success_bonus`, `energy_left`, `collision_to_drawer`

The `_symmetry`-suffixed terms are **DUPLICATES used for symmetric-learning augmentation ONLY** (they swap `robot`↔`robot_left` and the object/drawer contact-sensor sets so the same policy handles the left↔right mirror). **Drop them if you are not using symmetric learning.** The `_symmetry` duplicate terms are:

`reaching_object_symmetry`, `object_lifting_symmetry`, `object_goal_tracking_symmetry`, `object_in_drawer_symmetry`, `reset_robot_joint_pos_symmetry`, `reaching_handle_symmetry`, `moving_drawer_symmetry`, `moving_drawer_inside_symmetry`, `collision_to_table_symmetry`, `collision_to_drawer_symmetry`.

### Code

Base rewards (energy, `manager_based_env_cfg.py:142-154`):

```python
@configclass
class BaseRewardsCfg:
    """Reward terms for the MDP."""
    energy = RewTerm(func=energy_punishment,
                                  weight=0.000001,
                                  params={"asset_cfg": SceneEntityCfg("robot"), "actuator_name": ["allegro_hand_1", "allegro_hand_2", "allegro_hand_3", "allegro_hand_4", 
                                                                                                  "allegro_hand_thumb_1", "allegro_hand_thumb_2", "allegro_hand_thumb_3", "allegro_hand_thumb_4"]},
                                  )
    energy_left = RewTerm(func=energy_punishment,
                                  weight=0.000001,
                                  params={"asset_cfg": SceneEntityCfg("robot_left"), "actuator_name": ["allegro_hand_1", "allegro_hand_2", "allegro_hand_3", "allegro_hand_4", 
                                                                                                  "allegro_hand_thumb_1", "allegro_hand_thumb_2", "allegro_hand_thumb_3", "allegro_hand_thumb_4"]},
                                  )
```

Task rewards (`InsertDrawer/env_cfg.py:494-585`):

```python
@configclass
class InsertDrawerRewardsCfg(BaseRewardsCfg):
    """Reward terms for the MDP."""
    reaching_object = RewTerm(func=object_robot_distance, 
                              params={"weight": [1.0, 1.0, 1.0, 1.5], 
                                      "link_name": ["if5", "mf5", "pf5", "th5"], 
                                      "object_id": 0}, 
                                      weight=0.01)
    object_lifting = RewTerm(func=lift_distance,
                             params={"command_name": "target_pos", "object_id": 0, "sensor_names": ["contact_sensors_0", "contact_sensors_1", "contact_sensors_2", "contact_sensors_3"]},
                             weight=5.0,
                             )
    object_goal_tracking = RewTerm(func=object_goal_distance,
                                   params={"command_name": "target_pos", "object_id": 0, "sensor_names": ["contact_sensors_0", "contact_sensors_1", "contact_sensors_2", "contact_sensors_3"],},
                                   weight=50.0,
                                   )
    object_in_drawer = RewTerm(func=drawer.if_in_drawer,
                             params={"object_id": 0, "sensor_names": ["contact_sensors_0", "contact_sensors_1", "contact_sensors_2", "contact_sensors_3"]},
                             weight=100.0,
                             )
    reset_robot_joint_pos = RewTerm(func=drawer.robot_goal_distance, 
                              params={"target_pos": [-0.1277, -0.3174,  1.2583], 
                                      "target_link": "palm_link"}, 
                                      weight=1.0)           
    
    reaching_handle = RewTerm(func=drawer.drawer_handle_robot_distance, 
                              params={"weight": [1.0, 1.0], 
                                      "link_name": ["if5", "mf5"], 
                                      "asset_cfg": SceneEntityCfg("robot_left")}, 
                                      weight=0.01)
    moving_drawer = RewTerm(func=drawer.drawer_move, 
                            params={"joints": ["base_drawer_joint"], "asset_cfg": SceneEntityCfg("drawer"), "sensor_names": ["contact_sensors_0_left"]},
                            weight=10)
    moving_drawer_inside = RewTerm(func=drawer.drawer_move_inside, 
                            params={"joints": ["base_drawer_joint"], "asset_cfg": SceneEntityCfg("drawer"), "sensor_names": ["contact_sensors_0_left"]},
                            weight=10)
    success_bonus = RewTerm(func=drawer.success_bonus,
                            params={},
                            weight=5000,
                            )
    collision_to_table = RewTerm(func=collision_penalty,
                                params={"sensor_names": ["contact_sensors_0", "contact_sensors_1", "contact_sensors_2", "contact_sensors_3"]},
                                weight=0.000001,
                                )
    collision_to_drawer = RewTerm(func=collision_penalty,
                                params={"sensor_names": ["contact_sensors_0_left", "contact_sensors_1_left", "contact_sensors_2_left", "contact_sensors_3_left"]},
                                weight=0.000001,
                                )
    # symmetry terms
    reaching_object_symmetry = RewTerm(func=object_robot_distance,
                                      params={"weight": [1.0, 1.0, 1.0, 1.5], 
                                      "link_name": ["if5", "mf5", "pf5", "th5"], 
                                      "object_id": 0,
                                      "asset_cfg": SceneEntityCfg("robot_left")}, 
                                      weight=0.01)
    object_lifting_symmetry = RewTerm(func=lift_distance,
                             params={"command_name": "target_pos", "object_id": 0, "sensor_names": ["contact_sensors_0_symmetry", "contact_sensors_1_symmetry", "contact_sensors_2_symmetry", "contact_sensors_3_symmetry"]},
                             weight=5.0,
                             )
    object_goal_tracking_symmetry = RewTerm(func=object_goal_distance,
                                   params={"command_name": "target_pos", "object_id": 0, "sensor_names": ["contact_sensors_0_symmetry", "contact_sensors_1_symmetry", "contact_sensors_2_symmetry", "contact_sensors_3_symmetry"],},
                                   weight=50.0,
                                   )
    object_in_drawer_symmetry = RewTerm(func=drawer.if_in_drawer,
                             params={"object_id": 0, "sensor_names": ["contact_sensors_0_symmetry", "contact_sensors_1_symmetry", "contact_sensors_2_symmetry", "contact_sensors_3_symmetry"]},
                             weight=100.0,
                             )
    reset_robot_joint_pos_symmetry = RewTerm(func=drawer.robot_goal_distance, 
                              params={"target_pos": [-0.1277, 0.3174,  1.2583], 
                                      "target_link": "palm_link",
                                      "asset_cfg": SceneEntityCfg("robot_left")}, 
                                      weight=1.0)           
    
    reaching_handle_symmetry = RewTerm(func=drawer.drawer_handle_robot_distance, 
                              params={"weight": [1.0, 1.0], 
                                      "link_name": ["if5", "mf5"], 
                                      "asset_cfg": SceneEntityCfg("robot")}, 
                                      weight=0.01)
    moving_drawer_symmetry = RewTerm(func=drawer.drawer_move, 
                            params={"joints": ["base_drawer_joint"], "asset_cfg": SceneEntityCfg("drawer"), "sensor_names": ["contact_sensors_0_left_symmetry"]},
                            weight=10)
    moving_drawer_inside_symmetry = RewTerm(func=drawer.drawer_move_inside, 
                            params={"joints": ["base_drawer_joint"], "asset_cfg": SceneEntityCfg("drawer"), "sensor_names": ["contact_sensors_0_left_symmetry"]},
                            weight=10)
    collision_to_table_symmetry = RewTerm(func=collision_penalty,
                                params={"sensor_names": ["contact_sensors_0_symmetry", "contact_sensors_1_symmetry", "contact_sensors_2_symmetry", "contact_sensors_3_symmetry"]},
                                weight=0.000001,
                                )
    collision_to_drawer_symmetry = RewTerm(func=collision_penalty,
                                params={"sensor_names": ["contact_sensors_0_left_symmetry", "contact_sensors_1_left_symmetry", "contact_sensors_2_left_symmetry", "contact_sensors_3_left_symmetry"]},
                                weight=0.000001,
                                )
```

Task-specific reward functions (`InsertDrawer/mdps.py:15-126`):

```python
def drawer_handle_robot_distance(
    env: ManagerBasedRLEnv,
    weight: list,
    link_name: list,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward the agent for reaching the drawer handle using tanh-kernel."""
    weight = torch.tensor(weight, device=env.device)
    # extract the used quantities (to enable type-hinting)
    drawer: Articulation = env.scene["drawer"]
    # Target object position: (num_envs, 3)
    drawer_idx = drawer.find_bodies("handle_grip")[0]
    drawer_handle_pos_w = drawer.data.body_state_w[:, drawer_idx, :3]
    # Fingertip position: (num_envs, num_fingertip, 3)
    robot: Articulation = env.scene[asset_cfg.name]
    link_idx = robot.find_bodies(link_name)[0]
    link_w = robot.data.body_pos_w[:, link_idx, :3]
    # Distance of the fingertip to the object: (num_envs,)
    drawer_handle_link_distance = torch.norm(drawer_handle_pos_w - link_w, dim=-1) * weight
    drawer_handle_link_distance = torch.mean(drawer_handle_link_distance, dim=1)
    rew = 1 / drawer_handle_link_distance
    return rew

def drawer_move(
    env: ManagerBasedRLEnv,
    sensor_names: str = "contact_sensors_0_left",
    joints = None,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("drawer"),
) -> torch.Tensor:
    drawer: Articulation = env.scene[asset_cfg.name]
    if joints is None:
        joint_ids = asset_cfg.joint_ids
    else:
        joint_ids = drawer.find_joints(joints)[0]
    joint_pos = drawer.data.joint_pos[:, joint_ids].reshape(-1)
    # if object is inside the drawer, encourage the drawer to move inside
    max_joint_pos = drawer.data.default_joint_limits[:, joint_ids, 1].reshape(-1)
    is_inside = if_in_drawer(env, object_id=0)
    is_inside_idx = torch.where(is_inside)[0]
    joint_pos[is_inside_idx] = max_joint_pos[is_inside_idx]

    force = get_force(env, sensor_names, if_filter=True)
    is_contact = (torch.norm(force, dim=-1) > 1.0).reshape(-1)
    rew = joint_pos * is_contact
    return rew

def drawer_move_inside(
    env: ManagerBasedRLEnv,
    sensor_names: str = "contact_sensors_0_left",
    joints = None,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("drawer"),
) -> torch.Tensor:
    drawer: Articulation = env.scene[asset_cfg.name]
    if joints is None:
        joint_ids = asset_cfg.joint_ids
    else:
        joint_ids = drawer.find_joints(joints)[0]
    joint_pos = drawer.data.joint_pos[:, joint_ids].reshape(-1)
    rew = torch.zeros_like(joint_pos)
    # if object is inside the drawer, encourage the drawer to move inside
    max_joint_pos = drawer.data.default_joint_limits[:, joint_ids, 1].reshape(-1)
    is_inside = if_in_drawer(env, object_id=0)
    is_inside_idx = torch.where(is_inside)[0]
    rew[is_inside_idx] = 10 * (max_joint_pos[is_inside_idx] - joint_pos[is_inside_idx])

    force = get_force(env, sensor_names, if_filter=True)
    is_contact = (torch.norm(force, dim=-1) > 1.0).reshape(-1)
    rew = rew * is_contact
    return rew

def if_in_drawer(
    env: ManagerBasedRLEnv,
    object_id: int = 0,
    sensor_names: list = ["contact_sensors_0", "contact_sensors_1", "contact_sensors_2", "contact_sensors_3"],
) -> torch.Tensor:
    drawer: Articulation = env.scene["drawer"]
    drawer_idx = drawer.find_bodies("drawer")[0]
    drawer_pos_w = drawer.data.body_state_w[:, drawer_idx, :3].reshape(-1, 3)
    object: RigidObject = env.scene[f"object_{object_id}"]
    distance = torch.norm(object.data.root_pos_w[:, :3] - drawer_pos_w, dim=-1)
    rew = check_release(env, sensor_names) * (distance < 0.2) * (object.data.root_pos_w[:, 2] < 1.0)
    return rew

def reset_robot_joint_pos(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    robot: Articulation = env.scene[asset_cfg.name]
    robot_default_joint_pos = robot.data.default_joint_pos
    cur_joint_pos = robot.data.joint_pos
    distance = torch.norm(robot_default_joint_pos - cur_joint_pos, dim=-1)
    rew = 1 / (distance + 1e-6) * if_in_drawer(env, object_id=0).float()
    return rew

def robot_goal_distance(
    env: ManagerBasedRLEnv,
    target_pos: list,
    target_link: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward the agent for tracking the goal pose."""
    des_pos_w = torch.tensor(target_pos, device=env.device) + env.scene.env_origins
    target_link_idx = env.scene[asset_cfg.name].find_bodies([target_link])[0]
    distance = torch.norm(des_pos_w - env.scene[asset_cfg.name].data.body_state_w[:, target_link_idx, 0:3].squeeze(1), dim=1)
    rew = 1 / (distance + 1e-6) * if_in_drawer(env, object_id=0).float()
    return rew

def success_bonus(
    env: ManagerBasedRLEnv,
) -> torch.Tensor:
    rew = env.success_tracker
    return rew.float()
```

Shared reward functions used by task terms (`reward_mdps.py:14-131`):

```python
def object_robot_distance(
    env: ManagerBasedRLEnv,
    weight: list,
    link_name: list,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_id: int = 0,
) -> torch.Tensor:
    """Reward the agent for reaching the object using tanh-kernel."""
    weight = torch.tensor(weight, device=env.device)
    object: RigidObject = env.scene[f"object_{object_id}"]
    object_pos_w = object.data.root_pos_w[:, None, :] 
    robot: Articulation = env.scene[asset_cfg.name]
    link_idx = robot.find_bodies(link_name)[0]
    link_w = robot.data.body_pos_w[:, link_idx, :3]
    object_link_distance = torch.norm(object_pos_w - link_w, dim=-1) * weight
    object_link_distance = torch.mean(object_link_distance, dim=1)
    rew = 1 / object_link_distance
    return rew

def get_force(env: ManagerBasedRLEnv, sensor_names: list, if_filter: bool = False):
    if if_filter:
        forces = []
        for sensor_name in sensor_names:
            force = env.scene[sensor_name].data.force_matrix_w.squeeze(2)
            forces.append(force)
        return torch.cat(forces, dim=1)
    else:
        forces = []
        for sensor_name in sensor_names:
            force = env.scene[sensor_name].data.net_forces_w.squeeze(2)
            forces.append(force)
        return torch.cat(forces, dim=1)

def get_allegro_contact(env: ManagerBasedRLEnv, sensor_names: list):
    force = get_force(env, sensor_names, if_filter=True)
    is_contact = (torch.norm(force, dim=-1) > 1.0)
    is_contact_index_or_middle_or_ring = reduce(torch.logical_or, [is_contact[:, 0], is_contact[:, 1], is_contact[:, 2]])
    is_contact = reduce(torch.logical_and, [is_contact_index_or_middle_or_ring, is_contact[:, 3]])
    return is_contact

def check_release(env: ManagerBasedRLEnv, sensor_names: list):
    force = get_force(env, sensor_names, if_filter=True)
    is_contact = (torch.norm(force, dim=-1) < 1.0)
    is_release = reduce(torch.logical_and, [is_contact[:, 0], is_contact[:, 1], is_contact[:, 2], is_contact[:, 3]])
    return is_release

def get_actuator_energy_consumption(
    env: ManagerBasedEnv, robot_name: str, actuator_name: str | list[str]
):
    robot: Articulation = env.scene[robot_name]
    jnt_vel = robot.data.joint_vel
    jnt_effort = robot.data.applied_torque
    if isinstance(actuator_name, str):
        joint_ids = robot.actuators[actuator_name].joint_indices
    else:
        joint_ids = []
        for name in actuator_name:
            joint_ids.extend(robot.actuators[name].joint_indices)
    jnt_vel = jnt_vel[:, joint_ids]
    jnt_effort = jnt_effort[:, joint_ids]
    energy = torch.abs(jnt_vel * jnt_effort).sum(dim=-1)
    return energy

def energy_punishment(
    env: ManagerBasedRLEnv,
    actuator_name = None,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    if actuator_name is None:
        energy = get_energy_consumption(env=env, robot_name=asset_cfg.name)
    else:
        energy = get_actuator_energy_consumption(env=env, robot_name=asset_cfg.name, actuator_name=actuator_name)
    rew = -energy
    return rew

def collision_penalty(
    env: ManagerBasedRLEnv,
    sensor_names: list = ["contact_sensors_0", "contact_sensors_1", "contact_sensors_2", "contact_sensors_3"],
) -> torch.Tensor:
    filtered_is_contact = []
    for s in sensor_names:
        # collide with objects that are not the target object
        filtered_force = env.scene[s].data.force_matrix_w.mean(dim=tuple(range(1, env.scene[s].data.force_matrix_w.ndim))) == 0.0
        normal_force = env.scene[s].data.net_forces_w.mean(dim=tuple(range(1, env.scene[s].data.net_forces_w.ndim))) != 0.0
        filtered_is_contact.append(torch.logical_and(filtered_force, normal_force))
    
    is_contact = reduce(torch.logical_or, filtered_is_contact)
    return is_contact.float()

def lift_distance(
    env: ManagerBasedRLEnv,
    command_name: str,
    minimal_height = None,
    object_id: int = 0,
    sensor_names: list = ["contact_sensors_0", "contact_sensors_1", "contact_sensors_2", "contact_sensors_3"],
) -> torch.Tensor:
    command = env.command_manager.get_command(command_name)
    des_pos_w = command[:, :3] + env.scene.env_origins
    if minimal_height is None:
        minimal_height = des_pos_w[:, 2] + 0.05
    object: RigidObject = env.scene[f"object_{object_id}"]
    z_distance = (object.data.root_pos_w[:, 2] - env.object_init_pos[object_id][:, 2]) / (minimal_height - env.object_init_pos[object_id][:, 2]) # linear
    z_distance = torch.clamp(z_distance, min=0.0)
    rew = (object.data.root_pos_w[:, 2] < minimal_height) * z_distance * get_allegro_contact(env, sensor_names)
    return rew

def object_goal_distance(
    env: ManagerBasedRLEnv,
    command_name: str,
    object_id: int = 0,
    sensor_names: list = ["contact_sensors_0", "contact_sensors_1", "contact_sensors_2", "contact_sensors_3"],
) -> torch.Tensor:
    """Reward the agent for tracking the goal pose."""
    object: RigidObject = env.scene[f"object_{object_id}"]
    command = env.command_manager.get_command(command_name)
    des_pos_w = command[:, :3] + env.scene.env_origins
    distance = torch.norm(des_pos_w - object.data.root_pos_w[:, :3], dim=1)
    max_distance = torch.norm(des_pos_w - env.object_init_pos[object_id], dim=1)
    distance = torch.clamp(max_distance - distance, min=0.0)
    rew = distance * get_allegro_contact(env, sensor_names) * torch.where(object.data.root_pos_w[:, 2] > (des_pos_w[:, 2] - 0.05), 1.0, 0.0)
    return rew
```

---

## §7 DR

**Description.** No DR is wired into the static `InsertDrawerEnvCfg` — there is no randomization `EventTerm`, and all reset ranges are degenerate point intervals (only the object yaw ∈ [-π, π] varies; see §3). **However**, the base env carries an *optional runtime DR pathway* (`BaseEnv.update_randomization`, gated on `success_rate`, driven by `self.cfg.hydra_cfg.task.randomize`) that CAN randomize object mass, material friction/restitution, action_scale, reward weights (energy/collision curricula), external force/torque, and reset pose — but this is inactive unless the hydra `task.randomize` config enables it and the training loop calls `update_randomization`. In the shipped task cfg, **no DR terms are active** → treat as `<no DR>` for reproduction unless the training-side hydra config supplies a `randomize` block.

### Decisions resolved

| Decision | Value | Source |
|---|---|---|
| static DR event terms | none | `env_cfg.py:368-410` (no randomization term) |
| runtime DR pathway | `update_randomization` (mass/material/action_scale/rew-weight/force/reset-pose) — gated on hydra `task.randomize` + success_rate; inactive by default | `manager_based_env.py:178-205`, `randomization_mdps.py` |
| effective for this task | `<no DR>` (unless hydra `randomize` block enabled at train time) | — |

### Code (runtime DR pathway, for reference — NOT active in static cfg)

`BaseEnv.update_randomization` (`manager_based_env.py:178-205`):

```python
    def update_randomization(self, success_rate):
        self.domain_randomizer.update(success_rate)
        randomized_values, randomization_state, curriculum_state = self.domain_randomizer.sample()
        if "object_mass" in randomized_values:
            randomize_mass(self, randomized_values["object_mass"])
        if "static_friction" in randomized_values:
            randomize_material(self, static_friction=randomized_values["static_friction"],
                               static_friction_range=list(self.cfg.hydra_cfg.task.randomize.randomization.static_friction),
                               dynamic_friction=randomized_values["dynamic_friction"], 
                               dynamic_friction_range=list(self.cfg.hydra_cfg.task.randomize.randomization.dynamic_friction),
                               restitution=randomized_values["restitution"], 
                               restitution_range=list(self.cfg.hydra_cfg.task.randomize.randomization.restitution), 
                               num_buckets=250)
        if "action_scale" in randomized_values:
            self._scale[:6] = randomized_values["action_scale"]
        if "energy_penalty" in randomized_values:
            for rew_name in curriculum_state["energy_penalty"]["names"]:
                randomize_rew_weight(self, rew_name, randomized_values["energy_penalty"])
        if "collision_penalty" in randomized_values:
            for rew_name in curriculum_state["collision_penalty"]["names"]:
                randomize_rew_weight(self, rew_name, randomized_values["collision_penalty"])
        if "external_force_torque" in randomized_values:
            randomize_external_force_torque(self, force=randomized_values["external_force_torque"], torque=randomized_values["external_force_torque"])
        # reset pose
        for rand_name in randomized_values.keys():
            if "reset_pose" in rand_name:
                randomize_reset_pose(self, curriculum_state[rand_name]["names"], randomized_values[rand_name])
        return randomization_state, curriculum_state, self.domain_randomizer.best_so_far
```
