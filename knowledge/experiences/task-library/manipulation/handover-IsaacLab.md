# Handover — Implementation Spec

- robot: Bimanual UF850 arms + dual Allegro hands (44 DoF)
- simulator: IsaacLab (Isaac Sim, manager-based)
- objects: orange bottle, table
- bimanual: true
- summary: Grasp a bottle with one hand, lift it to a mid-air handover pose, and transfer it to the other hand.

> Source package name anonymized as `bimanual_suite`. This task comes from an internal
> bimanual manipulation suite rather than a public repo; the design below is otherwise verbatim.

## Task summary

`Handover` is a bimanual dexterous manipulation task: two UF850 6-DoF arms, each tipped with a 16-DoF Allegro hand (`robot` = right, `robot_left` = left), stand across a table. A single rigid `orange_bottle` starts on the table in front of the RIGHT hand. The RIGHT hand must reach, grasp, and lift the bottle to a mid-air "handover" pose (the `target_pos` command). Once the bottle has held that mid-air pose consecutively (the env tracks a `reach_middle` counter that increments while `target_pos` consecutive_success ≥ 5), the LEFT hand approaches, grasps the bottle from the RIGHT hand, and the RIGHT hand releases. Success = the bottle is at the handover goal pose (position error < 0.08, oriented within threshold), the RIGHT hand is NOT in contact, and the LEFT hand IS in contact — held consecutively for 10 steps (`success_tracker_step`), which fires the termination `max_consecutive_success(num_success=10)`.

The reward is a large staged bank (reach → grasp/lift → transfer-to-middle → left-hand align → left grasp → release → success bonus), and the entire bank is **mirrored** via a `_symmetry` set of terms so a C2-symmetric copy of every reward is available for symmetric environments (chosen per-reset via `symmetry_tracker`). Each `RewTerm` carries its own scalar weight (see §6); the term also fixes the *shape* of each reward and its internal per-link weight vectors. This task also depends on custom `env.py` step/reset bookkeeping (`reach_middle`, `success_tracker_step`) and on the shared `BaseEnv.step` that scales actions by a per-joint `action_scale` and reads `symmetry_tracker`.

Key design notes that are easy to miss:
- Gravity is **disabled** on both robot articulations (`disable_gravity=True`); the bottle keeps gravity.
- The staged logic is gated almost entirely on `env.reach_middle` (0 = pre-transfer, >0 = bottle reached middle at least once, >10 = deep in transfer/release phase).
- Contact is read from **filtered** contact-sensor force matrices (finger-vs-Object_0 only), with a fingertip-tip (`*f5`)/mid (`*f4`) + thumb AND-logic.
- The reward bank and the symmetry machinery use `cfg.hydra_cfg.task.symmetry` (a C2 group) and `cfg.hydra_cfg.task.randomize`.

---

## §1 Registration + Scene

**Description.** `HandoverEnv(BaseEnv)` overrides `step()` (to advance `reach_middle` and optionally visualize palms) and the pre/post-reset hooks (to allocate/zero `reach_middle` and `success_tracker_step`). Registration in bimanual_suite is by config class, not `gym.register` in the task dir — the env class + cfg class pair (`HandoverEnv` / `HandoverEnvCfg`) is the unit; `name="Handover"`. The scene declares BOTH robots (right/left UF850+Allegro USDs, gravity disabled, 16-iter position solver), the single handover object `object_0` (`orange_bottle.usd`, mass 1.0 kg, scale 1.0), a kinematic table (from `BaseSceneCfg`), ground @ z=-0.82, a dome light, 16 filtered contact sensors per hand-set (tip `*f5` + mid `*f4` × 4 fingers × 2 hands), and two `FrameTransformer` frames on the bottle (`bottle_top` @ +0.07 z, `bottle_bottom` @ −0.08 z) used as reach targets. `num_envs=4096`, `env_spacing=3.0`, `replicate_physics=False`.

**Decisions resolved.**

| Decision | Value | Source |
|---|---|---|
| Env class / cfg pair | `HandoverEnv` / `HandoverEnvCfg`, `name="Handover"` | env.py, env_cfg.py |
| Base class | `BaseEnv(ManagerBasedRLEnv)` | manager_based_env.py |
| num_envs / spacing | 4096 / 3.0 | `HandoverSceneCfg(num_envs=4096, env_spacing=3.0)` |
| replicate_physics | False | BaseSceneCfg |
| Right robot USD | `{LIB_PATH}/assets/ufactory850/uf850_allegro_right_colored.usd` | env_cfg.py |
| Left robot USD | `{LIB_PATH}/assets/ufactory850/uf850_allegro_left_colored.usd` | env_cfg.py |
| Right robot init pos | `(-0.274, -0.475, 0.01)` | env_cfg.py |
| Left robot init pos | `(-0.274, 0.475, 0.01)` | env_cfg.py |
| Robot gravity | **disabled** on both | `rigid_props.disable_gravity=True` |
| Object USD | `{LIB_PATH}/assets/object/orange_bottle.usd`, scale (1,1,1), mass 1.0, gravity ON | env_cfg.py object_0 |
| Object init state | pos (0,0,0) — actually placed by `reset_object` event (see §3) | env_cfg.py |
| Table | `{LIB_PATH}/assets/object/table.usd`, kinematic, rot (0.7071,0,0,0.7071) | BaseSceneCfg |
| Ground | GroundPlane @ z=-0.82 | BaseSceneCfg |
| Light | DomeLight color (0.75,0.75,0.75), intensity 2500 | BaseSceneCfg |
| Contact sensors | 8 per hand × 2 = 16, filtered to `Object_0` | env_cfg.py |
| Fingertip frames | `bottle_top` (+0.07 z), `bottle_bottom` (−0.08 z) | env_cfg.py |
| sim.dt | 1/120 s | BaseEnvCfg.__post_init__ |
| decimation | 6 | BaseEnvCfg.__post_init__ |
| episode_length_s | 8.3333 s (~166 control steps) | BaseEnvCfg.__post_init__ |
| render_interval | = decimation (6) | BaseEnvCfg.__post_init__ |
| PhysX | gpu_max_rigid_contact_count/patch = 2**24 | BaseEnvCfg.sim |
| Physics material | static_friction 1.5, dynamic 1.0, restitution 0.0 | BaseEnvCfg.sim |
| Arm actuators | ImplicitActuator, stiffness 2000, damping 16 (`joint[1-6]`) | env_cfg.py |
| Hand actuators | per-finger-knuckle groups, stiffness 100–1270 (see code) | env_cfg.py |

**Code (verbatim).**

`env.py` — full:
```python
from __future__ import annotations

import torch
from typing import Any, ClassVar

from isaacsim.core.version import get_version
from isaaclab.envs.common import VecEnvStepReturn

from bimanual_suite.env.tasks.manager_based_env import BaseEnv
from bimanual_suite.env.tasks.Handover.env_cfg import HandoverEnvCfg

class HandoverEnv(BaseEnv):
    is_vector_env: ClassVar[bool] = True
    """Whether the environment is a vectorized environment."""
    metadata: ClassVar[dict[str, Any]] = {
        "render_modes": [None, "human", "rgb_array"],
        "isaac_sim_version": get_version(),
    }
    """Metadata for the environment."""

    cfg: HandoverEnvCfg
    """Configuration for the environment."""
    
    def step(self, action: torch.Tensor) -> VecEnvStepReturn:
        super().step(action)
        self.reach_middle += self.command_manager.get_term("target_pos").metrics["consecutive_success"] >= 5
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
        self.reach_middle = torch.zeros(self.num_envs, device=self.device)
        self.success_tracker_step = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)

    def _post_reset_process(self, env_ids):
        super()._post_reset_process(env_ids)
        self.reach_middle[env_ids] = 0.0
        self.success_tracker_step[env_ids] = 0.0
```

`env_cfg.py` — imports + scene (robots, object, sensors, frames) — VERBATIM:
```python
from __future__ import annotations

from isaaclab.assets import RigidObjectCfg
from isaaclab.utils import configclass
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.sensors.frame_transformer import OffsetCfg
from isaaclab.markers.config import FRAME_MARKER_CFG 

import bimanual_suite
from bimanual_suite.env.tasks.manager_based_env_cfg import *
from bimanual_suite.env.mdps.obs_mdps import *
from bimanual_suite.env.mdps.reset_mdps import *
from bimanual_suite.env.mdps.reward_mdps import *
from bimanual_suite.env.mdps.termination_mdps import *
from bimanual_suite.env.mdps.command_mdps.grasp_command_cfg import TargetPositionCommandCfg
from bimanual_suite.env.mdps.command_mdps.reach_command_cfg import TargetPositionCommandCfg as ReachCommandCfg
from bimanual_suite.env.action_managers.actions_cfg import EMACumulativeRelativeJointPositionActionCfg
from bimanual_suite.env.tasks.Handover import mdps as handover

FRAME_MARKER_SMALL_CFG = FRAME_MARKER_CFG.copy()
FRAME_MARKER_SMALL_CFG.markers["frame"].scale = (0.10, 0.10, 0.10)

@configclass
class HandoverSceneCfg(BaseSceneCfg):
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
                "joint1": 0.05,
                "joint2": 0.0,
                "joint3": -0.5,
                "joint4": 1.4,
                "joint5": -1.0,
                "joint6": -3.14,
                # hand 
                "jif1": 0.0, "jif2": 0.4, "jif3": 0.4, "jif4": 0.0,
                "jmf1": 0.0, "jmf2": 0.4, "jmf3": 0.4, "jmf4": 0.0,
                "jpf1": 0.0, "jpf2": 0.4, "jpf3": 0.4, "jpf4": 1.2,
                "jth1": 0.364, "jth2": 0.0, "jth3": 0.2, "jth4": 0.0,
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
                "jif1": 0.0, "jif2": 0.4, "jif3": 0.4, "jif4": 0.0,
                "jmf1": 0.0, "jmf2": 0.4, "jmf3": 0.4, "jmf4": 0.0,
                "jpf1": 0.0, "jpf2": 0.4, "jpf3": 0.4, "jpf4": 0.0,
                "jth1": 1.3, "jth2": 0.0, "jth3": -0.1, "jth4": 0.0,
            },
            pos=(-0.274, 0.475, 0.01),
        ),
        actuators={
            # identical actuator groups/gains as `robot` (see right robot above)
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

    object_0 = RigidObjectCfg(
        prim_path=f"/World/envs/env_.*/Object_0",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{bimanual_suite.LIB_PATH}/assets/object/orange_bottle.usd",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=False,
                disable_gravity=False,
                max_linear_velocity=1000,
                max_angular_velocity=1000,
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=1,
                max_depenetration_velocity=1000.0,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
            activate_contact_sensors=True,
            scale=(1.0, 1.0, 1.0),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            lin_vel=(0.0, 0.0, 0.0),
            ang_vel=(0.0, 0.0, 0.0),
            pos=(0.0, 0.0, 0.0),
        ),
    )

    # sensors — RIGHT hand: tip (*f5) and mid (*f4) of index/middle/pinky/thumb, filtered to Object_0
    contact_sensors_0   = ContactSensorCfg(prim_path="/World/envs/env_.*/Robot/if5", update_period=0.0, debug_vis=True, filter_prim_paths_expr=["{ENV_REGEX_NS}/Object_0"])   # index tip
    contact_sensors_1   = ContactSensorCfg(prim_path="/World/envs/env_.*/Robot/mf5", update_period=0.0, debug_vis=True, filter_prim_paths_expr=["{ENV_REGEX_NS}/Object_0"])   # middle tip
    contact_sensors_2   = ContactSensorCfg(prim_path="/World/envs/env_.*/Robot/pf5", update_period=0.0, debug_vis=True, filter_prim_paths_expr=["{ENV_REGEX_NS}/Object_0"])   # pinky tip
    contact_sensors_3   = ContactSensorCfg(prim_path="/World/envs/env_.*/Robot/th5", update_period=0.0, debug_vis=True, filter_prim_paths_expr=["{ENV_REGEX_NS}/Object_0"])   # thumb tip
    contact_sensors_0_4 = ContactSensorCfg(prim_path="/World/envs/env_.*/Robot/if4", update_period=0.0, debug_vis=True, filter_prim_paths_expr=["{ENV_REGEX_NS}/Object_0"])   # index mid
    contact_sensors_1_4 = ContactSensorCfg(prim_path="/World/envs/env_.*/Robot/mf4", update_period=0.0, debug_vis=True, filter_prim_paths_expr=["{ENV_REGEX_NS}/Object_0"])   # middle mid
    contact_sensors_2_4 = ContactSensorCfg(prim_path="/World/envs/env_.*/Robot/pf4", update_period=0.0, debug_vis=True, filter_prim_paths_expr=["{ENV_REGEX_NS}/Object_0"])   # pinky mid
    contact_sensors_3_4 = ContactSensorCfg(prim_path="/World/envs/env_.*/Robot/th4", update_period=0.0, debug_vis=True, filter_prim_paths_expr=["{ENV_REGEX_NS}/Object_0"])   # thumb mid
    # LEFT hand: same 8 sensors on Robot_left (names: *_left and *_4_left). Note the source also declares
    # contact_sensors_0_left_4 (Robot_left/if4) which duplicates contact_sensors_0_4_left.
    contact_sensors_0_left_4 = ContactSensorCfg(prim_path="/World/envs/env_.*/Robot_left/if4", update_period=0.0, debug_vis=True, filter_prim_paths_expr=["{ENV_REGEX_NS}/Object_0"])
    contact_sensors_0_left   = ContactSensorCfg(prim_path="/World/envs/env_.*/Robot_left/if5", update_period=0.0, debug_vis=True, filter_prim_paths_expr=["{ENV_REGEX_NS}/Object_0"])
    contact_sensors_1_left   = ContactSensorCfg(prim_path="/World/envs/env_.*/Robot_left/mf5", update_period=0.0, debug_vis=True, filter_prim_paths_expr=["{ENV_REGEX_NS}/Object_0"])
    contact_sensors_2_left   = ContactSensorCfg(prim_path="/World/envs/env_.*/Robot_left/pf5", update_period=0.0, debug_vis=True, filter_prim_paths_expr=["{ENV_REGEX_NS}/Object_0"])
    contact_sensors_3_left   = ContactSensorCfg(prim_path="/World/envs/env_.*/Robot_left/th5", update_period=0.0, debug_vis=True, filter_prim_paths_expr=["{ENV_REGEX_NS}/Object_0"])
    contact_sensors_0_4_left = ContactSensorCfg(prim_path="/World/envs/env_.*/Robot_left/if4", update_period=0.0, debug_vis=True, filter_prim_paths_expr=["{ENV_REGEX_NS}/Object_0"])
    contact_sensors_1_4_left = ContactSensorCfg(prim_path="/World/envs/env_.*/Robot_left/mf4", update_period=0.0, debug_vis=True, filter_prim_paths_expr=["{ENV_REGEX_NS}/Object_0"])
    contact_sensors_2_4_left = ContactSensorCfg(prim_path="/World/envs/env_.*/Robot_left/pf4", update_period=0.0, debug_vis=True, filter_prim_paths_expr=["{ENV_REGEX_NS}/Object_0"])
    contact_sensors_3_4_left = ContactSensorCfg(prim_path="/World/envs/env_.*/Robot_left/th4", update_period=0.0, debug_vis=True, filter_prim_paths_expr=["{ENV_REGEX_NS}/Object_0"])

    bottle_top = FrameTransformerCfg(
        prim_path="{ENV_REGEX_NS}/Object_0",
        debug_vis=True,
        visualizer_cfg=FRAME_MARKER_SMALL_CFG.replace(prim_path="/Visuals/BottomTopFrameTransformer"),
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/Object_0",
                name="approach_frame",
                offset=OffsetCfg(pos=(0.0, 0.0, 0.07), rot=(1.0, 0.0, 0.0, 0.0)),
            ),
        ],
    )

    bottle_bottom = FrameTransformerCfg(
        prim_path="{ENV_REGEX_NS}/Object_0",
        debug_vis=True,
        visualizer_cfg=FRAME_MARKER_SMALL_CFG.replace(prim_path="/Visuals/BottomBottomFrameTransformer"),
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/Object_0",
                name="approach_frame",
                offset=OffsetCfg(pos=(0.0, 0.0, -0.08), rot=(1.0, 0.0, 0.0, 0.0)),
            ),
        ],
    )
```

Shared `BaseSceneCfg` (table/ground/light) — VERBATIM from `manager_based_env_cfg.py`:
```python
@configclass
class BaseSceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -0.82)),
    )
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=2500.0),
    )
    table: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Table",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{bimanual_suite.LIB_PATH}/assets/object/table.usd",
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True, disable_gravity=False,
                solver_position_iteration_count=16, solver_velocity_iteration_count=1,
                max_depenetration_velocity=10.0,
            ),
            scale=(1.0, 1.0, 1.0),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0), rot=(0.70710678, 0, 0., 0.70710678)),
    )
    replicate_physics = False
```

Sim timing / physx (`BaseEnvCfg` + `__post_init__`) — VERBATIM:
```python
    sim: SimulationCfg = SimulationCfg(
        physics_material=RigidBodyMaterialCfg(
            static_friction=1.5, dynamic_friction=1.0, restitution=0.0, restitution_combine_mode=min,
        ),
        physx=PhysxCfg(gpu_max_rigid_contact_count=2**24, gpu_max_rigid_patch_count=2**24),
    )
    def __post_init__(self):
        self.decimation = 6
        self.episode_length_s = 8.3333
        self.viewer.eye = (3.5, 3.5, 3.5)
        self.sim.dt = 1.0 / 120.0
        self.sim.render_interval = self.decimation
```

Top-level `HandoverEnvCfg` — VERBATIM:
```python
@configclass
class HandoverEnvCfg(BaseEnvCfg):
    name: str = "Handover"
    scene = HandoverSceneCfg(num_envs=4096, env_spacing=3.0)
    events = HandoverEventCfg()
    commands = HandoverCommandsCfg()
    observations = HandoverObservationsCfg()
    actions = HandoverActionsCfg()
    terminations = HandoverTerminationsCfg()
    rewards = HandoverRewardsCfg()
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
        super().__post_init__()
```

---

## §2 Actions

**Description.** Two action terms, one per arm, both `EMACumulativeRelativeJointPositionAction` over ALL joints (`joint_names=[".*"]` → 22 joints each: 6 arm + 16 hand). `action_dim = 44` (22+22). The per-step rule (see `actions.py::process_actions`): the raw policy action is first multiplied by `cfg.scale` (=1.0 here), then in `BaseEnv.step` the policy action is ALSO multiplied element-wise by the fixed `cfg.action_scale` vector (per-joint step sizes, 0.05 for arm joints, 0.03/0.015 for fingers) BEFORE being handed to the action manager. Inside `process_actions`: the scaled action is a **delta**; it accumulates into `del_action` (cumulative relative), adds the reset-time joint position `init_joint_pos`, then an EMA with `alpha=0.2` blends the new target with the previous applied target, and finally clamps to `[joint_lower_limit, joint_upper_limit]`. So the action is a **cumulative-relative EMA-smoothed absolute joint-position target**.

**Decisions resolved.**

| Decision | Value | Source |
|---|---|---|
| Action term class | `EMACumulativeRelativeJointPositionAction` (subclass of IsaacLab `JointPositionAction`) | actions.py |
| Right term | `arm_hand_action` on `robot`, `joint_names=[".*"]`, scale 1.0, alpha 0.2 | env_cfg.py |
| Left term | `arm_hand_action_left` on `robot_left`, same | env_cfg.py |
| use_default_offset | False | env_cfg.py |
| Per-joint limits | `JOINT_LOWER/UPPER_LIMIT` (right), `..._LEFT` (left) — see §2 code | manager_based_env_cfg.py |
| action_dim | 44 (= 22 right + 22 left) | env_cfg.py `action_dim=44` |
| EMA alpha | 0.2 (float, same for all joints) | env_cfg.py |
| action_scale | 44-vec, arm 0.05, fingers 0.03, jth3 0.015 | env_cfg.py |
| Where action_scale applied | `BaseEnv.step`: `action = action * self._scale` before `super().step` | manager_based_env.py |

**Code (verbatim).**

`HandoverActionsCfg` (env_cfg.py):
```python
@configclass
class HandoverActionsCfg:
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

`EMACumulativeRelativeJointPositionActionCfg` (actions_cfg.py):
```python
@configclass
class EMACumulativeRelativeJointPositionActionCfg(JointPositionActionCfg):
    class_type: type[ActionTerm] = EMACumulativeRelativeJointPositionAction
    alpha: float | dict[str, float] = 1.0
    joint_lower_limit: list[float] = None
    joint_upper_limit: list[float] = None
```

Per-step rule (`actions.py::EMACumulativeRelativeJointPositionAction`):
```python
    def __init__(self, cfg, env) -> None:
        super().__init__(cfg, env)
        if isinstance(cfg.alpha, float):
            if not 0.0 <= cfg.alpha <= 1.0:
                raise ValueError(...)
            self._alpha = cfg.alpha
        elif isinstance(cfg.alpha, dict):
            self._alpha = torch.ones((env.num_envs, self.action_dim), device=self.device)
            index_list, names_list, value_list = string_utils.resolve_matching_names_values(cfg.alpha, self._joint_names)
            for name, value in zip(names_list, value_list):
                if not 0.0 <= value <= 1.0: raise ValueError(...)
            self._alpha[:, index_list] = torch.tensor(value_list, device=self.device)
        else:
            raise ValueError(...)
        self._prev_applied_actions = torch.zeros_like(self.processed_actions)
        self.del_action = torch.zeros((self._env.num_envs, self.action_dim), device=self._env.device)
        self.init_joint_pos = self._asset.data.joint_pos[:, self._joint_ids].clone()
        self.joint_lower_limit = torch.tensor(cfg.joint_lower_limit, device=self.device) if cfg.joint_lower_limit is not None else None
        self.joint_upper_limit = torch.tensor(cfg.joint_upper_limit, device=self.device) if cfg.joint_upper_limit is not None else None

    def reset(self, env_ids=None) -> None:
        if env_ids is None: env_ids = slice(None)
        super().reset(env_ids)
        self._prev_applied_actions[env_ids, :] = self._asset.data.joint_pos[env_ids][:, self._joint_ids].clone()
        self.del_action[env_ids, :] = torch.zeros((env_ids.shape[0], self.action_dim), device=self.device)
        self.init_joint_pos[env_ids, :] = self._asset.data.joint_pos[env_ids][:, self._joint_ids].clone()

    def process_actions(self, actions: torch.Tensor):
        super().process_actions(actions)          # affine scale (cfg.scale=1.0)
        self._processed_actions += self.del_action # cumulative delta
        self.del_action = self._processed_actions.clone()
        self._processed_actions += self.init_joint_pos.clone()   # add reset-time abs pos
        ema_actions = self._alpha * self._processed_actions
        ema_actions += (1.0 - self._alpha) * self._prev_applied_actions   # EMA (alpha=0.2)
        if self.joint_lower_limit is not None and self.joint_upper_limit is not None:
            self._processed_actions[:] = torch.clamp(ema_actions, self.joint_lower_limit, self.joint_upper_limit)
        else:
            self._processed_actions[:] = ema_actions
        self._prev_applied_actions[:] = self._processed_actions[:]
```

Joint limit vectors (`manager_based_env_cfg.py`) — order is [joint1-6, then jif1/jmf1/jpf1/jth1, jif2/jmf2/jpf2/jth2, jif3.., jif4..]:
```python
JOINT_LOWER_LIMIT = [-6.283, -2.304, -4.224, -6.283, -2.164, -6.283,
                    -0.05, -0.05, -0.570, 0.364,
                    -0.296, -0.296, -0.296, -0.205,
                    -0.274, -0.274, -0.274, -0.290,
                    -0.327, -0.327, -0.327, -0.262]
JOINT_UPPER_LIMIT = [6.283, 2.304, 0.061, 6.283, 2.164, 6.283,
                    0.570, 0.05, 0.05, 1.497,
                    1.710, 1.710, 1.710, 1.130,
                    1.809, 1.809, 1.809, 1.633,
                    1.718, 1.718, 1.718, 1.820]
# LEFT differs only in the first hand row (jif1/jmf1/jpf1 signs flipped for the mirrored hand):
JOINT_LOWER_LIMIT_LEFT = [... arm same ..., -0.570, -0.05, -0.05, 0.364, ... rest same ...]
JOINT_UPPER_LIMIT_LEFT = [... arm same ...,  0.05,  0.05,  0.570, 1.497, ... rest same ...]
```

Action-scale application in the base env (`manager_based_env.py::BaseEnv.step`):
```python
    def step(self, action: torch.Tensor) -> VecEnvStepReturn:
        self.last_action = action.clone()
        action = action * self._scale     # self._scale = torch.tensor(cfg.action_scale)
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

**Description.** `HandoverEventCfg(BaseEventCfg)`. On reset (`mode="reset"`): (1) `reset_robot_joints` (inherited from `BaseEventCfg`) resets the RIGHT robot to its default pose scaled ×1.0 (i.e. exact default) via `reset_joints_by_symmetry` — which additionally MIRRORS the pose from the opposite robot for envs flagged symmetric (`symmetry_tracker==1`); (2) `reset_robot_joints_left` does the same for `robot_left`; (3) `reset_object` drops `object_0` at a FIXED offset `x=0.2, y=-0.25, z=0.12` (in front of the right hand), yaw randomized in `[-π, π]`. On startup (`mode="startup"`): `object_mass` scales the bottle mass by a uniform factor in `[0.05, 0.9]`. Note the fixed-value ranges (`[a, a]`) mean deterministic placement except the object yaw. The bottle's actual initial world pose is captured post-reset into `env.object_init_pos/orient` (used by reward normalizers).

**Decisions resolved.**

| Decision | Value | Source |
|---|---|---|
| Reset right robot joints | `reset_joints_by_symmetry`, position_range (1.0,1.0), vel (0,0) | BaseEventCfg |
| Reset left robot joints | `reset_joints_by_symmetry` on `robot_left`, same ranges | HandoverEventCfg |
| Reset object pose | x=0.2, y=-0.25, z=0.12 (fixed), yaw ∈ [-3.14, 3.14] | HandoverEventCfg |
| Object velocity range | {} (zero) | HandoverEventCfg |
| Object mass DR | startup, scale × U(0.05, 0.9) | HandoverEventCfg (see §7) |
| Symmetric reset | first reset never symmetric; thereafter 50% of reset envs mirrored (`_reset_idx`) | manager_based_env.py |

**Code (verbatim).**

`HandoverEventCfg` (env_cfg.py):
```python
@configclass
class HandoverEventCfg(BaseEventCfg):
    reset_robot_joints_left = EventTerm(
        func=reset_joints_by_symmetry,
        mode="reset",
        params={"position_range": (1.0, 1.0), "velocity_range": (0.0, 0.0), "asset_cfg": SceneEntityCfg("robot_left")},
    )
    reset_object = EventTerm(
        func=reset_object,
        mode="reset",
        params={
            "pose_range": {"x": [0.2, 0.2], "y": [-0.25, -0.25], "z": [0.12, 0.12], "yaw": [-3.14, 3.14]},
            "velocity_range": {},
            "object_id": 0,
        },
    )
    object_mass = EventTerm(
        func=randomize_rigid_body_mass,
        mode="startup",
        params={"mass_distribution_params": (0.05, 0.9), "operation": "scale"},
    )
```

Inherited `BaseEventCfg.reset_robot_joints` (manager_based_env_cfg.py):
```python
@configclass
class BaseEventCfg:
    reset_robot_joints = EventTerm(
        func=reset_joints_by_symmetry,
        mode="reset",
        params={"position_range": (1.0, 1.0), "velocity_range": (0.0, 0.0)},
    )
```

`reset_joints_by_symmetry` + `reset_object` (reset_mdps.py):
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

Post-reset capture of object init pose (`manager_based_env.py::_post_reset_process`):
```python
    def _post_reset_process(self, env_ids):
        for obj_id in range(self.num_object):
            self.object_init_pos[obj_id, env_ids, :3] = self.scene[f"object_{obj_id}"].data.root_pos_w[env_ids, :3]
            self.object_init_orient[obj_id, env_ids, :4] = self.scene[f"object_{obj_id}"].data.root_quat_w[env_ids, :4]
        if not self.cfg.hydra_cfg.task.symmetry.symmetric_envs:
            info = self.command_manager.reset(env_ids)
            self.extras["log"].update(info)
```

---

## §4 Goal + Termination

**Description.** Two commands. `target_pos` (grasp-style `TargetPositionCommand`) is the **handover goal pose** for the bottle: mid-air at env-frame `x=0.15, y=0.0, z=0.25` with a roll of −1.57 (bottle tipped), position success_threshold 0.05, orientation success_threshold 0.95 (dot-product on z-axis). `left_hand_target_pos` (reach-style command) is a target palm pose for the LEFT robot at `x=0.15, y=0.07, z=0.37, pitch=1.57`. Neither resamples on success (`update_goal_on_success=False`) nor on time (resampling_time_range = 1e6). Success predicate (the actual handover check) lives in reward `success_bonus` and termination `max_consecutive_success`:

- `env.reach_middle` (in `env.py::step`) increments each step that `target_pos` consecutive_success ≥ 5 → tracks "bottle reached the middle handover pose".
- `success_bonus` (§6) defines the TRUE handover event: `position_error < 0.08 AND orientation_error > 0.95 AND right-hand NOT in contact AND left-hand IS in contact`; it accumulates `env.success_tracker_step` for that condition held consecutively.
- Termination `max_consecutive_success(num_success=10)`: done when `env.success_tracker_step >= 10` (bottle held at handover pose by the LEFT hand, released by right, for 10 steps). Sets `env.success_tracker`.
- `time_out` (inherited): episode ends at `episode_length_s = 8.3333 s`.

**Decisions resolved.**

| Decision | Value | Source |
|---|---|---|
| Handover goal (`target_pos`) | pose x=0.15,y=0.0,z=0.25,roll=-1.57; thr 0.05 / orient 0.95 | HandoverCommandsCfg |
| Left-hand target (`left_hand_target_pos`) | pose x=0.15,y=0.07,z=0.37,pitch=1.57; thr 0.05 / orient 0.95 | HandoverCommandsCfg |
| Resample on success | False (both) | cfg |
| Resample on time | never (1e6, 1e6) | command cfg default |
| Success predicate | pos_err<0.08 & orient>0.95 & right no-contact & left contact, ×10 consecutive | success_bonus + max_consecutive_success |
| Termination (success) | `handover.max_consecutive_success(num_success=10)` on `success_tracker_step` | HandoverTerminationsCfg |
| Time-out | `mdp.time_out`, 8.3333 s | BaseTerminationsCfg |
| Failure termination | none beyond time-out | — |
| Goal markers | grasp-command frame_prim.usd; reach-command frame_prim.usd; both debug_vis=True | command cfgs |

**Code (verbatim).**

`HandoverCommandsCfg` (env_cfg.py):
```python
@configclass
class HandoverCommandsCfg(BaseCommandsCfg):
    target_pos = TargetPositionCommandCfg(
        object_id=0,
        success_threshold=0.05,
        success_threshold_orient=0.95, # 60 degree 
        pose_range={"x": [0.15, 0.15], "y": [0.0, 0.0], "z": [0.25, 0.25], "roll": [-1.57, -1.57]},
        update_goal_on_success=False,
        debug_vis=True,
    )
    left_hand_target_pos = ReachCommandCfg(
        asset_cfg=SceneEntityCfg("robot_left"),
        target_link="palm_link",
        success_threshold=0.05,
        success_threshold_orient=0.95, # 60 degree 
        pose_range={"x": [0.15, 0.15], "y": [0.07, 0.07], "z": [0.37, 0.37], "pitch": [1.57, 1.57]},
        update_goal_on_success=False,
        debug_vis=True,
    )
```

`HandoverTerminationsCfg` (env_cfg.py) + `time_out` (BaseTerminationsCfg):
```python
@configclass
class HandoverTerminationsCfg(BaseTerminationsCfg):
    max_consecutive_success = DoneTerm(func=handover.max_consecutive_success, params={"num_success": 10})
# inherited:
@configclass
class BaseTerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
```

`max_consecutive_success` (Handover/mdps.py) — success predicate driver:
```python
def max_consecutive_success(env: ManagerBasedRLEnv, num_success: int) -> torch.Tensor:
    success = env.success_tracker_step >= num_success
    env.success_tracker = success.float()
    return success
```

Grasp-command metric/success internals (`grasp_command.py::TargetPositionCommand`):
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

`reach_middle` update (env.py::step): `self.reach_middle += self.command_manager.get_term("target_pos").metrics["consecutive_success"] >= 5`

---

## §5 Observation

**Description.** Single `policy` group, concatenated, corruption DISABLED (`enable_corruption=False` — the commented `noise=` on several terms is inactive). Bimanual + object + command + bookkeeping. Robot joint count = 22 each (6 arm + 16 Allegro). `ee_pose` uses the "symmetry" rotation-matrix flattening (quat→3×3→9), so each EE pose is 12 (pos 3 + R_flat 9). `object_quat`/`generated_commands` likewise flatten to R (9). **Total obs dim = 181.**

**Decisions resolved (per-term dims).**

| Term | func | params | dim |
|---|---|---|---|
| ee_pose_right | `ee_pose` | ee_name="palm_link" (robot) | 12 (3+9 R_flat) |
| joint_pos_right | `joint_pos_limit_normalized` | joints=None, JOINT_*_LIMIT | 22 |
| joint_vel_right | `joint_vel` | joints=None | 22 |
| ee_pose_left | `ee_pose` | ee_name="palm_link", robot_left | 12 |
| joint_pos_left | `joint_pos_limit_normalized` | joints=None, *_LEFT, robot_left | 22 |
| joint_vel_left | `joint_vel` | joints=None, robot_left | 22 |
| bottle_pos | `object_pos` | object_id=0 | 3 |
| bottle_quat | `object_quat` | object_id=0, symmetry=True | 9 (R_flat) |
| handover_pos | `generated_commands` | command_name="target_pos", symmetry=True | 12 (3+9) |
| last_action | `last_action` | — | 44 |
| symmetry_tracker | `symmetry_tracker` | — | 1 |
| **Total** | | | **181** |

Flags: `enable_corruption=False`, `concatenate_terms=True`.

**Code (verbatim).**

`HandoverObservationsCfg.PolicyCfg` (env_cfg.py):
```python
@configclass
class HandoverObservationsCfg(BaseObservationsCfg):
    @configclass
    class PolicyCfg(ObsGroup):
        ee_pose_right = ObsTerm(func=ee_pose, params={"ee_name": "palm_link"})
        joint_pos_right = ObsTerm(func=joint_pos_limit_normalized, params={"joints": None,
                                    "joint_lower_limit": JOINT_LOWER_LIMIT, "joint_upper_limit": JOINT_UPPER_LIMIT}, )
        joint_vel_right = ObsTerm(func=joint_vel, params={"joints": None},)
        ee_pose_left = ObsTerm(func=ee_pose, params={"ee_name": "palm_link", "asset_cfg": SceneEntityCfg("robot_left")})
        joint_pos_left = ObsTerm(func=joint_pos_limit_normalized, params={"joints": None,
                                    "joint_lower_limit": JOINT_LOWER_LIMIT_LEFT, "joint_upper_limit": JOINT_UPPER_LIMIT_LEFT,
                                    "asset_cfg": SceneEntityCfg("robot_left")}, )
        joint_vel_left = ObsTerm(func=joint_vel, params={"joints": None, "asset_cfg": SceneEntityCfg("robot_left")},)
        bottle_pos = ObsTerm(func=object_pos, params={"object_id": 0})
        bottle_quat = ObsTerm(func=object_quat, params={"object_id": 0, "symmetry": True}, )
        handover_pos = ObsTerm(func=generated_commands, params={"command_name": "target_pos"})
        last_action = ObsTerm(func=last_action)
        symmetry_tracker = ObsTerm(func=symmetry_tracker)
        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True
    policy: PolicyCfg = PolicyCfg()
```

Observation functions (obs_mdps.py) — `ee_pose`, `object_quat`, `generated_commands`, `symmetry_tracker`:
```python
def ee_pose(env, ee_name, asset_cfg=SceneEntityCfg("robot"), symmetry=True):
    robot = env.scene[asset_cfg.name]
    ee_idx = robot.find_bodies(ee_name)[0]
    ee_w = robot.data.body_state_w[:, ee_idx, :7].clone().reshape(-1, 7)
    ee_w[:, :3] = ee_w[:, :3] - env.scene.env_origins
    if symmetry:
        ee_R = math_utils.matrix_from_quat(ee_w[:, 3:7])
        ee_R_flat = ee_R.transpose(1, 2).reshape(-1, 9)
        ee_w = torch.cat([ee_w[:, :3], ee_R_flat], dim=-1)
    return ee_w                                   # -> 12

def object_quat(env, make_quat_unique=False, object_id=0, symmetry=True):
    object = env.scene[f"object_{object_id}"]
    if symmetry:
        object_R = math_utils.matrix_from_quat(object.data.root_quat_w)
        object_R_flat = object_R.transpose(1, 2).reshape(-1, 9)
        return object_R_flat                       # -> 9
    else:
        return math_utils.quat_unique(object.data.root_quat_w) if make_quat_unique else object.data.root_quat_w

def generated_commands(env, command_name, symmetry=True):
    command = env.command_manager.get_command(command_name)
    if symmetry:
        command_R = math_utils.matrix_from_quat(command[:, 3:])
        command_R = command_R.transpose(1, 2).reshape(-1, 9)
    else:
        command_R = command[:, :4]
    return torch.cat([command[:, :3], command_R], dim=-1)   # -> 12

def symmetry_tracker(env):
    if hasattr(env, "symmetry_tracker") and env.symmetry_tracker is not None:
        symmetry_tracker = env.symmetry_tracker.clone()
        symmetry_tracker[symmetry_tracker == 0] = -1
        return symmetry_tracker.reshape(-1, 1)     # -> 1
    else:
        return torch.ones((env.num_envs, 1), device=env.device) * -1
```

---

## §6 Reward

**Description.** `HandoverRewardsCfg(BaseRewardsCfg)` — a large staged bank. Each `RewTerm` carries its own scalar weight (listed below); the term also fixes the reward *shape* and its internal per-link `params["weight"]` vectors. The full set exists TWICE: a primary block (right hand leads, left receives) and a `_symmetry` mirror block (roles swapped, used for envs where `symmetry_tracker==1`). Base rewards add `energy` / `energy_left` (hand-actuator energy penalty, also weight 0). The bank composes as a **weighted sum** (standard IsaacLab `RewardManager`). Nearly every term is gated on `env.reach_middle` and on filtered contact. Inherited base energy terms use `energy_punishment`.

Stage semantics (right-leading primary block):
- **reach** — `reaching_object` (`frame_marker_robot_distance` to `bottle_bottom`, per-link weights [1,1,1,1.5,2] over [if5,mf5,pf5,th5,palm], `1/dist` shaping when not-left).
- **grasp+lift to middle** — `object_goal_tracking` (`object_goal_distance` toward `target_pos`, gated on right-hand AND-contact + palm-close) and `object_goal_orient_tracking` (`object_goal_orient_distance`, z-axis).
- **middle hold** — `middle_success_bonus` (`cmd_success_bonus`, num_success=1, if_right — zeroed once consecutive_success>15 to stop farming).
- **release prep** — `contact_bottle_punish` (penalty proportional to right-hand contact force, +20 when released, gated `reach_middle>10`) and `reset_robot_joint_pos` (`robot_goal_distance` pulls right palm back to [0.0462,-0.5045,0.4468] once released & reach_middle>10).
- **left approach** — `left_align_hand_pose` (`align_hand_pose` to `left_hand_target_pos`), `left_align_finger_joint` (`align_finger_joint` back to default finger pose, gated reach_middle==0), `left_reaching_object` (`frame_marker_robot_distance` to `bottle_top`, if_left, gated reach_middle>0).
- **left grasp/track** — `left_object_goal_tracking`, `left_object_goal_orient_tracking` (if_left, left contact), `middle_success_bonus_left`.
- **success** — `success_bonus` (the true handover event; increments `success_tracker_step`).

**Decisions resolved (RewTerm inventory — 28 terms).**

| # | Term name | func | key params | RewTerm weight |
|---|---|---|---|---|
| B | energy | energy_punishment | robot hand actuators | (0.0 — not set) |
| B | energy_left | energy_punishment | robot_left hand actuators | (0.0 — not set) |
| 1 | reaching_object | frame_marker_robot_distance | w=[1,1,1,1.5,2], links [if5,mf5,pf5,th5,palm], bottle_bottom | 0.05 |
| 2 | object_goal_tracking | object_goal_distance | target_pos, object_id 0 | 10.0 |
| 3 | object_goal_orient_tracking | object_goal_orient_distance | target_pos, axis z | 2.0 |
| 4 | middle_success_bonus | cmd_success_bonus | num_success 1, if_right | 25.0 |
| 5 | contact_bottle_punish | contact_bottle_punish | right 8 sensors | 4.0 |
| 6 | reset_robot_joint_pos | robot_goal_distance | target [0.0462,-0.5045,0.4468], palm, right sensors | 200.0 |
| 7 | left_align_hand_pose | align_hand_pose | left_hand_target_pos, robot_left | 0.2 |
| 8 | left_align_finger_joint | align_finger_joint | 16 finger joints, robot_left | 1.0 |
| 9 | left_reaching_object | frame_marker_robot_distance | w=[1.5,1,1,2], [if5,mf5,pf5,th5], bottle_top, if_left, robot_left | 40.0 |
| 10 | left_object_goal_tracking | object_goal_distance | target_pos, if_left, left sensors | 20.0 |
| 11 | left_object_goal_orient_tracking | object_goal_orient_distance | target_pos, z, if_left, left sensors | 5.0 |
| 12 | middle_success_bonus_left | cmd_success_bonus | num_success 1, if_left, left sensors | 20.0 |
| 13 | success_bonus | success_bonus | num_success 10, right=not-contact, left=is-contact | 3000.0 |
| 14-25 | *_symmetry (mirror of 1-13, minus middle_success_bonus) | same funcs, robot/robot_left swapped | see code | same weight as base counterpart |

> Weights in the table above are each term's scalar `RewTerm.weight` (the `_symmetry` mirror terms 14-25 each take the same weight as their base counterpart). The `energy` / `energy_left` base terms carry `weight=0.0`. See the *Symmetric-learning reward terms* note below for the `_symmetry` duplicates.

### Symmetric-learning reward terms (drop if not using symmetric learning)

This task trains with **SYMMETRIC LEARNING** enabled (`base.yaml` → `symmetry.symmetric_envs: True`, C2 group). The **BASE reward set** = the right-hand terms **plus** their `left_`/`_left` counterparts — both are genuine and load-bearing for the bimanual handover (right hand lifts to the middle, left hand receives), so keep all of them.

The `_symmetry`-suffixed terms are **DUPLICATES** used for symmetric-learning data augmentation ONLY (the mirrored C2 copy applied to envs where `symmetry_tracker==1`). **If you are NOT using symmetric learning, drop every `_symmetry` term.** The `_symmetry` duplicate names:

- reaching_object_symmetry
- object_goal_tracking_symmetry
- object_goal_orient_tracking_symmetry
- contact_bottle_punish_symmetry
- reset_robot_joint_pos_symmetry
- left_align_hand_pose_symmetry
- left_align_finger_joint_symmetry
- left_reaching_object_symmetry
- left_object_goal_tracking_symmetry
- left_object_goal_orient_tracking_symmetry
- middle_success_bonus_left_symmetry
- success_bonus_symmetry

(Note: `middle_success_bonus` — the right-hand middle bonus — has no `_symmetry` duplicate in the Hydra config; the other 12 base terms each do.)

**Code (verbatim) — reward functions (`Handover/mdps.py`).**
```python
def get_allegro_contact(env, sensor_names: list):
    force = get_force(env, sensor_names, if_filter=True)
    is_contact = (torch.norm(force, dim=-1) > 1.0)
    if len(sensor_names) == 4:
        is_contact_index_or_middle_or_ring = reduce(torch.logical_or, [is_contact[:, 0], is_contact[:, 1], is_contact[:, 2]])
        is_contact = reduce(torch.logical_and, [is_contact_index_or_middle_or_ring, is_contact[:, 3]])
    elif len(sensor_names) == 8:
        is_contact_index_or_middle_or_ring = reduce(torch.logical_or, [is_contact[:, 0], is_contact[:, 1], is_contact[:, 2], is_contact[:, 4], is_contact[:, 5], is_contact[:, 6]])
        is_contact_thumb = reduce(torch.logical_or, [is_contact[:, 3], is_contact[:, 7]])
        is_contact = reduce(torch.logical_and, [is_contact_index_or_middle_or_ring, is_contact_thumb])
    return is_contact

def check_release_object(env, sensor_names: list):
    force = get_force(env, sensor_names, if_filter=True)
    is_not_contact = (torch.norm(force, dim=-1) < 1.0)
    is_not_contact = reduce(torch.logical_and, [is_not_contact[:, i] for i in range(len(sensor_names))])
    return is_not_contact, force

def contact_bottle_punish(env, sensor_names: list):
    is_not_contact, force = check_release_object(env, sensor_names)
    rew = -torch.mean(torch.abs(force), dim=(-1, -2))
    rew[is_not_contact] = 20.0
    return rew * (env.reach_middle > 10)

def robot_goal_distance(env, target_pos, target_link, sensor_names, asset_cfg=SceneEntityCfg("robot")):
    des_pos_w = torch.tensor(target_pos, device=env.device) + env.scene.env_origins
    target_link_idx = env.scene[asset_cfg.name].find_bodies([target_link])[0]
    distance = torch.norm(des_pos_w - env.scene[asset_cfg.name].data.body_state_w[:, target_link_idx, 0:3].squeeze(1), dim=1)
    is_not_contact, force = check_release_object(env, sensor_names)
    rew = torch.clamp(0.5 - distance, min=0.0) * (is_not_contact).float() * (env.reach_middle > 10).float()
    return rew

def frame_marker_robot_distance(env, weight, link_name, if_left=False, asset_cfg=SceneEntityCfg("robot"), frame_name="bottle_top"):
    weight = torch.tensor(weight, device=env.device)
    object_pos_w = env.scene[frame_name].data.target_pos_w.reshape(-1, 1, 3)
    robot = env.scene[asset_cfg.name]
    link_idx = robot.find_bodies(link_name)[0]
    link_w = robot.data.body_pos_w[:, link_idx, :3]
    object_link_distance = torch.norm(object_pos_w - link_w, dim=-1) * weight
    if if_left:
        weighted_object_link_distance = torch.sum(object_link_distance, dim=1)
        rew = torch.clamp(0.6 - weighted_object_link_distance, min=0.0) * (env.reach_middle > 0).float()
    else:
        weighted_object_link_distance = torch.mean(object_link_distance, dim=1)
        rew = 1 / weighted_object_link_distance
    return rew

def align_hand_pose(env, link_name, command_name="target_pos", asset_cfg=SceneEntityCfg("robot")):
    command_term = env.command_manager.get_term(command_name)
    robot = env.scene[asset_cfg.name]
    link_idx = robot.find_bodies(link_name)[0]
    robot_state_w = robot.data.body_state_w[:, link_idx, :7].reshape(-1, 7)
    ori_distance = math_utils.quat_error_magnitude(robot_state_w[:, 3:7], command_term.quat_command_w)
    distance = torch.norm(robot_state_w[:, :3] - command_term.pos_command_w, dim=-1)
    rew = -ori_distance - 2 * distance * (env.reach_middle == 0).float()
    return rew

def align_finger_joint(env, link_name, asset_cfg=SceneEntityCfg("robot")):
    joint_idx = env.scene[asset_cfg.name].find_joints(link_name)[0]
    joint_pos = env.scene[asset_cfg.name].data.joint_pos[:, joint_idx]
    default_joint_pos = env.scene[asset_cfg.name].data.default_joint_pos[:, joint_idx]
    distance = torch.norm(joint_pos - default_joint_pos, dim=-1)
    distance = -distance * (env.reach_middle == 0).float()
    return distance

def object_goal_distance(env, command_name, object_id=0, if_left=False, asset_cfg=SceneEntityCfg("robot"),
                         sensor_names=["contact_sensors_0","contact_sensors_1","contact_sensors_2","contact_sensors_3"]):
    object = env.scene[f"object_{object_id}"]
    command = env.command_manager.get_command(command_name)
    des_pos_w = command[:, :3] + env.scene.env_origins
    distance = torch.norm(des_pos_w - object.data.root_pos_w[:, :3], dim=1)
    if if_left:
        distance = torch.clamp(0.33 - distance, min=0.0)
        rew = distance * get_allegro_contact(env, sensor_names)
        rew = rew * (env.reach_middle > 0).float()
    else:
        max_distance = torch.norm(des_pos_w - env.object_init_pos[object_id], dim=1)
        distance = torch.clamp(max_distance - distance, min=0.0)
        rew = distance * get_allegro_contact(env, sensor_names)
        object_pos_w = env.scene["bottle_bottom"].data.target_pos_w.reshape(-1, 1, 3)
        robot = env.scene[asset_cfg.name]
        link_idx = robot.find_bodies("palm_link")[0]
        link_w = robot.data.body_pos_w[:, link_idx, :3]
        object_link_distance = torch.norm(object_pos_w - link_w, dim=-1)
        rew = rew * (object_link_distance[:, -1] < 0.15).float()
    return rew

def object_goal_orient_distance(env, command_name, object_id=0, if_left=False, axis="z",
                                sensor_names=["contact_sensors_0","contact_sensors_1","contact_sensors_2","contact_sensors_3"]):
    object = env.scene[f"object_{object_id}"]
    command_term = env.command_manager.get_term(command_name)
    des_orient_w = command_term.quat_command_w
    if axis is not None:
        from bimanual_suite.utils.isaac_utils import get_angle_from_quat
        target_axis = get_angle_from_quat(des_orient_w, axis=axis, normalize=True)
        cur_axis = get_angle_from_quat(object.data.root_quat_w, axis=axis, normalize=True)
        distance = torch.sum(target_axis * cur_axis, dim=-1) * (2**0.5)
        distance = torch.clamp(distance, min=0.0)
    else:
        distance = math_utils.quat_error_magnitude(object.data.root_quat_w, des_orient_w)
        max_distance = math_utils.quat_error_magnitude(env.object_init_orient[object_id], des_orient_w)
        distance = torch.clamp(max_distance - distance, min=0.0)
    if if_left:
        rew = distance * get_allegro_contact(env, sensor_names) * (env.reach_middle > 0).float()
    else:
        rew = distance * get_allegro_contact(env, sensor_names) * (object.data.root_pos_w[:, 2] > command_term.command[:, 2] - 0.08)
    return rew

def cmd_success_bonus(env, command_names, num_success=0, if_right=False, if_left=False,
                      sensor_names=["contact_sensors_0","contact_sensors_1","contact_sensors_2","contact_sensors_3"]):
    command_term = env.command_manager.get_term(command_names)
    rew = command_term.metrics["consecutive_success"] >= num_success
    if if_right:
        num_success = env.command_manager.get_term(command_names).metrics["consecutive_success"]
        rew[num_success > 15] = 0.0
    elif if_left:
        rew = rew * get_allegro_contact(env, sensor_names)
    return rew

def success_bonus(env, command_names, num_success=0, symmetry=False, not_contact_sensor_names=None, is_contact_sensor_names=None):
    is_not_contact, _ = check_release_object(env, not_contact_sensor_names)
    cmd_term = env.command_manager.get_term(command_names)
    within_range = torch.logical_and(cmd_term.metrics["position_error"] < 0.08, cmd_term.metrics["orientation_error"] > 0.95)
    success = within_range * is_not_contact * get_allegro_contact(env, is_contact_sensor_names)
    if symmetry:
        valid_envs = torch.where(env.symmetry_tracker == 1)[0]
    else:
        valid_envs = torch.where(env.symmetry_tracker == 0)[0]
    if len(valid_envs) > 0:
        success_idx = valid_envs[success[valid_envs].nonzero(as_tuple=True)[0]]
        failure_idx = valid_envs[(~success[valid_envs]).nonzero(as_tuple=True)[0]]
        old_success_tracker_step = env.success_tracker_step.clone()
        env.success_tracker_step[success_idx] += 1
        env.success_tracker_step[failure_idx] = 0
        rew = (old_success_tracker_step == num_success - 1) * (env.success_tracker_step == num_success)
        return rew.float()
    else:
        return torch.zeros_like(success).float()
```

Helper `get_force` (reward_mdps.py — filtered force matrix per sensor):
```python
def get_force(env, sensor_names: list, if_filter: bool = False):
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
```

Base energy reward (manager_based_env_cfg.py + reward_mdps.py):
```python
@configclass
class BaseRewardsCfg:
    energy = RewTerm(func=energy_punishment, weight=0.0,
        params={"asset_cfg": SceneEntityCfg("robot"), "actuator_name": ["allegro_hand_1","allegro_hand_2","allegro_hand_3","allegro_hand_4",
                "allegro_hand_thumb_1","allegro_hand_thumb_2","allegro_hand_thumb_3","allegro_hand_thumb_4"]})
    energy_left = RewTerm(func=energy_punishment, weight=0.0,
        params={"asset_cfg": SceneEntityCfg("robot_left"), "actuator_name": [... same 8 ...]})

def energy_punishment(env, actuator_name=None, asset_cfg=SceneEntityCfg("robot")):
    if actuator_name is None:
        energy = get_energy_consumption(env=env, robot_name=asset_cfg.name)
    else:
        energy = get_actuator_energy_consumption(env=env, robot_name=asset_cfg.name, actuator_name=actuator_name)
    return -energy
```

**Full `HandoverRewardsCfg` term list (env_cfg.py) — primary block + symmetry mirror:**
```python
@configclass
class HandoverRewardsCfg(BaseRewardsCfg):
    reaching_object = RewTerm(func=handover.frame_marker_robot_distance,
        params={"weight": [1.0, 1.0, 1.0, 1.5, 2.0], "link_name": ["if5","mf5","pf5","th5","palm_link"], "frame_name": "bottle_bottom"}, weight=0.05)
    object_goal_tracking = RewTerm(func=handover.object_goal_distance,
        params={"command_name": "target_pos", "object_id": 0}, weight=10.0)
    object_goal_orient_tracking = RewTerm(func=handover.object_goal_orient_distance,
        params={"command_name": "target_pos", "object_id": 0, "axis": "z"}, weight=2.0)
    middle_success_bonus = RewTerm(func=handover.cmd_success_bonus,
        params={"command_names": "target_pos", "num_success": 1, "if_right": True}, weight=25.0)
    contact_bottle_punish = RewTerm(func=handover.contact_bottle_punish,
        params={"sensor_names": ["contact_sensors_0","contact_sensors_1","contact_sensors_2","contact_sensors_3",
                                 "contact_sensors_0_4","contact_sensors_1_4","contact_sensors_2_4","contact_sensors_3_4"]}, weight=4.0)
    reset_robot_joint_pos = RewTerm(func=handover.robot_goal_distance,
        params={"target_pos": [0.0462, -0.5045, 0.4468], "target_link": "palm_link",
                "sensor_names": ["contact_sensors_0","contact_sensors_1","contact_sensors_2","contact_sensors_3",
                                 "contact_sensors_0_4","contact_sensors_1_4","contact_sensors_2_4","contact_sensors_3_4"]}, weight=200.0)
    left_align_hand_pose = RewTerm(func=handover.align_hand_pose,
        params={"link_name": "palm_link", "command_name": "left_hand_target_pos", "asset_cfg": SceneEntityCfg("robot_left")}, weight=0.2)
    left_align_finger_joint = RewTerm(func=handover.align_finger_joint,
        params={"link_name": ["jif1","jif2","jif3","jif4","jmf1","jmf2","jmf3","jmf4","jpf1","jpf2","jpf3","jpf4","jth1","jth2","jth3","jth4"],
                "asset_cfg": SceneEntityCfg("robot_left")}, weight=1.0)
    left_reaching_object = RewTerm(func=handover.frame_marker_robot_distance,
        params={"weight": [1.5, 1.0, 1.0, 2.0], "link_name": ["if5","mf5","pf5","th5"], "frame_name": "bottle_top",
                "if_left": True, "asset_cfg": SceneEntityCfg("robot_left")}, weight=40.0)
    left_object_goal_tracking = RewTerm(func=handover.object_goal_distance,
        params={"command_name": "target_pos", "object_id": 0, "if_left": True,
                "sensor_names": ["contact_sensors_0_left","contact_sensors_1_left","contact_sensors_2_left","contact_sensors_3_left",
                                 "contact_sensors_0_4_left","contact_sensors_1_4_left","contact_sensors_2_4_left","contact_sensors_3_4_left"]}, weight=20.0)
    left_object_goal_orient_tracking = RewTerm(func=handover.object_goal_orient_distance,
        params={"command_name": "target_pos", "object_id": 0, "axis": "z", "if_left": True,
                "sensor_names": [... 8 left sensors ...]}, weight=5.0)
    middle_success_bonus_left = RewTerm(func=handover.cmd_success_bonus,
        params={"command_names": "target_pos", "num_success": 1, "if_left": True,
                "sensor_names": [... 8 left sensors ...]}, weight=20.0)
    success_bonus = RewTerm(func=handover.success_bonus,
        params={"command_names": "target_pos", "num_success": 10,
                "not_contact_sensor_names": [... 8 RIGHT sensors ...],
                "is_contact_sensor_names": [... 8 LEFT sensors ...]}, weight=3000.0)
    # ---- symmetry mirror (roles swapped: robot<->robot_left, right<->left sensors) ----
    reaching_object_symmetry = RewTerm(func=handover.frame_marker_robot_distance,
        params={"weight": [1.0,1.0,1.0,1.5,2.0], "link_name": ["if5","mf5","pf5","th5","palm_link"],
                "frame_name": "bottle_bottom", "asset_cfg": SceneEntityCfg("robot_left")}, weight=0.05)
    object_goal_tracking_symmetry = RewTerm(func=handover.object_goal_distance,
        params={"command_name": "target_pos", "object_id": 0, "asset_cfg": SceneEntityCfg("robot_left"),
                "sensor_names": ["contact_sensors_0_left","contact_sensors_1_left","contact_sensors_2_left","contact_sensors_3_left"]}, weight=10.0)
    object_goal_orient_tracking_symmetry = RewTerm(func=handover.object_goal_orient_distance,
        params={"command_name": "target_pos", "object_id": 0, "axis": "z",
                "sensor_names": ["contact_sensors_0_left","contact_sensors_1_left","contact_sensors_2_left","contact_sensors_3_left"]}, weight=2.0)
    contact_bottle_punish_symmetry = RewTerm(func=handover.contact_bottle_punish,
        params={"sensor_names": [... 8 LEFT sensors ...]}, weight=4.0)
    reset_robot_joint_pos_symmetry = RewTerm(func=handover.robot_goal_distance,
        params={"target_pos": [0.0462, 0.5045, 0.4468], "target_link": "palm_link", "asset_cfg": SceneEntityCfg("robot_left"),
                "sensor_names": [... 8 LEFT sensors ...]}, weight=200.0)
    left_align_hand_pose_symmetry = RewTerm(func=handover.align_hand_pose,
        params={"link_name": "palm_link", "command_name": "left_hand_target_pos", "asset_cfg": SceneEntityCfg("robot")}, weight=0.2)
    left_align_finger_joint_symmetry = RewTerm(func=handover.align_finger_joint,
        params={"link_name": [... 16 joints ...], "asset_cfg": SceneEntityCfg("robot")}, weight=1.0)
    left_reaching_object_symmetry = RewTerm(func=handover.frame_marker_robot_distance,
        params={"weight": [1.5,1.0,1.0,2.0], "link_name": ["if5","mf5","pf5","th5"], "frame_name": "bottle_top",
                "if_left": True, "asset_cfg": SceneEntityCfg("robot")}, weight=40.0)
    left_object_goal_tracking_symmetry = RewTerm(func=handover.object_goal_distance,
        params={"command_name": "target_pos", "object_id": 0, "if_left": True,
                "sensor_names": [... 8 RIGHT sensors ...]}, weight=20.0)
    left_object_goal_orient_tracking_symmetry = RewTerm(func=handover.object_goal_orient_distance,
        params={"command_name": "target_pos", "object_id": 0, "axis": "z", "if_left": True,
                "sensor_names": [... 8 RIGHT sensors ...]}, weight=5.0)
    middle_success_bonus_left_symmetry = RewTerm(func=handover.cmd_success_bonus,
        params={"command_names": "target_pos", "num_success": 1, "if_left": True,
                "sensor_names": [... 8 RIGHT sensors ...]}, weight=20.0)
    success_bonus_symmetry = RewTerm(func=handover.success_bonus,
        params={"command_names": "target_pos", "num_success": 10, "symmetry": True,
                "not_contact_sensor_names": [... 8 LEFT sensors ...],
                "is_contact_sensor_names": [... 8 RIGHT sensors ...]}, weight=3000.0)
```

> NOTE for reproduction: the per-link `params["weight"]` vectors inside several terms are internal shaping weights, distinct from each term's scalar `RewTerm.weight`, and are load-bearing.

---

## §7 DR

**DR: yes (partial).**

Two mechanisms:
1. **Static event DR** wired in `HandoverEventCfg`: `object_mass` (`randomize_rigid_body_mass`, `mode="startup"`, scale × U(0.05, 0.9)) — the only always-on randomization declared in the task. Plus object-yaw randomization in `reset_object` (see §3), which is spatial DR of the initial bottle orientation.
2. **Adaptive DR** driven by `BaseEnv.update_randomization(success_rate)` through a `DomainRandomizer` built from `cfg.hydra_cfg.task.randomize`. It can randomize: object mass, physics material (static/dynamic friction, restitution), action_scale (arm), reward weights (energy/collision penalties), external force/torque, and reset-pose ranges.

**Code (verbatim).**

Static object-mass DR (env_cfg.py, in HandoverEventCfg):
```python
    object_mass = EventTerm(
        func=randomize_rigid_body_mass,
        mode="startup",
        params={"mass_distribution_params": (0.05, 0.9), "operation": "scale"},
    )
```

Adaptive DR dispatch (manager_based_env.py::BaseEnv.update_randomization):
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
