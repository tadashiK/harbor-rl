# UnitreeH1Stand-v1 — Implementation Spec

- robot: Unitree H1 humanoid, simplified (19 DoF, floating base)
- simulator: ManiSkill (SAPIEN, `BaseEnv` + `@register_env`)
- objects: none (flat ground plane)
- bimanual: false
- summary: Stand upright and stay balanced on a flat plane.

> NOTE: This is a SAPIEN ManiSkill task, NOT an IsaacLab manager-based task. There is no `mdp/` tree, no `RewardsCfg`/`ObservationsCfg`/`EventCfg` managers. All §1..§7 wiring lives directly on the `BaseEnv` subclass (`HumanoidStandEnv` / `UnitreeH1StandEnv`) and its agent (`UnitreeH1Simplified`). Sections below map the IsaacLab §1..§7 contract onto ManiSkill's hook methods.

---

## §1 Registration + Scene

### Description
`UnitreeH1Stand-v1` is registered with `@register_env(..., max_episode_steps=1000)` on the `UnitreeH1StandEnv` class, a thin per-robot subclass of the abstract `HumanoidStandEnv`. The scene is a bare infinite ground plane (`build_ground`) with no objects, no table, and no extra sensors. The only articulation is the Unitree H1 simplified humanoid (`unitree_h1_simplified`, 19 DoF), loaded by `BaseEnv._load_agent` from `SUPPORTED_ROBOTS` / `robot_uids`. No `_load_scene` object spawning beyond the ground.

### Decisions resolved
- env id: `UnitreeH1Stand-v1`; `max_episode_steps = 1000`
- `SUPPORTED_ROBOTS = ["unitree_h1_simplified"]`; `robot_uids = "unitree_h1_simplified"`
- robot total DoF: **19** (all body joints; `fix_root_link = False` → floating-base humanoid, gravity active)
- agent class: `UnitreeH1Simplified` (subclass of `UnitreeH1`); URDF: `{ASSET_DIR}/robots/unitree_h1/urdf/h1_simplified.urdf`
- `SUPPORTED_REWARD_MODES = ["sparse", "none"]` (NO dense reward — see §6)
- sim cfg: `GPUMemoryConfig(max_rigid_contact_count=2**22, max_rigid_patch_count=2**21)`, otherwise defaults
- scene: `build_ground(self.scene)` only — no objects, no table
- sensors: `_default_sensor_configs = []` (no per-obs cameras)
- render camera: `look_at([1.0, 1.0, 2.5], [0.0, 0.0, 0.75])`, 512×512
- actuators (PD joint pos): `body_stiffness = 1e3`, `body_damping = 1e2`, `body_force_limit = 100`
- `balance_passive_force=False` (gravity NOT compensated — required for a free-standing humanoid)
- standing keyframe pose: `sapien.Pose(p=[0, 0, 0.975])`

### Code (env class — `mani_skill/envs/tasks/humanoid/humanoid_stand.py`)
```python
from typing import Any, Union

import numpy as np
import sapien
import torch

from mani_skill.agents.robots import UnitreeG1Simplified, UnitreeH1Simplified
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import common, sapien_utils
from mani_skill.utils.building.ground import build_ground
from mani_skill.utils.registration import register_env
from mani_skill.utils.structs.types import GPUMemoryConfig, SimConfig


class HumanoidStandEnv(BaseEnv):
    SUPPORTED_REWARD_MODES = ["sparse", "none"]

    def __init__(
        self,
        *args,
        robot_uids="unitree_h1_simplified",
        robot_init_qpos_noise=0.02,
        **kwargs
    ):
        self.robot_init_qpos_noise = robot_init_qpos_noise
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    @property
    def _default_sensor_configs(self):
        return []

    @property
    def _default_human_render_camera_configs(self):
        pose = sapien_utils.look_at([1.0, 1.0, 2.5], [0.0, 0.0, 0.75])
        return CameraConfig("render_camera", pose, 512, 512, 1, 0.01, 100)

    def _load_scene(self, options: dict):
        build_ground(self.scene)

    # ... (_initialize_episode / evaluate / _get_obs_extra / reward — see §3..§6)


@register_env("UnitreeH1Stand-v1", max_episode_steps=1000)
class UnitreeH1StandEnv(HumanoidStandEnv):
    SUPPORTED_ROBOTS = ["unitree_h1_simplified"]
    agent: Union[UnitreeH1Simplified]

    def __init__(self, *args, robot_uids="unitree_h1_simplified", **kwargs):
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    @property
    def _default_sim_config(self):
        return SimConfig(
            gpu_memory_config=GPUMemoryConfig(
                max_rigid_contact_count=2**22, max_rigid_patch_count=2**21
            )
        )

    @property
    def _default_human_render_camera_configs(self):
        pose = sapien_utils.look_at([1.0, 1.0, 2.5], [0.0, 0.0, 0.75])
        return CameraConfig("render_camera", pose, 512, 512, 1, 0.01, 100)
```

### Code (agent — `mani_skill/agents/robots/unitree_h1/h1.py`)
```python
@register_agent(asset_download_ids=["unitree_h1"])
class UnitreeH1(BaseAgent):
    uid = "unitree_h1"
    urdf_path = f"{ASSET_DIR}/robots/unitree_h1/urdf/h1.urdf"
    urdf_config = dict()
    fix_root_link = False
    load_multiple_collisions = True

    keyframes = dict(
        standing=Keyframe(
            pose=sapien.Pose(p=[0, 0, 0.975]),
            qpos=np.array(
                [0, 0, 0, 0, 0, 0, 0, -0.4, -0.4, 0.0, 0.0, 0.8, 0.8,
                 0.0, 0.0, -0.4, -0.4, 0.0, 0.0]
            ) * 1,
        )
    )

    body_joints = [
        "left_hip_yaw_joint", "right_hip_yaw_joint", "torso_joint",
        "left_hip_roll_joint", "right_hip_roll_joint",
        "left_shoulder_pitch_joint", "right_shoulder_pitch_joint",
        "left_hip_pitch_joint", "right_hip_pitch_joint",
        "left_shoulder_roll_joint", "right_shoulder_roll_joint",
        "left_knee_joint", "right_knee_joint",
        "left_shoulder_yaw_joint", "right_shoulder_yaw_joint",
        "left_ankle_joint", "right_ankle_joint",
        "left_elbow_joint", "right_elbow_joint",
    ]
    body_stiffness = 1e3
    body_damping = 1e2
    body_force_limit = 100

    @property
    def _controller_configs(self):
        body_pd_joint_pos = PDJointPosControllerConfig(
            self.body_joints, lower=None, upper=None,
            stiffness=self.body_stiffness, damping=self.body_damping,
            force_limit=self.body_force_limit, normalize_action=False,
        )
        body_pd_joint_delta_pos = PDJointPosControllerConfig(
            self.body_joints, lower=-0.2, upper=0.2,
            stiffness=self.body_stiffness, damping=self.body_damping,
            force_limit=self.body_force_limit, use_delta=True,
        )
        # balance_passive_force=False otherwise gravity is disabled for the robot itself
        return dict(
            pd_joint_pos=dict(body=body_pd_joint_pos, balance_passive_force=False),
            pd_joint_delta_pos=dict(
                body=body_pd_joint_delta_pos, balance_passive_force=False
            ),
        )

    @property
    def _sensor_configs(self):
        return []

    def is_standing(self):
        """Checks if H1 is standing — torso height in [0.8, 1.2]."""
        return (self.robot.pose.p[:, 2] > 0.8) & (self.robot.pose.p[:, 2] < 1.2)

    def is_fallen(self):
        """Checks if H1 has fallen — torso below 0.3."""
        return self.robot.pose.p[:, 2] < 0.3


@register_agent(asset_download_ids=["unitree_h1"])
class UnitreeH1Simplified(UnitreeH1):
    uid = "unitree_h1_simplified"
    urdf_path = f"{ASSET_DIR}/robots/unitree_h1/urdf/h1_simplified.urdf"
```

### Asset paths (resolved relative to `ASSET_DIR` = `~/.maniskill/data`)
- `~/.maniskill/data/robots/unitree_h1/urdf/h1_simplified.urdf` (downloaded via `asset_download_ids=["unitree_h1"]`, source `ManiSkill-UnitreeH1` v0.1.0)

### Smoke (§1 build)
```bash
cd <ManiSkill-repo>
.venv/bin/python -c "import gymnasium as gym, mani_skill; e=gym.make('UnitreeH1Stand-v1'); print(e.observation_space, e.action_space); e.close()"
```
Expected stdout:
```
Box(-inf, inf, (1, 38), float32) Box([-0.43 -0.43 -2.35 -0.43 -0.43 -2.87 -2.87 -3.14 -3.14 -0.34 -3.11 -0.26 -0.26 -1.3 -4.45 -0.87 -0.87 -1.25 -1.25], [0.43 0.43 2.35 0.43 0.43 2.87 2.87 2.53 2.53 3.11 0.34 2.05 2.05 4.45 1.3 0.52 0.52 2.61 2.61], (19,), float32)
```

---

## §2 Actions

### Description
Action is a per-joint position command for the 19 body joints. The default `control_mode` resolved by `gym.make` is **`pd_joint_pos`** (absolute joint targets, `normalize_action=False` → action space equals the URDF joint limits, NOT [-1, 1]). An alternate `pd_joint_delta_pos` mode (clamped delta in [-0.2, 0.2], absolute joint limits irrelevant) is also registered. Both share `stiffness=1e3, damping=1e2, force_limit=100` and `balance_passive_force=False`.

### Decisions resolved
- control_mode (default): `pd_joint_pos` — absolute position targets, un-normalized
- alt control_mode: `pd_joint_delta_pos` — delta targets clamped to ±0.2
- action dim: **19** (one per body joint, ordered as `body_joints`)
- action space (pd_joint_pos): `Box(low=URDF joint lower, high=URDF joint upper, (19,))`; resolved low/high in §1 smoke stdout
- PD gains: stiffness 1e3, damping 1e2, force_limit 100; `balance_passive_force=False`

### Code
See `_controller_configs` in §1. Action space comes directly from `PDJointPosControllerConfig(... normalize_action=False)`, so bounds = `h1_simplified.urdf` joint limits.

### Smoke (§2 — action shape / step)
```bash
.venv/bin/python -c "import gymnasium as gym, mani_skill, torch; e=gym.make('UnitreeH1Stand-v1'); e.reset(seed=0); a=torch.from_numpy(e.action_space.sample()); o,r,te,tr,i=e.step(a); print(e.action_space.shape, o.shape); e.close()"
```
Expected: action shape `(19,)`, obs shape `(1, 38)`.

---

## §3 Reset (`_initialize_episode`)

### Description
On each episode reset, the H1's joint positions are set to the `standing` keyframe qpos plus Gaussian noise (std 0.05 per joint), and the base pose is teleported to `p=[0, 0, 0.975]` (upright at standing torso height). This is the per-robot override in `UnitreeH1StandEnv` (the abstract `HumanoidStandEnv._initialize_episode` is a no-op `pass`).

### Decisions resolved
- reset qpos = `standing` keyframe qpos + `randn(b, 19) * 0.05`
- reset base pose = `sapien.Pose(p=[0, 0, 0.975])`
- note: `robot_init_qpos_noise=0.02` is stored on the env but the per-robot `_initialize_episode` uses a hardcoded `0.05` Gaussian, not that field
- no object/goal randomization (no objects in scene)

### Code (`UnitreeH1StandEnv._initialize_episode`)
```python
def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
    with torch.device(self.device):
        b = len(env_idx)
        standing_keyframe = self.agent.keyframes["standing"]
        random_qpos = (
            torch.randn(size=(b, self.agent.robot.dof[0]), dtype=torch.float) * 0.05
        )
        random_qpos += common.to_tensor(standing_keyframe.qpos, device=self.device)
        self.agent.robot.set_qpos(random_qpos)
        self.agent.robot.set_pose(sapien.Pose(p=[0, 0, 0.975]))
```

### Smoke (§3 — reset variance)
```bash
.venv/bin/python -c "import gymnasium as gym, mani_skill; e=gym.make('UnitreeH1Stand-v1'); o0,_=e.reset(seed=0); o1,_=e.reset(seed=1); print((o0-o1).abs().sum() > 0); e.close()"
```
Expected: `True` (seed-distinct resets diverge).

---

## §4 Goal + Termination (`evaluate` + `max_episode_steps`)

### Description
There is no explicit success/done term object; ManiSkill derives termination from the `evaluate()` dict. The task "succeeds" each step the agent `is_standing()` (torso height in [0.8, 1.2]); `fail = ~is_standing` flags an early termination as soon as the humanoid leaves the standing height band. `is_fallen()` is called but its result is discarded (not used in the returned dict). Episodes time out at `max_episode_steps = 1000`.

### Decisions resolved
- `max_episode_steps = 1000` (set in `@register_env`)
- success signal: `is_standing` = `(torso_z > 0.8) & (torso_z < 1.2)`
- fail signal (early termination): `fail = ~is_standing`
- `is_fallen()` (torso_z < 0.3) is invoked but its return is unused
- no `CommandsCfg` (not goal-conditioned; stand-in-place balance)

### Code (`HumanoidStandEnv.evaluate`)
```python
def evaluate(self):
    is_standing = self.agent.is_standing()
    self.agent.is_fallen()
    return {"is_standing": is_standing, "fail": ~is_standing}
```
(`is_standing` / `is_fallen` defined on `UnitreeH1` — see §1.)

### Smoke (§4 — eval keys)
```bash
.venv/bin/python -c "import gymnasium as gym, mani_skill, torch; e=gym.make('UnitreeH1Stand-v1'); e.reset(seed=0); _,_,_,_,i=e.step(torch.from_numpy(e.action_space.sample())); print('is_standing' in i, 'fail' in i); e.close()"
```
Expected: `True True`.

---

## §5 Observation (`_get_obs_extra` + obs modes)

### Description
`_get_obs_extra` returns an empty dict — there are NO task-specific extra observations. The entire observation in `state` / `state_dict` mode is the agent's proprioception: `qpos` (19) + `qvel` (19) = **38**. ManiSkill's `BaseEnv._get_obs_state_dict` assembles `{agent: get_proprioception(), extra: {}}`; `get_proprioception` returns `{qpos, qvel}` (no controller state for PD joint-pos controllers). Image/pointcloud obs modes (`rgb`, `rgbd`, `pointcloud`) are available generically but the default registered mode is `state`.

### Decisions resolved
- obs_mode (default): `state` → flat `Box((1, 38), float32)`
- obs breakdown (state_dict): `agent.qpos: (1, 19)`, `agent.qvel: (1, 19)`, `extra: {}` → total **38**
- `_get_obs_extra(info) -> dict()` (empty)
- no sensors (`_default_sensor_configs = []`), so no camera obs by default

### Code
```python
def _get_obs_extra(self, info: dict):
    return dict()
```
Proprioception (base, `mani_skill/agents/base_agent.py`):
```python
def get_proprioception(self):
    obs = dict(qpos=self.robot.get_qpos(), qvel=self.robot.get_qvel())
    controller_state = self.controller.get_state()
    if len(controller_state) > 0:
        obs.update(controller=controller_state)
    return obs
```

### Smoke (§5 — obs dim)
```bash
.venv/bin/python -c "import gymnasium as gym, mani_skill; e=gym.make('UnitreeH1Stand-v1'); o,_=e.reset(seed=0); print(o.shape); e.close()"
```
Expected: `torch.Size([1, 38])`.

---

## §6 Reward (`compute_*_reward` + composer)

### Description
This task ships a **sparse** reward only. `SUPPORTED_REWARD_MODES = ["sparse", "none"]`; the default `reward_mode` resolved by `gym.make` is `sparse`. `compute_sparse_reward` returns `info["is_standing"]` directly — i.e. reward = 1.0 each step the torso height is in [0.8, 1.2], else 0.0. The dense reward and the normalized dense reward are **commented out in source** (present only as commented stubs). There is therefore no dense shaping, no per-term decomposition, and no multi-term composer.

### Decisions resolved
- reward_mode (default): `sparse`; supported: `["sparse", "none"]`
- sparse reward = `info["is_standing"]` (bool→float; 1.0 if standing else 0.0)
- composer: **none** — single scalar term, no sum/product of sub-terms
- dense / normalized-dense reward: NOT implemented (commented out)

### Code (verbatim, including commented stubs)
```python
def compute_sparse_reward(self, obs: Any, action: torch.Tensor, info: dict):
    return info["is_standing"]

# def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: dict):
#     return torch.zeros(self.num_envs, device=self.device)

# def compute_normalized_dense_reward(
#     self, obs: Any, action: torch.Tensor, info: dict
# ):
#     max_reward = 1.0
#     return self.compute_dense_reward(obs=obs, action=action, info=info) / max_reward
```

### Planning budget (retro-computed)
Single sparse term, per-step magnitude ∈ {0.0, 1.0}; max episodic return = `max_episode_steps` = 1000 (if standing every step). No saturation analysis applies (binary indicator).

### Smoke (§6 — reward finite + sparse)
```bash
.venv/bin/python -c "import gymnasium as gym, mani_skill, torch; e=gym.make('UnitreeH1Stand-v1'); e.reset(seed=0); _,r,_,_,_=e.step(torch.from_numpy(e.action_space.sample())); print(torch.isfinite(r.float()).all(), r); e.close()"
```
Expected: `True` and reward value in {0., 1.} (per env).

---

## §7 DR (domain randomization)

`<no DR>` — there are no `startup` / `interval` randomization events. The only per-episode stochasticity is the reset-time Gaussian qpos noise (std 0.05) documented in §3, which is reset randomization, not domain randomization. No mass / friction / actuator-gain / visual randomization is applied.

---

