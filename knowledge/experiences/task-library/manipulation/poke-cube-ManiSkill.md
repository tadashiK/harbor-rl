# PokeCube-v1 — Implementation Spec

- robot: Franka Panda (default; Fetch also supported)
- simulator: ManiSkill (SAPIEN, `BaseEnv` + `@register_env`)
- objects: cube, peg, goal marker, table
- bimanual: false
- summary: Use a peg to poke a cube to a goal position.

> ManiSkill task. Unlike IsaacLab manager-based tasks, all design choices live as methods on a single
> `BaseEnv` subclass (`PokeCubeEnv`) decorated with `@register_env`. There is no `RewardsCfg` / `ObservationsCfg` /
> `EventCfg` — instead the §-mapping below is onto methods: §1 `@register_env`/`SUPPORTED_ROBOTS`/`_load_agent`/`_load_scene`,
> §2 controller/control_mode, §3 `_initialize_episode`, §4 `evaluate()`/`max_episode_steps`, §5 `_get_obs_extra`,
> §6 `compute_dense_reward`/`compute_normalized_dense_reward`, §7 DR (none — randomization lives in §3 reset, not as
> separate startup/interval events).

## §1 Registration + Scene

**Description.** A single `PokeCubeEnv(BaseEnv)` is registered as `PokeCube-v1` with a 50-step episode cap. The agent (Panda by default, Fetch supported) is loaded at a fixed base pose offset behind the table. The scene is the standard `TableSceneBuilder` plus three actors: a dynamic red cube (the object to push), a dynamic two-color peg (the held tool), and a kinematic red/white circular goal target (collision disabled). A constant `peg_head_offsets` Pose marks the tip of the peg (peg half-length along +x in peg frame).

**Decisions resolved.**
- `@register_env("PokeCube-v1", max_episode_steps=50)`
- `SUPPORTED_ROBOTS = ["panda", "fetch"]`; default `robot_uids="panda"`; `robot_init_qpos_noise=0.02`
- Agent base pose offset: `sapien.Pose(p=[-0.615, 0, 0])`
- Geometry constants: `cube_half_size = 0.02`, `peg_half_width = 0.025`, `peg_half_length = 0.12`, `goal_radius = 0.05`
- cube: `actors.build_cube(half_size=0.02, color=[1,0,0,1], body_type="dynamic")`, initial_pose `p=[1,0,0.02]`
- peg: `actors.build_twocolor_peg(length=0.12, width=0.025, color_1=color_2=[12,42,160,255]/255, body_type="dynamic")`, initial_pose `p=[0,0,0.025]`
- goal_region: `actors.build_red_white_target(radius=0.05, thickness=1e-5, add_collision=False, body_type="kinematic")`
- `peg_head_offsets = Pose.create_from_pq(p=[0.12,0,0])`
- Sensor camera `base_camera`: `look_at(eye=[0.3,0,0.6], target=[-0.1,0,0.1])`, 128x128, fov π/2
- Render camera `render_camera`: `look_at([0.6,0.7,0.6],[0.2,0.2,0.35])`, 512x512, fov 1
- Sim config: defaults inherited from `BaseEnv` (no `_default_sim_config` override in this env).

**Code.**
```python
from typing import Any, Union

import numpy as np
import sapien
import torch
from transforms3d.euler import euler2quat

from mani_skill.agents.robots import Fetch, Panda
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.envs.utils import randomization
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from mani_skill.utils.building import actors
from mani_skill.utils.geometry import rotation_conversions
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.structs.pose import Pose


@register_env("PokeCube-v1", max_episode_steps=50)
class PokeCubeEnv(BaseEnv):
    _sample_video_link = "https://github.com/haosulab/ManiSkill/raw/main/figures/environment_demos/PokeCube-v1_rt.mp4"
    SUPPORTED_ROBOTS = ["panda", "fetch"]
    agent: Union[Panda, Fetch]

    cube_half_size = 0.02
    peg_half_width = 0.025
    peg_half_length = 0.12
    goal_radius = 0.05

    def __init__(self, *args, robot_uids="panda", robot_init_qpos_noise=0.02, **kwargs):
        self.robot_init_qpos_noise = robot_init_qpos_noise
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    @property
    def _default_sensor_configs(self):
        pose = sapien_utils.look_at(eye=[0.3, 0, 0.6], target=[-0.1, 0, 0.1])
        return [CameraConfig("base_camera", pose, 128, 128, np.pi / 2, 0.01, 100)]

    @property
    def _default_human_render_camera_configs(self):
        pose = sapien_utils.look_at([0.6, 0.7, 0.6], [0.2, 0.2, 0.35])
        return CameraConfig("render_camera", pose, 512, 512, 1, 0.01, 100)

    def _load_agent(self, options: dict):
        super()._load_agent(options, sapien.Pose(p=[-0.615, 0, 0]))

    def _load_scene(self, options: dict):
        self.table_scene = TableSceneBuilder(
            self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.table_scene.build()

        self.cube = actors.build_cube(
            self.scene,
            half_size=self.cube_half_size,
            color=[1, 0, 0, 1],
            name="cube",
            body_type="dynamic",
            initial_pose=sapien.Pose(p=[1, 0, self.cube_half_size]),
        )

        self.peg = actors.build_twocolor_peg(
            self.scene,
            length=self.peg_half_length,
            width=self.peg_half_width,
            color_1=np.array([12, 42, 160, 255]) / 255,
            color_2=np.array([12, 42, 160, 255]) / 255,
            name="peg",
            body_type="dynamic",
            initial_pose=sapien.Pose(p=[0, 0, self.peg_half_width]),
        )

        self.goal_region = actors.build_red_white_target(
            self.scene,
            radius=self.goal_radius,
            thickness=1e-5,
            name="goal_region",
            add_collision=False,
            body_type="kinematic",
            initial_pose=sapien.Pose(),
        )

        self.peg_head_offsets = Pose.create_from_pq(
            p=[self.peg_half_length, 0, 0], device=self.device
        )

    @property
    def peg_head_pos(self):
        return self.peg.pose.p + self.peg_head_offsets.p

    @property
    def peg_head_pose(self):
        return self.peg.pose * self.peg_head_offsets
```

**Smoke.** §1 build smoke:
```bash
cd <repo>
.venv/bin/python -c "import gymnasium as gym, mani_skill; e=gym.make('PokeCube-v1'); print(e.observation_space, e.action_space); e.close()"
# -> Box(-inf, inf, (1, 54), float32) Box(-1.0, 1.0, (8,), float32)
```

## §2 Actions

**Description.** PokeCube does not override `_controller_configs` — it uses the Panda agent's default controller set. The env is created with the registry/global default `control_mode` (`pd_joint_delta_pos` as confirmed by the build smoke). For Panda this yields a 7-DOF arm joint delta + 1-DOF gripper = 8-dim action, normalized to `[-1, 1]`.

**Decisions resolved.**
- control_mode (default): `pd_joint_delta_pos`
- action space: `Box(-1.0, 1.0, (8,), float32)` — 7 arm joint deltas + 1 gripper command
- No env-level controller override; controller config comes from `mani_skill.agents.robots.Panda`.
- Other available modes (Panda default set, selectable via `gym.make(..., control_mode=...)`): `pd_joint_pos`, `pd_joint_delta_pos`, `pd_ee_delta_pos`, `pd_ee_delta_pose`, `pd_joint_target_delta_pos`, etc. PokeCube does not constrain this.

**Code.** No action code in the env (inherited from Panda). The relevant fact is the absence of a `_controller_configs` override in `poke_cube.py`.

**Smoke.**
```bash
.venv/bin/python -c "import gymnasium as gym, mani_skill; e=gym.make('PokeCube-v1'); print(e.unwrapped.control_mode, e.action_space)"
# -> pd_joint_delta_pos Box(-1.0, 1.0, (8,), float32)
```

## §3 Reset

**Description.** `_initialize_episode` resets the table scene, then places peg, cube, and goal. The peg's xy is sampled uniformly in [-0.1, 0.1]² and laid flat (z = peg_half_width, identity quaternion). The cube's x is locked to `peg_x + peg_half_length + 0.1` (i.e. just beyond the peg tip), its y sampled uniformly in [-0.1, 0.1], z = cube_half_size, with a random z-axis rotation in [-π/6, π/6]. The goal target is placed at `cube_xy + [0.05 + goal_radius, 0]` (just past the cube along +x), flattened to z = 1e-3 and rotated to lie flat (`euler2quat(0, π/2, 0)`).

**Decisions resolved.**
- `b = len(env_idx)`; `self.table_scene.initialize(env_idx)` first.
- peg xyz `= rand((b,3))*0.2 - 0.1`; then `peg_xyz[...,2] = peg_half_width (0.025)`; quaternion `[1,0,0,0]`.
- cube xyz `= rand((b,3))*0.2 - 0.1`; then `cube_xyz[...,0] = peg_xyz[...,0] + peg_half_length(0.12) + 0.1`; `cube_xyz[...,2] = cube_half_size (0.02)`; quaternion from `randomization.random_quaternions(b, lock_x=True, lock_y=True, lock_z=False, bounds=(-π/6, π/6))`.
- goal xyz `= cube_xyz + [0.05 + goal_radius, 0, 0]`; then `goal_xyz[...,2] = 1e-3`; quaternion `euler2quat(0, π/2, 0)`.

**Code.**
```python
    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)
            # initialize the peg
            peg_xyz = torch.zeros((b, 3))
            peg_xyz = torch.rand((b, 3)) * 0.2 - 0.1
            peg_xyz[..., 2] = self.peg_half_width
            peg_q = [1, 0, 0, 0]
            peg_pose = Pose.create_from_pq(p=peg_xyz, q=peg_q)
            self.peg.set_pose(peg_pose)
            # initialize the cube
            cube_xyz = torch.zeros((b, 3))
            cube_xyz = torch.rand((b, 3)) * 0.2 - 0.1
            cube_xyz[..., 0] = peg_xyz[..., 0] + self.peg_half_length + 0.1
            cube_xyz[..., 2] = self.cube_half_size
            cube_q = randomization.random_quaternions(
                b,
                lock_x=True,
                lock_y=True,
                lock_z=False,
                bounds=(-np.pi / 6, np.pi / 6),
            )
            cube_pose = Pose.create_from_pq(p=cube_xyz, q=cube_q)
            self.cube.set_pose(cube_pose)
            # initialize the goal region
            goal_region_xyz = cube_xyz + torch.tensor([0.05 + self.goal_radius, 0, 0])
            goal_region_xyz[..., 2] = 1e-3
            goal_region_q = euler2quat(0, np.pi / 2, 0)
            goal_region_pose = Pose.create_from_pq(p=goal_region_xyz, q=goal_region_q)
            self.goal_region.set_pose(goal_region_pose)
```

**Smoke.**
```bash
.venv/bin/python -c "import gymnasium as gym, mani_skill; e=gym.make('PokeCube-v1'); o,_=e.reset(seed=0); o2,_=e.reset(seed=1); print('reset ok'); e.close()"
```

## §4 Goal + Termination

**Description.** `evaluate()` computes the success flag and intermediate booleans each step. Success = cube xy within `goal_radius` of the goal xy AND the robot is static (joint-velocity norm threshold 0.2). The episode also time-outs at `max_episode_steps=50`. Intermediate signals (peg–cube alignment, peg–cube head distance, peg grasped) feed §6's reward staging. There is no early termination on failure — only success + 50-step timeout.

**Decisions resolved.**
- `max_episode_steps = 50` (from `@register_env`).
- `is_cube_placed = ||cube.xy - goal.xy|| < goal_radius (0.05)`
- alignment: compare z-Euler of `peg_head_pose` vs `cube.pose`; `is_peg_cube_aligned = |angle_diff| < 0.05`
- closeness: `head_to_cube_dist = ||peg_head_pos.xy - cube.xy||`; `is_peg_cube_close = head_to_cube_dist <= cube_half_size + 0.005 (=0.025)`
- `is_peg_cube_fit = aligned AND close`
- `is_peg_grasped = agent.is_grasping(self.peg)`
- `is_robot_static = agent.is_static(0.2)`
- `success = is_cube_placed & is_robot_static`

**Code.**
```python
    def evaluate(self):
        is_cube_placed = (
            torch.linalg.norm(
                self.cube.pose.p[..., :2] - self.goal_region.pose.p[..., :2], axis=1
            )
            < self.goal_radius
        )
        peg_q = self.peg_head_pose.q
        peg_qmat = rotation_conversions.quaternion_to_matrix(peg_q)
        peg_euler = rotation_conversions.matrix_to_euler_angles(peg_qmat, "XYZ")
        cube_q = self.cube.pose.q
        cube_qmat = rotation_conversions.quaternion_to_matrix(cube_q)
        cube_euler = rotation_conversions.matrix_to_euler_angles(cube_qmat, "XYZ")
        angle_diff = torch.abs(peg_euler[:, 2] - cube_euler[:, 2])
        is_peg_cube_aligned = angle_diff < 0.05

        head_to_cube_dist = torch.linalg.norm(
            self.peg_head_pos[..., :2] - self.cube.pose.p[..., :2], axis=1
        )
        is_peg_cube_close = head_to_cube_dist <= self.cube_half_size + 0.005

        is_peg_cube_fit = torch.logical_and(is_peg_cube_aligned, is_peg_cube_close)
        is_peg_grasped = self.agent.is_grasping(self.peg)
        is_robot_static = self.agent.is_static(0.2)
        return {
            "success": is_cube_placed & is_robot_static,
            "is_cube_placed": is_cube_placed,
            "is_peg_cube_fit": is_peg_cube_fit,
            "is_peg_grasped": is_peg_grasped,
            "angle_diff": angle_diff,
            "head_to_cube_dist": head_to_cube_dist,
        }
```

**Smoke.**
```bash
.venv/bin/python -c "import gymnasium as gym, mani_skill; e=gym.make('PokeCube-v1'); e.reset(seed=0); import numpy as np; o,r,te,tr,info=e.step(e.action_space.sample()); print('success' in info, 'is_cube_placed' in info); e.close()"
```

## §5 Observation

**Description.** Beyond the standard agent proprioception (qpos+qvel, added by `BaseEnv`), `_get_obs_extra` always exposes `tcp_pose` (7-dim raw pose). When the obs mode includes privileged state (`obs_mode_struct.use_state`, true for the default `state` mode), it additionally exposes full cube/peg poses, goal position, and four relative-position vectors. The default `state` obs mode concatenates everything into a flat 54-dim vector.

**Decisions resolved.** Resolved total obs dim = **54** (default `state` mode), shape `(1, 54)` for 1 parallel env.
- Breakdown (state mode):
  - agent proprio: qpos(9) + qvel(9) = 18 (Panda: 7 arm + 2 finger joints)
  - `tcp_pose` (7)
  - `cube_pose` (7), `peg_pose` (7), `goal_pos` (3)
  - `tcp_to_peg_pos` (3), `peg_to_cube_pos` (3), `cube_to_goal_pos` (3), `peghead_to_cube_pos` (3)
  - = 18 + 7 + 7 + 7 + 3 + 3 + 3 + 3 + 3 = 54
- Obs modes: `state` (default, all of the above), `state_dict`, `sensor_data`/`rgbd`/`pointcloud` (drop privileged state vectors; only `tcp_pose` extra is kept).

**Code.**
```python
    def _get_obs_extra(self, info: dict):
        obs = dict(
            tcp_pose=self.agent.tcp.pose.raw_pose,
        )

        if self.obs_mode_struct.use_state:
            obs.update(
                cube_pose=self.cube.pose.raw_pose,
                peg_pose=self.peg.pose.raw_pose,
                goal_pos=self.peg.pose.p,
                tcp_to_peg_pos=self.peg.pose.p - self.agent.tcp.pose.p,
                peg_to_cube_pos=self.cube.pose.p - self.peg.pose.p,
                cube_to_goal_pos=self.goal_region.pose.p - self.cube.pose.p,
                peghead_to_cube_pos=self.peg_head_pos - self.cube.pose.p,
            )
        return obs
```
> Note: `goal_pos` is set to `self.peg.pose.p` here in source (uses peg position, not goal_region position) — reproduced verbatim.

**Smoke.**
```bash
.venv/bin/python -c "import gymnasium as gym, mani_skill; e=gym.make('PokeCube-v1'); print(e.observation_space.shape); e.close()"
# -> (1, 54)
```

## §6 Reward

**Description.** A staged dense reward with hard-overwrite stage gates (not additive across stages). Composer = **staged max/overwrite** (later stages replace, not sum). Stage 1 (always): reaching reward toward the peg, `2*(1 - tanh(5*tcp_to_peg_dist))`, range [0, 2). Stage 2 (peg grasped AND tcp reached peg): overwrite reward to `4 + close_reward + align_reward`, each in [0, 1), so [4, 6). Stage 3 (peg–cube fit AND grasped): overwrite to `7 + place_reward`, [7, 8). A static bonus `1 - tanh(5*||qvel[:-2]||)` is **added** when the cube is placed. On full success the reward is hard-set to 10. Normalized variant divides by `max_reward = 10.0`.

**Decisions resolved.**
- Composer: **staged overwrite** (boolean-mask assignment `reward[mask] = (...)[mask]`), with one additive static bonus and a final success override. Not a plain sum.
- `max_reward = 10.0` (normalization divisor).
- Per-stage saturated per-step magnitudes (retro-computed from the code):
  - Stage 1 reach: up to ~2.0
  - Stage 2 grasp+approach (`reached & is_peg_grasped`): 4.0 base + up to 1.0 (close) + up to 1.0 (align) → up to ~6.0
  - Stage 3 fit (`is_peg_cube_fit & grasped`): 7.0 base + up to 1.0 (place) → up to ~8.0
  - placed static bonus: +up to 1.0 (added, gives ~9.0 ceiling pre-success)
  - success override: exactly 10.0
- Note: `is_peg_grasped` in the reward is `info["is_peg_grasped"] * reached` (re-gated by reaching); `is_peg_cube_fit` is `info["is_peg_cube_fit"] * is_peg_grasped` (re-gated by the gated grasp).

**Code (verbatim, both functions).**
```python
    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: dict):
        # reach peg
        tcp_pos = self.agent.tcp.pose.p
        tgt_tcp_pose = self.peg.pose
        tcp_to_peg_dist = torch.linalg.norm(tcp_pos - tgt_tcp_pose.p, axis=1)
        reached = tcp_to_peg_dist < 0.01
        reaching_reward = 2 * (1 - torch.tanh(5.0 * tcp_to_peg_dist))
        reward = reaching_reward

        # peg to cube
        angle_diff = info["angle_diff"]
        align_reward = 1 - torch.tanh(5.0 * angle_diff)
        head_to_cube_dist = info["head_to_cube_dist"]
        close_reward = 1 - torch.tanh(5.0 * head_to_cube_dist)
        is_peg_grasped = info["is_peg_grasped"] * reached
        reward[is_peg_grasped] = (4 + close_reward + align_reward)[is_peg_grasped]

        # cube to goal
        cube_to_goal_dist = torch.linalg.norm(
            self.goal_region.pose.p - self.cube.pose.p, axis=1
        )
        place_reward = 1 - torch.tanh(5 * cube_to_goal_dist)
        is_peg_cube_fit = info["is_peg_cube_fit"] * is_peg_grasped
        reward[is_peg_cube_fit] = (7 + place_reward)[is_peg_cube_fit]

        static_reward = 1 - torch.tanh(
            5 * torch.linalg.norm(self.agent.robot.get_qvel()[..., :-2], axis=1)
        )
        reward[info["is_cube_placed"]] += static_reward[info["is_cube_placed"]]

        reward[info["success"]] = 10
        return reward

    def compute_normalized_dense_reward(
        self, obs: Any, action: torch.Tensor, info: dict
    ):
        max_reward = 10.0
        return self.compute_dense_reward(obs=obs, action=action, info=info) / max_reward
```

**Smoke.**
```bash
.venv/bin/python -c "import gymnasium as gym, mani_skill; e=gym.make('PokeCube-v1', reward_mode='dense'); e.reset(seed=0); o,r,te,tr,info=e.step(e.action_space.sample()); print('reward finite:', bool((r==r).all())); e.close()"
```

## §7 DR

`<no DR>` — `PokeCubeEnv` has no startup/interval domain-randomization events. All per-episode variation (peg xy, cube y + z-rotation, robot init qpos noise via `robot_init_qpos_noise=0.02`) is performed inside §3 `_initialize_episode` / `TableSceneBuilder`, not as separate DR events. There is no physical-parameter / material / mass randomization.

## Reproduce
```
/harbor:task-create name=<NewTaskID> from=<ManiSkill-repo>/harbor/create-task/pokecube-v1-implementation.md
```
