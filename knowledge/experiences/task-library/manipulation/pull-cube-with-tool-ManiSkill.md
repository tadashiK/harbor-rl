# PullCubeTool-v1 — Implementation Spec

- robot: Franka Panda (default; Fetch also supported)
- simulator: ManiSkill (SAPIEN, `BaseEnv` + `@register_env`)
- objects: cube, L-shaped tool, table
- bimanual: false
- summary: Use an L-shaped tool to pull an out-of-reach cube into reach.

---

## §1 Registration + Scene

### Description
`PullCubeTool-v1` is registered via the `@register_env` decorator with `max_episode_steps=100`. The class subclasses `mani_skill.envs.sapien_env.BaseEnv`. The scene is the standard `TableSceneBuilder` (table + ground + robot) plus two task actors: a dynamic blue **cube** (`actors.build_cube`, half_size 0.02) and a procedurally-built **L-shaped tool** (`l_shape_tool`) made of two box collisions/visuals (a long handle along +x and a short hook offset in +y). The tool is spawned within arm reach; the cube is spawned beyond direct arm reach but within reach of the tool's hook (see §3). Two robots are supported (`panda`, `fetch`); default is `panda`.

### Decisions resolved
- `@register_env("PullCubeTool-v1", max_episode_steps=100)`
- `SUPPORTED_ROBOTS = ["panda", "fetch"]`; `agent: Union[Panda, Fetch]`; default `robot_uids="panda"`.
- `SUPPORTED_REWARD_MODES = ("normalized_dense", "dense", "sparse", "none")`.
- `robot_init_qpos_noise = 0.02` (passed to `TableSceneBuilder`).
- Geometry constants (meters): `goal_radius=0.3`, `cube_half_size=0.02`, `handle_length=0.2`, `hook_length=0.05`, `width=0.05`, `height=0.05`, `cube_size=0.02`, `arm_reach=0.35`.
- Cube: `actors.build_cube(half_size=0.02, color=[12,42,160,255]/255, name="cube", body_type="dynamic")`.
- L-shaped tool (`name="l_shape_tool"`): handle box half-extents `[handle_length/2, width/2, height/2]` centered at `(handle_length/2, 0, 0)` density 500; hook box half-extents `[hook_length/2, width, height/2]` centered at `(handle_length - hook_length/2, width, 0)` (red metallic render material). No explicit density on hook box (SAPIEN default).
- Sim cfg: `GPUMemoryConfig(found_lost_pairs_capacity=2**25, max_rigid_patch_count=2**18)`.
- Sensor camera `base_camera`: `look_at(eye=[0.3,0,0.5], target=[-0.1,0,0.1])`, 128×128, fov π/2, near 0.01, far 100.
- Human render camera `render_camera`: `look_at([0.6,0.7,0.6],[0,0,0.35])`, 512×512, fov 1, near 0.01, far 100.

### Code
```python
@register_env("PullCubeTool-v1", max_episode_steps=100)
class PullCubeToolEnv(BaseEnv):
    _sample_video_link = "https://github.com/haosulab/ManiSkill/raw/main/figures/environment_demos/PullCubeTool-v1_rt.mp4"

    SUPPORTED_ROBOTS = ["panda", "fetch"]
    SUPPORTED_REWARD_MODES = ("normalized_dense", "dense", "sparse", "none")
    agent: Union[Panda, Fetch]

    goal_radius = 0.3
    cube_half_size = 0.02
    handle_length = 0.2
    hook_length = 0.05
    width = 0.05
    height = 0.05
    cube_size = 0.02
    arm_reach = 0.35

    def __init__(self, *args, robot_uids="panda", robot_init_qpos_noise=0.02, **kwargs):
        self.robot_init_qpos_noise = robot_init_qpos_noise
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    @property
    def _default_sim_config(self):
        return SimConfig(
            gpu_memory_config=GPUMemoryConfig(
                found_lost_pairs_capacity=2**25, max_rigid_patch_count=2**18
            )
        )

    @property
    def _default_sensor_configs(self):
        pose = sapien_utils.look_at(eye=[0.3, 0, 0.5], target=[-0.1, 0, 0.1])
        return [
            CameraConfig(
                "base_camera", pose=pose, width=128, height=128,
                fov=np.pi / 2, near=0.01, far=100,
            )
        ]

    @property
    def _default_human_render_camera_configs(self):
        pose = sapien_utils.look_at([0.6, 0.7, 0.6], [0.0, 0.0, 0.35])
        return [
            CameraConfig(
                "render_camera", pose=pose, width=512, height=512,
                fov=1, near=0.01, far=100,
            )
        ]

    def _build_l_shaped_tool(self, handle_length, hook_length, width, height):
        builder = self.scene.create_actor_builder()

        mat = sapien.render.RenderMaterial()
        mat.set_base_color([1, 0, 0, 1])
        mat.metallic = 1.0
        mat.roughness = 0.0
        mat.specular = 1.0

        builder.add_box_collision(
            sapien.Pose([handle_length / 2, 0, 0]),
            [handle_length / 2, width / 2, height / 2],
            density=500,
        )
        builder.add_box_visual(
            sapien.Pose([handle_length / 2, 0, 0]),
            [handle_length / 2, width / 2, height / 2],
            material=mat,
        )

        builder.add_box_collision(
            sapien.Pose([handle_length - hook_length / 2, width, 0]),
            [hook_length / 2, width, height / 2],
        )
        builder.add_box_visual(
            sapien.Pose([handle_length - hook_length / 2, width, 0]),
            [hook_length / 2, width, height / 2],
            material=mat,
        )

        return builder.build(name="l_shape_tool")

    def _load_scene(self, options: dict):
        self.scene_builder = TableSceneBuilder(
            self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.scene_builder.build()

        self.cube = actors.build_cube(
            self.scene,
            half_size=self.cube_half_size,
            color=np.array([12, 42, 160, 255]) / 255,
            name="cube",
            body_type="dynamic",
        )

        self.l_shape_tool = self._build_l_shaped_tool(
            handle_length=self.handle_length,
            hook_length=self.hook_length,
            width=self.width,
            height=self.height,
        )
```

Note: there is **no `_load_agent` override** — the robot is loaded by `BaseEnv` from `robot_uids` (default `panda`). Imports: `from mani_skill.agents.robots import Fetch, Panda`.

### Smoke
```bash
cd <repo>
.venv/bin/python -c "import gymnasium as gym, mani_skill; e=gym.make('PullCubeTool-v1'); print(e.observation_space, e.action_space); e.close()"
```
Expected stdout:
```
Box(-inf, inf, (1, 39), float32) Box(-1.0, 1.0, (8,), float32)
```

---

## §2 Actions

### Description
No custom controller is defined in the task. Action/control come entirely from the chosen robot agent (Panda). The default control mode is `pd_joint_delta_pos`: a delta over the 7 arm joints plus 1 gripper command = 8-D action, normalized to `[-1, 1]`.

### Decisions resolved
- `control_mode` (default): `pd_joint_delta_pos`.
- Action space: `Box(-1.0, 1.0, (8,), float32)` = 7 arm joint deltas + 1 gripper.
- Supported control modes (from Panda agent): `pd_joint_delta_pos`, `pd_joint_pos`, `pd_ee_delta_pos`, `pd_ee_delta_pose`, `pd_ee_pose`, `pd_joint_target_delta_pos`, `pd_ee_target_delta_pos`, `pd_ee_target_delta_pose`, `pd_joint_vel`, `pd_joint_pos_vel`, `pd_joint_delta_pos_vel`.
- The task does NOT override controller scales/limits; all controller config inherited from the Panda agent definition (`mani_skill/agents/robots/panda/panda.py`).

### Code
No task-level action code. To reproduce, set `robot_uids="panda"` and rely on agent defaults; pass `control_mode="pd_joint_delta_pos"` to `gym.make` (or omit — it is the agent default).

### Smoke
```bash
.venv/bin/python -c "import gymnasium as gym, mani_skill; e=gym.make('PullCubeTool-v1'); print(e.unwrapped.control_mode, e.action_space.shape); e.close()"
```
Expected: `pd_joint_delta_pos (8,)`

---

## §3 Reset

### Description
`_initialize_episode` runs vectorized over `env_idx`. It (1) re-initializes the table scene (sets robot home qpos with noise), (2) places the L-shaped tool within arm reach at a randomized base-side (x,y) location flat on the table, and (3) places the cube beyond direct arm reach but within hook reach, with a small in-plane yaw randomization.

### Decisions resolved
- `self.scene_builder.initialize(env_idx)` — robot/table reset (qpos noise 0.02).
- **Tool pose**: `tool_xyz[:, :2] = -rand(b,2)*0.2 - 0.1` → each of x,y ∈ `[-0.3, -0.1]` (toward the robot base/negative quadrant); `tool_xyz[:, 2] = height/2 = 0.025`; quaternion fixed identity `[1,0,0,0]`.
- **Cube pose**: `cube_xyz[:,0] = arm_reach + rand(b)*handle_length - 0.3` → x ∈ `[0.05, 0.25]` (i.e. `0.35 + [0,0.2] - 0.3`); `cube_xyz[:,1] = rand(b)*0.3 - 0.25` → y ∈ `[-0.25, 0.05]`; `cube_xyz[:,2] = cube_size/2 + 0.015 = 0.025`; quaternion `random_quaternions(lock_x=True, lock_y=True, lock_z=False, bounds=(-π/6, π/6))` — yaw-only jitter ±30°.

### Code
```python
def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
    with torch.device(self.device):
        b = len(env_idx)
        self.scene_builder.initialize(env_idx)

        tool_xyz = torch.zeros((b, 3), device=self.device)
        tool_xyz[..., :2] = -torch.rand((b, 2), device=self.device) * 0.2 - 0.1
        tool_xyz[..., 2] = self.height / 2
        tool_q = torch.tensor([1, 0, 0, 0], device=self.device).expand(b, 4)

        tool_pose = Pose.create_from_pq(p=tool_xyz, q=tool_q)
        self.l_shape_tool.set_pose(tool_pose)

        cube_xyz = torch.zeros((b, 3), device=self.device)
        cube_xyz[..., 0] = (
            self.arm_reach
            + torch.rand(b, device=self.device) * (self.handle_length)
            - 0.3
        )
        cube_xyz[..., 1] = torch.rand(b, device=self.device) * 0.3 - 0.25
        cube_xyz[..., 2] = self.cube_size / 2 + 0.015

        cube_q = randomization.random_quaternions(
            b,
            lock_x=True,
            lock_y=True,
            lock_z=False,
            bounds=(-np.pi / 6, np.pi / 6),
            device=self.device,
        )

        cube_pose = Pose.create_from_pq(p=cube_xyz, q=cube_q)
        self.cube.set_pose(cube_pose)
```

### Smoke
```bash
.venv/bin/python -c "import gymnasium as gym, mani_skill; e=gym.make('PullCubeTool-v1'); o,_=e.reset(seed=0); print('reset ok', tuple(o.shape)); e.close()"
```
Expected: `reset ok (1, 39)`

---

## §4 Goal + Termination

### Description
Success is geometric: the cube has been pulled close enough to the robot base (planar xy distance < 0.6 m). `evaluate()` returns the success flags plus auxiliary progress metrics; `max_episode_steps=100` (set in `@register_env`) governs the time-out / truncation. There is no separate `CommandsCfg`/goal site — the "goal region" is implicit in the base-distance threshold.

### Decisions resolved
- Success: `cube_to_base_dist = ||cube_pos[:, :2] - robot_base_pos[:, :2]|| < 0.6`.
- `evaluate()` returns `success`, `success_once`, `success_at_end` (all = `cube_pulled_close`), plus `cube_progress` (`mean(1 - tanh(3·dist_to_workspace_center))`), `cube_distance` (`mean(dist_to_workspace_center)`), and `reward` (the normalized dense reward).
- `workspace_center = robot_base_pos` with `x += arm_reach*0.1 = 0.035`.
- `max_episode_steps = 100` (truncation via gym TimeLimit; no early non-success termination term).

### Code
```python
def evaluate(self):
    cube_pos = self.cube.pose.p
    robot_base_pos = self.agent.robot.get_links()[0].pose.p
    cube_to_base_dist = torch.linalg.norm(
        cube_pos[:, :2] - robot_base_pos[:, :2], dim=1
    )
    # Success condition - cube is pulled close enough
    cube_pulled_close = cube_to_base_dist < 0.6

    workspace_center = robot_base_pos.clone()
    workspace_center[:, 0] += self.arm_reach * 0.1
    cube_to_workspace_dist = torch.linalg.norm(cube_pos - workspace_center, dim=1)
    progress = 1 - torch.tanh(3.0 * cube_to_workspace_dist)

    return {
        "success": cube_pulled_close,
        "success_once": cube_pulled_close,
        "success_at_end": cube_pulled_close,
        "cube_progress": progress.mean(),
        "cube_distance": cube_to_workspace_dist.mean(),
        "reward": self.compute_normalized_dense_reward(
            None, None, {"success": cube_pulled_close}
        ),
    }
```

### Smoke
```bash
.venv/bin/python -c "import gymnasium as gym, mani_skill; e=gym.make('PullCubeTool-v1'); e.reset(seed=0); import numpy as np; o,r,te,tr,info=e.step(e.action_space.sample()); print('success' in info, e.spec.max_episode_steps); e.close()"
```
Note: success key is exposed in `info`; `max_episode_steps=100` is set on the env spec by `@register_env`.

---

## §5 Observation

### Description
Default `obs_mode="state"` yields a flat 39-D vector. It concatenates the agent proprioception (qpos 9 + qvel 9) with the task extras: TCP pose (always) and, because state mode sets `use_state=True`, the cube pose and tool pose (each a 7-D `[xyz + quat]` raw pose).

### Decisions resolved
- `obs_mode` default: `state`. Total dim = **39**.
- Breakdown (state_dict):
  - `agent.qpos`: 9
  - `agent.qvel`: 9
  - `extra.tcp_pose`: 7 (always)
  - `extra.cube_pose`: 7 (only when `use_state`)
  - `extra.tool_pose`: 7 (only when `use_state`)
- Under non-state obs modes (e.g. `rgbd`, `pointcloud`, `sparse`) the `if self.obs_mode_struct.use_state:` branch is skipped, so `cube_pose`/`tool_pose` are omitted from `extra` (the agent provides visual obs instead). `tcp_pose` is always present.
- No `ObsTerm` noise / corruption applied at task level.

### Code
```python
def _get_obs_extra(self, info: dict):
    obs = dict(
        tcp_pose=self.agent.tcp.pose.raw_pose,
    )
    if self.obs_mode_struct.use_state:
        obs.update(
            cube_pose=self.cube.pose.raw_pose,
            tool_pose=self.l_shape_tool.pose.raw_pose,
        )
    return obs
```

### Smoke
```bash
.venv/bin/python -c "import gymnasium as gym, mani_skill; e=gym.make('PullCubeTool-v1'); print(e.observation_space.shape); e.close()"
```
Expected: `(1, 39)`

---

## §6 Reward

### Description
Three-stage hand-crafted dense reward, composed by **summation** with multiplicative grasp-gating. Stage 1 rewards reaching the tool grasp point and grasping it. Stage 2 (gated on grasping) rewards positioning the tool's hook behind the cube. Stage 3 (gated on grasping) rewards pulling the cube toward the robot workspace, but only once the tool is correctly positioned (`tool_positioned` boolean gate). A penalty subtracts 2.0 if the cube is pushed further away, and a +5.0 success bonus is added on success. The normalized variant divides the dense reward by `max_reward = 5.0`.

### Composer
**sum** (additive terms), with two multiplicative gates inside the sum:
- `positioning_reward * is_grasping`
- `pulling_reward * is_grasping`, where `pulling_reward = 3.0 * pulling_progress * tool_positioned` (so a second gate on `tool_positioned`).

### Decisions resolved (per-stage saturated per-step magnitudes — retro-computed from weights)
- Stage 1 reaching: `2.0 * (1 - tanh(5·tcp_to_tool_dist))` → saturates at **+2.0** when TCP at tool grasp point.
- Stage 1 grasping: `2.0 * is_grasping` → **+2.0** when grasping (`max_angle=20`).
- Stage 2 positioning: `1.5 * (1 - tanh(3·tool_positioning_dist))` → up to **+1.5**, gated ×`is_grasping`.
- Stage 3 pulling: `3.0 * pulling_progress * tool_positioned`, gated ×`is_grasping`; `pulling_progress = (initial_dist - cube_to_workspace_dist)/initial_dist` → up to **+3.0** at full pull (gated).
- Push-away penalty: **−2.0** when `cube_pos[:,0] > arm_reach + 0.15 = 0.5`.
- Success bonus: **+5.0** when `info["success"]`.
- `max_reward = 5.0` for normalization. (Note: nominal max summed dense reward exceeds 5.0; the normalizer divides by the success-bonus magnitude only.)
- Key offsets: `tool_grasp_pos = tool_pos + [0.02,0,0]`; `ideal_hook_pos = cube_pos + [-(hook_length+cube_half_size), -0.067, 0] = cube_pos + [-0.07, -0.067, 0]`; `tool_positioned = tool_positioning_dist < 0.05`; `workspace_target = robot_base_pos + [0.05,0,0]`; `initial_dist = ||[arm_reach+0.1, 0, cube_size/2] - workspace_target||`.

### Code (verbatim, both functions)
```python
def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: dict):

    tcp_pos = self.agent.tcp.pose.p
    cube_pos = self.cube.pose.p
    tool_pos = self.l_shape_tool.pose.p
    robot_base_pos = self.agent.robot.get_links()[0].pose.p

    # Stage 1: Reach and grasp tool
    tool_grasp_pos = tool_pos + torch.tensor([0.02, 0, 0], device=self.device)
    tcp_to_tool_dist = torch.linalg.norm(tcp_pos - tool_grasp_pos, dim=1)
    reaching_reward = 2.0 * (1 - torch.tanh(5.0 * tcp_to_tool_dist))

    # Add specific grasping reward
    is_grasping = self.agent.is_grasping(self.l_shape_tool, max_angle=20)
    grasping_reward = 2.0 * is_grasping

    # Stage 2: Position tool behind cube
    ideal_hook_pos = cube_pos + torch.tensor(
        [-(self.hook_length + self.cube_half_size), -0.067, 0], device=self.device
    )
    tool_positioning_dist = torch.linalg.norm(tool_pos - ideal_hook_pos, dim=1)
    positioning_reward = 1.5 * (1 - torch.tanh(3.0 * tool_positioning_dist))
    tool_positioned = tool_positioning_dist < 0.05

    # Stage 3: Pull cube to workspace
    workspace_target = robot_base_pos + torch.tensor(
        [0.05, 0, 0], device=self.device
    )
    cube_to_workspace_dist = torch.linalg.norm(cube_pos - workspace_target, dim=1)
    initial_dist = torch.linalg.norm(
        torch.tensor(
            [self.arm_reach + 0.1, 0, self.cube_size / 2], device=self.device
        )
        - workspace_target,
        dim=1,
    )
    pulling_progress = (initial_dist - cube_to_workspace_dist) / initial_dist
    pulling_reward = 3.0 * pulling_progress * tool_positioned

    # Combine rewards with staging and grasping dependency
    reward = reaching_reward + grasping_reward
    reward += positioning_reward * is_grasping
    reward += pulling_reward * is_grasping

    # Penalties
    cube_pushed_away = cube_pos[:, 0] > (self.arm_reach + 0.15)
    reward[cube_pushed_away] -= 2.0

    # Success bonus
    if "success" in info:
        reward[info["success"]] += 5.0

    return reward

def compute_normalized_dense_reward(
    self, obs: Any, action: torch.Tensor, info: dict
):
    """
    Normalizes the dense reward by the maximum possible reward (success bonus)
    """
    max_reward = 5.0  # Maximum possible reward from success bonus
    dense_reward = self.compute_dense_reward(obs=obs, action=action, info=info)
    return dense_reward / max_reward
```

Helper dependency: `self.agent.is_grasping(actor, max_angle=20)` (provided by the Panda/Fetch agent base — present in this repo). No task-local reward helpers / latch buffers.

### Smoke
```bash
.venv/bin/python -c "import gymnasium as gym, mani_skill; e=gym.make('PullCubeTool-v1', reward_mode='dense'); e.reset(seed=0); o,r,te,tr,info=e.step(e.action_space.sample()); print('reward', float(r.reshape(-1)[0])); e.close()"
```
Expected: a finite scalar reward (≈ in the low single digits at reset, dominated by Stage-1 reaching).

---

## §7 DR

`<no DR>`

No `startup`/`interval` randomization (no SAPIEN equivalent of an IsaacLab `EventCfg` DR term). The only per-episode randomization is the reset-time pose sampling of cube + tool and the robot qpos noise (`robot_init_qpos_noise=0.02`) — all documented in §3. No friction/mass/visual domain randomization is configured.

---

