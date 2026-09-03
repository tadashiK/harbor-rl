# PushCube-v1 — Implementation Spec

- robot: Franka Panda (default; Fetch also supported)
- simulator: ManiSkill (SAPIEN, `BaseEnv` + `@register_env`)
- objects: blue cube, goal region, table
- bimanual: false
- summary: Push a cube across the table into a goal region.

> ManiSkill maps to the §1..§7 sections as: §1 = `@register_env` + `SUPPORTED_ROBOTS` + `_load_agent` + `_load_scene` + sim cfg; §2 = controller / control_mode + action space; §3 = `_initialize_episode`; §4 = `evaluate()` + `max_episode_steps`; §5 = `_get_obs_extra` + obs modes; §6 = `compute_dense_reward` / `compute_normalized_dense_reward`; §7 = domain randomization.

---

## §1 Registration + Scene

**Description.** A `BaseEnv` subclass `PushCubeEnv` registered as `PushCube-v1` with `max_episode_steps=50`. The scene is a prebuilt table + floor (`TableSceneBuilder`), a single dynamic blue cube (`half_size=0.02`) to push, and a red/white circular target marking the goal region (`radius=0.1`, kinematic, collision-free — visual only). The Panda arm is loaded with its base at `p=[-0.615, 0, 0]`. Two cameras are configured: a 128×128 `base_camera` (for sensor obs) and a 512×512 `render_camera` (for `rgb_array` rendering). Table surface is at z=0.

**Decisions resolved.**
- `max_episode_steps = 50` (set in the `@register_env` decorator).
- `SUPPORTED_ROBOTS = ["panda", "fetch"]`; default `robot_uids="panda"`.
- `robot_init_qpos_noise = 0.02` (ctor kwarg, passed to `TableSceneBuilder`).
- `goal_radius = 0.1`, `cube_half_size = 0.02` (class attributes).
- Cube color = `np.array([12, 42, 160, 255]) / 255` (blue), `body_type="dynamic"`, initial pose `p=[0, 0, 0.02]`.
- Goal region: `build_red_white_target`, `radius=0.1`, `thickness=1e-5`, `add_collision=False`, `body_type="kinematic"`, initial pose `p=[0, 0, 1e-3]`.
- Agent base pose `sapien.Pose(p=[-0.615, 0, 0])`.
- Sim cfg: `GPUMemoryConfig(found_lost_pairs_capacity=2**25, max_rigid_patch_count=2**18)`.
- `base_camera`: `look_at(eye=[0.3,0,0.6], target=[-0.1,0,0.1])`, 128×128, `fov=np.pi/2`, `near=0.01`, `far=100`.
- `render_camera`: `look_at([0.6,0.7,0.6],[0.0,0.0,0.35])`, 512×512, `fov=1`, `near=0.01`, `far=100`.

**Code.**
```python
@register_env("PushCube-v1", max_episode_steps=50)
class PushCubeEnv(BaseEnv):
    SUPPORTED_ROBOTS = ["panda", "fetch"]
    agent: Union[Panda, Fetch]

    goal_radius = 0.1
    cube_half_size = 0.02

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
        pose = sapien_utils.look_at(eye=[0.3, 0, 0.6], target=[-0.1, 0, 0.1])
        return [
            CameraConfig(
                "base_camera", pose=pose, width=128, height=128,
                fov=np.pi / 2, near=0.01, far=100,
            )
        ]

    @property
    def _default_human_render_camera_configs(self):
        pose = sapien_utils.look_at([0.6, 0.7, 0.6], [0.0, 0.0, 0.35])
        return CameraConfig(
            "render_camera", pose=pose, width=512, height=512, fov=1, near=0.01, far=100
        )

    def _load_agent(self, options: dict):
        super()._load_agent(options, sapien.Pose(p=[-0.615, 0, 0]))

    def _load_scene(self, options: dict):
        self.table_scene = TableSceneBuilder(
            env=self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.table_scene.build()

        self.obj = actors.build_cube(
            self.scene,
            half_size=self.cube_half_size,
            color=np.array([12, 42, 160, 255]) / 255,
            name="cube",
            body_type="dynamic",
            initial_pose=sapien.Pose(p=[0, 0, self.cube_half_size]),
        )

        self.goal_region = actors.build_red_white_target(
            self.scene,
            radius=self.goal_radius,
            thickness=1e-5,
            name="goal_region",
            add_collision=False,
            body_type="kinematic",
            initial_pose=sapien.Pose(p=[0, 0, 1e-3]),
        )
```

**Smoke.**
```bash
cd <repo>
.venv/bin/python -c "import gymnasium as gym, mani_skill; e=gym.make('PushCube-v1'); print(e.observation_space, e.action_space); e.close()"
# -> Box(-inf, inf, (1, 35), float32) Box(-1.0, 1.0, (8,), float32)
```

---

## §2 Actions

**Description.** No `ActionsCfg` block — control is selected via ManiSkill's controller framework. The default control mode is `pd_joint_delta_pos`: a per-joint delta-position PD controller over the Panda's 7 arm joints + 1 gripper command = 8-D action, each component in [-1, 1]. The action is the only supported control mode reported by the agent in this build.

**Decisions resolved.**
- `control_mode = "pd_joint_delta_pos"` (default).
- Action space = `Box(-1.0, 1.0, (8,), float32)` (7 arm joint deltas + 1 gripper).
- Controller params (scale, stiffness, damping) come from the Panda agent config, not from the env file.

**Code.** (env file defines no actions; control is the Panda agent default)
```python
# control_mode resolves to "pd_joint_delta_pos" via the Panda agent's controller configs.
# Action: Box(-1.0, 1.0, (8,), float32)  ==  7 arm joint position deltas + 1 gripper command.
```

**Smoke.**
```bash
.venv/bin/python -c "import gymnasium as gym, mani_skill; e=gym.make('PushCube-v1'); print(e.action_space, e.unwrapped.control_mode); e.close()"
# -> Box(-1.0, 1.0, (8,), float32) pd_joint_delta_pos
```

---

## §3 Reset

**Description.** `_initialize_episode` does a partial reset over `env_idx`. It initializes the table scene, then randomizes the cube's xy on the table in `[-0.1, -0.1] x [0.1, 0.1]` (placed flat, `z = cube_half_size`), and deterministically places the goal region at `cube_xy + [0.1 + goal_radius, 0]` (i.e. `+0.2` in x), rotated 90° about y so the target faces up, at `z = 1e-3`.

**Decisions resolved.**
- Cube xy = `rand(b,2) * 0.2 - 0.1` → uniform in [-0.1, 0.1]^2; cube z = `cube_half_size` (0.02); quat = `[1,0,0,0]`.
- Goal xyz = cube_xyz + `[0.1 + goal_radius, 0, 0]` = `+[0.2, 0, 0]`; goal z overwritten to `1e-3`; goal quat = `euler2quat(0, np.pi/2, 0)`.
- Robot init qpos noise = 0.02 (via `TableSceneBuilder.initialize`).

**Code.**
```python
def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
    with torch.device(self.device):
        b = len(env_idx)
        self.table_scene.initialize(env_idx)

        xyz = torch.zeros((b, 3))
        xyz[..., :2] = torch.rand((b, 2)) * 0.2 - 0.1
        xyz[..., 2] = self.cube_half_size
        q = [1, 0, 0, 0]
        obj_pose = Pose.create_from_pq(p=xyz, q=q)
        self.obj.set_pose(obj_pose)

        target_region_xyz = xyz + torch.tensor([0.1 + self.goal_radius, 0, 0])
        target_region_xyz[..., 2] = 1e-3
        self.goal_region.set_pose(
            Pose.create_from_pq(
                p=target_region_xyz,
                q=euler2quat(0, np.pi / 2, 0),
            )
        )
```

**Smoke.**
```bash
.venv/bin/python -c "import gymnasium as gym, mani_skill; e=gym.make('PushCube-v1'); o,_=e.reset(seed=0); print(o.shape); e.close()"
# -> torch.Size([1, 35])
```

---

## §4 Goal + Termination

**Description.** Success when the cube's xy is within `goal_radius` (0.1) of the goal-region xy AND the cube is still on the table (`z < cube_half_size + 5e-3`). There is no failure termination; episodes end only by success or the `max_episode_steps=50` time-out.

**Decisions resolved.**
- Success: `||cube.xy - goal.xy|| < 0.1` AND `cube.z < 0.025`.
- `max_episode_steps = 50` (time-out termination).
- No explicit fail/early-termination condition.

**Code.**
```python
def evaluate(self):
    is_obj_placed = (
        torch.linalg.norm(
            self.obj.pose.p[..., :2] - self.goal_region.pose.p[..., :2], axis=1
        )
        < self.goal_radius
    ) & (self.obj.pose.p[..., 2] < self.cube_half_size + 5e-3)

    return {
        "success": is_obj_placed,
    }
```

**Smoke.**
```bash
.venv/bin/python -c "import gymnasium as gym, mani_skill; e=gym.make('PushCube-v1'); e.reset(seed=0); o,r,te,tr,info=e.step(e.action_space.sample()); print('success' in info, e.spec.max_episode_steps or 50); e.close()"
# -> True 50
```

---

## §5 Observation

**Description.** Default `obs_mode="state"` yields a flat 35-D vector. `_get_obs_extra` provides the task-specific extras: `tcp_pose` (always), plus `goal_pos` and `obj_pose` only when the obs mode uses state. The base env prepends Panda proprioception (qpos + qvel) to these extras.

**Decisions resolved.** Total obs dim = 35 (`state` mode), composed of:
- agent proprioception: qpos (9) + qvel (9) = 18
- `tcp_pose` = `agent.tcp.pose.raw_pose` → pos(3) + quat(4) = 7
- `goal_pos` = `goal_region.pose.p` = 3 (state only)
- `obj_pose` = `obj.pose.raw_pose` = 7 (state only)
- → 18 + 7 + 3 + 7 = **35**.

Other obs modes: `state_dict`, `sensor_data`, `rgbd`, `pointcloud`, etc. (visual modes drop `goal_pos`/`obj_pose` from extras and rely on `base_camera` 128×128). No observation noise is configured.

**Code.**
```python
def _get_obs_extra(self, info: dict):
    obs = dict(
        tcp_pose=self.agent.tcp.pose.raw_pose,
    )
    if self.obs_mode_struct.use_state:
        obs.update(
            goal_pos=self.goal_region.pose.p,
            obj_pose=self.obj.pose.raw_pose,
        )
    return obs
```

**Smoke.**
```bash
.venv/bin/python -c "import gymnasium as gym, mani_skill; e=gym.make('PushCube-v1'); print(e.observation_space.shape); e.close()"
# -> (1, 35)
```

---

## §6 Reward

**Description.** Staged dense reward composed as a **SUM** of three additive terms, with a flat success override.
1. **reaching_reward** = `1 - tanh(5 * dist(tcp, push_pose))`, where `push_pose` is just behind the cube (`cube.p + [-cube_half_size - 0.005, 0, 0]`) — the easiest place to push from.
2. **place_reward × reached** = `(1 - tanh(5 * ||obj.xy - goal.xy||)) * (tcp_to_push_dist < 0.01)` — only credited once the tcp has reached the push pose.
3. **place_reward × z_reward × reached** = additionally keeps the cube on the table (`z_reward = 1 - tanh(5 * |cube.z - cube_half_size|)`), gated by both `reached` and the placement progress.
On success, the total reward is overwritten to the flat maximum **4.0**. `compute_normalized_dense_reward` divides by `max_reward = 4.0`.

**Composer:** SUM (three additive terms; success replaces total with 4.0). Max reward = 4.0.

**Decisions resolved.**
- `push_pose = cube.p + [-(cube_half_size) - 0.005, 0, 0]` = `cube.p + [-0.025, 0, 0]`.
- `reached` gate threshold = `tcp_to_push_pose_dist < 0.01`.
- tanh sharpness = 5 for all three shaping terms.
- success override value = 4; `max_reward = 4.0`.
- Per-term saturated per-step magnitudes (max each ≈ 1.0): reaching → 1.0; place×reached → 1.0; place×z×reached → 1.0; total un-success max ≈ 3.0, success flat 4.0. (Note: the inline comment on the success line says "maximum of 3" but the code assigns 4 and `max_reward=4.0`.)

**Code (verbatim).**
```python
def compute_dense_reward(self, obs: Any, action: Array, info: dict):
    tcp_push_pose = Pose.create_from_pq(
        p=self.obj.pose.p
        + torch.tensor([-self.cube_half_size - 0.005, 0, 0], device=self.device)
    )
    tcp_to_push_pose = tcp_push_pose.p - self.agent.tcp.pose.p
    tcp_to_push_pose_dist = torch.linalg.norm(tcp_to_push_pose, axis=1)
    reaching_reward = 1 - torch.tanh(5 * tcp_to_push_pose_dist)
    reward = reaching_reward

    reached = tcp_to_push_pose_dist < 0.01
    obj_to_goal_dist = torch.linalg.norm(
        self.obj.pose.p[..., :2] - self.goal_region.pose.p[..., :2], axis=1
    )
    place_reward = 1 - torch.tanh(5 * obj_to_goal_dist)
    reward += place_reward * reached

    desired_obj_z = self.cube_half_size
    current_obj_z = self.obj.pose.p[..., 2]
    z_deviation = torch.abs(current_obj_z - desired_obj_z)
    z_reward = 1 - torch.tanh(5 * z_deviation)
    reward += place_reward * z_reward * reached

    reward[info["success"]] = 4
    return reward

def compute_normalized_dense_reward(self, obs: Any, action: Array, info: dict):
    max_reward = 4.0
    return self.compute_dense_reward(obs=obs, action=action, info=info) / max_reward
```

**Smoke.**
```bash
.venv/bin/python -c "import gymnasium as gym, mani_skill, torch; e=gym.make('PushCube-v1',reward_mode='dense'); e.reset(seed=0); o,r,te,tr,info=e.step(e.action_space.sample()); print(torch.isfinite(torch.as_tensor(r)).all().item()); e.close()"
# -> True
```

---

## §7 DR

`<no DR>`

The only stochasticity is episode-initialization randomization (§3): cube xy uniform in [-0.1, 0.1]^2 and `robot_init_qpos_noise=0.02`. There are no `startup` / `interval` domain-randomization events (no mass/friction/visual randomization). This matches ManiSkill's default for PushCube-v1.

---

