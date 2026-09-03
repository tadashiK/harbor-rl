# PullCube-v1 — Implementation Spec

- robot: Franka Panda (default; Fetch also supported)
- simulator: ManiSkill (SAPIEN, `BaseEnv` + `@register_env`)
- objects: blue cube, goal region, table
- bimanual: false
- summary: Pull a cube across the table toward the robot into a goal region.

A simple tabletop manipulation task: a Panda arm must pull a blue cube along the table surface onto a red-and-white target region using a pushing/pulling motion from behind the cube. Single-stage, dense-reward, 50-step episodes.

---

## §1 Registration + Scene

### Description
Registered as `PullCube-v1` (subclass of `BaseEnv`) with `max_episode_steps=50`. Supports the Panda (default) and Fetch robots. The scene is a standard ManiSkill table built by `TableSceneBuilder`, with one dynamic blue cube (`half_size=0.02`) and a kinematic red/white target disk (`radius=0.1`, near-zero thickness, no collision). The agent is loaded shifted back to `p=[-0.615, 0, 0]` so it sits at the table edge.

### Decisions resolved
- `@register_env("PullCube-v1", max_episode_steps=50)`
- `SUPPORTED_ROBOTS = ["panda", "fetch"]`; default `robot_uids="panda"`; `robot_init_qpos_noise=0.02`
- Class constants: `goal_radius = 0.1`, `cube_half_size = 0.02`
- Agent base pose offset: `sapien.Pose(p=[-0.615, 0, 0])`
- Cube: `actors.build_cube(half_size=0.02, color=[12,42,160,255]/255, body_type="dynamic", initial_pose p=[0,0,0.02])`
- Target: `actors.build_red_white_target(radius=0.1, thickness=1e-5, add_collision=False, body_type="kinematic")`
- Sim config: default `SimConfig()` → `sim_freq=100`, `control_freq=20` (5 physx steps per control step; control dt = 0.05 s). `_default_sim_config` is NOT overridden by this task.
- Sensor camera: `base_camera` `look_at(eye=[-0.5,0,0.25], target=[0.2,0,-0.5])`, 128×128, fov π/2.
- Human render camera: `render_camera` `look_at([0.6,0.7,0.6],[0,0,0.35])`, 512×512, fov 1.0.
- Panda robot: URDF `panda_v2.urdf`; rest keyframe qpos `[0, π/8, 0, -5π/8, 0, 3π/4, π/4, 0.04, 0.04]`; arm stiffness 1e3 / damping 1e2 / force_limit 100; gripper stiffness 1e3 / damping 1e2 / force_limit 100; ee link `panda_hand_tcp`.

### Code
```python
@register_env("PullCube-v1", max_episode_steps=50)
class PullCubeEnv(BaseEnv):
    _sample_video_link = "https://github.com/haosulab/ManiSkill/raw/main/figures/environment_demos/PullCube-v1_rt.mp4"
    SUPPORTED_ROBOTS = ["panda", "fetch"]
    agent: Union[Panda, Fetch]
    goal_radius = 0.1
    cube_half_size = 0.02

    def __init__(self, *args, robot_uids="panda", robot_init_qpos_noise=0.02, **kwargs):
        self.robot_init_qpos_noise = robot_init_qpos_noise
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    @property
    def _default_sensor_configs(self):
        pose = look_at(eye=[-0.5, 0.0, 0.25], target=[0.2, 0.0, -0.5])
        return [CameraConfig("base_camera", pose, 128, 128, np.pi / 2, 0.01, 100)]

    @property
    def _default_human_render_camera_configs(self):
        pose = look_at([0.6, 0.7, 0.6], [0.0, 0.0, 0.35])
        return CameraConfig("render_camera", pose, 512, 512, 1, 0.01, 100)

    def _load_agent(self, options: dict):
        super()._load_agent(options, sapien.Pose(p=[-0.615, 0, 0]))

    def _load_scene(self, options: dict):
        self.table_scene = TableSceneBuilder(
            env=self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.table_scene.build()

        # create cube
        self.obj = actors.build_cube(
            self.scene,
            half_size=self.cube_half_size,
            color=np.array([12, 42, 160, 255]) / 255,
            name="cube",
            body_type="dynamic",
            initial_pose=sapien.Pose(p=[0, 0, self.cube_half_size]),
        )

        # create target
        self.goal_region = actors.build_red_white_target(
            self.scene,
            radius=self.goal_radius,
            thickness=1e-5,
            name="goal_region",
            add_collision=False,
            body_type="kinematic",
        )
```

### Smoke
```bash
cd <ManiSkill-repo>
.venv/bin/python -c "import gymnasium as gym, mani_skill; e=gym.make('PullCube-v1'); print('OBS', e.observation_space); print('ACT', e.action_space); print('CTRL', e.unwrapped.control_mode); e.close()"
```
Expected stdout (literal, minus a benign goal_region initial-pose warning):
```
OBS Box(-inf, inf, (1, 35), float32)
ACT Box(-1.0, 1.0, (8,), float32)
CTRL pd_joint_delta_pos
```

---

## §2 Actions (controller / control_mode + action space)

### Description
No task-local `ActionsCfg` — the action interface comes entirely from the robot agent's controller configs. The default control mode is `pd_joint_delta_pos`: a 7-DoF arm PD joint-position-delta controller plus a 1-DoF gripper PD position controller. Action space is 8-dim, normalized to [-1, 1].

### Decisions resolved
- `control_mode = "pd_joint_delta_pos"` (Panda default).
- Action space: `Box(-1.0, 1.0, (8,), float32)` = 7 arm joint deltas + 1 gripper.
- Arm controller `arm_pd_joint_delta_pos` (`PDJointPosControllerConfig`): joints = 7 panda arm joints, `lower=-0.1`, `upper=0.1`, `use_delta=True`, stiffness=1e3, damping=1e2, force_limit=100. Normalized action maps [-1,1] → [-0.1, 0.1] rad delta per joint per control step.
- Gripper controller `gripper_pd_joint_pos` (PD joint position over the two finger joints).

### Code
```python
# mani_skill/agents/robots/panda/panda.py
arm_pd_joint_delta_pos = PDJointPosControllerConfig(
    self.arm_joint_names,         # 7 panda arm joints
    lower=-0.1,
    upper=0.1,
    stiffness=self.arm_stiffness, # 1e3
    damping=self.arm_damping,     # 1e2
    force_limit=self.arm_force_limit,  # 100
    use_delta=True,
)
...
controller_configs = dict(
    pd_joint_delta_pos=dict(
        arm=arm_pd_joint_delta_pos, gripper=gripper_pd_joint_pos
    ),
    ...
)
```

### Smoke
```bash
.venv/bin/python -c "import gymnasium as gym, mani_skill; e=gym.make('PullCube-v1'); print(e.action_space, e.unwrapped.control_mode); e.close()"
```
Expected: `Box(-1.0, 1.0, (8,), float32) pd_joint_delta_pos`

---

## §3 Reset (`_initialize_episode`)

### Description
On reset, the table scene is re-initialized (resets robot to rest keyframe with `robot_init_qpos_noise=0.02`), the cube's xy is randomized in `[-0.1, 0.1]²` on the table surface, and the target disk is placed `0.1 + goal_radius` (= 0.2 m) behind the cube in -x, rotated to lie flat (`euler2quat(0, π/2, 0)`).

### Decisions resolved
- Cube xy: `rand(b,2) * 0.2 - 0.1` → uniform in `[-0.1, 0.1]²`; z = `cube_half_size` (0.02); quat `[1,0,0,0]`.
- Target xyz: `cube_xyz - [0.1 + goal_radius, 0, 0]` = cube_xy shifted -0.2 m in x; z set to `1e-3`; quat `euler2quat(0, π/2, 0)`.
- Robot init via `TableSceneBuilder.initialize(env_idx)` with `robot_init_qpos_noise=0.02`.

### Code
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

        target_region_xyz = xyz - torch.tensor([0.1 + self.goal_radius, 0, 0])
        target_region_xyz[..., 2] = 1e-3
        self.goal_region.set_pose(
            Pose.create_from_pq(
                p=target_region_xyz,
                q=euler2quat(0, np.pi / 2, 0),
            )
        )
```

### Smoke
```bash
.venv/bin/python -c "import gymnasium as gym, mani_skill; e=gym.make('PullCube-v1'); e.reset(seed=0); e.reset(seed=1); print('reset ok'); e.close()"
```
Expected: `reset ok`

---

## §4 Goal + Termination (`evaluate()` + max_episode_steps)

### Description
Success when the cube's xy is within `goal_radius` (0.1 m) of the target's xy (Euclidean). There is no failure/early-termination condition; episodes end on success-or-timeout. `max_episode_steps=50` is the only time bound (truncation via the gym `TimeLimit`).

### Decisions resolved
- `evaluate()` returns `{"success": ||cube.xy - goal.xy|| < 0.1}`.
- `max_episode_steps = 50` (from `@register_env`). At control_freq 20 Hz → 2.5 s episodes.
- No explicit termination term beyond success; no fail condition.

### Code
```python
def evaluate(self):
    is_obj_placed = (
        torch.linalg.norm(
            self.obj.pose.p[..., :2] - self.goal_region.pose.p[..., :2], axis=1
        )
        < self.goal_radius
    )
    return {
        "success": is_obj_placed,
    }
```

### Smoke
```bash
.venv/bin/python -c "import gymnasium as gym, mani_skill; e=gym.make('PullCube-v1'); o,i=e.reset(seed=0); print('success' in i); e.close()"
```
Expected: `True`

---

## §5 Observation (`_get_obs_extra` + obs modes + dim)

### Description
Default obs_mode is `state` (first of `SUPPORTED_OBS_MODES`), which concatenates robot proprioception with the task-extra dict into a flat 35-dim vector. `_get_obs_extra` adds the TCP pose, the goal position, and — in state/state_dict modes — the cube pose.

### Decisions resolved
- `SUPPORTED_OBS_MODES = ("state", "state_dict", "none", "sensor_data", "any_textures", "pointcloud")`; default = `"state"`.
- Resolved total obs dim (state mode): `Box(-inf, inf, (1, 35), float32)` → 35 per env.
  - Analytic breakdown: proprio `agent.qpos`(9) + `qvel`(9) + tcp_pose(7) = base agent obs; extra: `tcp_pose`(7) + `goal_pos`(3) + `obj_pose`(7, state only). 9+9 (robot) + 7 (tcp_pose) + 3 (goal_pos) + 7 (obj_pose) = 35.
- `_get_obs_extra` keys: `tcp_pose` (`agent.tcp.pose.raw_pose`, 7-dim pq), `goal_pos` (`goal_region.pose.p`, 3-dim), and `obj_pose` (`obj.pose.raw_pose`, 7-dim) only when `obs_mode_struct.use_state` is True.

### Code
```python
def _get_obs_extra(self, info: dict):
    obs = dict(
        tcp_pose=self.agent.tcp.pose.raw_pose,
        goal_pos=self.goal_region.pose.p,
    )
    if self.obs_mode_struct.use_state:
        obs.update(
            obj_pose=self.obj.pose.raw_pose,
        )
    return obs
```

### Smoke
```bash
.venv/bin/python -c "import gymnasium as gym, mani_skill; e=gym.make('PullCube-v1'); print(e.observation_space.shape); e.close()"
```
Expected: `(1, 35)`

---

## §6 Reward (`compute_dense_reward` + `compute_normalized_dense_reward`, verbatim + composer)

### Description
Two-term shaped dense reward, composed by **summation** (with a gate). Term 1 is a continuous reaching reward toward a point just behind the cube (the "pull position"). Term 2 is a placement reward (cube→goal distance), gated so it only contributes once the TCP has reached the pull position (`reached`, dist < 0.01). On success the reward is overwritten to a flat 3.0. The normalized variant divides by `max_reward = 3.0`.

### Composer
**sum** — `reward = reaching_reward + place_reward * reached`, then `reward[success] = 3`. Per-term saturated per-step magnitudes:
- `reaching_reward` ∈ [0, 1] (1 - tanh(5·dist)); saturates at ~1 when TCP at pull pos.
- `place_reward * reached` ∈ {0} ∪ [0, 1]; only active once reached, saturates at ~1 when cube at goal.
- success override = 3.0 (dominates; equals max of the two summed terms at perfect placement).
(retro-computed from weights — no explicit planning-budget docstring in source.)

### Code (verbatim)
```python
def compute_dense_reward(self, obs: Any, action: Array, info: dict):
    # grippers should close and pull from behind the cube, not grip it
    # distance to backside of cube (+ 2*0.005) sufficiently encourages this
    tcp_pull_pos = self.obj.pose.p + torch.tensor(
        [self.cube_half_size + 2 * 0.005, 0, 0], device=self.device
    )
    tcp_to_pull_pose = tcp_pull_pos - self.agent.tcp.pose.p
    tcp_to_pull_pose_dist = torch.linalg.norm(tcp_to_pull_pose, axis=1)
    reaching_reward = 1 - torch.tanh(5 * tcp_to_pull_pose_dist)
    reward = reaching_reward

    reached = tcp_to_pull_pose_dist < 0.01
    obj_to_goal_dist = torch.linalg.norm(
        self.obj.pose.p[..., :2] - self.goal_region.pose.p[..., :2], axis=1
    )
    place_reward = 1 - torch.tanh(5 * obj_to_goal_dist)
    reward += place_reward * reached

    reward[info["success"]] = 3
    return reward

def compute_normalized_dense_reward(self, obs: Any, action: Array, info: dict):
    max_reward = 3.0
    return self.compute_dense_reward(obs=obs, action=action, info=info) / max_reward
```

### Smoke
```bash
.venv/bin/python -c "import gymnasium as gym, mani_skill, torch; e=gym.make('PullCube-v1', reward_mode='dense'); o,i=e.reset(seed=0); o,r,te,tr,i=e.step(e.action_space.sample()); print(torch.isfinite(torch.as_tensor(r)).all().item()); e.close()"
```
Expected: `True` (finite reward).

---

## §7 DR (Domain Randomization)

`<no DR>` — there is no `_default_sim_config` override, no episodic randomization beyond the cube/robot init pose noise in §3 (cube xy uniform, `robot_init_qpos_noise=0.02`), and no startup/interval physical-property randomization (mass, friction, damping, etc.). The task uses default sim/material parameters everywhere.

---

## Reproduce
`/harbor:task-create name=<NewTaskID> from=<ManiSkill-repo>/harbor/create-task/pullcube-v1-implementation.md`
