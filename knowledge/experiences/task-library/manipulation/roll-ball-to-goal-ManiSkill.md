# RollBall-v1 — Implementation Spec

- robot: Franka Panda
- simulator: ManiSkill (SAPIEN, `BaseEnv` + `@register_env`)
- objects: ball, goal region, table
- bimanual: false
- summary: Roll a ball across the table into a goal region.

> ManiSkill maps to the §1..§7 sections as: §1 = `@register_env` + `SUPPORTED_ROBOTS` + `_load_agent` + `_load_scene` + sim cfg; §2 = controller / control_mode + action space; §3 = `_initialize_episode`; §4 = `evaluate()` + `max_episode_steps`; §5 = `_get_obs_extra` + obs modes; §6 = `compute_dense_reward` / `compute_normalized_dense_reward`; §7 = domain randomization.

---

## §1 Registration + Scene

**Description.** A `BaseEnv` subclass `RollBallEnv` registered as `RollBall-v1` with `max_episode_steps=80`. The task is to push/roll a dynamic ball across the table into a goal region at the far end. The scene is a prebuilt table + floor (`TableSceneBuilder`), a single dynamic blue sphere (`radius=0.035`) to roll, and a red/white circular target marking the goal region (`radius=0.1`, kinematic, collision-free — visual only). The Panda arm is loaded with its base at `p=[-0.615, 0, 0]` (then repositioned in `_initialize_episode`). Two cameras: a 128×128 `base_camera` (sensor obs) and a 512×512 `render_camera` (rgb_array rendering). Table surface is at z=0. A per-env `reached_status` float buffer (shape `[num_envs]`) is allocated here — it is a reward-stage latch (see §6).

**Decisions resolved.**
- `max_episode_steps = 80` (set in the `@register_env` decorator).
- `SUPPORTED_ROBOTS = ["panda"]`; default `robot_uids="panda"`. (`Fetch` is imported but not in SUPPORTED_ROBOTS.)
- `robot_init_qpos_noise = 0.02` (ctor kwarg, passed to `TableSceneBuilder`).
- `goal_radius = 0.1`, `ball_radius = 0.035` (class attributes).
- Ball color = `[0, 0.2, 0.8, 1]` (blue), built via `actors.build_sphere`, `body_type` default (dynamic), initial pose `p=[0, 0, 0.1]`.
- Goal region: `build_red_white_target`, `radius=0.1`, `thickness=1e-5`, `add_collision=False`, `body_type="kinematic"`, initial pose `p=[0, 0, 0.1]`.
- Agent base load pose `sapien.Pose(p=[-0.615, 0, 0])` (overridden in §3 reset).
- `reached_status = torch.zeros(num_envs, dtype=torch.float32)` — reward latch buffer, allocated in `_load_scene`.
- Sim cfg: `GPUMemoryConfig(found_lost_pairs_capacity=2**25, max_rigid_patch_count=2**18)`.
- `base_camera`: `look_at(eye=[-0.1, 0.9, 0.3], target=[0.0, 0.0, 0.0])`, 128×128, `fov=np.pi/2`, `near=0.01`, `far=100`.
- `render_camera`: `look_at([-0.6, 1.3, 0.8], [0.0, 0.13, 0.0])`, 512×512, `fov=1`, `near=0.01`, `far=100`.

**Code.**
```python
@register_env("RollBall-v1", max_episode_steps=80)
class RollBallEnv(BaseEnv):
    _sample_video_link = "https://github.com/haosulab/ManiSkill/raw/main/figures/environment_demos/RollBall-v1_rt.mp4"
    SUPPORTED_ROBOTS = ["panda"]

    agent: Panda

    goal_radius: float = 0.1  # radius of the goal region
    ball_radius: float = 0.035  # radius of the ball
    reached_status: torch.Tensor

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
        pose = sapien_utils.look_at(eye=[-0.1, 0.9, 0.3], target=[0.0, 0.0, 0.0])
        return [CameraConfig("base_camera", pose, 128, 128, np.pi / 2, 0.01, 100)]

    @property
    def _default_human_render_camera_configs(self):
        pose = sapien_utils.look_at([-0.6, 1.3, 0.8], [0.0, 0.13, 0.0])
        return CameraConfig("render_camera", pose, 512, 512, 1, 0.01, 100)

    def _load_agent(self, options: dict):
        super()._load_agent(options, sapien.Pose(p=[-0.615, 0, 0]))

    def _load_scene(self, options: dict):
        self.table_scene = TableSceneBuilder(
            self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.table_scene.build()

        self.ball = actors.build_sphere(
            self.scene,
            radius=self.ball_radius,
            color=[0, 0.2, 0.8, 1],
            name="ball",
            initial_pose=sapien.Pose(p=[0, 0, 0.1]),
        )

        self.goal_region = actors.build_red_white_target(
            self.scene,
            radius=self.goal_radius,
            thickness=1e-5,
            name="goal_region",
            add_collision=False,
            body_type="kinematic",
            initial_pose=sapien.Pose(p=[0, 0, 0.1]),
        )
        self.reached_status = torch.zeros(self.num_envs, dtype=torch.float32)
```

**Smoke.**
```bash
cd <repo>
.venv/bin/python -c "import gymnasium as gym, mani_skill; e=gym.make('RollBall-v1'); print(e.observation_space, e.action_space); e.close()"
# -> Box(-inf, inf, (1, 44), float32) Box(-1.0, 1.0, (8,), float32)
```

---

## §2 Actions

**Description.** No `ActionsCfg` block — control is selected via ManiSkill's controller framework. The default control mode is `pd_joint_delta_pos`: a per-joint delta-position PD controller over the Panda's 7 arm joints + 1 gripper command = 8-D action, each component in [-1, 1]. (The gripper is unused functionally here — the arm pushes the ball — but the standard Panda action layout still exposes it.)

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
.venv/bin/python -c "import gymnasium as gym, mani_skill; e=gym.make('RollBall-v1'); print(e.action_space, e.unwrapped.control_mode); e.close()"
# -> Box(-1.0, 1.0, (8,), float32) pd_joint_delta_pos
```

---

## §3 Reset

**Description.** `_initialize_episode` does a partial reset over `env_idx`. It moves `reached_status` to device and zeros the slots for the reset envs, initializes the table scene, then **overrides the robot base pose** to `p=[-0.1, 1.0, 0]`, `q=[0.7071, 0, 0, -0.7072]` (the arm is placed at the +y near edge facing across the table). The ball is randomized on the table at one end and the goal region at the opposite end:
- Ball xy: `x ∈ [-0.4, 0.2]` (= `(rand*2-1)*0.3 - 0.1`), `y ∈ [0.5, 0.7]` (= `rand*0.2 + 0.5`); placed flat at `z = ball_radius`; quat `[1,0,0,0]`.
- Goal xy: `x ∈ [-0.4, 0.2]` (= `(rand*2-1)*0.3 - 0.1`), `y ∈ [-1.0+goal_radius, -0.8+goal_radius] = [-0.9, -0.7]` (= `rand*0.2 - 1.0 + goal_radius`); `z = 1e-3`; quat `euler2quat(0, np.pi/2, 0)` (target faces up).

So the ball starts near +y and the goal sits near −y at the far end of the table.

> Note: the docstring's stated randomization regions (`[0.2,0.5]x[-0.4,0.7]` for the ball, `[-0.4,-0.7]x[0.2,-0.9]` for the goal) are informal/approximate; the authoritative ranges are the code expressions above.

**Decisions resolved.**
- Robot base pose override: `Pose.create_from_pq(p=[-0.1, 1.0, 0], q=[0.7071, 0, 0, -0.7072])`.
- Ball xy ranges: x = `(rand(b)*2-1)*0.3 - 0.1` → uniform in [-0.4, 0.2]; y = `rand(b)*0.2 + 0.5` → uniform in [0.5, 0.7]; z = `ball_radius` (0.035); quat = `[1,0,0,0]`.
- Goal xy ranges: x = `(rand(b)*2-1)*0.3 - 0.1` → uniform in [-0.4, 0.2]; y = `rand(b)*0.2 - 1.0 + goal_radius` → uniform in [-0.9, -0.7]; z = `1e-3`; quat = `euler2quat(0, np.pi/2, 0)`.
- `reached_status[env_idx] = 0.0` (latch reset for the reset envs).
- Robot init qpos noise = 0.02 (via `TableSceneBuilder.initialize`).

**Code.**
```python
def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
    self.reached_status = self.reached_status.to(self.device)
    with torch.device(self.device):
        b = len(env_idx)
        self.table_scene.initialize(env_idx)

        robot_pose = Pose.create_from_pq(
            p=[-0.1, 1.0, 0], q=[0.7071, 0, 0, -0.7072]
        )
        self.agent.robot.set_pose(robot_pose)

        xyz = torch.zeros((b, 3))
        xyz[..., 0] = (torch.rand((b)) * 2 - 1) * 0.3 - 0.1
        xyz[..., 1] = torch.rand((b)) * 0.2 + 0.5
        xyz[..., 2] = self.ball_radius
        q = [1, 0, 0, 0]

        obj_pose = Pose.create_from_pq(p=xyz, q=q)
        self.ball.set_pose(obj_pose)

        xyz_goal = torch.zeros((b, 3))
        xyz_goal[..., 0] = (torch.rand((b)) * 2 - 1) * 0.3 - 0.1
        xyz_goal[..., 1] = torch.rand((b)) * 0.2 - 1.0 + self.goal_radius
        xyz_goal[..., 2] = 1e-3
        self.goal_region.set_pose(
            Pose.create_from_pq(
                p=xyz_goal,
                q=euler2quat(0, np.pi / 2, 0),
            )
        )
    self.reached_status[env_idx] = 0.0
```

**Smoke.**
```bash
.venv/bin/python -c "import gymnasium as gym, mani_skill; e=gym.make('RollBall-v1'); o,_=e.reset(seed=0); print(o.shape); e.close()"
# -> torch.Size([1, 44])
```

---

## §4 Goal + Termination

**Description.** Success when the ball's xy is within `goal_radius` (0.1) of the goal-region xy by Euclidean distance. There is no failure termination; episodes end only by success or the `max_episode_steps=80` time-out.

**Decisions resolved.**
- Success: `||ball.xy - goal.xy|| < 0.1` (no z gate).
- `max_episode_steps = 80` (time-out termination).
- No explicit fail/early-termination condition.

**Code.**
```python
def evaluate(self):

    is_obj_placed = (
        torch.linalg.norm(
            self.ball.pose.p[..., :2] - self.goal_region.pose.p[..., :2], axis=1
        )
        < self.goal_radius
    )

    return {
        "success": is_obj_placed,
    }
```

**Smoke.**
```bash
.venv/bin/python -c "import gymnasium as gym, mani_skill; e=gym.make('RollBall-v1'); e.reset(seed=0); o,r,te,tr,info=e.step(e.action_space.sample()); print('success' in info, e.spec.max_episode_steps); e.close()"
# -> True 80
```

---

## §5 Observation

**Description.** Default `obs_mode="state"` yields a flat 44-D vector. `_get_obs_extra` provides task-specific extras: `tcp_pose` (always), plus `goal_pos`, `ball_pose`, `ball_vel`, `tcp_to_ball_pos`, `ball_to_goal_pos` only when the obs mode uses state. The base env prepends Panda proprioception (qpos + qvel) to these extras. Note the velocity-aware term `ball_vel` (the ball's linear velocity) — relevant to rolling dynamics.

**Decisions resolved.** Total obs dim = 44 (`state` mode), composed of:
- agent proprioception: qpos (9) + qvel (9) = 18
- `tcp_pose` = `agent.tcp.pose.raw_pose` → pos(3) + quat(4) = 7  (always)
- `goal_pos` = `goal_region.pose.p` = 3 (state only)
- `ball_pose` = `ball.pose.raw_pose` = 7 (state only)
- `ball_vel` = `ball.linear_velocity` = 3 (state only)
- `tcp_to_ball_pos` = `ball.pose.p - agent.tcp.pose.p` = 3 (state only)
- `ball_to_goal_pos` = `goal_region.pose.p - ball.pose.p` = 3 (state only)
- → 18 + 7 + 3 + 7 + 3 + 3 + 3 = **44**.

Other obs modes: `state_dict`, `sensor_data`, `rgbd`, `pointcloud`, etc. (visual modes drop the state-only extras and rely on `base_camera` 128×128). No observation noise is configured.

**Code.**
```python
def _get_obs_extra(self, info: dict):

    obs = dict(
        tcp_pose=self.agent.tcp.pose.raw_pose,
    )
    if self.obs_mode_struct.use_state:
        obs.update(
            goal_pos=self.goal_region.pose.p,
            ball_pose=self.ball.pose.raw_pose,
            ball_vel=self.ball.linear_velocity,
            tcp_to_ball_pos=self.ball.pose.p - self.agent.tcp.pose.p,
            ball_to_goal_pos=self.goal_region.pose.p - self.ball.pose.p,
        )
    return obs
```

**Smoke.**
```bash
.venv/bin/python -c "import gymnasium as gym, mani_skill; e=gym.make('RollBall-v1'); print(e.observation_space.shape); e.close()"
# -> (1, 44)
```

---

## §6 Reward

**Description.** A two-stage latched dense reward composed as a **SUM** of mutually-gated terms, with a flat success override. The latch buffer `self.reached_status` (per-env float, 0.0 or 1.0; allocated in `_load_scene`, reset in `_initialize_episode`) records whether the TCP has ever reached the "hit pose" — a point just behind the ball on the ball→goal line where the arm should make contact to push it toward the goal.

Stage A (pre-contact, `reached_status == 0`):
- Compute `unit_vec = normalize(ball.p - goal.p)` (points from goal toward ball).
- `tcp_hit_pose = ball.p + unit_vec * (ball_radius + 0.05)` — a point on the far side of the ball from the goal, i.e. where the arm must be to push the ball goal-ward.
- `reaching_reward = 1 - tanh(2 * dist(tcp, tcp_hit_pose))`.
- When `dist(tcp, hit_pose) < 0.04`, latch `reached_status = 1.0` (sticky for the rest of the episode).
- Contribution while not reached: `reaching_reward * (1 - reached_status)`.

Stage B (post-contact, `reached_status == 1`):
- `obj_to_goal_dist = ||ball.xy - goal.xy||`.
- `reached_reward = 1 - tanh(obj_to_goal_dist)`.
- Contribution: `20 * reached_reward * reached_status` plus a flat `+ reached_status` bonus (=1 once latched).

Total: `reward = 20 * reached_reward * reached_status + reaching_reward * (1 - reached_status) + reached_status`. On success, the total is overwritten to the flat maximum **30.0**. `compute_normalized_dense_reward` divides by `max_reward = 30.0`.

**Composer:** SUM (gated additive terms via the `reached_status` latch; success replaces total with 30.0). Max reward = 30.0.

**Velocity-aware note.** The reward itself uses no velocity term (`ball_vel` appears only in the observation, §5). Shaping is purely positional + latch-gated. The rolling dynamics are driven implicitly by pushing the ball toward the goal.

**Decisions resolved.**
- `tcp_hit_pose = ball.p + normalize(ball.p - goal.p) * (ball_radius + 0.05)` = ball center + `0.085` along the ball-from-goal direction.
- Latch threshold: `dist(tcp, hit_pose) < 0.04` → `reached_status = 1.0` (sticky).
- tanh sharpness: 2 for the reaching term; 1 for the reached (place) term.
- Stage-B place weight = 20×; flat post-latch bonus = +1.
- Success override value = 30; `max_reward = 30.0`.
- Per-stage saturated per-step magnitudes (retro-computed): Stage A (not reached) max ≈ 1.0 (reaching_reward → 1). Stage B (reached) max ≈ `20 * 1.0 + 1.0 = 21.0` (when ball at goal but `evaluate` hasn't yet flagged success in the same step). Success flat = 30.0. The 30.0 override exceeds the un-success max (~21), so success is strictly rewarded above all shaping.

**Code (verbatim).**
```python
def compute_dense_reward(self, obs: Any, action: Array, info: dict):
    unit_vec = self.ball.pose.p - self.goal_region.pose.p
    unit_vec = unit_vec / torch.linalg.norm(unit_vec, axis=1, keepdim=True)
    tcp_hit_pose = Pose.create_from_pq(
        p=self.ball.pose.p + unit_vec * (self.ball_radius + 0.05),
    )
    tcp_to_hit_pose = tcp_hit_pose.p - self.agent.tcp.pose.p
    tcp_to_hit_pose_dist = torch.linalg.norm(tcp_to_hit_pose, axis=1)
    self.reached_status[tcp_to_hit_pose_dist < 0.04] = 1.0
    reaching_reward = 1 - torch.tanh(2 * tcp_to_hit_pose_dist)

    obj_to_goal_dist = torch.linalg.norm(
        self.ball.pose.p[..., :2] - self.goal_region.pose.p[..., :2], axis=1
    )

    reached_reward = 1 - torch.tanh(obj_to_goal_dist)

    reward = (
        20 * reached_reward * self.reached_status
        + reaching_reward * (1 - self.reached_status)
        + self.reached_status
    )

    reward[info["success"]] = 30.0
    return reward

def compute_normalized_dense_reward(self, obs: Any, action: Array, info: dict):
    max_reward = 30.0
    return self.compute_dense_reward(obs=obs, action=action, info=info) / max_reward
```

**Smoke.**
```bash
.venv/bin/python -c "import gymnasium as gym, mani_skill, torch; e=gym.make('RollBall-v1',reward_mode='dense'); e.reset(seed=0); o,r,te,tr,info=e.step(e.action_space.sample()); print(torch.isfinite(torch.as_tensor(r)).all().item()); e.close()"
# -> True
```

---

## §7 DR

`<no DR>`

The only stochasticity is episode-initialization randomization (§3): ball xy and goal xy uniform over the ranges above, plus `robot_init_qpos_noise=0.02`. There are no `startup` / `interval` domain-randomization events (no mass/friction/visual randomization). The `mani_skill.envs.utils.randomization` import is present in the file but unused for DR here. This matches ManiSkill's default for RollBall-v1.

---

