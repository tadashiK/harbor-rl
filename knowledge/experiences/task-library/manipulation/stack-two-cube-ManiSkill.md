# StackCube-v1 — Implementation Spec

- robot: Franka Panda with wrist camera (`panda_wristcam`; Panda / Fetch also supported)
- simulator: ManiSkill (SAPIEN, `BaseEnv` + `@register_env`)
- objects: two 4 cm cubes, table
- bimanual: false
- summary: Stack one cube on top of another.

ManiSkill maps to the Harbor §1..§7 schema as follows. A ManiSkill task is a single `BaseEnv` subclass — there is no separate per-robot `env_cfg`/`mdp/` tree. All design choices live in one file (`mani_skill/envs/tasks/tabletop/stack_cube.py`) plus shared base classes (`Panda` agent, `TableSceneBuilder`, `BaseEnv`). Code below is verbatim.

---

## §1 Registration + Scene

**Description.** Registers `StackCube-v1` with a 50-step episode cap. A Panda arm (wrist-cam variant by default) sits on a table at world x=-0.615. The scene adds a table workspace + ground (`TableSceneBuilder`) and two 4cm cubes built procedurally via `actors.build_cube`: `cubeA` red `[1,0,0,1]`, `cubeB` green `[0,1,0,1]`. No USD/mesh assets — cubes are primitive boxes (half_size 0.02); table mesh is `table.glb` shipped inside the package.

**Decisions resolved.**
- `id="StackCube-v1"`, `max_episode_steps=50`.
- `SUPPORTED_ROBOTS = ["panda_wristcam", "panda", "fetch"]`; default `robot_uids="panda_wristcam"`.
- `robot_init_qpos_noise=0.02` (Gaussian noise added to reset qpos, see §3).
- `cube_half_size = 0.02` (each cube 4cm), red cubeA / green cubeB.
- Robot base pose: `sapien.Pose(p=[-0.615, 0, 0])` (set both at load and at reset).
- Default sim config = `SimConfig()` (env does not override `_default_sim_config`).
- Sensors: `base_camera` 128×128 (look_at eye=[0.3,0,0.6] target=[-0.1,0,0.1]); human render `render_camera` 512×512.
- Cube initial poses at load are placeholders (`cubeA` p=[0,0,0.1], `cubeB` p=[1,0,0.1]); the real placement happens at reset (§3).

**Code.**
```python
@register_env("StackCube-v1", max_episode_steps=50)
class StackCubeEnv(BaseEnv):
    _sample_video_link = "https://github.com/haosulab/ManiSkill/raw/main/figures/environment_demos/StackCube-v1_rt.mp4"
    SUPPORTED_ROBOTS = ["panda_wristcam", "panda", "fetch"]
    agent: Union[Panda, Fetch]

    def __init__(
        self, *args, robot_uids="panda_wristcam", robot_init_qpos_noise=0.02, **kwargs
    ):
        self.robot_init_qpos_noise = robot_init_qpos_noise
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    @property
    def _default_sensor_configs(self):
        pose = sapien_utils.look_at(eye=[0.3, 0, 0.6], target=[-0.1, 0, 0.1])
        return [CameraConfig("base_camera", pose, 128, 128, np.pi / 2, 0.01, 100)]

    @property
    def _default_human_render_camera_configs(self):
        pose = sapien_utils.look_at([0.6, 0.7, 0.6], [0.0, 0.0, 0.35])
        return CameraConfig("render_camera", pose, 512, 512, 1, 0.01, 100)

    def _load_agent(self, options: dict):
        super()._load_agent(options, sapien.Pose(p=[-0.615, 0, 0]))

    def _load_scene(self, options: dict):
        self.cube_half_size = common.to_tensor([0.02] * 3, device=self.device)
        self.table_scene = TableSceneBuilder(
            env=self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.table_scene.build()
        self.cubeA = actors.build_cube(
            self.scene,
            half_size=0.02,
            color=[1, 0, 0, 1],
            name="cubeA",
            initial_pose=sapien.Pose(p=[0, 0, 0.1]),
        )
        self.cubeB = actors.build_cube(
            self.scene,
            half_size=0.02,
            color=[0, 1, 0, 1],
            name="cubeB",
            initial_pose=sapien.Pose(p=[1, 0, 0.1]),
        )
```

Shared agent constants (from `mani_skill/agents/robots/panda/panda.py`, inherited by `panda_wristcam`):
```python
arm_joint_names = ["panda_joint1", ..., "panda_joint7"]   # 7 arm DoF
gripper_joint_names = ["panda_finger_joint1", "panda_finger_joint2"]  # mimic'd
ee_link_name = "panda_hand_tcp"
arm_stiffness = 1e3; arm_damping = 1e2; arm_force_limit = 100
gripper_stiffness = 1e3; gripper_damping = 1e2; gripper_force_limit = 100
keyframes["rest"].qpos = [0.0, pi/8, 0, -5pi/8, 0, 3pi/4, pi/4, 0.04, 0.04]
# panda_wristcam reset qpos (TableSceneBuilder): [0.0, pi/8, 0, -5pi/8, 0, 3pi/4, -pi/4, 0.04, 0.04]
```

**Smoke (§1).** `cd <repo> && .venv/bin/python -c "import gymnasium as gym, mani_skill; e=gym.make('StackCube-v1'); print(e.observation_space, e.action_space); e.close()"`
Expected stdout: `Box(-inf, inf, (1, 48), float32) Box(-1.0, 1.0, (8,), float32)`

---

## §2 Actions

**Description.** Control is via the agent's `control_mode`. When `control_mode=None`, `BaseAgent` defaults to the first key of `Panda._controller_configs` → `pd_joint_delta_pos` (verified: `e.unwrapped.control_mode == "pd_joint_delta_pos"`). That is a 7-DoF per-joint delta-position controller (Δ∈[-0.1,0.1], use_delta=True, stiffness 1e3 / damping 1e2 / force_limit 100) plus a 1-DoF mimic gripper (`panda_finger_joint1` drives `panda_finger_joint2`; range [-0.01, 0.04]). Total action = 8, all normalized to [-1, 1] → `Box(-1, 1, (8,))`.

**Decisions resolved.**
- `control_mode = "pd_joint_delta_pos"` (resolved default).
- Action dim 8 = 7 arm joint deltas + 1 gripper.
- Arm delta bounds ±0.1 rad/step (`use_delta=True`); gripper mimic lower=-0.01 upper=0.04.

**Code (Panda `_controller_configs`, relevant entries).**
```python
arm_pd_joint_delta_pos = PDJointPosControllerConfig(
    self.arm_joint_names, lower=-0.1, upper=0.1,
    stiffness=self.arm_stiffness, damping=self.arm_damping,
    force_limit=self.arm_force_limit, use_delta=True,
)
gripper_pd_joint_pos = PDJointPosMimicControllerConfig(
    self.gripper_joint_names, lower=-0.01, upper=0.04,
    stiffness=self.gripper_stiffness, damping=self.gripper_damping,
    force_limit=self.gripper_force_limit,
    mimic={"panda_finger_joint2": {"joint": "panda_finger_joint1"}},
)
controller_configs = dict(
    pd_joint_delta_pos=dict(arm=arm_pd_joint_delta_pos, gripper=gripper_pd_joint_pos),
    pd_joint_pos=dict(arm=arm_pd_joint_pos, gripper=gripper_pd_joint_pos),
    pd_ee_delta_pos=dict(arm=arm_pd_ee_delta_pos, gripper=gripper_pd_joint_pos),
    pd_ee_delta_pose=dict(arm=arm_pd_ee_delta_pose, gripper=gripper_pd_joint_pos),
    ...  # also pd_ee_pose, pd_joint_*_target_delta_*, pd_joint_vel, pd_joint_pos_vel
)
```

**Smoke (§2).** action_space is `Box(-1.0, 1.0, (8,), float32)` (from §1 build). Reproduce: `e.action_space.shape == (8,)`.

---

## §3 Reset

**Description.** On `_initialize_episode`, the table scene resets the robot to its keyframe qpos plus Gaussian noise (σ=0.02), gripper forced open (0.04). Both cubes are placed on the table: a shared base xy is drawn uniformly in [-0.1, 0.1]², then each cube gets an independent non-colliding offset from a `UniformPlacementSampler` over region [[-0.1,-0.2],[0.1,0.2]] with collision radius `‖(0.02,0.02)‖ + 0.001`. Each cube's z-axis (yaw) rotation is randomized (`random_quaternions` with lock_x=lock_y=True, lock_z=False); roll/pitch locked. Cube z=0.02 (resting on table).

**Decisions resolved.**
- Robot reset: keyframe qpos + `normal(0, 0.02)`, fingers set to 0.04, base pose `[-0.615, 0, 0]`.
- Cube xy base: `rand(b,2)*0.2 - 0.1`.
- Per-cube placement offset: `UniformPlacementSampler(bounds=[[-0.1,-0.2],[0.1,0.2]])` sampled with `radius=‖(0.02,0.02)‖+0.001`, 100 candidate retries; cubeB sampled `verbose=False`.
- Cube yaw randomized (z free), roll/pitch locked; z height = 0.02.

**Code.**
```python
def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
    with torch.device(self.device):
        b = len(env_idx)
        self.table_scene.initialize(env_idx)

        xyz = torch.zeros((b, 3))
        xyz[:, 2] = 0.02
        xy = torch.rand((b, 2)) * 0.2 - 0.1
        region = [[-0.1, -0.2], [0.1, 0.2]]
        sampler = randomization.UniformPlacementSampler(
            bounds=region, batch_size=b, device=self.device
        )
        radius = torch.linalg.norm(torch.tensor([0.02, 0.02])) + 0.001
        cubeA_xy = xy + sampler.sample(radius, 100)
        cubeB_xy = xy + sampler.sample(radius, 100, verbose=False)

        xyz[:, :2] = cubeA_xy
        qs = randomization.random_quaternions(b, lock_x=True, lock_y=True, lock_z=False)
        self.cubeA.set_pose(Pose.create_from_pq(p=xyz.clone(), q=qs))

        xyz[:, :2] = cubeB_xy
        qs = randomization.random_quaternions(b, lock_x=True, lock_y=True, lock_z=False)
        self.cubeB.set_pose(Pose.create_from_pq(p=xyz, q=qs))
```

Robot reset (from `TableSceneBuilder.initialize`, `panda_wristcam` branch):
```python
qpos = np.array([0.0, np.pi/8, 0, -np.pi*5/8, 0, np.pi*3/4, -np.pi/4, 0.04, 0.04])
qpos = self.env._episode_rng.normal(0, self.robot_init_qpos_noise, (b, len(qpos))) + qpos
qpos[:, -2:] = 0.04
self.env.agent.reset(qpos)
self.env.agent.robot.set_pose(sapien.Pose([-0.615, 0, 0]))
```

**Smoke (§3).** `e.reset(seed=0)` twice with different seeds → cube poses differ; `e.reset(seed=0)` twice → identical.

---

## §4 Goal + Termination

**Description.** Success (computed every step in `evaluate()`) requires three conjuncts: (1) cubeA is on cubeB — xy offset within `‖half_size_xy‖+0.005` AND z offset within 0.005 of one cube edge (`2*half_size` = 0.04); (2) cubeA is static (lin<1e-2, ang<0.5 — ang threshold loosened for GPU sim instability); (3) cubeA is NOT grasped (robot let go). Episode terminates on `success=True` or truncates at `max_episode_steps=50`. There is no failure-termination term.

**Decisions resolved.**
- xy success tol: `‖cube_half_size[:2]‖ + 0.005`.
- z success tol: `|offset_z − 2*half_size| ≤ 0.005` (i.e. red exactly one cube-height above green).
- static thresholds: `lin_thresh=1e-2, ang_thresh=0.5`.
- ungrasped required (`~is_cubeA_grasped`).
- horizon: 50 steps (truncation); terminate on success.

**Code.**
```python
def evaluate(self):
    pos_A = self.cubeA.pose.p
    pos_B = self.cubeB.pose.p
    offset = pos_A - pos_B
    xy_flag = (
        torch.linalg.norm(offset[..., :2], axis=1)
        <= torch.linalg.norm(self.cube_half_size[:2]) + 0.005
    )
    z_flag = torch.abs(offset[..., 2] - self.cube_half_size[..., 2] * 2) <= 0.005
    is_cubeA_on_cubeB = torch.logical_and(xy_flag, z_flag)
    is_cubeA_static = self.cubeA.is_static(lin_thresh=1e-2, ang_thresh=0.5)
    is_cubeA_grasped = self.agent.is_grasping(self.cubeA)
    success = is_cubeA_on_cubeB * is_cubeA_static * (~is_cubeA_grasped)
    return {
        "is_cubeA_grasped": is_cubeA_grasped,
        "is_cubeA_on_cubeB": is_cubeA_on_cubeB,
        "is_cubeA_static": is_cubeA_static,
        "success": success.bool(),
    }
```

**Smoke (§4).** `obs, info = e.reset(); info["success"]` is a bool tensor, initially `False`. Step random actions for 50 steps → env truncates.

---

## §5 Observation

**Description.** obs_mode defaults to `state` (first of `SUPPORTED_OBS_MODES = ("state", "state_dict", "none", "sensor_data", "any_textures", "pointcloud")`). The flattened state vector = agent proprioception (qpos 9 + qvel 9 = 18) ⊕ task extras from `_get_obs_extra`. In `state`/`state_dict` mode, extras = `tcp_pose(7) + cubeA_pose(7) + cubeB_pose(7) + tcp_to_cubeA_pos(3) + tcp_to_cubeB_pos(3) + cubeA_to_cubeB_pos(3) = 30`. Total = 18 + 30 = **48** (verified build: `Box(-inf, inf, (1, 48))`). No observation noise (state mode).

**Decisions resolved.**
- obs_mode default = `state`.
- proprio: qpos(9) + qvel(9) = 18 (controller adds no extra state for `pd_joint_delta_pos`).
- extras in state mode: tcp_pose 7, cubeA_pose 7, cubeB_pose 7, tcp_to_cubeA_pos 3, tcp_to_cubeB_pos 3, cubeA_to_cubeB_pos 3.
- **Total obs dim = 48.**
- In non-state modes (e.g. `none`/`sensor_data`) only `tcp_pose` (7) is exposed from extras.

**Code.**
```python
def _get_obs_extra(self, info: dict):
    obs = dict(tcp_pose=self.agent.tcp.pose.raw_pose)
    if "state" in self.obs_mode:
        obs.update(
            cubeA_pose=self.cubeA.pose.raw_pose,
            cubeB_pose=self.cubeB.pose.raw_pose,
            tcp_to_cubeA_pos=self.cubeA.pose.p - self.agent.tcp.pose.p,
            tcp_to_cubeB_pos=self.cubeB.pose.p - self.agent.tcp.pose.p,
            cubeA_to_cubeB_pos=self.cubeB.pose.p - self.cubeA.pose.p,
        )
    return obs
```
Proprio (shared `BaseAgent.get_proprioception`):
```python
obs = dict(qpos=self.robot.get_qpos(), qvel=self.robot.get_qvel())
controller_state = self.controller.get_state()   # empty for pd_joint_delta_pos
```

**Smoke (§5).** observation_space is `Box(-inf, inf, (1, 48), float32)` (from §1 build). Reproduce: `e.observation_space.shape[-1] == 48`.

---

## §6 Reward

**Description.** Dense reward is a **staged max-of-stages** composer (NOT a weighted sum): a base reach reward, then in-place overwrite of grasped envs, then in-place overwrite of stacked envs, then a flat success value. Stages (monotonically increasing magnitude band):
1. **reach** (always): `2*(1 - tanh(5*d_tcp_to_cubeA))` ∈ [0, 2].
2. **grasp+place** (`is_cubeA_grasped`): `4 + place_reward`, where `place_reward = 1 - tanh(5*d_cubeA_to_goal)` ∈ [0,1]; goal = cubeB xy at z+0.04. Band [4, 5].
3. **ungrasp+static** (`is_cubeA_on_cubeB`): `6 + (ungrasp_reward + static_reward)/2`; `ungrasp_reward = sum(last-2 qpos)/gripper_width` (=1 when ungrasped), `static_reward = 1 - tanh(10*v + av)`. Band [6, 7].
4. **success**: flat `8`.

The composer is **overwrite-by-mask** (later stages mask out earlier ones for the same env), so it is neither a pure sum nor a pure product — it is a piecewise/staged select. `compute_normalized_dense_reward = compute_dense_reward / 8`.

**Decisions resolved (per-stage saturated per-step magnitudes, retro-computed from code).**
- reach: max 2.0 (tanh slope 5).
- grasp+place: 4.0 base + up to 1.0 place → 5.0.
- on-cubeB stage: 6.0 base + up to 1.0 (mean of ungrasp∈[0,1] + static∈[0,1]) → 7.0.
- success: 8.0 flat.
- Normalization divisor: 8.

**Code (verbatim, both functions).**
```python
def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: dict):
    # reaching reward
    tcp_pose = self.agent.tcp.pose.p
    cubeA_pos = self.cubeA.pose.p
    cubeA_to_tcp_dist = torch.linalg.norm(tcp_pose - cubeA_pos, axis=1)
    reward = 2 * (1 - torch.tanh(5 * cubeA_to_tcp_dist))

    # grasp and place reward
    cubeA_pos = self.cubeA.pose.p
    cubeB_pos = self.cubeB.pose.p
    goal_xyz = torch.hstack(
        [cubeB_pos[:, 0:2], (cubeB_pos[:, 2] + self.cube_half_size[2] * 2)[:, None]]
    )
    cubeA_to_goal_dist = torch.linalg.norm(goal_xyz - cubeA_pos, axis=1)
    place_reward = 1 - torch.tanh(5.0 * cubeA_to_goal_dist)

    reward[info["is_cubeA_grasped"]] = (4 + place_reward)[info["is_cubeA_grasped"]]

    # ungrasp and static reward
    gripper_width = (self.agent.robot.get_qlimits()[0, -1, 1] * 2).to(
        self.device
    )  # NOTE: hard-coded with panda
    is_cubeA_grasped = info["is_cubeA_grasped"]
    ungrasp_reward = (
        torch.sum(self.agent.robot.get_qpos()[:, -2:], axis=1) / gripper_width
    )
    ungrasp_reward[~is_cubeA_grasped] = 1.0
    v = torch.linalg.norm(self.cubeA.linear_velocity, axis=1)
    av = torch.linalg.norm(self.cubeA.angular_velocity, axis=1)
    static_reward = 1 - torch.tanh(v * 10 + av)
    reward[info["is_cubeA_on_cubeB"]] = (
        6 + (ungrasp_reward + static_reward) / 2.0
    )[info["is_cubeA_on_cubeB"]]

    reward[info["success"]] = 8

    return reward

def compute_normalized_dense_reward(self, obs: Any, action: torch.Tensor, info: dict):
    return self.compute_dense_reward(obs=obs, action=action, info=info) / 8
```

**Smoke (§6).** `obs, r, term, trunc, info = e.step(e.action_space.sample())` → `r` finite, in [0, 8] (dense) or [0, 1] (normalized via `reward_mode="normalized_dense"`). Non-constant across a rollout.

---

## §7 DR

**Description.** No startup/interval domain randomization. All randomization is reset-time only (§3): cube xy positions, cube yaw, and robot init qpos noise (σ=0.02). There are no physics-material / mass / friction / lighting randomizers applied at startup or on an interval.

`<no DR>` (reset-time randomization only — already captured in §3).

**Smoke (§7).** N/A. (No DR to toggle; reset-time randomization verified by §3 smoke.)

---

