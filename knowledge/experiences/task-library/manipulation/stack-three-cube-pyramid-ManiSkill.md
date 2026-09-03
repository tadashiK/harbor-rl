# StackPyramid-v1 — Implementation Spec

- robot: Franka Panda with wrist camera (`panda_wristcam`; Panda / Fetch also supported)
- simulator: ManiSkill (SAPIEN, `BaseEnv` + `@register_env`)
- objects: three cubes, table
- bimanual: false
- summary: Arrange three cubes into a pyramid.

ManiSkill maps onto the §1..§7 design-choice schema as follows. There is no IsaacLab-style `*Cfg` manager tree; every section is a method on the `BaseEnv` subclass. All section code is pasted verbatim from `mani_skill/envs/tasks/tabletop/stack_pyramid.py` unless noted.

---

## §1 Registration + Scene

### Description
A SAPIEN tabletop manipulation task. A single `@register_env("StackPyramid-v1", max_episode_steps=250)` decorator registers the env. `_load_scene` builds the standard `TableSceneBuilder` (table + ground + lights + robot) and three 4 cm cubes (A=red, B=green, C=blue), each a primitive `build_cube`. The pyramid: red (A) is placed next to green (B), blue (C) is stacked on top spanning both. The robot is loaded by the framework from `robot_uids` (default `panda_wristcam`) — there is no explicit `_load_agent` override; `BaseEnv._load_agent` instantiates the agent from `SUPPORTED_ROBOTS`.

### Decisions resolved
- `id = "StackPyramid-v1"`, `max_episode_steps = 250`.
- `SUPPORTED_ROBOTS = ["panda_wristcam", "panda", "fetch"]`; default `robot_uids="panda_wristcam"`.
- `robot_init_qpos_noise = 0.02` (passed into `TableSceneBuilder`).
- Cubes: primitive boxes, `half_size=0.02` (4 cm side), colors RGBA A=`[1,0,0,1]`, B=`[0,1,0,1]`, C=`[0,0,1,1]`. Initial (pre-episode) poses A `p=[0,0,0.2]`, B `p=[1,0,0.2]`, C `p=[-1,0,0.2]` (spread apart so build-time collision is avoided; real placement happens in §3).
- `self.cube_half_size = common.to_tensor([0.02]*3)` (used by `evaluate`).
- Sim config: no `_default_sim_config` override → inherits `BaseEnv` defaults (sim_freq=100 Hz, control_freq=20 Hz).
- Sensors: `base_camera` 128×128 at `look_at(eye=[0.3,0,0.4], target=[-0.05,0,0.1])`; human render cam `render_camera` 512×512 at `look_at([0.6,0.7,0.6],[0,0,0.35])`.

### Code
```python
@register_env("StackPyramid-v1", max_episode_steps=250)
class StackPyramidEnv(BaseEnv):
    SUPPORTED_ROBOTS = ["panda_wristcam", "panda", "fetch"]
    SUPPORTED_REWARD_MODES = ["none", "sparse"]

    agent: Union[Panda, Fetch]

    def __init__(
        self, *args, robot_uids="panda_wristcam", robot_init_qpos_noise=0.02, **kwargs
    ):
        self.robot_init_qpos_noise = robot_init_qpos_noise
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    @property
    def _default_sensor_configs(self):
        pose = sapien_utils.look_at(eye=[0.3, 0, 0.4], target=[-0.05, 0, 0.1])
        return [CameraConfig("base_camera", pose, 128, 128, np.pi / 2, 0.01, 100)]

    @property
    def _default_human_render_camera_configs(self):
        pose = sapien_utils.look_at([0.6, 0.7, 0.6], [0.0, 0.0, 0.35])
        return CameraConfig("render_camera", pose, 512, 512, 1, 0.01, 100)

    def _load_scene(self, options: dict):
        self.cube_half_size = common.to_tensor([0.02] * 3)
        self.table_scene = TableSceneBuilder(
            env=self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.table_scene.build()
        self.cubeA = actors.build_cube(
            self.scene,
            half_size=0.02,
            color=[1, 0, 0, 1],
            name="cubeA",
            initial_pose=sapien.Pose(p=[0, 0, 0.2]),
        )
        self.cubeB = actors.build_cube(
            self.scene,
            half_size=0.02,
            color=[0, 1, 0, 1],
            name="cubeB",
            initial_pose=sapien.Pose(p=[1, 0, 0.2]),
        )
        self.cubeC = actors.build_cube(
            self.scene,
            half_size=0.02,
            color=[0, 0, 1, 1],
            name="cubeC",
            initial_pose=sapien.Pose(p=[-1, 0, 0.2]),
        )
```

### Smoke
```bash
cd <repo>
.venv/bin/python -c "import gymnasium as gym, mani_skill; e=gym.make('StackPyramid-v1'); print(e.observation_space, e.action_space); e.close()"
# Box(-inf, inf, (1, 64), float32) Box(-1.0, 1.0, (8,), float32)
```

---

## §2 Actions

### Description
The env defines no controller — it uses the agent's controller configs. The default `control_mode` is the first key registered in `Panda._controller_configs`, namely `pd_joint_delta_pos`: 7 arm joints in delta-position mode (clamp ±0.1 rad per step) + a 1-DoF mimic gripper (`panda_finger_joint1` mirrored to `joint2`), giving an 8-D action space normalized to `[-1, 1]`.

### Decisions resolved
- `control_mode = "pd_joint_delta_pos"` (default; first key of `controller_configs`).
- Arm: `PDJointPosControllerConfig(arm_joint_names, lower=-0.1, upper=0.1, use_delta=True)` over the 7 Panda arm joints → 7 dims.
- Gripper: `PDJointPosMimicControllerConfig(gripper_joint_names, lower=-0.01, upper=0.04, mimic={"panda_finger_joint2": {"joint":"panda_finger_joint1"}})` → 1 dim.
- Total action dim = 8, `Box(-1.0, 1.0, (8,))`.
- Other available modes (not default): `pd_joint_pos`, `pd_ee_delta_pos`, `pd_ee_delta_pose`, `pd_ee_pose`, target variants, `pd_joint_vel`, `pd_joint_pos_vel`, `pd_joint_delta_pos_vel`.

### Code
(from `mani_skill/agents/robots/panda/panda.py`; StackPyramid adds nothing.)
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
    ...
)
```

### Smoke
```bash
.venv/bin/python -c "import gymnasium as gym, mani_skill; e=gym.make('StackPyramid-v1'); print(e.action_space)"
# Box(-1.0, 1.0, (8,), float32)
```

---

## §3 Reset

### Description
`_initialize_episode` resets the table scene, then samples non-colliding xy positions for all three cubes within a centered region using `UniformPlacementSampler`, and applies a random z-axis-only rotation to each cube. Cubes are placed flat on the table (z = 0.02 = half-size). The `robot_init_qpos_noise=0.02` perturbation of the robot's initial qpos is applied inside `table_scene.initialize`.

### Decisions resolved
- Placement region (xy bounds): `[[-0.1, -0.2], [0.1, 0.2]]` centered at origin.
- Sampler: `randomization.UniformPlacementSampler(bounds=region, batch_size=b)`, exclusion `radius = ||[0.02, 0.02]|| ≈ 0.0283`, 100 candidate samples each — ensures cubes don't collide.
- Cube z = 0.02 (resting on table top).
- Orientation: `random_quaternions(b, lock_x=True, lock_y=True, lock_z=False)` — yaw-only randomization, identical scheme for all three cubes.
- Robot: init qpos noise 0.02 via `TableSceneBuilder`.

### Code
```python
def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
    with torch.device(self.device):
        b = len(env_idx)
        self.table_scene.initialize(env_idx)

        xyz = torch.zeros((b, 3), device=self.device)
        xyz[:, 2] = 0.02
        xy = xyz[:, :2]
        region = [[-0.1, -0.2], [0.1, 0.2]]
        sampler = randomization.UniformPlacementSampler(
            bounds=region, batch_size=b, device=self.device
        )
        radius = torch.linalg.norm(torch.tensor([0.02, 0.02]))
        cubeA_xy = xy + sampler.sample(radius, 100)
        cubeB_xy = xy + sampler.sample(radius, 100, verbose=False)
        cubeC_xy = xy + sampler.sample(radius, 100, verbose=False)

        # Cube A
        xyz[:, :2] = cubeA_xy
        qs = randomization.random_quaternions(b, lock_x=True, lock_y=True, lock_z=False)
        self.cubeA.set_pose(Pose.create_from_pq(p=xyz.clone(), q=qs))

        # Cube B
        xyz[:, :2] = cubeB_xy
        qs = randomization.random_quaternions(b, lock_x=True, lock_y=True, lock_z=False)
        self.cubeB.set_pose(Pose.create_from_pq(p=xyz.clone(), q=qs))

        # Cube C
        xyz[:, :2] = cubeC_xy
        qs = randomization.random_quaternions(b, lock_x=True, lock_y=True, lock_z=False)
        self.cubeC.set_pose(Pose.create_from_pq(p=xyz, q=qs))
```

### Smoke
```bash
.venv/bin/python -c "import gymnasium as gym, mani_skill; e=gym.make('StackPyramid-v1', num_envs=4); e.reset(seed=0); print('reset ok'); e.close()"
# reset ok
```

---

## §4 Goal + Termination

### Description
`evaluate()` returns the success flag; `max_episode_steps=250` (set in the `@register_env` decorator) provides the time-out truncation. There are no explicit failure terms. Pyramid success requires THREE simultaneous geometric/state conditions, each gated by per-cube static + not-grasped checks:
1. A (red) is **next_to** B (green): xy-offset ≤ `||2·half_size[:2]|| + 0.005`.
2. C (blue) is **on top of** B (green): xy within the same tolerance AND |z-offset| > 0.02.
3. C (blue) is **on top of** A (red): same top condition.
Each pairwise check additionally requires the *first-named* cube to be static (`lin_thresh=1e-2, ang_thresh=0.5`) and NOT grasped by the agent. Overall success = AND of all three.

### Decisions resolved
- `max_episode_steps = 250` (truncation only; no `fail`/`out_of_bounds` term).
- xy tolerance for "aligned": `||2·cube_half_size[:2]|| + 0.005 = ||[0.04,0.04]|| + 0.005 ≈ 0.0616 m`.
- z separation for "on top": `|Δz| > 0.02 m`.
- Static thresholds: `lin_thresh=1e-2`, `ang_thresh=0.5`.
- Not-grasped: `~self.agent.is_grasping(cube)`.
- `info["success"]` is the only returned key (no `fail`).

### Code
```python
def evaluate(self):
    pos_A = self.cubeA.pose.p
    pos_B = self.cubeB.pose.p
    pos_C = self.cubeC.pose.p

    offset_AB = pos_A - pos_B
    offset_BC = pos_B - pos_C
    offset_AC = pos_A - pos_C

    def evaluate_cube_distance(offset, cube_a, cube_b, top_or_next):
        xy_flag = (
            torch.linalg.norm(offset[..., :2], axis=1)
            <= torch.linalg.norm(2 * self.cube_half_size[:2]) + 0.005
        )
        z_flag = torch.abs(offset[..., 2]) > 0.02
        if top_or_next == "top":
            is_cubeA_on_cubeB = torch.logical_and(xy_flag, z_flag)
        elif top_or_next == "next_to":
            is_cubeA_on_cubeB = xy_flag
        else:
            return NotImplementedError(
                f"Expect top_or_next to be either 'top' or 'next_to', got {top_or_next}"
            )
        is_cubeA_static = cube_a.is_static(lin_thresh=1e-2, ang_thresh=0.5)
        is_cubeA_grasped = self.agent.is_grasping(cube_a)
        success = is_cubeA_on_cubeB & is_cubeA_static & (~is_cubeA_grasped)
        return success.bool()

    success_A_B = evaluate_cube_distance(offset_AB, self.cubeA, self.cubeB, "next_to")
    success_C_B = evaluate_cube_distance(offset_BC, self.cubeC, self.cubeB, "top")
    success_C_A = evaluate_cube_distance(offset_AC, self.cubeC, self.cubeA, "top")
    success = torch.logical_and(
        success_A_B, torch.logical_and(success_C_B, success_C_A)
    )
    return {"success": success}
```

### Smoke
```bash
.venv/bin/python -c "import gymnasium as gym, mani_skill; e=gym.make('StackPyramid-v1', num_envs=2); e.reset(seed=0); o,r,term,trunc,info=e.step(e.action_space.sample()); print('success' in info, info['success'].shape); e.close()"
# True torch.Size([2])
```

---

## §5 Observation

### Description
The policy observation (`obs_mode="state"` default flat-state) concatenates the agent proprioception (`get_proprioception`: qpos/qvel/controller state) with `_get_obs_extra`. `_get_obs_extra` always exposes `tcp_pose` (7-D raw pose); under any `state` mode it additionally exposes the three cube raw poses (7-D each) and six relative-position vectors (3-D each: tcp→each cube, plus A→B, B→C, A→C). The framework flattens everything to a single vector. Resolved total dim = 64.

### Decisions resolved
- `obs_mode` default = `state` → flat `Box(-inf, inf, (1, 64), float32)` (leading dim is num_envs=1).
- Extra terms (state mode): `tcp_pose` (7), `cubeA_pose` (7), `cubeB_pose` (7), `cubeC_pose` (7), `tcp_to_cubeA_pos` (3), `tcp_to_cubeB_pos` (3), `tcp_to_cubeC_pos` (3), `cubeA_to_cubeB_pos` (3), `cubeB_to_cubeC_pos` (3), `cubeA_to_cubeC_pos` (3) = 46 dims of extra.
- Remaining 18 dims come from Panda proprioception (qpos 9 + qvel 9 under `pd_joint_delta_pos`). 46 + 18 = 64. ✓
- Other modes available: `state_dict`, `rgb`, `rgbd`, `pointcloud`, `sensor_data` (framework-provided; not task-specific).

### Code
```python
def _get_obs_extra(self, info: dict):
    obs = dict(tcp_pose=self.agent.tcp.pose.raw_pose)
    if "state" in self.obs_mode:
        obs.update(
            cubeA_pose=self.cubeA.pose.raw_pose,
            cubeB_pose=self.cubeB.pose.raw_pose,
            cubeC_pose=self.cubeC.pose.raw_pose,
            tcp_to_cubeA_pos=self.cubeA.pose.p - self.agent.tcp.pose.p,
            tcp_to_cubeB_pos=self.cubeB.pose.p - self.agent.tcp.pose.p,
            tcp_to_cubeC_pos=self.cubeC.pose.p - self.agent.tcp.pose.p,
            cubeA_to_cubeB_pos=self.cubeB.pose.p - self.cubeA.pose.p,
            cubeB_to_cubeC_pos=self.cubeC.pose.p - self.cubeB.pose.p,
            cubeA_to_cubeC_pos=self.cubeC.pose.p - self.cubeA.pose.p,
        )
    return obs
```

### Smoke
```bash
.venv/bin/python -c "import gymnasium as gym, mani_skill; e=gym.make('StackPyramid-v1'); print(e.observation_space); e.close()"
# Box(-inf, inf, (1, 64), float32)
```

---

## §6 Reward

### Description
**StackPyramid-v1 ships NO dense reward.** `SUPPORTED_REWARD_MODES = ["none", "sparse"]` and the class overrides neither `compute_dense_reward` nor `compute_normalized_dense_reward`. Therefore:
- Default reward mode (when `reward_mode` is unspecified) resolves to `SUPPORTED_REWARD_MODES[0]` = `"none"` → reward is always 0.
- With `reward_mode="sparse"`, the env uses the **base-class** `BaseEnv.compute_sparse_reward`: +1 on success (`info["success"]`), 0 otherwise (no `fail` key in this task, so the `-fail` branch never triggers).
- Requesting `reward_mode="dense"` / `"normalized_dense"` raises (not in `SUPPORTED_REWARD_MODES`).

**Composer:** sparse / scalar — a single term, no staging. (A staged dense reward would have to be authored from scratch; the canonical task has none.)

### Decisions resolved
- `SUPPORTED_REWARD_MODES = ["none", "sparse"]`.
- Default mode = `"none"` → constant 0.
- Sparse reward = `info["success"].float()` (1.0 on full pyramid success, else 0.0).
- No per-stage planning budget (no dense terms exist). Retro-note: a downstream dense reward would naturally stage as reach→grasp A→place A next-to B→reach/grasp C→place C atop A&B→release, mirroring the three-condition `evaluate()`.

### Code
StackPyramid defines no reward method. Inherited verbatim from `mani_skill/envs/sapien_env.py`:
```python
def compute_sparse_reward(self, obs: Any, action: torch.Tensor, info: dict):
    if "success" in info:
        if "fail" in info:
            if isinstance(info["success"], torch.Tensor):
                reward = info["success"].to(torch.float) - info["fail"].to(torch.float)
            else:
                reward = info["success"] - info["fail"]
        else:
            reward = info["success"]
    else:
        if "fail" in info:
            reward = -info["fail"]
        else:
            reward = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
    return reward
```
(`get_reward` with `reward_mode="none"` returns `torch.zeros((num_envs,))`.)

### Smoke
```bash
.venv/bin/python -c "import gymnasium as gym, mani_skill; e=gym.make('StackPyramid-v1', num_envs=2, reward_mode='sparse'); e.reset(seed=0); o,r,term,trunc,info=e.step(e.action_space.sample()); print(r.shape, float(r.sum())); e.close()"
# torch.Size([2]) 0.0   (sparse: 0 until full pyramid success)
```

---

## §7 DR

### Description
Init-only randomization, all inside §3 `_initialize_episode` (there is no startup/interval event system in ManiSkill — randomization is procedural per-reset):
- Per-cube xy position via `UniformPlacementSampler` over region `[[-0.1,-0.2],[0.1,0.2]]` with collision-avoidance radius ≈ 0.0283 m.
- Per-cube yaw (z-axis) rotation via `random_quaternions(..., lock_x=True, lock_y=True, lock_z=False)`.
- Robot init qpos noise `robot_init_qpos_noise = 0.02` applied in `TableSceneBuilder.initialize`.

No ongoing (interval/startup) domain randomization — no dynamics/friction/mass/visual DR. This is the standard ManiSkill init-only scheme; reproduced fully by §3 above.

### Code
See §3 (`UniformPlacementSampler` + `random_quaternions` + `robot_init_qpos_noise`). No separate DR block.

### Smoke
```bash
.venv/bin/python -c "import gymnasium as gym, mani_skill; e=gym.make('StackPyramid-v1', num_envs=1); e.reset(seed=0); a=e.unwrapped.cubeA.pose.p.clone(); e.reset(seed=1); b=e.unwrapped.cubeA.pose.p; print(bool((a-b).abs().sum()>1e-4)); e.close()"
# True   (different seeds → different cube placement)
```

---

## Reproduce
```
/harbor:task-create name=<NewTaskID> from=<ManiSkill-repo>/harbor/create-task/stackpyramid-v1-implementation.md
```
