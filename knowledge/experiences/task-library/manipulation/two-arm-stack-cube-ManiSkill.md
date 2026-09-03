# TwoRobotStackCube-v1 — Implementation Spec

- robot: Two Franka Panda arms with wrist cameras (`panda_wristcam` x2, multi-agent)
- simulator: ManiSkill (SAPIEN, `BaseEnv` + `@register_env`)
- objects: blue cubeA, green cubeB, target marker, table
- bimanual: true
- summary: Two arms cooperate to stack one cube on another at a target.

> Family note: ManiSkill tasks subclass `mani_skill.envs.sapien_env.BaseEnv` and self-register via `@register_env(id, max_episode_steps=...)`. There is **no** IsaacLab manager-based `*Cfg` split (no `ActionsCfg`/`ObservationsCfg`/`RewardsCfg`/`EventCfg`). All seven sections map onto `BaseEnv` method overrides:
> - §1 → `@register_env` + `SUPPORTED_ROBOTS` + `_load_agent` + `_load_scene` + `_default_sim_config`/`_default_sensor_configs`
> - §2 → `control_mode` (default-resolved, per-robot Panda controllers) + combined `MultiAgent` Dict action space
> - §3 → `_initialize_episode`
> - §4 → `evaluate()` + `max_episode_steps`
> - §5 → `_get_obs_extra` + obs modes
> - §6 → `compute_dense_reward` + `compute_normalized_dense_reward`
> - §7 → domain randomization (none beyond per-reset spawn randomization here)

---

## §1 Registration + Scene

**Description.** Cooperative two-arm tabletop stacking. A `TableSceneBuilder` lays a table (top at z=0) and instantiates the two Pandas. Two 0.02-half-size cubes (blue `cubeA`, green `cubeB`) and a kinematic red/white circular target region (`goal_region`, radius 0.06) are added. Sim memory config is enlarged for the two-robot contact load. Note the color/name mapping is deliberately crossed: `cubeA` is **blue** (RGB `[12,42,160]/255`), `cubeB` is **green** (`[0,1,0,1]`) — the docstring's "green cube near right robot / blue cube near left robot" refers to which arm reaches which, see §3.

**Decisions resolved.**
- `max_episode_steps = 100` (from `@register_env`).
- `SUPPORTED_ROBOTS = [("panda_wristcam", "panda_wristcam")]`; default `robot_uids=("panda_wristcam","panda_wristcam")`.
- Robot base placement (via `_load_agent`): `[sapien.Pose(p=[0,-1,0]), sapien.Pose(p=[0,1,0])]` → first agent (`agents[0]`, "left") at y=+1, second (`agents[1]`, "right") at y=−1. (The poses passed are in list order; the env comments label `agents[0]` left.)
- `robot_init_qpos_noise = 0.02`.
- `cube_half_size = 0.02` (3-vector). cubeA color `[12,42,160,255]/255` (blue), cubeB color `[0,1,0,1]` (green). Placeholder initial poses `cubeA p=[1,0,0.02]`, `cubeB p=[-1,0,0.02]` (overwritten in §3).
- `goal_radius = 0.06`; goal_region built via `actors.build_red_white_target(radius=0.06, thickness=1e-5, add_collision=False, body_type="kinematic")`.
- Sim cfg: `GPUMemoryConfig(found_lost_pairs_capacity=2**25, max_rigid_patch_count=2**19, max_rigid_contact_count=2**21)`.
- Sensors: base_camera `look_at(eye=[0.3,0,0.6], target=[-0.1,0,0.1])`, 128×128, fov π/2. Render camera `look_at(eye=[0.6,0.2,0.4], target=[-0.1,0,0.1])`, 512×512, fov 1.
- Each `panda_wristcam` also mounts a `hand_camera` (128×128, fov π/2) on `camera_link`.

**Code.**
```python
@register_env("TwoRobotStackCube-v1", max_episode_steps=100)
class TwoRobotStackCube(BaseEnv):
    _sample_video_link = "https://github.com/haosulab/ManiSkill/raw/main/figures/environment_demos/TwoRobotStackCube-v1_rt.mp4"
    SUPPORTED_ROBOTS = [("panda_wristcam", "panda_wristcam")]
    agent: MultiAgent[Tuple[Panda, Panda]]

    goal_radius = 0.06

    def __init__(self, *args, robot_uids=("panda_wristcam", "panda_wristcam"),
                 robot_init_qpos_noise=0.02, **kwargs):
        self.robot_init_qpos_noise = robot_init_qpos_noise
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    @property
    def _default_sim_config(self):
        return SimConfig(
            gpu_memory_config=GPUMemoryConfig(
                found_lost_pairs_capacity=2**25,
                max_rigid_patch_count=2**19,
                max_rigid_contact_count=2**21,
            )
        )

    @property
    def _default_sensor_configs(self):
        pose = sapien_utils.look_at(eye=[0.3, 0, 0.6], target=[-0.1, 0, 0.1])
        return [CameraConfig("base_camera", pose, 128, 128, np.pi / 2, 0.01, 100)]

    @property
    def _default_human_render_camera_configs(self):
        pose = sapien_utils.look_at(eye=[0.6, 0.2, 0.4], target=[-0.1, 0, 0.1])
        return CameraConfig("render_camera", pose, 512, 512, 1, 0.01, 100)

    def _load_agent(self, options: dict):
        super()._load_agent(
            options, [sapien.Pose(p=[0, -1, 0]), sapien.Pose(p=[0, 1, 0])]
        )

    def _load_scene(self, options: dict):
        self.cube_half_size = common.to_tensor([0.02] * 3, device=self.device)
        self.table_scene = TableSceneBuilder(
            env=self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.table_scene.build()
        self.cubeA = actors.build_cube(
            self.scene, half_size=0.02,
            color=np.array([12, 42, 160, 255]) / 255,
            name="cubeA", initial_pose=sapien.Pose(p=[1, 0, 0.02]),
        )
        self.cubeB = actors.build_cube(
            self.scene, half_size=0.02, color=[0, 1, 0, 1],
            name="cubeB", initial_pose=sapien.Pose(p=[-1, 0, 0.02]),
        )
        self.goal_region = actors.build_red_white_target(
            self.scene, radius=self.goal_radius, thickness=1e-5,
            name="goal_region", add_collision=False,
            body_type="kinematic", initial_pose=sapien.Pose(),
        )

    @property
    def left_agent(self) -> Panda:
        return self.agent.agents[0]

    @property
    def right_agent(self) -> Panda:
        return self.agent.agents[1]
```

**Asset paths (resolved).**
- Table mesh: `mani_skill/utils/scene_builder/table/assets/table.glb` (built via `TableSceneBuilder`; top surface at z=0, `add_box_collision`).
- Panda URDF: `<PACKAGE_ASSET_DIR>/robots/panda/panda_v3.urdf` (per `PandaWristCam.urdf_path`).
- Cubes + target: procedurally built (no external mesh) via `mani_skill.utils.building.actors.build_cube` / `build_red_white_target`.

**Smoke.**
```bash
cd <ManiSkill-repo>
.venv/bin/python -c "import gymnasium as gym, mani_skill; e=gym.make('TwoRobotStackCube-v1'); print(e.observation_space, e.action_space); e.close()"
```
Expected stdout:
```
Box(-inf, inf, (1, 76), float32) Dict('panda_wristcam-0': Box(-1.0, 1.0, (8,), float32), 'panda_wristcam-1': Box(-1.0, 1.0, (8,), float32))
```

---

## §2 Actions

**Description.** The env does not declare an `ActionsCfg`. Each Panda resolves its controllers from `Panda._controller_configs`; with no `control_mode` passed to `gym.make`, `BaseAgent` picks the **first** key, `pd_joint_delta_pos`. Each arm therefore exposes a 7-dim delta-joint-position arm controller (`PDJointPosControllerConfig`, use_delta) + a 1-dim gripper (`PDJointPosMimicControllerConfig`), → 8 dims per arm, normalized to `[-1,1]`. `MultiAgent` exposes a **Dict** action space keyed by per-robot uid (`panda_wristcam-0`, `panda_wristcam-1`), each a `Box(-1,1,(8,))`. Combined effective action dim = 16.

**Decisions resolved.**
- `control_mode` (default-resolved) = `pd_joint_delta_pos` for each robot.
- Per-arm action layout: indices `[0:7]` = delta joint positions (arm), index `[7]` = gripper open/close (mimic).
- Combined action space: `Dict('panda_wristcam-0': Box(-1,1,(8,)), 'panda_wristcam-1': Box(-1,1,(8,)))`.
- Other supported control modes (selectable via `gym.make(..., control_mode=...)`, same for both arms): `pd_joint_delta_pos` (default), `pd_joint_pos`, `pd_ee_delta_pos`, `pd_ee_delta_pose`, `pd_ee_pose`, `pd_joint_target_delta_pos`, `pd_ee_target_delta_pos`, `pd_ee_target_delta_pose`, `pd_joint_vel`, `pd_joint_pos_vel`, `pd_joint_delta_pos_vel`.
- `MultiAgent` cannot use `control_mode="*"` (BaseEnv raises NotImplementedError); both arms share the single chosen mode.

**Code (controller registry, from `mani_skill/agents/robots/panda/panda.py::_controller_configs`).**
```python
controller_configs = dict(
    pd_joint_delta_pos=dict(arm=arm_pd_joint_delta_pos, gripper=gripper_pd_joint_pos),  # default
    pd_joint_pos=dict(arm=arm_pd_joint_pos, gripper=gripper_pd_joint_pos),
    pd_ee_delta_pos=dict(arm=arm_pd_ee_delta_pos, gripper=gripper_pd_joint_pos),
    pd_ee_delta_pose=dict(arm=arm_pd_ee_delta_pose, gripper=gripper_pd_joint_pos),
    pd_ee_pose=dict(arm=arm_pd_ee_pose, gripper=gripper_pd_joint_pos),
    # ... target/vel variants ...
)
```
`MultiAgent.action_space` (from `mani_skill/agents/multi_agent.py`):
```python
@property
def action_space(self):
    return spaces.Dict(
        {uid: agent.action_space for uid, agent in self.agents_dict.items()}
    )
```

**Smoke.**
```bash
.venv/bin/python -c "import gymnasium as gym, mani_skill; e=gym.make('TwoRobotStackCube-v1'); print(e.action_space); print({k:v.shape for k,v in e.action_space.spaces.items()}); e.close()"
```
Expected: action space is a `Dict` of two `Box(-1,1,(8,))` entries (`panda_wristcam-0`, `panda_wristcam-1`).

---

## §3 Reset

**Description.** `_initialize_episode` re-initializes the table scene (which also re-seeds both robot qpos with `robot_init_qpos_noise=0.02`), then samples both cube xy positions in disjoint y-bands so each cube is reachable by exactly one arm, applies a random z-axis-only rotation to each cube, and randomizes the goal region's x while pinning its y on the robot midline. Cubes are dropped onto the table (z=0.02 = half size); goal sits at z=1e-3.

**Decisions resolved (per reset, batched over `b = len(env_idx)`).**
- `table_scene.initialize(env_idx)` — re-poses table + both robots (qpos += noise 0.02).
- cubeA (blue, right-arm side): `x ∈ U[-0.05, 0.05]`, `y = -0.15 - U[0,0.1] + 0.05` (i.e. `y ∈ [-0.20, -0.10]`), `z = 0.02`.
- cubeB (green, left-arm side): `x ∈ U[-0.05, 0.05]`, `y = 0.15 + U[0,0.1] - 0.05` (i.e. `y ∈ [+0.10, +0.20]`), `z = 0.02`.
- Each cube quaternion: `random_quaternions(b, lock_x=True, lock_y=True, lock_z=False)` (z-axis-only rotation).
- goal_region: `x ∈ U[-0.05, 0.05]`, `y = -0.1`, `z = 1e-3`, orientation fixed `euler2quat(0, π/2, 0)` (target disk lies flat on the table).

**Code.**
```python
def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
    with torch.device(self.device):
        b = len(env_idx)
        self.table_scene.initialize(env_idx)
        # self.agents[0] is left, self.agents[1] is right
        torch.zeros((b, 3))
        torch.rand((b, 2)) * 0.2 - 0.1
        cubeA_xyz = torch.zeros((b, 3))
        cubeA_xyz[:, 0] = torch.rand((b,)) * 0.1 - 0.05
        cubeA_xyz[:, 1] = -0.15 - torch.rand((b,)) * 0.1 + 0.05
        cubeB_xyz = torch.zeros((b, 3))
        cubeB_xyz[:, 0] = torch.rand((b,)) * 0.1 - 0.05
        cubeB_xyz[:, 1] = 0.15 + torch.rand((b,)) * 0.1 - 0.05
        cubeA_xyz[:, 2] = 0.02
        cubeB_xyz[:, 2] = 0.02

        qs = random_quaternions(b, lock_x=True, lock_y=True, lock_z=False)
        self.cubeA.set_pose(Pose.create_from_pq(p=cubeA_xyz, q=qs))
        qs = random_quaternions(b, lock_x=True, lock_y=True, lock_z=False)
        self.cubeB.set_pose(Pose.create_from_pq(p=cubeB_xyz, q=qs))

        target_region_xyz = torch.zeros((b, 3))
        target_region_xyz[:, 0] = torch.rand((b,)) * 0.1 - 0.05
        target_region_xyz[:, 1] = -0.1
        target_region_xyz[..., 2] = 1e-3
        self.goal_region.set_pose(
            Pose.create_from_pq(p=target_region_xyz, q=euler2quat(0, np.pi / 2, 0))
        )
```
(Note: the two bare `torch.zeros((b,3))` / `torch.rand((b,2))*0.2-0.1` lines on entry are dead — they compute nothing assigned. Kept verbatim for faithful reproduction.)

`random_quaternions` source (`mani_skill/envs/utils/randomization/pose.py:13`): samples uniform random quaternions with optional per-axis locking.

**Smoke.**
```bash
.venv/bin/python -c "
import gymnasium as gym, mani_skill, torch
e=gym.make('TwoRobotStackCube-v1'); o,_=e.reset(seed=0)
u=e.unwrapped
print('cubeA y', float(u.cubeA.pose.p[0,1]), 'cubeB y', float(u.cubeB.pose.p[0,1]))
print('goal y', float(u.goal_region.pose.p[0,1]))
e.close()"
```
Expected: cubeA y ∈ [-0.20,-0.10], cubeB y ∈ [+0.10,+0.20], goal y ≈ -0.1.

---

## §4 Goal + Termination

**Description.** No `TerminationsCfg`; termination is the env's `evaluate()` `success` flag plus the `max_episode_steps=100` timeout (enforced by the gym `TimeLimit` wrapper). Success is the conjunction of four cooperative conditions: cubeA stacked on cubeB (xy aligned within half-diagonal + 5mm, z gap ≈ one cube), cubeB centered on the goal region (within `goal_radius=0.06`), AND both cubes released (neither grasped). `evaluate()` also returns intermediate flags consumed by the staged reward (§6).

**Decisions resolved.**
- `max_episode_steps = 100`.
- `is_cubeA_on_cubeB`: `‖(A−B)_xy‖ ≤ ‖half_size_xy‖ + 0.005` AND `|(A−B)_z − 2·half_size_z| ≤ 0.005`.
- `cubeB_placed`: `‖B_xy − goal_xy‖ < goal_radius (0.06)`.
- `is_cubeA_grasped = left_agent.is_grasping(cubeA)`; `is_cubeB_grasped = right_agent.is_grasping(cubeB)` (grasp test: `Panda.is_grasping(obj, min_force=0.5, max_angle=85)`).
- `success = is_cubeA_on_cubeB ∧ cubeB_placed ∧ ¬is_cubeA_grasped ∧ ¬is_cubeB_grasped`.

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
    cubeB_to_goal_dist = torch.linalg.norm(
        self.cubeB.pose.p[:, :2] - self.goal_region.pose.p[..., :2], axis=1
    )
    cubeB_placed = cubeB_to_goal_dist < self.goal_radius
    is_cubeA_grasped = self.left_agent.is_grasping(self.cubeA)
    is_cubeB_grasped = self.right_agent.is_grasping(self.cubeB)
    success = (
        is_cubeA_on_cubeB * cubeB_placed * (~is_cubeA_grasped) * (~is_cubeB_grasped)
    )
    return {
        "is_cubeA_grasped": is_cubeA_grasped,
        "is_cubeB_grasped": is_cubeB_grasped,
        "is_cubeA_on_cubeB": is_cubeA_on_cubeB,
        "cubeB_placed": cubeB_placed,
        "success": success.bool(),
    }
```

**Smoke.**
```bash
.venv/bin/python -c "
import gymnasium as gym, mani_skill
e=gym.make('TwoRobotStackCube-v1'); e.reset(seed=0)
info=e.unwrapped.evaluate()
print(sorted(info.keys()))
e.close()"
```
Expected keys: `['cubeB_placed','is_cubeA_grasped','is_cubeA_on_cubeB','is_cubeB_grasped','success']`.

---

## §5 Observation

**Description.** No `ObservationsCfg`. The full observation is assembled by `BaseEnv` from (a) per-robot proprioception (qpos/qvel of both Pandas + controller state), and (b) the task-specific `_get_obs_extra(info)`. In the default `state` obs mode the extras include both arm TCP poses, the goal-region position, both cube poses, the two arm→cube relative positions, and the cube→cube relative position. All terms are concatenated → flat `(1, 76)` state vector.

**Decisions resolved.**
- obs_mode (default-resolved) = first of `SUPPORTED_OBS_MODES` → `state`. Other modes available: `state_dict`, `none`, `sensor_data`, `rgb`, `depth`, `rgbd`, `pointcloud`, etc.
- `_get_obs_extra` always returns `left_arm_tcp` (7), `right_arm_tcp` (7) (raw 7-dim poses = xyz + quat).
- When `"state" in obs_mode`, additionally: `goal_region_pos` (3), `cubeA_pose` (7), `cubeB_pose` (7), `left_arm_tcp_to_cubeA_pos` (3), `right_arm_tcp_to_cubeB_pos` (3), `cubeA_to_cubeB_pos` (3). Extras subtotal = 7+7+3+7+7+3+3+3 = **40**.
- Proprioception (both robots) subtotal = 76 − 40 = **36** (qpos + qvel + controller state across the two 9-DoF Pandas).
- Resolved total obs dim = **76** (per canonical build smoke).
- No observation noise / corruption configured (default obs mode is clean state).

**Code.**
```python
def _get_obs_extra(self, info: dict):
    obs = dict(
        left_arm_tcp=self.left_agent.tcp.pose.raw_pose,
        right_arm_tcp=self.right_agent.tcp.pose.raw_pose,
    )
    if "state" in self.obs_mode:
        obs.update(
            goal_region_pos=self.goal_region.pose.p,
            cubeA_pose=self.cubeA.pose.raw_pose,
            cubeB_pose=self.cubeB.pose.raw_pose,
            left_arm_tcp_to_cubeA_pos=self.cubeA.pose.p - self.left_agent.tcp.pose.p,
            right_arm_tcp_to_cubeB_pos=self.cubeB.pose.p - self.right_agent.tcp.pose.p,
            cubeA_to_cubeB_pos=self.cubeB.pose.p - self.cubeA.pose.p,
        )
    return obs
```

**Smoke.**
```bash
.venv/bin/python -c "import gymnasium as gym, mani_skill; e=gym.make('TwoRobotStackCube-v1'); print(e.observation_space); e.close()"
```
Expected: `Box(-inf, inf, (1, 76), float32)`.

---

## §6 Reward

**Description.** A single monotone **4-stage staged** dense reward (NOT a sum/product of independent terms — stages overwrite via boolean masking, so the composer is "staged max-by-mask", each stage adding a fixed base offset 0/2/4/8 plus a shaped within-stage bonus). Stages:
1. **Reach+grasp** (base 0, range ~[0,1]): both arms approach their targets — left arm reaches cubeA, right arm reaches a "push pose" 0.025m +y of cubeB; plus left-arm grasp of cubeA. `reward = (reach_reward + is_cubeA_grasped)/2`.
2. **Place bottom cube** (gate `is_cubeA_grasped`, base 2): right side places cubeB onto the goal while left keeps holding cubeA. `reward = 2 + (place_reward + is_cubeA_grasped)/2`.
3. **Place top cube** (gate `cubeB_placed ∧ is_cubeA_grasped`, base 4): stack cubeA on cubeB while right arm clears to y=0.2. `reward = 4 + place_reward·2 + right_arm_leave_reward`.
4. **Release both** (gate `is_cubeA_on_cubeB ∧ cubeB_placed`, base 8): open both grippers. `reward = 8 + (ungrasp_left + ungrasp_right)/2`.
- Terminal `success` overrides to constant **10**.
- `compute_normalized_dense_reward = compute_dense_reward / 10`.

**Composer.** Per-row masked stage override (a piecewise function selecting the highest reached stage), NOT additive across terms. Effective return range ≈ [0, 10]; normalized ≈ [0, 1].

**Decisions resolved (saturated per-step magnitudes, by stage).**
- Stage 1: ≈ 0–1 (reach 0–1 averaged with grasp indicator 0/1).
- Stage 2: base 2 + [0,1] → ≈ 2–3.
- Stage 3: base 4 + (place·2 ∈ [0,2]) + (leave ∈ [0,1]) → ≈ 4–7.
- Stage 4: base 8 + [0,1] → ≈ 8–9.
- Success: 10 (constant). Normalized: divide all by 10. (Magnitudes retro-computed from weights — no per-stage docstring budget in source.)
- Hard-coded Panda assumption: `gripper_width = left_agent.robot.get_qlimits()[0,-1,1]*2`; ungrasp uses the last two qpos entries (finger joints).

**Code (verbatim).**
```python
def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: dict):
    # Stage 1: Reach and grasp
    cubeA_to_left_arm_tcp_dist = torch.linalg.norm(
        self.left_agent.tcp.pose.p - self.cubeA.pose.p, axis=1
    )
    right_arm_push_pose = Pose.create_from_pq(
        p=self.cubeB.pose.p
        + torch.tensor([0, self.cube_half_size[0] + 0.005, 0], device=self.device)
    )
    right_arm_to_push_pose_dist = torch.linalg.norm(
        right_arm_push_pose.p - self.right_agent.tcp.pose.p, axis=1
    )
    reach_reward = (
        1
        - torch.tanh(5 * cubeA_to_left_arm_tcp_dist)
        + 1
        - torch.tanh(5 * right_arm_to_push_pose_dist)
    ) / 2

    # grasp reward for left robot which needs to lift cubeA up eventually
    cubeA_pos = self.cubeA.pose.p
    cubeB_pos = self.cubeB.pose.p
    reward = (reach_reward + info["is_cubeA_grasped"]) / 2

    # pass condition for stage 1
    place_stage_reached = info["is_cubeA_grasped"]

    # Stage 2: Place bottom cube and still hold to cube A
    cubeB_to_goal_dist = torch.linalg.norm(
        cubeB_pos[:, :2] - self.goal_region.pose.p[..., :2], axis=1
    )
    place_reward = 1 - torch.tanh(5 * cubeB_to_goal_dist)
    stage_2_reward = place_reward + info["is_cubeA_grasped"]
    reward[place_stage_reached] = 2 + stage_2_reward[place_stage_reached] / 2

    # pass condition for stage 2
    cubeB_placed_and_cubeA_grasped = info["cubeB_placed"] * info["is_cubeA_grasped"]

    # Stage 3: Place top cube while moving right arm away to give left arm space
    goal_xyz = torch.hstack(
        [cubeB_pos[:, :2], (cubeB_pos[:, 2] + self.cube_half_size[2] * 2)[:, None]]
    )
    cubeA_to_goal_dist = torch.linalg.norm(goal_xyz - cubeA_pos, axis=1)
    place_reward = 1 - torch.tanh(5 * cubeA_to_goal_dist)

    # move right arm as close as possible to the y=0.2 line
    right_arm_leave_reward = 1 - torch.tanh(
        5 * (self.right_agent.tcp.pose.p[:, 1] - 0.2).abs()
    )
    stage_3_reward = place_reward * 2 + right_arm_leave_reward
    reward[cubeB_placed_and_cubeA_grasped] = (
        4 + stage_3_reward[cubeB_placed_and_cubeA_grasped]
    )
    # pass condition for stage 3
    cubes_placed = info["is_cubeA_on_cubeB"] * info["cubeB_placed"]

    # Stage 4: get both robots to stop grasping
    gripper_width = (self.left_agent.robot.get_qlimits()[0, -1, 1] * 2).to(
        self.device
    )  # NOTE: hard-coded with panda
    ungrasp_reward_left = (
        torch.sum(self.left_agent.robot.get_qpos()[:, -2:], axis=1) / gripper_width
    )
    ungrasp_reward_left[~info["is_cubeA_grasped"]] = 1.0
    ungrasp_reward_right = (
        torch.sum(self.right_agent.robot.get_qpos()[:, -2:], axis=1) / gripper_width
    )
    ungrasp_reward_right[~info["is_cubeB_grasped"]] = 1.0

    reward[cubes_placed] = (
        8 + (ungrasp_reward_left + ungrasp_reward_right)[cubes_placed] / 2
    )

    reward[info["success"]] = 10

    return reward

def compute_normalized_dense_reward(self, obs: Any, action: torch.Tensor, info: dict):
    return self.compute_dense_reward(obs=obs, action=action, info=info) / 10
```

**Smoke.**
```bash
.venv/bin/python -c "
import gymnasium as gym, mani_skill, torch
e=gym.make('TwoRobotStackCube-v1', reward_mode='normalized_dense'); e.reset(seed=0)
a=e.action_space.sample()
_,r,_,_,_=e.step(a); print('reward', float(r), 'finite', bool(torch.isfinite(torch.as_tensor(r)).all()))
e.close()"
```
Expected: finite reward in ≈ [0, 1] (normalized).

---

## §7 DR

`<no DR>` — there is no `startup`/`interval` domain randomization. All variability comes from the per-reset spawn randomization in §3 (cube xy in disjoint bands, z-axis cube rotations, goal x) plus `robot_init_qpos_noise=0.02` applied to both robots by `TableSceneBuilder.initialize`. No friction/mass/visual/lighting randomization, no per-step disturbance.

---

