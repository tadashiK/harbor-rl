# PlaceSphere-v1 — Implementation Spec

- robot: Franka Panda (default; Fetch also supported)
- simulator: ManiSkill (SAPIEN, `BaseEnv` + `@register_env`)
- objects: sphere, bin, table
- bimanual: false
- summary: Pick up a sphere and place it inside a bin.

Task: place a dynamic sphere onto the top of a shallow kinematic bin; the robot must end static with the gripper open (not grasping).

---

## §1 Registration + Scene

**Description.** Registered as `PlaceSphere-v1` with `max_episode_steps=50`, entry point `PlaceSphereEnv(BaseEnv)`. Supports Panda and Fetch (default `panda`). The scene is a standard `TableSceneBuilder` table + robot, plus two added actors: a dynamic blue sphere (`build_sphere`) and a kinematic bin assembled from 5 boxes (1 bottom plate + 4 raised edge walls). Sim cfg only bumps GPU collision-pair / patch capacities; physics dt and substeps are SAPIEN defaults.

**Decisions resolved.**
- `register_env("PlaceSphere-v1", max_episode_steps=50)`
- `SUPPORTED_ROBOTS = ["panda", "fetch"]`; default `robot_uids="panda"`; `robot_init_qpos_noise=0.02`.
- Agent base pose: `sapien.Pose(p=[-0.615, 0, 0])` (robot placed behind the table, facing +x toward the workspace).
- Geometry constants: sphere `radius = 0.02`; bin `inner_side_half_len = 0.02`; `short_side_half_size = 0.0025`.
  - `block_half_size = [0.0025, 0.045, 0.045]` (bottom plate; `2*0.0025 + 0.02 = 0.045`)
  - `edge_block_half_size = [0.0025, 0.045, 0.005]` (the 4 walls)
- Sphere: `actors.build_sphere(radius=0.02, color=[12,42,160,255]/255, body_type="dynamic")`, name `"sphere"`.
- Bin: `_build_bin` → 5 box collisions+visuals, `build_kinematic(name="bin")`. Layout (local frame): bottom at origin; 4 edge walls offset by `dx = dy = block_half_size[1]-block_half_size[0] = 0.0425` and raised by `dz = edge_block_half_size[2]+block_half_size[0] = 0.0075`. Walls along ±x use `edge_block_half_size`; walls along ±y swap the x/y half-extents.
- Sim cfg: `SimConfig(gpu_memory_config=GPUMemoryConfig(found_lost_pairs_capacity=2**25, max_rigid_patch_count=2**18))`.
- Sensor camera `base_camera`: `look_at(eye=[0.3,0,0.2], target=[-0.1,0,0])`, 128×128, fov π/2, near 0.01, far 100.
- Human-render camera `render_camera`: `look_at([0.6,-0.2,0.2],[0.0,0.0,0.2])`, 512×512, fov 1.

**Code.**
```python
@register_env("PlaceSphere-v1", max_episode_steps=50)
class PlaceSphereEnv(BaseEnv):
    _sample_video_link = "https://github.com/haosulab/ManiSkill/raw/main/figures/environment_demos/PlaceSphere-v1_rt.mp4"
    SUPPORTED_ROBOTS = ["panda", "fetch"]
    agent: Union[Panda, Fetch]

    radius = 0.02  # radius of the sphere
    inner_side_half_len = 0.02  # side length of the bin's inner square
    short_side_half_size = 0.0025  # length of the shortest edge of the block
    block_half_size = [
        short_side_half_size,
        2 * short_side_half_size + inner_side_half_len,
        2 * short_side_half_size + inner_side_half_len,
    ]
    edge_block_half_size = [
        short_side_half_size,
        2 * short_side_half_size + inner_side_half_len,
        2 * short_side_half_size,
    ]

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
        pose = sapien_utils.look_at(eye=[0.3, 0, 0.2], target=[-0.1, 0, 0])
        return [
            CameraConfig("base_camera", pose=pose, width=128, height=128,
                         fov=np.pi / 2, near=0.01, far=100)
        ]

    @property
    def _default_human_render_camera_configs(self):
        pose = sapien_utils.look_at([0.6, -0.2, 0.2], [0.0, 0.0, 0.2])
        return CameraConfig("render_camera", pose=pose, width=512, height=512,
                            fov=1, near=0.01, far=100)

    def _build_bin(self, radius):
        builder = self.scene.create_actor_builder()
        dx = self.block_half_size[1] - self.block_half_size[0]
        dy = self.block_half_size[1] - self.block_half_size[0]
        dz = self.edge_block_half_size[2] + self.block_half_size[0]
        poses = [
            sapien.Pose([0, 0, 0]),
            sapien.Pose([-dx, 0, dz]),
            sapien.Pose([dx, 0, dz]),
            sapien.Pose([0, -dy, dz]),
            sapien.Pose([0, dy, dz]),
        ]
        half_sizes = [
            [self.block_half_size[1], self.block_half_size[2], self.block_half_size[0]],
            self.edge_block_half_size,
            self.edge_block_half_size,
            [self.edge_block_half_size[1], self.edge_block_half_size[0], self.edge_block_half_size[2]],
            [self.edge_block_half_size[1], self.edge_block_half_size[0], self.edge_block_half_size[2]],
        ]
        for pose, half_size in zip(poses, half_sizes):
            builder.add_box_collision(pose, half_size)
            builder.add_box_visual(pose, half_size)
        return builder.build_kinematic(name="bin")

    def _load_agent(self, options: dict):
        super()._load_agent(options, sapien.Pose(p=[-0.615, 0, 0]))

    def _load_scene(self, options: dict):
        self.table_scene = TableSceneBuilder(
            env=self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.table_scene.build()
        self.obj = actors.build_sphere(
            self.scene, radius=self.radius,
            color=np.array([12, 42, 160, 255]) / 255,
            name="sphere", body_type="dynamic",
        )
        self.bin = self._build_bin(self.radius)
```

**Smoke.** `.venv/bin/python -c "import gymnasium as gym, mani_skill; e=gym.make('PlaceSphere-v1'); print(e.observation_space, e.action_space)"` → `Box(-inf, inf, (1, 39), float32) Box(-1.0, 1.0, (8,), float32)`.

---

## §2 Actions

**Description.** No `ActionsCfg` — ManiSkill action space is defined by the agent's controller (`control_mode`). PlaceSphere does not override the control mode, so it uses the robot's default. For Panda the default (first key in `_controller_configs`) is `pd_joint_delta_pos`: 7 arm joints (delta position) + 1 gripper (`pd_joint_pos`), all normalized to `[-1, 1]` → action dim 8.

**Decisions resolved.**
- `control_mode` = `pd_joint_delta_pos` (Panda default; resolved via `BaseAgent`: `control_mode = self.supported_control_modes[0]`).
- Action space `Box(-1.0, 1.0, (8,), float32)` = 7 arm delta-joint + 1 gripper.
- Other supported Panda modes (selectable via `gym.make(..., control_mode=...)`): `pd_joint_delta_pos`, `pd_ee_delta_pos`, `pd_ee_delta_pose`, `pd_joint_pos`, `pd_joint_target_delta_pos`, `pd_ee_target_delta_pos`, `pd_ee_target_delta_pose`, `pd_joint_vel`, `pd_joint_pos_vel`, `pd_joint_delta_pos_vel`.

**Code.** (none in `place_sphere.py`; controller comes from `mani_skill/agents/robots/panda/panda.py::_controller_configs`, arm `arm_pd_joint_delta_pos = PDJointPosControllerConfig(..., use_delta=True)`, gripper `gripper_pd_joint_pos`.)

**Smoke.** action_space line from §1 build smoke: `Box(-1.0, 1.0, (8,), float32)`.

---

## §3 Reset

**Description.** `_initialize_episode` re-initializes the table scene (robot qpos with noise) and randomizes the planar positions of the sphere and the bin in disjoint x-bands so they never start overlapping. Both keep identity orientation. The sphere rests on the table at z=radius; the bin bottom sits at z=block_half_size[0].

**Decisions resolved.**
- `self.table_scene.initialize(env_idx)` — applies `robot_init_qpos_noise=0.02` to the robot.
- Sphere: `x ∈ [-0.1, -0.05]`, `y ∈ [-0.1, 0.1]`, `z = radius = 0.02`, quat `[1,0,0,0]`.
- Bin: `x ∈ [0.0, 0.1]`, `y ∈ [-0.1, 0.1]`, `z = block_half_size[0] = 0.0025`, quat `[1,0,0,0]`.
- Disjoint x-bands (sphere in the front quarter, bin in the back half) prevent initial collision.

**Code.**
```python
def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
    with torch.device(self.device):
        b = len(env_idx)
        self.table_scene.initialize(env_idx)

        # sphere
        xyz = torch.zeros((b, 3))
        xyz[..., 0] = (torch.rand((b, 1)) * 0.05 - 0.1)[..., 0]   # x in [-0.1, -0.05]
        xyz[..., 1] = (torch.rand((b, 1)) * 0.2 - 0.1)[..., 0]    # y in [-0.1, 0.1]
        xyz[..., 2] = self.radius
        q = [1, 0, 0, 0]
        obj_pose = Pose.create_from_pq(p=xyz, q=q)
        self.obj.set_pose(obj_pose)

        # bin
        pos = torch.zeros((b, 3))
        pos[:, 0] = (torch.rand((b, 1))[..., 0] * 0.1)            # x in [0, 0.1]
        pos[:, 1] = (torch.rand((b, 1))[..., 0] * 0.2 - 0.1)      # y in [-0.1, 0.1]
        pos[:, 2] = self.block_half_size[0]
        q = [1, 0, 0, 0]
        bin_pose = Pose.create_from_pq(p=pos, q=q)
        self.bin.set_pose(bin_pose)
```

**Smoke.** `.venv/bin/python -c "import gymnasium as gym, mani_skill; e=gym.make('PlaceSphere-v1'); e.reset(seed=0); e.reset(seed=1); print('reset ok')"` (expected `reset ok`; obj/bin poses differ across seeds).

---

## §4 Goal + Termination

**Description.** No explicit `TerminationsCfg` / `CommandsCfg`. Episodes end on `max_episode_steps=50` (time limit) or on success. Success (`evaluate()`) requires the sphere centered over the bin top within a 5 mm xy tolerance and the correct z gap (radius + bottom-plate half), AND the sphere is nearly static, AND the robot is no longer grasping it.

**Decisions resolved.**
- `max_episode_steps = 50` (from `@register_env`).
- `xy_flag`: `‖(pos_obj − pos_bin)[:2]‖ ≤ 0.005`.
- `z_flag`: `|offset_z − radius − block_half_size[0]| ≤ 0.005` (i.e. target gap `0.02 + 0.0025 = 0.0225`).
- `is_obj_on_bin = xy_flag ∧ z_flag`.
- `is_obj_static = obj.is_static(lin_thresh=1e-2, ang_thresh=0.5)`.
- `is_obj_grasped = agent.is_grasping(obj)`.
- `success = is_obj_on_bin ∧ is_obj_static ∧ ¬is_obj_grasped`.

**Code.**
```python
def evaluate(self):
    pos_obj = self.obj.pose.p
    pos_bin = self.bin.pose.p
    offset = pos_obj - pos_bin
    xy_flag = torch.linalg.norm(offset[..., :2], axis=1) <= 0.005
    z_flag = torch.abs(offset[..., 2] - self.radius - self.block_half_size[0]) <= 0.005
    is_obj_on_bin = torch.logical_and(xy_flag, z_flag)
    is_obj_static = self.obj.is_static(lin_thresh=1e-2, ang_thresh=0.5)
    is_obj_grasped = self.agent.is_grasping(self.obj)
    success = is_obj_on_bin & is_obj_static & (~is_obj_grasped)
    return {
        "is_obj_grasped": is_obj_grasped,
        "is_obj_on_bin": is_obj_on_bin,
        "is_obj_static": is_obj_static,
        "success": success,
    }
```

**Smoke.** `.venv/bin/python -c "import gymnasium as gym, mani_skill; e=gym.make('PlaceSphere-v1'); e.reset(seed=0); o,r,te,tr,i=e.step(e.action_space.sample()); print(sorted(i.keys()))"` (expected info keys include `is_obj_grasped`, `is_obj_on_bin`, `is_obj_static`, `success`).

---

## §5 Observation

**Description.** Default `obs_mode="state"` (first in `SUPPORTED_OBS_MODES`). The full state vector concatenates the ManiSkill base state (agent proprioception + object/actor states the base assembles) with `_get_obs_extra`. `_get_obs_extra` always exposes grasp flag, TCP pose, and bin position; under any `state`-family mode it additionally exposes the sphere pose and the TCP→sphere vector. Resolved total dim = 39 (per build smoke, obs_mode="state").

**Decisions resolved.**
- `SUPPORTED_OBS_MODES = ("state", "state_dict", "none", "sensor_data", "any_textures", "pointcloud")`; default `"state"`.
- `_get_obs_extra` always: `is_grasped` (1), `tcp_pose` (raw 7: xyz+quat), `bin_pos` (3).
- When `"state" in obs_mode`: adds `obj_pose` (raw 7) and `tcp_to_obj_pos` (3).
- Total flattened obs dim (state mode, 1 env) = **39** (verbatim from build smoke).

**Code.**
```python
def _get_obs_extra(self, info: dict):
    obs = dict(
        is_grasped=info["is_obj_grasped"],
        tcp_pose=self.agent.tcp.pose.raw_pose,
        bin_pos=self.bin.pose.p,
    )
    if "state" in self.obs_mode:
        obs.update(
            obj_pose=self.obj.pose.raw_pose,
            tcp_to_obj_pos=self.obj.pose.p - self.agent.tcp.pose.p,
        )
    return obs
```

**Smoke.** observation_space line from §1 build smoke: `Box(-inf, inf, (1, 39), float32)`.

---

## §6 Reward

**Description.** Single dense reward computed in `compute_dense_reward`, composed as a **staged overwrite** (NOT a sum/product): a base reaching reward, then progressively higher reward tiers that *overwrite* the value for envs that reach each stage (grasped → on-bin → success). Stages: (a) reach the sphere `2*(1−tanh(5·d_tcp_obj))` ∈ [0,2]; (b) once grasped, `4 + place_reward` where `place_reward = 1−tanh(5·d_obj_binTop)` ∈ [4,5]; (c) once on bin, `6 + (ungrasp + static + robot_static)/3` ∈ [6,7]; (d) on success, fixed `13`. Normalized variant divides by `max_reward = 13.0`.

**Composer:** `staged-overwrite` (each later stage `reward[mask] = ...` replaces the value for matching envs; not additive). `compute_normalized_dense_reward = compute_dense_reward / 13.0`.

**Planning budget (saturated per-step magnitudes, retro-computed from code):**
- Stage A (reaching, not grasped): up to `2.0`.
- Stage B (grasped, not yet on bin): `4.0` + place ∈ [0,1] → up to `5.0`.
- Stage C (on bin, not yet success): `6.0` + mean(ungrasp, static, robot_static) ∈ [0,1] → up to `7.0`. Note `ungrasp_reward = 16.0` when not grasped feeds this mean but the mean is divided by 3, and `robot_static`/`static` ∈ [0,1].
- Stage D (success): `13.0` (= `max_reward`).

**Code (verbatim).**
```python
def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: dict):
    # reaching reward
    tcp_pose = self.agent.tcp.pose.p
    obj_pos = self.obj.pose.p
    obj_to_tcp_dist = torch.linalg.norm(tcp_pose - obj_pos, axis=1)
    reward = 2 * (1 - torch.tanh(5 * obj_to_tcp_dist))

    # grasp and place reward
    obj_pos = self.obj.pose.p
    self.bin.pose.p
    bin_top_pos = self.bin.pose.p.clone()
    bin_top_pos[:, 2] = bin_top_pos[:, 2] + self.block_half_size[0] + self.radius
    obj_to_bin_top_dist = torch.linalg.norm(bin_top_pos - obj_pos, axis=1)
    place_reward = 1 - torch.tanh(5.0 * obj_to_bin_top_dist)
    reward[info["is_obj_grasped"]] = (4 + place_reward)[info["is_obj_grasped"]]

    # ungrasp and static reward
    gripper_width = (self.agent.robot.get_qlimits()[0, -1, 1] * 2).to(self.device)
    is_obj_grasped = info["is_obj_grasped"]
    ungrasp_reward = (
        torch.sum(self.agent.robot.get_qpos()[:, -2:], axis=1) / gripper_width
    )
    ungrasp_reward[
        ~is_obj_grasped
    ] = 16.0  # give ungrasp a bigger reward, so that it exceeds the robot static reward and the gripper can close
    v = torch.linalg.norm(self.obj.linear_velocity, axis=1)
    av = torch.linalg.norm(self.obj.angular_velocity, axis=1)
    static_reward = 1 - torch.tanh(v * 10 + av)
    robot_static_reward = self.agent.is_static(
        0.2
    )  # keep the robot static at the end state, since the sphere may spin when being placed on top
    reward[info["is_obj_on_bin"]] = (
        6 + (ungrasp_reward + static_reward + robot_static_reward) / 3.0
    )[info["is_obj_on_bin"]]

    # success reward
    reward[info["success"]] = 13
    return reward

def compute_normalized_dense_reward(self, obs: Any, action: Array, info: dict):
    # this should be equal to compute_dense_reward / max possible reward
    max_reward = 13.0
    return self.compute_dense_reward(obs=obs, action=action, info=info) / max_reward
```

Helpers used (all on agent/structs, no task-local defs): `self.agent.tcp.pose.p`, `self.obj.pose.p`, `self.bin.pose.p`, `self.agent.robot.get_qlimits()`, `self.agent.robot.get_qpos()`, `self.obj.linear_velocity`, `self.obj.angular_velocity`, `self.agent.is_static(thresh)`, `info["is_obj_grasped"/"is_obj_on_bin"/"success"]` (from `evaluate`).

**Smoke.** `.venv/bin/python -c "import gymnasium as gym, mani_skill, numpy as np; e=gym.make('PlaceSphere-v1', reward_mode='dense'); e.reset(seed=0); o,r,te,tr,i=e.step(e.action_space.sample()); print('reward', float(np.asarray(r).reshape(-1)[0]), 'finite', np.isfinite(np.asarray(r)).all())"` (expected finite reward in [0,13]).

---

## §7 DR

`<no DR>` — there is no `EventCfg`/`startup`/`interval` randomization. All randomization is reset-time only (sphere/bin planar positions in §3 and `robot_init_qpos_noise=0.02` applied by `TableSceneBuilder.initialize`). No startup or per-step domain randomization terms exist.

---

