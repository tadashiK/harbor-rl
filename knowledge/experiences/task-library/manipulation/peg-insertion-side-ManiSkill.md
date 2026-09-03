# PegInsertionSide-v1 — Implementation Spec

- robot: Franka Panda with wrist camera (`panda_wristcam`)
- simulator: ManiSkill (SAPIEN, `BaseEnv` + `@register_env`)
- objects: peg, box with hole, table
- bimanual: false
- summary: Insert a peg sideways into a matching hole in a box.

A precision peg-insertion task: pick up an orange-white peg laid flat on the table and insert its orange (head) end into a side hole of a box. Hole clearance is a tight `0.003 m` over the peg radius; success requires the peg head to be inserted past the mid-depth with sub-`box_hole_radii` lateral tolerance.

---

## §1 Registration + Scene

### Description
Registered with `@register_env("PegInsertionSide-v1", max_episode_steps=100)`. The agent is a `PandaWristCam` (the only `SUPPORTED_ROBOTS` entry), base-mounted at `Pose(p=[-0.615, 0, 0])`. The scene is the standard `TableSceneBuilder` plus, **per parallel env**, a freshly-built peg (two stacked half-boxes: orange head + white tail) and a kinematic box-with-hole assembled from 4 box walls. Geometry is randomized at reconfiguration (peg half-length 0.085–0.125 m, radius 0.015–0.025 m); the hole inner radius = peg radius + `_clearance` (0.003 m) — this is how tolerance is set. Per-env actors are merged via `Actor.merge` so the batch presents single `self.peg` / `self.box` structs. `reconfiguration_freq` defaults to 1 for `num_envs==1`, else 0.

### Decisions resolved
- `SUPPORTED_ROBOTS = ["panda_wristcam"]`, default `robot_uids="panda_wristcam"`.
- Robot base pose: `sapien.Pose(p=[-0.615, 0, 0])` (set in both `_load_agent` and `_initialize_episode`).
- `_clearance = 0.003` m (hole_inner_radius = peg_radius + 0.003).
- Peg: `add_box_collision(half_size=[length, radius, radius])`; orange head visual at `+length/2`, white tail visual at `-length/2`; `initial_pose = Pose(p=[0,0,0.1])`.
- Box: `_build_box_with_hole(inner_radius=radius+0.003, outer_radius=length, depth=length, center=centers[i])`; built `build_kinematic`; `initial_pose = Pose(p=[0,1,0.1])`.
- `peg_half_sizes = [length, radius, radius]` per env; `peg_head_offsets` = `+length` along peg x; `box_hole_offsets` = `[0, center_y, center_z]`; `box_hole_radii = radii + 0.003`.
- Sim: `_default_sim_config = SimConfig()` → defaults `sim_freq=100`, `control_freq=20` (5 physx substeps per control step).
- Sensors: `base_camera` 128×128 fov π/2 at `look_at([0,-0.3,0.2],[0,0,0.1])`; human render cam 512×512 at `look_at([0.5,-0.5,0.8],[0.05,-0.1,0.4])`.

### Code
```python
def _build_box_with_hole(
    scene: ManiSkillScene, inner_radius, outer_radius, depth, center=(0, 0)
):
    builder = scene.create_actor_builder()
    thickness = (outer_radius - inner_radius) * 0.5
    # x-axis is hole direction
    half_center = [x * 0.5 for x in center]
    half_sizes = [
        [depth, thickness - half_center[0], outer_radius],
        [depth, thickness + half_center[0], outer_radius],
        [depth, outer_radius, thickness - half_center[1]],
        [depth, outer_radius, thickness + half_center[1]],
    ]
    offset = thickness + inner_radius
    poses = [
        sapien.Pose([0, offset + half_center[0], 0]),
        sapien.Pose([0, -offset + half_center[0], 0]),
        sapien.Pose([0, 0, offset + half_center[1]]),
        sapien.Pose([0, 0, -offset + half_center[1]]),
    ]
    mat = sapien.render.RenderMaterial(
        base_color=sapien_utils.hex2rgba("#FFD289"), roughness=0.5, specular=0.5
    )
    for half_size, pose in zip(half_sizes, poses):
        builder.add_box_collision(pose, half_size)
        builder.add_box_visual(pose, half_size, material=mat)
    return builder


@register_env("PegInsertionSide-v1", max_episode_steps=100)
class PegInsertionSideEnv(BaseEnv):
    SUPPORTED_ROBOTS = ["panda_wristcam"]
    agent: Union[PandaWristCam]
    _clearance = 0.003

    def __init__(self, *args, robot_uids="panda_wristcam", num_envs=1,
                 reconfiguration_freq=None, **kwargs):
        if reconfiguration_freq is None:
            if num_envs == 1:
                reconfiguration_freq = 1
            else:
                reconfiguration_freq = 0
        super().__init__(*args, robot_uids=robot_uids, num_envs=num_envs,
                         reconfiguration_freq=reconfiguration_freq, **kwargs)

    @property
    def _default_sim_config(self):
        return SimConfig()

    @property
    def _default_sensor_configs(self):
        pose = sapien_utils.look_at([0, -0.3, 0.2], [0, 0, 0.1])
        return [CameraConfig("base_camera", pose, 128, 128, np.pi / 2, 0.01, 100)]

    @property
    def _default_human_render_camera_configs(self):
        pose = sapien_utils.look_at([0.5, -0.5, 0.8], [0.05, -0.1, 0.4])
        return CameraConfig("render_camera", pose, 512, 512, 1, 0.01, 100)

    def _load_agent(self, options: dict):
        super()._load_agent(options, sapien.Pose(p=[-0.615, 0, 0]))

    def _load_scene(self, options: dict):
        with torch.device(self.device):
            self.table_scene = TableSceneBuilder(self)
            self.table_scene.build()

            lengths = self._batched_episode_rng.uniform(0.085, 0.125)
            radii = self._batched_episode_rng.uniform(0.015, 0.025)
            centers = (
                0.5 * (lengths - radii)[:, None]
                * self._batched_episode_rng.uniform(-1, 1, size=(2,))
            )
            self.peg_half_sizes = common.to_tensor(np.vstack([lengths, radii, radii])).T
            peg_head_offsets = torch.zeros((self.num_envs, 3))
            peg_head_offsets[:, 0] = self.peg_half_sizes[:, 0]
            self.peg_head_offsets = Pose.create_from_pq(p=peg_head_offsets)

            box_hole_offsets = torch.zeros((self.num_envs, 3))
            box_hole_offsets[:, 1:] = common.to_tensor(centers)
            self.box_hole_offsets = Pose.create_from_pq(p=box_hole_offsets)
            self.box_hole_radii = common.to_tensor(radii + self._clearance)

            pegs = []
            boxes = []
            for i in range(self.num_envs):
                scene_idxs = [i]
                length = lengths[i]
                radius = radii[i]
                builder = self.scene.create_actor_builder()
                builder.add_box_collision(half_size=[length, radius, radius])
                mat = sapien.render.RenderMaterial(
                    base_color=sapien_utils.hex2rgba("#EC7357"),
                    roughness=0.5, specular=0.5)
                builder.add_box_visual(
                    sapien.Pose([length / 2, 0, 0]),
                    half_size=[length / 2, radius, radius], material=mat)
                mat = sapien.render.RenderMaterial(
                    base_color=sapien_utils.hex2rgba("#EDF6F9"),
                    roughness=0.5, specular=0.5)
                builder.add_box_visual(
                    sapien.Pose([-length / 2, 0, 0]),
                    half_size=[length / 2, radius, radius], material=mat)
                builder.initial_pose = sapien.Pose(p=[0, 0, 0.1])
                builder.set_scene_idxs(scene_idxs)
                peg = builder.build(f"peg_{i}")
                self.remove_from_state_dict_registry(peg)

                inner_radius, outer_radius, depth = (
                    radius + self._clearance, length, length)
                builder = _build_box_with_hole(
                    self.scene, inner_radius, outer_radius, depth, center=centers[i])
                builder.initial_pose = sapien.Pose(p=[0, 1, 0.1])
                builder.set_scene_idxs(scene_idxs)
                box = builder.build_kinematic(f"box_with_hole_{i}")
                self.remove_from_state_dict_registry(box)
                pegs.append(peg)
                boxes.append(box)
            self.peg = Actor.merge(pegs, "peg")
            self.box = Actor.merge(boxes, "box_with_hole")
            self.add_to_state_dict_registry(self.peg)
            self.add_to_state_dict_registry(self.box)
```

### Smoke
```bash
cd <ManiSkill-repo>
.venv/bin/python -c "import gymnasium as gym, mani_skill; e=gym.make('PegInsertionSide-v1'); print(e.observation_space, e.action_space)"
# Box(-inf, inf, (1, 43), float32) Box(-1.0, 1.0, (8,), float32)
```

---

## §2 Actions

### Description
No task-local `ActionsCfg`; the action space comes entirely from the robot's controller. `control_mode` is unspecified, so `BaseEnv` uses the agent's `_default_control_mode = supported_control_modes[0]` → **`pd_joint_delta_pos`** (first key in Panda's `_controller_configs`). Arm: 7-dim normalized PD joint delta-position (`arm_pd_joint_delta_pos`, lower=-0.1, upper=0.1). Gripper: 1-dim mimic PD joint position. Total action = 8, normalized to `[-1, 1]`.

### Decisions resolved
- control_mode = `pd_joint_delta_pos` (default = `supported_control_modes[0]`).
- Arm controller `PDJointPosControllerConfig` (delta variant): `lower=-0.1, upper=0.1, stiffness=1e3, damping=1e2, force_limit=100, use_delta=True, normalize_action=True`.
- Gripper `PDJointPosMimicControllerConfig` (`gripper_pd_joint_pos`): single mimic action over `panda_finger_joint1/2`, `stiffness=1e3, damping=1e2, force_limit=100`.
- Resolved action_space = `Box(-1.0, 1.0, (8,), float32)` (7 arm + 1 gripper).
- Other available control modes: `pd_joint_pos`, `pd_ee_delta_pos`, `pd_ee_delta_pose` (pos/rot bounds ±0.1), `pd_ee_pose`, `pd_joint_target_delta_pos`, `pd_ee_target_delta_pos/pose`, `pd_joint_vel`, `pd_joint_pos_vel`, `pd_joint_delta_pos_vel`.

### Code
```python
# mani_skill/agents/robots/panda/panda.py — Panda._controller_configs (relevant subset)
arm_pd_joint_delta_pos = PDJointPosControllerConfig(
    self.arm_joint_names, lower=-0.1, upper=0.1,
    stiffness=self.arm_stiffness, damping=self.arm_damping,
    force_limit=self.arm_force_limit, use_delta=True)
gripper_pd_joint_pos = PDJointPosMimicControllerConfig(
    self.gripper_joint_names, lower=-0.01, upper=0.04,
    stiffness=self.gripper_stiffness, damping=self.gripper_damping,
    force_limit=self.gripper_force_limit)
controller_configs = dict(
    pd_joint_delta_pos=dict(arm=arm_pd_joint_delta_pos, gripper=gripper_pd_joint_pos),
    ...  # default = first key
)
# arm_stiffness=1e3, arm_damping=1e2, arm_force_limit=100
# gripper_stiffness=1e3, gripper_damping=1e2, gripper_force_limit=100
```

### Smoke
Covered by §1 build smoke (`action_space == Box(-1.0, 1.0, (8,), float32)`).

---

## §3 Reset (`_initialize_episode`)

### Description
Per reset: re-initialize the table scene; place the peg flat on the table within a randomized xy box with z = peg radius and a z-axis rotation around π/2 (±π/3); place the box-with-hole flat within a separate xy box with z = peg half-length and z-rotation around π/2 (±π/8); set the robot to the canonical rest qpos + N(0, 0.02) noise (gripper fingers forced open to 0.04) and pin the base back at `[-0.615, 0, 0]`.

### Decisions resolved
- Peg xy ∈ `[-0.1,-0.3]`–`[0.1,0]`; peg z = `peg_half_sizes[:,2]` (= radius); quat random z-rot, `bounds=(π/2 - π/3, π/2 + π/3)`, lock_x/lock_y.
- Box xy ∈ `[-0.05,0.2]`–`[0.05,0.4]`; box z = `peg_half_sizes[:,0]` (= half-length); quat random z-rot, `bounds=(π/2 - π/8, π/2 + π/8)`.
- Robot qpos base = `[0, π/8, 0, -5π/8, 0, 3π/4, -π/4, 0.04, 0.04]` + `_episode_rng.normal(0, 0.02)`; fingers reset to 0.04. Base pose `Pose([-0.615, 0, 0])`.

### Code
```python
def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
    with torch.device(self.device):
        b = len(env_idx)
        self.table_scene.initialize(env_idx)

        xy = randomization.uniform(
            low=torch.tensor([-0.1, -0.3]), high=torch.tensor([0.1, 0]), size=(b, 2))
        pos = torch.zeros((b, 3)); pos[:, :2] = xy; pos[:, 2] = self.peg_half_sizes[env_idx, 2]
        quat = randomization.random_quaternions(
            b, self.device, lock_x=True, lock_y=True,
            bounds=(np.pi / 2 - np.pi / 3, np.pi / 2 + np.pi / 3))
        self.peg.set_pose(Pose.create_from_pq(pos, quat))

        xy = randomization.uniform(
            low=torch.tensor([-0.05, 0.2]), high=torch.tensor([0.05, 0.4]), size=(b, 2))
        pos = torch.zeros((b, 3)); pos[:, :2] = xy; pos[:, 2] = self.peg_half_sizes[env_idx, 0]
        quat = randomization.random_quaternions(
            b, self.device, lock_x=True, lock_y=True,
            bounds=(np.pi / 2 - np.pi / 8, np.pi / 2 + np.pi / 8))
        self.box.set_pose(Pose.create_from_pq(pos, quat))

        qpos = np.array([0.0, np.pi / 8, 0, -np.pi * 5 / 8, 0, np.pi * 3 / 4, -np.pi / 4, 0.04, 0.04])
        qpos = self._episode_rng.normal(0, 0.02, (b, len(qpos))) + qpos
        qpos[:, -2:] = 0.04
        self.agent.robot.set_qpos(qpos)
        self.agent.robot.set_pose(sapien.Pose([-0.615, 0, 0]))
```

---

## §4 Goal + Termination (`evaluate`)

### Description
Success = the peg head is inserted into the hole: in the box-hole frame, `x >= -0.015` (past mid-depth) AND `|y| <= box_hole_radii` AND `|z| <= box_hole_radii`. No failure termination; episodes also end at `max_episode_steps=100` (truncation). `goal_pose` (cached after reset) = box hole pose composed with inverse peg-head offset.

### Decisions resolved
- x_flag: `-0.015 <= peg_head_pos_at_hole[:,0]`.
- y/z flags: within `±box_hole_radii` (= peg_radius + 0.003).
- `max_episode_steps = 100`.

### Code
```python
@property
def peg_head_pose(self):
    return self.peg.pose * self.peg_head_offsets

@property
def box_hole_pose(self):
    return self.box.pose * self.box_hole_offsets

@property
def goal_pose(self):
    return self.box.pose * self.box_hole_offsets * self.peg_head_offsets.inv()

def has_peg_inserted(self):
    peg_head_pos_at_hole = (self.box_hole_pose.inv() * self.peg_head_pose).p
    x_flag = -0.015 <= peg_head_pos_at_hole[:, 0]
    y_flag = (-self.box_hole_radii <= peg_head_pos_at_hole[:, 1]) & (
        peg_head_pos_at_hole[:, 1] <= self.box_hole_radii)
    z_flag = (-self.box_hole_radii <= peg_head_pos_at_hole[:, 2]) & (
        peg_head_pos_at_hole[:, 2] <= self.box_hole_radii)
    return (x_flag & y_flag & z_flag, peg_head_pos_at_hole)

def evaluate(self):
    success, peg_head_pos_at_hole = self.has_peg_inserted()
    return dict(success=success, peg_head_pos_at_hole=peg_head_pos_at_hole)
```

---

## §5 Observation (`_get_obs_extra`)

### Description
Default `obs_mode="state"`. Total obs = 43 (flattened `Box(-inf, inf, (1,43))`). Composed of base proprioception (`_get_obs_agent`: qpos 9 + qvel 9) plus `_get_obs_extra`. In `state`/`state_dict` mode the extra dict includes privileged peg/box geometry; `tcp_pose` is always present even outside state mode.

### Decisions resolved
- obs_mode default = `state` (SUPPORTED_OBS_MODES[0]); env SUPPORTED_OBS_MODES = `("state","state_dict","none","sensor_data","any_textures","pointcloud")`.
- Resolved 43-dim breakdown (state mode):
  - agent qpos = 9, agent qvel = 9 (Panda 7 arm + 2 finger joints) → 18
  - `tcp_pose` = 7 (xyz + quat)
  - `peg_pose` = 7
  - `peg_half_size` = 3
  - `box_hole_pose` = 7
  - `box_hole_radius` = 1
  - 18 + 7 + 7 + 3 + 7 + 1 = **43** ✓

### Code
```python
def _get_obs_extra(self, info: dict):
    obs = dict(tcp_pose=self.agent.tcp.pose.raw_pose)
    if self.obs_mode_struct.use_state:
        obs.update(
            peg_pose=self.peg.pose.raw_pose,
            peg_half_size=self.peg_half_sizes,
            box_hole_pose=self.box_hole_pose.raw_pose,
            box_hole_radius=self.box_hole_radii,
        )
    return obs
```

### Smoke
Covered by §1 build smoke (`observation_space == Box(-inf, inf, (1, 43), float32)`).

---

## §6 Reward

### Description
Composer = **sum** (dense reward is an additive accumulation of staged terms; the normalized variant divides the dense reward by 10). Four staged shaping terms, gated by grasp/alignment so later stages only contribute once prerequisites hold:
- **Stage 1/2 (reach + grasp):** `reaching_reward = 1 - tanh(4 * gripper_to_peg_dist)` toward a `-0.06 m` x-offset grasp pose on the peg, plus `is_grasped` (1.0 when `is_grasping(peg, max_angle=20)`). Range ≈ [0, 2].
- **Stage 3 (align):** `pre_insertion_reward = 3 * (1 - tanh(0.5*(d_head+d_peg) + 4.5*max(d_head,d_peg)))` on the yz-distance of peg head & center w.r.t. goal frame; multiplied by `is_grasped`. Range ≈ [0, 3]. `pre_inserted` flag when both yz-dists < 0.01.
- **Stage 4 (insert):** `insertion_reward = 5 * (1 - tanh(5 * ||peg_head_in_hole_frame||))`; multiplied by `(is_grasped & pre_inserted)`. Range ≈ [0, 5].
- **Success override:** `reward[info["success"]] = 10` (caps total at 10; normalized → 1.0).

Per-stage saturated per-step magnitudes (retro-computed from weights): reach ~1, grasp +1, align +3, insert +5, success-clamp 10. Max non-success dense return ≈ 11 but is hard-clamped to 10 on success.

### Decisions resolved
- Grasp x-offset = `sapien.Pose([-0.06, 0, 0])`; `is_grasping` `max_angle=20`.
- reaching coeff 4.0; pre_insertion weight 3, coeffs 0.5 / 4.5 (max-term); insertion weight 5, coeff 5.0.
- `pre_inserted` threshold 0.01 (both head & center yz).
- success reward = 10; normalized = dense / 10.
- Composer: **sum** (with multiplicative gates `is_grasped`, `is_grasped & pre_inserted`).

### Code (verbatim)
```python
def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: dict):
    # Stage 1: Encourage gripper to be rotated to be lined up with the peg

    # Stage 2: Encourage gripper to move close to peg tail and grasp it
    gripper_pos = self.agent.tcp.pose.p
    tgt_gripper_pose = self.peg.pose
    offset = sapien.Pose(
        [-0.06, 0, 0]
    )  # account for panda gripper width with a bit more leeway
    tgt_gripper_pose = tgt_gripper_pose * (offset)
    gripper_to_peg_dist = torch.linalg.norm(
        gripper_pos - tgt_gripper_pose.p, axis=1
    )

    reaching_reward = 1 - torch.tanh(4.0 * gripper_to_peg_dist)

    # check with max_angle=20 to ensure gripper isn't grasping peg at an awkward pose
    is_grasped = self.agent.is_grasping(self.peg, max_angle=20)
    reward = reaching_reward + is_grasped

    # Stage 3: Orient the grasped peg properly towards the hole

    # pre-insertion award, encouraging both the peg center and the peg head to match the yz coordinates of goal_pose
    peg_head_wrt_goal = self.goal_pose.inv() * self.peg_head_pose
    peg_head_wrt_goal_yz_dist = torch.linalg.norm(
        peg_head_wrt_goal.p[:, 1:], axis=1
    )
    peg_wrt_goal = self.goal_pose.inv() * self.peg.pose
    peg_wrt_goal_yz_dist = torch.linalg.norm(peg_wrt_goal.p[:, 1:], axis=1)

    pre_insertion_reward = 3 * (
        1
        - torch.tanh(
            0.5 * (peg_head_wrt_goal_yz_dist + peg_wrt_goal_yz_dist)
            + 4.5 * torch.maximum(peg_head_wrt_goal_yz_dist, peg_wrt_goal_yz_dist)
        )
    )
    reward += pre_insertion_reward * is_grasped
    # stage 3 passes if peg is correctly oriented in order to insert into hole easily
    pre_inserted = (peg_head_wrt_goal_yz_dist < 0.01) & (
        peg_wrt_goal_yz_dist < 0.01
    )

    # Stage 4: Insert the peg into the hole once it is grasped and lined up
    peg_head_wrt_goal_inside_hole = self.box_hole_pose.inv() * self.peg_head_pose
    insertion_reward = 5 * (
        1
        - torch.tanh(
            5.0 * torch.linalg.norm(peg_head_wrt_goal_inside_hole.p, axis=1)
        )
    )
    reward += insertion_reward * (is_grasped & pre_inserted)

    reward[info["success"]] = 10

    return reward

def compute_normalized_dense_reward(
    self, obs: Any, action: torch.Tensor, info: dict
):
    return self.compute_dense_reward(obs, action, info) / 10
```

### Smoke
Reward verified by source read; runtime check at reproduction time: step env with `reward_mode="dense"`, assert finite, non-constant, and `compute_normalized_dense_reward == compute_dense_reward / 10`.

---

## §7 DR

`<no DR>` — there is no `interval`/`startup` event-based domain randomization. All variation is (a) per-reconfiguration geometry sampling in `_load_scene` (peg length/radius, hole center, clearance) and (b) per-reset pose/qpos randomization in `_initialize_episode`. These belong to §1/§3, not §7. No physical-parameter (mass/friction/PD-gain) randomization is applied.

---

