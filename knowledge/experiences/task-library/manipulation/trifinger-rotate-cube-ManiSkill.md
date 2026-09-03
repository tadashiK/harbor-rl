# TriFingerRotateCubeLevel1-v1 — Implementation Spec

- robot: TriFingerPro three-finger hand (9 DoF, fixed base)
- simulator: ManiSkill (SAPIEN, `BaseEnv` + `@register_env`)
- objects: cube, table, circular arena wall
- bimanual: false
- summary: Rotate a cube to a target orientation with a three-finger manipulator.

---

## §1 Registration + Scene

### Description
The env registers five level variants over a single `RotateCubeEnv(BaseEnv)` class. The scene = a fixed ground plane, a fixed table (URDF, root link fixed), a static non-convex "high table boundary" wall mesh forming a circular arena, a dynamic red cube (the manipulated object), and a kinematic green cube acting as the visual goal marker (collision off, hidden from sensor observations via `_hidden_objects`). The robot is `TriFingerPro` (3 fingers × 3 joints = 9 DoF, fixed base) mounted above the cube.

### Decisions resolved
- `SUPPORTED_ROBOTS = ["trifingerpro"]`; `robot_uids="trifingerpro"`.
- `register_env("TriFingerRotateCubeLevel1-v1", max_episode_steps=250)`.
- `difficulty_level = 1` (Level1 subclass). Base default would be `4`.
- `robot_init_qpos_noise = 0.02`.
- `sim_config = SimConfig(gpu_memory_config=GPUMemoryConfig(found_lost_pairs_capacity=2**25, max_rigid_patch_count=2**18))`.
- Geometry constants: `goal_radius=0.02`, `cube_half_size=0.02` (unused in scene; actual cube built with `half_size=size/2`), `size=0.065` m, `ARENA_RADIUS=0.195`, `radius_3d = 0.065*sqrt(3)/2 ≈ 0.0563`, `max_com_distance_to_center = 0.195 - radius_3d ≈ 0.1387`, `min_height = 0.0325`, `max_height = 0.1`.
- Cube (`obj`): `actors.build_colorful_cube(half_size=size/2=0.0325, color=[169,42,12,255]/255 (red), body_type="dynamic", add_collision=True)`, name `"cube"`.
- Goal cube (`obj_goal`): same builder, `color=[12,160,42,255]/255 (green)`, `body_type="kinematic"`, `add_collision=False`, name `"cube_goal"`, appended to `self._hidden_objects`.
- Table URDF: `{PACKAGE_ASSET_DIR}/robots/trifinger/table_without_border.urdf` (loaded via `create_urdf_loader()`, `fix_root_link=True`).
- Arena wall mesh (static, nonconvex collision + visual): `{PACKAGE_ASSET_DIR}/robots/trifinger/robot_properties_fingers/meshes/high_table_boundary.stl`, name `"table2"`.
- Ground: `build_ground(self.scene, altitude=0)`.
- Robot: `{PACKAGE_ASSET_DIR}/robots/trifinger/trifingerpro.urdf`, fixed base, 9 active joints (see §2). Tip-link friction material `static=2.0, dynamic=1.0, restitution=0.0` on `finger_tip_link_{0,120,240}`. Robot base pose set at reset (see §3) to `p=[0,0,size/2+0.022]=[0,0,0.0545]`, `q=[1,0,0,0]`.
- Sensor camera (`base_camera`): `look_at(eye=(0.7,0,0.7), target=(0,0,0))`, 128×128, fov π/2.
- Human render camera (`render_camera`): same pose, 512×512, fov 1.0.

### Decisions resolved — asset paths
- `mani_skill/assets/robots/trifinger/table_without_border.urdf` ✓
- `mani_skill/assets/robots/trifinger/robot_properties_fingers/meshes/high_table_boundary.stl` ✓
- `mani_skill/assets/robots/trifinger/trifingerpro.urdf` ✓
(`PACKAGE_ASSET_DIR` = `mani_skill/assets`.)

### Code — env class header + sim cfg + scene
```python
@register_env("TriFingerRotateCubeLevel1-v1", max_episode_steps=250)
class RotateCubeEnvLevel1(RotateCubeEnv):
    def __init__(self, *args, **kwargs):
        super().__init__(
            *args,
            robot_init_qpos_noise=0.02,
            difficulty_level=1,
            **kwargs,
        )


class RotateCubeEnv(BaseEnv):
    SUPPORTED_ROBOTS = ["trifingerpro"]
    agent: TriFingerPro

    sim_config = SimConfig(
        gpu_memory_config=GPUMemoryConfig(
            found_lost_pairs_capacity=2**25, max_rigid_patch_count=2**18
        )
    )

    goal_radius = 0.02
    cube_half_size = 0.02
    ARENA_RADIUS = 0.195
    size = 0.065  # m
    max_len = 0.065
    radius_3d = max_len * np.sqrt(3) / 2
    max_com_distance_to_center = ARENA_RADIUS - radius_3d
    min_height = 0.065 / 2
    max_height = 0.1

    def __init__(self, *args, robot_uids="trifingerpro", robot_init_qpos_noise=0.02,
                 difficulty_level: int = 4, **kwargs):
        self.robot_init_qpos_noise = robot_init_qpos_noise
        if (not isinstance(difficulty_level, int) or difficulty_level >= 5 or difficulty_level < 0):
            raise ValueError(f"Difficulty level must be a int within 0-4, but get {difficulty_level}")
        self.difficulty_level = difficulty_level
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    @property
    def _default_sensor_configs(self):
        pose = sapien_utils.look_at(eye=(0.7, 0.0, 0.7), target=(0.0, 0.0, 0.0))
        return [CameraConfig("base_camera", pose, 128, 128, np.pi / 2, 0.01, 100)]

    @property
    def _default_human_render_camera_configs(self):
        pose = sapien_utils.look_at(eye=(0.7, 0.0, 0.7), target=(0.0, 0.0, 0.0))
        return CameraConfig("render_camera", pose, 512, 512, 1, 0.01, 100)

    def _load_scene(self, options: dict):
        self.ground = build_ground(self.scene, altitude=0)
        loader1 = self.scene.create_urdf_loader()
        loader1.fix_root_link = True
        loader1.name = "table"
        urdf_path = f"{PACKAGE_ASSET_DIR}/robots/trifinger/table_without_border.urdf"
        table: Articulation = loader1.load(urdf_path)

        builder: ActorBuilder = self.scene.create_actor_builder()
        high_table_boundary_file_name = f"{PACKAGE_ASSET_DIR}/robots/trifinger/robot_properties_fingers/meshes/high_table_boundary.stl"
        builder.add_nonconvex_collision_from_file(filename=high_table_boundary_file_name, scale=[1, 1, 1], material=None)
        builder.add_visual_from_file(filename=high_table_boundary_file_name)
        table_boundary: Actor = builder.build_static("table2")

        self.obj = actors.build_colorful_cube(
            self.scene, half_size=self.size / 2, color=np.array([169, 42, 12, 255]) / 255,
            name="cube", body_type="dynamic", add_collision=True)

        self.obj_goal = actors.build_colorful_cube(
            self.scene, half_size=self.size / 2, color=np.array([12, 160, 42, 255]) / 255,
            name="cube_goal", body_type="kinematic", add_collision=False)
        self._hidden_objects.append(self.obj_goal)
```

### Code — robot agent (TriFingerPro)
```python
@register_agent()
class TriFingerPro(BaseAgent):
    uid = "trifingerpro"
    urdf_path = f"{PACKAGE_ASSET_DIR}/robots/trifinger/trifingerpro.urdf"
    urdf_config = dict(
        _materials=dict(tip=dict(static_friction=2.0, dynamic_friction=1.0, restitution=0.0)),
        link=dict(
            finger_tip_link_0=dict(material="tip"),
            finger_tip_link_120=dict(material="tip"),
            finger_tip_link_240=dict(material="tip"),
        ),
    )
    sensor_configs = {}

    def __init__(self, *args, **kwargs):
        self.joint_names = [
            "finger_base_to_upper_joint_0", "finger_upper_to_middle_joint_0", "finger_middle_to_lower_joint_0",
            "finger_base_to_upper_joint_120", "finger_upper_to_middle_joint_120", "finger_middle_to_lower_joint_120",
            "finger_base_to_upper_joint_240", "finger_upper_to_middle_joint_240", "finger_middle_to_lower_joint_240",
        ]
        self.joint_stiffness = 1e2
        self.joint_damping = 1e1
        self.joint_force_limit = 2e1
        self.tip_link_names = ["finger_tip_link_0", "finger_tip_link_120", "finger_tip_link_240"]
        self.root_joint_names = ["finger_base_to_upper_joint_0", "finger_base_to_upper_joint_120", "finger_base_to_upper_joint_240"]
        super().__init__(*args, **kwargs)

    def _after_init(self):
        self.tip_links = get_objs_by_names(self.robot.get_links(), self.tip_link_names)
        self.root_joints = [self.robot.find_joint_by_name(n) for n in self.root_joint_names]
        self.root_joint_indices = get_active_joint_indices(self.robot, self.root_joint_names)

    @property
    def tip_poses(self):
        tip_poses = [vectorize_pose(link.pose, device=self.device) for link in self.tip_links]
        return torch.stack(tip_poses, dim=-1)   # shape (N, 7, 3)

    def tip_velocities(self):
        tip_velocities = [link.linear_velocity for link in self.tip_links]
        return torch.stack(tip_velocities, dim=-1)   # shape (N, 3, 3)
```

### Smoke (§1)
```bash
cd <repo> && .venv/bin/python -c "import gymnasium as gym, mani_skill; e=gym.make('TriFingerRotateCubeLevel1-v1'); print(e.observation_space, e.action_space); print(e.unwrapped.control_mode); e.close()"
```
Expected stdout:
```
Box(-inf, inf, (1, 62), float32) Box(-1.0, 1.0, (9,), float32)
pd_joint_delta_pos
```

---

## §2 Actions

### Description
ManiSkill controller-based action space. The agent exposes six controller configs; the env does not override the default, so the FIRST key wins: `pd_joint_delta_pos` — a PD joint-position controller commanding a per-step DELTA on the 9 active finger joints. The external (gym) action space is normalized to `[-1, 1]^9`; internally `pd_joint_delta_pos` maps `[-1,1] → [-0.1, 0.1]` rad delta per joint (no `normalize_action` flag on the delta config, so the lower/upper -0.1/0.1 define the rescale range). Stiffness 100, damping 10, force limit 20 per joint.

### Decisions resolved
- `control_mode = "pd_joint_delta_pos"` (default = `supported_control_modes[0]`).
- Action dim = 9 (3 fingers × 3 joints). Action range `[-1, 1]`.
- Delta joint limits: `lower=-0.1`, `upper=0.1` rad.
- PD gains: stiffness `1e2`, damping `1e1`, force limit `2e1`.
- Other available control modes (not default): `pd_joint_pos` (absolute, `normalize_action=False`), `pd_joint_target_delta_pos`, `pd_joint_vel`, `pd_joint_pos_vel`, `pd_joint_delta_pos_vel`.

### Code — `_controller_configs`
```python
@property
def _controller_configs(self):
    joint_pos = PDJointPosControllerConfig(
        self.joint_names, None, None,
        self.joint_stiffness, self.joint_damping, self.joint_force_limit,
        normalize_action=False)
    joint_delta_pos = PDJointPosControllerConfig(
        self.joint_names, -0.1, 0.1,
        self.joint_stiffness, self.joint_damping, self.joint_force_limit,
        use_delta=True)
    joint_target_delta_pos = deepcopy(joint_delta_pos)
    joint_target_delta_pos.use_target = True
    pd_joint_vel = PDJointVelControllerConfig(
        self.joint_names, -1.0, 1.0, self.joint_damping, self.joint_force_limit)
    joint_pos_vel = PDJointPosVelControllerConfig(
        self.joint_names, None, None,
        self.joint_stiffness, self.joint_damping, self.joint_force_limit,
        normalize_action=False)
    joint_delta_pos_vel = PDJointPosVelControllerConfig(
        self.joint_names, -0.1, 0.1,
        self.joint_stiffness, self.joint_damping, self.joint_force_limit,
        use_delta=True)
    controller_configs = dict(
        pd_joint_delta_pos=dict(joint=joint_delta_pos),
        pd_joint_pos=dict(joint=joint_pos),
        pd_joint_target_delta_pos=dict(joint=joint_target_delta_pos),
        pd_joint_vel=dict(joint=pd_joint_vel),
        pd_joint_pos_vel=dict(joint=joint_pos_vel),
        pd_joint_delta_pos_vel=dict(joint=joint_delta_pos_vel),
    )
    return deepcopy_dict(controller_configs)
```

### Smoke (§2)
```bash
cd <repo> && .venv/bin/python -c "import gymnasium as gym, mani_skill, numpy as np; e=gym.make('TriFingerRotateCubeLevel1-v1'); e.reset(seed=0); o,r,te,tr,i=e.step(e.action_space.sample()); print('act_dim', e.action_space.shape, 'reward', float(np.asarray(r).reshape(-1)[0])); e.close()"
```
Expected: `act_dim (9,)` and a finite scalar reward.

---

## §3 Reset (`_initialize_episode`)

### Description
On reset, `_initialize_episode` calls `_initialize_actors` then `_initialize_agent`. Actors: the manipulated cube is placed at the arena center, resting on the table at `z = size/2 + 0.005`, identity orientation. The goal cube pose is sampled by `_sample_object_goal_poses(difficulty=1)` — for Level1: random (x,y) uniformly within the arena circle, `z = size/2` (on the table), and a yaw-only random quaternion (roll/pitch locked). `prev_norms` reward-latch buffer is reset to `None`. Agent: 9-DoF qpos set to zero plus Gaussian noise (`std = robot_init_qpos_noise = 0.02`); robot base pose fixed at `p=[0,0,size/2+0.022]`, `q=[1,0,0,0]`.

### Decisions resolved
- Cube spawn: `p=[0, 0, size/2 + 0.005] = [0,0,0.0375]`, `q=[1,0,0,0]`.
- Goal (Level1, difficulty=1): `random_xy()` → radius `= sqrt(U(0,1)) * max_com_distance_to_center`, theta `= 2π·U(0,1)`; `pos_z = size/2 = 0.0325`; `orientation = random_quaternions(b, lock_x=True, lock_y=True)` (yaw only).
- Agent init qpos: `zeros(b,9) + randn(b,9)*0.02`; robot base `p=[0,0,0.0545]`, `q=[1,0,0,0]`.

### Code
```python
def _initialize_episode(self, env_idx, options):
    self._initialize_actors(env_idx)
    self._initialize_agent(env_idx)

def _initialize_actors(self, env_idx):
    with torch.device(self.device):
        b = len(env_idx)
        xyz = torch.zeros((b, 3))
        xyz[..., 2] = self.size / 2 + 0.005
        obj_pose = Pose.create_from_pq(p=xyz, q=[1, 0, 0, 0])
        self.obj.set_pose(obj_pose)
        pos, orn = self._sample_object_goal_poses(env_idx, difficulty=self.difficulty_level)
        self.obj_goal.set_pose(Pose.create_from_pq(p=pos, q=orn))
        self.prev_norms = None

def _initialize_agent(self, env_idx):
    with torch.device(self.device):
        b = len(env_idx)
        dof = self.agent.robot.dof
        if isinstance(dof, torch.Tensor):
            dof = dof[0]
        init_qpos = torch.zeros((b, dof))
        init_qpos += torch.randn((b, dof)) * self.robot_init_qpos_noise
        self.agent.reset(init_qpos)
        self.agent.robot.set_pose(
            Pose.create_from_pq(
                torch.tensor([0.0, 0, self.size / 2 + 0.022]),
                torch.tensor([1, 0, 0, 0]),
            )
        )

def _sample_object_goal_poses(self, env_idx, difficulty):
    b = len(env_idx)
    default_orn = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float, device=self.device).repeat(b, 1)

    def random_xy():
        radius = torch.sqrt(torch.rand(b, dtype=torch.float, device=self.device))
        radius *= self.max_com_distance_to_center
        theta = 2 * np.pi * torch.rand(b, dtype=torch.float, device=self.device)
        x = radius * torch.cos(theta)
        y = radius * torch.sin(theta)
        return x, y

    def random_z(min_height, max_height):
        z = torch.rand(b, dtype=torch.float, device=self.device)
        z = (max_height - min_height) * z + min_height
        return z

    if difficulty == 0:
        pos_x, pos_y = random_xy(); pos_z = self.size / 2; orientation = default_orn
    elif difficulty == 1:                                  # <-- Level1
        pos_x, pos_y = random_xy(); pos_z = self.size / 2
        orientation = random_quaternions(b, lock_x=True, lock_y=True, device=self.device)
    elif difficulty == 2:
        pos_x, pos_y = 0.0, 0.0; pos_z = self.min_height + 0.05; orientation = default_orn
    elif difficulty == 3:
        pos_x, pos_y = random_xy(); pos_z = random_z(self.min_height, self.max_height); orientation = default_orn
    elif difficulty == 4:
        pos_x, pos_y = random_xy(); pos_z = random_z(self.radius_3d, self.max_height)
        orientation = random_quaternions(b, device=self.device)
    else:
        raise ValueError(f"Invalid difficulty index for task: {difficulty}.")

    pos_tensor = torch.zeros((b, 3), dtype=torch.float, device=self.device)
    pos_tensor[:, 0] = pos_x; pos_tensor[:, 1] = pos_y; pos_tensor[:, 2] = pos_z
    return pos_tensor, orientation
```

### Smoke (§3)
```bash
cd <repo> && .venv/bin/python -c "import gymnasium as gym, mani_skill, torch; e=gym.make('TriFingerRotateCubeLevel1-v1'); e.reset(seed=1); e.reset(seed=2); u=e.unwrapped; print('goal varies across resets:', u.obj_goal.pose.p); e.close()"
```
Expected: goal position differs between resets (random xy + yaw).

---

## §4 Goal + Termination (`evaluate` + `max_episode_steps`)

### Description
`evaluate()` defines success: the cube position is within `goal_radius = 0.02` m of the goal AND the quaternion geodesic angle to the goal is `< 0.1` rad. Returns `{"success": bool tensor}`. There is NO explicit failure/termination term beyond success; episodes truncate at `max_episode_steps = 250`. The success flag also short-circuits the reward to its max (15).

### Decisions resolved
- Success: `‖obj_p − goal_p‖₂ < 0.02` AND `quat_diff_rad(obj_q, goal_q) < 0.1`.
- `max_episode_steps = 250` (set in `@register_env`).
- No separate fail termination; `terminated = success`, `truncated` at step 250.

### Code
```python
def evaluate(self):
    obj_p = self.obj.pose.p
    goal_p = self.obj_goal.pose.p
    obj_q = self.obj.pose.q
    goal_q = self.obj_goal.pose.q
    is_obj_pos_close_to_goal = torch.linalg.norm(obj_p - goal_p, axis=1) < self.goal_radius
    is_obj_q_close_to_goal = common.quat_diff_rad(obj_q, goal_q) < 0.1
    is_success = is_obj_pos_close_to_goal & is_obj_q_close_to_goal
    return {"success": is_success}
```

### Smoke (§4)
```bash
cd <repo> && .venv/bin/python -c "import gymnasium as gym, mani_skill; e=gym.make('TriFingerRotateCubeLevel1-v1'); e.reset(seed=0); o,r,te,tr,i=e.step(e.action_space.sample()); print('success' in i, 'terminated', te); e.close()"
```
Expected: `True` (info has `success`), `terminated` False at step 1.

---

## §5 Observation (`_get_obs_extra` + proprioception)

### Description
Total flat obs dim = 62 in default `obs_mode="state"`. Composed of (a) agent proprioception from `TriFingerPro.get_proprioception()` and (b) task `_get_obs_extra`. Proprioception = base `qpos(9)` + `qvel(9)` plus the agent override adding `tip_poses(21)` (3 fingers × 7-dim pose) and `tip_velocities(9)` (3 fingers × 3-dim lin vel) = 48. Extra always includes `goal_pos(3)` + `goal_q(4)`; in a state obs mode it additionally includes the cube's own pose `obj_p(3)` + `obj_q(4)` = 14. 48 + 14 = 62. (`pd_joint_delta_pos` controller carries no extra controller state, so no controller obs term.)

### Decisions resolved
- `obs_mode = "state"` (default). Total dim 62, layout:
  - proprio: `qpos` 9, `qvel` 9, `tip_poses` 21, `tip_velocities` 9  → 48
  - extra: `goal_pos` 3, `goal_q` 4, `obj_p` 3, `obj_q` 4  → 14
- Goal pose is always observable; cube pose `obj_p`/`obj_q` only in state-based modes (`use_state`).
- No observation noise terms defined.

### Code
```python
# task extra obs
def _get_obs_extra(self, info: dict):
    obs = dict(
        goal_pos=self.obj_goal.pose.p,
        goal_q=self.obj_goal.pose.q,
    )
    if self.obs_mode_struct.use_state:
        obs.update(
            obj_p=self.obj.pose.p,
            obj_q=self.obj.pose.q,
        )
    return obs

# agent proprioception override (TriFingerPro)
def get_proprioception(self):
    obs = super().get_proprioception()          # {qpos(9), qvel(9)}
    obs.update({"tip_poses": self.tip_poses.view(-1, 21)})
    obs.update({"tip_velocities": self.tip_velocities().view(-1, 9)})
    return obs
```

### Smoke (§5)
```bash
cd <repo> && .venv/bin/python -c "import gymnasium as gym, mani_skill; e=gym.make('TriFingerRotateCubeLevel1-v1'); print(e.observation_space); e.close()"
```
Expected: `Box(-inf, inf, (1, 62), float32)`.

---

## §6 Reward (`compute_dense_reward` + `compute_normalized_dense_reward`)

### Description
Composer = SUM (additive). The dense reward is a sum of two groups:
1. **finger_reach_object_reward** — encourages all 3 fingertips to touch the cube. For each finger, `1 − tanh(5·‖tip_pos − obj_pos‖)`; averaged over 3 fingers and scaled by `object_dist_weight=5`.
2. **pose_reward** — drives the cube toward the goal pose, composed of:
   - `object_dist_reward = (1 − tanh(5·‖obj−goal‖)) − (1 − tanh(5·‖init_xyz − goal‖))` (baseline-subtracted full-3D distance; baseline anchor `init_xyz=[0,0,0.032]`).
   - `object_lift_reward = 5·(1 − tanh(5·|obj_z − goal_z|)) − 5·(1 − tanh(5·|init_z − goal_z|))` (baseline-subtracted height, anchor z=0.032).
   - `object_rot_reward = −|quat_diff_rad(obj_q, goal_q)|` (orientation penalty).
   - `pose_reward = object_dist_weight·(object_dist_reward + object_lift_reward) + object_rot_weight·object_rot_reward`, with both weights = 5.
`total_reward = finger_reach_object_reward + pose_reward`, clamped to `[−15, 15]`; on success forced to `+15`. Normalized variant: `dense/(2·15) + 0.5` → roughly `[0,1]`. Default `reward_mode = "normalized_dense"` (`SUPPORTED_REWARD_MODES[0]`).

### Per-stage saturated per-step magnitudes (retro-computed)
- finger reach: each finger reward ∈ [0,1]; group = `5·(mean of 3) ∈ [0,5]`.
- object_dist_reward (after baseline subtraction) ∈ roughly `[−1, 1]`; ×5 → `[−5,5]`.
- object_lift_reward (after baseline subtraction) ∈ roughly `[−5, 5]`; ×5 → `[−25,25]` pre-clamp.
- object_rot_reward = `−|angle|` ∈ `[−π, 0]`; ×5 → `[−5π, 0] ≈ [−15.7, 0]`.
- total clamped to `[−15, 15]`; success → exactly `+15`. Normalized: `[0, 1]`, success ≈ 1.0.

### Code (verbatim, full)
```python
def compute_dense_reward(self, obs: Any, action: Array, info: dict):
    obj_pos = self.obj.pose.p
    obj_q = self.obj.pose.q
    goal_pos = self.obj_goal.pose.p
    goal_q = self.obj_goal.pose.q

    object_dist_weight = 5
    object_rot_weight = 5

    # Reward penalising finger movement
    tip_poses = self.agent.tip_poses
    # shape (N, 3 + 4, 3 fingers)

    finger_reach_object_dist_1 = torch.norm(tip_poses[:, :3, 0] - obj_pos, p=2, dim=-1)
    finger_reach_object_dist_2 = torch.norm(tip_poses[:, :3, 1] - obj_pos, p=2, dim=-1)
    finger_reach_object_dist_3 = torch.norm(tip_poses[:, :3, 2] - obj_pos, p=2, dim=-1)
    finger_reach_object_reward1 = 1 - torch.tanh(5 * finger_reach_object_dist_1)
    finger_reach_object_reward2 = 1 - torch.tanh(5 * finger_reach_object_dist_2)
    finger_reach_object_reward3 = 1 - torch.tanh(5 * finger_reach_object_dist_3)
    finger_reach_object_reward = (
        object_dist_weight
        * (finger_reach_object_reward1 + finger_reach_object_reward2 + finger_reach_object_reward3)
        / 3
    )

    # Reward for object distance
    object_dist = torch.norm(obj_pos - goal_pos, p=2, dim=-1)

    init_xyz_tensor = torch.tensor([0, 0, 0.032], dtype=torch.float, device=self.device).reshape(1, 3)
    init_z_dist = torch.norm(init_xyz_tensor - goal_pos[..., ], p=2, dim=-1)

    # object_dist_reward = object_dist_weight * dt * lgsk_kernel(object_dist, scale=50., eps=2.)
    object_dist_reward = 1 - torch.tanh(5 * object_dist)
    object_init_dist_reward = 1 - torch.tanh(5 * init_z_dist)
    object_dist_reward -= object_init_dist_reward

    init_z_tensor = torch.tensor([0.032], dtype=torch.float, device=self.device).reshape(1, 1)
    object_z_dist = torch.norm(obj_pos[..., 2:3] - goal_pos[..., 2:3], p=2, dim=-1)
    init_z_dist = torch.norm(init_z_tensor - goal_pos[..., 2:3], p=2, dim=-1)
    object_lift_reward = 5 * ((1 - torch.tanh(5 * object_z_dist)))
    object_init_z_reward = 5 * ((1 - torch.tanh(5 * init_z_dist)))
    object_lift_reward -= object_init_z_reward

    # extract quaternion orientation
    angles = common.quat_diff_rad(obj_q, goal_q)
    object_rot_reward = -1 * torch.abs(angles)
    pose_reward = (
        object_dist_weight * (object_dist_reward + object_lift_reward)
        + object_rot_weight * object_rot_reward
    )
    total_reward = finger_reach_object_reward + pose_reward
    total_reward = total_reward.clamp(-15, 15)
    total_reward[info["success"]] = 15
    return total_reward

def compute_normalized_dense_reward(self, obs: Any, action: Array, info: dict):
    self.max_reward = 15
    dense_reward = self.compute_dense_reward(obs=obs, action=action, info=info)
    norm_dense_reward = dense_reward / (2 * self.max_reward) + 0.5
    return norm_dense_reward
```

Helper referenced: `mani_skill.utils.common.quat_diff_rad(q1, q2)` — geodesic angle between two quaternions (built-in to ManiSkill, resolves; no missing helper). `self.agent.tip_poses` is the `(N, 7, 3)` property defined in §1's agent code.

### Smoke (§6)
```bash
cd <repo> && .venv/bin/python -c "import gymnasium as gym, mani_skill, numpy as np; e=gym.make('TriFingerRotateCubeLevel1-v1', reward_mode='dense'); e.reset(seed=0); o,r,te,tr,i=e.step(e.action_space.sample()); v=float(np.asarray(r).reshape(-1)[0]); print('finite', np.isfinite(v), 'in_range', -15<=v<=15); e.close()"
```
Expected: `finite True in_range True`.

---

## §7 DR

`<no DR>`

No `startup`/`interval` randomization events. The only stochasticity is reset-time: goal-pose sampling (§3) and the `robot_init_qpos_noise=0.02` Gaussian on initial joint positions. No physics/material/mass/visual domain randomization registered on this env.

---

