# PickCube-v1 — Implementation Spec

- robot: Franka Panda (default; Fetch / xArm6-Robotiq / SO100 / WidowXAI also supported)
- simulator: ManiSkill (SAPIEN, `BaseEnv` + `@register_env`)
- objects: red cube, green goal sphere, table
- bimanual: false
- summary: Grasp a cube and move it to a goal position marked by a sphere.

> ManiSkill maps the IsaacLab §1..§7 sections onto methods of a single `BaseEnv` subclass (`PickCubeEnv`) rather than onto manager Cfg dataclasses. The mapping used below:
> - §1 = `@register_env` + `SUPPORTED_ROBOTS` + `__init__`/cfg + `_load_agent` + `_load_scene` + sim/scene/camera config
> - §2 = the agent's `_controller_configs` (control_mode) → resulting action space
> - §3 = `_initialize_episode`
> - §4 = `evaluate()` + `max_episode_steps`
> - §5 = `_get_obs_extra` + the default `_get_obs_agent` (proprioception) → resolved obs vector
> - §6 = `compute_dense_reward` + `compute_normalized_dense_reward` (composer = SUM of staged terms)
> - §7 = domain randomization (only episode-init pose randomization; no physical-property DR)

---

## §1 Registration + Scene

### Description
A tabletop pick-and-place env: a Franka **Panda** arm (default) must grasp a red cube and move it to a goal position marked by a green (collision-free, hidden-from-obs) sphere. The env supports 5 robots (panda / fetch / xarm6_robotiq / so100 / widowxai); per-robot geometry and camera params come from `PICK_CUBE_CONFIGS`. Table top is at world z = 0 (`TableSceneBuilder`). The agent base is offset to `p=[-0.615, 0, 0]`.

### Decisions resolved (panda defaults)
- `id = "PickCube-v1"`, `max_episode_steps = 50`
- `robot_uids = "panda"` (default), `robot_init_qpos_noise = 0.02`
- `cube_half_size = 0.02` (red cube, color `[1,0,0,1]`)
- `goal_thresh = 0.025` (also the green goal-sphere radius)
- `cube_spawn_half_size = 0.1`, `cube_spawn_center = (0, 0)`
- `max_goal_height = 0.3`
- agent base pose: `sapien.Pose(p=[-0.615, 0, 0])`
- table top at z = 0; cube initial_pose `p=[0,0,cube_half_size]`; goal_site is `kinematic`, `add_collision=False`, appended to `_hidden_objects` (excluded from rendered obs / hidden during sensor capture)
- sensor cam (`base_camera`): 128×128, fov π/2, eye `[0.3,0,0.6]` → target `[-0.1,0,0.1]`
- human render cam (`render_camera`): 512×512, fov 1, eye `[0.6,0.7,0.6]` → target `[0,0,0.35]`
- default `obs_mode = "state"` (`BaseEnv.SUPPORTED_OBS_MODES[0]`), default `reward_mode = SUPPORTED_REWARD_MODES[0]`

### Code

`@register_env` + class attrs + `__init__` + cameras + `_load_agent` + `_load_scene`:
```python
@register_env("PickCube-v1", max_episode_steps=50)
class PickCubeEnv(BaseEnv):

    _sample_video_link = "https://github.com/haosulab/ManiSkill/raw/main/figures/environment_demos/PickCube-v1_rt.mp4"
    SUPPORTED_ROBOTS = [
        "panda",
        "fetch",
        "xarm6_robotiq",
        "so100",
        "widowxai",
    ]
    agent: Union[Panda, Fetch, XArm6Robotiq, SO100, WidowXAI]
    goal_thresh = 0.025
    cube_spawn_half_size = 0.05
    cube_spawn_center = (0, 0)

    def __init__(self, *args, robot_uids="panda", robot_init_qpos_noise=0.02, **kwargs):
        self.robot_init_qpos_noise = robot_init_qpos_noise
        if robot_uids in PICK_CUBE_CONFIGS:
            cfg = PICK_CUBE_CONFIGS[robot_uids]
        else:
            cfg = PICK_CUBE_CONFIGS["panda"]
        self.cube_half_size = cfg["cube_half_size"]
        self.goal_thresh = cfg["goal_thresh"]
        self.cube_spawn_half_size = cfg["cube_spawn_half_size"]
        self.cube_spawn_center = cfg["cube_spawn_center"]
        self.max_goal_height = cfg["max_goal_height"]
        self.sensor_cam_eye_pos = cfg["sensor_cam_eye_pos"]
        self.sensor_cam_target_pos = cfg["sensor_cam_target_pos"]
        self.human_cam_eye_pos = cfg["human_cam_eye_pos"]
        self.human_cam_target_pos = cfg["human_cam_target_pos"]
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    @property
    def _default_sensor_configs(self):
        pose = sapien_utils.look_at(
            eye=self.sensor_cam_eye_pos, target=self.sensor_cam_target_pos
        )
        return [CameraConfig("base_camera", pose, 128, 128, np.pi / 2, 0.01, 100)]

    @property
    def _default_human_render_camera_configs(self):
        pose = sapien_utils.look_at(
            eye=self.human_cam_eye_pos, target=self.human_cam_target_pos
        )
        return CameraConfig("render_camera", pose, 512, 512, 1, 0.01, 100)

    def _load_agent(self, options: dict):
        super()._load_agent(options, sapien.Pose(p=[-0.615, 0, 0]))

    def _load_scene(self, options: dict):
        self.table_scene = TableSceneBuilder(
            self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.table_scene.build()
        self.cube = actors.build_cube(
            self.scene,
            half_size=self.cube_half_size,
            color=[1, 0, 0, 1],
            name="cube",
            initial_pose=sapien.Pose(p=[0, 0, self.cube_half_size]),
        )
        self.goal_site = actors.build_sphere(
            self.scene,
            radius=self.goal_thresh,
            color=[0, 1, 0, 1],
            name="goal_site",
            body_type="kinematic",
            add_collision=False,
            initial_pose=sapien.Pose(),
        )
        self._hidden_objects.append(self.goal_site)
```

`PICK_CUBE_CONFIGS["panda"]` (from `pick_cube_cfgs.py`):
```python
"panda": {
    "cube_half_size": 0.02,
    "goal_thresh": 0.025,
    "cube_spawn_half_size": 0.1,
    "cube_spawn_center": (0, 0),
    "max_goal_height": 0.3,
    "sensor_cam_eye_pos": [0.3, 0, 0.6],
    "sensor_cam_target_pos": [-0.1, 0, 0.1],
    "human_cam_eye_pos": [0.6, 0.7, 0.6],
    "human_cam_target_pos": [0.0, 0.0, 0.35],
},
```

Robot asset (Panda): `urdf_path = f"{PACKAGE_ASSET_DIR}/robots/panda/panda_v2.urdf"` → resolved `mani_skill/assets/robots/panda/panda_v2.urdf`. Rest keyframe qpos = `[0, π/8, 0, -5π/8, 0, 3π/4, π/4, 0.04, 0.04]`. EE link = `panda_hand_tcp`. Gripper friction material: static=dynamic=2.0.

Table asset: `mani_skill/utils/scene_builder/table/assets/table.glb`, kinematic, top surface at world z = 0, `initial_pose p=[-0.12,0,-0.9196429]`.

### Smoke (§1 build)
```bash
cd <ManiSkill-repo>
.venv/bin/python -c "import gymnasium as gym, mani_skill; e=gym.make('PickCube-v1'); print(e.observation_space, e.action_space); e.close()"
```
Expected stdout:
```
Box(-inf, inf, (1, 42), float32) Box(-1.0, 1.0, (8,), float32)
```

---

## §2 Actions

### Description
Action comes from the agent's selected `control_mode`. The default (first key of `Panda._controller_configs`) is **`pd_joint_delta_pos`**: 7 normalized arm joint-position deltas (each clamped to ±0.1 rad before adding to current target) + 1 gripper command driving a mimic joint pair. Action space is `Box(-1, 1, (8,))`.

### Decisions resolved
- `control_mode = "pd_joint_delta_pos"` (default)
- arm: `PDJointPosControllerConfig`, `use_delta=True`, `lower=-0.1`, `upper=0.1` rad, `stiffness=1e3`, `damping=1e2`, `force_limit=100`, over the 7 `panda_joint{1..7}`
- gripper: `PDJointPosMimicControllerConfig` over `panda_finger_joint{1,2}`, `lower=-0.01`, `upper=0.04`, `stiffness=1e3`, `damping=1e2`, `force_limit=100`, mimic `panda_finger_joint2 -> panda_finger_joint1` → 1 action dim
- total action dim = 7 + 1 = 8, range [-1, 1]
- other available control modes (full list from Panda): `pd_joint_pos`, `pd_ee_delta_pos`, `pd_ee_delta_pose`, `pd_ee_pose`, `pd_joint_target_delta_pos`, `pd_ee_target_delta_pos`, `pd_ee_target_delta_pose`, `pd_joint_vel`, `pd_joint_pos_vel`, `pd_joint_delta_pos_vel`

### Code (Panda `_controller_configs`, verbatim)
```python
@property
def _controller_configs(self):
    arm_pd_joint_pos = PDJointPosControllerConfig(
        self.arm_joint_names, lower=None, upper=None,
        stiffness=self.arm_stiffness, damping=self.arm_damping,
        force_limit=self.arm_force_limit, normalize_action=False,
    )
    arm_pd_joint_delta_pos = PDJointPosControllerConfig(
        self.arm_joint_names, lower=-0.1, upper=0.1,
        stiffness=self.arm_stiffness, damping=self.arm_damping,
        force_limit=self.arm_force_limit, use_delta=True,
    )
    arm_pd_joint_target_delta_pos = deepcopy(arm_pd_joint_delta_pos)
    arm_pd_joint_target_delta_pos.use_target = True

    arm_pd_ee_delta_pos = PDEEPosControllerConfig(
        joint_names=self.arm_joint_names, pos_lower=-0.1, pos_upper=0.1,
        stiffness=self.arm_stiffness, damping=self.arm_damping,
        force_limit=self.arm_force_limit, ee_link=self.ee_link_name,
        urdf_path=self.urdf_path,
    )
    arm_pd_ee_delta_pose = PDEEPoseControllerConfig(
        joint_names=self.arm_joint_names, pos_lower=-0.1, pos_upper=0.1,
        rot_lower=-0.1, rot_upper=0.1,
        stiffness=self.arm_stiffness, damping=self.arm_damping,
        force_limit=self.arm_force_limit, ee_link=self.ee_link_name,
        urdf_path=self.urdf_path,
    )
    arm_pd_ee_pose = PDEEPoseControllerConfig(
        joint_names=self.arm_joint_names, pos_lower=-2.0, pos_upper=2.0,
        stiffness=self.arm_stiffness, damping=self.arm_damping,
        force_limit=self.arm_force_limit, ee_link=self.ee_link_name,
        urdf_path=self.urdf_path, use_delta=False, normalize_action=False,
    )
    arm_pd_ee_target_delta_pos = deepcopy(arm_pd_ee_delta_pos)
    arm_pd_ee_target_delta_pos.use_target = True
    arm_pd_ee_target_delta_pose = deepcopy(arm_pd_ee_delta_pose)
    arm_pd_ee_target_delta_pose.use_target = True

    arm_pd_joint_vel = PDJointVelControllerConfig(
        self.arm_joint_names, -1.0, 1.0, self.arm_damping, self.arm_force_limit,
    )
    arm_pd_joint_pos_vel = PDJointPosVelControllerConfig(
        self.arm_joint_names, None, None, self.arm_stiffness,
        self.arm_damping, self.arm_force_limit, normalize_action=False,
    )
    arm_pd_joint_delta_pos_vel = PDJointPosVelControllerConfig(
        self.arm_joint_names, -0.1, 0.1, self.arm_stiffness,
        self.arm_damping, self.arm_force_limit, use_delta=True,
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
        pd_ee_pose=dict(arm=arm_pd_ee_pose, gripper=gripper_pd_joint_pos),
        pd_joint_target_delta_pos=dict(arm=arm_pd_joint_target_delta_pos, gripper=gripper_pd_joint_pos),
        pd_ee_target_delta_pos=dict(arm=arm_pd_ee_target_delta_pos, gripper=gripper_pd_joint_pos),
        pd_ee_target_delta_pose=dict(arm=arm_pd_ee_target_delta_pose, gripper=gripper_pd_joint_pos),
        pd_joint_vel=dict(arm=arm_pd_joint_vel, gripper=gripper_pd_joint_pos),
        pd_joint_pos_vel=dict(arm=arm_pd_joint_pos_vel, gripper=gripper_pd_joint_pos),
        pd_joint_delta_pos_vel=dict(arm=arm_pd_joint_delta_pos_vel, gripper=gripper_pd_joint_pos),
    )
    return deepcopy_dict(controller_configs)
```
Panda actuator constants: `arm_stiffness=1e3`, `arm_damping=1e2`, `arm_force_limit=100`; `gripper_stiffness=1e3`, `gripper_damping=1e2`, `gripper_force_limit=100`.

### Smoke (§2)
The `action_space` line from §1's build is the §2 verification: `Box(-1.0, 1.0, (8,), float32)`.

---

## §3 Reset

### Description
On reset, the table scene re-initializes the robot to the `rest` keyframe qpos with uniform noise of magnitude `robot_init_qpos_noise=0.02` (via `TableSceneBuilder.initialize`). The cube is dropped at a uniform-random xy in a 0.2 m × 0.2 m square centered at `cube_spawn_center=(0,0)` (i.e. xy ∈ ±0.1 m), z fixed at `cube_half_size`, with a random z-axis-only rotation (x/y euler locked to 0). The goal sphere xy is randomized in the same square; goal z = cube_z + uniform(0, max_goal_height=0.3).

### Decisions resolved
- cube xy: `U(-cube_spawn_half_size, +cube_spawn_half_size) = U(-0.1, 0.1)` per axis, offset by `cube_spawn_center=(0,0)`
- cube z = `cube_half_size = 0.02`; cube orientation = `random_quaternions(b, lock_x=True, lock_y=True)` → random yaw, euler convention "XYZ", bounds (0, 2π)
- goal xy: same `U(-0.1, 0.1)` square; goal z = `U(0, max_goal_height=0.3) + cube_z`
- robot reset: `rest` keyframe `[0, π/8, 0, -5π/8, 0, 3π/4, π/4, 0.04, 0.04]` + uniform noise 0.02 (applied inside `TableSceneBuilder.initialize`)

### Code (`_initialize_episode`, verbatim)
```python
def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
    with torch.device(self.device):
        b = len(env_idx)
        self.table_scene.initialize(env_idx)
        xyz = torch.zeros((b, 3))
        xyz[:, :2] = (
            torch.rand((b, 2)) * self.cube_spawn_half_size * 2
            - self.cube_spawn_half_size
        )
        xyz[:, 0] += self.cube_spawn_center[0]
        xyz[:, 1] += self.cube_spawn_center[1]

        xyz[:, 2] = self.cube_half_size
        qs = randomization.random_quaternions(b, lock_x=True, lock_y=True)
        self.cube.set_pose(Pose.create_from_pq(xyz, qs))

        goal_xyz = torch.zeros((b, 3))
        goal_xyz[:, :2] = (
            torch.rand((b, 2)) * self.cube_spawn_half_size * 2
            - self.cube_spawn_half_size
        )
        goal_xyz[:, 0] += self.cube_spawn_center[0]
        goal_xyz[:, 1] += self.cube_spawn_center[1]
        goal_xyz[:, 2] = torch.rand((b)) * self.max_goal_height + xyz[:, 2]
        self.goal_site.set_pose(Pose.create_from_pq(goal_xyz))
```

Helper `random_quaternions` (verbatim, `mani_skill/envs/utils/randomization/pose.py`):
```python
def random_quaternions(n, device=None, lock_x=False, lock_y=False, lock_z=False, bounds=(0, np.pi * 2)):
    dist = bounds[1] - bounds[0]
    xyz_angles = torch.rand((n, 3), device=device) * (dist) + bounds[0]
    if lock_x:
        xyz_angles[:, 0] *= 0
    if lock_y:
        xyz_angles[:, 1] *= 0
    if lock_z:
        xyz_angles[:, 2] *= 0
    return matrix_to_quaternion(euler_angles_to_matrix(xyz_angles, convention="XYZ"))
```

### Smoke (§3)
Reset twice and confirm cube + goal poses differ across resets (poses live on `env.unwrapped.cube.pose.p` / `env.unwrapped.goal_site.pose.p`).

---

## §4 Goal + Termination

### Description
Success = cube within `goal_thresh` (0.025 m) euclidean distance of the goal AND robot static (max |qvel| over arm joints ≤ 0.2). Episode is otherwise length-bounded at `max_episode_steps=50` (time-limit truncation; no fail/early-termination term). `evaluate()` also exposes intermediate booleans (`is_obj_placed`, `is_robot_static`, `is_grasped`) consumed by the reward.

### Decisions resolved
- `success = is_obj_placed & is_robot_static`
- `is_obj_placed = ||goal_p - cube_p|| <= goal_thresh (0.025)`
- `is_robot_static = max|qvel[arm]| <= 0.2` (Panda `is_static` drops the last 2 finger joints)
- `is_grasped = agent.is_grasping(cube)` (contact-force ≥ 0.5 N on both finger pads, contact angle ≤ 85°)
- truncation horizon: 50 steps; no DoneTerm beyond time-out

### Code (`evaluate`, verbatim)
```python
def evaluate(self):
    is_obj_placed = (
        torch.linalg.norm(self.goal_site.pose.p - self.cube.pose.p, axis=1)
        <= self.goal_thresh
    )
    is_grasped = self.agent.is_grasping(self.cube)
    is_robot_static = self.agent.is_static(0.2)
    return {
        "success": is_obj_placed & is_robot_static,
        "is_obj_placed": is_obj_placed,
        "is_robot_static": is_robot_static,
        "is_grasped": is_grasped,
    }
```

Panda `is_grasping` / `is_static` (verbatim, `panda.py`):
```python
def is_grasping(self, object, min_force=0.5, max_angle=85):
    l_contact_forces = self.scene.get_pairwise_contact_forces(self.finger1_link, object)
    r_contact_forces = self.scene.get_pairwise_contact_forces(self.finger2_link, object)
    lforce = torch.linalg.norm(l_contact_forces, axis=1)
    rforce = torch.linalg.norm(r_contact_forces, axis=1)
    ldirection = self.finger1_link.pose.to_transformation_matrix()[..., :3, 1]
    rdirection = -self.finger2_link.pose.to_transformation_matrix()[..., :3, 1]
    langle = common.compute_angle_between(ldirection, l_contact_forces)
    rangle = common.compute_angle_between(rdirection, r_contact_forces)
    lflag = torch.logical_and(lforce >= min_force, torch.rad2deg(langle) <= max_angle)
    rflag = torch.logical_and(rforce >= min_force, torch.rad2deg(rangle) <= max_angle)
    return torch.logical_and(lflag, rflag)

def is_static(self, threshold: float = 0.2):
    qvel = self.robot.get_qvel()[..., :-2]
    return torch.max(torch.abs(qvel), 1)[0] <= threshold
```

### Smoke (§4)
Step with zero actions for 50 steps; confirm `truncated=True` at step 50 and `info` contains `success/is_obj_placed/is_robot_static/is_grasped` boolean tensors.

---

## §5 Observation

### Description
`obs_mode="state"` (default) → a flat 42-D float32 vector = concatenation of agent proprioception (`_get_obs_agent`: qpos 9 + qvel 9) and task `extra` (`_get_obs_extra`). With "state" mode the extra dict includes the privileged ground-truth cube pose and relative vectors; with `state_dict` the same fields are returned un-flattened. The goal sphere itself is hidden from sensor obs (`_hidden_objects`), but its position is provided analytically via `goal_pos` / `obj_to_goal_pos`.

### Decisions resolved — obs vector layout (state mode, total = 42)
| field | source | dim |
|---|---|---|
| `agent.qpos` | proprioception | 9 (7 arm + 2 finger) |
| `agent.qvel` | proprioception | 9 |
| `extra.is_grasped` | `info["is_grasped"]` | 1 |
| `extra.tcp_pose` | `agent.tcp_pose.raw_pose` (pos 3 + quat 4) | 7 |
| `extra.goal_pos` | `goal_site.pose.p` | 3 |
| `extra.obj_pose` | `cube.pose.raw_pose` (state only) | 7 |
| `extra.tcp_to_obj_pos` | `cube.p - tcp.p` (state only) | 3 |
| `extra.obj_to_goal_pos` | `goal.p - cube.p` (state only) | 3 |
| **total** | | **42** |

Note: `is_grasped`/`tcp_pose`/`goal_pos` (11 dims) are present in all modes; the `obj_pose`/`tcp_to_obj_pos`/`obj_to_goal_pos` block (13 dims) is added only when `"state" in obs_mode`. No observation noise/corruption is applied by default.

### Code (`_get_obs_extra`, verbatim)
```python
def _get_obs_extra(self, info: dict):
    obs = dict(
        is_grasped=info["is_grasped"],
        tcp_pose=self.agent.tcp_pose.raw_pose,
        goal_pos=self.goal_site.pose.p,
    )
    if "state" in self.obs_mode:
        obs.update(
            obj_pose=self.cube.pose.raw_pose,
            tcp_to_obj_pos=self.cube.pose.p - self.agent.tcp_pose.p,
            obj_to_goal_pos=self.goal_site.pose.p - self.cube.pose.p,
        )
    return obs
```

Default proprioception (`base_agent.get_proprioception`, verbatim):
```python
def get_proprioception(self):
    obs = dict(qpos=self.robot.get_qpos(), qvel=self.robot.get_qvel())
    controller_state = self.controller.get_state()
    if len(controller_state) > 0:
        obs.update(controller=controller_state)
    return obs
```
(`pd_joint_delta_pos` controller carries no state, so `controller` key is absent.)

### Smoke (§5)
The `observation_space` line from §1's build is the §5 verification: `Box(-inf, inf, (1, 42), float32)`.

---

## §6 Reward

### Description
Dense reward is a **SUM of 4 staged shaping terms**, gated so later stages only count once earlier ones are achieved, plus a flat success override of 5. Stages: (1) reach the cube, (2) grasp bonus, (3) move grasped cube toward goal, (4) be static once placed. On success the whole reward is overridden to the max value 5. The normalized variant divides by 5 → reward ∈ [0, 1].

### Composer
SUM. `reward = reaching_reward + is_grasped + place_reward*is_grasped + static_reward*is_obj_placed`, then `reward[success] = 5`. Composer for any per-term reward log = **"sum"**.

### Per-stage saturated per-step magnitudes (retro-computed from the source, no docstring budget present)
- `reaching_reward` ∈ [0, 1] — `1 - tanh(5 * ||cube - tcp||)`; ≈1 when tcp touches cube
- grasp bonus ∈ {0, 1} — `is_grasped`
- `place_reward * is_grasped` ∈ [0, 1] — `(1 - tanh(5 * ||goal - cube||)) * is_grasped`; ≈1 when grasped cube at goal
- `static_reward * is_obj_placed` ∈ [0, 1] — `(1 - tanh(5 * ||qvel_arm||)) * is_obj_placed`; ≈1 when placed and still
- success override = **5** (dominates the max non-success return of ~4)
- normalized = raw / 5 → [0, 1]

### Code (verbatim)
```python
def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: dict):
    tcp_to_obj_dist = torch.linalg.norm(
        self.cube.pose.p - self.agent.tcp_pose.p, axis=1
    )
    reaching_reward = 1 - torch.tanh(5 * tcp_to_obj_dist)
    reward = reaching_reward

    is_grasped = info["is_grasped"]
    reward += is_grasped

    obj_to_goal_dist = torch.linalg.norm(
        self.goal_site.pose.p - self.cube.pose.p, axis=1
    )
    place_reward = 1 - torch.tanh(5 * obj_to_goal_dist)
    reward += place_reward * is_grasped

    qvel = self.agent.robot.get_qvel()
    if self.robot_uids in ["panda", "widowxai"]:
        qvel = qvel[..., :-2]
    elif self.robot_uids == "so100":
        qvel = qvel[..., :-1]
    static_reward = 1 - torch.tanh(5 * torch.linalg.norm(qvel, axis=1))
    reward += static_reward * info["is_obj_placed"]

    reward[info["success"]] = 5
    return reward

def compute_normalized_dense_reward(self, obs: Any, action: torch.Tensor, info: dict):
    return self.compute_dense_reward(obs=obs, action=action, info=info) / 5
```

### Smoke (§6)
With a per-term reward log wrapper, assert `sum(detailed_reward.values()) == env_reward` per step (composer="sum"), except on the success step where reward is overridden to 5. All terms finite and non-constant across a random rollout.

---

## §7 DR

### Description
No physical-property / actuator / mass / friction domain randomization. The only stochasticity is the per-episode pose randomization already documented in §3 (cube xy + yaw, goal xyz, robot init-qpos noise 0.02). Friction material on the gripper pads is fixed (static=dynamic=2.0), not randomized.

`<no DR>` (beyond §3 episode-init pose randomization).

---

