# TwoRobotPickCube-v1 — Implementation Spec

- robot: Two Franka Panda arms with wrist cameras (`panda_wristcam` x2, multi-agent)
- simulator: ManiSkill (SAPIEN, `BaseEnv` + `@register_env`)
- objects: red cube, green goal sphere, table
- bimanual: true
- summary: Two arms cooperate to pick a cube and move it to a goal sphere.

---

## §1 Registration + Scene

**Description.** A multi-arm tabletop task. Two `panda_wristcam` Panda arms face each other across a table (one at y=−1, one at y=+1 in world frame). A red rigid cube (half-size 0.02 m) and a green kinematic goal sphere (radius 0.025 m, no collision, hidden in non-render observations) are added on top of a standard `TableSceneBuilder` scene. Because the cube spawns only within the left arm's reach and the goal only within the right arm's reach, success requires cooperative handoff.

**Decisions resolved.**
- `@register_env("TwoRobotPickCube-v1", max_episode_steps=100)`
- `SUPPORTED_ROBOTS = [("panda_wristcam", "panda_wristcam")]`
- `robot_uids = ("panda_wristcam", "panda_wristcam")` (default ctor)
- `robot_init_qpos_noise = 0.02`
- `cube_half_size = 0.02`, `goal_thresh = 0.025`
- Cube: `actors.build_cube(half_size=0.02, color=[1,0,0,1], name="cube", initial_pose=Pose(p=[0,0,0.02]))`
- Goal: `actors.build_sphere(radius=0.025, color=[0,1,0,1], name="goal_site", body_type="kinematic", add_collision=False, initial_pose=Pose())`; appended to `self._hidden_objects` (rendered only in the human/render camera, masked from observations).
- Robot placement via `_load_agent(options, [sapien.Pose(p=[0,-1,0]), sapien.Pose(p=[0,1,0])])` — the list of per-agent base poses is the multi-arm hook.
- Scene base: `TableSceneBuilder(env=self, robot_init_qpos_noise=self.robot_init_qpos_noise)`.

**Robot asset / actuation (per arm, from `Panda` / `PandaWristCam`).**
- `PandaWristCam.uid = "panda_wristcam"`, `urdf_path = "{PACKAGE_ASSET_DIR}/robots/panda/panda_v3.urdf"` → resolved: `mani_skill/assets/robots/panda/panda_v3.urdf` (exists ✓). (Stock `Panda` uses `panda_v2.urdf`; the wristcam variant adds a wrist camera and uses `panda_v3.urdf`.)
- `arm_joint_names = panda_joint1..7`; `gripper_joint_names = [panda_finger_joint1, panda_finger_joint2]`; `ee_link_name = "panda_hand_tcp"`.
- `arm_stiffness=1e3, arm_damping=1e2, gripper_stiffness=1e3, gripper_damping=1e2`.
- Rest keyframe qpos = `[0, π/8, 0, -5π/8, 0, 3π/4, π/4, 0.04, 0.04]`.

**Sim config.**
```python
@property
def _default_sim_config(self):
    return SimConfig(
        gpu_memory_config=GPUMemoryConfig(
            found_lost_pairs_capacity=2**25,
            max_rigid_patch_count=2**19,
            max_rigid_contact_count=2**21,
        )
    )
```

**Sensors / render cameras.**
```python
@property
def _default_sensor_configs(self):
    pose = sapien_utils.look_at([1.0, 0, 0.75], [0.0, 0.0, 0.25])
    return [CameraConfig("base_camera", pose, 128, 128, np.pi / 2, 0.01, 100)]

@property
def _default_human_render_camera_configs(self):
    pose = sapien_utils.look_at([1.4, 0.8, 0.75], [0.0, 0.1, 0.1])
    return CameraConfig("render_camera", pose, 512, 512, 1, 0.01, 100)
```

**Code (verbatim).**
```python
@register_env("TwoRobotPickCube-v1", max_episode_steps=100)
class TwoRobotPickCube(BaseEnv):
    SUPPORTED_ROBOTS = [("panda_wristcam", "panda_wristcam")]
    agent: MultiAgent[Tuple[Panda, Panda]]
    cube_half_size = 0.02
    goal_thresh = 0.025

    def __init__(self, *args, robot_uids=("panda_wristcam", "panda_wristcam"),
                 robot_init_qpos_noise=0.02, **kwargs):
        self.robot_init_qpos_noise = robot_init_qpos_noise
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    def _load_agent(self, options: dict):
        super()._load_agent(options, [sapien.Pose(p=[0, -1, 0]), sapien.Pose(p=[0, 1, 0])])

    def _load_scene(self, options: dict):
        self.table_scene = TableSceneBuilder(env=self, robot_init_qpos_noise=self.robot_init_qpos_noise)
        self.table_scene.build()
        self.cube = actors.build_cube(
            self.scene, half_size=self.cube_half_size, color=[1, 0, 0, 1],
            name="cube", initial_pose=sapien.Pose(p=[0, 0, 0.02]))
        self.goal_site = actors.build_sphere(
            self.scene, radius=self.goal_thresh, color=[0, 1, 0, 1], name="goal_site",
            body_type="kinematic", add_collision=False, initial_pose=sapien.Pose())
        self._hidden_objects.append(self.goal_site)

    @property
    def left_agent(self) -> Panda:
        return self.agent.agents[0]

    @property
    def right_agent(self) -> Panda:
        return self.agent.agents[1]
```

**Smoke (§1, captured passing stdout).**
```bash
cd <repo>
.venv/bin/python -c "import gymnasium as gym, mani_skill; e=gym.make('TwoRobotPickCube-v1'); print(e.observation_space, e.action_space); e.close()"
# -> Box(-inf, inf, (1, 66), float32) Dict('panda_wristcam-0': Box(-1.0, 1.0, (8,), float32), 'panda_wristcam-1': Box(-1.0, 1.0, (8,), float32))
```

---

## §2 Actions

**Description.** Default per-arm controller is `pd_joint_delta_pos` (7 arm-joint position deltas + 1 gripper command = 8 dims per arm). The `MultiAgent` wrapper exposes the **combined** action space as a `gymnasium.spaces.Dict` keyed by `<uid>-<index>`, so a policy must emit a dict `{'panda_wristcam-0': a0(8,), 'panda_wristcam-1': a1(8,)}`. Total controllable action width = 16. Each arm's action box is normalized to `[-1, 1]`.

**Decisions resolved.**
- `control_mode = {'panda_wristcam-0': 'pd_joint_delta_pos', 'panda_wristcam-1': 'pd_joint_delta_pos'}` (captured at runtime).
- Per-arm action = 8 = 7 arm joint deltas + 1 gripper.
- Arm `pd_joint_delta_pos` controller: `PDJointPosControllerConfig(arm_joint_names, lower=-0.1, upper=0.1, stiffness=1e3, damping=1e2, use_delta=True)` — each normalized action component maps into a joint-position delta in `[-0.1, 0.1]` rad.
- Gripper controller: `PDJointPosMimicControllerConfig(gripper_joint_names, lower=-0.01, upper=0.04, stiffness=1e3, damping=1e2)` — mimic-coupled two-finger gripper, single command.
- Combined action space (verbatim from build smoke):
  `Dict('panda_wristcam-0': Box(-1.0, 1.0, (8,)), 'panda_wristcam-1': Box(-1.0, 1.0, (8,)))`.

**Smoke (§2).**
```bash
.venv/bin/python -c "import gymnasium as gym, mani_skill; e=gym.make('TwoRobotPickCube-v1'); print(e.unwrapped.agent.control_mode); print(e.action_space); e.close()"
# -> {'panda_wristcam-0': 'pd_joint_delta_pos', 'panda_wristcam-1': 'pd_joint_delta_pos'}
# -> Dict('panda_wristcam-0': Box(-1.0, 1.0, (8,)), 'panda_wristcam-1': Box(-1.0, 1.0, (8,)))
```

---

## §3 Reset (`_initialize_episode`)

**Description.** On reset the table scene re-initializes both arms (with `robot_init_qpos_noise=0.02`), the left arm's rest qpos is latched into `self.left_init_qpos` (used later by the §6 stage-3 "return left arm home" term), the cube is randomly placed on the **left/−y** half of the table with a z-axis-only random yaw, and the goal sphere is placed on the **right/+y** half at a randomized height above the cube. The asymmetric y-ranges are what enforce cooperation.

**Decisions resolved.**
- `b = len(env_idx)`; `self.table_scene.initialize(env_idx)`.
- `self.left_init_qpos = self.left_agent.robot.get_qpos()` (latched for §6).
- Cube xyz: `x ∈ rand*0.1 − 0.05` (i.e. `[-0.05, 0.05]`); `y = -0.15 - rand*0.1 + 0.05` (i.e. `[-0.20, -0.10]`, left side); `z = cube_half_size = 0.02`.
- Cube yaw: `randomization.random_quaternions(b, lock_x=True, lock_y=True)` (z-axis rotation only).
- Goal xyz: `x ∈ rand*0.1 − 0.05` (`[-0.05, 0.05]`); `y = 0.15 + rand*0.1 - 0.05` (i.e. `[0.10, 0.20]`, right side); `z = rand*0.3 + cube.z` (`[0.02, 0.32]`, in the air).

**Code (verbatim).**
```python
def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
    with torch.device(self.device):
        b = len(env_idx)
        self.table_scene.initialize(env_idx)
        self.left_init_qpos = self.left_agent.robot.get_qpos()
        xyz = torch.zeros((b, 3))
        xyz[:, 0] = torch.rand((b,)) * 0.1 - 0.05
        # ensure cube is spawned on the left side of the table
        xyz[:, 1] = -0.15 - torch.rand((b,)) * 0.1 + 0.05
        xyz[:, 2] = self.cube_half_size
        qs = randomization.random_quaternions(b, lock_x=True, lock_y=True)
        self.cube.set_pose(Pose.create_from_pq(xyz, qs))

        goal_xyz = torch.zeros((b, 3))
        goal_xyz[:, 0] = torch.rand((b,)) * 0.1 - 0.05
        goal_xyz[:, 1] = 0.15 + torch.rand((b,)) * 0.1 - 0.05
        goal_xyz[:, 2] = torch.rand((b,)) * 0.3 + xyz[:, 2]
        self.goal_site.set_pose(Pose.create_from_pq(goal_xyz))
```

---

## §4 Goal + Termination

**Description.** Success is cooperative: the cube must be within `goal_thresh = 0.025 m` of the goal sphere center **AND** the right arm must be (nearly) static (`is_static(0.2)`). `evaluate()` also returns the intermediate booleans `is_obj_placed` and `is_right_arm_static`, which the reward function consumes via `info`. Episode times out at `max_episode_steps = 100` (set on `@register_env`). There is no separate failure termination — only success + timeout.

**Decisions resolved.**
- `goal_thresh = 0.025`.
- `is_obj_placed = ‖goal_site.pose.p − cube.pose.p‖ ≤ 0.025`.
- `is_right_arm_static = right_agent.is_static(0.2)`.
- `success = is_obj_placed AND is_right_arm_static`.
- `max_episode_steps = 100` (timeout).

**Code (verbatim).**
```python
def evaluate(self):
    is_obj_placed = (
        torch.linalg.norm(self.goal_site.pose.p - self.cube.pose.p, axis=1)
        <= self.goal_thresh
    )
    is_right_arm_static = self.right_agent.is_static(0.2)
    return {
        "success": torch.logical_and(is_obj_placed, is_right_arm_static),
        "is_obj_placed": is_obj_placed,
        "is_right_arm_static": is_right_arm_static,
    }
```

---

## §5 Observation

**Description.** Observation modes follow ManiSkill convention (`state`, `state_dict`, `rgbd`, `pointcloud`, …); default is `state`. The flat state vector is **66-dim**. It is composed of (a) per-arm proprioception assembled by `MultiAgent.get_proprioception()` — one entry per arm keyed `panda_wristcam-0` / `panda_wristcam-1`, each = qpos(9) + qvel(9) = 18, total 36; plus (b) the task `_get_obs_extra` dict. In `state` mode `_get_obs_extra` adds privileged cube/goal terms; in non-state modes only the two TCP poses are exposed.

**Decisions resolved (66-dim breakdown, obs_mode="state").**
- Proprio, left arm (`panda_wristcam-0`): 18  (qpos 9 + qvel 9)
- Proprio, right arm (`panda_wristcam-1`): 18  (qpos 9 + qvel 9)
- `_get_obs_extra` always-on:
  - `left_arm_tcp` = left tcp `raw_pose` (7)
  - `right_arm_tcp` = right tcp `raw_pose` (7)
- `_get_obs_extra` state-only:
  - `cube_pose` = cube `raw_pose` (7)
  - `left_arm_tcp_to_cube_pos` (3)
  - `right_arm_tcp_to_cube_pos` (3)
  - `cube_to_goal_pos` (3)
- Total = 36 + 7 + 7 + 7 + 3 + 3 + 3 = **66** ✓ (matches `Box(-inf, inf, (1, 66))`).
- The goal sphere is in `_hidden_objects` (not directly observed); goal info reaches the policy only through `cube_to_goal_pos` in state mode.

**Code (verbatim).**
```python
def _get_obs_extra(self, info: dict):
    obs = dict(
        left_arm_tcp=self.left_agent.tcp.pose.raw_pose,
        right_arm_tcp=self.right_agent.tcp.pose.raw_pose,
    )
    if "state" in self.obs_mode:
        obs.update(
            cube_pose=self.cube.pose.raw_pose,
            left_arm_tcp_to_cube_pos=self.cube.pose.p - self.left_agent.tcp.pose.p,
            right_arm_tcp_to_cube_pos=self.cube.pose.p - self.right_agent.tcp.pose.p,
            cube_to_goal_pos=self.goal_site.pose.p - self.cube.pose.p,
        )
    return obs
```

---

## §6 Reward

**Description.** A single staged dense reward over 5 monotonically-increasing return bands (composer = a **piecewise `reward[mask] = base + shaped`** scheme, NOT a sum of RewTerms — ManiSkill writes one tensor and overwrites per-stage masks). The bands encode the cooperative handoff:
- **Stage 1 (base 0..2):** left arm reaches the cube and pushes it across the midline toward the right arm. `reward = (reaching_reward + cube_to_other_side_reward)/2`. Sub-goal: push cube past `y=0.05` (reward saturates), advance once `cube.y ≥ 0.0`.
- **Stage 2 (base 2, +0..~6):** right arm reaches & grasps the cube; reward shapes a good grasp (finger tips at equal height, ~0.07 m apart) and pushes the left arm out of the way toward `y=-0.2`; `+2` on grasp. Entered when `cube.y ≥ 0.0`.
- **Stage 3 (base 8, +shaped):** with cube grasped, bring it to the goal (`2*place_reward`) and return the left arm to its latched init qpos. Entered when `is_grasped`.
- **Stage 4 (base 12, +2*shaped):** same shaping as stage 3, stronger incentive, entered when cube is within 0.25 m of goal AND grasped.
- **Stage 5 (base 19, +0..1):** cube placed (`is_obj_placed`) — reward both arms for going static.
- **Success:** hard-set `reward = 21`.

`compute_normalized_dense_reward = compute_dense_reward / 21`.

**Decisions resolved.**
- Composer: **piecewise mask-overwrite** (each later stage's mask replaces, not adds to, the earlier reward). Per-stage saturated per-step magnitudes (the bands): stage1 ≤ 2, stage2 ≤ ~6 (added to base 2 → ≤ ~8), stage3 base 8, stage4 base 12, stage5 base 19..20, success 21. Max return per step = 21 (retro-computed from the literal band constants; no docstring planning-budget present in source).
- `tanh` slope constant = 5 throughout; left-leave target line `y = -0.2`; grasp tip-width target `0.07 m`; near-goal intermediate threshold `0.25 m`; static metric drops the last 2 (gripper) qvel entries via `[..., :-2]`.
- Latch dependency: stage-3 `left_qpos_reward` uses `self.left_init_qpos` latched in §3 reset.
- `info` dependency: reads `info["is_obj_placed"]` and `info["success"]` from `evaluate()` (§4).

**Code (verbatim — `compute_dense_reward` + `compute_normalized_dense_reward`).**
```python
def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: dict):
    # Stage 1: Reach and push cube to be near other robot
    tcp_to_obj_dist = torch.linalg.norm(
        self.cube.pose.p - self.left_agent.tcp.pose.p, axis=1
    )
    reaching_reward = 1 - torch.tanh(5 * tcp_to_obj_dist)

    # set a sub_goal here where we want the cube to first be pushed to close to the right arm robot
    # by moving cube past y = 0.05
    cube_to_other_side_reward = 1 - torch.tanh(
        5
        * (
            torch.max(
                0.05 - self.cube.pose.p[:, 1], torch.zeros_like(reaching_reward)
            )
        )
    )
    reward = (reaching_reward + cube_to_other_side_reward) / 2

    # stage 1 passes if cube is near a sub-goal
    cube_at_other_side = self.cube.pose.p[:, 1] >= 0.0

    # Stage 2: reach and grasp cube with right robot and make left robot leave space
    tcp_to_obj_dist = torch.linalg.norm(
        self.cube.pose.p - self.right_agent.tcp.pose.p, axis=1
    )
    reaching_reward = 1 - torch.tanh(5 * tcp_to_obj_dist)
    stage_2_reward = reaching_reward

    # condition for good grasp: both fingers are at the same height and open
    self.right_agent: Panda
    right_tip_1_height = self.right_agent.finger1_link.pose.p[:, 2]
    right_tip_2_height = self.right_agent.finger2_link.pose.p[:, 2]
    tip_height_reward = 1 - torch.tanh(
        5 * torch.abs(right_tip_1_height - right_tip_2_height)
    )
    tip_width_reward = 1 - torch.tanh(
        5
        * torch.abs(
            torch.linalg.norm(
                self.right_agent.finger1_link.pose.p
                - self.right_agent.finger2_link.pose.p,
                axis=1,
            )
            - 0.07
        )
    )
    tip_reward = (tip_height_reward + tip_width_reward) / 2
    stage_2_reward += tip_reward

    # make left arm move as close as possible to the y=-0.2 line
    left_arm_leave_reward = 1 - torch.tanh(
        5 * (self.left_agent.tcp.pose.p[:, 1] + 0.2).abs()
    )
    stage_2_reward += left_arm_leave_reward

    # stage 2 passes if cube is grasped
    is_grasped = self.right_agent.is_grasping(self.cube)
    stage_2_reward += 2 * is_grasped

    reward[cube_at_other_side] = 2 + stage_2_reward[cube_at_other_side]

    # Stage 3: bring cube towards goal
    obj_to_goal_dist = torch.linalg.norm(
        self.goal_site.pose.p - self.right_agent.tcp.pose.p, axis=1
    )
    place_reward = 1 - torch.tanh(5 * obj_to_goal_dist)
    stage_3_reward = 2 * place_reward

    # return left arm to original position
    left_qpos_reward = 1 - torch.tanh(
        torch.linalg.norm(
            self.left_agent.robot.get_qpos() - self.left_init_qpos, axis=1
        )
    )
    stage_3_reward += left_qpos_reward

    reward[is_grasped] = 8 + stage_3_reward[is_grasped]

    # stage 3 passes if object is near goal (within 0.25m) - intermediate reward
    is_obj_near = torch.logical_and(obj_to_goal_dist < 0.25, is_grasped)
    # Stage 4: reuse same reward as stage 3 but stronger incentive
    reward[is_obj_near] = 12 + 2 * stage_3_reward[is_obj_near]

    # stage 4 passes if object is placed
    is_obj_placed = info["is_obj_placed"]

    # Stage 5: keep robot static at the goal
    right_static_reward = 1 - torch.tanh(
        5 * torch.linalg.norm(self.right_agent.robot.get_qvel()[..., :-2], axis=1)
    )
    left_static_reward = 1 - torch.tanh(
        5 * torch.linalg.norm(self.left_agent.robot.get_qvel()[..., :-2], axis=1)
    )
    static_reward = (right_static_reward + left_static_reward) / 2

    reward[is_obj_placed] = 19 + static_reward[is_obj_placed]

    reward[info["success"]] = 21

    return reward

def compute_normalized_dense_reward(self, obs: Any, action: torch.Tensor, info: dict):
    return self.compute_dense_reward(obs=obs, action=action, info=info) / 21
```

**Smoke (§6).**
```bash
.venv/bin/python -c "import gymnasium as gym, mani_skill, torch; e=gym.make('TwoRobotPickCube-v1', reward_mode='dense'); o,_=e.reset(seed=0); a=e.action_space.sample(); o,r,te,tr,i=e.step(a); print('reward finite:', torch.isfinite(torch.as_tensor(r)).all().item(), 'r=', r); e.close()"
# expect a finite scalar/tensor reward in [0, 21]
```

---

## §7 DR

`<no DR>` — there are no `startup` / `interval` domain-randomization events. All randomization is reset-time only (cube xy + z-yaw, goal xyz, `robot_init_qpos_noise=0.02`), which is captured in §3. No physics-material / mass / friction / lighting randomization is applied.

---

