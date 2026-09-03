# UnitreeG1TransportBox-v1 — Implementation Spec

- robot: Unitree G1 humanoid, simplified upper body + head camera, bimanual (25 DoF, fixed base)
- simulator: ManiSkill (SAPIEN, `BaseEnv` + `@register_env`)
- objects: box, two tables
- bimanual: true
- summary: Lift a box from one table and carry it across to another.

ManiSkill tasks are SAPIEN `BaseEnv` subclasses decorated with `@register_env`. There is no IsaacLab-style manager-based config tree: scene / actions / reset / termination / observation / reward are all *methods* on the env class. The §1..§7 mapping below adapts the Harbor section model onto these methods.

---

## §1 Registration + Scene

**Description.** The env is registered as `UnitreeG1TransportBox-v1` with `max_episode_steps=100`. The scene is a ground plane, **two static tables** (`table-1` at `y=+0.66`, `table-2` at `y=-0.66`, each scaled 1.2× from `table.glb` with a box collider), and a single dynamic **cardboard box** that starts on `table-2` (the `y<0` side). The robot is a fixed-root Unitree G1 simplified upper body (torso + two 7-DoF arms + two articulated hands) loaded at `p=[-0.1, 0, 0.755]` with a tuned standing keyframe (elbows pre-bent: `qpos[3]=1.25`, `qpos[4]=-1.25`). Sim uses GPU contact buffers sized for many contact pairs and a `contact_offset=0.02`.

**Decisions resolved.**
- `register_env` id = `"UnitreeG1TransportBox-v1"`, `max_episode_steps=100`.
- `SUPPORTED_ROBOTS = ["unitree_g1_simplified_upper_body_with_head_camera"]`; `robot_uids` passed to `super().__init__`.
- Robot agent class: `UnitreeG1UpperBodyWithHeadCamera` (subclass of `UnitreeG1UpperBody`); URDF `{PACKAGE_ASSET_DIR}/robots/g1_humanoid/g1_simplified_upper_body.urdf`; `fix_root_link=True`, `load_multiple_collisions=False`.
- Robot init pose `p=[-0.1, 0, 0.755]`, init qpos = standing keyframe (all-zeros, 25 DoF) with `qpos[3]=1.25`, `qpos[4]=-1.25` (right/left shoulder-pitch pre-bent).
- `_load_agent` overridden to load at `sapien.Pose(p=[0,0,1])` (then repositioned at reset).
- Tables: model `mani_skill/utils/scene_builder/table/assets/table.glb`, `scale=1.2`, box collision `half_size=(0.829, 0.4145, 0.3153)` at center `z=0.3153`; `table-1.initial_pose = p=[0, 0.66, 0]`, `table-2.initial_pose = p=[0, -0.66, 0]`; both `build_static`.
- Box: box collision `half_size=(0.18, 0.12, 0.12)`, `density=200`; visual `mani_skill/envs/tasks/humanoid/assets/cardboard_box/textured.obj` at `scale=0.12`; `initial_pose = p=[-0.1, -0.37, 0.7508]`; dynamic (`builder.build`).
- Sim: `GPUMemoryConfig(max_rigid_contact_count=2**22, max_rigid_patch_count=2**21)`, `SceneConfig(contact_offset=0.02)`.
- Sensors: head camera mounted on `head_link` (128×128, fov π/2) + a `base_camera`/`render_camera` looking at the workspace (512×512, fov π/3).

**Code (registration + ctor + sim/sensor cfg).**
```python
@register_env("UnitreeG1TransportBox-v1", max_episode_steps=100)
class TransportBoxEnv(BaseEnv):
    _sample_video_link = "https://github.com/haosulab/ManiSkill/raw/main/figures/environment_demos/UnitreeG1TransportBox-v1_rt.mp4"
    SUPPORTED_ROBOTS = ["unitree_g1_simplified_upper_body_with_head_camera"]
    agent: UnitreeG1UpperBodyWithHeadCamera

    def __init__(self, *args, **kwargs):
        self.init_robot_pose = copy.deepcopy(
            UnitreeG1UpperBodyWithHeadCamera.keyframes["standing"].pose
        )
        self.init_robot_pose.p = [-0.1, 0, 0.755]
        self.init_robot_qpos = UnitreeG1UpperBodyWithHeadCamera.keyframes[
            "standing"
        ].qpos.copy()
        self.init_robot_qpos[4] = -1.25
        self.init_robot_qpos[3] = 1.25
        super().__init__(
            *args,
            robot_uids="unitree_g1_simplified_upper_body_with_head_camera",
            **kwargs
        )

    @property
    def _default_sim_config(self):
        return SimConfig(
            gpu_memory_config=GPUMemoryConfig(
                max_rigid_contact_count=2**22, max_rigid_patch_count=2**21
            ),
            scene_config=SceneConfig(contact_offset=0.02),
        )

    @property
    def _default_sensor_configs(self):
        pose = sapien_utils.look_at([1.0, 0.0, 1.6], [0, 0.0, 0.65])
        return [
            CameraConfig("base_camera", pose=pose, width=128, height=128, fov=np.pi / 3)
        ]

    @property
    def _default_human_render_camera_configs(self):
        pose = sapien_utils.look_at([1.0, 0.0, 1.6], [0, 0.0, 0.65])
        return CameraConfig(
            "render_camera", pose=pose, width=512, height=512, fov=np.pi / 3
        )

    def _load_agent(self, options: dict):
        super()._load_agent(options, sapien.Pose(p=[0, 0, 1]))
```

**Code (scene).**
```python
    def _load_scene(self, options: dict):
        self.ground = ground.build_ground(self.scene, mipmap_levels=7)
        # build two tables

        model_dir = Path(
            os.path.join(
                os.path.dirname(__file__), "../../../utils/scene_builder/table/assets"
            )
        )
        table_model_file = str(model_dir / "table.glb")
        scale = 1.2
        table_pose = sapien.Pose(q=euler2quat(0, 0, np.pi / 2))
        builder = self.scene.create_actor_builder()
        builder.add_visual_from_file(
            filename=table_model_file,
            scale=[scale] * 3,
            pose=sapien.Pose(q=euler2quat(0, 0, np.pi / 2)),
        )
        builder.add_box_collision(
            pose=sapien.Pose(p=[0, 0, 0.630612274 / 2]),
            half_size=(1.658057143 / 2, 0.829028571 / 2, 0.630612274 / 2),
        )
        builder.add_visual_from_file(
            filename=table_model_file, scale=[scale] * 3, pose=table_pose
        )
        builder.initial_pose = sapien.Pose(p=[0, 0.66, 0])
        self.table_1 = builder.build_static(name="table-1")
        builder = self.scene.create_actor_builder()
        builder.add_visual_from_file(
            filename=table_model_file,
            scale=[scale] * 3,
            pose=sapien.Pose(q=euler2quat(0, 0, np.pi / 2)),
        )
        builder.add_box_collision(
            pose=sapien.Pose(p=[0, 0, 0.630612274 / 2]),
            half_size=(1.658057143 / 2, 0.829028571 / 2, 0.630612274 / 2),
        )
        builder.add_visual_from_file(
            filename=table_model_file, scale=[scale] * 3, pose=table_pose
        )
        builder.initial_pose = sapien.Pose(p=[0, -0.66, 0])
        self.table_2 = builder.build_static(name="table-2")

        builder = self.scene.create_actor_builder()
        builder.add_box_collision(half_size=(0.18, 0.12, 0.12), density=200)
        visual_file = os.path.join(
            os.path.dirname(__file__), "assets/cardboard_box/textured.obj"
        )
        builder.add_visual_from_file(
            filename=visual_file,
            scale=[0.12] * 3,
            pose=sapien.Pose(q=euler2quat(0, 0, np.pi / 2)),
        )
        builder.initial_pose = sapien.Pose(p=[-0.1, -0.37, 0.7508])
        self.box = builder.build(name="box")
```

**Resolved asset paths (relative to the package root).**
- Table: `mani_skill/utils/scene_builder/table/assets/table.glb` — EXISTS.
- Box visual: `mani_skill/envs/tasks/humanoid/assets/cardboard_box/textured.obj` — EXISTS.
- Robot URDF: `mani_skill/assets/robots/g1_humanoid/g1_simplified_upper_body.urdf` (`PACKAGE_ASSET_DIR` = `mani_skill/assets`) — present in package asset dir.

**Smoke (§1 build).** `cd <repo> && .venv/bin/python -c "import gymnasium as gym, mani_skill; e=gym.make('UnitreeG1TransportBox-v1'); print(e.observation_space, e.action_space)"`
Expected stdout: `Box(-inf, inf, (1, 77), float32) Box(-1.0, 1.0, (25,), float32)`

---

## §2 Actions (control_mode + action space)

**Description.** The robot has **25 active DoF** (1 torso + 8 arm joints + 16 hand finger joints — see `body_joints` below). The default control mode is `pd_joint_delta_pos` (first key of `_controller_configs`): per-step delta joint-position targets, normalized to `[-1, 1]`, with per-joint delta limits of ±0.2 rad for the first 11 joints (torso + arm) and ±0.5 rad for the 14 finger joints. Passive (gravity-compensation) balance force is enabled.

**Decisions resolved.**
- `control_mode` default = `pd_joint_delta_pos`; also supports `pd_joint_pos`.
- Action space = `Box(-1.0, 1.0, (25,), float32)` — one delta target per active joint.
- `body_joints` ordering (25): `torso_joint`, `left_shoulder_pitch_joint`, `right_shoulder_pitch_joint`, `left_shoulder_roll_joint`, `right_shoulder_roll_joint`, `left_shoulder_yaw_joint`, `right_shoulder_yaw_joint`, `left_elbow_pitch_joint`, `right_elbow_pitch_joint`, `left_elbow_roll_joint`, `right_elbow_roll_joint`, then 14 finger joints: `left_zero_joint`, `left_three_joint`, `left_five_joint`, `right_zero_joint`, `right_three_joint`, `right_five_joint`, `left_one_joint`, `left_four_joint`, `left_six_joint`, `right_one_joint`, `right_four_joint`, `right_six_joint`, `left_two_joint`, `right_two_joint`.
- PD gains: `stiffness=1e3`, `damping=1e2`, `force_limit=100`.
- `pd_joint_delta_pos` limits: `lower=[-0.2]*11 + [-0.5]*14`, `upper=[0.2]*11 + [0.5]*14`, `use_delta=True`.
- `pd_joint_pos`: `lower=None, upper=None, normalize_action=False`.
- Legs are fixed (not actuated); robot root is fixed (`fix_root_link=True`) so qpos[0] = `torso_joint` (used by reward/eval as the "facing" proxy).

**Code (controller configs, from `g1_upper_body.py`).**
```python
    body_stiffness = 1e3
    body_damping = 1e2
    body_force_limit = 100

    @property
    def _controller_configs(self):
        body_pd_joint_pos = PDJointPosControllerConfig(
            self.body_joints,
            lower=None,
            upper=None,
            stiffness=self.body_stiffness,
            damping=self.body_damping,
            force_limit=self.body_force_limit,
            normalize_action=False,
        )
        body_pd_joint_delta_pos = PDJointPosControllerConfig(
            self.body_joints,
            lower=[-0.2] * 11 + [-0.5] * 14,
            upper=[0.2] * 11 + [0.5] * 14,
            stiffness=self.body_stiffness,
            damping=self.body_damping,
            force_limit=self.body_force_limit,
            use_delta=True,
        )
        return dict(
            pd_joint_delta_pos=dict(
                body=body_pd_joint_delta_pos, balance_passive_force=True
            ),
            pd_joint_pos=dict(body=body_pd_joint_pos, balance_passive_force=True),
        )
```

**Smoke (§2).** Sample a random action and step: `e.reset(); e.step(e.action_space.sample())` returns a 5-tuple without error; action dim == 25.

---

## §3 Reset (`_initialize_episode`)

**Description.** On reset, the robot qpos and pose are restored to the tuned standing keyframe, then the box is dropped onto `table-2` at a randomized xy offset and a randomized small z-rotation. No DR on the robot or tables.

**Decisions resolved.**
- Robot: `set_qpos(self.init_robot_qpos)`, `set_pose(self.init_robot_pose)` (pose `p=[-0.1, 0, 0.755]`).
- Box z fixed at `0.7508` (resting on `table-2`).
- Box x ~ `Uniform(-0.05, 0.2)`, then `+(-0.1)` → x ∈ `[-0.15, 0.1]`.
- Box y ~ `Uniform(-0.05, 0.05)`, then `+(-0.37)` → y ∈ `[-0.42, -0.32]`.
- Box z-rotation: `random_quaternions(lock_x=True, lock_y=True, bounds=(0, np.pi/6))` — yaw ∈ `[0, 30°]`.

**Code.**
```python
    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            self.agent.robot.set_qpos(self.init_robot_qpos)
            self.agent.robot.set_pose(self.init_robot_pose)
            xyz = torch.zeros((b, 3))
            xyz[:, 2] = 0.7508
            xyz[:, 0] = randomization.uniform(-0.05, 0.2, size=(b,))
            xyz[:, 1] = randomization.uniform(-0.05, 0.05, size=(b,))
            xyz[:, :2] += torch.tensor([-0.1, -0.37])
            quat = randomization.random_quaternions(
                n=b, device=self.device, lock_x=True, lock_y=True, bounds=(0, np.pi / 6)
            )
            self.box.set_pose(Pose.create_from_pq(xyz, quat))
```

**Smoke (§3).** `e.reset(seed=0)` twice with different seeds → box pose differs; with same seed → box pose matches.

---

## §4 Goal + Termination (`evaluate` + `max_episode_steps`)

**Description.** ManiSkill has no separate `TerminationsCfg`; termination is a time-out at `max_episode_steps=100` plus the `success` flag returned from `evaluate()`. Success = the box is **resting on the target table (`table-1`, the `y>0` side) and no longer grasped**. `evaluate()` also exports several intermediate booleans (`box_grasped`, `box_at_correct_table_xy`, `facing_table_with_box`, per-hand contact) that the dense reward consumes as stage gates.

**Decisions resolved.**
- `max_episode_steps = 100` (time-out termination).
- `success = ~box_grasped & box_at_correct_table` where:
  - `box_at_correct_table = box_at_correct_table_z & box_at_correct_table_xy`.
  - `box_at_correct_table_z`: `0.750 < box.z < 0.751` (resting on table-1 surface).
  - `box_at_correct_table_xy`: `-0.78 < box.x < 0.78` and `0.3 < box.y < 1.0` (over table-1).
- `box_grasped`: both hands contact box (per-hand summed contact force > 10 N across `*_five_link`, `*_three_link`, `*_palm_link`) **and** both TCPs are below the respective grasp point + 0.04 m.
- Grasp points: `box_left_grasp_point = box.pose * Pose([0.165, 0.07, 0.05])`, `box_right_grasp_point = box.pose * Pose([-0.165, 0.07, 0.05])`.
- `facing_table_with_box`: `-1.7 < qpos[:,0] < -1.4` (torso_joint angle proxy for "facing the box's table").

**Code (`evaluate` + grasp-point properties).**
```python
    def evaluate(self):
        l_contact_forces = (
            (
                self.scene.get_pairwise_contact_forces(
                    self.agent.robot.links_map["left_five_link"], self.box
                )
                + self.scene.get_pairwise_contact_forces(
                    self.agent.robot.links_map["left_three_link"], self.box
                )
                + self.scene.get_pairwise_contact_forces(
                    self.agent.robot.links_map["left_palm_link"], self.box
                )
            )
            .abs()
            .sum(dim=1)
        )
        r_contact_forces = (
            (
                self.scene.get_pairwise_contact_forces(
                    self.agent.robot.links_map["right_five_link"], self.box
                )
                + self.scene.get_pairwise_contact_forces(
                    self.agent.robot.links_map["right_three_link"], self.box
                )
                + self.scene.get_pairwise_contact_forces(
                    self.agent.robot.links_map["right_palm_link"], self.box
                )
            )
            .abs()
            .sum(dim=1)
        )
        left_hand_hit_box = l_contact_forces > 10
        right_hand_hit_box = r_contact_forces > 10
        box_grasped = (
            left_hand_hit_box
            & right_hand_hit_box
            & (
                self.agent.right_tcp.pose.p[:, 2]
                < self.box_right_grasp_point.p[:, 2] + 0.04
            )
            & (
                self.agent.left_tcp.pose.p[:, 2]
                < self.box_left_grasp_point.p[:, 2] + 0.04
            )
        )

        box_at_correct_table_z = (0.751 > self.box.pose.p[:, 2]) & (
            self.box.pose.p[:, 2] > 0.750
        )
        box_at_correct_table_xy = (
            (0.78 > self.box.pose.p[:, 0])
            & (self.box.pose.p[:, 0] > -0.78)
            & (1.0 > self.box.pose.p[:, 1])
            & (self.box.pose.p[:, 1] > 0.3)
        )
        box_at_correct_table = box_at_correct_table_z & box_at_correct_table_xy

        facing_table_with_box = (-1.7 < self.agent.robot.qpos[:, 0]) & (
            self.agent.robot.qpos[:, 0] < -1.4
        )
        return {
            "success": ~box_grasped & box_at_correct_table,
            "left_hand_hit_box": l_contact_forces > 0,
            "right_hand_hit_box": r_contact_forces > 0,
            "box_grasped": box_grasped,
            "box_at_correct_table_xy": box_at_correct_table_xy,
            "facing_table_with_box": facing_table_with_box,
        }

    @property
    def box_right_grasp_point(self):
        return self.box.pose * Pose.create_from_pq(
            torch.tensor([-0.165, 0.07, 0.05], device=self.device)
        )

    @property
    def box_left_grasp_point(self):
        return self.box.pose * Pose.create_from_pq(
            torch.tensor([0.165, 0.07, 0.05], device=self.device)
        )
```

**Smoke (§4).** `info = e.step(e.action_space.sample())[-1]` (or `e.unwrapped.evaluate()`) returns dict with keys `success, box_grasped, box_at_correct_table_xy, facing_table_with_box, left_hand_hit_box, right_hand_hit_box`; all boolean tensors of shape `(num_envs,)`.

---

## §5 Observation (`_get_obs_extra` + obs modes + dim)

**Description.** Default obs_mode = `"state"` (first of `SUPPORTED_OBS_MODES`). The flat 77-d state vector = base agent proprioception (qpos + qvel of the 25 DoF, etc., assembled by `BaseEnv._get_obs_state_dict`/`get_proprioception`) concatenated with the task `_get_obs_extra`. In non-state modes only the two TCP poses are exposed; in state/state_dict modes the privileged box pose + TCP-to-box vectors are added.

**Decisions resolved.**
- `SUPPORTED_OBS_MODES = ("state", "state_dict", "none", "sensor_data", "any_textures", "pointcloud")`; default `"state"`.
- `_get_obs_extra` always returns: `right_tcp_pose` (7), `left_tcp_pose` (7).
- When `"state" in obs_mode` (state / state_dict), additionally: `box_pose` (7), `right_tcp_to_box_pos` (3), `left_tcp_to_box_pos` (3).
- Resolved total flat obs dim (state mode) = **77** (verified via build smoke). Extra block contributes 7+7+7+3+3 = 27; the remaining 50 is agent proprioception assembled by the base env.

**Code.**
```python
    def _get_obs_extra(self, info: dict):
        obs = dict(
            right_tcp_pose=self.agent.right_tcp.pose.raw_pose,
            left_tcp_pose=self.agent.left_tcp.pose.raw_pose,
        )

        if "state" in self.obs_mode:
            obs.update(
                box_pose=self.box.pose.raw_pose,
                right_tcp_to_box_pos=self.box.pose.p - self.agent.right_tcp.pose.p,
                left_tcp_to_box_pos=self.box.pose.p - self.agent.left_tcp.pose.p,
            )
        return obs
```

**Smoke (§5).** `e.observation_space` → `Box(-inf, inf, (1, 77), float32)` in default state mode (verified).

---

## §6 Reward (`compute_dense_reward` + `compute_normalized_dense_reward`)

**Description.** A **4-stage staged dense reward** with hard overwrites (not additive composition) — each later stage's gate, when true, *replaces* the running reward for those envs with a higher band, producing a monotone reward ladder `Stage1 ≈ [0,1] → Stage2 ≈ [1,2] → Stage3 ≈ [3,4] → Stage4 ≈ [3,4] → success = 5`. Composer = **staged overwrite** (masked assignment by gate), NOT sum/product.

- **Stage 1 (turn to face box table):** base `reward = 1 - tanh(|qpos[0] + 1.4|)` — drives torso_joint toward -1.4 (facing table-2 / the box). Range [0, 1].
- **Stage 2 (gated by `facing_table_with_box`):** overwrite with `1 + reach-shaping/4` — encourages arms down (`qpos[3]`, `qpos[4]` → 0) and both TCPs to their grasp points. Range [1, 2].
- **Stage 3 (gated by `box_grasped`):** overwrite with `3 - tanh(|qpos[0] - 1.4|/5)` — drives torso_joint toward +1.4 (turn toward table-1). Range ~[3, 4].
- **Stage 4 (gated by `box_at_correct_table_xy`):** overwrite with `3 + release-shaping/2` — drives `qpos[3]→1.25`, `qpos[4]→-1.25` (open / release the box). Range [3, 4].
- **Success (gated by `success`):** overwrite with constant `5`.
- `compute_normalized_dense_reward = compute_dense_reward / 5`.

**Planning-budget (per-stage saturated per-step magnitudes, retro-computed from constants):** Stage1 max ≈ 1; Stage2 max ≈ 2; Stage3 max ≈ 4 (`3 + 1 - tanh(0) = 4`); Stage4 max ≈ 4 (`3 + 0.5 + 0.5`); success = 5. Normalized: divide all by 5.

**Code (verbatim, both functions).**
```python
    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: dict):
        # Stage 1, move to face the box on the table. Succeeds if facing_table_with_box
        reward = 1 - torch.tanh((self.agent.robot.qpos[:, 0] + 1.4).abs())

        # Stage 2, grasp the box stably. Succeeds if box_grasped
        # encourage arms to go down essentially and for tcps to be close to the edge of the box
        stage_2_reward = (
            1
            + (1 - torch.tanh((self.agent.robot.qpos[:, 3]).abs())) / 4
            + (1 - torch.tanh((self.agent.robot.qpos[:, 4]).abs())) / 4
            + (
                1
                - torch.tanh(
                    3
                    * torch.linalg.norm(
                        self.agent.right_tcp.pose.p - self.box_right_grasp_point.p,
                        dim=1,
                    )
                )
            )
            / 4
            + (
                1
                - torch.tanh(
                    3
                    * torch.linalg.norm(
                        self.agent.left_tcp.pose.p - self.box_left_grasp_point.p, dim=1
                    )
                )
            )
            / 4
        )
        reward[info["facing_table_with_box"]] = stage_2_reward[
            info["facing_table_with_box"]
        ]
        # Stage 3 transport box to above the other table, Succeeds if box_at_correct_table_xy
        stage_3_reward = (
            2 + 1 - torch.tanh((self.agent.robot.qpos[:, 0] - 1.4).abs() / 5)
        )
        reward[info["box_grasped"]] = stage_3_reward[info["box_grasped"]]
        # Stage 4 let go of the box. Succeeds if success (~box_grasped & box_at_correct_table)
        stage_4_reward = (
            3
            + (1 - torch.tanh((self.agent.robot.qpos[:, 3] - 1.25).abs())) / 2
            + (1 - torch.tanh((self.agent.robot.qpos[:, 4] + 1.25).abs())) / 2
        )
        reward[info["box_at_correct_table_xy"]] = stage_4_reward[
            info["box_at_correct_table_xy"]
        ]
        # encourage agent to stay close to a target qposition?
        reward[info["success"]] = 5
        return reward

    def compute_normalized_dense_reward(
        self, obs: Any, action: torch.Tensor, info: dict
    ):
        return self.compute_dense_reward(obs, action, info) / 5
```

**Composer.** Staged overwrite (masked assignment by gate booleans from `evaluate()`); not sum and not product. A per-term reward log must reproduce the final reward by replaying the same gate order (Stage1 → facing → grasped → at-table-xy → success), each later true gate overwriting earlier values.

**Smoke (§6).** Step a random rollout; assert `reward` is finite (no NaN/inf), non-constant across steps, and bounded in `[0, 5]` (dense) / `[0, 1]` (normalized).

---

## §7 DR (domain randomization beyond reset)

`<no DR>` — there are no `startup`/`interval` randomization events. The only per-episode randomization is the box xy + yaw in `_initialize_episode` (§3), which is initial-state reset randomization, not parameter DR. The robot, tables, masses, frictions, and PD gains are fixed across episodes.

---

