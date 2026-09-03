# UnitreeG1PlaceAppleInBowl-v1 — Implementation Spec

- robot: Unitree G1 humanoid, simplified upper body, bimanual (25 DoF, fixed base)
- simulator: ManiSkill (SAPIEN, `BaseEnv` + `@register_env`)
- objects: apple, bowl, kitchen counter
- bimanual: true
- summary: Pick an apple off a kitchen counter and place it in a bowl.

**Task:** Control the humanoid Unitree G1 (simplified upper body, fixed/seated base) to grasp an apple with its RIGHT arm and place it in a bowl beside it. Bimanual-capable robot (both arms + both 6-DoF hands actuated) but only the right side is used by the reward/success logic.

ManiSkill class hierarchy: `BaseEnv` → `HumanoidPickPlaceEnv` (base scene: kitchen counter + apple/bowl placeholders, sparse only) → `HumanoidPlaceAppleInBowl` (apple+bowl actors, dense reward, evaluate, obs) → `UnitreeG1PlaceAppleInBowlEnv` (`@register_env`, G1 robot wiring, episode init). Sections below note which class each block comes from.

---

## §1 Registration + Scene

### Description
A `@register_env` leaf class binds the task id to `max_episode_steps=100` and the Unitree G1 simplified-upper-body-with-head-camera robot. The scene is a `KitchenCounterSceneBuilder` (table/counter) with an apple (dynamic, convex collisions, graspable) and a bowl (kinematic, nonconvex collision) loaded from local `.ply`/`.glb` assets. The robot root is fixed (seated at the counter).

### Decisions resolved
- task id: `UnitreeG1PlaceAppleInBowl-v1`, `max_episode_steps=100`.
- robot: `unitree_g1_simplified_upper_body_with_head_camera` (25 active body joints: `torso_joint` + 8 arm joints per side split across both arms + 6 finger joints per hand). `fix_root_link=True`.
- `SUPPORTED_ROBOTS = ["unitree_g1_simplified_upper_body_with_head_camera"]`.
- `kitchen_scene_scale = 0.82`.
- robot init pose: standing keyframe pose with `p = [-0.3, 0, 0.755]`.
- apple: dynamic actor, scale `0.82 * 0.8` (shrunk to be graspable), initial_pose `p=[0, -0.4, 0.78]` (overridden at reset), nonconvex bowl via `add_nonconvex_collision_from_file`, apple via `add_multiple_convex_collisions_from_file`.
- bowl: kinematic actor (`build_kinematic`), scale `0.82`, initial_pose `p=[0, -0.4, 0.753]`.
- both assets rotated by `fix_rotation_pose = sapien.Pose(q=euler2quat(np.pi/2, 0, 0))`.
- sim cfg (leaf): `GPUMemoryConfig(max_rigid_contact_count=2**22, max_rigid_patch_count=2**21)`, `SceneConfig(contact_offset=0.01)` (reduced to limit dextrous-finger collision checks).
- `SUPPORTED_REWARD_MODES = ["normalized_dense", "dense", "sparse", "none"]`.
- `_load_agent`: G1 spawned at `sapien.Pose(p=[0, 0, 1])` (base class), then repositioned at reset.

### Resolved asset paths (relative to the `ManiSkill/` package root)
- `mani_skill/envs/tasks/humanoid/assets/frl_apartment_bowl_07.ply` (bowl collision)
- `mani_skill/envs/tasks/humanoid/assets/frl_apartment_bowl_07.glb` (bowl visual)
- `mani_skill/envs/tasks/humanoid/assets/apple_1.ply` (apple collision)
- `mani_skill/envs/tasks/humanoid/assets/apple_1.glb` (apple visual)
- robot URDF: `mani_skill/assets/robots/g1_humanoid/g1_simplified_upper_body.urdf` (referenced via `PACKAGE_ASSET_DIR`)
- kitchen counter: `KitchenCounterSceneBuilder` (`mani_skill/utils/scene_builder/kitchen_counter/`)

### Code
Registration + leaf class (`humanoid_pick_place.py:209-255`):
```python
@register_env("UnitreeG1PlaceAppleInBowl-v1", max_episode_steps=100)
class UnitreeG1PlaceAppleInBowlEnv(HumanoidPlaceAppleInBowl):
    """
    **Task Description:**
    Control the humanoid unitree G1 robot to grab an apple with its right arm and place it in a bowl to the side
    ...
    """
    _sample_video_link = "https://github.com/haosulab/ManiSkill/raw/main/figures/environment_demos/UnitreeG1PlaceAppleInBowl-v1_rt.mp4"

    SUPPORTED_ROBOTS = ["unitree_g1_simplified_upper_body_with_head_camera"]
    agent: UnitreeG1UpperBodyWithHeadCamera
    kitchen_scene_scale = 0.82

    def __init__(self, *args, **kwargs):
        self.init_robot_pose = copy.deepcopy(
            UnitreeG1UpperBodyWithHeadCamera.keyframes["standing"].pose
        )
        self.init_robot_pose.p = [-0.3, 0, 0.755]
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
            scene_config=SceneConfig(contact_offset=0.01),
        )
```

Base scene class (`humanoid_pick_place.py:23-69`):
```python
class HumanoidPickPlaceEnv(BaseEnv):
    SUPPORTED_REWARD_MODES = ["sparse", "none"]
    """sets up a basic scene with a apple to pick up and place on a dish"""
    kitchen_scene_scale = 1.0

    def __init__(self, *args, robot_init_qpos_noise=0.02, **kwargs):
        self.robot_init_qpos_noise = robot_init_qpos_noise
        super().__init__(*args, **kwargs)

    @property
    def _default_sim_config(self):
        return SimConfig(
            gpu_memory_config=GPUMemoryConfig(max_rigid_contact_count=2**22)
        )

    def _load_agent(self, options: dict):
        super()._load_agent(options, sapien.Pose(p=[0, 0, 1]))

    def _load_scene(self, options: dict):
        self.scene_builder = KitchenCounterSceneBuilder(self)
        self.kitchen_scene = self.scene_builder.build(scale=self.kitchen_scene_scale)
```

Apple + bowl loading (`humanoid_pick_place.py:103-136`):
```python
    def _load_scene(self, options: dict):
        super()._load_scene(options)
        scale = self.kitchen_scene_scale
        builder = self.scene.create_actor_builder()
        fix_rotation_pose = sapien.Pose(q=euler2quat(np.pi / 2, 0, 0))
        model_dir = os.path.dirname(__file__) + "/assets"
        builder.add_nonconvex_collision_from_file(
            filename=os.path.join(model_dir, "frl_apartment_bowl_07.ply"),
            pose=fix_rotation_pose,
            scale=[scale] * 3,
        )
        builder.add_visual_from_file(
            filename=os.path.join(model_dir, "frl_apartment_bowl_07.glb"),
            scale=[scale] * 3,
            pose=fix_rotation_pose,
        )
        builder.initial_pose = sapien.Pose(p=[0, -0.4, 0.753])
        self.bowl = builder.build_kinematic(name="bowl")

        builder = self.scene.create_actor_builder()
        model_dir = os.path.dirname(__file__) + "/assets"
        builder.add_multiple_convex_collisions_from_file(
            filename=os.path.join(model_dir, "apple_1.ply"),
            pose=fix_rotation_pose,
            scale=[scale * 0.8] * 3,  # scale down more to make apple a bit smaller to be graspable
        )
        builder.add_visual_from_file(
            filename=os.path.join(model_dir, "apple_1.glb"),
            scale=[scale * 0.8] * 3,
            pose=fix_rotation_pose,
        )
        builder.initial_pose = sapien.Pose(p=[0, -0.4, 0.78])
        self.apple = builder.build(name="apple")
```

### Smoke
```bash
cd <repo>
.venv/bin/python -c "import gymnasium as gym, mani_skill; e=gym.make('UnitreeG1PlaceAppleInBowl-v1'); print(e.observation_space, e.action_space); e.close()"
```
Expected stdout:
```
Box(-inf, inf, (1, 74), float32) Box(-1.0, 1.0, (25,), float32)
```

---

## §2 Actions

### Description
The G1 robot is controlled in joint space. The default control mode is `pd_joint_delta_pos`: each step the policy emits a normalized delta over all 25 active body joints (torso + both arms + both 6-DoF hands), added to the current joint targets, driven by a PD controller. There is no separate task-space / gripper sub-action — fingers are part of the same 25-D joint vector, so "gripper" control is implicit in the finger-joint dims.

### Decisions resolved
- control_mode: `pd_joint_delta_pos` (default; `pd_joint_pos` also available).
- action space: `Box(-1.0, 1.0, (25,))` (normalized).
- 25 joints (order = `body_joints`): `torso_joint`, then per-side arm joints (`{left,right}_shoulder_pitch`, `{left,right}_shoulder_roll`, `{left,right}_shoulder_yaw`, `{left,right}_elbow_pitch`, `{left,right}_elbow_roll`), then both hands' finger joints (`{left,right}_{zero,three,five,one,four,six,two}_joint`).
- delta limits (`pd_joint_delta_pos`): `lower = [-0.2]*11 + [-0.5]*14`, `upper = [0.2]*11 + [0.5]*14` (first 11 = torso+arms, last 14 = both hands' fingers).
- PD gains: `stiffness=1e3`, `damping=1e2`, `force_limit=100`, `use_delta=True`.
- `balance_passive_force=True` (gravity compensation on body joints).

### Code
Controller config (`g1_upper_body.py:99-124`):
```python
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
`body_joints` (25, `g1_upper_body.py:56-94`): `torso_joint`, `left_shoulder_pitch_joint`, `right_shoulder_pitch_joint`, `left_shoulder_roll_joint`, `right_shoulder_roll_joint`, `left_shoulder_yaw_joint`, `right_shoulder_yaw_joint`, `left_elbow_pitch_joint`, `right_elbow_pitch_joint`, `left_elbow_roll_joint`, `right_elbow_roll_joint`, `left_zero_joint`, `left_three_joint`, `left_five_joint`, `right_zero_joint`, `right_three_joint`, `right_five_joint`, `left_one_joint`, `left_four_joint`, `left_six_joint`, `right_one_joint`, `right_four_joint`, `right_six_joint`, `left_two_joint`, `right_two_joint`.
PD constants (`g1_upper_body.py:95-97`): `body_stiffness=1e3`, `body_damping=1e2`, `body_force_limit=100`.

### Smoke
```bash
.venv/bin/python -c "import gymnasium as gym, mani_skill; e=gym.make('UnitreeG1PlaceAppleInBowl-v1'); print(e.action_space, e.unwrapped.control_mode); e.close()"
```
Expected stdout:
```
Box(-1.0, 1.0, (25,), float32) pd_joint_delta_pos
```

---

## §3 Reset (`_initialize_episode`)

### Description
Per reset: build/reset the kitchen scene, set robot to the standing keyframe qpos at the fixed init pose, then randomize the apple and bowl xy positions on the tabletop and randomize the apple's z-axis (yaw) orientation. Apple/bowl z heights are fixed.

### Decisions resolved
- robot: `set_qpos(standing.qpos)` (all-zeros, 25 DoF), `set_pose(init_robot_pose)` with `p=[-0.3, 0, 0.755]`.
- apple xy: `uniform(low=-0.025, high=0.025, size=(b,2))`; z = `0.7335`; orientation = `random_quaternions(b, lock_x=True, lock_y=True)` (random yaw only).
- bowl xy: `uniform(low=-0.025, high=0.025, size=(b,2))` then `+= [0.0, -0.4]` (bowl offset 0.4 m away in -y); z = `0.753`; no orientation randomization.

### Code
Leaf `_initialize_episode` (`humanoid_pick_place.py:257-276`):
```python
    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        super()._initialize_episode(env_idx, options)
        with torch.device(self.device):
            b = len(env_idx)
            # initialize the robot
            self.agent.robot.set_qpos(self.agent.keyframes["standing"].qpos)
            self.agent.robot.set_pose(self.init_robot_pose)

            # initialize the apple to be within reach
            xyz = torch.zeros((b, 3))
            xyz[:, :2] = randomization.uniform(low=-0.025, high=0.025, size=(b, 2))
            qs = randomization.random_quaternions(b, lock_x=True, lock_y=True)
            xyz[:, 2] = 0.7335
            self.apple.set_pose(Pose.create_from_pq(xyz, qs))

            xyz = torch.zeros((b, 3))
            xyz[:, :2] = randomization.uniform(low=-0.025, high=0.025, size=(b, 2))
            xyz[:, :2] += torch.tensor([0.0, -0.4])
            xyz[:, 2] = 0.753
            self.bowl.set_pose(Pose.create_from_pq(xyz))
```
Base `_initialize_episode` (`humanoid_pick_place.py:59-60`): `self.scene_builder.initialize(env_idx)`.

### Smoke
```bash
.venv/bin/python -c "import gymnasium as gym, mani_skill; e=gym.make('UnitreeG1PlaceAppleInBowl-v1', num_envs=4); e.reset(seed=0); e.reset(seed=1); print('reset ok'); e.close()"
```
Expected: `reset ok` (no exception; apple/bowl poses differ across seeds).

---

## §4 Goal + Termination

### Description
Success = apple is inside the bowl AND the right hand has withdrawn above the bowl. There is no failure condition (always `False`). The episode terminates by time-out at 100 steps (`max_episode_steps=100`). `evaluate()` also exposes intermediate flags (`is_grasped`, `hand_outside_bowl`) used by the reward and observation.

### Decisions resolved
- `success = is_obj_placed & hand_outside_bowl` where:
  - `is_obj_placed`: `||bowl.pos - apple.pos|| <= 0.05` m.
  - `hand_outside_bowl`: `right_tcp.pos.z > bowl.pos.z + 0.125` m.
- `is_grasped = agent.right_hand_is_grasping(apple, max_angle=110)` (contact-force + angle based, see §5/§6 helper).
- No explicit fail term in the leaf (base `HumanoidPickPlaceEnv.evaluate` returns `fail=zeros`, but the leaf overrides `evaluate()` and does not return `fail`).
- `max_episode_steps = 100` (time-out termination).

### Code
Leaf `evaluate` (`humanoid_pick_place.py:138-150`):
```python
    def evaluate(self):
        is_obj_placed = (
            torch.linalg.norm(self.bowl.pose.p - self.apple.pose.p, axis=1) <= 0.05
        )
        hand_outside_bowl = (
            self.agent.right_tcp.pose.p[:, 2] > self.bowl.pose.p[:, 2] + 0.125
        )
        is_grasped = self.agent.right_hand_is_grasping(self.apple, max_angle=110)
        return {
            "success": is_obj_placed & hand_outside_bowl,
            "hand_outside_bowl": hand_outside_bowl,
            "is_grasped": is_grasped,
        }
```
No `CommandsCfg` — ManiSkill encodes the goal (bowl 3D position) directly in the observation, not via a command term.

### Smoke
```bash
.venv/bin/python -c "import gymnasium as gym, mani_skill; e=gym.make('UnitreeG1PlaceAppleInBowl-v1', num_envs=2); o,_=e.reset(seed=0); a=e.action_space.sample(); o,r,te,tr,i=e.step(a); print('success' in i, 'is_grasped' in i, e.unwrapped.max_episode_steps); e.close()"
```
Expected: `True True 100`.

---

## §5 Observation (`_get_obs_extra`)

### Description
Task-specific extra observation appended to the proprioceptive base obs. Always includes the grasp flag and the right-hand TCP pose; in `state`/`state_dict` obs modes it additionally exposes the bowl position, full apple pose, and the two relative vectors (tcp→apple, apple→goal). The default obs mode is `state` → flat vector of dim 74.

### Decisions resolved
- obs_mode default: `state`. Total dim = 74 (verified via build smoke).
- always-present extra: `is_grasped` (1), `tcp_pose` = right TCP raw_pose (7 = xyz + quat).
- state-only extra: `bowl_pos` (3), `obj_pose` = apple raw_pose (7), `tcp_to_obj_pos` (3), `obj_to_goal_pos` (3).
- extra-obs subtotal in state mode = 1+7+3+7+3+3 = 24; remaining 50 = base agent proprioception (25 qpos + 25 qvel for the 25-DoF body).

### Code
Leaf `_get_obs_extra` (`humanoid_pick_place.py:152-165`):
```python
    def _get_obs_extra(self, info: dict):
        # in reality some people hack is_grasped into observations by checking if the gripper can close fully or not
        obs = dict(
            is_grasped=info["is_grasped"],
            tcp_pose=self.agent.right_tcp.pose.raw_pose,
        )
        if "state" in self.obs_mode:
            obs.update(
                bowl_pos=self.bowl.pose.p,
                obj_pose=self.apple.pose.raw_pose,
                tcp_to_obj_pos=self.apple.pose.p - self.agent.right_tcp.pose.p,
                obj_to_goal_pos=self.bowl.pose.p - self.apple.pose.p,
            )
        return obs
```

### Smoke
```bash
.venv/bin/python -c "import gymnasium as gym, mani_skill; e=gym.make('UnitreeG1PlaceAppleInBowl-v1'); print(e.observation_space.shape); e.close()"
```
Expected stdout:
```
(1, 74)
```

---

## §6 Reward

### Description
Staged dense reward (composer = **sum** of additive terms, with overrides for late stages). Stages: (1) reach the apple, (2) grasp it, (3) carry it above the bowl, (4) when high above the bowl, switch to a fixed base of 4 + place + grasp-release shaping, (5) on success, base of 8 + place + grasp-release. The normalized variant divides the dense reward by 10. Supported reward modes include `sparse` (success flag) and `none`.

### Decisions resolved
- composer: additive (`reward = reaching + is_grasped + place*is_grasped`), with hard overrides via boolean indexing for the `obj_high_above_bowl` and `success` cohorts.
- `reaching_reward = 1 - tanh(5 * ||apple - right_tcp||)` (range ~[0,1]).
- grasp bonus: `+ is_grasped` (0/1).
- `place_reward = 1 - tanh(5 * ||(bowl + [0,0,0.15]) - apple||)`, added as `place_reward * is_grasped`.
- `obj_high_above_bowl = obj_to_goal_dist < 0.025`; for these envs `reward = 4 + place_reward + grasp_release_reward`.
- on `success`: `reward = 8 + place_reward + grasp_release_reward`.
- `grasp_release_reward = 1 - tanh(right_hand_dist_to_open_grasp())` (rewards opening the hand to drop the apple).
- normalized: `dense / 10` (max raw ≈ 10 → normalized ≈ [0,1]).
- planning budget (retro-computed, per-step saturated magnitudes): reach ≤ 1; +grasp 1; +place (gated by grasp) ≤ 1 → pre-drop max ≈ 3; high-above-bowl stage base 4 + place(≤1) + release(≤1) ≈ 6; success stage base 8 + place(≤1) + release(≤1) ≈ 10 (the /10 normalizer maps this to ~1.0).

### Code
Grasp-release helper (`humanoid_pick_place.py:167-169`):
```python
    def _grasp_release_reward(self):
        """a dense reward that rewards the agent for opening their hand"""
        return 1 - torch.tanh(self.agent.right_hand_dist_to_open_grasp())
```
Dense reward + normalized (`humanoid_pick_place.py:171-206`):
```python
    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: dict):
        tcp_to_obj_dist = torch.linalg.norm(
            self.apple.pose.p - self.agent.right_tcp.pose.p, axis=1
        )
        reaching_reward = 1 - torch.tanh(5 * tcp_to_obj_dist)
        reward = reaching_reward

        is_grasped = info["is_grasped"]
        reward += is_grasped

        # encourage to bring apple to above the bowl then drop it.
        obj_to_goal_dist = torch.linalg.norm(
            (self.bowl.pose.p + torch.tensor([0, 0, 0.15], device=self.device))
            - self.apple.pose.p,
            axis=1,
        )
        place_reward = 1 - torch.tanh(5 * obj_to_goal_dist)
        reward += place_reward * is_grasped

        # once above the goal, encourage to have the hand above the bowl still and begin releasing the grasp
        obj_high_above_bowl = obj_to_goal_dist < 0.025
        grasp_release_reward = self._grasp_release_reward()
        reward[obj_high_above_bowl] = (
            4
            + place_reward[obj_high_above_bowl]
            + grasp_release_reward[obj_high_above_bowl]
        )
        reward[info["success"]] = (
            8 + (place_reward + grasp_release_reward)[info["success"]]
        )
        return reward

    def compute_normalized_dense_reward(
        self, obs: Any, action: torch.Tensor, info: dict
    ):
        return self.compute_dense_reward(obs=obs, action=action, info=info) / 10
```
Reward-support helpers in the agent:
- `right_hand_dist_to_open_grasp` (`g1_upper_body.py:184-188`): `mean(|qpos[right_finger_joint_indexes]|)`.
- `right_hand_is_grasping(object, min_force=0.5, max_angle=85)` (`g1_upper_body.py:243-288`): contact-force magnitude + contact-angle test on `right_two_link` (one finger) AND (`right_four_link` OR `right_six_link`); the leaf calls it with `max_angle=110`.

### Smoke
```bash
.venv/bin/python -c "import gymnasium as gym, mani_skill, torch; e=gym.make('UnitreeG1PlaceAppleInBowl-v1', num_envs=4, reward_mode='dense'); e.reset(seed=0); _,r,_,_,_=e.step(e.action_space.sample()); print(torch.isfinite(torch.as_tensor(r)).all().item()); e.close()"
```
Expected: `True` (finite dense reward for all envs).

---

## §7 DR

`<no DR>`

There are no `startup` / `interval` domain-randomization events. All randomization is reset-time only (apple/bowl xy + apple yaw, in §3). The robot uses fixed PD gains, fixed friction (`finger` material static/dynamic 2.0), and a constant init qpos/pose; the kitchen scene is loaded deterministically. The base class accepts `robot_init_qpos_noise=0.02` but the leaf `_initialize_episode` sets robot qpos exactly to the standing keyframe (the noise param is unused here).

---

