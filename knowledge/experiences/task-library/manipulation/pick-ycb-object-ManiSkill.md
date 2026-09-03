# PickSingleYCB-v1 — Implementation Spec

- robot: Franka Panda with wrist camera (`panda_wristcam`; Panda / Fetch also supported)
- simulator: ManiSkill (SAPIEN, `BaseEnv` + `@register_env`)
- objects: random YCB object, green goal sphere, table
- bimanual: false
- summary: Lift a randomly sampled YCB object and move it to a goal position.

---

## §1 Registration + Scene

**Description.** A tabletop pick task where, per scene, a *random* YCB object is sampled (during reconfiguration) and must be lifted/moved to a random 3D goal position visualized by a green sphere. The robot is a `panda_wristcam` (Panda arm with a wrist RealSense camera), seated at `p=[-0.615, 0, 0]`. The scene is the standard `TableSceneBuilder` table + ground + lights. The object geometry itself is the per-scene randomization axis: each parallel sub-scene `i` loads `model_ids[i % n_models]`, and `reconfiguration_freq` controls whether a new object set is resampled on reset.

**Decisions resolved.**
- `id = "PickSingleYCB-v1"`, `max_episode_steps = 50`, `asset_download_ids = ["ycb"]`.
- `SUPPORTED_ROBOTS = ["panda", "panda_wristcam", "fetch"]`; default `robot_uids = "panda_wristcam"`.
- `goal_thresh = 0.025` (m) — both the success distance threshold and the goal-sphere radius.
- `robot_init_qpos_noise = 0.02` (default).
- Object set: all keys of `ASSET_DIR/assets/mani_skill2_ycb/info_pick_v0.json`, **minus** the 4 non-graspable ids `022_windex_bottle`, `028_skillet_lid`, `029_plate`, `059_chain`.
- Per-env object choice: `self._batched_episode_rng.choice(self.all_model_ids, replace=True)` → sub-scene `i` builds object `model_ids[i]` via `actors.get_actor_builder(self.scene, id=f"ycb:{model_id}")`; all per-scene actors merged into one `Actor` named `ycb_object`.
- `reconfiguration_freq` default: `1` when `num_envs == 1`, else `0` (objects fixed across resets unless `reconfigure=True` is passed or `reconfiguration_freq >= 1`).
- `goal_site`: kinematic sphere, radius `goal_thresh`, color green `[0,1,0,1]`, no collision; appended to `self._hidden_objects` (hidden in non-human renders).
- Robot root pose set in `_load_agent`: `sapien.Pose(p=[-0.615, 0, 0])`.
- Sim cfg: `GPUMemoryConfig(max_rigid_contact_count=2**20, max_rigid_patch_count=2**19)`.
- `_after_reconfigure` computes per-object `object_zs = -bbox.bounds[0,2]` so each object's bottom rests at table `z=0`.
- Sensors: base_camera 128×128 `look_at(eye=[0.3,0,0.6], target=[-0.1,0,0.1])`; human render cam 512×512 `look_at([0.6,0.7,0.6],[0,0,0.35])`. (wrist `hand_camera` 128×128 added by `PandaWristCam`, mounted on `camera_link`.)

**Code.**
```python
@register_env("PickSingleYCB-v1", max_episode_steps=50, asset_download_ids=["ycb"])
class PickSingleYCBEnv(BaseEnv):
    _sample_video_link = "https://github.com/haosulab/ManiSkill/raw/main/figures/environment_demos/PickSingleYCB-v1_rt.mp4"

    SUPPORTED_ROBOTS = ["panda", "panda_wristcam", "fetch"]
    agent: Union[Panda, PandaWristCam, Fetch]
    goal_thresh = 0.025

    def __init__(
        self,
        *args,
        robot_uids="panda_wristcam",
        robot_init_qpos_noise=0.02,
        num_envs=1,
        reconfiguration_freq=None,
        **kwargs,
    ):
        self.robot_init_qpos_noise = robot_init_qpos_noise
        self.model_id = None
        self.all_model_ids = np.array(
            [
                k
                for k in load_json(
                    ASSET_DIR / "assets/mani_skill2_ycb/info_pick_v0.json"
                ).keys()
                if k
                not in [
                    "022_windex_bottle",
                    "028_skillet_lid",
                    "029_plate",
                    "059_chain",
                ]  # NOTE (arth): ignore these non-graspable/hard to grasp ycb objects
            ]
        )
        if reconfiguration_freq is None:
            if num_envs == 1:
                reconfiguration_freq = 1
            else:
                reconfiguration_freq = 0
        super().__init__(
            *args,
            robot_uids=robot_uids,
            reconfiguration_freq=reconfiguration_freq,
            num_envs=num_envs,
            **kwargs,
        )

    @property
    def _default_sim_config(self):
        return SimConfig(
            gpu_memory_config=GPUMemoryConfig(
                max_rigid_contact_count=2**20, max_rigid_patch_count=2**19
            )
        )

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
        global WARNED_ONCE
        self.table_scene = TableSceneBuilder(
            env=self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.table_scene.build()

        # randomize the list of all possible models in the YCB dataset
        # then sub-scene i will load model model_ids[i % number_of_ycb_objects]
        model_ids = self._batched_episode_rng.choice(self.all_model_ids, replace=True)
        if (
            self.num_envs > 1
            and self.num_envs < len(self.all_model_ids)
            and self.reconfiguration_freq <= 0
            and not WARNED_ONCE
        ):
            WARNED_ONCE = True
            print(
                """There are less parallel environments than total available models to sample.
                Not all models will be used during interaction even after resets unless you call env.reset(options=dict(reconfigure=True))
                or set reconfiguration_freq to be >= 1."""
            )

        self._objs: list[Actor] = []
        self.obj_heights = []
        for i, model_id in enumerate(model_ids):
            # TODO: before official release we will finalize a metadata dataclass that these build functions should return.
            builder = actors.get_actor_builder(
                self.scene,
                id=f"ycb:{model_id}",
            )
            builder.initial_pose = sapien.Pose(p=[0, 0, 0])
            builder.set_scene_idxs([i])
            self._objs.append(builder.build(name=f"{model_id}-{i}"))
            self.remove_from_state_dict_registry(self._objs[-1])
        self.obj = Actor.merge(self._objs, name="ycb_object")
        self.add_to_state_dict_registry(self.obj)

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

    def _after_reconfigure(self, options: dict):
        self.object_zs = []
        for obj in self._objs:
            collision_mesh = obj.get_first_collision_mesh()
            # this value is used to set object pose so the bottom is at z=0
            self.object_zs.append(-collision_mesh.bounding_box.bounds[0, 2])
        self.object_zs = common.to_tensor(self.object_zs, device=self.device)
```

**Asset paths (resolved).**
- Object metadata JSON: `<ASSET_DIR>/assets/mani_skill2_ycb/info_pick_v0.json` → on this host `~/.maniskill/data/assets/mani_skill2_ycb/info_pick_v0.json`. **WARN: not present** — the `ycb` asset set has not been downloaded; download via `python -m mani_skill.utils.download_asset ycb`.
- Per-object meshes/collision: resolved internally by `actors.get_actor_builder(scene, id=f"ycb:{model_id}")`.

**Smoke (§1 build).** `not captured (verified via source read)`. Command (run after assets are downloaded + non-interactively):
```bash
cd <repo> && .venv/bin/python -c "import gymnasium as gym, mani_skill; e=gym.make('PickSingleYCB-v1'); print(e.observation_space, e.action_space); e.close()"
```
Expected (analytic, state obs_mode): `Box(-inf, inf, (45,), float32) Box(-1.0, 1.0, (8,), float32)`.

---

## §2 Actions

**Description.** Default controller is the Panda's `pd_joint_delta_pos` (first key in `controller_configs`, so it is the default control mode): a 7-D incremental joint-position command on the arm + a 1-D mimic gripper command, total action dim 8, each in `[-1, 1]`.

**Decisions resolved.**
- `control_mode = "pd_joint_delta_pos"` (default; first key of `Panda._controller_configs`).
- Arm: `PDJointPosControllerConfig` over the 7 `panda_joint{1..7}`, `use_delta=True`, `lower=-0.1`, `upper=0.1`, `stiffness=1e3`, `damping=1e2`, `force_limit=100`. Normalized action ⇒ a `+1` action maps to `+0.1` rad delta per joint.
- Gripper: `PDJointPosMimicControllerConfig` over `panda_finger_joint{1,2}`, `lower=-0.01` (trick for thin objects), `upper=0.04`, `stiffness=1e3`, `damping=1e2`, `force_limit=100`, mimic `panda_finger_joint2 -> panda_finger_joint1` ⇒ a single 1-D gripper action.
- `ee_link_name = "panda_hand_tcp"` (`self.agent.tcp`).
- Total action dim = 7 (arm) + 1 (gripper) = 8.

**Code (Panda controller configs, verbatim).**
```python
    arm_joint_names = [
        "panda_joint1", "panda_joint2", "panda_joint3", "panda_joint4",
        "panda_joint5", "panda_joint6", "panda_joint7",
    ]
    gripper_joint_names = ["panda_finger_joint1", "panda_finger_joint2"]
    ee_link_name = "panda_hand_tcp"

    arm_stiffness = 1e3
    arm_damping = 1e2
    arm_force_limit = 100
    gripper_stiffness = 1e3
    gripper_damping = 1e2
    gripper_force_limit = 100

    @property
    def _controller_configs(self):
        arm_pd_joint_delta_pos = PDJointPosControllerConfig(
            self.arm_joint_names,
            lower=-0.1,
            upper=0.1,
            stiffness=self.arm_stiffness,
            damping=self.arm_damping,
            force_limit=self.arm_force_limit,
            use_delta=True,
        )
        ...
        gripper_pd_joint_pos = PDJointPosMimicControllerConfig(
            self.gripper_joint_names,
            lower=-0.01,  # a trick to have force when the object is thin
            upper=0.04,
            stiffness=self.gripper_stiffness,
            damping=self.gripper_damping,
            force_limit=self.gripper_force_limit,
            mimic={"panda_finger_joint2": {"joint": "panda_finger_joint1"}},
        )

        controller_configs = dict(
            pd_joint_delta_pos=dict(
                arm=arm_pd_joint_delta_pos, gripper=gripper_pd_joint_pos
            ),
            pd_joint_pos=dict(arm=arm_pd_joint_pos, gripper=gripper_pd_joint_pos),
            pd_ee_delta_pos=dict(arm=arm_pd_ee_delta_pos, gripper=gripper_pd_joint_pos),
            pd_ee_delta_pose=dict(arm=arm_pd_ee_delta_pose, gripper=gripper_pd_joint_pos),
            pd_ee_pose=dict(arm=arm_pd_ee_pose, gripper=gripper_pd_joint_pos),
            ...
        )
        return deepcopy_dict(controller_configs)
```
(Full controller table in `mani_skill/agents/robots/panda/panda.py:77–218`; the env overrides nothing, so the default `pd_joint_delta_pos` is used.)

**Smoke (§2).** Covered by the §1 build: `action_space == Box(-1, 1, (8,), float32)`.

---

## §3 Reset

**Description.** Per-episode initialization (`_initialize_episode`) randomizes the object's xy on the table within a ±0.1 m square and its yaw (z-axis rotation only; x/y locked so it lies flat), sets a random 3D goal (xy in the same ±0.1 m square, z = object_z + U[0, 0.3] so the goal floats above the table), and resets the Panda to a "high" rest qpos with Gaussian joint noise.

**Decisions resolved.**
- Object xy: `rand(b,2) * 0.2 - 0.1` ⇒ each in `[-0.1, 0.1]`; z = `object_zs[env_idx]` (bottom on table).
- Object orientation: `random_quaternions(b, lock_x=True, lock_y=True)` ⇒ random yaw only.
- Goal xy: same `rand*0.2-0.1` ⇒ `[-0.1, 0.1]`; goal z = `object_z + rand*0.3` ⇒ `[object_z, object_z+0.3]`.
- Robot reset qpos (panda / panda_wristcam): `[0, 0, 0, -2π/3, 0, 2π/3, π/4, 0.04, 0.04]` (arm raised higher than other tabletop tasks), plus `N(0, robot_init_qpos_noise=0.02)` on the 7 arm joints (fingers untouched); root pose re-set to `[-0.615, 0, 0]`. Non-panda robots raise `NotImplementedError`.
- `self.table_scene.initialize(env_idx)` resets the table/robot scaffold.

**Code.**
```python
    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)
            xyz = torch.zeros((b, 3))
            xyz[:, :2] = torch.rand((b, 2)) * 0.2 - 0.1
            xyz[:, 2] = self.object_zs[env_idx]
            qs = random_quaternions(b, lock_x=True, lock_y=True)
            self.obj.set_pose(Pose.create_from_pq(p=xyz, q=qs))

            goal_xyz = torch.zeros((b, 3))
            goal_xyz[:, :2] = torch.rand((b, 2)) * 0.2 - 0.1
            goal_xyz[:, 2] = torch.rand((b)) * 0.3 + xyz[:, 2]
            self.goal_site.set_pose(Pose.create_from_pq(goal_xyz))

            if self.robot_uids == "panda" or self.robot_uids == "panda_wristcam":
                # fmt: off
                qpos = np.array(
                    [0.0, 0, 0, -np.pi * 2 / 3, 0, np.pi * 2 / 3, np.pi / 4, 0.04, 0.04]
                )
                # fmt: on
                qpos[:-2] += self._episode_rng.normal(
                    0, self.robot_init_qpos_noise, len(qpos) - 2
                )
                self.agent.reset(qpos)
                self.agent.robot.set_root_pose(sapien.Pose([-0.615, 0, 0]))
            else:
                raise NotImplementedError(self.robot_uids)
```

**Smoke (§3).** Behavioral: after two `env.reset(seed=k)` calls with different `k`, the object pose and goal_pos differ; with the same seed they match. (Run after assets present.)

---

## §4 Goal + Termination

**Description.** Success = object within `goal_thresh` (0.025 m) of the goal AND the robot is static (per-joint |qvel| < 0.2). There is no separate failure condition; episodes otherwise end by timeout at `max_episode_steps = 50`. `evaluate()` returns a per-env info dict consumed by both reward and the obs (`is_grasped`).

**Decisions resolved.**
- `success = is_obj_placed AND is_robot_static`.
- `is_obj_placed = ||goal.p - obj.p|| <= 0.025`.
- `is_robot_static = self.agent.is_static(0.2)` (all arm joint velocities below 0.2).
- `is_grasped = self.agent.is_grasping(self.obj)`.
- Timeout: `max_episode_steps = 50` (from `@register_env`).
- No goal/command term beyond the sampled `goal_site` position (exposed as `goal_pos` obs).

**Code.**
```python
    def evaluate(self):
        obj_to_goal_pos = self.goal_site.pose.p - self.obj.pose.p
        is_obj_placed = torch.linalg.norm(obj_to_goal_pos, axis=1) <= self.goal_thresh
        is_grasped = self.agent.is_grasping(self.obj)
        is_robot_static = self.agent.is_static(0.2)
        return dict(
            is_grasped=is_grasped,
            obj_to_goal_pos=obj_to_goal_pos,
            is_obj_placed=is_obj_placed,
            is_robot_static=is_robot_static,
            is_grasping=self.agent.is_grasping(self.obj),
            success=torch.logical_and(is_obj_placed, is_robot_static),
        )
```

**Smoke (§4).** Behavioral: `info["success"]` is bool, shape `(num_envs,)`; episodes truncate at step 50.

---

## §5 Observation

**Description.** `_get_obs_extra` exposes task-specific terms; in any `state`-containing obs mode it additionally exposes privileged object/relative-vector terms. The default obs mode is `state` (flattened `state_dict`), which concatenates proprioception (qpos/qvel) + the extra dict.

**Decisions resolved (default obs_mode = `state`).**
- `SUPPORTED_OBS_MODES = ("state", "state_dict", "none", "sensor_data", "any_textures", "pointcloud")` (default = `state`).
- Proprioception (`get_proprioception`): `qpos(9) + qvel(9) = 18` (Panda has 7 arm + 2 finger joints; `pd_joint_delta_pos` controller carries no extra controller state).
- `_get_obs_extra` (always): `tcp_pose(7) + goal_pos(3) + is_grasped(1) = 11`.
- `_get_obs_extra` (state-only additions): `tcp_to_goal_pos(3) + obj_pose(7) + tcp_to_obj_pos(3) + obj_to_goal_pos(3) = 16`.
- **Total state obs dim = 18 + 11 + 16 = 45** ⇒ `Box(-inf, inf, (45,), float32)`.
- `tcp_pose` / `obj_pose` are 7-D raw poses (xyz + wxyz quaternion). `is_grasped` is float 0/1.
- No observation noise / corruption is configured by this env (default ManiSkill behavior; no `enable_corruption` override).

**Code.**
```python
    def _get_obs_extra(self, info: dict):
        obs = dict(
            tcp_pose=self.agent.tcp.pose.raw_pose,
            goal_pos=self.goal_site.pose.p,
            is_grasped=info["is_grasped"],
        )
        if "state" in self.obs_mode:
            obs.update(
                tcp_to_goal_pos=self.goal_site.pose.p - self.agent.tcp.pose.p,
                obj_pose=self.obj.pose.raw_pose,
                tcp_to_obj_pos=self.obj.pose.p - self.agent.tcp.pose.p,
                obj_to_goal_pos=self.goal_site.pose.p - self.obj.pose.p,
            )
        return obs
```

**Smoke (§5).** Covered by §1 build: `observation_space == Box(-inf, inf, (45,), float32)` in state mode.

---

## §6 Reward

**Description.** Dense, staged, **sum-composed** shaping reward. It rewards (1) reaching the object, (2) grasping it, (3) — gated on grasp — moving it toward the goal, (4) a bonus once placed, and (5) — gated on placed+grasped — bringing the robot to rest. On success the reward is hard-set to the max value `6`. The normalized variant divides by `6`. Composer = **sum**; max attainable (pre-success-override) ≈ 6, and the success override pins exactly `6`.

**Decisions resolved.**
- Composer: **sum** (terms added into a single `reward` scalar).
- `success` bonus / cap: `reward[info["success"]] = 6` (overrides all shaping at success).
- Per-stage saturated per-step magnitudes (retro-computed from the code; each `1 - tanh(5·d)` term saturates to 1 as distance → 0):
  - reaching: 1 (always active).
  - grasp: 1 (when `is_grasped`).
  - place: 1 (when grasped, object at goal).
  - placed bonus: 1 (when grasped + placed).
  - static: 1 (when grasped + placed + robot at rest).
  - ⇒ shaping ceiling = 5; success override sets exactly 6 (normalized ⇒ 1.0).

**Code (verbatim, full).**
```python
    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: dict):
        tcp_to_obj_dist = torch.linalg.norm(
            self.obj.pose.p - self.agent.tcp.pose.p, axis=1
        )
        reaching_reward = 1 - torch.tanh(5 * tcp_to_obj_dist)
        reward = reaching_reward

        is_grasped = info["is_grasped"]
        reward += is_grasped

        obj_to_goal_dist = torch.linalg.norm(
            self.goal_site.pose.p - self.obj.pose.p, axis=1
        )
        place_reward = 1 - torch.tanh(5 * obj_to_goal_dist)
        reward += place_reward * is_grasped

        reward += info["is_obj_placed"] * is_grasped

        static_reward = 1 - torch.tanh(
            5 * torch.linalg.norm(self.agent.robot.get_qvel()[..., :-2], axis=1)
        )
        reward += static_reward * info["is_obj_placed"] * is_grasped

        reward[info["success"]] = 6
        return reward

    def compute_normalized_dense_reward(
        self, obs: Any, action: torch.Tensor, info: dict
    ):
        return self.compute_dense_reward(obs=obs, action=action, info=info) / 6
```

Helpers used: `info["is_grasped"]`, `info["is_obj_placed"]`, `info["success"]` (all from `evaluate()` above), `self.obj.pose.p`, `self.agent.tcp.pose.p`, `self.goal_site.pose.p`, `self.agent.robot.get_qvel()[..., :-2]` (arm qvel only, excluding the 2 fingers). No latch buffers; reward is purely state-functional.

**Smoke (§6).** Behavioral: per step `reward` is finite, non-constant across a random rollout, in `[0, 6]` (normalized in `[0, 1]`); at any env where `info["success"]` is True the dense reward equals exactly `6`.

---

## §7 DR

**Description.** `<no DR>` beyond episode-initialization randomization. ManiSkill randomizes only at episode init / reconfiguration; there are no `startup`/`interval` domain-randomization events in this env. The only randomized factors are:
- object geometry (sampled during reconfiguration, §1 `_load_scene`),
- object xy position + yaw (episode init, §3),
- goal position (episode init, §3),
- robot init qpos Gaussian noise `N(0, 0.02)` (episode init, §3).

No physics-parameter / friction / mass / observation-noise randomization is configured.

---

## Reproduce

```
probe-task: wrote <ManiSkill-repo>/harbor/create-task/picksingleycb-v1-implementation.md (sections §1..§7, 2 reward funcs, 7 obs terms)
             Reproduce via: /harbor:task-create name=<new_task_id> from=<ManiSkill-repo>/harbor/create-task/picksingleycb-v1-implementation.md
```
