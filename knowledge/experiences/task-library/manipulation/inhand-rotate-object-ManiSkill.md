# RotateSingleObjectInHandLevel1-v1 — Implementation Spec

- robot: Allegro right hand with FSR touch links (16 DoF, fixed base)
- simulator: ManiSkill (SAPIEN, `BaseEnv` + `@register_env`)
- objects: single dynamic object (cube or YCB object by level), table
- bimanual: false
- summary: Rotate an object in-hand to a target orientation without dropping it.

> **One env class, 4 levels.** `RotateSingleObjectInHand(BaseEnv)` is the shared base. The four registered ids (`Level0..3-v1`) are thin subclasses that pass a single `difficulty_level` int (0/1/2/3) and otherwise share identical noise (`robot_init_qpos_noise=0.02`, `obj_init_pos_noise=0.02`). **Level1 sets `difficulty_level=1`**, which selects a *size-randomized white box* (each parallel env gets an independently sampled half-size). Everything else below is the shared base behavior; per-level deltas are called out inline.

---

## §1 Registration + Scene

### Description
Registers the four difficulty levels of the in-hand rotation task. All use the same `BaseEnv` subclass, the same Allegro touch hand, a `TableSceneBuilder` table, and a single dynamic object floating at `z = hand_init_height + obj_height`. Level1 builds a per-env size-randomized white cube. Sim config bumps GPU contact-buffer capacities (dexterous contact is contact-rich).

### Decisions resolved
- `robot_uids = "allegro_hand_right_touch"` (forced in `super().__init__`).
- `_clearance = 0.003`, `hand_init_height = 0.25`.
- Level1: `difficulty_level=1`, `robot_init_qpos_noise=0.02`, `obj_init_pos_noise=0.02`.
- `max_episode_steps=300` (from `@register_env`).
- Default `sim_freq=100`, `control_freq=20` → `sim_steps_per_control=5`, control `dt=0.05 s`. (ManiSkill defaults; env does not override `sim_freq`/`control_freq`.)
- Default `obs_mode="state"`, default `control_mode="pd_joint_delta_pos"` (first key of the agent's controller dict).
- Level1 object: per-env half-size `= (randn()*0.1 + 1) * 0.04` → ~0.04 m half-size white box, one independent draw per parallel env via `_batched_episode_rng`. `obj_heights[i] = half_size[i]`.
- Reconfiguration freq: `None` for levels 0/1 (no per-reset reconfigure; geometry fixed after build).
- Asset: NO external download for Level1 (box built procedurally). Levels 2/3 need YCB (`asset_download_ids=["ycb"]`, reads `ASSET_DIR/assets/mani_skill2_ycb/info_pick_v0.json`). **WARN: YCB assets not present on this host** (`mani_skill/data` absent) — irrelevant to Level1.

### Code (registration)
```python
@register_env("RotateSingleObjectInHandLevel1-v1", max_episode_steps=300)
class RotateSingleObjectInHandLevel1(RotateSingleObjectInHand):
    def __init__(self, *args, **kwargs):
        super().__init__(
            *args,
            robot_init_qpos_noise=0.02,
            obj_init_pos_noise=0.02,
            difficulty_level=1,
            **kwargs,
        )
```

### Code (base __init__ + sim cfg + sensors)
```python
class RotateSingleObjectInHand(BaseEnv):
    agent: Union[AllegroHandRightTouch]
    _clearance = 0.003
    hand_init_height = 0.25

    def __init__(
        self,
        *args,
        robot_init_qpos_noise=0.02,
        obj_init_pos_noise=0.02,
        difficulty_level: int = -1,
        num_envs=1,
        reconfiguration_freq=None,
        **kwargs,
    ):
        self.robot_init_qpos_noise = robot_init_qpos_noise
        self.obj_init_pos_noise = obj_init_pos_noise
        self.obj_heights: torch.Tensor = torch.Tensor()
        if (
            not isinstance(difficulty_level, int)
            or difficulty_level >= 4
            or difficulty_level < 0
        ):
            raise ValueError(
                f"Difficulty level must be a int within 0-3, but get {difficulty_level}"
            )
        self.difficulty_level = difficulty_level
        if self.difficulty_level >= 2:
            if reconfiguration_freq is None:
                if num_envs == 1:
                    reconfiguration_freq = 1
                else:
                    reconfiguration_freq = 0
        super().__init__(
            *args,
            robot_uids="allegro_hand_right_touch",
            num_envs=num_envs,
            reconfiguration_freq=reconfiguration_freq,
            **kwargs,
        )

        with torch.device(self.device):
            self.prev_unit_vector = torch.zeros((self.num_envs, 3))
            self.cum_rotation_angle = torch.zeros((self.num_envs,))

    @property
    def _default_sim_config(self):
        return SimConfig(
            gpu_memory_config=GPUMemoryConfig(
                max_rigid_contact_count=self.num_envs * max(1024, self.num_envs) * 8,
                max_rigid_patch_count=self.num_envs * max(1024, self.num_envs) * 2,
                found_lost_pairs_capacity=2**26,
            )
        )

    @property
    def _default_sensor_configs(self):
        pose = sapien_utils.look_at(
            eye=[0.15, 0, 0.45], target=[-0.1, 0, self.hand_init_height]
        )
        return [CameraConfig("base_camera", pose, 128, 128, np.pi / 2, 0.01, 100)]

    @property
    def _default_human_render_camera_configs(self):
        pose = sapien_utils.look_at([0.2, 0.4, 0.6], [0.0, 0.0, 0.3])
        return CameraConfig("render_camera", pose, 512, 512, 1, 0.01, 100)
```

### Code (_load_scene — Level1 branch highlighted)
```python
    def _load_scene(self, options: dict):
        self.table_scene = TableSceneBuilder(
            env=self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.table_scene.build()

        obj_heights = []
        if self.difficulty_level == 0:
            self.obj = build_cube(
                self.scene, half_size=0.04,
                color=np.array([255, 255, 255, 255]) / 255,
                name="cube", body_type="dynamic",
            )
            obj_heights.append(0.03)
        elif self.difficulty_level == 1:                         # <-- LEVEL 1
            half_sizes = (self._batched_episode_rng.randn() * 0.1 + 1) * 0.04
            self._objs: list[Actor] = []
            for i, half_size in enumerate(half_sizes):
                builder = self.scene.create_actor_builder()
                builder.add_box_collision(half_size=[half_size] * 3)
                builder.add_box_visual(
                    half_size=[half_size] * 3,
                    material=sapien.render.RenderMaterial(
                        base_color=np.array([255, 255, 255, 255]) / 255,
                    ),
                )
                builder.set_scene_idxs([i])
                self._objs.append(builder.build(name=f"cube-{i}"))
                obj_heights.append(half_size)
            self.obj = Actor.merge(self._objs, name="cube")
        elif self.difficulty_level >= 2:
            all_model_ids = np.array(
                list(load_json(ASSET_DIR / "assets/mani_skill2_ycb/info_pick_v0.json").keys())
            )
            model_ids = self._batched_episode_rng.choice(all_model_ids)
            self._objs: list[Actor] = []
            for i, model_id in enumerate(model_ids):
                builder = actors.get_actor_builder(self.scene, id=f"ycb:{model_id}")
                builder.set_scene_idxs([i])
                self._objs.append(builder.build(name=f"{model_id}-{i}"))
            self.obj = Actor.merge(self._objs, name="ycb_object")
        else:
            raise ValueError(...)

        if self.difficulty_level < 2:
            self.obj_heights = common.to_tensor(obj_heights, device=self.device)
```

### Agent — AllegroHandRightTouch (robot)
- URDF: `{PACKAGE_ASSET_DIR}/robots/allegro/variation/allegro_hand_right_fsr_simple.urdf` (right Allegro with FSR touch sensor links).
- 16 revolute joints: `joint_0.0 … joint_15.0`.
- PD gains (per joint): `stiffness=4e2`, `damping=1e1`, `force_limit=5e1`.
- 4 fingertip links (order thumb, index, middle, ring): `link_15.0_tip, link_3.0_tip, link_7.0_tip, link_11.0_tip` (note `_load_agent`'s `tip_links` ordering vs. the touch FSR-link list differ — env uses `self.agent.tip_links`).
- Tip material: `static_friction=2.0, dynamic_friction=1.0, restitution=0.0`, `patch_radius=0.1`.
- FSR touch links (12 finger + 4 palm) drive `get_fsr_impulse()` → `fsr_impulse` (16,) proprioception term.

### Smoke (§1 build)
```bash
cd <ManiSkill-repo>
.venv/bin/python -c "import gymnasium as gym, mani_skill; e=gym.make('RotateSingleObjectInHandLevel1-v1'); print(e.observation_space, e.action_space); e.close()"
```
Expected stdout:
```
OBS Box(-inf, inf, (1, 105), float32)
ACT Box(-1.0, 1.0, (16,), float32)
```

---

## §2 Actions

### Description
The hand is position-controlled. The default (and only sensible RL) control mode is `pd_joint_delta_pos`: the 16-D action in `[-1, 1]` is a per-joint delta added to current joint targets, clamped to `±0.1 rad`, tracked by PD with the gains above. Two other modes exist (`pd_joint_pos`, `pd_joint_target_delta_pos`) but the env relies on `controller.config.{stiffness,damping,force_limit}` for the effort/torque reward terms regardless.

### Decisions resolved
- `control_mode = "pd_joint_delta_pos"` (default — first key).
- Action space: `Box(-1, 1, (16,))`.
- Delta range: `[-0.1, 0.1] rad` per step, `use_delta=True`, default `normalize_action=True` (so action ∈ [-1,1] maps to that range).
- `pd_joint_pos` variant: `normalize_action=False`, absolute targets, no limits given.
- Controller gains used by reward terms: `stiffness=4e2`, `damping=1e1`, `force_limit=5e1` (broadcast to (num_envs, 16) as `self.controller_param`).

### Code (controller configs — from AllegroHandRight)
```python
    @property
    def _controller_configs(self):
        joint_pos = PDJointPosControllerConfig(
            self.joint_names, None, None,
            self.joint_stiffness, self.joint_damping, self.joint_force_limit,
            normalize_action=False,
        )
        joint_delta_pos = PDJointPosControllerConfig(
            self.joint_names, -0.1, 0.1,
            self.joint_stiffness, self.joint_damping, self.joint_force_limit,
            use_delta=True,
        )
        joint_target_delta_pos = deepcopy(joint_delta_pos)
        joint_target_delta_pos.use_target = True
        controller_configs = dict(
            pd_joint_delta_pos=joint_delta_pos,
            pd_joint_pos=joint_pos,
            pd_joint_target_delta_pos=joint_target_delta_pos,
        )
        return deepcopy_dict(controller_configs)
```
(`self.joint_stiffness=4e2`, `self.joint_damping=1e1`, `self.joint_force_limit=5e1`.)

### Smoke (§2 action roundtrip)
```bash
.venv/bin/python -c "import gymnasium as gym, mani_skill, numpy as np; e=gym.make('RotateSingleObjectInHandLevel1-v1'); e.reset(seed=0); a=e.action_space.sample(); o,r,t,tr,i=e.step(a); print(a.shape, float(np.asarray(r).reshape(-1)[0])); e.close()"
```
Expected: action shape `(16,)`, a finite scalar reward.

---

## §3 Reset (`_initialize_episode`)

### Description
On reset, the table scene is initialized, the object is dropped to a randomized position just above the hand (`z = hand_init_height + obj_height`, x/y jittered by Gaussian noise `*obj_init_pos_noise`), with identity orientation. The rotation axis is fixed to the world Z axis for levels ≤2 (Level1: axis index 2). A unit tangent vector on that axis's plane is sampled and stored to measure incremental rotation. The hand is reset to zero qpos and posed palm-up at `z=0.25`. Cumulative rotation angle and per-env controller params are (re)initialized. Success threshold = `4π`.

### Decisions resolved
- Object xy noise: `randn((b,3)) * obj_init_pos_noise` (0.02); z forced positive + `hand_init_height(0.25) + obj_heights`.
- Object init quaternion: identity `[1,0,0,0]`.
- Rotation axis (Level1, `difficulty_level<=2`): all envs → axis index 2 (Z). `rot_dir = one_hot(2)`. (Level3 randomizes axis ∈ {0,1,2}.)
- Tangent unit vector: `one_hot((axis+1)%3)` → for Z axis that's the X unit vector.
- `success_threshold = π * 4`.
- Hand reset: `qpos = zeros(16)`, base pose `p=[0,0,0.25]`, `q=[-0.707,0,0.707,0]` (palm-up keyframe).
- Per-env controller params cached from `agent.controller.config.{stiffness,damping,force_limit}`, expanded to (num_envs, 16).

### Code
```python
    def _initialize_episode(self, env_idx, options):
        self._initialize_actors(env_idx)
        self._initialize_agent(env_idx)

    def _initialize_actors(self, env_idx):
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)
            new_pos = torch.randn((b, 3)) * self.obj_init_pos_noise
            new_pos[:, 2] = (torch.abs(new_pos[:, 2]) + self.hand_init_height + self.obj_heights)
            new_pose = torch.zeros((b, 7))
            new_pose[:, 0:3] = new_pos
            new_pose[:, 3:7] = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
            self.obj.set_pose(new_pose)

            if self.difficulty_level <= 2:
                axis = torch.ones((b,), dtype=torch.long) * 2
            else:
                axis = torch.randint(0, 3, (b,), dtype=torch.long)
            if not hasattr(self, 'rot_dir'):
                self.rot_dir = F.one_hot(axis, num_classes=3)
            else:
                self.rot_dir[env_idx] = F.one_hot(axis, num_classes=3)

            vector_axis = (axis + 1) % 3
            vector = F.one_hot(vector_axis, num_classes=3).float()

            self.success_threshold = torch.pi * 4
            stiffness = torch.tensor(self.agent.controller.config.stiffness)
            damping = torch.tensor(self.agent.controller.config.damping)
            force_limit = torch.tensor(self.agent.controller.config.force_limit)
            self.controller_param = (
                stiffness.expand(self.num_envs, self.agent.robot.dof[0]),
                damping.expand(self.num_envs, self.agent.robot.dof[0]),
                force_limit.expand(self.num_envs, self.agent.robot.dof[0]),
            )
            if not hasattr(self, 'unit_vector'):
                self.unit_vector = vector
                self.prev_unit_vector = vector.clone()
                self.cum_rotation_angle = torch.zeros((b,))
            else:
                self.unit_vector[env_idx] = vector
                self.prev_unit_vector[env_idx] = vector.clone()
                self.cum_rotation_angle[env_idx] = 0.0

    def _initialize_agent(self, env_idx):
        with torch.device(self.device):
            b = len(env_idx)
            dof = self.agent.robot.dof
            if isinstance(dof, torch.Tensor):
                dof = dof[0]
            init_qpos = torch.zeros((b, dof))
            self.agent.reset(init_qpos)
            self.agent.robot.set_pose(
                Pose.create_from_pq(
                    torch.tensor([0.0, 0, self.hand_init_height]),
                    torch.tensor([-0.707, 0, 0.707, 0]),
                )
            )
```

### Smoke (§3)
```bash
.venv/bin/python -c "import gymnasium as gym, mani_skill, torch; e=gym.make('RotateSingleObjectInHandLevel1-v1'); e.reset(seed=1); print('reset ok'); e.close()"
```
Expected: `reset ok`.

---

## §4 Goal + Termination (`evaluate` + max_episode_steps)

### Description
There is no `CommandsCfg` / goal command. The goal is implicit: rotate the object about the fixed axis until the **cumulative** signed rotation angle exceeds `4π` (~two full turns). Each step `evaluate()` measures the incremental rotation of the stored tangent unit vector projected onto the rotation plane, clips it to `±π/20` per step (anti-jump), accumulates it, and sets `success` when it crosses `success_threshold`. The episode also tracks falling (`obj_fall`/`fail`) when the object drops below `hand_init_height - 0.05`. Hard time limit: `max_episode_steps=300`.

### Decisions resolved
- Success: `cum_rotation_angle > 4π`.
- Per-step angle clip: `[-π/20, π/20]`.
- Fall / fail: `obj_pose.p[:,2] < hand_init_height - 0.05` (= 0.20). Returned as both `obj_fall` and `fail`.
- `evaluate()` also computes obj velocity, finger-tip→obj vectors/distances, controller torque `qf` and `power` (consumed by §6 reward and §5 obs).
- Time limit: 300 control steps (= 15 s at dt=0.05).

### Code
```python
    def evaluate(self, **kwargs) -> dict:
        with torch.device(self.device):
            obj_pose = self.obj.pose
            new_unit_vector = quaternion_apply(obj_pose.q, self.unit_vector)
            new_unit_vector -= (
                torch.sum(new_unit_vector * self.rot_dir, dim=-1, keepdim=True) * self.rot_dir
            )
            new_unit_vector = new_unit_vector / torch.linalg.norm(new_unit_vector, dim=-1, keepdim=True)
            angle = torch.acos(
                torch.clip(torch.sum(new_unit_vector * self.prev_unit_vector, dim=-1), 0, 1)
            )
            angle = torch.clip(angle, -torch.pi / 20, torch.pi / 20)
            self.prev_unit_vector = new_unit_vector

            obj_vel = torch.linalg.norm(self.obj.get_linear_velocity(), dim=-1)
            obj_fall = (obj_pose.p[:, 2] < self.hand_init_height - 0.05).to(torch.bool)

            tip_poses = [vectorize_pose(link.pose) for link in self.agent.tip_links]
            tip_poses = torch.stack(tip_poses, dim=1)              # (b, 4, 7)
            obj_tip_vec = tip_poses[..., :3] - obj_pose.p[:, None, :]   # (b, 4, 3)
            obj_tip_dist = torch.linalg.norm(obj_tip_vec, dim=-1)       # (b, 4)

            self.cum_rotation_angle += angle
            success = self.cum_rotation_angle > self.success_threshold

            qpos_target = self.agent.controller._target_qpos
            qpos_error = qpos_target - self.agent.robot.qpos
            qvel = self.agent.robot.qvel
            qf = qpos_error * self.controller_param[0] - qvel * self.controller_param[1]
            qf = torch.clip(qf, -self.controller_param[2], self.controller_param[2])
            power = torch.sum(qf * qvel, dim=-1)

        return dict(
            rotation_angle=angle, obj_vel=obj_vel, obj_fall=obj_fall,
            obj_tip_vec=obj_tip_vec, obj_tip_dist=obj_tip_dist,
            success=success, qf=qf, power=power, fail=obj_fall,
        )
```

### Smoke (§4)
```bash
.venv/bin/python -c "import gymnasium as gym, mani_skill, numpy as np; e=gym.make('RotateSingleObjectInHandLevel1-v1'); e.reset(seed=2); _,_,term,trunc,info=e.step(e.action_space.sample()); print('success' in info, 'fail' in info); e.close()"
```
Expected: `True True` (info carries `success`/`fail`).

---

## §5 Observation (`_get_obs_extra` + proprioception)

### Description
Default `obs_mode="state"` → flat 105-D vector. It concatenates the agent proprioception (qpos, qvel, palm pose, tip poses, FSR impulses) with task-extra terms (one-hot rotation axis, object pose, finger→object vectors). The one-hot `rotate_dir` is always present even when `use_state` is false; the object pose and tip-vec terms are gated on `use_state`.

### Decisions resolved — obs term breakdown (total 105)
Proprioception (`AllegroHandRightTouch.get_proprioception`, on top of base qpos/qvel):
- `qpos` (16) + `qvel` (16) = 32
- `palm_pose` (7)
- `tip_poses` (4 fingers × 7) = 28
- `fsr_impulse` (16) — `‖fsr contact impulse‖` per FSR link (12 finger + 4 palm)
- proprio subtotal = **83**

Task extra (`_get_obs_extra`, all included under `use_state`):
- `rotate_dir` (3) — one-hot rotation axis
- `obj_pose` (7) — `vectorize_pose(self.obj.pose)`
- `obj_tip_vec` (4×3 = 12) — finger-tip→object vectors, flattened
- extra subtotal = **22**

**Total = 83 + 22 = 105** ✓ (matches `Box(..., (1, 105))`).

### Code
```python
    def _get_obs_extra(self, info: dict):
        with torch.device(self.device):
            obs = dict(rotate_dir=self.rot_dir)
            if self.obs_mode_struct.use_state:
                obs.update(
                    obj_pose=vectorize_pose(self.obj.pose),
                    obj_tip_vec=info["obj_tip_vec"].view(self.num_envs, 12),
                )
            return obs
```
```python
    # AllegroHandRightTouch.get_proprioception (adds fsr on top of AllegroHandRight)
    def get_proprioception(self):
        obs = super().get_proprioception()                       # qpos, qvel, palm_pose, tip_poses
        fsr_impulse = self.get_fsr_impulse()
        obs.update({"fsr_impulse": torch.linalg.norm(fsr_impulse, dim=-1)})
        return obs
    # AllegroHandRight.get_proprioception
    def get_proprioception(self):
        obs = super().get_proprioception()                       # qpos (16) + qvel (16)
        obs.update({
            "palm_pose": self.palm_pose,                         # (7)
            "tip_poses": self.tip_poses.reshape(-1, len(self.tip_links) * 7),  # (28)
        })
        return obs
```

### Obs modes
`SUPPORTED_OBS_MODES = ("state", "state_dict", "none", "sensor_data", "any_textures", "pointcloud")`. Default `state`. `state_dict` exposes the same terms unflattened; visual modes add the `base_camera` 128×128 RGB-D sensor.

### Smoke (§5)
```bash
.venv/bin/python -c "import gymnasium as gym, mani_skill, numpy as np; e=gym.make('RotateSingleObjectInHandLevel1-v1'); o,_=e.reset(seed=3); print(np.asarray(o).shape); e.close()"
```
Expected: `(1, 105)`.

---

## §6 Reward (fully reproducible)

### Description
A weighted **SUM** of six dense terms computed entirely from `evaluate()`'s info dict (no `RewardManager` / `RewTerm` — ManiSkill rewards are imperative). Drives rotation forward (term 1), penalizes object velocity / falling / control effort / torque (terms 2–5), and shapes finger-object contact distance (term 6). The normalized variant divides the dense reward by `4.0`.

### Composer
**SUM** of the six terms (each `reward += ...`).

### Per-term magnitudes (per control step, retro-computed from weights)
- term1 rotation: `20 * angle`, angle clipped to `≤π/20≈0.157` → max ≈ **+3.14**/step (dominant forward signal).
- term2 velocity: `-0.1 * obj_vel` — small penalty.
- term3 fall: `-50.0` one-shot when object drops below z=0.20 (dominant negative).
- term4 power: `-0.0003 * |power|`.
- term5 torque: `-0.0003 * ‖qf‖`.
- term6 contact: `mean_fingers clip(0.1/(0.02+4·dist), 0, 1)` ∈ **[0, +1]** (max +1 when all 4 tips touch the object).
- Normalization divisor: `4.0` (≈ max attainable per-step dense reward ≈ 3.14 + 1.0).

### Code (verbatim)
```python
    def compute_dense_reward(self, obs: Any, action: Array, info: dict):
        # 1. rotation reward
        angle = info["rotation_angle"]
        reward = 20 * angle

        # 2. velocity penalty
        obj_vel = info["obj_vel"]
        reward += -0.1 * obj_vel

        # 3. falling penalty
        obj_fall = info["obj_fall"]
        reward += -50.0 * obj_fall

        # 4. effort penalty
        power = torch.abs(info["power"])
        reward += -0.0003 * power

        # 5. torque penalty
        qf = info["qf"]
        qf_norm = torch.linalg.norm(qf, dim=-1)
        reward += -0.0003 * qf_norm

        # 6. finger object distance reward
        obj_tip_dist = info["obj_tip_dist"]
        distance_rew = 0.1 / (0.02 + 4 * obj_tip_dist)
        reward += torch.mean(torch.clip(distance_rew, 0, 1), dim=-1)

        return reward

    def compute_normalized_dense_reward(self, obs: Any, action: Array, info: dict):
        # this should be equal to compute_dense_reward / max possible reward
        return self.compute_dense_reward(obs=obs, action=action, info=info) / 4.0
```
All inputs (`rotation_angle, obj_vel, obj_fall, power, qf, obj_tip_dist`) are produced verbatim by `evaluate()` in §4 — fully self-contained.

### Smoke (§6)
```bash
.venv/bin/python -c "import gymnasium as gym, mani_skill, numpy as np; e=gym.make('RotateSingleObjectInHandLevel1-v1', reward_mode='dense'); e.reset(seed=4); _,r,_,_,_=e.step(e.action_space.sample()); r=np.asarray(r).reshape(-1)[0]; print(np.isfinite(r)); e.close()"
```
Expected: `True`.

---

## §7 DR

`<no DR>` — there are no `startup` / `interval` randomization hooks. The only per-episode variation is in `_initialize_episode` (reset-time): object position Gaussian noise (`obj_init_pos_noise=0.02`), `TableSceneBuilder`'s `robot_init_qpos_noise=0.02`, and — the defining Level1 feature — per-env box half-size randomization (`(randn()*0.1+1)*0.04`) sampled once at build via `_batched_episode_rng`. No friction/mass/gravity/visual randomization, no observation noise (`obs_mode="state"`, no noise terms).

---

## Self-verification
- §1 build smoke PASSED on this host: `OBS Box(-inf, inf, (1, 105), float32)`, `ACT Box(-1.0, 1.0, (16,), float32)`.
- Obs dim 105 reconstructed analytically (83 proprio + 22 extra) and matches the live build.
- All §4/§6 helpers (`quaternion_apply`, `vectorize_pose`, controller param access) resolve in-repo.
- No external assets required for Level1 (procedural box). YCB assets (levels 2/3 only) absent on host — WARN, irrelevant to Level1.

## Reproduce
`/harbor:task-create name=<NewTaskId> from=<ManiSkill-repo>/harbor/create-task/rotatesingleobjectinhandlevel1-v1-implementation.md`
