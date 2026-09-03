# PlugCharger-v1 — Implementation Spec

- robot: Franka Panda with wrist camera (`panda_wristcam`)
- simulator: ManiSkill (SAPIEN, `BaseEnv` + `@register_env`)
- objects: charger, receptacle, table
- bimanual: false
- summary: Grasp a charger and plug it into a wall receptacle.

A precision two-prong insertion task: the robot must grasp a wall-charger (a base block with two thin metal prongs) and plug it into a matching wall receptacle (a kinematic socket with two prong holes). Success requires the charger to reach the goal pose (receptacle pose rotated 180° about z) within a very tight `5e-3 m` position tolerance AND `0.2 rad` orientation tolerance. Both charger and receptacle are procedurally built from primitive boxes (no external assets); prong clearance is `5e-4 m` single-sided. The receptacle is kinematic. **This env is sparse-only** — it declares `SUPPORTED_REWARD_MODES = ["none", "sparse"]` and provides no dense reward; staging (grasp→align→insert) must be supplied externally if dense shaping is desired.

---

## §1 Registration + Scene

### Description
Registered with `@register_env("PlugCharger-v1", max_episode_steps=200)`. The agent is a `PandaWristCam` (the only `SUPPORTED_ROBOTS` entry), base-mounted at `sapien.Pose(p=[-0.615, 0, 0])`. The scene is the standard `TableSceneBuilder` plus two procedurally-built actors:
- **charger** (`_build_charger`, dynamic): a base box (half size `[2e-2, 1.5e-2, 1.2e-2]`, white matte) with two thin metal prongs (peg half size `[8e-3, 0.75e-3, 3.2e-3]`, separated by `_peg_gap = 7e-3` in y, white metallic). The prongs extend in +x from the base. `initial_pose = Pose(p=[0,0,_base_size[2]])` (rests on table at z = base half-height).
- **receptacle** (`_build_receptacle`, **kinematic**): a socket box (half size `[1e-2, 5e-2, 5e-2]`) assembled from 4 wall boxes plus a gap-filler box, sized so the two prong holes match the charger prongs **plus `_clearance = 5e-4 m` single-side clearance** (the receptacle is built with peg dimensions `[_peg_size[0], _peg_size[1]+_clearance, _peg_size[2]+_clearance]`). Two `#DBB539` (gold) dummy visual boxes mark the hole faces. `initial_pose = Pose(p=[0,0,0.1])`.

The human render camera is **mounted on the receptacle** (`mount=self.receptacle`) so it tracks the socket. `reconfiguration_freq` is left at the `BaseEnv` default (no override in this env).

### Decisions resolved
- `SUPPORTED_ROBOTS = ["panda_wristcam"]`, default `robot_uids="panda_wristcam"`; `robot_init_qpos_noise=0.02`.
- `SUPPORTED_REWARD_MODES = ["none", "sparse"]` (NO dense/normalized_dense).
- Robot base pose: `sapien.Pose(p=[-0.615, 0, 0])` (set in `_load_agent` and re-pinned in `_initialize_episode`).
- Charger geometry: `_base_size=[2e-2,1.5e-2,1.2e-2]`, `_peg_size=[8e-3,0.75e-3,3.2e-3]`, `_peg_gap=7e-3`. Prongs at y=±gap, x-offset +peg_size[0]; base at x=−base_size[0]. Built dynamic via `builder.build(name="charger")`.
- Receptacle geometry: `_receptacle_size=[1e-2,5e-2,5e-2]`, `_clearance=5e-4`. Built kinematic via `builder.build_kinematic(name="receptacle")`. Holes sized = peg + clearance.
- Sim: `_default_sim_config = SimConfig()` → defaults `sim_freq=100`, `control_freq=20` (5 physx substeps per control step).
- Sensors: `base_camera` 128×128 fov π/2 at `look_at(eye=[0.3,0,0.6], target=[-0.1,0,0.1])`; human render cam `render_camera` 512×512 fov 1 at `look_at([0.3,0.4,0.1],[0,0,0])`, **mounted on `self.receptacle`**.

### Code
```python
@register_env("PlugCharger-v1", max_episode_steps=200)
class PlugChargerEnv(BaseEnv):
    _base_size = [2e-2, 1.5e-2, 1.2e-2]  # charger base half size
    _peg_size = [8e-3, 0.75e-3, 3.2e-3]  # charger peg half size
    _peg_gap = 7e-3  # charger peg gap
    _clearance = 5e-4  # single side clearance
    _receptacle_size = [1e-2, 5e-2, 5e-2]  # receptacle half size

    SUPPORTED_ROBOTS = ["panda_wristcam"]
    agent: Union[PandaWristCam]
    SUPPORTED_REWARD_MODES = ["none", "sparse"]

    def __init__(
        self, *args, robot_uids="panda_wristcam", robot_init_qpos_noise=0.02, **kwargs
    ):
        self.robot_init_qpos_noise = robot_init_qpos_noise
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    @property
    def _default_sim_config(self):
        return SimConfig()

    @property
    def _default_sensor_configs(self):
        pose = sapien_utils.look_at(eye=[0.3, 0, 0.6], target=[-0.1, 0, 0.1])
        return [
            CameraConfig("base_camera", pose=pose, width=128, height=128, fov=np.pi / 2)
        ]

    @property
    def _default_human_render_camera_configs(self):
        pose = sapien_utils.look_at([0.3, 0.4, 0.1], [0, 0, 0])
        return [
            CameraConfig(
                "render_camera",
                pose=pose,
                width=512,
                height=512,
                fov=1,
                mount=self.receptacle,
            )
        ]

    def _build_charger(self, peg_size, base_size, gap):
        builder = self.scene.create_actor_builder()

        # peg
        mat = sapien.render.RenderMaterial()
        mat.set_base_color([1, 1, 1, 1])
        mat.metallic = 1.0
        mat.roughness = 0.0
        mat.specular = 1.0
        builder.add_box_collision(sapien.Pose([peg_size[0], gap, 0]), peg_size)
        builder.add_box_visual(
            sapien.Pose([peg_size[0], gap, 0]), peg_size, material=mat
        )
        builder.add_box_collision(sapien.Pose([peg_size[0], -gap, 0]), peg_size)
        builder.add_box_visual(
            sapien.Pose([peg_size[0], -gap, 0]), peg_size, material=mat
        )

        # base
        mat = sapien.render.RenderMaterial()
        mat.set_base_color([1, 1, 1, 1])
        mat.metallic = 0.0
        mat.roughness = 0.1
        builder.add_box_collision(sapien.Pose([-base_size[0], 0, 0]), base_size)
        builder.add_box_visual(
            sapien.Pose([-base_size[0], 0, 0]), base_size, material=mat
        )
        builder.initial_pose = sapien.Pose(p=[0, 0, self._base_size[2]])
        return builder.build(name="charger")

    def _build_receptacle(self, peg_size, receptacle_size, gap):
        builder = self.scene.create_actor_builder()

        sy = 0.5 * (receptacle_size[1] - peg_size[1] - gap)
        sz = 0.5 * (receptacle_size[2] - peg_size[2])
        dx = -receptacle_size[0]
        dy = peg_size[1] + gap + sy
        dz = peg_size[2] + sz

        mat = sapien.render.RenderMaterial()
        mat.set_base_color([1, 1, 1, 1])
        mat.metallic = 0.0
        mat.roughness = 0.1

        poses = [
            sapien.Pose([dx, 0, dz]),
            sapien.Pose([dx, 0, -dz]),
            sapien.Pose([dx, dy, 0]),
            sapien.Pose([dx, -dy, 0]),
        ]
        half_sizes = [
            [receptacle_size[0], receptacle_size[1], sz],
            [receptacle_size[0], receptacle_size[1], sz],
            [receptacle_size[0], sy, receptacle_size[2]],
            [receptacle_size[0], sy, receptacle_size[2]],
        ]
        for pose, half_size in zip(poses, half_sizes):
            builder.add_box_collision(pose, half_size)
            builder.add_box_visual(pose, half_size, material=mat)

        # Fill the gap
        pose = sapien.Pose([-receptacle_size[0], 0, 0])
        half_size = [receptacle_size[0], gap - peg_size[1], peg_size[2]]
        builder.add_box_collision(pose, half_size)
        builder.add_box_visual(pose, half_size, material=mat)

        # Add dummy visual for hole
        mat = sapien.render.RenderMaterial()
        mat.set_base_color(sapien_utils.hex2rgba("#DBB539"))
        mat.metallic = 1.0
        mat.roughness = 0.0
        mat.specular = 1.0
        pose = sapien.Pose([-receptacle_size[0], -(gap * 0.5 + peg_size[1]), 0])
        half_size = [receptacle_size[0], peg_size[1], peg_size[2]]
        builder.add_box_visual(pose, half_size, material=mat)
        pose = sapien.Pose([-receptacle_size[0], gap * 0.5 + peg_size[1], 0])
        builder.add_box_visual(pose, half_size, material=mat)
        builder.initial_pose = sapien.Pose(p=[0, 0, 0.1])
        return builder.build_kinematic(name="receptacle")

    def _load_agent(self, options: dict):
        super()._load_agent(options, sapien.Pose(p=[-0.615, 0, 0]))

    def _load_scene(self, options: dict):
        self.scene_builder = TableSceneBuilder(
            self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.scene_builder.build()
        self.charger = self._build_charger(
            self._peg_size,
            self._base_size,
            self._peg_gap,
        )
        self.receptacle = self._build_receptacle(
            [
                self._peg_size[0],
                self._peg_size[1] + self._clearance,
                self._peg_size[2] + self._clearance,
            ],
            self._receptacle_size,
            self._peg_gap,
        )
```

### Smoke
```bash
cd <ManiSkill-repo>
.venv/bin/python -c "import gymnasium as gym, mani_skill; e=gym.make('PlugCharger-v1'); print(e.observation_space, e.action_space)"
# Box(-inf, inf, (1, 46), float32) Box(-1.0, 1.0, (8,), float32)
```

---

## §2 Actions

### Description
No task-local `ActionsCfg`; the action space comes entirely from the robot's controller. `control_mode` is unspecified, so `BaseEnv` uses the agent's `_default_control_mode = supported_control_modes[0]` → **`pd_joint_delta_pos`** (first key in Panda's `_controller_configs`). Arm: 7-dim normalized PD joint delta-position (lower=-0.1, upper=0.1). Gripper: 1-dim mimic PD joint position. Total action = 8, normalized to `[-1, 1]`.

### Decisions resolved
- control_mode = `pd_joint_delta_pos` (default = `supported_control_modes[0]`).
- Arm controller `PDJointPosControllerConfig` (delta variant): `lower=-0.1, upper=0.1, stiffness=1e3, damping=1e2, force_limit=100, use_delta=True, normalize_action=True`.
- Gripper `PDJointPosMimicControllerConfig` (`gripper_pd_joint_pos`): single mimic action over `panda_finger_joint1/2`, `stiffness=1e3, damping=1e2, force_limit=100`.
- Resolved action_space = `Box(-1.0, 1.0, (8,), float32)` (7 arm + 1 gripper).
- Other available control modes: `pd_joint_pos`, `pd_ee_delta_pos`, `pd_ee_delta_pose` (pos/rot bounds ±0.1), `pd_ee_pose`, `pd_joint_target_delta_pos`, `pd_ee_target_delta_pos/pose`, `pd_joint_vel`, `pd_joint_pos_vel`, `pd_joint_delta_pos_vel`.

### Smoke
Covered by §1 build smoke (`action_space == Box(-1.0, 1.0, (8,), float32)`).

---

## §3 Reset (`_initialize_episode`)

### Description
Per reset: re-initialize the table scene; set the robot to the canonical rest qpos + N(0, robot_init_qpos_noise=0.02) noise (gripper fingers forced open to 0.04) and pin the base at `[-0.615, 0, 0]`; place the **charger** on the table at z = base half-height within a randomized xy box (x in the **left** half of the table, allowing room for the prongs) with a z-axis rotation in ±π/3; place the **receptacle** in a separate xy box (x in the **right** half) at z = 0.1 with a z-rotation around π (≈180°, ±π/8) so the socket faces the charger. Finally cache `goal_pose = receptacle.pose * Pose(q=euler2quat(0,0,π))` — the receptacle pose rotated 180° about z, which is the target charger pose for a fully-plugged charger.

### Decisions resolved
- Charger xy ∈ `[-0.1, -0.2]`–`[-0.01 - _peg_size[0]*2, 0.2]`; charger z = `_base_size[2]` (= 1.2e-2). quat random z-rot, `bounds=(-π/3, π/3)`, lock_x/lock_y.
- Receptacle xy ∈ `[0.01, -0.1]`–`[0.1, 0.1]`; receptacle z = 0.1. quat random z-rot, `bounds=(π - π/8, π + π/8)`, lock_x/lock_y.
- Robot qpos base = `[0, π/8, 0, -5π/8, 0, 3π/4, π/4, 0.04, 0.04]` + `torch.normal(0, 0.02)`; fingers reset to 0.04. Base pose `Pose([-0.615, 0, 0])`.
- `goal_pose = receptacle.pose * sapien.Pose(q=euler2quat(0, 0, np.pi))`.

### Code
```python
def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
    with torch.device(self.device):
        b = len(env_idx)
        self.scene_builder.initialize(env_idx)

        # Initialize agent
        if self.agent.uid == "panda_wristcam":
            qpos = torch.tensor(
                [
                    0.0,
                    np.pi / 8,
                    0,
                    -np.pi * 5 / 8,
                    0,
                    np.pi * 3 / 4,
                    np.pi / 4,
                    0.04,
                    0.04,
                ]
            )
            qpos = (
                torch.normal(
                    0,
                    self.robot_init_qpos_noise,
                    (b, len(qpos)),
                    device=self.device,
                )
                + qpos
            )
            qpos[:, -2:] = 0.04
            self.agent.robot.set_qpos(qpos)
            self.agent.robot.set_pose(sapien.Pose([-0.615, 0, 0]))

        # Initialize charger
        xy = randomization.uniform(
            [-0.1, -0.2], [-0.01 - self._peg_size[0] * 2, 0.2], size=(b, 2)
        )
        pos = torch.zeros((b, 3))
        pos[:, :2] = xy
        pos[:, 2] = self._base_size[2]
        ori = randomization.random_quaternions(
            n=b, lock_x=True, lock_y=True, bounds=(-torch.pi / 3, torch.pi / 3)
        )
        self.charger.set_pose(Pose.create_from_pq(pos, ori))

        # Initialize receptacle
        xy = randomization.uniform([0.01, -0.1], [0.1, 0.1], size=(b, 2))
        pos = torch.zeros((b, 3))
        pos[:, :2] = xy
        pos[:, 2] = 0.1
        ori = randomization.random_quaternions(
            n=b,
            lock_x=True,
            lock_y=True,
            bounds=(torch.pi - torch.pi / 8, torch.pi + torch.pi / 8),
        )
        self.receptacle.set_pose(Pose.create_from_pq(pos, ori))

        self.goal_pose = self.receptacle.pose * (
            sapien.Pose(q=euler2quat(0, 0, np.pi))
        )
```

---

## §4 Goal + Termination (`evaluate`)

### Description
Success = the charger has reached the goal pose (receptacle rotated 180° about z) within a tight tolerance on BOTH position and orientation: `obj_to_goal_dist <= 5e-3 m` AND `obj_to_goal_angle <= 0.2 rad`. Distance is the L2 norm of the charger-to-goal position; the angle is the geodesic angle of the relative quaternion (`quaternion_multiply(invert(goal.q), obj.q)` → axis-angle → norm, folded to ≤ π). No failure/`fail` term, so the sparse reward never goes negative. Episodes end at `max_episode_steps=200` (truncation).

### Decisions resolved
- Position tolerance: `obj_to_goal_dist <= 5e-3`.
- Orientation tolerance: `obj_to_goal_angle <= 0.2` rad.
- `max_episode_steps = 200`.
- No `fail` term emitted by `evaluate()`.

### Code
```python
@property
def charger_base_pose(self):
    return self.charger.pose * (sapien.Pose([-self._base_size[0], 0, 0]))

def _compute_distance(self):
    obj_pose = self.charger.pose
    obj_to_goal_pos = self.goal_pose.p - obj_pose.p
    obj_to_goal_dist = torch.linalg.norm(obj_to_goal_pos, axis=1)

    obj_to_goal_quat = rotation_conversions.quaternion_multiply(
        rotation_conversions.quaternion_invert(self.goal_pose.q), obj_pose.q
    )
    obj_to_goal_axis = rotation_conversions.quaternion_to_axis_angle(
        obj_to_goal_quat
    )
    obj_to_goal_angle = torch.linalg.norm(obj_to_goal_axis, axis=1)
    obj_to_goal_angle = torch.min(
        obj_to_goal_angle, torch.pi * 2 - obj_to_goal_angle
    )

    return obj_to_goal_dist, obj_to_goal_angle

def evaluate(self):
    obj_to_goal_dist, obj_to_goal_angle = self._compute_distance()
    success = (obj_to_goal_dist <= 5e-3) & (obj_to_goal_angle <= 0.2)
    return dict(
        obj_to_goal_dist=obj_to_goal_dist,
        obj_to_goal_angle=obj_to_goal_angle,
        success=success,
    )
```

---

## §5 Observation (`_get_obs_extra`)

### Description
Default `obs_mode="state"`. Total obs = 46 (flattened `Box(-inf, inf, (1,46))`). Composed of base proprioception (`_get_obs_agent`: qpos 9 + qvel 9 = 18) plus `_get_obs_extra`. `tcp_pose` (7) is always present; in `state`/`state_dict` mode the extra dict additionally exposes privileged `charger_pose`, `receptacle_pose`, and `goal_pose` (7 each).

### Decisions resolved
- obs_mode default = `state` (`SUPPORTED_OBS_MODES[0]`); env inherits `SUPPORTED_OBS_MODES = ("state","state_dict","none","sensor_data","any_textures","pointcloud")` from `BaseEnv`.
- Resolved 46-dim breakdown (state mode):
  - agent qpos = 9, agent qvel = 9 (Panda 7 arm + 2 finger joints) → 18
  - `tcp_pose` = 7 (xyz + quat)
  - `charger_pose` = 7
  - `receptacle_pose` = 7
  - `goal_pose` = 7
  - 18 + 7 + 7 + 7 + 7 = **46** ✓

### Code
```python
def _get_obs_extra(self, info: dict):
    obs = dict(tcp_pose=self.agent.tcp.pose.raw_pose)
    if self.obs_mode_struct.use_state:
        obs.update(
            charger_pose=self.charger.pose.raw_pose,
            receptacle_pose=self.receptacle.pose.raw_pose,
            goal_pose=self.goal_pose.raw_pose,
        )
    return obs
```

### Smoke
Covered by §1 build smoke (`observation_space == Box(-inf, inf, (1, 46), float32)`).

---

## §6 Reward

### Description
**Sparse-only.** `SUPPORTED_REWARD_MODES = ["none", "sparse"]` and the env provides **NO** `compute_dense_reward` / `compute_normalized_dense_reward` override — calling either inherits `BaseEnv.compute_dense_reward` which raises `NotImplementedError`. There is therefore no grasp→align→insert staging baked into the env; that shaping must be authored externally if dense reward is desired (see reproduction note below).

- **Default reward (`reward_mode="none"`, since `SUPPORTED_REWARD_MODES[0] == "none"`):** constant `0` per step.
- **Sparse reward (`reward_mode="sparse"`):** `BaseEnv.compute_sparse_reward` returns `info["success"].float()`. Since `evaluate()` emits no `fail` key, the reward is `+1` on the step where the charger is plugged within tolerance, `0` otherwise. No negative reward.

### Decisions resolved
- Composer: **n/a** — no additive/multiplicative staged composition; the reward is the raw success indicator.
- Success indicator (from §4): `(obj_to_goal_dist <= 5e-3) & (obj_to_goal_angle <= 0.2)`.
- `reward_mode` default = `"none"` (must explicitly pass `reward_mode="sparse"` to obtain the +1 success signal).

### Code (verbatim)
PlugChargerEnv defines **no** reward method. The effective reward is `BaseEnv`'s sparse path:
```python
# mani_skill/envs/sapien_env.py — BaseEnv (inherited, NOT overridden by PlugChargerEnv)
def get_reward(self, obs, action, info):
    if self._reward_mode == "sparse":
        reward = self.compute_sparse_reward(obs=obs, action=action, info=info)
    elif self._reward_mode == "dense":
        reward = self.compute_dense_reward(obs=obs, action=action, info=info)        # -> NotImplementedError for PlugCharger
    elif self._reward_mode == "normalized_dense":
        reward = self.compute_normalized_dense_reward(obs=obs, action=action, info=info)  # -> NotImplementedError
    elif self._reward_mode == "none":
        reward = torch.zeros((self.num_envs,), dtype=torch.float, device=self.device)
    else:
        raise NotImplementedError(self._reward_mode)
    return reward

def compute_sparse_reward(self, obs, action, info):
    if "success" in info:
        if "fail" in info:
            reward = info["success"].to(torch.float) - info["fail"].to(torch.float)
        else:
            reward = info["success"]          # PlugCharger path: no "fail" -> +1 on success, else 0
    else:
        if "fail" in info:
            reward = -info["fail"]
        else:
            reward = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
    return reward

def compute_dense_reward(self, obs, action, info):
    raise NotImplementedError()
```

### Reproduction note
To reproduce identically, do NOT add a dense reward — keep `SUPPORTED_REWARD_MODES = ["none", "sparse"]` and omit both `compute_dense_reward` and `compute_normalized_dense_reward`. If the downstream consumer wants a dense signal (e.g. for RL training), author a separate staged reward (grasp the charger base via an x-offset grasp pose → align the prongs to the receptacle holes in the goal-frame yz plane → drive `obj_to_goal_dist`/`obj_to_goal_angle` to zero) and add `"dense"`/`"normalized_dense"` to `SUPPORTED_REWARD_MODES` — this is an ADDITION beyond the canonical PlugCharger-v1, not part of its spec.

### Smoke
Reward verified by source read. Runtime check at reproduction time: build with `reward_mode="sparse"`, step, assert reward is finite and ∈ {0, 1}; build with `reward_mode="dense"` and assert it raises `NotImplementedError` (proves no dense override leaked in).

---

## §7 DR

`<no DR>` — there is no `interval`/`startup` event-based domain randomization. All variation is per-reset pose/qpos randomization in `_initialize_episode` (charger xy + z-rot, receptacle xy + z-rot, robot qpos noise) — that belongs to §3, not §7. The class docstring's "Randomizations" section describes exactly these reset-time pose randomizations. No physical-parameter (mass/friction/PD-gain) randomization and no per-reconfiguration geometry sampling (charger/receptacle dimensions are fixed class constants).

---

