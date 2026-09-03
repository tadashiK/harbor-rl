# RotateValveLevel1-v1 — Implementation Spec

- robot: D'Claw three-finger hand (9 DoF, fixed base)
- simulator: ManiSkill (SAPIEN, `BaseEnv` + `@register_env`)
- objects: three-spoke valve, table
- bimanual: false
- summary: Rotate a three-spoke valve with a three-finger hand.

> NOTE: All five `RotateValveLevel{0..4}-v1` ids are registered on subclasses of one base `RotateValveEnv`. Behaviour differs only through `self.difficulty_level`, which gates (a) `success_threshold`, (b) the valve-head angle sampling in `_load_articulations`, (c) valve radius randomization (level ≥ 3), and (d) the rotation direction (level 4 randomizes sign). This spec captures the full base design and annotates exactly what Level1 selects.

---

## §1 Registration + Scene

**Description.** Five gym ids registered via `@register_env` on thin subclasses of `RotateValveEnv` (a `BaseEnv`). The scene is the standard `TableSceneBuilder` table + ground + lights, plus the D'Claw dexterous hand (`robot_uids="dclaw"`) mounted above the table, plus one procedurally-built ROBEL valve articulation per env. No USD/mesh asset files — both the hand (URDF) and the valve (programmatic articulation builder) are built from primitives/URDF shipped in the package.

**Decisions resolved.**
- `robot_uids = "dclaw"` (set in `RotateValveEnv.__init__` via `super().__init__(..., robot_uids="dclaw", ...)`).
- `agent: Union[DClaw]`. URDF: `{PACKAGE_ASSET_DIR}/robots/dclaw/dclaw_gripper_glb.urdf`. DOF = 9 (three 3-joint fingers).
- Robot root pose (set every reset in `_initialize_agent`): `p=[0,0,0.28]`, `q=[0,0,-1,0]` (hand points down at the valve).
- D'Claw actuator gains: stiffness `1e2`, damping `1e1`, force_limit `2e1`; fingertip links `link_f{1,2,3}_3` get a high-friction "tip" material (static 2.0 / dynamic 1.0).
- Valve built procedurally by `build_robel_valve` (a SAPIEN `ArticulationBuilder`): a fixed `mount` link (box base + bearing cylinder) + a `valve` revolute link carrying N capsule "heads". Geometry (Robel-derived, meters): `capsule_height=0.039854`, `capsule_length=0.061706*radius_scale`, `capsule_radius=0.0195*capsule_radius_scale`, `bottom_length=0.04`, `bottom_height=0.03`, `bearing_radius=0.007`, `bearing_height=0.032`. Valve joint: revolute, limits `[-inf, inf]`, `friction=0.02`, `damping=2`, built with `fix_root_link=True`. First head colored red, the rest white. `build_robel_valve` returns `(valve, capsule_length)`; the per-env `capsule_length` is stored in `self.capsule_lens` (used by the reward).
- **Level1 valve head angles**: `_load_articulations` cycles three fixed layouts across envs — 3 heads at 120° spacing, 4 heads at 90° spacing, 5 heads at 72° spacing (`np.arange(0, 2π, 2π/3)`, `np.arange(0, 2π, π/2)`, `np.arange(0, 2π, 2π/5)`), tiled `base_angles * (num_envs//3) + base_angles[:num_envs%3]`. (Level0 → fixed tri-valve; Level≥2 → random head count 3–5 via `sample_valve_angles`; Level≥3 → additionally random radius scales.) Per-env valves are `Articulation.merge`-d into `self.valve`; valve links into `self.valve_link`.
- Sim cfg: env does NOT override `_default_sim_config` → BaseEnv `SimConfig` defaults: `sim_freq=100`, `control_freq=20` (5 physx substeps / control step), `spacing=5`. `_clearance=0.003`, `capsule_offset=0.01`.
- Cameras: sensor `base_camera` 128×128 FOV π/2 at eye `[0.3,0,0.3]`→`[-0.1,0,0.05]`; human-render `render_camera` 512×512 at `[0.2,0.4,0.4]`→`[0,0,0.1]`.

**Code.**

```python
# mani_skill/envs/tasks/dexterity/rotate_valve.py
from mani_skill.agents.robots import DClaw
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.utils.building.articulations import build_robel_valve
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.structs.articulation import Articulation
from mani_skill.utils.structs.link import Link
from mani_skill.utils.structs.pose import Pose, vectorize_pose

class RotateValveEnv(BaseEnv):
    agent: Union[DClaw]
    _clearance = 0.003

    def __init__(self, *args, robot_init_qpos_noise=0.02, valve_init_pos_noise=0.02,
                 difficulty_level: int = -1, **kwargs):
        self.robot_init_qpos_noise = robot_init_qpos_noise
        self.valve_init_pos_noise = valve_init_pos_noise
        if (not isinstance(difficulty_level, int) or difficulty_level >= 5 or difficulty_level < 0):
            raise ValueError(f"Difficulty level must be a int within 0-4, but get {difficulty_level}")
        self.difficulty_level = difficulty_level
        if self.difficulty_level == 0:
            self.success_threshold = torch.pi / 2
        elif self.difficulty_level == 4:
            self.success_threshold = torch.pi * 2
        else:
            self.success_threshold = torch.pi * 1
        self.capsule_offset = 0.01
        super().__init__(*args, robot_uids="dclaw", **kwargs)

    @property
    def _default_sensor_configs(self):
        pose = sapien_utils.look_at(eye=[0.3, 0, 0.3], target=[-0.1, 0, 0.05])
        return [CameraConfig("base_camera", pose, 128, 128, np.pi / 2, 0.01, 100)]

    @property
    def _default_human_render_camera_configs(self):
        pose = sapien_utils.look_at([0.2, 0.4, 0.4], [0.0, 0.0, 0.1])
        return CameraConfig("render_camera", pose, 512, 512, 1, 0.01, 100)

    def _load_scene(self, options: dict):
        self.table_scene = TableSceneBuilder(env=self, robot_init_qpos_noise=self.robot_init_qpos_noise)
        self.table_scene.build()
        self._load_articulations()

    def _load_articulations(self):
        if self.difficulty_level == 0:
            valve_angles_list = [(0, np.pi / 3 * 2, np.pi / 3 * 4)] * self.num_envs
        elif self.difficulty_level == 1:
            base_angles = [
                np.arange(0, np.pi * 2, np.pi * 2 / 3),
                np.arange(0, np.pi * 2, np.pi / 2),
                np.arange(0, np.pi * 2, np.pi * 2 / 5),
            ]
            valve_angles_list = (base_angles * int(self.num_envs // 3)
                                 + base_angles[: int(self.num_envs % 3)])
        elif self.difficulty_level == 2:
            num_valve_head = self._batched_episode_rng.randint(3, 6)
            valve_angles_list = [sample_valve_angles(n, self._batched_episode_rng[i])
                                 for i, n in enumerate(num_valve_head)]
        elif self.difficulty_level >= 3:
            num_valve_head = self._batched_episode_rng.randint(3, 6)
            valve_angles_list = [sample_valve_angles(n, self._batched_episode_rng[i])
                                 for i, n in enumerate(num_valve_head)]
        else:
            raise ValueError(...)

        valves, capsule_lens, valve_links = [], [], []
        for i, valve_angles in enumerate(valve_angles_list):
            scene_idxs = [i]
            if self.difficulty_level < 3:
                valve, capsule_len = build_robel_valve(self.scene, valve_angles=valve_angles,
                                                       scene_idxs=scene_idxs, name=f"valve_station_{i}")
            else:
                scales = self._batched_episode_rng[i].randn(2) * 0.1 + 1
                valve, capsule_len = build_robel_valve(self.scene, valve_angles=valve_angles,
                                                       scene_idxs=scene_idxs, name=f"valve_station_{i}",
                                                       radius_scale=scales[0], capsule_radius_scale=scales[1])
            valves.append(valve); valve_links.append(valve.links_map["valve"]); capsule_lens.append(capsule_len)
        self.valve = Articulation.merge(valves, "valve_station")
        self.capsule_lens = torch.from_numpy(np.array(capsule_lens)).to(self.device)
        self.valve_link = Link.merge(valve_links, name="valve")
```

```python
# mani_skill/utils/building/articulations/robel.py — valve builder (verbatim)
def build_robel_valve(scene, valve_angles, name, radius_scale=1.0, capsule_radius_scale=1.0, scene_idxs=None):
    capsule_height = 0.039854
    capsule_length = 0.061706 * radius_scale
    capsule_radius = 0.0195 * capsule_radius_scale
    bottom_length = 0.04; bottom_height = 0.03; bearing_radius = 0.007; bearing_height = 0.032
    builder = scene.create_articulation_builder(); builder.set_scene_idxs(scene_idxs)
    mount_builder = builder.create_link_builder(parent=None); mount_builder.set_name("mount")
    mount_builder.add_box_collision(pose=sapien.Pose([0,0,bottom_height/2]),
        half_size=[bottom_length/2, bottom_length/2, bottom_height/2])
    mount_builder.add_box_visual(pose=sapien.Pose([0,0,bottom_height/2]),
        half_size=[bottom_length/2, bottom_length/2, bottom_height/2])
    mount_builder.add_cylinder_visual(pose=sapien.Pose([0,0,bottom_height+bearing_height/2],[-0.707,0,0.707,0]),
        half_length=bottom_height/2, radius=bearing_radius)
    mount_builder.add_cylinder_collision(pose=sapien.Pose([0,0,bottom_height+bearing_height/2],[-0.707,0,0.707,0]),
        half_length=bottom_height/2, radius=bearing_radius)
    valve_builder = builder.create_link_builder(mount_builder); valve_builder.set_name("valve")
    valve_angles = np.array(valve_angles)
    if np.min(valve_angles) < 0 or np.max(valve_angles) > 2*np.pi:
        raise ValueError(...)
    for i, angle in enumerate(valve_angles):
        rotate_pose = sapien.Pose([0,0,0]); rotate_pose.set_rpy([0,0,angle])
        capsule_pose = rotate_pose * sapien.Pose([capsule_length/2, 0, 0])
        color = np.array([1,1,1,1]) if i > 0 else np.array([1,0,0,1])
        viz_mat = sapien.render.RenderMaterial(base_color=color, roughness=0.5, specular=0.5)
        valve_builder.add_capsule_visual(pose=capsule_pose, radius=capsule_radius,
            half_length=capsule_length/2, material=viz_mat)
        physx_mat = physx.PhysxMaterial(1, 0.8, 0)
        valve_builder.add_capsule_collision(pose=capsule_pose, radius=capsule_radius,
            half_length=capsule_length/2, material=physx_mat, patch_radius=0.1, min_patch_radius=0.03)
    valve_builder.set_joint_name("valve_joint")
    valve_builder.set_joint_properties(type="revolute", limits=[[-np.inf, np.inf]],
        pose_in_parent=sapien.Pose([0,0,capsule_height+bottom_height],[0.707,0,0.707,0]),
        pose_in_child=sapien.Pose(q=[0.707,0,0.707,0]), friction=0.02, damping=2)
    valve = builder.build(name, fix_root_link=True)
    return valve, capsule_length
```

**Smoke (§1 build).**
```bash
cd ManiSkill
.venv/bin/python -c "import gymnasium as gym, mani_skill; e=gym.make('RotateValveLevel1-v1'); print(e.observation_space, e.action_space); e.close()"
```
Expected stdout (literal):
```
OBS Box(-inf, inf, (1, 51), float32)
ACT Box(-1.0, 1.0, (9,), float32)
CTRL pd_joint_delta_pos
```
(Build emits two benign WARN lines about "No initial pose set for articulation builder" for `dclaw` and `valve_station_0` — expected.)

---

## §2 Actions

**Description.** D'Claw is controlled in joint space. Three controller modes are registered; the default (= `supported_control_modes[0]` = first dict key) is `pd_joint_delta_pos`: a 9-DOF PD position controller where the action is a per-joint delta added each control step, clamped to ±0.1 rad, with the action space normalized to `[-1, 1]`.

**Decisions resolved.**
- DOF / action dim = 9 (`joint_f{1,2,3}_{0,1,2}`).
- Default `control_mode = "pd_joint_delta_pos"` (verified at runtime: `CTRL pd_joint_delta_pos`).
- `pd_joint_delta_pos`: `PDJointPosControllerConfig(joint_names, lower=-0.1, upper=0.1, stiffness=1e2, damping=1e1, force_limit=2e1, use_delta=True)` → action space `Box(-1, 1, (9,))` (normalized).
- Alternative modes: `pd_joint_pos` (absolute, `normalize_action=False`, unbounded action limits), `pd_joint_target_delta_pos` (delta with `use_target=True`).
- `root_joint_names = ["joint_f1_0","joint_f2_0","joint_f3_0"]` (the base joint of each finger).

**Code.**
```python
# mani_skill/agents/robots/dclaw/dclaw.py
@property
def _controller_configs(self):
    joint_pos = PDJointPosControllerConfig(
        self.joint_names, None, None, self.joint_stiffness, self.joint_damping,
        self.joint_force_limit, normalize_action=False)
    joint_delta_pos = PDJointPosControllerConfig(
        self.joint_names, -0.1, 0.1, self.joint_stiffness, self.joint_damping,
        self.joint_force_limit, use_delta=True)
    joint_target_delta_pos = deepcopy(joint_delta_pos)
    joint_target_delta_pos.use_target = True
    controller_configs = dict(
        pd_joint_delta_pos=dict(joint=joint_delta_pos),
        pd_joint_pos=dict(joint=joint_pos),
        pd_joint_target_delta_pos=dict(joint=joint_target_delta_pos),
    )
    return deepcopy_dict(controller_configs)
# joint_stiffness=1e2, joint_damping=1e1, joint_force_limit=2e1
```

**Smoke.** `e.action_space == Box(-1.0, 1.0, (9,), float32)` (from §1 stdout).

---

## §3 Reset (`_initialize_episode`)

**Description.** Each reset: rebuild the table, randomize the valve's planar position + yaw and give it a random initial joint angle (recorded as `rest_qpos`), set the rotation direction (always +1 for Level1), and reset the hand to a spread keyframe (root joints biased open) with small per-joint noise, placed above the valve facing down.

**Decisions resolved.**
- `robot_init_qpos_noise = 0.02`, `valve_init_pos_noise = 0.02` (Level1 ctor; note: `valve_init_pos_noise` is stored but the actual valve XY jitter below is hard-coded to ±0.02).
- Valve XY uniform `[-0.02, 0.02]`; valve yaw (`axis_angle[:,2]`) uniform `[π/6, 5π/6]` → quaternion.
- Valve initial joint qpos uniform `[-π, π]`, stored as `self.rest_qpos`.
- `rotate_direction`: Level≤3 → `ones(b)` (always positive/CCW). Level4 → random ±1.
- Hand init qpos = zeros, root-joint indices set to `[0.7, -0.7, -0.7]` (open spread), + `randn * 0.02` noise; then `agent.reset(init_qpos)` and root pose `p=[0,0,0.28], q=[0,0,-1,0]`.

**Code.**
```python
def _initialize_episode(self, env_idx, options):
    self._initialize_actors(env_idx); self._initialize_agent(env_idx)

def _initialize_actors(self, env_idx):
    with torch.device(self.device):
        b = len(env_idx); self.table_scene.initialize(env_idx)
        if self.difficulty_level <= 3:
            self.rotate_direction = torch.ones(b)
        else:
            self.rotate_direction = 1 - torch.randint(0, 2, (b,)) * 2
        xyz = torch.zeros((b, 3)); xyz[:, :2].uniform_(-0.02, 0.02)
        axis_angle = torch.zeros((b, 3)); axis_angle[:, 2].uniform_(torch.pi/6, torch.pi*5/6)
        pose = Pose.create_from_pq(xyz, axis_angle_to_quaternion(axis_angle))
        self.valve.set_pose(pose)
        qpos = torch.rand((b, 1)) * torch.pi * 2 - torch.pi
        self.valve.set_qpos(qpos); self.rest_qpos = qpos

def _initialize_agent(self, env_idx):
    with torch.device(self.device):
        b = len(env_idx); dof = self.agent.robot.dof
        if isinstance(dof, torch.Tensor): dof = dof[0]
        init_qpos = torch.zeros((b, dof))
        init_qpos[:, self.agent.root_joint_indices] = torch.tensor([0.7, -0.7, -0.7])
        init_qpos += torch.randn((b, dof)) * self.robot_init_qpos_noise
        self.agent.reset(init_qpos)
        self.agent.robot.set_pose(Pose.create_from_pq(
            torch.tensor([0.0, 0, 0.28]), torch.tensor([0, 0, -1, 0])))
```

---

## §4 Goal + Termination (`evaluate`)

**Description.** Success = the valve's joint has rotated, in the intended direction, by more than `success_threshold` from its reset angle. No early failure termination; episodes end on `success` (reported via `evaluate`) or the `max_episode_steps` time limit.

**Decisions resolved.**
- `success_threshold`: Level0 → π/2, Level4 → 2π, **Level1 (and 2,3) → π**.
- `valve_rotation = (valve.qpos - rest_qpos)[:, 0]`; `success = valve_rotation * rotate_direction > success_threshold`.
- `evaluate` returns `dict(success=..., valve_rotation=...)`; `valve_rotation` is consumed by the reward via `info`.
- `max_episode_steps = 150` (Level1). (Level0=80, Level2/3=150, Level4=300.)

**Code.**
```python
def evaluate(self, **kwargs) -> dict:
    valve_rotation = (self.valve.qpos - self.rest_qpos)[:, 0]
    success = valve_rotation * self.rotate_direction > self.success_threshold
    return dict(success=success, valve_rotation=valve_rotation)
```

---

## §5 Observation (`_get_obs_extra`)

**Description.** Default obs_mode = `"state"` → flat 51-dim float32 vector = D'Claw proprioception (qpos + qvel + 3 fingertip 7-D poses) ⊕ task-extra (valve angle/velocity + cos/sin embedding + rotate direction) ⊕ (state-only) the full valve 7-D pose.

**Decisions resolved (dim breakdown to 51).**
- Proprioception (`DClaw.get_proprioception` → base qpos/qvel + tip_poses): qpos 9 + qvel 9 + `tip_poses` 3×7 = 21 → **39**.
- `_get_obs_extra` always: `rotate_dir` 1 + `valve_qpos` 1 + `valve_qvel` 1 + `valve_x=cos(valve_qpos)` 1 + `valve_y=sin(valve_qpos)` 1 → **5**.
- state-only (`obs_mode_struct.use_state`): `valve_pose = vectorize_pose(valve.pose)` 7 → **7**.
- Total = 39 + 5 + 7 = **51** ✓ (matches `Box(-inf, inf, (1, 51))`).
- `SUPPORTED_OBS_MODES` (BaseEnv): `state` (default), `state_dict`, `none`, `sensor_data`, `any_textures` (rgb/depth/segmentation/rgbd…), `pointcloud`. The valve angle is exposed as cos/sin to avoid the ±π wraparound discontinuity.

**Code.**
```python
def _get_obs_extra(self, info: dict):
    with torch.device(self.device):
        valve_qpos = self.valve.qpos; valve_qvel = self.valve.qvel
        obs = dict(
            rotate_dir=self.rotate_direction.to(torch.float32),
            valve_qpos=valve_qpos, valve_qvel=valve_qvel,
            valve_x=torch.cos(valve_qpos[:, 0]), valve_y=torch.sin(valve_qpos[:, 0]),
        )
        if self.obs_mode_struct.use_state:
            obs.update(valve_pose=vectorize_pose(self.valve.pose))
        return obs

# DClaw.get_proprioception (adds tip_poses to base proprioception)
def get_proprioception(self):
    obs = super().get_proprioception()
    obs.update({"tip_poses": self.tip_poses.view(-1, len(self.tip_links) * 7)})  # 3*7=21
    return obs
```

---

## §6 Reward (`compute_dense_reward` / `compute_normalized_dense_reward`)

**Description.** Three additive shaping terms (composer = **sum**): (1) a fingertip-placement term that rewards each of the 3 fingertips being at the desired radius from the valve hub (so the fingers grasp the valve rim), (2) a directed-velocity term rewarding spinning the valve in the intended direction (dominant, ×4), (3) a progress term proportional to total rotation, clipped to [-1, 1]. The normalized reward divides the dense reward by `6.0` (the stated max).

**Decisions resolved.**
- Composer: **sum** of three terms.
- Term 1 (placement): `error = ||valve_tip_dist - (capsule_lens - capsule_offset)||` over the 3 fingertips (each fingertip's planar distance to the valve hub vs the head capsule length minus `capsule_offset=0.01`); `reward = 1 - tanh(10*error)`. Range ≈ [0, 1].
- Term 2 (directed velocity): `reward += tanh(5 * (valve_qvel[:,0] * rotate_direction)) * 4`. Range ≈ [-4, 4]; **dominant** signal.
- Term 3 (progress): `reward += clip(rotation / π / 2, -1, 1)` where `rotation = info["valve_rotation"]`. Range [-1, 1].
- Max possible ≈ 1 + 4 + 1 = **6** → `compute_normalized_dense_reward = compute_dense_reward / 6.0`.
- Default `reward_mode = "normalized_dense"` (BaseEnv `SUPPORTED_REWARD_MODES[0]`).
- (No per-stage planning-budget docstring in source; per-step saturated magnitudes retro-computed above.)

**Code (verbatim).**
```python
def compute_dense_reward(self, obs: Any, action: Array, info: dict):
    rotation = info["valve_rotation"]
    qvel = self.valve.qvel

    # Distance between fingertips and the circle grouned by valve tips
    tip_poses = self.agent.tip_poses  # (b, 3, 7)
    tip_pos = tip_poses[:, :, :2]  # (b, 3, 2)
    valve_pos = self.valve_link.pose.p[:, :2]  # (b, 2)
    valve_tip_dist = torch.linalg.norm(tip_pos - valve_pos[:, None, :], dim=-1)
    desired_valve_tip_dist = self.capsule_lens[:, None] - self.capsule_offset
    error = torch.norm(valve_tip_dist - desired_valve_tip_dist, dim=-1)
    reward = 1 - torch.tanh(error * 10)

    directed_velocity = qvel[:, 0] * self.rotate_direction
    reward += torch.tanh(5 * directed_velocity) * 4

    motion_reward = torch.clip(rotation / torch.pi / 2, -1, 1)
    reward += motion_reward

    return reward

def compute_normalized_dense_reward(self, obs: Any, action: Array, info: dict):
    # this should be equal to compute_dense_reward / max possible reward
    return self.compute_dense_reward(obs=obs, action=action, info=info) / 6.0
```
Helper `self.agent.tip_poses` (DClaw): stacks the 3 fingertip 7-D poses → `(b, 3, 7)`.

---

## §7 DR

`<no DR>` — there are no `startup`/`interval` randomization events. ManiSkill has no `EventCfg`; all stochasticity lives in `_initialize_episode` (reset-time): valve XY/yaw jitter, valve initial angle, and `robot_init_qpos_noise=0.02`. (Level ≥ 3 additionally randomizes valve radius scales at scene-load; Level1 does not.) No per-step / startup domain randomization.

---

## Reproduce
`/harbor:task-create name=<new_task_id> from=<ManiSkill-repo>/harbor/create-task/rotatevalvelevel1-v1-implementation.md`
