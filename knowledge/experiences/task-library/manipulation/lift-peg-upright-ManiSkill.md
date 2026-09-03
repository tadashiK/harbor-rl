# LiftPegUpright-v1 — Implementation Spec

- robot: Franka Panda (default; Fetch also supported)
- simulator: ManiSkill (SAPIEN, `BaseEnv` + `@register_env`)
- objects: two-color box peg, table
- bimanual: false
- summary: Move a peg lying flat on the table into an upright orientation.

ManiSkill task: move a peg lying flat on the table into an upright orientation. Subclasses `mani_skill.envs.sapien_env.BaseEnv`; registered with `@register_env("LiftPegUpright-v1", max_episode_steps=50)`. Robots supported: `panda` (default), `fetch`.

---

## §1 Registration + Scene

**Description.** The env is registered as `LiftPegUpright-v1` with a 50-step episode cap. The scene is the standard `TableSceneBuilder` (table + ground + lights + robot keyframe), plus a single dynamic two-color box peg of half-length 0.12 and half-width 0.025. The Panda agent base is offset to `[-0.615, 0, 0]`. No `_default_sim_config` override → base `SimConfig()` defaults apply (`sim_freq=100`, `control_freq=20`).

**Decisions resolved.**
- `@register_env("LiftPegUpright-v1", max_episode_steps=50)`
- `SUPPORTED_ROBOTS = ["panda", "fetch"]`; default `robot_uids="panda"`, `robot_init_qpos_noise=0.02`
- `peg_half_width = 0.025`, `peg_half_length = 0.12`
- Peg: `actors.build_twocolor_peg(length=0.12, width=0.025, color_1=[176,14,14,255]/255, color_2=[12,42,160,255]/255, name="peg", body_type="dynamic", initial_pose=sapien.Pose(p=[0,0,0.1]))` → box collision half_size `[0.12, 0.025, 0.025]`; two visual halves.
- Agent base pose: `sapien.Pose(p=[-0.615, 0, 0])`
- Sim defaults: `SimConfig()` → `sim_freq=100`, `control_freq=20` (no per-env override)
- Sensors: `base_camera` 128×128 at `look_at(eye=[0.3,0,0.6], target=[-0.1,0,0.1])`; human render `render_camera` 512×512 at `look_at([0.6,0.7,0.6],[0.0,0.0,0.35])`

**Code (registration + scene, verbatim).**
```python
@register_env("LiftPegUpright-v1", max_episode_steps=50)
class LiftPegUprightEnv(BaseEnv):
    _sample_video_link = "https://github.com/haosulab/ManiSkill/raw/main/figures/environment_demos/LiftPegUpright-v1_rt.mp4"
    SUPPORTED_ROBOTS = ["panda", "fetch"]
    agent: Union[Panda, Fetch]

    peg_half_width = 0.025
    peg_half_length = 0.12

    def __init__(self, *args, robot_uids="panda", robot_init_qpos_noise=0.02, **kwargs):
        self.robot_init_qpos_noise = robot_init_qpos_noise
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    @property
    def _default_sensor_configs(self):
        pose = look_at(eye=[0.3, 0, 0.6], target=[-0.1, 0, 0.1])
        return [CameraConfig("base_camera", pose, 128, 128, np.pi / 2, 0.01, 100)]

    @property
    def _default_human_render_camera_configs(self):
        pose = look_at([0.6, 0.7, 0.6], [0.0, 0.0, 0.35])
        return CameraConfig("render_camera", pose, 512, 512, 1, 0.01, 100)

    def _load_agent(self, options: dict):
        super()._load_agent(options, sapien.Pose(p=[-0.615, 0, 0]))

    def _load_scene(self, options: dict):
        self.table_scene = TableSceneBuilder(
            env=self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.table_scene.build()

        # the peg that we want to manipulate
        self.peg = actors.build_twocolor_peg(
            self.scene,
            length=self.peg_half_length,
            width=self.peg_half_width,
            color_1=np.array([176, 14, 14, 255]) / 255,
            color_2=np.array([12, 42, 160, 255]) / 255,
            name="peg",
            body_type="dynamic",
            initial_pose=sapien.Pose(p=[0, 0, 0.1]),
        )
```

**Peg builder (verbatim, `mani_skill/utils/building/actors/common.py:230`).**
```python
def build_twocolor_peg(
    scene: ManiSkillScene,
    length,
    width,
    color_1,
    color_2,
    name: str,
    body_type="dynamic",
    add_collision: bool = True,
    scene_idxs: Optional[Array] = None,
    initial_pose: Optional[Union[Pose, sapien.Pose]] = None,
):
    builder = scene.create_actor_builder()
    if add_collision:
        builder.add_box_collision(
            half_size=[length, width, width],
        )
    builder.add_box_visual(
        pose=sapien.Pose(p=[-length / 2, 0, 0]),
        half_size=[length / 2, width, width],
        material=sapien.render.RenderMaterial(base_color=color_1),
    )
    builder.add_box_visual(
        pose=sapien.Pose(p=[length / 2, 0, 0]),
        half_size=[length / 2, width, width],
        material=sapien.render.RenderMaterial(base_color=color_2),
    )
    return _build_by_type(builder, name, body_type, scene_idxs, initial_pose)
```

**Smoke (§1 build).**
```bash
cd <ManiSkill-repo>
.venv/bin/python -c "import gymnasium as gym, mani_skill; e=gym.make('LiftPegUpright-v1'); print(e.observation_space, e.action_space); e.close()"
```
Expected stdout:
```
Box(-inf, inf, (1, 32), float32) Box(-1.0, 1.0, (8,), float32)
```

---

## §2 Actions

**Description.** No `ActionsCfg` in ManiSkill — the action space is defined by the agent's controller. The env neither sets `control_mode` nor restricts `SUPPORTED_CONTROL_MODES`, so the agent's default controller (`pd_joint_delta_pos`, the first key in `Panda._controller_configs`) is selected. That is 7 arm joints (delta PD position, ±0.1 rad) + 1 mimicked gripper joint → 8-dim action, normalized to `[-1, 1]`.

**Decisions resolved.**
- Default `control_mode = "pd_joint_delta_pos"` (first entry in controller dict; env does not override)
- Arm: `arm_pd_joint_delta_pos` = `PDJointPosControllerConfig(arm_joint_names, lower=-0.1, upper=0.1, stiffness=1e3, damping=1e2, force_limit=100, use_delta=True)` → 7 dims
- Gripper: `gripper_pd_joint_pos` = `PDJointPosMimicControllerConfig(gripper_joint_names, lower=-0.01, upper=0.04, stiffness=1e3, damping=1e2, force_limit=100, mimic={"panda_finger_joint2": {"joint": "panda_finger_joint1"}})` → 1 dim
- Action space: `Box(-1.0, 1.0, (8,), float32)`
- `arm_joint_names = ["panda_joint1".."panda_joint7"]`, `gripper_joint_names = ["panda_finger_joint1","panda_finger_joint2"]`, `ee_link_name = "panda_hand_tcp"`

**Code (Panda controller defaults, verbatim relevant slice — `mani_skill/agents/robots/panda/panda.py`).**
```python
arm_stiffness = 1e3
arm_damping = 1e2
arm_force_limit = 100
gripper_stiffness = 1e3
gripper_damping = 1e2
gripper_force_limit = 100

arm_pd_joint_delta_pos = PDJointPosControllerConfig(
    self.arm_joint_names,
    lower=-0.1,
    upper=0.1,
    stiffness=self.arm_stiffness,
    damping=self.arm_damping,
    force_limit=self.arm_force_limit,
    use_delta=True,
)

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
    pd_joint_delta_pos=dict(arm=arm_pd_joint_delta_pos, gripper=gripper_pd_joint_pos),
    pd_joint_pos=dict(arm=arm_pd_joint_pos, gripper=gripper_pd_joint_pos),
    ...
)
```

**Smoke (§2 action space).** `e.action_space == Box(-1.0, 1.0, (8,), float32)` (captured in §1 build above).

---

## §3 Reset / Episode initialization

**Description.** `_initialize_episode` re-initializes the table scene (robot keyframe + `robot_init_qpos_noise=0.02`) and randomizes the peg: xy uniform in a 0.2×0.2 square centered at origin (`[-0.1,0.1]×[-0.1,0.1]`), z set to `peg_half_width` (0.025) so it rests flat, and a fixed orientation `euler2quat(pi/2, 0, 0)` (lying on its side along its length).

**Decisions resolved.**
- `self.table_scene.initialize(env_idx)` (resets robot to `rest` keyframe with qpos noise 0.02)
- Peg xy: `torch.rand((b,2)) * 0.2 - 0.1` → uniform in `[-0.1, 0.1]²`
- Peg z: `peg_half_width = 0.025` (flat on table)
- Peg orientation: `q = euler2quat(np.pi/2, 0, 0)` (constant; peg laid flat along its length)

**Code (verbatim).**
```python
def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
    with torch.device(self.device):
        b = len(env_idx)
        self.table_scene.initialize(env_idx)

        xyz = torch.zeros((b, 3))
        xyz[..., :2] = torch.rand((b, 2)) * 0.2 - 0.1
        xyz[..., 2] = self.peg_half_width
        q = euler2quat(np.pi / 2, 0, 0)

        obj_pose = Pose.create_from_pq(p=xyz, q=q)
        self.peg.set_pose(obj_pose)
```

**Smoke (§3).** Reset over a batch yields finite peg poses with xy ∈ [-0.1,0.1] and z ≈ 0.025; covered by the §1/§5 build + a `env.reset(seed=0)` rollout.

---

## §4 Goal + Termination

**Description.** Success = peg is upright (its long axis vertical) AND lifted so its center is one half-length above the table. Upright is measured by the peg's Z (third) Euler angle in `"XYZ"` convention being within 0.08 rad of π/2 (in absolute value). Lifted is measured by the peg z-position being within 0.005 of `peg_half_length` (0.12). There is no early termination term — the only horizon control is `max_episode_steps=50` (time-limit truncation).

**Decisions resolved.**
- `is_peg_upright = |  |euler_z| − π/2 | < 0.08`  (euler from `matrix_to_euler_angles(quaternion_to_matrix(q), "XYZ")`, index 2)
- `close_to_table = | peg.z − 0.12 | < 0.005`
- `success = is_peg_upright & close_to_table`
- `max_episode_steps = 50` (set in `@register_env`); no `DoneTerm` / fail term — termination is success flag in `info`; truncation via horizon.

**Code (verbatim).**
```python
def evaluate(self):
    q = self.peg.pose.q
    qmat = rotation_conversions.quaternion_to_matrix(q)
    euler = rotation_conversions.matrix_to_euler_angles(qmat, "XYZ")
    is_peg_upright = (
        torch.abs(torch.abs(euler[:, 2]) - np.pi / 2) < 0.08
    )  # 0.08 radians of difference permitted
    close_to_table = torch.abs(self.peg.pose.p[:, 2] - self.peg_half_length) < 0.005
    return {
        "success": is_peg_upright & close_to_table,
    }
```

**Smoke (§4).** `info["success"]` is a bool tensor of shape `(num_envs,)`, all `False` at reset.

---

## §5 Observation

**Description.** Default `obs_mode="state"` (first in `SUPPORTED_OBS_MODES`). The flattened state vector = agent proprioception (Panda qpos 9 + qvel 9 = 18) + `_get_obs_extra`. `_get_obs_extra` always provides `tcp_pose` (7-dim raw pose: xyz + wxyz quat); in state mode it additionally provides `obj_pose` (peg raw pose, 7-dim). Total = 18 + 7 + 7 = **32**.

**Decisions resolved.**
- `obs_mode` default = `"state"` (`SUPPORTED_OBS_MODES = ('state','state_dict','none','sensor_data','any_textures','pointcloud')`)
- `_get_obs_extra`: `tcp_pose = agent.tcp.pose.raw_pose` (7); if `use_state`: `obj_pose = peg.pose.raw_pose` (7)
- Resolved total obs dim (state mode): 32 → `Box(-inf, inf, (1, 32), float32)`
  - agent proprio: qpos(9) + qvel(9) = 18
  - tcp_pose: 7
  - obj_pose: 7

**Code (verbatim).**
```python
def _get_obs_extra(self, info: dict):
    obs = dict(
        tcp_pose=self.agent.tcp.pose.raw_pose,
    )
    if self.obs_mode_struct.use_state:
        obs.update(
            obj_pose=self.peg.pose.raw_pose,
        )
    return obs
```

**Smoke (§5).** Captured in §1 build: `observation_space == Box(-inf, inf, (1, 32), float32)`.

---

## §6 Reward

**Description.** Dense reward composed by **summation** of four additive shaping terms, then overridden to a flat 3.0 on success. Terms: (1) rotation alignment as |cosine similarity| between the peg's local x-axis (rotated into world) and world up `(0,0,1)`, range [0,1]; (2) height-progress `1 − tanh(5·|z − half_length|)`; (3) reaching `(1 − tanh(5·||peg − tcp||))/5`, set to 1 (pre-division) when grasping → max 0.2; (4) on success the entire reward is set to the max 3.0. Normalized reward divides the dense reward by `max_reward = 3.0`.

**Composer:** sum (`reward = rot_rew + height_rew + reaching_rew`, then `reward[success] = 3`).

**Planning-budget (per-step saturated magnitudes, retro-computed from code):**
- rotation term `rot_rew` ∈ [0, 1] (max 1.0 when perfectly upright)
- height term `1 − tanh(5·z_dist)` ∈ [0, 1] (max 1.0 when peg center at half_length)
- reaching term ∈ [0, 0.2] (`/5`; pinned to 0.2 when grasping)
- success override = 3.0 (flat); `max_reward = 3.0` = 1.0 (rot) + 1.0 (height) + 0.2 (reach) is the non-success ceiling ≈ 2.2, and success snaps to 3.0.

**Decisions resolved.**
- `max_reward = 3.0` (normalization divisor)
- rotation: local `vec=(1,0,0)` rotated by peg rot-matrix, dotted with `goal_vec=(0,0,1)`, `.abs()` (both ±z upright valid)
- height: `z_dist = |peg.z − 0.12|`; `reward += 1 − tanh(5·z_dist)`
- reaching: `to_grip_dist = ||peg.p − tcp.p||`; `reaching_rew = 1 − tanh(5·to_grip_dist)`; `reaching_rew[is_grasping] = 1`; then `/5`
- success snap: `reward[info["success"]] = 3`

**Code (verbatim — full).**
```python
def compute_dense_reward(self, obs: Any, action: Array, info: dict):
    # rotation reward as cosine similarity between peg direction vectors
    # peg center of mass to end of peg, (1,0,0), rotated by peg pose rotation
    # dot product with its goal orientation: (0,0,1) or (0,0,-1)
    qmats = rotation_conversions.quaternion_to_matrix(self.peg.pose.q)
    vec = torch.tensor([1.0, 0, 0], device=self.device)
    goal_vec = torch.tensor([0, 0, 1.0], device=self.device)
    rot_vec = (qmats @ vec).view(-1, 3)
    # abs since (0,0,-1) is also valid, values in [0,1]
    rot_rew = (rot_vec @ goal_vec).view(-1).abs()
    reward = rot_rew

    # position reward using common maniskill distance reward pattern
    # giving reward in [0,1] for moving center of mass toward half length above table
    z_dist = torch.abs(self.peg.pose.p[:, 2] - self.peg_half_length)
    reward += 1 - torch.tanh(5 * z_dist)

    # small reward to motivate initial reaching
    # initially, we want to reach and grip peg
    to_grip_vec = self.peg.pose.p - self.agent.tcp.pose.p
    to_grip_dist = torch.linalg.norm(to_grip_vec, axis=1)
    reaching_rew = 1 - torch.tanh(5 * to_grip_dist)
    # reaching reward granted if gripping block
    reaching_rew[self.agent.is_grasping(self.peg)] = 1
    # weight reaching reward less
    reaching_rew = reaching_rew / 5
    reward += reaching_rew

    reward[info["success"]] = 3
    return reward

def compute_normalized_dense_reward(self, obs: Any, action: Array, info: dict):
    max_reward = 3.0
    return self.compute_dense_reward(obs=obs, action=action, info=info) / max_reward
```

**Helper (`is_grasping`, verbatim — `mani_skill/agents/robots/panda/panda.py`).**
```python
def is_grasping(self, object: Actor, min_force=0.5, max_angle=85):
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
```

**Smoke (§6).** Step a random rollout; assert reward is finite, non-constant, and `compute_normalized_dense_reward == compute_dense_reward / 3.0`.

---

## §7 DR (Domain Randomization)

`<no DR>` — beyond the §3 reset randomization (peg xy in `[-0.1,0.1]²` and `robot_init_qpos_noise=0.02`), the env defines no startup/interval domain-randomization events (no mass/friction/visual randomization). All randomness lives in `_initialize_episode`.

---

