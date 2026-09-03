# OpenCabinetDoor-v1 — Implementation Spec

- robot: Fetch mobile manipulator
- simulator: ManiSkill (SAPIEN, `BaseEnv` + `@register_env`)
- objects: PartnetMobility cabinet (target door), ground
- bimanual: false
- summary: Drive to a cabinet and open its door at least three quarters of the way.

> **Class hierarchy note.** `OpenCabinetDoor-v1` is a *thin subclass* of `OpenCabinetDrawerEnv`. It overrides only `TRAIN_JSON` (door cabinets instead of drawer cabinets) and `handle_types = ["revolute", "revolute_unwrapped"]` (door = revolute joint vs. drawer = prismatic). Every other method (`_load_agent`, `_load_scene`, `_load_cabinets`, `_initialize_episode`, `evaluate`, `_get_obs_extra`, `compute_dense_reward`, `compute_normalized_dense_reward`) is inherited verbatim from `OpenCabinetDrawerEnv`. All code below is the inherited implementation unless noted.

---

## §1 Registration + Scene

### Description
A Fetch mobile manipulator (mobile base + 7-DoF arm + torso/head + parallel gripper) is spawned 1.6–1.8 m from a PartnetMobility cabinet and must open a target **door** (revolute joint) at least 75% of the way. The cabinet model and target door link are randomly sampled per episode. The target handle's center-of-mass is the goal (visualized with a green kinematic sphere).

### Decisions resolved
- `SUPPORTED_ROBOTS = ["fetch"]`; `robot_uids="fetch"`; `robot_init_qpos_noise=0.02`.
- `handle_types = ["revolute", "revolute_unwrapped"]` (DOOR — overrides drawer's `["prismatic"]`).
- `TRAIN_JSON = PACKAGE_ASSET_DIR / "partnet_mobility/meta/info_cabinet_door_train.json"` (overrides drawer's `info_cabinet_drawer_train.json`).
- `min_open_frac = 0.75` (target open fraction → `target_qpos = qmin + (qmax-qmin)*0.75`).
- `max_episode_steps = 100` (from `@register_env`).
- Cabinet asset: PartnetMobility, fetched via `articulations.get_articulation_builder(scene, f"partnet-mobility:{model_id}")`; model_ids sampled from `all_model_ids` (keys of `info_cabinet_door_train.json`). Requires `partnet_mobility_cabinet` asset download (`asset_download_ids=["partnet_mobility_cabinet"]` declared on the drawer parent's `@register_env`).
- Asset path (resolved): `<ManiSkill PACKAGE_ASSET_DIR>/partnet_mobility/meta/info_cabinet_door_train.json` + per-model PartnetMobility URDFs under the PartnetMobility download dir. **WARN: asset not present on disk at probe time.**
- Robot spawned with base placed at `[1,0,0]` in `_load_agent`, then repositioned in `_initialize_episode`. Cabinet initial pose `[0,0,0]`, z corrected post-reconfigure so its collision-mesh bottom sits at z=0.
- Collision groups: ground + cabinet links share `CABINET_COLLISION_BIT=29` (disables cabinet self-collision and cabinet↔ground); Fetch wheels use `FETCH_WHEELS_COLLISION_BIT=30`, base uses `FETCH_BASE_COLLISION_BIT=31`.
- Sim cfg: `SimConfig(spacing=5, gpu_memory_config=GPUMemoryConfig(max_rigid_contact_count=2**21, max_rigid_patch_count=2**19))`.
- `_default_sensor_configs = []` (no policy cameras by default).
- Human render camera: `look_at(eye=[-1.8,-1.3,1.8], target=[-0.3,0.5,0])`, 512×512, fov=1.
- `reconfiguration_freq`: 1 if `num_envs==1` else 0 (resamples cabinet/door each reset only in single-env mode).

### Code

```python
CABINET_COLLISION_BIT = 29

@register_env(
    "OpenCabinetDrawer-v1",
    asset_download_ids=["partnet_mobility_cabinet"],
    max_episode_steps=100,
)
class OpenCabinetDrawerEnv(BaseEnv):
    _sample_video_link = "https://github.com/haosulab/ManiSkill/raw/main/figures/environment_demos/OpenCabinetDrawer-v1_rt.mp4"

    SUPPORTED_ROBOTS = ["fetch"]
    agent: Union[Fetch]
    handle_types = ["prismatic"]
    TRAIN_JSON = (
        PACKAGE_ASSET_DIR / "partnet_mobility/meta/info_cabinet_drawer_train.json"
    )

    min_open_frac = 0.75

    def __init__(self, *args, robot_uids="fetch", robot_init_qpos_noise=0.02,
                 reconfiguration_freq=None, num_envs=1, **kwargs):
        self.robot_init_qpos_noise = robot_init_qpos_noise
        train_data = load_json(self.TRAIN_JSON)
        self.all_model_ids = np.array(list(train_data.keys()))
        if reconfiguration_freq is None:
            if num_envs == 1:
                reconfiguration_freq = 1
            else:
                reconfiguration_freq = 0
        super().__init__(*args, robot_uids=robot_uids,
                         reconfiguration_freq=reconfiguration_freq,
                         num_envs=num_envs, **kwargs)

    @property
    def _default_sim_config(self):
        return SimConfig(
            spacing=5,
            gpu_memory_config=GPUMemoryConfig(
                max_rigid_contact_count=2**21, max_rigid_patch_count=2**19
            ),
        )

    @property
    def _default_sensor_configs(self):
        return []

    @property
    def _default_human_render_camera_configs(self):
        pose = sapien_utils.look_at(eye=[-1.8, -1.3, 1.8], target=[-0.3, 0.5, 0])
        return CameraConfig("render_camera", pose=pose, width=512, height=512,
                            fov=1, near=0.01, far=100)

    def _load_agent(self, options: dict):
        super()._load_agent(options, sapien.Pose(p=[1, 0, 0]))

    def _load_scene(self, options: dict):
        self.ground = build_ground(self.scene)
        sapien.set_log_level("off")
        self._load_cabinets(self.handle_types)
        sapien.set_log_level("warn")
        from mani_skill.agents.robots.fetch import FETCH_WHEELS_COLLISION_BIT
        self.ground.set_collision_group_bit(group=2, bit_idx=FETCH_WHEELS_COLLISION_BIT, bit=1)
        self.ground.set_collision_group_bit(group=2, bit_idx=CABINET_COLLISION_BIT, bit=1)

    def _load_cabinets(self, joint_types: list[str]):
        model_ids = self._batched_episode_rng.choice(self.all_model_ids)
        link_ids = self._batched_episode_rng.randint(0, 2**31)
        self._cabinets: list[Articulation] = []
        handle_links: list[list[Link]] = []
        handle_links_meshes: list[list[trimesh.Trimesh]] = []
        for i, model_id in enumerate(model_ids):
            cabinet_builder = articulations.get_articulation_builder(
                self.scene, f"partnet-mobility:{model_id}")
            cabinet_builder.set_scene_idxs(scene_idxs=[i])
            cabinet_builder.initial_pose = sapien.Pose(p=[0, 0, 0], q=[1, 0, 0, 0])
            cabinet = cabinet_builder.build(name=f"{model_id}-{i}")
            self.remove_from_state_dict_registry(cabinet)
            for link in cabinet.links:
                link.set_collision_group_bit(group=2, bit_idx=CABINET_COLLISION_BIT, bit=1)
            self._cabinets.append(cabinet)
            handle_links.append([])
            handle_links_meshes.append([])
            for link, joint in zip(cabinet.links, cabinet.joints):
                if joint.type[0] in joint_types:
                    handle_links[-1].append(link)
                    handle_links_meshes[-1].append(
                        link.generate_mesh(
                            filter=lambda _, render_shape: "handle" in render_shape.name,
                            mesh_name="handle",
                        )[0]
                    )
        self.cabinet = Articulation.merge(self._cabinets, name="cabinet")
        self.add_to_state_dict_registry(self.cabinet)
        self.handle_link = Link.merge(
            [links[link_ids[i] % len(links)] for i, links in enumerate(handle_links)],
            name="handle_link",
        )
        self.handle_link_pos = common.to_tensor(
            np.array([
                meshes[link_ids[i] % len(meshes)].bounding_box.center_mass
                for i, meshes in enumerate(handle_links_meshes)
            ]),
            device=self.device,
        )
        self.handle_link_goal = actors.build_sphere(
            self.scene, radius=0.02, color=[0, 1, 0, 1], name="handle_link_goal",
            body_type="kinematic", add_collision=False,
            initial_pose=sapien.Pose(p=[0, 0, 0], q=[1, 0, 0, 0]),
        )

    def _after_reconfigure(self, options):
        self.cabinet_zs = []
        for cabinet in self._cabinets:
            collision_mesh = cabinet.get_first_collision_mesh()
            self.cabinet_zs.append(-collision_mesh.bounding_box.bounds[0, 2])
        self.cabinet_zs = common.to_tensor(self.cabinet_zs, device=self.device)
        target_qlimits = self.handle_link.joint.limits  # [b, 1, 2]
        qmin, qmax = target_qlimits[..., 0], target_qlimits[..., 1]
        self.target_qpos = qmin + (qmax - qmin) * self.min_open_frac

    def handle_link_positions(self, env_idx: Optional[torch.Tensor] = None):
        if env_idx is None:
            return transform_points(
                self.handle_link.pose.to_transformation_matrix().clone(),
                common.to_tensor(self.handle_link_pos, device=self.device))
        return transform_points(
            self.handle_link.pose[env_idx].to_transformation_matrix().clone(),
            common.to_tensor(self.handle_link_pos[env_idx], device=self.device))


@register_env("OpenCabinetDoor-v1", max_episode_steps=100)
class OpenCabinetDoorEnv(OpenCabinetDrawerEnv):
    TRAIN_JSON = (
        PACKAGE_ASSET_DIR / "partnet_mobility/meta/info_cabinet_door_train.json"
    )
    handle_types = ["revolute", "revolute_unwrapped"]
```

**Fetch agent (`mani_skill/agents/robots/fetch/fetch.py`), DoF layout (full robot = 15 DoF):**
- base: `root_x_axis_joint`, `root_y_axis_joint`, `root_z_rotation_joint` (3 — mobile base x/y translation + yaw)
- body: `head_pan_joint`, `head_tilt_joint`, `torso_lift_joint` (3)
- arm: `shoulder_pan_joint`, `shoulder_lift_joint`, `upperarm_roll_joint`, `elbow_flex_joint`, `forearm_roll_joint`, `wrist_flex_joint`, `wrist_roll_joint` (7)
- gripper: `l_gripper_finger_joint`, `r_gripper_finger_joint` (2, mimic-coupled)
- `ee_link_name = "gripper_link"`; `tcp` = gripper_link; arm stiffness 1e3 / damping 1e2 / force 100.
- URDF: `{PACKAGE_ASSET_DIR}/robots/fetch/fetch.urdf`; IK URDF `fetch_torso_up.urdf` (root link `torso_lift_link`).

### Smoke
```bash
cd <repo>
.venv/bin/python -c "import gymnasium as gym, mani_skill; e=gym.make('OpenCabinetDoor-v1'); print(e.observation_space, e.action_space); e.close()"
```
Expected stdout (after `python -m mani_skill.utils.download_asset partnet_mobility_cabinet`):
`Box(-inf, inf, (44,), float32) Box(<lo>, <hi>, (13,), float32)`
**WARN: not captured at probe time — asset missing.**

---

## §2 Actions

### Description
Default control mode is `pd_joint_delta_pos` (first key in Fetch's `controller_configs`). The action is the concatenation of four sub-controllers, in dict-iteration order **arm → gripper → body → base**.

### Decisions resolved
- `control_mode = "pd_joint_delta_pos"` (default).
- **arm** (7 dims): `PDJointPosController`, delta mode, per-joint range `[-0.1, 0.1]`, stiffness 1e3, damping 1e2, force 100. Normalized action mapped to that delta range.
- **gripper** (1 dim): `PDJointPosMimicController`, range `[-0.01, 0.05]`, stiffness 1e3, damping 1e2, force 100, mimic `r_gripper_finger_joint ← l_gripper_finger_joint` → 2 joints collapse to **1** action dim.
- **body** (3 dims): `PDJointPosController` over `[head_pan, head_tilt, torso_lift]`, delta `[-0.1, 0.1]`, stiffness 1e3, damping 1e2, force 100.
- **base** (2 dims): `PDBaseForwardVelController` over `[root_x, root_y, root_z_rotation]` but exposes a **2-dim** forward-velocity action `[forward_vel, yaw_vel]`, bounds `lower=[-1,-3.14] upper=[1,3.14]`, damping 1000, force 500. (Forward-velocity controller drives the holonomic root joints; action dim = 2.)
- **Total action dim = 7 + 1 + 3 + 2 = 13.**
- Other available control modes (not default): `pd_joint_pos`, `pd_ee_delta_pos`, `pd_ee_delta_pose`, `pd_ee_delta_pose_align`, `pd_joint_target_delta_pos`, `pd_ee_target_delta_pos`, `pd_ee_target_delta_pose`, `pd_joint_vel`, `pd_joint_pos_vel`, `pd_joint_delta_pos_vel`, `pd_joint_delta_pos_stiff_body`. All share the same gripper(1) + base(2); arm/body dims vary by mode.

### Code (Fetch `_controller_configs`, default-mode pieces)
```python
arm_pd_joint_delta_pos = PDJointPosControllerConfig(
    self.arm_joint_names, -0.1, 0.1,
    self.arm_stiffness, self.arm_damping, self.arm_force_limit, use_delta=True)

gripper_pd_joint_pos = PDJointPosMimicControllerConfig(
    self.gripper_joint_names, -0.01, 0.05,
    self.gripper_stiffness, self.gripper_damping, self.gripper_force_limit,
    mimic={"r_gripper_finger_joint": {"joint": "l_gripper_finger_joint"}})

body_pd_joint_delta_pos = PDJointPosControllerConfig(
    self.body_joint_names, -0.1, 0.1,
    self.body_stiffness, self.body_damping, self.body_force_limit, use_delta=True)

base_pd_joint_vel = PDBaseForwardVelControllerConfig(
    self.base_joint_names, lower=[-1, -3.14], upper=[1, 3.14],
    damping=1000, force_limit=500)

controller_configs = dict(
    pd_joint_delta_pos=dict(
        arm=arm_pd_joint_delta_pos,
        gripper=gripper_pd_joint_pos,
        body=body_pd_joint_delta_pos,
        base=base_pd_joint_vel),
    ...
)
```

### Smoke
`e.action_space.shape == (13,)`. **WARN: not captured at probe time.**

---

## §3 Reset (`_initialize_episode`)

### Description
Per reset: place cabinet (z so collision-mesh bottom sits on floor); set Fetch to a fixed home qpos but reposition its base on a ring 1.6–1.8 m from the cabinet, facing it (yaw = `theta - pi` plus ±9° noise); close all cabinet joints to their lower qlimit; on GPU sim do one settling step to suppress spurious self-opening; finally snap the green goal sphere to the handle COM.

### Decisions resolved
- Fetch home qpos (15) = `[0,0,0, 0,0,0,0, -pi/4,0,pi/4,0,pi/3,0, 0.015,0.015]` (base x/y/yaw overwritten below).
- Base placement: `dist ~ U(1.6, 1.8)`; `theta ~ U(0.9*pi, 1.1*pi)`; `x = cos(theta)*dist`, `y = sin(theta)*dist`; `yaw = (theta - pi) + U(-0.05*pi, 0.05*pi)` (i.e. base orientation jitter ±9°).
- Cabinet pose: `p = [0, 0, cabinet_zs[env_idx]]` (xy=0).
- All cabinet joints closed to `qlimits[..., 0]` (lower bound), qvel zeroed.
- GPU-sim settling workaround: `_gpu_apply_all → gpu_update_articulation_kinematics → px.step → _gpu_fetch_all`.
- `robot_init_qpos_noise = 0.02` is stored but **not** applied to the home qpos in this reset (the only randomization is base xy/yaw above).
- Goal sphere set to `handle_link_positions(env_idx)`.
- `_after_control_step`: every control step re-syncs the goal sphere to the (now-moving) handle COM (GPU kinematics update + fetch + apply).

### Code
```python
def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
    with torch.device(self.device):
        b = len(env_idx)
        xy = torch.zeros((b, 3))
        xy[:, 2] = self.cabinet_zs[env_idx]
        self.cabinet.set_pose(Pose.create_from_pq(p=xy))

        if self.robot_uids == "fetch":
            qpos = torch.tensor(
                [0, 0, 0, 0, 0, 0, 0, -np.pi / 4, 0, np.pi / 4, 0,
                 np.pi / 3, 0, 0.015, 0.015])
            qpos = qpos.repeat(b).reshape(b, -1)
            dist = randomization.uniform(1.6, 1.8, size=(b,))
            theta = randomization.uniform(0.9 * torch.pi, 1.1 * torch.pi, size=(b,))
            xy = torch.zeros((b, 2))
            xy[:, 0] += torch.cos(theta) * dist
            xy[:, 1] += torch.sin(theta) * dist
            qpos[:, :2] = xy
            noise_ori = randomization.uniform(-0.05 * torch.pi, 0.05 * torch.pi, size=(b,))
            ori = (theta - torch.pi) + noise_ori
            qpos[:, 2] = ori
            self.agent.robot.set_qpos(qpos)
            self.agent.robot.set_pose(sapien.Pose())

        qlimits = self.cabinet.get_qlimits()  # [b, max_dof, 2]
        self.cabinet.set_qpos(qlimits[env_idx, :, 0])
        self.cabinet.set_qvel(self.cabinet.qpos[env_idx] * 0)

        if self.gpu_sim_enabled:
            self.scene._gpu_apply_all()
            self.scene.px.gpu_update_articulation_kinematics()
            self.scene.px.step()
            self.scene._gpu_fetch_all()

        self.handle_link_goal.set_pose(
            Pose.create_from_pq(p=self.handle_link_positions(env_idx)))
```

### Smoke
`e.reset(seed=0)` runs without error; two resets with different seeds give different base xy/yaw. **WARN: not captured (asset missing).**

---

## §4 Goal + Termination (`evaluate`)

### Description
Success when the target door joint is open at least 75% of the way **and** the door link is nearly static (low angular + linear velocity). No failure/timeout DoneTerm in code beyond `max_episode_steps`.

### Decisions resolved
- `open_enough = handle_link.joint.qpos >= target_qpos`, where `target_qpos = qmin + (qmax-qmin)*0.75`.
- `link_is_static = (||angular_velocity|| <= 1.0) & (||linear_velocity|| <= 0.1)`.
- `success = open_enough & link_is_static`.
- `info` carries `handle_link_pos` (handle COM world pos) and `open_enough` — both consumed by reward (§6) and obs (§5).
- `max_episode_steps = 100` (from `@register_env`); no early-termination DoneTerm — episode ends on success-driven termination via the BaseEnv success→terminated wiring plus the 100-step truncation.

### Code
```python
def evaluate(self):
    open_enough = self.handle_link.joint.qpos >= self.target_qpos
    handle_link_pos = self.handle_link_positions()
    link_is_static = (
        torch.linalg.norm(self.handle_link.angular_velocity, axis=1) <= 1
    ) & (torch.linalg.norm(self.handle_link.linear_velocity, axis=1) <= 0.1)
    return {
        "success": open_enough & link_is_static,
        "handle_link_pos": handle_link_pos,
        "open_enough": open_enough,
    }
```

### Smoke
`info` from `e.step` contains keys `success`, `handle_link_pos`, `open_enough`; `success` is boolean tensor. **WARN: not captured (asset missing).**

---

## §5 Observation (`_get_obs_extra`)

### Description
Default `obs_mode = "state"`. Observation = agent proprioception (qpos + qvel of full 15-DoF robot) concatenated with task-extra terms. In state modes, extra terms add tcp pose, tcp→handle vector, target joint qpos, and target handle position. The goal handle pose is implicitly available via `target_handle_pos`.

### Decisions resolved
- `obs_mode = "state"` (default). Supported modes include `state`, `state_dict`, `none`, plus visual modes (`rgb`, `rgbd`, `pointcloud`, `sensor_data`) — but `_default_sensor_configs = []`, so visual modes carry only the human-render camera, not policy cameras.
- **Proprioception** (`get_proprioception`): `qpos` (15) + `qvel` (15) = **30**. No controller state for `pd_joint_delta_pos` (non-target controllers → empty `controller.get_state()`).
- **Extra obs** (always): `tcp_pose` = `agent.tcp.pose.raw_pose` → 7 (xyz + wxyz quat).
- **Extra obs** (state modes only): `tcp_to_handle_pos` (3) + `target_link_qpos` (1) + `target_handle_pos` (3) = 7.
- **Total state obs dim = 30 + 7 + 7 = 44.**

### Code
```python
def _get_obs_extra(self, info: dict):
    obs = dict(
        tcp_pose=self.agent.tcp.pose.raw_pose,
    )
    if "state" in self.obs_mode:
        obs.update(
            tcp_to_handle_pos=info["handle_link_pos"] - self.agent.tcp.pose.p,
            target_link_qpos=self.handle_link.joint.qpos,
            target_handle_pos=info["handle_link_pos"],
        )
    return obs
```
(Proprioception comes from `BaseAgent.get_proprioception`: `dict(qpos=robot.get_qpos(), qvel=robot.get_qvel())`.)

### Smoke
`e.observation_space.shape == (44,)` in state mode. **WARN: not captured (asset missing).**

---

## §6 Reward (`compute_dense_reward` + `compute_normalized_dense_reward`)

### Description
Two-stage dense reward: a reaching term pulls the TCP to the handle, then an opening term rewards joint progress toward `target_qpos`. Once the door starts to open, the reach term is saturated to 2; once `open_enough`, the open term is saturated to 3; on full `success`, the whole reward is overwritten to 5.0. Normalized variant divides by `max_reward = 5.0`.

### Decisions resolved
- **Composer = SUM**: `reward = reaching_reward + open_reward`, then hard overwrites for the saturated/success cases.
- `reaching_reward = 1 - tanh(5 * tcp_to_handle_dist)` ∈ (0,1], overwritten to **2** once `amount_to_open_left < 0.999` (door cracked open at all).
- `amount_to_open_left = (target_qpos - joint.qpos) / target_qpos`; `open_reward = 2 * (1 - amount_to_open_left)`, overwritten to **3** when `info["open_enough"]`.
- `reward[info["success"]] = 5.0` (terminal saturation).
- Per-stage saturated per-step magnitudes (retro-computed from the overwrites):
  - far from handle, door closed: `reward ≈ reaching_reward(0..1) + 0 ≈ 0..1`.
  - door cracked open, not enough: `reward = 2 (reach saturated) + open_reward(0..~1.5)`.
  - `open_enough` but moving: `reward = 2 + 3 = 5` (but `success=False` since link not static — note this already equals the success value).
  - full success (open + static): `reward = 5.0`.
  - `max_reward = 5.0` → normalized reward ∈ [~0, 1.0].

### Code (verbatim — both functions)
```python
def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: dict):
    tcp_to_handle_dist = torch.linalg.norm(
        self.agent.tcp.pose.p - info["handle_link_pos"], axis=1
    )
    reaching_reward = 1 - torch.tanh(5 * tcp_to_handle_dist)
    amount_to_open_left = torch.div(
        self.target_qpos - self.handle_link.joint.qpos, self.target_qpos
    )
    open_reward = 2 * (1 - amount_to_open_left)
    reaching_reward[
        amount_to_open_left < 0.999
    ] = 2  # if joint opens even a tiny bit, we don't need reach reward anymore
    open_reward[info["open_enough"]] = 3  # give max reward here
    reward = reaching_reward + open_reward
    reward[info["success"]] = 5.0
    return reward

def compute_normalized_dense_reward(self, obs: Any, action: torch.Tensor, info: dict):
    max_reward = 5.0
    return self.compute_dense_reward(obs=obs, action=action, info=info) / max_reward
```

### Smoke
Reward is finite and non-constant across a random rollout; for the normalized variant, reward ∈ [0, 1.0]. **WARN: not captured (asset missing).**

---

## §7 DR (Domain Randomization)

`<no DR>` — there is no `startup`/`interval`-mode randomization (no friction/mass/visual randomization). The only per-episode variation is the *task-level* randomization in `_initialize_episode` / `_load_cabinets` (cabinet model sampled from the door-cabinet set, door link sampled, robot base pose 1.6–1.8 m on a ring with ±9° yaw jitter). These are reset-time scene sampling, not physical-parameter DR.

---

## WARNINGs
- Cabinet asset (`partnet_mobility_cabinet`) not downloaded at probe time → `gym.make` raises; all spaces/dims/smokes derived analytically from source. Run `python -m mani_skill.utils.download_asset partnet_mobility_cabinet` before reproducing.
- Action/obs dims (13 / 44) assume default `control_mode="pd_joint_delta_pos"` and `obs_mode="state"`. Other modes change dims.
