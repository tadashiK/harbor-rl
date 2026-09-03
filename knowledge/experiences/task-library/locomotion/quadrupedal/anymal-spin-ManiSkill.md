# AnymalC-Spin-v1 — Implementation Spec

- robot: ANYbotics ANYmal-C quadruped (12 DoF)
- simulator: ManiSkill (SAPIEN, `BaseEnv` + `@register_env`)
- objects: none (flat ground plane)
- bimanual: false
- summary: Spin in place about the vertical axis as fast as possible without falling.

ManiSkill task: the Anymal-C quadruped (12 leg DoF) must spin in place about its vertical (yaw/z) axis as fast as possible. Reward is the base yaw angular velocity, minus stability/control penalties; a large terminal penalty applies if the body falls.

---

## §1 Registration + Scene

### Description
A SAPIEN `BaseEnv` subclass `QuadrupedSpinEnv` provides the shared quadruped-spin logic; the concrete `AnymalCSpinEnv` is registered as `AnymalC-Spin-v1` (default `robot_uids="anymal_c"`). Scene is just a flat ground plane (`build_ground`, width 400). The agent (Anymal-C, 12 leg actuated joints) is dropped in at z=1 and snapped to its `standing` keyframe on reset. No table, no manipulated objects. A base camera + render camera look at the robot from `[-1,1,2] → [0,0,0.5]`.

### Decisions resolved
- `register_env("AnymalC-Spin-v1", max_episode_steps=200)` on `AnymalCSpinEnv`.
- `SUPPORTED_ROBOTS = ["anymal_c", "unitree_go2_simplified_locomotion"]`; this task uses `anymal_c` (`agent: ANYmalC`).
- Sim cfg: `GPUMemoryConfig(max_rigid_contact_count=2**20)`, `SceneConfig(solver_position_iterations=4, solver_velocity_iterations=0)`.
- `_load_agent`: agent loaded at initial pose `sapien.Pose(p=[0, 0, 1])`.
- `_load_scene`: `build_ground(self.scene, floor_width=400)` (stored as `self.ground`).
- Sensor camera `base_camera` 128×128, fov π/2; render camera `render_camera` 512×512, fov 1; both `look_at([-1.0, 1.0, 2], [0, 0.0, 0.5])`.
- Anymal-C asset: `urdf_path = f"{ASSET_DIR}/robots/anymal_c/urdf/anymal.urdf"` (resolves to `<MS_ASSET_DIR>/robots/anymal_c/urdf/anymal.urdf`; default `~/.maniskill/data/...`). WARN: not on disk in this repo at probe time — must be downloaded.
- `fix_root_link = False`, `disable_self_collisions = True`; foot material static/dynamic friction 2.0; all links except root have gravity disabled (`_after_init`).
- 12 actuated joints (HAA/HFE/KFE × LF/RF/LH/RH). `standing` keyframe: pose `p=[0,0,0.545]`, qpos `[0.03,-0.03,0.03,-0.03, 0.4,0.4,-0.4,-0.4, -0.8,-0.8,0.8,0.8]`.

### Code
```python
# mani_skill/envs/tasks/quadruped/quadruped_spin.py
class QuadrupedSpinEnv(BaseEnv):
    SUPPORTED_ROBOTS = ["anymal_c", "unitree_go2_simplified_locomotion"]
    agent: ANYmalC
    default_qpos: torch.Tensor

    _UNDESIRED_CONTACT_LINK_NAMES: list[str] = None

    def __init__(self, *args, robot_uids="anymal-c", **kwargs):
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    @property
    def _default_sim_config(self):
        return SimConfig(
            gpu_memory_config=GPUMemoryConfig(max_rigid_contact_count=2**20),
            scene_config=SceneConfig(
                solver_position_iterations=4, solver_velocity_iterations=0
            ),
        )

    @property
    def _default_sensor_configs(self):
        pose = sapien_utils.look_at([-1.0, 1.0, 2], [0, 0.0, 0.5])
        return [
            CameraConfig(
                "base_camera", pose=pose, width=128, height=128,
                fov=np.pi / 2, near=0.01, far=100,
            )
        ]

    @property
    def _default_human_render_camera_configs(self):
        pose = sapien_utils.look_at([-1.0, 1.0, 2], [0, 0.0, 0.5])
        return [
            CameraConfig(
                "render_camera", pose=pose, width=512, height=512,
                fov=1, near=0.01, far=100,
            )
        ]

    def _load_agent(self, options: dict):
        super()._load_agent(options, sapien.Pose(p=[0, 0, 1]))

    def _load_scene(self, options: dict):
        self.ground = build_ground(self.scene, floor_width=400)


@register_env("AnymalC-Spin-v1", max_episode_steps=200)
class AnymalCSpinEnv(QuadrupedSpinEnv):
    _sample_video_link = "https://github.com/haosulab/ManiSkill/raw/main/figures/environment_demos/AnymalC-Spin-v1_rt.mp4"
    _UNDESIRED_CONTACT_LINK_NAMES = ["LF_KFE", "RF_KFE", "LH_KFE", "RH_KFE"]

    def __init__(self, *args, robot_uids="anymal_c", **kwargs):
        super().__init__(*args, robot_uids=robot_uids, **kwargs)
        self.default_qpos = torch.from_numpy(ANYmalC.keyframes["standing"].qpos).to(
            self.device
        )
```

```python
# mani_skill/agents/robots/anymal/anymal_c.py  (the agent)
@register_agent(asset_download_ids=["anymal_c"])
class ANYmalC(BaseAgent):
    uid = "anymal_c"
    urdf_path = f"{ASSET_DIR}/robots/anymal_c/urdf/anymal.urdf"
    urdf_config = dict(
        _materials=dict(
            foot=dict(static_friction=2.0, dynamic_friction=2.0, restitution=0.0)
        ),
        link=dict(
            LF_FOOT=dict(material="foot", patch_radius=0.1, min_patch_radius=0.1),
            LH_FOOT=dict(material="foot", patch_radius=0.1, min_patch_radius=0.1),
            RF_FOOT=dict(material="foot", patch_radius=0.1, min_patch_radius=0.1),
            RH_FOOT=dict(material="foot", patch_radius=0.1, min_patch_radius=0.1),
        ),
    )
    fix_root_link = False
    disable_self_collisions = True

    keyframes = dict(
        standing=Keyframe(
            pose=sapien.Pose(p=[0, 0, 0.545]),
            qpos=np.array(
                [0.03, -0.03, 0.03, -0.03, 0.4, 0.4, -0.4, -0.4, -0.8, -0.8, 0.8, 0.8]
            ),
        )
    )

    joint_names = [
        "LF_HAA", "RF_HAA", "LH_HAA", "RH_HAA",
        "LF_HFE", "RF_HFE", "LH_HFE", "RH_HFE",
        "LF_KFE", "RF_KFE", "LH_KFE", "RH_KFE",
    ]

    def _after_init(self):
        # disable gravity in all links but the root one
        for link in self.robot.links[1:]:
            link.disable_gravity = True

    def is_standing(self, ground_height=0):
        target_q = torch.tensor([1, 0, 0, 0], device=self.device)
        inner_prod = (self.robot.pose.q * target_q).sum(axis=1)
        angle_diff = torch.arccos(2 * (inner_prod**2) - 1)
        aligned = angle_diff < 0.349   # ~20 degrees
        high_enough = self.robot.pose.p[:, 2] > 0.35 + ground_height
        return aligned & high_enough

    def is_fallen(self):
        """This quadruped is considered fallen if its body contacts the ground"""
        forces = self.robot.get_net_contact_forces(["base"])
        return torch.norm(forces, dim=-1).max(-1).values > 1
```

### Smoke (§1 build)
```bash
cd <repo>
.venv/bin/python -c "import gymnasium as gym, mani_skill; e=gym.make('AnymalC-Spin-v1'); print(e.observation_space, e.action_space); e.close()"
```
Expected stdout: **not captured (verified via source read)** — aborts with `EOFError` from `download_asset.prompt_yes_no` until the `anymal_c` asset is downloaded. Analytic expectation once downloaded: `Box(..., (30,), float32) Box(-1.0, 1.0, (12,), float32)`.

---

## §2 Actions

### Description
Joint-space PD control over the 12 leg joints. ManiSkill does not declare an `ActionsCfg` block; the action space comes from the agent's controller (`ANYmalC._controller_configs`). Default `control_mode` is the first key, `pd_joint_delta_pos` — a normalized delta-position controller. The 12-D action ∈ [-1, 1] is rescaled to a per-step joint-position delta in [-0.225, 0.225] rad, then tracked by a PD law (stiffness 80, damping 2, force limit 100).

### Decisions resolved
- `control_mode` (default) = `"pd_joint_delta_pos"`; also available: `"pd_joint_pos"`.
- Action space: `Box(-1.0, 1.0, (12,))`, normalized (`normalize_action=True`, `use_delta=True`).
- Delta range: `lower=-0.225`, `upper=0.225` rad per step.
- PD gains: `stiffness=80.0`, `damping=2.0`, `force_limit=100`.
- `balance_passive_force=False` for both controllers.

### Code
```python
# mani_skill/agents/robots/anymal/anymal_c.py
@property
def _controller_configs(self):
    self.stiffness = 80.0
    self.damping = 2.0
    self.force_limit = 100
    # delta action scale for Omni Isaac Gym Envs is self.dt * self.action_scale = 1/60 * 13.5.
    pd_joint_delta_pos = PDJointPosControllerConfig(
        self.joint_names,
        -0.225,
        0.225,
        self.stiffness,
        self.damping,
        self.force_limit,
        normalize_action=True,
        use_delta=True,
    )
    pd_joint_pos = PDJointPosControllerConfig(
        self.joint_names,
        None,
        None,
        self.stiffness,
        self.damping,
        self.force_limit,
        normalize_action=False,
        use_delta=False,
    )
    controller_configs = dict(
        pd_joint_delta_pos=dict(body=pd_joint_delta_pos, balance_passive_force=False),
        pd_joint_pos=dict(body=pd_joint_pos, balance_passive_force=False),
    )
    return controller_configs
```

### Smoke (§2)
```bash
.venv/bin/python -c "import gymnasium as gym, mani_skill; e=gym.make('AnymalC-Spin-v1'); print(e.action_space); e.reset(); import numpy as np; e.step(np.zeros(12, dtype=np.float32)); e.close()"
```
Expected: `Box(-1.0, 1.0, (12,), float32)`, one zero-action step runs without error (after asset download).

---

## §3 Reset (_initialize_episode)

### Description
On reset every selected env is snapped to the Anymal-C `standing` keyframe — base pose `p=[0,0,0.545]` (identity orientation) and the standing leg qpos. No domain randomization of the initial state; deterministic rest pose.

### Decisions resolved
- Set base pose = `keyframe.pose` (`sapien.Pose(p=[0,0,0.545])`).
- Set qpos = `keyframe.qpos` = `[0.03,-0.03,0.03,-0.03, 0.4,0.4,-0.4,-0.4, -0.8,-0.8,0.8,0.8]`.
- No per-env randomized ranges (no position/velocity/pose jitter).

### Code
```python
def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
    with torch.device(self.device):
        len(env_idx)
        keyframe = self.agent.keyframes["standing"]
        self.agent.robot.set_pose(keyframe.pose)
        self.agent.robot.set_qpos(keyframe.qpos)
```

### Smoke (§3)
```bash
.venv/bin/python -c "import gymnasium as gym, mani_skill, torch; e=gym.make('AnymalC-Spin-v1'); e.reset(seed=0); print(e.unwrapped.agent.robot.get_qpos()[0]); e.close()"
```
Expected: qpos row equals the standing keyframe `[0.03,-0.03,0.03,-0.03,0.4,0.4,-0.4,-0.4,-0.8,-0.8,0.8,0.8]` (after asset download).

---

## §4 Goal + Termination

### Description
Pure time-out task with a fall failure. `evaluate()` returns `fail` / `is_fallen` flags computed from base-link contact forces (`is_fallen` = body contacts ground). There is no `success` condition — the goal is to maximize spin rate, not reach a discrete target. Episodes end after `max_episode_steps=200` (truncation) or are marked failed when the body falls.

### Decisions resolved
- `max_episode_steps = 200`.
- `fail = is_fallen = agent.is_fallen()` (base net contact force norm > 1).
- No `success` term, no `CommandsCfg`.
- Fail drives the §6 terminal penalty (`reward[info["fail"]] = -100`).

### Code
```python
def evaluate(self):
    is_fallen = self.agent.is_fallen()
    return {
        "fail": is_fallen,
        "is_fallen": is_fallen,
    }

# agent.is_fallen():
#   forces = self.robot.get_net_contact_forces(["base"])
#   return torch.norm(forces, dim=-1).max(-1).values > 1
```

### Smoke (§4)
```bash
.venv/bin/python -c "import gymnasium as gym, mani_skill; e=gym.make('AnymalC-Spin-v1'); print(e.spec.max_episode_steps); e.reset(); _,_,term,trunc,info=e.step(e.action_space.sample()); print('fail' in info, term.shape if hasattr(term,'shape') else term); e.close()"
```
Expected: `200`, `info` contains `fail`/`is_fallen` flags (after asset download).

---

## §5 Observation

### Description
State observation = proprioception (qpos + qvel of the 12 leg joints) plus task-extra base velocities. The `pd_joint_delta_pos` controller carries no extra controller state, so proprioception is just qpos(12) + qvel(12). `_get_obs_extra` adds root linear velocity (3) + root angular velocity (3). With `obs_mode="state"` (default; the first of `SUPPORTED_OBS_MODES`) everything is flattened/concatenated → 30-D.

### Decisions resolved
- `obs_mode` default = `"state"` (also: `state_dict`, `none`, `sensor_data`, `any_textures`, `pointcloud`).
- Proprioception (`_get_obs_agent` → `agent.get_proprioception`): `qpos` (12) + `qvel` (12) = 24.
- Extra (`_get_obs_extra`): `root_linear_velocity` (3) + `root_angular_velocity` (3) = 6.
- **Total state obs dim = 30.**
- No observation noise / corruption configured at the task level.

### Code
```python
def _get_obs_extra(self, info: dict):
    obs = dict(
        root_linear_velocity=self.agent.robot.root_linear_velocity,
        root_angular_velocity=self.agent.robot.root_angular_velocity,
    )
    return obs

# base BaseEnv:
#   _get_obs_state_dict -> dict(agent=_get_obs_agent(), extra=_get_obs_extra(info))
#   _get_obs_agent      -> agent.get_proprioception()  # dict(qpos=..., qvel=...)
```

### Smoke (§5)
```bash
.venv/bin/python -c "import gymnasium as gym, mani_skill; e=gym.make('AnymalC-Spin-v1', obs_mode='state'); o,_=e.reset(); print(o.shape); e.close()"
```
Expected: `(num_envs, 30)` (after asset download). Probe-time analytic value: 30.

---

## §6 Reward (fully reproducible)

### Description
Dense reward = `2 * yaw_rate - penalties`, where `yaw_rate` is the base angular velocity about z (the spin we want to maximize). Penalties discourage instability and wasted effort: vertical bounce (`lin_vel_z²`), off-axis tumbling (`ang_vel_xy²`), knee-to-ground contacts (undesired-contact indicator on the four KFE links), and deviation of the leg joints from the standing pose. Falling over zeroes everything and applies a flat `-100`. The normalized variant divides the dense reward by `max_reward = 2.0`.

### Composer
**Sum** — `reward = 2*rotation_reward + (sum of negatively-weighted penalty terms)`, with a terminal override `reward[fail] = -100`.

### Decisions resolved (per-term saturated per-step magnitudes — retro-computed from weights)
- `rotation_reward = root_angular_velocity[:, 2]` (yaw rate, rad/s) × weight **+2** → dominant positive term; a fast ~5 rad/s spin ⇒ ≈ +10/step.
- `lin_vel_z_l2 = root_linear_velocity[:, 2]²` × **-2** (vertical bounce penalty).
- `ang_vel_xy_l2 = sum(root_angular_velocity[:, :2]²)` × **-0.05** (roll/pitch rate penalty).
- undesired-contact indicator on `["LF_KFE","RF_KFE","LH_KFE","RH_KFE"]` (force-norm > 1.0) × **-1** (per knee-contact event).
- `||qpos - default_qpos||₂` × **-0.05** (posture / standing-pose deviation).
- Terminal: `reward[info["fail"]] = -100`.
- Normalized: `dense / max_reward` with `max_reward = 2.0`.

### Code (verbatim)
```python
def _compute_undesired_contacts(self, threshold=1.0):
    forces = self.agent.robot.get_net_contact_forces(
        self._UNDESIRED_CONTACT_LINK_NAMES
    )
    contact_exists = torch.norm(forces, dim=-1).max(-1).values > threshold
    return contact_exists

def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: dict):
    rotation_reward = self.agent.robot.root_angular_velocity[:, 2]
    # various penalties:
    lin_vel_z_l2 = torch.square(self.agent.robot.root_linear_velocity[:, 2])
    ang_vel_xy_l2 = (
        torch.square(self.agent.robot.root_angular_velocity[:, :2])
    ).sum(axis=1)
    penalties = (
        lin_vel_z_l2 * -2
        + ang_vel_xy_l2 * -0.05
        + self._compute_undesired_contacts() * -1
        + torch.linalg.norm(self.agent.robot.qpos - self.default_qpos, axis=1)
        * -0.05
    )
    reward = 2 * rotation_reward + penalties
    reward[info["fail"]] = -100
    return reward

def compute_normalized_dense_reward(
    self, obs: Any, action: torch.Tensor, info: dict
):
    max_reward = 2.0
    return self.compute_dense_reward(obs=obs, action=action, info=info) / max_reward
```

`self._UNDESIRED_CONTACT_LINK_NAMES = ["LF_KFE", "RF_KFE", "LH_KFE", "RH_KFE"]` (set on `AnymalCSpinEnv`).
`self.default_qpos = torch.from_numpy(ANYmalC.keyframes["standing"].qpos).to(self.device)` (set in `AnymalCSpinEnv.__init__`).

### Smoke (§6)
```bash
.venv/bin/python -c "import gymnasium as gym, mani_skill, torch; e=gym.make('AnymalC-Spin-v1', reward_mode='dense'); e.reset(); _,r,_,_,_=e.step(e.action_space.sample()); print(torch.isfinite(torch.as_tensor(r)).all()); e.close()"
```
Expected: `True` (reward finite); composer = sum (after asset download).

---

## §7 DR (domain randomization)

`<no DR>` — `_initialize_episode` snaps every env deterministically to the `standing` keyframe with no randomized ranges, and there are no `startup`/`interval` randomization hooks. (ManiSkill quadruped-spin has no `EventCfg`-style DR; the docstring's "Randomizations: Robot is initialized in a stable rest/standing position" is a fixed reset, not randomization.)

---

