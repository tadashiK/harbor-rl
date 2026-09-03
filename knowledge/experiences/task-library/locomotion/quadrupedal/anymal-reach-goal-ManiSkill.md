# AnymalC-Reach-v1 — Implementation Spec

- robot: ANYbotics ANYmal-C quadruped (12 DoF)
- simulator: ManiSkill (SAPIEN, `BaseEnv` + `@register_env`)
- objects: none (flat ground; commanded goal position)
- bimanual: false
- summary: Walk to a commanded goal position on flat ground.

> Task type: **goal-directed quadruped locomotion**. The Anymal-C quadruped must walk to a target sphere placed ~2.5 m in front of it (within ±0.5 m fwd, ±1 m lateral) and stop within 0.35 m of it without falling over. This is NOT velocity-command tracking — there is no commanded base velocity; the agent is rewarded purely on shrinking distance-to-goal plus stability penalties. The same env class (`QuadrupedReachEnv`) backs both `AnymalC-Reach-v1` (anymal_c) and `UnitreeGo2-Reach-v1` (go2); this spec documents the AnymalC subclass.

---

## §1 Registration + Scene

**Description.** `@register_env("AnymalC-Reach-v1", max_episode_steps=200)` registers a subclass `AnymalCReachEnv` of the shared `QuadrupedReachEnv(BaseEnv)`. `SUPPORTED_ROBOTS = ["anymal_c", "unitree_go2_simplified_locomotion"]`; the AnymalC subclass defaults `robot_uids="anymal_c"`. The agent is loaded 1 m above the origin (`sapien.Pose(p=[0,0,1])`) so it settles onto the ground. The scene is a 400 m-wide flat ground plus a single green kinematic sphere (radius 0.2 m, no collision) marking the goal. Sim uses a reduced PhysX solver (4 position iters, 0 velocity iters) and a 2^20 max-rigid-contact budget.

**Decisions resolved.**
- `max_episode_steps = 200`
- `SUPPORTED_ROBOTS = ["anymal_c", "unitree_go2_simplified_locomotion"]`
- AnymalC subclass: `robot_uids="anymal_c"`; `default_qpos = ANYmalC.keyframes["standing"].qpos`
- `_UNDESIRED_CONTACT_LINK_NAMES = ["LF_KFE", "RF_KFE", "LH_KFE", "RH_KFE"]` (knee links — penalized on contact)
- Agent spawn pose (in `_load_agent`): `sapien.Pose(p=[0, 0, 1])`
- Ground: `build_ground(self.scene, floor_width=400)`
- Goal marker: `actors.build_sphere(radius=0.2, color=[0,1,0,1], name="goal", add_collision=False, body_type="kinematic")`
- Sim: `solver_position_iterations=4, solver_velocity_iterations=0`, `GPUMemoryConfig(max_rigid_contact_count=2**20)`
- Robot (ANYmalC agent, `uid="anymal_c"`): `urdf_path = f"{ASSET_DIR}/robots/anymal_c/urdf/anymal.urdf"`; `fix_root_link=False`; `disable_self_collisions=True`; gravity disabled on all links except root (`_after_init`). Foot material: static/dynamic friction 2.0, restitution 0.0, patch_radius 0.1 on `{LF,LH,RF,RH}_FOOT`.
- 12 leg DoF (`joint_names`): `[LF_HAA, RF_HAA, LH_HAA, RH_HAA, LF_HFE, RF_HFE, LH_HFE, RH_HFE, LF_KFE, RF_KFE, LH_KFE, RH_KFE]`
- Standing keyframe: `pose=sapien.Pose(p=[0,0,0.545])`, `qpos=[0.03,-0.03,0.03,-0.03, 0.4,0.4,-0.4,-0.4, -0.8,-0.8,0.8,0.8]`
- Sensors: `base_camera` 128×128 fov π/2 mounted on `links[0]`; human-render `render_camera` 512×512 fov 1, eye `[-2,1.5,3]` → target `[1.5,0,0.5]`.

**Code.**
```python
@register_env("AnymalC-Reach-v1", max_episode_steps=200)
class AnymalCReachEnv(QuadrupedReachEnv):
    _sample_video_link = "https://github.com/haosulab/ManiSkill/raw/main/figures/environment_demos/AnymalC-Reach-v1_rt.mp4"
    _UNDESIRED_CONTACT_LINK_NAMES = ["LF_KFE", "RF_KFE", "LH_KFE", "RH_KFE"]

    def __init__(self, *args, robot_uids="anymal_c", **kwargs):
        super().__init__(*args, robot_uids=robot_uids, **kwargs)
        self.default_qpos = torch.from_numpy(ANYmalC.keyframes["standing"].qpos).to(
            self.device
        )


class QuadrupedReachEnv(BaseEnv):
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
        pose = sapien_utils.look_at(eye=[0.5, 0, 0.1], target=[1.0, 0, 0.0])
        return [
            CameraConfig(
                "base_camera", pose=pose, width=128, height=128,
                fov=np.pi / 2, near=0.01, far=100,
                mount=self.agent.robot.links[0],
            )
        ]

    @property
    def _default_human_render_camera_configs(self):
        pose = sapien_utils.look_at([-2.0, 1.5, 3], [1.5, 0.0, 0.5])
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
        self.goal = actors.build_sphere(
            self.scene, radius=0.2, color=[0, 1, 0, 1], name="goal",
            add_collision=False, body_type="kinematic",
        )
```

ANYmalC agent definition (verbatim, `mani_skill/agents/robots/anymal/anymal_c.py`):
```python
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
        # disable gravity / compensate gravity automatically in all links but the root one
        for link in self.robot.links[1:]:
            link.disable_gravity = True
```

**Smoke (§1 build).** not captured (verified via source read).
```bash
cd <repo>
.venv/bin/python -c "import gymnasium as gym, mani_skill; e=gym.make('AnymalC-Reach-v1'); print(e.observation_space, e.action_space); e.close()"
# Expected: Box(-inf, inf, (35,), float32) Box(-1.0, 1.0, (12,), float32)
# (Requires the `anymal_c` robot asset; run `python -m mani_skill.utils.download_asset anymal_c` first.)
```

---

## §2 Actions

**Description.** Control is per-leg-joint PD position targeting on the 12 leg DoF. The agent's default control mode is the first key of `_controller_configs`, i.e. `pd_joint_delta_pos`: a normalized delta-position controller. Each of the 12 action dims is in `[-1, 1]` (normalize_action=True) and maps to a joint target delta in `[-0.225, 0.225]` rad added to the current joint position, tracked by a PD controller (stiffness 80, damping 2, force limit 100). A non-delta absolute `pd_joint_pos` mode is also available but is not the default.

**Decisions resolved.**
- `control_mode` (default) = `pd_joint_delta_pos`
- Action space: `Box(-1, 1, (12,), float32)` — one normalized delta per leg joint
- Delta clip range: `[-0.225, 0.225]` rad; `use_delta=True`, `normalize_action=True`
- PD gains: `stiffness=80.0`, `damping=2.0`, `force_limit=100`
- `balance_passive_force=False` for both modes
- Alt mode `pd_joint_pos`: absolute joint targets, `normalize_action=False`, no clip (`None, None`)

**Code.**
```python
@property
def _controller_configs(self):
    self.stiffness = 80.0
    self.damping = 2.0
    self.force_limit = 100
    # delta action scale for Omni Isaac Gym Envs is self.dt * self.action_scale = 1/60 * 13.5.
    # NOTE that their self.dt value is not the same as the actual DT used in sim...., they use default of 1/100
    pd_joint_delta_pos = PDJointPosControllerConfig(
        self.joint_names,
        -0.225, 0.225,
        self.stiffness, self.damping, self.force_limit,
        normalize_action=True, use_delta=True,
    )
    pd_joint_pos = PDJointPosControllerConfig(
        self.joint_names,
        None, None,
        self.stiffness, self.damping, self.force_limit,
        normalize_action=False, use_delta=False,
    )
    controller_configs = dict(
        pd_joint_delta_pos=dict(body=pd_joint_delta_pos, balance_passive_force=False),
        pd_joint_pos=dict(body=pd_joint_pos, balance_passive_force=False),
    )
    return controller_configs
```

**Smoke (§2).** not captured (verified via source read).
```bash
.venv/bin/python -c "import gymnasium as gym, mani_skill; e=gym.make('AnymalC-Reach-v1'); o,_=e.reset(seed=0); a=e.action_space.sample(); o,r,te,tr,i=e.step(a); print(a.shape, r.shape); e.close()"
# Expected: (12,) (1,)
```

---

## §3 Reset (`_initialize_episode`)

**Description.** On reset, the robot is hard-set to the `standing` keyframe pose+qpos (no joint/pose noise). The goal sphere is randomized in the XY plane: forward x ∈ 2.5 ± 0.5 m (uniform), lateral y ∈ ±1 m (uniform), z = 0. The goal is set kinematically.

**Decisions resolved.**
- Robot pose/qpos = `agent.keyframes["standing"]` (no randomization on the robot)
- Goal x: `rand()*1 - 0.5 + 2.5` → uniform in **[2.0, 3.0]** m forward
- Goal y: `rand()*2 - 1` → uniform in **[-1.0, 1.0]** m lateral
- Goal z: 0
- Per-env batched via `b = len(env_idx)`

**Code.**
```python
def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
    with torch.device(self.device):
        b = len(env_idx)
        keyframe = self.agent.keyframes["standing"]
        self.agent.robot.set_pose(keyframe.pose)
        self.agent.robot.set_qpos(keyframe.qpos)
        # sample random goal
        xyz = torch.zeros((b, 3))
        xyz[:, 0] = 2.5
        noise_scale = 1
        xyz[:, 0] = torch.rand(size=(b,)) * noise_scale - noise_scale / 2 + 2.5
        noise_scale = 2
        xyz[:, 1] = torch.rand(size=(b,)) * noise_scale - noise_scale / 2
        self.goal.set_pose(Pose.create_from_pq(xyz))
```

**Smoke (§3).** not captured (verified via source read). Reset N envs, assert goal x ∈ [2.0,3.0], y ∈ [-1,1].

---

## §4 Goal + Termination (`evaluate` + `max_episode_steps`)

**Description.** `evaluate()` computes the planar (XY) distance from the robot base to the goal. **Success** = within 0.35 m of the goal AND not fallen. **Fail** = the robot has fallen (its `base` link net contact force exceeds 1 N, via `agent.is_fallen()`). The episode times out at `max_episode_steps=200`. There is no `CommandsCfg` (ManiSkill encodes the goal via the scene sphere + obs, not a command manager).

**Decisions resolved.**
- Success radius: `robot_to_goal_dist < 0.35` m (planar)
- Success requires `~is_fallen`
- Fail / early termination: `is_fallen` — `norm(net_contact_force(["base"])) > 1`
- `max_episode_steps = 200`
- Returned info keys: `success`, `fail`, `robot_to_goal_dist`, `reached_goal`, `is_fallen`

**Code.**
```python
def evaluate(self):
    is_fallen = self.agent.is_fallen()
    robot_to_goal_dist = torch.linalg.norm(
        self.goal.pose.p[:, :2] - self.agent.robot.pose.p[:, :2], axis=1
    )
    reached_goal = robot_to_goal_dist < 0.35
    return {
        "success": reached_goal & ~is_fallen,
        "fail": is_fallen,
        "robot_to_goal_dist": robot_to_goal_dist,
        "reached_goal": reached_goal,
        "is_fallen": is_fallen,
    }

# ANYmalC.is_fallen (mani_skill/agents/robots/anymal/anymal_c.py):
def is_fallen(self):
    """This quadruped is considered fallen if its body contacts the ground"""
    forces = self.robot.get_net_contact_forces(["base"])
    return torch.norm(forces, dim=-1).max(-1).values > 1
```

**Smoke (§4).** not captured (verified via source read). Step to time_out; assert `info["success"]/["fail"]` are bool tensors and `robot_to_goal_dist` is finite.

---

## §5 Observation (`_get_obs_extra` + obs modes + dim)

**Description.** Default `obs_mode="state"` (first of `SUPPORTED_OBS_MODES`), so `use_state=True`. The flattened state obs = agent proprioception (`qpos` 12 + `qvel` 12) ++ task extras. Task extras (`_get_obs_extra`) always include base root linear velocity (3), root angular velocity (3), and the success flag (1); when `use_state` is on (state / state_dict modes) it additionally exposes the 2D goal position and the 2D robot→goal vector. **Note:** unlike a typical IsaacLab locomotion obs, this env does NOT include a projected-gravity term or joint-target history — its base orientation signal comes purely from root angular velocity. There is no per-term observation noise (no corruption wrapper).

**Decisions resolved (obs_mode="state", default).**
- agent proprioception: `qpos` (12) + `qvel` (12) = **24**
- extras (always): `root_linear_velocity` (3) + `root_angular_velocity` (3) + `reached_goal` (1) = **7**
- extras (use_state only): `goal_pos` (2) + `robot_to_goal` (2) = **4**
- **Total state obs dim = 24 + 7 + 4 = 35**
- `SUPPORTED_OBS_MODES = ("state", "state_dict", "none", "sensor_data", "any_textures", "pointcloud")`; default `state`.
- In `obs_mode="state_dict"`: same content, nested `{agent:{qpos,qvel}, extra:{...}}`.
- In `obs_mode="none"`: empty.
- No `ObsTerm` noise / `enable_corruption` (ManiSkill state obs is ground-truth).

**Code.**
```python
def _get_obs_extra(self, info: dict):
    obs = dict(
        root_linear_velocity=self.agent.robot.root_linear_velocity,
        root_angular_velocity=self.agent.robot.root_angular_velocity,
        reached_goal=info["success"],
    )
    if self.obs_mode_struct.use_state:
        obs.update(
            goal_pos=self.goal.pose.p[:, :2],
            robot_to_goal=self.goal.pose.p[:, :2] - self.agent.robot.pose.p[:, :2],
        )
    return obs

# BaseEnv proprioception (mani_skill/envs/sapien_env.py / agents/base_agent.py):
def _get_obs_state_dict(self, info: dict):
    return dict(agent=self._get_obs_agent(), extra=self._get_obs_extra(info))

def _get_obs_agent(self):
    return self.agent.get_proprioception()  # -> dict(qpos=..., qvel=...)
```

**Smoke (§5).** not captured (verified via source read).
```bash
.venv/bin/python -c "import gymnasium as gym, mani_skill; e=gym.make('AnymalC-Reach-v1'); o,_=e.reset(seed=0); print(o.shape); e.close()"
# Expected: torch.Size([1, 35])
```

---

## §6 Reward (`compute_dense_reward` + `compute_normalized_dense_reward`)

**Description.** Single dense reward (no `RewardsCfg`/`RewTerm` manager — ManiSkill rewards are a plain method). Composer = **sum** of: a constant `+1` alive bonus, a `2×` weighted reaching term (`1 - tanh(dist)`, → 1 as the robot nears the goal), and a set of stability penalties. Penalties: vertical base velocity² (×−2), planar base angular velocity² summed (×−0.05), undesired knee-link ground contact (×−1 per env where any of the 4 KFE links exceeds 1 N), and joint deviation from the standing pose (L2 norm of `qpos − default_qpos`, ×−0.05). On `fail` (fallen) the whole reward is zeroed. The normalized variant divides by `max_reward = 3.0` (1 alive + 2×1 reaching max).

**Composer:** **sum** (`reward = 1 + 2 * reaching_reward + penalties`).

**Planning-budget (retro-computed, per-step saturated magnitudes):**
- alive bonus: constant **+1.0**
- reaching term: `2 * (1 - tanh(dist))` → **0** at far start (dist≈2.5, tanh≈0.987 → ~0.026) up to **+2.0** when on top of goal. So nominal max return per step ≈ **3.0** (= `max_reward`).
- `lin_vel_z_l2 * -2`: small near a stable gait (vz≈0); a 0.3 m/s vertical bounce → −0.18.
- `ang_vel_xy_l2 * -0.05`: roll/pitch rate penalty; 1 rad/s each → −0.10.
- undesired knee contact: **−1.0** whenever any KFE link touches ground (binary per env).
- joint-deviation `-0.05 * ||qpos − default_qpos||`: at the standing pose 0; a crouch of ~1 rad summed L2 → ~−0.05·(few tenths).
- On `fail`: reward forced to **0** (no negative — failure is just zero return, distinct from a penalty).

**Code (verbatim, full).**
```python
def _compute_undesired_contacts(self, threshold=1.0):
    forces = self.agent.robot.get_net_contact_forces(
        self._UNDESIRED_CONTACT_LINK_NAMES
    )
    contact_exists = torch.norm(forces, dim=-1).max(-1).values > threshold
    return contact_exists

def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: dict):
    robot_to_goal_dist = info["robot_to_goal_dist"]
    reaching_reward = 1 - torch.tanh(1 * robot_to_goal_dist)

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
    reward = 1 + 2 * reaching_reward + penalties
    reward[info["fail"]] = 0
    return reward

def compute_normalized_dense_reward(self, obs: Any, action: torch.Tensor, info: dict):
    max_reward = 3.0
    return self.compute_dense_reward(obs=obs, action=action, info=info) / max_reward
```

**Smoke (§6).** not captured (verified via source read).
```bash
.venv/bin/python -c "import gymnasium as gym, mani_skill; e=gym.make('AnymalC-Reach-v1', reward_mode='normalized_dense'); o,_=e.reset(seed=0); o,r,te,tr,i=e.step(e.action_space.sample()); print(r, bool((r<=1.0).all())); e.close()"
# Expected: finite reward tensor, all <= 1.0 (normalized by max_reward=3.0)
```

---

## §7 DR

`<no DR>` — there is no `startup`/`interval` domain-randomization. The only per-reset variation is the goal XY sampling in `_initialize_episode` (§3); the robot always starts from the fixed `standing` keyframe with no added pose/joint/mass/friction noise. Foot friction (2.0) and PD gains are fixed constants in the agent cfg.

---

