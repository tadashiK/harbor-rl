# UnitreeG1Stand-v1 — Implementation Spec

- robot: Unitree G1 humanoid, simplified-legs URDF (37 DoF, floating base)
- simulator: ManiSkill (SAPIEN, `BaseEnv` + `@register_env`)
- objects: none (flat ground plane)
- bimanual: false
- summary: Stand upright and stay balanced on a flat plane.

ManiSkill differs structurally from IsaacLab manager-based tasks: there is no `RewardsCfg`/`ObservationsCfg`/`EventCfg`/`TerminationsCfg`. Instead the task is a single `BaseEnv` subclass whose hooks (`_load_scene`, `_initialize_episode`, `evaluate`, `_get_obs_extra`, `compute_*_reward`) and the chosen robot agent (`UnitreeG1Simplified`) collectively define §1..§7. Sections below map those hooks onto the §1..§7 schema.

---

## §1 Registration + Scene

### Description
`UnitreeG1Stand-v1` is registered via `@register_env` on `UnitreeG1StandEnv`, a subclass of the shared `HumanoidStandEnv`. The scene is just a flat ground plane (no objects, no table). The robot is the Unitree G1 "simplified legs" embodiment — a full humanoid articulation (legs + torso + arms + hands) loaded from a URDF, 37 actuated DoF total. `fix_root_link=False` (free-floating base, so gravity/balance is in play). Sim cfg bumps GPU contact buffers for the high-contact humanoid.

### Decisions resolved
- `@register_env("UnitreeG1Stand-v1", max_episode_steps=1000)`
- `SUPPORTED_ROBOTS = ["unitree_g1_simplified_legs"]`
- default `robot_uids="unitree_g1_simplified_legs"`
- robot agent class: `UnitreeG1Simplified` (uid `unitree_g1_simplified_legs`), subclass of `UnitreeG1`
- `urdf_path = f"{PACKAGE_ASSET_DIR}/robots/g1_humanoid/g1_simplified_legs.urdf"` → resolved on disk: `mani_skill/assets/robots/g1_humanoid/g1_simplified_legs.urdf` (EXISTS, 36723 bytes)
- `fix_root_link = False`, `load_multiple_collisions = True`
- robot total DoF = **37** (verified: `agent.robot.dof == tensor([37])`)
- scene: `build_ground(self.scene)` only — no objects
- `_default_sim_config`: `SimConfig(gpu_memory_config=GPUMemoryConfig(max_rigid_contact_count=2**22, max_rigid_patch_count=2**21))`
- `_default_sensor_configs = []` (no obs cameras)
- human render camera: `look_at([1.0, 1.0, 2.0], [0.0, 0.0, 0.75])`, `CameraConfig("render_camera", pose, 512, 512, 1, 0.01, 100)`
- robot init pose at spawn: `sapien.Pose(p=[0, 0, 0.755])` (G1 standing base height; note H1 sibling uses 0.975)
- actuators (all body joints): `PDJointPosControllerConfig(body_joints, stiffness=50, damping=1, force_limit=100, normalize_action=False)` — see §2

### Code

Registration + env class (`mani_skill/envs/tasks/humanoid/humanoid_stand.py`):
```python
class HumanoidStandEnv(BaseEnv):
    SUPPORTED_REWARD_MODES = ["sparse", "none"]

    def __init__(
        self,
        *args,
        robot_uids="unitree_h1_simplified",
        robot_init_qpos_noise=0.02,
        **kwargs
    ):
        self.robot_init_qpos_noise = robot_init_qpos_noise
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    @property
    def _default_sensor_configs(self):
        return []

    @property
    def _default_human_render_camera_configs(self):
        pose = sapien_utils.look_at([1.0, 1.0, 2.5], [0.0, 0.0, 0.75])
        return CameraConfig("render_camera", pose, 512, 512, 1, 0.01, 100)

    def _load_scene(self, options: dict):
        build_ground(self.scene)


@register_env("UnitreeG1Stand-v1", max_episode_steps=1000)
class UnitreeG1StandEnv(HumanoidStandEnv):
    SUPPORTED_ROBOTS = ["unitree_g1_simplified_legs"]
    agent: Union[UnitreeG1Simplified]

    def __init__(self, *args, robot_uids="unitree_g1_simplified_legs", **kwargs):
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    @property
    def _default_sim_config(self):
        return SimConfig(
            gpu_memory_config=GPUMemoryConfig(
                max_rigid_contact_count=2**22, max_rigid_patch_count=2**21
            )
        )

    @property
    def _default_human_render_camera_configs(self):
        pose = sapien_utils.look_at([1.0, 1.0, 2.0], [0.0, 0.0, 0.75])
        return CameraConfig("render_camera", pose, 512, 512, 1, 0.01, 100)
```

Robot agent (`mani_skill/agents/robots/unitree_g1/g1.py`):
```python
@register_agent()
class UnitreeG1(BaseAgent):
    uid = "unitree_g1"
    urdf_path = f"{PACKAGE_ASSET_DIR}/robots/g1_humanoid/g1.urdf"
    urdf_config = dict()
    fix_root_link = False
    load_multiple_collisions = True

    keyframes = dict(
        standing=Keyframe(
            pose=sapien.Pose(p=[0, 0, 0.755]),
            qpos=np.array(
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.2, -0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.9, 0.9, 0.0, 0.0, 0.0, 0.0, 0.0, -0.77, -0.77, 0.0, 0.77, 0.77, 0.1, -0.92, -0.92, -0.1, 0.92, 0.92, 0.92, -0.92]
            ),
        ),
        # ... right_knee_up / left_knee_up keyframes also defined (unused by Stand task)
    )

    body_joints = [
        "left_hip_pitch_joint", "right_hip_pitch_joint", "torso_joint",
        "left_hip_roll_joint", "right_hip_roll_joint",
        "left_shoulder_pitch_joint", "right_shoulder_pitch_joint",
        "left_hip_yaw_joint", "right_hip_yaw_joint",
        "left_shoulder_roll_joint", "right_shoulder_roll_joint",
        "left_knee_joint", "right_knee_joint",
        "left_shoulder_yaw_joint", "right_shoulder_yaw_joint",
        "left_ankle_pitch_joint", "right_ankle_pitch_joint",
        "left_elbow_pitch_joint", "right_elbow_pitch_joint",
        "left_ankle_roll_joint", "right_ankle_roll_joint",
        "left_elbow_roll_joint", "right_elbow_roll_joint",
        "left_zero_joint", "left_three_joint", "left_five_joint",
        "right_zero_joint", "right_three_joint", "right_five_joint",
        "left_one_joint", "left_four_joint", "left_six_joint",
        "right_one_joint", "right_four_joint", "right_six_joint",
        "left_two_joint", "right_two_joint",
    ]   # 37 joints
    body_stiffness = 50
    body_damping = 1
    body_force_limit = 100


@register_agent()
class UnitreeG1Simplified(UnitreeG1):
    uid = "unitree_g1_simplified_legs"
    urdf_path = f"{PACKAGE_ASSET_DIR}/robots/g1_humanoid/g1_simplified_legs.urdf"
```

### Smoke
```bash
cd <repo>
.venv/bin/python -c "import gymnasium as gym, mani_skill; e=gym.make('UnitreeG1Stand-v1'); print(e.observation_space, e.action_space); print('DOF', e.unwrapped.agent.robot.dof); e.close()"
```
Expected stdout (literal, from passing run):
```
OBS Box(-inf, inf, (1, 74), float32)
ACT Box([...37 lower bounds...], [...37 upper bounds...], (37,), float32)
DOF tensor([37])
```

---

## §2 Actions

### Description
Single control mode `pd_joint_pos`: a PD joint-position controller over all 37 `body_joints`, absolute target positions (not delta), un-normalized (raw radians clamped to each joint's URDF limits). A `pd_joint_delta_pos` mode (±0.2 rad delta, normalized) is also defined on the agent but `pd_joint_pos` is the env default. Stiffness 50 / damping 1 / force-limit 100 across all joints. `balance_passive_force=False` so gravity acts on the robot (mandatory for a balance task).

### Decisions resolved
- control modes available: `pd_joint_pos` (default), `pd_joint_delta_pos`
- action dim = **37** (one per body joint)
- `pd_joint_pos`: `lower=None, upper=None` (uses URDF limits), `normalize_action=False`, `stiffness=50`, `damping=1`, `force_limit=100`, `balance_passive_force=False`
- `pd_joint_delta_pos`: `lower=-0.2, upper=0.2`, `use_delta=True`, same stiffness/damping/force_limit, `balance_passive_force=False`
- resolved action_space bounds (URDF joint limits) — full vector captured in §5 build output

### Code (`g1.py`)
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
        lower=-0.2,
        upper=0.2,
        stiffness=self.body_stiffness,
        damping=self.body_damping,
        force_limit=self.body_force_limit,
        use_delta=True,
    )
    # balance_passive_force=False otherwise gravity is disabled for the robot itself
    return dict(
        pd_joint_pos=dict(body=body_pd_joint_pos, balance_passive_force=False),
        pd_joint_delta_pos=dict(
            body=body_pd_joint_delta_pos, balance_passive_force=False
        ),
    )
```

### Smoke
`env.action_space.shape == (37,)`; `env.step(env.action_space.sample())` returns finite obs.

---

## §3 Reset / Initialize Episode

### Description
On reset, the robot is placed at the `standing` keyframe qpos plus small Gaussian noise (std 0.05 rad per joint), and the base is teleported to `p=[0,0,0.755]`. No object randomization (no objects). The G1 override resets qpos/pose directly (it does NOT use `robot_init_qpos_noise=0.02` — that base ctor param is overridden by the hard-coded 0.05 noise scale in `_initialize_episode`).

### Decisions resolved
- `random_qpos = randn(b, 37) * 0.05 + standing_keyframe.qpos`
- `set_qpos(random_qpos)`
- `set_pose(sapien.Pose(p=[0, 0, 0.755]))`
- standing qpos = the 37-vector in §1 keyframes
- the base `HumanoidStandEnv._initialize_episode` is a `pass`; the G1 subclass overrides it

### Code (`humanoid_stand.py`, `UnitreeG1StandEnv`)
```python
def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
    with torch.device(self.device):
        b = len(env_idx)
        standing_keyframe = self.agent.keyframes["standing"]
        random_qpos = (
            torch.randn(size=(b, self.agent.robot.dof[0]), dtype=torch.float) * 0.05
        )
        random_qpos += common.to_tensor(standing_keyframe.qpos, device=self.device)
        self.agent.robot.set_qpos(random_qpos)
        self.agent.robot.set_pose(sapien.Pose(p=[0, 0, 0.755]))
```

### Smoke
`env.reset(seed=0)` twice with same seed → identical initial obs; different seeds → divergent obs (Gaussian noise).

---

## §4 Goal + Termination

### Description
Goal: keep the humanoid torso/base upright within a height window (0.5 m < base z < 1.0 m). Success/standing is a per-step boolean; failure (`fail`) is its negation. There is no early `success` termination term — episodes run to the time limit `max_episode_steps=1000`. `is_standing` is a simple base-height-window heuristic; `is_fallen` (base z < 0.3) is computed but its result is discarded (called for side-effect/parity with H1).

### Decisions resolved
- `max_episode_steps = 1000` (set in `@register_env`)
- success criterion `is_standing`: `0.5 < base.z < 1.0`
- `fail = ~is_standing`
- no early termination on success; episodes terminate only via time-out at 1000 steps (or via the env's fail handling if configured downstream)
- `is_fallen`: `base.z < 0.3` (defined on agent, called in `evaluate` but return value unused)

### Code
`evaluate` (`humanoid_stand.py`, base `HumanoidStandEnv`):
```python
def evaluate(self):
    is_standing = self.agent.is_standing()
    self.agent.is_fallen()
    return {"is_standing": is_standing, "fail": ~is_standing}
```
Heuristics (`g1.py`, `UnitreeG1`):
```python
def is_standing(self):
    """Checks if G1 is standing by checking if the torso/base is within a height window"""
    return (self.robot.pose.p[:, 2] > 0.5) & (self.robot.pose.p[:, 2] < 1.0)

def is_fallen(self):
    """Checks if G1 has fallen (base too low)"""
    return self.robot.pose.p[:, 2] < 0.3
```

### Smoke
`info["is_standing"]` and `info["fail"]` present in `env.step(...)` info dict, both boolean tensors of shape `(num_envs,)`.

---

## §5 Observation

### Description
State-based proprioception only. `_get_obs_extra` returns an empty dict (no task-specific extras — no goal pose, no object state). The full obs is the agent proprioception: robot qpos (37) + qvel (37) = 74, flattened. No sensor/camera obs by default (`_default_sensor_configs = []`).

### Decisions resolved
- obs modes: default `state` (also `state_dict`; image modes possible but no cameras configured so they add nothing)
- `_get_obs_extra(info) -> dict()` (empty)
- `_get_obs_agent()` → `agent.get_proprioception()` → `dict(qpos=robot.get_qpos(), qvel=robot.get_qvel())`
- resolved total obs dim = **74** = 37 qpos + 37 qvel
- obs shape `(num_envs, 74)`; canonical single-env build → `Box(-inf, inf, (1, 74), float32)`
- no observation noise / corruption configured

### Code
Task hook (`humanoid_stand.py`, base):
```python
def _get_obs_extra(self, info: dict):
    return dict()
```
Base proprioception (`mani_skill/agents/base_agent.py`):
```python
def get_proprioception(self):
    obs = dict(qpos=self.robot.get_qpos(), qvel=self.robot.get_qvel())
    # + any controller state (PDJointPos has none)
    return obs
```

### Canonical build (literal stdout)
```
OBS Box(-inf, inf, (1, 74), float32)
ACT Box([-2.35     -2.35     -2.618    -0.26     -2.53     -2.9671   -2.9671
 -2.75     -2.75     -1.5882   -2.2515   -0.33489  -0.33489  -2.618
 -2.618    -0.68     -0.68     -0.2268   -0.2268   -0.2618   -0.2618
 -2.0943   -2.0943   -0.523598 -1.84     -1.84     -0.523598 -0.3
 -0.3      -1.       -1.84     -1.84     -1.2       0.        0.
  0.       -1.84    ], [3.05     3.05     2.618    2.53     0.26     2.7925   2.7925   2.75
 2.75     2.2515   1.5882   2.5449   2.5449   2.618    2.618    0.73
 0.73     3.4208   3.4208   0.2618   0.2618   2.0943   2.0943   0.523598
 0.3      0.3      0.523598 1.84     1.84     1.2      0.       0.
  1.       1.84     1.84     1.84     0.      ], (37,), float32)
DOF tensor([37])
```

### Smoke
`env.observation_space.shape == (1, 74)` (single env) / `(num_envs, 74)`.

---

## §6 Reward

### Description
The base `HumanoidStandEnv` declares `SUPPORTED_REWARD_MODES = ["sparse", "none"]` and implements ONLY a sparse reward: per step, reward = `info["is_standing"]` (1.0 while standing in the height window, else 0.0). There is **no dense reward** — `compute_dense_reward` / `compute_normalized_dense_reward` are commented out in the source. The composer is trivial (single sparse term; effectively a sum of one term). For a dense-reward reproduction you must AUTHOR upright + alive + control-penalty terms (the task ships sparse-only).

### Decisions resolved
- `SUPPORTED_REWARD_MODES = ["sparse", "none"]` (NO `"dense"` / `"normalized_dense"`)
- sparse reward = `info["is_standing"]` (boolean→float, max per-step magnitude 1.0)
- composer: single term (sum-of-one)
- dense reward: **not provided** — commented out (`compute_dense_reward` / `compute_normalized_dense_reward`)
- planning budget (retro-computed): per-step reward ∈ {0, 1}; episode return upper bound = 1000 (max_episode_steps) if standing every step

### Code (`humanoid_stand.py`, base `HumanoidStandEnv`)
```python
def compute_sparse_reward(self, obs: Any, action: torch.Tensor, info: dict):
    return info["is_standing"]

# def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: dict):
#     return torch.zeros(self.num_envs, device=self.device)

# def compute_normalized_dense_reward(
#     self, obs: Any, action: torch.Tensor, info: dict
# ):
#     max_reward = 1.0
#     return self.compute_dense_reward(obs=obs, action=action, info=info) / max_reward
```

### Smoke
With `reward_mode="sparse"`: `env.step(...)` reward ∈ {0.0, 1.0}, finite, shape `(num_envs,)`. With `reward_mode="none"`: reward == 0.

WARN: No dense reward in source. A `/harbor:task-create from=...` reproduction targeting a dense-reward training pipeline must add upright/alive/control-penalty terms; the canonical task is sparse-standing only.

---

## §7 DR (Domain Randomization)

`<no DR>` — no `startup`/`interval`-style domain randomization. The only stochasticity is the §3 reset noise (Gaussian std 0.05 rad on initial qpos). No physics/material/mass/external-force randomization is applied.

---

