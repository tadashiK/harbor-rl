# PushT-v1 — Implementation Spec

- robot: Franka Panda with stick end-effector (`panda_stick`)
- simulator: ManiSkill (SAPIEN, `BaseEnv` + `@register_env`)
- objects: T-shaped block, T target marker, white table
- bimanual: false
- summary: Push a T-shaped block into alignment with a target T outline.

A simulated version of the real-world push-T task from Diffusion Policy. The robot uses a stick end-effector (`panda_stick`) to precisely push a T-shaped block so that it covers ≥90% of a fixed target-T region on the table. Success is coverage-only (the "PushT-easy" variant — the ee end-zone return is not enforced by `evaluate()`).

---

## §1 Registration + Scene

### Description
Registered as `PushT-v1` with `max_episode_steps=100`, sole supported robot `panda_stick`. The scene uses a custom `WhiteTableSceneBuilder` (subclass of `TableSceneBuilder`) that (1) un-textures the table to flat white and (2) adds a `panda_stick` rest keyframe. The robot base is placed at `[-0.615, 0, 0]`. The scene builds three actors via a shared `create_tee` factory of two box primitives: a dynamic T block (`Tee`), a kinematic gray target-T outline (`goal_Tee`, near-zero thickness), and a kinematic gray cylinder marking the ee end-zone (`goal_ee`). The remainder of `_load_scene` precomputes a 64×64 UV grid and an egocentric T-shaped binary mask (`tee_render`) plus the world→goal homogeneous transform, used by the custom CUDA "pseudo-renderer" that measures T/goal overlap for success + (legacy) reward.

### Decisions resolved
- `@register_env("PushT-v1", max_episode_steps=100)`
- `SUPPORTED_ROBOTS = ["panda_stick"]`; default `robot_uids="panda_stick"`, `robot_init_qpos_noise=0.02`.
- Robot base pose: `sapien.Pose(p=[-0.615, 0, 0])` (set both in `_load_agent` and in the table-scene keyframe).
- panda_stick rest qpos for this task (in `WhiteTableSceneBuilder.initialize`): `[0.662, 0.212, 0.086, -2.685, -0.115, 2.898, 1.673]` + Gaussian noise `N(0, 0.02)`.
- T block (`create_tee`, `target=False`): two boxes — box1 half-size `[0.1, 0.025, 0.02]`, box2 half-size `[0.025, 0.075, 0.02]` (i.e. `box1_half_w=0.1`, `box1_half_h=0.025`, `half_thickness=0.02`); com_y offset `0.0375`; `_mass=0.8`; PhysxMaterial static/dynamic friction `3/3`, restitution `0`; color TARGET_RED `[194,19,22,255]/255`. Built as a dynamic rigid body.
- Target T (`goal_Tee`, `target=True`): same geometry but `half_thickness=1e-4`, no collision, gray `[128,128,128,255]/255`, built **kinematic**.
- EE end-zone marker (`goal_ee`): kinematic cylinder, `radius=0.02`, `half_length=1e-4`, gray.
- Sim cfg: `SimConfig(gpu_memory_config=GPUMemoryConfig(found_lost_pairs_capacity=2**25, max_rigid_patch_count=2**18))`.
- Sensors: `base_camera` 128×128 fov π/2; human render `render_camera` 512×512 fov 1; both `look_at(eye=[0.3,0,0.6], target=[-0.1,0,0.1])`.
- Pseudo-render constants: `res=64`, `uv_half_width=0.15`, `center_of_mass=(0, 0.0375)`, `intersection_thresh=0.90`, `goal_offset=[-0.156,-0.1]`, `goal_z_rot=(5/3)π`, `ee_starting_pos2D=[-0.321,0.284,1e-3]`, `ee_starting_pos3D=[-0.321,0.284,0.024]`.

### Code (verbatim — full registration + scene)
```python
from typing import Any

import numpy as np
import sapien
import torch
import torch.random
from transforms3d.euler import euler2quat

from mani_skill.agents.robots import PandaStick
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import common, sapien_utils
from mani_skill.utils.building import actors
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.structs import Pose
from mani_skill.utils.structs.types import Array, GPUMemoryConfig, SimConfig


# extending TableSceneBuilder and only making 2 changes:
# 1.Making table smooth and white, 2. adding support for keyframes of new robots - panda stick
class WhiteTableSceneBuilder(TableSceneBuilder):
    def initialize(self, env_idx: torch.Tensor):
        super().initialize(env_idx)
        b = len(env_idx)
        if self.env.robot_uids == "panda_stick":
            qpos = np.array(
                [
                    0.662,
                    0.212,
                    0.086,
                    -2.685,
                    -0.115,
                    2.898,
                    1.673,
                ]
            )
            qpos = (
                self.env._episode_rng.normal(
                    0, self.robot_init_qpos_noise, (b, len(qpos))
                )
                + qpos
            )
            self.env.agent.reset(qpos)
            self.env.agent.robot.set_pose(sapien.Pose([-0.615, 0, 0]))

    def build(self):
        super().build()
        # cheap way to un-texture table
        for part in self.table._objs:
            for triangle in (
                part.find_component_by_type(sapien.render.RenderBodyComponent)
                .render_shapes[0]
                .parts
            ):
                triangle.material.set_base_color(np.array([255, 255, 255, 255]) / 255)
                triangle.material.set_base_color_texture(None)
                triangle.material.set_normal_texture(None)
                triangle.material.set_emission_texture(None)
                triangle.material.set_transmission_texture(None)
                triangle.material.set_metallic_texture(None)
                triangle.material.set_roughness_texture(None)


@register_env("PushT-v1", max_episode_steps=100)
class PushTEnv(BaseEnv):
    _sample_video_link = "https://github.com/haosulab/ManiSkill/raw/main/figures/environment_demos/PushT-v1_rt.mp4"
    SUPPORTED_ROBOTS = ["panda_stick"]
    agent: PandaStick

    # Randomizations
    tee_spawnbox_xlength = 0.2
    tee_spawnbox_ylength = 0.3
    tee_spawnbox_xoffset = -0.1
    tee_spawnbox_yoffset = -0.1

    goal_offset = torch.tensor([-0.156, -0.1])
    goal_z_rot = (5 / 3) * np.pi

    ee_starting_pos2D = torch.tensor([-0.321, 0.284, 1e-3])
    ee_starting_pos3D = torch.tensor([-0.321, 0.284, 0.024])

    intersection_thresh = 0.90

    T_mass = 0.8
    T_dynamic_friction = 3
    T_static_friction = 3

    def __init__(
        self, *args, robot_uids="panda_stick", robot_init_qpos_noise=0.02, **kwargs
    ):
        self.robot_init_qpos_noise = robot_init_qpos_noise
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    @property
    def _default_sim_config(self):
        return SimConfig(
            gpu_memory_config=GPUMemoryConfig(
                found_lost_pairs_capacity=2**25, max_rigid_patch_count=2**18
            )
        )

    @property
    def _default_sensor_configs(self):
        pose = sapien_utils.look_at(eye=[0.3, 0, 0.6], target=[-0.1, 0, 0.1])
        return [
            CameraConfig(
                "base_camera", pose=pose, width=128, height=128,
                fov=np.pi / 2, near=0.01, far=100,
            )
        ]

    @property
    def _default_human_render_camera_configs(self):
        pose = sapien_utils.look_at(eye=[0.3, 0, 0.6], target=[-0.1, 0, 0.1])
        return CameraConfig(
            "render_camera", pose=pose, width=512, height=512, fov=1, near=0.01, far=100
        )

    def _load_agent(self, options: dict):
        super()._load_agent(options, sapien.Pose(p=[-0.615, 0, 0]))

    def _load_scene(self, options: dict):
        self.ee_starting_pos2D = self.ee_starting_pos2D.to(self.device)
        self.ee_starting_pos3D = self.ee_starting_pos3D.to(self.device)

        self.table_scene = WhiteTableSceneBuilder(
            env=self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.table_scene.build()

        TARGET_RED = np.array([194, 19, 22, 255]) / 255

        def create_tee(name="tee", target=False, base_color=TARGET_RED):
            box1_half_w = 0.2 / 2
            box1_half_h = 0.05 / 2
            half_thickness = 0.04 / 2 if not target else 1e-4
            com_y = 0.0375

            builder = self.scene.create_actor_builder()
            first_block_pose = sapien.Pose([0.0, 0.0 - com_y, 0.0])
            first_block_size = [box1_half_w, box1_half_h, half_thickness]
            if not target:
                builder._mass = self.T_mass
                tee_material = sapien.pysapien.physx.PhysxMaterial(
                    static_friction=self.T_dynamic_friction,
                    dynamic_friction=self.T_static_friction,
                    restitution=0,
                )
                builder.add_box_collision(
                    pose=first_block_pose, half_size=first_block_size, material=tee_material,
                )
            builder.add_box_visual(
                pose=first_block_pose, half_size=first_block_size,
                material=sapien.render.RenderMaterial(base_color=base_color),
            )

            second_block_pose = sapien.Pose([0.0, 4 * (box1_half_h) - com_y, 0.0])
            second_block_size = [box1_half_h, (3 / 4) * (box1_half_w), half_thickness]
            if not target:
                builder.add_box_collision(
                    pose=second_block_pose, half_size=second_block_size, material=tee_material,
                )
            builder.add_box_visual(
                pose=second_block_pose, half_size=second_block_size,
                material=sapien.render.RenderMaterial(base_color=base_color),
            )
            builder.initial_pose = sapien.Pose(p=[0, 0, 0.1])
            if not target:
                return builder.build(name=name)
            else:
                return builder.build_kinematic(name=name)

        self.tee = create_tee(name="Tee", target=False)
        self.goal_tee = create_tee(
            name="goal_Tee", target=True, base_color=np.array([128, 128, 128, 255]) / 255,
        )

        builder = self.scene.create_actor_builder()
        builder.add_cylinder_visual(
            radius=0.02, half_length=1e-4,
            material=sapien.render.RenderMaterial(
                base_color=np.array([128, 128, 128, 255]) / 255
            ),
        )
        builder.initial_pose = sapien.Pose(p=[0, 0, 0.1])
        self.ee_goal_pos = builder.build_kinematic(name="goal_ee")

        # Custom 2D "Pseudo-Rendering" setup
        res = 64
        uv_half_width = 0.15
        self.uv_half_width = uv_half_width
        self.res = res
        oned_grid = torch.arange(res, dtype=torch.float32).view(1, res).repeat(res, 1) - (res / 2)
        self.uv_grid = (
            torch.cat([oned_grid.unsqueeze(0), (-1 * oned_grid.T).unsqueeze(0)], dim=0) + 0.5
        ) / ((res / 2) / uv_half_width)
        self.uv_grid = self.uv_grid.to(self.device)
        self.homo_uv = torch.cat(
            [self.uv_grid, torch.ones_like(self.uv_grid[0]).unsqueeze(0)], dim=0
        )

        self.center_of_mass = (0, 0.0375)
        box1 = torch.tensor([[-0.1, 0.025], [0.1, 0.025], [-0.1, -0.025], [0.1, -0.025]])
        box2 = torch.tensor([[-0.025, 0.175], [0.025, 0.175], [-0.025, 0.025], [0.025, 0.025]])
        box1[:, 1] -= self.center_of_mass[1]
        box2[:, 1] -= self.center_of_mass[1]

        box1 *= (res / 2) / uv_half_width
        box1 += res / 2
        box2 *= (res / 2) / uv_half_width
        box2 += res / 2
        box1 = box1.long()
        box2 = box2.long()

        self.tee_render = torch.zeros(res, res)
        self.tee_render.T[box1[0, 0] : box1[1, 0], box1[2, 1] : box1[0, 1]] = 1
        self.tee_render.T[box2[0, 0] : box2[1, 0], box2[2, 1] : box2[0, 1]] = 1
        self.tee_render = self.tee_render.flip(0).to(self.device)

        goal_fake_quat = torch.tensor(
            [(torch.tensor([self.goal_z_rot]) / 2).cos(), 0, 0, 0.0]
        ).unsqueeze(0)
        zrot = self.quat_to_zrot(goal_fake_quat).squeeze(0)
        goal_trans = torch.eye(3)
        goal_trans[:2, :2] = zrot[:2, :2]
        goal_trans[0:2, 2] = self.goal_offset
        self.world_to_goal_trans = torch.linalg.inv(goal_trans).to(self.device)
```

### Smoke (§1 build)
```bash
cd <repo>
.venv/bin/python -c "import gymnasium as gym, mani_skill; e=gym.make('PushT-v1'); print(e.observation_space, e.action_space); print(e.unwrapped.control_mode); e.close()"
```
Expected stdout (literal, from a passing run):
```
Box(-inf, inf, (1, 31), float32) Box(-1.0, 1.0, (7,), float32)
pd_joint_delta_pos
```

---

## §2 Actions

### Description
PushT does **not** override the controller; it uses `panda_stick`'s default `_controller_configs`. The active control mode is `pd_joint_delta_pos` (7-DoF arm; the stick has no gripper). This is **NOT** a 2-D planar push controller — it is a full 7-joint delta-position controller. The action is a 7-vector of per-step joint-position deltas in `[-0.1, 0.1]` rad (normalized into `[-1, 1]` action space), tracked by a PD controller at `stiffness=1e3`, `damping=1e2`, `force_limit=100`. The robot indirectly produces planar pushes because the stick tip stays near the table, but the action space itself is joint-space.

### Decisions resolved
- `control_mode = "pd_joint_delta_pos"` (ManiSkill default; first key of `controller_configs`).
- Action space: `Box(-1.0, 1.0, (7,), float32)`.
- Arm controller: `PDJointPosControllerConfig(arm_joint_names, lower=-0.1, upper=0.1, stiffness=1e3, damping=1e2, force_limit=100, use_delta=True)`.
- ee_link / tcp: `panda_hand_tcp`. No gripper action.
- Other selectable modes (not default): `pd_joint_pos`, `pd_ee_delta_pos`, `pd_ee_delta_pose`, `pd_ee_delta_pose_align`, `pd_ee_pose`, and target/vel variants.

### Code (verbatim — relevant arm controller from `panda_stick.py`)
```python
arm_stiffness = 1e3
arm_damping = 1e2
arm_force_limit = 100

arm_pd_joint_delta_pos = PDJointPosControllerConfig(
    self.arm_joint_names,
    lower=-0.1,
    upper=0.1,
    stiffness=self.arm_stiffness,
    damping=self.arm_damping,
    force_limit=self.arm_force_limit,
    use_delta=True,
)
# ...
controller_configs = dict(
    pd_joint_delta_pos=dict(arm=arm_pd_joint_delta_pos),  # <-- default (first key)
    pd_joint_pos=dict(arm=arm_pd_joint_pos),
    pd_ee_delta_pos=dict(arm=arm_pd_ee_delta_pos),
    pd_ee_delta_pose=dict(arm=arm_pd_ee_delta_pose),
    ...
)
```

### Smoke (§2 action)
```bash
.venv/bin/python -c "import gymnasium as gym, mani_skill, numpy as np; e=gym.make('PushT-v1'); e.reset(seed=0); a=e.action_space.sample(); o,r,te,tr,i=e.step(a); print(a.shape, r); e.close()"
```
Expected: action shape `(7,)`, finite scalar reward.

---

## §3 Reset (`_initialize_episode`)

### Description
On reset: initialize the table scene (which also resets the robot qpos with noise + base pose). The fixed goal-T is placed at `goal_offset=[-0.156,-0.1]`, z=1e-3, rotated by `goal_z_rot=(5/3)π`. The dynamic T block is spawned in a randomized box relative to the goal: x ∈ goal_x + U[−0.1, 0.1], y ∈ goal_y + U[−0.1, 0.2] (spawnbox lengths 0.2/0.3 with offsets −0.1/−0.1), z = `0.04/2 + 1e-3` (half-thickness above table), with a uniform random z-rotation in `[0, 2π]`. The ee end-zone marker is placed at `ee_starting_pos2D`, rotated `euler(0, π/2, 0)`.

### Decisions resolved
- Goal-T pose: `p=[goal_offset_x, goal_offset_y, 1e-3]`, `q=euler2quat(0,0,(5/3)π)`. Fixed every episode.
- T block spawn: `x = goal_x + U[0,0.2] - 0.1`, `y = goal_y + U[0,0.3] - 0.1`, `z = 0.04/2 + 1e-3 = 0.021`.
- T block z-rotation: uniform `U[0, 2π]`, quaternion `[cos(θ/2), 0, 0, sin(θ/2)]`.
- Robot init qpos noise: `N(0, 0.02)` around the task rest qpos (handled in `WhiteTableSceneBuilder.initialize`).

### Code (verbatim)
```python
def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
    with torch.device(self.device):
        b = len(env_idx)
        self.table_scene.initialize(env_idx)

        target_region_xyz = torch.zeros((b, 3))
        target_region_xyz[:, 0] += self.goal_offset[0]
        target_region_xyz[:, 1] += self.goal_offset[1]
        target_region_xyz[..., 2] = 1e-3
        self.goal_tee.set_pose(
            Pose.create_from_pq(
                p=target_region_xyz,
                q=euler2quat(0, 0, self.goal_z_rot),
            )
        )

        target_region_xyz[..., 0] += (
            torch.rand(b) * (self.tee_spawnbox_xlength) + self.tee_spawnbox_xoffset
        )
        target_region_xyz[..., 1] += (
            torch.rand(b) * (self.tee_spawnbox_ylength) + self.tee_spawnbox_yoffset
        )
        target_region_xyz[..., 2] = 0.04 / 2 + 1e-3
        q_euler_angle = torch.rand(b) * (2 * torch.pi)
        q = torch.zeros((b, 4))
        q[:, 0] = (q_euler_angle / 2).cos()
        q[:, -1] = (q_euler_angle / 2).sin()

        obj_pose = Pose.create_from_pq(p=target_region_xyz, q=q)
        self.tee.set_pose(obj_pose)

        xyz = torch.zeros((b, 3))
        xyz[:] = self.ee_starting_pos2D
        self.ee_goal_pos.set_pose(
            Pose.create_from_pq(
                p=xyz,
                q=euler2quat(0, np.pi / 2, 0),
            )
        )
```

### Smoke (§3 reset)
```bash
.venv/bin/python -c "import gymnasium as gym, mani_skill; e=gym.make('PushT-v1'); e.reset(seed=0); e.reset(seed=1); print('ok'); e.close()"
```
Expected: two resets succeed.

---

## §4 Goal + Termination

### Description
Success is **coverage-only**: the dynamic T must cover ≥90% of the goal-T's 2D area, computed by the custom `pseudo_render_intersection()` (intersection_area / goal_area on a 64×64 binary raster in the goal frame, fully parallel/no-loop on CUDA). The ee end-zone return described in the docstring is the full PushT variant and is **not** enforced here (this is effectively PushT-easy). `max_episode_steps=100` provides the time-out truncation (set in `@register_env`, not via a termination function). There are no `CommandsCfg`-style command terms (ManiSkill has no command manager).

### Decisions resolved
- `evaluate()` returns `{"success": inter_area >= 0.90}`.
- `intersection_thresh = 0.90`.
- `max_episode_steps = 100` (truncation).
- No failure/early-termination condition; episodes end on success or time-out.

### Code (verbatim — `evaluate` + the pseudo-render intersection it depends on)
```python
def evaluate(self):
    # success is where the overlap is over intersection thresh and ee dist to start pos is less than it's own thresh
    inter_area = self.pseudo_render_intersection()
    tee_place_success = (inter_area) >= self.intersection_thresh
    success = tee_place_success
    return {"success": success}
```

```python
def quat_to_z_euler(self, quats):
    assert len(quats.shape) == 2 and quats.shape[-1] == 4
    signs = torch.ones_like(quats[:, -1])
    signs[quats[:, -1] < 0] = -1.0
    qw = quats[:, 0] * signs
    z_euler = 2 * qw.acos()
    return z_euler

def quat_to_zrot(self, quats):
    assert len(quats.shape) == 2 and quats.shape[-1] == 4
    alphas = self.quat_to_z_euler(quats)
    rot_mats = torch.zeros(quats.shape[0], 3, 3).to(quats.device)
    rot_mats[:, 2, 2] = 1
    rot_mats[:, 0, 0] = alphas.cos()
    rot_mats[:, 1, 1] = alphas.cos()
    rot_mats[:, 0, 1] = -alphas.sin()
    rot_mats[:, 1, 0] = alphas.sin()
    return rot_mats

def pseudo_render_intersection(self):
    """'pseudo render' algo for calculating the intersection
    made custom 'psuedo renderer' to compute intersection area
    all computation in parallel on cuda, zero explicit loops
    views blocks in 2d in the goal tee frame to see overlap"""
    tee_to_world_trans = self.quat_to_zrot(self.tee.pose.q)
    tee_to_world_trans[:, 0:2, 2] = self.tee.pose.p[:, :2]

    tee_to_goal_trans = self.world_to_goal_trans @ tee_to_world_trans

    b = tee_to_world_trans.shape[0]
    res = self.uv_grid.shape[1]
    homo_uv = self.homo_uv

    tees_in_goal_frame = (tee_to_goal_trans @ homo_uv.view(3, -1)).view(b, 3, res, res)
    tees_in_goal_frame = tees_in_goal_frame[:, 0:2, :, :] / tees_in_goal_frame[
        :, -1, :, :
    ].unsqueeze(1)

    tee_coords = tees_in_goal_frame[:, :, self.tee_render == 1].view(b, 2, -1)

    tee_indices = (
        (tee_coords * ((res / 2) / self.uv_half_width) + (res / 2)).long().view(b, 2, -1)
    )

    final_renders = torch.zeros(b, res, res).to(self.device)
    num_tee_pixels = tee_indices.shape[-1]
    batch_indices = (
        torch.arange(b).view(-1, 1).repeat(1, num_tee_pixels).to(self.device)
    )

    invalid_xs = (tee_indices[:, 0, :] < 0) | (tee_indices[:, 0, :] >= self.res)
    invalid_ys = (tee_indices[:, 1, :] < 0) | (tee_indices[:, 1, :] >= self.res)
    tee_indices[:, 0, :][invalid_xs] = 0
    tee_indices[:, 1, :][invalid_xs] = 0
    tee_indices[:, 0, :][invalid_ys] = 0
    tee_indices[:, 1, :][invalid_ys] = 0

    final_renders[batch_indices, tee_indices[:, 0, :], tee_indices[:, 1, :]] = 1
    final_renders = final_renders.permute(0, 2, 1).flip(1)

    intersection = (
        (final_renders.bool() & self.tee_render.bool()).sum(dim=[-1, -2]).float()
    )
    goal_area = self.tee_render.bool().sum().float()

    reward = intersection / goal_area
    return reward
```

### Smoke (§4 termination)
```bash
.venv/bin/python -c "import gymnasium as gym, mani_skill; e=gym.make('PushT-v1'); o,_=e.reset(seed=0); print(e.unwrapped.evaluate()['success'].shape); print('max_steps', e.spec.max_episode_steps); e.close()"
```
Expected: success tensor shape `(1,)`, `max_steps 100`.

---

## §5 Observation (`_get_obs_extra` + obs modes)

### Description
Default `obs_mode = "state"`. `_get_obs_extra` always returns `tcp_pose` (7: xyz + quat of the stick tip); in state-based modes it additionally returns `goal_pos` (3: goal-T position) and `obj_pose` (7: T block pose). These task-extra terms are concatenated with the base agent proprioception (qpos/qvel etc.) into a flat `(1, 31)` state vector. Supported reward modes: `('normalized_dense', 'dense', 'sparse', 'none')`.

### Decisions resolved
- `default_obs_mode = "state"`; flattened observation dim = **31**.
- Task-extra terms: `tcp_pose` (7) always; `goal_pos` (3) + `obj_pose` (7) when `use_state` → 17 task-extra dims.
- Remaining 14 dims come from `panda_stick` proprioception (qpos 7 + qvel 7) via the base `_get_obs_agent`.
- Other obs modes available via ManiSkill: `state_dict`, `sensor_data`, `rgb`, `rgbd`, `pointcloud`, etc. (the env adds the camera configs in §1 for visual modes).

### Code (verbatim)
```python
def _get_obs_extra(self, info: dict):
    # ee position is super useful for pandastick robot
    obs = dict(
        tcp_pose=self.agent.tcp.pose.raw_pose,
    )
    if self.obs_mode_struct.use_state:
        # state based gets info on goal position and t full pose - necessary to learn task
        obs.update(
            goal_pos=self.goal_tee.pose.p,
            obj_pose=self.tee.pose.raw_pose,
        )
    return obs
```

### Smoke (§5 observation)
```bash
.venv/bin/python -c "import gymnasium as gym, mani_skill; e=gym.make('PushT-v1'); o,_=e.reset(seed=0); print(o.shape); e.close()"
```
Expected stdout: `torch.Size([1, 31])` (or `(1, 31)` numpy).

---

## §6 Reward

### Description
**Composer = sum** of three shaping terms with a hard success override. Despite the elaborate pseudo-renderer (kept for evaluation), the *reward* is **pose-based, not coverage-based**: a comment notes the legacy coverage reward got stuck in 50–75% local maxima for PPO, so the dense reward uses (a) a squared z-rotation cosine-similarity term to the goal orientation (∈ [0, 0.5]), (b) a squared `tanh` translation term between T and goal (∈ [0, 0.5]), and (c) a small `sqrt(tanh)` shaping term rewarding the tcp being near the T center of mass (∈ [0, 0.05]). On success (`info["success"]`, i.e. ≥90% coverage from §4) reward is overwritten to the constant max `3`. `compute_normalized_dense_reward` divides by `max_reward = 3.0`.

### Decisions resolved
- Composer: **sum** of three terms, then `reward[success] = 3`.
- Term 1 (rotation): `(((cos(z_euler − goal_z_rot) + 1)/2) ** 2) / 2` → range `[0, 0.5]`.
- Term 2 (translation): `((1 − tanh(5·||tee_xy − goal_xy||))**2) / 2` → range `[0, 0.5]`.
- Term 3 (tcp-to-T assist): `sqrt(1 − tanh(5·||tee_p − tcp_p||)) / 20` → range `[0, 0.05]`.
- Non-success max ≈ `0.5 + 0.5 + 0.05 = 1.05`; success constant `3.0`.
- `max_reward = 3.0` (normalizer).

### Code (verbatim — FULL `compute_dense_reward` + `compute_normalized_dense_reward`)
```python
def compute_dense_reward(self, obs: Any, action: Array, info: dict):
    # reward for overlap of the tees

    # legacy reward
    # reward = self.pseudo_render_reward()
    # Pose based reward below is preferred over legacy reward
    # legacy reward gets stuck in local maxs of 50-75% intersection
    # and then fails to promote large explorations to perfectly orient the T, for PPO algorithm

    # new pose based reward: cos(z_rot_euler) + function of translation, between target and goal both in [0,1]
    # z euler cosine similarity reward: -- quat_to_z_euler guarenteed to reutrn value from [0,2pi]
    tee_z_eulers = self.quat_to_z_euler(self.tee.pose.q)
    # subtract the goal z rotatation to get relative rotation
    rot_rew = (tee_z_eulers - self.goal_z_rot).cos()
    # cos output [-1,1], we want reward of 0.5
    reward = (((rot_rew + 1) / 2) ** 2) / 2

    # x and y distance as reward
    tee_to_goal_pose = self.tee.pose.p[:, 0:2] - self.goal_tee.pose.p[:, 0:2]
    tee_to_goal_pose_dist = torch.linalg.norm(tee_to_goal_pose, axis=1)
    reward += ((1 - torch.tanh(5 * tee_to_goal_pose_dist)) ** 2) / 2

    # giving the robot a little help by rewarding it for having its end-effector close to the tee center of mass
    tcp_to_push_pose = self.tee.pose.p - self.agent.tcp.pose.p
    tcp_to_push_pose_dist = torch.linalg.norm(tcp_to_push_pose, axis=1)
    reward += ((1 - torch.tanh(5 * tcp_to_push_pose_dist)).sqrt()) / 20

    # assign rewards to parallel environments that achieved success to the maximum of 3.
    reward[info["success"]] = 3
    return reward

def compute_normalized_dense_reward(self, obs: Any, action: Array, info: dict):
    max_reward = 3.0
    return self.compute_dense_reward(obs=obs, action=action, info=info) / max_reward
```

### Planning budget (retro-computed from weights)
- Per-step saturated magnitudes: rotation-aligned ≈ 0.5; translation-aligned ≈ 0.5; tcp-assist ≈ 0.05; combined shaping ceiling ≈ 1.05.
- Terminal success: constant 3.0 (≈ 2.86× the shaping ceiling) — clearly dominates, so the policy is pulled toward true ≥90% coverage rather than parking at the shaping optimum.

### Smoke (§6 reward)
```bash
.venv/bin/python -c "import gymnasium as gym, mani_skill, torch; e=gym.make('PushT-v1', reward_mode='dense'); e.reset(seed=0); a=e.action_space.sample(); o,r,te,tr,i=e.step(a); print(float(r), bool(torch.isfinite(torch.as_tensor(r)).all())); e.close()"
```
Expected: finite scalar reward in roughly `[0, 1.05]` (3.0 only on success).

---

## §7 DR

`<no DR>` — There is no `startup`/`interval`-style domain randomization. All randomness lives in `_initialize_episode` (T block x/y/z-rotation spawn, robot qpos noise) and is reset-only, which belongs to §3, not §7. Physics material friction (3/3), mass (0.8), and the goal-T pose are fixed every episode.

---

