# Case Studies

Annotated env-build / action / frame-extract snippets per reference benchmark. The setup recipes are owned by dependency-generator (rendered into `harbor/dependency-generator/setup_uv.sh`); benchmark-generator only picks the smoke patterns below. When generating against a new benchmark, pick the closest match and copy its **unique** snippets for the smoke scripts.

Training scaffolding (sb3 / dispatcher / configs / data_logger) has moved to `rl-integration-generator`. These case studies cover what `benchmark-generator` owns: env-build expressions, action expressions, and render-frame extraction.

## ManiSkill (SAPIEN + Vulkan)

ManiSkill needs system-level Vulkan ICD configuration (`/usr/share/vulkan/icd.d/nvidia_icd.json`). That is a host-system requirement, not something dependency-generator can configure from inside a venv — dependency-generator's pre-render check refuses to render `setup_uv.sh` when the `needs_vulkan_icd` quirk fires. To run ManiSkill, the user must already be on a host with the right Vulkan setup.

Once the host is right, the smoke pattern is:

**`run_random.py` substitutions**:

- `SMOKE_ENV_BUILD`:
```python
import gymnasium as gym
import mani_skill.envs  # noqa: F401 — registers the envs
env = gym.make('PickCube-v1', render_mode=None)
```
- `SMOKE_ACTION_EXPR`: `env.action_space.sample()`

**`render_random.py` substitutions**:

- `SMOKE_ENV_BUILD_RENDER`:
```python
import gymnasium as gym
import mani_skill.envs  # noqa: F401
env = gym.make('PickCube-v1', render_mode='rgb_array', obs_mode='rgbd')
```
- `VIDEO_FRAME_EXTRACT`: `env.render_cameras()[0]['rgb']`

## loco-mujoco (uv + MuJoCo + JAX/MJX)

Pure uv path — no special host setup needed beyond what dependency-generator's harbor-extras block installs.

**`run_random.py` substitutions** (use `RLFactory`, **not** `ImitationFactory` — datasets and license-gated):

- `SMOKE_ENV_BUILD`:
```python
import numpy as np
from loco_mujoco import RLFactory
env = RLFactory.make('UnitreeH1')
```
- `SMOKE_ACTION_EXPR`: `np.random.randn(env.info.action_space.shape[0])`

**`render_random.py` substitutions**:

- `SMOKE_ENV_BUILD_RENDER`: same as L1 (loco-mujoco's MuJoCo backend renders via `env.physics.render()`; no separate constructor flag needed).
- `VIDEO_FRAME_EXTRACT`: `env.render()` (returns `(H, W, 3)` uint8 when MuJoCo is built with the offscreen GLFW backend).

Validation source for the L1 snippet: `tests/test_task_factories.py:55-57` (`test_RLFactory_numpy`) uses the same pattern.

**Anti-pattern (observed)**: picking `examples/tutorials/01_creating_mujoco_env.py` as the smoke test fails all contract criteria — it requires a license-gated model + HF download, calls `env.render()` (needs X11), and runs `while True` (no termination). Reject tutorials whose first 15 lines contain `render()` or `while True`.

## Picking the closest template

| Target benchmark signature | Use as starting point |
|----------------------------|-----------------------|
| SAPIEN / Vulkan (host-prepared) | ManiSkill |
| MuJoCo + JAX/MJX (or any pure-MuJoCo RL benchmark) | loco-mujoco |

IsaacGym benchmarks are NOT supported in uv mode — dependency-generator refuses the `is_isaacgym` quirk because the toolchain (Python 3.8 + CUDA 11.8 + Ubuntu 20.04 in lockstep) cannot be honored on an arbitrary host. Run on a host that already matches.

For non-IsaacGym MuJoCo-flavoured benchmarks (dm_control, gym MuJoCo envs, robosuite-derived), `loco-mujoco` is closest — adjust the env factory / action expression / `VIDEO_FRAME_EXTRACT` per the table at the top of `smoke-test-contract.md`.
