# Smoke Test Contract

Step 3 (formerly Step 4) runs after the env is ready (`<repo>/.venv/` populated). It proves the runtime is not just importable but exercises the env step + render pipeline. **Two tiers, both fatal.**

Both tiers run via `<repo>/.venv/bin/python scripts/{run_random,render_random}.py ...`.

Training-time scaffolding (sb3 / dispatcher / configs / data_logger) lives in `rl-integration-generator`. This file does not document training smokes — see `agents/rl-integration-generator.md` Phase 4 for that.

## Tier layout

| Tier | What it proves | Budget | Failure is fatal? |
|------|----------------|--------|-------------------|
| **L1 (random)** | `scripts/run_random.py` runs N steps and every reward is finite | < 30 s | Yes — proves env step + reward signal |
| **L2 (render)** | `scripts/render_random.py` writes a non-empty MP4 from N frames | < 30 s | Yes — proves the offscreen render pipeline |

## Why two tiers

- **L1** catches broken imports, missing deps, wrong path, AND a non-finite reward signal in one shot. It runs the actual rendered script — same artefact the user will hit on day 1. A `None` / `NaN` reward is recorded as FAIL but does NOT abort the rest of the pipeline (we still write the spec + receipts so the issue is visible).
- **L2** catches a class of bug L1 cannot: missing imageio / ffmpeg / libGL on the offscreen path, or a `{{VIDEO_FRAME_EXTRACT}}` expression that returns `None` for this benchmark. Without L2, the user's first attempt to `--save_video` would silently produce 0-byte MP4s.
- A separate headed-window check is unnecessary because the user is already running on the host's display — `python scripts/render_random.py` opens a GUI window directly when the env's renderer supports it.

## Contract (all tiers)

| Criterion | Rule |
|-----------|------|
| No side effects | No HuggingFace / S3 downloads, no license acceptance, no persistent file writes outside `/tmp` (L2 writes one MP4 to `/tmp/harbor-smoke-$$.mp4`; clean up after). The `$$` is required, not decoration: a fixed name in a world-writable directory is either owned by another user — so the write fails — or is another run's file, which passes the `>0` bytes check while telling you nothing about this task. |
| Headless | L1 + L2 — no `env.render()` to a screen, no `--gui` flag. Explicit `render_mode='rgb_array'` (L2) / `None` (L1) on the env constructor. |
| Finite termination | Exactly N steps (L1: 10, L2: 30), print `Lx OK`, exit 0. Never `while True`. |
| Deterministic where possible | `env.reset(seed=0)` when the API supports it. |
| Budget | L1 < 30 s, L2 < 30 s on a single NVIDIA L4 / RTX 3060 class GPU. |

## Selection rules

### `SMOKE_ENV_BUILD` (L1 — headless)

Pick the benchmark's **plain RL env factory** with rendering disabled.

- loco-mujoco → `from loco_mujoco import RLFactory; env = RLFactory.make('UnitreeH1')`
- ManiSkill → `import gymnasium as gym, mani_skill.envs; env = gym.make('PickCube-v1', render_mode=None)`
- IsaacLab → `import gymnasium as gym; env = gym.make('Isaac-Lift-Cube-Franka-v0', num_envs=4, headless=True)`
- dm_control + shimmy → use the helper that pins cwd off the source tree (e.g. `from scripts._dm_env import make_dm_control_env; env = make_dm_control_env('cartpole/swingup')`)

### `SMOKE_ENV_BUILD_RENDER` (L2 — offscreen rendering enabled)

Same shape as `SMOKE_ENV_BUILD` but with rendering wired:

- gymnasium-style → swap `render_mode=None` → `render_mode='rgb_array'`
- robosuite / LIBERO → `has_offscreen_renderer=True, use_camera_obs=True`
- SAPIEN → `obs_mode='rgbd'` (so cameras are populated)
- dm_control → `env.physics.render(...)` works regardless; can reuse `SMOKE_ENV_BUILD`

### `SMOKE_ACTION_EXPR` (L1 / L2)

In order of preference:

1. `env.action_space.sample()` — standard gym/gymnasium
2. `np.random.randn(env.info.action_space.shape[0])` — custom Space interface (loco-mujoco)
3. `np.zeros(env.action_dim)` — last-resort fallback (zero action can be a no-op on clipped envs; L1 reward will be uninformative)

### `VIDEO_FRAME_EXTRACT` (L2)

Single expression returning one RGB `(H, W, 3)` `uint8` ndarray or `None`, given `env`, `obs`, `info` in scope after `env.step(...)`. Examples:

- gymnasium: `env.render() if getattr(env, 'render_mode', None) == 'rgb_array' else None`
- SAPIEN / ManiSkill: `env.render_cameras()[0]['rgb']`
- robosuite / LIBERO: `obs.get('robot0_agentview_left_image')`
- RLBench: `getattr(_obs, 'front_rgb', None)`
- dm_control + shimmy: `env.render()`

If no obvious offscreen path exists, set to `None` and document the limitation in `benchmark.md`. L2 will then fail with `no RGB frames captured`; that's a deliberate, visible failure.

### Headed visualization (manual)

Out of scope for the smoke contract — the user can launch `python scripts/render_random.py --task <task> --n-steps 500 --output /tmp/viz.mp4` directly under their host display when they want to watch the simulator. No automated tier.

## Validation sources

Every generated `run_random.py` / `render_random.py` snippet must be backed by a pattern that already exists in the target repo's tests or examples. `grep tests/` for `env.reset() ... env.step(...)` without dataset imports; if the pattern exists, snippet is proven. For loco-mujoco, `tests/test_task_factories.py:55-57` is the validation source for `RLFactory.make(...)`.

## Anti-patterns (observed)

- Using `examples/tutorials/*.py` blindly — they often call `env.render()` in a `while True` loop, or require a dataset. Reject if first 15 lines contain `render()` or `while True`.
- `env.step(np.zeros(action_dim))` on envs with bounded clipping that makes zero a no-op — L1 `reward` will be constant and uninformative. Use `.sample()` or `randn`.
- Skipping L2 because "L1 already covers env stepping" — L2 catches a different bug (offscreen render path). They are not redundant.
- Treating L2 failure as image-invalid when the benchmark genuinely has no camera obs — surface the `{{VIDEO_FRAME_EXTRACT}} = None` situation in `benchmark.md` Troubleshooting; do not silently delete `render_random.py`.

## Reporting into history.md

Step 5 pulls the stdout tail of each tier (last 5 lines) and formats as:

```markdown
| Tier | Status | Last output |
|------|--------|-------------|
| L1 (random) | ✅ OK | `L1 OK: 10 steps, all rewards finite` |
| L2 (render) | ✅ OK | `L2 OK: wrote 30 frames to /tmp/harbor-smoke-4711.mp4 (412.7 KB)` |
```

Even a FAIL tier goes into the doc — silent omission is worse than a visible failure.
