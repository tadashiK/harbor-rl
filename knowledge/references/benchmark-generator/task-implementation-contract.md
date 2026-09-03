# Task Implementation Doc — Authoring Contract

Read by `benchmark-generator` when filling `knowledge/templates/benchmark-generator/task-implementation.md.template` into `<repo>/harbor/create-task/task-implementation.md`. The output of that step is the **input** to three future authoring agents (`task-generator`, `reward-tuning-agent`, `dr-generator`) dispatched by `/harbor:task-create` — so the rules here exist to keep downstream behaviour stable.

## What the doc is for

Each `/harbor:task-create` invocation boots three agents in sequence. None re-scan the upstream repo — they all read this single doc. So the doc has to be:

1. **Self-sufficient** — the agents must be able to write a working task using only this doc + the user's task description, without re-discovering the repo's layout.
2. **Family-faithful** — the patterns in the code templates must compile against the venv at `<repo>/.venv/`. If the family uses `mdp.JointPositionActionCfg`, do not invent `mdp.JointPositionActionConfig`.
3. **Canonical-example-anchored** — every section must point at a real, smoke-passing task whose layout future agents copy.

If you cannot anchor a section to a smoke-passing example, the section's "File pointers" + "Code template" must say so explicitly (`<unsupported in this benchmark>`). Future agents will refuse to scaffold the corresponding step rather than guess.

## Where the doc lives

`<repo>/harbor/create-task/task-implementation.md`

The folder is `create-task/` — the stable workspace root for the task-authoring flow (the `/harbor:task-create` command and friends write here). Future per-task scaffolding may also drop artefacts here — keep the folder; never delete it.

## Hard rules

1. **English-only**, regardless of chat language. Same constraint as `install.md` / `benchmark.md` / `history.md`.
2. **Regenerated on every benchmark-generator re-run** — overwrite, do not append. Hand-edits to this file are lost on re-run; permanent corrections belong in the upstream `task-implementation-contract.md` (this file).
3. **Skeleton is fixed** — seven sections in the order §1..§7 below. Adding sections breaks the read contract for downstream agents. Removing sections is acceptable only by leaving the section header and writing `<unsupported in this benchmark>` under each sub-block.
4. **Each code template stays under ~60 lines.** Prose can be longer; code must be drop-in skeleton, not a paste of the canonical task. Future agents copy these blocks verbatim with placeholder substitution; long blocks invite copy-paste-then-forget bugs.
5. **Use `<TaskName>`, `<Robot>`, `<Object>` placeholders in code templates** — never a real task ID. The future agent renames placeholders in one pass.
6. **The smoke-check command for each section must be runnable as-is, against an existing canonical task.** This is the smoke the corresponding future agent will replay. If the smoke depends on a file the future task hasn't been created yet, prefix it with `# after task is registered:` and parameterise on `<NewTaskID>`.

## The seven sections (fixed schema)

| § | Section | Owner agent at /task-create | Smoke verifies |
|---|---------|-------------------------------|----------------|
| 1 | Register task + setup scene | task-generator | `gym.make` (or family equivalent) returns a working env |
| 2 | Action types | task-generator | action_space matches the chosen mode; one step does not error |
| 3 | Init / reset | task-generator | 5 resets succeed; randomized fields differ |
| 4 | Goal + termination | task-generator | episode ends within max_episode_steps; success/failure flag present |
| 5 | Observation | task-generator | obs has expected keys/shape; all values finite |
| 6 | Reward | reward-tuning-agent | reward finite at every step; per-term decomposition (if any) sums to total |
| 7 | Domain randomization | dr-generator | seed-matched obs trajectories diverge with vs without DR |

Each section in the rendered doc has the same five sub-blocks: **Description**, **File pointers**, **Code template**, **Decisions**, **Smoke check**.

## Per-family evidence-gathering rules

The `BENCHMARK_FAMILY` placeholder picks the playbook. Use the matrix below — if the repo doesn't fit, ask the user before guessing.

### isaaclab-manager-based

Detect: `source/isaaclab_tasks/manager_based/` exists; canonical task imports `ManagerBasedRLEnvCfg`. Examples in this repo: `Isaac-Reach-Franka-v0`, `Isaac-Lift-Cube-Franka-v0`.

| Section | Where to look |
|---------|---------------|
| §1 | `source/isaaclab_tasks/manager_based/<domain>/<task>/__init__.py` (`gym.register`), `..._env_cfg.py` (`@configclass class <Task>SceneCfg(InteractiveSceneCfg): ...`), `isaaclab_assets/...` for robot cfgs |
| §2 | `isaaclab.envs.mdp.actions` — `JointPositionActionCfg`, `JointEffortActionCfg`, `DifferentialInverseKinematicsActionCfg` (delta + abs variants live in the same class via `.use_relative_mode`). Citations: `source/isaaclab/isaaclab/envs/mdp/actions/__init__.py` |
| §3 | `EventCfg` with `mode="reset"` terms in `<task>_env_cfg.py`. Common reset funcs: `mdp.reset_root_state_uniform`, `mdp.reset_joints_by_offset` |
| §4 | `TerminationsCfg` (`mdp.time_out`, `mdp.illegal_contact`), `CommandsCfg` for goal sampling (`UniformPoseCommandCfg`, `UniformVelocityCommandCfg`) |
| §5 | `ObservationsCfg` with `@configclass class PolicyCfg(ObsGroup): ...`. Cite the example task's policy group. |
| §6 | `RewardsCfg` with `@configclass` of `RewTerm(func=mdp.<reward_func>, weight=...)`. Composer is **sum** (RewardManager sums weighted terms). Cross-link `/harbor:reward-add-log` — manager-based gets per-term decomposition for free. |
| §7 | `EventCfg` with `mode="startup" / "interval"` terms. Common: `mdp.randomize_rigid_body_mass`, `mdp.randomize_rigid_body_material` |

**Gotchas — propagate verbatim into the §1 smoke command**:

1. `gym.make('<id>')` alone raises `ManagerBasedRLEnv.__init__() missing 1 required positional argument: 'cfg'`. Always pair with `parse_env_cfg`:
   ```python
   from isaaclab_tasks.utils import parse_env_cfg
   env_cfg = parse_env_cfg('<id>', device='cuda:0', num_envs=1, use_fabric=True)
   env     = gym.make('<id>', cfg=env_cfg)
   ```
2. `AppLauncher(headless=True)` MUST run before any `import isaaclab_tasks` — Kit needs to boot or USD imports fail.
3. Do **not** `source _isaac_sim/setup_conda_env.sh` in smoke commands. It exports a stale `CARB_APP_PATH` that breaks the pip-installed `isaacsim 5.1` and surfaces as opaque `omni.kit.commands` errors. Just `source .venv/bin/activate` and run python.

### isaaclab-direct

Detect: `source/isaaclab_tasks/direct/` exists; canonical task imports `DirectRLEnvCfg`. Examples: `Isaac-Cartpole-Direct-v0`, `Isaac-Ant-Direct-v0`.

| Section | Where to look |
|---------|---------------|
| §1 | `<task>_env.py` defining a `class <Task>Env(DirectRLEnv)` + `<task>_env_cfg.py:<Task>EnvCfg`. Scene goes in `_setup_scene()`. |
| §2 | `action_space = gym.spaces.Box(...)` in EnvCfg + `_apply_action(actions)` body. No off-the-shelf delta/EE wrappers — note this in the doc. |
| §3 | `_reset_idx(env_ids)` body |
| §4 | `_get_dones()` returning `(terminated, truncated)` |
| §5 | `_get_observations()` returning a `dict` |
| §6 | `_get_rewards()` returning a 1D tensor. Composer typically **sum** of in-line term tensors. No RewardManager — `/harbor:reward-add-log` adds passthrough wrapper (`detailed_reward = {"total": reward}`). |
| §7 | DR is hand-coded inside `_setup_scene()` / `_reset_idx()`. There is no EventCfg surface. |

### dexteroushands

Detect: `bidexhands/` package, tasks under `bidexhands/tasks/`. Examples: `ShadowHandOver`, `ShadowHandLift`.

| Section | Where to look |
|---------|---------------|
| §1 | `bidexhands/tasks/<task>.py` — class subclasses `VecTask`. Asset loading via `gym.create_actor` with URDFs under `bidexhands/assets/`. |
| §2 | `self.cfg["env"]["controlType"]` ∈ `{"joint", "ee", "torque"}` switched in `pre_physics_step`. Note: dexterous-hands uses 24-DOF position-target by default. |
| §3 | `reset_idx(env_ids)` |
| §4 | `compute_reward()` emits `resets` tensor; `compute_observations()` does not. |
| §5 | `compute_observations()` builds `self.obs_buf` |
| §6 | `compute_reward()` — composer is **sum** of in-line terms |
| §7 | `randomization_params` from cfg, applied in `apply_randomizations()` (inherited from VecTask) |

### loco-mujoco

Detect: `loco_mujoco/` package + `RLFactory.make`. Examples: `Atlas.walk`, `UnitreeH1.run`.

| Section | Where to look |
|---------|---------------|
| §1 | `RLFactory.make("<env>", **env_params)`. Robots are MJX/MuJoCo XMLs under `loco_mujoco/environments/<robot>/`. |
| §2 | `action_type` env_param. Default is joint torque; position control via wrapper. |
| §3 | `reset_type` env_param ∈ {`init`, `random`, `traj`} |
| §4 | `terminal_state_type` env_param + `goal_type` env_param |
| §5 | `observation_spec` env_param — list of obs-key strings. Concat order matters. |
| §6 | `reward_type` env_param — composer is **sum** by default; some types override to **product** |
| §7 | DR via `domain_randomization_type` env_param — values are presets like `"None"`, `"DefaultMass"`, `"DefaultMassFriction"` |

### dm_control

Detect: `dm_control/suite/` exists; canonical task `cartpole/swingup`. Examples: any `(domain, task)` from `dm_control.suite.ALL_TASKS`.

| Section | Where to look |
|---------|---------------|
| §1 | `dm_control/suite/<domain>.py` — `@SUITE.add(...) def <task>(...): ...`. Scene defined in matching XML under `dm_control/suite/<domain>.xml`. |
| §2 | `Physics.named.data.ctrl[:]` in `before_step`. dm_control natively exposes torque control; abs/delta joint position via wrapper (`shimmy`). |
| §3 | `Task.initialize_episode(physics)` |
| §4 | `Task.get_termination(physics)` |
| §5 | `Task.get_observation(physics)` returning OrderedDict |
| §6 | `Task.get_reward(physics)` — composer is whatever the body computes. Most suite tasks use shaped scalars; `_REWARD_TERM_SPECS` is a dm_control extension some forks add. |
| §7 | dm_control suite ships limited DR; family-level DR handled by user wrapper |

### gymnasium-generic

Detect: env registers via plain `gym.register` and inherits `gymnasium.Env`; no manager / suite scaffolding. Use as fallback.

| Section | Where to look |
|---------|---------------|
| §1 | `gym.register(id=..., entry_point=...)` + `class <Task>Env(gymnasium.Env)` |
| §2 | `self.action_space = gym.spaces.Box(...)`; control happens in `step()` |
| §3 | `reset(seed, options)` |
| §4 | `done = terminated or truncated` returned from `step()` |
| §5 | `_get_obs()` returning array or dict |
| §6 | `_get_reward()` — composer is whatever the body computes |
| §7 | Hand-coded inside `reset()` / `step()` — no standard surface |

## Failure modes (what NOT to do)

- **Do not paste the canonical task verbatim** as a code template. The template is a minimal skeleton, not an example.
- **Do not list every reward term** in §6 if the canonical task has 12. List the minimum set (one shaping term + one regularizer) and say "see canonical example file for full term list".
- **Do not invent action modes the family does not support.** If isaaclab-direct has no DifferentialIK action class, mark §2's "Delta EE pose" row as `n/a`.
- **Do not embed paths absolute to the user's machine** (`/home/<user>/...`). All paths must be repo-relative.
- **Do not promise smoke checks that require a file you haven't told the agent to create.** If the smoke needs `<NewTaskID>`, parameterise it; don't hardcode a name future-you can't predict.

## Authoring loop

1. Fill `BENCHMARK_FAMILY` and `CANONICAL_EXAMPLE_TASK_ID` first — both must come from `<repo>/harbor/benchmark-generator/benchmark-spec.json` and the example must be a smoke-passing task. If neither is true, escalate to the user.
2. For each of §1..§7, in order:
   - Locate the canonical example's section (open the example file, find the registration / scene / actions / reset / etc.).
   - Fill **File pointers** with repo-relative paths that exist on disk (verify with `Read`).
   - Write **Description** in 3-6 sentences. Cite the class/function names you used in File pointers.
   - Write **Code template** as a minimal skeleton ≤ 60 lines, with `<TaskName>` style placeholders.
   - Enumerate 2-4 **Decisions** the future agent will need to ask the user about.
   - Write the **Smoke check** as a one-liner that runs against the canonical example today (so you know it works) and is parameterised on `<NewTaskID>` for future agents to swap in.
3. Don't render the doc unless §1's smoke command actually passes when you run it inside the venv. Run it once, paste the literal expected output into `*_SMOKE_EXPECTED`. (You can skip running §2..§7's smoke at this stage — that becomes the future agents' job.)
4. After writing, check the doc loads: `head -5 <repo>/harbor/create-task/task-implementation.md` should show the title and family table.
