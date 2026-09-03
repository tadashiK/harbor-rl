# §1 — Registration + scene

The task's identity and its physical world: the `gym.register` entry, the env cfg class, and
the scene the simulator builds from it.

## What §1 authors

- The `gym.register(...)` call — id, entry point, `env_cfg_entry_point`, and any kwargs the
  benchmark's own registrations carry. Mirror the canonical example's idiom exactly; a
  registration that differs structurally is the most common reason `gym.make` fails later.
- The env cfg class (`@configclass class <Task>EnvCfg(ManagerBasedRLEnvCfg)`) and its
  `__post_init__`.
- The scene cfg: robot articulation, manipulated objects, table / ground plane, lights,
  sensors, and any frame transformers the later sections read.
- Sim parameters — `dt`, `decimation`, `episode_length_s`, physics material and solver
  settings — when the task needs values other than the family's defaults.
- **The collision geometry of every mesh asset the scene spawns** — see *Asset preparation*
  below. This is authored, not inherited: an asset handed to the simulator unprepared gets a
  collision shape nobody chose.

## Asset preparation — run this BEFORE authoring the scene cfg

Every mesh asset is decomposed by CoACD ahead of time to at most **32 hulls**, and each part is
spawned as its own `convexHull` collider. Three steps; skipping the middle one silently undoes
the first:

```bash
P="${CLAUDE_PLUGIN_ROOT}/scripts/task-generator"
# 1. decompose      mesh -> part_*.obj + <asset>.coacd.json   (raise --threshold for fewer hulls)
.venv/bin/python "$P/coacd_decompose.py" --input mug.obj --out-dir mug_coacd
# 2. author the USD  parts -> one convexHull collider prim EACH
.venv/bin/python "$P/build_coacd_usd.py" --parts-dir mug_coacd --source-mesh mug.obj \
    --out mug_coacd.usd --name Mug --mass 0.15 [--kinematic]
# 3. point the scene cfg at the built USD; C6 + C8 verify it on the loaded stage
```

For an asset that arrives as a **USD or MJCF** rather than a mesh, `--inspect` it first; if it
is runtime-cooked or carries one collider for a visibly concave part, decompose its SOURCE mesh
and rebuild with step 2. Why this is mandatory, and what C8 checks, is under *C8 in detail*.

## Analysis terms

Recorded in `task-history.md` §1 **Analysis** before authoring — the scaffold enumerates them
and each must carry a real answer:

1. **Task interpretation** — the full reading of what the task is, beyond restating the prompt:
   what has to be true at the end, and what the intermediate stages are.
2. **Desired robot behavior** — the trajectory a competent policy would follow, stage by stage.
3. **Potential failures** — the physical ways this scene defeats a policy, especially
   collisions. *Insert-drawer:* the arm grasps the object but strikes the opened drawer front on
   the way in. *Hang-mug:* the handle catches the peg tip instead of clearing it. This is what
   §1 exists to get right — the geometry that makes those failures likely is chosen here, and
   §4's termination and §6's reward can only work around it, not undo it.

## Decisions to resolve

In precedence order: the user's `description` → the canonical example's value → a closer
sibling found by scanning the repo → ambiguous (batch into one `AskUserQuestion`).

| Decision | Notes |
|---|---|
| Robot asset | A user-supplied asset path is binding, never a preference (task-experience #2). |
| Object assets | Prefer the benchmark's stock cfgs; note spawn source (USD / URDF import path). |
| Collision decomposition | Per mesh asset: CoACD hull count (≤ 32), or `PRIMITIVE_ASSETS` + the reason an analytic collider is right instead. See *Asset preparation*. |
| Init poses | Where every entity starts. This is task **design**, not robot idiom — see traps. |
| Table / ground | The workspace surface the reward's height terms will key on. |
| Sensors | Contact sensors, frame transformers. Author them here if any later section reads them. |
| `dt` / `decimation` / `episode_length_s` | Deviate from the family default only with a reason. |

## API surface

`knowledge/references/task-generator/isaaclab-code-reference.md` → **Boot + instantiate** (how a smoke
brings the env up) and **Scene state** (reading entity state back out).

## Smoke S1

Eight checks. Every one runs on every invocation and each reports its own `[Cn] PASS|FAIL`
line —
none may be skipped, and a failure does not stop the rest, so one run tells you everything that
is wrong rather than only the first thing.

| # | Check | What it proves |
|---|---|---|
| C1 | registration + build | the id is in `gym.registry`, its `env_cfg_entry_point` resolves, `gym.make` returns an env at the requested `num_envs` |
| C2 | action space | `single_action_space` equals the summed dims of the action manager's active terms — a mismatch means an action term is silently dropped or double-counted |
| C3 | observation space | every declared group is present in what `reset()` returns, with matching per-env shape and finite values; nothing is returned that was never declared |
| C4 | max episode length | `max_episode_length > 0` and equals `ceil(episode_length_s / step_dt)` |
| C5 | timing | `sim.dt`, `decimation` and `step_dt` agree, and `episode_length_s` divides evenly by `step_dt` (a non-integral episode truncates mid-step) |
| C6 | collision geometry | every robot and object has at least one collider, and no concave collision mesh is being silently approximated by its convex hull |
| C7 | articulation limits | every articulated object's joints reach their limits under full drive, stop there, and stay there |
| C8 | collision provenance | every mesh collider is a **pre-decomposed CoACD part** within a 32-hull budget — not one prim handed to the simulator's own cooker |

- Runs **first**, always, in every mode — nothing downstream can be verified until the env builds.
- Substitutions: see `knowledge/templates/task-generator/smokes/smoke_s1.py.template`.
- It is also the build check a `reward-candidate-agent` runs before anything else, whether or
  not its candidate touched §1.

**C6 in detail.** For every collider prim under each robot and object, the smoke reads the
PhysX approximation and measures the mesh's *solidity* — its own volume over its convex hull's.
A convex mesh scores ≈ 1.0; a mug with a handle, a bin, an open drawer, or a gripper's fingers
score far below it. A concave mesh (`solidity < CONVEXITY_MIN`) whose approximation is
`convexHull` / `boundingCube` / `boundingSphere` is a **RISK** and fails the check: the scene
renders correctly while the contacts are wrong, and no later smoke can see it — the policy just
learns against geometry that does not exist.

Two legitimate resolutions, both of which are decisions to record in §1 Analysis:

1. **Re-decompose the asset** with CoACD (see C8) at a lower `--threshold`, so the split keeps
   the feature open, then re-run and check the solidity the smoke reports actually moved.
   Do **not** reach for the simulator's own `convexDecomposition` — that is what C8 rejects,
   and it is how the hull count reaches the hundreds.
2. **Accept it** by adding the prim path to `COLLISION_REVIEWED`, when the concavity is
   irrelevant to the task (a table's underside, a decorative rim nothing ever touches).

The list starts empty on purpose: accepting a risky collider costs a line and a reason, and
"the smoke was noisy" is not one.

**C8 in detail — decompose meshes yourself, ahead of time.**

Every mesh asset a task spawns must be decomposed by
`scripts/task-generator/coacd_decompose.py` **before** it reaches the simulator, to at most
**32 hulls**, with each part spawned as its own `convexHull` collider.

The alternative — hand the raw mesh over and set `mesh_approximation_name="convexDecomposition"`
— is the trap. The simulator's cooker optimises for fidelity, not shape count, so preserving one
small feature costs a hull budget in the hundreds. A hang-mug task cooked at
`max_convex_hulls=128` put half a million convex shapes in a scene at `num_envs=4096` and
overflowed the contact buffers. CoACD's near-convex decomposition keeps the topology that
matters (the hole stays open) at a fraction of the count.

The three-step pipeline is under *Asset preparation* above.

**Do not concatenate the parts into one multi-object OBJ and convert that.** IsaacLab's
`MeshConverter` merges the objects into a single mesh and PhysX wraps the result in ONE convex
hull — filling in exactly the concavity the decomposition preserved, while still satisfying a
manifest check, a hull-budget check and a no-runtime-cooker check. This was tried; C6 caught it
at `solidity=0.598`, and C8 now compares the manifest's part count against the collider prims on
the stage so it cannot recur silently.

**`--inspect` cannot read a binary `usdc`** — its token table is compressed and `pxr` is not
importable outside a simulator, so it returns *unknown* rather than a misleading "fine". C8,
which runs inside the sim, is the authority on what actually loaded.

The two are indistinguishable in a render, so C8 checks the two things that do differ:

1. **no collider prim asks the simulator to cook** (`convexDecomposition` / `meshSimplification`
   on the loaded stage), and
2. **a `<asset>.coacd.json` manifest sits beside the asset**, naming CoACD and a hull count
   within budget, cross-checked against the collider prims actually on the stage.

Without (2), "we used CoACD" is a claim nobody can check after the fact — which is the whole
reason the manifest exists rather than a comment.

`PRIMITIVE_ASSETS` exempts assets whose colliders are analytic or intentionally exact — a
table's box, a kinematic peg's cylinder. Analytic primitives are *cheaper and more accurate*
than any decomposition, so exempting them is correct; each entry is a decision to record.

Getting fewer hulls: raise CoACD's `--threshold`. Re-run C6 after any re-decomposition — a
coarser split can close a hole that the finer one kept open, and C6 is what notices.

**C7 in detail.** Drawers, cabinet doors, levers, valves — anything with a limited joint. A
joint limit is a constraint the solver *enforces*, not a wall the body cannot cross. C7 resets
the articulation, drives each limited joint toward each stop at the strongest pull its actuator
can deliver (position target half a range beyond the stop, plus the signed effort limit), steps
physics directly, and records the whole trajectory. Three ways that ends badly:

| Symptom | What it means | Where the fix goes |
|---|---|---|
| **overshoot** — travels past the stop, then snaps back | the limit is too soft to hold under full drive; the drawer opens further than it can and jitters, and a policy that learns to yank is rewarded for a state the task forbids | raise the joint's limit stiffness / damping, or the articulation's solver iteration counts |
| **short** — never reaches the stop | damping, friction or armature swallow the actuator's whole effort; "fully open" is unreachable, so a success predicate keyed on the limit can never fire however good the policy is | lower joint friction / armature / damping, or raise the effort limit |
| **unsettled** — still moving at the end | resting *on* the stop was never achieved — it is bouncing off it | same as overshoot; the stop is under-damped |

Bypassing `env.step` is deliberate: the action manager would overwrite the target every step and
a termination would reset the scene mid-test.

The robot goes in `ARTICULATION_EXCLUDE` — S2 and S2.5 own its joints. Everything else is
**discovered**, not listed, so a cabinet added to the scene later is covered without anyone
remembering to add it. Excluding a specific joint (`"cabinet.drawer_top_joint"`) is a decision
to record, same as accepting a collider.

None of this shows up in a random rollout: random actions almost never drive a joint to its
stop, so §1 is the only place it can be caught before a policy finds it for you.

## Failure → diagnosis → fix

| Symptom | Likely cause | First-pass fix |
|---|---|---|
| `gym.error.NameNotFound` | registration not imported, or id typo | check the package `__init__` actually imports the module holding `gym.register` |
| USD / asset not found | wrong path, or an asset that needs the URDF-import path | resolve the path against the benchmark's asset root; use the family's URDF importer for URDF sources |
| Entity missing from the scene at runtime | declared on the wrong cfg, or shadowed by `__post_init__` | confirm the attribute is on the scene cfg and that `__post_init__` does not overwrite it |
| Env builds but everything sinks / explodes | collision props, mass, or solver settings | check collision approximation and physics material before touching `dt` |
| C2 mismatch | an action term was added to `mdp/` but never listed on `ActionsCfg`, or two terms drive the same joints | reconcile `ActionsCfg` against the term dims the smoke prints |
| C3 group returned but undeclared | the group is built by the observation manager but missing from `single_observation_space` | usually a `__post_init__` that mutates `ObservationsCfg` after the space is computed |
| C4 off by one | `episode_length_s` does not divide by `step_dt` | fix `episode_length_s` (or `decimation`) — C5 names the exact ratio |
| C6 `NO COLLIDER` | the asset spawned without `collision_props`, or the USD has visual geometry only | add `sim_utils.CollisionPropertiesCfg()` to the spawn cfg; for a visual-only USD, the mesh needs cooking |
| C6 `RISK` | a concave mesh is cooked to its convex hull | pre-decompose with `coacd_decompose.py` and spawn the parts; confirm the solidity the smoke reports actually moved |
| C8 "left to the simulator's cooker" | the spawn cfg sets `convexDecomposition` | decompose the source mesh with CoACD and spawn each part as its own `convexHull` |
| C8 "exceeds the hull budget" | CoACD threshold too low | raise `--threshold` and re-run; then re-check C6, since a coarser split can close a feature |
| C8 "no .coacd.json beside" | the asset was converted by some other path | `--inspect` it; if it is runtime-cooked, rebuild from CoACD parts |
| C7 exceeded / bouncing | limit too soft for the drive force | raise limit stiffness+damping on the joint, or the articulation's `solver_position_iteration_count` |
| C7 short of the limit | effort swallowed by friction / armature / damping | lower those, or raise `effort_limit`, until the joint reaches its own stop |
| C7 "no joint position limits exposed" | the asset spawned without joint drive/limit properties | check the articulation cfg's `actuators` and the USD's `UsdPhysics.PrismaticJoint` / `RevoluteJoint` limit attributes |

## Known traps

- **Init qpos does not come with the robot.** When you substitute a robot because the base
  task's asset is missing, `SomeRobotCfg.replace(...)` silently inherits the substitute's
  default home pose. Port the base's `init_state.joint_pos` — joint values map 1:1 across
  same-family arms. See `knowledge/experiences/task-generator/task-experience.md` #1 for the full rule,
  including which values are embodiment geometry and must *not* be ported.
- **Actuator params are a §1 concern.** When S2.5 fails, the fix lands here — raise
  `stiffness` / `damping` / `effort_limit` (implicit) or `kp` / `kd` (explicit) toward the
  benchmark's own example for that robot. See `s2-actions.md`.
- **A render cannot show you who cooked the collision.** A runtime-cooked asset and a
  pre-decomposed one look identical; only the manifest and the prim count distinguish them.
  This is why C8 exists next to C6 rather than inside it — C6 asks whether the collision matches
  the mesh, C8 asks who made it and in how many pieces.
- **`convexDecomposition` in the cfg does not mean the collider is decomposed.** The token
  selects the cooker; PhysX then cooks at *its* defaults, and those defaults routinely bridge a
  handle hole or a narrow slot with a single hull. Grepping the USD for the token proves the
  intent, not the result — only C6's measured solidity proves the result. This has already cost
  a full tune: nine reward candidates trained against a mug whose handle hole was filled in.
- **A decomposed mesh needs a bigger PhysX broadphase.** Aggregate pairs scale with collider
  prims × `num_envs`, so a CoACD asset (up to 32 prims per mesh) overruns the default and PhysX
  silently drops contacts — which surfaces as a reward that will not converge, not as a config
  error (HangMug: return 1416 → 2292, std ±1113 → ±10, on the same checkpoint).

  ```python
  def __post_init__(self):
      super().__post_init__()
      self.sim.physx.gpu_total_aggregate_pairs_capacity = 256 * 1024   # 64 * 1024 is undersized
      self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 1024 * 1024 * 4
  ```

  Size above the MAX it asks for, not the first — the demand grows with contact activity
  (HangMug peaked at 143360, so `128 * 1024` would still have been too small):

  ```bash
  grep -o "totalAggregatePairsCapacity to [0-9]*" <log> | awk '{print $NF}' | sort -n | tail -1
  ```
