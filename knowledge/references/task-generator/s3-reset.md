# §3 — Reset / events

What the world looks like at the start of every episode, expressed as `EventCfg` terms with
`mode="reset"`.

## What §3 authors

- The `EventCfg` reset terms: robot joint state, object poses, and anything else re-randomized
  per episode.
- **Always with their full range parameters present**, even when the ranges are point
  intervals — `dr-generator` widens the numbers later (§7) and needs the terms to exist.

## Analysis terms

Recorded in `task-history.md` §3 **Analysis** before authoring:

1. **Reset layout — feasibility.** Read §1's *desired robot behavior* and *potential failures*
   first; this term is where you act on them. The prompt fixes some of the layout and leaves the
   rest free, and **the free part is a design choice, not a default**. Among the layouts the
   prompt permits, pick the one that makes the task most achievable, and say why.

   Answer three things:

   - **What the prompt constrains, and what it leaves free.** "A mug on the table and a rack"
     fixes two objects and a surface. It says nothing about handle orientation, the distance
     between them, or which side of the workspace each sits on — all free.
   - **At least two candidate layouts, and which one you chose.** A layout is not just poses:
     it is what the first stage of the trajectory has to do before any progress is possible.
   - **Why the chosen one is easier, in terms of §1's failure list.** Not "it seemed cleaner" —
     which named failure it removes or makes less likely.

   Easier means: reachable without contorting, oriented so the first stage does not begin with a
   regrasp, clear of the obstacles §1 identified, and far enough from the goal that the task is
   not already solved at t=0. A layout that starts inside the success condition is worse than a
   hard one.

   This is the cheapest lever in the whole task. §6 spends real GPU hours discovering that a
   reward will not converge; §3 can remove the reason for a few minutes of thought — and an
   infeasible start is the most common reason a well-formed reward never learns. It is also the
   only lever that is *free*: the reward has to work around the layout, the layout does not have
   to work around anything.

## Decisions to resolve

| Decision | Notes |
|---|---|
| Which entities reset | Every entity whose start state matters to the reward or the success condition. |
| Layout | The chosen candidate from the analysis above, with the reason recorded. |
| Range params | In **create mode**, point intervals: `(K, K)` / `(1.0, 1.0)` so every episode is deterministic and S3 can assert exact read-back. |
| Reset ordering | If one entity's reset depends on another's (an object placed relative to the table), the term order matters. |

**Create mode is DR-aware but no-op.** Include the terms with full range params, set the
ranges to point intervals. The same applies to observation noise in §5
(`Unoise(n_min=0, n_max=0)`). This is what lets §7 widen ranges later without re-authoring.

## API surface

`knowledge/references/task-generator/isaaclab-code-reference.md` → **Forcing known reset / goal values**
(pinning a term's range to a point interval before `gym.make`) and **Scene state** (reading
the value back out of the simulator).

## Smoke S3 — two passes, both mandatory

A reset layout fails in two unrelated ways, and one pass cannot see both.

- **Numeric** (`smoke_s3.py`) proves the layout is the one you configured.
- **Visual** (`smoke_s3_render.py`) proves the layout you configured is the one you meant.

A mug whose handle faces away from the rack, a cube behind the arm's shoulder, a drawer front
sunk into the table — every one of those reads back exactly as configured and passes all five
numeric checks. Only a picture shows it. Run both; record both.

### Pass 1 — numeric

| # | Check | What it proves |
|---|---|---|
| C1 | injected values | every pinned reset value reads back out of the sim exactly |
| C2 | buffers populated | every articulation, rigid object and sensor has finite, correctly-shaped data after reset |
| C3 | robot configuration | the robot starts finite, **inside its joint limits**, and identical in every env |
| C4 | layout settles | objects rest where they were placed over `SETTLE_STEPS` of zero action |
| C5 | repeatable | two seed-0 resets produce the same start state |

- `{{INIT_OVERRIDES_BLOCK}}` — lines pinning `cfg.events.<term>.params["pose_range"]` /
  `["position_range"]` to point intervals.
- `{{EXPECTED_STATE_CHECKS}}` — one `expect(name, condition, detail)` call per injected value.
  **Use `expect()`, not bare `assert`**: an assert stops at the first failure, so a run tells you
  about one wrong pose instead of all of them.

**C4 is the check that earns its keep.** An object spawned intersecting the table is legal at
t=0 and gets ejected on the first physics step; an object spawned a millimetre above nothing
falls. Both read back perfectly and ruin every episode. C3's limit check is the same class of
bug on the robot: an init pose outside the joint limits has the articulation fighting its own
stops from step 0, which looks like an unstable robot and gets misdiagnosed as an actuator
problem in §2.

Full substitution list: `knowledge/templates/task-generator/smokes/smoke_s3.py.template`.

### Pass 2 — visual

Resets the env `N_RESETS` times, captures each frame, writes `reset_NN.png` into
`{{FRAMES_DIR}}`. Two mechanical checks — **V1** frames captured, **V2** frames not blank (an
all-black render reads as "captured" without being inspectable) — and then **the judgement is
yours**: open the PNGs and check each against §1's *desired robot behavior* and
*potential failures*. Specifically:

- Is every object **reachable** from the robot's start pose?
- Is each object **oriented** so the first stage of the trajectory can begin without a regrasp?
- Is the path §1 described **clear** of the obstacles §1 named?
- Is anything **intersecting** or **floating**?
- Is the task **already solved** at t=0?

Record what you saw, not that you looked. "Handle faces the rack, 18 cm of clear table between
them" is a validation; "layout looks correct" is not.

The camera is the shared harbor framing (same numbers as `render.py` and `smoke_s6_render.py`).
Do not set a per-task angle — comparability across tasks and across attempts is the point.

Full substitution list: `knowledge/templates/task-generator/smokes/smoke_s3_render.py.template`.

## Failure → diagnosis → fix

| Symptom | Likely cause | First-pass fix |
|---|---|---|
| Read-back is the cfg default, not the injected value | the override was applied after the cfg was consumed | pin the range **before** `gym.make`, not after |
| Read-back is close but not equal | reading before the sim settled, or reading a derived field | read immediately after `reset()`, from the raw `data.<field>` |
| `KeyError` on the event term | term name differs from the cfg | read `unw.event_manager.active_terms` off the built env |
| Value correct in env 0, wrong elsewhere | a range that is not actually a point interval | check both ends of every tuple |
| C3 "init pose is OUTSIDE the joint limits" | the ported init `joint_pos` belongs to a different robot | port the base task's init pose for **this** arm; see `s1-scene.md` traps |
| C3 "envs do not agree on the start pose" | a reset term applies to a subset of envs | check the term's `asset_cfg` / `env_ids` handling — a reset written for one env silently covers only env 0 |
| C4 object moved with nothing acting on it | spawned intersecting geometry (ejected) or unsupported (fell) | raise the spawn height above the support surface, or move it clear of what it overlaps |
| C5 two identical resets differ | state carried across the episode boundary | a reset term that reads live state instead of the configured default |
| Visual: layout looks nothing like the numbers | reading a pose in the wrong frame | `root_pos_w` is world; env-relative layouts need the env origin subtracted |

## Known traps

- **Dropping the range params because create mode does not randomize** leaves `dr-generator`
  nothing to widen, and §7 then has to re-author §3.
- **Position vs pose ranges** are different parameter names on different event terms; copying
  a block between terms without checking is a silent no-op.
- **A layout that passes every numeric check can still be unlearnable.** The numbers only say
  the sim got what you configured. Whether what you configured leaves the task achievable is the
  visual pass and the feasibility analysis — and it is the one §3 failure that costs §6 a full
  tuning run to discover.
- **Making the layout easier is not the same as making the task easier.** The prompt is the
  spec; the freedom is only in what the prompt did not fix. Starting the mug already on the peg
  is not a feasible layout, it is a different task.
