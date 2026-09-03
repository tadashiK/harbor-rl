# S6 — Render + visual check

Not a task section: the **whole-task gate** that runs after the section smokes. Everything up
to here checks one section in isolation; S6 checks that the assembled task is a physically
sane scene that looks like what was asked for.

Two stages, and the second is the point:

1. the script asserts **mechanical stability** — no explosion, no floor penetration, no
   non-finite state, and the video shows motion;
2. **the agent reads the keyframe PNGs** and judges whether the rollout matches the task
   description.

A scene that passes stage 1 and fails stage 2 is a fail. Mechanical stability is cheap to
satisfy and says nothing about whether the robot is standing inside the table.

## Analysis terms

Recorded in `task-history.md` §6 **Analysis** before running the gate. §6 authors nothing, so
these record what the gate is being held against — written **before** you look at the frames,
so the judgement is against a stated expectation rather than against whatever the frames happen
to show.

1. **What the rollout should show.** Under random actions: which entities are in frame, what
   the robot does, what stays put. Not "the task being solved" — a random policy solves nothing
   — but the scene being *physically right* while nothing useful happens.
2. **Camera.** Which framing the render used: the shared one, or a `{{VIEWER_OVERRIDE_BLOCK}}`
   (name it, and say why the shared angle could not show the scene — the render is no longer
   comparable to any other).

## When it runs

| Mode | When |
|---|---|
| `task-generator` create | **last**, after every section smoke has passed |
| `task-generator` edit | when an edited section can change the rendered scene — §1 / §2 / §3 |
| `reward-candidate-agent` | when the candidate's `task_changes.sections` include any of §1 / §2 / §3 |

## Smoke S6

Renders a random-action rollout to MP4 plus keyframe PNGs, into the per-task workspace next
to `task-history.md` — **not** `/tmp`; the MP4 is a user-facing artifact.

- **Camera: do not choose one.** The template already sets harbor's shared framing —
  `eye=(2.5, 2.5, 1.6)`, `lookat=(0.30, 0.0, 0.4)`, env-relative — the same numbers
  `render.py` uses for every trained-policy video. One angle across every render is what
  makes keyframes comparable between tasks AND between candidates of one task; a candidate
  that also moved the camera cannot be told apart from one that changed behavior.
  `{{VIEWER_OVERRIDE_BLOCK}}` is **optional and normally empty** — use it only when the
  shared angle genuinely cannot show the scene (a tall fixture it crops), and log that you
  did, because the resulting render is no longer comparable to any other.
  (`tests/contract/test_shared_camera.py` keeps the two definitions in sync.)
- `{{OUTPUT_MP4}}` = `<task_dir>/smoke_s6_render.mp4`, `{{FRAMES_DIR}}` = `<task_dir>/smoke_s6_frames`
- `{{N_STEPS}}` default `120`, `{{N_KEYFRAMES}}` default `8`
- `{{STABILITY_CHECKS}}` — OPTIONAL extra per-step asserts for task-specific penetration, e.g.

  ```python
  assert (unw.scene["object"].data.root_pos_w[:, 2]
          - unw.scene.env_origins[:, 2]).min() > TABLE_TOP_Z - 0.02
  ```

  Leave it empty for the generic explosion / floor checks only.

Full substitution list: `knowledge/templates/task-generator/smokes/smoke_s6_render.py.template`.

## What to look for in the keyframes

`Read` each `keyframe_*.png` and judge:

- **Does the rollout match the description?** Right robot, right objects, right workspace.
- **Penetration** — gripper through the object, object through the table, robot through itself.
- **Sinking** — an object slowly descending into a surface across keyframes is a collision or
  mass problem, not a rendering artifact.
- **Jitter / instability** — high-frequency shake means solver or actuator settings, not the reward.
- **Scale** — an object an order of magnitude too large or small usually means the wrong USD
  or a missing scale factor.

**Loop until the scene is stable AND visually matches the description.** This is the one smoke
whose pass criterion is a judgement, so it is also the one that gets waved through under time
pressure; a task that fails here trains fine and produces a policy that solves the wrong problem.

## Failure → diagnosis → fix

| Symptom | Likely cause | First-pass fix |
|---|---|---|
| Explosion / non-finite state | solver settings, or interpenetrating init poses | separate the init poses (§1/§3) before touching `dt` / substeps |
| Object sinks through the table | collision approximation or missing collider | check the collision mesh approximation on both bodies |
| Object rests visibly inside a surface | init pose z below the surface | fix §1 / §3 init poses |
| Video shows no motion | actions not applied, or the robot is fully constrained | confirm §2 first — S2 passing rules out the action mapping |
| Scene correct but empty of the object | asset failed to spawn silently | check the spawn path resolved (§1) |

## Known traps

- **Passing the asserts is not passing S6.** The visual judgement is a required step, not an
  optional extra.
- **Diagnosing an unstable scene as a reward problem.** A task that fails S6 cannot be tuned
  into working; the fix is always upstream in §1 / §2 / §3.
