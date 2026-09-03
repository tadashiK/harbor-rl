# §2 — Action terms

How a policy action becomes a controller target, and whether the robot can actually reach it.
Two smokes, because those are two different failures: S2 checks the **mapping**, S2.5 checks
the **physics**.

## What §2 authors

- The `ActionsCfg` block: one term per controllable group (arm, gripper, base), each with its
  mode, `scale`, and any offset / clamp the mode takes.
- Custom action terms, when the design calls for a mode IsaacLab does not ship.

## Analysis terms

Recorded in `task-history.md` §2 **Analysis** before authoring:

1. **Does the desired behavior require end-effector orientation?** Answer from §1's *desired
   robot behavior* and *potential failures*, not from the mode list. Then follow the answer:

   | Answer | Mode |
   |---|---|
   | No — one fixed wrist orientation serves the whole trajectory | **fixed-orientation relative EE pose control**. Pin the orientation and expose only position (+ gripper). The smaller action space is the point: every DoF you do not expose is one the policy cannot waste exploration on. |
   | Yes — the wrist has to rotate, tilt, or align to something | **joint position control or EE pose control**, whichever the relevant task-library example used. Both are fine; matching the example beats picking freshly. |
   | Yes, but there is no relevant example | **joint position control.** |

   State which branch you took and the evidence for it. "Fixed orientation" is a claim about
   geometry — that no stage of the trajectory needs the wrist anywhere else — so say which
   orientation and why it clears every stage.

   This is the decision that has actually gone wrong: a hang-mug task was authored with position
   + binary gripper and no rotation DoF, and the locked wrist consumed 0.027 m of the handle's
   0.072 m clearance against an angled peg. Nothing downstream could recover it — no reward
   fixes an action space that cannot express the motion. Constrain the orientation when the task
   permits it; do not constrain it *because* the action space is smaller.

## Decisions to resolve

| Decision | Notes |
|---|---|
| Mode, per term | One of: `abs_joint_default_offset`, `abs_joint_no_offset`, `delta_joint`, `abs_ee_pose`, `delta_ee_pose`, `binary_gripper`, `joint_effort`, `non_holonomic`, `ema_delta_joint_pos`, `ema_delta_ee_pose`. |
| `scale` | Sets how far one unit of policy output moves the target. Carry the base task's value when adapting. |
| Default offset | Whether the action is relative to the robot's default joint pos. Changes the S2 expected formula. |
| EMA `alpha` | `ema_*` modes only — the smoothing constant. |

The two `ema_*` modes are the custom `EMACumulative*` action terms shipped with the plugin;
their render templates live at `knowledge/templates/task-generator/action_terms/`.

## API surface

`knowledge/references/task-generator/isaaclab-code-reference.md` → **Action manager**, and for the
custom terms, **Custom action term: EMA cumulative-delta joint position** and **Custom action
term: EMA cumulative-delta EE pose**.

Both smokes below are **mandatory**. They check two different things and neither substitutes
for the other: S2 checks the **mapping** (does an action become the target you meant?), S2.5
checks the **physics** (can the robot actually get there?). A task can pass one and fail the
other, and the failure that survives is the one nobody looked for.

## Smoke S2

Proves a known random action maps to the **expected controller target** for the mode chosen —
delta vs absolute, with or without the default offset. The template already carries the
per-mode expected formula; the agent picks the mode.

Substitutions: see `knowledge/templates/task-generator/smokes/smoke_s2.py.template`.

## Smoke S2.5 — actuator tracking

Proves the actuator **reaches** the target S2 verified: hold a reachable action, let it settle,
then assert achieved `joint_pos` ≈ commanded `joint_pos_target`. A large residual means the
robot's physics parameters are wrong, not the action term.

- **Runs after S2 passes, every time.** All three modes §2's analysis recommends are position
  control, so there is always a target to track.
- **A failure is fixed by tuning the parameters, not by widening the tolerance.** Raise
  `stiffness` / `damping` / `effort_limit` (implicit actuators) or `kp` / `kd` (explicit) in §1
  toward the benchmark's own example for that robot, re-run, repeat. Loosening
  `{{TRACKING_TOL}}` to make it pass is the one thing that is not a fix — it ships a robot that
  cannot hold the pose the policy will spend every episode commanding.
- **Multi-arm tasks run one S2.5 per arm** (`{{ROBOT_ASSET}}` = `robot_0`, `robot_1`, …).
- `{{HOLD_ACTION_EXPR}}` must be a **reachable static** target. `torch.zeros((NUM_ENVS,
  action_dim))` holds the home pose — the hardest gravity hold, and the right default for
  `abs_joint_default_offset` / `delta_joint` / `ema_*`. For `abs_ee_pose` /
  `abs_joint_no_offset`, zeros may command something unreachable; set a reachable target instead.
- Defaults: `{{SETTLE_STEPS}}` `150`, `{{TRACKING_TOL}}` `0.05` rad for joint space — tighten
  or loosen to the robot.

Full substitution list: `knowledge/templates/task-generator/smokes/smoke_s2_5.py.template`.

## Failure → diagnosis → fix

| Symptom | Likely cause | First-pass fix |
|---|---|---|
| S2 target off by the default joint pos | wrong offset variant of the mode | switch between `abs_joint_default_offset` and `abs_joint_no_offset` |
| S2 target off by a constant factor | `scale` mismatch | align `scale` with the base task's value |
| S2 `KeyError` on the term name | `{{ACTION_TERM_NAME}}` is not in `action_manager._terms` | read the active term list off the built env, not the cfg source |
| **S2.5 residual too large** | improper robot physics params | **fix §1**: raise `stiffness` / `damping` / `effort_limit` (implicit actuators) or `kp` / `kd` (explicit) toward the benchmark's own example for that robot, then re-run. Iterate on the parameters until it passes — never on the tolerance |
| S2.5 residual large only under load | `effort_limit` saturating | raise the effort limit before touching gains |
| S2.5 residual oscillates / overshoots | damping too low for the stiffness you just raised | raise `damping` alongside `stiffness`; a stiff undamped joint rings instead of settling |

## Known traps

- **S2.5 is the one section smoke whose fix is not in its own section.** Its failure is a §1
  robot/actuator cfg problem; editing the action term to make it pass hides a robot that
  cannot hold its own arm up.
- **Zeros are not always reachable.** For task-space modes, a zero action is a pose command at
  the origin, not "stay put".
- **A constrained action space is only free when the constraint is true.** Dropping orientation
  shrinks what the policy must explore, which is real value — but only if no stage of the
  trajectory needed the wrist elsewhere. When it did, the cost lands in §6, where it looks like
  a reward that will not converge and cannot be made to.
