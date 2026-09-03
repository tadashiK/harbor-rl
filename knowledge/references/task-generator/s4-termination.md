# §4 — Goal + termination

What the task is asking for, and when an episode ends.

## What §4 authors

- `CommandsCfg`, when the goal is a sampled pose / target the policy observes.
- `TerminationsCfg`: the success term, failure terms (object dropped, robot out of bounds),
  and the time-out.
- Any **subgoal predicate** the design implements as a checkable condition.

## Analysis terms

Recorded in `task-history.md` §4 **Analysis** before authoring. §1 described the trajectory in
prose; §4 is where it becomes a set of predicates, and this analysis is the bridge.

1. **Subgoal decomposition.** Enumerate **every** subgoal on the way from the reset state to
   completion — the full decomposition, not only the ones you intend to implement. One row
   each: the subgoal, and its **success criterion as a testable predicate** (entity, quantity,
   threshold), not as a description. "Mug is grasped" is a description; "both finger joints
   below 0.02 m of closure AND mug height above the table by > 0.03 m" is a criterion.

2. **Final goal + success criterion.** The end condition, stated the same way. Then answer the
   question that decides whether the whole task is measurable: **what degenerate state also
   satisfies this predicate?** A mug resting *beside* the peg, an object at the right height
   because it is still in the gripper, a drawer "closed" because it was never opened. Name them
   — this is what §6 will exploit if you leave it open, and an inflated `success_rate` is the
   most expensive kind of wrong number in the pipeline.

3. **Which subgoals are implemented, and why the rest are not.** Not every subgoal from term 1
   becomes a term here — some belong in §6 as shaping, some are implied by others, some are not
   worth the cost. State which are implemented as `§4` predicates and give a reason for each
   omission. An unimplemented subgoal is a legitimate choice; an unnoticed one is not.

4. **Terminations.** Every `DoneTerm`: its name, its trigger condition, and its **kind** —
   success, failure, or time-out. Include the time-out even when it is the only one. Failure
   terms are where episodes are silently thrown away, so say what each one costs the policy if
   it fires early.

Terms 3 and 4 together define the predicate list the smoke tests: N implemented predicates → 2N
checks, one numeric and one visual each.

## Decisions to resolve

| Decision | Notes |
|---|---|
| Success condition | The geometric / state predicate that means "done, correctly". The reward's `success_term` will key on this. |
| Failure conditions | Dropping, tipping, leaving the workspace. Each becomes a `DoneTerm`. |
| Time-out | `episode_length_s` from §1; whether time-out is the *only* termination. |
| Goal representation | A command manager term, a fixed target, or implicit in the scene. Determines whether §5 exposes a goal observation. |

## API surface

`knowledge/references/task-generator/isaaclab-code-reference.md` → **Termination / Command managers**
and **Forcing known reset / goal values**.

## Smoke S4 — every predicate, twice

The predicate list from analysis terms 3 and 4 — implemented subgoals, the final success
condition, every termination term — is tested **one by one**, and each one **twice**:

| | |
|---|---|
| `C<i>` | **numeric** — predicate *i* is FALSE at the start state and TRUE once the scene is driven into the state that should trigger it |
| `V<i>` | **visual** — a keyframe of that forced state, written to `FRAMES_DIR` for you to open |

N predicates → 2N checks. Plus one outside the pairs:

| | |
|---|---|
| `G1` | the goal is sampled deterministically and, if §5 exposes it, lands at the right obs slice |

**Why each half exists.** A numeric pass says the predicate fired in a state you constructed —
not that the state you constructed is the one you meant. A success predicate satisfiable by a
mug resting *beside* the peg passes every assertion, then reports an inflated `success_rate`
for an entire tuning run. The keyframe is how that gets caught, at one render per predicate.

**The FALSE-at-start half is not a formality.** A predicate that is always true is the single
most expensive §4 bug: §6 optimises against it for hours and reports success the whole time.

### Filling it

- `{{PREDICATE_DEFS}}` — one module-level `def force_<name>(unw)` per predicate, driving every
  env into the triggering state. Zero velocities as well as poses, or the object arrives at the
  right place already moving away from it.
- `{{PREDICATE_LIST}}` — the `PREDICATES = [(label, force_fn, fired_fn), ...]` list.
  `fired_term("<term>")` is provided for `TerminationsCfg` terms; a subgoal that is not a
  termination term takes any `unw -> bool tensor (num_envs,)`.
- `{{GOAL_CHECK_BLOCK}}` — `expect_goal(name, condition, detail)` calls. **Not every task has a
  CommandManager**: `commands = None` (implicit, fixed goal) is a standard manipulation idiom —
  the task-library's insert-drawer base uses it — and `command_manager.get_command(...)` does
  not exist for those. Derive the goal from scene state instead.

`write_root_pose_to_sim` **rejects a `(1, 7)` tensor when `num_envs > 1`** — broadcast:

```python
pose = torch.tensor([px, py, pz, qw, qx, qy, qz], device=unw.device)
unw.scene[<obj>].write_root_pose_to_sim(pose.expand(unw.num_envs, -1).contiguous())
```

Camera is the shared harbor framing — these keyframes are read against §3's reset frames and
§6's rollout frames, which only means something from one viewpoint.

Full substitution list: `knowledge/templates/task-generator/smokes/smoke_s4.py.template`.

### Reading the keyframes

Per predicate, against §1's *desired behavior* and *potential failures*:

- Is the forced state the one the subgoal actually describes, or a pose that merely satisfies
  the arithmetic?
- Would a policy reaching this state be **making progress**, or has it found a shortcut?
- For the final goal: is this a state you would accept as the task being done?

Record what each frame showed, per predicate. "Mug hangs by the handle, peg through the hole,
mug not touching the table" is a validation; "success looks right" is not.

**Visualize sibling.** `smoke_success_visualize.py` is the headed, interactive version for the
final-goal predicate: `HEADLESS=0`, `NUM_ENVS=1`, never exits, re-pinning every relevant entity
each frame so gravity and integration drift never accumulate. It does **not** run during Phase B
regression — it is a tool for the user:

```bash
cd <repo>
HEADLESS=0 NUM_ENVS=1 .venv/bin/python -u <task_dir>/smokes/smoke_success_visualize.py
```

Its `{{SUCCESS_HOLD_BLOCK}}` must **also pin the anchor / goal entity** — the one the S4 force
function can skip because S4 runs a single step — and must zero linear *and* angular velocities
every frame.

## Failure → diagnosis → fix

| Symptom | Likely cause | First-pass fix |
|---|---|---|
| `C<i>` "did not fire in every env" | the forced state does not satisfy the predicate | print the predicate's inputs at the forced state **before** adjusting thresholds — the usual answer is the force block, not the predicate |
| `C<i>` "TRUE at the start state" | the predicate is trivially satisfiable | tighten it. This is not a smoke problem; it is the bug the check exists for |
| `C<i>` fires in some envs only | the force block writes a `(1, 7)` pose that lands on env 0 | broadcast as above |
| Some other term fires first | the forced state trips a failure term too | force a state that is unambiguous for the term under test |
| `write_root_pose_to_sim` shape error | `(1, 7)` at `num_envs > 1` | broadcast as above |
| `V<i>` shows a state you did not intend | the force block satisfies the arithmetic, not the intent | fix the force block, then re-read whether the predicate should have accepted it |
| `V<i>` blank / no frame | `enable_cameras` or `render_mode` | both are set in the template; a blank frame means the camera sees nothing |
| `G1` goal absent from obs | §5 does not expose the command | fix §5, not the termination |

## Known traps

- **"A termination fired" is not "the right termination fired."** Every predicate is checked by
  name against the per-term dones, never by "the episode ended".
- **A success predicate that is too loose passes its numeric check and then makes §6's success
  term fire trivially** during tuning, producing an inflated `success_rate` that survives until
  someone watches a video. Analysis term 2 — *what degenerate state also satisfies this?* — is
  where that gets caught cheaply; `V<i>` is the second chance.
- **An unimplemented subgoal is a choice; an unnoticed one is a gap.** Term 1 enumerates all of
  them precisely so term 3 can say, on the record, which ones §4 does not implement and why.
