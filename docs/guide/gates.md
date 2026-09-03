# Gates

A gate is an executable check that decides whether a stage may advance. Gates are the reason HARBOR is more reliable than a capable agent working without one: removing them makes the pipeline *faster* and less reliable, and lets a silent render-path defect ship undetected.

Gates come in two kinds. **Hard interface checks** verify that something is structurally correct — an import resolves, a shape matches, a value reads back. **Semantic checks** verify that something behaved — a rollout produced finite rewards, a controller tracked its command, rendered frames actually changed. The strongest of them end in a judgement the agent makes by looking at rendered frames; see [semantic correctness](/guide/semantic-correctness).

The distinction matters when one fails. An interface failure names the thing to fix. A semantic failure only tells you the behavior was wrong, which is why those gates report the *measured value* rather than a pass/fail bit — a tracking error of 0.4 rad and one of 0.02 rad are different problems.

## The catalogue

| Stage | Gate | What it catches |
|---|---|---|
| Dependency | Import + device | A venv that builds but cannot see the GPU, or a CPU-only torch build |
| Benchmark | Random rollout | Wrong observation or action shapes, non-finite rewards |
| Benchmark | Render to MP4 | A scene that never moves, which a scalar return hides completely |
| §1 Scene | Collision solidity, articulation limits | Objects that interpenetrate, joints outside their range |
| §2 Actions | Actuator tracking error | Commanded pose ≠ achieved pose: controller, dynamics, or IK errors |
| §3 Reset | Per-reset keyframe render | A start distribution that looks wrong to a human eye |
| §4 Termination | One check/validation pair per implemented predicate | A success predicate that never fires, or fires immediately |
| §5 Observation | Shape and finiteness | Terms that silently emit NaN |
| §6 Reward | Finite, non-constant, composition assertion | A reward that is passthrough, constant, or whose terms do not sum back to it |
| §7 DR *(coming soon)* | Exact value read-back at `num_envs=16` | Randomization that is configured but not actually applied |
| RL integration | Five-tier smoke | An algorithm that trains but produces no checkpoint, curves, or metrics |
| Render | Inference-moved + frame difference | A zeroed policy, a frozen IK solve |

The §1 row is the densest: it runs eight checks, not two. Registration and build, action-space dimension against the summed action terms, observation groups present and finite, a positive episode length that matches `ceil(episode_length_s / step_dt)`, timing consistency between `sim.dt`, decimation and `step_dt`, collider presence, articulation limits under full drive, and collision *provenance*.

That last one is worth naming. A mesh collider must be a pre-decomposed convex-hull set within a 32-hull budget — not a single concave mesh the physics engine will silently hull for you. A silently-hulled mug is a solid blob with no handle opening, and every downstream stage behaves as though the geometry were fine while the task is quietly impossible.

## Why frame differencing

Several gates check that rendered frames differ from one another, which looks paranoid until you have shipped one. A policy outputting all zeros and a simulation that never steps both produce a perfectly valid MP4 of a perfectly still scene. Every scalar metric agrees that nothing is wrong. The only signal that separates "the robot is standing still because that is optimal" from "the robot is standing still because nothing is running" is whether the pixels change.

The check is a mean per-pixel L1 between sampled frames against a threshold — cheap enough to run on every render, which is the property that matters. A check expensive enough to skip is a check that will be skipped.

## Why the reward composition assertion

Per-term reward logging is only useful if the terms are the reward. HARBOR asserts, every step:

```python
composer(info["detailed_reward"].values()) == env_reward
```

with `composer` being `sum` or `product` per task. If that fails, the decomposition being read during tuning is not what the policy is optimizing — so every conclusion drawn from it is wrong. It is a hard failure rather than a warning for that reason.

This is what makes the per-term curves trustworthy as evidence. A reward designer reads "the grasp term never rises" and concludes the gate never fires; if the terms did not actually compose back to the reward, that conclusion would be unfounded and the next candidate would be designed against a fiction.

## Verdicts are machine truth

Every smoke writes a `<smoke>.verdict.json` as it runs, eagerly. Agents read results out of that file rather than out of their own recollection of the output, and `check_task_history.py` diffs every claimed verdict in the design history against the file the smoke actually wrote.

This closes a specific failure mode: an agent that ran a check, misread it, and recorded a pass. Transcription stops being load-bearing.

Writing eagerly rather than at the end is deliberate too — a smoke that crashes halfway has still recorded the checks that ran, so a partial result is evidence rather than a blank.

## Retries are bounded by progress

A failing gate does not end a stage. The agent is handed the failed check, the error, and the observed value, and retries. What bounds the loop is **movement in the measured quantity**, not an attempt count: two consecutive attempts that fail to move it escalate to you, with a hard backstop at ten.

Attempt-count budgets get this wrong in both directions — they let an agent repeat the same failed edit nine more times, and they cut off a loop that was converging. An agent repeating itself is information, and the loop is designed to read it.

## Tools report honestly

The tool layer follows a convention that makes gate failures legible:

> **Exit 0 means the tool ran. Non-zero means it could not run at all.**

`score_iter.py` is the worked example. No `metrics.jsonl` exits 0 with `{"success_rate": null, "gate": "no_metrics"}` — the tool worked, and the honest answer is "ungradable". A missing `design.json` exits non-zero, because the caller passed something broken. Backwards, and an ungradable candidate reads as a broken script — or worse, a fabricated zero reads as a real result.

The distinction is load-bearing for the search above it: an ungradable candidate and a candidate that scored zero call for opposite next moves. One means re-run, the other means redesign.
