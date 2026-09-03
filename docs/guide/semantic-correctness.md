# Semantic correctness

A check can confirm that something is structurally valid and still tell you nothing about whether it is the thing you meant.

A rollout can produce finite rewards, correctly shaped observations, and a success predicate that fires at exactly the configured threshold — while the gripper closes on empty air, the mug spawns inside the rack it is supposed to hang on, or the humanoid "jumps" by scooting along the floor. Every number agrees. The task is wrong.

Numbers cannot close this gap, because the gap is between *what you configured* and *what you meant*, and only one of those is written down anywhere a script can read. So HARBOR renders the simulation to images and has the agent look at them.

## Two questions per stage

Wherever a design choice can be misconfigured *and* misconceived, the check is split in two:

| | Question | Form |
|---|---|---|
| Numeric pass | Is the state what you configured? | Assertions on tensors — positions, shapes, finiteness, thresholds |
| Visual pass | Is what you configured what you meant? | Rendered PNGs the agent opens and judges |

The reset section is the clearest case. Its numeric pass verifies that object positions fall inside the configured spawn ranges. Its visual pass renders one frame per reset and asks whether that start layout is a task worth solving — whether the block is reachable, whether the target is visible, whether the arm begins in self-collision. A spawn range can be perfectly implemented and still describe a scene nobody would want.

## Where visuals feed back

Visual judgement is not a final review pass. It is wired into the stages where a wrong answer would otherwise propagate:

| Stage | Artifact | What the agent judges |
|---|---|---|
| Benchmark sanity | `render_random.py` → MP4 | The scene moves at all — a frozen simulation renders a perfectly valid still video |
| §3 Reset | `smoke_s3_frames/` — one PNG per reset | The start distribution is a sane task |
| §4 Termination | `smoke_s4_frames/` — one keyframe per predicate | The predicate fires in the state it was meant to describe |
| §6 Render | `smoke_s6_frames/` — rollout keyframes | Nothing explodes, sinks through the floor, or drifts off-scene |
| Reward candidate | `render.mp4` → extracted `frames/` | What the trained policy *actually does*, written into the verdict as `behavior` |
| Checkpoint render | `/harbor:rl-render` | Inference moved, and consecutive frames differ |

Two properties make the table hold together. The frames come from a **fixed shared camera** — `smoke_s3_render`, `smoke_s6_render` and `render.py` all use the same framing on purpose, so layouts from different tasks, different attempts, and different reward candidates are comparable by eye rather than each inventing an angle. And the numeric half always runs first: a visual pass on a scene that failed its asserts is judging a broken simulation.

The reward loop is where this matters most. A candidate's scorer owns every number — success rate, per-term curves, peak detection. What it cannot compute is what the robot is doing, so the candidate agent extracts frames from its own rollout and reports behavior in those terms. "Reward went up" is not a behavior. "The arm reaches the cube and hovers, gripper never closes" is — and it tells the designer that the next candidate needs a grasp term, which no curve would have said.

## "Looks correct" is not a validation

A visual pass ends in a written judgement, and the agents are required to record *what they saw* rather than that they looked:

> "handle faces the rack, 18 cm of clear table between them" — a validation
>
> "looks correct" — not one

The distinction is enforceable. A specific claim can be contradicted by the frame it describes; a vague one cannot be wrong, which makes it worthless as evidence. This is the same reason [verdicts are machine truth](/guide/gates) — a check whose result cannot be disputed later is not a check.

In the reward loop this is not left to discipline. A candidate's render analysis is a **checklist with a fixed set of aspects**, each answered from the frames, and the scorer refuses to grade a candidate whose checklist has a hole:

| Aspect | What it answers |
|---|---|
| `checkpoint_watched` | `peak` or `final` — on a collapsed run these are different policies |
| `frames_usable` | is the subject actually in frame; if not, every answer below is void |
| `behavior` | what the policy does, against the requested description |
| `stage_reached` | the furthest rung of the term ladder, and where it stalls |
| `time_allocation` | where the frames cluster — "reaches at frame 2, hovers 3–11" |
| `reward_hacking` | a term being farmed instead of progressed |
| `physical_validity` | penetration, sinking, jitter, explosion |
| `termination` | fires as intended, never, constantly, or on a wrong-looking state |
| `actuation_quality` | jitter, oscillation, saturation |

Coverage is the half a machine can judge. It cannot tell whether "the gripper never closes" is true, but it can tell that nobody addressed `termination` — and an unanswered aspect is the common way a rollout analysis misleads the search. So a missing key, an unknown key (a typo would otherwise leave an aspect unanswered while the checklist looks full), and placeholder answers like `"clean"` or `"ok"` are all refused outright.

The aspects are not interchangeable. `physical_validity` routes to §1–§3 while every other answer points at the reward, so getting that one wrong sends the next iteration to repair the wrong layer entirely.

## What it costs

Almost nothing, which is why it is everywhere rather than reserved for a final review. The frames are a by-product of a render the pipeline already performs, the extraction is `ffmpeg`, and the judgement is a handful of images the agent already has the ability to read.

The expensive version of this check is the one that does not exist: discovering after a twelve-hour training run that the mug spawned inside the rack. §3's visual pass costs a few seconds and removes that class of failure before any GPU time is spent.

## Why the agent is the judge

Frame differencing catches a frozen scene, and a scalar threshold catches a predicate that never fires. Neither can catch a scene that moves incorrectly, or a predicate that fires on the wrong state. That judgement needs a model that can look at a picture of a robot and say whether it is doing the task — which is precisely what a vision-capable agent is for, and is the one check in the pipeline with no deterministic substitute.

Two rules keep it honest. The agent must open the images itself rather than infer from the numeric verdict beside them — a green assertion next to a wrong-looking frame means the assertion is measuring the wrong thing, and that is a finding, not a pass. And when a training run peaked and collapsed, the rendered checkpoint is the best one rather than the last, so the behavior report has to say which policy was actually watched. Conflating them misreports what the reward produced.

## Where it does not apply

The video is one environment — env 0, with a follow-cam for locomotion — so nothing about cross-episode variance is observable from it. A question like "does this succeed in most episodes or one in ten?" belongs to `success_rate`, computed from `metrics.jsonl` across every environment, not to the frames. Asking the visual pass for it would produce a confident guess, which is the one output this check must never generate.

That boundary is why the analysis schema accepts explicit uncertainty. "Unclear: only the arm is in frame" is a valid answer, and a schema with no way to say it would manufacture something worse.

## Related

- [Gates](/guide/gates) — the full catalogue of checks, and why frame differencing exists
- [Authoring tasks](/guide/tasks) — where the per-section frames are written
- [Tuning rewards](/guide/rewards) — how `behavior` feeds the next candidate's design
