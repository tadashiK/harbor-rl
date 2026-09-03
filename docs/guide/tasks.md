# Authoring tasks

A task in HARBOR is authored in seven numbered sections, and each one is gated separately. That decomposition is deliberate: a reward can only be as good as the signals the task exposes, so the sections that define those signals are settled and verified before reward design begins.

| § | Section | What it settles |
|:--:|---|---|
| 1 | Register / scene | Assets, collisions, articulation limits, the gym id |
| 2 | Actions | What the action space can express, and whether the controller tracks it |
| 3 | Reset | The start-state distribution |
| 4 | Goal / termination | The success predicate and the subgoal decomposition |
| 5 | Observation | What a reward term is allowed to key on |
| 6 | Reward | The term ladder, validated by actual training |
| 7 | Domain randomization | **Coming soon** — not authored in this release |

The numbering is a dependency order, not a filing convention. §4's subgoal decomposition *is* the reward's term ladder, and §5 fixes what a reward term is even allowed to key on — so a reward designed before those are settled is designed against signals that may not exist. Working the other direction is how a reward gets blamed for a task's defect.

## Create a task

```text
/harbor:task-create name=Isaac-Push-Block-Franka-v0 \
  description="Franka panda pushes a 5 cm wooden block from the table center to a
               target marker. Success when block-to-marker xy distance < 5 cm;
               horizon 200 steps."
```

Pre-flight verifies the dependency → benchmark → RL-integration chain and dispatches any missing stage first. Then `task-generator` authors §1–§5, running that section's behavioral smoke after each one and looping until it passes.

Two of those smokes are worth knowing about individually. §2 runs both S2 and S2.5, because they check different things: S2 verifies the *mapping* (a known action becomes the controller target you meant) and S2.5 verifies the *physics* (the actuator actually reaches it). A task can pass one and fail the other, and the failure that survives is the one nobody looked for. When S2.5 fails, the fix is §1's actuator parameters — never widening the tolerance until it passes.

The smoke loop is bounded by **progress, not attempts**: it escalates to you after two consecutive attempts that fail to move the measured quantity, rather than burning ten identical retries. It also escalates at a hard backstop of ten.

## What you get back

Each run writes a workspace under `harbor/create-task/<task-slug>/`:

| File | What it holds |
|---|---|
| `task-history.md` | The design record: per-section analysis of *why*, plus the validations that passed |
| `task-analysis.md` | The rationale half alone, with validations stripped — what reward design reads |
| `test-checklist.md` | Every check that actually ran, which is task-specific |
| `smokes/` | The rendered smoke scripts and the `verdict.json` each one wrote |
| `smoke_s{3,4,6}_frames/` | Reset layouts, per-predicate states, and rollout keyframes |

`task-history.md` is gated before the agent returns: every analysis term must be answered, every table cell filled, and every claimed verdict diffed against the smoke's own `verdict.json`. Transcription stops being load-bearing — the history reports machine truth, not the agent's recollection.

## The design record

`task-history.md` is not a log of what happened; it is the record of *why*, written before the code. Each section answers a fixed set of analysis terms first — §1's failure modes, §2's question of what the action space can express, §3's start layout, §4's subgoal decomposition, §5's permitted signals — and only then authors.

§3's analysis is the cheapest lever in the whole task, and it is where the format earns its keep. The prompt fixes part of the layout and leaves the rest free, and the free part is a design choice rather than a default: the agent must state what was constrained versus free, give at least two candidate layouts, and justify the chosen one against §1's named failure list. "It seemed cleaner" is rejected. An infeasible start is the most common reason a well-formed reward never learns, and §3 can remove it in minutes where §6 would spend GPU hours discovering it.

## Edit an existing task

Re-author individual sections surgically, leaving the rest untouched:

```text
/harbor:task-create name=Isaac-Push-Block-Franka-v0 sections=2,5 \
  description="Switch to EMA delta end-effector pose control; add object velocity
               to the observation."
```

## Reproduce a task elsewhere

Probe an existing task into a portable specification, then rebuild it from that spec:

```text
/harbor:probe-task task=Isaac-Insert-Drawer-Franka-v0
/harbor:task-create name=Isaac-Insert-Drawer-UR10-v0 \
  from=harbor/create-task/isaac-insert-drawer-implementation.md
```

The spec captures every design choice with verbatim code — scene, actions, reset, termination, observation, reward, and DR. In reproduce mode the reward is pasted verbatim at iteration 0 of the reward-tune loop and then validated by real training like any other candidate, because a reward that worked in one simulator's contact model is a hypothesis in the next one, not a guarantee.

Reproducing into a repo of the **same** simulator family works today. Reproducing across families — the sim-to-sim path, where the same task is adapted to a different simulator's APIs and physics with its intent preserved — is **coming soon** and is refused in this release.

## Clone a task

For A/B variants or isolated editing:

```text
/harbor:task-clone op=create source=<TaskID> dest=<TaskID>-variant1
```

The clone copies only the editable surface, rewires imports, registers the new id, and runs build, rollout, and per-term-logging smokes. A manifest records every created file so `op=delete` reverses it cleanly. The source is never edited.

## Next

[Tune the reward →](/guide/rewards)
