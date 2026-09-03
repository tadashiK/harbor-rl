# The end-to-end workflow

Every other page in this section covers one stage. This one is the map: what runs, in what order, from an empty checkout to a tuned policy.

The order is not a matter of taste. Each stage consumes an artifact the previous one wrote, and each ends in a gate that must pass before the next may start. A reward cannot be designed against observations that do not exist yet; a training run cannot be launched against a task that does not build.

```
  install          probe            author           design           train
  packages   →   benchmark    →      task      →     reward     →    policy
     │              │                 │                │               │
 dependency-    benchmark-       task-generator   reward-tuning-    rl-run /
 generator      generator        §1..§5           agent §6          rl-tune
     │              │                 │                │               │
  imports +     rollout +        per-section      success_rate     checkpoint +
  device        render MP4       smokes           ≥ threshold      curves + MP4
```

## The one-sentence path

You do not have to run these individually. Point Claude Code at a repository and describe what you want:

```text
/harbor:task-create Set up the env for https://github.com/isaac-sim/IsaacLab, then create a task where
a Franka pushes a 5 cm block to a target marker. Success is block-to-marker
distance under 5 cm. Then train PPO on it.
```

HARBOR walks the chain and stops at the first gate that fails, leaving the workspace intact so you can inspect what happened and resume from there.

`/harbor:task-create` runs its own pre-flight over stages 1–3 and dispatches whatever is missing, so pointing it at a fresh checkout works — you do not have to run the setup stages first, only to know that they happened.

## The explicit path

### 1. Environment

```text
/harbor:env-install-uv
```

`dependency-generator` probes the repository, builds an installation plan from its own docs, renders `harbor/dependency-generator/setup_uv.sh`, and executes it to create `.venv/`.

**Gate:** the package imports and the expected device is visible.

### 2. Benchmark sanity

`benchmark-generator` renders `scripts/run_random.py` and `scripts/render_random.py`, then runs a random-action rollout and a render to MP4. It captures `harbor/benchmark-generator/benchmark-spec.json` — the task inventory everything downstream reads.

**Gate:** finite rewards with correct shapes, and rendered frames that differ from one another.

### 3. Training stack

`rl-integration-generator` renders `harbor/scripts/rl/<impl>/`, `harbor/configs/rl/`, and `rl-suite-spec.json`. The default algorithm source is `custom_torch`, a self-contained tree with no external RL library to install.

**Gate:** each algorithm passes a five-tier smoke that produces a checkpoint, metrics, curves, and a render through the same code paths production uses.

Stages 1–3 are covered in detail in [your first benchmark](/guide/first-benchmark).

### 4. Probe the benchmark

```text
/harbor:probe-benchmark
```

This reads the repository's own task code and writes `harbor/create-task/task-implementation.md` — the per-family authoring guide, describing how *this* benchmark registers tasks, configures scenes, wires actions, and composes rewards.

It is what makes the next stage benchmark-agnostic. Task authoring reads this file rather than assuming IsaacLab conventions, which is why the same task description works against ManiSkill or Genesis. `/harbor:task-create` runs it automatically if the file is missing.

**Gate:** none — the output is a guide, not an executable artifact. It is validated indirectly, by the section smokes that follow.

Run it once per benchmark, not per task. Every task authored in that repository reads the same guide.

### 5. Author the task (§1–§5)

```text
/harbor:task-create name=Isaac-Push-Block-Franka-v0 \
  description="Franka pushes a 5 cm block to a target marker; success when
               xy distance < 5 cm; horizon 200 steps."
```

`task-generator` authors scene, actions, reset, termination, and observation, running that section's behavioral smoke after each and looping until it passes.

**Gate:** one per section — collision solidity, actuator tracking, reset layout, one check/validation pair per success predicate, observation shape and finiteness. Several end in a [visual judgement](/guide/semantic-correctness).

Before authoring anything, the orchestrator searches the task library for the closest prior task and passes it down as the design base. That is its only design act — §1–§5 decisions belong to the authoring agent, not to the thing dispatching it. Starting from the nearest of 52 existing task specifications is why a new task rarely starts from a blank file.

Details in [authoring tasks](/guide/tasks).

### 6. Design the reward (§6)

The same command continues into the reward loop, or run it standalone against an existing task:

```text
/harbor:reward-tune task=Isaac-Push-Block-Franka-v0 pool_size=4 gpus=4
```

`reward-tuning-agent` designs candidates — each a bounded §1–§5 task delta plus a complete reward — and dispatches one `reward-candidate-agent` per candidate to implement, smoke, train, render, and score it.

**Gate:** `success_rate ≥ success_threshold`, measured by actual training rather than by inspection. The winning design is then promoted onto the source task and re-verified with its own smoke set.

This is the only stage whose gate costs GPU hours, which is why everything before it is gated cheaply: each earlier check exists to keep a defect from being discovered here, where it looks like a reward problem and costs a training run to find.

Details in [tuning rewards](/guide/rewards).

### 7. Domain randomization (§7) — coming soon

Not available in this release; `dr-generator` is never dispatched, and asking for DR gets you a note saying so rather than a silent skip. When it ships it will wire every available randomization term across robot, object, and observation-noise groups.

The slot is left DR-aware in the meantime — §3 range params and §5 observation noise are authored at point intervals — so DR can be added later without re-authoring §1–§6.

**Gate:** exact value read-back at `num_envs=16`, re-checked after reset — which catches randomization that is configured but never actually applied.

### 8. Train

```text
/harbor:rl-run task=Isaac-Push-Block-Franka-v0 algorithm=ppo
```

Artifacts land in `harbor/outputs/<algo>_<task>_<timestamp>/`: checkpoint, `metrics.jsonl`, TensorBoard logs, curves, and `render.mp4`.

### 9. Tune

```text
/harbor:rl-tune task=<list> algorithm=<list>
```

One `rl-tuning-agent` per (task, algorithm) cell, each running an open-ended loop: default-config baseline, then a tricks pass, then log-driven hyperparameter edits, stopping when the running best is not beaten for N consecutive iterations. `/harbor:rl-sweep` is the fixed-grid alternative when you already know what you want to compare.

`/harbor:rl-sweep` is the fixed-grid alternative — use it when you already know the comparison you want; use `rl-tune` when you do not.

Details in [training and tuning](/guide/training).

## Resuming

Every stage writes its state to disk before advancing, so a chain that stops at stage 6 does not cost you stages 1–5. Re-running a completed stage is safe: `setup_uv.sh` is idempotent, and `/harbor:task-create` with `sections=` re-authors only what you name.

This is what [artifact-centric](/guide/harness) means in practice — the workflow survives a killed agent, a resumed session, or a twelve-hour training job, because none of its state lives in a conversation.

One asymmetry to keep in mind: killing the agent does not kill work it has already handed to something else. A submitted SLURM job outlives the agent that submitted it, which is why the jobid is written to disk — so it can be cancelled deliberately rather than orphaned.

## Next

[Author a task →](/guide/tasks)
