# The harness

A harness is the structured execution environment around an agent: how it reaches tools, preserves state, observes feedback, and verifies progress. Long-horizon autonomy depends on it at least as much as on the model, because most failures are not reasoning failures — they are underspecified executions, where the agent lacked the abstractions and feedback to pursue a goal reliably.

Robot RL is an unusually good fit for this view. The MDP already gives you stable interfaces:

$$\mathcal{M} = (\mathcal{S}, \mathcal{A}, P, r, \rho_0, \gamma, T)$$

State and action spaces must match simulator observations and control interfaces. Dynamics are determined by assets, control frequency, and physical parameters. Every one of those is separately checkable — and simulators produce executable feedback for checking them.

HARBOR specializes the general harness pattern into five pieces:

$$\mathcal{H}_{\text{RL}} = (\mathcal{H}_A,\ \mathcal{C},\ \mathcal{M},\ \mathcal{G},\ \mathcal{K})$$

## Agents

Context-isolated subprocesses assigned to bounded stages. Each operates on stage-local artifacts and retrieved knowledge, does its own implementation, and returns a compact summary to its caller.

Nine ship today, and each owns exactly one stage: `dependency-generator` (environment), `benchmark-generator` (sanity), `rl-integration-generator` (training stack), `task-generator` (§1–§5), `reward-tuning-agent` and `reward-candidate-agent` (§6), `dr-generator` (§7), `task-cloner` (isolation), and `rl-tuning-agent` (hyperparameters).

Isolation is the point. When a reward candidate fails a smoke test, the traceback, the log tail, and the failed frames belong to that candidate — not to the agent deciding what to try next. That separation is why `reward-tuning-agent` never writes reward code itself: it designs, and a worker implements, so implementation noise never enters the context making design decisions.

Dispatch depth is capped at two — main thread → orchestrator → worker — and exactly one agent, `reward-tuning-agent`, carries the ability to dispatch at all. Every other agent is a leaf that does its own delegated work. This is not a convention but an enforced invariant: a contract test fails the build if any other agent acquires the dispatch tool, because unbounded nesting is how an agent tree becomes impossible to reason about or to cancel.

## Commands

Reproducible operations exposed to agents, from primitives like `rl-run` to composed loops like `reward-tune`. Twenty-three ship, grouped by filename prefix — `env-*`, `probe-*`/`task-*`, `reward-*`, `rl-*`, and top-level utilities — because a filename prefix is the only grouping mechanism the runtime actually offers.

The same surface is invocable by different agents with different artifacts and gates — and by you, directly, which is what lets you assemble a workflow HARBOR does not ship.

Composability is why almost every command stays model-invocable. `rl-sweep` calls `rl-run`; `task-create` calls `reward-tune`; `test` drives the whole chain. A command that Claude cannot invoke cannot be called by another command either, so gating a building block silently breaks every chain above it. Exactly one command is gated — `reset-workspace`, the only irreversible operation — and the chains that need its behavior read its body and execute it rather than invoking it.

## Artifacts

Workflow state externalized into persistent, inspectable objects. They are the communication substrate between agents, which reduces reliance on transient context.

Three JSON specs carry the interfaces between stages — `install_plan.json`, `benchmark-spec.json`, `rl-suite-spec.json` — and each is written by one agent and read by everything downstream. Around them sit the per-run records: design histories, smoke verdicts, keyframes, metrics, checkpoints.

This is what makes long runs survivable. A tuning loop checkpoints its state every iteration; a killed agent, an interrupted session, or a resumed conversation picks up from the file rather than from memory. It is also what makes the system auditable: the artifact is the same thing you would have written by hand.

It has a second effect that is easy to miss. Because state is in files, an agent can be given a *small* view of it — the analysis without the validations, the verdict without the training log — which is what keeps a long loop's context from growing with its own history.

## Gates

Executable checks that decide whether a stage may advance. [See the gate catalogue →](/guide/gates)

## Knowledge

Templates, references, scripts, human heuristics, and accumulated experience from previous runs. Knowledge constrains generation, encodes simulator- and algorithm-specific contracts, and gives later agents access to what earlier attempts learned.

It splits three ways:

**Templates** are rendered into your repository — training scripts, configs, the per-section smokes, the SLURM launchers. Rendering rather than generating is deliberate: a smoke that is authored fresh each run is a smoke whose failures are novel each run.

**References** are decision aids an agent loads only when it needs them. Twenty-five files, organized so an agent authoring the action space reads `s2-actions.md` and not the observation guide. Each task-section reference carries the same six headings — what it authors, the decisions to resolve, API pointers, its smoke, failure → diagnosis → fix, and traps — so an agent knows where to look inside a file it has never read.

**Experiences** are append-only ledgers that accumulate across runs: one per authoring agent, plus a task library of self-contained task specifications indexed by embodiment. The library is what a new task is adapted *from* — a task-design search runs before any design work begins, so a new task starts from the closest prior one rather than from nothing.

Beneath all three sits the tool layer: 27 deterministic scripts invoked with plain Bash rather than through a server, because the heavy operations are backgrounded GPU jobs whose state must survive a killed agent. Files and exit codes give you that; an in-process server does not.

## The execution protocol

Every stage runs the same way:

1. The main agent retrieves relevant knowledge and current artifacts
2. It spawns a stage-local agent with bounded context
3. That agent creates or edits artifacts through standardized commands
4. Gates evaluate the result
5. **Pass** → commit the artifact, write logs and metrics back into knowledge
6. **Fail** → return the failed check, error message, and observed values to the stage agent for repair
7. After a fixed retry budget, the stage is marked unresolved and you are asked to intervene

Step 6 matters more than it looks. The agent is handed a diagnosis — *which* check failed and *what value* it observed — not a stack trace to interpret.

Step 7's budget is bounded by **progress, not attempts**, wherever a measurement exists. The section smoke loop escalates after two consecutive attempts that fail to move the measured quantity, rather than burning ten identical retries — and still has a hard backstop at ten. An agent repeating itself is a signal, and spending the budget before reading it wastes both.

## What this does and does not guarantee

HARBOR does not guarantee the semantic correctness of your final policy. No gate can prove the task you described is the task you meant.

What it does is turn a large class of RL engineering failures into **observable gate failures, before they propagate downstream**. A wrong action scale caught by an actuator tracking check costs ten minutes. The same bug caught after a twelve-hour training run costs a day, and is usually diagnosed as a reward problem.

The trade is not subtle, and it runs the direction people expect least: removing the gates makes the pipeline *faster*, and less reliable — a silent render-path defect ships undetected. Speed without verification is what produces the failure you find a week later.
