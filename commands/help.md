---
description: Show the full harbor plugin surface — slash commands, subagents, hooks. Use when the user types /harbor:help or asks "what can harbor do", "harbor help", "harbor overview", "list harbor features", "list harbor commands".
---

# /harbor:help — Plugin Overview

The user wants a tour of everything this plugin offers. **Print the static block below verbatim.** Do not paraphrase it — it is the canonical surface description.

## Action

1. Print the static block (everything between the `BEGIN STATIC` and `END STATIC` markers below) verbatim, omitting the markers themselves.

---

<!-- BEGIN STATIC -->

**harbor** — Set up Python simulation repos with uv and author RL tasks end-to-end.

## Slash commands

Grouped by prefix: `env-*` · task (`task-*`/`probe-*`) · `reward-*` · `rl-*` · utilities.

| Command | What it does |
|---|---|
| `/harbor:help` | This overview. |
| **env** | |
| `/harbor:env-install-uv [path]` | Set up an isolated `.venv/` for a simulation repo via uv; renders `harbor/dependency-generator/setup_uv.sh`, runs the import smoke, then dispatches `benchmark-generator`. |
| **task** | |
| `/harbor:probe-benchmark [repo=<path>]` | Author the family-level `create-task/task-implementation.md` guide for a benchmark repo. |
| `/harbor:probe-task task=<id> [output=<path>]` | Emit a portable per-task spec (verbatim §1–§7 code). Runs in a subagent to save context. |
| `/harbor:task-create name=<TaskID> (description=… \| from=<spec.md>)` | Author a NEW task, or reproduce one from a probe-task spec. Auto-bootstraps missing prerequisites (dependency-generator → benchmark-generator → rl-integration-generator, defaulting to custom_torch), then runs task-generator (§1–§5) → the reward-tune training loop (§6) → dr-generator (§7, opt-in — skipped unless DR is explicitly requested). |
| `/harbor:task-list [<task-id>]` | List / inspect tasks in the cwd-local benchmark (falls back to the registry via `list_tasks`). |
| `/harbor:task-clone op=create source=<id> dest=<id>` | Clone a task into an isolated, independently-editable copy under a new suffixed gym id (delete with `op=delete`). The collision-free isolation primitive behind parallel reward-tune candidates. |
| **reward** | |
| `/harbor:reward-tune task=<id> [algorithm=<algo>] [pool_size=N] [gpus=N] [mode=local\|cluster]` | Async-pool §6 reward tuning. Thin orchestrator dispatches `reward-tuning-agent` (designs + decides), which dispatches one `reward-candidate-agent` per candidate (implements + trains + renders + scores); repeat until success, then promote the winner onto the source task. `pool_size>1` runs candidates in parallel, each on its own slot clone; a sequential tune edits the task directly. |
| `/harbor:reward-add-log` | Wire per-reward-term decomposition into a benchmark repo without changing the env's native reward — asserts `composer(terms) == reward` every step. |
| **rl** | |
| `/harbor:rl-run task=<id> algorithm=<algo> [k=v ...]` | Train one trial. Wraps `harbor/scripts/rl/<impl>/train.py` with the repo's `<repo>/.venv/bin/python` and Hydra overrides. |
| `/harbor:rl-eval checkpoint=<path> [k=v ...]` | Evaluate a single trained checkpoint. Auto-infers task and algorithm from saved config. Writes `metrics.json` next to checkpoint. |
| `/harbor:rl-render checkpoint=<path> [k=v ...]` | Render a checkpoint to MP4 with inference-moved + frame-difference sanity checks. |
| `/harbor:rl-visualize checkpoint=<path> [k=v ...]` | Open a HEADED GLFW viewer for a trained agent. Requires `$DISPLAY`. |
| `/harbor:rl-sweep task=<list> algorithm=<list> [k=v1,v2,...]` | Cartesian-product sweep — each combination dispatches a sub-agent that runs `/harbor:rl-run`. |
| `/harbor:rl-tune task=<list> algorithm=<list> [mode=local\|cluster]` | Cartesian-product grid TUNING. One `rl-tuning-agent` subagent per cell (open-ended hyperparameter loop). |
| `/harbor:rl-add-trick <trick> [algorithm=<algo>]` | Apply an RL training trick to a chosen algorithm config in-place. |
| `/harbor:rl-list-tricks` | List available RL training tricks with descriptions + applicability (read-only). |
| `/harbor:rl-add-log` | Print the canonical metric-key contract every algorithm under `harbor/scripts/rl/<impl>/` must emit. |
| **utilities** | |
| `/harbor:plot spec=<yaml>` | Multi-panel mean±std W&B learning curves grouped by task × baseline. |
| `/harbor:wandb-setup` | Inspect / re-login / logout the host's Weights & Biases credentials (`~/.netrc`). |
| `/harbor:update-experience target=<name> (experience="…" \| file=<path>)` | Append a numbered bullet to an agent experience ledger (≤5-line hand-written bullets), or file a probe-task spec into the right `task-library/` embodiment folder. |
| `/harbor:reset-workspace repo=<path> [clean_inbenchmark_tasks=true\|false]` | **Destructive.** Remove all plugin output from a benchmark repo (`harbor/`, `.venv/`, `scripts/` carve-outs, caches) and (default) git-reset it back to its original clone. Runs in a subagent; dry-run + confirm, then a git-based smoke (incl. hidden files) must fully pass. |
| `/harbor:test [layers=1,2,3] [repo=<path>] [task=<id>]` | Plugin test runner. L1 contract + L2 unit = `pytest tests/`; L3 = e2e pipeline driving the task-create→train→reset chain module-by-module on an isolated clean worktree (resumable, fail-fast, per-module smokes + checks). |

## Subagents (heavy, multi-step work; main thread dispatches)

Invoke via `Task('<agent-name>')`. Dispatch depth is capped at 2: the main thread orchestrates, and exactly one agent — `reward-tuning-agent` — dispatches a worker of its own (`reward-candidate-agent`). Every other agent is a leaf. Requires Claude Code ≥ 2.1.219.

| Agent | Purpose |
|---|---|
| `dependency-generator` | **Entry point** for any "set up env for \<repo\>" task. Probes the repo, renders `harbor/dependency-generator/setup_uv.sh`, creates `<repo>/.venv/`, runs import smoke test, returns to main. |
| `benchmark-generator` | After dependency-generator finishes. Renders `scripts/{run_random,render_random}.py`, runs 2-tier smoke (L1 random / L2 render), captures suite spec into `harbor/benchmark-generator/benchmark-spec.json`. Training scaffolding is owned by `rl-integration-generator` (dispatched directly as a subagent once the spec is written). |
| `rl-integration-generator` | After benchmark-generator finishes. Renders `harbor/scripts/rl/<impl>/{train,eval,render,env_wrapper}.py`, `harbor/configs/rl/{ppo,sac,td3}{,.parallel}.yaml`, and `harbor/rl-integration-generator/rl-suite-spec.json`. Smokes each algorithm. |
| `rl-tuning-agent` | Per-algorithm hyperparameter tuning loop: train → eval → render → analyze metrics + behavior → suggest next config. Per-cell tune state under `harbor/rl_experiments/tunes/<tune_id>/<wandb_project>/`. |
| `task-generator` | Authors §1–§5 of a task (register/scene · actions · reset · goal+termination · observation) with per-section smokes. Dispatched by `/harbor:task-create`. |
| `reward-tuning-agent` | The §6 **designer**: decides each candidate's reward spec and what the results mean, dispatches `reward-candidate-agent` per candidate, promotes the winner onto the source task. The one agent that dispatches a worker of its own. |
| `reward-candidate-agent` | One candidate end to end: implement its §1–§5 task delta + reward → run the smokes for every section touched, plus S6 → train + render → score curves + frames → verdict. Never designs. |
| `dr-generator` | Authors §7 domain randomization across robot / object / observation-noise groups, then verifies each term by exact value read-back. Opt-in stage of `/harbor:task-create`. |
| `task-cloner` | Clones a task's editable surface into an independently-editable copy under a new gym id. The isolation primitive behind `/harbor:task-clone`. |

## Lifecycle hooks

| Hook | Effect |
|---|---|
| `PostToolUse` | Truncate noisy Bash stdout to keep the conversation lean. |
| `Stop` / `SubagentStop` | Append a one-line audit entry. |

## Where to learn more

- `README.md` — architecture, layout, prerequisites, contribution flow.
- `CLAUDE.md` — 6-layer mental model + "where things live".
- `agents/<name>.md` — each subagent's contract, phase breakdown, exit-code semantics.
- `knowledge/references/{env,benchmark,rl-integration}-generator/` — decision matrices, smoke contracts, install-plan schema, RL suite spec, decision protocol.

<!-- END STATIC -->
