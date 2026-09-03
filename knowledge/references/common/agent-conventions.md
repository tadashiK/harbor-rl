# Shared agent conventions

Conventions common to the authoring/scaffolding subagents (`dependency-generator`,
`benchmark-generator`, `rl-integration-generator`, `task-generator`, `reward-tuning-agent`,
`dr-generator`, `task-cloner`). Each agent keeps its OWN specifics inline (smoke names, file
paths, retry budgets, schemas); this file is the canonical statement of the generic shape so
those don't drift across agents. When an agent's body and this file disagree on a SPECIFIC
(e.g. a retry budget), the agent's body wins — this is the default, not an override.

## Smoke pass-criterion

A smoke = render the template into the agent's workspace, run it inside the repo's
`<repo>/.venv/bin/python`, and judge the result by a single line of stdout:

Smoke templates set `sys.stdout.reconfigure(line_buffering=True)`: a GPU-sim teardown can
kill the process before stdout flushes, and the verdict line IS the pass criterion — a
passing smoke that loses its line reads as a failure. Run them with `python -u` as well.

> **PASS = process exits 0 AND the final stdout line reads `<NAME> OK: ...`** (e.g. `S1 OK: ...`,
> `L2 OK: ...`, `CLONE OK: ...`). Anything else is FAIL.

Render smokes into the agent's own subdir (e.g. `<task_dir>/smokes/`), never the repo root.

### Smoke `num_envs` + indexing

`{{NUM_ENVS}}` is set per benchmark from `harbor/benchmark-generator/benchmark-spec.json:gpu_sim`:

- **gpu-sim (`gpu_sim == true`) → `2`.** Small enough to stay fast, ≥2 so per-env indexing
  bugs that hide at 1 env still surface. Render smokes render env 0.
- **non-gpu-sim → `1`.**

In an agent-filled check block, index `[0]` when comparing a scalar read — every env sees the
same point-interval reset at no-op DR, so any single env is representative. For batched
assertions use `torch.allclose` over the full `(NUM_ENVS, dim)` tensor.

### A smoke contract outranks a doc

Per-section smoke commands in a repo's `task-implementation.md` are reference material, not
the contract. Where the doc's smoke is weaker than the section's contract, use the contract
and (optionally) patch the doc surgically to match.

When a non-IsaacLab family is in play, substitute the family-equivalent API on the same
contract surface — the *checks* stay the same; the *call sites* differ.

## Diagnose-and-retry

On a step/smoke error, do NOT patch blindly:

1. Read the **actual stderr + the rendered file** that failed.
2. Form ONE focused hypothesis from the symptom — reason from the error, not from precedent.
3. Apply the **minimal surgical fix** and retry.
4. **Loop until it passes.** The exit condition is a passing smoke, not an exhausted counter.
5. Escalate via `AskUserQuestion` when the retry stops making progress — the agent's body states
   what that means for it (`task-generator`: two consecutive attempts that fail to move the
   measured quantity, or a 10-attempt backstop).

A retry that fails is not automatically a reason to stop; a retry that fails **the same way** is.
When a smoke reports a number — a residual, an overshoot, a frame-diff — that number is the
signal: still moving means the diagnosis is right and the fix was too small, unchanged twice
means the diagnosis is wrong and further attempts will not find it.

Never silently rewrite env/config files to make a smoke pass; fix the actual cause. In
particular, never make a smoke pass by weakening what it checks — a widened tolerance, a lowered
threshold, or a shortened settle time is not a fix, it is the failure with the alarm turned off.

## Does the task build?

`gym.make("<task>")` on its own is **not** a build check. Manager-based families construct the
env from a cfg object, so the bare call raises
`TypeError: __init__() missing 1 required positional argument: 'cfg'` for EVERY task in the
family — canonical ones included. Using it as a gate rejects healthy tasks.

Build through the family's own constructor. For `isaaclab-manager-based` (the app must be up
before the ids register):

```bash
.venv/bin/python -u - <<'PY' || exit 1
import argparse, gymnasium as gym
from isaaclab.app import AppLauncher
_p = argparse.ArgumentParser(); AppLauncher.add_app_launcher_args(_p)
_app = AppLauncher(_p.parse_args(["--headless"])).app
import isaaclab_tasks                                  # registers the Isaac-* / <benchmark> ids
from isaaclab_tasks.utils import parse_env_cfg
env = gym.make("<task>", cfg=parse_env_cfg("<task>", num_envs=2))
print("build ok", env.observation_space, env.action_space, flush=True)
env.close(); _app.close()
PY
```

`-u` **and** `flush=True` are both load-bearing: Kit's shutdown discards buffered stdout, so
without them the check exits 0 having printed nothing — and the spaces on that line are the
canonical build record. Same reason the smoke templates set `line_buffering`.

Other families substitute their own construction call on the same contract: build it, print the
spaces, close it. Budget ~2 min — a GPU-sim app launch is not free, so run this once per command
invocation, never per iteration.

Smokes, trainings and candidates all outlive a single turn. **Only an API request touches the
prompt cache**, so how you wait decides what the wait costs:

- a gap longer than the cache TTL → the whole conversation is re-cached at **write** price (1.25×)
- a tick inside the TTL → the conversation is a **read** (0.1×), 12.5× cheaper

A subagent's TTL is **5 minutes** (the 1h allowlist is `repl_main_thread*` / `sdk` /
`auto_mode` / `memdir_relevance`; subagents are not on it, and `ENABLE_PROMPT_CACHING_1H` must
be set before the process starts, so no agent can change this for itself).

**One protocol, everywhere:**

1. **Launch detached** — `setsid`, output to a log, a sentinel file written last.
   Without `setsid` a harness-side process-group cleanup can kill a multi-hour job mid-run.
2. **Stamp the start** (`date +%s > <dir>/started_at`) so every tick knows `elapsed`.
3. **Tick under the TTL** — 240 s is the default. Give the `Bash` call an explicit `timeout`
   above the interval; the default is 120 s and would cut the sleep short. Use
   `python3 -c "import time; time.sleep(N)"` — foreground `sleep` is neutered here and returns
   instantly, which spins the loop.
4. **Judge liveness by CPU, not output.** A computing process always accrues CPU; a frozen one
   never does, whatever it is or isn't printing. Log silence is never a kill criterion — some
   backends load assets or JIT-compile for minutes without emitting a line. Calibrate every
   other threshold from what the run exhibits between two ticks; any hardcoded cadence is wrong
   for some simulator.
5. **Stop at ~1.5× the expected budget**, and fail immediately on a dead process with no
   sentinel, or on non-finite metrics.

**Three ways of waiting are forbidden:**

| | why |
|---|---|
| foreground blocking call | caps at ~600 s, so a long job is killed mid-run *and* the gap re-caches |
| one backgrounded `until [ -f ... ]` | clears the cap but leaves the API idle, so the cache dies anyway |
| continuous polling (e.g. `Read`-ing a background task's `.output`) | keeps the cache warm at absurd cost — one candidate spent 439M cache-read tokens, 96 % of its total, watching a file that was not changing |

Measured before this protocol existed: 13.58M, 6.16M and 5.93M cache-write tokens burned by
three agents waiting the first two ways, and 439M cache-read by one waiting the third.

**Above ~50 minutes of waiting, a single rebuild is marginally cheaper than ticking** (12.5
ticks × 0.1 = 1.25). Tick anyway. The premium is small — and ticking is what makes the wait
survive a lost notification, and what lets a dead job be caught in minutes instead of hours.
Consistency and recoverability beat a marginal saving.

Each agent keeps only its own specifics inline: the sentinel path, the progress signal to read,
and any phase where a shorter interval is warranted.

## Process log (`*-history.md`)

Agents that keep a workspace history file write it **append-only as work progresses** (not in
one dump at the end):

- On entry: a header table (task_id / mode / sections / started_at / status).
- Per phase/section: an authoring block (decisions + files touched, with the source of each
  decision) and, where applicable, a smoke block (command + last ~50 lines of stdout + verdict).
- On exit: a final verdict block; update the header status in place.

## Universal rules

- **English-only** for all generated comments, logs, and receipts (CLAUDE.md constraint #1) —
  regardless of chat language.
- **Dispatch depth ≤ 2** (CLAUDE.md constraint #2). Exactly one agent carries the `Agent`
  tool — `reward-tuning-agent`, which dispatches `reward-candidate-agent`. Every other
  agent is a leaf and does its delegated work itself.
- **`AskUserQuestion` is unavailable inside a subagent** — Claude Code strips it whether or
  not the `tools:` list names it. An agent that needs a decision returns it to its caller
  (a `needs_decision` status plus enough context to ask), and the caller asks.
  *Migration status:* only `reward-tuning-agent` follows this. `dependency-generator`,
  `benchmark-generator`, `rl-integration-generator`, `task-generator`, `dr-generator`, and
  `rl-tuning-agent` still escalate through `AskUserQuestion`, so those escalation paths do
  not reach the user. Not yet fixed.
- **Surgical edits** — touch only what the task requires; don't refactor adjacent code or
  silently edit sibling tasks. Log every cross-section edit in the agent's history file.
