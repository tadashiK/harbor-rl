---
description: >-
  Iteratively tune the §6 reward of an EXISTING task with an ASYNC fixed-pool controller. A thin orchestrator: the main agent runs pre-flight + (standalone only) picks the design base, then dispatches `reward-tuning-agent`, which designs each candidate — a bounded §1–§5 task delta plus a complete reward — and dispatches one `reward-candidate-agent` per candidate to implement both, smoke every section touched, train + render, and score per-term curves + rendered frames → success_rate. Keeps `pool_size` candidates in flight (capped by `gpus` locally), loops until `success_rate ≥ success_threshold`, then promotes the winning design onto the source task. Isolation follows the effective pool: sequential over a base snapshot when it is 1, one slot clone per candidate when it is more. Use when the user types /harbor:reward-tune task=<id> [algorithm=<algo>] [pool_size=N] [gpus=N], or asks "tune the reward for task X".
argument-hint: "task=<id> [algorithm=<ppo|sac|td3>] [wandb=<project>] [pool_size=N] [gpus=N] [success_threshold=0.5] [timesteps_per_iter=N] [seed=N] [monitor_early_stop=true|false] [monitor_interval=240]"
---

# /harbor:reward-tune — Async-Pool Reward Tuning

A **thin orchestrator**. The main agent does pre-flight, selects the design base (standalone
only), and owns every question put to the user — then hands the loop to
`reward-tuning-agent`. It makes NO reward-design decisions.

```
Step 0  (main agent)          → pre-flight: nesting support / venv / specs / build / ffmpeg / wandb
Step 1  (main agent)          → standalone-only: task-library-search → library_refs
Step 2  (reward-tuning-agent) → DESIGN + DECIDE; dispatches reward-candidate-agent per candidate
Step 3  (main agent)          → answer a `needs_decision` return, or persist the final verdict
```

Dispatch depth is main → `reward-tuning-agent` → `reward-candidate-agent`, i.e. **2**.

## Required arguments

| Arg | Notes |
|---|---|
| `task` | Task ID. The task must build (see *Does the task build?*). Its §6 may be a placeholder or a real reward. |

## Optional arguments

**Rule: if an arg isn't passed, don't override the config — leave its default untouched.**
Only forward a train-command override (`seed=`, `total_timesteps=`, …) for args the user
explicitly set. The loop-control args (`pool_size`, `gpus`, `on_success`,
`success_threshold`, `n_frames`, `prompt_every_n_stuck`, `monitor_*`) are reward-tune's own
logic, so their defaults always apply.

| Arg | Default | Effect |
|---|---|---|
| `algorithm` | `ppo` | Picks `harbor/configs/rl/<algo>.parallel.yaml`. |
| `wandb` | `reward-tune-<task>` | W&B project; run names `iter_000`, … |
| `mode` | `local` | `local` (background bash on this host) or `cluster` (SLURM `sbatch` per candidate). |
| `pool_size` | `1` | Candidates in flight (`1` = serial). |
| `gpus` | detected | **`mode=local` only**: GPUs this tune may use. Caps the pool — `effective_pool = min(pool_size, gpus)` — because a candidate trains at full `num_envs`. In `cluster` mode SLURM does the allocating and `pool_size` stands. |
| `on_success` | `cancel` | First convergence: `cancel` or `drain` in-flight. |
| `success_threshold` | `0.5` | Stop when `success_rate ≥` this. |
| `timesteps_per_iter` | config | Per-candidate budget; unset → config default. |
| `seed` | config | Unset → config default (random). |
| `n_frames` | `12` | Frames read per render. |
| `prompt_every_n_stuck` | `5` | Ask the user after N non-improving completions. |
| `monitor_early_stop` | `false` | Default `false` = train every candidate to full budget. `true` opts each candidate into the mid-run curve monitor that early-stops an unambiguously doomed run. |
| `monitor_interval` | `240` | Seconds between monitor ticks. Must stay UNDER the prompt-cache TTL (300 s for a subagent): a tick landing after expiry pays a full context rebuild instead of a cheap read, so rarer ticking is dearer, not cheaper. |
| `monitor_soft_floor` | `0.5` | Budget fraction below which soft bad patterns are only watched, never killed; hard fails ignore it. |
| `spec_section` | (none) | §6 code block (reproduce mode); seeds iter 0 verbatim. |

## Step 0 — Pre-flight

```bash
cd "$(pwd)"
# Nested dispatch (main → tuner → candidate) requires Claude Code >= 2.1.219.
claude --version
test -x .venv/bin/python                                   || exit 1
test -f harbor/benchmark-generator/benchmark-spec.json     || exit 1
test -f harbor/rl-integration-generator/rl-suite-spec.json || exit 1
test -f harbor/create-task/task-implementation.md          || exit 1
command -v ffmpeg >/dev/null                               || exit 1
grep -q "machine api.wandb.ai" ~/.netrc                    || { echo "run /harbor:wandb-setup"; exit 1; }
[ "<mode>" = cluster ] && { command -v sbatch >/dev/null || exit 1; } || nvidia-smi -L || exit 1
```

Then build `<task>` per *Does the task build?* in `agent-conventions.md`. A bare
`gym.make('<task>')` is NOT a build check — it raises
`TypeError: missing 1 required positional argument: 'cfg'` for every manager-based task,
canonical ones included, so gating on it aborts the tune before it starts.

**Nesting is a hard requirement, not a preference.** On Claude Code < 2.1.219 a subagent
cannot dispatch its own subagent, so `reward-tuning-agent` would silently do the
candidates' work in its own context — exactly the contamination this design removes. If
`claude --version` is below 2.1.219, STOP and tell the user to upgrade
(`claude update`). Do not fall back to an in-line loop.

Resolve `task_dir = harbor/create-task/<slug>` (must already exist from
`/harbor:task-create` or a prior tune). Per-term reward logging is the
`reward-tuning-agent`'s STEP 0 — it runs the `/harbor:reward-add-log` flow itself if
`info["detailed_reward"]` is missing.

## Step 1 — Pick the design base

Do NOT hand-author `tune-state.json` — the agent owns it (creates on first run, resumes if
present). The orchestrator's only Step-1 job is the **design base**:

- **Standalone** (`/harbor:reward-tune` typed directly, no `spec_section`): run
  `${CLAUDE_PLUGIN_ROOT}/knowledge/references/task-library-search.md` once and pass the result as
  `library_refs`. (Empty library → `library_refs=[]`, pure creation.)
- **Reproduce** (`spec_section` given): skip the search — the spec IS the base; pass
  `library_refs=[]`.
- **Called from `/harbor:task-create`** (which already ran task-library-search at Step 1.5):
  forward the caller's `library_refs` unchanged; do NOT re-search.

## Step 2 — Dispatch the reward-tuning-agent

```
Agent(reward-tuning-agent, prompt={
  repo_path:          "<abs>",
  task:               "<task>",
  task_dir:           "<task_dir>",
  description:        "<from spec.json — the behavior to match>",
  algorithm, wandb, mode, pool_size, gpus, on_success, success_threshold,
  timesteps_per_iter, seed, n_frames, prompt_every_n_stuck,   # forward only the ones the user set
  monitor_early_stop, monitor_interval, monitor_soft_floor,
  library_refs:       [<Step 1 base(s)>],
  spec_section:       "<§6 Code block>"                        # reproduce mode only
})
```

It designs each candidate, dispatches a `reward-candidate-agent` to implement + train +
render + score it, keeps `effective_pool` in flight, promotes the winner onto the source
task, and returns `{status, best_iter, best_success_rate, best_total_return, iters}`.

## Step 3 — Answer decisions, then persist

**`status: "needs_decision"`** — the loop is stuck (`prompt_every_n_stuck` non-improving
completions). `AskUserQuestion` is unavailable inside subagents, so this is the main
thread's job: put `decision_context` to the user via `AskUserQuestion`
(continue / abort / change strategy), then resume the SAME agent with `SendMessage` so its
design context survives. Do not spawn a fresh tuner — that discards every design decision
made so far.

**`status: "converged" | "aborted"`** — confirm `tune-state.json:status` + `finished_at`
are set and `reward-history.md` has its "Final summary" block (the agent writes both).
Confirm the PROMOTE step landed: the source task carries the winning design (its §1–§5
delta and its reward), and the winner's own smoke set passed against the source. Then print:

```
reward-tune : <task>  (status: converged|aborted)
iters       : <N>  best=<best_iter>  best_success=<v>  best_total=<v>
pool / gpus : <effective_pool> of <pool_size> / <gpus>
task_dir    : <task_dir>
history     : <task_dir>/reward-history.md
best reward : promoted to the source task (verified by S6)
```

## Constraints

- **The orchestrator makes NO reward-design decisions.** Design, implementation, and
  scoring all live below it; the main agent does pre-flight, base selection, user
  questions, and final persistence.
- **Isolation follows `effective_pool`, not `pool_size`.** At 1 the tune is sequential: no
  clone, the task edited directly, restored from a `base/` snapshot before each candidate
  so candidates never stack. Above 1 each candidate gets its own slot clone covering the
  full editable surface, since candidates vary §1–§5 as well as the reward. A `pool_size=4`
  request on a one-GPU host therefore runs sequentially AND skips cloning.
- **In `local` mode the pool is capped by GPUs.** A candidate trains at the config's
  `num_envs`; two on one GPU thrash or OOM. In `cluster` mode SLURM allocates, so the cap
  is `pool_size` and rendering stays compute-side — never on a login node.
- **Train at default num_envs**; **render is folded into the candidate's job**.
- **No hard cap.** Stop on `success_rate ≥ threshold` or user abort; the user is asked
  every `prompt_every_n_stuck` non-improving completions.
- **Single source of truth per file** (all owned by `reward-tuning-agent`):
  `reward-history.md` (cumulative), `handoff-reward-tuning-agent.md` (latest best),
  `memories.jsonl` (findings), `tune-state.json` (metadata + pool + in_flight + per-iter).
  Candidates write only inside their own `iter_<NNN>/`.

## Examples

```text
# Serial (pool_size=1) — the default
/harbor:reward-tune task=IsaacLab-Franka-StackCup wandb=Isaac_exp

# Four candidates in flight, one per GPU, each on its own slot clone
/harbor:reward-tune task=IsaacLab-Franka-StackCup pool_size=4 gpus=4

# Same, on SLURM — one sbatch job per candidate, train + render on the compute node
/harbor:reward-tune task=IsaacLab-Franka-StackCup pool_size=4 mode=cluster

# Tighter threshold; let in-flight candidates drain on first success
/harbor:reward-tune task=IsaacLab-Franka-StackCup success_threshold=0.8 on_success=drain

# Resume (tune-state.json present)
/harbor:reward-tune task=IsaacLab-Franka-StackCup
```
