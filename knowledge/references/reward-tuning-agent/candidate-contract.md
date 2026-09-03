# Designer ↔ candidate contract

The only channel between `reward-tuning-agent` (designs, decides) and
`reward-candidate-agent` (implements, smokes, trains, scores). Read by both.

The split exists to keep implementation noise out of the context that designs the next
candidate. That only holds if the boundary is respected, so the boundary rule comes first.

## Boundary rule

**Crosses:** the request below, and the verdict below. Nothing else.

**Never crosses:** rendered frames, log tails, tracebacks, smoke-retry churn, file diffs.
Those stay in the candidate's context and on disk under `iter_<NNN>/`. The designer may
`Read` an artifact **on demand** when a verdict looks surprising — pull, never push.

## Write scopes

| Writer | May write |
|---|---|
| candidate | `iter_<NNN>/` **plus** the paths in `isolation.editable_files` |
| designer | every shared file — `tune-state.json`, `reward-history.md`, `memories.jsonl`, `handoff-reward-tuning-agent.md` — plus slot creation/teardown, the `base/` snapshot, and the source task at PROMOTE |

`editable_files` is what makes concurrency safe, and it means different things per mode:

- **`kind: "clone"`** (`effective_pool > 1`) — the candidate's own copies, from
  `clone_task.py`'s manifest. Private to this slot; siblings cannot see the edits.
- **`kind: "sequential"`** (`effective_pool == 1`) — the task's own files. Safe because
  nothing else is in flight. The designer restores them from `base/` **before** dispatching,
  so each candidate starts from the base task rather than stacking on its predecessor.

A candidate that appends to `memories.jsonl` races its siblings even in clone mode. It
returns `findings` instead; the designer is the single writer.

## Request (designer → candidate)

```json
{ "repo_path": "/abs/path/to/benchmark",
  "task": "<gym id this candidate builds, trains and smokes — its slot clone, or the source task when sequential>",
  "task_dir": "<repo>/harbor/create-task/<slug>",
  "iter": 7,
  "mode": "local | cluster",
  "isolation": { "kind": "clone | sequential",
                 "editable_files": ["<repo-relative>", "..."] },
  "design": { "...": "design.json verbatim — see below" },
  "train": { "algorithm": "ppo", "config_name": "ppo.parallel",
             "timesteps": 2000000, "seed": 3,
             "wandb_project": "reward-tune-<task>", "wandb_run_name": "iter_007",
             "cuda_device": 1 },
  "description": "<the behavior to match — SCORE judges the rollout against this>",
  "n_frames": 12,
  "monitor_early_stop": false, "monitor_interval": 240, "monitor_soft_floor": 0.5 }
```

`train` keys the caller did not set are absent; the candidate then leaves the config
default untouched rather than passing an override. `cuda_device` is present in
`mode: "local"`, where the designer assigns it, and absent in `mode: "cluster"`, where
SLURM allocates.

## `design.json` — a candidate is a task design AND a reward design

```json
{ "kind": "structured",
  "task_changes": {
    "sections": [1, 4],
    "changes": [{ "section": 1,
                  "what": "add fingertip ContactSensorCfg on panda_{left,right}finger",
                  "why": "the grasp gate needs real contact, not palm proximity" }] },
  "reward": { "composer": "sum",
              "success_term": "<term whose firing = task success>",
              "terms": [{"name": "...", "weight": 0.0, "shape": "...", "gate": null}],
              "budget_rationale": "..." } }
```

`task_changes.sections` is the list of §1–§5 sections this candidate modifies. It drives
smoke selection, so it must be accurate: a section changed but not listed goes unverified.
`sections: []` is a reward-only candidate — the common case, and the cheapest.

`task_changes` are **bounded deltas** with a stated reason (add a sensor, add an
observation term, widen a bound, adjust a termination), not a redesign. Wholesale §1–§5
authoring belongs to `task-generator`; here it exists to unblock a reward that cannot
otherwise express the behavior.

## Verdict (candidate → designer)

Written to `iter_<NNN>/verdict.json` **and** returned as the agent's final message. The
numeric fields come from `scripts/reward-tuning-agent/score_iter.py`, which owns the
`success_rate` arithmetic so every candidate is scored identically.

```json
{ "iter": 7,
  "status": "scored | task_smoke_failed | reward_smoke_failed | train_failed | early_stopped",
  "smokes": {"S1": "pass", "S4": "fail", "S6": "skipped"},
  "success_rate": 0.31,
  "total_return": 123.4,
  "per_term": {"reach": 41.2, "grasp": 6.1, "lift": 0.0},
  "gate": "ok | total_only | success_term_missing | bad_weight | no_metrics | non_finite",
  "behavior": "<one paragraph: what the policy actually does, vs `description`>",
  "failure_mode": "<one line, or null when it converged>",
  "findings": ["<0-3 lines the next design should act on>"],
  "analysis": {"checkpoint_watched": "peak | final",
               "frames_usable": "...", "stage_reached": "...", "time_allocation": "...",
               "reward_hacking": "...", "physical_validity": "...",
               "termination": "...", "actuation_quality": "..."},
  "notes": ["<scorer-emitted caveats>"],
  "artifacts": {"render_mp4": "...", "frames_dir": "...", "curves_dir": "...",
                "metrics_jsonl": "...", "run_log": "...", "trial_dir": "..."} }
```

Every artifact but `trial_dir` lives **inside `iter_<NNN>/`**, copied there by the candidate,
so an iteration is reviewable on its own without resolving a timestamped path under
`harbor/outputs/`. `trial_dir` still points at the canonical training output (checkpoints,
TensorBoard events). `score_iter.py` **drops any declared path that does not exist** and says
so in `notes` — the designer is never sent to a dead file.

**`verdict.json` is the ONLY analysis channel.** `behavior` / `failure_mode` / `findings` are
the write-up; there is no companion prose file. A candidate that has more to say says it in
those fields.

`behavior`, `failure_mode`, `findings`, and `analysis` all come from the candidate's
`iter_<NNN>/analysis.json`, passed to `score_iter.py --analysis-json`. That file is a
**checklist**: eight aspects of the rollout, each answered in prose. The scorer validates
coverage — a missing key, an unknown key, or a placeholder answer exits non-zero rather
than producing a verdict with quiet holes, and `--analysis-json` is mandatory whenever
`status` is `scored`. Coverage is the half a machine can check: it cannot tell whether
"the gripper never closes" is true, but it can tell that nobody addressed `termination`.

Read `analysis.physical_validity` before concluding anything about the reward — penetration
and sinking are §1–§3 defects, and every other aspect points at the reward, so the two lead
to opposite repairs. `artifacts` are raw evidence — the render, the frames, the metrics, the run log —
for the designer to pull when a verdict doesn't add up, never a second narrative to maintain.

The durable copy on disk is what makes the loop resume-safe: a crashed session, a killed
agent, or a lost notification all recover by reading `verdict.json`.

## Status semantics

| `status` | Means | Designer's move |
|---|---|---|
| `scored` | trained to budget, curves + rollout read | compare against best; refill |
| `task_smoke_failed` | a §1–§5 smoke never passed — **the task design is not realizable** | non-improving; steer the next design away from that structural idea. Re-issuing the same `task_changes` with different weights wastes an iteration |
| `reward_smoke_failed` | S6 never passed — the reward is malformed, the task is fine | non-improving; the reward spec asked for something the env cannot express |
| `train_failed` | launched, no checkpoint sentinel | non-improving; infrastructure unless `findings` says otherwise |
| `early_stopped` | killed mid-run by the monitor | non-improving; `failure_mode` steers the next design |

Splitting the two smoke failures is the point of the `smokes` map: a broken contact sensor
and a broken reward term produce the same "it didn't train" symptom, and the designer's
next move is completely different in each case.

`success_rate: null` (any `gate` but `ok`) means **ungradable, not zero**. Rank it below
every graded candidate and never write it into `best_success_rate` — a fabricated score
poisons every later comparison.
