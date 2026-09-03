# rl-tuning-agent — Tuning Instructions

Read once at Phase 0 (the first `submit` for a cell). The 4 hard constraints below are non-negotiable.

## Procedure

### Phase 0 — Bootstrap (first `submit` only)

1. Read this file + `tuning-experience.md` ONCE. Reference both from memory thereafter.
2. Derive `wandb_project = f"tuning-{benchmark_name}-{task}-{algorithm}"` (read `benchmark_name` from `<repo>/harbor/benchmark-generator/benchmark-spec.json`). Verify W&B creds (`grep -q machine.api.wandb.ai ~/.netrc`); else surface `/harbor:wandb-setup` and stop.
3. Mint per-cell directory `<tune_dir>/<wandb_project>/`. Standalone fallback: `<repo>/harbor/rl_experiments/tunes/standalone_<UTC-ts>/<wandb_project>/`.
4. Render `<cell_dir>/tuning-history.md` from `${CLAUDE_PLUGIN_ROOT}/knowledge/templates/rl-tuning-agent/tuning-history.md.template`.

### Phase 0.5 — Default-config baseline (iter 0)

Train with the unmodified suite config. Run name `v1_baseline`. Record final_return / sample_eff / wall_clock / paths / plateau check (constraint #2) into the iter-0 row.

### Phase 1 — Tricks (iter 1..N_tricks)

For each available trick (`python3 ${CLAUDE_PLUGIN_ROOT}/scripts/rl-tricks/list_tricks.py --algo <algo>`, filter by `backend == algorithm_slug`):

- Snapshot the suite yaml. Apply `${CLAUDE_PLUGIN_ROOT}/scripts/rl-tricks/apply_trick.py --trick <name> --algorithm <algo> --repo <repo>`.
- Run one trial under the same `total_timesteps` as baseline.
- Improved → keep, update running best. Regressed → revert, log, next trick.

One trick at a time. No stacking until each is judged.

### Phase 2 — Hyperparameter hill-climb (iter > N_tricks)

Read the running-best `metrics.jsonl` ONCE and propose ONE candidate per iteration using these heuristics:

| Symptom (from log) | Edit |
|---|---|
| train_loss / actor_loss spike or NaN | halve `learning_rate` OR double `batch_size` |
| eval return stuck low (≤ baseline ± 5%) | double `n_epochs` (PPO) / `gradient_steps` (SAC/TD3) |
| value_loss climbs unbounded (PPO) | enable `value_clip_torch` (else halve LR) |
| policy entropy < 0.01 in first 10% (PPO) | 2× `entropy_coef` |
| eval return still rising at end | constraint #2: bump `total_timesteps` 1.5–2× and re-run BEFORE counting iter |

Write candidate as `iter_<NNN>/overrides.yaml` (diff vs running best). Use Hydra overrides on `train.py`; do NOT mutate the suite yaml.

### Phase 3 — Finalize (runs INSIDE the score that triggered stop)

There is no separate `finalize` Agent dispatch. When `score` determines `stopped_by != null` (`stuck_threshold` / `max_iterations` / fatal), the SAME `score` invocation:

1. Appends Best-vs-baseline + Best-config + Takeaways sections to `tuning-history.md` (template at the bottom of the template file).
2. Writes `<cell_dir>/result.json` per the agent body's output contract.
3. Returns the score JSON with `stopped_by` set + full `result` inlined (so the orchestrator can persist it without re-reading).

## Stop conditions

Stop when EITHER:
- (a) `stuck >= stuck_threshold` (default 3) — running best not beaten for that many consecutive iters, OR
- (b) `iter >= max_iterations` (default 10) — hard cap including iter_0 baseline + Phase 1 tricks.

Failed iterations (constraint #4) DO count toward both. Set `result.json:stopped_by` accordingly.

## W&B run names: `v<N>_<change_slug>`

Iteration `i` (0-indexed) → run name `v<i+1>_<slug>`. Slug rules:

- ≤ 30 chars, snake_case, alphanumeric + underscore only.
- Imperative — describe the CHANGE, not absolute values: `lr_halved` ✓ / `lr_15e_4` ✗.
- Phase 1 tricks: use the trick's manifest name (`obs_rms_jax`, `value_clip_torch`).
- Multi-key edits: join dominant changes with `_` (`lr_halved_grad_x2`).

The slug must equal the iteration's `candidate_label` in `tuning-history.md` so a reviewer can cross-reference W&B ↔ audit-trail.

Pass on every train command:
```
wandb=<wandb_project> wandb_run_name=v<i+1>_<slug>
```

## Cross-cell findings

The `score` JSON output may include `findings: [string]` — 0 to 3 short imperatives this iteration revealed. The orchestrator appends them (with `cell` + `iter` tags) to `<tune_dir>/_findings.jsonl` and injects the latest 10 into the next `submit` prompt as `sibling_findings`.

Use `sibling_findings` to BIAS candidate selection — advisory, not prescriptive (sibling cells run different combos).

**Emit a finding** when you discovered:
- A failure-mode cause/fix (`"halve learning_rate when actor_loss spikes — saw NaN at lr=3e-3"`).
- A cross-run-relevant trick effect (positive or negative).
- A monitoring signal with a clear mechanism (`"obs_rms is essential for SAC on multi-scale state vectors"`).

**Do NOT emit** for: baseline result, score deltas without mechanism, task-specific values.

Format: present-tense imperative, ≤ 120 chars. The orchestrator wraps with metadata; emit only the bare string.

## Hard constraints (never bypass)

| # | Rule |
|---|------|
| 1 | `num_envs` ∈ `{0.5×, 1×, 2×}` of default. Reject any other value. |
| 2 | A run must show convergence. If eval return is still rising at the final step, bump `total_timesteps` 1.5–2× and re-run BEFORE scoring. |
| 3 | Reject a candidate whose wall-clock > 1.5× running best AND chosen-metric improvement < 10%. |
| 4 | Scan the slurm/training log ONCE after the job exits for traceback / `Error` / `Exception` / NaN. Do NOT tail logs or inspect `metrics.jsonl` mid-run. |

A trial whose post-run scan finds traceback / NaN / Error → `status: failed`. Counts toward `stuck_threshold`. Three consecutive failures → pause and surface to the user.

## Run-to-completion policy

Every trial (iter 0 + Phase 1 tricks + Phase 2) runs to its full `total_timesteps`. The agent does NOT poll mid-flight — every probe re-replays full context. Wait for the SLURM job (cluster) or `train.py` (local) to exit; then read the curve ONCE from `metrics.jsonl` or the saved `curves/` plot. Score, decide, propose next.

## Token efficiency

- **References ONCE.** This file + `tuning-experience.md` at Phase 0 only.
- **History index, not full file.** Read top ~30 lines (`Read offset=0 limit=30`); audit blocks below are append-only (you wrote them).
- **`Edit`, not `Read+Write`.** Append iteration blocks via single `Edit` (anchor on previous-iter row).
- **Batch Bash probes.** One call with `&&` / `;` — every separate Bash invocation = full-context replay.

## Scoring

```
score = w_se · normalized(sample_efficiency) + w_fr · normalized(final_return)
```

- `normalized(x)` = min-max across iterations seen so far this tune (single-point baseline → 1.0).
- `sample_efficiency = 1.0 / env_steps_to_threshold` (larger = better; threshold = 0.8 × final_return of running best).
- Defaults: `w_se = w_fr = 0.5`.

## Tune-directory layout

```
<repo>/harbor/rl_experiments/tunes/<tune_id>/  ← orchestrator owns
├── history.md, manifest.json, _findings.jsonl
└── <wandb_project>/                             ← per-cell, name = wandb project
    ├── tuning-history.md, result.json, state.json
    └── iter_<NNN>/overrides.yaml,
        run.sh|launch.sh,                         ← orchestrator launches
        run.log|slurm-*.out,                      ← orchestrator captures
        .done                                     ← orchestrator's signal
```

## Slash-command mirroring (subagent has Bash, not Skill)

For canonical wording, read each `commands/<name>.md`. Bash equivalents:

### `/harbor:rl-run` (the per-iter trial command)
```bash
"<repo>/.venv/bin/python" harbor/scripts/rl/<algorithm_slug>/train.py \
    --config-name=<algo>.parallel \
    task=<t> wandb=<wandb_project> wandb_run_name=v<N>_<slug> \
    <hydra k=v ...>
```
`algorithm_slug` and `parallel` flag come from `harbor/rl-integration-generator/rl-suite-spec.json`. The `wandb` and `wandb_run_name` overrides are mandatory.

### `/harbor:rl-add-trick`
```bash
python3 "${CLAUDE_PLUGIN_ROOT}/scripts/rl-tricks/apply_trick.py" \
    --trick <name> --algorithm <algo> --repo "<repo>"
[ -f "${CLAUDE_PLUGIN_ROOT}/knowledge/templates/rl-tricks/<name>/smoke.py" ] && \
    "<repo>/.venv/bin/python" "${CLAUDE_PLUGIN_ROOT}/knowledge/templates/rl-tricks/<name>/smoke.py" --repo "<repo>"
```

The cluster-mode SLURM wait is OUTSIDE this agent — the orchestrator runs `while squeue -h -j $JOBID; do sleep 600; done` in `Bash run_in_background=true`. Do NOT replicate that loop here.

## Per-iteration block in tuning-history.md

Append-only. One section per iteration, structured per the template file. Each block has: Rationale (cite the heuristic), Config diff vs parent, Analysis (final_return / sample_eff / wall_clock + Δ, log signals), Decision (Accepted / Rejected / Failed + stuck counter + next move). Update the index table row in the same `Edit`.
