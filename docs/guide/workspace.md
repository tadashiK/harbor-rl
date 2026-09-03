# Your workspace

Everything HARBOR generates lands inside the target repository under a single root, `harbor/`. Nothing is hidden in a cache directory or a conversation transcript, and setting up a second benchmark is the same pipeline run again.

```
<your-repo>/                          your repo, untouched except for the carve-outs below
├── .venv/                            uv-managed environment
├── scripts/
│   ├── _<family>_env.py              smoke-helper convention, kept at root
│   ├── run_random.py                 L1 smoke entry point
│   └── render_random.py              L2 smoke entry point
└── harbor/
    ├── dependency-generator/         setup_uv.sh, probe.json, install_plan.json, install.md
    ├── benchmark-generator/          benchmark-spec.json, task_overview.md, benchmark.md, history.md
    ├── rl-integration-generator/     rl-suite-spec.json, rl-integration.md, history.md
    ├── create-task/
    │   ├── task-implementation.md    the family-level authoring guide
    │   └── <task-slug>/              one folder per task-create run
    ├── scripts/rl/<impl>/            train.py, eval.py, render.py, env_wrapper.py
    ├── configs/rl/                   ppo.yaml, sac.yaml, td3.yaml
    ├── outputs/<algo>_<task>_<ts>/   checkpoint, metrics.jsonl, tb/, curves/, render.mp4
    ├── rl_experiments/{sweeps,tunes}/
    └── utils/data_logger.py
```

Only three files land outside `harbor/`, all under `scripts/`: `_<family>_env.py`, `run_random.py`, and `render_random.py`. They stay at the repository root because they are the user-facing smoke entry points and because some GPU-batched simulators refuse to import if torch is loaded first — the `_<family>_env.py` helper does the pre-import dance, and it has to be importable from the root the way the repo's own code expects.

The folder is `harbor/` with no leading dot, deliberately: it doubles as a valid Python package. Rendered scripts walk up from their own location — `harbor/` is `parents[3]` of `harbor/scripts/rl/<impl>/train.py`, the repo root is `parents[4]` — and insert both on `sys.path`, so `from utils.data_logger import DataLogger` and `from scripts._<family>_env import ...` both resolve without any installation step. Hydra's `config_path="../../../configs/rl"` lands in the same place by the same arithmetic.

## The three specs

Three JSON files are the interfaces between stages. Each is written by one agent and read by the ones downstream, which is what lets a stage run months after the one before it:

| File | Written by | Carries |
|---|---|---|
| `dependency-generator/probe.json` + `install_plan.json` | dependency-generator | what the repo needs and how it was installed; editing the plan and re-rendering is how you change the install, since `setup_uv.sh` is regenerated every run and must never be hand-edited |
| `benchmark-generator/benchmark-spec.json` | benchmark-generator | the task inventory — `tasks[]` is what `/harbor:task-list` and every downstream stage read |
| `rl-integration-generator/rl-suite-spec.json` | rl-integration-generator | the algorithm slug, `scripts_dir`, whether the suite is parallel, and the config name |

`rl-suite-spec.json` is read through one shared reader, `scripts/common/resolve_suite.py`, rather than by each caller parsing it independently — a single source so the key path cannot drift across the commands that depend on it.

## Three kinds of file

**Receipts** are the end-of-run summary written for you — `install.md`, `benchmark.md`, `rl-integration.md`. Read these first.

**Process logs** are the append-only engineering record, one `history.md` inside each agent's own subdirectory. One short section per run: what tool, which agent, what command, one line of result. Read these when something went wrong and you want to know what was tried. There is no shared run-log folder — each agent's log lives with its own outputs, so a stage's record cannot be separated from the artifacts it produced.

**Working artifacts** are everything else — specs, configs, smokes, verdicts, checkpoints. These are the substrate agents actually communicate through.

## Per-task workspace

Each `task-create` run gets its own folder:

```
harbor/create-task/<task-slug>/
├── spec.json                  arguments and per-phase status
├── task-history.md            §1–§6 design record: analysis + validations
├── task-analysis.md           the rationale half alone, read by reward design
├── test-checklist.md          every check that actually ran
├── smokes/                    rendered smokes + the verdict.json each wrote
├── smoke_s{3,4,6}_frames/     reset layouts · per-predicate states · rollout keyframes
├── reward-history.md          the §6 tuning log
├── dr-history.md              the §7 log (coming soon — not written yet)
└── handoff-dr-generator.md    available and effective DR terms, modes, ranges, results
```

`spec.json` carries the run's arguments plus a per-phase status, which is what makes the chain resumable: a run that stopped after §5 records that, and re-entering picks up from the first phase that is not `pass`.

The split between `task-history.md` and `task-analysis.md` is a context decision, not an editorial one. The history is the full record including every validation; the analysis is the same file with validations stripped. Reward design and every reward candidate read the analysis, because the full history runs to tens of kilobytes and the validations are exactly the part they do not need — see [context optimization](/guide/context-optimization).

`test-checklist.md` is task-specific rather than boilerplate: §4 emits one check/validation pair per predicate the design actually implemented, so the checklist describes *this* task's verification, not a generic template.

Each smoke writes its own `<smoke>.verdict.json` as it runs, eagerly. Agents read results out of that file rather than out of their recollection of the output, and `check_task_history.py` gates the history by diffing every claimed verdict against the file the smoke actually wrote. Transcription stops being load-bearing.

## Resuming

Because state lives in files rather than context, an interrupted run resumes. Tuning loops checkpoint `tune-state.json` every iteration. Long training jobs launch detached and signal completion with a sentinel file, so a trainer that outlives its agent still counts. Pick the conversation back up and HARBOR reads where it was from disk.

Two consequences worth knowing. A killed agent does **not** kill a submitted SLURM job — the jobid is recorded so it can be cancelled deliberately. And re-running a completed stage is safe: `setup_uv.sh` is idempotent, and `/harbor:task-create` with `sections=` re-authors only the sections you name.

## Removing it

```text
/harbor:reset-workspace repo=<path> [clean_inbenchmark_tasks=true|false]
```

Removes all HARBOR output — `harbor/`, `.venv/`, the `scripts/` carve-outs, caches — and by default restores the repository to its cloned HEAD with `git reset --hard` and `git clean -fdx`. That default also removes tasks HARBOR authored *inside* the benchmark's own source tree; pass `clean_inbenchmark_tasks=false` to keep them.

This is the one destructive command, and the only one gated against model invocation: it runs in a subagent behind a dry-run and an explicit confirmation, then verifies with a git-based check that includes hidden and ignored files before reporting success. Gating it is why other commands that need its behavior read its body and execute it rather than slash-invoking it — a gated command cannot be invoked by another command at all.
