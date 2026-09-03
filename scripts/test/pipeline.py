#!/usr/bin/env python3
"""Stage-DAG state engine for /harbor:test Layer 3 (resumable, Docker-layer-style).

The L3 e2e runs the task-create pipeline module-by-module on a clean benchmark
worktree. Each stage is a build layer with a fingerprint:

    fp(stage) = sha256( the stage's PLUGIN definition files
                        + the parent stage's fp        # chain → upstream changes cascade
                        + repo HEAD commit + task id )

A stage is SKIPPED on resume iff its cached status is `pass` AND its fingerprint
is unchanged. Because each fp chains the parent's, editing an upstream module's
.md invalidates it and everything downstream (rebuild from there); editing a
downstream module rebuilds only it onward — exactly like recompiling a Dockerfile.

Subcommands (all read/write a JSON state file the orchestrator keeps in the worktree):
  stages [--skip a,b]
      print the effective ordered stage names (after skips), one per line.
  plan --repo R --task T --state S [--skip a,b]
      print JSON: per stage {name, fp, cached, action: run|skip}, plus first_to_run.
  mark --state S --stage X --status pass|fail --fp F [--note "..."]
      record a stage result (use the fp from `plan`).

PLUGIN_ROOT is derived from this file's location (scripts/test/pipeline.py).
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

PLUGIN_ROOT = Path(__file__).resolve().parents[2]

# Ordered pipeline stages → the plugin files that DEFINE each (globs, relative to PLUGIN_ROOT).
STAGES = [
    ("env-install",    ["commands/env-install-uv.md", "agents/dependency-generator.md",
                        "scripts/dependency-generator/*.py", "knowledge/references/dependency-generator/*.md"]),
    ("benchmark",      ["agents/benchmark-generator.md", "commands/probe-benchmark.md",
                        "scripts/benchmark-generator/*.py", "knowledge/references/benchmark-generator/*.md"]),
    ("rl-integration", ["agents/rl-integration-generator.md", "scripts/rl-integration-generator/*.py",
                        "scripts/common/resolve_suite.py", "knowledge/templates/rl-integration-generator/**/*",
                        "knowledge/references/rl-integration-generator/*.md"]),
    ("task-generator", ["agents/task-generator.md", "commands/task-create.md",
                        "knowledge/templates/task-generator/**/*", "knowledge/references/task-generator/*.md"]),
    ("reward-add-log", ["commands/reward-add-log.md", "scripts/reward-add-log/*.py",
                        "knowledge/templates/reward-add-log/*"]),
    # reward-tune candidates run the task-generator smokes for every section they touch,
    # and clone the task when the effective pool exceeds 1 — so both are stage inputs.
    ("reward-tune",    ["commands/reward-tune.md", "agents/reward-tuning-agent.md",
                        "agents/reward-candidate-agent.md",
                        "scripts/reward-tuning-agent/*.py",
                        "scripts/task-cloner/clone_task.py",
                        "knowledge/templates/reward-tuning-agent/**/*",
                        "knowledge/templates/task-generator/smokes/*",
                        "knowledge/references/reward-tuning-agent/*.md"]),
    ("dr-generator",   ["agents/dr-generator.md", "knowledge/templates/dr-generator/**/*",
                        "knowledge/references/dr-generator/*.md"]),
    ("rl-run",         ["commands/rl-run.md", "scripts/rl-run/*.py", "scripts/common/resolve_suite.py"]),
    ("rl-eval",        ["commands/rl-eval.md"]),
    ("rl-render",      ["commands/rl-render.md"]),
    ("rl-tricks",      ["commands/rl-add-trick.md", "commands/rl-list-tricks.md",
                        "scripts/rl-tricks/*.py", "knowledge/templates/rl-tricks/**/*"]),
    ("reset",          ["commands/reset-workspace.md"]),
]
# Not ready yet / headless — excluded from the e2e chain by default.
DEFAULT_SKIP = ["dr-generator"]

_STAGE_DEPS = dict(STAGES)
_ORDER = [name for name, _ in STAGES]


def _effective(skip):
    skip = set(skip or [])
    return [n for n in _ORDER if n not in skip]


def _dep_files(stage):
    files = set()
    for pat in _STAGE_DEPS[stage]:
        files.update(p for p in PLUGIN_ROOT.glob(pat) if p.is_file())
    return sorted(files)


def _fp(stage, parent_fp, repo_head, task):
    h = hashlib.sha256()
    for f in _dep_files(stage):
        h.update(f.relative_to(PLUGIN_ROOT).as_posix().encode())
        h.update(f.read_bytes())
    h.update(parent_fp.encode())
    h.update(repo_head.encode())
    h.update(task.encode())
    return h.hexdigest()[:16]


def _load_state(path):
    p = Path(path)
    if p.is_file():
        return json.loads(p.read_text())
    return {"stages": {}}


def cmd_stages(args):
    for n in _effective(args.skip.split(",") if args.skip else []):
        print(n)


def cmd_plan(args):
    skip = args.skip.split(",") if args.skip else []
    state = _load_state(args.state)
    cache = state.get("stages", {})
    parent_fp, out, first = "", [], None
    for stage in _effective(skip):
        fp = _fp(stage, parent_fp, args.repo_head, args.task)
        c = cache.get(stage)
        action = "skip" if (c and c.get("status") == "pass" and c.get("fp") == fp) else "run"
        if action == "run" and first is None:
            first = stage
        out.append({"name": stage, "fp": fp, "cached": (c or {}).get("status"), "action": action})
        parent_fp = fp
    print(json.dumps({"first_to_run": first, "stages": out}, indent=2))


def cmd_mark(args):
    p = Path(args.state)
    state = _load_state(args.state)
    state.setdefault("stages", {})[args.stage] = {
        "status": args.status, "fp": args.fp, "note": args.note or "",
    }
    p.write_text(json.dumps(state, indent=2))
    print(f"marked {args.stage}={args.status}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("stages"); s.add_argument("--skip", default=",".join(DEFAULT_SKIP)); s.set_defaults(fn=cmd_stages)
    p = sub.add_parser("plan")
    p.add_argument("--repo-head", required=True); p.add_argument("--task", required=True)
    p.add_argument("--state", required=True); p.add_argument("--skip", default=",".join(DEFAULT_SKIP))
    p.set_defaults(fn=cmd_plan)
    m = sub.add_parser("mark")
    m.add_argument("--state", required=True); m.add_argument("--stage", required=True)
    m.add_argument("--status", required=True, choices=["pass", "fail"])
    m.add_argument("--fp", required=True); m.add_argument("--note", default="")
    m.set_defaults(fn=cmd_mark)
    args = ap.parse_args()
    args.fn(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
