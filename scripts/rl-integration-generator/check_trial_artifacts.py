#!/usr/bin/env python3
"""check_trial_artifacts.py — the deterministic half of the rl-suite smoke (T4 + T5).

Tiers T1-T3 deliberately mirror the /harbor:rl-run, /harbor:rl-eval and /harbor:rl-render
command bodies verbatim, so they stay with the agent. What does NOT need an agent is
checking what those runs left on disk — curve PNGs, TensorBoard events, metrics.jsonl — and
resolving the trial dir out of the trainer's stdout, which is fiddly enough to be worth
having in one place.

Usage:
  check_trial_artifacts.py --trial <trial_dir> [--logging-mode wandb]
  check_trial_artifacts.py --log <T1 stdout log> [--logging-mode wandb]   # resolve trial dir

Prints a JSON verdict. Exits 0 whenever it could look — a trial that produced nothing is a
RESULT (`"T4": "fail"`), not a crash. Exits non-zero only when no trial dir can be resolved,
which means the caller passed something broken.
"""
import argparse
import json
import re
import sys
from pathlib import Path

# train.py prints this line on save; the trial dir is the checkpoint's parent.
_CKPT_LINE = re.compile(r"saved checkpoint to\s+(.+?)\s*$", re.MULTILINE)


def resolve_trial_dir(log_path):
    """Trial dir from a T1 stdout log, or None. Last match wins — the final save."""
    try:
        text = Path(log_path).read_text(errors="replace")
    except OSError:
        return None
    matches = _CKPT_LINE.findall(text)
    if not matches:
        return None
    return Path(matches[-1].strip()).parent


def inspect(trial: Path, logging_mode: str = ""):
    curves = sorted(trial.glob("curves/*.png"))
    tb_events = sorted(trial.glob("tb/**/events.out.tfevents.*"))
    metrics = trial / "metrics.jsonl"
    metric_lines = 0
    if metrics.is_file():
        with metrics.open(errors="replace") as f:
            metric_lines = sum(1 for line in f if line.strip())

    checkpoints = sorted(p.name for p in trial.glob("checkpoint.*"))
    wandb_dirs = [str(p.relative_to(trial)) for p in trial.glob("**/wandb") if p.is_dir()]

    out = {
        "trial_dir": str(trial),
        "exists": trial.is_dir(),
        "checkpoints": checkpoints,
        "render_mp4": (trial / "render.mp4").is_file(),
        "curves": {"count": len(curves), "dir": str(trial / "curves")},
        "tb": {"events": len(tb_events)},
        "metrics": {"lines": metric_lines, "path": str(metrics)},
        "wandb": {"dirs": len(wandb_dirs)},
        "tiers": {
            # T4 — train.py auto-plots at the end of training; this checks the artifacts.
            "T4": "pass" if curves else "fail",
            # T5 — both logging sinks must have produced something.
            "T5": "pass" if (tb_events and metric_lines) else "fail",
        },
        "notes": [],
    }
    if not trial.is_dir():
        out["notes"].append(f"trial dir does not exist: {trial}")
    if not curves:
        out["notes"].append("no PNG curves — train.py's end-of-training plot did not run")
    if not tb_events:
        out["notes"].append("no TensorBoard events")
    if not metric_lines:
        out["notes"].append("metrics.jsonl missing or empty")
    if logging_mode == "wandb" and not wandb_dirs:
        out["notes"].append("logging_mode=wandb but no wandb run dir under the trial")
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--trial", type=Path, help="trial dir to inspect")
    p.add_argument("--log", type=Path, help="T1 stdout log; the trial dir is read from it")
    p.add_argument("--logging-mode", default="", help="'wandb' adds a W&B run-dir check")
    a = p.parse_args()

    trial = a.trial
    if trial is None:
        if a.log is None:
            sys.exit("pass --trial or --log")
        trial = resolve_trial_dir(a.log)
        if trial is None:
            sys.exit(f"no 'saved checkpoint to ...' line in {a.log} — cannot resolve trial dir")

    print(json.dumps(inspect(trial, a.logging_mode), indent=2))


if __name__ == "__main__":
    main()
