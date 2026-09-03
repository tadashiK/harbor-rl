#!/usr/bin/env python3
"""score_iter.py — turn a finished candidate's training log into the verdict JSON.

The single source of the `success_rate` arithmetic. Every candidate is scored by the
same code path, so two iterations of the same tune can be compared; when the agent
re-derived the formula per iteration, they could not.

Reads `metrics.jsonl` + the candidate's `design.json`, writes the complete verdict
object defined in knowledge/references/reward-tuning-agent/candidate-contract.md. The numeric
fields are computed here; the rollout analysis comes from `--analysis-json`, written by
the candidate agent, which is the thing that watched the rollout.

The analysis is a CHECKLIST, and this script enforces that every aspect was answered.
Coverage is the part a machine can check: it cannot tell whether "the gripper never
closes" is true, but it can tell that nobody addressed `termination` at all, and an
omitted aspect is the common way a rollout analysis misleads the search. Answers are
prose, because the useful content ("reaches at frame 2, hovers 3-12") does not fit an
enum. Vague placeholders are rejected for the same reason "looks correct" is not a
validation elsewhere in the harness.

Usage:
  score_iter.py --metrics <trial>/metrics.jsonl --design <iter>/design.json --iter 7
                [--status scored|task_smoke_failed|reward_smoke_failed|train_failed|early_stopped]
                [--smoke S1=pass]... [--analysis-json <iter>/analysis.json]
                [--artifact render_mp4=<path>]... [--out <iter>/verdict.json]

`--analysis-json` is REQUIRED when --status is `scored`: a scored candidate is one whose
rollout was watched. The other statuses mean there was no rollout worth watching.

Exits 0 with a verdict even when the run is ungradable — an ungradable run is a
RESULT the designer must see (`success_rate: null` + a gate note), not a crash.
Exits non-zero only on bad inputs (missing/unreadable files).
"""
import argparse
import json
import math
import os
import sys

from _metrics import final_values, load_series, peak_values

PREFIX, SUFFIX = "reward/", "/episodic_return_mean"
TOTAL_KEY = PREFIX + "total" + SUFFIX


# The rollout-analysis checklist. Each key is one aspect of the video the designer acts on;
# the order is the order the candidate is asked to work through them.
ANALYSIS_PROSE_KEYS = (
    "frames_usable",       # is the subject actually in frame — if not, the rest is void
    "behavior",            # what the policy does, against `description`
    "stage_reached",       # furthest rung of the term ladder, and where it stalls
    "time_allocation",     # where the frames cluster
    "reward_hacking",      # a term being farmed instead of progress
    "physical_validity",   # penetration, sinking, jitter, explosion -> routes to §1-§3
    "termination",         # fires as intended / never / constantly / on a wrong state
    "actuation_quality",   # jitter, oscillation, saturation -> action-rate or §2
)
ANALYSIS_KEYS = ("checkpoint_watched",) + ANALYSIS_PROSE_KEYS + ("failure_mode", "findings")
CHECKPOINT_VALUES = ("peak", "final")
MAX_FINDINGS = 3

# A real answer does not fit in a word. The floor is deliberately low — it rejects "ok" and
# "clean", not a terse honest answer like "none observed" — and uncertainty has an escape
# hatch ("unclear: only the arm is in frame"), so nothing here pushes toward confabulation.
ANALYSIS_MIN_CHARS = 12
VAGUE_ANSWERS = {
    "ok", "okay", "fine", "good", "bad", "yes", "no", "none", "n/a", "na", "nil", "-",
    "normal", "correct", "clean", "nothing", "unknown", "unclear", "tbd", "todo",
    "as expected", "looks correct", "looks good", "looks fine", "no issues", "all good",
}


def _rollout_aspects(analysis):
    """The checklist minus the three fields that are promoted to the verdict's top level."""
    if not analysis:
        return None
    promoted = ("behavior", "failure_mode", "findings")
    aspects = {k: analysis[k] for k in ANALYSIS_KEYS
               if k in analysis and k not in promoted}
    return aspects or None


def validate_analysis(obj, status="scored"):
    """Return a list of human-readable problems; empty means the checklist is complete.

    The rollout aspects are required only for a `scored` candidate, because only a scored
    candidate has a rollout. One whose smokes failed never trained, and demanding eight
    observations of a video that does not exist would just manufacture them — but it still
    owes a `failure_mode` and `findings`, which is what the designer acts on.
    """
    if not isinstance(obj, dict):
        return ["analysis JSON must be an object"]
    errs = []
    required = ANALYSIS_KEYS if status == "scored" else ("failure_mode", "findings")
    unknown = sorted(set(obj) - set(ANALYSIS_KEYS))
    if unknown:
        # A typo'd key would otherwise drop that aspect silently while looking answered.
        errs.append(f"unknown key(s) {unknown}; allowed: {list(ANALYSIS_KEYS)}")
    for k in required:
        if k not in obj:
            errs.append(f"missing required key '{k}'"
                        + ("" if status == "scored" else f" (required for status {status!r})"))

    ckpt = obj.get("checkpoint_watched")
    if "checkpoint_watched" in obj and ckpt not in CHECKPOINT_VALUES:
        errs.append(
            f"checkpoint_watched must be one of {list(CHECKPOINT_VALUES)}, got {ckpt!r} — "
            "on a run that peaked and collapsed these are different policies"
        )

    for k in ANALYSIS_PROSE_KEYS:
        if k not in obj:
            continue
        v = obj[k]
        if not isinstance(v, str) or not v.strip():
            errs.append(f"'{k}' must be a non-empty string")
        elif v.strip().lower().rstrip(".") in VAGUE_ANSWERS:
            errs.append(
                f"'{k}' is {v.strip()!r} — say what you saw. An answer that cannot be "
                "contradicted by the frames is not evidence."
            )
        elif len(v.strip()) < ANALYSIS_MIN_CHARS:
            errs.append(f"'{k}' is too short to be an answer ({v.strip()!r})")

    fm = obj.get("failure_mode")
    if "failure_mode" in obj and fm is not None and not (isinstance(fm, str) and fm.strip()):
        errs.append("'failure_mode' must be a non-empty string, or null when it converged")

    f = obj.get("findings")
    if "findings" in obj:
        if not isinstance(f, list):
            errs.append("'findings' must be a list")
        elif len(f) > MAX_FINDINGS:
            errs.append(f"'findings' has {len(f)} entries; at most {MAX_FINDINGS}")
        elif any(not isinstance(x, str) or not x.strip() for x in f):
            errs.append("'findings' entries must be non-empty strings")
    return errs


def per_term_returns(final):
    """{term: episodic return} for every reward/<term>/episodic_return_mean key but total."""
    out = {}
    for k, v in final.items():
        if k.startswith(PREFIX) and k.endswith(SUFFIX) and isinstance(v, (int, float)):
            term = k[len(PREFIX):-len(SUFFIX)]
            if term != "total":
                out[term] = v
    return out


def reward_spec(design):
    """The reward half of a candidate design.

    A design is `{task_changes: {...}, reward: {...}}`. The flat legacy shape (reward keys
    at the top level) is still read so an in-flight tune's older design.json still scores.
    """
    return design.get("reward") or design


def term_weight(spec, name):
    for t in spec.get("terms", []):
        if t.get("name") == name:
            return t.get("weight")
    return None


def score(final, design):
    """Return (success_rate, gate, notes). success_rate is None whenever ungradable."""
    notes = []
    spec = reward_spec(design)
    per_term = per_term_returns(final)
    if not per_term:
        return None, "total_only", [
            "metrics.jsonl carries only reward/total/... — per-term logging regressed, "
            "so no success_rate can be computed. Do NOT infer one from the total curve."
        ]

    name = spec.get("success_term")
    if not name:
        return None, "no_success_term", ["design.json names no success_term"]
    if name not in per_term:
        return None, "success_term_missing", [
            f"success_term '{name}' is absent from the logged terms {sorted(per_term)} — "
            "the term was renamed in the implementation, or it never fired and the logger "
            "dropped it."
        ]

    weight = term_weight(spec, name)
    if not isinstance(weight, (int, float)) or weight == 0:
        return None, "bad_weight", [
            f"success_term '{name}' has weight {weight!r} in design.json; a non-zero "
            "numeric weight is required to normalize its episodic return into a rate."
        ]

    rate = per_term[name] / weight
    if not math.isfinite(rate):
        return None, "non_finite", [f"success_rate is not finite (term={per_term[name]}, weight={weight})"]
    if rate > 1.0:
        notes.append(
            f"success_rate {rate:.3f} > 1 — the success term fired more than once per "
            "episode on average; treat it as a rate ceiling, not a fraction."
        )
    return rate, "ok", notes


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--metrics", required=True, help="<trial_dir>/metrics.jsonl")
    p.add_argument("--design", required=True, help="<iter_dir>/design.json")
    p.add_argument("--iter", type=int, required=True)
    p.add_argument("--status", default="scored",
                   choices=["scored", "task_smoke_failed", "reward_smoke_failed",
                            "train_failed", "early_stopped"])
    p.add_argument("--smoke", action="append", default=[], metavar="NAME=RESULT",
                   help="repeatable, e.g. S1=pass S4=fail S6=skipped")
    p.add_argument("--analysis-json", default=None,
                   help="the candidate's rollout-analysis checklist (see module docstring); "
                        "required when --status is 'scored'")
    p.add_argument("--artifact", action="append", default=[],
                   metavar="KEY=PATH", help="repeatable, e.g. render_mp4=/abs/render.mp4")
    p.add_argument("--out", default=None, help="write the verdict here (also printed)")
    a = p.parse_args()

    # A malformed analysis is bad INPUT, not an ungradable run: exit non-zero so the candidate
    # fixes it, rather than emitting a verdict whose checklist quietly has holes.
    analysis = None
    if a.analysis_json:
        try:
            with open(a.analysis_json) as f:
                analysis = json.load(f)
        except (OSError, ValueError) as e:
            sys.exit(f"score_iter: cannot read --analysis-json {a.analysis_json}: {e}")
        problems = validate_analysis(analysis, a.status)
        if problems:
            sys.exit("score_iter: incomplete rollout analysis in "
                     + a.analysis_json + "\n  - " + "\n  - ".join(problems))
    elif a.status == "scored":
        sys.exit("score_iter: --analysis-json is required when --status is 'scored' — a "
                 "scored candidate is one whose rollout was watched. Use another --status "
                 "if there was no rollout to analyse.")

    if not os.path.isfile(a.design):
        sys.exit(f"design.json not found: {a.design}")
    with open(a.design) as f:
        design = json.load(f)

    if os.path.isfile(a.metrics):
        ser = load_series(a.metrics)
        final = final_values(ser)
        rate, gate, notes = score(final, design)
        per_term = per_term_returns(final)
        total = final.get(TOTAL_KEY)

        # Peak-vs-final. A run that peaked and collapsed is scored on the collapse unless the
        # designer is shown the gap, and the end-of-training checkpoint is then not the policy
        # that earned the peak.
        #
        # Each key peaks on its own schedule, so the total's best step and the success term's
        # best step are two different moments and are reported as two fields. Collapsing them
        # into one `step` invited the reader — the candidate agent picking which checkpoint it
        # watched — to attribute the peak success_rate to the step where the TOTAL peaked, a
        # policy that may never have existed.
        peak = peak_values(ser)
        peak_total, peak_total_step = peak.get(TOTAL_KEY, (None, None))
        peak_rate, _, _ = score({k: v for k, (v, _) in peak.items()}, design)
        success_term = reward_spec(design).get("success_term")
        peak_rate_step = peak.get(PREFIX + str(success_term) + SUFFIX, (None, None))[1]
        if (peak_total is not None and total is not None
                and peak_total > 0 and total < 0.7 * peak_total):
            notes.append(
                f"run PEAKED at total_return {peak_total:.1f} @step {peak_total_step} and ended "
                f"at {total:.1f} ({total / peak_total:.0%} of peak) — the final checkpoint is "
                f"not the best policy this run produced. Score the run on both; render "
                f"checkpoint_best.pth, not checkpoint.pth."
            )
        if (peak_rate_step is not None and peak_total_step is not None
                and peak_rate_step != peak_total_step):
            notes.append(
                f"peak success_rate is at step {peak_rate_step} but peak total_return is at "
                f"step {peak_total_step} — different policies. checkpoint_best.pth tracks the "
                f"total, so it is NOT the highest-success checkpoint here."
            )
    else:
        rate, gate, notes = None, "no_metrics", [f"metrics.jsonl not found: {a.metrics}"]
        per_term, total = {}, None
        peak_total = peak_total_step = peak_rate = peak_rate_step = None

    def _pairs(items):
        out = {}
        for item in items:
            k, _, v = item.partition("=")
            if k:
                out[k] = v
        return out

    # Artifact existence gate. A verdict that points at a path which was never written sends
    # the designer to a dead file exactly when it is already suspicious of the numbers — and a
    # silent `cp` failure is invisible otherwise. Missing paths are REPORTED, not carried.
    declared = _pairs(a.artifact)
    artifacts = {k: v for k, v in declared.items() if v and os.path.exists(v)}
    missing = {k: v for k, v in declared.items() if k not in artifacts}
    if missing:
        notes.append(
            "declared artifact(s) do not exist and were dropped from the verdict: "
            + ", ".join(f"{k}={v}" for k, v in sorted(missing.items()))
            + " — the designer cannot pull evidence that was never written."
        )

    verdict = {
        "iter": a.iter,
        "status": a.status,
        "smokes": _pairs(a.smoke),
        "success_rate": rate,
        "total_return": total,
        "peak": {"success_rate": peak_rate, "success_rate_step": peak_rate_step,
                 "total_return": peak_total, "total_return_step": peak_total_step},
        "per_term": {k: round(v, 4) for k, v in sorted(per_term.items())},
        "gate": gate,
        # `behavior` / `failure_mode` / `findings` stay top level: they are what the designer
        # reads first, and the checklist sits beside them rather than displacing them.
        "behavior": (analysis or {}).get("behavior", ""),
        "failure_mode": (analysis or {}).get("failure_mode"),
        "findings": (analysis or {}).get("findings", []),
        # Only the rollout aspects that were actually answered — a smoke-failed candidate has
        # no rollout, so this is null rather than a row of empty strings pretending otherwise.
        "analysis": _rollout_aspects(analysis),
        "notes": notes,
        "artifacts": artifacts,
    }

    out = json.dumps(verdict, indent=2)
    if a.out:
        with open(a.out, "w") as f:
            f.write(out + "\n")
    print(out)


if __name__ == "__main__":
    main()
