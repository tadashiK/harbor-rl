"""Unit: scripts/reward-tuning-agent/score_iter.py — the one success_rate formula.

Locks in the two behaviors the reward-tune loop depends on:

1. `success_rate = <success term's episodic return> / <its design weight>`, computed
   identically for every candidate so iterations are comparable.
2. An ungradable run yields `success_rate: null` + a gate reason and still exits 0 —
   the designer must SEE that a run was ungradable rather than receive a number
   fabricated from the total-only curve.
"""
import json
import subprocess
import sys

from _pluginmeta import ROOT

SCRIPT = ROOT / "scripts" / "reward-tuning-agent" / "score_iter.py"

DESIGN = {
    "kind": "structured",
    "task_changes": {"sections": [], "changes": []},
    "reward": {
        "composer": "sum",
        "success_term": "stack_success",
        "terms": [
            {"name": "reach", "weight": 1.0},
            {"name": "stack_success", "weight": 200.0},
        ],
    },
}


def _write(tmp_path, metrics_rows, design=DESIGN):
    m = tmp_path / "metrics.jsonl"
    m.write_text("".join(json.dumps(r) + "\n" for r in metrics_rows))
    d = tmp_path / "design.json"
    d.write_text(json.dumps(design))
    return m, d


VALID_ANALYSIS = {
    "checkpoint_watched": "final",
    "frames_usable": "robot, cube and marker in frame for all 12 frames",
    "behavior": "the arm reaches the cube and hovers; the gripper never closes",
    "stage_reached": "reach (rung 1 of 2); never advances to lift",
    "time_allocation": "frames 0-2 approach, frames 3-11 stationary hover",
    "reward_hacking": "parks at the reach term's maximum instead of grasping",
    "physical_validity": "no penetration; the cube rests on the table throughout",
    "termination": "never fires; every episode runs the full horizon",
    "actuation_quality": "smooth approach, slight wrist jitter while hovering",
    "failure_mode": "reach saturates before grasp is attempted",
    "findings": ["gate reach on gripper-closed"],
}


def _analysis(tmp_path, **overrides):
    """A complete checklist, so tests about the NUMBERS need not restate it."""
    obj = dict(VALID_ANALYSIS)
    for k, v in overrides.items():
        if v is _DROP:
            obj.pop(k, None)
        else:
            obj[k] = v
    p = tmp_path / "analysis.json"
    p.write_text(json.dumps(obj))
    return p


_DROP = object()


def _run(metrics, design, *extra, expect_rc=0):
    extra = list(extra)
    # --analysis-json is mandatory under the default `scored` status. Tests that are about
    # the arithmetic get a valid one injected so they stay about the arithmetic.
    if "--analysis-json" not in extra and "--status" not in extra:
        extra += ["--analysis-json", str(_analysis(metrics.parent))]
    r = subprocess.run(
        [sys.executable, str(SCRIPT), "--metrics", str(metrics),
         "--design", str(design), "--iter", "7", *extra],
        capture_output=True, text=True,
    )
    assert r.returncode == expect_rc, f"rc={r.returncode}\n{r.stderr}"
    if expect_rc:
        return r.stderr
    return json.loads(r.stdout)


def test_success_rate_is_term_return_over_weight(tmp_path):
    m, d = _write(tmp_path, [
        {"step": 100, "reward/total/episodic_return_mean": 10.0,
         "reward/reach/episodic_return_mean": 8.0,
         "reward/stack_success/episodic_return_mean": 20.0},
        {"step": 200, "reward/total/episodic_return_mean": 74.0,
         "reward/reach/episodic_return_mean": 12.0,
         "reward/stack_success/episodic_return_mean": 62.0},
    ])
    v = _run(m, d)
    assert v["gate"] == "ok"
    assert v["success_rate"] == 62.0 / 200.0     # last step, not the max or the mean
    assert v["total_return"] == 74.0
    assert v["per_term"] == {"reach": 12.0, "stack_success": 62.0}
    assert v["iter"] == 7 and v["status"] == "scored"


def test_long_format_metrics_parse(tmp_path):
    m, d = _write(tmp_path, [
        {"key": "reward/total/episodic_return_mean", "step": 1, "value": 5.0},
        {"key": "reward/stack_success/episodic_return_mean", "step": 1, "value": 50.0},
        {"key": "reward/stack_success/episodic_return_mean", "step": 2, "value": 100.0},
        {"key": "reward/total/episodic_return_mean", "step": 2, "value": 120.0},
    ])
    v = _run(m, d)
    assert v["success_rate"] == 0.5 and v["total_return"] == 120.0


def test_total_only_curves_are_ungradable(tmp_path):
    m, d = _write(tmp_path, [
        {"step": 1, "reward/total/episodic_return_mean": 55.0},
    ])
    v = _run(m, d)
    assert v["success_rate"] is None
    assert v["gate"] == "total_only"
    assert v["per_term"] == {}


def test_missing_success_term_is_ungradable(tmp_path):
    m, d = _write(tmp_path, [
        {"step": 1, "reward/total/episodic_return_mean": 9.0,
         "reward/reach/episodic_return_mean": 9.0},
    ])
    v = _run(m, d)
    assert v["success_rate"] is None
    assert v["gate"] == "success_term_missing"
    assert "reach" in v["notes"][0]


def test_missing_metrics_file_is_a_result_not_a_crash(tmp_path):
    _, d = _write(tmp_path, [])
    v = _run(tmp_path / "absent.jsonl", d, "--status", "train_failed")
    assert v["success_rate"] is None and v["gate"] == "no_metrics"
    assert v["status"] == "train_failed"


def test_task_and_reward_smoke_failures_stay_distinguishable(tmp_path):
    """A broken sensor and a broken reward term look identical downstream unless the
    status and the per-smoke map say which one it was."""
    _, d = _write(tmp_path, [])
    a = tmp_path / "smoke_fail.json"
    a.write_text(json.dumps({"failure_mode": "contact sensor never reports a hit",
                             "findings": ["add a contact sensor to the gripper"]}))
    v = _run(tmp_path / "absent.jsonl", d, "--status", "task_smoke_failed",
             "--smoke", "S1=pass", "--smoke", "S4=fail", "--smoke", "S6=skipped",
             "--analysis-json", str(a))
    assert v["failure_mode"] == "contact sensor never reports a hit"
    assert v["status"] == "task_smoke_failed"
    assert v["smokes"] == {"S1": "pass", "S4": "fail", "S6": "skipped"}
    assert v["success_rate"] is None

    v = _run(tmp_path / "absent.jsonl", d, "--status", "reward_smoke_failed",
             "--smoke", "S1=pass", "--smoke", "S6=fail")
    assert v["status"] == "reward_smoke_failed"
    assert v["smokes"]["S6"] == "fail"


def test_flat_legacy_design_still_scores(tmp_path):
    """An in-flight tune's design.json predates the task_changes/reward nesting."""
    flat = {"kind": "structured", "composer": "sum", "success_term": "stack_success",
            "terms": [{"name": "stack_success", "weight": 50.0}]}
    m, d = _write(tmp_path, [
        {"step": 1, "reward/total/episodic_return_mean": 30.0,
         "reward/stack_success/episodic_return_mean": 25.0},
    ], design=flat)
    v = _run(m, d)
    assert v["gate"] == "ok" and v["success_rate"] == 0.5


def test_prose_and_artifacts_pass_through(tmp_path):
    m, d = _write(tmp_path, [
        {"step": 1, "reward/total/episodic_return_mean": 1.0,
         "reward/stack_success/episodic_return_mean": 100.0},
    ])
    out = tmp_path / "verdict.json"
    mp4 = tmp_path / "render.mp4"; mp4.write_bytes(b"mp4")   # must exist — see the gate below
    a = _analysis(tmp_path, behavior="arm reaches but never closes the gripper",
                  failure_mode="no grasp", findings=["contact gate never fires"])
    v = _run(m, d, "--analysis-json", str(a),
             f"--artifact", f"render_mp4={mp4}",
             "--out", str(out))
    assert v["behavior"].startswith("arm reaches")
    assert v["failure_mode"] == "no grasp"
    assert v["findings"] == ["contact gate never fires"]
    assert v["analysis"]["termination"].startswith("never fires")
    assert v["artifacts"] == {"render_mp4": str(mp4)}
    assert json.loads(out.read_text()) == v      # --out and stdout agree


def test_peak_is_reported_alongside_final_when_a_run_collapses(tmp_path):
    """A run that peaks and degrades is scored on the collapse unless the designer sees the
    gap — observed in a real tune: peak 772, scored 158."""
    m, d = _write(tmp_path, [
        {"step": 100, "reward/total/episodic_return_mean": 100.0,
         "reward/stack_success/episodic_return_mean": 100.0},
        {"step": 200, "reward/total/episodic_return_mean": 772.0,
         "reward/stack_success/episodic_return_mean": 160.0},
        {"step": 300, "reward/total/episodic_return_mean": 158.0,
         "reward/stack_success/episodic_return_mean": 20.0},
    ])
    v = _run(m, d)
    assert v["total_return"] == 158.0, "final must stay the headline number"
    assert v["peak"]["total_return"] == 772.0 and v["peak"]["total_return_step"] == 200
    assert v["peak"]["success_rate"] == 160.0 / 200.0
    assert v["peak"]["success_rate_step"] == 200
    assert any("PEAKED" in n for n in v["notes"])
    assert any("checkpoint_best.pth" in n for n in v["notes"])
    assert not any("different policies" in n for n in v["notes"]), \
        "both peaks are at step 200 here — nothing to warn about"


def test_peak_success_and_peak_total_are_reported_as_separate_moments(tmp_path):
    """Each key peaks on its own schedule. The success term topping out at step 100 while the
    total tops out at step 300 describes two different policies, and `checkpoint_best.pth`
    tracks the total — so a reader told only `step` would render the wrong one and attribute
    the peak success_rate to it."""
    m, d = _write(tmp_path, [
        {"step": 100, "reward/total/episodic_return_mean": 200.0,
         "reward/stack_success/episodic_return_mean": 180.0},
        {"step": 200, "reward/total/episodic_return_mean": 400.0,
         "reward/stack_success/episodic_return_mean": 100.0},
        {"step": 300, "reward/total/episodic_return_mean": 900.0,
         "reward/stack_success/episodic_return_mean": 40.0},
    ])
    v = _run(m, d)
    assert v["peak"]["success_rate"] == 180.0 / 200.0
    assert v["peak"]["success_rate_step"] == 100
    assert v["peak"]["total_return"] == 900.0
    assert v["peak"]["total_return_step"] == 300
    assert any("different policies" in n for n in v["notes"]), \
        "the divergence is the whole point — it must be stated, not left to be inferred"


def test_no_peak_note_when_the_run_ends_at_its_best(tmp_path):
    m, d = _write(tmp_path, [
        {"step": 100, "reward/total/episodic_return_mean": 10.0,
         "reward/stack_success/episodic_return_mean": 10.0},
        {"step": 200, "reward/total/episodic_return_mean": 90.0,
         "reward/stack_success/episodic_return_mean": 100.0},
    ])
    v = _run(m, d)
    assert v["peak"]["total_return"] == 90.0
    assert not any("PEAKED" in n for n in v["notes"])


def test_declared_artifacts_that_do_not_exist_are_dropped_and_reported(tmp_path):
    """A silent `cp` failure would otherwise leave the verdict pointing at a dead path —
    discovered exactly when the designer is already suspicious of the numbers."""
    m, d = _write(tmp_path, [
        {"step": 1, "reward/total/episodic_return_mean": 1.0,
         "reward/stack_success/episodic_return_mean": 100.0},
    ])
    real = tmp_path / "render.mp4"; real.write_bytes(b"mp4")
    v = _run(m, d,
             "--artifact", f"render_mp4={real}",
             "--artifact", f"curves_dir={tmp_path / 'curves'}")     # never written
    assert v["artifacts"] == {"render_mp4": str(real)}, "dead path must not survive"
    assert any("curves_dir" in n and "do not exist" in n for n in v["notes"])


def test_all_artifacts_present_produces_no_note(tmp_path):
    m, d = _write(tmp_path, [
        {"step": 1, "reward/total/episodic_return_mean": 1.0,
         "reward/stack_success/episodic_return_mean": 100.0},
    ])
    mp4 = tmp_path / "render.mp4"; mp4.write_bytes(b"mp4")
    curves = tmp_path / "curves"; curves.mkdir()
    v = _run(m, d, "--artifact", f"render_mp4={mp4}", "--artifact", f"curves_dir={curves}")
    assert set(v["artifacts"]) == {"render_mp4", "curves_dir"}
    assert not any("do not exist" in n for n in v["notes"])


def test_scoring_without_an_analysis_is_refused(tmp_path):
    """A `scored` candidate is one whose rollout was watched. Scoring one with no analysis
    would publish numbers with no account of what the policy actually did."""
    m, d = _write(tmp_path, [
        {"step": 1, "reward/total/episodic_return_mean": 1.0,
         "reward/stack_success/episodic_return_mean": 100.0},
    ])
    err = _run(m, d, "--status", "scored", expect_rc=1)
    assert "--analysis-json is required" in err


def test_an_unanswered_aspect_is_refused(tmp_path):
    """Coverage is the half a machine can check. It cannot tell whether 'the gripper never
    closes' is true; it can tell that nobody addressed termination at all."""
    m, d = _write(tmp_path, [
        {"step": 1, "reward/total/episodic_return_mean": 1.0,
         "reward/stack_success/episodic_return_mean": 100.0},
    ])
    a = _analysis(tmp_path, termination=_DROP)
    err = _run(m, d, "--analysis-json", str(a), expect_rc=1)
    assert "missing required key 'termination'" in err


def test_a_placeholder_answer_is_refused(tmp_path):
    """Same rule the visual passes use: an answer that cannot be contradicted by the frames
    is not evidence."""
    m, d = _write(tmp_path, [
        {"step": 1, "reward/total/episodic_return_mean": 1.0,
         "reward/stack_success/episodic_return_mean": 100.0},
    ])
    for junk in ("clean", "ok", "Looks correct.", "  n/a "):
        a = _analysis(tmp_path, physical_validity=junk)
        err = _run(m, d, "--analysis-json", str(a), expect_rc=1)
        assert "physical_validity" in err, junk


def test_explicit_uncertainty_is_accepted(tmp_path):
    """The checklist must not push toward confabulation: 'I could not see' is a real answer,
    and is exactly what a forced-choice schema would destroy."""
    m, d = _write(tmp_path, [
        {"step": 1, "reward/total/episodic_return_mean": 1.0,
         "reward/stack_success/episodic_return_mean": 100.0},
    ])
    a = _analysis(tmp_path, reward_hacking="unclear: only the arm is in frame")
    v = _run(m, d, "--analysis-json", str(a))
    assert v["analysis"]["reward_hacking"].startswith("unclear")


def test_a_typoed_aspect_is_refused_rather_than_silently_dropped(tmp_path):
    """A misspelled key would otherwise leave that aspect unanswered while the checklist
    looks full — the exact failure the checklist exists to prevent."""
    m, d = _write(tmp_path, [
        {"step": 1, "reward/total/episodic_return_mean": 1.0,
         "reward/stack_success/episodic_return_mean": 100.0},
    ])
    a = _analysis(tmp_path, terminaton="never fires", termination=_DROP)
    err = _run(m, d, "--analysis-json", str(a), expect_rc=1)
    assert "unknown key(s) ['terminaton']" in err


def test_checkpoint_watched_must_say_which_policy(tmp_path):
    """On a run that peaked and collapsed, the rendered best checkpoint and the final one are
    different policies; conflating them misreports what the reward produced."""
    m, d = _write(tmp_path, [
        {"step": 1, "reward/total/episodic_return_mean": 1.0,
         "reward/stack_success/episodic_return_mean": 100.0},
    ])
    a = _analysis(tmp_path, checkpoint_watched="the good one")
    err = _run(m, d, "--analysis-json", str(a), expect_rc=1)
    assert "checkpoint_watched" in err


def test_smoke_failed_candidate_needs_no_rollout_aspects(tmp_path):
    """It never trained, so there is no video. Demanding eight observations of a rollout that
    does not exist would only manufacture them — but it still owes a failure_mode."""
    _, d = _write(tmp_path, [])
    a = tmp_path / "sf.json"
    a.write_text(json.dumps({"failure_mode": "contact sensor never reports a hit",
                             "findings": ["add a contact sensor"]}))
    v = _run(tmp_path / "absent.jsonl", d, "--status", "reward_smoke_failed",
             "--analysis-json", str(a))
    assert v["failure_mode"].startswith("contact sensor")
    assert v["analysis"] is None
