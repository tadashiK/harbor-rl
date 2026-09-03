"""Unit: scripts/reward-tuning-agent/curve_health.py — evidence for the early-stop rubric.

This tool decides nothing, but the candidate agent kills live training runs based on its
`concern` field, so the escalation ladder has to be exactly right. The expensive mistake is
a FALSE kill: a dip at 20-40 % of budget routinely recovers, and killing there costs a whole
iteration and pollutes the search. So the asymmetry below is the point —

  hard_fail    NaN / dead policy — act immediately, at any budget fraction
  confident_bad soft pattern, past the soft floor, not improving — act only if it persists
  watch        the same soft pattern too early, or contradicted by real progress — never act
  none         nothing to see
"""
import json
import subprocess
import sys

from _pluginmeta import ROOT

SCRIPT = ROOT / "scripts" / "reward-tuning-agent" / "curve_health.py"


def _metrics(tmp_path, rows):
    p = tmp_path / "metrics.jsonl"
    p.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return p


def _run(metrics, *args):
    r = subprocess.run([sys.executable, str(SCRIPT), "--metrics", str(metrics), *args],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    return json.loads(r.stdout)


def _curve(values, entropy=None, step0=100, stride=100):
    rows = []
    for i, v in enumerate(values):
        row = {"step": step0 + i * stride, "reward/total/episodic_return_mean": v}
        if entropy is not None:
            row["train/entropy"] = entropy[i]
        rows.append(row)
    return rows


def test_healthy_rising_run_is_no_concern(tmp_path):
    m = _metrics(tmp_path, _curve([1, 5, 12, 25, 44, 70, 96, 130, 170, 210]))
    v = _run(m, "--total-steps", "1000")
    assert v["concern"] == "none"
    assert "improving" in v["flags"]


def test_nan_is_a_hard_fail_at_any_budget_fraction(tmp_path):
    """NaN does not recover, so the soft floor must not gate it."""
    m = _metrics(tmp_path, _curve([1.0, 2.0, float("nan")]))
    v = _run(m, "--total-steps", "100000")           # ~0.3 % of budget
    assert v["budget_frac"] < 0.05
    assert v["concern"] == "hard_fail" and "nan_inf" in v["flags"]


def test_dead_policy_is_a_hard_fail(tmp_path):
    """Entropy collapsed to ~0 with a flat return — the policy stopped exploring and
    learned nothing. Also must bypass the floor."""
    m = _metrics(tmp_path, _curve([1.0] * 8, entropy=[2.0, 1.0, 0.4, 0.1, 0.02, 0.01, 0.005, 0.001]))
    v = _run(m, "--total-steps", "100000")
    assert v["concern"] == "hard_fail" and "dead_policy" in v["flags"]


def test_flat_run_early_is_only_watch(tmp_path):
    """The dip-and-recover trap: the same evidence early must never be a kill."""
    m = _metrics(tmp_path, _curve([1.0] * 6))
    v = _run(m, "--total-steps", "100000")
    assert "no_learning" in v["flags"]
    assert v["concern"] == "watch", "a flat early curve must not be actionable"


def test_flat_run_past_the_soft_floor_is_confident_bad(tmp_path):
    m = _metrics(tmp_path, _curve([1.0] * 10))
    v = _run(m, "--total-steps", "1000", "--soft-floor", "0.5")
    assert v["budget_frac"] >= 0.5
    assert v["concern"] == "confident_bad"


def test_soft_floor_is_configurable(tmp_path):
    """Same curve, same budget — only the floor moves. Guards the knob the agent passes."""
    m = _metrics(tmp_path, _curve([1.0] * 10))       # last step 1000, i.e. 50 % of 2000
    assert _run(m, "--total-steps", "2000")["budget_frac"] == 0.5
    assert _run(m, "--total-steps", "2000", "--soft-floor", "0.5")["concern"] == "confident_bad"
    assert _run(m, "--total-steps", "2000", "--soft-floor", "0.9")["concern"] == "watch"


def test_real_progress_suppresses_a_soft_flag(tmp_path):
    """A run that is still climbing is never a kill candidate, however late it is."""
    rows = _curve([1, 4, 9, 16, 25, 36, 49, 64, 81, 100])
    for r in rows:
        r["train/success_rate"] = r["reward/total/episodic_return_mean"] / 200.0
    v = _run(_metrics(tmp_path, rows), "--total-steps", "1000")
    assert "improving" in v["flags"]
    assert v["concern"] in ("none", "watch")
    assert v["concern"] != "confident_bad"


def test_success_rate_proxy_is_the_term_return_over_its_weight(tmp_path):
    rows = [{"step": 100, "reward/total/episodic_return_mean": 30.0,
             "reward/stack_success/episodic_return_mean": 50.0}]
    v = _run(_metrics(tmp_path, rows), "--success-term", "stack_success",
             "--success-weight", "200")
    assert v["success_signal"]["success_rate_proxy"] == 0.25
    assert v["success_signal"]["source"] == "reward/stack_success/episodic_return_mean"


def test_partial_trailing_line_is_tolerated(tmp_path):
    """The file is read while training is still writing it."""
    p = tmp_path / "metrics.jsonl"
    p.write_text('{"step": 1, "reward/total/episodic_return_mean": 1.0}\n'
                 '{"step": 2, "reward/total/epi')          # torn write
    v = _run(p, "--total-steps", "100")
    assert v["n_points"] >= 1 and v["concern"] in ("none", "watch")


def test_no_budget_means_no_floor_and_so_no_confident_bad(tmp_path):
    """Without --total-steps there is no budget fraction, so a soft flag can never be
    confident — the agent would otherwise kill on evidence it cannot place in time."""
    m = _metrics(tmp_path, _curve([1.0] * 10))
    v = _run(m)
    assert v["budget_frac"] is None
    assert v["concern"] == "watch"
