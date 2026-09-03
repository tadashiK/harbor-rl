"""Unit: scripts/reward-tuning-agent/_metrics.py — the shared metrics.jsonl reader.

Both `score_iter.py` (the verdict a candidate is judged on) and `curve_health.py` (the
mid-run early-stop decision) read the training log through this module, so a
misinterpretation here does not surface as an error — it surfaces as a wrong score or a
wrongly-killed run. The cases below are the ones where a plausible file shape used to be
read as something other than what it says.
"""
import sys

from _pluginmeta import ROOT

SCRIPT_DIR = ROOT / "scripts" / "reward-tuning-agent"
sys.path.insert(0, str(SCRIPT_DIR))
import _metrics as M  # noqa: E402


def test_long_format_is_read_as_key_step_value():
    rows = [{"key": "reward/total/episodic_return_mean", "step": 10, "value": 1.0},
            {"key": "reward/total/episodic_return_mean", "step": 20, "value": 3.0}]
    assert M.series(rows) == {"reward/total/episodic_return_mean": [(10, 1.0), (20, 3.0)]}


def test_wide_format_is_read_as_one_row_per_tick():
    rows = [{"step": 10, "a": 1.0, "b": 2.0}, {"step": 20, "a": 3.0, "b": 4.0}]
    assert M.series(rows) == {"a": [(10, 1.0), (20, 3.0)], "b": [(10, 2.0), (20, 4.0)]}


def test_a_wide_preamble_row_does_not_flip_a_long_file_to_wide():
    """Detecting the format from rows[0] alone made one header line — a config dump, a run
    banner — decide how the whole file was parsed. Every real row is long here, so a
    majority vote must keep reading it as long; the old check returned {} and the caller
    scored the run `no_metrics` on a log that was completely intact."""
    rows = [{"run": "ppo_stack", "num_envs": 4096}] + [
        {"key": "reward/total/episodic_return_mean", "step": s, "value": float(s)}
        for s in (10, 20, 30)
    ]
    out = M.series(rows)
    assert out["reward/total/episodic_return_mean"] == [(10, 10.0), (20, 20.0), (30, 30.0)]


def test_an_explicit_null_step_falls_back_to_row_order():
    """`.get("step", i)` only covers an ABSENT key. A logger that writes `"step": null`
    before the first global step is set left None in the tuples, and sorting None against
    an int raises TypeError — turning a readable log into a crash."""
    rows = [{"step": None, "a": 1.0}, {"step": 20, "a": 3.0}]
    assert M.series(rows) == {"a": [(0, 1.0), (20, 3.0)]}


def test_final_values_takes_the_largest_step_not_the_last_line():
    rows = [{"step": 30, "a": 3.0}, {"step": 10, "a": 1.0}]
    assert M.final_values(M.series(rows)) == {"a": 3.0}


def test_peak_values_reports_each_key_at_its_own_best_step():
    """The property score_iter depends on: two keys peaking at different steps are two
    different moments, and the reader must not collapse them onto one."""
    rows = [{"step": 10, "a": 9.0, "b": 1.0}, {"step": 20, "a": 2.0, "b": 8.0}]
    assert M.peak_values(M.series(rows)) == {"a": (9.0, 10), "b": (8.0, 20)}


def test_a_half_written_trailing_line_is_ignored(tmp_path):
    """The file is read while training is still appending to it."""
    p = tmp_path / "metrics.jsonl"
    p.write_text('{"step": 10, "a": 1.0}\n{"step": 20, "a":')
    assert M.load_series(str(p)) == {"a": [(10, 1.0)]}
