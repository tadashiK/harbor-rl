"""Unit: scripts/rl-integration-generator/check_trial_artifacts.py — T4/T5 artifact checks."""
import json
import subprocess
import sys

from _pluginmeta import ROOT

SCRIPT = ROOT / "scripts" / "rl-integration-generator" / "check_trial_artifacts.py"


def _run(*args, expect_ok=True):
    r = subprocess.run([sys.executable, str(SCRIPT), *args], capture_output=True, text=True)
    if expect_ok:
        assert r.returncode == 0, r.stderr
        return json.loads(r.stdout)
    assert r.returncode != 0
    return r.stderr


def _trial(tmp_path, curves=1, tb=1, metric_lines=2, ckpt=True):
    t = tmp_path / "ppo_Task_2026"
    (t / "curves").mkdir(parents=True)
    (t / "tb").mkdir()
    for i in range(curves):
        (t / "curves" / f"c{i}.png").write_bytes(b"\x89PNG")
    for i in range(tb):
        (t / "tb" / f"events.out.tfevents.{i}").write_bytes(b"tb")
    if metric_lines:
        (t / "metrics.jsonl").write_text("".join('{"step":%d}\n' % i for i in range(metric_lines)))
    if ckpt:
        (t / "checkpoint.pth").write_bytes(b"ckpt")
    return t


def test_complete_trial_passes_both_tiers(tmp_path):
    v = _run("--trial", str(_trial(tmp_path)))
    assert v["tiers"] == {"T4": "pass", "T5": "pass"}
    assert v["checkpoints"] == ["checkpoint.pth"]
    assert v["metrics"]["lines"] == 2 and v["curves"]["count"] == 1
    assert v["notes"] == []


def test_missing_curves_fails_t4_only(tmp_path):
    v = _run("--trial", str(_trial(tmp_path, curves=0)))
    assert v["tiers"] == {"T4": "fail", "T5": "pass"}
    assert any("PNG curves" in n for n in v["notes"])


def test_empty_metrics_fails_t5(tmp_path):
    v = _run("--trial", str(_trial(tmp_path, metric_lines=0)))
    assert v["tiers"]["T5"] == "fail"


def test_absent_trial_is_a_result_not_a_crash(tmp_path):
    """A run that produced nothing must report, not blow up — the caller needs the verdict."""
    v = _run("--trial", str(tmp_path / "never_ran"))
    assert v["exists"] is False
    assert v["tiers"] == {"T4": "fail", "T5": "fail"}


def test_trial_dir_resolved_from_training_log(tmp_path):
    t = _trial(tmp_path)
    log = tmp_path / "T1.log"
    log.write_text(
        "epoch 1\n"
        f"saved checkpoint to {t}/checkpoint_100.pth\n"
        "epoch 2\n"
        f"saved checkpoint to {t}/checkpoint.pth\n"      # last save wins
        "done\n"
    )
    v = _run("--log", str(log))
    assert v["trial_dir"] == str(t) and v["tiers"]["T4"] == "pass"


def test_unresolvable_log_is_a_usage_error(tmp_path):
    log = tmp_path / "T1.log"
    log.write_text("training crashed before saving\n")
    err = _run("--log", str(log), expect_ok=False)
    assert "cannot resolve trial dir" in err


def test_wandb_mode_flags_a_missing_run_dir(tmp_path):
    t = _trial(tmp_path)
    assert _run("--trial", str(t))["notes"] == []
    v = _run("--trial", str(t), "--logging-mode", "wandb")
    assert any("wandb" in n for n in v["notes"])
    (t / "wandb").mkdir()
    assert _run("--trial", str(t), "--logging-mode", "wandb")["notes"] == []
