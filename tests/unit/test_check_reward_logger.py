"""Unit: scripts/rl-run/check_reward_logger.py — task_overview.md rc classification."""
import subprocess
import sys

from _pluginmeta import ROOT

SCRIPT = ROOT / "scripts" / "rl-run" / "check_reward_logger.py"


def _overview(tmp_path, rows):
    d = tmp_path / "harbor" / "benchmark-generator"
    d.mkdir(parents=True)
    lines = ["| Task | Reward logger added | Note |", "|---|---|---|"]
    lines += [f"| `{task}` | {token} | x |" for task, token in rows]
    (d / "task_overview.md").write_text("\n".join(lines) + "\n")
    return tmp_path


def _rc(repo, task):
    return subprocess.run([sys.executable, str(SCRIPT), "--repo", str(repo), "--task", task]).returncode


def test_yes_is_rc0(tmp_path):
    assert _rc(_overview(tmp_path, [("A", "yes")]), "A") == 0


def test_total_only_is_rc3(tmp_path):
    assert _rc(_overview(tmp_path, [("B", "total only")]), "B") == 3


def test_no_is_rc1(tmp_path):
    assert _rc(_overview(tmp_path, [("C", "no")]), "C") == 1


def test_task_not_in_table_is_rc2(tmp_path):
    assert _rc(_overview(tmp_path, [("A", "yes")]), "Missing") == 2


def test_missing_overview_is_rc2(tmp_path):
    assert _rc(tmp_path, "A") == 2
