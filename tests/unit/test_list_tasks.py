"""Unit: scripts/benchmark-generator/list_tasks.py — task-universe detection.

The live listers (isaaclab / dm_control / gymnasium) need those packages importable and are
out of reach in CI — but they return `None` when the import fails, which is exactly what
happens here. That leaves the parts that actually encode decisions fully testable: the
path-anchored Bi-DexHands lister, the spec-only fallback, and the detection ORDER.

Order is the subtle one. `_detect` tries path-anchored families before the generic gymnasium
registry, because Bi-DexHands envs are not gym-registered at all — flip it and a repo with
gymnasium installed reports the wrong universe.
"""
import importlib.util
import json
import subprocess
import sys

import pytest

from _pluginmeta import ROOT

SRC = ROOT / "scripts" / "benchmark-generator" / "list_tasks.py"


def _importable(name):
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


# "Out of reach in CI" above is an assumption, not a guarantee. Run this suite with a benchmark
# venv rather than a bare interpreter and the live listers DO import, so `_detect` takes a live
# branch instead of falling back: gymnasium returns its own registry, and isaaclab is worse —
# `_list_isaaclab` calls `AppLauncher(headless=True)` and imports `isaaclab_tasks` IN-PROCESS,
# booting Isaac Sim inside pytest. It never returns, and leaves Kit subprocesses that outlive
# the run. Skip rather than hang the suite.
_LIVE_LISTERS = [n for n in ("isaaclab", "gymnasium", "dm_control") if _importable(n)]
needs_bare_interpreter = pytest.mark.skipif(
    bool(_LIVE_LISTERS),
    reason=f"needs a bare interpreter; live listers importable: {', '.join(_LIVE_LISTERS)}",
)


def _mod():
    spec = importlib.util.spec_from_file_location("list_tasks", SRC)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


M = _mod()

PARSE_TASK = """\
from bidexhands.tasks.hand_base.vec_task import VecTaskPython
from bidexhands.tasks.shadow_hand_over import ShadowHandOver
from bidexhands.tasks.shadow_hand_catch import ShadowHandCatch
from bidexhands.tasks.shadow_hand_meta_ml1 import ShadowHandMetaML1
"""


def _bidex_repo(tmp_path, tasks=("ShadowHandOver", "ShadowHandCatch")):
    cfg = tmp_path / "bidexhands" / "cfg"
    cfg.mkdir(parents=True)
    utils = tmp_path / "bidexhands" / "utils"
    utils.mkdir(parents=True)
    (utils / "parse_task.py").write_text(PARSE_TASK)
    for t in tasks:
        (cfg / f"{t}.yaml").write_text(f"env:\n  episodeLength: 125\nname: {t}\n")
    (tmp_path / "harbor").mkdir(exist_ok=True)
    return tmp_path


def _spec_repo(tmp_path, tasks):
    d = tmp_path / "harbor" / "benchmark-generator"
    d.mkdir(parents=True)
    (d / "benchmark-spec.json").write_text(json.dumps({"tasks": tasks}))
    return tmp_path


# --- bidexhands (path-anchored, fully testable) ------------------------------

def test_bidexhands_lists_only_instantiable_tasks(tmp_path):
    """A cfg yaml with no matching import in parse_task.py cannot be built by `bi.make`,
    so listing it would hand the agent a task that fails at construction."""
    repo = _bidex_repo(tmp_path, tasks=("ShadowHandOver", "NotImported"))
    out = M._list_bidexhands(repo)
    assert [t["id"] for t in out["tasks"]] == ["ShadowHandOver"]


def test_bidexhands_drops_meta_variants(tmp_path):
    repo = _bidex_repo(tmp_path, tasks=("ShadowHandOver", "ShadowHandMetaML1"))
    ids = [t["id"] for t in M._list_bidexhands(repo)["tasks"]]
    assert "ShadowHandMetaML1" not in ids, "meta tasks need task_type=Meta and are not listed"


def test_bidexhands_reads_episode_length(tmp_path):
    out = M._list_bidexhands(_bidex_repo(tmp_path))
    assert all(t["max_episode_steps"] == 125 for t in out["tasks"])


def test_bidexhands_absent_returns_none(tmp_path):
    assert M._list_bidexhands(tmp_path) is None


def test_bidexhands_with_no_matching_tasks_returns_none(tmp_path):
    """None (not an empty result) is what lets _detect fall through to the next family."""
    assert M._list_bidexhands(_bidex_repo(tmp_path, tasks=("NotImported",))) is None


# --- spec-only fallback ------------------------------------------------------

def test_spec_only_fallback_reports_captured_tasks(tmp_path):
    repo = _spec_repo(tmp_path, [{"id": "Demo-A-v0", "max_episode_steps": 300}])
    out = M._spec_only_fallback(repo)
    assert out["family"] == "spec_only"
    assert out["tasks"][0]["id"] == "Demo-A-v0"
    assert out["tasks"][0]["max_episode_steps"] == 300


def test_spec_only_fallback_with_no_spec_is_empty_not_an_error(tmp_path):
    out = M._spec_only_fallback(tmp_path)
    assert out["family"] == "spec_only" and out["tasks"] == []


# --- detection order ---------------------------------------------------------

def test_path_anchored_family_wins_over_the_generic_fallback(tmp_path):
    """Bi-DexHands envs are not gym-registered, so a path-anchored hit must be preferred."""
    repo = _bidex_repo(tmp_path)
    _spec_repo(repo, [{"id": "Should-Not-Win-v0"}])
    out = M._detect(repo)
    assert out["family"] == "bidexhands"


@needs_bare_interpreter
def test_detect_falls_back_to_spec_when_nothing_is_importable(tmp_path):
    repo = _spec_repo(tmp_path, [{"id": "Demo-A-v0"}])
    out = M._detect(repo)
    assert out["family"] == "spec_only" and out["tasks"][0]["id"] == "Demo-A-v0"


@needs_bare_interpreter
def test_every_result_carries_the_template_fields(tmp_path):
    """task_overview.md.template substitutes these directly; a missing key renders a hole."""
    for out in (M._detect(_bidex_repo(tmp_path)),
                M._detect(_spec_repo(tmp_path / "s", [{"id": "X-v0"}]))):
        assert {"family", "listing_function", "id_prefix", "tasks"} <= set(out)
        assert out["listing_function"], "listing_function is embedded in the doc; must be set"


# --- CLI ---------------------------------------------------------------------

def test_cli_refuses_a_non_harbor_repo(tmp_path):
    r = subprocess.run([sys.executable, str(SRC), "--repo", str(tmp_path)],
                       capture_output=True, text=True)
    assert r.returncode == 2
    assert "not a harbor benchmark repo" in r.stderr


@needs_bare_interpreter
def test_cli_writes_json_to_output_keeping_stdout_clean(tmp_path):
    """IsaacLab's Kit logs all over stdout, so --output is how the JSON stays parseable."""
    repo = _spec_repo(tmp_path, [{"id": "Demo-A-v0"}])
    out = tmp_path / "nested" / "tasks.json"
    r = subprocess.run(
        [sys.executable, str(SRC), "--repo", str(repo), "--output", str(out)],
        capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert json.loads(out.read_text())["tasks"][0]["id"] == "Demo-A-v0"
    assert r.stdout.strip() == "", "JSON leaked to stdout despite --output"
