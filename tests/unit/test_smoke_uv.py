"""Unit: scripts/dependency-generator/smoke_uv.py — the pure half of the env smoke.

The tiers themselves shell into a real `.venv` and can't run in CI, but the logic that
decides what they MEAN can: which third-party imports get probed, how the project's
importable name is resolved, and how the two tier verdicts combine into the exit code.

That last one is the one worth pinning. The agent branches on `overall`, so getting the
precedence wrong turns a broken environment into a "partial" that the chain happily builds on.
"""
import importlib.util

import pytest

from _pluginmeta import ROOT

SRC = ROOT / "scripts" / "dependency-generator" / "smoke_uv.py"


def _mod():
    spec = importlib.util.spec_from_file_location("smoke_uv", SRC)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


M = _mod()


# --- overall verdict ---------------------------------------------------------

@pytest.mark.parametrize("t1,t2,expected", [
    ("pass", "pass",    "pass"),
    ("pass", "skipped", "pass"),      # not every repo exposes an importable package
    ("pass", "partial", "partial"),
    ("pass", "fail",    "fail"),
    ("fail", "pass",    "fail"),      # a broken env dominates whatever tier2 said
    ("fail", "skipped", "fail"),
    ("skipped", "pass", "partial"),   # tier1 never ran — cannot claim a pass
])
def test_overall_verdict_precedence(t1, t2, expected):
    assert M.overall_verdict(t1, t2) == expected


def test_a_broken_env_is_never_reported_as_partial():
    """`partial` is a non-blocking result the chain continues past — tier1 failures must
    not land there."""
    for t2 in ("pass", "partial", "fail", "skipped"):
        assert M.overall_verdict("fail", t2) == "fail"


# --- import collection -------------------------------------------------------

def _repo(tmp_path, pkg="demo_pkg", init_body="", scripts=None):
    (tmp_path / pkg).mkdir()
    (tmp_path / pkg / "__init__.py").write_text(init_body)
    for name, body in (scripts or {}).items():
        p = tmp_path / "scripts" / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(body)
    return tmp_path


def test_stdlib_imports_are_not_probed(tmp_path):
    """Probing stdlib would always pass and tell you nothing about the install."""
    repo = _repo(tmp_path, init_body="import os\nimport sys\nimport json\nimport torch\n")
    mods = M._collect_imports(repo, "demo_pkg")
    assert "torch" in mods
    assert not {"os", "sys", "json"} & set(mods)


def test_the_package_itself_is_probed_first(tmp_path):
    """Importing the project is the actual point of tier2; it must lead."""
    repo = _repo(tmp_path, init_body="import numpy\n")
    assert M._collect_imports(repo, "demo_pkg")[0] == "demo_pkg"


def test_from_and_plain_imports_both_counted(tmp_path):
    repo = _repo(tmp_path, init_body="from scipy.linalg import norm\nimport pandas.io\n")
    mods = M._collect_imports(repo, "demo_pkg")
    assert "scipy" in mods and "pandas" in mods, "submodule imports must reduce to their root"


def test_private_and_self_imports_are_skipped(tmp_path):
    repo = _repo(tmp_path, init_body="import _internal\nimport demo_pkg.sub\nimport click\n")
    mods = M._collect_imports(repo, "demo_pkg")
    assert "click" in mods
    assert "_internal" not in mods
    assert mods.count("demo_pkg") == 1, "self-import double-counted"


def test_scripts_dir_is_scanned_too(tmp_path):
    repo = _repo(tmp_path, init_body="", scripts={"run.py": "import gymnasium\n"})
    assert "gymnasium" in M._collect_imports(repo, "demo_pkg")


def test_unreadable_file_does_not_abort_collection(tmp_path):
    repo = _repo(tmp_path, init_body="import numpy\n",
                 scripts={"bad.py": ""})
    (repo / "scripts" / "bad.py").write_bytes(b"\xff\xfe\x00 import broken")
    assert "numpy" in M._collect_imports(repo, "demo_pkg")


# --- name resolution ---------------------------------------------------------

def test_project_name_read_from_pyproject(tmp_path):
    (tmp_path / "pyproject.toml").write_text('[project]\nname = "loco-mujoco"\nversion = "0.1"\n')
    assert M._project_name(tmp_path) == "loco-mujoco"


def test_project_name_falls_back_to_the_directory(tmp_path):
    assert M._project_name(tmp_path) == tmp_path.name


def test_hyphenated_project_resolves_to_the_underscore_package(tmp_path):
    """pyproject names hyphenate; importable packages do not."""
    (tmp_path / "loco_mujoco").mkdir()
    (tmp_path / "loco_mujoco" / "__init__.py").write_text("")
    assert M._import_name(tmp_path, "loco-mujoco") == "loco_mujoco"


def test_import_name_prefers_what_is_actually_on_disk(tmp_path):
    (tmp_path / "weird-name").mkdir()
    (tmp_path / "weird-name" / "__init__.py").write_text("")
    assert M._import_name(tmp_path, "weird-name") == "weird-name"


def test_import_name_guesses_underscore_when_nothing_is_on_disk(tmp_path):
    assert M._import_name(tmp_path, "not-here") == "not_here"
