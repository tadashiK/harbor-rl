"""Unit: scripts/task-generator/coacd_decompose.py — pre-decomposition and its provenance.

The decomposition itself needs the `coacd` package and a real mesh, so what is covered here is
everything around it: the manifest that PROVES CoACD produced a collision, the hull budget, and
the inspection that tells a USD or MJCF apart from a mesh. Those are the parts S1's C8 depends
on, and the parts that would silently accept a runtime-cooked asset if they were wrong.
"""
import json
import subprocess
import sys

import pytest

from _pluginmeta import ROOT

SCRIPT = ROOT / "scripts" / "task-generator" / "coacd_decompose.py"

sys.path.insert(0, str(SCRIPT.parent))
import coacd_decompose as cd  # noqa: E402


def _run(*args):
    r = subprocess.run([sys.executable, str(SCRIPT), *args], capture_output=True, text=True)
    return r.returncode, json.loads(r.stdout) if r.stdout.strip() else {}


# ---------------------------------------------------------------- the hull budget

def test_a_manifest_over_budget_is_rejected():
    """32 is the whole point: a 128-hull decomposition renders fine and overflows the contact
    buffers at high num_envs, which is how it was discovered."""
    data = {"tool": "coacd", "hulls": 128, "parts": [f"p{i}" for i in range(128)]}
    problems = cd.validate_manifest(data)
    assert any("exceeds the 32" in p for p in problems)
    assert any("threshold" in p for p in problems), "must say how to get fewer hulls"


def test_a_manifest_within_budget_passes():
    data = {"tool": "coacd", "hulls": 24, "parts": [f"p{i}" for i in range(24)]}
    assert cd.validate_manifest(data) == []


def test_a_manifest_from_another_tool_is_rejected():
    """The manifest exists to prove provenance; accepting any tool would defeat it."""
    data = {"tool": "isaac_convex_decomposition", "hulls": 8, "parts": ["a"] * 8}
    assert any("not 'coacd'" in p for p in cd.validate_manifest(data))


def test_part_count_must_match_the_claimed_hull_count():
    data = {"tool": "coacd", "hulls": 32, "parts": ["a", "b"]}
    assert any("lists 2 parts" in p for p in cd.validate_manifest(data))


def test_manifest_path_is_one_rule_so_the_check_can_find_it(tmp_path):
    assert cd.manifest_path(tmp_path / "mug.usd").name == "mug.coacd.json"
    assert cd.manifest_path(tmp_path / "mug.obj").name == "mug.coacd.json"


# ---------------------------------------------------------------- inspection

def test_a_usd_asking_the_simulator_to_cook_is_flagged(tmp_path):
    """This is the case the whole check exists for — and it is invisible in a render."""
    usd = tmp_path / "mug.usda"          # TEXT usd — a binary usdc cannot be scanned, see below
    usd.write_bytes(b"#usda 1.0\nstuff physics:approximation = \"convexDecomposition\"")
    _, out = _run("--input", str(usd), "--inspect")
    assert out["runtime_cooked"] is True
    assert not out["ok"]
    assert any("cooks this at" in p for p in out["problems"])
    assert any("coacd_decompose" in p or "Re-decompose" in p or "pre-decompose" in p.lower()
               for p in out["problems"])


def test_a_usd_of_convex_hull_parts_with_a_manifest_passes(tmp_path):
    usd = tmp_path / "mug.usda"
    usd.write_bytes(b'#usda 1.0\n convexHull convexHull convexHull')
    (tmp_path / "mug.coacd.json").write_text(json.dumps(
        {"tool": "coacd", "hulls": 3, "parts": ["a", "b", "c"]}))
    _, out = _run("--input", str(usd), "--inspect")
    assert out["ok"], out["problems"]
    assert out["runtime_cooked"] is False


def test_an_asset_with_no_manifest_is_flagged(tmp_path):
    """No manifest means nothing records what produced the collision — unverifiable."""
    usd = tmp_path / "rack.usda"
    usd.write_bytes(b"#usda 1.0\n convexHull")
    _, out = _run("--input", str(usd), "--inspect")
    assert not out["ok"]
    assert any(".coacd.json" in p for p in out["problems"])


def test_a_manifest_over_budget_fails_inspection_too(tmp_path):
    usd = tmp_path / "mug.usda"
    usd.write_bytes(b"#usda 1.0\n convexHull")
    (tmp_path / "mug.coacd.json").write_text(json.dumps(
        {"tool": "coacd", "hulls": 128, "parts": ["p"] * 128}))
    _, out = _run("--input", str(usd), "--inspect")
    assert not out["ok"] and any("exceeds the 32" in p for p in out["problems"])


def test_mjcf_inspection_reports_its_mesh_geoms(tmp_path):
    xml = tmp_path / "robot.xml"
    xml.write_text('<mujoco><asset><mesh name="m" file="mug.obj"/></asset>'
                   '<worldbody><geom mesh="m"/><geom mesh="m"/></worldbody></mujoco>')
    _, out = _run("--input", str(xml), "--inspect")
    assert out["format"] == "mjcf"
    assert out["mesh_assets"] == ["mug.obj"] and out["mesh_geoms"] == 2


# ---------------------------------------------------------------- CLI contract

def test_decomposing_a_usd_is_refused_with_the_right_advice(tmp_path):
    """You cannot CoACD a USD — you decompose the source mesh and rebuild from the parts."""
    usd = tmp_path / "mug.usd"
    usd.write_bytes(b"PXR-USDC\x00")
    rc, out = _run("--input", str(usd), "--out-dir", str(tmp_path / "o"))
    assert rc != 0 and "not a mesh" in out["error"] and "--inspect" in out["error"]


def test_decompose_requires_an_out_dir(tmp_path):
    mesh = tmp_path / "m.obj"
    mesh.write_text("v 0 0 0\n")
    rc, out = _run("--input", str(mesh))
    assert rc != 0 and "--out-dir" in out["error"]


def test_missing_input_is_a_caller_error(tmp_path):
    rc, out = _run("--input", str(tmp_path / "nope.obj"), "--inspect")
    assert rc != 0 and "no such file" in out["error"]


def test_missing_coacd_says_how_to_install_it(tmp_path):
    """The venv routinely lacks it; a bare ImportError traceback is not an instruction."""
    pytest.importorskip("trimesh", reason="only meaningful when trimesh is present")
    if cd is None:  # pragma: no cover
        pytest.skip("module unavailable")
    try:
        import coacd  # noqa: F401
        pytest.skip("coacd IS installed here, so the guard cannot fire")
    except ImportError:
        pass
    mesh = tmp_path / "m.obj"
    mesh.write_text("v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n")
    rc, out = _run("--input", str(mesh), "--out-dir", str(tmp_path / "o"))
    assert rc != 0 and "pip install coacd" in out["error"]


@pytest.mark.parametrize("flag", ["-h", "--help"])
def test_help_works(flag):
    r = subprocess.run([sys.executable, str(SCRIPT), flag], capture_output=True, text=True)
    assert r.returncode == 0 and "CoACD" in r.stdout


def test_binary_usdc_reports_unknown_rather_than_ok(tmp_path):
    """The bug this replaces: a byte scan of a binary usdc finds no tokens and reported
    `runtime_cooked: False` — "fine" — for a real asset cooked at 128 hulls. The original
    fixture hid it by writing plaintext tokens into a fake usdc, which is not a usdc."""
    usd = tmp_path / "mug.usd"
    usd.write_bytes(b"PXR-USDC\x00" + b"\x1f\x8b compressed token table")
    _, out = _run("--input", str(usd), "--inspect")
    assert out["encoding"] == "usdc-binary"
    assert out["runtime_cooked"] is None, "unknown must not be reported as False"
    assert not out["ok"]
    assert any("binary usdc" in p and "C8" in p for p in out["problems"])
