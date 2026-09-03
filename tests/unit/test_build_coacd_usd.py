"""Unit: scripts/task-generator/build_coacd_usd.py — the OBJ reader feeding collision geometry.

Authoring the USD needs `pxr`, which only exists inside a running simulator, so that half is
exercised by S1's C8 on a real stage. What is covered here is the mesh parsing underneath it —
and it matters more than it looks: every vertex this reader drops or mis-indexes becomes a
wrong collision shape, silently, in an asset that still renders correctly.
"""
import sys

import pytest

from _pluginmeta import ROOT

SCRIPT = ROOT / "scripts" / "task-generator" / "build_coacd_usd.py"
sys.path.insert(0, str(SCRIPT.parent))
import build_coacd_usd as b  # noqa: E402


def _write(tmp_path, text):
    p = tmp_path / "m.obj"
    p.write_text(text)
    return p


def test_triangles_are_read_with_obj_one_based_indices(tmp_path):
    """OBJ indexes from 1; an off-by-one here rotates every face onto the wrong vertices."""
    verts, faces = b._load_obj(_write(tmp_path, "v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n"))
    assert verts == [[0, 0, 0], [1, 0, 0], [0, 1, 0]]
    assert faces == [[0, 1, 2]]


def test_quads_are_fan_triangulated(tmp_path):
    """CoACD emits convex parts, which are routinely quads. Dropping the second triangle would
    leave a hole in the collider — invisible, since the visual mesh is a different prim."""
    obj = "v 0 0 0\nv 1 0 0\nv 1 1 0\nv 0 1 0\nf 1 2 3 4\n"
    _, faces = b._load_obj(_write(tmp_path, obj))
    assert faces == [[0, 1, 2], [0, 2, 3]]


def test_ngons_are_fan_triangulated(tmp_path):
    _, faces = b._load_obj(_write(tmp_path, "v 0 0 0\n" * 5 + "f 1 2 3 4 5\n"))
    assert faces == [[0, 1, 2], [0, 2, 3], [0, 3, 4]]


def test_face_indices_with_texture_and_normal_slots_are_handled(tmp_path):
    """`f v/vt/vn` is the common export form; splitting on '/' is what keeps it from
    ValueError-ing on a mesh that came out of any real DCC tool."""
    _, faces = b._load_obj(_write(tmp_path, "v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1/1/1 2/2/2 3/3/3\n"))
    assert faces == [[0, 1, 2]]


def test_non_geometry_lines_are_ignored(tmp_path):
    """Object/group/material/comment lines must not be mistaken for vertices."""
    obj = ("# comment\no part_000\nmtllib x.mtl\nusemtl m\ns off\n"
           "v 0 0 0\nv 1 0 0\nv 0 1 0\nvn 0 0 1\nvt 0 0\nf 1 2 3\n")
    verts, faces = b._load_obj(_write(tmp_path, obj))
    assert len(verts) == 3 and faces == [[0, 1, 2]]


def test_a_multi_object_obj_reads_as_one_merged_mesh(tmp_path):
    """This is the behavior that made the merged-OBJ shortcut fail: `o` does NOT separate
    meshes for a consumer that ignores it. It is why each CoACD part is authored as its own
    prim instead of being concatenated into one file."""
    obj = ("o a\nv 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n"
           "o b\nv 5 0 0\nv 6 0 0\nv 5 1 0\nf 4 5 6\n")
    verts, faces = b._load_obj(_write(tmp_path, obj))
    assert len(verts) == 6 and faces == [[0, 1, 2], [3, 4, 5]]


def test_missing_parts_dir_is_reported_not_raised(tmp_path):
    out = b.build(tmp_path / "nope", tmp_path / "src.obj", tmp_path / "o.usd", "Mug", 1.0, False)
    assert out["ok"] is False and "no part_*.obj" in out["error"]


@pytest.mark.parametrize("flag", ["-h", "--help"])
def test_help_works(flag):
    import subprocess
    r = subprocess.run([sys.executable, str(SCRIPT), flag], capture_output=True, text=True)
    assert r.returncode == 0 and "CoACD" in r.stdout
