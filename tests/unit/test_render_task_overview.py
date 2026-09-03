"""Unit: scripts/benchmark-generator/render_task_overview.py — the task_overview.md renderer.

Pure helpers, tested directly: the per-family task categorizer and the markdown row escaper.
The categoriser is what groups the overview table, so a family whose ids stop matching
silently collapses every task into one bucket — visible only by reading the doc.
"""
import importlib.util

import pytest

from _pluginmeta import ROOT

SRC = ROOT / "scripts" / "benchmark-generator" / "render_task_overview.py"


def _mod():
    spec = importlib.util.spec_from_file_location("render_task_overview", SRC)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


M = _mod()


@pytest.mark.parametrize("family,task_id", [
    ("isaaclab", "Isaac-Lift-Cube-Franka-v0"),
    ("dmcontrol", "walker-walk"),
    ("bidexhands", "ShadowHandOver"),
    ("generic", "SomeTask-v3"),
])
def test_every_family_categorizes_to_a_non_empty_label(family, task_id):
    label = M._categorize(family, task_id)
    assert isinstance(label, str) and label.strip(), f"{family}/{task_id} produced no category"


def test_categories_actually_discriminate():
    """If everything mapped to one bucket the overview table would be useless."""
    ids = ["Isaac-Lift-Cube-Franka-v0", "Isaac-Velocity-Flat-Anymal-C-v0",
           "Isaac-Repose-Cube-Allegro-v0"]
    labels = {M._categorize("isaaclab", i) for i in ids}
    assert len(labels) > 1, f"all isaaclab ids collapsed into {labels}"


def test_unknown_family_falls_back_to_generic_without_raising():
    assert M._categorize("no-such-family", "Whatever-v0")


def test_markdown_row_is_pipe_delimited():
    row = M._md_table_row(["a", "b", "c"])
    assert row.startswith("|") and row.rstrip().endswith("|")
    assert row.count("|") >= 4


def test_row_cells_do_not_break_the_table():
    """A cell containing a pipe would silently split the column, so it must be escaped."""
    row = M._md_table_row(["ok", "has | pipe", "end"])
    assert r"has \| pipe" in row
    assert row.count("|") - row.count(r"\|") == 4, f"unescaped pipe leaked into: {row}"
