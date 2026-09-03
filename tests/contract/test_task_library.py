"""Contract: every task-library entry is uniform enough to be searched and replicated.

The library is what `task-generator` and `reward-tuning-agent` search before designing
anything, via `knowledge/references/task-library-search.md`. That search matches on the
metadata header, so an entry that states it differently is not "slightly inconsistent" —
it is invisible to the search that exists to find it.

This is not hypothetical. 51 of the 52 entries arrived in one bulk import, and the import
set the de-facto standard: only one of them carried the metadata block the template
specified, 45 carried an origin-repo pointer section, and roughly half used the per-section
subblocks. Nothing checked, so nothing converged. These tests are the check.

What is deliberately NOT asserted: the §-section bodies. Their prose and depth vary with the
task and should — a 305-line push-cube and a 1777-line hang-mug are both correct. Only the
things a consumer parses are pinned.
"""
import re

import pytest

from _pluginmeta import ROOT

LIBRARY = ROOT / "knowledge" / "experiences" / "task-library"
ENTRIES = sorted(p for p in LIBRARY.rglob("*.md") if p.name != "README.md")

# The header every entry states identically, because it is what the library is searched on.
# Order is fixed too: a reader scanning many entries reads position, not labels.
REQUIRED_FIELDS = ("robot", "simulator", "objects", "bimanual", "summary")

SECTIONS = ("§1", "§2", "§3", "§4", "§5", "§6", "§7")

# Provenance. An entry is a task design that is assumed correct and replicable; how it came
# to exist is not part of that, and these fields rot (a commit sha for a repo the reader does
# not have, a timestamp that says nothing about whether the design still holds).
BANNED_FIELDS = ("benchmark_family", "source_repo", "probed_from_commit",
                 "probed_at", "canonical_build")

TITLE_RE = re.compile(r"^#\s+\S.*—\s*Implementation Spec\s*$")


def _field(text, name):
    m = re.search(rf"^-\s*{name}\s*:\s*(.+?)\s*$", text, re.M)
    return m.group(1) if m else None


def test_the_library_is_not_empty():
    """A glob that silently matches nothing would make every test below vacuously pass."""
    assert len(ENTRIES) > 40, f"only {len(ENTRIES)} entries found under {LIBRARY}"


@pytest.mark.parametrize("path", ENTRIES, ids=lambda p: p.stem)
def test_entry_has_the_metadata_header(path):
    text = path.read_text(encoding="utf-8")
    missing = [f for f in REQUIRED_FIELDS if _field(text, f) is None]
    assert not missing, (
        f"{path.relative_to(LIBRARY)}: missing metadata field(s) {missing}. "
        f"Every entry states {list(REQUIRED_FIELDS)} — that header is what "
        f"task-library-search.md matches on, so an entry missing one cannot be found."
    )


@pytest.mark.parametrize("path", ENTRIES, ids=lambda p: p.stem)
def test_metadata_fields_are_in_the_canonical_order(path):
    text = path.read_text(encoding="utf-8")
    seen = [f for f in re.findall(r"^-\s*(\w+)\s*:", text, re.M) if f in REQUIRED_FIELDS]
    # first occurrence only — later sections may legitimately mention a field name
    first = []
    for f in seen:
        if f not in first:
            first.append(f)
    assert first == list(REQUIRED_FIELDS), (
        f"{path.relative_to(LIBRARY)}: metadata order is {first}, expected "
        f"{list(REQUIRED_FIELDS)}"
    )


@pytest.mark.parametrize("path", ENTRIES, ids=lambda p: p.stem)
def test_bimanual_is_a_boolean(path):
    """It is a filter, not prose: 'two arms (sort of)' cannot be matched on."""
    v = _field(path.read_text(encoding="utf-8"), "bimanual")
    assert v in ("true", "false"), (
        f"{path.relative_to(LIBRARY)}: bimanual is {v!r}; must be exactly 'true' or 'false'"
    )


@pytest.mark.parametrize("path", ENTRIES, ids=lambda p: p.stem)
def test_robot_simulator_and_objects_are_answered(path):
    """Guards the failure that leaves the field present and the answer absent."""
    text = path.read_text(encoding="utf-8")
    for f in ("robot", "simulator", "objects", "summary"):
        v = _field(text, f)
        assert v and len(v) >= 4 and v.lower() not in {"tbd", "todo", "n/a", "none", "-"}, (
            f"{path.relative_to(LIBRARY)}: '{f}' is {v!r} — state it. A locomotion task with "
            f"no objects says so as 'none (flat terrain)', naming the terrain."
        )


@pytest.mark.parametrize("path", ENTRIES, ids=lambda p: p.stem)
def test_entry_carries_every_section_anchor(path):
    """`/harbor:task-create from=<spec>` passes each section's block to a subagent; a
    missing anchor silently drops that section from the reproduction."""
    text = path.read_text(encoding="utf-8")
    missing = [s for s in SECTIONS if not re.search(rf"^##\s+{s}\b", text, re.M)]
    assert not missing, f"{path.relative_to(LIBRARY)}: missing section anchor(s) {missing}"


@pytest.mark.parametrize("path", ENTRIES, ids=lambda p: p.stem)
def test_title_is_the_implementation_spec_form(path):
    first = path.read_text(encoding="utf-8").split("\n", 1)[0]
    assert TITLE_RE.match(first), (
        f"{path.relative_to(LIBRARY)}: first line is {first!r}; expected "
        f"'# <TaskID> — Implementation Spec'"
    )


@pytest.mark.parametrize("path", ENTRIES, ids=lambda p: p.stem)
def test_entry_carries_no_provenance(path):
    """An entry describes a task, never how the file came to exist."""
    text = path.read_text(encoding="utf-8")
    found = [f for f in BANNED_FIELDS if f in text]
    assert not found, (
        f"{path.relative_to(LIBRARY)}: carries provenance {found}. Entries are assumed "
        f"correct and replicable; where the spec came from is not part of the design."
    )
    assert not re.search(r"^##\s+Source files", text, re.M), (
        f"{path.relative_to(LIBRARY)}: has a 'Source files' section — origin-repo paths "
        f"point at a tree the reader may not have, and the verbatim code is already inline."
    )


def test_filename_simulator_suffix_matches_the_simulator_field():
    """Filenames end in the simulator (`-IsaacLab.md`), and update-experience derives that
    suffix from the `simulator:` field. If the two can disagree, the naming rule is fiction."""
    bad = []
    for p in ENTRIES:
        sim = _field(p.read_text(encoding="utf-8"), "simulator") or ""
        # `IsaacLab (Isaac Sim, manager-based)` -> `IsaacLab`
        head = sim.split("(")[0].strip()
        stem = re.sub(r"-v\d+$", "", p.stem)      # -v2 collision suffix is not the simulator
        if head and not stem.endswith(head):
            bad.append(f"{p.name} declares simulator {head!r}")
    assert not bad, "filename/simulator mismatch:\n  " + "\n  ".join(bad)
