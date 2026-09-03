"""Contract: the per-section reference files stay wired and uniform.

`knowledge/references/task-generator/s<N>-*.md` is the shared §1–§5 knowledge both `task-generator` (authoring)
and `reward-candidate-agent` (applying a bounded delta) load one file at a time. Two ways it
rots, neither of which `test_references.py` can see — that check only catches links pointing
at files that don't exist, not files that nothing points at:

  - an **orphan**: a section file no agent routes to, so the detail in it is never read;
  - a **drifted shape**: a file missing the common headings, so an agent that has read one
    section file cannot navigate the next one.
"""
import re

import pytest

from _pluginmeta import AGENTS, COMMANDS, REFERENCES, ROOT

SECTIONS_DIR = ROOT / "knowledge" / "references" / "task-generator"
SECTION_FILES = sorted(SECTIONS_DIR.glob("s[0-9]*.md"))

# Both consumers must route to the sections; a third would be added here deliberately.
CONSUMERS = ["agents/task-generator.md", "agents/reward-candidate-agent.md"]

REQUIRED_HEADINGS = [
    "## What",                      # what this section authors
    "## Analysis terms",
    "## Decisions to resolve",
    "## API surface",
    "## Smoke",
    "## Failure → diagnosis → fix",
    "## Known traps",
]

# Sections whose Analysis terms have not been written yet — now empty, and it MAY ONLY SHRINK.
# Every section file states the numbered questions its `### Analysis` block in the task-history
# scaffold enumerates; a new section file must bring its own before it can be added.
ANALYSIS_PENDING = set()

# s6-render is the whole-task render GATE, not a section: it authors nothing, so it has no
# decisions and no API of its own. Forcing it to carry empty headings would be padding, which
# is what the shared shape exists to prevent. It still shares the diagnosis + traps headings,
# which the shape test below checks for every file.
SHAPE_EXEMPT = {"s6-render.md"}
UNIVERSAL_HEADINGS = ["## Failure → diagnosis → fix", "## Known traps"]

_ALL_DOCS = COMMANDS + AGENTS + REFERENCES


def test_sections_dir_is_populated():
    assert SECTION_FILES, "knowledge/references/task-generator/ has no s<N>-*.md section files"
    assert (SECTIONS_DIR / "README.md").exists(), "task-generator/ needs its README (filing rule)"


@pytest.mark.parametrize("path", SECTION_FILES, ids=lambda p: p.name)
def test_no_orphan_section_files(path):
    """Every section file is routed to by at least one agent or command."""
    ref = f"knowledge/references/task-generator/{path.name}"
    referrers = [d.relative_to(ROOT).as_posix() for d in _ALL_DOCS
                 if ref in d.read_text(encoding="utf-8")]
    assert referrers, (
        f"{ref} is an orphan — no agent or command routes to it, so nothing it contains "
        f"is ever read. Add it to a consumer's section table or delete it."
    )


@pytest.mark.parametrize("path", SECTION_FILES, ids=lambda p: p.name)
def test_section_files_share_the_common_shape(path):
    text = path.read_text(encoding="utf-8")
    required = UNIVERSAL_HEADINGS if path.name in SHAPE_EXEMPT else REQUIRED_HEADINGS
    if path.name in ANALYSIS_PENDING:
        required = [h for h in required if h != "## Analysis terms"]
    missing = [h for h in required if h not in text]
    assert not missing, (
        f"{path.name} is missing heading(s) {missing}. The shared shape is what lets an "
        f"agent navigate a section file it has never read; see task-generator/README.md."
    )


@pytest.mark.parametrize("consumer", CONSUMERS)
def test_consumers_route_to_every_section(consumer):
    """A consumer that lists some sections but not others silently skips the rest."""
    text = (ROOT / consumer).read_text(encoding="utf-8")
    missing = [p.name for p in SECTION_FILES
               if f"knowledge/references/task-generator/{p.name}" not in text]
    assert not missing, f"{consumer} routes to no section file for: {missing}"


@pytest.mark.parametrize("path", SECTION_FILES, ids=lambda p: p.name)
def test_sections_point_at_their_smoke_template(path):
    """The template docstring owns the complete substitution list — one copy, no drift.

    A section file legitimately NAMES individual slots while explaining what makes a good
    value (`{{HOLD_ACTION_EXPR}}` must be reachable, and so on). What it must not become is
    the second home for the full enumerated list. Enforcing that positively — every section
    that describes a smoke points at that smoke's template — is checkable; a regex for
    "is this a restatement or a pointer" is not.
    """
    text = path.read_text(encoding="utf-8")
    if "## Smoke" not in text:
        pytest.skip("no smoke described in this file")
    assert re.search(r"knowledge/templates/task-generator/smokes/\S+\.template", text), (
        f"{path.name} describes a smoke but never points at its template, so a reader has "
        f"nowhere to get the substitution list; see task-generator/README.md filing rule."
    )


def test_analysis_pending_list_only_names_real_files():
    """A stale name in the ratchet quietly exempts nothing, and hides that the debt is paid."""
    names = {p.name for p in SECTION_FILES}
    assert ANALYSIS_PENDING <= names, f"ANALYSIS_PENDING names non-existent files: " \
                                      f"{sorted(ANALYSIS_PENDING - names)}"
    done = names - ANALYSIS_PENDING
    for name in sorted(done):
        text = (SECTIONS_DIR / name).read_text(encoding="utf-8")
        assert "## Analysis terms" in text, (
            f"{name} is not in ANALYSIS_PENDING but has no `## Analysis terms` heading — the "
            f"ratchet may only shrink, so this is a removed section, not a pending one."
        )


@pytest.mark.parametrize("path", SECTION_FILES, ids=lambda p: p.name)
def test_documented_checks_match_the_smoke_that_emits_them(path):
    """A smoke's numbered checks are its contract with the agent reading the section file.

    The smoke prints one `[Cn] PASS|FAIL` line per check so none can be skipped; the section
    file is where the agent learns what each one proves. A check that exists in only one of the
    two is either an undocumented failure the agent cannot diagnose, or a documented check that
    silently stopped running.
    """
    text = path.read_text(encoding="utf-8")
    for rel in set(re.findall(r"knowledge/templates/task-generator/smokes/\S+?\.template", text)):
        template = ROOT / rel
        if not template.exists():
            continue
        emitted = set(re.findall(r'check\(\s*"([A-Z]\d+)"',
                                 template.read_text(encoding="utf-8")))
        undocumented = sorted(c for c in emitted if not re.search(rf"\b{c}\b", text))
        assert not undocumented, (
            f"{path.name} does not document check(s) {undocumented} emitted by {rel} — an "
            f"agent seeing that FAIL line has nothing to look them up in."
        )
