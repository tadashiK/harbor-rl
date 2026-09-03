"""Contract: the task-history scaffold covers every section, with both subsections.

`task-history.md` is scaffolded at agent entry rather than appended to as work happens, so that
a slot left empty is *visible*. That only holds if the scaffold actually has a slot for every
section the agent can be asked to author: a §N missing from the template is a §N whose analysis
and smoke results quietly go unrecorded, and no smoke fails when they do.

The section set is derived from the section reference files (`s1-scene.md` … `s6-render.md`)
rather than hardcoded, so adding a §7 to that directory fails here until the scaffold gains a
block for it.
"""
import re

import pytest

from _pluginmeta import ROOT

TEMPLATE = ROOT / "knowledge" / "templates" / "task-generator" / "task-history.md.template"
SECTIONS_DIR = ROOT / "knowledge" / "references" / "task-generator"
# s6-render is the whole-task GATE, not a numbered section — its scaffold block is `## S6`,
# deliberately not `## §6`, because task-create's `sections` uses 6 for the REWARD.
GATE_FILES = {"s6-render.md"}
SECTION_NUMBERS = sorted(int(m.group(1)) for p in SECTIONS_DIR.glob("s[0-9]*.md")
                         if p.name not in GATE_FILES and (m := re.match(r"s(\d+)-", p.name)))

REQUIRED_BLOCKS = ["## Adaptation delta", "## S6", "## Doc patches", "## Final verdict"]


def _blocks(text):
    """{section number: body} for each `## §N — ...` heading, up to the next `## `."""
    out = {}
    parts = re.split(r"^## ", text, flags=re.M)[1:]
    for part in parts:
        m = re.match(r"§(\d+)\b", part)
        if m:
            out[int(m.group(1))] = part
    return out


def test_template_exists_and_is_routed_to():
    assert TEMPLATE.exists(), f"{TEMPLATE.relative_to(ROOT)} is missing"
    agent = (ROOT / "agents" / "task-generator.md").read_text(encoding="utf-8")
    assert "knowledge/templates/task-generator/task-history.md.template" in agent, (
        "task-generator does not route to the scaffold, so it would hand-roll the log's "
        "structure again and the per-section slots stop being guaranteed"
    )


def test_every_section_has_a_block():
    assert SECTION_NUMBERS, "no section reference files found to derive the section set from"
    present = set(_blocks(TEMPLATE.read_text(encoding="utf-8")))
    missing = [n for n in SECTION_NUMBERS if n not in present]
    assert not missing, (
        f"task-history scaffold has no block for §{missing} — work on those sections would "
        f"leave no trace, since the agent is told never to append new section headings."
    )


@pytest.mark.parametrize("number", SECTION_NUMBERS)
def test_each_block_has_analysis_and_validations(number):
    body = _blocks(TEMPLATE.read_text(encoding="utf-8"))[number]
    for sub in ("### Analysis", "### Validations"):
        assert sub in body, f"§{number} block is missing its `{sub}` subsection"
    # Validations is where smoke results land; a table header is what makes "which smokes ran"
    # answerable at a glance instead of buried in prose.
    validations = body.split("### Validations", 1)[1]
    assert "| Smoke |" in validations, (
        f"§{number}'s Validations has no smoke table header, so per-smoke verdicts have "
        f"nowhere consistent to go"
    )


def test_scaffold_carries_the_run_level_blocks():
    text = TEMPLATE.read_text(encoding="utf-8")
    missing = [b for b in REQUIRED_BLOCKS if b not in text]
    assert not missing, f"task-history scaffold is missing run-level block(s): {missing}"
