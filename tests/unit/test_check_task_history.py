"""Unit: scripts/task-generator/check_task_history.py — the gate task-generator passes before returning.

What matters here is that the checker cannot be satisfied by a plausible-looking file. The
failure it exists to catch is not a missing document — it is a complete-looking one whose
analysis is boilerplate and whose smoke verdicts were written from memory by the agent that
just failed them.
"""
import json
import subprocess
import sys

import pytest

from _pluginmeta import ROOT

SCRIPT = ROOT / "scripts" / "task-generator" / "check_task_history.py"
TEMPLATE = ROOT / "knowledge" / "templates" / "task-generator" / "task-history.md.template"

ANSWER = "a real answer long enough to clear the tripwire"
PENDING_TOKEN = "_pending_"


def _section(n, title, terms, rows):
    body = [f"## §{n} — {title}", "", "### Analysis", ""]
    for i, (label, answer) in enumerate(terms, 1):
        body += [f"{i}. **{label}**", f"   {answer}", ""]
    body += ["### Validations", "", "| Smoke | Attempts | Verdict | Evidence |",
             "|---|---|---|---|"]
    body += rows
    return "\n".join(body) + "\n\n"


def _history(**over):
    """A minimal but VALID history covering §1 only."""
    head = ("# Task history — `T`\n\n"
            "| Field | Value |\n|---|---|\n| task_id | `T` |\n"
            "| finished_at | 2026-01-01 |\n| status | pass |\n\n")
    s1 = _section(1, "Registration + scene",
                  over.get("terms", [("Task interpretation", ANSWER),
                                     ("Desired robot behavior", ANSWER),
                                     ("Potential failures", ANSWER),
                                     ("User Q&A", "_none_")]),
                  over.get("rows", ["| S1 | 1 | pass | stdout tail |"]))
    checks = over.get("checks", "| Check | Verdict | Evidence |\n|---|---|---|\n"
                                "| C6 collision geometry | PASS | 2 assets |\n\n")
    return head + s1 + checks


def _run(tmp_path, text, *extra, verdicts=None):
    hist = tmp_path / "task-history.md"
    hist.write_text(text, encoding="utf-8")
    smokes = tmp_path / "smokes"
    smokes.mkdir(exist_ok=True)
    for name, data in (verdicts or {}).items():
        (smokes / f"{name}.verdict.json").write_text(json.dumps(data), encoding="utf-8")
    r = subprocess.run([sys.executable, str(SCRIPT), "--history", str(hist),
                        "--sections", "1", *extra], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    return json.loads(r.stdout)


def test_a_complete_history_passes(tmp_path):
    out = _run(tmp_path, _history())
    assert out["ok"] and out["problems"] == []


def test_pending_term_is_caught(tmp_path):
    out = _run(tmp_path, _history(terms=[("Task interpretation", "_pending_")]))
    assert not out["ok"]
    assert any("_pending_" in p and "Task interpretation" in p for p in out["problems"])


def test_template_prose_quoting_pending_does_not_trip_the_gate(tmp_path):
    """Regression: the scaffold must not fail its own gate.

    The header is prose (the format rules) followed by the field table, and the rules have to
    quote `_pending_` in order to name it. Scanning the whole header for the token therefore
    flagged every history rendered from the template, including finished ones -- two ports
    hand-reworded the bullets to get past it. Only table ROWS can hold an unfilled field.

    The prose is read from the real template so this cannot silently pass by drifting away
    from what agents actually render.
    """
    header_prose = TEMPLATE.read_text(encoding="utf-8").split("\n| Field |", 1)[0]
    assert PENDING_TOKEN in header_prose, "template prose no longer quotes the token — retarget this test"
    text = _history()
    body = text.split("\n| Field |", 1)[1]
    out = _run(tmp_path, header_prose + "\n| Field |" + body)
    assert out["ok"], out["problems"]


def test_pending_in_a_header_table_row_is_still_caught(tmp_path):
    """The other half: relaxing the scan must not stop it catching an unfilled field."""
    out = _run(tmp_path, _history().replace("| status | pass |", "| status | _pending_ |"))
    assert not out["ok"]
    assert any("header table" in p for p in out["problems"])


def test_token_answer_is_caught(tmp_path):
    """`TODO` clears "is it non-empty" and answers nothing."""
    out = _run(tmp_path, _history(terms=[("Potential failures", "TODO")]))
    assert not out["ok"]
    assert any("too short" in p for p in out["problems"])


def test_explicit_none_is_accepted(tmp_path):
    out = _run(tmp_path, _history(terms=[("User Q&A", "_none_")]))
    assert out["ok"], out["problems"]


def test_empty_table_cell_is_caught(tmp_path):
    out = _run(tmp_path, _history(rows=["| S1 | 1 |  | |"]))
    assert not out["ok"]
    assert any("empty cell" in p for p in out["problems"])


def test_a_table_with_no_data_rows_is_fine(tmp_path):
    out = _run(tmp_path, _history() + "| Prim path | Why accepted |\n|---|---|\n")
    assert out["ok"], out["problems"]


def test_unrequested_section_must_say_so(tmp_path):
    text = _history() + _section(2, "Action terms", [("Decisions resolved", ANSWER)],
                                 ["| S2 | 1 | pass | ok |"])
    out = _run(tmp_path, text)                       # --sections 1, so §2 is unrequested
    assert not out["ok"]
    assert any("not requested" in p for p in out["problems"])

    text = _history() + "## §2 — Action terms\n\n### Analysis\n\n_not requested_\n"
    assert _run(tmp_path, text)["ok"]


def test_requested_section_with_no_block_is_caught(tmp_path):
    out = _run(tmp_path, _history(), "--sections", "1,4")
    assert not out["ok"]
    assert any("§4 was requested" in p for p in out["problems"])


# ---------------------------------------------------------------- the point of the whole file

def test_history_that_contradicts_the_smoke_is_caught(tmp_path):
    """The agent wrote PASS; smoke_s1.verdict.json says C6 FAILED. This is the case that a
    markdown-only artifact cannot catch, and the reason smokes record their own verdicts."""
    out = _run(tmp_path, _history(),
               verdicts={"smoke_s1": {"smoke": "S1", "status": "fail",
                                      "checks": {"C6": {"title": "collision geometry",
                                                        "verdict": "FAIL",
                                                        "detail": "concave mesh hulled"}}}})
    assert not out["ok"]
    assert any("C6" in p and "FAIL" in p for p in out["problems"])


def test_agreeing_verdicts_pass(tmp_path):
    out = _run(tmp_path, _history(),
               verdicts={"smoke_s1": {"smoke": "S1", "status": "pass",
                                      "checks": {"C6": {"title": "collision geometry",
                                                        "verdict": "PASS", "detail": "ok"}}}})
    assert out["ok"], out["problems"]


def test_check_the_smoke_ran_but_the_history_never_recorded(tmp_path):
    out = _run(tmp_path, _history(checks=""),
               verdicts={"smoke_s1": {"smoke": "S1", "status": "pass",
                                      "checks": {"C3": {"title": "obs", "verdict": "PASS",
                                                        "detail": "ok"}}}})
    assert not out["ok"]
    assert any("no row for it" in p for p in out["problems"])


def test_smoke_killed_mid_flight_is_not_a_pass(tmp_path):
    """Kit can take the process down before the smoke finishes; a file still reading 'running'
    means the result is unknown, which must not read as green."""
    out = _run(tmp_path, _history(),
               verdicts={"smoke_s1": {"smoke": "S1", "status": "running", "checks": {}}})
    assert not out["ok"]
    assert any("did not finish" in p for p in out["problems"])


# ---------------------------------------------------------------- deleted-term + real scaffold

def test_deleting_a_term_is_caught_with_the_template(tmp_path):
    """Without --template, dropping a term you cannot answer is invisible."""
    text = _history(terms=[("Task interpretation", ANSWER)])
    assert _run(tmp_path, text)["ok"]

    tmpl = tmp_path / "scaffold.md"
    tmpl.write_text(_history(terms=[("Task interpretation", "_pending_"),
                                    ("Potential failures", "_pending_")]), encoding="utf-8")
    out = _run(tmp_path, text, "--template", str(tmpl))
    assert not out["ok"]
    assert any("deleted rather than answered" in p and "Potential failures" in p
               for p in out["problems"])


def test_the_shipped_scaffold_fails_until_it_is_filled_in(tmp_path):
    """The scaffold as rendered must NOT pass — every slot is still a placeholder."""
    out = _run(tmp_path, TEMPLATE.read_text(encoding="utf-8"), "--sections", "1,2,3,4,5,6")
    assert not out["ok"]
    assert any("_pending_" in p for p in out["problems"])


# ---------------------------------------------------------------- the ran-checks checklist

def test_checklist_lists_the_checks_that_actually_ran(tmp_path):
    """Which checks exist is task-specific — §4 emits one C<i>/V<i> pair per implemented
    predicate — so the checklist must come from the verdict files, never from the templates."""
    out = _run(tmp_path, _history(), "--checklist-out", str(tmp_path / "test-checklist.md"),
               verdicts={"smoke_s4": {"smoke": "S4", "status": "pass", "checks": {
                   "G1": {"title": "goal representation", "verdict": "PASS", "detail": "1/1"},
                   "C1": {"title": "predicate fires: mug on peg", "verdict": "PASS", "detail": "2/2"},
                   "V1": {"title": "keyframe: mug on peg", "verdict": "PASS", "detail": "01_mug.png"},
               }}})
    assert out["checklist"]["passed"] == 3 and out["checklist"]["failed"] == 0

    body = (tmp_path / "test-checklist.md").read_text()
    for expected in ("C1 predicate fires: mug on peg", "V1 keyframe: mug on peg",
                     "G1 goal representation", "3 check(s)"):
        assert expected in body, f"{expected!r} missing from the checklist"


def test_checklist_reports_failures_rather_than_omitting_them(tmp_path):
    out = _run(tmp_path, _history(), "--checklist-out", str(tmp_path / "cl.md"),
               verdicts={"smoke_s1": {"smoke": "S1", "status": "fail", "checks": {
                   "C6": {"title": "collision geometry", "verdict": "FAIL",
                          "detail": "concave mesh hulled"},
               }}})
    assert out["checklist"] == {"path": str(tmp_path / "cl.md"), "passed": 0, "failed": 1}
    assert "FAIL" in (tmp_path / "cl.md").read_text()


def test_checklist_is_not_written_unless_asked(tmp_path):
    out = _run(tmp_path, _history())
    assert "checklist" not in out


def test_missing_history_is_a_caller_error_not_a_finding(tmp_path):
    r = subprocess.run([sys.executable, str(SCRIPT), "--history", str(tmp_path / "nope.md")],
                       capture_output=True, text=True)
    assert r.returncode != 0
    assert "no history file" in json.loads(r.stdout)["error"]


@pytest.mark.parametrize("flag", ["-h", "--help"])
def test_help_works(flag):
    r = subprocess.run([sys.executable, str(SCRIPT), flag], capture_output=True, text=True)
    assert r.returncode == 0 and "task-history" in r.stdout


def test_single_assertion_smokes_are_counted_not_just_listed(tmp_path):
    """A smoke with no enumerated checks is still one result. Listing it as a row while omitting
    it from the totals made the header count disagree with the table it heads."""
    out = _run(tmp_path, _history(), "--checklist-out", str(tmp_path / "cl.md"),
               verdicts={"smoke_s5": {"smoke": "S5", "status": "pass", "checks": {},
                                      "summary": "7 terms, order + shapes match"},
                         "smoke_s2": {"smoke": "S2", "status": "fail", "checks": {}}})
    assert out["checklist"]["passed"] == 1 and out["checklist"]["failed"] == 1
    body = (tmp_path / "cl.md").read_text()
    assert "2 check(s)" in body
    assert body.count("\n| ") - 1 == 2, "row count must match the stated total"


def test_analysis_digest_keeps_rationale_and_drops_smoke_evidence(tmp_path):
    """§6's designer and every reward candidate need WHY the task is shaped as it is, not the
    per-check Validations — which are a third of the history and pure smoke evidence."""
    # A realistic Validations block: in the real run it is 24.7k of the 72.4k file. The
    # minimal fixture would make the digest LARGER than its source (its preamble outweighs a
    # 3-row table), which says nothing about the property under test.
    bulky = _history(rows=[f"| S1 | 1 | pass | {'stdout tail ' * 40} |" for _ in range(20)])
    out = _run(tmp_path, bulky, "--analysis-out", str(tmp_path / "task-analysis.md"))
    body = (tmp_path / "task-analysis.md").read_text()
    assert "### Analysis" in body and "Task interpretation" in body
    assert "### Validations" not in body, "smoke evidence must not ride along"
    assert "stdout tail" not in body, "the evidence itself must not survive"
    assert out["analysis"]["chars"] < out["analysis"]["of_history"] * 0.5


def test_digest_is_not_written_unless_asked(tmp_path):
    assert "analysis" not in _run(tmp_path, _history())
