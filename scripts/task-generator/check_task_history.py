#!/usr/bin/env python3
"""Gate `task-history.md` before task-generator returns.

The history is the only durable record of why a task is shaped the way it is, and the §1
analysis exists specifically so later sections have something to build on — a §4 termination
predicate and a §6 reward are both written against "what would defeat a policy here". A history
whose analysis is blank does not fail loudly; it just makes every later section guess.

Five failure modes, none of which any smoke can see:

  H1  a requested section left `_pending_`, or an unrequested one not marked `_not requested_`
  H2  an enumerated analysis term with no answer under it
  H3  an analysis term deleted rather than answered (needs --template)
  H4  a table row with an empty cell — a check nobody recorded
  H5  a verdict that DISAGREES with the smoke's own <smoke>.verdict.json

H5 is the one worth the file. The agent that just spent three attempts on S4 is the one least
suited to transcribing S4's result, so the smokes record their own verdicts and this diffs the
claim against the record.

Exit 0 means the check RAN — read `ok` for the answer. Non-zero means it could not run at all
(no history file), which is a caller error, not a finding.
"""
import argparse
import json
import re
import sys
from pathlib import Path

PENDING = "_pending_"
NOT_REQUESTED = "_not requested_"
NONE_ANSWER = "_none_"
MIN_ANSWER_CHARS = 25          # tripwire for "TODO" / "ok" / "-", not a quality bar

_COMMENT = re.compile(r"<!--.*?-->", re.S)
_TERM = re.compile(r"^(\d+)\.\s+\*\*(.+?)\*\*\s*$", re.M)
_SECTION = re.compile(r"^## +(?:§(\d+)\b[^\n]*|([^\n]+?))\s*$", re.M)


def split_sections(text):
    """{key: body} for each `## ` block; key is an int for `## §N`, else the heading text."""
    out, marks = {}, list(_SECTION.finditer(text))
    for i, m in enumerate(marks):
        end = marks[i + 1].start() if i + 1 < len(marks) else len(text)
        out[int(m.group(1)) if m.group(1) else m.group(2).strip()] = text[m.end():end]
    return out


def subsection(body, name):
    """Body of a `### <name>` block within a section, up to the next `### ` or the end."""
    m = re.search(rf"^### +{re.escape(name)}\s*$", body, re.M)
    if not m:
        return None
    rest = body[m.end():]
    nxt = re.search(r"^### ", rest, re.M)
    return rest[:nxt.start()] if nxt else rest


def terms(body):
    """{label: answer} for each `N. **Label**` and the lines beneath it, comments stripped."""
    out, marks = {}, list(_TERM.finditer(body))
    for i, m in enumerate(marks):
        end = marks[i + 1].start() if i + 1 < len(marks) else len(body)
        out[m.group(2)] = _COMMENT.sub("", body[m.end():end]).strip()
    return out


def check_terms(where, body, problems, expected=None):
    """H2 + H3 — every scaffolded term is still present and carries a real answer."""
    found = terms(body)
    for label in sorted(set(expected or ()) - set(found)):
        problems.append(f"{where}: term **{label}** was deleted rather than answered")
    for label, answer in found.items():
        at = f"{where} term **{label}**"
        if not answer:
            problems.append(f"{at}: no answer")
        elif PENDING in answer:
            problems.append(f"{at}: still {PENDING}")
        elif answer == NONE_ANSWER:
            continue                                   # explicit claim of nothing — allowed
        elif len(answer) < MIN_ANSWER_CHARS:
            problems.append(f"{at}: answer is {len(answer)} chars ({answer!r}) — too short to "
                            f"be one; use {NONE_ANSWER} to claim there is nothing")


def _cells(line):
    return [c.strip() for c in line.strip().strip("|").split("|")]


def _is_separator(line):
    cells = _cells(line)
    return bool(cells) and all(set(c) <= set("-: ") and c for c in cells)


def tables(text):
    """Yield (headers, rows) for every markdown table — a header row, its separator, its rows."""
    lines = [ln.strip() for ln in text.splitlines()]
    i = 0
    while i < len(lines):
        if (lines[i].startswith("|") and i + 1 < len(lines)
                and lines[i + 1].startswith("|") and _is_separator(lines[i + 1])):
            headers, i, rows = _cells(lines[i]), i + 2, []
            while i < len(lines) and lines[i].startswith("|"):
                rows.append(_cells(lines[i]))
                i += 1
            yield headers, rows
        else:
            i += 1


def check_tables(where, body, problems):
    """H4 — no data row with an empty cell. A table with no data rows is fine."""
    for _, rows in tables(body):
        for cells in rows:
            if any(not c for c in cells):
                problems.append(f"{where}: table row has an empty cell — | {' | '.join(cells)} |")


def load_verdicts(smokes_dir):
    out = {}
    for path in sorted(Path(smokes_dir).glob("*.verdict.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            data = {"smoke": path.stem, "status": "unreadable", "checks": {}, "error": str(exc)}
        out[data.get("smoke") or path.stem] = data
    return out


def row_verdict(text, label):
    """The Verdict cell of the row whose first cell's first token is `label`.

    Located by COLUMN NAME, not position — the per-check table and the per-smoke table carry
    different columns, and reading the wrong one silently compares a verdict to an attempt count.
    """
    for headers, rows in tables(text):
        col = next((i for i, h in enumerate(headers) if h.lower() == "verdict"), None)
        if col is None:
            continue
        for cells in rows:
            if cells and cells[0].split()[:1] == [label]:
                return cells[col] if col < len(cells) else ""
    return None


def check_verdicts(text, verdicts, problems):
    """H5 — the history's claim matches what the smoke recorded about itself."""
    for smoke_id, data in sorted(verdicts.items()):
        status = data.get("status")
        if status not in ("pass", "fail"):
            problems.append(f"{smoke_id}: verdict file reads {status!r} — the smoke did not "
                            f"finish, so its result is unknown")
        for cid, check in sorted(data.get("checks", {}).items()):
            claimed = row_verdict(text, cid)
            if claimed is None:
                problems.append(f"{smoke_id}/{cid}: the smoke reported {check['verdict']} but "
                                f"the history has no row for it")
            elif check["verdict"].lower() not in claimed.lower():
                problems.append(f"{smoke_id}/{cid}: history says {claimed!r}, the smoke "
                                f"recorded {check['verdict']!r}")
        claimed = row_verdict(text, smoke_id)
        if claimed is not None and status in ("pass", "fail") and status not in claimed.lower():
            problems.append(f"{smoke_id}: history says {claimed!r}, the smoke recorded {status!r}")


def render_checklist(verdicts, task_id):
    """The list of checks that ACTUALLY RAN, built from the smokes' own verdict files.

    Which checks exist is task-specific: §4 emits one C<i>/V<i> pair per predicate the design
    implemented, so a task with three terminations and two subgoals runs a different set from
    one with a single time-out. The templates describe what CAN be tested; only the verdict
    files know what WAS. Nothing here is authored — it is a rendering of machine truth.
    """
    rows, passed, failed = [], 0, 0
    for smoke_id, data in sorted(verdicts.items()):
        checks = data.get("checks", {})
        if not checks:
            # A smoke with no enumerated checks is still ONE result. Listing it as a row but
            # omitting it from the totals made the stated count disagree with the table.
            ok = data.get("status") == "pass"
            passed += ok
            failed += not ok
            rows.append((smoke_id, "—", "PASS" if ok else "FAIL",
                         data.get("summary") or data.get("error") or "single-assertion smoke"))
        for cid, check in sorted(checks.items(), key=lambda kv: (kv[0][0], len(kv[0]), kv[0])):
            rows.append((smoke_id, f"{cid} {check['title']}", check["verdict"], check["detail"]))
            ok = check["verdict"] == "PASS"
            passed += ok
            failed += not ok

    out = [f"# Test checklist — `{task_id}`", "",
           "Every check that actually ran, rendered from the smokes' own `*.verdict.json`. Which",
           "checks exist depends on the design — §4 emits one `C<i>`/`V<i>` pair per predicate it",
           "implemented — so this is the list for THIS task, not the list the templates offer.", "",
           f"**{passed + failed} check(s) across {len(verdicts)} smoke(s): "
           f"{passed} PASS, {failed} FAIL.**", "",
           "| # | Smoke | Check | Verdict | Detail |", "|---|---|---|---|---|"]
    for i, (smoke, check, verdict, detail) in enumerate(rows, 1):
        detail = " ".join(str(detail).split())
        if len(detail) > 300:
            detail = detail[:297] + "..."
        detail = detail.replace("|", "\\|")      # escaped OUTSIDE the f-string: a backslash
        # inside an f-string expression is a SyntaxError before Python 3.12, and this script
        # runs under the benchmark repo's venv, which is routinely older than the host.
        out.append(f"| {i} | {smoke} | {check} | {verdict} | {detail} |")
    return "\n".join(out) + "\n", passed, failed


def render_analysis(text, task_id):
    """The design-rationale half of the history, for the agents that come after.

    §6's designer and every reward candidate need to know WHY the task is shaped as it is —
    §1's failure modes, §4's subgoal decomposition and degenerate states, §5's observability.
    They do not need the Validations tables: per-check verdicts and stdout tails are evidence
    about the smokes, not about the design, and they are a third of the file. Reading the whole
    history to reach the analysis costs every downstream agent ~25k chars of irrelevance.
    """
    keep = [f"# Task analysis — `{task_id}`", "",
            "The design rationale behind this task, extracted from task-history.md at Phase C.",
            "Validations (per-check smoke evidence) are deliberately excluded — see",
            "test-checklist.md for what was verified, and task-history.md for the full record.",
            ""]
    header = text.split("\n## ", 1)[0]
    for line in header.splitlines():
        if line.startswith("|"):
            keep.append(line)
    for key, body in split_sections(text).items():
        if key == "Adaptation delta":
            keep += ["", "## Adaptation delta", body.rstrip()]
        elif isinstance(key, int) or str(key).startswith("S6"):
            sub = subsection(body, "Analysis")
            if sub:
                head = f"## §{key}" if isinstance(key, int) else f"## {key}"
                keep += ["", head, "", "### Analysis", sub.rstrip()]
    return "\n".join(keep).rstrip() + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--history", required=True, help="path to task-history.md")
    ap.add_argument("--smokes-dir", help="dir holding <smoke>.verdict.json "
                                         "(default: <history dir>/smokes)")
    ap.add_argument("--template", help="the scaffold it was rendered from; enables the "
                                       "deleted-term check")
    ap.add_argument("--sections", default="1,2,3,4,5,6",
                    help="comma-separated sections that were requested (default: all)")
    ap.add_argument("--checklist-out", help="write the ran-checks checklist here, e.g. "
                                            "<task_dir>/test-checklist.md")
    ap.add_argument("--analysis-out", help="write the design-rationale digest here, e.g. "
                                           "<task_dir>/task-analysis.md — what §6's designer "
                                           "and every reward candidate read instead of the "
                                           "full history")
    a = ap.parse_args()

    history = Path(a.history)
    if not history.is_file():
        print(json.dumps({"ok": False, "error": f"no history file at {history}"}))
        return 2

    text = history.read_text(encoding="utf-8")
    requested = {int(s) for s in a.sections.split(",") if s.strip()}
    smokes_dir = Path(a.smokes_dir) if a.smokes_dir else history.parent / "smokes"
    problems = []

    scaffold = {}
    if a.template:
        tmpl = Path(a.template)
        if not tmpl.is_file():
            print(json.dumps({"ok": False, "error": f"no template at {tmpl}"}))
            return 2
        scaffold = split_sections(tmpl.read_text(encoding="utf-8"))

    def expected_terms(key, sub_name=None):
        body = scaffold.get(key)
        if body is None:
            return None
        if sub_name:
            body = subsection(body, sub_name) or ""
        return set(terms(body))

    sections = split_sections(text)
    for n in sorted(requested - {k for k in sections if isinstance(k, int)}):
        problems.append(f"§{n} was requested but has no block in the history")

    header = text.split("\n## ", 1)[0]
    # Only the header's TABLE rows can hold an unfilled field. The prose above the table
    # documents the format rules and necessarily quotes `_pending_` to name it, so scanning
    # the whole header makes the scaffold trip its own gate.
    if any(PENDING in ln for ln in header.splitlines() if ln.lstrip().startswith("|")):
        problems.append("header table still has a _pending_ field (finished_at / status)")
    check_tables("header", header, problems)

    for key, body in sections.items():
        if isinstance(key, int):
            if key not in requested:
                if NOT_REQUESTED not in body:
                    problems.append(f"§{key} was not requested but is not marked {NOT_REQUESTED}")
                continue
            for name in ("Analysis", "Validations"):
                sub = subsection(body, name)
                if sub is None:
                    problems.append(f"§{key}: `### {name}` subsection is missing")
                elif name == "Analysis":
                    check_terms(f"§{key} Analysis", sub, problems, expected_terms(key, name))
                else:
                    check_tables(f"§{key} Validations", sub, problems)
        else:
            check_terms(key, body, problems, expected_terms(key))
            check_tables(key, body, problems)

    verdicts = load_verdicts(smokes_dir)
    check_verdicts(text, verdicts, problems)

    result = {"ok": not problems, "history": str(history),
              "sections": sorted(requested), "problems": problems}

    if a.checklist_out:
        m = re.search(r"^#\s*Task history\s*—\s*`?([^`\n]+)`?", text, re.M)
        body, passed, failed = render_checklist(verdicts, (m.group(1).strip() if m else "task"))
        out = Path(a.checklist_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(body, encoding="utf-8")
        result["checklist"] = {"path": str(out), "passed": passed, "failed": failed}

    if a.analysis_out:
        m = re.search(r"^#\s*Task history\s*—\s*`?([^`\n]+)`?", text, re.M)
        out = Path(a.analysis_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        body = render_analysis(text, m.group(1).strip() if m else "task")
        out.write_text(body, encoding="utf-8")
        result["analysis"] = {"path": str(out), "chars": len(body),
                              "of_history": len(text)}

    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
