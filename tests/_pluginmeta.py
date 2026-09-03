"""Shared helpers + constants for the harbor plugin test suite.

No heavy deps: the contract layer parses frontmatter with a tolerant line parser
(matching how Claude Code reads it) rather than strict YAML, so a `:` inside a
description never causes a false failure.
"""
import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parents[1]  # plugin root (parent of tests/)

COMMANDS = sorted((ROOT / "commands").glob("*.md"))
AGENTS = sorted((ROOT / "agents").glob("*.md"))
REFERENCES = sorted((ROOT / "knowledge" / "references").rglob("*.md"))

# files scanned by the reference + hygiene checks
_TEXT_GLOBS = [
    "commands/*.md", "agents/*.md", "knowledge/references/**/*.md",
    "knowledge/templates/**/*.template", "knowledge/templates/**/*.yaml", "knowledge/templates/**/*.py",
    "scripts/**/*.py", "scripts/**/*.sh", "*.md",
]

KNOWN_TOOLS = {
    "Read", "Write", "Edit", "Bash", "Glob", "Grep",
    "AskUserQuestion", "WebFetch", "WebSearch", "NotebookEdit",
    "Agent", "TaskStop",
}
KNOWN_MODELS = {"opus", "sonnet", "haiku", "fable", "inherit"}
BUILTIN_AGENTS = {"general-purpose", "claude", "Explore", "Plan", "statusline-setup"}


def load_frontmatter(path):
    """Return (frontmatter dict | None, body str). Tolerant line parser."""
    text = path.read_text(encoding="utf-8")
    m = re.match(r"^---\n(.*?)\n---\n?", text, re.DOTALL)
    if not m:
        return None, text
    lines = m.group(1).split("\n")
    fm, i = {}, 0
    while i < len(lines):
        km = re.match(r"^([A-Za-z][\w-]*):\s?(.*)$", lines[i])
        if not km:
            i += 1
            continue
        key, val = km.group(1), km.group(2)
        # block scalar, incl. chomping/indent indicators (`|`, `>`, `>-`, `|+`, ...)
        if re.fullmatch(r"[|>][-+]?\d?", val.strip()):
            i += 1
            block = []
            while i < len(lines) and (lines[i].startswith("  ") or not lines[i].strip()):
                block.append(lines[i][2:] if lines[i].startswith("  ") else lines[i])
                i += 1
            fm[key] = "\n".join(block).strip()
            continue
        fm[key] = val.strip()
        i += 1
    tools = fm.get("tools")
    if isinstance(tools, str) and tools.startswith("[") and tools.endswith("]"):
        fm["tools"] = [t.strip() for t in tools[1:-1].split(",") if t.strip()]
    return fm, text[m.end():]


def iter_text_files():
    seen = set()
    for g in _TEXT_GLOBS:
        for p in ROOT.glob(g):
            if p.is_file() and p not in seen and ".git" not in p.parts:
                seen.add(p)
                yield p
