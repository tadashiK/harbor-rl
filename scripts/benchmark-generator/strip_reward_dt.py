#!/usr/bin/env python3
"""
Strip ALL dt scaling from IsaacLab's ``RewardManager.compute()`` — and HARD-GATE on it.

Stock IsaacLab's ``RewardManager.compute(dt)`` scales every reward term by the control ``dt``
(``value = func(...) * weight * dt``) and stores a ``1/dt``-scaled copy in the per-term log
buffer (``_step_reward = value / dt``). Harbor's convention is NOMINAL weights + logs that show
the real per-step reward, so both dt factors are stripped ONCE here, at the source.

Robustness (why this is more than a string match): IsaacLab may restructure ``compute()`` across
versions (rename the ``dt`` param, reorder the multiply, split the line, move the file). A safety
gate must therefore **fail closed** — the guarantee is a structure-agnostic POSTCONDITION, not a
literal pattern:

    after this runs, ``compute()`` must contain NO ``[*/]`` scaling by a dt-like token
    (``dt`` / ``step_dt`` / ``self.*dt``) anywhere in its code.

Flow: (1) best-effort strip of the recognized patterns, then (2) re-scan ``compute()`` and assert
the postcondition. If ANYTHING dt-scaling survives — because the stripper didn't recognize a new
structure — the file is left untouched and the script EXITS 1 (``DT-SCALE SMOKE FAIL``), telling
the human to fix ``compute()`` by hand. It never claims PASS on an unrecognized layout, and it
never silently writes a half-stripped file.

Fail-closed matrix:
  * isaaclab importable, compute() found, dt fully stripped  -> PASS (exit 0)
  * isaaclab importable, dt token still in compute()         -> FAIL (exit 1)
  * isaaclab importable, but reward_manager / compute() not found -> FAIL (exit 1, structure changed)
  * isaaclab NOT importable (not an IsaacLab repo)           -> no-op PASS (exit 0)

Usage:
    python strip_reward_dt.py --repo /abs/repo [--python /abs/repo/.venv/bin/python]
"""
import argparse
import re
import subprocess
import sys
from pathlib import Path

MARKER = "# harbor: dt-scaling stripped"

# A dt-like scaling token: the compute() ``dt`` param, a bare ``step_dt``/``_dt`` local, or any
# ``self.<...>dt`` attribute (self.step_dt, self._dt, self._sim_step_dt, ...).
DT_TOKEN = r"(?:dt|step_dt|_dt|self\.[A-Za-z_]*dt)"
# The dt SCALING signature we must not leave behind: a ``*`` or ``/`` (optionally ``*=``/``/=``)
# applied to a dt-like token.
DT_SCALE = re.compile(r"[*/]=?\s*" + DT_TOKEN + r"\b")


def strip_inline_comment(line: str) -> str:
    """Best-effort drop of a trailing ``# ...`` comment (RewardManager.compute has no string
    literals containing ``#`` on the scaling lines, so a simple split is safe here)."""
    return line.split("#", 1)[0]


def locate_reward_manager(repo: Path, python: str | None):
    """Return (path_or_None, isaaclab_present: bool).

    isaaclab_present distinguishes 'not an IsaacLab repo' (safe no-op) from 'IsaacLab is here but
    reward_manager moved' (structure changed -> fail closed)."""
    py = python or str(repo / ".venv" / "bin" / "python")
    present = False
    if Path(py).exists():
        try:
            present = "OK" in subprocess.run(
                [py, "-c", "import isaaclab; print('OK')"],
                capture_output=True, text=True, timeout=120,
            ).stdout
        except Exception:
            present = False
        try:
            out = subprocess.run(
                [py, "-c", "import isaaclab.managers.reward_manager as m; print(m.__file__)"],
                capture_output=True, text=True, timeout=120,
            )
            path = out.stdout.strip().splitlines()[-1] if out.stdout.strip() else ""
            if path and Path(path).is_file():
                return Path(path), True
        except Exception:
            pass
    candidate = repo / "source" / "isaaclab" / "isaaclab" / "managers" / "reward_manager.py"
    if candidate.is_file():
        return candidate, True
    return None, present


def find_compute_span(lines):
    """Return (start, end) line indices covering the body of ``def compute(``, or None."""
    for i, ln in enumerate(lines):
        if re.match(r"\s*def compute\(", ln):
            indent = len(ln) - len(ln.lstrip())
            j = i + 1
            while j < len(lines):
                s = lines[j]
                if s.strip() and (len(s) - len(s.lstrip())) <= indent and re.match(r"\s*(def|class)\b", s):
                    break
                j += 1
            return i, j
    return None


def dt_scaling_lines(lines, span):
    """Lines within compute() (comment-stripped) that still apply dt scaling — the postcondition."""
    i, j = span
    hits = []
    for k in range(i, j):
        code = strip_inline_comment(lines[k])
        if DT_SCALE.search(code):
            hits.append((k + 1, lines[k].rstrip()))
    return hits


def best_effort_strip(lines, span):
    """Remove the recognized dt factors in place. Returns True if anything changed."""
    i, j = span
    changed = False
    for k in range(i, j):
        line = lines[k]
        if MARKER in line:
            continue
        code = strip_inline_comment(line).rstrip("\n")
        # (a) per-term VALUE line: ``... term_cfg.func(...) * term_cfg.weight * dt`` -> drop ``* dt``
        if "term_cfg.func(" in code and "term_cfg.weight" in code:
            m = re.search(r"\s*\*\s*" + DT_TOKEN + r"\s*$", code)
            if m:
                lines[k] = f"{code[:m.start()].rstrip()}  {MARKER} (weights are nominal per-step)\n"
                changed = True
                continue
        # (b) per-term LOG buffer: ``_step_reward[...] = value / dt`` -> drop ``/ dt``
        if "_step_reward" in code and "=" in code:
            m = re.search(r"\s*/\s*" + DT_TOKEN + r"\s*$", code)
            if m:
                lines[k] = f"{code[:m.start()].rstrip()}  {MARKER} (real per-step reward)\n"
                changed = True
                continue
    return changed


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", required=True)
    ap.add_argument("--python", default=None)
    args = ap.parse_args()

    repo = Path(args.repo).resolve()
    rm, present = locate_reward_manager(repo, args.python)
    if rm is None:
        if present:
            print("strip_reward_dt: DT-SCALE SMOKE FAIL — isaaclab is importable but "
                  "reward_manager.py could not be located (structure changed). Verify by hand.",
                  file=sys.stderr)
            return 1
        print("strip_reward_dt: no isaaclab — not an IsaacLab repo, no-op.")
        return 0

    lines = rm.read_text().splitlines(keepends=True)
    span = find_compute_span(lines)
    if span is None:
        print(f"strip_reward_dt: DT-SCALE SMOKE FAIL — could not locate RewardManager.compute() "
              f"in {rm} (unrecognized structure). Verify by hand.", file=sys.stderr)
        return 1

    changed = best_effort_strip(lines, span)
    # Re-derive the span (line count is unchanged by our in-place edits) and check the POSTCONDITION.
    remaining = dt_scaling_lines(lines, span)
    if remaining:
        # Do NOT write a half-stripped file. Fail closed and show what we couldn't handle.
        print(f"strip_reward_dt: DT-SCALE SMOKE FAIL — dt scaling still present in {rm} "
              f"after best-effort strip (unrecognized structure). Fix compute() by hand:",
              file=sys.stderr)
        for lineno, text in remaining:
            print(f"    {rm}:{lineno}: {text}", file=sys.stderr)
        return 1

    if changed:
        rm.write_text("".join(lines))
        print(f"strip_reward_dt: stripped dt scaling from RewardManager.compute() — {rm}")
    else:
        print(f"strip_reward_dt: no dt scaling to strip — {rm}")
    print("strip_reward_dt: DT-SCALE SMOKE PASS — compute() has no dt scaling; declared weights "
          "are nominal and logs show the real per-step reward.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
