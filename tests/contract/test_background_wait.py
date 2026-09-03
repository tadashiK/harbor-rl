"""Contract: an agent that waits for a long job waits in the BACKGROUND.

A foreground `Bash` call caps at ~600 s. Every harbor job worth waiting on — a GPU training
run, a render, a SLURM submission — outlives that, so a foreground wait does not wait: it
times out and gets re-issued, five to seven times per job. Each re-issue is a fresh turn, and
the prompt cache has expired in the gap, so each one re-caches the agent's ENTIRE context at
full price.

This is not a micro-optimization. Measured on one 10-candidate reward tune:

    reward-tuning-agent      51 poll turns   13.58M cache-write tokens   94% of its total
    reward-candidate-agent   48 poll turns    6.16M cache-write tokens   62% of its total
                                             ------
                                             19.74M — ~45% of the whole run, spent asking
                                             "is it done yet"

`run_in_background=true` has no cap and re-invokes the agent when the job exits, so the wait
costs nothing. The sentinel FILE stays the source of truth either way, which is what keeps the
wait robust against a lost notification, a killed agent, or a resumed session.
"""
import re

from _pluginmeta import AGENTS, COMMANDS

# The loop spans lines — `until ... do` on one, `[ -f "$d" ]` on the next — so a per-line match
# misses it. Scanned as a small window rather than one multi-line regex: the regex version of
# this, `(?:[^\n]*\n?){0,4}?[^\n]*\bsleep`, is a nested quantifier that backtracks exponentially
# and hung the whole suite for 20 minutes.
LOOP_START = re.compile(r"\b(?:until|while)\b")
SLEEP = re.compile(r"\bsleep\s+(\d+)")
BACKGROUND = "run_in_background=true"
LOOKAHEAD = 4          # lines from `until` to its `sleep`
MIN_WAIT = 20          # a shorter sleep is a race-avoidance pause, not a wait


def _wait_sites(text):
    """Yield (line_no, context) for each foreground-wait shape: a loop with a sleep in it,
    or a bare long sleep."""
    lines = text.splitlines()
    for i, line in enumerate(lines):
        window = lines[i:i + LOOKAHEAD + 1]
        m = SLEEP.search(line)
        hit = (LOOP_START.search(line) and any(SLEEP.search(w) for w in window)) \
            or (m and int(m.group(1)) >= MIN_WAIT)
        if hit:
            yield i + 1, "\n".join(lines[max(0, i - 8):i + 9])


def test_wait_loops_are_backgrounded():
    """Every wait-for-file loop in an agent must be issued with run_in_background=true."""
    offenders = []
    for path in list(AGENTS) + list(COMMANDS):
        text = path.read_text(encoding="utf-8")
        for line_no, context in _wait_sites(text):
            if BACKGROUND not in context:
                offenders.append(f"{path.name}:{line_no}")
    assert not offenders, (
        "foreground wait loop — caps at 600 s, so it becomes a re-poll that re-caches the "
        f"whole context every turn. Issue it with {BACKGROUND}: " + ", ".join(offenders)
    )


def test_no_agent_prescribes_a_re_poll_loop():
    """The instruction itself must not tell an agent that re-polling is normal.

    `reward-tuning-agent` used to say "this is a RE-POLL loop ... expect tens of iterations
    over a long candidate — that is normal, not a hang", and it did exactly that.
    """
    offenders = [p.name for p in list(AGENTS) + list(COMMANDS)
                 if re.search(r"re-?poll", p.read_text(encoding="utf-8"), re.IGNORECASE)]
    assert not offenders, (
        "agent prescribes re-polling; background the wait instead: " + ", ".join(offenders)
    )
