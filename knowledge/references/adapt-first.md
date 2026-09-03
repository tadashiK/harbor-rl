# Adapt-first — build from the selected base

Followed by the authoring agents — `task-generator` (§1–§5), `reward-tuning-agent` (§6),
`dr-generator` (§7) — on the `design_base` that `task-library-search.md` already returned. Selection is
done; this is how to use it.

**Precedence when sources disagree:** (1) the user's description / explicit constraints, (2) the base
spec's design, (3) the destination repo's expression (in-repo canonical / `task-implementation.md` — for
*how* to write it, never *what* to design), (4) experience-ledger heuristics. A library spec flagged
"analytic / not build-verified" downgrades only trust in its *code* (re-verify it compiles / re-measure
dims), not its *design*.

## Step 1 — Read the matching experience ledger

Read the caller's own append-only ledger (heuristics distilled across runs):

| Caller | Ledger |
|---|---|
| `task-generator` | `knowledge/experiences/task-generator/task-experience.md` |
| `reward-tuning-agent` | `knowledge/experiences/reward-tuning-agent/reward-experience.md` |
| `dr-generator` | `knowledge/experiences/dr-generator/dr-experience.md` |

## Step 2 — Adapt-first (BINDING)

**Port everything; change only what the goal task overrides.** Adopt the base's whole design — scene,
action term, reset, reward structure, obs, gating — and swap only what the prompt explicitly differs on
(base grasps a dog, prompt says a cube → change only the object, keep the rest). The action term is part
of the design, not a repo idiom: keep the base's even if the destination repo has a "native equivalent".
Every gratuitous deviation from a proven base is an unforced risk.

- **Ledger heuristics do not override the base.** The base is proven; the ledgers are not. Don't
  rebalance proven weights / remove proven terms / restructure proven gating on a ledger hunch — let
  training on the CURRENT run falsify the base first, then change it citing that evidence.
- **Express in the destination repo's idioms.** When the base spec and the destination family disagree
  on *how* to express something (imports, cfg field names, action-term API), follow the destination;
  borrow the *design* from the base.
- **Idiom ≠ redesign; a missing signal is ADDED, not proxied.** "Destination idioms" above is syntax
  only. It is NOT license to change a reward term's mathematical FORM (an unbounded `1/d` attractor
  re-expressed as a bounded `tanh`; a contact-gated grasp re-expressed as a proximity gate) or to drop the
  SIGNAL a term reads. If the destination scene lacks a signal the base's function needs — a fingertip
  contact sensor, a command manager, a force/link sensor — **port it** (declare the added §1–§5 field;
  `reward-tuning-agent` names it in `design.json:env_changes` so its IMPLEMENT step wires it in) rather than
  re-derive the function around a weaker proxy. Re-expressing a proven reward's math is a gratuitous
  deviation — each is a `changed:` bullet with a goal-task justification, and "the scene didn't have the
  sensor" is never that justification; it is a trigger to add the sensor.
- **Byte-identical → reproduce.** If the base is byte-identical to what's wanted, prefer
  `/harbor:task-create from=<that spec>` (reproduce mode) over re-authoring.

**Pure creation mode** activates ONLY when search returned **none** (no spec shares the
verb / object-class / embodiment). "The match isn't perfect" is a reason to adapt with a larger delta,
not to go pure-create.

## Step 3 — Document the adaptation delta

Record in the history file (`task-history.md` / `reward-history.md` / the `## Iter <N>` section):

```markdown
### Adaptation delta
- base: <abs path of the selected spec>  (or "none — pure creation mode: no relevant task in <folder>")
- kept as-is: <what was reused unchanged — e.g. action term, composer, gating structure>
- changed: <one bullet per deviation from the base, each with WHY the goal task requires it>
```

The delta shows which knobs actually had to move between two sibling tasks — that's what makes the next
run's search useful.
