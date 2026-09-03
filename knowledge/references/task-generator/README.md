# task-generator references

Two kinds of file live here:

| File | Role |
|---|---|
| `s1-scene.md` … `s6-render.md` | one per task section — everything an agent needs to work on that section and nothing about any other |
| `isaaclab-code-reference.md` | the API surface every section file points into |

Despite the folder name, the section files are **cross-cutting**: two consumers read them, and
both load only the files for the sections they touch.

| Consumer | Loads |
|---|---|
| `task-generator` | the section files for the `sections` it was asked to author (all five in create mode) |
| `reward-candidate-agent` | the section files named by `design.task_changes.sections` — usually none, since most candidates are reward-only |

The files are written **consumer-neutral**: they describe the section, its smoke, and how it
fails. Authoring a section from scratch and applying a bounded delta to it need the same
facts; only the scope differs, and each agent's own body says which it is doing.

## Shape

Every file has the same headings, so navigating a section you have not read before costs
nothing:

1. **What §N authors** — the cfg blocks that belong to this section
2. **Analysis terms** — the numbered questions the agent must answer *before* authoring, and
   which it records under that section's **Analysis** in `task-history.md`. Understanding, not
   choices — "what would defeat a policy here" rather than "which asset"
3. **Decisions to resolve** — what must be chosen, and in what order of precedence
4. **API surface** — pointers into `knowledge/references/task-generator/isaaclab-code-reference.md`
5. **Smoke S\<N\>** — the numbered checks it runs, what each proves, what the agent must fill
6. **Failure → diagnosis → fix** — symptom, likely cause, first-pass fix
7. **Known traps** — the mistakes that have actually been made

Analysis terms and smoke checks are both **enumerated and non-skippable**: each carries its own
number, each gets its own answer or its own `PASS|FAIL` line, and neither collapses into a
summary. `tests/contract/test_task_sections.py` holds the list of sections that still lack
their Analysis terms — it may only shrink.

## Filing rule

Where a new fact goes, so this structure survives being grown:

| Kind of detail | Home |
|---|---|
| API call / signature / idiom | `knowledge/references/task-generator/isaaclab-code-reference.md`, by concern |
| A smoke's substitution slot spec | that smoke template's **docstring** — the agent must open it to render it, so it is free there and duplicating it here would drift |
| What to decide for §N · what its smoke proves · failure → fix | the section file here |
| A heuristic learned from an actual run | `knowledge/experiences/task-generator/task-experience.md` (numbered, append-only) |

§6 (reward) and §7 (DR) are not here: they are owned by `reward-tuning-agent` and
`dr-generator`, and their contracts live under those agents' own reference directories.
