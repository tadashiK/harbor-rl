---
description: List or inspect tasks within a Harbor benchmark. Use when the user types /harbor:task-list or asks "list tasks", "show task spec", "what tasks does this benchmark have", "show me task <id>". Reads the cwd-local `harbor/benchmark-generator/benchmark-spec.json` (populated by benchmark-generator).
argument-hint: "[list] | <task-id>"
---

# /harbor:task-list — Task Browser

The user invoked `/harbor:task-list` with optional sub-command args. **Parse the args and route to ONE action below.** Do not execute multiple actions in one invocation.

## Dispatch table

| User input | Route to |
|---|---|
| `/harbor:task-list` (no args) | **list local** |
| `/harbor:task-list list` | **list local** |
| `/harbor:task-list <task-id>` (arg contains `/`, e.g. `cartpole/swingup`) | **show one** |

Anything else: print the dispatch table above and stop.

---

## Action: list local

Tasks captured by `benchmark-generator` into the **current repo's** `harbor/benchmark-generator/benchmark-spec.json` — the rich path: each task carries `id`, `max_episode_steps`, `success_metric`, `reward_implemented`, `reward_metric`, plus benchmark-specific fields.

1. Locate the spec:
   ```bash
   ls "$(pwd)/harbor/benchmark-generator/benchmark-spec.json"
   ```
   If missing, print:
   > `harbor/benchmark-generator/benchmark-spec.json` not found in `<cwd>`. `cd` into a benchmark repo that has been processed by `benchmark-generator` and retry. If this repo has not been set up yet, run `/harbor:env-install-uv` and then the `benchmark-generator` subagent first.

2. Tabulate via inline Bash:
   ```bash
   python3 - <<'PY'
   import json, sys
   from pathlib import Path
   spec = json.loads(Path("harbor/benchmark-generator/benchmark-spec.json").read_text())
   tasks = spec.get("tasks") or []
   bench = spec.get("benchmark_name") or "(unknown)"
   if not tasks:
       print(f"benchmark '{bench}' has no enumerated tasks in benchmark-spec.json.")
       sys.exit(0)
   headers = ["id", "max_episode_steps", "success_metric", "reward_implemented", "reward_metric"]
   widths = [len(h) for h in headers]
   rows = [[str(t.get(h, "")) for h in headers] for t in tasks]
   for row in rows:
       for i, c in enumerate(row):
           widths[i] = max(widths[i], len(c))
   fmt = "  ".join(f"{{:<{w}}}" for w in widths)
   print(f"benchmark: {bench}  ({len(tasks)} tasks)")
   print(fmt.format(*headers))
   print(fmt.format(*("-"*w for w in widths)))
   for r in rows:
       print(fmt.format(*r))
   PY
   ```

3. Print the script's stdout verbatim. Do not paraphrase.

---

## Action: show one

Show the full task entry for `<task-id>` from the cwd-local benchmark-spec.

1. Verify `harbor/benchmark-generator/benchmark-spec.json` exists in cwd; if not, follow the same fallback as **list local**.

2. Inline:
   ```bash
   python3 - "<task-id>" <<'PY'
   import json, sys
   from pathlib import Path
   tid = sys.argv[1]
   spec = json.loads(Path("harbor/benchmark-generator/benchmark-spec.json").read_text())
   for t in spec.get("tasks") or []:
       if t.get("id") == tid:
           print(json.dumps(t, indent=2))
           sys.exit(0)
   print(f"task '{tid}' not found in benchmark '{spec.get('benchmark_name')}'. "
         f"Run /harbor:task-list to see the full list.", file=sys.stderr)
   sys.exit(2)
   PY
   ```

3. Print stdout verbatim. If exit code is non-zero, surface stderr and stop.

---

## Constraints

- The cwd-local `benchmark-spec.json` is the only source — there is no central index to fall back to.
- Do NOT modify `harbor/benchmark-generator/benchmark-spec.json` — this command is read-only.
