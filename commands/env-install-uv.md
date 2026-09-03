---
description: Generate an isolated Python environment for a simulation repo using uv on the host (creates a `.venv/`). Use when the user types /harbor:env-install-uv [path], or asks "set up env for X", "make a venv".
argument-hint: "[path]"
---

# /harbor:env-install-uv — Environment Generator (uv only)

The user has invoked `/harbor:env-install-uv` with an optional `[path]` argument. Resolve it and dispatch the dependency-generator agent.

## Dispatch table

| User input | Route to |
|---|---|
| `/harbor:env-install-uv` | dispatch dependency-generator agent with `repo_path=$(pwd)` |
| `/harbor:env-install-uv <path>` | dispatch with `repo_path=<absolute_path>` |

---

## Action: dispatch

1. Resolve `repo_path`:
   - If an argument is present, treat it as a path (resolve to absolute via `realpath` / `Path.resolve()`).
   - Otherwise use `$(pwd)`.
   - If the resolved path is missing or not a directory, print the error and stop.

2. Verify the repo:
   ```bash
   ls "<repo_path>/.git" >/dev/null 2>&1 && echo "is git repo" || echo "warning: not a git repo"
   ls "<repo_path>"/{pyproject.toml,setup.py,requirements.txt,uv.lock} 2>/dev/null
   ```
   If none of these exist, ask the user to confirm `<repo_path>` is the right repo (Python project missing).

3. Dispatch the dependency-generator agent with the resolved inputs:
   ```
   Skill('dependency-generator')
     repo_path = <absolute>
     force?    = false
   ```

4. **Print the agent's structured JSON output verbatim** (its `smoke`, `next_action`, etc.). If the build + smoke passed, dispatch `benchmark-generator` next (the env is the entry point of the chain); if they failed, stop and surface the errors.

## Constraints

- **Pass the resolved absolute path** to the agent — never relative.
- **Dispatch `benchmark-generator` only after a clean build + smoke.** If the env didn't come up, stop and report instead.
- The agent produces these artifacts in `<repo>/`:
  - `<repo>/.venv/` — the actual venv (created by `uv venv` + `uv pip install`).
  - `<repo>/harbor/dependency-generator/{setup_uv.sh, install_plan.json, install.md, probe.json}` — re-runnable setup script + the install receipt + metadata.
