---
description: >-
  Reset a benchmark repo to its original cloned state by removing ALL harbor-plugin-generated files — the entire <repo>/harbor/ tree, the uv .venv/, the scripts/ carve-outs (run_random.py / render_random.py / _<family>_env.py), plugin caches — and (by default) reverting any task code the plugin wrote directly into the benchmark's own source tree. Runs in a subagent. DESTRUCTIVE + irreversible: shows a dry-run plan and asks to confirm first, then verifies via a git-based smoke (including hidden / ignored files) that the repo is byte-identical to its original clone before reporting success. Use when the user types /harbor:reset-workspace repo=<path> [clean_inbenchmark_tasks=true|false] or asks "reset/clean the workspace", "remove all harbor files from this repo", "restore the benchmark to its cloned state".
argument-hint: "repo=<path> [clean_inbenchmark_tasks=true|false]"
disable-model-invocation: true
---

# /harbor:reset-workspace — Restore a Benchmark Repo to Its Cloned State

Remove everything the harbor plugin generated in a benchmark repo and return it to its original `git clone` state. **DESTRUCTIVE and irreversible** — it deletes `.venv/` + the `harbor/` tree and (by default) `git reset --hard` + `git clean -fdx` the benchmark back to HEAD. Only run on a checkout you are willing to return to its clean state.

## Execution model — run in a subagent (REQUIRED)

Cleaning lists + removes many files and runs the verification smoke; that work bloats the main context and is never needed again. The main thread:
1. Resolves args (`repo`, `clean_inbenchmark_tasks`).
2. Dispatches ONE subagent (`Agent(subagent_type="general-purpose")`) whose prompt is this command body + the resolved args. The subagent does plan → confirm → clean → smoke and returns only the final one-line verdict.
3. Relays that verdict to the user.

## Required argument

| Arg | Notes |
|---|---|
| `repo` | Absolute path to the benchmark repo to reset. **No default** — you must name the target explicitly (this command is destructive). Must be a git working tree. |

## Optional argument

| Arg | Default | Effect |
|---|---|---|
| `clean_inbenchmark_tasks` | `true` | `true` → ALSO revert/remove task code the plugin authored directly inside the benchmark's own source tree (new `gym.register` entries, new `*_env_cfg.py` / `mdp/` files), i.e. a full reset to the original clone. `false` → KEEP those in-benchmark task modifications; remove only the `harbor/` tree, `.venv/`, the `scripts/` carve-outs, and plugin caches. |

The plugin's normal output all lives under `<repo>/harbor/`; the only things it writes OUTSIDE `harbor/` are `.venv/`, the `scripts/{run_random,render_random,_<family>_env}.py` smoke entry points, plugin caches, and — when you author tasks with `/harbor:task-create` — task code inside the benchmark's own source tree. `clean_inbenchmark_tasks` controls only that last category.

## Removal target

| Path | Tracked? | Removed when |
|---|---|---|
| `<repo>/harbor/` (whole tree) | untracked | always |
| `<repo>/.venv/` | gitignored | always |
| `<repo>/scripts/run_random.py`, `scripts/render_random.py`, `scripts/_*_env.py` | untracked | always |
| plugin caches (`__pycache__/`, `*.egg-info/`, `.pytest_cache/`, `.ruff_cache/`) | gitignored/untracked | always |
| new task registrations in existing `__init__.py` (modified) + new `*_env_cfg.py` / `mdp/` files inside the benchmark source | modified/untracked | only when `clean_inbenchmark_tasks=true` |

## Procedure (subagent)

### Step 0 — Pre-flight

```bash
test -d "<repo>" || { echo "repo path does not exist: <repo>"; exit 1; }
git -C "<repo>" rev-parse --is-inside-work-tree >/dev/null 2>&1 \
  || { echo "<repo> is not a git working tree — cannot define 'original cloned state'; aborting"; exit 1; }
```

Not a git repo ⇒ abort: without git there is no well-defined "original clone" to verify against.

### Step 1 — Build the removal plan (DRY-RUN — nothing deleted yet)

```bash
cd "<repo>" || exit 1
echo "== modified/deleted tracked files (would be reverted in default mode) =="
git status --porcelain
echo "== untracked + ignored files that would be removed (incl. hidden: .venv/, __pycache__/) =="
git clean -ndx          # -n dry-run · -d dirs · -x ignored
```

Summarize the counts + notable paths. If `clean_inbenchmark_tasks=false`, call out which in-source task paths will be **PRESERVED**.

### Step 2 — Confirm (mandatory destructive gate)

`AskUserQuestion` with the plan summary: **Proceed / Abort**. Default mode wipes `.venv/` and discards every local change + untracked/ignored file in the repo — irreversible. Abort on anything but explicit proceed.

### Step 3 — Execute

Always remove the plugin's out-of-`harbor/` artifacts + caches. **`cd` must be guarded** —
every path below is repo-relative, so a `cd` that silently fails (typo'd `repo=`, a directory
removed between the plan and the execute step) points the whole block at whatever directory the
shell happened to be in, and the next line is a recursive delete:
```bash
cd "<repo>" || exit 1
rm -rf harbor/ .venv/
rm -f scripts/run_random.py scripts/render_random.py scripts/_*_env.py
find . -path ./.git -prune -o \( -name '__pycache__' -o -name '*.egg-info' -o -name '.pytest_cache' -o -name '.ruff_cache' \) -exec rm -rf {} +
```

Run this block as ONE Bash invocation. Splitting the `cd` into a call of its own puts the
guard in a different shell from the thing it guards.

Then, for the in-source task code:
- **`clean_inbenchmark_tasks=true` (default) — full reset to the cloned HEAD:**
  ```bash
  git reset --hard HEAD     # revert plugin edits to tracked files (task registrations, etc.)
  git clean -fdx            # remove ALL remaining untracked + ignored (new task files, leftover caches)
  ```
  After this the working tree equals the original clone.
- **`clean_inbenchmark_tasks=false` — keep in-source task code:** do NOT run `git reset --hard` / `git clean -fdx`; leave the benchmark's source task files + registrations in place.

### Step 4 — Smoke check (ALL applicable checks must pass)

Run every check and report each. Finish with success ONLY if all pass.

**Always (artifact removal):**
```bash
cd "<repo>" || exit 1   # unguarded, every `test ! -e` below passes in the wrong directory
test ! -e harbor  && echo "OK harbor/ gone"  || echo "FAIL harbor/ remains"
test ! -e .venv   && echo "OK .venv/ gone"   || echo "FAIL .venv/ remains"
ls scripts/run_random.py scripts/render_random.py scripts/_*_env.py 2>/dev/null \
  && echo "FAIL scripts carve-outs remain" || echo "OK scripts carve-outs gone"
find . -path ./.git -prune -o \( -name '__pycache__' -o -name '*.egg-info' -o -name '.pytest_cache' -o -name '.ruff_cache' \) -print | grep -q . \
  && echo "FAIL plugin caches remain" || echo "OK no plugin caches"
```

**Default mode only — identical-to-clone (the strong check):**
```bash
git -C "<repo>" diff --quiet HEAD              && echo "OK working tree == HEAD"  || echo "FAIL tree differs from HEAD"
[ -z "$(git -C "<repo>" status --porcelain)" ] && echo "OK no modified/untracked" || echo "FAIL dirty working tree"
[ -z "$(git -C "<repo>" clean -ndx)" ]         && echo "OK nothing left to clean (incl. hidden/ignored)" || echo "FAIL untracked/ignored remain"
```

The default-mode trio IS the definition of "identical to original cloned state": working tree matches HEAD, no modified/untracked files, and `git clean -ndx` (which sees hidden + ignored dirs like `.venv/`, `__pycache__/`) finds nothing to remove. In `clean_inbenchmark_tasks=false` mode the git trio is EXPECTED to be dirty (the kept task files) — skip it; only the artifact-removal block must pass.

### Step 5 — Return

`reset-workspace: <repo> — <full reset to clone | harbor artifacts removed, N task files kept> — all smoke checks PASSED`. If ANY check FAILED, return the failing checks and do NOT claim success.

## Constraints

- **DESTRUCTIVE + irreversible.** Default mode runs `git reset --hard` + `git clean -fdx` — discards ALL local changes + untracked/ignored files in `<repo>`, not only plugin ones. The Step 2 confirmation is mandatory.
- **Git working tree required** — abort otherwise.
- **Never touch anything outside `<repo>`** (no plugin-side files, no other repos).
- **English-only** output.
- **Success only when ALL applicable smoke checks pass.** A partial clean is a FAIL, reported as such.

## Examples

```text
# Full reset — benchmark identical to its original clone (default)
/harbor:reset-workspace repo=/home/me/IsaacLab

# Remove harbor/ + .venv but KEEP tasks authored directly in the benchmark source
/harbor:reset-workspace repo=/home/me/IsaacLab clean_inbenchmark_tasks=false
```
