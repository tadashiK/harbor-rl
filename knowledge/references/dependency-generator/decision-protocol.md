# Decision protocol (dependency-generator)

Three-tier resolution order used in `agents/dependency-generator.md` Step 4 (InstallationPlan confirmation).

## The three tiers

1. **Main-thread pre-election (highest priority).**
   Scan the dispatch prompt the subagent received from the main thread. If it contains any of the step's pre-elect keywords, apply the directive directly and proceed. Do **not** call `AskUserQuestion`.

2. **`AskUserQuestion` (interactive fallback).**
   If main-thread pre-election is silent, call `AskUserQuestion` with the step's question text and options.

3. **Default-on-no-response (non-interactive fallback).**
   If `AskUserQuestion` returns `null` / fails / is unsupported in the current execution context (e.g. the surrounding harness lacks `SendMessage`-style continuation), or if no answer arrives within the tool call, **apply the step's default branch and continue silently**. Do not exit. Do not block. Record the fallback in the final receipt so the operator can see it fired.

## Step-specific bindings

Each step that uses this protocol must specify:

- **pre-elect keywords** — phrases in the dispatch prompt that bypass `AskUserQuestion`
- **question** — exact question text + options for `AskUserQuestion`
- **default branch** — which option to apply when no response arrives
- **receipt field** — JSON field name where the chosen path is recorded

### Step 4 — InstallationPlan confirmation

Bindings:

| Binding | Value |
|---------|-------|
| pre-elect keywords | `"trust readme"` / `"auto-confirm install plan"` → option (a)<br>`"plan-only"` / `"skip readme"` / `"probe-only install"` → option (c) |
| question | "Generated InstallationPlan from README has N steps. Top 3: \<step summaries\>. Confirm, edit, or fall back to probe-only?" |
| options | (a) Confirm — render setup_uv.sh from this plan<br>(b) Edit — let the user paste edits to install_plan.json before render<br>(c) Fall back to probe-only — ignore plan, use heuristic install (uv sync if uv.lock present, else uv pip install) |
| default branch | (a) Confirm |
| receipt field | `install_plan_confidence` (values: `"high_user_confirmed"`, `"high_user_edited"`, `"high_pre_elected"`, `"fallback_probe_only"`, `"default_no_user_response"`) |

## Implementation notes

- The protocol is a **prompt convention**, not code. There is no shared library function; each step in the agent's prompt references this file for the pattern and lists its bindings inline.
- `receipt field` values become part of the JSON returned to the main thread (Step 4 → `install_plan_confidence`). The operator inspects these to detect when interactive prompts didn't reach the user.
- When adding a new step that needs three-tier resolution, add a new sub-table here and reference it from the step.
