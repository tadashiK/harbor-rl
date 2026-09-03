# Contributing to HARBOR

Thanks for your interest. The most valuable contribution is usually not a code change — it is running HARBOR against a simulator or task family we have not covered and reporting exactly what broke. The harness improves by accumulating that kind of evidence.

## Ways to help

**Ask first if you are not sure.** [Discord](https://discord.gg/W3ywA3jUKs) is the place for setup trouble, a simulator that will not cooperate, or a question about how to shape a task. Not everything that goes wrong is a bug, and a conversation is faster than an issue when it is not.

**Report a failure with its artifacts.** HARBOR writes an inspectable record of every run. A good issue includes the relevant `history.md`, the failing smoke's `verdict.json`, and what you asked for. That is usually enough to diagnose without a reproduction.

**Add a simulator.** If you get HARBOR working against a benchmark it has not seen, the `probe-benchmark` guide and any new references you needed are worth upstreaming.

**Contribute experience.** Heuristics that made a stage work are first-class content here, not folklore:

```text
/harbor:update-experience target=reward-tuning-agent experience="..."
```

**File a task.** A `probe-task` specification filed into `knowledge/experiences/task-library/` makes that task reproducible in any other benchmark.

## Development setup

```bash
git clone https://github.com/supersglzc/harbor-rl.git
cd harbor-rl
pip install -r tests/requirements-test.txt
pytest tests/contract tests/unit -q
```

Layers 1 and 2 need no GPU, no simulator, and no network. Layer 3 — the end-to-end agentic pipeline — needs a GPU benchmark host and is run through `/harbor:test layers=3`.

## Architecture

Read [`CLAUDE.md`](CLAUDE.md) before making a structural change. It defines the six-layer model and, more usefully, a decision procedure for where a new module belongs. Placing something at the wrong layer is the most common review comment.

The short version:

| Layer | What lives there |
|---|---|
| L2 entry points | `commands/<name>.md` — anything the user invokes directly |
| L3 subagents | `agents/<name>.md` — multi-step reasoning in an isolated context |
| L4 tools | `scripts/<owner>/*.py` — one deterministic input → output |
| L5 knowledge | `knowledge/{templates,references,experiences}/` — read-only material loaded on demand |

## Hard constraints

These are enforced by the contract tests, not by review, so a violation fails CI:

1. **Dispatch depth ≤ 2.** Exactly one agent (`reward-tuning-agent`) carries the `Agent` tool. Every other agent is a leaf.
2. **All generated files live under `<repo>/harbor/`**, except the three user-facing smoke entry points at `scripts/`.
3. **Generated receipts are English-only**, regardless of the language of the conversation that produced them.
4. **Every script gets a unit test.** `UNTESTED_DEBT` in `tests/contract/test_script_conventions.py` may only shrink. Scripts that genuinely cannot run in CI go in `UNTESTABLE` with a stated reason.
5. **Tools report honestly.** Exit 0 means the tool ran; non-zero means it could not run at all. A tool must emit a verdict object even when the answer is "it failed".

## Code style

- **Think before coding.** State assumptions; if a choice is genuinely ambiguous, ask rather than picking silently.
- **Simplicity first.** The minimum code that solves the problem. No speculative abstraction, configurability, or error handling for impossible states.
- **Surgical changes.** Touch only what the task requires. Do not reformat adjacent code or delete pre-existing dead code unless that is the task.
- **Goal-driven.** Define the verifiable success criterion up front and loop until it passes. "Make it work" is not a criterion.

## Documentation

The command and agent reference pages are generated from the plugin source:

```bash
python3 tools/gen_docs_reference.py
```

CI fails if what is committed differs from what the generator produces, so run it after changing any command or agent frontmatter. The generated pages are `docs/guide/{commands,agents,task-library}.md` — do not hand-edit those three. Every other page under `docs/guide/` is written by hand.

## Pull requests

Branch from `main`, keep the change scoped to one concern, and make sure `pytest tests/contract tests/unit` passes. If your change alters behavior an agent depends on, say so explicitly in the description — agent instructions are load-bearing here in a way that ordinary code comments are not.

## License

By contributing you agree that your contributions are licensed under the [Apache License 2.0](LICENSE).

Parts of HARBOR are derived from upstream projects under their own licenses — the `custom_torch`
and `custom_jax` algorithm trees, and the verbatim code excerpts in the task library. Those are
recorded in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md), and every derived template file
carries a header pointing there. If you add code adapted from somewhere else, add its notice
too: a header that gets dropped in the next refactor is how the attribution went missing the
first time.
