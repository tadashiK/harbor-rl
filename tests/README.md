# harbor plugin tests

Three layers, because the plugin is two kinds of thing — deterministic code
(`scripts/`, rendered templates) and LLM instruction files
(`commands/`, `agents/`).

| Layer | Dir | Tests | Needs |
|---|---|---|---|
| **1. Contract** | `tests/contract/` | frontmatter valid · in-repo references resolve · CLAUDE.md map ↔ disk · open-source hygiene | nothing (any machine, < 1s) |
| **2. Unit** | `tests/unit/` | `resolve_suite.py` behavior (canonical keys) · every `scripts/*.py` compiles · every `*.py.template` compiles | nothing / light deps |
| **3. E2E pipeline** | (via `/harbor:test-pipeline`) | the full create→train→reset chain on one task | GPU box + a benchmark + the Claude runtime |

Layers 1–2 are plain pytest and run in CI on any machine. Layer 3 is agentic +
GPU-gated and runs through the `/harbor:test-pipeline` command (or a headless
`claude -p` driver) on the benchmark host — not part of `pytest`.

## Run (Layers 1–2)

```bash
pip install -r tests/requirements-test.txt
pytest tests/ -q
```

Run just the fast contract layer:

```bash
pytest tests/contract -q
```

## Next increments

- **Unit (behavioral):** fixtures + tests for `apply_trick.py`, `validate_rl_suite.py`,
  `discover_{algorithms,rl_tasks}.py`, `check_reward_logger.py`, `list_tricks.py`,
  Today's unit layer covers `resolve_suite.py` behavior
  plus syntax-compile gates for all scripts + templates.
- **Layer 3 (e2e):** `/harbor:test-pipeline repo=<path> task=<id>` — run the full
  env-install → benchmark → rl-integration → task-create → rl-run/eval/render →
  reset-workspace chain on one IsaacLab task, asserting each stage's embedded smoke
  + artifacts. GPU-gated; runs through the Claude runtime, not pytest.
