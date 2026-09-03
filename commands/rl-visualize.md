---
description: Open a HEADED viewer (GLFW window) showing a trained RL policy in action. Loads a checkpoint, runs N steps on the CPU MuJoCo backend, displays each frame live. Use when the user types /harbor:rl-visualize checkpoint=<path> [task=<id>] or asks "show me the trained policy", "open a viewer", "watch the agent run live".
argument-hint: "checkpoint=<path> [task=<id>] [n_steps=N]"
---

# /harbor:rl-visualize — Open a Headed Viewer

Loads a trained checkpoint and opens a live MuJoCo GLFW window showing the policy in action. **Different from `/harbor:rl-eval`** — this is a visual sanity check, not a metric collector. **Different from `render.py`** — that writes an offscreen MP4; this opens a live window.

## Required argument

| Arg | Notes |
|---|---|
| `checkpoint` | Absolute or repo-relative path to `<trial_dir>/AgentXXX_saved.pkl`. |

## Optional arguments

| Arg | Default | Effect |
|---|---|---|
| `task` | inferred from checkpoint's `resolved_config.yaml` | Override task identifier. The CPU backend is used (the `Mjx` prefix is stripped automatically). |
| `n_steps` | 2000 | How many env steps to play. Auto-resets on terminal. |
| `seed` | 0 | Env reset seed. |

## Action

1. **Pre-flight**:
   ```bash
   test -n "${DISPLAY:-}" || echo "[warn] DISPLAY unset — the GLFW window won't open. Use ssh -X or run on a display host."
   test -e <resolved_checkpoint_path> || { echo "checkpoint not found"; exit 1; }
   test -f harbor/rl-integration-generator/rl-suite-spec.json || { echo "no rl-suite-spec.json"; exit 1; }
   ```

2. **Resolve checkpoint to absolute path.** Auto-infer `algorithm` and `task` from `<checkpoint_dir>/resolved_config.yaml` (same as `/harbor:rl-eval`).

3. **Confirm visualize.py exists**:
   ```bash
   slug=$(python3 "${CLAUDE_PLUGIN_ROOT}/scripts/common/resolve_suite.py" --field slug)
   test -f "harbor/scripts/rl/${slug}/visualize.py" || { \
     echo "no visualize.py in harbor/scripts/rl/${slug}/. Available only for algorithm_source=custom_jax."; \
     exit 1; }
   ```
   `visualize.py` is currently rendered only for `custom_jax`. If the user is on `custom_torch` / `stable_baseline3`, fall back to `render.py` and tell them.

4. **Resolve run prefix**: `<repo>/.venv/bin/python`. Error if `.venv/` is missing.

5. **Launch**:
   ```bash
   <prefix> harbor/scripts/rl/<slug>/visualize.py \
       --config-name=<algo>.parallel \
       task=<task> \
       checkpoint=<abs_path> \
       n_steps=<value>
   ```
   Stream stdout live; the window closes when the process exits.

6. **On exit**, print the per-episode return summary (visualize.py emits this).

## Constraints

- **Headed only.** For an offscreen MP4 use the existing `render.py` (write `…/render.mp4`) — that's a separate concern.
- **CPU backend.** The viewer uses the upstream MuJoCo GLFW backend. The JAX-trained checkpoint is portable across CPU and MJX backends because the network only sees the obs vector.
- **Do NOT call this in headless CI.** It will fail with "cannot open display". Use `render.py` or `/harbor:rl-eval` instead.
- **Single env (n=1) hard-coded** — the viewer can't show 4096 vmap'd envs simultaneously.
- For `custom_torch` / `stable_baseline3` (which don't ship `visualize.py`), point the user at `render.py` for now; a viewer for those sources is a future addition.
