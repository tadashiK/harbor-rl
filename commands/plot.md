---
description: Plot mean ± std curves from wandb runs grouped by task × baseline. Reads a YAML spec describing one or more subplots; each subplot averages multiple seeds for the same (task, baseline). Use when the user types /harbor:plot spec=<path> or asks "plot these wandb runs", "compare baselines on these tasks", "make a multi-panel learning curve".
argument-hint: "spec=<path-to-yaml>  [or no args to walk through writing one]"
---

# /harbor:plot — Multi-panel Learning Curves from W&B

Wraps `scripts/plot/render_plot.py`: fetches wandb runs, groups by (task, baseline), smooths each run with Savitzky-Golay, interpolates onto a common x-grid, plots mean as a line and ±1 std as a shaded band, lays subplots out in a `rows × cols` grid.

## Action

### Step 0 — Resolve the spec + output dir

Every invocation writes spec + plot artifacts under a fresh, timestamped folder so results never overwrite each other:

```
<cwd>/plot_output/<YYYYMMDD-HHMMSS>/
   ├── harbor-plot-spec.yaml   (the spec used for this run)
   └── harbor-plot.<ext>       (the rendered figure)
```

Add `plot_output/` to `.gitignore` (it's data, not source). First, compute the run directory:

```bash
RUN_DIR="$(pwd)/plot_output/$(date +%Y%m%d-%H%M%S)"
mkdir -p "$RUN_DIR"
```

Two paths to populate the spec:

1. **`spec=<path>` provided** → copy that file to `$RUN_DIR/harbor-plot-spec.yaml`, rewrite its `output` field to `$RUN_DIR/harbor-plot.<ext>` (preserve the user's chosen extension), and proceed.
2. **No spec arg** → walk the user through building one. Ask via `AskUserQuestion`:
   - **Source**: `explicit run IDs (paste a list)` | `wandb project URL (auto-group by config keys)`
   - **Tasks**: comma-separated names (used as subplot titles)
   - **Baselines**: comma-separated names (used as legend labels + legend order)
   - For source = explicit: ask for the run-ID list per (task × baseline) cell (user can paste `entity/project/run_id` strings).
   - For source = project: ask for the project path (`entity/project`) plus the two config keys to group by (e.g. `env_name`, `algorithm`).
   - **X axis key** (default `_step`)
   - **Y axis key** (default `eval/success_rate`)
   - **Layout**: `rows`, `cols` (must satisfy `rows*cols >= len(tasks)`)
   - **xlim, ylim** (optional; xlim may be a single number or per-figure list)
   - **Output format** (`.pdf` / `.png` / `.svg` / `.html`) — the file always lives at `$RUN_DIR/harbor-plot.<ext>`.
   Write the spec to `$RUN_DIR/harbor-plot-spec.yaml` with `output: $RUN_DIR/harbor-plot.<ext>` and continue.

A reference spec lives at `${CLAUDE_PLUGIN_ROOT}/knowledge/templates/plot/spec.example.yaml` — copy + edit if the user prefers to hand-write.

### Step 1 — Pre-flight

```bash
spec_path="$RUN_DIR/harbor-plot-spec.yaml"
test -f "${spec_path}" || { echo "spec not found: ${spec_path}"; exit 1; }
command -v uv >/dev/null || { echo "uv not on PATH — run install_prerequisites.sh"; exit 1; }
```

W&B credentials are read from `~/.netrc`. If `wandb.Api()` errors with auth failure, surface `/harbor:wandb-setup` as the fix and stop.

### Step 2 — Run the renderer

```bash
uv run --no-project \
  --with wandb --with plotly --with kaleido --with pyyaml --with scipy --with numpy --with pandas \
  python "${CLAUDE_PLUGIN_ROOT}/scripts/plot/render_plot.py" --spec "${spec_path}"
```

`kaleido` is needed only for PDF/PNG/SVG output; HTML output skips it. Stream stdout/stderr live.

### Step 3 — Report

Print BOTH the spec path (`$RUN_DIR/harbor-plot-spec.yaml`) and the rendered output path (verbatim from the script's `[ok] wrote <path>` line). Surface any `[warn]` from a run fetch so the user knows which run was skipped (transient API errors, missing keys, all-NaN history).

## Spec schema (summary — see `knowledge/templates/plot/spec.example.yaml` for the full version)

| Key | Required | Notes |
|---|---|---|
| `figures` OR (`project` + `tasks` + `baselines` + `group_by`) | one of | Run sources. Explicit IDs vs project enumeration. |
| `x_axis`, `y_axis` | yes | wandb history keys (e.g. `_step`, `eval/success_rate`). |
| `xlim`, `ylim` | no | scalar / per-figure list; `[lo, hi]` for ylim. |
| `layout` | yes | `{rows: N, cols: M}`; `rows*cols` must cover `len(figures)`. |
| `smoothing` | no | `{window: 11, order: 3}`; set `window: 1` to disable. |
| `colors` | no | List of `[r, g, b]` (0-255); cycles per baseline. |
| `baseline_styles` | no | Per-baseline overrides: `{<label>: {color: [r,g,b], dash: solid\|dash\|dot\|dashdot}}`. Use to pair labels (same color, different dash). |
| `output` | yes | Absolute path. Suffix decides format (`.pdf` / `.png` / `.svg` via kaleido, `.html` direct). |
| `samples` | no | Cap on `run.history(samples=...)`. Default 10M. |
| `interp_points` | no | Resolution of common x-grid after interpolation. Default 100. |

## Constraints

- **Do NOT modify the user's spec.** If validation fails (e.g. `rows*cols < len(figures)`), surface the error and let them edit the file.
- **Do NOT leak the W&B API key.** It lives in `~/.netrc`; never echo or copy it.
- **Do NOT silently swap axis keys** if a run is missing them. The renderer skips that run and prints a `[warn]`; let the user decide whether to retry.
- **Do NOT auto-install `wandb`/`plotly` into a project venv.** Always invoke via `uv run --no-project --with ...` so the host stays clean.
- **HTML output** needs no `kaleido`; for `.html`-only, drop `--with kaleido` for a faster cold start.
