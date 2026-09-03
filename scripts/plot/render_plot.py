#!/usr/bin/env python3
"""Render plotly subplot grid of wandb runs grouped by task × baseline.

Reads a YAML spec describing one or more figures (subplots), each with runs
grouped under named baselines. For each (task, baseline) cell:
  1. fetch each run's history[(x_axis, y_axis)] via wandb.Api
  2. apply Savitzky-Golay smoothing
  3. interpolate every run to a common x grid
  4. plot mean as a line + std band as a shaded region

Spec schema — see `knowledge/templates/plot/spec.example.yaml`.
"""
from __future__ import annotations

import argparse
import json
import sys
from math import factorial
from pathlib import Path

import numpy as np
import plotly.graph_objs as go
import plotly.io as pio
import scipy.interpolate
import wandb
import yaml
from plotly.subplots import make_subplots


def savitzky_golay(y: np.ndarray, window_size: int, order: int) -> np.ndarray:
    window_size = abs(int(window_size))
    order = abs(int(order))
    if window_size % 2 != 1 or window_size < 1:
        raise ValueError("window_size must be a positive odd integer")
    if window_size < order + 2:
        raise ValueError("window_size too small for the polynomial order")
    half = (window_size - 1) // 2
    b = np.array([[k ** i for i in range(order + 1)] for k in range(-half, half + 1)])
    m = np.linalg.pinv(b)[0] * factorial(0)
    firstvals = y[0] - np.abs(y[1:half + 1][::-1] - y[0])
    lastvals = y[-1] + np.abs(y[-half - 1:-1][::-1] - y[-1])
    y_pad = np.concatenate((firstvals, y, lastvals))
    return np.convolve(m[::-1], y_pad, mode="valid")


def interpolate_xy(xs: np.ndarray, ys: np.ndarray, x_target: np.ndarray) -> np.ndarray:
    f = scipy.interpolate.interp1d(xs, ys, bounds_error=False, fill_value="extrapolate")
    return f(x_target)


def mean_std_bounds(stack: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    arr = np.array(stack)
    mean = arr.mean(axis=0)
    std = arr.std(axis=0)
    return mean, mean - std, mean + std


def _dotted_get(d: dict, path: str):
    """Look up a possibly-dotted key in a nested dict. `algo.name` walks d['algo']['name']."""
    cur = d
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur


def discover_from_project(project: str, tasks: list[str], baselines: list[str],
                          group_by: list[str], api: wandb.Api,
                          task_labels: list[str] | None = None,
                          baseline_labels: list[str] | None = None) -> list[dict]:
    """Walk every run in a wandb project and bucket by (task, baseline) using config keys.

    `group_by` is a 2-tuple of config keys like ["task", "algo.name"]. Dotted paths
    walk into nested dicts. Runs whose looked-up values are in `tasks` / `baselines`
    are kept; everything else is dropped. Returns one figure per task in the order given.

    `task_labels` / `baseline_labels` (optional) override the displayed subplot title
    and legend label respectively, while matching still uses the raw config values.
    """
    if len(group_by) != 2:
        sys.exit("group_by must be exactly [task_key, baseline_key]")
    if task_labels is not None and len(task_labels) != len(tasks):
        sys.exit("task_labels must be the same length as tasks")
    if baseline_labels is not None and len(baseline_labels) != len(baselines):
        sys.exit("baseline_labels must be the same length as baselines")
    task_key, base_key = group_by
    t_label = task_labels or tasks
    b_label = baseline_labels or baselines
    base_to_label = dict(zip(baselines, b_label))
    figures: list[dict] = [{"title": t_label[i], "runs": {b: [] for b in b_label}} for i in range(len(tasks))]
    task_to_idx = {t: i for i, t in enumerate(tasks)}
    for run in api.runs(project):
        cfg = run.config or {}
        t = _dotted_get(cfg, task_key)
        b = _dotted_get(cfg, base_key)
        if t in task_to_idx and b in baselines:
            figures[task_to_idx[t]]["runs"][base_to_label[b]].append(f"{run.entity}/{run.project}/{run.id}")
    return figures


def fetch_curve(run_path: str, x_axis: str, y_axis: str, xlim: float | None,
                samples: int, api: wandb.Api) -> tuple[np.ndarray, np.ndarray]:
    run = api.run(run_path)
    history = run.history(samples=samples, keys=[x_axis, y_axis])
    df = history[[x_axis, y_axis]].dropna()
    if xlim is not None:
        df = df[df[x_axis] < xlim]
    if len(df) == 0:
        raise RuntimeError(f"no points for {run_path} (after NaN/xlim filter)")
    return df[x_axis].values, df[y_axis].values


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--spec", required=True, help="Path to YAML/JSON plot spec")
    args = p.parse_args()

    spec_path = Path(args.spec)
    text = spec_path.read_text()
    spec = yaml.safe_load(text) if spec_path.suffix in (".yaml", ".yml") else json.loads(text)

    x_axis = spec["x_axis"]
    y_axis = spec["y_axis"]
    xlim_spec = spec.get("xlim")
    ylim = spec.get("ylim")
    layout = spec.get("layout", {})
    rows = int(layout.get("rows", 1))
    cols = int(layout.get("cols", 1))
    smoothing = spec.get("smoothing", {"window": 11, "order": 3})
    sg_window = int(smoothing.get("window", 11))
    sg_order = int(smoothing.get("order", 3))
    colors = spec.get("colors") or [
        [55, 126, 184], [255, 127, 0], [77, 175, 74],
        [152, 78, 163], [228, 26, 28], [166, 86, 40],
    ]
    output = spec["output"]
    samples = int(spec.get("samples", 10_000_000))
    interp_points = int(spec.get("interp_points", 100))
    template = spec.get("template", "gridon")
    title_font_size = int(spec.get("font_size", 28))
    subtitle_font_size = int(spec.get("subtitle_font_size", title_font_size))
    tick_font_size = int(spec.get("tick_font_size", title_font_size))
    legend_y = float(spec.get("legend_y", -0.18))
    legend_itemwidth = spec.get("legend_itemwidth")
    legend_font_size = spec.get("legend_font_size")
    width_per_col = int(spec.get("width_per_col", 600))
    height_per_row = int(spec.get("height_per_row", 500))
    baseline_styles = spec.get("baseline_styles", {}) or {}
    y_label_first_col_only = bool(spec.get("y_label_first_col_only", False))

    api = wandb.Api()

    # Resolve `figures`
    if "figures" in spec:
        figures = spec["figures"]
    elif "project" in spec:
        figures = discover_from_project(
            project=spec["project"],
            tasks=spec["tasks"],
            baselines=spec["baselines"],
            group_by=spec.get("group_by", ["task", "baseline"]),
            task_labels=spec.get("task_labels"),
            baseline_labels=spec.get("baseline_labels"),
            api=api,
        )
    else:
        sys.exit("spec must contain either `figures` or (`project` + `tasks` + `baselines` + `group_by`)")

    # Per-figure xlim: scalar broadcast or list
    if isinstance(xlim_spec, list):
        xlims = xlim_spec
    else:
        xlims = [xlim_spec] * len(figures)
    if len(xlims) < len(figures):
        sys.exit(f"xlim list has {len(xlims)} entries but {len(figures)} figures")

    pio.templates.default = template
    titles = [f.get("title", f"fig{i}") for i, f in enumerate(figures)]
    fig = make_subplots(
        rows=rows, cols=cols, subplot_titles=titles,
        horizontal_spacing=spec.get("horizontal_spacing", 0.06),
        vertical_spacing=spec.get("vertical_spacing", 0.15),
    )

    # Stable color assignment per baseline name (first occurrence wins)
    baseline_color: dict[str, list[int]] = {}

    for k, figdef in enumerate(figures):
        runs_dict = figdef["runs"]
        xlim = xlims[k]
        row = k // cols + 1
        col = k % cols + 1

        for label, run_paths in runs_dict.items():
            if not run_paths:
                print(f"[warn] {titles[k]} / {label}: no runs, skipping", file=sys.stderr)
                continue
            style = baseline_styles.get(label, {})
            if "color" in style:
                color = style["color"]
            else:
                if label not in baseline_color:
                    baseline_color[label] = colors[len(baseline_color) % len(colors)]
                color = baseline_color[label]
            dash = style.get("dash", "solid")

            xs_list, ys_list = [], []
            for path in run_paths:
                try:
                    x, y = fetch_curve(path, x_axis, y_axis, xlim, samples, api)
                except Exception as e:
                    print(f"[warn] {path}: {e}", file=sys.stderr)
                    continue
                if sg_window > 1 and len(y) >= sg_window:
                    y = savitzky_golay(y, sg_window, sg_order)
                xs_list.append(x)
                ys_list.append(y)

            if not xs_list:
                continue

            x_min = max(x.min() for x in xs_list)
            x_max = min(x.max() for x in xs_list)
            n_points = min(max(30, min(len(x) for x in xs_list)), interp_points)
            new_x = np.linspace(x_min, x_max, n_points)
            new_ys = [interpolate_xy(xs_list[i], ys_list[i], new_x) for i in range(len(xs_list))]
            mean, lower, upper = mean_std_bounds(new_ys)

            showlegend = (k == 0)
            fig.add_trace(
                go.Scatter(
                    x=new_x, y=mean,
                    line=dict(color=f"rgb({color[0]},{color[1]},{color[2]})", width=3, dash=dash),
                    mode="lines",
                    name=label,
                    showlegend=showlegend,
                ),
                row=row, col=col,
            )
            fig.add_trace(
                go.Scatter(
                    x=np.concatenate((new_x, new_x[::-1])),
                    y=np.concatenate((upper, lower[::-1])),
                    fill="toself",
                    fillcolor=f"rgba({color[0]},{color[1]},{color[2]},0.2)",
                    line=dict(color="rgba(255,255,255,0)"),
                    hoverinfo="skip",
                    showlegend=False,
                ),
                row=row, col=col,
            )

        x_axis_kw = dict(showline=True, linewidth=2, linecolor="black",
                         gridcolor="#c2c2c2", title_text=spec.get("x_label", x_axis),
                         tickfont=dict(size=tick_font_size),
                         row=row, col=col)
        if xlim is not None:
            x_axis_kw["range"] = [0, xlim]
        fig.update_xaxes(**x_axis_kw)

        y_label_text = spec.get("y_label", y_axis)
        if y_label_first_col_only and col != 1:
            y_label_text = ""
        y_axis_kw = dict(showline=True, linewidth=2, linecolor="black",
                         gridcolor="#c2c2c2", title_text=y_label_text,
                         tickfont=dict(size=tick_font_size),
                         row=row, col=col)
        if ylim is not None:
            y_axis_kw["range"] = ylim
        fig.update_yaxes(**y_axis_kw)

    legend_kw = dict(orientation="h", yanchor="top", y=legend_y,
                     xanchor="center", x=0.5)
    if legend_itemwidth is not None:
        legend_kw["itemwidth"] = float(legend_itemwidth)
    if legend_font_size is not None:
        legend_kw["font"] = dict(size=int(legend_font_size))
    fig.update_layout(
        width=width_per_col * cols,
        height=height_per_row * rows,
        title={"text": spec.get("title", ""), "x": 0.5},
        font=dict(family="Arial", size=title_font_size),
        legend=legend_kw,
    )
    fig.update_annotations(font_size=subtitle_font_size)

    out = Path(output).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.suffix.lower() == ".html":
        fig.write_html(str(out))
    else:
        fig.write_image(str(out))
    print(f"[ok] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
