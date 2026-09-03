#!/usr/bin/env python3
"""
Render data_logger.py from the harbor template, with feature flags.

Used by rl-integration-generator to render <repo>/harbor/utils/data_logger.py.
Applies mustache-style block substitution:
  {{#FLAG}} ... {{/FLAG}}    — included only when FLAG is true
  {{ACTIVE_FEATURES}}        — replaced with a human-readable feature summary

Usage:
    python3 render_data_logger.py \
        --backend tb|wandb|both \
        --features scalar[,image][,text] \
        --output path/to/data_logger.py

Example:
    python3 render_data_logger.py --backend both --features scalar,image,text \
        --output /tmp/myproject/utils/data_logger.py
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path


PLUGIN_ROOT = Path(__file__).resolve().parents[2]
TEMPLATE = PLUGIN_ROOT / "knowledge/templates/rl-integration-generator/data_logger.py.template"


def render(template: str, flags: dict[str, bool], scalars: dict[str, str]) -> str:
    """
    Apply mustache-style block conditionals + scalar substitutions.

    Block syntax: {{#FLAG}} ... {{/FLAG}}  (greedy across newlines).
    Scalar syntax: {{NAME}}  (single-line replacement).
    """
    out = template
    # Iterate until no more block markers remain (handles nested-by-name
    # blocks of the same flag at multiple sites — they don't actually nest
    # by name in our template, but loop anyway for safety).
    for flag, enabled in flags.items():
        pattern = re.compile(r"\{\{#" + re.escape(flag) + r"\}\}(.*?)\{\{/" + re.escape(flag) + r"\}\}", re.DOTALL)
        if enabled:
            out = pattern.sub(r"\1", out)
        else:
            out = pattern.sub("", out)

    # Scalars (single-line)
    for name, value in scalars.items():
        out = out.replace("{{" + name + "}}", value)

    # Sanity: no leftover mustache markers
    leftover = re.findall(r"\{\{[^}]+\}\}", out)
    if leftover:
        sys.stderr.write("WARNING: unprocessed mustache markers: {}\n".format(set(leftover)))

    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backend", required=True, choices=["tb", "wandb", "both"],
                    help="Which logging backend(s) to include.")
    ap.add_argument("--features", required=True,
                    help="Comma-separated feature flags from {scalar,image,text}. 'scalar' is mandatory.")
    ap.add_argument("--output", required=True,
                    help="Absolute output path for the rendered data_logger.py.")
    args = ap.parse_args()

    feats = [f.strip() for f in args.features.split(",") if f.strip()]
    valid_feats = {"scalar", "image", "text"}
    unknown = set(feats) - valid_feats
    if unknown:
        ap.error("unknown features: {}; valid: {}".format(unknown, valid_feats))
    if "scalar" not in feats:
        ap.error("'scalar' is mandatory (DataLogger has no point without it)")

    flags = {
        "USE_TB":    args.backend in ("tb", "both"),
        "USE_WANDB": args.backend in ("wandb", "both"),
        "LOG_IMAGE": "image" in feats,
        "LOG_TEXT":  "text" in feats,
    }
    scalars = {
        "ACTIVE_FEATURES": "{} backend(s); features: {}".format(args.backend, ",".join(sorted(feats))),
    }

    if not TEMPLATE.exists():
        ap.error("template not found at {}".format(TEMPLATE))

    rendered = render(TEMPLATE.read_text(), flags, scalars)

    out_path = Path(args.output).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(rendered)

    print("Wrote {} ({} lines)".format(out_path, rendered.count("\n") + 1))
    print("Backend: {} | Features: {}".format(args.backend, ",".join(sorted(feats))))
    print()
    print("Install dependencies:")
    deps = ["numpy"]
    if flags["USE_TB"]:
        deps.append("tensorboardX")
    if flags["USE_WANDB"]:
        deps.append("wandb")
    print("  pip install {}".format(" ".join(deps)))


if __name__ == "__main__":
    main()
