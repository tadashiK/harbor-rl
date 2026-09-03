#!/usr/bin/env python3
"""Resolve the rendered RL suite spec to its canonical fields — single source of truth.

Every command/agent that needs the algorithm slug, scripts dir, parallel flag, or a
config name reads it THROUGH this helper, so the JSON key path can never drift again.

The spec (`harbor/rl-integration-generator/rl-suite-spec.json`, rendered from
rl-suite-spec.json.template) stores:
  - the algorithm slug at  algorithm_source.slug      (NOT `algorithm_slug`, NOT top-level)
  - the parallel flag at   algorithm_source.parallel  (NOT top-level)
  - scripts_dir at the top level (defaults to harbor/scripts/rl/<slug>)

Usage:
  resolve_suite.py [--repo DIR] [--algo ALGO] [--field {slug,scripts_dir,parallel,config_name}]

Default (no --field): print shell-evalable KEY=value lines —
  SLUG=<slug>
  SCRIPTS_DIR=<scripts_dir>
  PARALLEL=true|false
  CONFIG_NAME=<algo>[.parallel]   (only when --algo is given)
  e.g.  eval "$(python3 .../resolve_suite.py --algo ppo)"

With --field: print just that value (config_name requires --algo) —
  e.g.  slug=$(python3 .../resolve_suite.py --field slug)

Exits 1 with a message on missing spec / missing slug / bad usage.
"""
import argparse
import json
import sys
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=".")
    ap.add_argument("--algo")
    ap.add_argument("--field", choices=["slug", "scripts_dir", "parallel", "config_name"])
    args = ap.parse_args()

    spec_path = Path(args.repo) / "harbor" / "rl-integration-generator" / "rl-suite-spec.json"
    if not spec_path.is_file():
        sys.stderr.write(f"no rl-suite-spec.json at {spec_path} — run rl-integration-generator first\n")
        return 1

    spec = json.loads(spec_path.read_text())
    src = spec.get("algorithm_source", {})
    slug = src.get("slug")
    if not slug:
        sys.stderr.write("rl-suite-spec.json: algorithm_source.slug missing\n")
        return 1
    scripts_dir = spec.get("scripts_dir", f"harbor/scripts/rl/{slug}")
    parallel = bool(src.get("parallel", False))

    def config_name() -> str:
        if not args.algo:
            sys.stderr.write("--algo is required for config_name\n")
            sys.exit(1)
        return f"{args.algo}.parallel" if parallel else args.algo

    if args.field:
        value = {
            "slug": slug,
            "scripts_dir": scripts_dir,
            "parallel": "true" if parallel else "false",
            "config_name": config_name() if args.field == "config_name" else "",
        }[args.field]
        print(value)
    else:
        print(f"SLUG={slug}")
        print(f"SCRIPTS_DIR={scripts_dir}")
        print(f"PARALLEL={'true' if parallel else 'false'}")
        if args.algo:
            print(f"CONFIG_NAME={config_name()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
