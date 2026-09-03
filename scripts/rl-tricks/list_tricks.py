#!/usr/bin/env python3
"""
List every RL trick available under ${CLAUDE_PLUGIN_ROOT}/knowledge/templates/rl-tricks/.

Each trick has a `manifest.yaml` that declares name, description, applicable
algorithms, backend, and config_patches. This script just enumerates them.

Usage:
    python list_tricks.py [--plugin-root <path>]

The plugin root is resolved from this file's own location. CLAUDE_PLUGIN_ROOT is
deliberately not consulted — see `_plugin_root` for why. Pass --plugin-root to
point at a different tree.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _plugin_root(override: str | None) -> Path:
    if override:
        return Path(override).resolve()
    # A script's own location is the only reliable answer to "which plugin tree am I in".
    # CLAUDE_PLUGIN_ROOT is deliberately NOT consulted: it is ambient state that a stale or
    # foreign value would silently win, pointing this script at another tree. Callers that
    # genuinely need a different root pass --plugin-root.
    return Path(__file__).resolve().parents[2]   # <plugin>/scripts/rl-tricks/<this>.py


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--plugin-root", default=None)
    p.add_argument("--algo", default=None,
                   help="filter to tricks that apply to this algorithm")
    p.add_argument("--json", action="store_true",
                   help="emit JSON instead of human-readable text")
    args = p.parse_args()

    root = _plugin_root(args.plugin_root)
    tricks_dir = root / "knowledge" / "templates" / "rl-tricks"
    if not tricks_dir.is_dir():
        print(f"[error] no tricks directory at {tricks_dir}", file=sys.stderr)
        return 2

    try:
        import yaml
    except ImportError:
        print("[error] PyYAML required (pip install pyyaml)", file=sys.stderr)
        return 2

    records = []
    for child in sorted(tricks_dir.iterdir()):
        manifest = child / "manifest.yaml"
        if not manifest.is_file():
            continue
        try:
            data = yaml.safe_load(manifest.read_text())
        except Exception as e:
            print(f"[warn] could not parse {manifest}: {e}", file=sys.stderr)
            continue
        if args.algo and args.algo not in (data.get("algorithms") or []):
            continue
        records.append({
            "name": data.get("name", child.name),
            "description": (data.get("description") or "").strip(),
            "algorithms": data.get("algorithms") or [],
            "backend": data.get("backend"),
            "config_patches": data.get("config_patches") or [],
            "caveats": data.get("caveats") or [],
            "references": data.get("references") or [],
        })

    if args.json:
        print(json.dumps(records, indent=2))
        return 0

    if not records:
        print("(no tricks available)")
        return 0

    print(f"Available RL tricks ({len(records)}):\n")
    for r in records:
        print(f"  {r['name']}")
        print(f"    backend:    {r['backend']}")
        print(f"    algorithms: {', '.join(r['algorithms'])}")
        first_line = r['description'].splitlines()[0] if r['description'] else ''
        if first_line:
            print(f"    summary:    {first_line}")
        print()
    print("Use /harbor:rl-add-trick <name> [algorithm=<algo>] to apply a trick.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
