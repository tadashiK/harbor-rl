#!/usr/bin/env python3
"""Deterministic same-repo task clone tool for /harbor:task-clone (task-cloner).

This is the ONE source of clone file-ops. It copies a source task's editable
surface (its ``*_env_cfg.py`` + the task-local ``mdp/`` package the surface
touches), rewires the cloned cfg's imports to the copies, mirrors the source's
``gym.register`` for ``<dest>`` (suffix BEFORE any ``-vN`` token, no ``#``), and
writes a manifest of every created file + the registration anchor.

It encodes the clone-contract checks that are DETERMINISTIC — id legality (the
registration rule), independence (SC2: the cloned cfg imports its copied surface,
not the source's), and dest-registered — as asserts. The checks that need a
running simulator (SC1 build / SC3 per-term logging / SC4 rollout) stay OUT:
the caller (task-cloner agent) runs those as sim-side smokes.

    clone_task.py --op create --repo <path> --source <TaskID> --dest <TaskID> \
        [--surface reward] [--manifest <path>]
    clone_task.py --op delete --repo <path> [--source <TaskID>] --manifest <path>

Prints a JSON verdict to stdout. Exits 0 on success, non-zero on any hard
check failure (leaving the repo clean — create rolls back its own partial files).
"""
from __future__ import annotations

import argparse
import ast
import json
import re
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

SKIP_DIRS = {".venv", ".git", "__pycache__", "outputs", "rl_experiments", "node_modules"}


# ---------------------------------------------------------------------------
# id rules (clone-contract "Registration rule")
# ---------------------------------------------------------------------------

_VN = re.compile(r"-v\d+$")
_LEGAL_ID = re.compile(r"^[\w:.\-]+$")


def derive_suffix(source_id: str, dest_id: str) -> str:
    """The token dest inserts before any -vN. dest MUST equal source with a
    single `-<suffix>` inserted before the version token (or appended if none)."""
    if "#" in dest_id or not _LEGAL_ID.match(dest_id):
        raise SystemExit(f"illegal dest id (bad chars / '#'): {dest_id}")
    if dest_id == source_id:
        raise SystemExit("dest id must differ from source id")
    m = _VN.search(source_id)
    if m:
        stem, ver = source_id[: m.start()], source_id[m.start():]
        if not (dest_id.startswith(stem + "-") and dest_id.endswith(ver)):
            raise SystemExit(
                f"dest must be '{stem}-<suffix>{ver}' (suffix before {ver}); got {dest_id}")
        suffix = dest_id[len(stem) + 1: len(dest_id) - len(ver)]
    else:
        if not dest_id.startswith(source_id + "-"):
            raise SystemExit(f"dest must be '{source_id}-<suffix>'; got {dest_id}")
        suffix = dest_id[len(source_id) + 1:]
    if not suffix or _VN.search("-" + suffix) or not re.match(r"^[\w]+$", suffix):
        raise SystemExit(f"suffix must be a single word-char token; got '{suffix}'")
    return suffix


# ---------------------------------------------------------------------------
# discovery — grep the repo tree for the gym.register site (no simulator needed)
# ---------------------------------------------------------------------------

def _py_files(repo: Path):
    for p in repo.rglob("*.py"):
        if not any(part in SKIP_DIRS for part in p.parts):
            yield p


def find_registration(repo: Path, source_id: str):
    """Locate the file + AST node registering source_id and pull its
    env_cfg_entry_point (raw source segment, so `f"{__name__}..."` is preserved)."""
    for path in _py_files(repo):
        text = path.read_text(encoding="utf-8", errors="ignore")
        if source_id not in text:
            continue
        try:
            tree = ast.parse(text)
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if not (isinstance(node, ast.Call) and _is_register(node.func)):
                continue
            kw = {k.arg: k.value for k in node.keywords if k.arg}
            if "id" not in kw or _literal(kw["id"]) != source_id:
                continue
            entry = _find_entry_point(kw.get("kwargs"))
            if entry is None:
                raise SystemExit(f"register for {source_id} has no env_cfg_entry_point")
            return {
                "file": path,
                "text": text,
                "call_segment": ast.get_source_segment(text, node),
                "entry_point": ast.get_source_segment(text, entry),  # raw
            }
    raise SystemExit(f"no gym.register(id={source_id!r}) found under {repo}")


def _is_register(func) -> bool:
    if isinstance(func, ast.Attribute):
        return func.attr == "register"
    return isinstance(func, ast.Name) and func.id == "register"


def _literal(node):
    return node.value if isinstance(node, ast.Constant) else None


def _find_entry_point(kwargs_node):
    if not isinstance(kwargs_node, ast.Dict):
        return None
    for k, v in zip(kwargs_node.keys, kwargs_node.values):
        if _literal(k) == "env_cfg_entry_point":
            return v
    return None


def resolve_cfg(repo: Path, register_file: Path, entry_point: str):
    """From an env_cfg_entry_point of the form '<module>:<Class>' (module may be
    `f"{__name__}..."`) resolve the cfg module FILE + class name, without importing.
    `__name__` is the package of the register file (its dir's __init__.py)."""
    raw = entry_point.strip()
    for q in ('f"', "f'", '"', "'"):
        if raw.startswith(q):
            raw = raw[len(q):]
            break
    raw = raw.rstrip("\"'")
    if ":" not in raw:
        raise SystemExit(f"malformed env_cfg_entry_point: {entry_point}")
    module, cls = raw.rsplit(":", 1)
    module = module.replace("{__name__}", "").lstrip(".")  # tail after the package
    tail = module.replace(".", "/") + ".py"
    # cfg module is a sibling chain under the register file's package dir.
    base = register_file.parent
    cand = base / tail
    if cand.exists():
        return {"file": cand, "class": cls, "module_base": module.rsplit(".", 1)[-1]}
    # Fallback: search the repo for a file whose path ends with the module tail.
    for p in _py_files(repo):
        if str(p).endswith("/" + tail) or p.name == tail:
            return {"file": p, "class": cls, "module_base": module.rsplit(".", 1)[-1]}
    raise SystemExit(f"cannot locate cfg module file for entry_point {entry_point}")


# ---------------------------------------------------------------------------
# layout discovery
#
# A task's editable surface is NOT always beside its registered cfg. Real IsaacLab
# manipulation tasks split across two levels:
#
#     stack_cube/
#     ├── stack_cube_env_cfg.py        <- RewardsCfg / ActionsCfg / … live HERE
#     ├── mdp/rewards.py               <- reward functions live HERE
#     └── config/franka/
#         └── joint_pos_env_cfg.py     <- the REGISTERED cfg: a thin robot subclass
#
# Copying only the registered file leaves both clones sharing the family's mdp and
# RewardsCfg — the exact collision cloning exists to prevent, with no error raised.
# ---------------------------------------------------------------------------

# The cfg class each surface name edits, used to find which file defines it.
_SURFACE_CLASS = {
    "reward": "RewardsCfg",
    "actions": "ActionsCfg",
    "observations": "ObservationsCfg",
    "events": "EventCfg",
    "terminations": "TerminationsCfg",
}


def find_mdp_package(cfg_file: Path, repo: Path) -> Path | None:
    """Nearest ancestor directory holding an `mdp/` package, walking up from the cfg."""
    d = cfg_file.parent
    while True:
        cand = d / "mdp"
        if (cand / "__init__.py").is_file():
            return cand
        if d == repo or d.parent == d:
            return None
        d = d.parent


def find_surface_files(cfg_file: Path, task_root: Path, surface: list[str]) -> list[Path]:
    """Files under *task_root* (besides the registered cfg) that define a surface class."""
    wanted = {_SURFACE_CLASS[s] for s in surface if s in _SURFACE_CLASS}
    if not wanted:
        return []
    hits = []
    for p in sorted(task_root.glob("*.py")):
        if p == cfg_file:
            continue
        text = p.read_text(encoding="utf-8", errors="ignore")
        if any(re.search(rf"^class {n}\b", text, re.M) for n in wanted):
            hits.append(p)
    return hits


def rewire_mdp(body: str, mdp_name: str) -> str:
    """Point relative mdp imports at the copy, at ANY dot depth (`.`, `...`, …)."""
    body = re.sub(r"\bfrom (\.+) import mdp\b", rf"from \1 import {mdp_name} as mdp", body)
    body = re.sub(r"\bfrom (\.+)mdp\b", rf"from \1{mdp_name}", body)
    return body


def rewire_module(body: str, stem: str, new_stem: str) -> str:
    """Point a relative import of a sibling/ancestor module at its copy."""
    return re.sub(rf"\bfrom (\.+){re.escape(stem)}\b", rf"from \1{new_stem}", body)


# ---------------------------------------------------------------------------
# create
# ---------------------------------------------------------------------------

def op_create(repo: Path, source_id: str, dest_id: str, surface: list[str],
              manifest_path: Path) -> dict:
    suffix = derive_suffix(source_id, dest_id)
    reg = find_registration(repo, source_id)
    cfg = resolve_cfg(repo, reg["file"], reg["entry_point"])

    created: list[Path] = []
    created_dirs: list[Path] = []
    try:
        # 1. Copy the task-local mdp package, found by walking UP from the registered
        #    cfg — it sits at the task-package root, which for `config/<robot>/` layouts
        #    is two levels above the cfg, not beside it.
        mdp_src = find_mdp_package(cfg["file"], repo)
        mdp_dst_name = f"mdp_{suffix}"
        copied_mdp = False
        task_root = None
        if mdp_src is not None:
            task_root = mdp_src.parent
            mdp_dst = task_root / mdp_dst_name
            shutil.copytree(mdp_src, mdp_dst,
                            ignore=shutil.ignore_patterns("__pycache__"))
            created.extend(sorted(mdp_dst.rglob("*.py")))
            created_dirs.append(mdp_dst)
            copied_mdp = True

        # 2. Copy every OTHER file defining a surface class (the family base cfg that
        #    holds RewardsCfg & friends). Without this the clone inherits the source's
        #    reward by subclassing it, and two slots edit one file.
        surface_files = find_surface_files(cfg["file"], task_root, surface) if task_root else []
        copied_surface = {}
        for src in surface_files:
            dst = src.with_name(f"{src.stem}_{suffix}.py")
            body = src.read_text(encoding="utf-8")
            if copied_mdp:
                body = rewire_mdp(body, mdp_dst_name)
            dst.write_text(body, encoding="utf-8")
            created.append(dst)
            copied_surface[src.stem] = dst.stem

        # 3. Copy the registered cfg -> dest-named sibling; rename its class; rewire its
        #    imports to the copies made above.
        cfg_dir = cfg["file"].parent
        new_cfg = cfg_dir / f"{cfg['file'].stem}_{suffix}.py"
        new_cls = f"{cfg['class']}_{suffix}"
        body = cfg["file"].read_text(encoding="utf-8")
        body = re.sub(rf"\b{re.escape(cfg['class'])}\b", new_cls, body)
        if copied_mdp:
            body = rewire_mdp(body, mdp_dst_name)
        for stem, new_stem in copied_surface.items():
            body = rewire_module(body, stem, new_stem)
        new_cfg.write_text(body, encoding="utf-8")
        created.append(new_cfg)

        # 3. Mirror the registration for <dest> (suffix before -vN). One entry,
        #    appended to the source's register file — the sanctioned edit.
        anchor = _build_register_entry(reg, source_id, dest_id, cfg, new_cfg, new_cls)
        with reg["file"].open("a", encoding="utf-8") as f:
            f.write(anchor)

        # --- deterministic clone-contract asserts (build/rollout stay for smokes) ---
        #
        # SC2 independence FAILS CLOSED. These checks used to sit behind `if copied_mdp:`,
        # so a layout where the mdp package was not found produced zero checks and still
        # reported "independence": "pass" — a clone that silently shared its reward with
        # every sibling. An unverifiable clone is a failed clone.
        assert new_cfg.exists(), "cloned cfg not written"
        assert mdp_src is not None, (
            f"no mdp package found above {cfg['file']} — cannot give this clone its own "
            f"editable surface, so it would share the source's. Refusing to report success."
        )
        for name in (_SURFACE_CLASS[s] for s in surface if s in _SURFACE_CLASS):
            defined = any(re.search(rf"^class {name}\b", p.read_text(encoding="utf-8"), re.M)
                          for p in created if p.suffix == ".py")
            assert defined, (
                f"surface '{name}' is not defined in any cloned file — the clone would "
                f"inherit it from the source and edits would leak across slots"
            )
        # No cloned file OUTSIDE the copied mdp package may still reach the source's mdp.
        for p in [c for c in created if c.suffix == ".py" and mdp_dst_name not in c.parts]:
            text = p.read_text(encoding="utf-8")
            assert re.search(r"\bfrom \.+mdp\b", text) is None, \
                f"{p.name} still imports the source's .mdp"
            assert re.search(r"\bfrom \.+ import mdp\b", text) is None, \
                f"{p.name} still imports the source's mdp package"
        assert dest_id in reg["file"].read_text(encoding="utf-8"), "dest not registered"
    except BaseException:
        _rollback(created_dirs, created, reg, dest_id)
        raise

    manifest = {
        "source_id": source_id,
        "dest_id": dest_id,
        "surface": surface,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "cloned_files": [str(p.relative_to(repo)) for p in created],
        "cloned_dirs": [str(p.relative_to(repo)) for p in created_dirs],
        "registration": {"file": str(reg["file"].relative_to(repo)), "anchor": anchor},
        # Where the candidate actually edits the reward: the cloned file defining
        # RewardsCfg (the family base for config/<robot>/ layouts), not the registered
        # subclass — plus the cloned reward functions.
        "reward_path": str(_reward_path(created, repo, new_cfg)),
        "mdp_path": str((task_root / mdp_dst_name).relative_to(repo)) if copied_mdp else None,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return {
        "op": "create", "status": "pass", "dest_id": dest_id,
        "cloned_files": manifest["cloned_files"],
        "reward_path": manifest["reward_path"],
        "manifest": str(manifest_path),
        "checks": {"id_rule": "pass", "independence": "pass", "registered": "pass"},
    }


def _reward_path(created: list[Path], repo: Path, fallback: Path) -> Path:
    """The cloned file defining RewardsCfg — where a candidate writes its reward."""
    for p in created:
        if p.suffix != ".py":
            continue
        if re.search(r"^class RewardsCfg\b", p.read_text(encoding="utf-8"), re.M):
            return p.relative_to(repo)
    return fallback.relative_to(repo)


def _build_register_entry(reg, source_id, dest_id, cfg, new_cfg, new_cls) -> str:
    """Duplicate the source's register call, substituting id, cfg module + class."""
    seg = reg["call_segment"]
    seg = seg.replace(f'"{source_id}"', f'"{dest_id}"').replace(f"'{source_id}'", f"'{dest_id}'")
    seg = re.sub(rf"\b{re.escape(cfg['file'].stem)}\b", new_cfg.stem, seg)
    seg = re.sub(rf"\b{re.escape(cfg['class'])}\b", new_cls, seg)
    return f"\n\n# --- harbor clone: {dest_id} (from {source_id}) ---\n{seg}\n"


def _rollback(created_dirs, created, reg, dest_id):
    for d in created_dirs:
        shutil.rmtree(d, ignore_errors=True)
    for p in created:
        try:
            p.unlink()
        except OSError:
            pass
    _remove_registration(reg["file"], dest_id)


# ---------------------------------------------------------------------------
# delete
# ---------------------------------------------------------------------------

def op_delete(repo: Path, source_id: str | None, manifest_path: Path) -> dict:
    if not manifest_path.exists():
        return {"op": "delete", "status": "pass", "note": "manifest absent — no-op"}
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    source_id = source_id or manifest.get("source_id")
    for rel in manifest.get("cloned_dirs", []):
        shutil.rmtree(repo / rel, ignore_errors=True)
    for rel in manifest.get("cloned_files", []):
        try:
            (repo / rel).unlink()
        except FileNotFoundError:
            pass
    reg = manifest.get("registration", {})
    reg_file = repo / reg["file"] if reg.get("file") else None
    if reg_file and reg_file.exists() and reg.get("anchor"):
        _remove_text(reg_file, reg["anchor"])
    manifest_path.unlink(missing_ok=True)
    # Deterministic source-intact check (the running-sim gym.make is the caller's).
    src_ok = source_id is None or (
        reg_file is not None and reg_file.exists()
        and source_id in reg_file.read_text(encoding="utf-8"))
    return {
        "op": "delete", "status": "pass" if src_ok else "fail",
        "source_id": source_id, "source_registration_intact": src_ok,
    }


def _remove_registration(reg_file: Path, dest_id: str):
    if not reg_file.exists():
        return
    text = reg_file.read_text(encoding="utf-8")
    text = re.sub(rf"\n*# --- harbor clone: {re.escape(dest_id)} .*?(?=\n# ---|\Z)",
                  "", text, flags=re.DOTALL)
    reg_file.write_text(text, encoding="utf-8")


def _remove_text(reg_file: Path, anchor: str):
    text = reg_file.read_text(encoding="utf-8")
    if anchor in text:
        reg_file.write_text(text.replace(anchor, ""), encoding="utf-8")


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--op", required=True, choices=["create", "delete"])
    ap.add_argument("--repo", required=True, type=Path)
    ap.add_argument("--source")
    ap.add_argument("--dest")
    ap.add_argument("--surface", default="reward",
                    help="comma list of sections whose surface to copy (default reward)")
    ap.add_argument("--manifest", type=Path)
    args = ap.parse_args()

    repo = args.repo.resolve()
    if args.op == "create":
        if not (args.source and args.dest):
            ap.error("--source and --dest are required for create")
        manifest = args.manifest or (repo / "harbor" / "clones" / f"{args.dest}.json")
        surface = [s.strip() for s in args.surface.split(",") if s.strip()]
        out = op_create(repo, args.source, args.dest, surface, manifest.resolve())
    else:
        if not args.manifest:
            ap.error("--manifest is required for delete")
        out = op_delete(repo, args.source, args.manifest.resolve())

    print(json.dumps(out, indent=2))
    sys.exit(0 if out.get("status") == "pass" else 1)


if __name__ == "__main__":
    main()
