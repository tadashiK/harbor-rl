#!/usr/bin/env python3
"""Decompose a collision mesh with CoACD, and record that CoACD is what produced it.

Every mesh asset a task spawns needs a collision approximation. The convenient path — hand the
raw mesh to the simulator and let it cook `convexDecomposition` at load time — is a trap at
scale: PhysX's cooker is tuned for fidelity, not for shape count, so preserving one small
feature (a mug's handle hole) costs a hull budget in the hundreds. Multiply by `num_envs` and
the contact buffers overflow. A hang-mug task cooked at 128 hulls put half a million convex
shapes in a scene at 4096 envs.

CoACD produces a **coarse, near-convex** decomposition instead: it keeps the topology that
matters (the hole stays open) at a fraction of the shape count. So the rule is:

    every mesh input is decomposed by CoACD, ahead of time, to at most MAX_HULLS parts,
    and each part is spawned as its own convexHull collider.

The second half matters as much as the first. A pre-decomposed asset and a runtime-cooked one
look identical in a render; what distinguishes them is that the pre-decomposed one has N
convexHull collider prims and a manifest, while the runtime-cooked one has ONE prim marked
`convexDecomposition`. This script writes that manifest, and S1's C8 check reads it — otherwise
"we used CoACD" is a claim nobody can verify after the fact.

Modes:
  (default)   decompose --input <mesh> --out-dir <dir>
  --inspect   report an existing asset's collision provenance without changing it. Use on a
              USD or MJCF input: if it carries a single high-hull `convexDecomposition`, it
              needs the same treatment as a raw mesh.

Exit 0 = the tool ran (read the JSON). Non-zero = bad input, or CoACD is not installed.
"""
import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

MAX_HULLS = 32
MANIFEST_SUFFIX = ".coacd.json"

MESH_SUFFIXES = {".obj", ".stl", ".ply", ".glb", ".gltf", ".dae"}
USD_SUFFIXES = {".usd", ".usda", ".usdc"}
XML_SUFFIXES = {".xml", ".mjcf"}

INSTALL_HINT = "pip install coacd  (inside the repo's .venv)"


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def manifest_path(asset):
    """Where the provenance record for `asset` lives. One rule, so the check can find it."""
    return Path(asset).with_suffix(MANIFEST_SUFFIX)


def build_manifest(source, parts, params, version=None):
    return {
        "tool": "coacd",
        "tool_version": version,
        "source": Path(source).name,
        "source_sha256": sha256(source),
        "params": params,
        "hulls": len(parts),
        "parts": [Path(p).name for p in parts],
    }


def validate_manifest(data, max_hulls=MAX_HULLS):
    """Reasons this manifest does NOT prove an acceptable CoACD decomposition."""
    problems = []
    if data.get("tool") != "coacd":
        problems.append(f"tool is {data.get('tool')!r}, not 'coacd' — the collision was not "
                        f"produced by the pre-decomposition step")
    hulls = data.get("hulls")
    if not isinstance(hulls, int) or hulls < 1:
        problems.append(f"hulls is {hulls!r}")
    elif hulls > max_hulls:
        problems.append(f"{hulls} hulls exceeds the {max_hulls} budget — raise the CoACD "
                        f"threshold and re-run; a high hull count is what overflows the "
                        f"contact buffers at high num_envs")
    if len(data.get("parts") or []) != (hulls or -1):
        problems.append(f"manifest lists {len(data.get('parts') or [])} parts but claims "
                        f"{hulls} hulls")
    return problems


# ---------------------------------------------------------------- inspect an existing asset

def inspect_xml(path):
    """MJCF: report every mesh-backed geom and whether it looks pre-decomposed.

    MJCF has no `convexDecomposition` token — MuJoCo convexifies each mesh geom. So the signal
    is the same one in spirit: ONE mesh geom for a concave part means its concavity is gone.
    """
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    meshes = re.findall(r'<mesh\b[^>]*\bfile\s*=\s*"([^"]+)"', text)
    geoms = re.findall(r'<geom\b[^>]*\bmesh\s*=\s*"([^"]+)"', text)
    return {"format": "mjcf", "mesh_assets": sorted(set(meshes)),
            "mesh_geoms": len(geoms),
            "note": "each mesh geom is convexified by MuJoCo; a concave part needs to arrive "
                    "already split into parts"}


def inspect_usd(path):
    """USD: read the approximation tokens — but ONLY when they are actually readable.

    `pxr` is not importable outside a running simulator (Isaac Sim bundles it into the app, not
    the venv), so this scans bytes. That works for text `usda`, and NOT for binary `usdc`, whose
    token table is compressed: a byte scan of a usdc finds nothing and would report
    `runtime_cooked: False` for an asset cooked at 128 hulls. Reporting "fine" for an asset you
    cannot read is worse than reporting nothing, so binary returns UNKNOWN and says where the
    answer actually lives — S1's C8, which runs inside the sim where pxr exists.
    """
    raw = Path(path).read_bytes()
    if raw[:8] == b"PXR-USDC":
        return {"format": "usd", "encoding": "usdc-binary", "runtime_cooked": None,
                "approximation_tokens": None,
                "unreadable": "binary usdc — its token table is compressed, so approximation "
                              "cannot be read without pxr. Run S1's C8 (inside the sim) for the "
                              "real answer, or `usdcat asset.usd -o asset.usda` and re-inspect."}
    counts = {tok: raw.count(tok.encode()) for tok in
              ("convexDecomposition", "convexHull", "boundingCube", "sdf", "meshSimplification")}
    return {"format": "usd", "encoding": "usda-text",
            "approximation_tokens": {k: v for k, v in counts.items() if v},
            "runtime_cooked": counts["convexDecomposition"] > 0}


def inspect(path, max_hulls=MAX_HULLS):
    p = Path(path)
    suffix = p.suffix.lower()
    out = {"input": str(p), "exists": p.is_file()}
    if not p.is_file():
        return out

    if suffix in USD_SUFFIXES:
        out.update(inspect_usd(p))
    elif suffix in XML_SUFFIXES:
        out.update(inspect_xml(p))
    elif suffix in MESH_SUFFIXES:
        out.update({"format": "mesh", "runtime_cooked": None})
    else:
        out.update({"format": "unknown"})

    man = manifest_path(p)
    if man.is_file():
        try:
            data = json.loads(man.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            out["manifest"] = {"path": str(man), "error": str(exc)}
        else:
            out["manifest"] = {"path": str(man), "hulls": data.get("hulls"),
                               "problems": validate_manifest(data, max_hulls)}
    else:
        out["manifest"] = None

    problems = []
    if out.get("unreadable"):
        problems.append(out["unreadable"])
    if out.get("runtime_cooked"):
        problems.append("asset carries `convexDecomposition` — the SIMULATOR cooks this at "
                        "load time, with a hull budget you do not control. Re-decompose the "
                        "source mesh with this script and spawn the parts as convexHull.")
    if out["manifest"] is None and out.get("format") in ("usd", "mesh", "mjcf"):
        problems.append(f"no {MANIFEST_SUFFIX} beside the asset — nothing records what "
                        f"produced its collision geometry")
    elif out.get("manifest", {}) and out["manifest"].get("problems"):
        problems += out["manifest"]["problems"]
    out["problems"] = problems
    out["ok"] = not problems
    return out


# ---------------------------------------------------------------- decompose

def decompose(source, out_dir, max_hulls=MAX_HULLS, threshold=0.05, preprocess_resolution=50):
    try:
        import coacd
        import trimesh
    except ImportError as exc:
        # Returned, not raised: SystemExit(str) writes to stderr, and every caller of this
        # tool parses stdout. A message nobody can parse is a message nobody acts on.
        return {"ok": False, "error": f"{exc.name} is not installed — {INSTALL_HINT}"}

    mesh = trimesh.load(str(source), force="mesh")
    parts = coacd.run_coacd(
        coacd.Mesh(mesh.vertices, mesh.faces),
        max_convex_hull=max_hulls,
        threshold=threshold,
        preprocess_resolution=preprocess_resolution,
    )

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for i, (verts, faces) in enumerate(parts):
        part = out_dir / f"part_{i:03d}.obj"
        trimesh.Trimesh(vertices=verts, faces=faces).export(str(part))
        written.append(part)

    params = {"max_convex_hull": max_hulls, "threshold": threshold,
              "preprocess_resolution": preprocess_resolution}
    data = build_manifest(source, written, params, getattr(coacd, "__version__", None))
    man = manifest_path(source)
    man.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return {"ok": not validate_manifest(data, max_hulls), "manifest": str(man),
            "out_dir": str(out_dir), "hulls": data["hulls"],
            "parts": [str(p) for p in written],
            "problems": validate_manifest(data, max_hulls)}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--input", required=True, help="mesh to decompose, or asset to --inspect")
    ap.add_argument("--out-dir", help="where the part meshes go (decompose mode)")
    ap.add_argument("--max-hulls", type=int, default=MAX_HULLS,
                    help=f"hull budget (default {MAX_HULLS})")
    ap.add_argument("--threshold", type=float, default=0.05,
                    help="CoACD concavity threshold; RAISE it to get fewer hulls")
    ap.add_argument("--preprocess-resolution", type=int, default=50)
    ap.add_argument("--inspect", action="store_true",
                    help="report an existing asset's collision provenance, change nothing")
    a = ap.parse_args()

    src = Path(a.input)
    if not src.is_file():
        print(json.dumps({"ok": False, "error": f"no such file: {src}"}))
        return 2

    if a.inspect:
        print(json.dumps(inspect(src, a.max_hulls), indent=2))
        return 0

    if not a.out_dir:
        print(json.dumps({"ok": False, "error": "--out-dir is required when decomposing"}))
        return 2
    if src.suffix.lower() not in MESH_SUFFIXES:
        print(json.dumps({"ok": False,
                          "error": f"{src.suffix} is not a mesh; use --inspect on it first, "
                                   f"then decompose the SOURCE mesh it was built from"}))
        return 2

    result = decompose(src, a.out_dir, a.max_hulls, a.threshold, a.preprocess_resolution)
    print(json.dumps(result, indent=2))
    # exit 0 = the tool RAN (a hull-budget failure is a finding); non-zero = it could not run.
    return 2 if "error" in result else 0


if __name__ == "__main__":
    sys.exit(main())
