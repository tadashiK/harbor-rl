#!/usr/bin/env python3
"""Build a USD whose collision is the CoACD parts, one convexHull prim each.

This is the middle of the pipeline. `coacd_decompose.py` produces the parts and the manifest;
S1's C8 checks what the stage actually loaded; this is what connects them, and without it the
decomposition never reaches the simulator.

The obvious shortcut does not work and fails *quietly*: concatenating the parts into one
multi-object OBJ and handing it to IsaacLab's `MeshConverter` merges them into a single mesh,
which PhysX then wraps in ONE convex hull — filling in the very concavity the decomposition was
run to preserve. That satisfies a manifest check, a hull-budget check and a no-runtime-cooker
check simultaneously, which is why C8 also compares the manifest's part count against the
collider prims on the stage.

So the parts are authored directly:

    /<Name>                      Xform, RigidBodyAPI (+ kinematic) + MassAPI
      /visual                    the original mesh — rendered, no collider
      /collisions/part_NNN       one Mesh per CoACD part, each CollisionAPI +
                                 MeshCollisionAPI(approximation="convexHull")

Each part is already convex, so `convexHull` adds no approximation error — it is a statement of
fact about that part, not a request for the simulator to approximate anything.

Needs a running simulator app for `pxr`; run it with the benchmark's `.venv/bin/python`.
"""
import argparse
import json
import sys
from pathlib import Path

# Kit's shutdown can kill the process before stdout flushes — the same trap the smoke templates
# guard against. The result is ALSO written to a file, because a verdict that only exists on a
# stream the simulator can truncate is a verdict that sometimes does not exist.
sys.stdout.reconfigure(line_buffering=True)


def _load_obj(path):
    """Minimal OBJ reader — verts + triangulated faces. Avoids a trimesh dependency here."""
    verts, faces = [], []
    for line in Path(path).read_text().splitlines():
        if line.startswith("v "):
            verts.append([float(x) for x in line.split()[1:4]])
        elif line.startswith("f "):
            idx = [int(tok.split("/")[0]) - 1 for tok in line.split()[1:]]
            for k in range(1, len(idx) - 1):
                faces.append([idx[0], idx[k], idx[k + 1]])
    return verts, faces


def build(parts_dir, source_mesh, out_usd, name, mass, kinematic):
    # Validate BEFORE importing pxr: bad input should be reportable without a simulator.
    parts = sorted(Path(parts_dir).glob("part_*.obj"))
    if not parts:
        return {"ok": False, "error": f"no part_*.obj in {parts_dir}"}

    from pxr import Usd, UsdGeom, UsdPhysics, Gf   # noqa: E402  (needs the app)

    stage = Usd.Stage.CreateNew(str(out_usd))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)

    root = UsdGeom.Xform.Define(stage, f"/{name}")
    stage.SetDefaultPrim(root.GetPrim())

    rb = UsdPhysics.RigidBodyAPI.Apply(root.GetPrim())
    if kinematic:
        rb.CreateKinematicEnabledAttr(True)
    UsdPhysics.MassAPI.Apply(root.GetPrim()).CreateMassAttr(float(mass))

    def _mesh(path, prim_path, collider):
        verts, faces = _load_obj(path)
        m = UsdGeom.Mesh.Define(stage, prim_path)
        m.CreatePointsAttr([Gf.Vec3f(*v) for v in verts])
        m.CreateFaceVertexIndicesAttr([i for f in faces for i in f])
        m.CreateFaceVertexCountsAttr([3] * len(faces))
        if collider:
            UsdPhysics.CollisionAPI.Apply(m.GetPrim())
            # Each CoACD part IS convex, so this asserts a fact rather than requesting an
            # approximation. It is also what keeps the simulator's own cooker out of the loop.
            UsdPhysics.MeshCollisionAPI.Apply(m.GetPrim()).CreateApproximationAttr("convexHull")
        else:
            m.CreatePurposeAttr(UsdGeom.Tokens.render)
        return len(faces)

    _mesh(source_mesh, f"/{name}/visual", collider=False)
    UsdGeom.Scope.Define(stage, f"/{name}/collisions")
    for i, p in enumerate(parts):
        _mesh(p, f"/{name}/collisions/part_{i:03d}", collider=True)

    stage.GetRootLayer().Save()

    # Verify what was written, from the file — not from the objects still in memory.
    check = Usd.Stage.Open(str(out_usd))
    colliders, approximations = 0, set()
    for prim in check.Traverse():
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            colliders += 1
            if prim.HasAPI(UsdPhysics.MeshCollisionAPI):
                approximations.add(
                    UsdPhysics.MeshCollisionAPI(prim).GetApproximationAttr().Get())

    problems = []
    if colliders != len(parts):
        problems.append(f"wrote {len(parts)} parts but the saved stage has {colliders} "
                        f"collider prim(s)")
    if approximations - {"convexHull"}:
        problems.append(f"unexpected approximation(s) {sorted(approximations - {'convexHull'})}")
    return {"ok": not problems, "usd": str(out_usd), "parts": len(parts),
            "collider_prims": colliders, "approximations": sorted(approximations),
            "problems": problems}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--parts-dir", required=True, help="dir of part_*.obj from coacd_decompose")
    ap.add_argument("--source-mesh", required=True, help="original mesh, used as the visual")
    ap.add_argument("--out", required=True, help="USD to write")
    ap.add_argument("--name", required=True, help="root prim name, e.g. Mug")
    ap.add_argument("--mass", type=float, default=1.0)
    ap.add_argument("--kinematic", action="store_true")
    a = ap.parse_args()

    for p in (a.parts_dir, a.source_mesh):
        if not Path(p).exists():
            print(json.dumps({"ok": False, "error": f"no such path: {p}"}))
            return 2

    from isaaclab.app import AppLauncher
    parser = argparse.ArgumentParser()
    AppLauncher.add_app_launcher_args(parser)
    app_args = parser.parse_args([])
    app_args.headless = True
    app = AppLauncher(app_args).app

    out = Path(a.out)
    if out.exists():
        out.unlink()                      # Usd.Stage.CreateNew refuses to overwrite
    result = build(a.parts_dir, a.source_mesh, out, a.name, a.mass, a.kinematic)
    receipt = out.with_suffix(".build.json")
    receipt.write_text(json.dumps(result, indent=2), encoding="utf-8")
    result["receipt"] = str(receipt)
    print("BUILD_RESULT " + json.dumps(result), flush=True)
    app.close()
    return 0 if result.get("ok") else 1


if __name__ == "__main__":
    sys.exit(main())
