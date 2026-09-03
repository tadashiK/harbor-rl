"""Contract: one camera angle, shared by every render harbor produces.

Renders are compared BY EYE — the S6 keyframe judgement, and a reward candidate's rollout
against its siblings. That comparison only means something if the camera stood in the same
place. Two tasks shot from different angles look different when they are not; two candidates
of the SAME task shot differently make a behavior change indistinguishable from a framing
change.

The framing is angle AND resolution: three benchmarks of one task at 512x512 / 640x480 /
1280x720 read as three different things even shot from the same point, so both are pinned.

The angle is defined in render.py (trained-policy videos) and mirrored in the S6 render smoke.
They live in different template trees and cannot import each other, so this asserts the
numbers agree. It exists because a refactor once turned the smoke's camera into a free-form
agent-filled slot, and every agent duly invented its own angle.
"""
import re

from _pluginmeta import ROOT

RENDER_PY = (ROOT / "knowledge/templates/rl-integration-generator/custom_torch/scripts"
             / "render.py.template")
SMOKE_S6 = ROOT / "knowledge/templates/task-generator/smokes/smoke_s6_render.py.template"


def _triple(text, pattern):
    m = re.search(pattern, text)
    assert m, f"camera definition not found for pattern {pattern!r}"
    return tuple(round(float(v), 6) for v in re.findall(r"-?\d+\.?\d*", m.group(1)))


def test_render_and_smoke_share_one_camera():
    render = RENDER_PY.read_text(encoding="utf-8")
    smoke = SMOKE_S6.read_text(encoding="utf-8")

    render_eye = _triple(render, r'"viewer_eye"\s*,\s*(\[[^\]]+\])')
    render_target = _triple(render, r'"viewer_target"\s*,\s*(\[[^\]]+\])')
    smoke_eye = _triple(smoke, r'cfg\.viewer\.eye\s*=\s*(\([^)]+\))')
    smoke_lookat = _triple(smoke, r'cfg\.viewer\.lookat\s*=\s*(\([^)]+\))')

    assert smoke_eye == render_eye, (
        f"S6 smoke eye {smoke_eye} != render.py viewer_eye {render_eye}. Every harbor render "
        f"must share one angle, or keyframes stop being comparable across tasks and across "
        f"candidates of the same task."
    )
    assert smoke_lookat == render_target, (
        f"S6 smoke lookat {smoke_lookat} != render.py viewer_target {render_target}."
    )

    render_res = _triple(render, r'"render_resolution"\s*,\s*(\([^)]+\))')
    smoke_res = _triple(smoke, r'cfg\.viewer\.resolution\s*=\s*(\([^)]+\))')
    assert smoke_res == render_res, (
        f"S6 smoke resolution {smoke_res} != render.py render_resolution {render_res}. "
        f"Resolution is part of the framing: the same task rendered at different sizes is "
        f"not visually comparable, which is what these renders are FOR."
    )


def test_smoke_camera_is_not_an_agent_filled_slot():
    """The framing must be concrete in the template. A required free-form slot is how the
    per-task drift got introduced — each agent picks its own numbers."""
    smoke = SMOKE_S6.read_text(encoding="utf-8")
    body = smoke.split('"""', 2)[-1]        # skip the docstring's slot listing
    for line in ("cfg.viewer.eye", "cfg.viewer.lookat", "cfg.viewer.resolution"):
        assert re.search(rf"^{re.escape(line)}\s*=\s*\(", body, re.M), (
            f"{line} is not concretely set in the template body"
        )
    assert "{{VIEWER_BLOCK}}" not in smoke, (
        "VIEWER_BLOCK was the required free-form slot that caused the drift; the shared "
        "framing is now concrete and only VIEWER_OVERRIDE_BLOCK (optional) remains"
    )


# ---------------------------------------------------------------- ManiSkill follow-cam rig
#
# Regression guard for a defect that silently affected EVERY GPU-sim ManiSkill render.
#
# Under `sim_backend=physx_cuda`, ManiSkill renders human cameras through a sapien
# RenderCameraGroup whose pose is `gpu_pose[cam.gpu_pose_batch_index] * cam.local_pose`.
# `local_pose` is read once, at group-creation time, and the group is built exactly once behind
# `MSScene._human_render_cameras_initialized`. So `RenderCamera.set_local_pose()` is accepted,
# reads back correctly, and never moves a pixel. render.py called it, got no exception, and
# reported success — so the follow-cam no-opped for every ManiSkill task and a walking robot
# simply left frame. Harmless for fixed-base tabletop tasks, which is why it went unnoticed.
#
# The fix binds the camera to the GPU pose row of a 1 mm kinematic `cam_rig` body, which IS
# re-read every frame. These tests pin the three things that make that work, because each of
# them fails SILENTLY rather than loudly.


def test_maniskill_cam_rig_is_installed_before_the_env_is_built():
    """The rig body must be added in `_load_scene`, so the patch has to precede env creation."""
    src = RENDER_PY.read_text()
    assert "def _maniskill_install_cam_rig" in src, (
        "render.py.template lost the ManiSkill camera-rig installer; the follow-cam will "
        "silently no-op on physx_cuda"
    )
    # Match the CALL SITE, not the `def` line — an unanchored search finds the definition,
    # which is always near the top of the file and makes this comparison vacuous.
    call = re.search(r"^\s+_maniskill_install_cam_rig\(cfg\)", src, re.M)
    assert call, "no call to _maniskill_install_cam_rig(cfg) in main()"
    install = call.start()
    build = src.index("env = create_render_env(cfg)")
    assert install < build, (
        "_maniskill_install_cam_rig(cfg) must be called BEFORE create_render_env(cfg) — the rig "
        "body is added in _load_scene, so patching after the scene is built does nothing"
    )


def test_maniskill_cam_rig_bind_forces_a_render_first():
    """The camera group is created lazily on the first render; binding before it is a no-op."""
    src = RENDER_PY.read_text()
    body = src[src.index("def _maniskill_bind_cam_rig"):]
    body = body[: body.index("\ndef ", 1)]
    assert "u.render()" in body, (
        "_maniskill_bind_cam_rig must force one render before rebinding — the render system "
        "group does not exist until the first render, so an early bind silently returns False"
    )
    assert "set_gpu_pose_batch_index" in body and "create_camera_group" in body, (
        "the bind must point the camera at the rig's GPU pose row AND rebuild the camera group"
    )


def test_set_viewer_camera_does_not_claim_success_on_the_noop_path():
    """A camera setter that lies is worse than one that fails: the caller cannot tell."""
    src = RENDER_PY.read_text()
    body = src[src.index("def _set_viewer_camera"):]
    body = body[: body.index("\ndef ", 1)]
    setter_at = body.index("setter(pose)")
    after = body[setter_at:]
    assert "return True" not in after.split("except")[0], (
        "_set_viewer_camera must not unconditionally `return True` after set_local_pose(): on "
        "physx_cuda that call changes no pixels, and reporting success is what hid the broken "
        "follow-cam. Return the rig result, or gate on the backend."
    )
