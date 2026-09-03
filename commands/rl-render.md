---
description: Render a trained RL checkpoint to MP4 and verify the policy actually moves (frame-difference sanity check). Use when the user types /harbor:rl-render checkpoint=<path> [task=<id>] [key=value ...] or asks "render this checkpoint to a video", "make an MP4 of the trained policy", "show the policy as a video file (not a viewer)". For a headed live viewer use /harbor:rl-visualize instead.
argument-hint: "checkpoint=<path> [task=<id>] [render_max_steps=N] [key=value ...]"
---

# /harbor:rl-render — Render an RL Checkpoint to MP4

Loads a checkpoint, runs the rendered `render.py` (which captures frames and writes `<checkpoint_dir>/render.mp4`), then performs two sanity checks:

1. **Checkpoint loaded + inference produced actions** (not stuck on init / NaN / silent failure).
2. **Frames at different timesteps actually differ** (the rendered policy is not a static image — a common failure when obs normalization isn't restored or the IK target is frozen).

## Required argument

| Arg | Notes |
|---|---|
| `checkpoint` | Absolute or repo-relative path to `<trial_dir>/checkpoint.pth` (custom_torch) / `<trial_dir>/AgentXXX_saved.pkl` (custom_jax / SB3). |

## Optional arguments

| Arg | Default | Effect |
|---|---|---|
| `task` | inferred from `<trial_dir>/resolved_config.yaml` | Task ID. Override for cross-task render. |
| `render_max_steps` | inferred (suite spec default; usually 1000) | Number of env steps to capture. |
| `frame_diff_threshold` | 1.0 | Minimum mean per-pixel L1 between two sampled frames to count as "moving". Lower for very-low-amplitude policies; raise to be stricter. |
| any other `key=value` | — | Forwarded to `render.py` as a Hydra override. |

## Action

1. **Pre-flight**:
   ```bash
   cd "$(pwd)"
   test -f harbor/rl-integration-generator/rl-suite-spec.json   || { echo "no rl-suite-spec.json — run rl-integration-generator first"; exit 1; }
   test -e "<resolved_checkpoint_path>"  || { echo "checkpoint not found"; exit 1; }
   ```

2. **Resolve checkpoint path** to absolute (relative → relative to `$(pwd)`).

3. **Auto-infer algorithm + task** from `<checkpoint_dir>/resolved_config.yaml` (same recipe as `/harbor:rl-eval`). If the user passed `task=`, use their value (cross-task render).

4. **Load suite spec** via the canonical reader:
   ```bash
   eval "$(python3 "${CLAUDE_PLUGIN_ROOT}/scripts/common/resolve_suite.py" --algo <algo>)"
   # → SLUG, SCRIPTS_DIR, PARALLEL, CONFIG_NAME
   ```

5. `CONFIG_NAME` (resolved above) is `<algo>.parallel` when `PARALLEL=true`, else `<algo>`.

6. **Resolve run prefix**: `<repo>/.venv/bin/python`. Error if `.venv/` missing.

7. **Build + run the render command**:
   ```bash
   <prefix> "${SCRIPTS_DIR}/render.py" \
       --config-name=${CONFIG_NAME} \
       task=<task> \
       checkpoint=<abs_checkpoint_path> \
       <user_overrides...>
   ```
   Capture stdout to `/tmp/rl-render-<ts>.log`. Stream live so the user sees `[render] step= N` progress.

8. **Sanity Check #1 — checkpoint loaded + inference produced actions.** The render scripts already print one or more of these on success:
   - `[render] restored obs_rms from checkpoint` (custom_torch w/ obs normalization)
   - `[render] step= N  action.mean|abs|=X.XXXX  action[0]=[...]`
   - `[render] wrote <path>/render.mp4 (<N> frames, return=<R>, action|abs|.mean=X max=Y min=Z)`

   The check passes when **both**:
   - rc=0 from the render command, AND
   - the final `[render] wrote ...` line is present in stdout AND reports `action|abs|.mean > 0` (a strictly-zero action mean usually means inference fell through to a default-init policy because the checkpoint state-dict didn't load).

   If either fails, surface the LAST 30 LINES of stdout/stderr and stop. Do not proceed to Check #2.

9. **Sanity Check #2 — frames at different timesteps differ.** Use ffmpeg to extract 5 frames spread across the MP4 and compare consecutive pairs by mean per-pixel L1 distance. The MP4 is the file the render script just wrote (path is in the `[render] wrote ...` line; default `<checkpoint_dir>/render.mp4`):

   ```bash
   MP4="<checkpoint_dir>/render.mp4"
   <prefix> - <<'PY'
   import subprocess, sys, json, os, tempfile, pathlib
   mp4    = os.environ["MP4"]
   thr    = float(os.environ.get("THR", "1.0"))
   tmp    = pathlib.Path(tempfile.mkdtemp(prefix="rl-render-check-"))
   # Probe frame count via ffprobe.
   nframes = int(subprocess.check_output([
       "ffprobe", "-v", "error", "-select_streams", "v:0",
       "-count_frames", "-show_entries", "stream=nb_read_frames",
       "-of", "csv=p=0", mp4,
   ]).decode().strip())
   if nframes < 5:
       print(f"FAIL: only {nframes} frames in MP4 — render emitted too few steps"); sys.exit(2)
   # Sample 5 evenly-spaced frame indices (skip frame 0 / first ~10% since reset often holds a static frame).
   import numpy as np
   idxs = np.linspace(int(nframes * 0.10), nframes - 1, 5).astype(int).tolist()
   # Dump each one as PNG via ffmpeg select filter.
   for i, fidx in enumerate(idxs):
       subprocess.check_call([
           "ffmpeg", "-y", "-loglevel", "error", "-i", mp4,
           "-vf", f"select=eq(n\\,{fidx})", "-vframes", "1",
           str(tmp / f"f{i:02d}.png"),
       ])
   # Load + compare consecutive pairs.
   from PIL import Image
   imgs = [np.asarray(Image.open(tmp / f"f{i:02d}.png"), dtype=np.float32) for i in range(5)]
   diffs = [float(np.mean(np.abs(imgs[i+1] - imgs[i]))) for i in range(len(imgs)-1)]
   print(f"frame-diff: indices={idxs}, mean_l1={[round(d,3) for d in diffs]}, threshold={thr}")
   # ALL pairs must differ (allow exactly one frozen pair as tolerance for terminal-state holds).
   below = [d for d in diffs if d < thr]
   if len(below) >= 2:
       print(f"FAIL: {len(below)}/{len(diffs)} consecutive frame pairs below threshold — policy appears static"); sys.exit(3)
   print("OK: frames vary across timesteps"); sys.exit(0)
   PY
   ```

   Forward env vars: `MP4=<resolved_mp4_path> THR=<frame_diff_threshold>`.

   - Exit 0 → pass.
   - Exit 2 → MP4 too short (< 5 frames); the render likely crashed mid-rollout.
   - Exit 3 → too many static pairs; the policy is not moving (silently-zero action, frozen IK target, obs_rms not restored, etc.) — same root causes Check #1 catches at the action level.

   On any failure, surface stdout and stop.

10. **On full success**, print a compact summary:
   ```
   render: <checkpoint_dir>/render.mp4
     frames:   <N>
     return:   <R>
     |action|: mean=<X> min=<min> max=<max>
     sanity:   inference OK, frame-diff OK (mean_l1=[d1, d2, d3, d4])
   ```

## Constraints

- **Do NOT background** — the user wants the verification result before they look at the MP4.
- **`render.py` is the source of truth** for inference + obs-normalization restoration; do not re-implement those checks here. This command only assertively verifies the *artifact* (MP4 frames) AFTER `render.py` says it succeeded.
- **Do NOT silently overwrite** existing `render.mp4` files at a different path than the rendered script chose. The script always writes to `<checkpoint_dir>/render.mp4`; trust that path.
- **`ffprobe` / `ffmpeg`** must be available in the host PATH. They ship with the `imageio[ffmpeg]` extras pinned by dependency-generator's harbor-extras block; if missing, surface a one-line `apt install ffmpeg` hint and stop.
- For `algorithm_source == local_implementation`, the shim's `render.py` may not print the canonical `[render] step=` / `[render] wrote` lines. In that case Check #1 falls back to: rc=0 AND `<checkpoint_dir>/render.mp4` exists with size > 0; Check #2 (frame-diff) runs unchanged.
