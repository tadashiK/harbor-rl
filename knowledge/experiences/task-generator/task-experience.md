# task-generator — Cross-run Task-Authoring Experience

Cross-run heuristics for authoring §1–§5 (scene / actions / reset / goal / observation). Read at
Phase 0 of every new task. These are **heuristics, subordinate to a matched task-library base spec**
(see `knowledge/references/adapt-first.md` precedence): library tasks are proven successful. **Each
entry is numbered for stable cross-reference; never renumber existing entries — only append.**

1. **Embodiment swaps must port the base task's init qpos / init pose.** Init pose is task design (it places the EE over the workspace at t=0), and joint values map 1:1 across same-family arms — never inherit the substitute robot cfg's default home pose. Conversely, TCP/ee-frame offsets are embodiment geometry: follow the new robot, keeping the IK `body_offset` equal to the `ee_frame` offset.

2. **A user-supplied asset path is a binding constraint, never a preference.** Wire the asset the user named (URDF assets spawn via the family's URDF-import path); never silently substitute a stock cfg because it is the "proven path". If the asset genuinely cannot be wired, fail loudly with the reason — even in no-questions mode.
