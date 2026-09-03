# Reward Tuning Experience

Cross-run heuristics from human-in-the-loop reward tuning. These are **heuristics, not rules** —
weigh them against the task at hand. They are **subordinate to a matched task-library base spec**
(see `knowledge/references/adapt-first.md` precedence): library tasks are proven successful, and a
heuristic is never a reason to modify a proven base design. **Each entry is numbered for stable
cross-reference; never renumber existing entries — only append.**

1. **Stage long-horizon rewards.** Decompose into stages (reach → grasp → lift → align → place) and gate each later stage on a prior-stage-complete predicate, so the policy learns one behaviour at a time.

2. **Plan a global per-stage magnitude budget before tuning any single weight.** Per-step income at saturation should strictly increase from earlier to later stages (e.g. reach 2–3 → lift 5–10 → align 10–30 → place 30–100); sparse landmarks sit an order of magnitude above the dense steady-state episode sum; regularizers stay at |weight| ≤ 0.1. Write the budget into the §6 docstring so the global balance is auditable.

3. **Keep the reward as simple as the task allows.** Every extra term is another local optimum; add terms only on a clean diagnosis of a specific bootstrap failure.

4. **Diagnose per term, not per total.** After a run, read each stage's episodic return separately; the stage with the lowest mean-to-weight ratio is usually the blocker, and fixing its signal beats re-weighting unrelated terms.

5. **Mux shared structure.** When sub-tasks are the same operation on different operands, expose one shared observation interface switched by a stage predicate, so the policy learns the operation once.

6. **Simplest controller, clamped workspace.** Use the lowest-DOF action mode the task allows and clamp the action's reachable workspace to the task's actual extent plus a small margin.

7. **Gate horizontal attractors on obstacle clearance.** When a payload must pass over a vertical obstacle, gate the horizontal-attractor reward on the payload clearing the obstacle's top (plus margin), or the policy drags the payload through it.

8. **Monotone income across stage boundaries.** No transition on the task path may reduce per-step income: the next stage's payment must start at the boundary event at a rate ≥ the previous stage's saturation, or the policy learns to avoid the transition. Enumerate per-step income at each waypoint and verify it is non-decreasing.

9. **Never reduce a proven gradient when refining.** Add refinement kernels on top of a working attractor (`coarse + α·fine`); averaging them in halves the long-range gradient.

10. **Re-check farmability when rescaling.** A ladder that works at small absolute scale can produce camping on risk-free terms at a larger scale; a total that declines from mid-run to final is the signature of a converged reward hack.

11. **Only reshape against a converged run.** If per-term curves are still rising at cutoff, extend the training budget before touching the reward — mid-flight verdicts mis-attribute exploration timing to reward structure.

12. **Income must require progress.** Continuity offsets that pay for merely existing in a post-stage state become idling income; break penalties on policy-built structures should be small deterrents, not catastrophic — otherwise next-stage attempts become EV-negative and the policy idles.

13. **Audit the env start state before blaming the reward.** A "stage-0 never fires" per-term log is as often a scene/reset (§1/§3) defect — e.g. a lost init pose — as a §6 one.

16. **A structurally-correct reward that won't train is usually a TRAINER problem, not a reward problem — read the RL curves before reshaping.** When the reward smoke is green, magnitudes are right, and the reward is proven (a reproduced task-library base), and it still scores ~0, inspect `train/entropy` / `approx_kl` / the policy std before touching a single weight. The tell: a skill is *discovered then abandoned* (a gated stage's return rises then decays to 0) while **entropy RISES** — that is a too-strong entropy bonus / growing learnable action-std / too-short rollout horizon washing out a fragile precise skill, not a reward defect. Fix on the RL axis (lower `ent_coef`, freeze/shrink learnable std, lengthen `n_steps`), leave the reward untouched. Corollary: a source repo's "solves in N steps" assumes ITS hyperparameters; the destination's generic PPO config is NOT per-task tuned, so "reward is proven" ≠ "converges under the default trainer." Reshaping a proven reward to compensate for a trainer mismatch makes a correct reward worse.

17. **Verify a gated reward's sensor/signal actually fires before diagnosing the reward.** When a whole ladder hangs off one predicate (contact grasp, height, proximity), a blind gate silently kills every downstream stage. Before concluding the shape is wrong, FORCE the gating event and read the raw signal — e.g. for a contact-gated grasp reward, teleport the object into the closed hand and read `sensor.data.force_matrix_w[:,0,0,:]` (expect newtons on the filtered object). Confirm the gate CAN fire; only then reason about why the policy isn't making it fire.

18. **Adapt-first governs the reward's FORM and SIGNALS, not just its ladder.** Porting a library base means keeping its term shape functions (an unbounded `1/d` attractor, a contact-gated lift) AND the §1–§5 signals they read (fingertip contact sensors, a command manager) — not re-expressing the math as a bounded proxy (`tanh`, palm-proximity) nor dropping the signal. "Express in destination idioms" is syntax only (imports, cfg names, RewTerm/API), never the function's form. If the scene lacks a signal the base needs, ADD it via `design.json:env_changes` (IMPLEMENT wires it into §1–§5), don't proxy around it; `env_changes:[]` when the base needed a sensor is a design bug, and a proxied signal is a common root of a persistent failure (grasp fragility from a proximity gate). Distinct from #17: that assumes the sensor exists and checks it fires; this is about the sensor never being wired at all.

19. **A mis-firing high-magnitude success/placement bonus destroys value learning — verify its firing, don't just trust the predicate.** A large sparse landmark (an in-container / place bonus at weight ~1e3) whose accept test is too loose fires on non-successes — an object still GRASPED and merely held or hovering inside the accept region, or transiting it — because a bare position box has no release/settle gate. Every spurious fire injects a huge return spike that blows up the value target and its normalization, corrupts advantages, and collapses the policy; the signature (return spikes then decays to ~0) masquerades as a grasp/exploration failure and mis-routes the fix to reset curricula or RL-axis tweaks when the real defect is the success predicate + the bonus scale. During tuning: (a) MONITOR the success/placement term's per-step firing and cross-check each fire against the rendered frames — it must correspond to a genuine, settled, released success and be unsatisfiable by a held or passing object; (b) gate the predicate on release / rest / settle (low target z + low object speed, or fingertips-not-in-contact), not mere position, BEFORE giving it a large weight; (c) keep the landmark within the value-normalization budget — even a correctly-gated but enormous bonus wrecks value learning, so prefer a modest weight and/or value normalization over a giant spike.

20. **`ent_coef` is single-peaked, not monotone — too low fails as hard as too high, so sweep it before reshaping.** #16's "entropy rises while a gated stage decays -> lower `ent_coef`" is only the upper half of the curve. Below the peak the policy collapses onto a partial/single-limb strategy and never explores the coordinated behaviour at all, which looks exactly like a reward-structure failure. The high-side failure has a distinct mechanism: a hard `log_std` ceiling means entropy pins at the cap permanently and the policy can only sharpen by growing the actor mean. When a structurally-correct reward will not train, sweep `ent_coef` across an order of magnitude in BOTH directions before touching a single weight.
