# Tuning rewards

Reward design is a search, and HARBOR runs it as one — with real training runs as the fitness function. A candidate is not scored by asking a model whether the reward looks reasonable; it is scored by training a policy on it and measuring the success rate.

```text
/harbor:reward-tune task=<TaskID> algorithm=ppo pool_size=4 gpus=4
```

The loop runs until `success_rate ≥ success_threshold`, then promotes the winning design onto the source task and re-verifies it there.

It is an **async fixed pool**, not a sequence of rounds. `pool_size` candidates stay in flight at once; when one finishes, its verdict comes back and a replacement is designed against everything known so far, rather than waiting for a whole generation to complete. `on_success=cancel` stops the in-flight candidates as soon as one clears the threshold; `on_success=drain` lets them finish, which is what you want when you would rather compare several winners than take the first.

## How the search is structured

Two agents, with a boundary between them that exists for a specific reason.

**`reward-tuning-agent` designs and decides.** It reads the §1–§5 analysis first — the failure modes, what the action space can express, the start layout, the subgoal decomposition that *is* the term ladder, and the degenerate states that form the reward-hacking surface. Then it proposes each candidate: a bounded task delta plus a complete reward with concrete weights, gates, and a composer. It never writes reward code.

**`reward-candidate-agent` implements and scores.** One candidate end to end: implement it into its own isolated task, run the smokes for every section it touched, train, render, score the per-term curves and rendered frames, and return a compact verdict.

That split is what keeps the design context clean. Every traceback, log tail, and failed smoke stays on the candidate's side; the agent choosing what to try next sees verdicts, not noise.

This is the one place in HARBOR where an agent dispatches another agent, and it is the reason the two-level cap exists at all. Depth two buys the isolation; depth three would make the search impossible to cancel coherently.

Candidates are designed to be **distinct from what is already running**, not just from what has finished. With four in flight, proposing a fifth that duplicates an in-flight hypothesis wastes a GPU for the length of a training run — so the designer reasons about the pool's live contents, not only its history.

## Why candidates change the task, not just the reward

A candidate is a *bounded §1–§5 task delta plus a complete reward*, because a reward can only key on what the observation exposes and can only be achieved by what the action space can express. A search restricted to reweighting terms cannot escape a task whose observation is missing the quantity the reward needs. So the search covers both, and any section a candidate touches must re-pass that section's own smoke.

## Isolation follows the pool

| `pool_size` | Isolation |
|:--:|---|
| 1 | Sequential, editing the source task over a `base/` snapshot |
| > 1 | One slot clone per candidate, each with its own GPU |

In local mode the effective pool is capped by `gpus`. In cluster mode candidates are submitted to SLURM and the pool is bounded by your allocation.

## What a candidate reports

The verdict is the only channel between the two agents, and it carries both halves of the evidence. The numbers come from the scorer; what the policy actually *did* comes from the candidate agent watching its own rollout.

That second half is a checklist rather than a paragraph — which stage of the term ladder was reached, where the frames cluster in time, whether a term is being farmed instead of progressed, whether the scene stayed physically valid, whether termination fired as intended, and which checkpoint was watched. The scorer validates that every aspect was answered and refuses placeholder answers, because an aspect nobody addressed reads exactly like an aspect that was fine.

The physical-validity answer routes differently from all the others: penetration and sinking are §1–§3 defects, while everything else points at the reward. Getting that one wrong sends the next iteration to repair the wrong layer.

## Scoring

`scripts/reward-tuning-agent/score_iter.py` holds the single success-rate formula, so no two callers can compute it differently. It also refuses to invent a number: a run with no `metrics.jsonl` returns `success_rate: null` with a gate reason like `no_metrics`, rather than a fabricated zero. An ungradable candidate and a failed candidate lead to different next moves, so they must not look alike.

Candidates also report `task_smoke_failed` and `reward_smoke_failed` distinctly — the first means the task delta was wrong, the second means the reward was. Opposite fixes.

The scorer reports a **peak** alongside the final numbers, because runs routinely climb and then collapse. Scoring only the end silently grades a policy the reward did not produce at its best: one observed candidate peaked at 772 and was scored at 158. When a run peaked, the rendered checkpoint is the best one rather than the last, and the verdict has to say which was watched.

An optional mid-run monitor can stop an unambiguously doomed candidate early, but it is off by default and deliberately hard to trigger — a hard failure like NaN acts immediately, while a soft concern needs two consecutive ticks. Early RL curves are noisy and a dip at 20–40% of budget routinely recovers. A wasted run costs compute; a wrongly killed good candidate costs an iteration *and* misleads the search.

## Per-term visibility

Tuning a reward you cannot decompose is guesswork. HARBOR wires per-term logging so every term is visible in the metrics stream, and asserts the decomposition holds every step:

```python
composer(info["detailed_reward"].values()) == env_reward
```

where `composer` is `sum` or `product` per task. If that assertion ever fails, the terms you are reading are not the reward the policy is optimizing — so it is a hard failure, not a warning.

You can add this to any benchmark without changing its native reward:

```text
/harbor:reward-add-log
```

## Experience

After each round, recurring patterns — effective term shapes, unstable ranges, common failure modes — are distilled into an append-only ledger. Later runs retrieve entries filtered by simulator, task, and algorithm before designing. In the paper's measurement this cut a reward redesign on the hardest task from four hours to thirty minutes.

Entries marked `[MUST]` are binding on their reader. For example, magnitude-budget discipline: terms are assigned a share of a fixed total, so no single term can silently dominate the ladder.

## Promotion

Clearing the threshold is not the end. The winning design is **promoted onto the source task** — the clone was scaffolding — and then re-verified with that winner's own smoke set, because a design that passed inside its slot clone has not yet been shown to pass where it will live.

## Next

[Train and tune →](/guide/training)
