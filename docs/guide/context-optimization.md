# Context optimization

An agent that authors a robot task is not a chatbot with tools. It runs for hours, waits on work that takes longer than it does, and produces artifacts far larger than anything it can hold. Those three properties decide what a run costs — far more than how much the model thinks.

One `/harbor:task-create` run — a Franka mug-hanging task on IsaacLab, four agents, five hours — cost **73.1M input-token-equivalents**. Only **4%** of that was the model generating anything. The rest was conversations being re-sent, and caches being rebuilt after they expired.

## Why agentic RL is the hard case

Most agent workloads are call-and-response: a tool returns in a second, the context grows a little, the loop repeats. Reinforcement-learning workflows break all three assumptions at once.

**The work is slower than the agent.** A PPO run is 1–3 hours. A physics smoke test is ~10 minutes. Mesh convex decomposition is minutes. The agent spends most of its wall clock *waiting*, and waiting is not free — an idle gap expires the prompt cache, so the next request re-sends everything at write price.

**The artifacts are enormous and heterogeneous.** One training run here produced a 37,641-line log, per-term reward curves, a rendered MP4, checkpoints, and a metrics stream. Naively, these flow through context. They must not.

**The loop iterates.** Reward tuning is *design → implement → train → score*, repeated until success. Every per-iteration inefficiency multiplies by the number of candidates.

**The reasoning must outlive the agent that did it.** Why the rack peg sits at 25.6° is a §1 decision that a §6 reward designer needs hours later, in a different context. Handoffs are load-bearing.

That gives three distinct cost surfaces, and they fail in different ways:

| surface | question | failure |
|---|---|---|
| **Context management** | what does an agent carry, and for how long? | a reference read once, re-sent 346 times |
| **Artifact management** | where do work products live? | a 12k template routed through context to substitute two strings |
| **Waiting** | how do you bridge work slower than a turn? | an hour of silence, then a full cache rebuild |

## Measuring it

Nothing here is provider-specific. Any LLM API with prompt caching bills four classes of token, and the *ratios* between them are what matter:

| class | typical weight |
|---|---|
| input | 1.0 |
| output | ~5× input |
| cache read | ~0.1× input |
| cache write | ~1.25× input (short TTL), ~2× (long TTL) |

Quote everything in **input-token-equivalents (ITE)** and the analysis survives a price change or a provider switch.

Per-request usage is normally reported alongside each response, and that is all you need:

- **cost by component** — sum output, cache-read and cache-write per agent.
- **context size at request *i*** — `cache_read + cache_creation`. This is the whole cached prefix.
- **what a turn added** — diff that between consecutive requests. Exact, no tokenizer estimate.
- **what a turn *cost*** — its growth × the number of requests that followed it.
- **wait failures** — a large cache-write with a cache-read of zero is a rebuilt cache. The gap before it tells you what you were waiting on.

### Two laws

**The carry law.** A token added to context costs `0.1 × (requests remaining)`. Position matters as much as size: 26.6k read at request 5 of 352 costs 0.92M ITE; the same read at request 300 costs 0.13M.

**The wait law.** An idle gap longer than the cache TTL costs a rebuild at `ctx × 1.25`. A tick inside the TTL costs `ctx × 0.1`. They cross at **12.5 ticks ≈ 50 minutes**.

Every principle below is a corollary of one of these.

The two laws pull in opposite directions, which is why neither alone is a strategy. The carry law says read less; the wait law says do not go quiet. An agent that minimizes reads but sleeps through the cache TTL pays more than one that read too much and kept ticking — and the reverse is equally true. What the numbers below settle is *where* the crossover sits.

## Context management

**Delegate to isolate context, not to parallelize.** HARBOR's `reward-tuning-agent` designs rewards but never implements them:

> implementation noise (tracebacks, smoke retries, log tails, rendered frames) never enters the context that designs the next reward.

The split exists for context hygiene first. It works: the designer costs 2.49M while the candidate it dispatches costs 45.97M. The expensive noise is quarantined where it can be discarded.

**Position is cost.** The five most expensive turns of one agent's 352, ranked by what carrying them cost afterwards:

```
req   5  +26,644 tok  Read design-base spec  [FULL]  → re-read 346× = 9.2M
req   8  +14,501      Read task-implementation.md    → 343× = 5.0M
req  20  +12,870      Read canonical example cfg     → 331× = 4.3M
req  47  +12,319      Read _verdict.py.template      → 304× = 3.7M
req  40  +11,246      Read task-history.md.template  → 311× = 3.5M
```

Reading the *one selected example* end to end is correct — adapt-first authoring requires it. The problem is duration, not fullness: those files were loaded in the first 50 requests and carried through 250 more requests of smoke-debugging that never referenced them again.

**Bound what enters context, never what is captured.** 118 unbounded shell results cost 4.02M ITE — a repo-wide `grep -rn` (8.0k), an `ls` of a populated directory (9.2k). The fix is not truncation: Python puts the real error at the *end* of a traceback, so a blanket `head -50` discards exactly what a retry loop needs.

```bash
cmd > out.log 2>&1; echo "rc=$?"; tail -40 out.log
grep -n "Error\|Traceback" out.log | head -20     # more, on demand
```

Nothing is destroyed; it simply isn't resident.

## Artifact management

**Disk is the medium; context is the channel.** Each tuning iteration writes a self-contained directory — design, smokes, logs, curves, frames, MP4. Only a verdict crosses into context. Sentinel files and verdict JSON carry state between turns while occupying none.

**Distill an artifact for the consumer.** `task-generator` emits `task-analysis.md` beside its full history: the design rationale with every validation table stripped.

> §6's designer and every reward candidate read it instead of the full history: they need why the task is shaped as it is, not per-check smoke evidence, and that is a third of the file they would otherwise carry.

Write for whoever reads next, not for the record.

**Never route a string substitution through context.** Two of the five costliest turns above are templates, read only to replace `{{TASK_ID}}` and write the result back. The agent makes no decision from that text. Seven such turns cost **12.1M** to carry. Render on disk instead — ~200 tokens instead of ~50,000:

```bash
python3 - <<'PY'
src = open("<plugin>/knowledge/templates/.../smoke_s1.py.template").read()
open("<task_dir>/smokes/smoke_s1.py", "w").write(
    src.replace("{{TASK_ID}}", "...").replace("{{NUM_ENVS}}", "2"))
assert "{{" not in open("<task_dir>/smokes/smoke_s1.py").read()
PY
```

Keep reading the template's *docstring* — it defines which slots exist — and assert no placeholder survives, or you have traded tokens for a silent bad-render bug.

## Waiting

Before there was a law, four agents improvised four strategies and landed in two opposite ditches:

| how it waited | requests | dominant cost |
|---|---|---|
| foreground poll, 51 turns | too few | 13.58M cache-**write**, 94% of that agent |
| foreground `tail`, 48 turns | too few | 6.16M cache-**write**, 62% of each candidate |
| 9 blocking ~608 s smokes | too few | 5.93M cache-**write**, 38% of that agent |
| `Read` a background `.output` 841× at ~3 s | far too many | 439M cache-**read**, 96% of that candidate |

The first three go idle past the TTL and pay to rebuild. The fourth never lets the cache expire but re-sends a 433k conversation 1,723 times to watch a file that wasn't changing. Opposite errors, opposite cost columns.

**Tick just under the TTL.** The consequence is counter-intuitive, at a 433k context:

| strategy | polls | cost |
|---|---:|---:|
| every 4 min (cache warm) | 24 | **1.04M** |
| every 20 min (cache cold) | 5 | 2.71M |
| every ~3 s | 1,723 | 43.97M |

**Polling every 4 minutes is 2.6× cheaper than every 20 minutes**, and gives 5× finer early-stop granularity. Below the TTL each poll is a cheap read; above it, every poll pays a rebuild no matter how rare. Backgrounding alone doesn't help — it frees the call slot but leaves the API idle, so the cache still dies. The *tick* is what saves it.

**Liveness is CPU, not output.** Deciding whether a silent job is working or hung cannot use log output — log frequency belongs to the logger, and a backend that loads assets quietly or JIT-compiles emits nothing while working perfectly. Two silent processes, six seconds apart:

| process | log bytes | CPU delta |
|---|---:|---:|
| computing | 0 | **6 s** |
| hung | 0 | **0 s** |

A computing process always accrues CPU; a frozen one never does. Silence is never a kill criterion. Every other threshold — iteration cadence, startup time — should be calibrated from what the run exhibits between two ticks, because any constant is wrong for some simulator.

**Prohibitions don't work; protocols do.** The candidate agent already said *"never a foreground poll loop"* and *"do not inspect a running trial"*. It then polled 1,723 times. Its `monitor_interval` was 300 s — but that parameter was consumed only by a monitor gated behind a flag defaulting to false, so it was dead config, and the default path had **two prohibitions and no sanctioned way to wait**. Told what not to do and not what to do, the agent improvised the worst option available.

All three agents now share one protocol: **launch detached → stamp the start → tick under the TTL → judge liveness by CPU → stop at ~1.5× budget.** One deliberate exception is written down rather than hidden: above the 50-minute crossover a single rebuild is *cheaper* than ticking, and the tuning agent ticks anyway, because the premium buys a wait that survives a lost notification and surfaces a stuck job in minutes rather than hours.

## Case study: hanging a mug on a rack

The task: a Franka arm picks up a mug and hangs it on a rack peg, succeeding only once the mug is on the peg *and* the gripper has let go. Authored §1–§5 by `task-generator` (28/28 checks), reward tuned by `reward-tuning-agent` (converged on the first candidate, `success_rate 0.522`).

**Where the 73.1M went:**

| part | ITE | share | dominated by |
|---|---:|---:|---|
| `reward-candidate-agent` | 45.97M | 62.9% | cache read (94%) |
| `task-generator` | 19.34M | 26.5% | write 38%, read 54% |
| main orchestrator | 5.28M | 7.2% | cache write (51%) |
| `reward-tuning-agent` | 2.49M | 3.4% | cache write (68%) |

**What each agent's context actually held.** `task-generator` grew from 12,344 to 525,707 tokens across 352 requests. Attributing every turn's growth exactly:

| source | turns | added | carry cost |
|---|---:|---:|---:|
| shell output | 118 | 217,617 | 4.02M |
| written files | 22 | 85,443 | 1.95M |
| repo source reads | 19 | 67,178 | 1.72M |
| template reads | 7 | 50,654 | 1.21M |
| plugin reference reads | 5 | 41,478 | 1.20M |

Note what is *absent*: the agent's own instructions are 12,342 tokens — 2% of the peak. Trimming prompts would have saved nothing. The context is made of what the agent did, not what it was told.

**One structural finding.** The API reports which TTL each cache write used. Every subagent wrote to the 5-minute cache (7.8M tokens); the orchestrator wrote to the 1-hour one (2.2M). The agents running ten-minute simulations held the *short* cache while the orchestrator that mostly sits idle held the long one — exactly backwards. Nine of `task-generator`'s twelve cache rebuilds followed gaps of 607–612 s: smokes blocking just past a five-minute TTL.

**After the fixes:**

| | measured | projected |
|---|---:|---:|
| `reward-candidate-agent` | 45.97M | ~4.5M |
| `task-generator` | 19.34M | ~6.1M |
| **run total** | **73.1M** | **~20M** |

Roughly **3.5×**, with no change to what any agent decides, trains, or verifies — the same 28 checks, the same reward, the same policy. The candidate figure comes from replacing 1,723 polls with ~30 ticks; the task-generator figure from ticking its smokes and splitting §1–§5 into per-section contexts so §5 stops re-reading §1's debugging.

The right-hand column is a projection, not a measurement — the polling and tick changes are in, the per-section split is not. And underneath it all sits a floor that no context work touches: 655k output tokens, the cost of actually thinking.

## Where this shows up in the rest of the pipeline

The laws are not advice held separately from the design; most of HARBOR's structural choices are them in another form.

| Mechanism | Which law | What it buys |
|---|---|---|
| `reward-tuning-agent` never writing reward code | carry | Tracebacks and log tails stay in the candidate's context, not the designer's |
| `task-analysis.md` split out of `task-history.md` | carry | Reward design reads the rationale without the validation tables — a third of the file |
| One reference file per task section | carry | An agent authoring actions never loads the observation guide |
| `<smoke>.verdict.json` | carry | A result is re-read as a small object rather than re-derived from the full output |
| Rendering templates on disk | carry | ~200 tokens instead of ~50,000, for text no decision is made from |
| Sentinel files + backgrounded waits | wait | A multi-hour run costs ticks, not a context rebuild per check |
| `tune-state.json` checkpointing | wait | An interrupted loop resumes from disk instead of reconstructing itself |

Read that list in reverse and it is also the answer to "why is this system built out of files": the artifact-centric design in [the harness](/guide/harness) is what makes the cheap option available. You cannot hand an agent a small view of state that only exists in its own transcript.
