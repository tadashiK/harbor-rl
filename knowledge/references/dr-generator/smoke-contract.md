# §7 smoke contract

Loaded by `dr-generator`. Template at `${CLAUDE_PLUGIN_ROOT}/knowledge/templates/dr-generator/smokes/smoke_s7.py.template`.

## What it verifies

Build **one** env at `num_envs=16` with every effective group-1/2 DR term's range collapsed to a
**point interval `[k, k]`** (uniform sample → exactly `k`). Reset, then read each randomized property
straight off the physics view and assert it equals the original modified by `k`:

| Mode | `operation` | Assert |
|---|---|---|
| multiplicative | `scale` | `prop ≈ default * k` |
| additive | `add` | `prop ≈ default + k` |
| direct | `abs` | `prop ≈ k` |

Each physical check runs **twice** — after a first reset and after a second reset (with steps in
between) — to prove the randomization re-applies **once per episode per env** (`mode="reset"`).

Group-3 obs-noise terms are verified separately: build a paired **no-noise** cfg (`noise=None` on every
obs term), roll out both, assert the noisy obs diverge from the clean obs and (gaussian) that the
empirical std over `16 envs × steps` is within ~0.3×–3× of the configured `std`.

This is an **exact-value** contract, not the old "DR ON vs OFF diverges" heuristic — a no-op or
mis-scaled term now fails loudly instead of passing on incidental divergence.

## Pass criterion

Script exits 0, no `Traceback`/`AssertionError`. `results` is non-empty (at least one effective term
was checked), every entry is OK, final stdout line reads `S7 OK: ...`.

## Num envs

`num_envs=16`. Point intervals make the sampled value deterministic, so a single pass per reset is
enough — no statistics needed for the physical checks. The 16 envs catch per-env indexing bugs (a term
that randomizes env 0 only, or broadcasts one env's value to all).

## Substitutions

| Slot | Notes |
|---|---|
| `{{TASK_ID}}` | gym task id |
| `{{POINT_OVERRIDES}}` | one line per effective group-1/2 term collapsing its range to `[k, k]` on `cfg`, recording `k`. Reset terms (`reset_*`) are §3 — never touched. |
| `{{PHYS_CHECKS}}` | body of `check_physical()` — read each property off the physics view, compare to `default`-modified-by-`k`, append `(name, ok, detail)` to `results`. Recipes in `isaaclab-dr-reference` → "Read-back recipes". |
| `{{OBS_NOISE_CHECK}}` | optional top-level block: paired no-noise rollout + divergence/std verdicts appended to `results`. Empty if no group-3 terms. |

## When the contract fails

| Symptom | Likely cause | First-pass fix |
|---|---|---|
| `prop != default * k` (off by a factor) | wrong `operation` (used `add` where `scale` meant) or read wrong default buffer | match `operation` to the asserted mode; read `data.default_*` not the live buffer |
| value correct env 0, wrong elsewhere | term randomizes/broadcasts a single env | pass `env_ids=None` / index with `env_ids[:, None]`; re-check at 16 envs |
| `after_reset_2` differs from `after_reset_1` for a `scale` term | randomization stacking on already-randomized values | confirm the term re-applies on `default_*`, not the previous sample (built-in funcs do this) |
| obs-noise term shows no divergence | `noise=` not set, or obs term inactive | attach `Gnoise`/`Unoise` to the active term; confirm it's in `observation_manager.active_terms` |
| Build/`assert_close` error on material | bucketing vs point interval | point interval makes all buckets equal `k`; if still off, set `num_buckets` ≥ 1 and re-check |
| Tracebacks during reset | range too aggressive (mass→0, friction<0) | shrink the real range; point interval `k` must stay > 0 for scaled quantities |

If the canonical task fails the same smoke, `task-implementation.md` is stale — patch surgically, log in
`dr-history.md`, retry.

## When DR is skipped

The agent skips §7 entirely (and never renders this template) only if all three hold:

1. User description does not request randomization, robustness, sim-to-real, or noise, AND gives no DR-term overrides.
2. §7 reference task in `task-implementation.md` has no DR wired (`<unsupported>` or zero terms).
3. Family default = no-DR (e.g. `dm_control`, `gymnasium-generic`).

`status: skipped` is a valid success outcome.
