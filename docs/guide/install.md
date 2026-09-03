# Installation

HARBOR installs its one host-side tool itself. Everything else goes into the target repository's own virtual environment, so it never touches your system Python.

## Prerequisites

| Requirement | Why | You install it? |
|---|---|:--:|
| Claude Code ≥ 2.1.219 | Nested agent dispatch, which reward tuning depends on | yes |
| NVIDIA driver | GPU simulation; `nvidia-smi` must print your device | yes |
| [`uv`](https://docs.astral.sh/uv/) | Drives the generated `setup_uv.sh` that builds each repo's `.venv/` | no — auto |
| CUDA toolkit | Only if your simulator builds CUDA extensions | no — warned |

Only the first two are yours to arrange. The generated `setup_uv.sh` installs `uv` to
`~/.local/bin` on first use when the host lacks it, and `dependency-generator` warns rather
than failing when a host has the CUDA runtime but no compiler.

The version floor is a real gate, not advice. Reward tuning is the one place HARBOR dispatches
agents two levels deep — `reward-tuning-agent` designs candidates and dispatches one
`reward-candidate-agent` each — and that requires 2.1.219. `/harbor:reward-tune` asserts it in
pre-flight and stops, because degrading to a serial loop would quietly change what you asked
for into something slower and differently-behaved.

The CUDA row is a warning rather than a requirement because the distinction only matters for
some repositories: a host with the CUDA *runtime* can run most simulators, but packages that
compile extensions at install time (`flash-attn` is the usual one) need the full toolkit.
`dependency-generator` detects a runtime-only host and says so instead of failing a setup that
would have worked.

## Install `uv` yourself (optional)

The bootstrap covers this, but installing ahead of time is harmless and keeps the first run
offline-free:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
exec $SHELL
uv --version && nvidia-smi
```

If `uv` is not found after `exec $SHELL`, log out and back in so the PATH change takes effect.

::: tip Scripted path
Or run the bundled installer, which does the same thing and nothing else:

```bash
git clone https://github.com/supersglzc/harbor-rl.git ~/harbor
~/harbor/scripts/install/install_prerequisites.sh
```

Pass `--skip-uv` if `uv` is already on PATH, or `--help` for details. Every step is idempotent,
so re-running it is safe.
:::

## Install the plugin

Inside Claude Code:

```text
/plugin marketplace add supersglzc/harbor-rl
/plugin install harbor@harbor
```

The marketplace bundles a single plugin named `harbor`. Confirm it loaded:

```text
/harbor:help
```

That prints the full surface — every command, agent, and hook.

## What installing actually adds

The plugin is markdown and scripts, not a service. Three things come with it beyond the
commands:

**Hooks.** `pretool_safety_check.sh` refuses obviously destructive Bash patterns — a recursive
force-delete of `/` or `~` (in any flag spelling, with or without a trailing glob), fork bombs,
`mkfs` on a device — before they reach the shell. `stop_audit_log.sh` appends a one-line audit
entry whenever an agent or subagent finishes, and is **off unless you set `HARBOR_AUDIT_LOG=1`**:
hooks fire in every project you open, not only the benchmark repos you installed HARBOR for, so
it does not record unrelated work to `~/.claude/audit/` without being asked.

**A permission allowlist.** `settings.json` pre-approves the `uv` and `python` invocations the
subagents need in order to run unattended. This removes the prompt, not the safety net — the
pre-tool hook still blocks destructive commands.

**One gated command.** `/harbor:reset-workspace` carries `disable-model-invocation`, so Claude
cannot invoke it on its own initiative. It is the only irreversible operation in the plugin.

## Verify

Run the deterministic test layers, which need no GPU and no simulator:

```text
/harbor:test layers=1,2
```

**Layer 1 (contract)** checks the plugin's own invariants: that dispatch depth never exceeds
two, that exactly one command is gated against model invocation and no other command
slash-invokes it, that every frontmatter block is valid YAML, that every script follows the
tool conventions, and that every task-library entry carries its metadata. **Layer 2 (unit)**
tests the tool scripts directly.

**Layer 3** is a full end-to-end pipeline and does need a GPU and a benchmark repo:

```text
/harbor:test layers=3 repo=<path>
```

It drives the whole `task-create → train → reset` chain module-by-module on an isolated git
**worktree**, so your working tree is never touched, using a benchmark-agnostic fixture — a
robot arm stacking one cube on another — that works against any simulator family. Stages are
fingerprinted Docker-layer style, so a re-run only repeats the stage that changed and
everything downstream of it.

## Troubleshooting

**`uv: command not found` after install.** The installer appends to your shell profile;
`exec $SHELL` only reloads an interactive shell. Log out and back in.

**Reward tuning reports a version error.** Nested dispatch requires Claude Code 2.1.219 or
newer. `/harbor:reward-tune` asserts this in pre-flight and stops rather than silently
degrading to a serial loop.

**`nvidia-smi` prints but training cannot see the GPU.** The repo's `.venv/` may have installed
a CPU-only torch build. Re-run `/harbor:env-install-uv` — the dependency generator's import
smoke checks device availability and will report the mismatch.

**An install step failed and you want to change it.** Do not hand-edit
`harbor/dependency-generator/setup_uv.sh`; it is regenerated on every run. Edit
`install_plan.json` and re-render, which is what makes the fix survive.

**W&B logs nothing from a SLURM job.** Compute nodes usually cannot reach `api.wandb.ai`
directly. The rendered launcher exports `HTTP(S)_PROXY` and `WANDB_API_KEY` from the submitting
environment; set them before `sbatch`, or pass `proxy=` to `/harbor:rl-sweep`. Without a route,
logging falls back to offline mode silently — training completes and the dashboard stays empty.

## Next

[Set up your first benchmark →](/guide/first-benchmark)
