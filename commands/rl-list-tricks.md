---
description: List all available RL training tricks (in-network obs RMS, reward normalization, etc.) with descriptions, applicable algorithms, and references. Use when the user types /harbor:rl-list-tricks or asks "list tricks", "show available rl tricks", "what tricks can I apply", "what tricks are available".
argument-hint: "(no args)"
---

# /harbor:rl-list-tricks — List Available RL Tricks

Each trick is a small, well-scoped algorithmic enhancement that can be enabled
or disabled per-algorithm via a config flag. Tricks live under
`${CLAUDE_PLUGIN_ROOT}/knowledge/templates/rl-tricks/<name>/manifest.yaml` and are applied
to a benchmark repo via `/harbor:rl-add-trick <name> [algorithm=<algo>]`.

The default `custom_jax` implementation does NOT enable any tricks — apply them
explicitly when you want them. The two starter tricks (`obs_rms_jax` and
`reward_norm_jax`) together close the gap to loco-mujoco's reference PPOJax on
shaped-reward locomotion tasks.

## Action

Run the trick lister and print its output:

```bash
python3 "${CLAUDE_PLUGIN_ROOT}/scripts/rl-tricks/list_tricks.py"
```

If the user passed an argument that looks like an algorithm name (`ppo`, `sac`,
`td3`), filter:

```bash
python3 "${CLAUDE_PLUGIN_ROOT}/scripts/rl-tricks/list_tricks.py" --algo <name>
```

The script reads every `manifest.yaml` under `knowledge/templates/rl-tricks/`, prints
name + backend + algorithms + first-line summary for each. Print the script's
output verbatim, then suggest the apply command:

> Use `/harbor:rl-add-trick <name> [algorithm=<algo>]` to apply one to the
> current repo's RL configs.

## Constraints

- **Don't paraphrase the descriptions** — they may include caveats (off-policy
  warnings, sparse-reward warnings) that the user needs verbatim.
- **Don't apply** anything from this command — `/harbor:rl-list-tricks` is read-only.
- If the trick directory is empty / missing, say so plainly; don't make up
  trick names.
