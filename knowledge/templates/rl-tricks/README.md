# rl-tricks — the trick library

One directory per trick, read by `/harbor:rl-add-trick` and listed by
`/harbor:rl-list-tricks`. Each holds:

```
<trick>/
├── manifest.yaml     name · description · algorithms · backend · references · caveats
├── patches.yaml      file_patches[] (literal find/replace edits) + config_patches[]
├── smoke.py          the check that proves the trick is actually in effect
└── edits/            the raw text of every find/replace pair
```

`patches.yaml` is the index: each `file_patches[].file` names the rendered target in the
benchmark repo (`{slug}` resolves from `rl-suite-spec.json`), and each `edits[]` entry points
at the two files holding its `find` and `replace` text. `apply_trick.py` reads them verbatim
and does a literal single-occurrence replace, in the order `patches.yaml` lists them — the
`NN` in a fragment's name is there so a human reading `edits/` sees the same order, not
because anything sorts on it.

Edits are idempotent by construction: once applied, the `find` text no longer matches and the
post-trick text is detected instead, so a re-run reports `[noop]` rather than double-applying.
A `find` block that matches more than once is refused as ambiguous — tighten its context
rather than relying on which one comes first.

## `edits/ppo.py` is a directory, not a module

The fragments for a target file live in a directory named after that file, so
`edits/ppo.py/ppo_01_pass_use_obs_rms.{find,replace}` reads as "the first edit to `ppo.py`".
It keeps the layout legible, and it is the shape `patches.yaml` already points at.

The cost is that a plain `**/*.py` sweep walks into it and hits `IsADirectoryError` instead of
finding source. Anything traversing the tree — a linter, a packaging step, a coverage run —
has to filter on `Path.is_file()`. The plugin's own template-compile test does; a new tool
will not unless you make it.
