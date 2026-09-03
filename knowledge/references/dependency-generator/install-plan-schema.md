# InstallationPlan schema

`<repo>/harbor/dependency-generator/install_plan.json` is the agent's structured digest of install-relevant instructions from `README.md` and friends. The renderer (`render_uv.py`) consumes it together with `probe.json` to emit the install block in `setup_uv.sh`. If `install_plan.json` is missing the renderer falls back to a heuristic install (uv.lock present → `uv sync --frozen`; else `uv pip install -r requirements.txt` and / or `uv pip install -e .`).

## Top-level fields

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `primary_install_strategy` | enum | yes | `uv-sync-frozen` / `uv-pip-requirements` / `uv-pip-editable` / `mixed`. Audit-only — renderer derives actual commands from `installation_steps`. |
| `system_apt_packages` | `list[str]` | no | Extra apt packages the user must install manually (no sudo from the venv). Surfaced in `install.md` as a `sudo apt install ...` hint. |
| `third_party_clones` | `list[ThirdPartyClone]` | no | Repos cloned to `<repo>/.harbor_thirdparty/<name>`. See schema below. |
| `installation_steps` | `list[InstallStep]` | yes | Ordered shell lines emitted into `setup_uv.sh`. **The order is preserved verbatim.** |
| `post_install_hooks` | `list[PostHook]` | no | Commands appended to `setup_uv.sh` under a first-run sentinel (`<repo>/.venv/.harbor_first_run_done`). |
| `evidence` | `Evidence` | yes | Audit trail: README quotes + flagged ambiguities for AskUserQuestion. |

### `InstallStep`

```json
{"kind": "uv_sync|uv_pip_install|apt_install|git_clone|shell", "args": [...], "cmd": "..."}
```

| `kind` | Required field | Renders to (in `setup_uv.sh`) |
|--------|----------------|---------------------------------|
| `uv_sync` | `args: list[str]` | `uv sync {args}` |
| `uv_pip_install` | `args: list[str]` | `uv pip install {args}` |
| `apt_install` | `pkgs: list[str]` | comment `# sudo apt install -y {pkgs}` (NOT executed; user runs manually if imports fail) |
| `git_clone` | `url`, `dest`, optional `commit` | `git clone {url} {dest} && (cd {dest} && git checkout {commit})` |
| `shell` | `cmd: str` | the cmd verbatim (caller is responsible for any `&&` chaining; `/workspace/<repo>` paths are rewritten to host absolute) |

### `ThirdPartyClone`

```json
{"url": "...", "commit": "...", "dest": ".harbor_thirdparty/<name>", "post_clone": "<shell>"}
```

Convenience wrapper. Equivalent to two `installation_steps`: a `git_clone` followed by a `shell`. Use it when the README explicitly groups them as "clone X then build it".

### `PostHook`

```json
{"kind": "shell", "cmd": "...", "when": "first_run|every_run"}
```

`first_run` is gated by a `<repo>/.venv/.harbor_first_run_done` sentinel inside the rendered `setup_uv.sh`. `every_run` is unconditional.

### `Evidence`

```json
{
  "readme_quotes": ["...", "..."],
  "ambiguities":   ["..."]
}
```

`readme_quotes` should contain ≥1 quote per `installation_steps` entry that came from prose (not from `pyproject.toml` / `requirements.txt`). `ambiguities` lists decisions where the README was vague and the agent picked a default — these are surfaced to the user during the Step 4 confirmation.

## quirks → InstallStep mapping

`probe.json` exposes `quirks: list[str]`. The InstallationPlan extractor must check each entry and emit the corresponding `InstallStep` *unless* the README already prescribes the same fix. In that case the extractor records the README quote in `evidence.readme_quotes` and skips the synthesised step (no double-install).

| Quirk | Auto-injected step (if README silent) |
|-------|---------------------------------------|
| `needs_setuptools_pin` | `uv_pip_install --no-deps "setuptools>=62.3,<81"` *before* any other install |
| `has_flash_attn` | `uv_pip_install --no-build-isolation flash-attn==<version-from-deps>` |
| `needs_render_libs` | `apt_install libegl1 libosmesa6 libgl1 libglib2.0-0` (emitted as a comment; user runs `sudo apt install ...` manually) |
| `needs_devel_base` | warns the user that the host's CUDA toolkit (`nvcc`) must be present; no auto-install |
| `needs_vulkan_icd` | NOT supported in uv mode — dependency-generator aborts with a clear error |
| `is_isaacgym` | NOT supported in uv mode — dependency-generator aborts with a clear error |
| `has_third_party_sim` | (informational; extractor should emit a `git_clone` step pinned by commit) |

## Worked example 1 — pure uv.lock simple repo

`evidence.readme_quotes` is empty because no prose-driven steps exist; the plan reduces to one `uv_sync` step. This is the back-compat path — output is bit-for-bit identical to today's `_OPTION_A_BLOCK`.

```json
{
  "primary_install_strategy": "uv-sync-frozen",
  "system_apt_packages": [],
  "third_party_clones": [],
  "installation_steps": [
    {"kind": "uv_sync", "args": ["--frozen", "--no-install-project", "--active"]}
  ],
  "post_install_hooks": [],
  "evidence": {
    "readme_quotes": [],
    "ambiguities": []
  }
}
```

## Worked example 2 — flash-attn + system libs + HF post-install (openvla-oft style)

```json
{
  "primary_install_strategy": "mixed",
  "system_apt_packages": ["libegl1", "libosmesa6"],
  "third_party_clones": [],
  "installation_steps": [
    {"kind": "apt_install", "pkgs": ["libegl1", "libosmesa6"]},
    {"kind": "uv_pip_install", "args": ["--no-deps", "setuptools>=62.3,<81"]},
    {"kind": "uv_sync", "args": ["--frozen", "--no-install-project", "--active"]},
    {"kind": "uv_pip_install", "args": ["--no-build-isolation", "flash-attn==2.5.6"]}
  ],
  "post_install_hooks": [
    {"kind": "shell",
     "cmd": "huggingface-cli download openvla/openvla-7b --local-dir /workspace/ckpt",
     "when": "first_run"}
  ],
  "evidence": {
    "readme_quotes": [
      "## Installation\n```\npip install --no-build-isolation flash-attn==2.5.6\n```",
      "Download the OpenVLA checkpoint: `huggingface-cli download openvla/openvla-7b ...`"
    ],
    "ambiguities": [
      "README pins flash-attn==2.5.6 but our pyproject.toml has it unpinned — used README's pin"
    ]
  }
}
```

Note: `needs_setuptools_pin` is in `quirks` AND the README is silent on it → extractor auto-injected the `setuptools>=62.3,<81` step. `has_flash_attn` is in `quirks` AND the README mentions flash-attn → extractor took the README's exact command (with `==2.5.6` pin) instead of synthesising one, recorded the quote.

## Worked example 3 — third-party C++ build (ManiSkill / SAPIEN style)

```json
{
  "primary_install_strategy": "uv-pip-editable",
  "system_apt_packages": ["libvulkan1", "libvulkan-dev", "vulkan-tools"],
  "third_party_clones": [
    {"url": "https://github.com/haosulab/SAPIEN.git",
     "commit": "v3.0.0.dev",
     "dest": "/opt/third_party/SAPIEN",
     "post_clone": "cd /opt/third_party/SAPIEN && pip install -e ."}
  ],
  "installation_steps": [
    {"kind": "apt_install", "pkgs": ["libvulkan1", "libvulkan-dev", "vulkan-tools"]},
    {"kind": "git_clone",
     "url": "https://github.com/haosulab/SAPIEN.git",
     "commit": "v3.0.0.dev",
     "dest": "/opt/third_party/SAPIEN"},
    {"kind": "shell", "cmd": "cd /opt/third_party/SAPIEN && uv pip install -e . --index-strategy unsafe-best-match"},
    {"kind": "uv_pip_install", "args": ["-e", "."]}
  ],
  "post_install_hooks": [],
  "evidence": {
    "readme_quotes": [
      "## SAPIEN install\nClone https://github.com/haosulab/SAPIEN.git, checkout v3.0.0.dev, then `pip install -e .`",
      "## Vulkan\nMake sure libvulkan1 and vulkan-tools are installed before running ..."
    ],
    "ambiguities": []
  }
}
```

## What NOT to put in `installation_steps`

`installation_steps` are rendered into `setup_uv.sh` and run against the host (with `<repo>/.venv` activated). Container-only paths (`/workspace/<project>`) are auto-rewritten to the host repo path, but inventing build-time-only commands still wastes a step.

If the repo's editable install needs special flags (`--no-deps`, `--no-build-isolation`), put it in `installation_steps` as a `uv_pip_install` step with the right flags. The fallback heuristic only kicks in when no plan is present.

## Renderer contract

`render_setup_uv_sh(repo)` in `render_uv.py`:

1. If `plan` is `None` → fall back to heuristic install (uv.lock present → `uv sync --frozen`; else `uv pip install -r requirements.txt` and / or `uv pip install -e .`).
2. Else iterate `installation_steps` in order, emit one shell line per step using the `kind →` mapping above.
3. After the user's plan, append the "harbor extras" block (`uv pip install wandb tensorboardX imageio[ffmpeg] matplotlib hydra-core omegaconf 'stable_baselines3[extra]'`) idempotently — these are required by downstream subagents (benchmark-generator, rl-integration-generator).
4. `apt_install` steps are emitted as comments only — `setup_uv.sh` does not have sudo. The user runs the apt commands manually if Tier 2 imports fail.
5. `quirks`-derived auto-steps the extractor synthesised are already present in `installation_steps`; the renderer does not re-inject anything from `probe.quirks`.
