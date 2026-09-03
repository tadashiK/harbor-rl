---
description: Show the host's current W&B login info (account name + masked API key) and offer to re-login, logout, or just inspect. Use when the user types /harbor:wandb-setup or asks "show wandb account", "am I logged into wandb", "wandb login", "wandb relogin", "wandb logout", "switch wandb account", "reconfigure wandb".
---

# /harbor:wandb-setup — Inspect / re-login / logout W&B credentials

W&B credentials live in `~/.netrc` on the host (the rl-integration container mounts it read-only). This command shows the active identity and offers re-login, logout, or inspect.

The API key is **never** printed in full — only `***<last4> (length=N)`. Output is safe to share.

## Action

1. **Print the current state** — run the credential check below, surface the masked identity:

```bash
NETRC="${HOME}/.netrc"
if [ -f "${NETRC}" ] && grep -q "machine api.wandb.ai" "${NETRC}" 2>/dev/null; then
    LOGIN=$(awk '/^machine api\.wandb\.ai/{f=1; next} f && /^[ \t]*login/{print $2; exit}' "${NETRC}")
    KEY=$(awk   '/^machine api\.wandb\.ai/{f=1; next} f && /^[ \t]*password/{print $2; exit}' "${NETRC}")
    KEY_LEN=${#KEY}
    KEY_TAIL=${KEY: -4}
    echo "[wandb-creds] ✅ FOUND in ${NETRC}"
    echo "  account (login): ${LOGIN:-<unknown>}"
    echo "  api_key:         ***${KEY_TAIL} (length=${KEY_LEN})"
    if command -v wandb >/dev/null 2>&1; then
        ENT=$(wandb status 2>/dev/null | grep -iE 'entity|user|api_key' | head -3 | sed 's/^/  /')
        if [ -n "${ENT}" ]; then
            echo "  wandb status:"
            echo "${ENT}"
        fi
    fi
    STATUS=found
else
    echo "[wandb-creds] ❌ MISSING — no api.wandb.ai entry in ${NETRC}"
    STATUS=missing
fi
```

2. **Ask what the user wants to do** via `AskUserQuestion`:

   - `inspect` — just print the state above; do nothing else (default if `STATUS=found`)
   - `re-login` — start fresh: prompt the user to run `wandb login --relogin` on the host
   - `logout` — remove credentials from `~/.netrc` via `wandb logout` (or strip the entry manually if the `wandb` CLI is missing)
   - `abort` — bail out

3. **On `re-login`** — print these instructions and wait via a follow-up `AskUserQuestion`:

   > 1. Open https://wandb.ai/authorize in your browser and copy the API key.
   > 2. On the host (NOT inside any container), run:
   >    ```
   >    wandb login --relogin
   >    ```
   >    Paste the key when prompted.
   > 3. Reply `ready` when done, or `abort` to cancel.

   On `ready`, re-run the credential block from step 1. If the new identity differs (compare `LOGIN` + `KEY_TAIL` before/after), confirm: "Switched from `<old_login>` to `<new_login>`. Continue?" — purely informational.

4. **On `logout`** — confirm once (`AskUserQuestion: [yes, logout / cancel]`), then:

   ```bash
   if command -v wandb >/dev/null 2>&1; then
       wandb logout
   else
       # Fallback: surgically strip the api.wandb.ai entry from ~/.netrc.
       # Make a backup first so the user can restore.
       cp "${HOME}/.netrc" "${HOME}/.netrc.bak.$(date +%Y%m%d-%H%M%S)"
       awk '
         /^machine api\.wandb\.ai/ { skip=3; next }
         skip > 0 { skip--; next }
         { print }
       ' "${HOME}/.netrc.bak."* | tail -1 > "${HOME}/.netrc.new"
       mv "${HOME}/.netrc.new" "${HOME}/.netrc"
       echo "[wandb-logout] api.wandb.ai entry removed; backup at ${HOME}/.netrc.bak.*"
   fi
   ```

   Then re-run the credential block from step 1 to confirm `STATUS=missing`.

5. **On `inspect`** — nothing more to do; the print from step 1 was the whole answer.

## What this command does NOT do

- **Does NOT touch any container** — credentials live on the host. The rl-integration container reads them via the bind-mount on next start.
- **Does NOT clear cached W&B run dirs** under `outputs/<run>/wandb/` — those are tied to specific past runs, not credentials. Delete them manually if you want.
- **Does NOT switch entities/teams** — that's per-run via `WANDB_ENTITY=<team>` env or `wandb.init(entity=...)`. This command only manages the API key.

## When the user runs this mid-workflow

If `rl-integration-generator` is running and the user wants to swap accounts:

1. Let the in-flight smoke complete (or abort it).
2. Run `/harbor:wandb-setup` → `re-login`.
3. Re-dispatch `rl-integration-generator` — it picks up the new credentials at the next training run.

A live training process started before the credential change keeps the old identity until restarted.
