#!/usr/bin/env bash
# harbor PreToolUse hook for the Bash tool.
# Refuses obviously-destructive commands; harbor runs docker / pytest a lot,
# so this catches accidents before they reach the shell.
#
# Contract (see https://code.claude.com/docs/en/hooks):
#   stdin  -> JSON payload of the tool_use call (Claude Code passes this)
#   stdout -> hookSpecificOutput JSON carrying permissionDecision, only when denying
#   exit 0 -> always. Exit 0 IS the intended code when printing structured JSON: the
#             decision lives in the payload, not in the exit status. Exiting 2 instead
#             makes Claude Code take the stderr path and treat the JSON as raw text, so
#             the reason reaches the model as an unparsed blob rather than a decision.

set -euo pipefail

input=$(cat)

cmd=$(echo "$input" | python3 -c '
import json, sys
try:
    payload = json.load(sys.stdin)
    print(payload.get("tool_input", {}).get("command", ""))
except Exception:
    pass
' 2>/dev/null || true)

if [[ -z "$cmd" ]]; then
    cmd=$(echo "$input" | grep -oE '"command"\s*:\s*"[^"]*"' | head -n1 | sed 's/.*"command"\s*:\s*"\(.*\)"/\1/' || true)
fi

# Flags are order-free, so match `-[a-z]*[rf][a-z]*` rather than a fixed `-rf?`: the
# force-recursive delete has several equally destructive spellings. The path alternation
# has to accept a trailing `/*` as well — a glob expanding to every top-level entry is
# the classic form, and anchoring on end-or-space alone lets it through untouched.
declare -a DANGEROUS_PATTERNS=(
    'rm[[:space:]]+(-[a-z]*[rf][a-z]*[[:space:]]+)+/(\*)?($|[[:space:]])'   # force-delete of /
    'rm[[:space:]]+(-[a-z]*[rf][a-z]*[[:space:]]+)+~(/\*)?($|[[:space:]])'  # force-delete of ~
    'rm[[:space:]]+.*--recursive.*--force[[:space:]]+/($|[[:space:]])'      # long-flag spelling
    ':\(\)\{[[:space:]]*:\|:&[[:space:]]*\};:'                              # fork bomb
    'mkfs\.[a-z]+[[:space:]]+/dev/'                                         # mkfs on a device
    'dd[[:space:]]+.*of=/dev/sd'                                            # raw write to disk
    '>[[:space:]]*/dev/sd[a-z]'                                             # stdout to disk
    'docker[[:space:]]+system[[:space:]]+prune[[:space:]]+.*--volumes.*-f'  # nuke all volumes
    'docker[[:space:]]+volume[[:space:]]+prune[[:space:]]+.*-f'             # delete all volumes
)

for pat in "${DANGEROUS_PATTERNS[@]}"; do
    if [[ "$cmd" =~ $pat ]]; then
        # Built with json.dumps rather than a heredoc: the reason embeds the pattern, which
        # is full of quotes and backslashes that would otherwise produce invalid JSON and
        # silently downgrade the deny into an unparsed blob.
        python3 -c '
import json, sys
print(json.dumps({"hookSpecificOutput": {
    "hookEventName": "PreToolUse",
    "permissionDecision": "deny",
    "permissionDecisionReason":
        "harbor PreToolUse hook: refusing dangerous pattern " + repr(sys.argv[1])
        + ". If you really mean it, run it from a shell outside Claude Code.",
}}))' "$pat"
        exit 0
    fi
done

exit 0
