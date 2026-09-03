#!/usr/bin/env bash
# harbor Stop / SubagentStop hook — append a per-turn audit record to disk.
#
# OPT-IN. Hooks ship with the plugin and fire in every project the user opens, not only
# the benchmark repos harbor was installed for, so a logger that is on by default writes
# a record of unrelated work to the user's home directory without ever being asked. Set
#
#     HARBOR_AUDIT_LOG=1
#
# in your environment to enable it. Records land in $HOME/.claude/audit/<date>.jsonl:
# one line per turn carrying the session id, the agent name, and the transcript path.
# Nothing is written, and no directory is created, while it is unset.
#
# Budget: <1 second.

set -u

if [ "${HARBOR_AUDIT_LOG:-0}" != "1" ]; then
    exit 0
fi

INPUT=$(cat)
LOG_DIR="$HOME/.claude/audit"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/$(date +%Y-%m-%d).jsonl"

if command -v jq >/dev/null 2>&1; then
    echo "$INPUT" | jq -c '{
        ts: now,
        session_id: (.session_id // ""),
        agent: (.agent // ""),
        stop_hook_active: (.stop_hook_active // false),
        transcript_path: (.transcript_path // "")
    }' >> "$LOG_FILE" 2>/dev/null || true
else
    # jq is not a harbor prerequisite, so the fallback has to stand on its own. python3 is
    # already required by the pre-tool hook, and it emits correctly-escaped JSON where the
    # hand-rolled sed version produced invalid lines on any input containing a backslash.
    echo "$INPUT" | python3 -c '
import json, sys, time
try:
    p = json.load(sys.stdin)
except Exception:
    p = {}
print(json.dumps({
    "ts": time.time(),
    "session_id": p.get("session_id", ""),
    "agent": p.get("agent", ""),
    "stop_hook_active": p.get("stop_hook_active", False),
    "transcript_path": p.get("transcript_path", ""),
}))' >> "$LOG_FILE" 2>/dev/null || true
fi

exit 0
