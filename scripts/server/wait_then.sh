#!/usr/bin/env bash
# Run a command once a pattern has appeared in a log file:  wait_then.sh <log file> <grep -E pattern> <command...>
# Launch detached: setsid nohup bash repo/scripts/server/wait_then.sh logs/a.log "TRAIN_DONE" bash repo/... > logs/b.log 2>&1 < /dev/null &
set -uo pipefail
. "$(dirname "$0")/env.sh"
LOG="$1"; PAT="$2"; shift 2
until grep -qE "$PAT" "$LOG" 2>/dev/null; do sleep 60; done
log "condition met in $LOG, starting: $*"
"$@"
