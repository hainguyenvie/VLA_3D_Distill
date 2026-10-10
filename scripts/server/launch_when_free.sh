#!/usr/bin/env bash
# Start the given launcher commands one after the other, each only when fewer than MAX trainings of ours
# (scripts/train_pi05_ocd.py) are running, so the shared machine's CPUs are not oversubscribed.
#   MAX=7 launch_when_free.sh "<env assignments> pi05_round19.sh ..." "..."
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
. "$HERE/env.sh"
cd "$W"
MAX="${MAX:-7}"
n=0
for cmd in "$@"; do
  while [ "$(pgrep -fc 'bin/python -u .*scripts/train_pi05_ocd.py' || true)" -ge "$MAX" ]; do sleep 300; done
  n=$((n + 1))
  log "LAUNCH $cmd"
  setsid nohup bash -c "cd $W && $cmd" > "logs/when_free_$(date +%s)_$n.log" 2>&1 < /dev/null &
  sleep 900  # let it reach its training phase before counting again
done
log "LAUNCH_QUEUE_DONE"
