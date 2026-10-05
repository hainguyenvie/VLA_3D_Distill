#!/usr/bin/env bash
# LIBERO-Plus (Object, the fixed 420-task sample) for the iteration-10 adapter of each given run, one after another.
#   plus_iter10.sh <run name> [<run name> ...]     waits for each adapter to exist
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
. "$HERE/env.sh"
cd "$W"
O=outputs/week1
STUDENT="$W/checkpoints/Haozhan72__Openvla-oft-SFT-libero-object-traj1"
for run in "$@"; do
  A="$O/$run/adapter_iter0010"
  until [ -f "$A/adapter_config.json" ]; do sleep 60; done
  sleep 30  # let the adapter files finish writing
  [ -f "$O/plus_${run}_iter10/summary.json" ] && { log "skip $run"; continue; }
  LIBERO_VARIANT=plus bash "$HERE/run_py.sh" 0 scripts/eval_libero_plus.py --ckpt "$STUDENT" --lora "$A" --suite libero_object \
    --per_category 60 --num_envs "${NUM_ENVS:-5}" --out "$O/plus_${run}_iter10" || log "FAILED $run"
done
log "PLUS_ITER10_DONE $*"
