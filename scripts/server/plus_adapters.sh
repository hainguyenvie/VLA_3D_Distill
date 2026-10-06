#!/usr/bin/env bash
# LIBERO-Plus (Object, the fixed 420-task sample) for saved adapters of distillation runs, one after another.
#   plus_adapters.sh <run name>:<iteration> [<run name>:<iteration> ...]     waits for each adapter to exist
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
. "$HERE/env.sh"
cd "$W"
O=outputs/week1
STUDENT="$W/checkpoints/Haozhan72__Openvla-oft-SFT-libero-object-traj1"
for spec in "$@"; do
  run="${spec%%:*}"; it="${spec##*:}"
  A="$O/$run/adapter_iter$(printf %04d "$it")"
  until [ -f "$A/adapter_config.json" ]; do sleep 60; done
  sleep 30  # let the adapter files finish writing
  [ -f "$O/plus_${run}_iter${it}/summary.json" ] && { log "skip $spec"; continue; }
  LIBERO_VARIANT=plus bash "$HERE/run_py.sh" "${GPU:-0}" scripts/eval_libero_plus.py --ckpt "$STUDENT" --lora "$A" --suite libero_object \
    --per_category 60 --num_envs "${NUM_ENVS:-5}" --out "$O/plus_${run}_iter${it}" || log "FAILED $spec"
done
log "PLUS_ADAPTERS_DONE $*"
