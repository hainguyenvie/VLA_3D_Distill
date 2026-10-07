#!/usr/bin/env bash
# Failure analysis of trained pi0.5 adapters: greedy rollouts with per-query states on the LIBERO-PRO swap and position
# cells (same 200 episodes as the pi0.5 diagnosis), then the failure taxonomy.   pi05_failure_runs.sh <gpu> <run>:<iter> [...]
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
export PY_ENV=pi05
. "$HERE/env.sh"
cd "$W"
G="$1"; shift
O=outputs/week1; CK="$W/checkpoints/lerobot__pi05_libero_finetuned"
for spec in "$@"; do
  run="${spec%%:*}"; it=$(printf %04d "${spec##*:}"); A="$O/$run/adapter_iter$it"
  for cell in swap temp; do
    out="$O/${run}_iter${it}_pro_${cell}_steps"
    [ -f "$out/summary.json" ] || LIBERO_VARIANT=pro bash "$HERE/run_py.sh" "$G" scripts/eval_libero.py --ckpt "pi05:$CK" --lora "$A" \
      --suite "libero_object_$cell" --trials 20 --num_envs 10 --max_steps 280 --out "$out" || { log "FAILED $out"; continue; }
    [ -f "$out/failures.csv" ] || LIBERO_VARIANT=pro bash "$HERE/run_py.sh" "$G" scripts/analyze_failures.py --run "$out" \
      --suite "libero_object_$cell" || log "FAILED taxonomy $out"
  done
done
log "FAILURE_RUNS_DONE $*"
