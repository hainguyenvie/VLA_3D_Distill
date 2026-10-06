#!/usr/bin/env bash
# Early read of pi0.5 runs: once <run>/adapter_iter<NNNN> exists, evaluate it on one LIBERO-PRO Object cell.
#   pi05_peek.sh <gpu> <cell> <iteration> <run> [<run> ...]
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
export PY_ENV=pi05
. "$HERE/env.sh"
cd "$W"
G="$1"; CELL="$2"; IT=$(printf %04d "$3"); shift 3
O=outputs/week1; CK="$W/checkpoints/lerobot__pi05_libero_finetuned"
for run in "$@"; do
  A="$O/$run/adapter_iter$IT"
  until [ -f "$A/adapter_model.safetensors" ]; do sleep 60; done
  sleep 20
  out="$O/${run}_iter${IT}_pro_$CELL"
  [ -f "$out/summary.json" ] || LIBERO_VARIANT=pro bash "$HERE/run_py.sh" "$G" scripts/eval_libero.py --ckpt "pi05:$CK" --lora "$A" \
    --suite "libero_object_$CELL" --trials 20 --num_envs 8 --max_steps 280 --no_steps --out "$out" || log "FAILED $out"
done
log "PEEK_DONE $CELL $IT"
