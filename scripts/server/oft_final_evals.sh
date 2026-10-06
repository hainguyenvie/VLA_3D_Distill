#!/usr/bin/env bash
# Final scoring of finished standard-OFT distillation runs: last adapter on LIBERO-Object (500 episodes, 280-step
# horizon) and on the fixed LIBERO-Plus sample.   oft_final_evals.sh <run name> [...]   (waits for TRAIN_DONE)
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
. "$HERE/env.sh"
cd "$W"
O=outputs/week1
CKPT="oft:$W/checkpoints/moojink__openvla-7b-oft-finetuned-libero-object"
for run in "$@"; do
  until grep -q "TRAIN_DONE" "logs/$run.log" 2>/dev/null; do sleep 60; done
  last="$(ls -d "$O/$run"/adapter_iter* | sort | tail -1)"; it="$(basename "$last" | sed 's/adapter_iter0*//')"
  out="$O/${run}_iter$(printf %04d "$it")_eval500"
  [ -f "$out/summary.json" ] || bash "$HERE/run_py.sh" "${GPU:-0}" scripts/eval_libero.py --ckpt "$CKPT" --lora "$last" --suite libero_object \
    --num_envs "${NUM_ENVS:-7}" --max_steps 280 --no_steps --out "$out" || log "FAILED $out"
  out="$O/plus_${run}_iter$it"
  [ -f "$out/summary.json" ] || LIBERO_VARIANT=plus bash "$HERE/run_py.sh" "${GPU:-0}" scripts/eval_libero_plus.py --ckpt "$CKPT" --lora "$last" \
    --suite libero_object --per_category 60 --num_envs "${NUM_ENVS:-7}" --max_steps 280 --out "$out" || log "FAILED $out"
done
log "OFT_FINAL_EVALS_DONE $*"
