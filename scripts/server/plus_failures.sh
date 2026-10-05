#!/usr/bin/env bash
# LIBERO-Plus (Object, the fixed 420-task sample) re-run with per-query simulator states kept, followed by the
# failure taxonomy per perturbation type.   plus_failures.sh <oft|fullsft|student> [...]     one job at a time
# Launch detached: setsid nohup bash repo/scripts/server/plus_failures.sh oft > logs/plus_failures.log 2>&1 < /dev/null &
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
. "$HERE/env.sh"
cd "$W"
C="$W/checkpoints"; O=outputs/week1
export LIBERO_VARIANT=plus
for name in "$@"; do
  case "$name" in
    oft) CKPT="oft:$C/moojink__openvla-7b-oft-finetuned-libero-object"; STEPS=280 ;;  # openvla-oft's horizon for LIBERO-Object
    fullsft) CKPT="$C/Haozhan72__Openvla-oft-SFT-libero-object-trajall"; STEPS=512 ;;
    student) CKPT="$C/Haozhan72__Openvla-oft-SFT-libero-object-traj1"; STEPS=512 ;;
    *) log "unknown policy $name"; continue ;;
  esac
  OUT="$O/plus_${name}_steps"
  if [ ! -f "$OUT/summary.json" ]; then
    bash "$HERE/run_py.sh" 0 scripts/eval_libero_plus.py --ckpt "$CKPT" --suite libero_object --per_category 60 \
      --num_envs "${NUM_ENVS:-7}" --max_steps "$STEPS" --save_steps --out "$OUT" || { log "FAILED $name"; continue; }
  fi
  bash "$HERE/run_py.sh" 0 scripts/analyze_failures.py --run "$OUT" || log "FAILED taxonomy $name"
done
log "PLUS_FAILURES_DONE $*"
