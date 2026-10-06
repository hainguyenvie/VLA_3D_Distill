#!/usr/bin/env bash
# pi0.5 (LeRobot checkpoint) anchors in this harness: LIBERO-Object standard, the LIBERO-PRO Object cells and the
# LIBERO-Plus Robot-init / Layout tasks. One job at a time; finished evaluations are skipped.
#   pi05_anchors.sh [trials per task, default 10]
# Launch detached: PY_ENV=pi05 setsid nohup bash repo/scripts/server/pi05_anchors.sh > logs/pi05_anchors.log 2>&1 < /dev/null &
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
export PY_ENV=pi05 MIN_FREE_GPU_GB="${MIN_FREE_GPU_GB:-16}"
. "$HERE/env.sh"
cd "$W"
TRIALS="${1:-10}"; O=outputs/week1; N="${NUM_ENVS:-5}"
CKPT="pi05:$W/checkpoints/lerobot__pi05_libero_finetuned"
run() {  # run <variant> <suite> <out name> [extra args]
  local variant="$1" suite="$2" out="$3"; shift 3
  [ -f "$O/$out/summary.json" ] && { log "skip $out"; return 0; }
  LIBERO_VARIANT="$variant" bash "$HERE/run_py.sh" 0 scripts/eval_libero.py --ckpt "$CKPT" --suite "$suite" --trials "$TRIALS" \
    --num_envs "$N" --max_steps 280 --no_steps --out "$O/$out" "$@" || log "FAILED $out"
}
run "" libero_object pi05_object
for cell in swap task lan object temp; do run pro "libero_object_$cell" "pi05_pro_object_$cell"; done
[ -f "$O/plus_pi05_robotlayout/summary.json" ] || LIBERO_VARIANT=plus bash "$HERE/run_py.sh" 0 scripts/eval_libero_plus.py --ckpt "$CKPT" \
  --suite libero_object --per_category 60 --categories "Robot Initial States,Objects Layout" --num_envs "$N" --max_steps 280 \
  --out "$O/plus_pi05_robotlayout" || log "FAILED plus_pi05_robotlayout"
log "PI05_ANCHORS_DONE"
