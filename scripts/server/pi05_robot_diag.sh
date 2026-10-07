#!/usr/bin/env bash
# Why does the rotation counterfactual lower LIBERO-Plus "Robot Initial States" (78, mix 75; base 83, coshift 90)?
# Rollouts with per-query states on that category (Object, 60 tasks), then the failure taxonomy.
#   pi05_robot_diag.sh <gpu>
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
export PY_ENV=pi05
. "$HERE/env.sh"
cd "$W"
G="$1"; O=outputs/week1; CK="$W/checkpoints/lerobot__pi05_libero_finetuned"
for run in p05_opd_base_s8 p05_ocd_rotate_s8 p05_ocd_coshift_s8 p05_ocd_mix_s8; do
  out="$O/plusrobot_${run}_steps"; A="$(ls -d $O/$run/adapter_iter* | sort | tail -1)"
  [ -f "$out/summary.json" ] || LIBERO_VARIANT=plus bash "$HERE/run_py.sh" "$G" scripts/eval_libero_plus.py --ckpt "pi05:$CK" --lora "$A" \
    --suite libero_object --per_category 60 --categories "Robot Initial States" --num_envs 8 --max_steps 280 --save_steps \
    --out "$out" || { log "FAILED $out"; continue; }
  [ -f "$out/failures.csv" ] || LIBERO_VARIANT=plus bash "$HERE/run_py.sh" "$G" scripts/analyze_failures.py --run "$out" \
    --suite libero_object || log "FAILED taxonomy $out"
done
log "ROBOT_DIAG_DONE"
