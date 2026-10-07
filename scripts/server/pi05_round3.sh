#!/usr/bin/env bash
# pi0.5 round 3: mirror counterfactuals (exact labels, the target at a different place), with and without the
# early-state restriction, and a second seed of the matched on-policy baseline. Same budget / evaluations as before.
#   pi05_round3.sh <gpu mirror s7> <gpu mirror early s7> <gpu base s8>
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
export PY_ENV=pi05
. "$HERE/env.sh"
cd "$W"
O=outputs/week1; CK="$W/checkpoints/lerobot__pi05_libero_finetuned"
COMMON="--ckpt $CK --suite libero_object --iters 20 --states_per_iter 1024 --episodes_per_batch 20 --num_envs 8 --eval_every 4 --eval_trials 10"
evals() {
  local g="$1" run="$2" last; last="$(ls -d "$O/$run"/adapter_iter* | sort | tail -1)"
  bash "$HERE/run_py.sh" "$g" scripts/eval_libero.py --ckpt "pi05:$CK" --lora "$last" --suite libero_object --trials 20 --num_envs 8 \
    --max_steps 280 --no_steps --out "$O/${run}_object" || log "FAILED $run object"
  for cell in swap temp task lan object; do
    LIBERO_VARIANT=pro bash "$HERE/run_py.sh" "$g" scripts/eval_libero.py --ckpt "pi05:$CK" --lora "$last" --suite "libero_object_$cell" \
      --trials 20 --num_envs 8 --max_steps 280 --no_steps --out "$O/${run}_pro_$cell" || log "FAILED $run pro $cell"
  done
  LIBERO_VARIANT=plus bash "$HERE/run_py.sh" "$g" scripts/eval_libero_plus.py --ckpt "pi05:$CK" --lora "$last" --suite libero_object \
    --per_category 60 --categories "Robot Initial States,Objects Layout" --num_envs 8 --max_steps 280 --out "$O/plus_${run}_robotlayout" \
    || log "FAILED $run plus"
  log "EVALS_DONE $run"
}
arm() { local g="$1" run="$2"; shift 2
  bash "$HERE/run_py.sh" "$g" scripts/train_pi05_ocd.py $COMMON --out "$O/$run" "$@" > "logs/$run.log" 2>&1 && evals "$g" "$run"; }
arm "$1" p05_ocd_mirror_s7 --cf_mode mirror --lambda_cf 1.0 --seed 7 &
arm "$2" p05_ocd_mirror_early_s7 --cf_mode mirror --lambda_cf 1.0 --cf_max_query 4 --seed 7 &
arm "$3" p05_opd_base_s8 --no_cf --seed 8 &
wait
log "PI05_ROUND3_DONE"
