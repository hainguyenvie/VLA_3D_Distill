#!/usr/bin/env bash
# pi0.5 round 5: the coshift counterfactual (target and hand moved together; the target ends up elsewhere relative
# to the other objects and the container, the label is the nominal chunk until it carries the target), two seeds.
# Same budget / evaluations as rounds 3-4.
#   pi05_round5.sh <gpu coshift s7> <gpu coshift s8>
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
arm "$1" p05_ocd_coshift_s7 --cf_mode coshift --lambda_cf 1.0 --seed 7 &
arm "$2" p05_ocd_coshift_s8 --cf_mode coshift --lambda_cf 1.0 --seed 8 &
wait
log "PI05_ROUND5_DONE"
