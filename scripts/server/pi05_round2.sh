#!/usr/bin/env bash
# pi0.5 round 2: a gentler swap arm (lr 2e-5, image encoder frozen) and the off-policy control of the swap arm (pairs
# built on the frozen model's own states instead of the student's). Same budget and evaluations as round 1.
#   pi05_round2.sh <gpu for the gentle arm> <gpu for the control>
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
export PY_ENV=pi05
. "$HERE/env.sh"
cd "$W"
O=outputs/week1; CK="$W/checkpoints/lerobot__pi05_libero_finetuned"
COMMON="--ckpt $CK --suite libero_object --iters 20 --states_per_iter 1024 --episodes_per_batch 20 --num_envs 8 --eval_every 4 --eval_trials 10 --seed 7"
evals() {
  local g="$1" run="$2" last; last="$(ls -d "$O/$run"/adapter_iter* | sort | tail -1)"
  bash "$HERE/run_py.sh" "$g" scripts/eval_libero.py --ckpt "pi05:$CK" --lora "$last" --suite libero_object --trials 20 --num_envs 8 \
    --max_steps 280 --no_steps --out "$O/${run}_object" || log "FAILED $run object"
  for cell in swap temp task lan object; do
    LIBERO_VARIANT=pro bash "$HERE/run_py.sh" "$g" scripts/eval_libero.py --ckpt "pi05:$CK" --lora "$last" --suite "libero_object_$cell" \
      --trials 20 --num_envs 8 --max_steps 280 --no_steps --out "$O/${run}_pro_$cell" || log "FAILED $run pro $cell"
  done
  log "EVALS_DONE $run"
}
arm() { local g="$1" run="$2"; shift 2
  bash "$HERE/run_py.sh" "$g" scripts/train_pi05_ocd.py $COMMON --out "$O/$run" "$@" > "logs/$run.log" 2>&1 && evals "$g" "$run"; }
arm "$1" p05_ocd_swap_gentle_s7 --cf_mode swap --lambda_cf 1.0 --lr 2e-5 --lr_min 2e-6 --lora_scope llm &
arm "$2" p05_ocd_swap_offpolicy_s7 --cf_mode swap --lambda_cf 1.0 --state_source teacher &
wait
log "PI05_ROUND2_DONE"
