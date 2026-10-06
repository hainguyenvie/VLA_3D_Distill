#!/usr/bin/env bash
# pi0.5 round 1 on the 8x H200 node: on-policy counterfactual distillation (swap worlds) against the matched on-policy
# distillation without counterfactuals, and the shift-world variant; then the evaluations of each last adapter.
# Launch detached: setsid nohup bash repo/scripts/server/pi05_round1.sh > logs/pi05_round1.log 2>&1 < /dev/null &
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
export PY_ENV=pi05
. "$HERE/env.sh"
cd "$W"
O=outputs/week1; CK="$W/checkpoints/lerobot__pi05_libero_finetuned"
COMMON="--ckpt $CK --suite libero_object --iters 20 --states_per_iter 1024 --episodes_per_batch 20 --num_envs 10 --eval_every 4 --eval_trials 10 --seed 7"
evals() {  # evals <gpu> <run>: last adapter on Object (20 trials/task), the LIBERO-PRO Object cells and LIBERO-Plus Robot/Layout
  local g="$1" run="$2" last; last="$(ls -d "$O/$run"/adapter_iter* | sort | tail -1)"
  bash "$HERE/run_py.sh" "$g" scripts/eval_libero.py --ckpt "pi05:$CK" --lora "$last" --suite libero_object --trials 20 --num_envs 10 \
    --max_steps 280 --no_steps --out "$O/${run}_object" || log "FAILED $run object"
  for cell in swap temp task lan object; do
    LIBERO_VARIANT=pro bash "$HERE/run_py.sh" "$g" scripts/eval_libero.py --ckpt "pi05:$CK" --lora "$last" --suite "libero_object_$cell" \
      --trials 20 --num_envs 10 --max_steps 280 --no_steps --out "$O/${run}_pro_$cell" || log "FAILED $run pro $cell"
  done
  LIBERO_VARIANT=plus bash "$HERE/run_py.sh" "$g" scripts/eval_libero_plus.py --ckpt "pi05:$CK" --lora "$last" --suite libero_object \
    --per_category 60 --categories "Robot Initial States,Objects Layout" --num_envs 10 --max_steps 280 --out "$O/plus_${run}_robotlayout" \
    || log "FAILED $run plus"
  log "EVALS_DONE $run"
}
arm() {  # arm <gpu> <run> <args...>
  local g="$1" run="$2"; shift 2
  bash "$HERE/run_py.sh" "$g" scripts/train_pi05_ocd.py $COMMON --out "$O/$run" "$@" > "logs/$run.log" 2>&1 && evals "$g" "$run"
}
arm 5 p05_ocd_swap_s7 --cf_mode swap --lambda_cf 1.0 &
arm 6 p05_opd_base_s7 --no_cf &
arm 7 p05_ocd_shift_s7 --cf_mode shift --cf_delta 0.02,0.10 --lambda_cf 1.0 &
wait
log "PI05_ROUND1_DONE"
