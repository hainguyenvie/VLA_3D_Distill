#!/usr/bin/env bash
# pi0.5 round 4: the rotation counterfactual (world rotated about the first joint's axis, label = rotated nominal
# chunk; needs no symmetric scene, no renamed instruction, no target identity), two seeds, and a second seed of the
# mirror counterfactual. Same budget / evaluations as round 3. Starts once the failure rollouts have finished.
#   pi05_round4.sh <gpu rotate s7> <gpu rotate s8> <gpu mirror s8>
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
export PY_ENV=pi05
. "$HERE/env.sh"
cd "$W"
for _ in $(seq 1 120); do  # the failure rollouts share the cards and CPUs (at most 2 h)
  [ "$(cat logs/fr_swap20.log logs/fr_base20.log logs/fr_swap8.log 2>/dev/null | grep -c FAILURE_RUNS_DONE)" -ge 3 ] && break
  sleep 60
done
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
arm "$1" p05_ocd_rotate_s7 --cf_mode rotate --cf_theta 0.1,0.3 --lambda_cf 1.0 --seed 7 &
arm "$2" p05_ocd_rotate_s8 --cf_mode rotate --cf_theta 0.1,0.3 --lambda_cf 1.0 --seed 8 &
arm "$3" p05_ocd_mirror_s8 --cf_mode mirror --lambda_cf 1.0 --seed 8 &
wait
log "PI05_ROUND4_DONE"
