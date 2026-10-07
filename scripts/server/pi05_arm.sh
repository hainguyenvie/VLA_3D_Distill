#!/usr/bin/env bash
# One pi0.5 arm with the round 3-5 budget and evaluations: train (resumes from <run>/state.pt if present, so a run can
# be moved to another card), then the evaluations of its last adapter. The training log is appended to.
#   [SUITE=libero_spatial MAX_STEPS=220] pi05_arm.sh <gpu> <run> <train_pi05_ocd.py args...>
# SUITE (default libero_object) and MAX_STEPS (default 280, the openpi horizons are Spatial 220, Goal 300, Long 520) set
# the training suite and every evaluation; the LIBERO-PRO position cell (temp) exists for Object only.
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
export PY_ENV=pi05
. "$HERE/env.sh"
cd "$W"
O=outputs/week1; CK="$W/checkpoints/lerobot__pi05_libero_finetuned"
SUITE="${SUITE:-libero_object}"; MS="${MAX_STEPS:-280}"
CELLS="swap task lan object"; [ "$SUITE" = libero_object ] && CELLS="swap temp task lan object"
COMMON="--ckpt $CK --suite $SUITE --max_steps $MS --iters 20 --states_per_iter 1024 --episodes_per_batch 20 --num_envs 8 --eval_every 4 --eval_trials 10"
evals() {
  local g="$1" run="$2" last; last="$(ls -d "$O/$run"/adapter_iter* | sort | tail -1)"
  bash "$HERE/run_py.sh" "$g" scripts/eval_libero.py --ckpt "pi05:$CK" --lora "$last" --suite "$SUITE" --trials 20 --num_envs 8 \
    --max_steps "$MS" --no_steps --out "$O/${run}_${SUITE#libero_}" || log "FAILED $run standard"
  for cell in $CELLS; do
    LIBERO_VARIANT=pro bash "$HERE/run_py.sh" "$g" scripts/eval_libero.py --ckpt "pi05:$CK" --lora "$last" --suite "${SUITE}_$cell" \
      --trials 20 --num_envs 8 --max_steps "$MS" --no_steps --out "$O/${run}_pro_$cell" || log "FAILED $run pro $cell"
  done
  LIBERO_VARIANT=plus bash "$HERE/run_py.sh" "$g" scripts/eval_libero_plus.py --ckpt "pi05:$CK" --lora "$last" --suite "$SUITE" \
    --per_category 60 --categories "Robot Initial States,Objects Layout" --num_envs 8 --max_steps "$MS" --out "$O/plus_${run}_robotlayout" \
    || log "FAILED $run plus"
  log "EVALS_DONE $run"
}
G="$1"; RUN="$2"; shift 2
bash "$HERE/run_py.sh" "$G" scripts/train_pi05_ocd.py $COMMON --out "$O/$RUN" "$@" >> "logs/$RUN.log" 2>&1 && evals "$G" "$RUN"
