#!/usr/bin/env bash
# The evaluations of pi05_arm.sh for a trained run, skipping the cells that already have a summary.json (e.g. to move the
# rest of a run's evaluations to another card).
#   [SUITE=libero_spatial MAX_STEPS=220 EVAL_EXTRA=--geo] pi05_evals.sh <gpu> <run>
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
export PY_ENV=pi05
. "$HERE/env.sh"
cd "$W"
O=outputs/week1; CK="$W/checkpoints/lerobot__pi05_libero_finetuned"
SUITE="${SUITE:-libero_object}"; MS="${MAX_STEPS:-280}"; EX="${EVAL_EXTRA:-}"
CELLS="swap task lan object"; [ "$SUITE" = libero_object ] && CELLS="swap temp task lan object"
G="$1"; run="$2"; last="$(ls -d "$O/$run"/adapter_iter* | sort | tail -1)"
out="$O/${run}_${SUITE#libero_}"
[ -f "$out/summary.json" ] || bash "$HERE/run_py.sh" "$G" scripts/eval_libero.py --ckpt "pi05:$CK" --lora "$last" --suite "$SUITE" \
  --trials 20 --num_envs 8 --max_steps "$MS" --no_steps $EX --out "$out" || log "FAILED $run standard"
for cell in $CELLS; do
  out="$O/${run}_pro_$cell"
  [ -f "$out/summary.json" ] || LIBERO_VARIANT=pro bash "$HERE/run_py.sh" "$G" scripts/eval_libero.py --ckpt "pi05:$CK" --lora "$last" \
    --suite "${SUITE}_$cell" --trials 20 --num_envs 8 --max_steps "$MS" --no_steps $EX --out "$out" || log "FAILED $run pro $cell"
done
out="$O/plus_${run}_robotlayout"
[ -f "$out/summary.json" ] || LIBERO_VARIANT=plus bash "$HERE/run_py.sh" "$G" scripts/eval_libero_plus.py --ckpt "pi05:$CK" --lora "$last" \
  --suite "$SUITE" --per_category 60 --categories "Robot Initial States,Objects Layout" --num_envs 8 --max_steps "$MS" $EX \
  --out "$out" || log "FAILED $run plus"
log "EVALS_DONE $run"
