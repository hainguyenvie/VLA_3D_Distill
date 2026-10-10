#!/usr/bin/env bash
# Re-evaluation with the furniture placement fixed per (task, trial) (eval_libero.py --fixture_seed), so that every model
# meets the same cabinet / stove position in the same trial. LIBERO re-draws the furniture at every reset and the
# scheduler decides which env (and how many episodes before) runs a trial, so the default protocol only matches the
# placement distribution across models, not the trials. Object has no furniture and is not affected.
#   fs_evals.sh <gpu> <suite> <max steps> "<tag>=<adapter dir or ->[ <tag>=<adapter> ...]" [cells, default "std swap"]
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
export PY_ENV=pi05
. "$HERE/env.sh"
cd "$W"
G="$1"; SUITE="$2"; MS="$3"; LIST="$4"; CELLS="${5:-std swap}"
CK="$W/checkpoints/lerobot__pi05_libero_finetuned"
for item in $LIST; do
  tag="${item%%=*}"; lora="${item#*=}"
  LA=""; [ "$lora" != - ] && LA="--lora $lora"
  for cell in $CELLS; do
    out="outputs/week1/fs_${tag}_${cell}"
    [ -f "$out/summary.json" ] && continue
    if [ "$cell" = std ]; then
      bash "$HERE/run_py.sh" "$G" scripts/eval_libero.py --ckpt "pi05:$CK" $LA --suite "$SUITE" --trials 20 --num_envs 8 \
        --max_steps "$MS" --no_steps --fixture_seed --out "$out" > "logs/fs_${tag}_${cell}.log" 2>&1 || log "FAILED fs $tag $cell"
    else
      LIBERO_VARIANT=pro bash "$HERE/run_py.sh" "$G" scripts/eval_libero.py --ckpt "pi05:$CK" $LA --suite "${SUITE}_$cell" \
        --trials 20 --num_envs 8 --max_steps "$MS" --no_steps --fixture_seed --out "$out" > "logs/fs_${tag}_${cell}.log" 2>&1 \
        || log "FAILED fs $tag $cell"
    fi
    log "FS $tag $cell $(python3 -c "import json;s=json.load(open('$out/summary.json'));print(round(100*s['success_rate'],1), ''.join(str(min(9,int(10*v['success_rate']))) for k,v in sorted(s['per_task'].items(), key=lambda x:int(x[0]))))" 2>/dev/null)"
  done
done
log "FS_EVALS_DONE $SUITE"
