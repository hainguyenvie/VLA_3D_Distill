#!/usr/bin/env bash
# Diagnostic: success of drawer tasks against the furniture's position. LIBERO re-places the furniture at every reset
# (the initial states do not hold it); --fixture_pin fixes it where a fresh env puts it, shifted by an offset (m).
#   fixture_sweep.sh <gpu> <suite> <max steps> <tasks> <tag> <lora or "-" for the base model> [offsets "dx,dy;dx,dy;..."]
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
export PY_ENV=pi05
. "$HERE/env.sh"
cd "$W"
G="$1"; SUITE="$2"; MS="$3"; TASKS="$4"; TAG="$5"; LORA="$6"
OFFS="${7:-0,0;0.02,0;-0.02,0;0,0.02;0,-0.02}"
CK="$W/checkpoints/lerobot__pi05_libero_finetuned"
LA=""; [ "$LORA" != - ] && LA="--lora $LORA"
IFS=';' read -ra LIST <<< "$OFFS"
for off in "${LIST[@]}"; do
  out="outputs/week1/fix_${TAG}_$(echo "$off" | tr ',.-' '_pm')"
  [ -f "$out/summary.json" ] && continue
  bash "$HERE/run_py.sh" "$G" scripts/eval_libero.py --ckpt "pi05:$CK" $LA --suite "$SUITE" --tasks "$TASKS" --trials 20 \
    --num_envs 8 --max_steps "$MS" --no_steps --fixture_pin --fixture_offset "$off" --out "$out" > "logs/fix_${TAG}.log" 2>&1 \
    || log "FAILED fixture sweep $TAG $off"
  log "FIXSWEEP $TAG $off $(python3 -c "import json;s=json.load(open('$out/summary.json'));print(s['success_rate'], {k: v['success_rate'] for k, v in s['per_task'].items()})" 2>/dev/null)"
done
log "FIXSWEEP_DONE $TAG"
