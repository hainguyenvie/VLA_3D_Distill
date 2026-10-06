#!/usr/bin/env bash
# Diagnosis of pi0.5 before any training (one card): greedy rollouts with per-query states on LIBERO-Object and on the
# LIBERO-PRO position cells (swap, x-shift), failure taxonomy of each, and the counterfactual object-displacement
# probe (does the action chunk follow the target?). Usage: GPU=5 pi05_diag.sh
# Launch detached: GPU=5 setsid nohup bash repo/scripts/server/pi05_diag.sh > logs/pi05_diag.log 2>&1 < /dev/null &
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
export PY_ENV=pi05
. "$HERE/env.sh"
cd "$W"
G="${GPU:-5}"; O=outputs/week1; CK="pi05:$W/checkpoints/lerobot__pi05_libero_finetuned"
roll() {  # roll <variant> <suite> <out> <trials>
  local v="$1" s="$2" out="$3" n="$4"
  [ -f "$O/$out/summary.json" ] || LIBERO_VARIANT="$v" bash "$HERE/run_py.sh" "$G" scripts/eval_libero.py --ckpt "$CK" --suite "$s" \
    --trials "$n" --num_envs 10 --max_steps 280 --out "$O/$out" || { log "FAILED $out"; return 1; }
  [ -f "$O/$out/failures.csv" ] || LIBERO_VARIANT="$v" bash "$HERE/run_py.sh" "$G" scripts/analyze_failures.py --run "$O/$out" --suite "$s" || log "FAILED taxonomy $out"
}
roll "" libero_object pi05_object_steps 20
roll pro libero_object_swap pi05_pro_swap_steps 20
roll pro libero_object_temp pi05_pro_temp_steps 20
for r in pi05_object_steps:libero_object: pi05_pro_swap_steps:libero_object_swap:pro; do
  run="${r%%:*}"; rest="${r#*:}"; suite="${rest%%:*}"; v="${rest#*:}"
  [ -f "$O/cf_$run/summary.json" ] || LIBERO_VARIANT="$v" bash "$HERE/run_py.sh" "$G" scripts/probe_counterfactual.py --run "$O/$run" \
    --suite "$suite" --ckpt "$CK" --episodes_per_task 3 --states_per_episode 3 --num_envs 4 --out "$O/cf_$run" || log "FAILED cf $run"
done
log "PI05_DIAG_DONE"
