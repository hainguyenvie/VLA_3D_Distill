#!/usr/bin/env bash
# "Odd one out" probe: standard LIBERO-Object episodes in which one non-target object starts ~20 cm off its usual spot.
# A policy that learned "pick the object that is not where it belongs" (a shortcut the retarget counterfactual could
# teach, since there the target is always the displaced object) picks the displaced distractor.   pi05_oddone_probe.sh <gpu>
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
export PY_ENV=pi05
. "$HERE/env.sh"
cd "$W"
G="$1"; O=outputs/week1; CK="$W/checkpoints/lerobot__pi05_libero_finetuned"
for spec in base:p05_opd_base_s8 coshift:p05_ocd_coshift_s7 retarget:p05_ocd_retarget_s7 relocate:p05_ocd_relocate4cap_s7; do
  name="${spec%%:*}"; run="${spec#*:}"; A="$(ls -d $O/$run/adapter_iter* | sort | tail -1)"
  out="$O/oddone_${name}_steps"
  [ -f "$out/summary.json" ] || bash "$HERE/run_py.sh" "$G" scripts/eval_libero.py --ckpt "pi05:$CK" --lora "$A" --suite libero_object \
    --trials 10 --num_envs 8 --max_steps 280 --displace_distractor 0.2 --out "$out" || { log "FAILED $out"; continue; }
  bash "$HERE/run_py.sh" "$G" scripts/analyze_failures.py --run "$out" --suite libero_object || log "FAILED taxonomy $out"
done
log "ODDONE_PROBE_DONE"
