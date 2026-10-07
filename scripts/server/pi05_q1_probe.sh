#!/usr/bin/env bash
# Shortcut probe for the rotation counterfactual: LIBERO-Object episodes that start with the arm turned about its first
# joint while the scene is not (what the rotation counterfactual never shows). A policy that learned "arm turned =>
# rotate the action" should fail more here than the others.   pi05_q1_probe.sh <gpu>
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
export PY_ENV=pi05
. "$HERE/env.sh"
cd "$W"
G="$1"; O=outputs/week1; CK="$W/checkpoints/lerobot__pi05_libero_finetuned"
for off in 0.25 -0.25; do
  for spec in pi05: base:p05_opd_base_s8 rotate:p05_ocd_rotate_s8 coshift:p05_ocd_coshift_s8 mix:p05_ocd_mix_s8; do
    name="${spec%%:*}"; run="${spec#*:}"
    lora=""; [ -n "$run" ] && lora="--lora $(ls -d $O/$run/adapter_iter* | sort | tail -1)"
    out="$O/q1probe_${name}_${off}"
    [ -f "$out/summary.json" ] || bash "$HERE/run_py.sh" "$G" scripts/eval_libero.py --ckpt "pi05:$CK" $lora --suite libero_object \
      --trials 10 --num_envs 8 --max_steps 280 --no_steps --q1_offset "$off" --out "$out" || log "FAILED $out"
  done
done
log "Q1_PROBE_DONE"
