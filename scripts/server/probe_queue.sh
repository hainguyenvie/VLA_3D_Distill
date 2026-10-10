#!/usr/bin/env bash
# Sequential mechanism probes (scripts/probe_mechanism.py) on one card, at the logged states of a steps directory.
#   probe_queue.sh <gpu> <suite> <steps dir> "<tag>=<ckpt spec>,<lora dir or ->,<noise seed or -1>" ...
# e.g. probe_queue.sh 3 libero_object outputs/week1/pi05_object_steps "base_n0=pi05:<ckpt>,-,0" "oft=oft:<ckpt>,-,-1"
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
. "$HERE/env.sh"
cd "$W"
G="$1"; SUITE="$2"; STEPS="$3"; shift 3
for item in "$@"; do
  tag="${item%%=*}"; rest="${item#*=}"
  IFS=',' read -r ckpt lora nseed <<< "$rest"
  out="outputs/week1/mech2_${SUITE#libero_}_$tag"
  [ -f "$out/summary.json" ] && continue
  case "$ckpt" in oft:*) export PY_ENV=oft ;; *) export PY_ENV=pi05 ;; esac
  LA=""; [ "$lora" != - ] && LA="--lora $lora"
  bash "$HERE/run_py.sh" "$G" scripts/probe_mechanism.py --steps "$STEPS" --suite "$SUITE" --ckpt "$ckpt" $LA \
    --noise_seed "$nseed" --out "$out" > "logs/mech2_$tag.log" 2>&1 || log "FAILED probe $tag"
  log "PROBE $tag $(grep -A1 '^carrying' logs/mech2_$tag.log | tail -1 | cut -c1-120)"
done
log "PROBE_QUEUE_DONE $SUITE"
