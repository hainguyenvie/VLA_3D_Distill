#!/usr/bin/env bash
# Move results from the retired machines (1x H200, 2x L40) to the 8x H200 node, through the laptop (the servers
# are never connected to each other).   scripts/relay_outputs.sh pull | push      (re-runnable: rsync skips what is there)
#   pull: small result files (json/jsonl/csv/txt, no rollout arrays), logs, the adapters listed below and the
#         rollouts the probes need -> $RELAY/{h200,l40}
#   push: $RELAY/{h200,l40}/outputs -> <tensara workspace>/outputs ; logs -> logs/{h200,l40}/
set -euo pipefail
cd "$(dirname "$0")/.."
. ./infra.env
RELAY="${RELAY:-$HOME/h2n/vla3d_relay}"
O=outputs/week1
H200_ADAPTERS="b2_student_states_rkl_s7/adapter_iter0020 b2p_teacher_states_rkl_s7/adapter_iter0020
  b3_teacher_states_depth_rkl_s7/adapter_iter0020 b4_student_states_depth_rkl_s7/adapter_iter0020
  cos_b2_student_states_rkl_s7/adapter_iter0020 cos_b3_teacher_states_depth_rkl_s7/adapter_iter0020
  long_b2_student_states_rkl_s7/adapter_iter0040 v1_mixed_viewaug_rkl_s7/adapter_iter0020
  v1_mixed_viewaug_rkl_s8/adapter_iter0020 oft_a2_viewaug_s7/adapter_iter0004 ${EXTRA_H200:-}"
H200_ROLLOUTS="plus_oft_steps"
L40_ADAPTERS="v2_mixed_viewaug_depth_rkl_s7/adapter_iter0020 cos_v1_mixed_viewaug_rkl_s7/adapter_iter0018
  cos_v1_mixed_viewaug_rkl_s7/adapter_iter0020 ${EXTRA_L40:-}"
L40_ROLLOUTS="b0_object_student plus_fullsft_steps probe_offset_teacher"
SMALL=(--include='*/' --include='*.json' --include='*.jsonl' --include='*.csv' --include='*.txt' --include='*.md' --exclude='*')

pull() {  # pull <ssh alias> <remote root> <label> <adapters> <rollouts>
  local ssh="$1" root="$2" dst="$RELAY/$3" a r
  mkdir -p "$dst/$O" "$dst/logs"
  rsync -a --prune-empty-dirs "${SMALL[@]}" --exclude='adapter_*/' --exclude='steps/' "$ssh:$root/$O/" "$dst/$O/"
  rsync -a "$ssh:$root/logs/" "$dst/logs/"
  for a in $4; do mkdir -p "$dst/$O/$(dirname "$a")"; rsync -a "$ssh:$root/$O/$a" "$dst/$O/$(dirname "$a")/"; echo "pulled $3:$a"; done
  for r in $5; do rsync -a "$ssh:$root/$O/$r" "$dst/$O/"; echo "pulled $3:$r"; done
}
push() {  # push <label>
  local src="$RELAY/$1" root="$TENSARA_REMOTE_ROOT" rs="--rsync-path=$TENSARA_REMOTE_ROOT/envs/tools/bin/rsync"
  ssh -n "$TENSARA_SSH" "mkdir -p $root/$O $root/logs/$1"
  rsync -a "$rs" "$src/$O/" "$TENSARA_SSH:$root/$O/"
  rsync -a "$rs" "$src/logs/" "$TENSARA_SSH:$root/logs/$1/"
  echo "pushed $1"
}
case "${1:-}" in
  pull) pull "$H200_SSH" "$H200_REMOTE_ROOT" h200 "$H200_ADAPTERS" "$H200_ROLLOUTS"
        pull "$L40_SSH" "$L40_REMOTE_ROOT" l40 "$L40_ADAPTERS" "$L40_ROLLOUTS"; echo RELAY_PULL_DONE ;;
  push) push h200; push l40; echo RELAY_PUSH_DONE ;;
  *) echo "usage: scripts/relay_outputs.sh pull|push" >&2; exit 2 ;;
esac
