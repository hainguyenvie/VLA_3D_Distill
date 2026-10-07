#!/usr/bin/env bash
# Usage: DRY=1 bash cleanup_workspace.sh <workspace>   (DRY=0 deletes). The lists below are the 07/10 cleanup.
# Workspace cleanup. DRY=1 (default) only lists what would go and its size. Keeps every log, summary.json,
# episodes.jsonl, failures.*, train_log.jsonl, config. Never touches runs that are still training.
set -uo pipefail
cd "$1" || exit 1
DRY="${DRY:-1}"
O=outputs/week1
LIST=$(mktemp)
add() { for p in "$@"; do [ -e "$p" ] && echo "$p" >> "$LIST"; done; }
running="p05sp_ocd_coshift_post_s7 p05_ocd_coshift_both_s7"   # round 10, still training

# 1. abandoned or superseded directions: all weights and optimizer state (logs and summaries stay)
for r in p05_ocd_swap_s7 p05_ocd_swap_gentle_s7 p05_ocd_swap_offpolicy_s7 p05_ocd_shift_s7 \
         p05_ocd_mirror_s7 p05_ocd_mirror_s8 p05_ocd_mirror_early_s7 \
         oft_spatial_a1_clean_s7 oft_spatial_a2_viewaug_s7 oft_goal_a1_clean_s7 oft_goal_a2_viewaug_s7 \
         oft_10_a1_clean_s7 oft_10_a2_viewaug_s7 \
         cos_b2_student_states_rkl_s7 cos_b2p_teacher_states_rkl_s7 long_cos_b2_student_states_rkl_s7; do
  add $O/$r/adapter_* $O/$r/state.pt $O/$r/*.pt
done
add $O/smoke_oft_a1_clean_s7 $O/smoke_oft_a2_viewaug_s7
# 2. finished runs that are kept: optimizer state, the duplicate adapter_last, and intermediate adapters
#    (pi0.5: keep iterations 8, 12 and the last; OFT: keep the last)
for d in $O/p05*_s[789] $O/oft_a[12]_*_s[78]; do
  r=$(basename "$d"); case " $running " in *" $r "*) continue ;; esac
  [ -d "$d" ] || continue
  add "$d/state.pt" "$d/adapter_last"
  last=$(ls -d "$d"/adapter_iter* 2>/dev/null | sort | tail -1)
  for a in "$d"/adapter_iter*; do
    [ -e "$a" ] || continue
    case "$r" in p05*) keep="adapter_iter0008 adapter_iter0012" ;; *) keep="" ;; esac
    [ "$a" = "$last" ] && continue
    case " $keep " in *" $(basename "$a") "*) continue ;; esac
    add "$a"
  done
done
# 3. per-query rollout states of analyses already done (keep the pi0.5 baselines used by the gates / diagnosis)
for s in $O/*/steps; do
  case "$s" in $O/pi05_object_steps/steps|$O/pi05_spatial_pro_swap_steps/steps|$O/pi05_pro_swap_steps/steps|$O/pi05_pro_temp_steps/steps) continue ;; esac
  [ -f "$(dirname "$s")/summary.json" ] || continue  # an evaluation still writing
  add "$s"
done
# 4. outputs relayed from the retired machines: weights and rollout states only
find outputs/old_h200 outputs/old_l40 -type f \( -name "*.safetensors" -o -name "*.pt" -o -name "*.bin" -o -name "*.npz" \) >> "$LIST" 2>/dev/null
# 5. base checkpoints of directions not pursued (all public on HuggingFace, scripts/server/fetch_checkpoints.sh)
add checkpoints/Haozhan72__Openvla-oft-SFT-libero-object-traj1 checkpoints/Haozhan72__Openvla-oft-SFT-libero-object-trajall \
    checkpoints/Haozhan72__Openvla-oft-SFT-libero10-traj1 checkpoints/Haozhan72__openvla-oft-libero10-traj1-rl \
    checkpoints/haofuly__spatial-forcing-7b-finetuned-libero-object

sort -u "$LIST" -o "$LIST"
echo "items: $(wc -l < "$LIST")"
du -sch $(cat "$LIST") 2>/dev/null | tail -1
awk -F/ '{print $1"/"$2"/"$3}' "$LIST" | sort | uniq -c | sort -rn | head -60
if [ "$DRY" = 0 ]; then
  xargs -d '\n' rm -rf < "$LIST"
  echo "DELETED"; df -h . | tail -1
fi
rm -f "$LIST"
