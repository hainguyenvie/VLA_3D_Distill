#!/usr/bin/env bash
# Laptop-side watcher of the 8xH200 workspace: every 5 minutes, print the milestone / result lines not seen before
# (training done, evaluations done, failures, new summary.json success rates). The first poll only records what is
# already there unless FIRST=1.   SEEN=<state file> bash scripts/watch_h8.sh
cd "$(dirname "$0")/.." && source infra.env
SEEN="${SEEN:-/tmp/watch_h8_seen.txt}"
touch "$SEEN"
FIRST=${FIRST:-0}
while true; do
  cur=$(timeout 90 ssh -o ConnectTimeout=30 $H8_SSH 'cd ~/'"$H8_REMOTE_ROOT"'; for f in logs/pi05_round*.log logs/pi05_arm_*.log logs/check_*.log logs/p05*_s[789].log; do [ -f $f ] && grep -HE "CHECK_[A-Z_]*(OK|FAILED)|TRAIN_DONE|EVALS_DONE|ROUND[0-9]+_DONE|[A-Z0-9_]+_DONE |PEEK_DONE|FAILED|Traceback|Killed|out of memory|exit [1-9]" $f | grep -v "exit 143" | sed "s/^logs\///"; done; for d in outputs/week1/p05*_s[789]_* outputs/week1/plus_p05*_s[789]_robotlayout outputs/week1/q1probe_* outputs/week1/plusrobot_*; do [ -f $d/summary.json ] && echo "RESULT $(basename $d) $(python3 -c "import json;s=json.load(open(\"$d/summary.json\"));print(round(100*s[\"success_rate\"],1) if \"success_rate\" in s else {k[:5]: round(100*v[\"success_rate\"]) for k,v in s[\"by_category\"].items()})" 2>/dev/null)"; done; echo END_OF_LIST' 2>/dev/null)
  if echo "$cur" | grep -q END_OF_LIST; then
    new=$(echo "$cur" | grep -v END_OF_LIST | grep -Fxv -f "$SEEN")
    if [ -n "$new" ]; then
      [ "$FIRST" = 1 ] && echo "$new"
      echo "$new" >> "$SEEN"
    fi
    FIRST=1
  fi
  sleep 300
done
