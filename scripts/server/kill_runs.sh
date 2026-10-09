#!/usr/bin/env bash
# Kill the processes of the named runs (their pi05_arm.sh, run_py.sh and python): matched on the run's output directory or
# arm arguments in /proc/<pid>/cmdline, never this script, its parent or its children.  bash scratch/kill_runs.sh <run> ...
me=$$; pp=$PPID
for run in "$@"; do
  for p in /proc/[0-9]*; do
    pid=${p#/proc/}
    [ "$pid" = "$me" ] || [ "$pid" = "$pp" ] && continue
    [ -O "$p" ] || continue
    cmd=$(tr '\0' ' ' < "$p/cmdline" 2>/dev/null) || continue
    case "$cmd" in
      *kill_runs.sh*) continue ;;
      *"pi05_arm.sh "*" $run "*|*"/$run "*|*"/$run") echo "kill $pid: ${cmd:0:120}"; kill "$pid" 2>/dev/null ;;
    esac
  done
done
