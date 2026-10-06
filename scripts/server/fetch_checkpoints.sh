#!/usr/bin/env bash
# Every checkpoint this project uses, downloaded on the machine itself (re-runnable; finished repos are skipped).
# Launch detached: setsid nohup bash repo/scripts/server/fetch_checkpoints.sh > logs/hf_fetch_all.log 2>&1 < /dev/null &
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
. "$HERE/env.sh"
cd "$W"
export HF_CONNS="${HF_CONNS:-8}" HF_FILES="${HF_FILES:-4}"
python3 "$HERE/hf_fetch.py" checkpoints \
  Haozhan72/Openvla-oft-SFT-libero-object-traj1 Haozhan72/Openvla-oft-SFT-libero-object-trajall \
  moojink/openvla-7b-oft-finetuned-libero-object lerobot/pi05_libero_finetuned \
  Haozhan72/Openvla-oft-SFT-libero10-traj1 Haozhan72/openvla-oft-libero10-traj1-rl \
  haofuly/spatial-forcing-7b-finetuned-libero-object
