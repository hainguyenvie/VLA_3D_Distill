#!/usr/bin/env bash
# LIBERO-PRO benchmark code (third_party/LIBERO-PRO) plus its official bddl / init files from the HF dataset
# zhouxueyang/LIBERO-Pro, and the position-perturbation suite `libero_object_temp` (x 0.3 shift).
# Launch detached: setsid nohup bash repo/scripts/server/setup_libero_pro.sh > logs/setup_libero_pro.log 2>&1 < /dev/null &
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
. "$HERE/env.sh"
cd "$W"
[ -d third_party/LIBERO-PRO/.git ] || git clone -q --depth 1 https://github.com/Zxy-MLlab/LIBERO-PRO third_party/LIBERO-PRO || { log "PRO_FAILED clone"; exit 1; }
python3 "$HERE/hf_fetch.py" --dataset checkpoints zhouxueyang/LIBERO-Pro || { log "PRO_FAILED fetch"; exit 1; }
D=checkpoints/zhouxueyang__LIBERO-Pro; P=third_party/LIBERO-PRO/libero/libero
for sub in bddl_files init_files; do
  for d in "$D/$sub"/*/; do n=$(basename "$d"); mkdir -p "$P/$sub/$n"; cp -rn "$d"* "$P/$sub/$n/" 2>/dev/null; done
  [ -d "$P/$sub/libero_object_temp" ] || { mkdir -p "$P/$sub/libero_object_temp"; cp -r "$P/$sub/libero_object_temp_x0.3/"* "$P/$sub/libero_object_temp/"; }
done
log "PRO_SETUP_DONE"
