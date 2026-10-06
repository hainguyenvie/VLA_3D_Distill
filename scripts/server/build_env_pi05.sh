#!/usr/bin/env bash
# Environment for the pi0.5 track (LeRobot PyTorch port of pi0.5), beside envs/oft: envs/pi05, the LIBERO-finetuned
# pi0.5 checkpoint (lerobot/pi05_libero_finetuned, PyTorch) and the LIBERO-PRO benchmark code.
# Launch detached: setsid nohup bash repo/scripts/server/build_env_pi05.sh > logs/build_env_pi05.log 2>&1 < /dev/null &
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
. "$HERE/env.sh"
cd "$W"
[ -x envs/pi05/bin/python ] || "$W/envs/oft/bin/python" -m venv envs/pi05
envs/pi05/bin/pip install -q --upgrade pip
envs/pi05/bin/pip install -q "lerobot[pi]" || envs/pi05/bin/pip install -q lerobot || { log "PI05_ENV_FAILED pip"; exit 1; }
envs/pi05/bin/python -c "import lerobot, torch; print('lerobot', lerobot.__version__, 'torch', torch.__version__, 'cuda', torch.cuda.is_available())" || { log "PI05_ENV_FAILED import"; exit 1; }
python3 "$HERE/hf_fetch.py" checkpoints lerobot/pi05_libero_finetuned || { log "PI05_ENV_FAILED fetch"; exit 1; }
[ -d third_party/LIBERO-PRO ] || git clone -q --depth 1 https://github.com/Zxy-MLlab/LIBERO-PRO third_party/LIBERO-PRO || log "LIBERO-PRO clone failed"
log "PI05_ENV_DONE"
