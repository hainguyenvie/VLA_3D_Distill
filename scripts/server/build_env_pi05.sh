#!/usr/bin/env bash
# Environment for the pi0.5 track (LeRobot PyTorch port of pi0.5), beside envs/oft: envs/pi05, the LIBERO-finetuned
# pi0.5 checkpoint (lerobot/pi05_libero_finetuned, PyTorch) and the LIBERO-PRO benchmark code.
# Launch detached: setsid nohup bash repo/scripts/server/build_env_pi05.sh > logs/build_env_pi05.log 2>&1 < /dev/null &
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
. "$HERE/env.sh"
cd "$W"
# venv on top of the oft environment (conda): robosuite / mujoco / LIBERO deps and the OSMesa libs come from there,
# lerobot and its torch live in the venv and shadow the older torch of the oft environment
if [ "${REBUILD:-0}" = 1 ] && [ -d envs/pi05 ]; then mv envs/pi05 "_trash_$(date +%Y%m%d)/envs_pi05_$(date +%H%M%S)" 2>/dev/null || rm -rf envs/pi05; fi
[ -x envs/pi05/bin/python ] || "$W/envs/oft/bin/python" -m venv --system-site-packages envs/pi05
envs/pi05/bin/pip install -q --upgrade pip
envs/pi05/bin/pip install -q lerobot "transformers>=4.53,<4.58" sentencepiece || { log "PI05_ENV_FAILED pip"; exit 1; }
envs/pi05/bin/python - <<'PY' || { log "PI05_ENV_FAILED import"; exit 1; }
import lerobot, torch, numpy, robosuite, mujoco
print("lerobot", lerobot.__version__, "torch", torch.__version__, "cuda", torch.cuda.is_available(), "numpy", numpy.__version__,
      "robosuite", robosuite.__version__, "mujoco", mujoco.__version__)
from lerobot.policies.pi05 import PI05Policy  # noqa: F401
PY
python3 "$HERE/hf_fetch.py" checkpoints lerobot/pi05_libero_finetuned || { log "PI05_ENV_FAILED fetch"; exit 1; }
[ -d third_party/LIBERO-PRO ] || git clone -q --depth 1 https://github.com/Zxy-MLlab/LIBERO-PRO third_party/LIBERO-PRO || log "LIBERO-PRO clone failed"
log "PI05_ENV_DONE"
