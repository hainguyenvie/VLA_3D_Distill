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
envs/pi05/bin/pip install -q lerobot sentencepiece || { log "PI05_ENV_FAILED pip"; exit 1; }
# pi0.5 refuses PyPI transformers ("incorrect transformer version"): lerobot's pyproject pins this fork branch;
# numpy back to 1.26 (lerobot pulls numpy 2, which breaks the robosuite / opencv of the shared site-packages)
envs/pi05/bin/pip install -q "transformers @ git+https://github.com/huggingface/transformers.git@fix/lerobot_openpi" \
  "scipy>=1.10.1,<1.15" "numpy==1.26.4" || { log "PI05_ENV_FAILED transformers fork"; exit 1; }
envs/pi05/bin/python - <<'PY' || { log "PI05_ENV_FAILED import"; exit 1; }
import lerobot, torch, numpy, robosuite, mujoco
print("lerobot", lerobot.__version__, "torch", torch.__version__, "cuda", torch.cuda.is_available(), "numpy", numpy.__version__,
      "robosuite", robosuite.__version__, "mujoco", mujoco.__version__)
from lerobot.policies.pi05 import PI05Policy  # noqa: F401
from transformers.models.siglip import check
assert check.check_whether_transformers_replace_is_installed_correctly()
PY
python3 "$HERE/hf_fetch.py" checkpoints lerobot/pi05_libero_finetuned || { log "PI05_ENV_FAILED fetch"; exit 1; }
# the PaliGemma tokenizer is gated: an ungated copy of the same Gemma tokenizer files, BOS on, right padding
T=checkpoints/paligemma_tokenizer_fallback
if [ ! -s "$T/tokenizer.json" ]; then
  mkdir -p "$T"
  for f in tokenizer.model tokenizer_config.json special_tokens_map.json tokenizer.json; do
    curl -sSL -o "$T/$f" "https://huggingface.co/CWKSC/paligemma-cord-demo/resolve/main/$f" || { log "PI05_ENV_FAILED tokenizer"; exit 1; }
  done
  envs/pi05/bin/python -c "
import json; p='$T/tokenizer_config.json'; d=json.load(open(p)); d.update(add_bos_token=True, add_eos_token=False, padding_side='right'); json.dump(d, open(p, 'w'))"
fi
[ -d third_party/LIBERO-PRO ] || git clone -q --depth 1 https://github.com/Zxy-MLlab/LIBERO-PRO third_party/LIBERO-PRO || log "LIBERO-PRO clone failed"
log "PI05_ENV_DONE"
