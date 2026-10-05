#!/usr/bin/env bash
# One-time environment build for the workspace (LIBERO + OpenVLA-OFT). Re-runnable: finished steps are skipped.
# Usage (on the server, from the workspace): setsid nohup bash repo/scripts/server/setup_env.sh > logs/setup_env.log 2>&1 < /dev/null &
set -uo pipefail
. "$(dirname "$0")/env.sh"
mkdir -p "$W"/{envs,checkpoints/hf,data,logs,outputs,third_party,.cache,.libero}
cd "$W/third_party"

clone() {  # clone <url> <dir>
  [ -d "$2/.git" ] || git clone -q "$1" "$2" || { log "CLONE_FAILED $1"; exit 1; }
  log "$2 @ $(git -C "$2" rev-parse --short HEAD)"
}
clone https://github.com/moojink/openvla-oft.git openvla-oft
clone https://github.com/Lifelong-Robot-Learning/LIBERO.git LIBERO
clone https://github.com/PRIME-RL/SimpleVLA-RL.git SimpleVLA-RL

if [ ! -x "$PY" ]; then
  log "creating conda env"
  "$HOME/miniconda3/bin/conda" create -y -q -p "$W/envs/oft" -c conda-forge --override-channels python=3.10 || { log "CONDA_FAILED"; exit 1; }
fi
PIP="$PY -m pip"
$PIP install -q --upgrade pip
log "installing torch"
$PIP install -q torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 || { log "TORCH_FAILED"; exit 1; }
log "installing openvla-oft"
$PIP install -q -e "$W/third_party/openvla-oft" || { log "OFT_FAILED"; exit 1; }
log "installing LIBERO"
$PIP install -q -r "$W/third_party/openvla-oft/experiments/robot/libero/libero_requirements.txt" || { log "LIBERO_REQ_FAILED"; exit 1; }
$PIP install -q -e "$W/third_party/LIBERO" || log "LIBERO editable install failed (PYTHONPATH fallback is set in env.sh)"

# LIBERO asks interactive questions on first import unless its config already exists
if [ ! -f "$LIBERO_CONFIG_PATH/config.yaml" ]; then
  L="$W/third_party/LIBERO/libero/libero"
  cat > "$LIBERO_CONFIG_PATH/config.yaml" <<YAML
benchmark_root: $L
bddl_files: $L/bddl_files
init_states: $L/init_files
datasets: $W/data/libero_hdf5
assets: $L/assets
YAML
fi

$PY - <<'PY'
import torch, transformers, numpy
print("torch", torch.__version__, "cuda", torch.cuda.is_available(), torch.cuda.device_count(), "transformers", transformers.__version__, "numpy", numpy.__version__)
import robosuite, mujoco
print("robosuite", robosuite.__version__, "mujoco", mujoco.__version__)
from libero.libero import benchmark
print("libero suites", sorted(benchmark.get_benchmark_dict().keys()))
PY
log "SETUP_DONE"
