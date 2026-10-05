# Source this at the top of every job script on the server: . "$(dirname "$0")/env.sh"
# Layout: <workspace>/repo is the rsync mirror of this git repo; envs, weights, data, logs, outputs sit next to it.
# Everything (caches, configs, weights) stays inside the workspace; nothing is written to the shared home.
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
W="$(dirname "$REPO")"
export W REPO
export PY="$W/envs/oft/bin/python"
export HF_HOME="$W/checkpoints/hf"
export TORCH_HOME="$W/checkpoints/torch"
export PIP_CACHE_DIR="$W/.cache/pip"
export XDG_CACHE_HOME="$W/.cache"
export TFDS_DATA_DIR="$W/data/tfds"
export LIBERO_CONFIG_PATH="$W/.libero"
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
# 96 cores, shared: without these every process starts ~96 BLAS / OpenMP / TF threads (load average in the hundreds)
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export TF_NUM_INTRAOP_THREADS=2 TF_NUM_INTEROP_THREADS=2
export PYTHONPATH="$W/third_party/LIBERO:$W/third_party/openvla-oft:$REPO${PYTHONPATH:+:$PYTHONPATH}"
# machine-specific settings live next to the repo mirror, outside git (e.g. the CPU renderer on the H200 machine)
[ -f "$W/machine.env" ] && . "$W/machine.env"
log() { echo "[$(date +%F\ %T%z)] $*"; }
