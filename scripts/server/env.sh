# Source this at the top of every job script on the server: . "$(dirname "$0")/env.sh"
# Layout: <workspace>/repo is the rsync mirror of this git repo; envs, weights, data, logs, outputs sit next to it.
# Everything (caches, configs, weights) stays inside the workspace; nothing is written to the shared home.
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
W="$(dirname "$REPO")"
export W REPO
export PY="$W/envs/${PY_ENV:-oft}/bin/python"  # PY_ENV=pi05 selects the pi0.5 environment
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
# LIBERO_VARIANT=plus swaps in LIBERO-Plus (same package name `libero`, perturbed tasks) with its own config dir
if [ "${LIBERO_VARIANT:-}" = plus ]; then
  LIBERO_DIR="$W/third_party/LIBERO-plus"
  export LD_LIBRARY_PATH="$W/envs/oft/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"  # ImageMagick (wand) lives in the env
  export LIBERO_CONFIG_PATH="$W/.libero_plus"
  if [ ! -f "$LIBERO_CONFIG_PATH/config.yaml" ]; then
    mkdir -p "$LIBERO_CONFIG_PATH"
    printf 'benchmark_root: %s\nbddl_files: %s/bddl_files\ninit_states: %s/init_files\ndatasets: %s\nassets: %s/assets\n' \
      "$LIBERO_DIR/libero/libero" "$LIBERO_DIR/libero/libero" "$LIBERO_DIR/libero/libero" "$W/data/libero_hdf5" "$LIBERO_DIR/libero/libero" \
      > "$LIBERO_CONFIG_PATH/config.yaml"
  fi
else
  LIBERO_DIR="$W/third_party/LIBERO"
fi
export PYTHONPATH="$LIBERO_DIR:$W/third_party/openvla-oft:$REPO${PYTHONPATH:+:$PYTHONPATH}"
# machine-specific settings live next to the repo mirror, outside git (e.g. the CPU renderer on the H200 machine)
[ -f "$W/machine.env" ] && . "$W/machine.env"
log() { echo "[$(date +%F\ %T%z)] $*"; }
