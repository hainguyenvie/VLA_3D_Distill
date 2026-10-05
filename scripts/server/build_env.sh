#!/usr/bin/env bash
# One-time environment build for the workspace (LIBERO + OpenVLA-OFT). Re-runnable: finished steps are skipped.
# Usage (on the server, from the workspace): setsid nohup bash repo/scripts/server/build_env.sh > logs/build_env.log 2>&1 < /dev/null &
# Network facts behind the layout (05/10): HF, conda-forge, download.pytorch.org are fast; PyPI and GitHub
# are throttled per connection (30-100 KB/s), so torch comes from the pytorch index, every other wheel is
# resolved by a pip dry run and fetched with parallel range requests, and GitHub repos come as shallow
# clones / zip archives.
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
. "$HERE/env.sh"
mkdir -p "$W"/{envs,checkpoints/hf,data,logs,outputs,third_party,.cache/wheels,.libero}
TP="$W/third_party"
die() { log "$1"; exit 1; }

cloned() { git -C "$1" rev-parse -q --verify HEAD > /dev/null 2>&1 && [ ! -f "$1/.git/index.lock" ] && [ -n "$(ls "$1" 2>/dev/null)" ]; }
clone_bg() {  # clone_bg <url> <dir>: shallow clone in the background unless present or already being cloned
  [ -d "$TP/$2/.git" ] || (cd "$TP" && git clone -q --depth 1 "$1" "$2" > "$W/logs/clone_$2.log" 2>&1 &)
}
clone_bg https://github.com/moojink/openvla-oft.git openvla-oft
clone_bg https://github.com/Lifelong-Robot-Learning/LIBERO.git LIBERO
clone_bg https://github.com/PRIME-RL/SimpleVLA-RL.git SimpleVLA-RL

if [ ! -x "$PY" ]; then
  log "creating conda env"
  if [ -x "$HOME/miniconda3/bin/conda" ]; then  # L40 machine
    "$HOME/miniconda3/bin/conda" create -y -q -p "$W/envs/oft" -c conda-forge --override-channels python=3.10 pip > /dev/null || die CONDA_FAILED
  else  # H200 machine: micromamba
    "$HOME/.local/bin/micromamba" create -y -q -p "$W/envs/oft" -c conda-forge python=3.10 pip > /dev/null || die CONDA_FAILED
  fi
fi
$PY -m pip --version > /dev/null 2>&1 || $PY -m ensurepip --upgrade > /dev/null || die PIP_FAILED
# CPU renderer (OSMesa, removed from Mesa >= 25.1) for MUJOCO_GL=osmesa; selected per machine in <workspace>/machine.env
if [ ! -e "$W/envs/oft/lib/libOSMesa.so" ]; then
  log "installing OSMesa"
  if [ -x "$HOME/miniconda3/bin/conda" ]; then
    "$HOME/miniconda3/bin/conda" install -y -q -p "$W/envs/oft" -c conda-forge --override-channels "mesalib=24" > /dev/null || die OSMESA_FAILED
  else
    "$HOME/.local/bin/micromamba" install -y -q -p "$W/envs/oft" -c conda-forge "mesalib=24" > /dev/null || die OSMESA_FAILED
  fi
fi
# the bundled pip 23.0 downloads whole wheels just to resolve; newer pip reads the PEP 658 metadata files
$PY -c "import pip, sys; sys.exit(int(pip.__version__.split('.')[0]) < 25)" || $PY -m pip install -q --upgrade pip || die PIP_UPGRADE_FAILED
PIP="$PY -m pip"
if ! $PY -c "import torch" 2>/dev/null; then
  log "installing torch + CUDA runtime wheels (pytorch index, no deps)"
  $PIP install -q --no-deps --only-binary :all: --index-url https://download.pytorch.org/whl/cu121 \
    torch==2.2.0+cu121 torchvision==0.17.0+cu121 torchaudio==2.2.0+cu121 triton==2.2.0 \
    nvidia-cuda-nvrtc-cu12==12.1.105 nvidia-cuda-runtime-cu12==12.1.105 nvidia-cuda-cupti-cu12==12.1.105 \
    nvidia-cudnn-cu12==8.9.2.26 nvidia-cublas-cu12==12.1.3.1 nvidia-cufft-cu12==11.0.2.54 nvidia-curand-cu12==10.3.2.106 \
    nvidia-cusolver-cu12==11.4.5.107 nvidia-cusparse-cu12==12.1.0.106 nvidia-nccl-cu12==2.19.3 nvidia-nvtx-cu12==12.1.105 \
    nvidia-nvjitlink-cu12==12.1.105 || die TORCH_FAILED
fi

# GitHub archives are small but single-connection: start them now, use them at the end
WH="$W/.cache/wheels"; ZIPS="$W/.cache/zips"; mkdir -p "$WH" "$ZIPS"
branch() { git ls-remote --symref "https://github.com/$1.git" HEAD | awk '/^ref:/{sub("refs/heads/","",$2); print $2}'; }
zip_bg() {  # zip_bg <owner/repo> <name>
  [ -s "$ZIPS/$2.zip" ] && return
  pgrep -u "$USER" -f "curl.*$ZIPS/$2.zip.part" > /dev/null && return  # an earlier run is still downloading it
  (curl -sL --retry 5 -o "$ZIPS/$2.zip.part" "https://github.com/$1/archive/refs/heads/$(branch "$1").zip" && mv "$ZIPS/$2.zip.part" "$ZIPS/$2.zip") &
}
zip_bg moojink/transformers-openvla-oft transformers
zip_bg moojink/dlimp_openvla dlimp

cat > "$W/.cache/constraints.txt" <<'TXT'
torch==2.2.0
torchvision==0.17.0
torchaudio==2.2.0
numpy==1.26.4
TXT
if [ ! -f "$W/.cache/deps_installed" ]; then
  [ -s "$W/.cache/pip_report.json" ] || { log "resolving python deps (pip dry run)"; $PIP install -q --dry-run --report "$W/.cache/pip_report.json.tmp" -c "$W/.cache/constraints.txt" \
    filelock typing-extensions sympy networkx jinja2 fsspec pillow requests safetensors regex packaging pyyaml tqdm \
    tensorflow==2.15.0 tensorflow_datasets==4.9.3 tensorflow_graphics==2021.12.3 tqdm-multiprocess \
    "accelerate>=0.25.0" draccus==0.8.0 einops huggingface_hub json-numpy jsonlines matplotlib peft==0.11.1 protobuf rich \
    sentencepiece==0.1.99 timm==0.9.10 tokenizers==0.19.1 wandb diffusers==0.30.3 "imageio[ffmpeg]" uvicorn fastapi \
    robosuite==1.4.1 bddl easydict cloudpickle gym h5py pandas scipy && mv "$W/.cache/pip_report.json.tmp" "$W/.cache/pip_report.json" || die RESOLVE_FAILED; }
  log "fetching wheels (parallel ranges)"
  $PY "$HERE/fetch_report.py" "$W/.cache/pip_report.json" "$WH" "$W/.cache/pinned.txt" || die FETCH_FAILED
  log "installing python deps from local wheels"
  $PIP install -q --no-index --find-links "$WH" --no-build-isolation -r "$W/.cache/pinned.txt" || die DEPS_FAILED
  touch "$W/.cache/deps_installed"
fi
$PY -c "import torch; assert torch.cuda.is_available(); print('torch', torch.__version__)" || die TORCH_IMPORT_FAILED

# Pins that override what the resolver picked (installed without deps, after everything else):
#  mujoco 3.2.3            robosuite 1.4.1 breaks on recent mujoco (3.14: AssertionError in get_joint_qpos_addr
#                          when the env is built); 3.2.3 is what other LIBERO setups of early 2025 pin
#  tensorflow-metadata 1.15.0  newer releases ship protobuf-5 generated code, TF 2.15 needs protobuf < 5
PINS="mujoco==3.2.3 tensorflow-metadata==1.15.0"
PIN_MARK="$W/.cache/pins_$(echo "$PINS" | md5sum | cut -c1-8)"
if [ ! -f "$PIN_MARK" ]; then
  log "installing version pins: $PINS"
  $PIP install -q --dry-run --no-deps --ignore-installed --report "$W/.cache/pins_report.json" $PINS || die PIN_RESOLVE_FAILED
  $PY "$HERE/fetch_report.py" "$W/.cache/pins_report.json" "$WH" "$W/.cache/pins.txt" || die PIN_FETCH_FAILED
  $PIP install -q --no-deps --no-index --find-links "$WH" -r "$W/.cache/pins.txt" || die PIN_INSTALL_FAILED
  touch "$PIN_MARK"
fi

log "waiting for GitHub archives"
until [ -s "$ZIPS/transformers.zip" ] && [ -s "$ZIPS/dlimp.zip" ]; do sleep 10; done
if [ ! -f "$W/.cache/transformers_fork_installed" ]; then
  # must replace the PyPI transformers pulled in as a dependency: only the fork has bidirectional attention
  log "installing transformers fork (zip archive)"
  $PIP install -q --no-deps --no-index --no-build-isolation --force-reinstall "$ZIPS/transformers.zip" || die TRANSFORMERS_FAILED
  touch "$W/.cache/transformers_fork_installed"
fi
if ! $PY -c "import dlimp" 2>/dev/null; then
  log "installing dlimp (zip archive)"
  $PIP install -q --no-deps --no-index --no-build-isolation "$ZIPS/dlimp.zip" || die DLIMP_FAILED
fi

log "waiting for clones"
for d in openvla-oft LIBERO SimpleVLA-RL; do
  until cloned "$TP/$d"; do sleep 20; done
  log "$d @ $(git -C "$TP/$d" rev-parse --short HEAD)"
done
$PIP install -q --no-deps --no-index --no-build-isolation -e "$TP/openvla-oft" || die OFT_FAILED
$PIP install -q --no-deps --no-index --no-build-isolation -e "$TP/LIBERO" || log "LIBERO editable install failed (PYTHONPATH fallback is set in env.sh)"

# LIBERO asks interactive questions on first import unless its config already exists
if [ ! -f "$LIBERO_CONFIG_PATH/config.yaml" ]; then
  L="$TP/LIBERO/libero/libero"
  cat > "$LIBERO_CONFIG_PATH/config.yaml" <<YAML
benchmark_root: $L
bddl_files: $L/bddl_files
init_states: $L/init_files
datasets: $W/data/libero_hdf5
assets: $L/assets
YAML
fi

$PY - <<'PY' || die SANITY_FAILED
import torch, transformers, numpy
print("torch", torch.__version__, "cuda", torch.cuda.is_available(), torch.cuda.device_count(), "transformers", transformers.__version__, "numpy", numpy.__version__)
import tensorflow as tf
print("tensorflow", tf.__version__)
import robosuite, mujoco
print("robosuite", robosuite.__version__, "mujoco", mujoco.__version__)
import dlimp, tensorflow_datasets  # noqa: F401  (RLDS loader chain; catches protobuf mismatches)
import experiments.robot.openvla_utils  # noqa: F401  (everything the policy wrapper imports)
from libero.libero import benchmark
print("libero suites", sorted(benchmark.get_benchmark_dict().keys()))
PY
log "BUILD_DONE"
