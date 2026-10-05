#!/usr/bin/env bash
# One-time setup of LIBERO-Plus next to LIBERO (selected at run time with LIBERO_VARIANT=plus). Re-runnable.
# Launch detached: setsid nohup bash repo/scripts/server/setup_libero_plus.sh > logs/setup_libero_plus.log 2>&1 < /dev/null &
set -uo pipefail
. "$(dirname "$0")/env.sh"
P="$W/third_party/LIBERO-plus"; L="$P/libero/libero"
[ -d "$P/.git" ] || git clone -q --depth 1 https://github.com/sylvestf/LIBERO-plus.git "$P" || { log CLONE_FAILED; exit 1; }
if [ ! -d "$L/assets/new_objects" ]; then
  mkdir -p "$W/data/libero_plus"
  Z="$W/data/libero_plus/assets.zip"
  [ -s "$Z" ] || { curl -sL --retry 5 -o "$Z.part" https://huggingface.co/datasets/Sylvest/LIBERO-plus/resolve/main/assets.zip && mv "$Z.part" "$Z"; } || { log ASSETS_FAILED; exit 1; }
  (cd "$L" && unzip -q -o "$Z") || { log UNZIP_FAILED; exit 1; }
  # the archive carries the authors' absolute path: find the assets folder inside and put it where LIBERO expects it
  A="$(find "$L" -mindepth 2 -type d -name assets -path '*LIBERO-plus-0*' | head -1)"
  [ -n "$A" ] && rm -rf "$L/assets" && mv "$A" "$L/assets" && rm -rf "$L/inspire"
  [ -d "$L/assets/new_objects" ] || { log ASSETS_LAYOUT_FAILED; exit 1; }
  rm -f "$Z"
fi
if ! LD_LIBRARY_PATH="$W/envs/oft/lib" "$PY" -c "from wand.image import Image; import skimage" 2>/dev/null; then
  if [ -x "$HOME/miniconda3/bin/conda" ]; then
    "$HOME/miniconda3/bin/conda" install -y -q -p "$W/envs/oft" -c conda-forge --override-channels imagemagick > /dev/null || { log IMAGEMAGICK_FAILED; exit 1; }
  else
    "$HOME/.local/bin/micromamba" install -y -q -p "$W/envs/oft" -c conda-forge imagemagick > /dev/null || { log IMAGEMAGICK_FAILED; exit 1; }
  fi
  "$PY" -m pip install -q wand scikit-image || { log PIP_FAILED; exit 1; }
fi
LD_LIBRARY_PATH="$W/envs/oft/lib" "$PY" -c "from wand.image import Image; import skimage, torch, numpy; print('wand + skimage ok; torch', torch.__version__, 'numpy', numpy.__version__)" || { log IMPORT_FAILED; exit 1; }
log "SETUP_PLUS_DONE"
