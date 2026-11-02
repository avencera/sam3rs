#!/usr/bin/env bash
# Runs the Python package's tests against several releases of the nvidia-cublas wheel the
# `cuda` extra accepts: by default the oldest 13.x, the one PyTorch's cu130 wheels pin and
# the newest. On a machine with a CUDA toolkit whose cuBLAS differs from the wheel's, it
# also checks that the wheel's copy is the only one loaded. Needs an NVIDIA GPU, the
# weights in models/sam3 and the parity fixtures in models/parity
# (scripts/fetch-models.sh, scripts/reference.py dump)
#
#   scripts/check-cublas-versions.sh                       # the default releases
#   scripts/check-cublas-versions.sh 13.2.0.9 13.4.0.1     # these releases
set -euo pipefail

cd "$(dirname "$0")/.."
root=$PWD
work=target/cublas-versions
mkdir -p "$work"

versions=("$@")
if [[ ${#versions[@]} -eq 0 ]]; then
  # "latest" resolves to the newest release the extra allows
  versions=(13.0.0.19 13.1.1.3 latest)
fi

# pytest runs from crates/sam3rs-python, so a relative model directory is resolved here
SAM3RS_MODEL_DIR=$(realpath -m "${SAM3RS_MODEL_DIR:-models/sam3}")
export SAM3RS_MODEL_DIR
# the GPU tests skip without these, which would count as a pass here
for file in "$SAM3RS_MODEL_DIR/model.safetensors" models/parity/dogs.safetensors; do
  if [[ ! -f $file ]]; then
    echo "missing $file; run scripts/fetch-models.sh and scripts/reference.py dump" >&2
    exit 2
  fi
done

rm -rf "$work/wheel"
uvx maturin build --release -m crates/sam3rs-python/Cargo.toml -o "$work/wheel"
wheel=$(ls "$work"/wheel/sam3rs-*.whl)

for version in "${versions[@]}"; do
  venv="$work/venv-$version"
  spec=("nvidia-cublas==$version")
  # a reused environment already satisfies the range, so ask for the newest explicitly
  [[ $version == latest ]] && spec=("nvidia-cublas>=13,<14" --upgrade-package nvidia-cublas)
  uv venv -q --allow-existing --python 3.12 "$venv"
  uv pip install -q --python "$venv/bin/python" --reinstall-package sam3rs "$wheel" "${spec[@]}" numpy pytest
done

failed=()
passed=()
for version in "${versions[@]}"; do
  venv="$root/$work/venv-$version"
  installed=$("$venv/bin/python" -c 'from importlib.metadata import version; print(version("nvidia-cublas"))')
  echo "== nvidia-cublas $installed"
  if ! (cd crates/sam3rs-python && "$venv/bin/python" -m pytest -q -rs); then
    failed+=("$installed")
  else
    passed+=("$installed")
  fi
done

if [[ ${#failed[@]} -gt 0 ]]; then
  echo "failed with nvidia-cublas ${failed[*]}" >&2
  exit 1
fi
echo "passed with nvidia-cublas ${passed[*]}"
