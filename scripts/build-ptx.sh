#!/usr/bin/env bash
# Regenerates the committed PTX in kernels/ptx from kernels/*.cu, then its source manifest.
# nvcc runs in a pinned CUDA 13.0 image: 13.0 emits PTX ISA 9.0, which any driver from
# r580 can JIT, while newer toolkits raise that floor, and the image also sidesteps host
# glibc headers that nvcc 13.0 rejects. compute_80 is the floor because the attention
# kernel uses BF16 mma.sync
#
#   scripts/build-ptx.sh
set -euo pipefail

root="$(cd "$(dirname "$0")/.." && pwd)"
image="nvidia/cuda:13.0.1-devel-ubuntu24.04"
cd "$root"

docker run --rm -u "$(id -u):$(id -g)" -v "$root:/src" -w /src "$image" \
  nvcc -ptx -arch=compute_80 -O3 -std=c++17 --expt-relaxed-constexpr \
  -o kernels/ptx/sam3.sm80.ptx kernels/sam3.cu
scripts/check-ptx.sh --write
