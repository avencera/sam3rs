#!/usr/bin/env bash
# Phase 0 baselines on this GPU: osam ORT CUDA and official PyTorch (BF16 and FP32),
# timed with the osam protocol, then nsys captures of each and an ORT node profile.
# Run through the homebased GPU queue so nothing else shares the GPU
set -euo pipefail

root="$(cd "$(dirname "$0")/.." && pwd)"
out="${1:-$root/_scratch/native-runtime/baseline}"
profiles="$root/_scratch/native-runtime/profiles"
mkdir -p "$out" "$profiles"
torch_py="$root/_scratch/venv-torch/bin/python"
ort_py="$root/_scratch/venv-ort/bin/python"
# osam reads its ONNX blobs from ~/.cache/osam, so give it a HOME inside the repo
osam_home="$root/_scratch/osam-home"
ckpt="$root/models/sam3/sam3.pt"

nvidia-smi --query-gpu=name,clocks.sm,clocks.mem,power.draw,utilization.gpu,memory.used --format=csv > "$out/gpu-before.csv"
nvidia-smi --query-compute-apps=pid,name,used_memory --format=csv >> "$out/gpu-before.csv"

cd "$root"
HOME="$osam_home" "$ort_py" -I scripts/bench_osam.py 2>&1 | grep -v "^\s*$" | tee "$out/osam.txt"
"$torch_py" -I scripts/reference.py bench --ckpt "$ckpt" --precision bf16 2>&1 | grep "^|" | tee "$out/torch-bf16.txt"
"$torch_py" -I scripts/reference.py bench --ckpt "$ckpt" --precision fp32 2>&1 | grep "^|" | tee "$out/torch-fp32.txt"

HOME="$osam_home" "$ort_py" -I scripts/bench_osam.py --ort-profile "$out/ort-profile" 2>&1 | grep -v "^\s*$" | tee "$out/osam-ort-profile.txt"

nsys profile -t cuda,nvtx,cublas,cudnn,osrt --force-overwrite true -o "$profiles/baseline-osam" \
  env HOME="$osam_home" "$ort_py" -I scripts/bench_osam.py > "$out/nsys-osam.log" 2>&1
nsys profile -t cuda,nvtx,cublas,cudnn,osrt --force-overwrite true -o "$profiles/baseline-torch-bf16" \
  "$torch_py" -I scripts/reference.py bench --ckpt "$ckpt" --precision bf16 > "$out/nsys-torch-bf16.log" 2>&1

nvidia-smi --query-gpu=name,clocks.sm,clocks.mem,power.draw,utilization.gpu,memory.used --format=csv > "$out/gpu-after.csv"
echo baseline done
