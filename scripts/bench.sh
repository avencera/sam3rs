#!/usr/bin/env bash
# Times a snapshot of the sam3rs bench binary (osam protocol) in both precisions, three
# runs each, then captures an nsys profile per precision. Run it through the homebased
# GPU queue so nothing else shares the GPU
#
#   scripts/bench.sh RUN_DIR    # RUN_DIR/bench is the binary to time
set -euo pipefail

root="$(cd "$(dirname "$0")/.." && pwd)"
run="$(cd "$1" && pwd)"
name="$(basename "$run")"
profiles="$root/_scratch/native-runtime/profiles"
mkdir -p "$profiles"
cd "$root"

nvidia-smi --query-gpu=name,clocks.sm,clocks.mem,power.draw,utilization.gpu,memory.used --format=csv > "$run/gpu-before.csv"
nvidia-smi --query-compute-apps=pid,name,used_memory --format=csv >> "$run/gpu-before.csv"

for precision in fp32 bf16; do
  for _ in 1 2 3; do
    "$run/bench" --precision "$precision" | tail -n 1
  done | tee "$run/bench-$precision.txt"
done
for precision in fp32 bf16; do
  nsys profile -t cuda,nvtx,cublas --force-overwrite true -o "$profiles/sam3rs-$name-$precision" \
    "$run/bench" --precision "$precision" > "$run/nsys-$precision.log" 2>&1
done

nvidia-smi --query-compute-apps=pid,name,used_memory --format=csv > "$run/gpu-after.csv"
echo bench done
