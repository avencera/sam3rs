#!/usr/bin/env bash
# Fails when the committed PTX is stale: kernels/ptx/sources.sha256 records the hashes of
# every kernel source, the build script and the PTX it produced, so editing any of them
# without rerunning scripts/build-ptx.sh changes a hash. Needs no CUDA toolkit
#
#   scripts/check-ptx.sh            # check
#   scripts/check-ptx.sh --write    # record the current files (build-ptx.sh does this)
set -euo pipefail

cd "$(dirname "$0")/.."
manifest=kernels/ptx/sources.sha256
files=(kernels/*.cu scripts/build-ptx.sh kernels/ptx/*.ptx)

if [[ "${1:-}" == "--write" ]]; then
  sha256sum "${files[@]}" > "$manifest"
  exit 0
fi

if ! diff <(sha256sum "${files[@]}") "$manifest"; then
  echo "kernels/ptx is stale: run scripts/build-ptx.sh and commit the result" >&2
  exit 1
fi
echo "kernels/ptx matches its sources"
