#!/usr/bin/env bash
# Fails when the committed PTX is stale: kernels/ptx/sources.sha256 records the hashes of
# the kernel crate's sources, manifest, lockfile and toolchain pin, the build script and
# the PTX it produced, so editing any of them without rerunning scripts/build-ptx.sh
# changes a hash. Needs no CUDA toolkit or cuda-oxide
#
#   scripts/check-ptx.sh            # check
#   scripts/check-ptx.sh --write    # record the current files (build-ptx.sh does this)
set -euo pipefail

cd "$(dirname "$0")/.."
manifest=kernels/ptx/sources.sha256
crate=crates/sam3rs-kernels
mapfile -t sources < <(find "$crate/src" -name '*.rs' | LC_ALL=C sort)
files=("${sources[@]}" "$crate/Cargo.toml" "$crate/Cargo.lock" "$crate/rust-toolchain.toml"
  scripts/build-ptx.sh kernels/ptx/*.ptx)

if [[ "${1:-}" == "--write" ]]; then
  sha256sum "${files[@]}" > "$manifest"
  exit 0
fi

if ! diff <(sha256sum "${files[@]}") "$manifest"; then
  echo "kernels/ptx is stale: run scripts/build-ptx.sh and commit the result" >&2
  exit 1
fi
echo "kernels/ptx matches its sources"
