#!/usr/bin/env bash
# Regenerates the committed PTX in kernels/ptx from the cuda-oxide kernel crate in
# crates/sam3rs-kernels, then its source manifest. The crate is built once per tier:
# sm_80 is the floor because the attention kernels use BF16 mma.sync, and sm_120 is what
# RTX 50 series GPUs load. Both stay at PTX ISA 9.0 or below, which any driver from r580
# can JIT
#
# cargo-oxide and its codegen backend come from the cuda-rust commit that
# crates/sam3rs-kernels/Cargo.lock pins, installed under their own CARGO_HOME so other
# projects' cuda-oxide installs at other commits are left alone. Needs Linux, the crate's
# rust-toolchain.toml toolchain, LLVM 21's llc (CUDA_OXIDE_LLC) with the opt and llvm-link
# beside it, libclang, and a CUDA toolkit's libdevice (CUDA_OXIDE_LIBDEVICE). Their
# versions and libdevice's hash go to kernels/ptx/tools.txt, which the source manifest
# covers, so a rebuild with other tools shows up in the diff
#
#   scripts/build-ptx.sh
set -euo pipefail

root="$(cd "$(dirname "$0")/.." && pwd)"
crate="$root/crates/sam3rs-kernels"
toolchain="$(sed -n 's/^channel = "\(.*\)"/\1/p' "$crate/rust-toolchain.toml")"
rev="$(sed -n 's|^source = "git+https://github.com/NVIDIA/cuda-rust.git?branch=main#\(.*\)"|\1|p' "$crate/Cargo.lock" | sort -u)"
if [[ ! "$rev" =~ ^[0-9a-f]{40}$ ]]; then
  echo "expected one cuda-rust commit in $crate/Cargo.lock, found: $rev" >&2
  exit 1
fi

export CARGO_HOME="${SAM3RS_OXIDE_HOME:-$HOME/.cache/sam3rs/cargo-oxide}"
export PATH="$CARGO_HOME/bin:$PATH"
export CUDA_OXIDE_LLC="${CUDA_OXIDE_LLC:-/usr/bin/llc-21}"
export LIBCLANG_PATH="${LIBCLANG_PATH:-/usr/lib/llvm-21/lib}"
export CUDA_OXIDE_LIBDEVICE="${CUDA_OXIDE_LIBDEVICE:-${CUDA_HOME:-/usr/local/cuda}/nvvm/libdevice/libdevice.10.bc}"
# a cargo or rustup environment inherited from a parent build would pick another toolchain
unset RUSTUP_TOOLCHAIN CARGO RUSTC

stamp="$CARGO_HOME/cargo-oxide.rev"
if [[ "$(cat "$stamp" 2>/dev/null)" != "$rev" ]]; then
  cargo "+$toolchain" install --locked --force --git https://github.com/NVIDIA/cuda-rust.git --rev "$rev" cargo-oxide
  echo "$rev" > "$stamp"
fi

# cuda-oxide runs the opt and llvm-link beside llc and links libdevice into kernels that
# call math functions, so all four shape the PTX
llvm_version() {
  "$1" --version | sed -n 's/^ *\(.*LLVM version .*\)$/\1/p'
}
{
  echo "llc: $(llvm_version "$CUDA_OXIDE_LLC")"
  echo "opt: $(llvm_version "${CUDA_OXIDE_LLC/llc/opt}")"
  echo "llvm-link: $(llvm_version "${CUDA_OXIDE_LLC/llc/llvm-link}")"
  echo "libdevice: $(sha256sum < "$CUDA_OXIDE_LIBDEVICE" | cut -d' ' -f1)"
} > "$root/kernels/ptx/tools.txt"

cd "$crate"
for arch in 80 120; do
  out="$root/target/sam3rs-kernels/sm$arch"
  mkdir -p "$out/ptx"
  # cargo-oxide only writes PTX when the crate itself recompiles
  cargo "+$toolchain" clean -q --target-dir "$out/cargo" -p sam3rs-kernels
  CARGO_TARGET_DIR="$out/cargo" CUDA_OXIDE_PTX_DIR="$out/ptx" cargo "+$toolchain" oxide build --arch "sm_$arch"
  ptx="$out/ptx/sam3rs_kernels.ptx"
  version="$(sed -n 's/^\.version \([0-9]*\.[0-9]*\)$/\1/p' "$ptx")"
  if ! awk -v v="$version" 'BEGIN { exit !(v != "" && v + 0 <= 9.0) }'; then
    echo "sm_$arch PTX has ISA version '$version', above the 9.0 that r580 drivers JIT" >&2
    exit 1
  fi
  # a call left in the PTX is a body that wasn't inlined, such as a generic #[device] fn,
  # whose pointers then reach it as generic ones: 25% slower in the flash kernels
  if grep -q 'call\.uni\|^\s*call\b' "$ptx"; then
    echo "sm_$arch PTX has calls; make the callee a plain #[inline(always)] fn:" >&2
    grep -o 'call\.uni [A-Za-z0-9_]*' "$ptx" | sort -u >&2
    exit 1
  fi
  cp "$ptx" "$root/kernels/ptx/sam3.sm$arch.ptx"
done
"$root/scripts/check-ptx.sh" --write
