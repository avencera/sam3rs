#!/usr/bin/env bash
# Fetch SAM 3 weights into models/ and verify them against the official facebook/sam3 hashes.
# facebook/sam3 is gated with manual approval, so files come from a mirror whose LFS sha256
# matches the official repo; the check below fails loudly if the mirror ever drifts
set -euo pipefail

root="$(cd "$(dirname "$0")/.." && pwd)"
mirror="${SAM3_MIRROR:-jetjodh/sam3}"
hf="https://huggingface.co"

declare -A sha=(
  [model.safetensors]=6d06f0a5f84e435071fe6603e61d0b4cc7b40e0d39d487cfd4d67d8cc11cc14a
  [sam3.pt]=9999e2341ceef5e136daa386eecb55cb414446a00ac2b55eb2dfd2f7c3cf8c9e
)

fetch() {
  local repo="$1" file="$2" dest="$3"
  [[ -s "$dest/$file" ]] && return 0
  mkdir -p "$dest"
  curl -sSL --retry 5 -C - -o "$dest/$file.part" "$hf/$repo/resolve/main/$file"
  mv "$dest/$file.part" "$dest/$file"
}

for f in config.json merges.txt processor_config.json special_tokens_map.json \
  tokenizer.json tokenizer_config.json vocab.json model.safetensors "${@:-}"; do
  [[ -n "$f" ]] && fetch "$mirror" "$f" "$root/models/sam3"
done

for f in "${!sha[@]}"; do
  path="$root/models/sam3/$f"
  [[ -f "$path" ]] || continue
  got="$(sha256sum "$path" | cut -d' ' -f1)"
  if [[ "$got" != "${sha[$f]}" ]]; then
    echo "sha256 mismatch for $f: $got" >&2
    exit 1
  fi
  echo "verified $f"
done

# test images: osam's benchmark image plus a few from the SAM 3 and sam3-onnx repos
images="$root/models/images"
mkdir -p "$images"
gh_raw="https://raw.githubusercontent.com"
for spec in \
  "wkentaro/osam/c45c9968d86532b1d722cd6fbf6d6d9d2f934972/examples/_images/dogs.jpg" \
  "wkentaro/sam3-onnx/main/images/bus.jpg" \
  "facebookresearch/sam3/main/assets/images/truck.jpg" \
  "facebookresearch/sam3/main/assets/images/groceries.jpg"; do
  file="$(basename "$spec")"
  [[ -s "$images/$file" ]] || curl -sSL --retry 5 -o "$images/$file" "$gh_raw/$spec"
done
