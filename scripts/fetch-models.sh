#!/usr/bin/env bash
# Fetch SAM 3 weights into models/sam3 and the test images into models/images, then check
# the weights against the official facebook/sam3 hashes
#
# facebook/sam3 is gated: accept Meta's SAM License at https://huggingface.co/facebook/sam3
# while logged in, then run `hf auth login` or set HF_TOKEN. `--mirror` instead downloads
# from an ungated mirror (SAM3_MIRROR, default jetjodh/sam3) for our own testing; the
# sha256 check catches a mirror that drifts from the official files
#
#   scripts/fetch-models.sh [--mirror] [extra files, e.g. sam3.pt]
set -euo pipefail

root="$(cd "$(dirname "$0")/.." && pwd)"
repo=facebook/sam3
if [[ "${1:-}" == "--mirror" ]]; then
  repo="${SAM3_MIRROR:-jetjodh/sam3}"
  shift
fi
hf="https://huggingface.co"

auth=()
if [[ "$repo" == facebook/sam3 ]]; then
  token="${HF_TOKEN:-}"
  # huggingface_hub's precedence: HF_TOKEN_PATH, then HF_HOME, then XDG_CACHE_HOME
  token_file="${HF_TOKEN_PATH:-${HF_HOME:-${XDG_CACHE_HOME:-$HOME/.cache}/huggingface}/token}"
  [[ -z "$token" && -s "$token_file" ]] && token="$(<"$token_file")"
  if [[ -z "$token" ]]; then
    echo "facebook/sam3 is gated: accept Meta's SAM License at $hf/facebook/sam3, then run" >&2
    echo "\`hf auth login\` or set HF_TOKEN (or pass --mirror for our own testing)" >&2
    exit 1
  fi
  auth=(-H "Authorization: Bearer $token")
fi

declare -A sha=(
  [model.safetensors]=6d06f0a5f84e435071fe6603e61d0b4cc7b40e0d39d487cfd4d67d8cc11cc14a
  [sam3.pt]=9999e2341ceef5e136daa386eecb55cb414446a00ac2b55eb2dfd2f7c3cf8c9e
)

fetch() {
  local repo="$1" file="$2" dest="$3"
  [[ -s "$dest/$file" ]] && return 0
  mkdir -p "$dest"
  local status
  status="$(curl -sSL --retry 5 -C - "${auth[@]}" -w '%{http_code}' -o "$dest/$file.part" "$hf/$repo/resolve/main/$file")"
  # a failed response leaves its error body in .part, which `-C -` would resume onto
  [[ "$status" == 200 || "$status" == 206 ]] || rm -f "$dest/$file.part"
  if [[ "$status" == 401 || "$status" == 403 ]]; then
    echo "HTTP $status for $repo/$file: check that your token is valid and that its account" >&2
    echo "has accepted Meta's SAM License at $hf/$repo" >&2
    exit 1
  fi
  [[ "$status" == 200 || "$status" == 206 ]] || { echo "HTTP $status for $repo/$file" >&2; exit 1; }
  mv "$dest/$file.part" "$dest/$file"
}

for f in config.json merges.txt processor_config.json special_tokens_map.json \
  tokenizer.json tokenizer_config.json vocab.json model.safetensors "${@:-}"; do
  [[ -n "$f" ]] && fetch "$repo" "$f" "$root/models/sam3"
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
