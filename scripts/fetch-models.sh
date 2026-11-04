#!/usr/bin/env bash
# Fetch SAM 3 weights into models/sam3 and the test images into models/images, then check
# the weights against the official facebook/sam3 hashes
#
# The weights come from avencera/sam3-safetensors, an ungated mirror holding byte-identical
# copies of facebook/sam3's files, so no Hugging Face token is needed. `--official` fetches
# from the gated facebook/sam3 instead, which also holds the original sam3.pt checkpoint:
# accept Meta's SAM License at https://huggingface.co/facebook/sam3 while logged in, then
# run `hf auth login` or set HF_TOKEN
#
#   scripts/fetch-models.sh [--official] [extra files, e.g. sam3.pt with --official]
set -euo pipefail

root="$(cd "$(dirname "$0")/.." && pwd)"
repo=avencera/sam3-safetensors
# the same commit the crate and the Python package load, read from src/hub.rs so the two
# can't drift
revision="$(sed -n 's/^pub const MODEL_REVISION: &str = "\([0-9a-f]*\)";$/\1/p' "$root/src/hub.rs")"
[[ -n "$revision" ]] || { echo "MODEL_REVISION not found in src/hub.rs" >&2; exit 1; }
if [[ "${1:-}" == "--official" ]]; then
  repo=facebook/sam3
  revision=main
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
    echo "\`hf auth login\` or set HF_TOKEN (or drop --official to use the ungated mirror)" >&2
    exit 1
  fi
  # curl reads the header from a file only this user can read, since a command-line
  # argument shows up in other users' ps
  header_file="$(mktemp)"
  trap 'rm -f "$header_file"' EXIT
  printf 'Authorization: Bearer %s\n' "$token" > "$header_file"
  auth=(-H "@$header_file")
fi

declare -A sha=(
  [model.safetensors]=6d06f0a5f84e435071fe6603e61d0b4cc7b40e0d39d487cfd4d67d8cc11cc14a
  [vocab.json]=5047b556ce86ccaf6aa22b3ffccfc52d391ea4accdab9c2f2407da5b742d4363
  [merges.txt]=9fd691f7c8039210e0fced15865466c65820d09b63988b0174bfe25de299051a
  [sam3.pt]=9999e2341ceef5e136daa386eecb55cb414446a00ac2b55eb2dfd2f7c3cf8c9e
)

fetch() {
  local repo="$1" file="$2" dest="$3"
  [[ -s "$dest/$file" ]] && return 0
  mkdir -p "$dest"
  local status
  status="$(curl -sSL --retry 5 -C - "${auth[@]}" -w '%{http_code}' -o "$dest/$file.part" "$hf/$repo/resolve/$revision/$file")"
  # a failed response leaves its error body in .part, which `-C -` would resume onto
  [[ "$status" == 200 || "$status" == 206 ]] || rm -f "$dest/$file.part"
  if [[ "$repo" == facebook/sam3 && ("$status" == 401 || "$status" == 403) ]]; then
    echo "HTTP $status for $repo/$file: check that your token is valid and that its account" >&2
    echo "has accepted Meta's SAM License at $hf/$repo" >&2
    exit 1
  fi
  [[ "$status" == 200 || "$status" == 206 ]] || { echo "HTTP $status for $repo/$file" >&2; exit 1; }
  mv "$dest/$file.part" "$dest/$file"
}

for f in merges.txt vocab.json model.safetensors "${@:-}"; do
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
