#!/usr/bin/env bash
# Fetches the video benchmark clips from Wikimedia Commons into models/video-src, checks
# their sha256, and cuts H.264 test clips into models/video at 1080p and 720p, 64 and 256
# frames each. All clips allow commercial use; credits are in models/video/CREDITS.md
#
#   FFMPEG=/path/to/ffmpeg scripts/fetch-clips.sh
set -euo pipefail

root="$(cd "$(dirname "$0")/.." && pwd)"
src="$root/models/video-src"
out="$root/models/video"
ffmpeg="${FFMPEG:-ffmpeg}"
mkdir -p "$src" "$out"

# name|url|sha256|start seconds|license|author|page
clips=(
  "horses|https://upload.wikimedia.org/wikipedia/commons/a/af/Horses_running_in_a_pasture.webm|a845625850b2b899fb70e377154cc1a20e8f977f7a0542ba6ec5f6dbcd636ff4|4|CC BY 3.0|Field of Dreams|https://commons.wikimedia.org/wiki/File:Horses_running_in_a_pasture.webm"
  "cyclists|https://upload.wikimedia.org/wikipedia/commons/7/7f/Fietsers_in_de_IJ-tunnel_%28video%29.webm|b3efc0786f679bd8d629c6e94e1834c6751f2af4307efebcfc4d1049eaec2151|4|CC BY 4.0|Hay Kranen|https://commons.wikimedia.org/wiki/File:Fietsers_in_de_IJ-tunnel_(video).webm"
  "grazing|https://upload.wikimedia.org/wikipedia/commons/3/3a/A_grazing_horse_at_the_%C3%89comus%C3%A9e_de_la_Bintinais.webm|3549ff43db03b209fd8f5ed05d833054ff775086b8156b15fe2d4a15229279bc|0|CC0 1.0|Seb35|https://commons.wikimedia.org/wiki/File:A_grazing_horse_at_the_%C3%89comus%C3%A9e_de_la_Bintinais.webm"
)

{
  echo "# Video benchmark clips"
  echo
  echo "Cut from Wikimedia Commons originals by scripts/fetch-clips.sh (re-encoded and trimmed)."
  echo
} > "$out/CREDITS.md"

for entry in "${clips[@]}"; do
  IFS='|' read -r name url sha start license author page <<< "$entry"
  mkdir -p "$src/$name"
  file="$src/$name/src.webm"
  if [ ! -f "$file" ] || ! echo "$sha  $file" | sha256sum -c --status; then
    curl -sSfL -o "$file.part" "$url"
    mv "$file.part" "$file"
  fi
  echo "$sha  $file" | sha256sum -c --quiet
  echo "- $name: \"$page\" by $author, $license" >> "$out/CREDITS.md"

  for height in 1080 720; do
    for frames in 64 256; do
      clip="$out/${name}-${height}p-${frames}.mp4"
      [ -f "$clip" ] && continue
      # constant quality and a fixed GOP so every runtime decodes the same frames
      "$ffmpeg" -hide_banner -loglevel error -ss "$start" -i "$file" -frames:v "$frames" -an \
        -vf "scale=-2:$height:flags=lanczos" -c:v libx264 -preset slow -crf 18 -g 32 \
        -pix_fmt yuv420p "$clip.part.mp4"
      mv "$clip.part.mp4" "$clip"
    done
  done
done
ls -l "$out"
