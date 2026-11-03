#!/bin/sh
# Export the draw.io sources to the files the manuscript reads: figures/<id>.pdf (cropped) and
# figures/_png/<id>.png. Run from anywhere: sh figures/drawio/export.sh
set -e
DRAWIO=/Applications/draw.io.app/Contents/MacOS/draw.io
cd "$(dirname "$0")/.."
for src in drawio/*.drawio; do
  id=$(basename "$src" .drawio)
  "$DRAWIO" -x -f pdf --crop -o "$id.pdf" "$src"
  "$DRAWIO" -x -f png -s 2 -o "_png/$id.png" "$src"
done
