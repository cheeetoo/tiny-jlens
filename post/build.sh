#!/bin/sh
# post.md is the source. This regenerates post.html (a local preview with the figures inline).
cd "$(dirname "$0")" && pandoc post.md -f gfm -t html5 -s \
  --metadata pagetitle="GPT-2 Small Mostly Has a Functional Global Workspace (draft)" \
  -c https://cdn.jsdelivr.net/npm/water.css@2/out/light.css -o post.html && echo "wrote post/post.html"
