#!/usr/bin/env bash
# Run a command in the css-eval image on one or more GPUs:  ./in_container.sh <gpu> python detect.py ...
#   <gpu> is one index (3) or a comma list (0,1) for models split over several GPUs.
# The code folder is /work; the dataset is mounted read-only; every download (released weights, repos,
# HF / CLIP / torch backbones) goes to $CSS_WEIGHTS on the host, default <dataset>/TRIAGE/weights.
# $CSS_IMAGE picks another image (default css-eval; css-vlm for the VLMs, see Containerfile.vlm).
set -euo pipefail
gpus=$1; shift
DATA=${CSS_DATA:?set CSS_DATA to the dataset folder, e.g. /path/to/css_dataset}
WEIGHTS=${CSS_WEIGHTS:-$DATA/TRIAGE/weights}
mkdir -p "$WEIGHTS"
devices=()
for g in ${gpus//,/ }; do devices+=(--device "nvidia.com/gpu=$g"); done
exec podman run --rm --security-opt label=disable --group-add keep-groups "${devices[@]}" \
  -v "$PWD:/work" -v "$DATA:$DATA:ro" -v "$WEIGHTS:/root/.cache" \
  "${CSS_IMAGE:-css-eval}" "$@"
