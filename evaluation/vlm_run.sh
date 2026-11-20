#!/usr/bin/env bash
# One VLM on the test split, several instances in parallel (data parallel), then the metrics of every VLM
# in scores_vlm_test/. Runs ONE model and stops: the next one is started by hand.
#   CSS_MANIFEST=... IMAGE_DIR=... ./vlm_run.sh <model> [gpus=0,1,2,3,4,5,6] [seed=0]
# GPUs per instance: 1 for the 8B models, 2 for InternVL3.5-30B-A3B and Kimi-VL, 3 for Qwen3-VL-30B and
# Qwen2.5-VL-32B (32 GB cards). Resumable. Outputs: scores_vlm_test/, results_vlm_test/, logs/vlm/ for seed 0,
# scores_vlm_test_seed<k>/, results_vlm_test_seed<k>/, logs/vlm/seed<k>/ otherwise (never mixed).
# Run it again with the same GPUs to complete a model: failed images are retried (same number of instances needed).
set -uo pipefail
MODEL=$1; GPUS=(${2:-0,1,2,3,4,5,6}); GPUS=(${GPUS//,/ }); SEED=${3:-0}
TAG=vlm_test; LOGS=logs/vlm
[ "$SEED" != 0 ] && { TAG=vlm_test_seed$SEED; LOGS=logs/vlm/seed$SEED; }
M=${CSS_MANIFEST:?set CSS_MANIFEST=/path/to/TRIAGE/triage_manifest.json}
I=${IMAGE_DIR:?set IMAGE_DIR=/path/to/all_images}
case $MODEL in
  qwen3vl_8b|qwen25vl_7b|internvl3_8b) PER=1 ;;
  internvl35_30b_a3b|kimivl_a3b) PER=2 ;;
  qwen3vl_30b_a3b|qwen25vl_32b) PER=3 ;;
  *) echo "modello sconosciuto: $MODEL"; exit 1 ;;
esac
IMAGE=css-vlm; [ "$MODEL" = kimivl_a3b ] && IMAGE=css-vlm-kimi
N=$(( ${#GPUS[@]} / PER ))
mkdir -p "$LOGS" "scores_$TAG"
{
  echo "$(date) $MODEL seed $SEED: $N istanze x $PER GPU (${GPUS[*]})"
  for k in $(seq 0 $((N - 1))); do
    g=$(IFS=,; echo "${GPUS[*]:$((k * PER)):$PER}")
    CSS_IMAGE=$IMAGE ./in_container.sh "$g" env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python vlm_detect.py --retry-failed --model "$MODEL" --manifest "$M" --subset test \
      --image-dir "$I" --out "scores_$TAG" --shard "$k/$N" --seed "$SEED" >> "$LOGS/${MODEL}_$k.txt" 2>&1 &
  done
  wait
  echo "$(date) $MODEL finito, immagini fallite: $(cat "$LOGS"/${MODEL}_*.txt | grep -c 'failed on')"
  ./in_container.sh 0 python score.py --manifest "$M" --subset test --image-dir "$I" --scores "scores_$TAG" \
    --out "results_$TAG"
  echo "$(date) metriche: results_$TAG/"
} 2>&1 | tee -a "$LOGS/vlm_run.out"
