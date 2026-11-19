#!/usr/bin/env bash
# Cheap CSS-4 screen for large real pools (VisualNews extra buckets, Incidents1M):
#   1. Qwen3-VL alone scores the screen manifest (same prompt as the full annotation);
#   2. images it scores 4 are copied to a regular manifest (promote_screened);
#   3. the full annotation (all 3 proposers + aggregator) runs on those only.
# Qwen's answer is reused in step 3, so screened images get exactly the same treatment as the others.
#
#   GPUS="0 1 2 3 4 5 6" bash scripts/screen_pool.sh \
#       metadata/screen/manifest_incidents_top.json metadata/manifest_incidents.json incidents
set -euo pipefail
cd "$(dirname "$0")/.."
SCREEN=$1; OUT=$2; TAG=${3:-screen}
read -r -a G <<< "${GPUS:-0}"
mkdir -p logs
echo "$(date) Qwen screen on $SCREEN (GPUs ${G[*]})"
for i in "${!G[@]}"; do
    python -m greve.annotate.proposer_qwen --manifest "$SCREEN" --gpu "${G[$i]}" \
        --num_shards ${#G[@]} --shard "$i" > "logs/${TAG}_screen_qwen$i.out" 2>&1 &
done
wait
python -m greve.screen.promote_screened "$SCREEN" "$OUT" --proposer qwen3-vl-8b --min_score 4
exec bash scripts/score_pool.sh "$OUT" "$TAG"
