#!/usr/bin/env bash
# One detector on all GPUs (one shard per GPU), then the scoring.
#   ./run_all.sh <detector> /path/to/TRIAGE/triage_manifest.json [split=0.7] [gpus=8]
set -euo pipefail
det=$1; M=$2; split=${3:-0.7}; G=${4:-8}
mkdir -p logs
./in_container.sh 0 python detect.py --detector "$det" --fetch-only   # download once, not G times in parallel
for g in $(seq 0 $((G - 1))); do
  ./in_container.sh "$g" python detect.py --detector "$det" --manifest "$M" --split "$split" \
    --out scores --shard "$g/$G" > "logs/${det}_$g.txt" 2>&1 &
done
wait
grep -h "failed on" logs/${det}_*.txt | head || true
./in_container.sh 0 python score.py --manifest "$M" --split "$split" --scores scores --out results
