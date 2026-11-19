#!/usr/bin/env bash
# Full CSS annotation of one pool (one manifest): the three VLM proposers in parallel, then the
# text-only aggregator, which also applies the code guards (majority ceiling, context cap, fiction cap).
# Resumable: images already scored with the current prompts are skipped.
#
#   GPUS="0 1 2 3 4 5 6" bash scripts/score_pool.sh metadata/manifest_medic.json medic
#
# GPUS: space-separated GPU ids (default "0"). With 7 GPUs the proposers get Qwen x2, InternVL x2,
# Pixtral x3 (Pixtral is the slowest); with fewer, every proposer gets every GPU in turn.
set -euo pipefail
cd "$(dirname "$0")/.."
M=$1; TAG=${2:-pool}
read -r -a G <<< "${GPUS:-0}"
mkdir -p logs

run_proposer() {  # name, then the GPUs to shard it over
    local name=$1; shift; local gpus=("$@"); local n=${#gpus[@]}
    for i in "${!gpus[@]}"; do
        python -m greve.annotate.proposer_$name --manifest "$M" --gpu "${gpus[$i]}" \
            --num_shards "$n" --shard "$i" > "logs/${TAG}_${name}$i.out" 2>&1 &
    done
}

echo "$(date) proposers on $M (GPUs ${G[*]})"
if [ ${#G[@]} -ge 7 ]; then
    run_proposer qwen     "${G[0]}" "${G[1]}"
    run_proposer internvl "${G[2]}" "${G[3]}"
    run_proposer pixtral  "${G[@]:4}"
    wait
else  # one proposer at a time over all GPUs (two models do not fit on one 32 GB card)
    for p in qwen internvl pixtral; do run_proposer $p "${G[@]}"; wait; done
fi

echo "$(date) aggregator"
for i in "${!G[@]}"; do
    python -m greve.annotate.aggregator --manifest "$M" --gpu "${G[$i]}" \
        --num_shards ${#G[@]} --shard "$i" > "logs/${TAG}_aggregator$i.out" 2>&1 &
done
wait
echo "$(date) $TAG done"
