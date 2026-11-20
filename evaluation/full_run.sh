#!/usr/bin/env bash
# Full run: every detector on one subset of the manifest (triage_split, default test), sharded over
# all GPUs, then the metrics once at the end.
#   ./full_run.sh /path/to/TRIAGE/triage_manifest.json [subset=test] [gpus=8] [seed=0]
# Env: DETECTORS="gapl pgc dgs_net" to run only those; IMAGE_DIR=/path/to/all_images to read
# the images from a flat folder instead of <root>/<triage_path>.
# Resumable: rerun the same command after a stop, finished images are skipped.
# Outputs: scores_<subset>/ and results_<subset>/ for seed 0, scores_<subset>_seed<k>/ and
# results_<subset>_seed<k>/ otherwise (never mixed); logs in logs/ (seed 0) or logs/seed<k>/; full_run.out.
set -uo pipefail
M=$1; SUBSET=${2:-test}; G=${3:-8}; SEED=${4:-0}
TAG=$SUBSET; [ "$SEED" != 0 ] && TAG=${SUBSET}_seed$SEED
SCORES=scores_$TAG; RESULTS=results_$TAG; LOGS=logs; [ "$SEED" != 0 ] && LOGS=logs/seed$SEED
# two queues of similar length, two detectors per GPU; the slowest (dire, fusing) at the end
QUEUE_A=(community_forensics bfree drct_convb effort npr rine clipdet c2p_clip gapl dgs_net dire)
QUEUE_B=(safe fatformer cnnspot fredect gramnet univfd patchcraft pgc lgrad fusing)
# memory hungry (OOM next to another detector): alone, one per GPU, after the queues
ALONE=(aide spai lnp)
# DETECTORS="gapl pgc dgs_net" ./full_run.sh ...  runs only those, one after the other (one per GPU shard)
if [ -n "${DETECTORS:-}" ]; then QUEUE_A=($DETECTORS); QUEUE_B=(); ALONE=(); fi
mkdir -p "$LOGS" "$SCORES"

run_detector() {  # one detector, one shard per GPU; extra args go to detect.py
  local d=$1 tag=$2 ok=0; shift 2
  ./in_container.sh 0 python detect.py --detector "$d" --fetch-only > "$LOGS/full_${d}_fetch.txt" 2>&1 \
    || { echo "$(date +%T) DOWNLOAD FALLITO: $d"; return 1; }
  for g in $(seq 0 $((G - 1))); do
    ./in_container.sh "$g" env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python detect.py --detector "$d" \
      --manifest "$M" --subset "$SUBSET" --out "$SCORES" --shard "$g/$G" --seed "$SEED" \
      ${IMAGE_DIR:+--image-dir "$IMAGE_DIR"} "$@" \
      > "$LOGS/${tag}_${d}_$g.txt" 2>&1 &
  done
  for job in $(jobs -p); do wait "$job" || ok=1; done
  local failed; failed=$(cat "$LOGS"/${tag}_${d}_[0-9]*.txt | grep -c "failed on")
  if [ $ok -ne 0 ]; then echo "$(date +%T) FALLITO: $d ($tag)"; else echo "$(date +%T) fatto: $d ($tag, batch falliti: $failed)"; fi
}

run_queue() { for d in "$@"; do echo "$(date +%T) inizio: $d"; run_detector "$d" full; done; }

{
  echo "$(date) subset=$SUBSET gpus=$G seed=$SEED -> $SCORES"
  run_queue "${QUEUE_A[@]}" &
  [ ${#QUEUE_B[@]} -gt 0 ] && run_queue "${QUEUE_B[@]}" &
  wait
  for d in "${ALONE[@]}"; do echo "$(date +%T) inizio: $d"; run_detector "$d" full; done
  # images that failed (OOM on very large images): once more one at a time
  for d in "${ALONE[@]}"; do run_detector "$d" retry --retry-failed --batch-size 1; done
  # with DETECTORS: retry the failed images of those too
  [ -n "${DETECTORS:-}" ] && for d in $DETECTORS; do run_detector "$d" retry --retry-failed --batch-size 1; done
  echo "$(date) metriche"
  ./in_container.sh 0 python score.py --manifest "$M" --subset "$SUBSET" --scores "$SCORES" --out "$RESULTS" \
    ${IMAGE_DIR:+--image-dir "$IMAGE_DIR"}
  echo "$(date) fine: $RESULTS/summary.csv"
} 2>&1 | tee -a "$LOGS/full_run.out"
