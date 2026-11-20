#!/usr/bin/env bash
# MoE-JEPA on the TRIAGE test split, GPUs 0-6 (GPU 7 stays free), seeds 0, 1, 2 like the other detectors:
# seed 0 -> scores_test/, results_test/; seed k -> scores_test_seed<k>/, results_test_seed<k>/ (same format).
# It needs transformers >= 4.53, so it runs in the css-vlm image. Stops at the end.
#   tmux new -s moejepa;  CSS_MANIFEST=... IMAGE_DIR=... ./moejepa_run.sh 2>&1 | tee logs/moejepa_run.out
# To stop: Ctrl-c in the session, then  podman stop -t 5 $(podman ps -q)
cd "$(dirname "$0")"
M=${CSS_MANIFEST:?set CSS_MANIFEST=/path/to/TRIAGE/triage_manifest.json}
export CSS_IMAGE=css-vlm IMAGE_DIR=${IMAGE_DIR:?set IMAGE_DIR=/path/to/all_images} DETECTORS="moe_jepa"
for s in 0 1 2; do
  ./full_run.sh "$M" test 7 "$s"
done
echo "$(date) moe_jepa finito"
