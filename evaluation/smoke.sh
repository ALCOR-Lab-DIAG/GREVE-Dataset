#!/usr/bin/env bash
# Smoke test: every detector on ~200 images, the detectors spread over the GPUs.
#   ./smoke.sh /path/to/TRIAGE/triage_manifest.json [gpus=8]
set -uo pipefail
M=$1; G=${2:-8}
DETS=(cnnspot fredect fusing gramnet lgrad lnp dire univfd patchcraft drct_convb npr fatformer rine
      clipdet c2p_clip aide safe effort bfree spai community_forensics)
mkdir -p logs
for d in "${DETS[@]}"; do  # downloads one at a time: two processes must not write the same files
  ./in_container.sh 0 python detect.py --detector "$d" --fetch-only > "logs/fetch_$d.txt" 2>&1 || echo "DOWNLOAD FALLITO: $d"
done
for g in $(seq 0 $((G - 1))); do (
  for i in "${!DETS[@]}"; do
    [ $((i % G)) -eq "$g" ] || continue
    d=${DETS[$i]}
    ./in_container.sh "$g" python detect.py --detector "$d" --manifest "$M" --subset test --split 0.01 --out smoke \
      > "logs/smoke_$d.txt" 2>&1 || echo "FALLITO: $d"
  done ) & done
wait
./in_container.sh 0 python score.py --manifest "$M" --subset test --split 0.01 --scores smoke --out smoke_results --n-boot 0
echo "log con errori:"; grep -l "failed on\|Error" logs/*.txt
