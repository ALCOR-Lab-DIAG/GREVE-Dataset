#!/usr/bin/env bash
# Resumable MEDIC download: retries curl -C - until the archive has the expected size.
mkdir -p "${GREVE_RAW_DIR:-data/raw}/medic" && cd "${GREVE_RAW_DIR:-data/raw}/medic"
EXPECTED=10865298409
for try in $(seq 1 200); do
    SIZE=$(stat -c %s MEDIC.tar.gz 2>/dev/null || echo 0)
    [ "$SIZE" -eq "$EXPECTED" ] && { echo "$(date) complete"; exit 0; }
    echo "$(date) try $try, resuming from $SIZE bytes"
    curl -sS -L -C - --speed-limit 10000 --speed-time 120 -o MEDIC.tar.gz https://crisisnlp.qcri.org/data/MEDIC/MEDIC.tar.gz
    sleep 20
done
echo "$(date) giving up"; exit 1
