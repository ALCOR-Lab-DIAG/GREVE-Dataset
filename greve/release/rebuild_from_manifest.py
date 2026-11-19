"""
Rebuild the GREVE image folders from the released manifest and the ORIGINAL image files.

Only the manifest and our generated images are distributed by us; every other image belongs to its
source dataset and must be downloaded from there (see the README, "Replicating the dataset"). Put
each original file in --originals_dir under the name in the manifest's `filename` field, then run:

  python -m greve.release.rebuild_from_manifest --manifest manifest/greve_manifest_*.json.gz \
      --originals_dir data/images --out data/images/GREVE

For every entry the script:
  1. checks the original: decoded RGB pixels must match `pixel_sha256` (same hash as
     greve.curate.dedup_phash), so a wrong or re-compressed download is caught;
  2. writes <out>/<greve_path>:
       greve_action == "copied"     the original JPEG bytes, unchanged
       greve_action == "converted"  RGB (transparency over white, first frame of animations),
                                     JPEG at quality `greve_jpeg_q`, 4:2:0 chroma, optimize=True —
                                     exactly what greve.curate.build_release did.

Missing originals and hash mismatches are listed in <out>/rebuild_report.json; nothing is written
for them. Resumable: files already in --out are skipped.
"""

import argparse
import gzip
import hashlib
import json
import os
import shutil
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

from PIL import Image

from greve.curate.build_release import to_rgb
from greve.paths import IMAGES_DIR, RELEASE_DIR


def load_manifest(paths):
    entries = []
    for p in paths:
        opener = gzip.open if str(p).endswith(".gz") else open
        with opener(p, "rt", encoding="utf-8") as f:
            entries.extend(json.load(f))
    return entries


def pixel_sha256(path):
    with Image.open(path) as im:
        rgb = im.convert("RGB")
        return hashlib.sha256(rgb.tobytes() + f"{rgb.size}".encode()).hexdigest()


def work(job):
    src, dst, action, q, expected_sha, verify = job
    src, dst = Path(src), Path(dst)
    if dst.exists():
        return "exists"
    if not src.exists():
        return "missing"
    if verify and expected_sha and pixel_sha256(src) != expected_sha:
        return "hash_mismatch"
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.name + ".tmp")
    if action == "copied":
        shutil.copyfile(src, tmp)
    else:
        with Image.open(src) as im:
            to_rgb(im).save(tmp, "JPEG", quality=q, subsampling=2, optimize=True)
    os.replace(tmp, dst)
    return action


def main():
    parser = argparse.ArgumentParser(description="Rebuild GREVE from the manifest and the original files")
    parser.add_argument("--manifest", nargs="+", required=True, help="manifest file(s), .json or .json.gz")
    parser.add_argument("--originals_dir", default=IMAGES_DIR, help="folder with the original files (`filename`)")
    parser.add_argument("--out", default=RELEASE_DIR)
    parser.add_argument("--sources", nargs="*", default=None, help="only these source_dataset values")
    parser.add_argument("--no_verify", action="store_true", help="skip the pixel_sha256 check")
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()

    entries = load_manifest(args.manifest)
    if args.sources:
        entries = [e for e in entries if e["source_dataset"] in args.sources]
    out, orig = Path(args.out), Path(args.originals_dir)
    jobs = [(str(orig / e["filename"]), str(out / e["greve_path"]), e["greve_action"], e.get("greve_jpeg_q"),
             e.get("pixel_sha256"), not args.no_verify) for e in entries]
    with Pool(args.workers) as pool:
        results = pool.map(work, jobs, chunksize=64)

    stats = Counter(results)
    problems = {r: [e["image_id"] for e, x in zip(entries, results) if x == r] for r in ("missing", "hash_mismatch")}
    out.mkdir(parents=True, exist_ok=True)
    report = out / "rebuild_report.json"
    report.write_text(json.dumps({"counts": dict(stats), **problems}, indent=1))
    print(f"{len(entries)} entries: {dict(stats)} — details in {report}")


if __name__ == "__main__":
    main()
