"""
Phase 1 — Extract the DGM4 candidate pool for the GREVE sampling plan
(sampling plan, "DGM4" row) into manifest_dgm4.json.

Source: rshaojimmy/DGM4. Metadata in metadata/{train,val,test}.json, images
zipped per outlet (origin/<outlet>.zip, manipulation/<tool>.zip).

What the plan takes from DGM4: pristine REAL photos only ("12K pristine",
role CSS-2: politics, celebrities). The dataset never includes tampered
images in any form, so DGM4's manipulated images are never extracted. The default target
overshoots the plan (15K vs 12K) because some sampled photos won't land at
the CSS level they're needed for.

DGM4 is built on VisualNews: a row's `id` IS the VisualNews image id (same
photo, same path — verified for every pristine row), and a pristine photo
can appear in several rows (one per caption), so the pool keys on that id:
one entry per photo, with source_image_id = the VisualNews id, which is what
the plan's VisualNews dedup ("key on the VisualNews image ID") joins on.

Tamper bases: a face-manipulated DGM4 image is named after the id of the
pristine photo it was made from (manipulation/simswap/683133-013337-
simswap.jpg is made from photo 683133); 71,373 of the 77,426 pristine
photos have such a version. The plan's rule "a photo used as a DGM4 tamper
base is never used as real" only guards against overlap with a tampered
track, and there is none, so every pristine photo is eligible. Each entry
still records is_tamper_base (and has_text_manipulation, for photos DGM4
also pairs with a falsified caption) for reference.

Grayscale originals are excluded by default (as for SID-Set), and images are
saved as their ORIGINAL bytes. Sampling is a seeded random draw over the
eligible photos of all splits; resumable (photos whose file exists in
--output_dir count as done, deleted ones are saved again).
"""

import argparse
import json
import os
import random
import re
import sys
import zipfile
from collections import Counter
from pathlib import Path

from huggingface_hub import hf_hub_download
from tqdm import tqdm

from greve.common import ManifestWriter, existing_ids, is_grayscale, make_entry, open_image, save_original
from greve.paths import IMAGES_DIR

HF_REPO = "rshaojimmy/DGM4"
SPLITS = ["train", "val", "test"]


def load_rows(hf_token):
    rows = []
    for split in SPLITS:
        path = hf_hub_download(HF_REPO, f"metadata/{split}.json", repo_type="dataset", token=hf_token)
        with open(path) as f:
            rows.extend(dict(r, split=split) for r in json.load(f))
    return rows


def pristine_photos(rows):
    """One record per pristine photo (VisualNews id), with what DGM4 did to it elsewhere."""
    photos = {}
    for r in rows:
        if r["fake_cls"] == "orig":
            photos.setdefault(str(r["id"]), {"id": str(r["id"]), "image": r["image"], "split": r["split"],
                                             "caption": r["text"]})
    face_bases = {re.match(r"\d+", r["image"].rsplit("/", 1)[-1]).group(0)
                  for r in rows if r["image"].startswith("DGM4/manipulation/")}
    text_manipulated = {str(r["id"]) for r in rows
                        if r["fake_cls"] != "orig" and r["image"].startswith("DGM4/origin/")}
    for pid, p in photos.items():
        p["face_base"] = pid in face_bases
        p["text_manipulated"] = pid in text_manipulated
    return photos


class OriginZips:
    """Downloads each origin/<outlet>.zip once (HF cache) and keeps it open."""

    def __init__(self, hf_token):
        self.hf_token = hf_token
        self.open = {}

    def read(self, img_path):
        # "DGM4/origin/bbc/0303/427.jpg" -> zip "origin/bbc.zip", member "bbc/0303/427.jpg"
        parts = img_path.split("/")
        zip_repo_path = "/".join(parts[1:3]) + ".zip"
        if zip_repo_path not in self.open:
            local = hf_hub_download(HF_REPO, zip_repo_path, repo_type="dataset", token=self.hf_token)
            self.open[zip_repo_path] = zipfile.ZipFile(local)
        return self.open[zip_repo_path].read("/".join(parts[2:]))

    def close(self):
        for zf in self.open.values():
            zf.close()


def main():
    parser = argparse.ArgumentParser(description="Phase 1 — Extract the DGM4 pristine-real pool (GREVE plan)")
    parser.add_argument("--output_dir", type=str, default=IMAGES_DIR)
    parser.add_argument("--metadata_path", type=str, default="./metadata/manifest_dgm4.json")
    parser.add_argument("--hf_token", type=str, default=os.environ.get("HF_TOKEN"),
                        help="Hugging Face token (default: $HF_TOKEN from the environment)")
    parser.add_argument("--n_real", type=int, default=15000, help="Pool size (plan: 12K, plus overshoot)")
    parser.add_argument("--seed", type=int, default=9999, help="Seeds the sampling order (and is recorded per entry)")
    parser.add_argument("--keep_grayscale", action="store_true", help="Don't exclude grayscale originals")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    Path(args.metadata_path).parent.mkdir(parents=True, exist_ok=True)

    photos = pristine_photos(load_rows(args.hf_token))
    n_face = sum(p["face_base"] for p in photos.values())
    print(f"DGM4: {len(photos)} pristine photos ({n_face} also have a face-manipulated version, not used)")

    eligible = sorted(photos.values(), key=lambda p: int(p["id"]))
    random.Random(args.seed).shuffle(eligible)
    print(f"By split: {dict(Counter(p['split'] for p in eligible))}")

    manifest = ManifestWriter(args.metadata_path)
    seen = existing_ids(manifest, "dgm4", output_dir)
    count = sum(1 for e in manifest.entries if e["image_id"] in seen)
    rejected = Counter()
    print(f"Manifest has {count} DGM4 photos on disk (target {args.n_real})")

    zips = OriginZips(args.hf_token)
    progress = tqdm(total=args.n_real, initial=min(count, args.n_real), desc="DGM4")
    for p in eligible:
        if count >= args.n_real:
            break
        image_id = f"dgm4_real_{p['id']}"
        if image_id in seen:
            continue
        try:
            raw_bytes = zips.read(p["image"])
            img = open_image(raw_bytes)
        except Exception as e:
            rejected["unreadable"] += 1
            tqdm.write(f"  [skip] {p['image']} -> {type(e).__name__}: {e}")
            continue
        if not args.keep_grayscale and is_grayscale(img):
            rejected["grayscale"] += 1
            continue

        filename, info = save_original(raw_bytes, output_dir, image_id, img=img)
        manifest.add(make_entry(
            image_id=image_id,
            filename=filename,
            source="dgm4",
            label="real",
            sensitivity="MEDIUM-HIGH",
            scenario="news_politics",
            split=p["split"],
            generator="original",
            original_label="orig",
            source_image_id=p["id"],
            condition="original",
            hf_repo=HF_REPO,
            seed=args.seed,
            info=info,
            extra={
                "outlet": p["image"].split("/")[2],
                "caption": p["caption"],
                "is_tamper_base": p["face_base"],
                "has_text_manipulation": p["text_manipulated"],
            },
        ))
        seen.add(image_id)
        count += 1
        progress.update(1)
        progress.set_description(f"DGM4 | saved={count} rejected={sum(rejected.values())}")
    progress.close()
    zips.close()

    total = manifest.export_json()
    manifest.close()
    print(f"\nRejected: {dict(rejected)}")
    if count < args.n_real:
        print(f"  [short] only {count}/{args.n_real} usable photos")
    print(f"DONE — {total} total entries in manifest ({args.metadata_path}); on disk: {count}")
    sys.exit(0)


if __name__ == "__main__":
    main()
