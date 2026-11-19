"""
Phase 1 — Extract UnsafeBench (all of it, train + test) into
manifest_unsafebench.json, for the GREVE sampling plan
(sampling plan, "UnsafeBench" row: CSS-3/4 candidates).

Source: yiting/UnsafeBench (gated: accept its Data Use Agreement on the Hub
and set HF_TOKEN). 10,146 rows: train 8,109 + test 2,037.
  source   Laion5B -> label "real" (real-world photos from LAION-5B)
           Lexica  -> label "fake" (AI-generated images from the Lexica
                      gallery — mostly Stable Diffusion; the exact model per
                      image isn't given, so generator_model = "Lexica")
  safety_label  "Safe" / "Unsafe", per the UnsafeBench annotators
  category      one of 11 unsafe-content types the image was collected for
                (Hate, Harassment, Violence, Self-harm, Sexual, Shocking,
                Illegal activity, Deception, Political, Public and personal
                health, Spam) — set for Safe images too
  text          LAION caption (real) or Lexica prompt (fake), when available

The dataset has no image ids, and 269 rows are byte-identical copies of other
rows (sometimes with a different category or safety label). Each image is
therefore keyed on the SHA-256 of its bytes: image_id =
unsafebench_<label>_<first 16 hex>. A duplicate becomes one entry whose
`unsafebench_annotations` lists every row it appeared in (split, row index,
safety_label, category); `safety_label` is "Unsafe" if any of them is.

Grayscale originals are excluded by default (as for SID-Set and DGM4), and
images are saved as their ORIGINAL bytes. Resumable: images whose file
exists in --output_dir are skipped.
"""

import argparse
import hashlib
import os
import sys
from collections import Counter
from pathlib import Path

from datasets import Image as HFImage
from datasets import load_dataset
from tqdm import tqdm

from greve.common import ManifestWriter, existing_ids, is_grayscale, make_entry, open_image, save_original
from greve.paths import IMAGES_DIR

HF_REPO = "yiting/UnsafeBench"
SPLITS = ["train", "test"]
LABELS = {"Laion5B": "real", "Lexica": "fake"}


def main():
    parser = argparse.ArgumentParser(description="Phase 1 — Extract UnsafeBench (train + test) into manifest_unsafebench.json")
    parser.add_argument("--output_dir", type=str, default=IMAGES_DIR)
    parser.add_argument("--metadata_path", type=str, default="./metadata/manifest_unsafebench.json")
    parser.add_argument("--hf_token", type=str, default=os.environ.get("HF_TOKEN"),
                        help="Hugging Face token with access to the gated dataset (default: $HF_TOKEN)")
    parser.add_argument("--splits", type=str, nargs="+", default=SPLITS)
    parser.add_argument("--keep_grayscale", action="store_true", help="Don't exclude grayscale originals")
    parser.add_argument("--seed", type=int, default=9999, help="Recorded per entry for provenance; everything is taken")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    Path(args.metadata_path).parent.mkdir(parents=True, exist_ok=True)

    manifest = ManifestWriter(args.metadata_path)
    seen = existing_ids(manifest, "unsafebench", output_dir)
    entries = {e["image_id"]: e for e in manifest.entries if e["image_id"] in seen}
    print(f"Manifest has {len(entries)} UnsafeBench images on disk")

    stats = Counter()
    for split in args.splits:
        ds = load_dataset(HF_REPO, split=split, streaming=True, token=args.hf_token)
        ds = ds.cast_column("image", HFImage(decode=False))  # keep the original bytes
        for index, row in enumerate(tqdm(ds, desc=f"UnsafeBench[{split}]")):
            label = LABELS.get(row["source"])
            if label is None:
                stats[f"unknown source {row['source']}"] += 1
                continue
            raw_bytes = row["image"]["bytes"]
            image_id = f"unsafebench_{label}_{hashlib.sha256(raw_bytes).hexdigest()[:16]}"
            annotation = {"split": split, "row_index": index,
                          "safety_label": row["safety_label"], "category": row["category"]}

            if image_id in entries:  # a byte-identical duplicate, or already saved by an earlier run
                entry = entries[image_id]
                if annotation not in entry["unsafebench_annotations"]:
                    entry["unsafebench_annotations"].append(annotation)
                    entry["unsafebench_categories"] = sorted({a["category"] for a in entry["unsafebench_annotations"]})
                    if row["safety_label"] == "Unsafe":
                        entry["safety_label"] = "Unsafe"
                    manifest.add(entry)  # the journal keeps the latest version of each image_id
                    stats["duplicate rows merged"] += 1
                continue

            try:
                img = open_image(raw_bytes)
            except Exception as e:
                stats["unreadable"] += 1
                tqdm.write(f"  [skip] {split}/{index} -> {type(e).__name__}: {e}")
                continue
            if not args.keep_grayscale and is_grayscale(img):
                stats[f"grayscale_{label}"] += 1
                continue

            filename, info = save_original(raw_bytes, output_dir, image_id, img=img)
            text = (row.get("text") or "").strip() or None
            entry = make_entry(
                image_id=image_id,
                filename=filename,
                source="unsafebench",
                label=label,
                sensitivity="HIGH" if row["safety_label"] == "Unsafe" else "LOW",
                scenario=row["category"],
                split=split,
                generator=None if label == "real" else "Lexica",
                original_label=row["source"],
                source_image_id=f"{split}/{index}",
                condition="original",
                hf_repo=HF_REPO,
                seed=args.seed,
                info=info,
                extra={
                    "real_source": "LAION-5B" if label == "real" else None,
                    "safety_label": row["safety_label"],
                    "unsafebench_categories": [row["category"]],
                    "unsafebench_annotations": [annotation],
                    "caption": text if label == "real" else None,
                    "prompt": text if label == "fake" else None,
                    "image_sha256": hashlib.sha256(raw_bytes).hexdigest(),
                },
            )
            manifest.add(entry)
            entries[image_id] = entry
            stats[f"saved_{label}"] += 1

    total = manifest.export_json()
    manifest.close()
    labels = Counter(e["label"] for e in entries.values())
    print(f"\nThis run: {dict(stats)}")
    print(f"DONE — {total} entries in {args.metadata_path} (real={labels['real']}, fake={labels['fake']})")
    sys.exit(0)


if __name__ == "__main__":
    main()
