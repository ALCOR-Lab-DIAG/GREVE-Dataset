"""
Phase 1 — Extract the SID-Set candidate pool for the GREVE sampling plan
(sampling plan, "SID-Set" row) into manifest_sidset.json.

Source: saberzl/SID_Set — label: 0 = real (OpenImages V7), 1 = fake
(full-synthetic, FLUX), 2 = tampered (always skipped: the dataset never
includes tampered images).
Splits on the Hub: train (210K) and validation (30K); the official 60K
test split is distributed separately by the SIDA authors.

The plan needs ~6K real + ~4K fake SID-Set images; the defaults overshoot
that (8K real + 5K fake) because some sampled images won't turn out to be
CSS-1 after labeling.

Sampling: SID-Set's parquet row groups (~93 rows, ~55MB each) mix real,
fake and tampered rows, and reading one row downloads its whole row group.
So the script walks the row groups of the chosen splits in a random
(seeded) order and takes every eligible real/fake row from each until both
quotas are met — a random sample that downloads ~15GB instead of 140GB.

Grayscale originals (mode L/LA/1/I, or RGB with identical channels) are
excluded by default, so "grayscale" can't become a real-vs-fake shortcut
(every FLUX fake is color). Images are saved as their ORIGINAL bytes (FLUX
fakes stay PNG, reals keep their original JPEG compression).

Of the GREVE fields make_entry() defines, SID-Set fills source_image_id
(real: OpenImages V7 id), generator_type, condition and the orig_* fields;
the rest stay null.

Resumable: a re-run walks the same row-group order, counts manifest rows
whose file exists in --output_dir and continues (rows whose file was
deleted are downloaded again).
"""

import argparse
import os
import random
import sys
from collections import Counter, deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import HfFileSystem
from tqdm import tqdm

from greve.common import ManifestWriter, existing_ids, is_grayscale, make_entry, open_image, sanitize_key, save_original
from greve.paths import IMAGES_DIR

HF_REPO = "saberzl/SID_Set"
LABELS = {0: "real", 1: "fake"}  # 2 = tampered: never used


def list_row_groups(fs, splits, workers):
    files = sorted(f for f in fs.ls(f"datasets/{HF_REPO}/data", detail=False)
                   if f.endswith(".parquet") and f.rsplit("/", 1)[-1].split("-", 1)[0] in splits)

    def n_groups(f):
        with fs.open(f, block_size=2**20) as fh:
            return [(f, rg) for rg in range(pq.ParquetFile(fh).metadata.num_row_groups)]

    with ThreadPoolExecutor(workers) as ex:
        return [g for groups in ex.map(n_groups, files) for g in groups]


def read_row_group(fs, path, rg):
    with fs.open(path, block_size=2**24) as fh:
        rows = pq.ParquetFile(fh).read_row_group(rg, columns=["img_id", "label", "image"]).to_pylist()
    split = path.rsplit("/", 1)[-1].split("-", 1)[0]
    return split, [r for r in rows if r["label"] in LABELS]


def main():
    parser = argparse.ArgumentParser(description="Phase 1 — Extract the SID-Set candidate pool (GREVE plan)")
    parser.add_argument("--output_dir", type=str, default=IMAGES_DIR)
    parser.add_argument("--metadata_path", type=str, default="./metadata/manifest_sidset.json")
    parser.add_argument("--hf_token", type=str, default=os.environ.get("HF_TOKEN"),
                        help="Hugging Face token (default: $HF_TOKEN from the environment)")
    parser.add_argument("--n_real", type=int, default=8000)
    parser.add_argument("--n_fake", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=9999, help="Seeds the row-group order (and is recorded per entry)")
    parser.add_argument("--splits", type=str, nargs="+", default=["train"],
                        help="Splits to sample from (default: train validation)")
    parser.add_argument("--keep_grayscale", action="store_true", help="Don't exclude grayscale originals")
    parser.add_argument("--workers", type=int, default=8, help="Row groups downloaded in parallel")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    Path(args.metadata_path).parent.mkdir(parents=True, exist_ok=True)

    manifest = ManifestWriter(args.metadata_path)
    seen = existing_ids(manifest, "sidset", output_dir)
    quota = {"real": args.n_real, "fake": args.n_fake}
    counts = Counter(e["label"] for e in manifest.entries if e["image_id"] in seen)
    rejected = Counter()
    print(f"Manifest has {counts['real']} real / {counts['fake']} fake SID-Set images on disk "
          f"(target {args.n_real} / {args.n_fake})")

    fs = HfFileSystem(token=args.hf_token)
    groups = list_row_groups(fs, set(args.splits), args.workers)
    random.Random(args.seed).shuffle(groups)
    print(f"{len(groups)} row groups in {args.splits}; walking them in seeded random order")

    def done():
        return all(counts[l] >= quota[l] for l in quota)

    progress = tqdm(total=args.n_real + args.n_fake,
                    initial=min(counts["real"], args.n_real) + min(counts["fake"], args.n_fake), desc="SID-Set")
    used_groups = 0
    with ThreadPoolExecutor(args.workers) as ex:
        it = iter(groups)
        pending = deque()

        def submit_next():
            g = next(it, None)
            if g is not None:
                pending.append(ex.submit(read_row_group, fs, *g))

        for _ in range(args.workers * 2):
            submit_next()
        while pending and not done():
            split, rows = pending.popleft().result()
            used_groups += 1
            for r in rows:
                label = LABELS[r["label"]]
                image_id = f"sidset_{label}_{sanitize_key(f'{split}_{r['img_id']}')}"
                if image_id in seen or counts[label] >= quota[label]:
                    continue
                raw_bytes = r["image"]["bytes"]
                try:
                    img = open_image(raw_bytes)
                except Exception as e:
                    rejected["unreadable"] += 1
                    tqdm.write(f"  [skip] {r['img_id']} -> {type(e).__name__}: {e}")
                    continue
                if not args.keep_grayscale and is_grayscale(img):
                    rejected[f"grayscale_{label}"] += 1
                    continue

                is_real = label == "real"
                filename, info = save_original(raw_bytes, output_dir, image_id, img=img)
                entry = make_entry(
                    image_id=image_id,
                    filename=filename,
                    source="sidset",
                    label=label,
                    sensitivity="LOW",
                    scenario="social_media",
                    split=split,
                    generator="original" if is_real else "FLUX",
                    generator_type=None if is_real else "base",
                    original_label=str(r["label"]),
                    source_image_id=r["img_id"],
                    condition="original",
                    hf_repo=HF_REPO,
                    seed=args.seed,
                    info=info,
                )
                manifest.add(entry)
                seen.add(image_id)
                counts[label] += 1
                progress.update(1)
            progress.set_description(f"SID-Set | real={counts['real']} fake={counts['fake']} "
                                     f"groups={used_groups} rejected={sum(rejected.values())}")
            if not done():
                submit_next()
        for f in pending:
            f.cancel()
    progress.close()

    total = manifest.export_json()
    manifest.close()
    print(f"\nRow groups downloaded: {used_groups} | rejected: {dict(rejected)}")
    if not done():
        print(f"  [short] ran out of row groups: real={counts['real']}/{args.n_real} fake={counts['fake']}/{args.n_fake}")
    print(f"DONE — {total} total entries in manifest ({args.metadata_path}); "
          f"on disk: real={counts['real']}, fake={counts['fake']}")
    sys.exit(0)


if __name__ == "__main__":
    main()
