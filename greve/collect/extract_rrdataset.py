"""
Phase 1 — Extract RRDataset ("original" condition only) into
manifest_rrdataset.json, for the GREVE sampling plan
(sampling plan, "RRDataset" row: every CSS level, backbone of
the CSS-3/4 fakes).

Source: the official release, Zenodo record 14963880 (ICCV 2025, CC-BY-4.0),
mirrored on the Hub as alecrespi/RRDataset-CV-Project2:
  - RRDataset_original_train_val.tar.gz (2.2GB): train 1,250 ai + 1,250 real,
    val 250 + 250. Streamed from Zenodo, nothing but the kept images is
    written to disk.
  - RRDataset_test.tar.gz (20GB): already extracted locally in --test_dir
    (RRDataset_final/), whose original/{ai,real} hold 8,500 + 8,500 images.
    Its transfer/ and redigital/ conditions are NOT extracted.
Together: 10,000 real + 10,000 AI-generated originals.

What is kept per image (beyond the common manifest fields):
  original_filename / source_path   the file's name and path in the release,
                                    to join the per-image generator list the
                                    authors may provide later
  rr_scenario   fake images only, from the filename: War_&_Conflict_Scenes,
                Natural_Disasters_&_Accidents, Political_&_Social_Events,
                Medical_&_Public_Health, Culture_&_Religion, production
                (Labor & Production), normal (everyday life). `scenario` holds
                the readable name. Real files carry no scenario.
  generator_model   INFERRED from the resolution, since the release has no
                    generator labels (generator_inferred = true):
                      512x512   -> "sd1.x"  (SD v1.4 / v1.5)
                      1024x1024 -> "1024px: sd3.5L/flux1/dalle3/midjourney"
                      other     -> "chameleon" (photographic AI images taken
                                   from the Chameleon dataset; a mix of models)
  generator_type    the same group, coarse ("sd1.x", "1024px", "chameleon")

Dropped: 256x256 AI images (the paper's StyleGAN/ProGAN images, all in the
"normal" scenario — the plan drops GANs) and grayscale originals (as for the
other datasets). Images are saved as their ORIGINAL bytes as
rrdataset_<real|fake>_<split>_<file stem>.<ext>. Resumable.
"""

import argparse
import sys
import tarfile
import urllib.request
from collections import Counter
from pathlib import Path

from tqdm import tqdm

from greve.common import ManifestWriter, existing_ids, is_grayscale, make_entry, open_image, sanitize_key, save_original
from greve.paths import IMAGES_DIR, RAW_DIR

ZENODO_TRAIN_VAL = "https://zenodo.org/api/records/14963880/files/RRDataset_original_train_val.tar.gz/content"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
LABELS = {"ai": "fake", "real": "real"}
SCENARIOS = {
    "War_&_Conflict_Scenes": "War & Conflict",
    "Natural_Disasters_&_Accidents": "Natural Disasters & Accidents",
    "Political_&_Social_Events": "Political & Social Events",
    "Medical_&_Public_Health": "Medical & Public Health",
    "Culture_&_Religion": "Culture & Religion",
    "production": "Labor & Production",
    "normal": "Everyday life",
}
GAN_SIZE = (256, 256)


def rr_scenario(stem):
    """'War_&_Conflict_Scenes_000542' -> 'War_&_Conflict_Scenes'; 'real_000001' -> None."""
    prefix = stem.rsplit("_", 1)[0]
    return prefix if prefix in SCENARIOS else None


def generator_group(size):
    if size == (512, 512):
        return "sd1.x", "sd1.x"
    if size == (1024, 1024):
        return "1024px: sd3.5L/flux1/dalle3/midjourney", "1024px"
    return "chameleon", "chameleon"


def iter_train_val(url):
    """Stream the official train/val archive: yields (split, ai|real, filename, path in archive, bytes)."""
    with urllib.request.urlopen(url) as response:
        with tarfile.open(fileobj=response, mode="r|gz") as tf:
            for member in tf:
                parts = member.name.split("/")  # RRDataset_original_train_val/<split>/<ai|real>/<file>
                if not member.isfile() or len(parts) != 4 or Path(parts[3]).suffix.lower() not in IMAGE_EXTENSIONS:
                    continue
                yield parts[1], parts[2], parts[3], member.name, tf.extractfile(member).read()


def iter_test(test_dir):
    """The local test split's original condition: yields the same tuples as iter_train_val."""
    for kind in ("real", "ai"):
        folder = Path(test_dir) / "original" / kind
        for path in sorted(folder.iterdir()):
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
                yield "test", kind, path.name, f"RRDataset_test/original/{kind}/{path.name}", path.read_bytes()


def main():
    parser = argparse.ArgumentParser(description="Phase 1 — Extract RRDataset originals (train/val/test) into manifest_rrdataset.json")
    parser.add_argument("--output_dir", type=str, default=IMAGES_DIR)
    parser.add_argument("--metadata_path", type=str, default="./metadata/manifest_rrdataset.json")
    parser.add_argument("--test_dir", type=str, default=str(Path(RAW_DIR) / "rrdataset" / "RRDataset_final"),
                        help="Local copy of the extracted RRDataset_test archive (contains original/{ai,real})")
    parser.add_argument("--train_val_url", type=str, default=ZENODO_TRAIN_VAL)
    parser.add_argument("--splits", type=str, nargs="+", default=["train_val", "test"], choices=["train_val", "test"])
    parser.add_argument("--keep_grayscale", action="store_true", help="Don't exclude grayscale originals")
    parser.add_argument("--keep_gan", action="store_true", help="Don't drop the 256x256 GAN images")
    parser.add_argument("--seed", type=int, default=9999, help="Recorded per entry for provenance; everything is taken")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    Path(args.metadata_path).parent.mkdir(parents=True, exist_ok=True)

    manifest = ManifestWriter(args.metadata_path)
    seen = existing_ids(manifest, "rrdataset", output_dir)
    print(f"Manifest has {len(seen)} RRDataset images on disk")

    stats = Counter()
    sources = []
    if "train_val" in args.splits:
        sources.append(("train/val (Zenodo, streamed)", iter_train_val(args.train_val_url), 3000))
    if "test" in args.splits:
        sources.append(("test (local)", iter_test(args.test_dir), 17000))

    for desc, rows, total in sources:
        for split, kind, filename, source_path, raw_bytes in tqdm(rows, desc=f"RRDataset {desc}", total=total):
            label = LABELS.get(kind)
            if label is None:
                continue
            stem = Path(filename).stem
            image_id = f"rrdataset_{label}_{split}_{sanitize_key(stem)}"
            if image_id in seen:
                stats["already saved"] += 1
                continue
            try:
                img = open_image(raw_bytes)
            except Exception as e:
                stats["unreadable"] += 1
                tqdm.write(f"  [skip] {source_path} -> {type(e).__name__}: {e}")
                continue
            if label == "fake" and img.size == GAN_SIZE and not args.keep_gan:
                stats["dropped_gan_256px"] += 1
                continue
            if not args.keep_grayscale and is_grayscale(img):
                stats[f"dropped_grayscale_{label}"] += 1
                continue

            scen = rr_scenario(stem) if label == "fake" else None
            generator, generator_type = generator_group(img.size) if label == "fake" else (None, None)
            filename_out, info = save_original(raw_bytes, output_dir, image_id, img=img)
            manifest.add(make_entry(
                image_id=image_id,
                filename=filename_out,
                source="rrdataset",
                label=label,
                sensitivity="HIGH" if scen and scen not in ("normal", "production") else "MEDIUM",
                scenario=SCENARIOS.get(scen) if scen else None,
                split=split,
                generator=generator,
                generator_type=generator_type,
                original_label=kind,
                source_image_id=source_path,
                condition="original",
                hf_repo=None,
                seed=args.seed,
                info=info,
                extra={
                    "original_filename": filename,
                    "source_path": source_path,
                    "rr_scenario": scen,
                    "generator_inferred": label == "fake",
                    "release": "Zenodo 14963880 (RRDataset, ICCV 2025)",
                },
            ))
            seen.add(image_id)
            stats[f"saved_{label}"] += 1

    total = manifest.export_json()
    labels = Counter(e["label"] for e in manifest.entries)
    manifest.close()
    print(f"\nThis run: {dict(stats)}")
    print(f"DONE — {total} entries in {args.metadata_path} (real={labels['real']}, fake={labels['fake']})")
    sys.exit(0)


if __name__ == "__main__":
    main()
