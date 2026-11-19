"""
Phase 1 — Extract a pre-filtered MEDIC pool (real disaster photos, CSS-3/4
candidates) into manifest_medic.json, for the GREVE sampling plan
(sampling plan, "MEDIC" row: CSS-4 real that is not VisualNews;
"severe damage; affected, injured or dead").

Source: MEDIC v1.0 (Alam et al., Neural Computing and Applications 2023),
https://crisisnlp.qcri.org/medic/ — 71,198 social-media disaster images from
CrisisMMD, AIDR, the Damage Assessment Dataset (ASONAM17) and the Damage
Multimodal Dataset (DMD). License CC BY-NC-SA 4.0 (non-commercial,
share-alike). Labels per image (MEDIC_{train,dev,test}.tsv):
  damage_severity  severe | mild | little_or_none
  humanitarian     affected_injured_or_dead_people | infrastructure_and_utility_damage |
                   rescue_volunteering_or_donation_effort | not_humanitarian
  (the TSV's image_id column holds the source name and event_name the
  per-image id — tweet id or file name; the event is read from the path)
  disaster_types   earthquake | fire | flood | hurricane | landslide | other_disaster | not_disaster
  informative      informative | not_informative

Selection (--n images, nothing is CSS-labelled here):
  tier 1  humanitarian = affected_injured_or_dead_people — all of them (the
          CSS-4 candidates)
  tier 2  damage_severity = severe — fills the rest, spread evenly over the
          MEDIC sub-datasets x events (CSS-3: damage without victims)
  tier 3  (--include_rescue) rescue_volunteering_or_donation_effort
  Only images MEDIC marks informative (drops memes and unrelated pictures;
  --keep_not_informative to keep them). Byte-identical copies are kept once
  (the sub-datasets overlap); grayscale originals are skipped.

Reads straight from MEDIC.tar.gz (no unpacking): one streaming pass for the
label files, one to save the selected images as their ORIGINAL bytes as
medic_real_<sub-dataset>_<file stem>.<ext>. Resumable.

  python -m greve.collect.extract_medic --dry_run    # pool sizes only
  python -m greve.collect.extract_medic --n 12000
"""

import argparse
import hashlib
import io
import random
import sys
import tarfile
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from greve.common import ManifestWriter, existing_ids, is_grayscale, make_entry, open_image, sanitize_key, save_original
from greve.paths import IMAGES_DIR, RAW_DIR

ARCHIVE = str(Path(RAW_DIR) / "medic" / "MEDIC.tar.gz")
SPLITS = ["train", "dev", "test"]
LICENSE = "CC BY-NC-SA 4.0"


def event_of(path):
    """The event (or DMD category) is a folder in the path; the TSV's event_name column holds the per-image id."""
    parts = path.split("/")  # data/<subset>/[data_image|multimodal/]<event>/...
    return parts[3] if parts[1] in ("crisismmd", "multimodal-deep-learning-disaster-response-mouzannar") else parts[2]


def norm(path):
    return path[2:] if path.startswith("./") else path


def read_labels(archive):
    """Stream the archive until the three TSVs are read (they sit near its start)."""
    found = {}
    with tarfile.open(archive, mode="r|gz") as tf:
        for m in tf:
            name = norm(m.name)
            for s in SPLITS:
                if name == f"MEDIC_{s}.tsv":
                    found[s] = pd.read_csv(io.BytesIO(tf.extractfile(m).read()), sep="\t")
            if len(found) == len(SPLITS):
                break
    missing = [s for s in SPLITS if s not in found]
    if missing:
        sys.exit(f"Label files missing from the archive: {missing}")
    return pd.concat([d.assign(split=s) for s, d in found.items()], ignore_index=True)


def select(labels, n, include_rescue, keep_not_informative, rng):
    d = labels.copy()
    d["subset"] = d.image_path.str.split("/").str[1]
    d["event"] = d.image_path.map(event_of)
    stats = Counter({"labelled images": len(d)})
    if not keep_not_informative:
        stats["dropped: not informative"] = int((d.informative != "informative").sum())
        d = d[d.informative == "informative"]
    d["tier"] = None
    d.loc[d.damage_severity == "severe", "tier"] = 2
    if include_rescue:
        d.loc[d.tier.isna() & (d.humanitarian == "rescue_volunteering_or_donation_effort"), "tier"] = 3
    d.loc[d.humanitarian == "affected_injured_or_dead_people", "tier"] = 1
    d = d[d.tier.notna()]
    for t, c in d.tier.value_counts().sort_index().items():
        stats[f"tier {int(t)} candidates"] = int(c)

    chosen = []
    for tier in (1, 2, 3):
        left = n - len(chosen)
        group = d[d.tier == tier]
        if left <= 0 or group.empty:
            continue
        if len(group) <= left:
            chosen += group.to_dict("records")
            continue
        # spread over sub-dataset x event: round-robin over shuffled groups
        cells = defaultdict(list)
        for r in group.to_dict("records"):
            cells[(r["subset"], r["event"])].append(r)
        for v in cells.values():
            rng.shuffle(v)
        keys = sorted(cells)
        rng.shuffle(keys)
        while left > 0 and keys:
            for k in list(keys):
                if left == 0:
                    break
                if cells[k]:
                    chosen.append(cells[k].pop())
                    left -= 1
                else:
                    keys.remove(k)
    return chosen, stats


def main():
    parser = argparse.ArgumentParser(description="Phase 1 — Extract a pre-filtered MEDIC pool into manifest_medic.json")
    parser.add_argument("--n", type=int, default=12000, help="Pool size (plan: 8K; overshoot for CSS-3/4 real)")
    parser.add_argument("--archive", type=str, default=ARCHIVE)
    parser.add_argument("--output_dir", type=str, default=IMAGES_DIR)
    parser.add_argument("--metadata_path", type=str, default="./metadata/manifest_medic.json")
    parser.add_argument("--include_rescue", action="store_true", help="Add rescue/volunteering images as tier 3")
    parser.add_argument("--keep_not_informative", action="store_true")
    parser.add_argument("--keep_grayscale", action="store_true", help="Don't exclude grayscale originals")
    parser.add_argument("--dry_run", action="store_true", help="Print the pool, extract nothing")
    parser.add_argument("--seed", type=int, default=9999)
    args = parser.parse_args()

    labels = read_labels(args.archive)
    chosen, stats = select(labels, args.n, args.include_rescue, args.keep_not_informative, random.Random(args.seed))
    for k, v in stats.items():
        print(f"  {k}: {v}")
    print(f"  selected: {len(chosen)} — " + ", ".join(f"tier {t}: {c}" for t, c in
                                                     sorted(Counter(int(r['tier']) for r in chosen).items())))
    if args.dry_run:
        return

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    Path(args.metadata_path).parent.mkdir(parents=True, exist_ok=True)
    manifest = ManifestWriter(args.metadata_path)
    seen = existing_ids(manifest, "medic", output_dir)
    hashes = {e.get("image_sha256") for e in manifest.entries if e["image_id"] in seen}
    print(f"Manifest has {len(seen)} MEDIC images on disk")

    wanted = {r["image_path"]: r for r in chosen}
    run = Counter()
    progress = tqdm(total=len(wanted), desc="MEDIC")
    with tarfile.open(args.archive, mode="r|gz") as tf:
        for m in tf:
            path = norm(m.name)
            r = wanted.get(path)
            if r is None or not m.isfile():
                continue
            progress.update(1)
            image_id = f"medic_real_{sanitize_key(r['subset'], 40)}_{sanitize_key(Path(path).stem)}"
            if image_id in seen:
                run["already saved"] += 1
                continue
            raw = tf.extractfile(m).read()
            sha = hashlib.sha256(raw).hexdigest()
            if sha in hashes:
                run["duplicate bytes"] += 1
                continue
            try:
                img = open_image(raw)
            except Exception:
                run["unreadable"] += 1
                continue
            if not args.keep_grayscale and is_grayscale(img):
                run["grayscale"] += 1
                continue
            filename, info = save_original(raw, output_dir, image_id, img=img)
            tier = int(r["tier"])
            manifest.add(make_entry(
                image_id=image_id,
                filename=filename,
                source="medic",
                label="real",
                sensitivity="HIGH",
                scenario=r["disaster_types"],
                split=r["split"],
                original_label=r["humanitarian"],
                source_image_id=path,
                condition="original",
                hf_repo=None,
                seed=args.seed,
                info=info,
                extra={
                    "original_filename": Path(path).name,
                    "medic_subset": r["subset"],
                    "medic_source_name": str(r["image_id"]),   # TSV image_id column: the source's name
                    "medic_image_id": str(r["event_name"]),    # TSV event_name column: tweet id / file name
                    "medic_event": r["event"],
                    "damage_severity": r["damage_severity"],
                    "humanitarian": r["humanitarian"],
                    "disaster_types": r["disaster_types"],
                    "informative": r["informative"],
                    "selection_tier": tier,
                    "image_sha256": sha,
                    "license": LICENSE,
                },
            ))
            seen.add(image_id)
            hashes.add(sha)
            run[f"saved tier {tier}"] += 1
    progress.close()

    total = manifest.export_json()
    manifest.close()
    missing = len(wanted) - progress.n
    print(f"\nThis run: {dict(run)}" + (f"; {missing} selected images not found in the archive" if missing else ""))
    print(f"DONE — {total} entries in {args.metadata_path}")
    sys.exit(0)


if __name__ == "__main__":
    main()
