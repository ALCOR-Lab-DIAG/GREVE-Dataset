"""
Phase 1 — Extract a CSS-4-oriented pool from Incidents1M (real web photos of
disasters, damage and incidents) into manifest_incidents.json.

Source: Incidents1M (Weber et al., ECCV 2020; extended paper arXiv 2201.04236),
https://github.com/ethanweber/IncidentsDataset — the label files are released
on request (form + email to incidentsdataset@googlegroups.com):
  multi_label_train.json, multi_label_val.json   (latest, multi-label)
  eccv_train.json, eccv_val.json                 (ECCV 2020, single-label)
Each maps an image name to {"url": ..., "incidents": {class: 1|0}, "places":
{class: 1|0}}; 1 = the class is present, 0 = verified absent. Only URLs are
released: the images are downloaded from the original sites (some are gone).
Labels MIT; the images keep their owners' rights (research use only).

Selection — only images with a POSITIVE incident label that can depict CSS-4
content (explosion, a large fire in a populated place, transport disasters,
catastrophic destruction with victims); purely natural scenes (drought, fog,
snow, storms, floods without damage) and incidents that can't reach CSS-4
(traffic jam, under construction, contamination) are never selected:
  tier 1  explosion / fire / transport disasters anywhere, and burned,
          collapsed or damaged structures in a populated place
  tier 2  the same damage classes elsewhere, natural disasters in a populated
          place (earthquake, tornado, volcanic eruption, landslide, ...)
Tier 1 is taken whole, then tier 2, each spread round-robin over (incident x
place) cells so no single class dominates; --n caps the pool. Nothing is
CSS-scored here: the proposers do that (or the Qwen screen first).

Downloads run in parallel threads; an image is kept if it opens, its short
side is >= --min_side, it is not grayscale and not a byte duplicate. Resumable:
images already in the manifest are skipped, and failed URLs are remembered in
<manifest>.failed.txt so a re-run doesn't retry them (--retry_failed to do so).

  python -m greve.collect.extract_incidents --labels data/raw/incidents/multi_label_*.json --dry_run
  python -m greve.collect.extract_incidents --labels incidents_train.json incidents_val.json
"""

import argparse
import hashlib
import json
import random
import sys
import threading
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
from tqdm import tqdm

from greve.common import ManifestWriter, existing_ids, is_grayscale, make_entry, open_image, sanitize_key, save_original
from greve.paths import IMAGES_DIR

LICENSE = "labels MIT (Incidents1M); images: original owners' rights, research use"

# incidents that can show CSS-4 content in any setting ("nuclear explosion" and
# "fire whirl" left out: mushroom clouds / nature shots, no victims)
TIER1_ANY = {"on fire", "airplane accident", "train accident", "bus accident", "ship boat accident"}
# damage classes: tier 1 in a populated place, tier 2 elsewhere
DAMAGE = {"burned", "collapsed", "damaged", "with smoke"}
# natural disasters / road accidents: tier 2 in a populated place, else never
# (tornado, tropical cyclone, storm surge, volcanic eruption, wildfire, oil spill
# left out: debris / landscape shots, CSS-4 yield too low to be worth screening)
TIER2_POPULATED = {"earthquake", "landslide", "mudslide mudflow", "car accident", "truck accident",
                   "van accident", "motorcycle accident", "bicycle accident"}
POPULATED = {"street", "downtown", "residential neighborhood", "building facade", "building outdoor",
             "house", "village", "slum", "skyscraper", "religious building", "parking lot", "gas station",
             "fire station", "industrial area", "highway", "bridge", "port", "railroad track"}

HEADERS = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) research-dataset-builder"}


def positives(d):
    return sorted(k for k, v in (d or {}).items() if v == 1)


def tier_of(incidents, places):
    populated = bool(set(places) & POPULATED)
    inc = set(incidents)
    if inc & TIER1_ANY or (inc & DAMAGE and populated):
        return 1
    if inc & DAMAGE or (inc & TIER2_POPULATED and populated):
        return 2
    return None


def load_labels(paths):
    items, stats = {}, Counter()
    for path in paths:
        data = json.load(open(path))
        split = "val" if "val" in Path(path).stem else "train"
        stats[f"images in {Path(path).name}"] = len(data)
        for name, d in data.items():
            if name in items or not d.get("url"):
                continue
            incidents, places = positives(d.get("incidents")), positives(d.get("places"))
            if not incidents:
                stats["no positive incident"] += 1
                continue
            tier = tier_of(incidents, places)
            if tier is None:
                stats["incident not CSS-4-related"] += 1
                continue
            items[name] = {"name": name, "url": d["url"], "incidents": incidents, "places": places,
                           "tier": tier, "split": split}
    return list(items.values()), stats


def select(items, n, rng):
    chosen = []
    for tier in (1, 2):
        left = n - len(chosen)
        group = [it for it in items if it["tier"] == tier]
        if left <= 0 or not group:
            continue
        if len(group) <= left:
            chosen += group
            continue
        cells = defaultdict(list)
        for it in group:
            cells[(it["incidents"][0], (it["places"] or ["-"])[0])].append(it)
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
    return chosen


def fetch(url, timeout, max_bytes):
    with requests.get(url, headers=HEADERS, timeout=timeout, stream=True) as r:
        r.raise_for_status()
        if "image" not in r.headers.get("Content-Type", "image"):
            raise ValueError(f"not an image: {r.headers.get('Content-Type')}")
        raw = b""
        for chunk in r.iter_content(65536):
            raw += chunk
            if len(raw) > max_bytes:
                raise ValueError("too large")
    return raw


def main():
    parser = argparse.ArgumentParser(description="Phase 1 — Extract a CSS-4-oriented Incidents1M pool")
    parser.add_argument("--labels", type=str, nargs="+", required=True, help="Incidents1M label JSON files")
    parser.add_argument("--n", type=int, default=130000, help="Max images to download (tier 1 first)")
    parser.add_argument("--output_dir", type=str, default=IMAGES_DIR)
    parser.add_argument("--metadata_path", type=str, default="./metadata/screen/manifest_incidents_screen.json")
    parser.add_argument("--min_side", type=int, default=256, help="Drop images whose short side is smaller")
    parser.add_argument("--keep_grayscale", action="store_true")
    parser.add_argument("--workers", type=int, default=32, help="Parallel downloads")
    parser.add_argument("--timeout", type=float, default=20)
    parser.add_argument("--max_mb", type=float, default=25)
    parser.add_argument("--retry_failed", action="store_true", help="Retry URLs that failed in earlier runs")
    parser.add_argument("--dry_run", action="store_true", help="Print the pool, download nothing")
    parser.add_argument("--seed", type=int, default=9999)
    args = parser.parse_args()

    items, stats = load_labels(args.labels)
    for k, v in stats.items():
        print(f"  {k}: {v}")
    print(f"  CSS-4-related candidates: {len(items)} — tier 1: {sum(i['tier'] == 1 for i in items)}, "
          f"tier 2: {sum(i['tier'] == 2 for i in items)}")
    inc = Counter(x for i in items for x in i["incidents"])
    print("  by incident: " + ", ".join(f"{k} {v}" for k, v in inc.most_common()))
    chosen = select(items, args.n, random.Random(args.seed))
    print(f"  selected: {len(chosen)} — " + ", ".join(f"tier {t}: {c}" for t, c in
                                                     sorted(Counter(i['tier'] for i in chosen).items())))
    if args.dry_run:
        return

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    Path(args.metadata_path).parent.mkdir(parents=True, exist_ok=True)
    manifest = ManifestWriter(args.metadata_path)
    seen = existing_ids(manifest, "incidents", output_dir)
    hashes = {e.get("image_sha256") for e in manifest.entries if e["image_id"] in seen}
    failed_path = Path(args.metadata_path).with_suffix(".failed.txt")
    failed = set() if args.retry_failed or not failed_path.exists() else set(failed_path.read_text().split("\n"))
    print(f"Manifest has {len(seen)} Incidents images on disk; {len(failed)} URLs failed before")

    def image_id_of(it):
        return f"incidents_real_{sanitize_key(it['name'].rsplit('.', 1)[0].replace('/', '_'))}"

    todo = [it for it in chosen if image_id_of(it) not in seen and it["url"] not in failed]
    lock = threading.Lock()
    run = Counter()
    failed_fh = open(failed_path, "a")

    def work(it):
        try:
            raw = fetch(it["url"], args.timeout, args.max_mb * 1e6)
            img = open_image(raw)
        except Exception as e:
            return it, None, None, f"download failed ({type(e).__name__})"
        if min(img.size) < args.min_side:
            return it, None, None, "too small"
        if not args.keep_grayscale and is_grayscale(img):
            return it, None, None, "grayscale"
        return it, raw, img, None

    with ThreadPoolExecutor(args.workers) as pool:
        futures = [pool.submit(work, it) for it in todo]
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Incidents1M", mininterval=30):
            it, raw, img, reason = fut.result()
            if reason:
                run[reason] += 1
                if reason.startswith("download failed"):
                    failed_fh.write(it["url"] + "\n")
                continue
            sha = hashlib.sha256(raw).hexdigest()
            with lock:
                if sha in hashes:
                    run["duplicate bytes"] += 1
                    continue
                hashes.add(sha)
            image_id = image_id_of(it)
            try:
                filename, info = save_original(raw, output_dir, image_id, img=img)
            except ValueError:
                run[f"unsupported format ({img.format})"] += 1
                continue
            manifest.add(make_entry(
                image_id=image_id,
                filename=filename,
                source="incidents",
                label="real",
                sensitivity="HIGH",
                scenario=it["incidents"][0],
                split=it["split"],
                original_label=",".join(it["incidents"]),
                source_image_id=it["name"],
                condition="original",
                hf_repo=None,
                seed=args.seed,
                info=info,
                extra={
                    "url": it["url"],
                    "incidents": it["incidents"],
                    "places": it["places"],
                    "selection_tier": it["tier"],
                    "image_sha256": sha,
                    "license": LICENSE,
                },
            ))
            run[f"saved tier {it['tier']}"] += 1
    failed_fh.close()

    total = manifest.export_json()
    manifest.close()
    print(f"\nThis run: {dict(sorted(run.items()))}")
    print(f"DONE — {total} entries in {args.metadata_path}")
    sys.exit(0)


if __name__ == "__main__":
    main()
