"""
Phase 1 — Mine VisualNews for sensitive real photos (CSS-3/4 candidates)
into manifest_visualnews.json, for the GREVE sampling plan
(sampling plan, "VisualNews (mined)" row: CSS-4 real top-up,
caption keywords, v1 method).

Source: the local VisualNews origin.tar (91GB; Guardian, BBC, USA Today,
Washington Post; https://www.cs.rice.edu/~vo9/visualnews/). Its data.json
gives, per image: id, caption, topic, source (outlet), image_path.

Selection (nothing is labelled here — the VLMs score the pool later):
  1. Keyword match on the caption, whole words, in 10 buckets: the v1
     buckets (war_conflict, mass_casualty, graphic_violence, weapons) plus
     attack_explosion, disaster, injury_rescue, unrest, displacement, grief
     (the CSS-3 content of proposer_prompt.txt).
  2. Drop entertainment/sport/lifestyle topics (film, music, sport,
     arts_culture, ...) and captions with fiction/entertainment words (Star
     Wars, film, premiere, video game, re-enactment, ...) — the usual false
     positives ("offensive lineman", "The Walking Dead").
  3. Drop VisualNews ids already in manifest_dgm4.json (DGM4 reals are
     VisualNews photos: one real image, one appearance).
  4. Rank by visual_score — how many terms describe something VISIBLE
     (wounded, rubble, smoke, coffin, tear gas, ...) rather than the story
     ("fatalities in road accidents") — and fill --n images spread evenly
     over the buckets (best visual_score first, ties random), giving a
     bucket's unused share to the others.
Grayscale photos are skipped (as for the other datasets) and the next
candidate is taken. Images are saved as their ORIGINAL bytes (mostly
~256px on the short side). Resumable.

The archive is read through an index of member offsets
(metadata/cache/visualnews_tar_index.pkl), built on the first run (~4 min).

  python -m greve.collect.extract_visualnews --n 15000
  python -m greve.collect.extract_visualnews --dry_run      # only print the pool sizes
"""

import argparse
import json
import pickle
import random
import re
import sys
import tarfile
from collections import Counter
from pathlib import Path

from tqdm import tqdm

from greve.common import ManifestWriter, existing_ids, is_grayscale, make_entry, open_image, save_original
from greve.paths import IMAGES_DIR, RAW_DIR

TAR_PATH = str(Path(RAW_DIR) / "visualnews" / "origin.tar")
INDEX_PATH = Path("metadata") / "cache" / "visualnews_tar_index.pkl"

BUCKETS = {
    "war_conflict": ["war", "wars", "warfare", "battle", "battlefield", "combat", "airstrike", "airstrikes",
                     "air strike", "air strikes", "shelling", "shelled", "bombing", "bombings", "bombed",
                     "gunfire", "firefight", "insurgent", "insurgents", "militant", "militants",
                     "rebel fighters", "invasion", "troops", "frontline", "front line"],
    "mass_casualty": ["killed", "dead", "death toll", "casualty", "casualties", "corpse", "corpses", "bodies",
                      "body bag", "body bags", "massacre", "massacres", "genocide", "fatality", "fatalities",
                      "dying", "slain", "mass grave", "mass graves"],
    "graphic_violence": ["blood", "bloodied", "bloody", "wounded", "injured", "mutilated", "torture", "tortured",
                         "execution", "executions", "executed", "beheaded", "atrocity", "atrocities", "stabbed",
                         "stabbing", "beaten"],
    "attack_explosion": ["attack", "attacks", "terror attack", "terrorist", "terrorists", "explosion", "explosions",
                         "blast", "suicide bomber", "car bomb", "shooting", "gunman", "gunmen", "shooter",
                         "rampage", "hostage", "hostages"],
    "weapons": ["gun", "guns", "rifle", "rifles", "firearm", "firearms", "grenade", "grenades", "explosive",
                "explosives", "bomb", "bombs", "missile", "missiles", "artillery", "weapon", "weapons",
                "ammunition"],
    "disaster": ["earthquake", "quake", "tsunami", "flood", "floods", "flooding", "floodwaters", "hurricane",
                 "typhoon", "cyclone", "tornado", "wildfire", "wildfires", "bushfire", "landslide", "mudslide",
                 "rubble", "debris", "collapsed", "wreckage", "destroyed", "devastation", "devastated", "crash",
                 "derailed"],
    "injury_rescue": ["rescue", "rescuers", "rescued", "paramedic", "paramedics", "ambulance", "stretcher",
                      "survivor", "survivors", "victim", "victims", "field hospital", "evacuated", "evacuation"],
    "unrest": ["clash", "clashes", "riot", "riots", "rioters", "riot police", "tear gas", "teargas", "arrested",
               "detained", "police violence", "crackdown", "water cannon"],
    "displacement": ["refugee", "refugees", "migrants", "asylum seekers", "displaced", "fleeing", "fled",
                     "refugee camp", "starvation", "famine", "malnourished"],
    "grief": ["funeral", "funerals", "mourners", "mourn", "mourning", "coffin", "coffins", "vigil",
              "memorial service", "grieving"],
}
# Terms that describe something visible in the photo (used only for ranking).
VISUAL_TERMS = ["wounded", "injured", "bodies", "body", "corpse", "corpses", "blood", "bloodied", "rubble", "debris",
                "smoke", "flames", "burning", "fire", "ablaze", "destroyed", "collapsed", "wreckage", "damaged",
                "airstrike", "explosion", "blast", "shelling", "clashes", "tear gas", "riot police", "water cannon",
                "refugee camp", "refugees", "coffin", "coffins", "mourners", "stretcher", "ambulance", "rescuers",
                "rescue workers", "firefighters", "paramedics", "carried", "carries", "lies", "lying", "flooded",
                "floodwaters", "survivors", "scene", "aftermath", "soldiers", "fighters", "troops", "gunmen",
                "armed", "weeps", "cries", "crying", "grieve", "grieving", "hospital", "bandaged"]
BAD_TOPICS = {"sport", "football", "sports", "arts_culture", "music", "film", "artanddesign", "books", "stage",
              "fashion", "lifeandstyle", "travel", "tv-and-radio", "money", "media", "culture", "technology",
              "science_technology", "business", "business_economy", "food", "games", "education", "commentisfree"}
NEGATIVE = ["star wars", "walking dead", "grateful dead", "war horse", "world of warcraft", "video game", "videogame",
            "film", "movie", "premiere", "actor", "actress", "actors", "cast", "trailer", "season", "episode", "album",
            "concert", "festival", "fashion", "exhibition", "museum", "painting", "sculpture", "novel", "musical",
            "theatre", "theater", "re-enactment", "reenactment", "reenactors", "rehearsal", "scene from", "still from",
            "play", "game of thrones", "call of duty", "fortnite", "cosplay"]


def _pattern(words):
    return re.compile(r"\b(?:" + "|".join(re.escape(w) for w in words) + r")\b")


BUCKET_RE = {k: _pattern(v) for k, v in BUCKETS.items()}
VISUAL_RE = _pattern(VISUAL_TERMS)
NEGATIVE_RE = _pattern(NEGATIVE)


def load_index(tar_path):
    if INDEX_PATH.exists():
        with open(INDEX_PATH, "rb") as f:
            return pickle.load(f)
    print("Indexing the archive once (~4 min)...")
    index = {}
    with tarfile.open(tar_path, "r:") as tf:
        for m in tqdm(tf, desc="index", unit=" files"):
            if m.isfile():
                index[m.name] = (m.offset_data, m.size)
    INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(INDEX_PATH, "wb") as f:
        pickle.dump(index, f)
    return index


def read_member(fh, index, name):
    offset, size = index[name]
    fh.seek(offset)
    return fh.read(size)


def dgm4_visualnews_ids(path):
    if not Path(path).exists():
        return set()
    with open(path) as f:  # dgm4_real_<VisualNews id>
        return {e["image_id"].rsplit("_", 1)[-1] for e in json.load(f)}


def build_pool(entries, exclude_ids, stats):
    pool = {k: [] for k in BUCKETS}
    for e in entries:
        caption = (e.get("caption") or "").lower()
        matched = {k: sorted(set(p.findall(caption))) for k, p in BUCKET_RE.items()}
        matched = {k: v for k, v in matched.items() if v}
        if not matched:
            continue
        stats["keyword match"] += 1
        if e.get("topic") in BAD_TOPICS:
            stats["dropped: entertainment/sport/lifestyle topic"] += 1
            continue
        if NEGATIVE_RE.search(caption):
            stats["dropped: fiction/entertainment words"] += 1
            continue
        if str(e["id"]) in exclude_ids:
            stats["dropped: already in DGM4"] += 1
            continue
        stats["candidates"] += 1
        item = {"entry": e, "buckets": matched, "visual_score": len(set(VISUAL_RE.findall(caption)))}
        for k in matched:
            pool[k].append(item)
    return pool


def bucket_quotas(n, sizes):
    """Even split of n over the buckets; a bucket's unused share goes to the others."""
    quotas, left, open_ = {k: 0 for k in sizes}, n, [k for k in sizes if sizes[k]]
    while left > 0 and open_:
        share = max(1, left // len(open_))
        for k in list(open_):
            take = min(share, sizes[k] - quotas[k], left)
            quotas[k] += take
            left -= take
            if quotas[k] >= sizes[k]:
                open_.remove(k)
            if left == 0:
                break
    return quotas


def main():
    parser = argparse.ArgumentParser(description="Phase 1 — Mine sensitive VisualNews photos into manifest_visualnews.json")
    parser.add_argument("--n", type=int, default=15000, help="Images to extract (the plan's VisualNews-mined pool)")
    parser.add_argument("--tar", type=str, default=TAR_PATH)
    parser.add_argument("--output_dir", type=str, default=IMAGES_DIR)
    parser.add_argument("--metadata_path", type=str, default="./metadata/manifest_visualnews.json")
    parser.add_argument("--dgm4_manifest", type=str, default="./metadata/manifest_dgm4.json",
                        help="VisualNews ids already used as DGM4 reals are excluded")
    parser.add_argument("--keep_grayscale", action="store_true", help="Don't exclude grayscale originals")
    parser.add_argument("--dry_run", action="store_true", help="Print the pool and quotas, extract nothing")
    parser.add_argument("--buckets", type=str, nargs="+", default=None,
                        help="With --all_remaining: only these buckets (default: all)")
    parser.add_argument("--all_remaining", action="store_true",
                        help="Take every candidate of --buckets not in --exclude_manifest (ignores --n), "
                             "e.g. to screen the rest of the pool for CSS-4")
    parser.add_argument("--exclude_manifest", type=str, nargs="*", default=[],
                        help="VisualNews manifests whose images are already taken")
    parser.add_argument("--seed", type=int, default=9999)
    args = parser.parse_args()

    index = load_index(args.tar)
    data_name = next(n for n in index if n.endswith("data.json"))
    prefix = data_name[: -len("data.json")]
    fh = open(args.tar, "rb")
    entries = json.loads(read_member(fh, index, data_name))
    print(f"VisualNews: {len(entries)} captioned images")

    stats = Counter()
    pool = build_pool(entries, dgm4_visualnews_ids(args.dgm4_manifest), stats)
    rng = random.Random(args.seed)
    for items in pool.values():
        rng.shuffle(items)
        items.sort(key=lambda it: -it["visual_score"])  # stable: random within equal scores
    exclude = set()
    for path in args.exclude_manifest:
        exclude |= {e["vn_id"] for e in json.load(open(path))}
    if args.all_remaining:
        if args.buckets:
            unknown = set(args.buckets) - set(BUCKETS)
            if unknown:
                sys.exit(f"Unknown buckets: {sorted(unknown)}")
        chosen = set(args.buckets or BUCKETS)
        for k in pool:
            pool[k] = [it for it in pool[k] if it["entry"]["id"] not in exclude] if k in chosen else []
        args.n = len({it["entry"]["id"] for v in pool.values() for it in v})
        quotas = {k: len(v) for k, v in pool.items()}
        print(f"  --all_remaining: {args.n} unique images in {sorted(chosen)} not in {args.exclude_manifest}")
    else:
        quotas = bucket_quotas(args.n, {k: len(v) for k, v in pool.items()})
    for k, v in stats.items():
        print(f"  {k}: {v}")
    print("  bucket          candidates  quota")
    for k in BUCKETS:
        print(f"  {k:16s}{len(pool[k]):10d}{quotas[k]:7d}")
    if args.dry_run:
        return

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    Path(args.metadata_path).parent.mkdir(parents=True, exist_ok=True)
    manifest = ManifestWriter(args.metadata_path)
    seen = existing_ids(manifest, "visualnews", output_dir)
    taken = {e["vn_id"]: e.get("selection_bucket") for e in manifest.entries if e["image_id"] in seen}
    print(f"Manifest has {len(seen)} VisualNews images on disk")

    counts = Counter(taken.values())
    run = Counter()
    progress = tqdm(total=args.n, initial=len(taken), desc="VisualNews (mined)")
    for bucket in BUCKETS:
        for item in pool[bucket]:
            if counts[bucket] >= quotas[bucket]:
                break
            e = item["entry"]
            if e["id"] in taken or e["id"] in exclude:
                continue
            name = prefix + e["image_path"].lstrip("./")
            if name not in index:
                run["missing in archive"] += 1
                continue
            raw = read_member(fh, index, name)
            try:
                img = open_image(raw)
            except Exception:
                run["unreadable"] += 1
                continue
            if not args.keep_grayscale and is_grayscale(img):
                run["grayscale"] += 1
                continue
            image_id = f"visualnews_real_{e['id']}"
            filename, info = save_original(raw, output_dir, image_id, img=img)
            manifest.add(make_entry(
                image_id=image_id,
                filename=filename,
                source="visualnews",
                label="real",
                sensitivity="HIGH",
                scenario=e.get("topic"),
                split=None,
                original_label=e.get("topic"),
                source_image_id=e["image_path"],
                condition="original",
                hf_repo=None,
                seed=args.seed,
                info=info,
                extra={
                    "vn_id": e["id"],
                    "caption": e.get("caption"),
                    "topic": e.get("topic"),
                    "outlet": e.get("source"),
                    "article_path": e.get("article_path"),
                    "selection_bucket": bucket,
                    "keyword_buckets": sorted(item["buckets"]),
                    "keywords_matched": sorted({w for ws in item["buckets"].values() for w in ws}),
                    "visual_score": item["visual_score"],
                },
            ))
            taken[e["id"]] = bucket
            counts[bucket] += 1
            run["saved"] += 1
            progress.update(1)
    progress.close()
    fh.close()

    total = manifest.export_json()
    manifest.close()
    print(f"\nThis run: {dict(run)}")
    print(f"Per bucket: {dict(counts)}")
    print(f"DONE — {total} entries in {args.metadata_path}")
    sys.exit(0)


if __name__ == "__main__":
    main()
