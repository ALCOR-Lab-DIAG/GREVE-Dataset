"""
Dedup pass of the GREVE sampling plan, steps (1) and (2): exact duplicates (SHA-256 of the decoded
RGB pixels) and near duplicates (64-bit pHash, Hamming distance <= --max_dist), within each label
(real / fake) over every counted manifest (metadata/manifest_*.json, our generated fakes included).
Step (3) of the plan (DINOv2/CLIP cosine, for crops) is not done here.

Read-only for the manifests: results go to metadata/dedup/
  hashes.json      image_id -> {"phash64": 16 hex, "pixel_sha256": hex}   (cache: only new images are hashed)
  pairs.json       [[image_id_a, image_id_b, distance], ...] near/exact duplicate pairs within a label
  clusters.json    image_id -> dup_cluster id (connected components of pairs.json; information only:
                   chains link images that are not similar, so select_final.py drops an image only
                   when it is DIRECTLY similar to an image it keeps)
  pairs_cross_label.json  real/fake pairs within --max_dist (reported, not clustered: caption pairs are intended)
Threshold: a visual check of sample pairs (2026-10-17) showed distance <= 6 is almost always the same
photo (recompressed, resized, small crop or overlay), 8 is mixed and 10 mostly different images with
flat backgrounds or dark scenes, hence --max_dist 6 (the plan's 10 is too loose here). Even at <= 6,
dark night scenes and objects on white backgrounds collide, so a pHash pair is confirmed only if the
16x16 RGB thumbnails are also close (mean absolute difference <= --max_thumb_mae, 0-255 scale): all 37
real-fake pHash pairs, all checked by eye and all different images, had MAE >= 10.6; same-photo pairs
(recompressions, crops, burst frames) are mostly < 25.
pHash = imagehash's phash: grayscale, 32x32 Lanczos, 2-D DCT, the 8x8 lowest frequencies vs their median.

  python -m greve.curate.dedup_phash --workers 64
"""

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")  # one thread per worker: 64 workers x 128 BLAS threads hit the pids limit

import argparse
import glob
import hashlib
import json
from collections import Counter, defaultdict
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.fft import dctn
from greve.paths import IMAGES_DIR

IMG_DIR = Path(IMAGES_DIR)
OUT = Path("metadata/dedup")


def hash_one(item):
    image_id, filename = item
    try:
        with Image.open(IMG_DIR / filename) as im:
            rgb = im.convert("RGB")
            sha = hashlib.sha256(rgb.tobytes() + f"{rgb.size}".encode()).hexdigest()
            small = np.asarray(rgb.convert("L").resize((32, 32), Image.LANCZOS), dtype=np.float64)
        low = dctn(small, type=2, norm="ortho")[:8, :8].flatten()
        bits = low > np.median(low)
        return image_id, {"phash64": f"{int(''.join('1' if b else '0' for b in bits), 2):016x}", "pixel_sha256": sha}
    except Exception as ex:  # unreadable file: report, never silently drop
        return image_id, {"error": repr(ex)}


def thumb_one(item):
    image_id, filename = item
    with Image.open(IMG_DIR / filename) as im:
        return image_id, np.asarray(im.convert("RGB").resize((16, 16), Image.BILINEAR), dtype=np.float32)


def near_pairs(hashes, max_dist, block=2048):
    """All index pairs (i < j) with Hamming(hashes[i], hashes[j]) <= max_dist."""
    h = np.array(hashes, dtype=np.uint64)
    out = []
    for s in range(0, len(h), block):
        d = np.bitwise_count(h[s:s + block, None] ^ h[None, s:])  # block x (n - s)
        i, j = np.nonzero(d <= max_dist)
        keep = j > i  # j is relative to s, i to s as well
        out += list(zip((i[keep] + s).tolist(), (j[keep] + s).tolist()))
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_dist", type=int, default=6)
    parser.add_argument("--max_thumb_mae", type=float, default=25.0)
    parser.add_argument("--workers", type=int, default=64)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    entries = {}
    for path in sorted(glob.glob("metadata/manifest_*.json")):
        for e in json.load(open(path)):
            entries[e["image_id"]] = (e["filename"], e["label"], e["source_dataset"])
    cache_path = OUT / "hashes.json"
    cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}
    todo = [(i, f) for i, (f, _, _) in entries.items() if i not in cache or "error" in cache[i]]
    print(f"{len(entries)} images in the counted manifests, {len(todo)} to hash")
    if todo:
        with Pool(args.workers) as pool:
            for n, (i, h) in enumerate(pool.imap_unordered(hash_one, todo, chunksize=64), 1):
                cache[i] = h
                if n % 20000 == 0:
                    print(f"  hashed {n}/{len(todo)}", flush=True)
        cache_path.write_text(json.dumps(cache))
    errors = [i for i in entries if "error" in cache[i]]
    if errors:
        print(f"WARNING: {len(errors)} unreadable images, e.g. {errors[:3]}")

    ids = [i for i in entries if "error" not in cache[i]]
    clusters, stats = {}, Counter()
    parent = {}

    def find(x):
        while parent.get(x, x) != x:
            parent[x] = parent.get(parent[x], parent[x])
            x = parent[x]
        return x

    pair_out = []
    by_label = defaultdict(list)
    for i in ids:
        by_label[entries[i][1]].append(i)
    for label, lid in by_label.items():
        # exact pixel duplicates
        by_sha = defaultdict(list)
        for i in lid:
            by_sha[cache[i]["pixel_sha256"]].append(i)
        for group in by_sha.values():
            for x in group[1:]:
                parent[find(x)] = find(group[0])
                pair_out.append([group[0], x, 0])
                stats[f"{label}: exact pixel duplicates"] += 1
        # near duplicates: pHash, confirmed by the thumbnails
        pairs = near_pairs([int(cache[i]["phash64"], 16) for i in lid], args.max_dist)
        need = sorted({lid[x] for p in pairs for x in p})
        with Pool(args.workers) as pool:
            thumbs = dict(pool.imap_unordered(thumb_one, [(i, entries[i][0]) for i in need], chunksize=32))
        confirmed = []
        for a, b in pairs:
            if float(np.abs(thumbs[lid[a]] - thumbs[lid[b]]).mean()) <= args.max_thumb_mae:
                confirmed.append((a, b))
            else:
                stats[f"{label}: pHash pairs rejected by the thumbnail check"] += 1
        for a, b in confirmed:
            stats[f"{label}: pHash pairs <= {args.max_dist}"] += 1
            pair_out.append([lid[a], lid[b], int(bin(int(cache[lid[a]]["phash64"], 16) ^ int(cache[lid[b]]["phash64"], 16)).count("1"))])
            ra, rb = find(lid[a]), find(lid[b])
            if ra != rb:
                parent[ra] = rb
    groups = defaultdict(list)
    for i in ids:
        groups[find(i)].append(i)
    n = 0
    src_pairs = Counter()
    for root, members in groups.items():
        if len(members) < 2:
            continue
        n += 1
        cid = f"dup{n:06d}"
        for m in members:
            clusters[m] = cid
        stats[f"{entries[members[0]][1]}: clusters"] += 1
        stats[f"{entries[members[0]][1]}: images in clusters"] += len(members)
        src_pairs[(entries[members[0]][1],) + tuple(sorted({entries[m][2] for m in members}))] += 1
    (OUT / "clusters.json").write_text(json.dumps(clusters))
    (OUT / "pairs.json").write_text(json.dumps(pair_out))

    # real-fake near pairs (reported only)
    real, fake = by_label.get("real", []), by_label.get("fake", [])
    hr = np.array([int(cache[i]["phash64"], 16) for i in real], dtype=np.uint64)
    cross = []
    for s in range(0, len(fake), 2048):
        hf = np.array([int(cache[i]["phash64"], 16) for i in fake[s:s + 2048]], dtype=np.uint64)
        d = np.bitwise_count(hf[:, None] ^ hr[None, :])
        for a, b in zip(*np.nonzero(d <= args.max_dist)):
            cross.append({"fake": fake[s + a], "real": real[b], "dist": int(d[a, b])})
    if cross:
        need = sorted({c[k] for c in cross for k in ("fake", "real")})
        with Pool(args.workers) as pool:
            thumbs = dict(pool.imap_unordered(thumb_one, [(i, entries[i][0]) for i in need], chunksize=32))
        for c in cross:
            c["thumb_mae"] = round(float(np.abs(thumbs[c["fake"]] - thumbs[c["real"]]).mean()), 1)
        cross = [c for c in cross if c["thumb_mae"] <= args.max_thumb_mae]
    (OUT / "pairs_cross_label.json").write_text(json.dumps(cross))

    for k, v in sorted(stats.items()):
        print(f"  {k}: {v}")
    sizes = Counter(Counter(clusters.values()).values())
    print(f"  cluster sizes: {dict(sorted(sizes.items())[:10])} ... largest {max(sizes) if sizes else 0}")
    print(f"  real-fake pairs <= {args.max_dist}: {len(cross)}  "
          f"(by fake source: {dict(Counter(entries[c['fake']][2] for c in cross).most_common())})")
    print("  clusters by (label, sources):")
    for k, v in src_pairs.most_common(25):
        print(f"    {v:6d}  {k}")


if __name__ == "__main__":
    main()
