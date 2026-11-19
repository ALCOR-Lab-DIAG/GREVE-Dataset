"""
Build the GREVE folder: the final GREVE selection (metadata/final/css_v2_selection.json), every image
as JPEG, so the file format can't become a real/fake shortcut (2026-10-17).

  $GREVE_RELEASE_DIR/{CSS-1,CSS-2,CSS-3,CSS-4}/{real,fake}/<original name>.jpg
  CSS-1/2/3: 12,000 real + 12,000 fake; CSS-4: 3,000 + 3,000

CSS-4: the selection has 3,081 per side matched per content cue; 81 per side are dropped, the same
number per cue on both sides (largest remainder), taken from the largest sources/generators first
(select_final.allocate_cell), so the content mix stays identical on both sides.

Per image (the originals are only read):
  - JPEG originals: the bytes are copied unchanged (same file name)
  - every other format (PNG, WEBP, GIF, MPO, BMP): converted to RGB (transparency over white, first
    frame of animations / multi-picture files) and saved as JPEG, 4:2:0 chroma, with a quality drawn
    from the quality of the REAL JPEGs of the same CSS level (orig_jpeg_q), so real and fake have the
    same quality mix; the draw is seeded by the image_id (reproducible). Name: <original stem>.jpg

$GREVE_RELEASE_DIR/greve_manifest.json: the selection entries + greve_path, greve_action (copied/converted),
greve_from_format, greve_jpeg_q (converted only). Resumable: existing files are kept.

  python -m greve.curate.build_release --workers 48
"""

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import hashlib
import json
import random
import shutil
from collections import Counter, defaultdict
from multiprocessing import Pool
from pathlib import Path

from PIL import Image, ImageSequence

from greve.curate.select_final import allocate_cell
from greve.paths import IMAGES_DIR, RELEASE_DIR

SRC = Path(IMAGES_DIR)
DST = Path(RELEASE_DIR)
TARGET = {1: 12000, 2: 12000, 3: 12000, 4: 3000}
SEED = 2026


def trim_css4(sel, n, rng):
    """Reduce CSS-4 to n per side with the same count per content cue on both sides."""
    keep = [e for e in sel if e["css_score"] != 4]
    by_cue = defaultdict(lambda: defaultdict(list))
    for e in sel:
        if e["css_score"] == 4:
            by_cue[e["css4_cue"]][e["label"]].append(e)
    have = {c: len(v["real"]) for c, v in by_cue.items()}
    assert all(len(v["real"]) == len(v["fake"]) for v in by_cue.values())
    total = sum(have.values())
    exact = {c: have[c] * n / total for c in have}
    quota = {c: int(x) for c, x in exact.items()}
    for c in sorted(have, key=lambda c: (-(exact[c] - quota[c]), c))[:n - sum(quota.values())]:
        quota[c] += 1
    for c in sorted(by_cue):
        for label in ("real", "fake"):
            keep += allocate_cell(by_cue[c][label], label, 4, quota[c], rng, capped=False)
    return keep, quota


def quality_of(image_id, qualities):
    h = int(hashlib.sha256(f"{SEED}:{image_id}".encode()).hexdigest()[:16], 16)
    return qualities[h % len(qualities)]


def to_rgb(im):
    if getattr(im, "n_frames", 1) > 1:
        im = next(iter(ImageSequence.Iterator(im)))
    if im.mode in ("RGBA", "LA") or (im.mode == "P" and "transparency" in im.info):
        im = im.convert("RGBA")
        bg = Image.new("RGB", im.size, (255, 255, 255))
        bg.paste(im, mask=im.getchannel("A"))
        return bg
    if im.mode.startswith("I;16") or im.mode == "I":
        im = im.point(lambda v: v / 256).convert("L")
    return im.convert("RGB")


def work(job):
    src, dst, q, is_jpeg = job
    dst = Path(dst)
    if dst.exists():
        return str(dst), "exists", None
    tmp = dst.with_name(dst.name + ".tmp")
    with Image.open(src) as im:
        fmt = im.format
        if fmt == "JPEG" and is_jpeg:
            shutil.copyfile(src, tmp)
            action = "copied"
        else:
            to_rgb(im).save(tmp, "JPEG", quality=q, subsampling=2, optimize=True)
            action = "converted"
    os.replace(tmp, dst)
    return str(dst), action, fmt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=48)
    parser.add_argument("--dst", default=str(DST), help="output folder (a scratch folder for tests)")
    parser.add_argument("--limit", type=int, default=0, help="test: only the first N images")
    args = parser.parse_args()
    dst_root = Path(args.dst)
    rng = random.Random(SEED)
    sel = json.load(open("metadata/final/css_v2_selection.json"))
    sel, quota4 = trim_css4(sel, TARGET[4], rng)
    cells = Counter((e["css_score"], e["label"]) for e in sel)
    assert all(cells[(l, lab)] == TARGET[l] for l in TARGET for lab in ("real", "fake")), cells

    # real JPEG qualities per level (the copied reals), to draw the conversion quality from
    qualities = {l: sorted(e["orig_jpeg_q"] for e in sel if e["css_score"] == l and e["label"] == "real"
                           and e.get("orig_format") == "JPEG" and e.get("orig_jpeg_q")) for l in TARGET}
    jobs, out = [], []
    names = Counter()
    for e in sorted(sel, key=lambda e: e["image_id"]):
        is_jpeg = e.get("orig_format") == "JPEG"
        name = e["filename"] if is_jpeg else Path(e["filename"]).stem + ".jpg"
        rel = Path(f"CSS-{e['css_score']}") / e["label"] / name
        names[str(rel)] += 1
        q = quality_of(e["image_id"], qualities[e["css_score"]])  # used only if the image is converted
        jobs.append((str(SRC / e["filename"]), str(dst_root / rel), q, is_jpeg))
        out.append(dict(e, greve_path=str(rel), greve_from_format=e.get("orig_format"), greve_jpeg_q=q))
    dup = [n for n, c in names.items() if c > 1]
    assert not dup, f"name collisions: {dup[:5]}"
    for l in TARGET:
        for lab in ("real", "fake"):
            (dst_root / f"CSS-{l}" / lab).mkdir(parents=True, exist_ok=True)
    if args.limit:
        pick = set(random.Random(1).sample(range(len(jobs)), args.limit))
        jobs = [j for i, j in enumerate(jobs) if i in pick]
        out = [o for i, o in enumerate(out) if i in pick]

    stats, fmt_seen, actions = Counter(), Counter(), {}
    with Pool(args.workers) as pool:
        for n, (path, action, fmt) in enumerate(pool.imap_unordered(work, jobs, chunksize=32), 1):
            stats[action] += 1
            actions[str(Path(path).relative_to(dst_root))] = (action, fmt)
            if fmt:
                fmt_seen[(action, fmt)] += 1
            if n % 10000 == 0:
                print(f"  {n}/{len(jobs)}", flush=True)
    # the action actually taken: a file the manifest calls JPEG that decodes as something else is converted;
    # files already present from an earlier run: converted iff the source was not a JPEG
    for o in out:
        action, fmt = actions[o["greve_path"]]
        if action == "exists":
            action = "copied" if o["greve_from_format"] == "JPEG" else "converted"
        elif fmt and fmt != o["greve_from_format"]:
            o["greve_from_format"] = fmt
        o["greve_action"] = action
        if action == "copied":
            o["greve_jpeg_q"] = None
    tmp = dst_root / "greve_manifest.json.tmp"
    tmp.write_text(json.dumps(out, ensure_ascii=False))
    tmp.replace(dst_root / "greve_manifest.json")

    print(f"GREVE: {len(jobs)} images -> {dst_root}  {dict(stats)}  formats: {dict(fmt_seen)}")
    print("CSS-4 per content cue (per side):", quota4)
    for l in TARGET:
        for lab in ("real", "fake"):
            c = [o for o in out if o["css_score"] == l and o["label"] == lab]
            if not c:
                continue
            qs = sorted([o["greve_jpeg_q"] or o.get("orig_jpeg_q") or 0 for o in c])
            print(f"  CSS-{l} {lab}: {len(c)} files, {sum(o['greve_action'] == 'converted' for o in c)} converted, "
                  f"JPEG quality p10/p50/p90 {qs[len(qs) // 10]}/{qs[len(qs) // 2]}/{qs[len(qs) * 9 // 10]}")
    s4 = {lab: Counter(o["source_dataset"] for o in out if o["css_score"] == 4 and o["label"] == lab) for lab in ("real", "fake")}
    if s4["real"]:
        top = s4["real"].most_common(1)[0]
        print(f"  CSS-4 real largest source: {top[0]} {top[1]} ({top[1] / sum(s4['real'].values()):.1%})")


if __name__ == "__main__":
    main()
