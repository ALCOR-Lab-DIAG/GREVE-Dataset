"""
Final balanced selection of GREVE (sampling plan, section 1), from every counted manifest
(metadata/manifest_*.json, our generated fakes = manifest_cssgen.json).

Eligible images: css_score 1-4, in colour, not flagged sexual_content by >= 2 proposers (both labels,
so neither becomes a real/fake shortcut), and one image per near-duplicate cluster within a label
(metadata/dedup/pairs.json from dedup_phash.py; an image is dropped when it is directly similar to an
image kept before it, in the order: original condition, higher resolution, smaller source).

Quotas: real = fake in every level; CSS-1/2/3 = 12,000 per side. CSS-4 = as many as both sides allow
with the SAME content mix (2026-10-16: the CSS-4 fakes match the real content mix, no gore
oversampling): each CSS-4 image gets a content cue from the proposers' scene descriptions (the first that
matches, in order: blood, dead_body, injury, violence, combat_explosion, fire, rubble, else none), and
every cue gets min(real, fake) images per side, so both sides have exactly the same mix. css4_content
groups the cues as in the brief: harm_to_people (blood, dead_body, injury, violence), fire_explosion
(combat_explosion, fire), other (rubble, none).

Source caps per level x label cell: no source above 50% (60% for OpenFake in CSS-2 fake and for any
source in CSS-4 real); each of our generator models is its own source. Inside a cell the quota is
spread as evenly as possible (water-filling): over sources for reals, over (source, generator) for
fakes (most generators), small sources are used in full; images inside a source are drawn at random
(--seed).

Output (the source manifests are not modified):
  metadata/final/css_v2_selection.json   selected entries + phash64, dup_cluster, css4_content, cell
  metadata/final/selection_report.txt    per-cell sources/generators, rules check, exclusions

  python -m greve.curate.select_final
"""

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import glob
import json
import random
import re
from collections import Counter, defaultdict
from multiprocessing import Pool
from pathlib import Path

from PIL import Image

from greve.common import is_grayscale
from greve.generate.topup_plan import sexual
from greve.paths import IMAGES_DIR

IMG_DIR = Path(IMAGES_DIR)
DEDUP = Path("metadata/dedup")
OUT = Path("metadata/final")
TARGET = {1: 12000, 2: 12000, 3: 12000}
CUES = [("blood", r"blood|bleed|gore|mutilat|dismember|severed"),
        ("dead_body", r"corpse|dead bod|bodies|body bag|lifeless|deceased|\bdead\b"),
        ("injury", r"injur|wound"),
        ("violence", r"beat|stab|shooting|fight|assault|execution|tortur|riot|clash"),
        ("combat_explosion", r"explo|bomb|missile|airstrike|combat|shelling"),
        ("fire", r"\bfire\b(?!d)|flame|blaze|burning"),
        ("rubble", r"rubble|collapse")]
STRATUM = {"blood": "harm_to_people", "dead_body": "harm_to_people", "injury": "harm_to_people",
           "violence": "harm_to_people", "combat_explosion": "fire_explosion", "fire": "fire_explosion",
           "rubble": "other", None: "other"}


def content_cue(e):
    resp = e.get("css_proposer_responses") or {}
    text = " ".join(str((resp.get(k) or {}).get("scene_description") or "") for k in resp)
    for name, rx in CUES:
        if re.search(rx, text, re.I):
            return name
    return None


def cap(label, level, source):
    if level == 4 and label == "real":
        return 0.6
    if level == 2 and label == "fake" and source == "openfake":
        return 0.6
    return 0.5


def gray_one(item):
    image_id, filename = item
    with Image.open(IMG_DIR / filename) as im:
        return image_id, bool(is_grayscale(im))


def grayscale_all(entries, workers):
    path = DEDUP / "grayscale.json"
    cache = json.loads(path.read_text()) if path.exists() else {}
    gen_cache = Path("metadata/gen/grayscale_cache.json")
    if gen_cache.exists():
        cache.update({k: v for k, v in json.loads(gen_cache.read_text()).items() if k not in cache})
    todo = [(e["image_id"], e["filename"]) for e in entries if e["image_id"] not in cache]
    if todo:
        print(f"grayscale check of {len(todo)} images ...", flush=True)
        with Pool(workers) as pool:
            cache.update(dict(pool.imap_unordered(gray_one, todo, chunksize=64)))
        path.write_text(json.dumps(cache))
    return cache


def waterfill(counts, n, rng):
    """Integer allocation of n over units, as even as possible, unit u getting at most counts[u]."""
    units = [u for u in counts if counts[u] > 0]
    if n >= sum(counts[u] for u in units):
        return {u: counts[u] for u in units}
    lo, hi = 0, max(counts[u] for u in units)
    while lo < hi:  # smallest level L with sum min(c, L) >= n
        mid = (lo + hi) // 2
        lo, hi = (lo, mid) if sum(min(counts[u], mid) for u in units) >= n else (mid + 1, hi)
    alloc = {u: min(counts[u], lo - 1) for u in units}
    rest = n - sum(alloc.values())
    grow = sorted(u for u in units if counts[u] >= lo)
    rng.shuffle(grow)
    for u in grow[:rest]:
        alloc[u] += 1
    return alloc


def allocate_cell(pool, label, level, n, rng, capped=True):
    """pool: list of entries of one cell. Returns the selected entries (as many as possible up to n)."""
    groups = defaultdict(lambda: defaultdict(list))  # cap source -> unit -> entries
    for e in pool:
        src = e["source_dataset"] if e["source_dataset"] != "cssgen" else "gen:" + e["generator_model"]
        unit = src if label == "real" or src.startswith("gen:") else (src, e.get("generator_model") or "?")
        groups[src][unit].append(e)
    caps = {s: int(cap(label, level, s) * n) if capped else n for s in groups}
    size = {s: {u: len(v) for u, v in us.items()} for s, us in groups.items()}

    def take(s, L):
        return min(caps[s], sum(min(c, L) for c in size[s].values()))

    top = max(len(v) for us in groups.values() for v in us.values())
    lo, hi = 0, top
    while lo < hi:
        mid = (lo + hi) // 2
        lo, hi = (lo, mid) if sum(take(s, mid) for s in groups) >= n else (mid + 1, hi)
    quota = {s: take(s, max(lo - 1, 0)) for s in groups}
    rest = min(n, sum(take(s, lo) for s in groups)) - sum(quota.values())
    grow = sorted(s for s in groups if take(s, lo) > quota[s])
    rng.shuffle(grow)
    i = 0
    while rest > 0 and grow:
        s = grow[i % len(grow)]
        if quota[s] < take(s, lo):
            quota[s] += 1
            rest -= 1
        else:
            grow.remove(s)
            continue
        i += 1
    chosen = []
    for s in sorted(groups):
        per_unit = waterfill(size[s], quota[s], rng)
        for u in sorted(groups[s], key=str):
            items = sorted(groups[s][u], key=lambda e: e["image_id"])
            rng.shuffle(items)
            chosen += items[:per_unit.get(u, 0)]
    return chosen


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--workers", type=int, default=48)
    parser.add_argument("--keep_sexual", action="store_true")
    parser.add_argument("--css4_match", choices=["cue", "group"], default="cue",
                        help="cue: same count per content cue (default); group: per css4_content group only "
                             "(more CSS-4 images, but more blood on the fake side)")
    args = parser.parse_args()
    rng = random.Random(args.seed)
    OUT.mkdir(parents=True, exist_ok=True)

    entries = [e for p in sorted(glob.glob("metadata/manifest_*.json")) for e in json.load(open(p))]
    assert len({e["image_id"] for e in entries}) == len(entries), "an image_id is in two manifests"
    hashes = json.loads((DEDUP / "hashes.json").read_text())
    clusters = json.loads((DEDUP / "clusters.json").read_text())
    scored = [e for e in entries if e.get("css_score") in (1, 2, 3, 4)]
    gray = grayscale_all(scored, args.workers)

    drop = Counter()
    ok = []
    for e in scored:
        if gray[e["image_id"]]:
            drop[(e["label"], "grayscale")] += 1
        elif not args.keep_sexual and sexual(e):
            drop[(e["label"], "sexual_content")] += 1
        elif "error" in hashes.get(e["image_id"], {"error": 1}):
            drop[(e["label"], "unreadable")] += 1
        else:
            ok.append(e)
    # near duplicates (within a label): go through the images best first and drop an image only when it
    # is directly similar to one already kept (clusters chain images that are not similar to each other)
    pool_size = Counter(e["source_dataset"] for e in ok)
    neighbours = defaultdict(set)
    for a, b, _ in json.loads((DEDUP / "pairs.json").read_text()):
        neighbours[a].add(b)
        neighbours[b].add(a)
    kept, dup_out = set(), set()
    for e in sorted((e for e in ok if e["image_id"] in neighbours),
                    key=lambda e: (e.get("condition") not in (None, "original"),
                                   -((e.get("orig_w") or 0) * (e.get("orig_h") or 0)),
                                   pool_size[e["source_dataset"]], e["image_id"])):
        if neighbours[e["image_id"]] & kept:
            dup_out.add(e["image_id"])
            drop[(e["label"], "near_duplicate")] += 1
        else:
            kept.add(e["image_id"])
    ok = [e for e in ok if e["image_id"] not in dup_out]

    cells = defaultdict(list)
    for e in ok:
        cells[(e["label"], e["css_score"])].append(e)
    selected, report = [], []
    for level in (1, 2, 3):
        for label in ("real", "fake"):
            chosen = allocate_cell(cells[(label, level)], label, level, TARGET[level], rng)
            selected += [dict(e, cell=f"{label}_css{level}") for e in chosen]
    # CSS-4: same content strata on both sides
    strata = defaultdict(lambda: defaultdict(list))
    for label in ("real", "fake"):
        for e in cells[(label, 4)]:
            cue = content_cue(e)
            key = (cue or "none") if args.css4_match == "cue" else STRATUM[cue]
            strata[key][label].append(dict(e, css4_content=STRATUM[cue], css4_cue=cue or "none"))
    css4 = {}
    for s in sorted(strata):
        n_s = min(len(strata[s]["real"]), len(strata[s]["fake"]))
        css4[s] = n_s
        for label in ("real", "fake"):
            # caps are checked on the whole CSS-4 cell below; inside a stratum only spread evenly
            chosen = allocate_cell(strata[s][label], label, 4, n_s, rng, capped=False) if n_s < len(strata[s][label]) \
                else strata[s][label]
            assert len(chosen) == n_s
            selected += [dict(e, cell=f"{label}_css4") for e in chosen]
    for e in selected:
        h = hashes[e["image_id"]]
        e["phash64"], e["pixel_sha256"], e["dup_cluster"] = h["phash64"], h["pixel_sha256"], clusters.get(e["image_id"])

    # report + rules check
    def src_of(e):
        return e["source_dataset"] if e["source_dataset"] != "cssgen" else "gen:" + e["generator_model"]

    problems = []
    report.append(f"GREVE final selection (seed {args.seed}): {len(selected)} images")
    report.append("excluded before selection: " + ", ".join(f"{l} {k} {v}" for (l, k), v in sorted(drop.items())))
    report.append(f"CSS-4 matched per {args.css4_match}: " + ", ".join(f"{s} {n}" for s, n in css4.items()))
    real4 = [e for e in selected if e["cell"] == "real_css4"]
    groups4 = Counter(e["css4_content"] for e in real4)
    report.append("CSS-4 content groups (real side): " + ", ".join(
        f"{g} {n} ({n / len(real4):.1%})" for g, n in groups4.most_common()))
    by_cell = defaultdict(list)
    for e in selected:
        by_cell[e["cell"]].append(e)
    for level in (1, 2, 3, 4):
        n = {lab: len(by_cell[f"{lab}_css{level}"]) for lab in ("real", "fake")}
        if n["real"] != n["fake"]:
            problems.append(f"CSS-{level}: real {n['real']} != fake {n['fake']}")
        if level in TARGET and min(n.values()) < TARGET[level]:
            problems.append(f"CSS-{level}: below target {TARGET[level]}: {n}")
        srcs = {lab: Counter(src_of(e) for e in by_cell[f"{lab}_css{level}"]) for lab in ("real", "fake")}
        both = sorted({s for s in srcs["real"]} & {s for s in srcs["fake"]})
        if not both:
            problems.append(f"CSS-{level}: no source gives both real and fake")
        for lab in ("real", "fake"):
            cell = by_cell[f"{lab}_css{level}"]
            report.append(f"\nCSS-{level} {lab}: {len(cell)} images (pool after exclusions {len(cells[(lab, level)])})")
            for s, c in srcs[lab].most_common():
                report.append(f"    {s:22} {c:6d}  {c / len(cell):5.1%}")
                if c > cap(lab, level, s) * len(cell) + 1:
                    problems.append(f"CSS-{level} {lab}: {s} {c / len(cell):.1%} above its cap")
            if len(srcs[lab]) < 3:
                problems.append(f"CSS-{level} {lab}: fewer than 3 sources")
            if lab == "fake":
                gens = Counter((e["source_dataset"], e.get("generator_model")) for e in cell)
                top = gens.most_common(1)[0]
                report.append(f"    generators: {len(gens)}; largest {top[0][0]}/{top[0][1]} {top[1] / len(cell):.1%}")
            if level == 4:
                report.append("    content: " + ", ".join(f"{k} {v}" for k, v in
                                                          sorted(Counter(e['css4_cue'] for e in cell).items())))
        report.append(f"  sources with both real and fake at CSS-{level}: {', '.join(both) or 'NONE'}")
    report.append("\nRULES CHECK: " + ("all passed" if not problems else "PROBLEMS:\n  " + "\n  ".join(problems)))
    tmp = OUT / "css_v2_selection.json.tmp"
    tmp.write_text(json.dumps(selected, ensure_ascii=False))
    tmp.replace(OUT / "css_v2_selection.json")
    (OUT / "selection_report.txt").write_text("\n".join(report) + "\n")
    print("\n".join(report))


if __name__ == "__main__":
    main()
