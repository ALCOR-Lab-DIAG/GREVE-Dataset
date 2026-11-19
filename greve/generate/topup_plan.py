"""
Top-up waves for the fake side of GREVE: count the usable fakes per CSS level, and if CSS-3 or
CSS-4 (or CSS-2, which our generations only fill by spillover) is still short, write the prompts
and the job queue of the next wave.

Usable fakes per level = the largest N that fits the source caps (OpenFake <= 60% in CSS-2 and
<= 50% elsewhere, any other source <= 50%; each of our generator models is its own source).
Our generated images count only when scored, in colour (grayscale is excluded from every pool),
not flagged sexual_content by >= 2 proposers, and not from CogView4 (dropped).

New prompts, always the same recipe (documentary prefix + a VLM scene description of a real
image), never a (description, model) pair used before:
  CSS-3: real CSS-3 images not used by any earlier prompt (qwen3-vl description first)
  CSS-4: real CSS-4 images are all used already, so a wave takes the description of ANOTHER
         proposer (internvl3 / pixtral) and/or another model; images with the fewest fakes first.
         Harm prompts showing children are rewritten to adults (prompt_edit=minor_to_adult).
Each wave's prompts go to the models by their measured hit rate for that level (every model gets
some, so the generators stay many).

  python -m greve.generate.topup_plan --wave 1          # writes metadata/gen/topup_w1_prompts.jsonl + queue
  python -m greve.generate.topup_plan --status          # counts only
Exit code 0 = a wave was written, 3 = all levels complete (nothing written).
"""

import argparse
import glob
import json
import math
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

from greve.generate.gen_prompts import PREFIX, aspect, clean
from greve.paths import IMAGES_DIR

TARGET = {1: 12000, 2: 12000, 3: 12000, 4: 3000}
GEN_MANIFESTS = ["metadata/gen/manifest_gen_pilot.json", "metadata/gen/manifest_gen_full.json",
                 "metadata/gen/manifest_gen_topup.json"]
PROMPT_FILES = ["metadata/gen/pilot_prompts.jsonl", "metadata/gen/full_prompts.jsonl"] + \
               sorted(glob.glob("metadata/gen/topup_w*_prompts.jsonl"))
MODELS = ["flux-dev", "flux-schnell", "flux-krea", "sdxl", "pixart-sigma", "sana-1.6b", "lumina-2",
          "hunyuandit", "chroma1-hd", "z-image-turbo", "playground-v2.5", "kolors", "auraflow", "kandinsky-3"]
SEC_PER_IMG = {"chroma1-hd": 64, "auraflow": 40, "kandinsky-3": 24, "flux-krea": 20, "flux-dev": 20,
               "hunyuandit": 17.4, "lumina-2": 13, "kolors": 8, "playground-v2.5": 7.6, "z-image-turbo": 5.5,
               "sdxl": 4.9, "pixart-sigma": 3.4, "flux-schnell": 3.0, "sana-1.6b": 2.1}
PROPOSER_ORDER = ["qwen3-vl-8b", "internvl3-8b", "pixtral-12b"]
GRAY_CACHE = Path("metadata/gen/grayscale_cache.json")
OUT_DIR = IMAGES_DIR
SAFETY = 1.15        # more prompts than the expected need (hit rates vary)
MIN_PROMPTS = 40     # a level short by less than this still gets this many prompts

HARM = r"blood|bleed|gore|mutilat|corpse|bodies|body bag|lifeless|\bdead\b|injur|wound|beat|stab|shooting|execution|tortur"
KID = r"\b(child|children|kid|kids|boy|boys|girl|girls|baby|babies|infant|infants|toddler|toddlers|teen|teens|teenager|teenagers)\b"
ADULT = [(r"\b(a|an) young (child|kid|toddler|infant|baby)\b", "an adult"), (r"\byoung (child|kid)\b", "adult"),
         (r"\b(a|an) young boy\b", "a young man"), (r"\b(a|an) young girl\b", "a young woman"),
         (r"\byoung boys\b", "young men"), (r"\byoung girls\b", "young women"),
         (r"\byoung boy\b", "young man"), (r"\byoung girl\b", "young woman"),
         (r"\b(children|kids|infants|babies|toddlers|teenagers|teens)\b", "adults"),
         (r"\b(child|kid|infant|baby|toddler|teenager|teen)'s\b", "person's"),
         (r"\b(a|an) (child|kid|infant|baby|toddler|teenager|teen)\b", "an adult"),
         (r"\b(child|kid|infant|baby|toddler|teenager|teen)\b", "adult"),
         (r"\bboys\b", "men"), (r"\bgirls\b", "women"), (r"\bboy\b", "man"), (r"\bgirl\b", "woman"),
         (r"\badults(,)? (and|or|including) adults\b", "adults")]


def to_adults(s):
    for pat, rep in ADULT:
        s = re.sub(pat, lambda m, rep=rep: rep[0].upper() + rep[1:] if m.group(0)[0].isupper() else rep, s,
                   flags=re.IGNORECASE)
    return re.sub(r"\b([Aa]) adult", lambda m: m.group(1) + "n adult", s)


def source_manifests():
    """The counted manifests of the external sources (manifest_cssgen.json repeats GEN_MANIFESTS)."""
    return [p for p in sorted(glob.glob("metadata/manifest_*.json")) if not p.endswith("manifest_cssgen.json")]


def load(path):
    return json.load(open(path)) if Path(path).exists() else []


def grayscale_of(entries):
    """image_id -> is grayscale, for generated images (cached; generated PNGs never change)."""
    from PIL import Image
    from common import is_grayscale
    cache = json.loads(GRAY_CACHE.read_text()) if GRAY_CACHE.exists() else {}
    new = 0
    for e in entries:
        if e["image_id"] not in cache:
            with Image.open(Path(OUT_DIR) / e["filename"]) as im:
                cache[e["image_id"]] = bool(is_grayscale(im))
            new += 1
    if new:
        GRAY_CACHE.write_text(json.dumps(cache))
    return cache


def sexual(e):
    return sum("sexual_content" in ((r or {}).get("flags") or [])
               for r in (e.get("css_proposer_responses") or {}).values()) >= 2


def usable_n(sources, lvl):
    """Largest N with sum over sources of min(count, cap * N) >= N."""
    def ok(n):
        return sum(min(c, int((0.6 if s == "openfake" and lvl == 2 else 0.5) * n)) for s, c in sources.items()) >= n
    lo, hi = 0, sum(sources.values())
    while lo < hi:
        mid = (lo + hi + 1) // 2
        lo, hi = (mid, hi) if ok(mid) else (lo, mid - 1)
    return lo


def need_images(sources, lvl):
    """Extra images of our own (spread over many generators, each under its cap) to reach the target."""
    t = TARGET[lvl]
    return max(0, t - sum(min(c, int((0.6 if s == "openfake" and lvl == 2 else 0.5) * t)) for s, c in sources.items()))


def count(verbose=True):
    cells = defaultdict(Counter)          # level -> source -> n
    for path in source_manifests():
        for e in load(path):
            if e["label"] == "fake" and e.get("css_score") in TARGET:
                cells[e["css_score"]][e["source_dataset"]] += 1
    gen = [e for p in GEN_MANIFESTS for e in load(p)]
    scored = [e for e in gen if e.get("css_score") in TARGET and e["generator_model"] != "cogview4"]
    gray = grayscale_of(scored)
    drop = Counter()
    for e in scored:
        if gray.get(e["image_id"]):
            drop["grayscale"] += 1
        elif sexual(e):
            drop["sexual_content"] += 1
        else:
            cells[e["css_score"]]["gen:" + e["generator_model"]] += 1
    unscored = sum(e.get("css_score") is None for e in gen)
    usable = {lvl: usable_n(cells[lvl], lvl) for lvl in TARGET}
    if verbose:
        print(f"generated images: {len(gen)} ({unscored} not scored yet); excluded: "
              + ", ".join(f"{k} {v}" for k, v in drop.items()))
        for lvl in sorted(TARGET):
            gen_n = sum(v for s, v in cells[lvl].items() if s.startswith("gen:"))
            print(f"  CSS-{lvl} fakes: {sum(cells[lvl].values()):6d} total, {gen_n:5d} ours, usable {usable[lvl]:6d}"
                  f" / target {TARGET[lvl]}  -> {need_images(cells[lvl], lvl)} more images needed")
    return cells, usable, gen, unscored


def hit_rates(gen):
    """model -> target level -> {final level: share}, from all scored generations."""
    c = defaultdict(lambda: defaultdict(Counter))
    for e in gen:
        if e.get("css_score") in TARGET and e["generator_model"] in MODELS:
            c[e["generator_model"]][e["target_level"]][e["css_score"]] += 1
    rates = {}
    for m in MODELS:
        rates[m] = {}
        for t in (3, 4):
            n = sum(c[m][t].values())
            rates[m][t] = {l: (c[m][t][l] + 1) / (n + 4) for l in TARGET}  # smoothed
    return rates


def real_pool():
    pool = {3: [], 4: []}
    for path in source_manifests():
        for e in load(path):
            if e["label"] == "real" and e.get("css_score") in pool:
                resp = e.get("css_proposer_responses") or {}
                scenes = [(p, (resp.get(p) or {}).get("scene_description")) for p in PROPOSER_ORDER]
                scenes = [(p, s) for p, s in scenes if s and len(s) > 40]
                if scenes:
                    pool[e["css_score"]].append((e, scenes))
    return pool


def plan(wave, seed, dry=None):
    cells, usable, gen, unscored = count()
    if unscored > 0.02 * len(gen) and not dry:  # a few may fail scoring for good
        sys.exit(f"{unscored} generated images not scored yet: score them before planning")
    short = {lvl: need_images(cells[lvl], lvl) for lvl in TARGET}
    if not any(short[l] for l in (2, 3, 4)):
        print("all CSS levels complete")
        return 3
    rates = hit_rates(gen)
    # the extra fakes a level needs, in usable terms: CSS-4 from CSS-4 prompts; CSS-3 from CSS-3 prompts
    # plus CSS-4 spillover; CSS-2 only as spillover (no CSS-2-targeted generation)
    weights = {t: {m: rates[m][t][t] for m in MODELS if not (t == 4 and m == "sdxl")} for t in (3, 4)}
    mean = {t: sum(weights[t].values()) / len(weights[t]) for t in (3, 4)}
    spill = {t: {l: sum(rates[m][t][l] for m in weights[t]) / len(weights[t]) for l in TARGET} for t in (3, 4)}
    n4 = math.ceil(short[4] / mean[4] * SAFETY) if short[4] else 0
    need3 = max(0, short[3] - n4 * spill[4][3])
    n3 = math.ceil(need3 / mean[3] * SAFETY) if need3 else 0
    if short[2] > (n3 * spill[3][2] + n4 * spill[4][2]):
        # CSS-2 only fills by spillover: add CSS-3 prompts until the expected CSS-2 spillover covers it
        n3 = max(n3, math.ceil((short[2] - n4 * spill[4][2]) / spill[3][2] * SAFETY))
    if short[4]:
        n4 = max(n4, MIN_PROMPTS)
    if short[3] or short[2]:
        n3 = max(n3, MIN_PROMPTS)
    print(f"wave {wave}: {n3} CSS-3 prompts, {n4} CSS-4 prompts "
          f"(mean hit rate CSS-3 {mean[3]:.0%}, CSS-4 {mean[4]:.0%})")

    # used (description text, model) pairs, and fakes per real image
    used_pair, used_img, fakes_per_img = set(), set(), Counter()
    for path in PROMPT_FILES:
        for l in open(path):
            p = json.loads(l)
            for m in ([p["model"]] if "model" in p else MODELS):  # pilot prompts went to every model
                used_pair.add((p.get("prompt_original", p["prompt"]), m))
            used_img.add(p["from_image_id"])
            fakes_per_img[p["from_image_id"]] += 1
    pool = real_pool()
    rng = random.Random(seed * 1000 + wave)

    def split(n, t):
        """n prompts over the models by hit rate (largest remainder)."""
        w = weights[t]
        tot = sum(w.values())
        share = {m: n * w[m] / tot for m in w}
        alloc = {m: int(share[m]) for m in w}
        for m in sorted(w, key=lambda m: share[m] - alloc[m], reverse=True)[:n - sum(alloc.values())]:
            alloc[m] += 1
        return alloc

    rows = []

    def add(t, m, e, proposer, scene):
        text = PREFIX + clean(scene)
        row = {"prompt_id": f"w{wave}l{t}_{e['image_id']}__{m}__{proposer.split('-')[0]}", "model": m,
               "target_level": t, "from_image_id": e["image_id"], "from_source": e["source_dataset"],
               "aspect": aspect(e), "scene_proposer": proposer, "wave": wave, "prompt": text}
        if t == 4 and re.search(HARM, text, re.I) and re.search(KID, text, re.I):
            row.update(prompt_original=text, prompt=to_adults(text), prompt_edit="minor_to_adult")
        rows.append(row)
        used_pair.add((text, m))
        fakes_per_img[e["image_id"]] += 1

    # CSS-3: fresh real images
    fresh = [(e, s) for e, s in pool[3] if e["image_id"] not in used_img]
    fresh.sort(key=lambda t: t[0]["image_id"])
    rng.shuffle(fresh)
    k = 0
    for m, n in split(n3, 3).items():
        for e, scenes in fresh[k:k + n]:
            add(3, m, e, *scenes[0])
        k += n
    if k > len(fresh):
        print(f"WARNING: only {len(fresh)} unused real CSS-3 images for {k} CSS-3 prompts")
    # CSS-4: least-used real images first; for each model the first (description, model) pair not used yet
    imgs = pool[4][:]
    imgs.sort(key=lambda t: t[0]["image_id"])
    rng.shuffle(imgs)
    for m, n in split(n4, 4).items():
        got = 0
        for e, scenes in sorted(imgs, key=lambda t: fakes_per_img[t[0]["image_id"]]):
            if got == n:
                break
            for proposer, scene in scenes:
                if (PREFIX + clean(scene), m) not in used_pair:
                    add(4, m, e, proposer, scene)
                    got += 1
                    break
        if got < n:
            print(f"WARNING: {m}: only {got} new CSS-4 (description, model) pairs for {n} prompts")

    out = Path(dry or "metadata/gen") / f"topup_w{wave}_prompts.jsonl"
    out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    # queue: shard so no job is much longer than ~1.5 h, longest jobs first
    per_model = Counter(r["model"] for r in rows)
    jobs = []
    for m, n in per_model.items():
        shards = max(1, min(7, math.ceil(n * SEC_PER_IMG[m] / 5400)))
        jobs += [(n * SEC_PER_IMG[m] / shards, f"{m} {shards} {k}") for k in range(shards)]
    (out.parent / f"topup_w{wave}_queue_all.txt").write_text(
        "".join(j + "\n" for _, j in sorted(jobs, reverse=True)))
    print(f"{len(rows)} prompts -> {out} ({sum(r.get('prompt_edit') is not None for r in rows)} rewritten to adults); "
          f"{len(jobs)} jobs, ~{sum(t for t, _ in jobs) / 3600 / 7:.1f} h on 7 GPUs")
    print("  per model: " + ", ".join(f"{m} {n}" for m, n in per_model.most_common()))
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--wave", type=int)
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--dry", default=None, help="test: ignore unscored images, write into this folder")
    a = ap.parse_args()
    if a.status or not a.wave:
        count()
        sys.exit(0)
    sys.exit(plan(a.wave, a.seed, a.dry))
