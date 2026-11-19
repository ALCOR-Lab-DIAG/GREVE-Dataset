"""
Prompts for the full generation run (CSS-3 / CSS-4 fakes), each assigned to one model.

Same prompt recipe as the pilot (gen_prompts.py: the real image's scene description + the
documentary-photo prefix + its aspect ratio). Real images used by the pilot are skipped.

  CSS-3: one prompt per real image (plenty of real CSS-3), --css3 N per model (--css3_sdxl for SDXL,
         which is weak at CSS-4 and gets no CSS-4 prompts)
  CSS-4: --css4 N per model; real CSS-4 images are few, so the list is cycled and an image can
         be used again, always by a different model (prompt_id carries the model)

  python -m greve.generate.gen_prompts_full --out metadata/gen/full_prompts.jsonl
"""

import argparse
import glob
import json
import random
from collections import Counter

from greve.generate.gen_prompts import PREFIX, PROPOSERS, aspect, clean

MODELS = ["flux-dev", "flux-schnell", "flux-krea", "sdxl", "pixart-sigma", "sana-1.6b", "lumina-2",
          "hunyuandit", "chroma1-hd", "z-image-turbo", "playground-v2.5", "kolors", "auraflow", "kandinsky-3"]
NO_CSS4 = {"sdxl"}


def main():
    parser = argparse.ArgumentParser(description="Build model-assigned prompts for the full generation run")
    parser.add_argument("--models", nargs="+", default=MODELS)
    parser.add_argument("--css3", type=int, default=600)
    parser.add_argument("--css3_sdxl", type=int, default=950)
    parser.add_argument("--css4", type=int, default=355)
    parser.add_argument("--manifests", default="./metadata/manifest_*.json")
    parser.add_argument("--exclude_used", nargs="*", default=["./metadata/gen/pilot_prompts.jsonl"])
    parser.add_argument("--out", default="./metadata/gen/full_prompts.jsonl")
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    used = set()
    for path in args.exclude_used:
        used |= {json.loads(l)["from_image_id"] for l in open(path)}
    pool = {3: [], 4: []}
    for path in sorted(glob.glob(args.manifests)):
        for e in json.load(open(path)):
            if e["label"] != "real" or e.get("css_score") not in pool or e["image_id"] in used:
                continue
            responses = e.get("css_proposer_responses") or {}
            scene = next(((responses.get(m) or {}).get("scene_description") for m in PROPOSERS
                          if (responses.get(m) or {}).get("scene_description")), None)
            if scene and len(scene) > 40:
                pool[e["css_score"]].append((e, scene))
    rng = random.Random(args.seed)
    for lvl in pool:
        pool[lvl].sort(key=lambda t: t[0]["image_id"])
        rng.shuffle(pool[lvl])

    rows = []
    def add(lvl, model, e, scene):
        rows.append({"prompt_id": f"l{lvl}_{e['image_id']}__{model}", "model": model, "target_level": lvl,
                     "from_image_id": e["image_id"], "from_source": e["source_dataset"],
                     "aspect": aspect(e), "prompt": PREFIX + clean(scene)})

    # CSS-3: distinct real images
    k = 0
    for m in args.models:
        n = args.css3_sdxl if m == "sdxl" else args.css3
        for e, scene in pool[3][k:k + n]:
            add(3, m, e, scene)
        k += n
    if k > len(pool[3]):
        raise SystemExit(f"only {len(pool[3])} real CSS-3 images for {k} prompts")
    # CSS-4: cycle the real images; a model gets a contiguous block shorter than the pool,
    # so an image used twice is always used by two different models
    k = 0
    for m in args.models:
        if m in NO_CSS4:
            continue
        for _ in range(args.css4):
            e, scene = pool[4][k % len(pool[4])]
            add(4, m, e, scene)
            k += 1
    with open(args.out, "w") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    c = Counter((r["target_level"], r["model"]) for r in rows)
    print(f"real pool: CSS-3 {len(pool[3])}, CSS-4 {len(pool[4])} (pilot images excluded)")
    for lvl in (3, 4):
        print(f"CSS-{lvl}: {sum(v for (l, _), v in c.items() if l == lvl)} prompts; sources "
              + ", ".join(f"{s} {n}" for s, n in Counter(r['from_source'] for r in rows if r['target_level'] == lvl).most_common()))
    print(f"{len(rows)} prompts -> {args.out}")


if __name__ == "__main__":
    main()
