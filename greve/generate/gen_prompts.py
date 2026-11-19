"""
Build text-to-image prompts for the self-generated fakes from REAL images of a
target CSS level, so the fakes show the same kinds of scenes as the reals of
that level (content can't become a real-vs-fake shortcut).

The prompt is the real image's scene description written by Qwen3-VL (falling
back to InternVL / Pixtral), not its caption: the proposers don't know who is
in a photo, so descriptions carry no names of real people or events, while
captions do. A short documentary-photo prefix keeps the output photographic
(and away from the film-still look the aggregator caps as staged/fictional).

Each prompt also gets the aspect ratio of its real image (landscape / portrait
/ square), so image shape follows the real cell too.

Text-to-image only: the real image itself is never given to a generator, so
every generated image is fully synthetic.

  python -m greve.generate.gen_prompts --levels 3 4 --per_level 150 --out metadata/gen/pilot_prompts.jsonl
"""

import argparse
import glob
import json
import random
import re
from collections import Counter
from pathlib import Path

PROPOSERS = ["qwen3-vl-8b", "internvl3-8b", "pixtral-12b"]
PREFIX = "Candid documentary news photograph, photojournalism, realistic, natural light. "
DROP = [r"^(the|this) (image|photo|photograph|picture) (shows|depicts|captures|features)\s*",
        r"\bin (the|this) (image|photo|photograph|picture)\b", r"\bthe image\b"]


def clean(scene):
    s = scene.strip()
    for pat in DROP:
        s = re.sub(pat, "", s, flags=re.IGNORECASE)
    s = re.sub(r"\s+", " ", s).strip(" ,")
    return s[:1].upper() + s[1:]


def aspect(e):
    w, h = e.get("orig_w") or 1, e.get("orig_h") or 1
    r = w / h
    return "landscape" if r > 1.15 else "portrait" if r < 0.87 else "square"


def main():
    parser = argparse.ArgumentParser(description="Build T2I prompts from real images of target CSS levels")
    parser.add_argument("--levels", type=int, nargs="+", default=[3, 4])
    parser.add_argument("--per_level", type=int, default=150)
    parser.add_argument("--manifests", type=str, default="./metadata/manifest_*.json")
    parser.add_argument("--exclude_used", type=str, nargs="*", default=[], help="prompt files whose source images to skip")
    parser.add_argument("--out", type=str, default="./metadata/gen/pilot_prompts.jsonl")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    used = set()
    for path in args.exclude_used:
        used |= {json.loads(l)["from_image_id"] for l in open(path)}
    pool = {lvl: [] for lvl in args.levels}
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
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(args.out, "w") as fh:
        for lvl in args.levels:
            chosen = rng.sample(pool[lvl], min(args.per_level, len(pool[lvl])))
            print(f"CSS-{lvl}: {len(chosen)} of {len(pool[lvl])} real images; sources "
                  + ", ".join(f"{k} {v}" for k, v in Counter(e["source_dataset"] for e, _ in chosen).most_common()))
            for e, scene in chosen:
                fh.write(json.dumps({"prompt_id": f"l{lvl}_{e['image_id']}", "target_level": lvl,
                                     "from_image_id": e["image_id"], "from_source": e["source_dataset"],
                                     "aspect": aspect(e), "prompt": PREFIX + clean(scene)}, ensure_ascii=False) + "\n")
                n += 1
    print(f"{n} prompts -> {args.out}")


if __name__ == "__main__":
    main()
