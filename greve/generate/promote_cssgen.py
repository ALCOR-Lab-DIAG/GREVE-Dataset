"""
Promote our generated fakes (pilot + full run + top-up waves) into one counted manifest,
metadata/manifest_cssgen.json, for the final GREVE selection.

An image is promoted only if it can be used:
  - scored (css_score 1-4)
  - not from CogView4 (dropped: it softens harmful prompts)
  - in colour (grayscale is excluded from every pool; cache metadata/gen/grayscale_cache.json)
  - not flagged sexual_content by >= 2 proposers
The others go to metadata/gen/cssgen_excluded.json with the reason. Entries are copied unchanged,
plus gen_run (pilot / full / topup) and the prompt edits recorded in the prompt files (prompt_edit,
prompt_original: the CSS-4 harm prompts whose children were rewritten to adults; joined on model + prompt
text, since the full run started before generate.py copied these fields). Re-running rebuilds both files.

  python -m greve.generate.promote_cssgen
"""

import json
from collections import Counter
from pathlib import Path

from greve.generate.topup_plan import GEN_MANIFESTS, PROMPT_FILES, grayscale_of, load, sexual

PROMPT_FIELDS = ("prompt_edit", "prompt_original")

OUT = Path("metadata/manifest_cssgen.json")
EXCLUDED = Path("metadata/gen/cssgen_excluded.json")


def main():
    entries = []
    for path in GEN_MANIFESTS:
        run = Path(path).stem.replace("manifest_gen_", "")
        entries += [{**e, "gen_run": run} for e in load(path)]
    edits = {}
    for path in PROMPT_FILES:
        for line in open(path):
            p = json.loads(line)
            if any(k in p for k in PROMPT_FIELDS):
                edits[(p.get("model"), p["prompt"])] = {k: p[k] for k in PROMPT_FIELDS if k in p}
    n_edit = 0
    for e in entries:
        extra = edits.get((e["generator_model"], e.get("prompt")))
        if extra:
            e.update(extra)
            n_edit += 1
    print(f"prompt edits joined: {n_edit} images (of {len(edits)} edited prompts)")
    gray = grayscale_of([e for e in entries if e.get("css_score") in (1, 2, 3, 4)])
    keep, excluded, stats = [], [], Counter()
    for e in entries:
        if e.get("css_score") not in (1, 2, 3, 4):
            reason = "not_scored"
        elif e["generator_model"] == "cogview4":
            reason = "cogview4"
        elif gray.get(e["image_id"]):
            reason = "grayscale"
        elif sexual(e):
            reason = "sexual_content"
        else:
            keep.append(e)
            stats[f"CSS-{e['css_score']}"] += 1
            continue
        excluded.append({"image_id": e["image_id"], "filename": e["filename"], "generator_model": e["generator_model"],
                         "css_score": e.get("css_score"), "reason": reason})
        stats["excluded: " + reason] += 1
    assert len({e["image_id"] for e in keep}) == len(keep), "duplicate image_id across generation runs"
    for path, data in ((OUT, keep), (EXCLUDED, excluded)):
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(data, ensure_ascii=False))
        tmp.replace(path)
    print(f"{len(entries)} generated images -> {len(keep)} promoted to {OUT}, {len(excluded)} excluded ({EXCLUDED})")
    for k, v in sorted(stats.items()):
        print(f"  {k}: {v}")
    print("  promoted per generator:", dict(Counter(e["generator_model"] for e in keep).most_common()))


if __name__ == "__main__":
    main()
