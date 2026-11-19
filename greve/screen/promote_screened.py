"""
Move the images that passed a single-proposer CSS-4 screen into a regular
manifest, so the other proposers and the aggregator score only those.

The screen manifest lives in metadata/screen/ (not picked up by the default
manifest list or the totals). An image passes when the screening
proposer gave it css_score >= --min_score. Its entry (with that proposer's
answer) is copied unchanged; resumable — images already promoted are skipped.

  python -m greve.screen.promote_screened metadata/screen/manifest_visualnews_screen.json \
      metadata/manifest_visualnews_extra.json --proposer qwen3-vl-8b --min_score 4
"""

import argparse
import json
from collections import Counter

from greve.common import ManifestWriter


def main():
    parser = argparse.ArgumentParser(description="Promote images that passed a CSS-4 screen")
    parser.add_argument("screen_manifest")
    parser.add_argument("out_manifest")
    parser.add_argument("--proposer", default="qwen3-vl-8b")
    parser.add_argument("--min_score", type=int, default=4)
    args = parser.parse_args()

    entries = json.load(open(args.screen_manifest))
    stats = Counter()
    out = ManifestWriter(args.out_manifest)
    have = {e["image_id"] for e in out.entries}
    for e in entries:
        score = ((e.get("css_proposer_responses") or {}).get(args.proposer) or {}).get("css_score")
        stats["no answer" if score is None else f"score {score}"] += 1
        if score is not None and score >= args.min_score and e["image_id"] not in have:
            out.add({**e, "screened_by": args.proposer, "screen_min_score": args.min_score})
            stats["promoted"] += 1
    total = out.export_json()
    out.close()
    print(f"{args.screen_manifest}: {len(entries)} images, {dict(sorted(stats.items()))}")
    print(f"{total} entries in {args.out_manifest}")


if __name__ == "__main__":
    main()
