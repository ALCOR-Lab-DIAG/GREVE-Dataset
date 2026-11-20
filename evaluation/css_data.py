"""The dataset manifest (JSON list or JSON Lines, one record per image) and the evaluation split.

Fields used:  image_id -> id, triage_path -> image file (relative to --root), label "real"/"fake",
css_score -> CSS level 1..4. Breakdown columns: source_dataset, generator_model, generator_type,
condition, scenario.
Subset: `--subset test` keeps only the records whose `triage_split` is "test" (train/val/test folders).
Split: `--split 0.7` keeps 70% of the images, chosen by a hash of the pair (a fake goes with the real
it is `paired_with`), so the same images are used by every detector and by the scoring.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

LEVELS = (1, 2, 3, 4)
BREAKDOWN = ("source_dataset", "generator_model", "generator_type", "condition", "scenario")


def read_records(path) -> list[dict]:
    text = Path(path).read_text(encoding="utf-8").strip()
    if text.startswith("["):
        return json.loads(text)
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def in_split(rec: dict, frac: float, seed: int = 0) -> bool:
    group = rec.get("paired_with") if rec["label"] == "fake" and rec.get("paired_with") else rec["image_id"]
    h = int(hashlib.sha256(f"{seed}:{group}".encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    return h < frac


def load(manifest, root=None, split: float = 1.0, seed: int = 0, subset: str | None = None,
         image_dir=None) -> list[dict]:
    """Rows with id, path (absolute), label 0/1, css 1..4 and the breakdown columns.
    image_dir: read every image from this flat folder (file name of triage_path) instead of root/triage_path."""
    root = (Path(root) if root else Path(manifest).parent).resolve()
    rows = []
    for rec in read_records(manifest):
        if subset and rec.get("triage_split") != subset:
            continue
        if split < 1.0 and not in_split(rec, split, seed):
            continue
        if rec["label"] not in ("real", "fake"):
            raise ValueError(f"{rec['image_id']}: label must be real/fake, got {rec['label']!r}")
        css = int(rec["css_score"])
        if css not in LEVELS:
            raise ValueError(f"{rec['image_id']}: css_score must be one of {LEVELS}, got {css!r}")
        path = Path(image_dir) / Path(rec["triage_path"]).name if image_dir else root / rec["triage_path"]
        rows.append({"id": rec["image_id"], "path": str(path),
                     "label": int(rec["label"] == "fake"), "css": css,
                     **{c: "" if rec.get(c) is None else str(rec[c]) for c in BREAKDOWN}})
    return rows
