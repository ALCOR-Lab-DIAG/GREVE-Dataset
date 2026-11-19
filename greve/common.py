"""
Shared helpers for the extract_<dataset>.py scripts (Phase 1 extraction).

Each extract_<dataset>.py pulls one source dataset end to end and appends
its rows to a shared manifest (an append-only manifest.jsonl journal,
exported to manifest.json at the end of each run). Because the scripts are
meant to be run independently of each other — including at the same time,
one per dataset — ManifestWriter takes an exclusive file lock around every
append so concurrent runs can't interleave each other's entries.
"""

import fcntl
import hashlib
import io
import json
import os
import re
from contextlib import contextmanager
from datetime import date
from pathlib import Path

import numpy as np
from PIL import Image

# Pillow refuses images over ~179 megapixels as a "decompression bomb" guard.
# Some source images (e.g. RRDataset reals, up to ~180MP) are legitimately
# that large and come from trusted datasets, so lift the cap for every script
# that imports this module (extractors, proposers).
Image.MAX_IMAGE_PIXELS = None


# Original container format -> file extension. MPO is a multi-picture JPEG
# (some cameras), stored as .jpg.
FORMAT_EXT = {"JPEG": ".jpg", "MPO": ".jpg", "PNG": ".png", "WEBP": ".webp",
              "BMP": ".bmp", "GIF": ".gif", "TIFF": ".tif"}

# Standard JPEG luminance quantization table (quality 50). Sums are
# order-independent, so zigzag vs natural ordering doesn't matter.
_STD_LUMA_SUM = sum([
    16, 11, 10, 16, 24, 40, 51, 61, 12, 12, 14, 19, 26, 58, 60, 55,
    14, 13, 16, 24, 40, 57, 69, 56, 14, 17, 22, 29, 51, 87, 80, 62,
    18, 22, 37, 56, 68, 109, 103, 77, 24, 35, 55, 64, 81, 104, 113, 92,
    49, 64, 78, 87, 103, 121, 120, 101, 72, 92, 95, 98, 112, 100, 103, 99,
])


def estimate_jpeg_quality(img):
    """Estimate the IJG quality factor a JPEG was saved with, from its luminance quantization table."""
    q = getattr(img, "quantization", None)
    if not q or 0 not in q:
        return None
    scale = 100.0 * sum(q[0]) / _STD_LUMA_SUM
    quality = (200 - scale) / 2 if scale <= 100 else 5000 / scale
    return int(round(min(max(quality, 1), 100)))


def image_info(img):
    """Original-format fields recorded in every manifest entry."""
    return {
        "orig_format": img.format,
        "orig_mode": img.mode,
        "orig_w": img.width,
        "orig_h": img.height,
        "orig_jpeg_q": estimate_jpeg_quality(img) if img.format in ("JPEG", "MPO") else None,
    }


def open_image(raw_bytes):
    """Open and fully decode image bytes (raises on corrupt/truncated data)."""
    img = Image.open(io.BytesIO(raw_bytes))
    img.load()
    return img


def is_grayscale(img):
    """
    True for grayscale originals: a grayscale mode, or RGB whose three
    channels are (near-)identical. Pools exclude these so "grayscale" can't
    become a real-vs-fake shortcut (generator outputs are virtually all color).
    """
    if img.mode in ("1", "L", "LA", "I", "I;16", "F"):
        return True
    rgb = np.asarray(img.convert("RGB").resize((64, 64)), dtype=np.int16)
    return int(np.abs(rgb[..., 0] - rgb[..., 1]).max()) <= 2 and int(np.abs(rgb[..., 1] - rgb[..., 2]).max()) <= 2


def save_original(raw_bytes, output_dir, image_id, img=None):
    """
    Write the image's ORIGINAL bytes unchanged — no decoding/re-encoding, so
    compression and generator traces are preserved — as <image_id><ext>,
    where ext matches the real container format (.jpg, .png, ...).

    The bytes are decoded once first to make sure they're a valid image.
    Returns (filename, image_info dict). Pass `img` if already opened.
    """
    if img is None:
        img = open_image(raw_bytes)
    ext = FORMAT_EXT.get(img.format)
    if ext is None:
        raise ValueError(f"unsupported image format {img.format!r}")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{image_id}{ext}"
    tmp_path = output_dir / (filename + ".tmp")
    with open(tmp_path, "wb") as f:
        f.write(raw_bytes)
    os.replace(tmp_path, output_dir / filename)
    return filename, image_info(img)


ORIG_FIELDS = ("orig_format", "orig_mode", "orig_w", "orig_h", "orig_jpeg_q")


def make_entry(image_id, filename, source, label, sensitivity, scenario, split,
               generator=None, original_label=None, hf_repo=None, seed=42, info=None,
               source_image_id=None, base_image_id=None, paired_with=None,
               generator_type=None, generator_release=None, condition=None, extra=None):
    """
    Build a manifest entry. Every entry has the same fields, for every source:

      - common fields, and the Phase 2 CSS fields pre-populated as null
      - GREVE provenance (sampling plan): source_image_id (the
        source dataset's own id), base_image_id (the photo a fake/tamper was
        made from), paired_with (image_id of a caption-paired real/fake),
        generator_type, generator_release, condition (e.g. RRDataset's
        original/transfer/redigital) — null when a source can't fill them
      - orig_format/orig_mode/orig_w/orig_h/orig_jpeg_q from `info`
        (save_original() returns it alongside `filename`)
      - phash64 and dup_cluster, filled later by the dedup pass

    `extra` holds fields only one source has (e.g. OpenFake's prompt).
    """
    entry = {
        "image_id": image_id,
        "filename": filename,
        "source_dataset": source,
        "hf_repo": hf_repo,
        "split": split,
        "label": label,                        # "real" or "fake"
        "sensitivity_tier": sensitivity,
        "scenario": scenario,
        "generator_model": generator,
        "original_label_value": original_label,
        "css_score": None,
        "css_proposer_responses": {},          # proposer key -> parsed response (proposer_common.py)
        "css_aggregated_response": None,
        "sampling_date": str(date.today()),
        "random_seed": seed,
        "source_image_id": source_image_id,
        "base_image_id": base_image_id,
        "paired_with": paired_with,
        "generator_type": generator_type,
        "generator_release": generator_release,
        "condition": condition,
        **{k: (info or {}).get(k) for k in ORIG_FIELDS},
        "phash64": None,
        "dup_cluster": None,
    }
    entry.update(extra or {})
    return entry


def sanitize_key(key, maxlen=80):
    """
    Turn an arbitrary dataset-native identifier into something safe to use
    inside a filename. Non-alphanumeric runs become a single underscore;
    identifiers longer than `maxlen` are truncated and given a short hash
    suffix so distinct long keys can't collide after truncation.
    """
    key = re.sub(r"[^A-Za-z0-9]+", "_", str(key)).strip("_")
    if len(key) > maxlen:
        h = hashlib.sha1(key.encode()).hexdigest()[:10]
        key = f"{key[:maxlen]}_{h}"
    return key


class ManifestWriter:
    """
    Appends entries to a manifest that other extract_<dataset>.py processes
    may be writing to at the same time.

    The source of truth is an append-only JSON Lines journal next to the
    manifest (manifest.json -> manifest.jsonl), one entry per line. add()
    appends a single line under an exclusive lock on a sidecar `.lock`
    file, so its cost is constant no matter how many entries exist —
    unlike rewriting the whole JSON array on every add, which is O(N) per
    image and O(N^2) per run.

    export_json() rebuilds the plain JSON-array manifest.json (the format
    downstream consumers read) from the journal; the extract scripts call
    it once at the end of a run. If a journal doesn't exist yet but an
    old-style manifest.json does, it is converted to a journal on first use.

    `entries` reflects the journal as of the last load; call refresh() to
    pick up other processes' writes.
    """

    def __init__(self, metadata_path):
        self.metadata_path = Path(metadata_path)
        self.journal_path = self.metadata_path.with_suffix(".jsonl")
        self.lock_path = self.metadata_path.with_suffix(self.metadata_path.suffix + ".lock")
        self.metadata_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock_f = open(self.lock_path, "a")
        with self._locked():
            if not self.journal_path.exists() and self.metadata_path.exists():
                self._migrate_from_json()
            self._terminate_torn_line()
        self._journal_f = open(self.journal_path, "a", encoding="utf-8")
        self.refresh()

    @contextmanager
    def _locked(self):
        fcntl.flock(self._lock_f, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(self._lock_f, fcntl.LOCK_UN)

    def _migrate_from_json(self):
        with open(self.metadata_path) as f:
            entries = json.load(f)
        tmp_path = self.journal_path.with_suffix(".jsonl.tmp")
        with open(tmp_path, "w", encoding="utf-8") as f:
            for e in entries:
                f.write(json.dumps(e, ensure_ascii=False) + "\n")
        os.replace(tmp_path, self.journal_path)

    def _terminate_torn_line(self):
        """If a crash left a partial last line, end it so the next append starts clean."""
        if self.journal_path.exists() and self.journal_path.stat().st_size > 0:
            with open(self.journal_path, "rb+") as f:
                f.seek(-1, os.SEEK_END)
                if f.read(1) != b"\n":
                    f.write(b"\n")

    def _read_journal(self):
        """Read every entry; later duplicates of an image_id replace earlier ones."""
        by_id = {}
        if self.journal_path.exists():
            with open(self.journal_path, encoding="utf-8") as f:
                for line in f:
                    try:
                        e = json.loads(line)
                    except json.JSONDecodeError:
                        continue  # torn final line from a crash mid-write
                    by_id[e["image_id"]] = e
        return list(by_id.values())

    def refresh(self):
        """Reload entries from disk (picks up other processes' writes)."""
        self.entries = self._read_journal()
        return self.entries

    def _reopen_if_replaced(self):
        """
        update() swaps in a rewritten journal (temp file + rename). A handle
        opened before that points at the old, unlinked file, so appends
        through it would be lost: reopen when the path's inode has changed.
        Call with the lock held.
        """
        try:
            replaced = os.stat(self.journal_path).st_ino != os.fstat(self._journal_f.fileno()).st_ino
        except FileNotFoundError:
            replaced = True
        if replaced:
            self._journal_f.close()
            self._journal_f = open(self.journal_path, "a", encoding="utf-8")

    def add(self, entry):
        line = json.dumps(entry, ensure_ascii=False) + "\n"
        with self._locked():
            self._reopen_if_replaced()
            self._journal_f.write(line)
            self._journal_f.flush()
        self.entries.append(entry)

    def update(self, fn):
        """
        Apply fn(entry) -> bool (True if it changed the entry) to every
        entry, under the lock, on a fresh read of the journal — so concurrent
        writers (e.g. several CSS proposers filling in their own responses)
        never overwrite each other's changes. The journal is rewritten
        compacted (one line per image_id) only if something changed.
        Returns the number of changed entries.
        """
        with self._locked():
            self.refresh()
            changed = sum(bool(fn(e)) for e in self.entries)
            if changed:
                tmp_path = self.journal_path.with_suffix(".jsonl.tmp")
                with open(tmp_path, "w", encoding="utf-8") as f:
                    for e in self.entries:
                        f.write(json.dumps(e, ensure_ascii=False) + "\n")
                os.replace(tmp_path, self.journal_path)
                self._reopen_if_replaced()
        return changed

    def export_json(self):
        """Rebuild manifest.json (a plain JSON array) from the journal, atomically."""
        with self._locked():
            self.refresh()
            tmp_path = self.metadata_path.with_suffix(self.metadata_path.suffix + ".tmp")
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(self.entries, f, indent=2, ensure_ascii=False)
            os.replace(tmp_path, self.metadata_path)
        return len(self.entries)

    def close(self):
        self._journal_f.close()
        self._lock_f.close()

    def __len__(self):
        return len(self.entries)


def existing_ids(manifest, source, output_dir=None):
    """
    image_ids already recorded for `source`, used to skip already-saved rows
    on resume. With `output_dir`, only ids whose file still exists there
    count, so deleted images are saved again (the new manifest line then
    replaces the stale one, since the journal keeps the last entry per id).
    """
    ids = set()
    for e in manifest.entries:
        if e.get("source_dataset") != source:
            continue
        if output_dir is not None and not (Path(output_dir) / e["filename"]).exists():
            continue
        ids.add(e["image_id"])
    return ids
