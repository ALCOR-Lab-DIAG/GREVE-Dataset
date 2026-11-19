"""
Where the pipeline reads and writes data. Every script is run from the repository root
(`python -m greve.<stage>.<script>`), so relative paths below resolve against it.

Override any of them with an environment variable:

  GREVE_IMAGES_DIR   all collected and generated images, one flat folder, original filenames
                     (default: data/images)
  GREVE_RAW_DIR      raw source archives / downloads (MEDIC tarball, RRDataset test set,
                     VisualNews origin.tar, ...)            (default: data/raw)
  GREVE_RELEASE_DIR  the released dataset (JPEG, CSS-x/{real,fake} folders + manifest)
                     (default: data/images/GREVE)
  GREVE_CACHE_DIR    small caches (CLIP embeddings / probe)  (default: data/cache)

Metadata (manifests, VLM answer journals, dedup and selection files) always lives in
./metadata/ under the repository root.
"""

import os
from pathlib import Path

IMAGES_DIR = os.environ.get("GREVE_IMAGES_DIR", "data/images")
RAW_DIR = os.environ.get("GREVE_RAW_DIR", "data/raw")
RELEASE_DIR = os.environ.get("GREVE_RELEASE_DIR", str(Path(IMAGES_DIR) / "GREVE"))
CACHE_DIR = os.environ.get("GREVE_CACHE_DIR", "data/cache")
METADATA_DIR = Path("metadata")
