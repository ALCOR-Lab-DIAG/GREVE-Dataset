"""Toy dataset in the manifest format: python make_toy.py toy/   (fakes are brighter, so `dummy` works)."""
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

out = Path(sys.argv[1] if len(sys.argv) > 1 else "toy")
rng = np.random.default_rng(0)
recs = []
for css, n in [(1, 40), (2, 40), (3, 40), (4, 10)]:
    for label in ("real", "fake"):
        (out / f"CSS-{css}" / label).mkdir(parents=True, exist_ok=True)
        for j in range(n):
            i = f"toy_{label}_{css}_{j}"
            fake = label == "fake"
            base = 100 + 40 * fake - 25 * fake * (css == 4) + rng.normal(0, 22)  # harder on CSS-4
            path = f"CSS-{css}/{label}/{i}.jpg"
            Image.fromarray(rng.normal(base, 40, (48, 48, 3)).clip(0, 255).astype("uint8")).save(out / path)
            recs.append({"image_id": i, "label": label, "css_score": css, "triage_path": path,
                         "source_dataset": f"src{j % 3}", "generator_model": f"gen{j % 2}" if fake else None,
                         "paired_with": f"toy_real_{css}_{j}" if fake else None, "condition": "original"})
(out / "manifest.json").write_text(json.dumps(recs, indent=1))
print(out / "manifest.json", len(recs))
