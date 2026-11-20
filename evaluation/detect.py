"""Run one detector over the dataset and write <out>/<detector>.csv with id, score, pred.

    python detect.py --detector npr --manifest manifest.json --root /data/css --split 0.7 --out scores/
    python detect.py --detector npr ... --shard 3/8      # images 3, 11, 19, ... (one GPU each)
    python detect.py --list

The first run clones the official repo and downloads the released weights into $CSS_CACHE.
Resumes: images already in the output file are skipped. An image that fails is written with an
empty score and pred (score.py counts it as an error) and the error is printed.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from PIL import Image

import css_data as data
import detectors  # noqa: F401  (fills the registry)
from base import REGISTRY


def run(name, manifest, out_dir, root=None, split=1.0, device="cuda", shard="0/1", subset=None, batch_size=None,
        seed=0, retry_failed=False, image_dir=None, crash_safe=False):
    import random
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    det = REGISTRY[name]()
    det.seed = seed
    if batch_size:
        det.batch_size = batch_size
    rows = data.load(manifest, root, split, subset=subset, image_dir=image_dir)
    i, n = map(int, shard.split("/"))
    rows = rows[i::n]
    suffix = "" if n == 1 else f".shard{i}of{n}"
    out = Path(out_dir).resolve() / f"{name}{suffix}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        with open(out, newline="", encoding="utf-8") as f:
            old = list(csv.DictReader(f))
        if retry_failed:  # drop the failed images (empty score) so they are run again
            old = [r for r in old if r["score"] != ""]
            with open(out, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["id", "score", "pred"])
                w.writerows([[r["id"], r["score"], r["pred"]] for r in old])
        done = {r["id"] for r in old}
    # crash_safe: an image that kills the process (segfault in native code) is noted in <out>.running; the next
    # run moves it to <out>.crashed and skips it, so it stays missing (an error for score.py) and the rest goes on
    running, crashed = out.with_suffix(".running"), out.with_suffix(".crashed")
    if crash_safe and running.exists():
        with open(crashed, "a", encoding="utf-8") as f:
            f.write(running.read_text(encoding="utf-8"))
        running.unlink()
    skip = set(crashed.read_text(encoding="utf-8").split()) if crash_safe and crashed.exists() else set()
    todo = [r for r in rows if r["id"] not in done and r["id"] not in skip]
    det.fetch()
    det.load(device)
    new = not out.exists()
    with open(out, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["id", "score", "pred"])
        for b in range(0, len(todo), det.batch_size):
            batch = todo[b:b + det.batch_size]
            if crash_safe:
                running.write_text("".join(r["id"] + "\n" for r in batch), encoding="utf-8")
            try:
                x = [det.preprocess(Image.open(r["path"]).convert("RGB")) for r in batch]
                scores = [float(s) for s in det.forward(x)]
                for r, s in zip(batch, scores):
                    w.writerow([r["id"], s, int(s >= det.threshold)])
            except Exception as e:  # noqa: BLE001  one bad image must not stop the run
                print(f"{name}: failed on {[r['id'] for r in batch]}: {type(e).__name__}: {e}")
                for r in batch:
                    w.writerow([r["id"], "", ""])
            f.flush()
            if crash_safe:
                running.unlink()
            if (b // det.batch_size) % 200 == 0:
                print(f"{name}: {min(b + det.batch_size, len(todo))}/{len(todo)}", flush=True)
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--detector")
    p.add_argument("--manifest")
    p.add_argument("--root", default=None, help="dataset folder (triage_path is relative to it); default: the manifest's folder")
    p.add_argument("--split", type=float, default=1.0, help="fraction of the dataset to use, e.g. 0.7")
    p.add_argument("--image-dir", default=None, help="flat folder with all the images (overrides --root)")
    p.add_argument("--subset", default=None, help="keep only records with this triage_split, e.g. test")
    p.add_argument("--out", default="scores")
    p.add_argument("--device", default="cuda")
    p.add_argument("--shard", default="0/1", help="i/n: run only every n-th image starting at i")
    p.add_argument("--batch-size", type=int, default=None, help="override the detector's batch size (e.g. 1 after OOM)")
    p.add_argument("--seed", type=int, default=0, help="seed of python, numpy and torch (random test-time steps)")
    p.add_argument("--retry-failed", action="store_true", help="run again the images written with an empty score")
    p.add_argument("--crash-safe", action="store_true", help="skip, on the next run, an image that killed the process")
    p.add_argument("--fetch-only", action="store_true", help="only download code and weights")
    p.add_argument("--list", action="store_true", help="list the registered detectors")
    a = p.parse_args(argv)
    if a.list:
        for n, c in REGISTRY.items():
            print(f"{n:22s} {c.checkpoint}")
        return
    if a.fetch_only:
        for n in ([a.detector] if a.detector else REGISTRY):
            REGISTRY[n]().fetch()
        return
    if not (a.detector and a.manifest):
        p.error("--detector and --manifest are required")
    print(run(a.detector, a.manifest, a.out, a.root, a.split, a.device, a.shard, a.subset, a.batch_size, a.seed,
              a.retry_failed, a.image_dir, a.crash_safe))


if __name__ == "__main__":
    main()
