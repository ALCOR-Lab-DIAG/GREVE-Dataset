"""
CLIP pre-ranking for CSS-4 mining: score every image of a pool by how much
more it matches "graphic" prompts (dead bodies, severe injuries, blood) than
"non-graphic" disaster prompts (fire, smoke, damage, wreckage without people),
so the Qwen screen only has to look at the top of the ranking.

  calibrate  on already-scored manifests: recall of the final CSS-4 images
             (label real by default) at each top fraction of the ranking
  probe      train a logistic regression on the CLIP embeddings of the scored
             manifests (CSS-4 vs not), report recall on held-out sources
  select     rank a screen pool (zero-shot prompts, or --probe) and write its top --keep_fraction (or
             --keep_n) to a new manifest for the Qwen screen

Image embeddings are cached per manifest in --cache_dir, so re-running with
other prompts or cutoffs costs no GPU time for the images.

  python -m greve.screen.clip_rank calibrate metadata/manifest_medic.json metadata/manifest_visualnews*.json --gpu 1
  python -m greve.screen.clip_rank probe metadata/manifest_*.json --gpu 1
  python -m greve.screen.clip_rank select metadata/screen/manifest_incidents_screen.json \
      metadata/screen/manifest_incidents_top.json --probe data/cache/css4_probe.npz \
      --keep_fraction 0.3 --gpu 1
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from transformers import CLIPModel, CLIPProcessor

from greve.common import ManifestWriter
from greve.paths import IMAGES_DIR, CACHE_DIR

MODEL = "openai/clip-vit-large-patch14"
GRAPHIC = [
    "a photo of a dead body",
    "a photo of corpses lying on the ground",
    "a photo of a body covered with a sheet at a disaster scene",
    "a photo of a severely injured person covered in blood",
    "a photo of blood and gore",
    "a photo of a mutilated body",
    "a photo of burned bodies",
    "a graphic photo of casualties of an accident",
    "a graphic war photo with dead people",
    "a photo of a wounded person bleeding on a stretcher",
]
NOT_GRAPHIC = [
    "a photo of a building on fire",
    "a photo of thick smoke",
    "a photo of a collapsed building",
    "a photo of a damaged house",
    "a photo of a car crash",
    "a photo of a plane crash wreckage",
    "a photo of a train derailment",
    "a photo of firefighters at work",
    "a photo of rescue workers",
    "a news photo of a crowd of people",
    "a photo of soldiers",
    "a photo of a protest",
    "a photo of a destroyed city",
    "a photo of a landscape",
]


def features(out):
    """transformers >= 5 returns an output object (pooler_output = projected embedding), older a tensor."""
    return out if torch.is_tensor(out) else out.pooler_output


class Images(Dataset):
    def __init__(self, paths, processor):
        self.paths, self.processor = paths, processor

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        try:
            img = Image.open(self.paths[i]).convert("RGB")
            ok = True
        except Exception:
            img, ok = Image.new("RGB", (224, 224)), False
        return self.processor(images=img, return_tensors="pt")["pixel_values"][0], ok


def embed(entries, images_dir, cache_dir, manifest_path, model, processor, device, batch, workers):
    key = hashlib.md5(str(Path(manifest_path).resolve()).encode()).hexdigest()[:10]
    cache = Path(cache_dir) / f"{Path(manifest_path).stem}_{key}.npz"
    ids = [e["image_id"] for e in entries]
    have = {}
    if cache.exists():
        z = np.load(cache, allow_pickle=True)
        have = dict(zip(z["ids"], z["emb"]))
    todo = [e for e in entries if e["image_id"] not in have]
    if todo:
        loader = DataLoader(Images([str(Path(images_dir) / e["filename"]) for e in todo], processor),
                            batch_size=batch, num_workers=workers)
        out, done = [], 0
        with torch.no_grad():
            for pix, ok in loader:
                f = features(model.get_image_features(pixel_values=pix.to(device, torch.float16)))
                f = torch.nn.functional.normalize(f.float(), dim=-1).cpu().numpy()
                f[~ok.numpy()] = np.nan
                out.append(f)
                done += len(pix)
                print(f"  {Path(manifest_path).name}: {done}/{len(todo)} embedded", file=sys.stderr, end="\r")
        print(file=sys.stderr)
        for e, f in zip(todo, np.concatenate(out)):
            have[e["image_id"]] = f
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez(cache, ids=np.array(list(have)), emb=np.stack(list(have.values())))
    return np.stack([have[i] for i in ids])


def text_features(model, processor, device):
    with torch.no_grad():
        t = processor(text=GRAPHIC + NOT_GRAPHIC, return_tensors="pt", padding=True).to(device)
        f = torch.nn.functional.normalize(features(model.get_text_features(**t)).float(), dim=-1).cpu().numpy()
    return f[:len(GRAPHIC)], f[len(GRAPHIC):]


def graphic_score(emb, pos, neg):
    """log-sum-exp similarity to the graphic prompts minus to the others (CLIP logit scale 100)."""
    lse = lambda s: np.log(np.exp(100 * s - (100 * s).max(1, keepdims=True)).sum(1)) + (100 * s).max(1)
    score = lse(emb @ pos.T) - lse(emb @ neg.T)
    return np.where(np.isnan(score), -np.inf, score)


def source_family(path):
    """manifest_visualnews_extra -> visualnews: held out together."""
    return Path(path).stem.replace("manifest_", "").split("_")[0]


def probe(args, model, processor, device):
    """Logistic regression on the CLIP embeddings, CSS-4 vs not, trained on the scored manifests
    (real + fake); recall measured on each --holdout source with that source left out of training."""
    from sklearn.linear_model import LogisticRegression
    X, y, fam = [], [], []
    for path in args.manifests:
        entries = [e for e in json.load(open(path)) if e.get("css_score") is not None]
        X.append(embed(entries, args.images_dir, args.cache_dir, path, model, processor, device, args.batch, args.workers))
        y.append(np.array([e["css_score"] == 4 for e in entries]))
        fam += [source_family(path)] * len(entries)
    X, y, fam = np.concatenate(X), np.concatenate(y), np.array(fam)
    ok = np.isfinite(X).all(1)
    X, y, fam = X[ok], y[ok], fam[ok]
    fit = lambda X, y: LogisticRegression(C=args.C, class_weight="balanced", max_iter=3000).fit(X, y)
    fractions = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5]
    print(f"{'held-out source':20s} {'images':>7s} {'CSS-4':>6s}  recall of CSS-4 in the top " + " ".join(f"{f:>5.0%}" for f in fractions))
    for h in args.holdout:
        test = fam == h
        clf = fit(X[~test], y[~test])
        s = clf.decision_function(X[test])
        order = np.argsort(-s)
        rec = [y[test][order[:max(1, int(f * test.sum()))]].sum() / max(1, y[test].sum()) for f in fractions]
        print(f"{h:20s} {int(test.sum()):7d} {int(y[test].sum()):6d}  {'':27s}" + " ".join(f"{r:5.0%}" for r in rec))
    clf = fit(X, y)
    out = Path(args.cache_dir) / "css4_probe.npz"
    np.savez(out, coef=clf.coef_[0], intercept=clf.intercept_, n=len(y), pos=int(y.sum()))
    print(f"Probe trained on {len(y)} images ({int(y.sum())} CSS-4) -> {out}")


def main():
    parser = argparse.ArgumentParser(description="CLIP pre-ranking for CSS-4 mining")
    parser.add_argument("mode", choices=["calibrate", "probe", "select"])
    parser.add_argument("manifests", nargs="+", help="calibrate: scored manifests; select: <screen manifest> <out manifest>")
    parser.add_argument("--labels", nargs="+", default=["real"])
    parser.add_argument("--holdout", nargs="+", default=["medic", "visualnews"], help="probe: sources to test on")
    parser.add_argument("--C", type=float, default=1.0, help="probe: inverse regularization")
    parser.add_argument("--probe", type=str, default=None, help="select: rank with this trained probe (.npz) "
                        "instead of the zero-shot prompts")
    parser.add_argument("--keep_fraction", type=float, default=None)
    parser.add_argument("--keep_n", type=int, default=None)
    parser.add_argument("--images_dir", type=str, default=IMAGES_DIR)
    parser.add_argument("--cache_dir", type=str, default=CACHE_DIR)
    parser.add_argument("--gpu", type=int, default=1)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()

    device = f"cuda:{args.gpu}"
    model = CLIPModel.from_pretrained(MODEL, torch_dtype=torch.float16).to(device).eval()
    processor = CLIPProcessor.from_pretrained(MODEL)
    pos, neg = text_features(model, processor, device)

    if args.mode == "probe":
        return probe(args, model, processor, device)
    if args.mode == "calibrate":
        scores, is4, names = [], [], []
        for path in args.manifests:
            entries = [e for e in json.load(open(path)) if e["label"] in args.labels and e.get("css_score") is not None]
            if not entries:
                continue
            s = graphic_score(embed(entries, args.images_dir, args.cache_dir, path, model, processor, device,
                                    args.batch, args.workers), pos, neg)
            y = np.array([e["css_score"] == 4 for e in entries])
            scores.append(s)
            is4.append(y)
            names.append((Path(path).stem, s, y))
        names.append(("ALL", np.concatenate(scores), np.concatenate(is4)))
        fractions = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5]
        print(f"{'manifest':34s} {'images':>7s} {'CSS-4':>6s}  recall of CSS-4 in the top " + " ".join(f"{f:>5.0%}" for f in fractions))
        for name, s, y in names:
            order = np.argsort(-s)
            rec = [y[order[:max(1, int(f * len(s)))]].sum() / max(1, y.sum()) for f in fractions]
            print(f"{name:34s} {len(s):7d} {int(y.sum()):6d}  {'':27s}" + " ".join(f"{r:5.0%}" for r in rec))
        return

    src, out = args.manifests
    entries = json.load(open(src))
    emb = embed(entries, args.images_dir, args.cache_dir, src, model, processor, device, args.batch, args.workers)
    if args.probe:
        p = np.load(args.probe)
        s = np.where(np.isfinite(emb).all(1), np.nan_to_num(emb) @ p["coef"] + p["intercept"][0], -np.inf)
    else:
        s = graphic_score(emb, pos, neg)
    order = np.argsort(-s)
    keep = args.keep_n or int(round((args.keep_fraction or 0.3) * len(entries)))
    writer = ManifestWriter(out)
    have = {e["image_id"] for e in writer.entries}
    added = 0
    for rank, i in enumerate(order[:keep]):
        if not np.isfinite(s[i]) or entries[i]["image_id"] in have:
            continue
        e = dict(entries[i], clip_graphic_score=float(s[i]), clip_rank=rank)
        writer.add(e)
        added += 1
    total = writer.export_json()
    writer.close()
    print(f"Kept the top {keep} of {len(entries)} by CLIP graphic score (cutoff {s[order[keep - 1]]:.2f}); "
          f"{added} added, {total} in {out}")


if __name__ == "__main__":
    main()
