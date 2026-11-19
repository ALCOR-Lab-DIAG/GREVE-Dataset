"""Split GREVE into train / val / test (25 / 15 / 60) and move the files to GREVE/<split>/CSS-<x>/<real|fake>.

Decided (2026-10-20):
- 60% test, 25% train, 15% val, for every level including CSS-4;
- real = fake inside every split and level (e.g. test CSS-1: 7,200 real + 7,200 fake);
- random draw, stratified by source (source_dataset; generator_model for our cssgen images),
  generated fakes and their source reals are NOT forced into the same split;
- files are MOVED; greve_path is updated, the old path kept in greve_path_presplit.

Usage: python -m greve.curate.split_release            # dry run: plan + counts, nothing touched
       python -m greve.curate.split_release --apply    # write manifest backup, move files, update manifest
"""
import json, os, random, sys
from collections import Counter, defaultdict
from pathlib import Path
from greve.paths import RELEASE_DIR

ROOT = Path(RELEASE_DIR)
MAN = ROOT / "greve_manifest.json"
SEED = 20261020
FRAC = {"test": 12, "train": 5, "val": 3}  # out of 20 = 60 / 25 / 15
APPLY = "--apply" in sys.argv


def pattern():
    """20 split labels spread evenly with the exact 12/5/3 proportions."""
    out, done = [], Counter()
    for k in range(20):
        s = max(FRAC, key=lambda s: FRAC[s] * (k + 1) / 20 - done[s])
        out.append(s); done[s] += 1
    return out


def stratum(e):
    return "gen:" + e["generator_model"] if e["source_dataset"] == "cssgen" else e["source_dataset"]


T = json.load(open(MAN))
assert not any("greve_split" in e for e in T), "manifest already split"
rng = random.Random(SEED)
PAT = pattern()
cells = defaultdict(list)
for e in T:
    cells[(int(e["css_score"]), e["label"])].append(e)

assign = {}
for key in sorted(cells):
    by = defaultdict(list)
    for e in cells[key]:
        by[stratum(e)].append(e)
    order = sorted(by); rng.shuffle(order)  # random stratum order, each stratum shuffled, then laid end to end
    seq = []
    for s in order:
        g = sorted(by[s], key=lambda e: e["image_id"]); rng.shuffle(g); seq += g
    assert len(seq) % 20 == 0
    for i, e in enumerate(seq):
        assign[e["image_id"]] = PAT[i % 20]

# checks: exact counts and source shares
counts = Counter((assign[e["image_id"]], int(e["css_score"]), e["label"]) for e in T)
for sp in ("test", "train", "val"):
    for lv in (1, 2, 3, 4):
        r, f = counts[(sp, lv, "real")], counts[(sp, lv, "fake")]
        n = (12000 if lv < 4 else 3000) * FRAC[sp] // 20
        assert r == f == n, (sp, lv, r, f, n)
        print(f"{sp:5s} CSS-{lv}: {r} real + {f} fake")
worst = 0
for key, ents in cells.items():
    tot = Counter(stratum(e) for e in ents)
    for sp in FRAC:
        got = Counter(stratum(e) for e in ents if assign[e["image_id"]] == sp)
        for s, n in tot.items():
            worst = max(worst, abs(got[s] - n * FRAC[sp] / 20))
print("largest deviation of a source from its exact share in any split/cell:", round(worst, 1), "images")

moves = []
for e in T:
    old = e["greve_path"]; lv, lab, name = old.split("/")
    assert lv == f"CSS-{e['css_score']}" and lab == e["label"]
    moves.append((old, f"{assign[e['image_id']]}/{old}"))
print(len(moves), "files to move, e.g.", moves[0])
if not APPLY:
    sys.exit("dry run: nothing changed")

backup = ROOT / "greve_manifest.pre_split.json"
if not backup.exists():
    os.link(MAN, backup)  # hard link: the backup keeps the current manifest bytes after we replace MAN
for old, new in moves:
    src, dst = ROOT / old, ROOT / new
    if dst.exists() and not src.exists():
        continue  # already moved by an earlier interrupted run
    dst.parent.mkdir(parents=True, exist_ok=True)
    os.rename(src, dst)
for e in T:
    e["greve_path_presplit"] = e["greve_path"]
    e["greve_split"] = assign[e["image_id"]]
    e["greve_path"] = f"{e['greve_split']}/{e['greve_path_presplit']}"
tmp = ROOT / "greve_manifest.json.tmp"
json.dump(T, open(tmp, "w"))
os.replace(tmp, MAN)
for lv in (1, 2, 3, 4):  # remove the old folders only if empty
    for lab in ("real", "fake"):
        d = ROOT / f"CSS-{lv}" / lab
        if d.exists() and not any(d.iterdir()):
            d.rmdir()
    d = ROOT / f"CSS-{lv}"
    if d.exists() and not any(d.iterdir()):
        d.rmdir()
print("applied")
