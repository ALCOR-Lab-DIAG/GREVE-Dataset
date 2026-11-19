"""
Phase 1 — Build the OpenFake candidate pool for the GREVE sampling plan
into its own manifest (manifest_openfake.json).

Source: ComplexDataLab/OpenFake (v2), config "core" — splits validation
(~58K rows), test (~91K) and train (~2.3M, ~3.3TB). License: CC-BY-NC-4.0.

Plan targets (sampling plan, "OpenFake" row; the select defaults
overshoot each by ~30%: 13K / 13K / 20K / 20K):
  - Real: 20K LAION photos = 10K random + 10K prompt-selected (Pexels skipped;
    test-split reals are DOCCI/ImageNet, so reals come from validation/train)
  - Fake: 30K = 15K spread across generators (<= 300 per generator)
          + 15K paired with the selected LAION reals (same prompt/caption)
  - Prefer the validation/test splits; photos only (anime/illustration
    generators excluded by --exclude_model_regex, posters/covers/drawings/
    screenshots... by --exclude_prompt_regex on the prompt/caption)

The parquet files store each row group (~70-150 shuffled rows, ~100MB)
together, so fetching even one row downloads its whole row group. The work
is therefore split into three steps:

  index   Read only the metadata columns of every row (no image bytes) and
          cache them, with each row's file/row group, in --index_path.
  select  Apply the filters and quotas on the cached metadata and write the
          chosen rows to --selection_path. Prints the per-generator spread,
          the pairing yield and how many GB the fetch will download. Rows are
          taken split by split (validation, test, train) and, within a split,
          row group by row group in a random row-group order, so the draw is
          still random but touches as few row groups as possible.
          Re-run it freely with other quotas — it doesn't download anything.
  fetch   Download only the row groups that hold selected rows, save those
          images' original bytes (no re-encoding) to --output_dir and add them to
          the manifest.
          Resumable: already-saved image_ids are skipped, and so are row
          groups whose selected rows are all saved.

  python -m greve.collect.extract_openfake index
  python -m greve.collect.extract_openfake select
  python -m greve.collect.extract_openfake fetch

Of the GREVE fields make_entry() defines, OpenFake fills source_image_id
(row_id), paired_with, generator_type, generator_release and the orig_*
fields; OpenFake-only fields go in `extra`: image_sha256, prompt,
prompt_kind, selection_role, real_source. orig_jpeg_q is estimated from the
JPEG quantization tables (null for non-JPEG originals). Images are saved as
their original bytes, as <image_id>.png / .jpg / ...
"""

import argparse
import heapq
import os
import random
import re
import shutil
import sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
from huggingface_hub import HfFileSystem
from tqdm import tqdm

from greve.common import ManifestWriter, existing_ids, make_entry, sanitize_key, save_original
from greve.paths import IMAGES_DIR

HF_REPO = "ComplexDataLab/OpenFake"
HF_CONFIG = "core"
SPLITS = ["validation", "train"]  # preference order
META_COLUMNS = ["label", "model", "type", "release_date", "row_id", "image_sha256", "prompt", "prompt_kind"]

# Non-photographic generators (anime/illustration fine-tunes, a stylized
# video LoRA) plus tiny-random-sana, a random-weights test model. Checked
# against the generator list of the v2 validation/test splits; cyberrealistic-
# pony-* are photorealistic Pony fine-tunes and are kept on purpose.
# Rows whose prompt/caption describes a non-photographic image (posters,
# covers, illustrations, screenshots, memes...), for either label.
DEFAULT_EXCLUDE_PROMPT_REGEX = (r"\b(?:poster|movie poster|magazine cover|book cover|album cover|cover art|"
                                r"illustration|illustrated|cartoon|comic|anime|manga|painting|drawing|sketch|"
                                r"watercolor|logo|infographic|screenshot|meme|diagram|chart|map of|"
                                r"3d render|render of|digital art|vector|clipart|clip art|icon)\b")

DEFAULT_EXCLUDE_MODEL_REGEX = (r"illustrious|animagine|novaanime|anima-preview|one-obsession|"
                               r"studio-ghibli|tiny-random|cartoon|manga|hentai|\banime")

# Used for the 10K prompt-selected reals: captions describing content likely
# to land at CSS 2-4 (public figures/events, conflict, disasters, crises).
DEFAULT_KEYWORDS = [
    "president", "prime minister", "minister", "politician", "senator", "parliament", "congress",
    "government", "election", "campaign", "rally", "press conference", "summit", "court", "judge",
    "protest", "protesters", "demonstration", "demonstrators", "march", "riot", "clash", "police",
    "arrest", "soldier", "soldiers", "military", "army", "troops", "tank", "weapon", "gun", "rifle",
    "war", "battle", "conflict", "attack", "bombing", "explosion", "missile", "airstrike", "terrorist",
    "fire", "wildfire", "blaze", "smoke", "flood", "flooding", "earthquake", "hurricane", "tornado",
    "storm", "disaster", "rubble", "destroyed", "collapsed", "wreckage", "crash", "accident",
    "injured", "wounded", "victim", "victims", "casualties", "blood", "dead", "body", "bodies",
    "hospital", "ambulance", "paramedic", "stretcher", "emergency", "rescue", "firefighter",
    "refugee", "refugees", "migrants", "camp", "funeral", "mourners", "coffin", "memorial",
    "pandemic", "vaccine", "mask",
]


def image_id_for(label, row_id):
    row_id = re.sub(r"^openfake-", "", row_id)  # OpenFake row_ids already carry an "openfake-" prefix
    return f"openfake_{label}_{sanitize_key(row_id)}"


def norm_prompt(p):
    return re.sub(r"\s+", " ", (p or "").strip().lower())


def keyword_regex(spec):
    """--keywords: a path to a file (one keyword per line) or a comma-separated list."""
    if spec is None:
        words = DEFAULT_KEYWORDS
    elif Path(spec).is_file():
        words = Path(spec).read_text().splitlines()
    else:
        words = spec.split(",")
    words = [w.strip() for w in words if w.strip() and not w.strip().startswith("#")]
    return re.compile(r"\b(?:" + "|".join(re.escape(w) for w in words) + r")\b", re.IGNORECASE)


def free_gb(path):
    return shutil.disk_usage(path).free / 1e9


# ──────────────────────────────────────────────────────────────
# INDEX
# ──────────────────────────────────────────────────────────────

def index_file(path):
    fs = HfFileSystem()
    with fs.open(path, block_size=2**20) as fh:
        pf = pq.ParquetFile(fh)
        md = pf.metadata
        frames = []
        for rg in range(md.num_row_groups):
            t = pf.read_row_group(rg, columns=META_COLUMNS).to_pandas()
            t["row_in_rg"] = range(len(t))
            t["row_group"] = rg
            t["rg_bytes"] = md.row_group(rg).total_byte_size
            frames.append(t)
    df = pd.concat(frames, ignore_index=True)
    df["file"] = path
    df["split"] = path.rsplit("/", 1)[-1].split("-", 1)[0]
    return df


def cmd_index(args):
    fs = HfFileSystem()
    files = sorted(f for f in fs.ls(f"datasets/{HF_REPO}/{HF_CONFIG}", detail=False)
                   if f.endswith(".parquet") and f.rsplit("/", 1)[-1].split("-", 1)[0] in args.splits)
    print(f"Indexing metadata of {len(files)} parquet files ({', '.join(args.splits)}) — no image bytes are read")

    index_path = Path(args.index_path)
    parts_dir = index_path.with_suffix(".parts")  # one cached part per file, so an interrupted index resumes
    parts_dir.mkdir(parents=True, exist_ok=True)

    def part_path(f):
        return parts_dir / (f.rsplit("/", 1)[-1] + ".meta.parquet")

    todo = [f for f in files if not part_path(f).exists()]
    with ThreadPoolExecutor(args.workers) as pool:
        futures = {pool.submit(index_file, f): f for f in todo}
        for fut in tqdm(as_completed(futures), total=len(futures), desc="index",
                        initial=len(files) - len(todo)):
            fut.result().to_parquet(part_path(futures[fut]), index=False)

    df = pd.concat([pd.read_parquet(part_path(f)) for f in files], ignore_index=True)
    df.to_parquet(index_path, index=False)
    print(f"Indexed {len(df)} rows -> {index_path}")
    print(df.groupby(["split", "label"]).size().to_string())


# ──────────────────────────────────────────────────────────────
# SELECT
# ──────────────────────────────────────────────────────────────

def lazy_greedy(rgs, score, take, rg_bytes, budget, spent, rng):
    """
    Repeatedly take the row group with the highest score (rows it would still
    contribute) until no row group adds anything or the GB budget is spent.
    Scores only ever drop as needs get filled, so stale heap entries are
    re-scored lazily. Returns the bytes spent.
    """
    heap = [(-score(g), rng.random(), g) for g in rgs]
    heap = [h for h in heap if h[0] < 0]
    heapq.heapify(heap)
    while heap:
        neg, tie, g = heapq.heappop(heap)
        s = score(g)
        if s <= 0:
            continue
        if s < -neg:
            heapq.heappush(heap, (-s, tie, g))
            continue
        if budget is not None and spent + rg_bytes[g] > budget:
            continue
        take(g)
        spent += rg_bytes[g]
    return spent


def cmd_select(args):
    df = pd.read_parquet(args.index_path)
    rng = random.Random(args.seed)

    split_rank = {s: i for i, s in enumerate(args.splits)}
    df = df[df["split"].isin(split_rank)].copy()
    df["rg"] = pd.factorize(pd.Series(list(zip(df["file"], df["row_group"]))))[0]
    rg_bytes = df.groupby("rg")["rg_bytes"].first().to_dict()
    # Draw order inside the cheap splits: split preference, random row-group order, random within.
    rg_rank = list(rg_bytes)
    rng.shuffle(rg_rank)
    rg_rank = {g: i for i, g in enumerate(rg_rank)}
    df["order"] = [(split_rank[s], rg_rank[g], rng.random()) for s, g in zip(df["split"], df["rg"])]
    df = df.sort_values("order").reset_index(drop=True)
    df["image_id"] = [image_id_for(l, r) for l, r in zip(df["label"], df["row_id"])]
    df["nprompt"] = df["prompt"].map(norm_prompt)

    if args.exclude_prompt_regex:
        nonphoto = df["prompt"].fillna("").str.contains(re.compile(args.exclude_prompt_regex, re.IGNORECASE))
        print(f"Excluded {int(nonphoto.sum())} rows whose prompt describes a non-photographic image "
              f"(real: {int((nonphoto & (df['label'] == 'real')).sum())}, "
              f"fake: {int((nonphoto & (df['label'] == 'fake')).sum())})")
        df = df[~nonphoto]

    exclude_re = re.compile(args.exclude_model_regex, re.IGNORECASE) if args.exclude_model_regex else None
    reals = df[(df["label"] == "real") & (df["model"] == "laion")].copy()
    fakes = df[df["label"] == "fake"]
    if exclude_re is not None:
        excluded = fakes[fakes["model"].fillna("").str.contains(exclude_re)]
        fakes = fakes.drop(excluded.index)
        print(f"Excluded {len(excluded)} non-photographic fakes from "
              f"{excluded['model'].nunique()} generators: {sorted(excluded['model'].unique())}")
    if args.exclude_types:
        fakes = fakes[~fakes["type"].isin(args.exclude_types)]
    reals["kw"] = reals["prompt"].fillna("").str.contains(keyword_regex(args.keywords))

    costly = set(args.costly_splits)
    budget = args.max_download_gb * 1e9 if args.max_download_gb is not None else None
    role = {}  # row index -> selection_role

    # ── Phase 1: cheap splits (taken in preference order, as before) ──
    cheap_reals = reals[~reals["split"].isin(costly)]
    cheap_fakes = fakes[~fakes["split"].isin(costly)]
    for i in cheap_reals[cheap_reals["kw"]].index[:args.n_real_prompt]:
        role[i] = "real_prompt_selected"
    for i in [i for i in cheap_reals.index if i not in role][:args.n_real_random]:
        role[i] = "real_random"
    taken = Counter()
    by_gen = {g: list(rows.index) for g, rows in cheap_fakes.groupby("model", sort=False)}
    n_spread = 0
    progressed = True
    while n_spread < args.n_fake_spread and progressed:
        progressed = False
        for g, idxs in by_gen.items():
            if n_spread >= args.n_fake_spread:
                break
            if taken[g] < min(args.per_generator_cap, len(idxs)):
                role[idxs[taken[g]]] = "fake_spread"
                taken[g] += 1
                n_spread += 1
                progressed = True
    chosen = {df.at[i, "rg"] for i in role}
    spent = sum(rg_bytes[g] for g in chosen)

    # ── Phase 2: fill the remaining real/spread quotas from the costly splits, fewest row groups ──
    need = {"kw": args.n_real_prompt - sum(r == "real_prompt_selected" for r in role.values()),
            "rand": args.n_real_random - sum(r == "real_random" for r in role.values()),
            "spread": args.n_fake_spread - n_spread}
    need_gen = defaultdict(lambda: args.per_generator_cap, {g: args.per_generator_cap - c for g, c in taken.items()})
    cand = defaultdict(lambda: {"kw": [], "oth": [], "fk": defaultdict(list)})
    costly_reals = reals[reals["split"].isin(costly)]
    costly_fakes = fakes[fakes["split"].isin(costly)]
    for i, g, k in zip(costly_reals.index, costly_reals["rg"], costly_reals["kw"]):
        cand[g]["kw" if k else "oth"].append(i)
    for i, g, m in zip(costly_fakes.index, costly_fakes["rg"], costly_fakes["model"]):
        cand[g]["fk"][m].append(i)

    def score2(g):
        c = cand[g]
        kw = min(len(c["kw"]), need["kw"])
        rand = min(len(c["kw"]) - kw + len(c["oth"]), need["rand"])
        spread = min(sum(min(len(v), need_gen[m]) for m, v in c["fk"].items()), need["spread"])
        return kw + rand + spread

    def take2(g):
        c = cand[g]
        for i in c["kw"][:need["kw"]]:
            role[i] = "real_prompt_selected"
        need["kw"] -= min(len(c["kw"]), need["kw"])
        rest = [i for i in c["kw"] + c["oth"] if i not in role][:need["rand"]]
        for i in rest:
            role[i] = "real_random"
        need["rand"] -= len(rest)
        for m, v in sorted(c["fk"].items(), key=lambda kv: -need_gen[kv[0]]):
            k = min(len(v), need_gen[m], need["spread"])
            for i in v[:k]:
                role[i] = "fake_spread"
            need_gen[m] -= k
            need["spread"] -= k
        chosen.add(g)

    spent = lazy_greedy([g for g in cand if g not in chosen], score2, take2, rg_bytes, budget, spent, rng)

    # ── Phase 3: pairs (fake generated from a LAION real's caption) ──
    # Free pairs first (both rows in row groups already chosen), then add the row
    # groups that complete the most pairs, within the budget.
    avail_fakes = fakes[~fakes.index.isin(role.keys()) & (fakes["nprompt"] != "")]
    pr = reals[reals["nprompt"] != ""][["nprompt", "rg"]].reset_index().rename(columns={"index": "ri", "rg": "rrg"})
    pf = avail_fakes[["nprompt", "rg", "model"]].reset_index().rename(columns={"index": "fi", "rg": "frg"})
    pairs = pr.merge(pf, on="nprompt")
    pairs_by_rg = defaultdict(list)
    for k, (a, b) in enumerate(zip(pairs["rrg"], pairs["frg"])):
        pairs_by_rg[a].append(k)
        if b != a:
            pairs_by_rg[b].append(k)
    P_ri, P_fi = pairs["ri"].to_numpy(), pairs["fi"].to_numpy()
    P_rrg, P_frg, P_model = pairs["rrg"].to_numpy(), pairs["frg"].to_numpy(), pairs["model"].to_numpy()
    real_paired, fake_paired, pair_gen = set(), set(), Counter()
    paired_with = {}
    n_pairs = [0]

    def open_pairs(ks):
        """Pairs among `ks` whose both row groups are chosen and both rows unused, grouped by real."""
        by_real = defaultdict(list)
        for k in ks:
            if P_rrg[k] in chosen and P_frg[k] in chosen and P_ri[k] not in real_paired and P_fi[k] not in fake_paired:
                by_real[P_ri[k]].append(k)
        return by_real

    def pair_up(ks):
        # Reals already in the pool first, so pairing adds as few extra reals as possible.
        by_real = open_pairs(ks)
        for ri in sorted(by_real, key=lambda r: r not in role):
            if n_pairs[0] >= args.n_fake_paired:
                return
            ks_r = [k for k in by_real[ri] if P_fi[k] not in fake_paired]
            if not ks_r:
                continue
            k = min(ks_r, key=lambda k: pair_gen[P_model[k]])
            real_paired.add(ri)
            fake_paired.add(P_fi[k])
            pair_gen[P_model[k]] += 1
            role[P_fi[k]] = "fake_paired"
            role.setdefault(ri, "real_paired")
            paired_with[P_fi[k]] = df.at[ri, "image_id"]
            paired_with[ri] = df.at[P_fi[k], "image_id"]
            n_pairs[0] += 1

    pair_up(range(len(pairs)))

    def score3(g):
        if n_pairs[0] >= args.n_fake_paired:
            return 0
        reals_done = set()
        for k in pairs_by_rg[g]:
            other = P_frg[k] if P_rrg[k] == g else P_rrg[k]
            if (other in chosen or other == g) and P_ri[k] not in real_paired and P_fi[k] not in fake_paired:
                reals_done.add(P_ri[k])
        return min(len(reals_done), args.n_fake_paired - n_pairs[0])

    def take3(g):
        chosen.add(g)
        pair_up(pairs_by_rg[g])

    spent = lazy_greedy([g for g in pairs_by_rg if g not in chosen], score3, take3, rg_bytes, budget, spent, rng)

    sel = df.loc[list(role)].copy()
    sel["selection_role"] = [role[i] for i in sel.index]
    sel["paired_with"] = [paired_with.get(i) for i in sel.index]
    sel = sel.drop(columns=["order", "nprompt", "kw", "rg"], errors="ignore")
    sel.to_parquet(args.selection_path, index=False)

    # Report
    print(f"\nSelected {len(sel)} rows -> {args.selection_path}")
    print(sel.groupby(["selection_role", "split"]).size().unstack(fill_value=0).to_string())
    print(f"\nCandidate pairs (LAION real + same-prompt fake): {pairs['ri'].nunique()} reals / "
          f"{pairs['fi'].nunique()} fakes; paired: {n_pairs[0]}")
    gens = sel[sel["label"] == "fake"]["model"].value_counts()
    print(f"Fake generators: {len(gens)}; max per generator: {gens.max()}")
    targets = {"real_prompt_selected": args.n_real_prompt, "real_random": args.n_real_random,
               "fake_spread": args.n_fake_spread, "fake_paired": args.n_fake_paired}
    got = sel["selection_role"].value_counts()
    for r, want in targets.items():
        if got.get(r, 0) < want:
            print(f"  [short] {r}: {got.get(r, 0)}/{want}")
    rgs = sel.drop_duplicates(["file", "row_group"])
    print(f"\nFetch will download {len(rgs)} row groups ≈ {rgs['rg_bytes'].sum() / 1e9:.0f} GB "
          f"(by split: {(rgs.groupby('split')['rg_bytes'].sum() / 1e9).round(0).to_dict()})"
          + (f" | budget {args.max_download_gb:.0f} GB" if budget is not None else ""))


# ──────────────────────────────────────────────────────────────
# FETCH
# ──────────────────────────────────────────────────────────────

def fetch_row_group(path, rg, wanted, output_dir):
    """Download one row group, save the wanted rows' original bytes.
    Returns [(row_id, (filename, info) | None, error | None)]."""
    fs = HfFileSystem()
    with fs.open(path, block_size=2**24) as fh:
        t = pq.ParquetFile(fh).read_row_group(rg, columns=["image", "row_id"]).to_pylist()
    results = []
    for row in t:
        rid = row["row_id"]
        if rid not in wanted:
            continue
        try:
            results.append((rid, save_original(row["image"]["bytes"], output_dir, wanted[rid]), None))
        except Exception as e:
            results.append((rid, None, f"{type(e).__name__}: {e}"))
    return results


def cmd_fetch(args):
    sel = pd.read_parquet(args.selection_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    Path(args.metadata_path).parent.mkdir(parents=True, exist_ok=True)

    manifest = ManifestWriter(args.metadata_path)
    print(f"Loaded {len(manifest)} existing entries from manifest | free space: {free_gb(output_dir):.0f} GB")
    seen = existing_ids(manifest, "openfake", output_dir)
    todo = sel[~sel["image_id"].isin(seen)]
    # Missing values come out of pandas as NaN; turn them into None so the
    # manifest gets JSON null (NaN isn't valid JSON, and it's truthy in Python).
    todo = todo.astype(object).where(todo.notna(), None)
    rows_by_id = {r.row_id: r for r in todo.itertuples(index=False)}
    jobs = [(f, rg, {r: rows_by_id[r].image_id for r in g["row_id"]})
            for (f, rg), g in todo.groupby(["file", "row_group"])]
    print(f"{len(todo)} of {len(sel)} selected images left to fetch, in {len(jobs)} row groups")

    counts = Counter(e["label"] for e in manifest.entries if e.get("source_dataset") == "openfake")
    saved, failed = 0, 0
    progress = tqdm(total=len(todo), desc="OpenFake fetch")
    with ThreadPoolExecutor(args.workers) as pool:
        futures = [pool.submit(fetch_row_group, f, rg, wanted, output_dir) for f, rg, wanted in jobs]
        try:
            for fut in as_completed(futures):
                for rid, saved_file, err in fut.result():
                    r = rows_by_id[rid]
                    if err:
                        failed += 1
                        tqdm.write(f"  [skip] {rid} -> {err}")
                        continue
                    is_real = r.label == "real"
                    filename, info = saved_file
                    entry = make_entry(
                        image_id=r.image_id,
                        filename=filename,
                        source="openfake",
                        label=r.label,
                        sensitivity="MEDIUM-HIGH",
                        scenario="news_politics",
                        split=r.split,
                        generator=None if is_real else r.model,
                        generator_type=r.type,
                        generator_release=r.release_date,
                        original_label=r.label,
                        source_image_id=r.row_id,
                        paired_with=r.paired_with,
                        hf_repo=HF_REPO,
                        seed=args.seed,
                        info=info,
                        extra={
                            "image_sha256": r.image_sha256,
                            "prompt": r.prompt,
                            "prompt_kind": r.prompt_kind,
                            "selection_role": r.selection_role,
                            "real_source": r.model if is_real else None,
                        },
                    )
                    manifest.add(entry)
                    counts[r.label] += 1
                    saved += 1
                progress.update(len(fut.result()))
                progress.set_description(f"OpenFake fetch | real={counts['real']} fake={counts['fake']} "
                                         f"saved={saved} failed={failed}")
                if free_gb(output_dir) < args.min_free_gb:
                    tqdm.write(f"  Free space below --min_free_gb={args.min_free_gb}, stopping")
                    for f in futures:
                        f.cancel()
                    break
        finally:
            progress.close()

    total = manifest.export_json()
    manifest.close()
    print(f"\nDONE — saved {saved} new images ({failed} failed), "
          f"{total} total entries in manifest ({args.metadata_path})")


# ──────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Phase 1 — Build the OpenFake candidate pool (GREVE sampling plan)")
    parser.add_argument("--index_path", type=str, default="./metadata/openfake_index.parquet")
    parser.add_argument("--selection_path", type=str, default="./metadata/openfake_selection.parquet")
    parser.add_argument("--splits", type=str, nargs="+", default=SPLITS,
                        help="Splits to use, in preference order (default: validation test train)")
    parser.add_argument("--seed", type=int, default=9999)
    parser.add_argument("--workers", type=int, default=8, help="Parallel downloads (index and fetch)")
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("index", help="Cache the metadata columns of every row (no images)")

    p = sub.add_parser("select", help="Apply filters and quotas to the cached metadata")
    # Defaults overshoot the plan's targets (10K / 10K / 15K / 15K) by ~30%,
    # because some selected images won't land at the CSS level they're needed for.
    p.add_argument("--n_real_random", type=int, default=13_000, help="Plan: 10K")
    p.add_argument("--n_real_prompt", type=int, default=13_000, help="Plan: 10K")
    p.add_argument("--n_fake_spread", type=int, default=20_000, help="Plan: 15K")
    p.add_argument("--n_fake_paired", type=int, default=20_000, help="Plan: 15K")
    p.add_argument("--per_generator_cap", type=int, default=300)
    p.add_argument("--keywords", type=str, default=None,
                   help="Keywords for the prompt-selected reals: comma-separated or a file, one per line "
                        "(default: built-in CSS 2-4 list)")
    p.add_argument("--exclude_model_regex", type=str, default=DEFAULT_EXCLUDE_MODEL_REGEX,
                   help="Fakes whose generator name matches are dropped (non-photographic); '' to keep all")
    p.add_argument("--exclude_prompt_regex", type=str, default=DEFAULT_EXCLUDE_PROMPT_REGEX,
                   help="Rows whose prompt matches are dropped (posters, covers, illustrations...); '' to keep all")
    p.add_argument("--costly_splits", type=str, nargs="*", default=["train"],
                   help="Splits filled with as few row groups as possible (the rest are taken in order)")
    p.add_argument("--max_download_gb", type=float, default=600,
                   help="Stop adding row groups once the fetch would download this many GB")
    p.add_argument("--exclude_types", type=str, nargs="*", default=[],
                   help="Generator types to drop, e.g. video lora")

    p = sub.add_parser("fetch", help="Download the selected images and write the manifest")
    p.add_argument("--output_dir", type=str, default=IMAGES_DIR)
    p.add_argument("--metadata_path", type=str, default="./metadata/manifest_openfake.json")
    p.add_argument("--min_free_gb", type=float, default=50.0,
                   help="Stop cleanly when free space in --output_dir drops below this many GB")

    args = parser.parse_args()
    {"index": cmd_index, "select": cmd_select, "fetch": cmd_fetch}[args.cmd](args)
    sys.exit(0)


if __name__ == "__main__":
    main()
