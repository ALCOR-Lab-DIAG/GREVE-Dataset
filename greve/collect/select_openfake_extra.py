"""
Select extra OpenFake FAKES for the CSS-2 / CSS-3 cells (keyword-targeted),
as a selection file for `extract_openfake.py fetch`.

Candidates: fake rows of the cached index (metadata/openfake_index.parquet)
that are not in manifest_openfake.json yet, pass the same photo-only filters
as the main pool (--exclude_model_regex / --exclude_prompt_regex) and whose
prompt matches the CSS 2-4 keyword list (public figures, police, protests,
military, disasters, accidents...). Taken row group by row group, densest
first (fewest GB downloaded), at most --per_generator_cap per generator.

  python -m greve.collect.select_openfake_extra --n 8000
  python -m greve.collect.extract_openfake --selection_path metadata/openfake_selection_extra.parquet \
      fetch --metadata_path metadata/manifest_openfake_extra.json
"""

import argparse
import json
from collections import Counter

import pandas as pd

from greve.collect import extract_openfake as O


def main():
    parser = argparse.ArgumentParser(description="Select extra keyword-targeted OpenFake fakes")
    parser.add_argument("--n", type=int, default=8000)
    parser.add_argument("--per_generator_cap", type=int, default=400)
    parser.add_argument("--index_path", default="./metadata/openfake_index.parquet")
    parser.add_argument("--used_manifests", nargs="+", default=["./metadata/manifest_openfake.json"])
    parser.add_argument("--out", default="./metadata/openfake_selection_extra.parquet")
    args = parser.parse_args()

    idx = pd.read_parquet(args.index_path)
    used = set()
    for path in args.used_manifests:
        used |= {e["source_image_id"] for e in json.load(open(path))}
    f = idx[(idx["label"] == "fake") & ~idx["row_id"].isin(used)]
    f = f[~f["model"].str.contains(O.DEFAULT_EXCLUDE_MODEL_REGEX, case=False, regex=True)]
    prompts = f["prompt"].fillna("")
    f = f[~prompts.str.contains(O.DEFAULT_EXCLUDE_PROMPT_REGEX, case=False, regex=True)]
    f = f[f["prompt"].fillna("").str.contains(O.keyword_regex(None))]
    print(f"Candidates: {len(f)} unused keyword-matching fakes, {f['model'].nunique()} generators")

    f = f.assign(rg=f["file"] + ":" + f["row_group"].astype(str))
    density = f.groupby("rg").size().sort_values(ascending=False)
    taken, chosen, rgs = Counter(), [], set()
    groups = dict(list(f.groupby("rg")))
    for rg in density.index:
        for i, r in groups[rg].iterrows():
            if len(chosen) >= args.n:
                break
            if taken[r["model"]] < args.per_generator_cap:
                taken[r["model"]] += 1
                chosen.append(i)
                rgs.add(rg)
        if len(chosen) >= args.n:
            break

    sel = f.loc[chosen].drop(columns=["rg"]).copy()
    sel["image_id"] = [O.image_id_for("fake", r) for r in sel["row_id"]]
    sel["selection_role"] = "fake_keyword_extra"
    sel["paired_with"] = None
    sel.to_parquet(args.out)
    gb = idx.assign(rg=idx["file"] + ":" + idx["row_group"].astype(str)).drop_duplicates("rg")
    gb = gb[gb["rg"].isin(rgs)]["rg_bytes"].sum() / 1e9
    print(f"Selected {len(sel)} fakes from {len(rgs)} row groups (~{gb:.0f} GB to download), "
          f"{len(taken)} generators, max {max(taken.values())} per generator -> {args.out}")
    print("  " + ", ".join(f"{g} {c}" for g, c in taken.most_common()))


if __name__ == "__main__":
    main()
