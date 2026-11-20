"""Scoring of the CSS benchmark, from the per-image score files written by detect.py.

    python score.py --manifest manifest.json --root /data/css --split 0.7 --scores scores/ --out results/

Inputs
  manifest.json            the dataset manifest (see css_data.py); --split must be the one used by detect.py
  scores/<detector>.csv    id, score, pred (also <detector>.shard<i>of<n>.csv, merged)
                 score = how fake the detector thinks the image is (higher = more fake);
                 pred  = 0/1 decision at the threshold released by the authors.
                 Images missing from the files, or with an empty or non-finite score, count as errors
                 in the accuracy metrics and are left out of AUC (the count is reported).
Outputs (in --out)
  summary.csv    one row per detector, the headline metrics:
                   acc_real_css1..4   reals called real, per CSS level
                   acc_fake_css1..4   fakes called fake, per CSS level
                   acc_pooled         accuracy on all images (= sum_k n_k/N * acc_css_k)
                   overall_acc        plain mean of the accuracies on CSS 1..4
                   auc                ROC AUC on all images
                   caAcc, caAUC       content-aware accuracy and AUC, centroid weights, with 95% CI
  per_css.csv    one row per detector x CSS level (acc, AUC, AP, TPR, FPR, ...)
  predictions.csv  one row per detector x image: image_path, label_gt, label_pred (real/fake), css, score
  by_<col>.csv   one row per detector x source_dataset / generator_model / ... value
  <detector>.json   everything, including the ablation weightings, the band and the CIs
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

import css_data as data

LEVELS = data.LEVELS
K = len(LEVELS)


# ---------------------------------------------------------------------------------- weights
def weights() -> dict[str, np.ndarray]:
    """Weights over CSS-1..4 (increasing sensitivity). 'centroid' is the primary one."""
    k = np.arange(1, K + 1, dtype=float)
    raw = {
        "centroid": np.array([sum(1 / j for j in range(K - kk + 1, K + 1)) / K for kk in range(1, K + 1)]),
        "uniform": np.ones(K),
        "linear": k,
        "exponential": 2.0 ** (k - 1),
        "fibonacci": np.array([1.0, 2.0, 3.0, 5.0]),
        "reciprocal": 1 / (K - k + 1),
    }
    return {n: w / w.sum() for n, w in raw.items()}


# ------------------------------------------------------------------------------- primitives
def auc(y: np.ndarray, s: np.ndarray) -> float | None:
    """ROC AUC = P(score of a fake > score of a real), ties count 1/2."""
    pos, neg = int(y.sum()), int(len(y) - y.sum())
    if pos == 0 or neg == 0:
        return None
    _, inv, cnt = np.unique(s, return_inverse=True, return_counts=True)
    rank = (np.cumsum(cnt) - (cnt - 1) / 2)[inv]
    return float((rank[y == 1].sum() - pos * (pos + 1) / 2) / (pos * neg))


def ap(y: np.ndarray, s: np.ndarray) -> float | None:
    """Average precision (area under the precision-recall curve, step-wise; same as scikit-learn)."""
    pos = int(y.sum())
    if pos == 0 or pos == len(y):
        return None
    order = np.argsort(-s, kind="mergesort")
    s, y = s[order], y[order]
    last = np.r_[np.flatnonzero(np.diff(s)), len(s) - 1]  # last index of each distinct score
    tp = np.cumsum(y)[last]
    fp = (last + 1) - tp
    precision, recall = tp / (tp + fp), tp / pos
    return float(np.sum(np.diff(np.r_[0.0, recall]) * precision))


def rates(y: np.ndarray, correct: np.ndarray) -> dict:
    """Threshold metrics on a group of images. correct = decision at the released threshold is right."""
    n_f, n_r = int(y.sum()), int(len(y) - y.sum())
    tpr = float(correct[y == 1].mean()) if n_f else None  # fakes called fake
    tnr = float(correct[y == 0].mean()) if n_r else None  # reals called real
    return {
        "n": int(len(y)), "n_real": n_r, "n_fake": n_f,
        "acc": float(correct.mean()) if len(y) else None,
        "bacc": (tpr + tnr) / 2 if tpr is not None and tnr is not None else None,
        "tpr": tpr, "fnr": None if tpr is None else 1 - tpr,  # fnr = missed fakes
        "tnr": tnr, "fpr": None if tnr is None else 1 - tnr,  # fpr = reals called fake
    }


def ranking_metrics(y, s) -> dict:
    ok = ~np.isnan(s)
    return {"auc": auc(y[ok], s[ok]), "ap": ap(y[ok], s[ok]), "n_scored": int(ok.sum())}


# ---------------------------------------------------------------------------------- one detector
def headline(y, s, correct, css, W) -> dict:
    """Pooled metrics, per-level metrics and the content-aware ones."""
    out = {"pooled": {**rates(y, correct), **ranking_metrics(y, s)}}
    per = {}
    for lv in LEVELS:
        m = css == lv
        per[lv] = {**rates(y[m], correct[m]), **ranking_metrics(y[m], s[m])}
    out["per_css"] = per
    out["unbalanced_levels"] = [lv for lv in LEVELS if per[lv]["n_real"] != per[lv]["n_fake"]]  # then acc != bacc
    acc_k = [per[lv]["acc"] for lv in LEVELS]
    if any(a is None for a in acc_k):
        out["error"] = "a CSS level has no images: content-aware metrics undefined"
        return out
    m = np.array(acc_k)
    T = np.cumsum(m[::-1]) / np.arange(1, K + 1)  # T_j: mean over the j most sensitive levels
    out["caAcc"] = {n: float(w @ m) for n, w in W.items()}
    out["band"] = [float(T.min()), float(T.max())]
    out["T"] = T.tolist()
    out["worst_css"] = float(m.min())
    out["gap_css4_css1"] = float(m[-1] - m[0])
    out["overall_acc"] = float(m.mean())  # plain mean over the levels (the pooled one is out["pooled"]["acc"])
    auc_k = [per[lv]["auc"] for lv in LEVELS]
    if all(a is not None for a in auc_k):
        out["caAUC"] = {n: float(w @ np.array(auc_k)) for n, w in W.items()}
    return out


def bootstrap(y, s, correct, css, W, n_boot, seed) -> dict:
    """95% intervals, resampling images within each CSS level x label cell (keeps the design fixed)."""
    if n_boot <= 0:
        return {}
    rng = np.random.default_rng(seed)
    cells = [np.flatnonzero((css == lv) & (y == c)) for lv in LEVELS for c in (0, 1)]
    cells = [c for c in cells if len(c)]
    keys = {"acc": [], "auc": [], "overall_acc": [], "caAcc_centroid": [], "caAUC_centroid": [],
            "band_min": [], "band_max": []} | {f"acc_css{lv}": [] for lv in LEVELS}
    for _ in range(n_boot):
        idx = np.concatenate([rng.choice(c, size=len(c), replace=True) for c in cells])
        r = headline(y[idx], s[idx], correct[idx], css[idx], {"centroid": W["centroid"]})
        if "caAcc" not in r:
            continue
        keys["acc"].append(r["pooled"]["acc"])
        keys["overall_acc"].append(r["overall_acc"])
        keys["caAcc_centroid"].append(r["caAcc"]["centroid"])
        keys["band_min"].append(r["band"][0])
        keys["band_max"].append(r["band"][1])
        for lv in LEVELS:
            keys[f"acc_css{lv}"].append(r["per_css"][lv]["acc"])
        if r["pooled"]["auc"] is not None:
            keys["auc"].append(r["pooled"]["auc"])
        if "caAUC" in r:
            keys["caAUC_centroid"].append(r["caAUC"]["centroid"])
    return {k: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] for k, v in keys.items() if len(v) > 1}


def score_detector(rows: list[dict], scores: dict[str, tuple], by: list[str], n_boot: int, seed: int) -> dict:
    y = np.array([int(r["label"]) for r in rows])
    css = np.array([int(r["css"]) for r in rows])
    s = np.array([scores[r["id"]][0] if r["id"] in scores else np.nan for r in rows], dtype=float)
    pred = np.array([scores[r["id"]][1] if r["id"] in scores else -1 for r in rows])
    correct = pred == y  # missing or failed predictions (-1) count as errors
    W = weights()
    res = headline(y, s, correct, css, W)
    res["missing_predictions"] = int((pred < 0).sum())
    res["ci95"] = bootstrap(y, s, correct, css, W, n_boot, seed)
    res["weights"] = {n: w.round(4).tolist() for n, w in W.items()}
    for col in by:
        groups = {}
        for v in sorted({r.get(col, "") for r in rows}):
            m = np.array([r.get(col, "") == v for r in rows])
            groups[v or "(empty)"] = {**rates(y[m], correct[m]), **ranking_metrics(y[m], s[m])}
        res[f"by_{col}"] = groups
    return res


# ------------------------------------------------------------------------------------------ io
def read_scores(folder, det) -> dict[str, tuple]:
    out = {}
    for path in sorted(Path(folder).glob(f"{det}.csv")) + sorted(Path(folder).glob(f"{det}.shard*.csv")):
        with open(path, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                if r.get("score") in ("", None) or r.get("pred") in ("", None):
                    continue  # failed image, or a line cut short (e.g. disk full): counted as an error
                if not np.isfinite(float(r["score"])):
                    continue  # NaN/inf is no score, whatever pred says: an error, out of the AUC, in missing
                out[r["id"]] = (float(r["score"]), int(r["pred"]))
    return out


def _r(x):
    return round(x, 4) if isinstance(x, float) else x


def write_csv(path, rows):
    keys = []
    for r in rows:
        keys += [k for k in r if k not in keys]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows([{k: _r(v) for k, v in r.items()} for r in rows])


def write_table(out, summary, per_css):
    """table.csv / table.md: per CSS level accuracy, real and fake accuracy; then AUC, caAcc, caAUC, overall acc."""
    acc = {(r["detector"], r["css"]): r["acc"] for r in per_css}
    rows = []
    for s in summary:
        row = {"detector": s["detector"]}
        for lv in LEVELS:
            row |= {f"css{lv}_acc": acc[s["detector"], lv], f"css{lv}_real": s[f"acc_real_css{lv}"],
                    f"css{lv}_fake": s[f"acc_fake_css{lv}"]}
        row |= {k: s[k] for k in ("auc", "caAcc", "caAUC", "acc_pooled", "overall_acc", "missing")}
        rows.append(row)
    write_csv(out / "table.csv", rows)
    f = lambda x: "" if x is None else f"{x:.3f}"
    head = "| Detector | " + " | ".join(f"CSS{lv} Acc | CSS{lv} Real | CSS{lv} Fake" for lv in LEVELS) \
        + " | AUC | caAcc | caAUC | Acc (pooled) | Overall Acc |"
    lines = [head, "|" + "---|" * (head.count("|") - 1)]
    for r in rows:
        lines.append(f"| {r['detector']} | " + " | ".join(f(v) for k, v in r.items() if k not in ("detector", "missing"))
                     + " |")
    (out / "table.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_latex(out, summary, acc)


NAMES = {"aide": "AIDE", "bfree": "B-Free", "c2p_clip": "C2P-CLIP", "clipdet": "CLIPDet", "cnnspot": "CNNSpot",
         "community_forensics": "CommFor", "dire": "DIRE", "drct_convb": "DRCT", "effort": "Effort",
         "fatformer": "FatFormer", "fredect": "FreDect", "fusing": "Fusing", "gramnet": "GramNet", "lgrad": "LGrad",
         "lnp": "LNP", "npr": "NPR", "patchcraft": "PatchCraft", "rine": "RINE", "safe": "SAFE", "spai": "SPAI",
         "univfd": "UnivFD"}


def write_latex(out, summary, acc):
    """table.tex (formulas + two tables) and report.tex (the same as a complete document).
    Table 1: pooled Acc, AUC | accuracy per CSS level | caAcc, caAUC, overall Acc (mean over the levels). Table 2: real / fake accuracy per level.
    Percentages, rows sorted by caAUC; in table 1 best in bold and second underlined per column."""
    rows, missing = [], []
    for s in summary:
        r = {"name": NAMES.get(s["detector"], s["detector"]), "acc_pooled": s["acc_pooled"], "auc": s["auc"],
             "caAcc": s["caAcc"], "caAUC": s["caAUC"], "overall_acc": s["overall_acc"]}
        for lv in LEVELS:
            r |= {f"acc{lv}": acc[s["detector"], lv], f"real{lv}": s[f"acc_real_css{lv}"],
                  f"fake{lv}": s[f"acc_fake_css{lv}"]}
        if s.get("missing"):  # images the detector failed on, counted as errors
            missing.append(f"{r['name']} {int(s['missing'])}")
            r["name"] += r"$^\dagger$"
        # half width of the 95% bootstrap interval, shown as a subscript in table 1
        r["ci"] = {c: (s.get(f"{b}_hi"), s.get(f"{b}_lo")) for c, b in
                   [("acc_pooled", "acc"), ("overall_acc", "overall_acc"), ("auc", "auc"), ("caAcc", "caAcc"), ("caAUC", "caAUC")]
                   + [(f"acc{lv}", f"acc_css{lv}") for lv in LEVELS]}
        rows.append(r)
    rows.sort(key=lambda r: -(r["caAUC"] or 0))
    main = ["acc_pooled", "auc"] + [f"acc{lv}" for lv in LEVELS] + ["caAcc", "caAUC", "overall_acc"]
    split = [f"{m}{lv}" for lv in LEVELS for m in ("real", "fake")]
    # highlight only in table 1: a detector that calls everything real (or fake) would win real/fake accuracy
    rank = {c: sorted({r[c] for r in rows if r[c] is not None}, reverse=True)[:2] for c in main}

    def cell(r, c):
        v = r[c]
        if v is None:
            return "--"
        t = f"{100 * v:.1f}"
        if c in rank:
            t = rf"\textbf{{{t}}}" if v == rank[c][0] else rf"\underline{{{t}}}" if v == rank[c][1] else t
        hi, lo = r["ci"].get(c, (None, None))
        return t if hi is None or lo is None else rf"{t}$_{{\pm {50 * (hi - lo):.1f}}}$"

    def body(cols):
        lines = []
        for i, r in enumerate(rows):
            if i and i % 5 == 0:
                lines.append(r"\addlinespace")
            lines.append(r["name"] + " & " + " & ".join(cell(r, c) for c in cols) + r" \\")
        return lines

    levels = " & ".join(f"CSS {lv}" for lv in LEVELS)
    t1 = [
        r"\begin{table}[t]", r"\centering", r"\small",
        r"\caption{Detection on the TRIAGE test split (\%), sorted by caAUC. Acc$_\text{pooled}$ and AUC on all test images; "
        r"accuracy on each CSS level (1 = least, 4 = most sensitive content); caAcc and caAUC weight the levels "
        r"with the centroid weights; Overall Acc is the plain mean of the four level accuracies. "
        r"Subscripts: half width of the 95\% bootstrap interval (1000 resamples within each CSS level $\times$ "
        r"label cell). Best in bold, second underlined. Inference is deterministic (fixed test split, released "
        r"weights), so the intervals, not seeds, carry the uncertainty."
        + (r" $^\dagger$Images without a score, counted as errors: " + ", ".join(missing) + "." if missing else "")
        + "}",
        r"\label{tab:css_main}",
        r"\resizebox{\linewidth}{!}{%", r"\begin{tabular}{l cc cccc ccc}", r"\toprule",
        r"& \multicolumn{2}{c}{All images} & \multicolumn{4}{c}{Accuracy per CSS level} "
        r"& \multicolumn{3}{c}{Over CSS levels} \\",
        r"\cmidrule(lr){2-3} \cmidrule(lr){4-7} \cmidrule(lr){8-10}",
        r"Detector & Acc$_\text{pooled}$ & AUC & " + levels + r" & caAcc & caAUC & Overall Acc \\", r"\midrule",
        *body(main), r"\bottomrule", r"\end{tabular}}", r"\end{table}"]
    t2 = [
        r"\begin{table}[t]", r"\centering", r"\small",
        r"\caption{Accuracy on real images (Real, reals called real) and on fake images (Fake, fakes called fake) "
        r"for each CSS level (\%), at the threshold released by the authors; same order as "
        r"Table~\ref{tab:css_main}. A detector near 100 / 0 calls almost everything real.}",
        r"\label{tab:css_real_fake}",
        r"\begin{tabular}{l " + " ".join(["cc"] * K) + "}", r"\toprule",
        "& " + " & ".join(rf"\multicolumn{{2}}{{c}}{{CSS {lv}}}" for lv in LEVELS) + r" \\",
        " ".join(rf"\cmidrule(lr){{{2 + 2 * i}-{3 + 2 * i}}}" for i in range(K)),
        "Detector & " + " & ".join(["Real & Fake"] * K) + r" \\", r"\midrule",
        *body(split), r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    tables = "\n".join(t1) + "\n\n" + "\n".join(t2) + "\n"
    (out / "table.tex").write_text(FORMULAS + tables, encoding="utf-8")
    (out / "report.tex").write_text(  # the same, as a complete document: pdflatex report.tex
        "\\documentclass[a4paper,10pt]{article}\n\\usepackage[margin=2cm]{geometry}\n"
        "\\usepackage{amsmath,booktabs,graphicx}\n\\begin{document}\n\\section*{CSS benchmark: detectors on the "
        "TRIAGE test split}\n\\subsection*{Metrics}\n" + FORMULAS + "\\subsection*{Results}\n"
        + tables.replace("[t]", "[h]") + "\\end{document}\n", encoding="utf-8")


# the metrics as computed by rates(), auc(), headline() and weights() above
FORMULAS = r"""% Metrics of the tables, as computed by score.py. Needs amsmath, booktabs and graphicx.
Each test image $i$ has label $y_i\in\{0,1\}$ (1 = fake), CSS level $c_i\in\{1,\dots,K\}$ ($K=4$), score $s_i$
(higher = more fake) and prediction $\hat y_i=\mathbf{1}[s_i\ge\tau]$ at the threshold $\tau$ released by the
authors; an image the detector fails on counts as an error. With $\mathcal{I}_k=\{i: c_i=k\}$, $n_k=|\mathcal{I}_k|$,
$N=\sum_k n_k$:
\begin{align}
\mathrm{Acc}_{\mathrm{pooled}} &= \frac{1}{N}\sum_{i=1}^{N}\mathbf{1}[\hat y_i=y_i] = \sum_{k=1}^{K}\frac{n_k}{N}\,\mathrm{Acc}_k,
\qquad \mathrm{Acc}_k = \frac{1}{n_k}\sum_{i\in\mathcal{I}_k}\mathbf{1}[\hat y_i=y_i],\\
\mathrm{Real}_k &= \frac{\sum_{i\in\mathcal{I}_k}\mathbf{1}[y_i=0,\ \hat y_i=0]}{\sum_{i\in\mathcal{I}_k}\mathbf{1}[y_i=0]},
\qquad
\mathrm{Fake}_k = \frac{\sum_{i\in\mathcal{I}_k}\mathbf{1}[y_i=1,\ \hat y_i=1]}{\sum_{i\in\mathcal{I}_k}\mathbf{1}[y_i=1]},\\
\mathrm{AUC} &= \frac{1}{|P|\,|Q|}\sum_{i\in P}\sum_{j\in Q}\Big(\mathbf{1}[s_i>s_j]+\tfrac12\,\mathbf{1}[s_i=s_j]\Big),
\qquad P=\{i:y_i=1\},\ Q=\{j:y_j=0\},\\
\mathrm{caAcc} &= \sum_{k=1}^{K} w_k\,\mathrm{Acc}_k, \qquad
\mathrm{caAUC} = \sum_{k=1}^{K} w_k\,\mathrm{AUC}_k, \qquad
\mathrm{Acc}_{\mathrm{overall}} = \frac{1}{K}\sum_{k=1}^{K}\mathrm{Acc}_k,
\end{align}
where $\mathrm{AUC}_k$ is the AUC restricted to $\mathcal{I}_k$ (images without a score are left out of the AUC)
and the centroid (rank-order centroid) weights, increasing with the sensitivity of the content, are
\begin{equation}
w_k = \frac{1}{K}\sum_{j=K-k+1}^{K}\frac{1}{j},\qquad
(w_1,w_2,w_3,w_4) = \Big(\tfrac{1}{16},\ \tfrac{7}{48},\ \tfrac{13}{48},\ \tfrac{25}{48}\Big)\approx(0.063,\ 0.146,\ 0.271,\ 0.521).
\end{equation}

"""


def main(argv=None):
    ap_ = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap_.add_argument("--manifest", required=True)
    ap_.add_argument("--root", default=None)
    ap_.add_argument("--split", type=float, default=1.0)
    ap_.add_argument("--image-dir", default=None, help="flat folder with all the images (paths in predictions.csv)")
    ap_.add_argument("--subset", default=None, help="keep only records with this triage_split, e.g. test")
    ap_.add_argument("--scores", required=True, help="folder with the <detector>.csv files")
    ap_.add_argument("--out", default="results")
    ap_.add_argument("--n-boot", type=int, default=1000)
    ap_.add_argument("--seed", type=int, default=0)
    a = ap_.parse_args(argv)

    rows = data.load(a.manifest, a.root, a.split, subset=a.subset, image_dir=a.image_dir)
    by = [c for c in data.BREAKDOWN if any(r[c] for r in rows)]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    dets = sorted({f.name.split(".")[0] for f in Path(a.scores).glob("*.csv")})
    summary, per_css, by_rows = [], [], {c: [] for c in by}
    pf = open(out / "predictions.csv", "w", newline="", encoding="utf-8")
    pw = csv.writer(pf)
    pw.writerow(["detector", "image_path", "label_gt", "label_pred", "css", "score"])
    for det in dets:
        scores = read_scores(a.scores, det)
        name = {0: "real", 1: "fake"}
        for r in rows:  # one line per image; failed or missing images have an empty prediction
            sc, pr = scores.get(r["id"], ("", None))
            pw.writerow([det, r["path"], name[r["label"]], name.get(pr, ""), r["css"], sc])
        res = score_detector(rows, scores, by, a.n_boot, a.seed)
        (out / f"{det}.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
        p, ci, per = res["pooled"], res["ci95"], res["per_css"]
        row = {"detector": det, "n": p["n"], "missing": res["missing_predictions"]}
        row |= {f"acc_real_css{lv}": per[lv]["tnr"] for lv in data.LEVELS}
        row |= {f"acc_fake_css{lv}": per[lv]["tpr"] for lv in data.LEVELS}
        row |= {"acc_pooled": p["acc"], "overall_acc": res.get("overall_acc"), "auc": p["auc"],
                "caAcc": res.get("caAcc", {}).get("centroid"),
                "caAcc_lo": ci.get("caAcc_centroid", [None])[0], "caAcc_hi": ci.get("caAcc_centroid", [None, None])[1],
                "caAUC": res.get("caAUC", {}).get("centroid"),
                "caAUC_lo": ci.get("caAUC_centroid", [None])[0], "caAUC_hi": ci.get("caAUC_centroid", [None, None])[1]}
        row |= {f"{k}_{e}": ci.get(b, [None, None])[j] for k, b in
                [("acc", "acc"), ("overall_acc", "overall_acc"), ("auc", "auc")] + [(f"acc_css{lv}", f"acc_css{lv}") for lv in data.LEVELS]
                for j, e in enumerate(("lo", "hi"))}
        summary.append(row)
        for lv, m in per.items():
            per_css.append({"detector": det, "css": lv, **m})
        for c in by:
            for v, m in res[f"by_{c}"].items():
                by_rows[c].append({"detector": det, c: v, **m})
        print(f"{det}: acc={_r(p['acc'])} auc={_r(p['auc'])} caAcc={_r(row['caAcc'])} caAUC={_r(row['caAUC'])}")
    pf.close()
    if not summary:
        raise SystemExit(f"no score files in {a.scores}")
    write_csv(out / "summary.csv", summary)
    write_csv(out / "per_css.csv", per_css)
    write_table(out, summary, per_css)
    for c, rr in by_rows.items():
        write_csv(out / f"by_{c}.csv", rr)


if __name__ == "__main__":
    main()
