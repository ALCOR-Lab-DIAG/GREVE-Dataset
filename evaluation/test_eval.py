"""python test_eval.py : checks the metrics against hand values and scikit-learn (if installed)."""
import numpy as np

import score as S


def test_weights():
    W = S.weights()
    assert np.allclose(W["centroid"], [1 / 16, 7 / 48, 13 / 48, 25 / 48])
    assert np.allclose(W["linear"], [.1, .2, .3, .4])
    assert all(abs(w.sum() - 1) < 1e-12 and np.all(np.diff(w) >= 0) for w in W.values())


def test_auc_ap():
    y = np.array([0, 0, 1, 1]); s = np.array([0.1, 0.4, 0.35, 0.8])
    assert S.auc(y, s) == 0.75 and np.isclose(S.ap(y, s), 0.5 * 1 + 0.5 * 2 / 3)
    try:
        from sklearn.metrics import average_precision_score, roc_auc_score
    except ImportError:
        return
    rng = np.random.default_rng(0)
    for _ in range(50):
        y = rng.integers(0, 2, 300); s = rng.integers(0, 20, 300) / 20  # many ties
        assert np.isclose(S.auc(y, s), roc_auc_score(y, s))
        assert np.isclose(S.ap(y, s), average_precision_score(y, s))


def test_detector_metrics():
    # per level: (n_real, n_fake, reals right, fakes right)
    cells = {1: (10, 10, 9, 8), 2: (10, 10, 10, 6), 3: (10, 10, 7, 7), 4: (5, 5, 4, 2)}
    rows, scores = [], {}
    for lv, (nr, nf, kr, kf) in cells.items():
        for lab, n, k in ((0, nr, kr), (1, nf, kf)):
            for j in range(n):
                i = f"{lv}{lab}{j}"
                rows.append({"id": i, "label": str(lab), "css": str(lv), "source": "a"})
                pred = lab if j < k else 1 - lab
                scores[i] = (0.9 if pred else 0.1, pred)
    del scores["400"]  # one missing prediction: counts as an error
    r = S.score_detector(rows, scores, ["source"], n_boot=50, seed=0)
    acc = np.array([17 / 20, 16 / 20, 14 / 20, 5 / 10])
    assert np.allclose([r["per_css"][lv]["acc"] for lv in S.LEVELS], acc)
    assert np.isclose(r["caAcc"]["centroid"], S.weights()["centroid"] @ acc)
    assert np.isclose(r["caAcc"]["uniform"], acc.mean())
    pooled = (17 + 16 + 14 + 5) / 70
    assert np.isclose(r["pooled"]["acc"], pooled)
    assert np.isclose(r["overall_acc"], acc.mean())  # plain mean over the levels, not the pooled accuracy
    T = [acc[3], acc[2:].mean(), acc[1:].mean(), acc.mean()]
    assert np.allclose(r["band"], [min(T), max(T)]) and np.isclose(r["gap_css4_css1"], acc[3] - acc[0])
    assert r["missing_predictions"] == 1 and r["per_css"][4]["fpr"] == 0.4
    assert r["ci95"]["caAcc_centroid"][0] <= r["caAcc"]["centroid"] <= r["ci95"]["caAcc_centroid"][1]
    assert r["by_source"]["a"]["n"] == 70


def test_non_finite_scores(tmp="/tmp/css_nan_test"):
    # a NaN/inf score with a pred is no score: an error in the accuracy, out of the AUC, counted in missing
    import csv
    from pathlib import Path
    Path(tmp).mkdir(exist_ok=True)
    rows = [{"id": f"i{i}", "label": i % 2, "css": i % 4 + 1} for i in range(16)]
    with open(f"{tmp}/det.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "score", "pred"])
        for r in rows:  # everything right, except that i0 (real) is NaN and i1 (fake) is inf
            bad = {"i0": "nan", "i1": "inf"}.get(r["id"])
            w.writerow([r["id"], bad or (0.9 if r["label"] else 0.1), r["label"]])
    scores = S.read_scores(tmp, "det")
    assert set(scores) == {r["id"] for r in rows} - {"i0", "i1"}
    res = S.score_detector(rows, scores, [], 0, 0)
    assert res["missing_predictions"] == 2 and res["pooled"]["n_scored"] == 14
    assert np.isclose(res["pooled"]["acc"], 14 / 16) and res["pooled"]["auc"] == 1.0
    assert np.isclose(res["per_css"][1]["tnr"], 3 / 4) and np.isclose(res["per_css"][2]["tpr"], 3 / 4)


def test_manifest_and_split(tmp="/tmp/css_toy_test"):
    import json, subprocess, sys
    import css_data as data
    subprocess.run([sys.executable, "make_toy.py", tmp], check=True, capture_output=True)
    rows = data.load(f"{tmp}/manifest.json")
    assert len(rows) == 260 and {r["label"] for r in rows} == {0, 1} and rows[0]["path"].endswith(".jpg")
    part = data.load(f"{tmp}/manifest.json", split=0.7)
    assert 0.55 < len(part) / len(rows) < 0.85 and part == data.load(f"{tmp}/manifest.json", split=0.7)
    ids = {r["id"] for r in part}  # a fake is kept iff its paired real is kept
    recs = json.load(open(f"{tmp}/manifest.json"))
    assert all((r["image_id"] in ids) == (r["paired_with"] in ids) for r in recs if r["paired_with"])
    for i, r in enumerate(recs):
        r["triage_split"] = ("train", "val", "test")[i % 3]
    json.dump(recs, open(f"{tmp}/manifest.json", "w"))
    test = data.load(f"{tmp}/manifest.json", subset="test")
    assert {r["id"] for r in test} == {r["image_id"] for r in recs if r["triage_split"] == "test"}

if __name__ == "__main__":
    test_weights(); test_auc_ap(); test_detector_metrics(); test_non_finite_scores(); test_manifest_and_split()
    print("all tests passed")
