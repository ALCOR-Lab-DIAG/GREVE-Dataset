# Evaluation code: detectors, VLMs, caAcc / caAUC

Code to run AI-generated image detectors and zero-shot VLMs on the TRIAGE split of GREVE and to score
them per content-sensitivity level (CSS 1..4), with the content-aware metrics caAcc and caAUC.
Only released checkpoints are used: no training.

| file | what it does |
|---|---|
| `base.py`, `detectors.py`, `adapters/*.py` | one adapter per detector, written from the authors' official test code (cited in the docstrings); downloads code and weights on first use |
| `detect.py` | runs one detector on the manifest, writes `<out>/<detector>.csv` (id, score, pred) |
| `vlm_detect.py` | zero-shot VLM detectors (Qwen2.5-VL, Qwen3-VL, InternVL3 / 3.5, Kimi-VL), same output format |
| `css_data.py` | manifest loading, `--subset` / deterministic `--split`, sharding |
| `score.py` | metrics: per-CSS accuracy / AUC / AP, overall accuracy and AUC, caAcc, caAUC, stability band, bootstrap CIs, breakdowns |
| `full_run.sh`, `run_all.sh`, `smoke.sh`, `vlm_run.sh`, `moejepa_run.sh` | multi-GPU run scripts (one shard per GPU, resumable) |
| `in_container.sh` | runs a command in the podman image on given GPUs |
| `Containerfile`, `Containerfile.vlm`, `Containerfile.kimi` | images: `css-eval` (detectors), `css-vlm` (VLMs, MoE-JEPA), `css-vlm-kimi` (Kimi-VL) |
| `test_eval.py`, `make_toy.py` | unit tests of the metrics and a toy dataset to try the pipeline |

## Setup

```bash
podman build -t css-eval -f Containerfile .
podman build -t css-vlm -f Containerfile.vlm .        # VLMs and MoE-JEPA
podman build -t css-vlm-kimi -f Containerfile.kimi .  # Kimi-VL only
```
GPUs are passed with CDI (`--device nvidia.com/gpu=N`, needs `nvidia-container-toolkit`).

Environment variables:

| variable | meaning |
|---|---|
| `CSS_DATA` | dataset folder, mounted read-only in the container (required by `in_container.sh`) |
| `CSS_WEIGHTS` | where weights, detector repos and HF / CLIP / torch backbones are downloaded (default `$CSS_DATA/TRIAGE/weights`, mounted as `/root/.cache`) |
| `CSS_MANIFEST` | path to `triage_manifest.json` (used by `vlm_run.sh`, `moejepa_run.sh`) |
| `IMAGE_DIR` | optional flat folder with all the images, instead of `<root>/<triage_path>` |
| `CSS_IMAGE` | podman image for `in_container.sh` (default `css-eval`) |
| `CSS_CACHE` | download cache inside the container (default `~/.cache/css-eval`) |

## Running

```bash
export CSS_DATA=/path/to/css_dataset
M=/path/to/TRIAGE/triage_manifest.json

# one detector, one GPU
./in_container.sh 0 python detect.py --detector npr --manifest "$M" --subset test --out scores_test
python detect.py --list                                   # registered detectors

# all detectors on the test split, 8 GPUs, seed 0 (then the metrics)
./full_run.sh "$M" test 8 0

# one VLM, data parallel over the given GPUs
CSS_MANIFEST="$M" IMAGE_DIR=/path/to/all_images ./vlm_run.sh qwen3vl_8b 0,1,2,3

# metrics only
./in_container.sh 0 python score.py --manifest "$M" --subset test --scores scores_test --out results_test
```
`score` in the score files is "how fake" (higher = more fake); `pred` is the decision at the threshold released
by the authors. Missing or failed images count as errors in the accuracies and are left out of AUC (the count is
reported). Outputs: `summary.csv`, `per_css.csv`, `predictions.csv`, `by_<column>.csv`, `<detector>.json`
(see the docstring of `score.py`).

Manifest fields used: `image_id`, `triage_path`, `label` (real / fake), `css_score` (1..4), `triage_split`;
breakdowns on `source_dataset`, `generator_model`, `generator_type`, `condition`, `scenario`.

## Metrics

With `acc_k` and `AUC_k` the accuracy and ROC AUC on the images of CSS level k (1 = least, 4 = most sensitive):

- caAcc = Σ_k w_k · acc_k, caAUC = Σ_k w_k · AUC_k, with the rank-order centroid weights
  w = (3, 7, 13, 25) / 48 ≈ (0.063, 0.146, 0.271, 0.521), i.e. w_k = (1/K) Σ_{j=K-k+1..K} 1/j with K = 4.
- Stability band: T_j = mean of acc over the j most sensitive levels (j = 1..4); the band is
  [min_j T_j, max_j T_j].
- 95% CIs by bootstrap over images (`--n-boot`, default 1000).

## Tests

```bash
python test_eval.py                     # metrics against hand-computed values and scikit-learn, manifest and split
python make_toy.py toy && python detect.py --detector dummy --manifest toy/manifest.json --device cpu
```
