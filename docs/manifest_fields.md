# Manifest fields

The pipeline keeps one JSON manifest per source pool in `metadata/manifest_<source>.json`, with one entry per image. Every source uses the same fields, built by `greve.common.make_entry`; a field a source cannot fill is `null`. The released manifest (`manifest/greve_manifest_<split>.json.gz`; `greve_manifest.json` when built locally) holds the 78,000 selected entries plus the release fields listed at the end.

## Identity and provenance

| Field | Meaning |
|---|---|
| `image_id` | unique id: `<source>_<label>_<source id>` |
| `filename` | file name in the flat image folder; for released files, see `greve_path` |
| `source_dataset` | `openfake`, `sidset`, `rrdataset`, `dgm4`, `unsafebench`, `visualnews`, `medic`, `incidents`, or `cssgen` for our generated fakes |
| `label` | `real` or `fake`; fakes are always fully synthetic, and tampered images are never included |
| `original_label_value` | the label as the source dataset encodes it |
| `hf_repo`, `split`, `source_image_id` | where the image came from in its source |
| `condition` | RRDataset condition (only `original` is used) |
| `generator_model`, `generator_type`, `generator_release` | for fakes, as given by the source or by our generation run |
| `orig_format`, `orig_mode`, `orig_w`, `orig_h`, `orig_jpeg_q` | the original file as collected (`orig_jpeg_q` is the estimated JPEG quality) |
| `pixel_sha256` | SHA-256 of the decoded RGB pixels (from `greve.curate.dedup_phash`); `greve.release.rebuild_from_manifest` uses it to check downloaded originals |

## Annotation

| Field | Meaning |
|---|---|
| `css_proposer_responses` | `{proposer key: parsed JSON answer}` for `qwen3-vl-8b`, `internvl3-8b` and `pixtral-12b` (schema in [annotation.md](annotation.md)) |
| `css_aggregated_response` | the aggregator's parsed answer plus `guards`, `guarded_css_score`, `fiction_votes`, `fiction_cap_applied` and `uncapped_css_score` |
| `css_score` | **final CSS level (1–4)** after all guards |

The manifests store the parsed answers. The raw text of every model answer is kept in the per-model journals `metadata/css_scores/<model>.jsonl`, which are written while the pipeline runs and are not part of the release.

## Release fields (released manifest only)

| Field | Meaning |
|---|---|
| `greve_path` | path inside the release: `<split>/CSS-<level>/<real\|fake>/<name>.jpg` |
| `greve_split` | `train` (25%), `val` (15%) or `test` (60%) |
| `greve_action` | `copied` (original JPEG bytes kept unchanged) or `converted` (re-encoded to JPEG) |
| `greve_from_format` | format of the original file |
| `greve_jpeg_q` | JPEG quality used for converted images, drawn from the qualities of the real JPEGs at the same CSS level |
| `greve_path_presplit` | path before the split step |

## Pipeline-only fields (not in the released manifest)

The pipeline manifests in `metadata/` also carry these fields. They are used while sampling, generating, deduplicating and selecting, and are removed from the released manifest.

| Field | Meaning |
|---|---|
| `scenario`, `sensitivity_tier` | sampling bucket used when building the candidate pool (not the CSS) |
| `prompt`, `negative_prompt`, `gen_params`, `random_seed` | generation settings for our fakes; for OpenFake, `prompt` is that dataset's own prompt |
| `paired_with`, `prompt_from_source`, `target_level` | our fakes only: the real image whose scene description gave the prompt, its source, and the CSS level the prompt targeted. A fake counts at the level it is scored at, not at `target_level`. |
| `gen_run` | our fakes only: `pilot` or `full` |
| `prompt_edit`, `prompt_original` | our fakes only, when the prompt was edited: `minor_to_adult` (a CSS-4 harm prompt that described minors was rewritten to adults) and the original text |
| `sampling_date` | date the entry was created |
| `phash64`, `dup_cluster` | 64-bit pHash and near-duplicate cluster from `greve.curate.dedup_phash` |
| `cell` | selection cell, e.g. `fake_css3` |
