# CSS annotation: protocol and outputs

Every image in GREVE has a **Content Sensitivity Score (CSS)** from 1 to 4. The score describes how much harm a wrong authenticity verdict on the image could cause. It is assigned from the depicted content only, as if the image were an authentic photograph, and it never looks at whether the image is real or generated.

| CSS | Name | Fires when the image shows… |
|---|---|---|
| 1 | Everyday | nothing harmful, distressing or newsworthy |
| 2 | Public-interest, non-distressing | identifiable public figures, institutions, events, peaceful protests, uniformed forces not in confrontation, routine healthcare, minor incidents |
| 3 | Distressing, non-graphic | injured people receiving care, rescue in progress, disaster aftermath without casualties, confrontations and arrests, threatening weapons, displacement, grief, hate content, suffering animals |
| 4 | Graphic or catastrophic | dead bodies, severe injury or blood, active violence, combat or explosions in populated places, mass-casualty scenes, catastrophic destruction with victims |

The full rubric, with the edge rules (artwork, staged and fictional content, media covers, combat sports, weapons, medical scenes, animals), is [`greve/annotate/prompts/proposer_prompt.txt`](../greve/annotate/prompts/proposer_prompt.txt).

## Three stages

```
image ──► Qwen3-VL-8B ──┐
      ──► InternVL3-8B ─┼─► Qwen2.5-7B (text only) ──► code guards ──► css_score
      ──► Pixtral-12B ──┘     aggregator
```

### 1. Proposers (`greve/annotate/proposer_*.py`, shared code in `proposer_common.py`)

| Key in the manifest | Hugging Face model |
|---|---|
| `qwen3-vl-8b` | `Qwen/Qwen3-VL-8B-Instruct` |
| `internvl3-8b` | `OpenGVLab/InternVL3-8B-hf` |
| `pixtral-12b` | `mistral-community/pixtral-12b` (transformers conversion of `mistralai/Pixtral-12B-2409`) |

Settings shared by all three proposers:
- bf16;
- greedy decoding (`do_sample=False`), `max_new_tokens=1024`;
- images resized so the longer side is at most 1536 px (aspect ratio kept);
- seed 9999.

The rubric is a decision procedure: the model applies Test 4, then Test 3, then Test 2, and the first test that fires sets the score. Each proposer answers with one JSON object:

```json
{
  "scene_description": "...",
  "key_sensitivity_factors": ["..."],
  "test_fired": "T4_graphic | T3_distressing | T2_public_interest | T1_none",
  "evidence_for_test": "...",
  "public_figure_or_event": "... or null",
  "css_score": 1,
  "confidence": "high | medium | low",
  "flags": ["staged_or_fictional | media_cover | sexual_content | low_quality | none"],
  "justification": "..."
}
```

An answer is accepted only if it parses as JSON and `css_score` is an integer from 1 to 4. A failed answer is recorded and retried on the next run.

### 2. Aggregator (`greve/annotate/aggregator.py`)

`Qwen/Qwen2.5-7B-Instruct` runs text only (bf16, greedy decoding, `max_new_tokens=768`). It receives the three proposer answers, formatted as "Analyst 1..3", together with [`aggregator_prompt.txt`](../greve/annotate/prompts/aggregator_prompt.txt). That prompt uses the same scale and the same tests as the proposer rubric. The aggregator returns:

- `image_context`: photograph, artwork, media cover, …;
- `context_cap`: 2 for artwork, 3 for a media cover, otherwise null;
- `decisive_evidence`, `evidence_check`, `consensus_level`, `agreement_analysis`;
- `final_test_fired`, `final_css_score`, `final_justification`.

### 3. Code guards (`apply_guards`, `apply_fiction_cap` in `aggregator.py`)

The guards run in this order on the aggregator's `final_css_score`:

1. **Majority ceiling.** If at least 2 of the 3 proposers agree, the final score can go below their score but never above it. If all three disagree, the final score is clamped to the range of their scores. Disable with `--no_majority_ceiling`.
2. **Context cap.** The final score never exceeds the `context_cap` the aggregator itself declared. Disable with `--no_context_cap`.
3. **Fiction cap.** If at least 2 proposers flag `staged_or_fictional`, the score is capped at 2. Set the threshold with `--fiction_min_votes`; 0 disables the cap.

The guard results are stored in `css_aggregated_response`:
- `guards`: e.g. `["majority_ceiling 3->2"]`;
- `guarded_css_score`;
- `fiction_votes` and `fiction_cap_applied`;
- `uncapped_css_score`: the aggregator's raw answer.

The final value is written to the entry's top-level `css_score`.

## Running it

Each script reads one or more manifests (default: every `metadata/manifest_*.json`) and writes its results back into them.

- **Resume.** An image is skipped once it has a parsed answer produced with the current prompt. The aggregator skips an image when the proposer answers it would read and its own prompt are both unchanged.
- **Sharding.** `--num_shards N --shard i` runs N copies of a script, one per GPU. Images are split by a stable hash of `image_id`. All copies share one append-only `.jsonl` journal in `metadata/css_scores/`, and manifest writes are file-locked, so the copies do not overwrite each other.
- **Interruption.** `kill` and Ctrl-C trigger a final merge into the manifests before the process exits.
- **Memory.** Every model fits on one 32 GB GPU. Pixtral needs `--batch_size 2`, which is its default.

`scripts/score_pool.sh <manifest> <tag>` runs the whole chain (three proposers, then the aggregator) on one pool.

Measured cost on 7 × 32 GB GPUs for about 82K images:
- proposers: about 2.5 days, set by Pixtral;
- aggregator: about 5 hours.
