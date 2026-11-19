"""
CSS proposer — Pixtral-12B (2409).

Scores every image in the manifests with the prompt in proposer_prompt.txt and writes
the parsed answers into each entry as css_proposer_responses["pixtral-12b"].
Raw answers are kept in metadata/css_scores/pixtral-12b.json(l). Resuming is
automatic; see proposer_common.py for the details.

Uses mistral-community/pixtral-12b, the transformers conversion of
mistralai/Pixtral-12B-2409 (same weights), so it needs neither
mistral_inference nor a local copy of the weights. In bf16 the model takes
~25GB, so on a 32GB card the default batch is 2 (4 runs out of memory).
One GPU per process is fastest; to use more GPUs, run shards
(--num_shards / --shard) rather than splitting the model across cards.

  python -m greve.annotate.proposer_pixtral --gpu 2
"""

from greve.annotate.proposer_common import build_parser, hf_generate, load_hf_model, run

PROPOSER = "pixtral-12b"
MODEL_NAME = "mistral-community/pixtral-12b"


def main():
    parser = build_parser("Pixtral-12B CSS proposer", PROPOSER, default_gpu="0,1", default_model=MODEL_NAME,
                          default_batch_size=2)
    run(parser.parse_args(), PROPOSER, load_hf_model, hf_generate)


if __name__ == "__main__":
    main()
