"""
CSS proposer — Qwen3-VL-8B-Instruct.

Scores every image in the manifests with the prompt in proposer_prompt.txt and writes
the parsed answers into each entry as css_proposer_responses["qwen3-vl-8b"].
Raw answers are kept in metadata/css_scores/qwen3-vl-8b.json(l). Resuming is
automatic; see proposer_common.py for the details.

  python -m greve.annotate.proposer_qwen --gpu 0
"""

from greve.annotate.proposer_common import build_parser, hf_generate, load_hf_model, run

PROPOSER = "qwen3-vl-8b"
MODEL_NAME = "Qwen/Qwen3-VL-8B-Instruct"


def main():
    parser = build_parser("Qwen3-VL CSS proposer", PROPOSER, default_gpu="0,1", default_model=MODEL_NAME)
    run(parser.parse_args(), PROPOSER, load_hf_model, hf_generate)


if __name__ == "__main__":
    main()
