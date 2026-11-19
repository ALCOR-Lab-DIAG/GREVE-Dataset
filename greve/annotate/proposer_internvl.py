"""
CSS proposer — InternVL3-8B.

Scores every image in the manifests with the prompt in proposer_prompt.txt and writes
the parsed answers into each entry as css_proposer_responses["internvl3-8b"].
Raw answers are kept in metadata/css_scores/internvl3-8b.json(l). Resuming is
automatic; see proposer_common.py for the details.

Uses OpenGVLab/InternVL3-8B-hf, the transformers-native release of the same
InternVL3-8B weights: no trust_remote_code, so no extra dependencies (timm,
flash_attn) and no breakage with transformers 5. Its processor does
InternVL's dynamic 448px tiling itself.

  python -m greve.annotate.proposer_internvl --gpu 1
"""

from greve.annotate.proposer_common import build_parser, hf_generate, load_hf_model, run

PROPOSER = "internvl3-8b"
MODEL_NAME = "OpenGVLab/InternVL3-8B-hf"


def main():
    parser = build_parser("InternVL3 CSS proposer", PROPOSER, default_gpu="0", default_model=MODEL_NAME)
    run(parser.parse_args(), PROPOSER, load_hf_model, hf_generate)


if __name__ == "__main__":
    main()
