"""
Generate fully synthetic fakes (text-to-image only) for the CSS-3 / CSS-4
cells with open models, from the prompts of gen_prompts.py.

Each image is saved as PNG (the generator's raw output, lossless) to
--output_dir as cssgen_fake_<model>_<prompt_id>.png and added to the manifest
(source "cssgen", label fake, generator_model = the model key, paired_with =
the real image the prompt was written from). The CSS score comes later from
the same proposers + aggregator as every other image, and an image counts in
the level it gets, whatever level its prompt targeted.

Resumable (images already in the manifest are skipped); shards split the
prompts by md5(prompt_id) so several GPUs can share one model.

  python -m greve.generate.generate --model flux-dev --prompts metadata/gen/pilot_prompts.jsonl \
      --metadata_path metadata/gen/manifest_gen_pilot.json --gpu 1 --num_shards 2 --shard 0
"""

import argparse
import hashlib
import io
import json
import sys
import time
from pathlib import Path

import os

# generator weights live on /data (the root disk is nearly full); the VLM scorers keep the default cache
# HF_HUB_CACHE: set it in the environment to choose where generator weights go
# the container allows 2048 threads in all: keep each generator's CPU thread pools small
for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "RAYON_NUM_THREADS"):
    os.environ.setdefault(_var, "4")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch

from greve.common import ManifestWriter, make_entry, sanitize_key, save_original
from greve.paths import IMAGES_DIR

NEGATIVE = ("cartoon, illustration, painting, drawing, anime, 3d render, cgi, video game, movie still, "
            "watermark, text, blurry, deformed")
# Chroma is trained uncensored: keep nudity out of the CSS fakes
NSFW_NEGATIVE = ", nudity, naked, nsfw, sexual content"
SIZES = {"landscape": (1216, 832), "portrait": (832, 1216), "square": (1024, 1024)}
HUNYUAN_SIZES = {"landscape": (1280, 768), "portrait": (768, 1280), "square": (1024, 1024)}

# key: (repo, pipeline class, dtype, offload, call kwargs, release)
# offload: True = enable_model_cpu_offload, False = whole pipeline on the GPU, "phased" = FLUX / Chroma:
#   encode every prompt first with the text encoders on the GPU, then free them and keep only
#   transformer + VAE on the GPU (no per-image weight transfers; ~2x faster than offload on 32 GB)
# cogview4 was in the pilot only (dropped: it softens harmful prompts, 19% CSS-4 hit rate)
MODELS = {
    "flux-dev": ("black-forest-labs/FLUX.1-dev", "FluxPipeline", torch.bfloat16, "phased",
                 dict(num_inference_steps=28, guidance_scale=3.5, max_sequence_length=512), "2024-08"),
    "flux-schnell": ("black-forest-labs/FLUX.1-schnell", "FluxPipeline", torch.bfloat16, "phased",
                     dict(num_inference_steps=4, guidance_scale=0.0, max_sequence_length=256), "2024-08"),
    "sdxl": ("stabilityai/stable-diffusion-xl-base-1.0", "StableDiffusionXLPipeline", torch.float16, False,
             dict(num_inference_steps=30, guidance_scale=6.0, negative_prompt=NEGATIVE), "2023-07"),
    "pixart-sigma": ("PixArt-alpha/PixArt-Sigma-XL-2-1024-MS", "PixArtSigmaPipeline", torch.float16, False,
                     dict(num_inference_steps=20, guidance_scale=4.5, negative_prompt=NEGATIVE), "2024-03"),
    "sana-1.6b": ("Efficient-Large-Model/Sana_1600M_1024px_BF16_diffusers", "SanaPipeline", torch.bfloat16, False,
                  dict(num_inference_steps=20, guidance_scale=4.5, negative_prompt=NEGATIVE), "2024-11"),
    "lumina-2": ("Alpha-VLLM/Lumina-Image-2.0", "Lumina2Pipeline", torch.bfloat16, False,
                 dict(num_inference_steps=40, guidance_scale=4.0, cfg_trunc_ratio=0.25, cfg_normalization=True,
                      negative_prompt=NEGATIVE), "2025-01"),
    "cogview4": ("THUDM/CogView4-6B", "CogView4Pipeline", torch.bfloat16, True,
                 dict(num_inference_steps=40, guidance_scale=3.5, negative_prompt=NEGATIVE), "2025-03"),
    "hunyuandit": ("Tencent-Hunyuan/HunyuanDiT-v1.2-Diffusers", "HunyuanDiTPipeline", torch.float16, False,
                   dict(num_inference_steps=40, guidance_scale=5.0, negative_prompt=NEGATIVE), "2024-07"),
    "flux-krea": ("black-forest-labs/FLUX.1-Krea-dev", "FluxPipeline", torch.bfloat16, "phased",
                  dict(num_inference_steps=28, guidance_scale=4.5, max_sequence_length=512), "2025-07"),
    "chroma1-hd": ("lodestones/Chroma1-HD", "ChromaPipeline", torch.bfloat16, "phased",
                   dict(num_inference_steps=40, guidance_scale=3.0, negative_prompt=NEGATIVE + NSFW_NEGATIVE), "2025-08"),
    "z-image-turbo": ("Tongyi-MAI/Z-Image-Turbo", "ZImagePipeline", torch.bfloat16, False,
                      dict(num_inference_steps=9, guidance_scale=0.0), "2025-11"),
    "playground-v2.5": ("playgroundai/playground-v2.5-1024px-aesthetic", "StableDiffusionXLPipeline", torch.float16, False,
                        dict(num_inference_steps=50, guidance_scale=3.0, negative_prompt=NEGATIVE), "2024-02"),
    "kolors": ("Kwai-Kolors/Kolors-diffusers", "KolorsPipeline", torch.float16, False,
               dict(num_inference_steps=50, guidance_scale=5.0, negative_prompt=NEGATIVE), "2024-07"),
    "auraflow": ("fal/AuraFlow-v0.3", "AuraFlowPipeline", torch.float16, False,
                 dict(num_inference_steps=50, guidance_scale=3.5, negative_prompt=NEGATIVE), "2024-09"),
    "kandinsky-3": ("kandinsky-community/kandinsky-3", "Kandinsky3Pipeline", torch.float16, True,  # 21.7 GB text encoder: offload
                    dict(num_inference_steps=25, guidance_scale=3.0, negative_prompt=NEGATIVE), "2023-12"),
}
FP16_VARIANT = {"sdxl", "playground-v2.5", "kolors", "kandinsky-3", "auraflow"}


def auraflow_transformer(repo, dtype):
    """The AuraFlow fp16 shards are indexed by a legacy file name (...safetensors.fp16.index.json)
    that diffusers does not look for, so the transformer is built and its shards loaded by hand."""
    from diffusers import AuraFlowTransformer2DModel
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file
    d = Path(hf_hub_download(repo, "transformer/config.json", local_files_only=True)).parent
    with torch.device("meta"):
        model = AuraFlowTransformer2DModel.from_config(AuraFlowTransformer2DModel.load_config(d))
    shards = sorted(set(json.load(open(d / "diffusion_pytorch_model.safetensors.fp16.index.json"))["weight_map"].values()))
    state = {}
    for f in shards:
        state.update(load_file(d / f))
    model.load_state_dict(state, assign=True, strict=True)
    return model.to(dtype).eval()


def load(key, gpu):
    import diffusers
    repo, cls, dtype, offload, _, _ = MODELS[key]
    kw = {"variant": "fp16"} if key in FP16_VARIANT else {}
    if key == "auraflow":
        kw["transformer"] = auraflow_transformer(repo, dtype)
    pipe = getattr(diffusers, cls).from_pretrained(repo, torch_dtype=dtype, **kw)
    if offload == "phased":
        pass  # moved to the GPU stage by stage in phased()
    elif offload:
        pipe.enable_model_cpu_offload(gpu_id=gpu)
    else:
        pipe.to(f"cuda:{gpu}")
    pipe.set_progress_bar_config(disable=True)
    return pipe


def phased(pipe, gpu, todo, call):
    """FLUX / Chroma on one 32 GB GPU without offload: all prompt embeddings first (text encoders
    on the GPU), then the text encoders go back to the CPU and transformer + VAE stay on the GPU.
    Yields (prompt, kwargs to pass to pipe() instead of the prompt)."""
    dev = f"cuda:{gpu}"
    chroma = type(pipe).__name__ == "ChromaPipeline"
    encoders = [pipe.text_encoder] if chroma else [pipe.text_encoder, pipe.text_encoder_2]
    for m in encoders:
        m.to(dev)
    embs = []
    with torch.no_grad():
        for p in todo:
            if chroma:
                pe, _, mask, npe, _, nmask = pipe.encode_prompt(
                    prompt=p["prompt"], negative_prompt=call["negative_prompt"], device=dev,
                    do_classifier_free_guidance=True, max_sequence_length=512)
                embs.append(dict(prompt_embeds=pe.cpu(), prompt_attention_mask=mask.cpu(),
                                 negative_prompt_embeds=npe.cpu(), negative_prompt_attention_mask=nmask.cpu()))
            else:
                pe, pooled, _ = pipe.encode_prompt(prompt=p["prompt"], prompt_2=None, device=dev,
                                                   max_sequence_length=call["max_sequence_length"])
                embs.append(dict(prompt_embeds=pe.cpu(), pooled_prompt_embeds=pooled.cpu()))
    for m in encoders:
        m.to("cpu")
    torch.cuda.empty_cache()
    pipe.transformer.to(dev)
    pipe.vae.to(dev)
    print(f"  {len(embs)} prompts encoded; transformer + VAE on {dev}", flush=True)
    for p, e in zip(todo, embs):
        yield p, {k: v.to(dev) for k, v in e.items()}


def main():
    parser = argparse.ArgumentParser(description="Generate GREVE fakes with an open T2I model")
    parser.add_argument("--model", required=True, choices=sorted(MODELS))
    parser.add_argument("--prompts", required=True)
    parser.add_argument("--metadata_path", required=True)
    parser.add_argument("--output_dir", default=IMAGES_DIR)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None, help="Stop after this many images (smoke tests)")
    args = parser.parse_args()

    prompts = [json.loads(l) for l in open(args.prompts)]
    # prompt files of the full run assign each prompt to one model ("model"); the pilot's don't
    prompts = [p for p in prompts if p.get("model", args.model) == args.model]
    prompts = [p for p in prompts
               if int(hashlib.md5(p["prompt_id"].encode()).hexdigest(), 16) % args.num_shards == args.shard]
    manifest = ManifestWriter(args.metadata_path)
    have = {e["image_id"] for e in manifest.entries}

    def image_id_of(p):
        return f"cssgen_fake_{args.model}_{sanitize_key(p['prompt_id'], maxlen=100)}"

    todo = [p for p in prompts if image_id_of(p) not in have][:args.limit]
    print(f"{args.model} shard {args.shard}/{args.num_shards}: {len(prompts)} prompts, {len(todo)} to generate", flush=True)
    if not todo:
        return
    pipe = load(args.model, args.gpu)
    call = MODELS[args.model][4]
    sizes = HUNYUAN_SIZES if args.model == "hunyuandit" else SIZES
    if MODELS[args.model][3] == "phased":
        stream = phased(pipe, args.gpu, todo, call)
        call = {k: v for k, v in call.items() if k != "negative_prompt"}  # already in the embeddings
    else:
        stream = ((p, {"prompt": p["prompt"]}) for p in todo)
    t0 = time.time()
    for i, (p, prompt_kw) in enumerate(stream, 1):
        if i == 1:
            t0 = time.time()
        seed = int(hashlib.md5(f"{args.model}:{p['prompt_id']}".encode()).hexdigest(), 16) % 2**31
        w, h = sizes[p.get("aspect", "landscape")]
        g = torch.Generator("cpu").manual_seed(seed)
        img = pipe(width=w, height=h, generator=g, **prompt_kw, **call).images[0]
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        image_id = image_id_of(p)
        filename, info = save_original(buf.getvalue(), Path(args.output_dir), image_id)
        manifest.add(make_entry(
            image_id=image_id,
            filename=filename,
            source="cssgen",
            label="fake",
            sensitivity="HIGH",
            scenario=f"target_css{p['target_level']}",
            split=None,
            generator=args.model,
            generator_type="text-to-image",
            generator_release=MODELS[args.model][5],
            original_label="fake",
            source_image_id=p["prompt_id"],
            paired_with=p["from_image_id"],
            condition="original",
            hf_repo=MODELS[args.model][0],
            seed=seed,
            info=info,
            extra={"prompt": p["prompt"], "target_level": p["target_level"], "prompt_from_source": p["from_source"],
                   "gen_params": {k: v for k, v in call.items() if k != "negative_prompt"} | {"width": w, "height": h},
                   "negative_prompt": MODELS[args.model][4].get("negative_prompt")}
                  | {k: p[k] for k in ("prompt_edit", "prompt_original", "scene_proposer", "wave") if k in p},
        ))
        if i % 10 == 0 or i == len(todo):
            print(f"  {i}/{len(todo)} images, {(time.time() - t0) / i:.1f} s/img", flush=True)
    total = manifest.export_json()
    manifest.close()
    print(f"DONE — {args.model} shard {args.shard}: {len(todo)} generated, {total} entries in {args.metadata_path}")


if __name__ == "__main__":
    main()
