"""
Shared machinery for the CSS proposers (proposer_qwen.py, proposer_internvl.py,
proposer_pixtral.py): each one asks a VLM to assign every image a Content
Sensitivity Score using the prompt in proposer_prompt.txt, and records the answer.

Per proposer, for every image in the manifests:
  1. The prompt is read from --prompt_file (default prompts/proposer_prompt.txt next to this
     file); a leading "<image>" line is dropped since the image is passed as
     its own chat content block. Its SHA-256 (first 12 hex) is recorded as
     prompt_sha with every answer.
  2. Answers go to a per-proposer results file (--output, default
     metadata/css_scores/<proposer>.json): an append-only journal (.jsonl),
     one line per image, exported as a JSON array at the end — so 80K+
     images cost one small append each, not a full rewrite.
  3. Every --merge_every answers and at the end (also on Ctrl-C / kill), the
     parsed answers are written into each manifest entry as
     css_proposer_responses[<proposer>] = <parsed JSON>. Merges run under
     the manifest lock on a fresh read, so the three proposers can run at
     the same time (on different GPUs) without overwriting each other.

Resuming is automatic: an image counts as done when its last answer parsed,
has css_score in 1-4, and was produced with the current prompt (same
prompt_sha) — so failed/unparsed answers are retried, and editing
proposer_prompt.txt makes every image get re-scored. --rescore ignores prior answers.

To use more GPUs for one model, run several copies with --num_shards N and
--shard 0..N-1, one GPU each (data parallel; splitting one model across GPUs
is slower). They share the results journal and merge safely.

All status messages and the tqdm bar go to a log file (--log, default
<output>.log, or <output>.shard<i>of<N>.log), so a run can be left detached.
"""

import argparse
import glob
import hashlib
import json
import logging
import os
import random
import re
import signal
import sys
from pathlib import Path

from PIL import Image
from tqdm import tqdm

from greve.common import ManifestWriter
from greve.paths import IMAGES_DIR

HERE = Path(__file__).resolve().parent
PROMPT_PATH = HERE / "prompts" / "proposer_prompt.txt"
DEFAULT_MANIFESTS = sorted(glob.glob(str(Path("metadata") / "manifest_*.json")))  # run from the repo root
DEFAULT_IMAGES_DIR = IMAGES_DIR


# ──────────────────────────────────────────────────────────────
# PROMPT / PARSING
# ──────────────────────────────────────────────────────────────

def load_user_prompt(path):
    """proposer_prompt.txt leads with a literal "<image>" placeholder line; drop it."""
    text = Path(path).read_text()
    text = re.sub(r"^\s*<image>\s*\n+", "", text)
    return text.strip()


def prompt_sha(prompt):
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:12]


def parse_json_response(text):
    """Extract the JSON object from a model answer; None unless it has css_score in 1-4."""
    if not text:
        return None
    text = re.sub(r"```(?:json)?\s*", "", text).strip()
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None
    try:
        parsed = json.loads(match.group())
    except json.JSONDecodeError:
        return None
    try:
        score = int(parsed.get("css_score"))
    except (TypeError, ValueError):
        return None
    if score not in (1, 2, 3, 4):
        return None
    parsed["css_score"] = score
    return parsed


def load_image(path, max_side):
    """RGB PIL image, downscaled (aspect kept) so its longer side is <= max_side."""
    img = Image.open(path)
    img.load()
    img = img.convert("RGB")
    if max_side and max(img.size) > max_side:
        img.thumbnail((max_side, max_side), Image.LANCZOS)
    return img


# ──────────────────────────────────────────────────────────────
# GENERIC TRANSFORMERS BACKEND
# ──────────────────────────────────────────────────────────────

def load_hf_model(model_id):
    """Any image-text-to-text model transformers supports natively (Qwen3-VL, InternVL3-hf, Pixtral...)."""
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor

    model = AutoModelForImageTextToText.from_pretrained(
        model_id, dtype=torch.bfloat16, device_map="auto",
    ).eval()
    processor = AutoProcessor.from_pretrained(model_id)
    processor.tokenizer.padding_side = "left"  # required for batched generation
    if processor.tokenizer.pad_token is None:  # e.g. Pixtral ships without one
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
    return model, processor


def hf_generate(model, processor, images, prompt, max_new_tokens):
    """Greedy-decode one answer per image, as a batch."""
    import torch

    conversations = [
        [{"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": prompt}]}]
        for img in images
    ]
    inputs = processor.apply_chat_template(
        conversations, add_generation_prompt=True, tokenize=True,
        return_dict=True, return_tensors="pt", processor_kwargs={"padding": True},
    ).to(model.device, dtype=torch.bfloat16)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False,
                             pad_token_id=processor.tokenizer.pad_token_id)
    return processor.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)


# ──────────────────────────────────────────────────────────────
# MANIFEST MERGE
# ──────────────────────────────────────────────────────────────

def merge_into_manifests(manifest_paths, proposer, answers):
    """Write css_proposer_responses[proposer] for every image in `answers` (image_id -> parsed)."""
    def apply(entry):
        parsed = answers.get(entry["image_id"])
        if parsed is None:
            return False
        responses = entry.get("css_proposer_responses")
        if not isinstance(responses, dict):
            responses = {}
        if responses.get(proposer) == parsed:
            return False
        responses[proposer] = parsed
        entry["css_proposer_responses"] = responses
        return True

    total = 0
    for path in manifest_paths:
        manifest = ManifestWriter(path)
        changed = manifest.update(apply)
        if changed:
            manifest.export_json()
        manifest.close()
        total += changed
    return total


# ──────────────────────────────────────────────────────────────
# RUN LOOP
# ──────────────────────────────────────────────────────────────

def build_parser(description, proposer, default_gpu, default_model, default_batch_size=8):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--manifest", type=str, nargs="+", default=DEFAULT_MANIFESTS,
                        help="Manifest(s) to score and update (default: every metadata/manifest_*.json)")
    parser.add_argument("--images_dir", type=str, default=DEFAULT_IMAGES_DIR)
    parser.add_argument("--output", type=str, default=str(Path("metadata") / "css_scores" / f"{proposer}.json"),
                        help="Per-proposer results file (a .jsonl journal is kept next to it)")
    parser.add_argument("--prompt_file", type=str, default=str(PROMPT_PATH))
    parser.add_argument("--log", type=str, default=None, help="Log file (default: <output> with a .log suffix)")
    parser.add_argument("--gpu", type=str, default=default_gpu, help="CUDA_VISIBLE_DEVICES, e.g. 0 or 0,1")
    parser.add_argument("--model", type=str, default=default_model, help="Hugging Face model id or local path")
    parser.add_argument("--batch_size", type=int, default=default_batch_size)
    parser.add_argument("--num_shards", type=int, default=1,
                        help="Split the images into N shards, to run N copies of this proposer on different GPUs")
    parser.add_argument("--shard", type=int, default=0, help="Which shard (0..num_shards-1) this process scores")
    parser.add_argument("--max_side", type=int, default=1536,
                        help="Images are downscaled so their longer side is at most this (same for every proposer)")
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--merge_every", type=int, default=1000,
                        help="Write parsed answers into the manifests every N answers (and at the end)")
    parser.add_argument("--limit", type=int, default=None, help="Score at most N images this run (for testing)")
    parser.add_argument("--rescore", action="store_true", help="Ignore previous answers and score everything again")
    parser.add_argument("--resume", action="store_true", help=argparse.SUPPRESS)  # resuming is now the default
    parser.add_argument("--cpu_threads", type=int, default=8,
                        help="CPU threads per process (the models run on the GPU)")
    parser.add_argument("--seed", type=int, default=9999)
    return parser


def limit_cpu_threads(n):
    """
    Cap the CPU threads this process may start. Must run before torch or
    transformers are imported. The models run on the GPU, so a few CPU threads
    are plenty — and each process otherwise starts one per core (~325 threads
    on a 128-core machine), so several processes side by side exceed the
    container's thread limit and crash ("ThreadPoolBuildError").
    """
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "RAYON_NUM_THREADS"):
        os.environ[var] = str(n)


def is_fatal(e):
    """Errors that must stop the run. Anything else (incl. non-Exception panics raised
    by Rust extensions such as the tokenizers library) is recorded and skipped."""
    return isinstance(e, (KeyboardInterrupt, SystemExit, MemoryError))


def _raise_keyboard_interrupt(*_):
    raise KeyboardInterrupt


def run(args, proposer, load_model, generate):
    """
    load_model(model_id) -> (model, processor);
    generate(model, processor, images, prompt, max_new_tokens) -> list of answers.
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu  # before torch touches CUDA

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not 0 <= args.shard < args.num_shards:
        sys.exit(f"--shard must be in 0..{args.num_shards - 1}")
    shard_tag = f".shard{args.shard}of{args.num_shards}" if args.num_shards > 1 else ""
    log_path = Path(args.log) if args.log else output_path.with_suffix(shard_tag + ".log")
    log_file = open(log_path, "a", buffering=1)
    handler = logging.StreamHandler(log_file)
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logging.getLogger().setLevel(logging.INFO)
    logging.getLogger().addHandler(handler)
    for noisy in ("httpx", "httpcore", "huggingface_hub", "urllib3", "filelock"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    print(f"[{proposer}] GPU(s) {args.gpu} — progress is logged to {log_path}", file=sys.stderr)

    # Make `kill` behave like Ctrl-C, so the final merge into the manifests still runs.
    signal.signal(signal.SIGTERM, _raise_keyboard_interrupt)

    limit_cpu_threads(args.cpu_threads)
    import numpy as np
    import torch
    torch.set_num_threads(args.cpu_threads)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    prompt = load_user_prompt(args.prompt_file)
    sha = prompt_sha(prompt)
    logging.info(f"[{proposer}] model={args.model} gpu={args.gpu} prompt={args.prompt_file} (sha {sha}) "
                 f"batch_size={args.batch_size} max_side={args.max_side}")

    if not args.manifest:
        sys.exit("No manifests found — pass --manifest")
    entries = []
    for path in args.manifest:
        with open(path) as f:
            entries.extend(json.load(f))
    logging.info(f"Loaded {len(entries)} entries from {len(args.manifest)} manifest(s): {args.manifest}")

    results = ManifestWriter(output_path)
    answers = {}  # image_id -> parsed, current prompt only
    if not args.rescore:
        for r in results.entries:
            if r.get("prompt_sha") == sha and r.get("parsed") is not None:
                answers[r["image_id"]] = r["parsed"]
    todo = [e for e in entries if e["image_id"] not in answers]
    if args.num_shards > 1:  # stable split: an image always lands in the same shard
        todo = [e for e in todo
                if int(hashlib.md5(e["image_id"].encode()).hexdigest(), 16) % args.num_shards == args.shard]
    if args.limit is not None:
        todo = todo[:args.limit]
    logging.info(f"{len(answers)} already scored with this prompt; {len(todo)} to score now"
                 + (f" (shard {args.shard} of {args.num_shards})" if args.num_shards > 1 else ""))
    if not todo:
        merge_into_manifests(args.manifest, proposer, answers)
        results.close()
        print(f"[{proposer}] nothing to score — {len(answers)} images already scored with this prompt", file=sys.stderr)
        return

    model, processor = load_model(args.model)
    logging.info("Model loaded")

    images_dir = Path(args.images_dir)
    since_merge = 0
    n_ok = n_fail = 0

    def record(entry, raw, error=None):
        nonlocal since_merge, n_ok, n_fail
        parsed = parse_json_response(raw)
        results.add({
            "image_id": entry["image_id"], "proposer": proposer, "model": args.model, "prompt_sha": sha,
            "raw_response": raw, "parsed": parsed, "error": error,
        })
        if parsed is not None:
            answers[entry["image_id"]] = parsed
            since_merge += 1
            n_ok += 1
        else:
            n_fail += 1
            logging.warning(f"{entry['image_id']} → {'ERROR ' + error if error else 'unparseable'} | "
                            f"raw: {(raw or '')[:120]!r}")

    def merge():
        nonlocal since_merge
        changed = merge_into_manifests(args.manifest, proposer, answers)
        logging.info(f"Merged into manifests: {changed} entries updated")
        since_merge = 0

    progress = tqdm(total=len(todo), desc=proposer, file=log_file, mininterval=10.0)
    try:
        for start in range(0, len(todo), args.batch_size):
            batch, images = [], []
            for entry in todo[start:start + args.batch_size]:
                try:
                    images.append(load_image(images_dir / entry["filename"], args.max_side))
                    batch.append(entry)
                except BaseException as e:
                    if is_fatal(e):
                        raise
                    record(entry, None, f"image load failed: {type(e).__name__}: {e}")
            if batch:
                try:
                    raws = generate(model, processor, images, prompt, args.max_new_tokens)
                    for entry, raw in zip(batch, raws):
                        record(entry, raw)
                except BaseException as e:
                    if is_fatal(e):
                        raise
                    # e.g. CUDA OOM on a batch of large images: retry one at a time
                    logging.warning(f"Batch failed ({type(e).__name__}: {e}); retrying one by one")
                    torch.cuda.empty_cache()
                    for entry, img in zip(batch, images):
                        try:
                            record(entry, generate(model, processor, [img], prompt, args.max_new_tokens)[0])
                        except BaseException as e1:
                            if is_fatal(e1):
                                raise
                            torch.cuda.empty_cache()
                            record(entry, None, f"{type(e1).__name__}: {e1}")
            progress.update(len(todo[start:start + args.batch_size]))
            progress.set_postfix(ok=n_ok, failed=n_fail, refresh=False)
            if since_merge >= args.merge_every:
                merge()
    except KeyboardInterrupt:
        logging.warning("Interrupted — merging what was scored so far")
    finally:
        progress.close()
        merge()
        results.export_json()
        results.close()

    msg = f"[{proposer}] done — {n_ok} scored, {n_fail} failed this run; {len(answers)} total with this prompt"
    logging.info(msg)
    print(msg, file=sys.stderr)
