"""
CSS aggregator — Qwen2.5-7B-Instruct (text-only).

Reads the three proposers' answers from each manifest entry
(css_proposer_responses, filled by proposer_qwen.py / proposer_internvl.py /
proposer_pixtral.py), asks a text-only LLM to reconcile them with the prompt
in aggregator_prompt.txt (same scale and tests as proposer_prompt.txt), and writes the
result back into the manifests:

  css_aggregated_response   the aggregator's parsed JSON answer, plus the
                            fiction-cap fields below
  css_score                 its final_css_score (1-4), after the fiction cap

Fiction cap: drawings, digital art, film stills, games, costumes and other
staged or fictional content never score above CSS-2 (rule in proposer_prompt.txt).
The proposers do not always obey it, so it is enforced in code: when at least
--fiction_min_votes proposers (default 2 of 3) put "staged_or_fictional" in
their flags, css_score is capped at 2. A single flag is not enough because
Qwen3-VL adds it to many real photos (surgeries, protests, war scenes). The
aggregator's own answer is kept; css_aggregated_response gets
fiction_votes, fiction_cap_applied and uncapped_css_score.

Guards against the aggregator overriding the proposers (applied before the
fiction cap; recorded in css_aggregated_response.guards):
  majority ceiling  when >= 2 proposers agree, the final score may go BELOW
                    their score (context, contradicted evidence) but not
                    above it; when all differ it stays within their range.
                    The 7B aggregator otherwise raised ~10K majority votes to
                    a single analyst's higher score. --no_majority_ceiling
                    turns it off.
  context cap       the final score never exceeds the context_cap the
                    aggregator itself declared (artwork 2, media cover 3).
                    --no_context_cap turns it off.

Raw answers are kept in metadata/css_scores/aggregator.json(l), an
append-only journal like the proposers' (one line per image), so runs are
cheap to resume.

  - By default an image is aggregated only once all of --proposers have
    answered it (--min_proposers to relax), so it can run while proposers
    are still going: images become eligible as their answers come in.
  - Resuming is automatic: an image is skipped when it was already
    aggregated from exactly the same proposer answers and the same
    aggregator prompt (inputs_sha). If a proposer re-scores an image, it is
    re-aggregated. --reaggregate ignores previous results.
  - Manifest updates run under the manifest lock on a fresh read (like the
    proposers'), every --merge_every answers and at the end, also on
    Ctrl-C / kill. A result is only written if the entry's proposer answers
    still match the ones it was computed from.

To use several GPUs, run N copies with --num_shards N --shard 0..N-1 (one GPU
each); they share the results journal and merge safely.

All status messages and the tqdm bar go to a log file (--log, default
<output>.log, or <output>.shard<i>of<N>.log), so a run can be left detached.

  python -m greve.annotate.aggregator --gpu 3
"""

import argparse
import hashlib
from collections import Counter
import json
import logging
import os
import random
import re
import signal
import sys
from pathlib import Path

from tqdm import tqdm

from greve.common import ManifestWriter
from greve.annotate.proposer_common import DEFAULT_MANIFESTS, HERE, is_fatal, limit_cpu_threads, prompt_sha

MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
PROMPT_PATH = HERE / "prompts" / "aggregator_prompt.txt"
PROPOSERS = ["qwen3-vl-8b", "internvl3-8b", "pixtral-12b"]


# ──────────────────────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────────────────────

def parse_aggregator_response(text):
    """Extract the JSON object; None unless final_css_score is in 1-4."""
    if not text:
        return None
    text = re.sub(r"```(?:json)?\s*", "", text).strip()
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None
    blob = match.group()
    try:
        parsed = json.loads(blob)
    except json.JSONDecodeError:
        try:  # \' is not a valid JSON escape, but the model copies it from quoted image text
            parsed = json.loads(blob.replace("\\'", "'"))
        except json.JSONDecodeError:
            parsed = _salvage_fields(blob)  # e.g. unescaped "..." inside a string
    try:
        score = int(parsed.get("final_css_score"))
    except (AttributeError, TypeError, ValueError):
        return None
    if score not in (1, 2, 3, 4):
        return None
    parsed["final_css_score"] = score
    return parsed


def _salvage_fields(blob):
    """Last resort for malformed JSON: read the fields one by one (strings up to the next key)."""
    keys = ["image_context", "context_cap", "decisive_evidence", "evidence_check", "consensus_level",
            "agreement_analysis", "final_test_fired", "final_css_score", "final_justification"]
    found = {k: m.start() for k in keys if (m := re.search(rf'"{k}"\s*:', blob))}
    if "final_css_score" not in found:
        return None
    order = sorted(found, key=found.get)
    out = {"parse_salvaged": True}
    for k, nxt in zip(order, order[1:] + [None]):
        start = re.search(rf'"{k}"\s*:', blob).end()
        end = found[nxt] if nxt else blob.rfind("}")
        value = blob[start:end].strip().rstrip(",").strip()
        if value.startswith('"') and value.endswith('"'):
            out[k] = value[1:-1]
        else:
            try:
                out[k] = json.loads(value)
            except json.JSONDecodeError:
                out[k] = value
    return out


def proposer_answers(entry, proposers):
    responses = entry.get("css_proposer_responses")
    if not isinstance(responses, dict):
        return {}
    return {k: responses[k] for k in proposers if isinstance(responses.get(k), dict)}


def inputs_sha(answers, agg_prompt_sha):
    blob = json.dumps(answers, sort_keys=True, ensure_ascii=False) + agg_prompt_sha
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def format_assessments(answers):
    def field(r, k):
        v = r.get(k)
        if isinstance(v, list):
            v = ", ".join(str(x) for x in v) or "none"
        return "N/A" if v is None or v == "" else v

    blocks = []
    for i, (name, r) in enumerate(answers.items(), 1):
        blocks.append(
            f"Analyst {i} ({name}):\n"
            f"  CSS score: {field(r, 'css_score')}\n"
            f"  Test fired: {field(r, 'test_fired')}\n"
            f"  Evidence for test: {field(r, 'evidence_for_test')}\n"
            f"  Public figure or event: {field(r, 'public_figure_or_event')}\n"
            f"  Confidence: {field(r, 'confidence')}\n"
            f"  Flags: {field(r, 'flags')}\n"
            f"  Scene: {field(r, 'scene_description')}\n"
            f"  Key factors: {field(r, 'key_sensitivity_factors')}\n"
            f"  Justification: {field(r, 'justification')}"
        )
    return "\n\n".join(blocks)


def load_model(model_id):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16, device_map="auto").eval()
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.padding_side = "left"  # required for batched generation
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def generate(model, tokenizer, prompts, max_new_tokens):
    import torch

    texts = [tokenizer.apply_chat_template([{"role": "user", "content": p}], tokenize=False,
                                           add_generation_prompt=True) for p in prompts]
    inputs = tokenizer(texts, return_tensors="pt", padding=True).to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False,
                             pad_token_id=tokenizer.pad_token_id)
    return tokenizer.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)


FICTION_FLAG = "staged_or_fictional"
FICTION_CAP = 2


def apply_guards(parsed, answers, majority_ceiling=True, context_cap=True):
    """(score, guards): the aggregator's final score after the majority ceiling and its own context cap."""
    score, guards = parsed["final_css_score"], []
    votes = [r.get("css_score") for r in answers.values() if r.get("css_score") in (1, 2, 3, 4)]
    if majority_ceiling and votes:
        top, n = Counter(votes).most_common(1)[0]
        ceiling = top if n >= 2 else max(votes)
        floor = min(votes) if n < 2 else 1
        if score > ceiling:
            guards.append(f"majority_ceiling {score}->{ceiling}")
            score = ceiling
        elif score < floor:
            guards.append(f"range_floor {score}->{floor}")
            score = floor
    cap = parsed.get("context_cap")
    if context_cap and isinstance(cap, int) and not isinstance(cap, bool) and 1 <= cap < score:
        guards.append(f"context_cap {score}->{cap}")
        score = cap
    return score, guards


def apply_fiction_cap(parsed, answers, min_votes, majority_ceiling=True, context_cap=True):
    """Copy of parsed with the guard and fiction-cap fields; css_score = guarded score, capped at 2 if >= min_votes proposers flag fiction."""
    votes = sum(FICTION_FLAG in (r.get("flags") or []) for r in answers.values())
    score, guards = apply_guards(parsed, answers, majority_ceiling, context_cap)
    capped = min_votes > 0 and votes >= min_votes and score > FICTION_CAP
    return {**parsed, "guards": guards, "guarded_css_score": score, "fiction_votes": votes,
            "fiction_cap_applied": capped, "uncapped_css_score": parsed["final_css_score"],
            "css_score": FICTION_CAP if capped else score}


def merge_into_manifests(manifest_paths, proposers, final, agg_sha, fiction_min_votes=2,
                         majority_ceiling=True, context_cap=True):
    """final: image_id -> (inputs_sha, parsed). Writes css_aggregated_response + css_score (fiction-capped)."""
    def apply(entry):
        item = final.get(entry["image_id"])
        if item is None:
            return False
        sha, parsed = item
        # Only if the entry still has the proposer answers this was computed from.
        answers = proposer_answers(entry, proposers)
        if inputs_sha(answers, agg_sha) != sha:
            return False
        capped = apply_fiction_cap(parsed, answers, fiction_min_votes, majority_ceiling, context_cap)
        score = capped.pop("css_score")
        if entry.get("css_aggregated_response") == capped and entry.get("css_score") == score:
            return False
        entry["css_aggregated_response"] = capped
        entry["css_score"] = score
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


def _raise_keyboard_interrupt(*_):
    raise KeyboardInterrupt


# ──────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="CSS aggregator")
    parser.add_argument("--manifest", type=str, nargs="+", default=DEFAULT_MANIFESTS,
                        help="Manifest(s) to read proposer answers from and update (default: every metadata/manifest_*.json)")
    parser.add_argument("--output", type=str, default=str(Path("metadata") / "css_scores" / "aggregator.json"))
    parser.add_argument("--prompt_file", type=str, default=str(PROMPT_PATH))
    parser.add_argument("--log", type=str, default=None, help="Log file (default: <output> with a .log suffix)")
    parser.add_argument("--gpu", type=str, default="3", help="CUDA_VISIBLE_DEVICES")
    parser.add_argument("--model", type=str, default=MODEL_NAME)
    parser.add_argument("--proposers", type=str, nargs="+", default=PROPOSERS,
                        help="Proposer keys in css_proposer_responses to aggregate")
    parser.add_argument("--min_proposers", type=int, default=None,
                        help="Aggregate images with at least this many proposer answers (default: all of --proposers)")
    parser.add_argument("--fiction_min_votes", type=int, default=2,
                        help="Cap css_score at 2 when at least this many proposers flag staged_or_fictional (0 = no cap)")
    parser.add_argument("--no_majority_ceiling", action="store_true",
                        help="Let the aggregator raise the score above a 2-of-3 proposer majority")
    parser.add_argument("--no_context_cap", action="store_true",
                        help="Don't enforce the context_cap the aggregator declares")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_shards", type=int, default=1,
                        help="Split the images into N shards, to run N copies of the aggregator on different GPUs")
    parser.add_argument("--shard", type=int, default=0, help="Which shard (0..num_shards-1) this process aggregates")
    parser.add_argument("--max_new_tokens", type=int, default=768)
    parser.add_argument("--merge_every", type=int, default=1000)
    parser.add_argument("--limit", type=int, default=None, help="Aggregate at most N images this run (for testing)")
    parser.add_argument("--reaggregate", action="store_true", help="Ignore previous results and aggregate everything again")
    parser.add_argument("--resume", action="store_true", help=argparse.SUPPRESS)  # resuming is now the default
    parser.add_argument("--cpu_threads", type=int, default=8,
                        help="CPU threads per process (the model runs on the GPU)")
    parser.add_argument("--seed", type=int, default=9999)
    args = parser.parse_args()
    min_proposers = args.min_proposers or len(args.proposers)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
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
    print(f"[aggregator] GPU(s) {args.gpu} — progress is logged to {log_path}", file=sys.stderr)
    signal.signal(signal.SIGTERM, _raise_keyboard_interrupt)

    limit_cpu_threads(args.cpu_threads)
    import numpy as np
    import torch
    torch.set_num_threads(args.cpu_threads)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    template = Path(args.prompt_file).read_text()
    agg_sha = prompt_sha(template)
    logging.info(f"[aggregator] model={args.model} gpu={args.gpu} prompt={args.prompt_file} (sha {agg_sha}) "
                 f"proposers={args.proposers} min_proposers={min_proposers} fiction_min_votes={args.fiction_min_votes} "
                 f"majority_ceiling={not args.no_majority_ceiling} context_cap={not args.no_context_cap}")

    entries = []
    for path in args.manifest:
        with open(path) as f:
            entries.extend(json.load(f))

    results = ManifestWriter(output_path)
    final = {}  # image_id -> (inputs_sha, parsed)
    done = set()
    if not args.reaggregate:
        for r in results.entries:
            if r.get("parsed") is not None:
                done.add((r["image_id"], r.get("inputs_sha")))

    todo, waiting = [], 0
    for e in entries:
        answers = proposer_answers(e, args.proposers)
        if len(answers) < min_proposers:
            waiting += 1
            continue
        sha = inputs_sha(answers, agg_sha)
        if (e["image_id"], sha) in done:
            continue
        todo.append((e, answers, sha))
    if args.num_shards > 1:  # stable split, same rule as the proposers
        todo = [t for t in todo
                if int(hashlib.md5(t[0]["image_id"].encode()).hexdigest(), 16) % args.num_shards == args.shard]
    if args.limit is not None:
        todo = todo[:args.limit]
    logging.info(f"Loaded {len(entries)} entries from {len(args.manifest)} manifest(s); "
                 f"{waiting} still waiting for proposer answers; {len(todo)} to aggregate now")

    model, tokenizer = load_model(args.model)
    logging.info("Model loaded")

    n_ok = n_fail = since_merge = 0

    def record(entry, answers, sha, raw, error=None):
        nonlocal n_ok, n_fail, since_merge
        parsed = parse_aggregator_response(raw)
        results.add({
            "image_id": entry["image_id"], "model": args.model, "prompt_sha": agg_sha, "inputs_sha": sha,
            "proposer_scores": {k: v.get("css_score") for k, v in answers.items()},
            "raw_response": raw, "parsed": parsed, "error": error,
        })
        if parsed is not None:
            final[entry["image_id"]] = (sha, parsed)
            n_ok += 1
            since_merge += 1
        else:
            n_fail += 1
            logging.warning(f"{entry['image_id']} → {'ERROR ' + error if error else 'unparseable'} | "
                            f"raw: {(raw or '')[:120]!r}")

    def merge():
        nonlocal since_merge
        changed = merge_into_manifests(args.manifest, args.proposers, final, agg_sha, args.fiction_min_votes,
                                       not args.no_majority_ceiling, not args.no_context_cap)
        logging.info(f"Merged into manifests: {changed} entries updated")
        since_merge = 0

    progress = tqdm(total=len(todo), desc="aggregator", file=log_file, mininterval=10.0)
    try:
        for start in range(0, len(todo), args.batch_size):
            batch = todo[start:start + args.batch_size]
            prompts = [template.format(n_analysts=len(a), assessments=format_assessments(a)) for _, a, _ in batch]
            try:
                raws = generate(model, tokenizer, prompts, args.max_new_tokens)
                for (entry, answers, sha), raw in zip(batch, raws):
                    record(entry, answers, sha, raw)
            except BaseException as e:
                if is_fatal(e):
                    raise
                logging.warning(f"Batch failed ({type(e).__name__}: {e}); retrying one by one")
                torch.cuda.empty_cache()
                for (entry, answers, sha), p in zip(batch, prompts):
                    try:
                        record(entry, answers, sha, generate(model, tokenizer, [p], args.max_new_tokens)[0])
                    except BaseException as e1:
                        if is_fatal(e1):
                            raise
                        torch.cuda.empty_cache()
                        record(entry, answers, sha, None, f"{type(e1).__name__}: {e1}")
            progress.update(len(batch))
            progress.set_postfix(ok=n_ok, failed=n_fail, refresh=False)
            if since_merge >= args.merge_every:
                merge()
    except KeyboardInterrupt:
        logging.warning("Interrupted — merging what was aggregated so far")
    finally:
        progress.close()
        merge()
        results.export_json()
        results.close()

    msg = f"[aggregator] done — {n_ok} aggregated, {n_fail} failed this run; {waiting} waiting for proposers"
    logging.info(msg)
    print(msg, file=sys.stderr)


if __name__ == "__main__":
    main()
