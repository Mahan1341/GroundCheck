"""Reviewed FactCG input adapter; explicit FP32, bounded runs, reusable exact-pair cache."""

import argparse
import hashlib
import json
import time
from dataclasses import asdict

import numpy as np

from grounded_hallucination_lab.research.protocol import (
    ROOT,
    digest,
    read_json,
    read_rows,
    write_json,
    write_rows,
)
from grounded_hallucination_lab.research.segments import clean_segments
from grounded_hallucination_lab.research.verifiers import (
    memory_report,
    punkt_spans,
    setup_runtime,
)

BASE = ROOT / "models/discovery_loop_20261026"
PRED = ROOT / "predictions/discovery_loop_20261026"
REPORT = ROOT / "reports/discovery_loop_20261026"
PROMPT = (
    '{text_a}\n\nChoose your answer: based on the paragraph above can we conclude that "{text_b}"?'
    "\n\nOPTIONS:\n- Yes\n- No\nI think the answer is "
)


def pair_key(document, claim):
    return hashlib.sha256(json.dumps([document, claim], ensure_ascii=False).encode()).hexdigest()


def chunks(text):
    """Author 550 NLTK word-token sentence packing, with original provenance."""
    from nltk import word_tokenize

    groups, current, count = [], [], 0
    for span in punkt_spans(text):
        size = len(word_tokenize(span.text))
        if current and count + size > 550:
            groups.append(current)
            current, count = [], 0
        current.append(span)
        count += size
    if current:
        groups.append(current)
    return [
        {"text": "\n".join(s.text for s in group), "pieces": [[s.start, s.end] for s in group]}
        for group in groups
    ] or [{"text": "", "pieces": []}]


def prepare(views, offset_rows):
    pairs, layouts = {}, []
    for v, offset_row in zip(views, offset_rows, strict=True):
        assert v["id"] == offset_row["id"]
        contexts = chunks(v["evidence"])
        segments, dropped = clean_segments(v["response"])
        row = {
            "id": v["id"],
            "source_id": v["source_id"],
            "offsets": offset_row["offsets"],
            "dropped": dropped,
            "segments": [],
        }
        for segment in segments:
            keys = []
            for c in contexts:
                key = pair_key(c["text"], segment.text)
                pairs.setdefault(key, {"document": c["text"], "claim": segment.text})
                keys.append({"key": key, "pieces": c["pieces"]})
            row["segments"].append({**asdict(segment), "evidence": keys})
        layouts.append(row)
    return pairs, layouts


def project(layouts, cache):
    result = []
    for row in layouts:
        segments = []
        for segment in row["segments"]:
            outputs = [cache[e["key"]] for e in segment["evidence"]]
            segments.append({**segment, "risk": 1 - max(p["support"] for p in outputs)})
        probabilities = [
            max((s["risk"] for s in segments if a < s["end"] and b > s["start"]), default=0)
            for a, b in row["offsets"]
        ]
        result.append(
            {
                **row,
                "segments": segments,
                "token_probabilities": probabilities,
                "score": max((s["risk"] for s in segments), default=0),
            }
        )
    return result


def read_cache(path):
    if not path.exists():
        return {}
    rows = read_rows(path)
    assert len({r["key"] for r in rows}) == len(rows)
    return {r["key"]: r for r in rows}


def run(name, pairs_path, *, minutes, pilot=False):
    """No label access. Append-only cache, exclusive manifests, exact resume by cache keys."""
    torch = setup_runtime()
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    source = {
        str(p.relative_to(ROOT)): digest(p)
        for p in [
            ROOT / "scripts/discovery_verifier.py",
            pairs_path,
            ROOT / "docs/DISCOVERY_CYCLE2_PREREG.md",
        ]
    }
    write_json(
        BASE / f"{name}_run_freeze.json",
        {
            "files": source,
            "minutes_cap": minutes,
            "precision": "fp32",
            "batch": 1,
            "tf32": False,
            "pilot": pilot,
        },
    )
    pairs = read_json(pairs_path)
    tokenizer = AutoTokenizer.from_pretrained(BASE / "factcg", local_files_only=True)
    tokenizer.pad_token = tokenizer.eos_token
    model = (
        AutoModelForSequenceClassification.from_pretrained(
            BASE / "factcg",
            local_files_only=True,
            trust_remote_code=False,
            torch_dtype=torch.float32,
        )
        .to("cuda")
        .eval()
    )
    assert model.config.num_labels == 2
    cache_path = PRED / "factcg_pair_cache.jsonl"
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache = read_cache(cache_path)
    todo = [k for k in pairs if k not in cache]
    lengths = {}
    for k in todo:
        p = pairs[k]
        lengths[k] = len(
            tokenizer(PROMPT.format(text_a=p["document"], text_b=p["claim"]), truncation=False)[
                "input_ids"
            ]
        )
    if pilot and todo:
        longest = max(todo, key=lengths.get)
        todo = list(dict.fromkeys([*todo[:63], longest]))
    start = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    times, completed, failure = [], 0, None
    try:
        with cache_path.open("a", encoding="utf-8", newline="\n") as stream:
            for key in todo:
                if time.perf_counter() - start >= minutes * 60:
                    failure = "preregistered_time_cap"
                    break
                p = pairs[key]
                text = PROMPT.format(text_a=p["document"], text_b=p["claim"])
                inputs = tokenizer(text, truncation=True, max_length=2048, return_tensors="pt").to(
                    "cuda"
                )
                torch.cuda.synchronize()
                t = time.perf_counter()
                with torch.inference_mode():
                    prob = model(**inputs).logits.float().softmax(-1)[0].cpu().tolist()
                torch.cuda.synchronize()
                elapsed = time.perf_counter() - t
                times.append(elapsed)
                used = int(inputs["attention_mask"].sum())
                value = {
                    "key": key,
                    "support": prob[1],
                    "probabilities": prob,
                    "input_tokens": lengths[key],
                    "used_tokens": used,
                    "truncated_tokens": max(0, lengths[key] - used),
                    "forward_seconds": elapsed,
                }
                stream.write(json.dumps(value) + "\n")
                stream.flush()
                completed += 1
                if completed % 100 == 0:
                    print(f"{name}: {completed}/{len(todo)}", flush=True)
                if torch.cuda.max_memory_allocated() > 10 * 1024**3:
                    failure = "preregistered_10GiB_memory_guard"
                    break
    except (torch.cuda.OutOfMemoryError, RuntimeError) as exc:
        failure = f"{type(exc).__name__}: {exc}"
    seconds = time.perf_counter() - start
    report = {
        "name": name,
        "completed_new_pairs": completed,
        "pending_before": len(lengths),
        "seconds": seconds,
        "gpu_minutes_conservative": seconds / 60,
        "failure": failure,
        "max_input_tokens": max(lengths.values(), default=0),
        "forward_p50_s": float(np.median(times)) if times else None,
        "forward_p95_s": float(np.percentile(times, 95)) if times else None,
        "estimated_remaining_seconds": (len(lengths) - completed) * seconds / max(1, completed),
        **memory_report(),
    }
    write_json(REPORT / f"{name}_resources.json", report)
    for filename, sha in source.items():
        assert digest(ROOT / filename) == sha
    print(json.dumps(report), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["prepare", "pilot", "run", "project"])
    parser.add_argument("--name", default="cycle2")
    parser.add_argument("--minutes", type=float, default=37)
    args = parser.parse_args()
    pairs_path = BASE / f"{args.name}_pairs.json"
    if args.mode == "prepare":
        from research_r10 import records

        pairs = {}
        files = {}
        for label, record_path in {
            "natural": ROOT / "models/phase2b_r10/natural_records.jsonl",
            "r1": ROOT / "models/phase2b_r10/r1_records.jsonl",
            "r6": ROOT / "predictions/phase2b_r6/transfer_records.jsonl",
        }.items():
            rr = records(record_path)
            old_path = ROOT / "predictions/phase2b_r10" / f"B7_{label}.jsonl"
            old = read_rows(old_path)
            views = [
                {k: getattr(r, k) for k in ("id", "source_id", "response", "evidence")} for r in rr
            ]
            pp, layouts = prepare(views, old)
            pairs.update(pp)
            write_json(BASE / f"{args.name}_{label}_layout.json", layouts)
            files[str(record_path.relative_to(ROOT))] = digest(record_path)
            files[str(old_path.relative_to(ROOT))] = digest(old_path)
        write_json(pairs_path, pairs)
        write_json(
            BASE / f"{args.name}_data_freeze.json",
            {
                "input_files": files,
                "unique_pairs": len(pairs),
                "pairs_sha256": digest(pairs_path),
                "source": digest(ROOT / "scripts/discovery_verifier.py"),
                "prereg": digest(ROOT / "docs/DISCOVERY_CYCLE2_PREREG.md"),
            },
        )
        print({"unique_pairs": len(pairs)}, flush=True)
    elif args.mode in ("pilot", "run"):
        run(
            args.name + "_" + args.mode,
            pairs_path,
            minutes=args.minutes,
            pilot=args.mode == "pilot",
        )
    else:
        cache = read_cache(PRED / "factcg_pair_cache.jsonl")
        for label in ("natural", "r1", "r6"):
            layouts = read_json(BASE / f"{args.name}_{label}_layout.json")
            rows = project(layouts, cache)
            write_rows(PRED / f"{args.name}_FactCG_{label}.jsonl", rows)


if __name__ == "__main__":
    main()
