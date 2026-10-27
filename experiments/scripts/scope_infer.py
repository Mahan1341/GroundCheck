"""Only missing allowlisted pairs, immutable historical cache, bounded FP32 inference."""

import argparse
import json
import time

import numpy as np
from discovery_verifier import PROMPT
from scope_inputs import BASE, OLD, PRED, REPORT, scoped_cache

from grounded_hallucination_lab.research.protocol import ROOT, digest, read_json, write_json
from grounded_hallucination_lab.research.verifiers import memory_report, setup_runtime


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["pilot", "run"])
    args = parser.parse_args()
    freeze = read_json(BASE / "freeze.json")
    for name, sha in freeze["source"].items():
        assert digest(ROOT / name) == sha
    assert digest(BASE / "pairs.json") == freeze["pairs_sha256"]
    assert digest(OLD / "factcg/model.safetensors") == (
        "379a343d1ca25c9dfe24ae149f76cc958ea4f09c00dae88bd49bbbf58c749d5e"
    )
    started = time.perf_counter()
    torch = setup_runtime()
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    pairs = read_json(BASE / "pairs.json")
    cache = scoped_cache(PRED / "pair_cache.jsonl", set(pairs))
    todo = [k for k in pairs if k not in cache]
    tok = AutoTokenizer.from_pretrained(OLD / "factcg", local_files_only=True)
    tok.pad_token = tok.eos_token
    lengths = {
        k: len(
            tok(
                PROMPT.format(text_a=pairs[k]["document"], text_b=pairs[k]["claim"]),
                truncation=False,
            )["input_ids"]
        )
        for k in todo
    }
    assert max(lengths.values(), default=0) <= 2048, "no silent claim truncation"
    if args.mode == "pilot" and todo:
        todo = list(dict.fromkeys([*todo[:32], max(todo, key=lengths.get)]))
    if args.mode == "run":
        pilot = read_json(REPORT / "pilot_resources.json")
        assert pilot["passed"]
    model = (
        AutoModelForSequenceClassification.from_pretrained(
            OLD / "factcg",
            local_files_only=True,
            trust_remote_code=False,
            torch_dtype=torch.float32,
        )
        .to("cuda")
        .eval()
    )
    torch.cuda.reset_peak_memory_stats()
    durations, completed, failure = [], 0, None
    limit = 60 if args.mode == "pilot" else 540
    with (PRED / "pair_cache.jsonl").open("a", encoding="utf-8", newline="\n") as stream:
        for key in todo:
            if time.perf_counter() - started > limit:
                failure = "time_cap"
                break
            p = pairs[key]
            inputs = tok(
                PROMPT.format(text_a=p["document"], text_b=p["claim"]),
                return_tensors="pt",
                truncation=False,
            ).to("cuda")
            torch.cuda.synchronize()
            t = time.perf_counter()
            with torch.inference_mode():
                probs = model(**inputs).logits.float().softmax(-1)[0].cpu().tolist()
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - t
            durations.append(elapsed)
            value = {
                "key": key,
                "support": probs[1],
                "probabilities": probs,
                "input_tokens": lengths[key],
                "used_tokens": lengths[key],
                "truncated_tokens": 0,
                "forward_seconds": elapsed,
            }
            stream.write(json.dumps(value) + "\n")
            stream.flush()
            completed += 1
            if completed % 100 == 0:
                print(f"{args.mode}: {completed}/{len(todo)}", flush=True)
    seconds = time.perf_counter() - started
    resources = memory_report()
    estimate = (len(lengths) - completed) * sum(durations) / max(1, completed)
    write_json(
        REPORT / f"{args.mode}_resources.json",
        {
            "seconds_with_load": seconds,
            "new_pairs": completed,
            "pending_before": len(lengths),
            "failure": failure,
            "max_input_tokens": max(lengths.values(), default=0),
            "estimated_remaining_seconds": estimate,
            "forward_p50": float(np.median(durations)),
            "forward_p95": float(np.percentile(durations, 95)),
            **resources,
            "passed": failure is None
            and estimate < 500
            and resources["cuda_peak_allocated_bytes"] < 10 * 1024**3,
            "protocol": "FP32,batch1,TF32off,no truncation",
            "freeze_sha256": digest(BASE / "freeze.json"),
        },
    )
    print({"new_pairs": completed, "seconds": seconds, "remaining_estimate": estimate})


if __name__ == "__main__":
    main()
