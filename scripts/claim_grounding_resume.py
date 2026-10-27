"""One exact-key continuation after the preserved C2 CUDA failure; no new experiment."""

import json
import time

from claim_grounding_c2 import MODEL, PROMPT
from claim_grounding_c2_bounded import BASE, PRED, REPORT
from scope_analysis import ROOT, read, rows, write

from grounded_hallucination_lab.research.protocol import digest


def main():
    failure = read(REPORT / "c2_run_resources.json")
    assert "CUDA error: unknown error" in failure["failure"]
    destination = REPORT / "c2_resume1_resources.json"
    assert not destination.exists(), "only one continuation permitted"
    for name, sha in read(BASE / "c2_freeze.json")["files"].items():
        assert digest(ROOT / name) == sha, name
    assert digest(MODEL / "model.safetensors") == (
        "379a343d1ca25c9dfe24ae149f76cc958ea4f09c00dae88bd49bbbf58c749d5e")
    pairs = read(BASE / "c2_pairs.json")
    cache_path = PRED / "c2_pair_cache.jsonl"
    existing = rows(cache_path)
    done = {r["key"] for r in existing}
    assert len(done) == len(existing) and done <= set(pairs)
    todo = sorted(set(pairs) - done)
    limit = 840 - failure["seconds_with_load"]
    write(BASE / "c2_resume1_freeze.json", {
        "source_sha256": digest(ROOT / "scripts/claim_grounding_resume.py"),
        "original_freeze_sha256": digest(BASE / "c2_freeze.json"),
        "failed_report_sha256": digest(REPORT / "c2_run_resources.json"),
        "cache_prefix_sha256": digest(cache_path), "cached_pairs": len(done),
        "pending_keys": todo, "remaining_run_seconds": limit,
        "change": "new CUDA context; same FP32/batch1/TF32off/weights/prompt; no duplicate pairs"})
    started = time.perf_counter()
    from grounded_hallucination_lab.research.verifiers import memory_report, setup_runtime

    torch = setup_runtime()
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    completed, error = 0, None
    try:
        tok = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
        tok.pad_token = tok.eos_token
        model = AutoModelForSequenceClassification.from_pretrained(
            MODEL, local_files_only=True, trust_remote_code=False,
            torch_dtype=torch.float32).to("cuda").eval()
        torch.cuda.reset_peak_memory_stats()
        with cache_path.open("a", encoding="utf-8") as stream:
            for key in todo:
                if time.perf_counter() - started > limit:
                    error = "remaining_time_cap"
                    break
                pair = pairs[key]
                encoded = tok(PROMPT.format(text_a=pair["document"], text_b=pair["claim"]),
                              return_tensors="pt", truncation=False).to("cuda")
                assert encoded["input_ids"].shape[1] <= 2048
                with torch.inference_mode():
                    logits = model(**encoded).logits[0].float()
                    value = {"key": key, "logits": logits.cpu().tolist(),
                             "margin": float(logits[0] - logits[1]),
                             "probabilities": logits.softmax(-1).cpu().tolist(),
                             "input_tokens": int(encoded["input_ids"].shape[1]),
                             "truncated_tokens": 0}
                stream.write(json.dumps(value) + "\n")
                stream.flush()
                completed += 1
                if completed % 200 == 0:
                    print(f"resume {completed}/{len(todo)}", flush=True)
                if torch.cuda.max_memory_allocated() > 10 * 1024**3:
                    error = "VRAM_cap"
                    break
    except (RuntimeError, AssertionError) as exc:
        error = f"{type(exc).__name__}: {exc}"
    elapsed = time.perf_counter() - started
    write(destination, {"new_pairs": completed, "seconds_with_load": elapsed,
                        "limit_seconds": limit, "failure": error, **memory_report()})
    print({"completed": completed, "seconds": elapsed, "failure": error})
    if error:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
