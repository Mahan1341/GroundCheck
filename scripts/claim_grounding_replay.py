"""Six fixed native-logit replays after the C2 CUDA reset; never a research cycle.

Run only after the one permitted resume completes. No labels, thresholds, weight
updates, new input construction, cache writes or selection based on model outcomes.
"""

import hashlib
import json
import math
import time
from pathlib import Path

from claim_grounding_c2 import MODEL, PROMPT
from claim_grounding_c2_bounded import BASE, PRED, REPORT
from scope_analysis import ROOT, read, write

from grounded_hallucination_lab.research.protocol import digest

OUT = ROOT / "reports/claim_grounding_20261026/c2_replay_resources.json"
FREEZE = ROOT / "models/claim_grounding_20261026/c2_replay_freeze.json"
SNAPSHOT = ROOT / "models/claim_grounding_20261026/c2_replay_source"
LIMIT_SECONDS = 60.0
TOLERANCE = 1e-5
MODEL_SHA = "379a343d1ca25c9dfe24ae149f76cc958ea4f09c00dae88bd49bbbf58c749d5e"


def select_sentinels(cache_bytes, prefix_count, prefix_sha, pending_keys):
    """Validate the unchanged resume prefix and completed suffix before choosing six."""
    lines = cache_bytes.splitlines(keepends=True)
    if prefix_count < 3 or len(lines) < prefix_count + 3:
        raise ValueError("Need three pre-reset and three post-reset entries")
    if hashlib.sha256(b"".join(lines[:prefix_count])).hexdigest() != prefix_sha:
        raise ValueError("Pre-reset cache prefix changed")
    entries = [json.loads(line) for line in lines]
    keys = [entry["key"] for entry in entries]
    if len(keys) != len(set(keys)):
        raise ValueError("Duplicate cache keys")
    if keys[prefix_count:] != pending_keys:
        raise ValueError("Resume must finish its exact frozen pending-key sequence")
    selected = entries[prefix_count - 3 : prefix_count + 3]
    for entry in selected:
        if len(entry.get("logits", [])) != 2 or not all(
            math.isfinite(value) for value in entry["logits"]
        ):
            raise ValueError("Each sentinel needs two finite original native logits")
    return selected, keys


def prepare():
    """CPU-only guards and immutable pre-forward snapshot; no predictions executed."""
    if OUT.exists() or FREEZE.exists() or SNAPSHOT.exists():
        raise ValueError("One integrity replay only; preserve any previous attempt")
    failure_path = REPORT / "c2_run_resources.json"
    resume_path = REPORT / "c2_resume1_resources.json"
    failed, resumed = read(failure_path), read(resume_path)
    if "CUDA error: unknown error" not in failed["failure"] or resumed["failure"] is not None:
        raise ValueError("Replay requires the documented CUDA failure and successful resume")
    used = failed["seconds_with_load"] + resumed["seconds_with_load"]
    remaining = 840.0 - used
    if not math.isfinite(remaining) or remaining <= LIMIT_SECONDS:
        raise ValueError("Insufficient remaining original C2 runtime for a 60-second replay")
    resume_freeze_path = BASE / "c2_resume1_freeze.json"
    resume_freeze = read(resume_freeze_path)
    if digest(failure_path) != resume_freeze["failed_report_sha256"]:
        raise ValueError("Preserved failure report changed")
    if digest(BASE / "c2_freeze.json") != resume_freeze["original_freeze_sha256"]:
        raise ValueError("Original C2 freeze changed")
    for name, expected in read(BASE / "c2_freeze.json")["files"].items():
        if digest(ROOT / name) != expected:
            raise ValueError(f"Frozen dependency changed: {name}")
    if digest(MODEL / "model.safetensors") != MODEL_SHA:
        raise ValueError("FactCG weights changed")
    cache_path = PRED / "c2_pair_cache.jsonl"
    selected, keys = select_sentinels(
        cache_path.read_bytes(), resume_freeze["cached_pairs"],
        resume_freeze["cache_prefix_sha256"], resume_freeze["pending_keys"],
    )
    pairs_path = BASE / "c2_pairs.json"
    pairs = read(pairs_path)
    if set(keys) != set(pairs) or resumed["new_pairs"] != len(resume_freeze["pending_keys"]):
        raise ValueError("Resume/cache is incomplete or contains unapproved pairs")
    source = [
        Path(__file__), ROOT / "scripts/claim_grounding_c2.py",
        ROOT / "scripts/claim_grounding_c2_bounded.py", ROOT / "scripts/claim_grounding_resume.py",
        ROOT / "scripts/discovery_verifier.py", ROOT / "scripts/scope_analysis.py",
        ROOT / "src/grounded_hallucination_lab/research/verifiers.py",
        ROOT / "src/grounded_hallucination_lab/research/protocol.py",
    ]
    files = source + [
        failure_path, resume_path, resume_freeze_path, BASE / "c2_freeze.json",
        cache_path, pairs_path,
    ] + [p for p in MODEL.iterdir() if p.is_file()]
    hashes = {str(path.relative_to(ROOT)): digest(path) for path in files}
    SNAPSHOT.mkdir(exist_ok=False)
    for path in source:
        target = SNAPSHOT / path.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(path.read_bytes())
    write(FREEZE, {
        "files": hashes, "selection": "last3 original prefix + first3 resumed suffix",
        "cached_pairs_at_reset": resume_freeze["cached_pairs"],
        "sentinels": [{"key": row["key"], "original_logits": row["logits"]} for row in selected],
        "absolute_logit_tolerance": TOLERANCE, "replay_cap_seconds_including_load": LIMIT_SECONDS,
        "original_run_cap_seconds": 840, "already_used_seconds": used,
        "remaining_seconds_before_replay": remaining,
        "purpose": "hardware-reset integrity check; no model/threshold/data selection",
        "protocol": "native FactCG FP32, batch1, TF32 off, exact prompt, no truncation",
    })
    return selected, pairs, hashes, used


def main():
    selected, pairs, hashes, prior_seconds = prepare()
    started = time.perf_counter()
    measurements, failure, memory = [], None, {}
    try:
        from grounded_hallucination_lab.research.verifiers import memory_report, setup_runtime

        torch = setup_runtime()
        torch.backends.cudnn.allow_tf32 = False
        torch.set_float32_matmul_precision("highest")
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        tok = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False)
        tok.pad_token = tok.eos_token
        model = AutoModelForSequenceClassification.from_pretrained(
            MODEL, local_files_only=True, trust_remote_code=False,
            torch_dtype=torch.float32,
        ).to("cuda").eval()
        torch.cuda.reset_peak_memory_stats()
        for index, original in enumerate(selected):
            if time.perf_counter() - started >= LIMIT_SECONDS:
                raise TimeoutError("Replay load-inclusive 60-second cap")
            pair = pairs[original["key"]]
            encoded = tok(
                PROMPT.format(text_a=pair["document"], text_b=pair["claim"]),
                return_tensors="pt", truncation=False,
            ).to("cuda")
            tokens = int(encoded["input_ids"].shape[1])
            if tokens > 2048 or tokens != original["input_tokens"]:
                raise ValueError("Sentinel native tokenization changed or exceeds budget")
            with torch.inference_mode():
                logits = model(**encoded).logits[0].float().cpu().tolist()
            torch.cuda.synchronize()
            if not all(math.isfinite(value) for value in logits):
                raise RuntimeError("Non-finite replay logits")
            delta = [abs(a - b) for a, b in zip(logits, original["logits"], strict=True)]
            measurements.append({
                "key": original["key"], "side": "pre_reset" if index < 3 else "post_reset",
                "original_logits": original["logits"], "replay_logits": logits,
                "absolute_logit_deltas": delta, "max_absolute_logit_delta": max(delta),
                "input_tokens": tokens, "passed": all(math.isfinite(d) and d <= TOLERANCE
                                                       for d in delta),
            })
            if time.perf_counter() - started >= LIMIT_SECONDS:
                raise TimeoutError("Replay load-inclusive 60-second cap")
            if torch.cuda.max_memory_allocated() > 10 * 1024**3:
                raise RuntimeError("Original 10GiB VRAM guard exceeded")
        memory = memory_report()
    except Exception as exc:
        failure = f"{type(exc).__name__}: {exc}"
        try:
            memory = memory_report()
        except Exception as memory_exc:
            memory = {"memory_report_failure": f"{type(memory_exc).__name__}: {memory_exc}"}
    changed = [name for name, expected in hashes.items() if digest(ROOT / name) != expected]
    elapsed = time.perf_counter() - started
    if changed:
        failure = f"Preserved dependencies changed: {changed}"
    if elapsed >= LIMIT_SECONDS:
        failure = "Replay load-inclusive 60-second cap"
    passed = failure is None and len(measurements) == 6 and all(m["passed"] for m in measurements)
    write(OUT, {
        "passed": passed, "failure": failure, "count": len(measurements),
        "seconds_with_load": elapsed, "limit_seconds": LIMIT_SECONDS,
        "prior_run_seconds": prior_seconds, "run_plus_replay_seconds": prior_seconds + elapsed,
        "remaining_original_run_seconds": 840 - prior_seconds - elapsed,
        "max_absolute_logit_delta": max(
            (m["max_absolute_logit_delta"] for m in measurements), default=None,
        ),
        "absolute_logit_tolerance": TOLERANCE, "measurements": measurements,
        "preserved_checksums_verified": not changed, "freeze_sha256": digest(FREEZE), **memory,
    })
    print({"passed": passed, "count": len(measurements), "seconds_with_load": elapsed,
           "failure": failure})
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
