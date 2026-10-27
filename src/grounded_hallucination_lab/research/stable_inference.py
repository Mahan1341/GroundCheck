"""Explicit precision/padding inference for R6; historical scorer stays unchanged."""

import time

import numpy as np

from grounded_hallucination_lab.research.span_model import collate
from grounded_hallucination_lab.research.verifiers import memory_report


def score(model, tokenizer, examples, metadata, *, precision="fp32", batch_size=1, pad_to=0):
    """Average overlapping windows, then unchanged max-answer-token risk."""
    import torch
    import torch.nn.functional as functional

    if precision not in ("fp32", "bf16") or batch_size not in (1, 2):
        raise ValueError("explicit audited precision/batch policy required")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    model.eval()
    torch.cuda.reset_peak_memory_stats()
    sums = {m["id"]: np.zeros(m["answer_tokens"]) for m in metadata}
    counts = {m["id"]: np.zeros(m["answer_tokens"]) for m in metadata}
    ordered = sorted(examples, key=lambda r: len(r["input_ids"]))
    start, times = time.perf_counter(), []
    for offset in range(0, len(ordered), batch_size):
        batch = ordered[offset : offset + batch_size]
        inputs = collate(batch, tokenizer.pad_token_id)
        if pad_to:
            length = inputs["input_ids"].shape[1]
            if length > pad_to:
                raise ValueError("padding budget shorter than input")
            inputs["input_ids"] = functional.pad(
                inputs["input_ids"], (0, pad_to - length), value=tokenizer.pad_token_id
            )
            inputs["attention_mask"] = functional.pad(
                inputs["attention_mask"], (0, pad_to - length), value=0
            )
        torch.cuda.synchronize()
        t = time.perf_counter()
        with (
            torch.inference_mode(),
            torch.autocast("cuda", dtype=torch.bfloat16, enabled=precision == "bf16"),
        ):
            logits = model(
                input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"]
            ).logits
        probabilities = logits.float().softmax(-1)[:, :, 1].cpu().numpy()
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t)
        for j, row in enumerate(batch):
            for k, index in enumerate(row["answer_indices"]):
                if index >= 0:
                    sums[row["record_id"]][index] += probabilities[j, k]
                    counts[row["record_id"]][index] += 1
    rows = []
    for m in metadata:
        assert np.all(counts[m["id"]] > 0), "lost answer tokens"
        values = sums[m["id"]] / counts[m["id"]]
        rows.append(
            {**m, "token_probabilities": values.tolist(), "score": float(max(values, default=0))}
        )
    seconds = time.perf_counter() - start
    return rows, {
        "precision": precision,
        "batch_size": batch_size,
        "pad_to": pad_to,
        "tf32": False,
        "responses": len(rows),
        "pipeline_seconds": seconds,
        "responses_per_second": len(rows) / seconds,
        "batch_forward_latency_p50_s": float(np.median(times)),
        "batch_forward_latency_p95_s": float(np.percentile(times, 95)),
        **memory_report(),
    }
