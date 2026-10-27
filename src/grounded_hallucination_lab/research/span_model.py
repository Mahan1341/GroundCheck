"""Train-only answer token classification with complete evidence and extractive offsets."""

import math
import random
import time
from collections import Counter
from pathlib import Path

import numpy as np

from grounded_hallucination_lab.data.ragtruth import gold_label
from grounded_hallucination_lab.eval.scored import scored_metrics
from grounded_hallucination_lab.eval.selection import select_threshold
from grounded_hallucination_lab.research.protocol import (
    ROOT,
    InferenceInput,
    digest,
    provenance,
    write_json,
)
from grounded_hallucination_lab.research.verifiers import memory_report, setup_runtime


def token_targets(offsets: list[tuple[int, int]], spans: list[tuple[int, int]]) -> list[int]:
    """Any nonempty character overlap is positive; special tokens have no target."""
    return [
        int(any(a < end and b > start for start, end in spans)) if b > a else -100
        for a, b in offsets
    ]


def windows(
    view: InferenceInput,
    tokenizer,
    *,
    max_length: int = 2048,
    answer_only: bool = False,
    spans: list[tuple[int, int]] | None = None,
) -> tuple[list, dict]:
    prefix = "" if answer_only else f"Question: {view.question}\nEvidence: {view.evidence}\nAnswer:"
    prefix_ids = tokenizer(prefix, add_special_tokens=False)["input_ids"]
    answer = tokenizer(view.response, add_special_tokens=False, return_offsets_mapping=True)
    answer_ids, offsets = answer["input_ids"], answer["offset_mapping"]
    capacity = min(512, max_length - len(prefix_ids) - 3)
    if capacity < 32:
        raise ValueError(
            f"complete evidence does not fit: prefix={len(prefix_ids)}, limit={max_length}"
        )
    step = capacity - min(64, capacity // 4)
    slices = []
    for start in range(0, max(1, len(answer_ids)), step):
        end = min(start + capacity, len(answer_ids))
        slices.append((start, end))
        if end == len(answer_ids):
            break
    counts = Counter(i for a, b in slices for i in range(a, b))
    targets = token_targets(offsets, spans) if spans is not None else [-100] * len(answer_ids)
    result = []
    for start, end in slices:
        ids = [
            tokenizer.cls_token_id,
            *prefix_ids,
            tokenizer.sep_token_id,
            *answer_ids[start:end],
            tokenizer.sep_token_id,
        ]
        lead = len(prefix_ids) + 2
        result.append(
            {
                "input_ids": ids,
                "attention_mask": [1] * len(ids),
                "labels": [-100] * lead + targets[start:end] + [-100],
                "loss_weights": [0.0] * lead + [1 / counts[i] for i in range(start, end)] + [0.0],
                "answer_indices": [-1] * lead + list(range(start, end)) + [-1],
                "record_id": view.id,
            }
        )
    return result, {
        "id": view.id,
        "source_id": view.source_id,
        "offsets": offsets,
        "answer_tokens": len(answer_ids),
        "prefix_tokens": len(prefix_ids),
        "windows": len(result),
        "answer_coverage": 1.0,
        "evidence_coverage": 0.0 if answer_only else 1.0,
        "truncated_tokens": 0,
        "max_window_tokens": max(len(w["input_ids"]) for w in result),
    }


def encode_records(records, tokenizer, max_length, answer_only, *, training=False):
    examples, metadata = [], []
    for record in records:
        view = InferenceInput.from_record(record)
        spans = [(a.start, a.end) for a in record.annotations] if training else None
        encoded, info = windows(
            view, tokenizer, max_length=max_length, answer_only=answer_only, spans=spans
        )
        examples.extend(encoded)
        metadata.append(info)
    return examples, metadata


def collate(batch, pad_token_id):
    import torch

    size = max(len(row["input_ids"]) for row in batch)
    result = {}
    for key, pad, dtype in (
        ("input_ids", pad_token_id, torch.long),
        ("attention_mask", 0, torch.long),
        ("labels", -100, torch.long),
        ("loss_weights", 0.0, torch.float32),
    ):
        result[key] = torch.tensor(
            [r[key] + [pad] * (size - len(r[key])) for r in batch], dtype=dtype, device="cuda"
        )
    return result


def load_model(path: Path | None = None, seed: int = 42):
    torch = setup_runtime()
    from transformers import AutoModelForTokenClassification, AutoTokenizer, set_seed

    set_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    path = path or ROOT / "models/pretrained/backbone"
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
    model = AutoModelForTokenClassification.from_pretrained(
        path,
        num_labels=2,
        local_files_only=True,
        attn_implementation="sdpa",
        reference_compile=False,
        id2label={0: "not_annotated", 1: "hallucination"},
        label2id={"not_annotated": 0, "hallucination": 1},
    )
    return model.to("cuda"), tokenizer


def score(model, tokenizer, examples, metadata, batch_size=2):
    import torch

    model.eval()
    torch.cuda.reset_peak_memory_stats()
    sums = {m["id"]: np.zeros(m["answer_tokens"]) for m in metadata}
    counts = {m["id"]: np.zeros(m["answer_tokens"]) for m in metadata}
    ordered = sorted(examples, key=lambda r: len(r["input_ids"]))
    start = time.perf_counter()
    times = []
    for offset in range(0, len(ordered), batch_size):
        batch = ordered[offset : offset + batch_size]
        inputs = collate(batch, tokenizer.pad_token_id)
        torch.cuda.synchronize()
        t = time.perf_counter()
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            outputs = model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])
        probabilities = outputs.logits.float().softmax(-1)[:, :, 1].cpu().numpy()
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t)
        for j, row in enumerate(batch):
            for k, token_index in enumerate(row["answer_indices"]):
                if token_index >= 0:
                    sums[row["record_id"]][token_index] += probabilities[j, k]
                    counts[row["record_id"]][token_index] += 1
    rows = []
    for m in metadata:
        assert np.all(counts[m["id"]] > 0), "lost answer tokens"
        values = sums[m["id"]] / counts[m["id"]]
        rows.append(
            {**m, "token_probabilities": values.tolist(), "score": float(max(values, default=0))}
        )
    seconds = time.perf_counter() - start
    return rows, {
        "responses": len(rows),
        "pipeline_seconds": seconds,
        "responses_per_second": len(rows) / seconds,
        "amortized_seconds_per_response": seconds / len(rows),
        "batch_forward_latency_p50_s": float(np.median(times)),
        "batch_forward_latency_p95_s": float(np.percentile(times, 95)),
        "windows": len(examples),
        "batch_size": batch_size,
        **memory_report(),
    }


def train(
    train_records,
    val_records,
    destination,
    *,
    epochs=3,
    seed=42,
    max_length=2048,
    microbatch=1,
    answer_only=False,
    select_epoch=True,
    initial_checkpoint=None,
    learning_rate=2e-5,
    positive_weight_override=None,
):
    import torch
    import torch.nn.functional as functional
    from transformers import get_linear_schedule_with_warmup

    if destination.exists():
        raise FileExistsError(destination)
    destination.mkdir(parents=True)
    model, tokenizer = load_model(initial_checkpoint, seed=seed)
    model.gradient_checkpointing_enable()
    examples, metadata = encode_records(
        train_records, tokenizer, max_length, answer_only, training=True
    )
    val_examples, val_meta = encode_records(val_records, tokenizer, max_length, answer_only)
    # Weighted window multiplicity counts each original answer token once.
    pos = sum(
        w for r in examples for y, w in zip(r["labels"], r["loss_weights"], strict=True) if y == 1
    )
    neg = sum(
        w for r in examples for y, w in zip(r["labels"], r["loss_weights"], strict=True) if y == 0
    )
    positive_weight = (
        min(10.0, math.sqrt(neg / max(pos, 1)))
        if positive_weight_override is None
        else float(positive_weight_override)
    )
    if not math.isfinite(positive_weight) or positive_weight <= 0:
        raise ValueError("positive class weight must be positive and finite")
    class_weights = torch.tensor([1.0, positive_weight], device="cuda")
    accumulation = max(1, 8 // microbatch)
    total_steps = epochs * math.ceil(math.ceil(len(examples) / microbatch) / accumulation)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    scheduler = get_linear_schedule_with_warmup(optimizer, int(total_steps * 0.1), total_steps)
    config = {
        "seed": seed,
        "max_epochs": epochs,
        "max_length": max_length,
        "microbatch": microbatch,
        "accumulation": accumulation,
        "answer_only": answer_only,
        "select_epoch_on_validation": select_epoch,
        "learning_rate": learning_rate,
        "initial_checkpoint": str(initial_checkpoint) if initial_checkpoint else None,
        "initial_checkpoint_sha256": (
            digest(Path(initial_checkpoint) / "model.safetensors") if initial_checkpoint else None
        ),
        "positive_weight_override": positive_weight_override,
        "precision": "FP32 weights/optimizer, BF16 autocast, FP32 loss",
        "deterministic_algorithms": True,
        "positive_weight": positive_weight,
        "training_target_mass": {"positive": pos, "negative": neg},
        "train_ids": [r.id for r in train_records],
        "train_sources": sorted({r.source_id for r in train_records}),
        "validation_ids": [r.id for r in val_records],
        "provenance": provenance(),
        "coverage": {
            "windows": len(examples),
            "answer_coverage": 1.0,
            "evidence_coverage": 0.0 if answer_only else 1.0,
            "max_window_tokens": max(len(r["input_ids"]) for r in examples),
        },
    }
    write_json(destination / "training_config.json", config)
    rng = random.Random(seed)
    history, best, best_epoch = [], -1.0, 0
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    for epoch in range(1, epochs + 1):
        model.train()
        indices = list(range(len(examples)))
        rng.shuffle(indices)
        optimizer.zero_grad(set_to_none=True)
        losses = []
        batches = math.ceil(len(indices) / microbatch)
        for step, offset in enumerate(range(0, len(indices), microbatch), 1):
            batch = collate(
                [examples[i] for i in indices[offset : offset + microbatch]], tokenizer.pad_token_id
            )
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(
                    input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]
                ).logits
                loss_tokens = functional.cross_entropy(
                    logits.float().reshape(-1, 2),
                    batch["labels"].reshape(-1),
                    weight=class_weights,
                    ignore_index=-100,
                    reduction="none",
                ).reshape_as(batch["labels"])
                loss = (loss_tokens * batch["loss_weights"]).sum() / batch[
                    "loss_weights"
                ].sum().clamp_min(1)
            # Final partial accumulation uses its actual size.
            divisor = min(accumulation, batches - ((step - 1) // accumulation) * accumulation)
            (loss / divisor).backward()
            losses.append(float(loss.detach()))
            if step % accumulation == 0 or step == batches:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
            if step % 100 == 0:
                print(
                    f"seed{seed} epoch{epoch} step{step}/{batches} "
                    f"loss={np.mean(losses[-100:]):.4f}",
                    flush=True,
                )
        train_memory = memory_report()
        info = {
            "epoch": epoch,
            "loss": float(np.mean(losses)),
            "elapsed_seconds": time.perf_counter() - start,
            "training_memory": train_memory,
        }
        if val_records:
            rows, speed = score(model, tokenizer, val_examples, val_meta, batch_size=microbatch)
            labels = [gold_label(r) for r in val_records]
            risks = [r["score"] for r in rows]
            selection = select_threshold(labels, risks)
            value = selection["macro_f1"] if select_epoch else float(epoch)
            info.update(
                {
                    "inner_selection": selection,
                    "inner_metrics": scored_metrics(
                        labels, [int(s >= selection["threshold"]) for s in risks], risks
                    ),
                    "validation_runtime": speed,
                }
            )
        else:
            value = float(epoch)  # Final fixed-epoch fitting, no held-out selection.
        history.append(info)
        label = "inner_macro_f1" if val_records and select_epoch else "fixed_epoch"
        print(f"seed{seed} epoch{epoch} {label}={value:.6f}", flush=True)
        if value > best:
            best, best_epoch = value, epoch
            model.save_pretrained(destination / "best", safe_serialization=True)
            tokenizer.save_pretrained(destination / "best")
        write_json(
            destination / "history.json",
            {"epochs": history, "selected_epoch": best_epoch},
            overwrite=True,
        )
    result = {
        "config": config,
        "selected_epoch": best_epoch,
        "history": history,
        "training_seconds_including_validation": time.perf_counter() - start,
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "training_peak_cuda_allocated_bytes": max(
            h["training_memory"]["cuda_peak_allocated_bytes"] for h in history
        ),
        "training_peak_cuda_reserved_bytes": max(
            h["training_memory"]["cuda_peak_reserved_bytes"] for h in history
        ),
        **memory_report(),
    }
    write_json(destination / "result.json", result)
    return result
