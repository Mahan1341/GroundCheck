"""Matched CE versus CE + aligned evidence-pair loss. Inference remains unchanged."""

import hashlib
import json
import math
import random
import time
from collections import defaultdict

import numpy as np

from grounded_hallucination_lab.research.protocol import (
    ROOT,
    digest,
    load_parts,
    partition,
    write_json,
)
from grounded_hallucination_lab.research.relations import content_hash, relation_suite
from grounded_hallucination_lab.research.span_model import (
    collate,
    encode_records,
    load_model,
)
from grounded_hallucination_lab.research.verifiers import memory_report

BASE = ROOT / "models/phase2b_r2"
INITIAL = ROOT / "models/phase2b/span_inner_s42/best"
INITIAL_SHA = "7c6fb6f5f842863175f6422a84acd9c85d5056c3144ec73738b3e3a83464b4f4"


def slot_mask(example, offsets, start, end):
    """Map authored character slot to ordinary input tokens via answer indices."""
    if not 0 <= start < end:
        raise ValueError("empty or invalid claim slot")
    mask = [
        i >= 0 and offsets[i][0] < end and offsets[i][1] > start for i in example["answer_indices"]
    ]
    if not any(mask):
        raise ValueError("claim slot not covered by answer window")
    if any(m and y == -100 for m, y in zip(mask, example["labels"], strict=True)):
        raise ValueError("pair supervision includes a masked prefix/special token")
    return mask


def pair_objective(logits, masks, margin=1.0):
    """Rows ordered supported, contradicted; mean slot log odds, not global max."""
    import torch
    import torch.nn.functional as F

    mask = torch.as_tensor(masks, device=logits.device, dtype=torch.bool)
    if logits.shape[0] != 2 or mask.shape != logits.shape[:2] or not mask.any(1).all():
        raise ValueError("two nonempty aligned masks required")
    odds = logits.float()[..., 1] - logits.float()[..., 0]
    local = (odds * mask).sum(1) / mask.sum(1)
    return F.softplus(margin - (local[1] - local[0]))


def batches(tokenizer, seed):
    from grounded_hallucination_lab.research.protocol import read_json

    natural = partition(load_parts(), "inner_train")
    synthetic, design = relation_suite("fit")
    manifest = read_json(BASE / "suite_manifest.json")
    if content_hash(synthetic, design) != manifest["splits"]["fit"]["content_sha256"]:
        raise ValueError("frozen fit suite changed")
    forbidden = {r.source_id for r in partition(load_parts(), "inner_val")}
    for split in ("selection", "diagnostic"):
        forbidden |= set(manifest["splits"][split]["sources"])
    if {r.source_id for r in natural + synthetic} & forbidden:
        raise ValueError("training source overlaps held-out source")
    ex, meta = encode_records(natural, tokenizer, 2048, False, training=True)
    sx, sm = encode_records(synthetic, tokenizer, 2048, False, training=True)
    if len(sx) != len(synthetic) or any(m["windows"] != 1 for m in sm):
        raise ValueError("synthetic paired answer must be wholly covered in one window")
    by_id = {m["id"]: m for m in sm}
    by_example = {e["record_id"]: e for e in sx}
    pair_map = defaultdict(dict)
    for d in design:
        pair_map[d["pair_id"]][d["gold"]] = d
    rng = random.Random(seed)
    rng.shuffle(ex)
    output = [{"examples": ex[i : i + 2], "masks": None} for i in range(0, len(ex), 2)]
    for pair in pair_map.values():
        ds = [pair[0], pair[1]]
        examples = [by_example[d["id"]] for d in ds]
        masks = [
            slot_mask(e, by_id[d["id"]]["offsets"], d["slot_start"], d["slot_end"])
            for e, d in zip(examples, ds, strict=True)
        ]
        # Match the complete answer tokens, not token positions in distinct prefixes.
        answer_ids = [
            [t for t, i in zip(e["input_ids"], e["answer_indices"], strict=True) if i >= 0]
            for e in examples
        ]
        slots = [
            [t for t, m in zip(e["input_ids"], mask, strict=True) if m]
            for e, mask in zip(examples, masks, strict=True)
        ]
        if answer_ids[0] != answer_ids[1] or slots[0] != slots[1]:
            raise ValueError("supported/contradictory answer token alignment changed")
        output.append({"examples": examples, "masks": masks})
    rng.shuffle(output)
    all_examples = ex + sx
    order = [[e["record_id"] for e in b["examples"]] for b in output]
    targets = [(e["record_id"], e["labels"], e["loss_weights"]) for e in all_examples]

    def stable_hash(v):
        return hashlib.sha256(json.dumps(v, sort_keys=True).encode()).hexdigest()

    info = {
        "train_ids": [r.id for r in natural + synthetic],
        "train_sources": sorted({r.source_id for r in natural + synthetic}),
        "natural_rows": len(natural),
        "synthetic_rows": len(synthetic),
        "batch_order_sha256": stable_hash(order),
        "target_sha256": stable_hash(targets),
        "batches": len(output),
        "optimizer_steps": math.ceil(len(output) / 4),
        "max_window_tokens": max(len(e["input_ids"]) for e in all_examples),
        "answer_coverage": 1.0,
        "evidence_coverage": 1.0,
        "truncated_tokens": 0,
        "target_mass": {
            str(y): sum(
                w
                for e in all_examples
                for t, w in zip(e["labels"], e["loss_weights"], strict=True)
                if t == y
            )
            for y in (0, 1)
        },
    }
    return output, info


def batch_loss(model, tokenizer, item, weight):
    import torch
    import torch.nn.functional as F

    batch = collate(item["examples"], tokenizer.pad_token_id)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        logits = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).logits
    tokens = F.cross_entropy(
        logits.float().reshape(-1, 2),
        batch["labels"].reshape(-1),
        weight=torch.tensor([1.0, 3.370202392225359], device="cuda"),
        ignore_index=-100,
        reduction="none",
    ).reshape_as(batch["labels"])
    ce = (tokens * batch["loss_weights"]).sum() / batch["loss_weights"].sum().clamp_min(1)
    pair = logits.float().sum() * 0
    if item["masks"] is not None:
        size = logits.shape[1]
        masks = [m + [False] * (size - len(m)) for m in item["masks"]]
        pair = pair_objective(logits, masks)
    return ce + weight * pair, ce, pair


def run(command, seed, weight):
    import torch
    from research_r2 import state
    from transformers import get_linear_schedule_with_warmup

    from grounded_hallucination_lab.research.protocol import read_json

    if digest(INITIAL / "model.safetensors") != INITIAL_SHA:
        raise ValueError("B7 initialization hash changed")
    if seed not in (42, 43, 44) or weight not in (0.0, 0.1):
        raise ValueError("outside preregistered configuration set")
    if command != "profile":
        profile = read_json(BASE / "profile/result.json")
        if any(p["cuda_peak_allocated_bytes"] > 11 * 1024**3 for p in profile["profile"]):
            raise ValueError("profile exceeds reserved GPU budget")
        if seed != 42:
            gate = read_json(ROOT / "reports/phase2b_r2/seed42_gate.json")
            if not gate["repeat_allowed"]:
                raise ValueError("preregistered seed repetition gate failed")
    destination = BASE / ("profile" if command == "profile" else f"R2C{int(weight > 0)}s{seed}")
    if destination.exists():
        raise FileExistsError(destination)
    destination.mkdir(parents=True)
    model, tokenizer = load_model(INITIAL, seed=seed)
    model.gradient_checkpointing_enable()
    items, info = batches(tokenizer, seed)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=0.01)
    config = {
        **info,
        "seed": seed,
        "pair_weight": weight,
        "margin_log_odds": 1.0,
        "initial_checkpoint_sha256": INITIAL_SHA,
        "epochs": 1,
        "microbatch": 2,
        "accumulation": 4,
        "learning_rate": 1e-5,
        "positive_weight": 3.370202392225359,
        "max_length": 2048,
        "precision": "BF16 autocast, FP32 parameters/AdamW/loss",
        "inference": "unchanged max-token risk; no claim masks",
        "provenance": state(),
    }
    write_json(destination / "training_config.json", config)
    if command == "profile":
        natural = sorted(
            [e for item in items if item["masks"] is None for e in item["examples"]],
            key=lambda e: len(e["input_ids"]),
            reverse=True,
        )[:2]
        paired = max(
            (b for b in items if b["masks"] is not None),
            key=lambda b: max(len(e["input_ids"]) for e in b["examples"]),
        )
        profile = []
        for name, item in [
            ("longest_natural", {"examples": natural, "masks": None}),
            ("longest_pair", paired),
        ]:
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.reset_peak_memory_stats()
            t = time.perf_counter()
            loss, ce, pair = batch_loss(model, tokenizer, item, 0.1)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            torch.cuda.synchronize()
            profile.append(
                {
                    "case": name,
                    "seconds": time.perf_counter() - t,
                    "tokens": [len(e["input_ids"]) for e in item["examples"]],
                    "ce": float(ce.detach()),
                    "pair": float(pair.detach()),
                    **memory_report(),
                }
            )
        write_json(destination / "result.json", {"profile": profile, "config": config})
        print(profile, flush=True)
        return
    scheduler = get_linear_schedule_with_warmup(
        optimizer, int(info["optimizer_steps"] * 0.1), info["optimizer_steps"]
    )
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    optimizer.zero_grad(set_to_none=True)
    logs = []
    for step, item in enumerate(items, 1):
        loss, ce, pair = batch_loss(model, tokenizer, item, weight)
        divisor = min(4, len(items) - ((step - 1) // 4) * 4)
        (loss / divisor).backward()
        logs.append(
            {
                "step": step,
                "ce": float(ce.detach()),
                "pair": float(pair.detach()),
                "synthetic_pair": item["masks"] is not None,
            }
        )
        if step % 4 == 0 or step == len(items):
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
        if step % 100 == 0:
            print(
                f"seed{seed} lambda{weight} batch{step}/{len(items)} "
                f"CE={np.mean([x['ce'] for x in logs[-100:]]):.4f}",
                flush=True,
            )
    torch.cuda.synchronize()
    seconds = time.perf_counter() - start
    model.save_pretrained(destination / "best", safe_serialization=True)
    tokenizer.save_pretrained(destination / "best")
    result = {
        "config": config,
        "training_seconds": seconds,
        "batches_completed": len(items),
        "checkpoint_sha256": digest(destination / "best/model.safetensors"),
        "mean_ce": float(np.mean([x["ce"] for x in logs])),
        "mean_pair": float(np.mean([x["pair"] for x in logs if x["synthetic_pair"]])),
        "parameter_count": sum(p.numel() for p in model.parameters()),
        **memory_report(),
    }
    write_json(destination / "batch_log.json", logs)
    write_json(destination / "result.json", result)
    print({k: v for k, v in result.items() if k != "config"}, flush=True)
