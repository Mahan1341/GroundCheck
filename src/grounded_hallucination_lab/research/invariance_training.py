"""R3 matched positive invariance. Reuses the established fixed training loop."""

import hashlib
import json
import math
import random
import time

import numpy as np

from grounded_hallucination_lab.research.pair_training import INITIAL, INITIAL_SHA, slot_mask
from grounded_hallucination_lab.research.protocol import (
    ROOT,
    digest,
    load_parts,
    partition,
    write_json,
)
from grounded_hallucination_lab.research.span_model import collate, encode_records, load_model
from grounded_hallucination_lab.research.verifiers import memory_report

BASE = ROOT / "models/phase2b_r3"


def pair_objective(logits, masks):
    """Token-aligned MSE on supported equivalent views; no constraint on truth flips."""
    import torch

    mask = torch.as_tensor(masks, device=logits.device, dtype=torch.bool)
    if logits.shape[0] != 2 or mask.shape != logits.shape[:2] or not mask.any(1).all():
        raise ValueError("two aligned nonempty masks required")
    if int(mask[0].sum()) != int(mask[1].sum()):
        raise ValueError("different token counts")
    probs = logits.float().softmax(-1)[..., 1]
    return ((probs[0][mask[0]] - probs[1][mask[1]]) ** 2).mean()


def batches(tokenizer, seed):
    from research_r3 import frozen

    natural = partition(load_parts(), "inner_train")
    synthetic, design = frozen("fit")
    ex, _ = encode_records(natural, tokenizer, 2048, False, training=True)
    sx, sm = encode_records(synthetic, tokenizer, 2048, False, training=True)
    if len(sx) != len(synthetic) or any(m["windows"] != 1 for m in sm):
        raise ValueError("paired answer not fully covered")
    forbidden = {r.source_id for r in partition(load_parts(), "inner_val")}
    for split in ("selection", "diagnostic"):
        forbidden |= {r.source_id for r in frozen(split)[0]}
    if {r.source_id for r in natural + synthetic} & forbidden:
        raise ValueError("source overlap")
    by_id = {e["record_id"]: e for e in sx}
    meta = {m["id"]: m for m in sm}
    lookup = {
        (d["source_id"], d["variant"], d["order"], d["answer_index"], d["evidence_index"]): d
        for d in design
    }
    rng = random.Random(seed)
    rng.shuffle(ex)
    output = [{"examples": ex[i : i + 2], "masks": None} for i in range(0, len(ex), 2)]
    used = set()
    for d in design:
        if d["variant"] not in ("words", "units"):
            continue
        base = lookup[(d["source_id"], "exact", d["order"], d["answer_index"], d["evidence_index"])]
        ds = [base, d]
        es = [by_id[x["id"]] for x in ds]
        masks = None
        if not d["gold"]:
            if base["gold"] or base["values"] != d["values"]:
                raise ValueError("not a supported equivalent pair")
            masks = [
                slot_mask(e, meta[x["id"]]["offsets"], x["claim_start"], x["claim_end"])
                for e, x in zip(es, ds, strict=True)
            ]
            ids = [
                [t for t, m in zip(e["input_ids"], mask, strict=True) if m]
                for e, mask in zip(es, masks, strict=True)
            ]
            if ids[0] != ids[1]:
                raise ValueError("claim token alignment changed")
        output.append({"examples": es, "masks": masks})
        used.update(x["id"] for x in ds)
    remaining = [e for e in sx if e["record_id"] not in used]
    output += [
        {"examples": remaining[i : i + 2], "masks": None} for i in range(0, len(remaining), 2)
    ]
    rng.shuffle(output)
    all_examples = [e for b in output for e in b["examples"]]

    def h(v):
        return hashlib.sha256(json.dumps(v, sort_keys=True).encode()).hexdigest()

    info = {
        "train_ids": [r.id for r in natural + synthetic],
        "train_sources": sorted({r.source_id for r in natural + synthetic}),
        "natural_rows": len(natural),
        "synthetic_unique_rows": len(synthetic),
        "synthetic_effective_rows": len(all_examples) - len(ex),
        "duplication_policy": (
            "canonical evidence repeated equally under both labels for word and unit pairs"
        ),
        "batch_order_sha256": h([[e["record_id"] for e in b["examples"]] for b in output]),
        "target_sha256": h(
            [(e["record_id"], e["labels"], e["loss_weights"]) for e in all_examples]
        ),
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
    from research_r3 import state
    from transformers import get_linear_schedule_with_warmup

    from grounded_hallucination_lab.research.protocol import read_json

    if digest(INITIAL / "model.safetensors") != INITIAL_SHA:
        raise ValueError("B7 initialization hash changed")
    if seed not in (42, 43, 44) or weight not in (0.0, 1.0):
        raise ValueError("outside preregistered configuration set")
    if command != "profile":
        profile = read_json(BASE / "profile/result.json")
        if any(p["cuda_peak_allocated_bytes"] > 11 * 1024**3 for p in profile["profile"]):
            raise ValueError("profile exceeds reserved GPU budget")
        if seed != 42:
            gate = read_json(ROOT / "reports/phase2b_r3/seed42_gate.json")
            if not gate["repeat_allowed"]:
                raise ValueError("preregistered seed repetition gate failed")
    destination = BASE / (
        "profile" if command == "profile" else f"R3{'B' if weight else 'A'}s{seed}"
    )
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
        "consistency": "mean squared aligned claim token probabilities; supported pairs only",
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
            loss, ce, pair = batch_loss(model, tokenizer, item, 1.0)
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
