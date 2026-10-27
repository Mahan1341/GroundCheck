"""Matched R6 relation-renderer experiment. No outer evaluation."""

import argparse
import hashlib
import json
import random
import time
from dataclasses import asdict

from grounded_hallucination_lab.data.ragtruth import gold_label
from grounded_hallucination_lab.research.binding import suite
from grounded_hallucination_lab.research.metrics import metrics, span_metrics
from grounded_hallucination_lab.research.pair_training import INITIAL, INITIAL_SHA
from grounded_hallucination_lab.research.protocol import (
    ROOT,
    digest,
    load_parts,
    partition,
    preserve_code,
    read_json,
    write_json,
    write_rows,
)
from grounded_hallucination_lab.research.relations import content_hash

BASE = ROOT / "models/phase2b_r6"
PRED = ROOT / "predictions/phase2b_r6"
REPORT = ROOT / "reports/phase2b_r6"
THRESHOLD = 0.37612172961235046
DATASETS = ("natural", "transfer")
FROZEN_MODELS = {
    "B7": (INITIAL, INITIAL_SHA),
    "R5A": (
        ROOT / "models/phase2b_r5/R5As42/best",
        "085c2941e7afac27cafd387df914748b851d04d92afa3c70820d8f29ab723de3",
    ),
}


def data(name):
    if name == "natural":
        return partition(load_parts(), "inner_val"), []
    if name == "transfer":
        return suite()
    raise ValueError("only frozen inner controls allowed")


def train_records(arm):
    return partition(load_parts(), "inner_train") + suite("fit", arm)[0]


def state():
    return {"source": preserve_code(), "plan_sha256": digest(ROOT / "EXPERIMENT_R6_PLAN.md")}


def checked():
    frozen = read_json(BASE / "freeze.json")
    assert frozen["state"] == state(), "source/plan changed; do not bypass hashes"
    assert digest(INITIAL / "model.safetensors") == INITIAL_SHA
    assert digest(BASE / "audit/result.json") == frozen["inference_audit_sha256"]
    return frozen


def freeze():
    previous = read_json(ROOT / "models/phase2b_r5/R5_artifact_manifest.json")
    for name, spec in previous["files"].items():
        assert digest(ROOT / name) == spec["sha256"], name
    audit = read_json(BASE / "audit/result.json")
    assert audit["stable"], "inference audit gate failed"
    result = {
        "state": state(),
        "R5_verified_files": len(previous["files"]),
        "inference_audit_sha256": digest(BASE / "audit/result.json"),
        "train_hash": {},
        "datasets": {},
        "models": {},
    }
    for arm in ("A", "B"):
        records = train_records(arm)
        result["train_hash"][arm] = content_hash(records, [])
    sources = {r.source_id for r in records}
    for method, (path, expected) in FROZEN_MODELS.items():
        assert digest(path / "model.safetensors") == expected
        result["models"][method] = expected
    for name in DATASETS:
        records, design = data(name)
        assert not sources & {r.source_id for r in records}
        result["datasets"][name] = {
            "sha256": content_hash(records, design),
            "n": len(records),
            "positive": sum(gold_label(r) for r in records),
            "sources": sorted({r.source_id for r in records}),
        }
        write_rows(PRED / f"{name}_records.jsonl", [asdict(r) for r in records])
        write_json(PRED / f"{name}_design.json", design)
    write_json(BASE / "freeze.json", result)
    print({"frozen": {k: v["n"] for k, v in result["datasets"].items()}}, flush=True)


def batches(tokenizer, seed, arm):
    from grounded_hallucination_lab.research.span_model import encode_records, token_targets

    frozen = checked()
    views = {}
    for label in ("A", "B"):
        records = train_records(label)
        assert content_hash(records, []) == frozen["train_hash"][label]
        rows, metas = encode_records(records, tokenizer, 2048, False, training=True)
        assert len(rows) == len(metas) == len(records) == 3336
        for r, row, meta in zip(records, rows, metas, strict=True):
            assert meta["windows"] == 1 and meta["truncated_tokens"] == 0
            assert meta["answer_coverage"] == meta["evidence_coverage"] == 1
            actual = [
                y for idx, y in zip(row["answer_indices"], row["labels"], strict=True) if idx >= 0
            ]
            assert actual == token_targets(
                meta["offsets"], [(a.start, a.end) for a in r.annotations]
            )
            assert all(
                y == -100 and w == 0
                for idx, y, w in zip(
                    row["answer_indices"], row["labels"], row["loss_weights"], strict=True
                )
                if idx < 0
            )
        views[label] = (records, rows, metas)
    for a, b in zip(views["A"][0], views["B"][0], strict=True):
        assert (a.id, a.response, a.question, a.annotations) == (
            b.id,
            b.response,
            b.question,
            b.annotations,
        )
    assert [m["offsets"] for m in views["A"][2]] == [m["offsets"] for m in views["B"][2]]
    records, rows, metas = views[arm]
    order = list(range(len(rows)))
    random.Random(seed).shuffle(order)
    items = [{"examples": [rows[i], rows[i]], "masks": None} for i in order]
    info = {
        "train_ids": [r.id for r in records],
        "train_sources": sorted({r.source_id for r in records}),
        "record_order_sha256": hashlib.sha256(
            json.dumps([records[i].id for i in order]).encode()
        ).hexdigest(),
        "batches": len(items),
        "optimizer_steps": len(items) // 4,
        "max_tokens": max(len(row["input_ids"]) for row in rows),
        "positive_tokens": sum(row["labels"].count(1) for row in rows),
        "answer_tokens": sum(m["answer_tokens"] for m in metas),
        "answer_targets_sha256": hashlib.sha256(
            json.dumps([[y for y in row["labels"] if y >= 0] for row in rows]).encode()
        ).hexdigest(),
        "input_tokens": {
            a: {
                "sum": sum(len(r["input_ids"]) for r in v[1]),
                "max": max(len(r["input_ids"]) for r in v[1]),
            }
            for a, v in views.items()
        },
        "coverage": "100% answer/evidence; one window; prefix/special loss masked",
    }
    return items, info, []


def train(command, arm, seed):
    import torch
    from transformers import get_linear_schedule_with_warmup

    from grounded_hallucination_lab.research.invariance_training import batch_loss
    from grounded_hallucination_lab.research.span_model import load_model
    from grounded_hallucination_lab.research.verifiers import memory_report

    frozen = checked()
    if seed != 42:
        assert read_json(REPORT / "seed42_gate.json")["passed"], "repeat-seed gate closed"
    if command == "train":
        for candidate in ("A", "B"):
            profile = read_json(BASE / f"profile{candidate}/result.json")
            assert profile["gate_passed"]
    destination = BASE / (f"profile{arm}" if command == "profile" else f"R6{arm}s{seed}")
    destination.mkdir(parents=True, exist_ok=False)
    model, tokenizer = load_model(INITIAL, seed)
    model.gradient_checkpointing_enable()
    items, info, audits = batches(tokenizer, seed, arm)
    config = {
        "arm": arm,
        "seed": seed,
        "initial_sha256": INITIAL_SHA,
        "lr": 1e-5,
        "weight_decay": 0.01,
        "epochs": 1,
        "microbatch": 2,
        "accumulation": 4,
        "positive_weight": 3.370202392225359,
        "pair_weight": 0,
        "max_length": 2048,
        "precision": "BF16 autocast FP32 loss/weights",
        "state": frozen["state"],
        **info,
    }
    write_json(destination / "training_config.json", config)
    write_json(destination / "alignment.json", audits)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=0.01)
    if command == "profile":
        groups = (items, [b for b in items if b["examples"][0]["record_id"].startswith("r6:")])
        results = []
        for group in groups:
            item = max(group, key=lambda b: max(len(e["input_ids"]) for e in b["examples"]))
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.reset_peak_memory_stats()
            start = time.perf_counter()
            loss, _, _ = batch_loss(model, tokenizer, item, 0)
            assert torch.isfinite(loss)
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            assert torch.isfinite(norm)
            optimizer.step()
            torch.cuda.synchronize()
            results.append(
                {
                    "seconds": time.perf_counter() - start,
                    "loss": float(loss.detach()),
                    "tokens": [len(e["input_ids"]) for e in item["examples"]],
                    **memory_report(),
                }
            )
        result = {
            "profile": results,
            "gate_passed": all(r["cuda_peak_allocated_bytes"] < 11 * 2**30 for r in results),
        }
    else:
        scheduler = get_linear_schedule_with_warmup(
            optimizer, int(info["optimizer_steps"] * 0.1), info["optimizer_steps"]
        )
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        logs = []
        for step, item in enumerate(items, 1):
            loss, ce, _ = batch_loss(model, tokenizer, item, 0)
            assert torch.isfinite(loss)
            (loss / 4).backward()
            logs.append({"step": step, "ce": float(ce.detach())})
            if step % 4 == 0:
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
                assert torch.isfinite(norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
            if step % 200 == 0:
                print(
                    f"R6{arm} seed{seed} {step}/{len(items)} CE={float(ce.detach()):.5f}",
                    flush=True,
                )
        torch.cuda.synchronize()
        seconds = time.perf_counter() - start
        model.save_pretrained(destination / "best", safe_serialization=True)
        tokenizer.save_pretrained(destination / "best")
        write_json(destination / "batch_log.json", logs)
        result = {
            "training_seconds": seconds,
            "batches_completed": len(items),
            "checkpoint_sha256": digest(destination / "best/model.safetensors"),
            **memory_report(),
        }
    assert state() == frozen["state"]
    write_json(destination / "result.json", result)
    print(result, flush=True)


def evaluate(method, seed):
    from grounded_hallucination_lab.research.span_model import encode_records, load_model
    from grounded_hallucination_lab.research.stable_inference import score

    frozen = checked()
    if method in FROZEN_MODELS:
        path, expected = FROZEN_MODELS[method]
        label = method
    else:
        path = BASE / f"{method}s{seed}/best"
        expected = read_json(path.parent / "result.json")["checkpoint_sha256"]
        label = f"{method}s{seed}"
    assert digest(path / "model.safetensors") == expected
    model, tokenizer = load_model(path)
    for dataset in DATASETS:
        records, design = data(dataset)
        assert content_hash(records, design) == frozen["datasets"][dataset]["sha256"]
        examples, metadata = encode_records(records, tokenizer, 2048, False)
        rows, runtime = score(model, tokenizer, examples, metadata)
        assert [r["id"] for r in rows] == [r.id for r in records]
        write_rows(PRED / f"{label}_{dataset}.jsonl", rows)
        report = {
            "method": label,
            "dataset": dataset,
            "checkpoint_sha256": expected,
            "metrics": metrics(
                [gold_label(r) for r in records], [r["score"] for r in rows], THRESHOLD
            ),
            "localization": span_metrics(records, rows),
            "runtime": runtime,
        }
        write_json(REPORT / f"{label}_{dataset}.json", report)
        print(f"{label} {dataset} macroF1={report['metrics']['macro_f1']:.6f}", flush=True)
    assert checked()["state"] == frozen["state"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("freeze", "profile", "train", "evaluate"))
    parser.add_argument("--arm", choices=("A", "B"), default="A")
    parser.add_argument("--method", choices=("B7", "R5A", "R6A", "R6B"), default="B7")
    parser.add_argument("--seed", type=int, choices=(42, 43, 44), default=42)
    args = parser.parse_args()
    if args.command == "freeze":
        freeze()
    elif args.command == "evaluate":
        evaluate(args.method, args.seed)
    else:
        train(args.command, args.arm, args.seed)
