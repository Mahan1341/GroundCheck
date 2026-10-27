"""Matched R5 CE experiment. Local checkpoints, train-side data, no outer scoring."""

import argparse
import hashlib
import json
import random
import time
from dataclasses import asdict, replace

from grounded_hallucination_lab.data.ragtruth import gold_label
from grounded_hallucination_lab.research.invariance import suite as r3_suite
from grounded_hallucination_lab.research.metrics import metrics, span_metrics
from grounded_hallucination_lab.research.numeric_local import (
    normalize_numbers,
    project_probabilities,
)
from grounded_hallucination_lab.research.numeric_local_suite import suite as r4_suite
from grounded_hallucination_lab.research.numerical_views import canonical_record
from grounded_hallucination_lab.research.numerical_views_suite import suite as r5_suite
from grounded_hallucination_lab.research.pair_training import INITIAL, INITIAL_SHA
from grounded_hallucination_lab.research.protocol import (
    ROOT,
    digest,
    load_parts,
    partition,
    preserve_code,
    read_json,
    read_rows,
    write_json,
    write_rows,
)
from grounded_hallucination_lab.research.relations import content_hash

BASE = ROOT / "models/phase2b_r5"
PRED = ROOT / "predictions/phase2b_r5"
REPORT = ROOT / "reports/phase2b_r5"
THRESHOLD = 0.37612172961235046
DATASETS = ("natural", "old", "fresh", "new")
FROZEN_MODELS = {
    "B7": (INITIAL, INITIAL_SHA),
    "R3A": (
        ROOT / "models/phase2b_r3/R3As42/best",
        "a0116cda0d54be0a97fa70940b59c2ef1c734229c7fdea856dd52c64078e1d26",
    ),
}


def data(name):
    if name == "natural":
        return partition(load_parts(), "inner_val"), []
    if name == "old":
        return r3_suite("diagnostic")
    if name == "fresh":
        return r4_suite()
    if name == "new":
        return r5_suite()
    raise ValueError("only fixed inner controls allowed")


def train_records():
    return partition(load_parts(), "inner_train") + r3_suite("fit")[0]


def state():
    return {"source": preserve_code(), "plan_sha256": digest(ROOT / "EXPERIMENT_R5_PLAN.md")}


def checked():
    frozen = read_json(BASE / "freeze.json")
    assert frozen["state"] == state(), "source/plan changed; do not bypass frozen hashes"
    assert digest(INITIAL / "model.safetensors") == INITIAL_SHA
    return frozen


def freeze():
    old = read_json(ROOT / "models/phase2b_r4/R4_artifact_manifest.json")
    for name, spec in old["files"].items():
        assert digest(ROOT / name) == spec["sha256"], name
    train = train_records()
    sources = {r.source_id for r in train}
    result = {
        "state": state(),
        "R4_verified_files": len(old["files"]),
        "train_hash": content_hash(train, []),
        "datasets": {},
        "models": {},
    }
    for method, (path, expected) in FROZEN_MODELS.items():
        assert digest(path / "model.safetensors") == expected
        result["models"][method] = expected
    for name in DATASETS:
        records, design = data(name)
        assert not sources & {r.source_id for r in records}
        assert len({r.id for r in records}) == len(records)
        result["datasets"][name] = {
            "sha256": content_hash(records, design),
            "n": len(records),
            "positive": sum(gold_label(r) for r in records),
            "sources": sorted({r.source_id for r in records}),
        }
        write_rows(PRED / f"{name}_records.jsonl", [asdict(r) for r in records])
        write_json(PRED / f"{name}_design.json", design)
    write_json(BASE / "freeze.json", result)
    print(
        {
            "frozen": {k: v["n"] for k, v in result["datasets"].items()},
            "R4_files": len(old["files"]),
        },
        flush=True,
    )


def batches(tokenizer, seed, arm):
    from grounded_hallucination_lab.research.span_model import encode_records, token_targets

    records = train_records()
    assert content_hash(records, []) == checked()["train_hash"]
    canonical, audits = zip(*(canonical_record(r) for r in records), strict=True)
    raw, raw_meta = encode_records(records, tokenizer, 2048, False, training=True)
    canon, canon_meta = encode_records(canonical, tokenizer, 2048, False, training=True)
    assert all(m["windows"] == 1 for m in raw_meta + canon_meta), "unmatched window membership"
    assert len(raw) == len(canon) == len(records) == 3528
    for view, rows, metas in ((records, raw, raw_meta), (canonical, canon, canon_meta)):
        for record, row, meta in zip(view, rows, metas, strict=True):
            expected = token_targets(
                meta["offsets"], [(a.start, a.end) for a in record.annotations]
            )
            actual = [
                target
                for idx, target in zip(row["answer_indices"], row["labels"], strict=True)
                if idx >= 0
            ]
            assert actual == expected
            assert all(
                target == -100 and weight == 0
                for idx, target, weight in zip(
                    row["answer_indices"], row["labels"], row["loss_weights"], strict=True
                )
                if idx < 0
            )
            assert (
                meta["answer_coverage"] == meta["evidence_coverage"] == 1
                and meta["truncated_tokens"] == 0
            )
    order = list(range(len(raw)))
    random.Random(seed).shuffle(order)
    other = canon if arm == "B" else raw
    items = [{"examples": [raw[i], other[i]], "masks": None} for i in order]
    info = {
        "train_ids": [r.id for r in records],
        "train_sources": sorted({r.source_id for r in records}),
        "record_order_sha256": hashlib.sha256(
            json.dumps([records[i].id for i in order]).encode()
        ).hexdigest(),
        "batches": len(items),
        "optimizer_steps": len(items) // 4,
        "max_tokens": max(len(e["input_ids"]) for b in items for e in b["examples"]),
        "canonical_changed_answer": sum(bool(a["answer_edits"]) for a in audits),
        "canonical_changed_evidence": sum(bool(a["evidence_edits"]) for a in audits),
        "annotation_expansions": sum(a["expanded_annotation_intervals"] for a in audits),
        "raw_positive_tokens": sum(e["labels"].count(1) for e in raw),
        "canonical_positive_tokens": sum(e["labels"].count(1) for e in canon),
        "raw_answer_tokens": sum(m["answer_tokens"] for m in raw_meta),
        "canonical_answer_tokens": sum(m["answer_tokens"] for m in canon_meta),
        "targets_sha256": hashlib.sha256(
            json.dumps([[e["labels"] for e in b["examples"]] for b in items]).encode()
        ).hexdigest(),
        "coverage": "all evidence and answer tokens; prefix/special loss masked",
    }
    return items, info, list(audits)


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
    destination = BASE / (f"profile{arm}" if command == "profile" else f"R5{arm}s{seed}")
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
        groups = (items, [b for b in items if b["examples"][0]["record_id"].startswith("r3:")])
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
                    f"R5{arm} seed{seed} {step}/{len(items)} CE={float(ce.detach()):.5f}",
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
    from grounded_hallucination_lab.research.span_model import encode_records, load_model, score

    frozen = checked()
    symmetric = method == "R4sym"
    if method in ("R5A", "R5B"):
        path = BASE / f"{method}s{seed}/best"
        assert (
            digest(path / "model.safetensors")
            == read_json(path.parent / "result.json")["checkpoint_sha256"]
        )
        label = f"{method}s{seed}"
    else:
        path, expected = FROZEN_MODELS["R3A" if symmetric else method]
        assert digest(path / "model.safetensors") == expected
        label = method
    model, tokenizer = load_model(path)
    for dataset in DATASETS:
        records, design = data(dataset)
        assert content_hash(records, design) == frozen["datasets"][dataset]["sha256"]
        if method in ("B7", "R3A", "R4sym") and dataset != "new":
            oldname = (
                f"{'R3A' if symmetric else method}_{dataset}_{'symmetric' if symmetric else 'raw'}"
            )
            source = ROOT / "predictions/phase2b_r4" / f"{oldname}.jsonl"
            manifest = read_json(ROOT / "models/phase2b_r4/R4_artifact_manifest.json")
            assert (
                digest(source)
                == manifest["files"][str(source.relative_to(ROOT)).replace("\\", "/")]["sha256"]
            )
            rows = read_rows(source)
            runtime = {
                "reused_immutable": str(source.relative_to(ROOT)),
                "sha256": digest(source),
                "original_report": read_json(ROOT / "reports/phase2b_r4" / f"{oldname}.json"),
            }
        else:
            originals = None
            inference_records = records
            if symmetric:
                _, original_meta = encode_records(records, tokenizer, 2048, False)
                originals = {m["id"]: m for m in original_meta}
                inference_records = [
                    replace(
                        r,
                        evidence=normalize_numbers(r.evidence).text,
                        response=normalize_numbers(r.response).text,
                    )
                    for r in records
                ]
            examples, metadata = encode_records(inference_records, tokenizer, 2048, False)
            rows, runtime = score(model, tokenizer, examples, metadata, batch_size=2)
            if symmetric:
                write_rows(PRED / f"{label}_{dataset}_native.jsonl", rows)
                mapping = {r.id: normalize_numbers(r.response) for r in records}
                rows = [
                    {
                        **originals[r["id"]],
                        "score": r["score"],
                        "token_probabilities": project_probabilities(
                            originals[r["id"]]["offsets"],
                            mapping[r["id"]],
                            r["offsets"],
                            r["token_probabilities"],
                        ),
                    }
                    for r in rows
                ]
        assert [r["id"] for r in rows] == [r.id for r in records]
        write_rows(PRED / f"{label}_{dataset}.jsonl", rows)
        report = {
            "method": label,
            "dataset": dataset,
            "checkpoint_sha256": digest(path / "model.safetensors"),
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
    parser.add_argument("--method", choices=("B7", "R3A", "R4sym", "R5A", "R5B"), default="B7")
    parser.add_argument("--seed", type=int, choices=(42, 43, 44), default=42)
    args = parser.parse_args()
    if args.command == "freeze":
        freeze()
    elif args.command == "evaluate":
        evaluate(args.method, args.seed)
    else:
        train(args.command, args.arm, args.seed)
