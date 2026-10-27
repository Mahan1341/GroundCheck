"""Matched natural permutation CE, frozen train-side controls, no outer loader."""

import argparse
import hashlib
import json
import random
import time
from dataclasses import asdict, replace

from grounded_hallucination_lab.data.ragtruth import Annotation, Record
from grounded_hallucination_lab.research.controls import controls
from grounded_hallucination_lab.research.pair_training import INITIAL, INITIAL_SHA
from grounded_hallucination_lab.research.passage_probe import evidence_view
from grounded_hallucination_lab.research.protocol import (
    ROOT,
    digest,
    preserve_code,
    read_json,
    read_rows,
    write_json,
    write_rows,
)

BASE = ROOT / "models/phase2b_r10"
PRED = ROOT / "predictions/phase2b_r10"
REPORT = ROOT / "reports/phase2b_r10"
THRESHOLD = 0.37612172961235046


def records(path):
    return [
        Record(**{**r, "annotations": tuple(Annotation(**a) for a in r["annotations"])})
        for r in read_rows(path)
    ]


def h(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def state():
    return {
        "source": preserve_code(),
        "proposal": digest(ROOT / "docs/R10_RESEARCH_PROPOSAL.md"),
        "input_manifest": digest(BASE / "inputs.json"),
    }


def old_integrity():
    manifest = read_json(ROOT / "models/phase2b_r9/R9_artifact_manifest.json")
    for name, spec in manifest["files"].items():
        assert digest(ROOT / name) == spec["sha256"], name
    for files in manifest["checkpoint_files"].values():
        for name, sha in files.items():
            assert digest(ROOT / name) == sha, name
    return len(manifest["files"])


def freeze():
    old = old_integrity()
    train, natural = records(BASE / "train_records.jsonl"), records(BASE / "natural_records.jsonl")
    assert len(train) == 2568 and len(natural) == 642
    assert all(r.split == "train" and r.labels_available for r in train + natural)
    assert not {r.source_id for r in train} & {r.source_id for r in natural}
    manifest = read_json(ROOT / "models/phase2b_r6/R6_artifact_manifest.json")
    for filename in ("transfer_records.jsonl", "transfer_design.json"):
        path = ROOT / "predictions/phase2b_r6" / filename
        assert digest(path) == manifest["files"][path.relative_to(ROOT).as_posix()]["sha256"]
    old_source = read_json(ROOT / "models/phase2b_r9/freeze_v4.json")["state"]["source"]["code"]
    path = ROOT / "src/grounded_hallucination_lab/research/controls.py"
    assert digest(path) == old_source[str(path.relative_to(ROOT))], "no new authored templates"
    rr, dd = controls()
    selected = [
        (r, d)
        for r, d in zip(rr, dd, strict=True)
        if d["condition"] in ("supported", "contradicted")
    ]
    write_rows(BASE / "r1_records.jsonl", [asdict(r) for r, _ in selected])
    design = []
    for r, d in selected:
        design.append(
            {
                **d,
                "source_id": r.source_id,
                "pair_id": f"{d['group']}:{d['padding_sentences']}:{d['position']}",
            }
        )
    write_json(BASE / "r1_design.json", design)
    assert len(selected) == 150
    blank_sources = sorted(
        {r.source_id for r in natural},
        key=lambda s: hashlib.sha256(("R10blank:42:" + s).encode()).hexdigest(),
    )[:16]
    write_json(BASE / "blank_sources.json", blank_sources)
    files = [
        BASE / n
        for n in (
            "train_records.jsonl",
            "natural_records.jsonl",
            "eligibility_frozen.json",
            "r1_records.jsonl",
            "r1_design.json",
            "blank_sources.json",
            "semantic_review_notes.json",
        )
    ]
    files += [
        ROOT / "predictions/phase2b_r6" / n
        for n in ("transfer_records.jsonl", "transfer_design.json", "B7_transfer.jsonl")
    ]
    files += [
        ROOT / "predictions/phase2b_r9_final" / f"B7_{c}.jsonl"
        for c in ("full", "rotate1", "rotate2")
    ]
    write_json(BASE / "inputs.json", {p.relative_to(ROOT).as_posix(): digest(p) for p in files})
    write_json(
        BASE / "freeze.json",
        {
            "state": state(),
            "old_R9_verified": old,
            "initial_sha256": INITIAL_SHA,
            "train_n": len(train),
            "natural_n": len(natural),
            "screening": read_json(BASE / "screening_summary.json"),
        },
    )
    print({"frozen": True, "old_verified": old}, flush=True)


def checked():
    f = read_json(BASE / "freeze.json")
    assert state() == f["state"], "R10 source/plan mismatch; do not bypass"
    for name, sha in read_json(BASE / "inputs.json").items():
        assert digest(ROOT / name) == sha, name
    assert digest(INITIAL / "model.safetensors") == INITIAL_SHA
    return f


def batches(tokenizer, seed, arm):
    from grounded_hallucination_lab.research.span_model import encode_records, token_targets

    rr = records(BASE / "train_records.jsonl")
    screen = read_json(BASE / "eligibility_frozen.json")["train"]
    changed = [
        replace(r, evidence=evidence_view(r.evidence, f"rotate{screen[r.source_id]['rotation']}"))
        if screen[r.source_id]["eligible"]
        else r
        for r in rr
    ]
    raw, meta = encode_records(rr, tokenizer, 2048, False, training=True)
    aug, other = encode_records(changed, tokenizer, 2048, False, training=True)
    assert len(raw) == len(aug) == len(rr) == 2568
    assert [m["offsets"] for m in meta] == [m["offsets"] for m in other]
    for r, a, b, m in zip(rr, raw, aug, meta, strict=True):
        targets = [
            [y for i, y in zip(row["answer_indices"], row["labels"], strict=True) if i >= 0]
            for row in (a, b)
        ]
        assert targets[0] == targets[1]
        assert targets[0] == token_targets(m["offsets"], [(x.start, x.end) for x in r.annotations])
        for row in (a, b):
            assert all(
                y == -100 and w == 0
                for i, y, w in zip(
                    row["answer_indices"], row["labels"], row["loss_weights"], strict=True
                )
                if i < 0
            )
    for m, n in zip(meta, other, strict=True):
        assert m["windows"] == n["windows"] == 1
        assert m["answer_coverage"] == m["evidence_coverage"] == 1
        assert n["answer_coverage"] == n["evidence_coverage"] == 1
    order = list(range(len(rr)))
    random.Random(seed).shuffle(order)
    items = [{"examples": [raw[i], raw[i] if arm == "A" else aug[i]], "masks": None} for i in order]
    info = {
        "train_ids": [r.id for r in rr],
        "train_sources": sorted({r.source_id for r in rr}),
        "order_sha256": h([rr[i].id for i in order]),
        "answer_targets_sha256": h([[y for y in row["labels"] if y >= 0] for row in raw]),
        "answer_offsets_sha256": h([m["offsets"] for m in meta]),
        "batches": len(items),
        "updates": len(items) // 4,
        "exposures": 2 * len(items),
        "changed_second_views": sum(
            a["input_ids"] != b["input_ids"] for a, b in zip(raw, aug, strict=True)
        ),
        "max_input_tokens": max(len(x["input_ids"]) for x in raw + aug),
        "length_differences": {
            str(d): sum(
                len(b["input_ids"]) - len(a["input_ids"]) == d
                for a, b in zip(raw, aug, strict=True)
            )
            for d in sorted(
                {len(b["input_ids"]) - len(a["input_ids"]) for a, b in zip(raw, aug, strict=True)}
            )
        },
        "coverage": "complete answer/evidence, no truncation; masked prefix/special tokens",
    }
    return items, info


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
        profiles = [read_json(BASE / f"profile{x}/result.json") for x in ("A", "B")]
        assert all(p["passed"] for p in profiles)
        assert sum(p["projected_fit_seconds"] for p in profiles) < 1800
    destination = BASE / (f"profile{arm}" if command == "profile" else f"R10{arm}s{seed}")
    destination.mkdir(parents=True, exist_ok=False)
    model, tok = load_model(INITIAL, seed)
    model.gradient_checkpointing_enable()
    items, info = batches(tok, seed, arm)
    cfg = {
        "arm": arm,
        "seed": seed,
        "initial_sha256": INITIAL_SHA,
        "lr": 1e-5,
        "weight_decay": 0.01,
        "positive_weight": 3.370202392225359,
        "microbatch": 2,
        "accumulation": 4,
        "warmup": 64,
        "epochs": 1,
        "precision": "BF16 autocast; FP32 weights/optimizer/loss; inference FP32",
        "state": frozen["state"],
        **info,
    }
    write_json(destination / "training_config.json", cfg)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=0.01)
    if command == "profile":
        ordered = sorted(items, key=lambda b: max(len(e["input_ids"]) for e in b["examples"]))
        timings = []
        for q in (1.0, 0.25, 0.5, 0.75, 1.0):
            item = ordered[min(len(ordered) - 1, int(q * len(ordered)))]
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.reset_peak_memory_stats()
            start = time.perf_counter()
            loss, _, _ = batch_loss(model, tok, item, 0)
            assert torch.isfinite(loss)
            loss.backward()
            assert torch.isfinite(torch.nn.utils.clip_grad_norm_(model.parameters(), 1))
            optimizer.step()
            torch.cuda.synchronize()
            timings.append(
                {
                    "quantile": q,
                    "seconds": time.perf_counter() - start,
                    "loss": float(loss.detach()),
                    **memory_report(),
                }
            )
        estimate = sum(t["seconds"] for t in timings[1:]) / 4 * len(items)
        result = {
            "timings": timings,
            "projected_fit_seconds": estimate,
            "passed": all(t["cuda_peak_allocated_bytes"] < 11 * 2**30 for t in timings),
        }
    else:
        scheduler = get_linear_schedule_with_warmup(optimizer, 64, info["updates"])
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        with (destination / "batch_log.jsonl").open("x", encoding="utf-8") as stream:
            for step, item in enumerate(items, 1):
                loss, ce, _ = batch_loss(model, tok, item, 0)
                assert torch.isfinite(loss)
                (loss / 4).backward()
                stream.write(json.dumps({"step": step, "ce": float(ce.detach())}) + "\n")
                if step % 4 == 0:
                    assert torch.isfinite(torch.nn.utils.clip_grad_norm_(model.parameters(), 1))
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                if step % 200 == 0:
                    stream.flush()
                    print(
                        f"R10{arm}s{seed} {step}/{len(items)} CE={float(ce.detach()):.5f}",
                        flush=True,
                    )
        torch.cuda.synchronize()
        seconds = time.perf_counter() - start
        model.save_pretrained(destination / "final", safe_serialization=True)
        tok.save_pretrained(destination / "final")
        result = {
            "training_seconds": seconds,
            "batches_completed": len(items),
            "updates_completed": info["updates"],
            "checkpoint_sha256": digest(destination / "final/model.safetensors"),
            **memory_report(),
        }
    assert state() == frozen["state"]
    write_json(destination / "result.json", result)
    print(result, flush=True)


def evaluate(method):
    from grounded_hallucination_lab.research.span_model import encode_records, load_model
    from grounded_hallucination_lab.research.stable_inference import score

    checked()
    if method == "B7":
        path, sha = INITIAL, INITIAL_SHA
    else:
        folder = BASE / method
        path = folder / "final"
        sha = read_json(folder / "result.json")["checkpoint_sha256"]
    assert digest(path / "model.safetensors") == sha
    model, tok = load_model(path)
    natural = records(BASE / "natural_records.jsonl")
    screen = read_json(BASE / "eligibility_frozen.json")["natural"]
    eligible = [r for r in natural if screen[r.source_id]["eligible"]]
    blank_sources = set(read_json(BASE / "blank_sources.json"))
    datasets = {
        "natural": natural,
        "rotate1": [replace(r, evidence=evidence_view(r.evidence, "rotate1")) for r in eligible],
        "rotate2": [replace(r, evidence=evidence_view(r.evidence, "rotate2")) for r in eligible],
        "r6": records(ROOT / "predictions/phase2b_r6/transfer_records.jsonl"),
        "r1": records(BASE / "r1_records.jsonl"),
        "blank": [replace(r, evidence="") for r in natural if r.source_id in blank_sources],
    }
    for name, rr in datasets.items():
        target = PRED / f"{method}_{name}.jsonl"
        assert not target.exists()
        if method == "B7" and name in ("natural", "rotate1", "rotate2", "r6"):
            old = (
                ROOT / "predictions/phase2b_r6/B7_transfer.jsonl"
                if name == "r6"
                else ROOT
                / f"predictions/phase2b_r9_final/B7_{'full' if name == 'natural' else name}.jsonl"
            )
            ids = {r.id for r in rr}
            rows = [r for r in read_rows(old) if r["id"] in ids]
            timing = {"reused_verified": old.relative_to(ROOT).as_posix(), "new_gpu": False}
        else:
            enc, meta = encode_records(rr, tok, 2048, False)
            assert all(m["answer_coverage"] == m["evidence_coverage"] == 1 for m in meta)
            rows, timing = score(model, tok, enc, meta)
        assert [p["id"] for p in rows] == [r.id for r in rr]
        write_rows(target, rows)
        write_json(REPORT / f"{method}_{name}_runtime.json", {"checkpoint_sha256": sha, **timing})
        print({"model": method, "dataset": name, "n": len(rows)}, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("freeze", "profile", "train", "evaluate", "verify-old"))
    parser.add_argument("--arm", choices=("A", "B"), default="A")
    parser.add_argument("--seed", type=int, choices=(42, 43, 44), default=42)
    parser.add_argument("--method", default="B7")
    args = parser.parse_args()
    if args.command == "freeze":
        freeze()
    elif args.command in ("profile", "train"):
        train(args.command, args.arm, args.seed)
    elif args.command == "evaluate":
        evaluate(args.method)
    else:
        print(old_integrity())
