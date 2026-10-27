"""One preregistered paired continuation experiment; inner partitions only."""

import argparse
import time
from dataclasses import asdict

import numpy as np

from grounded_hallucination_lab.data.ragtruth import gold_label
from grounded_hallucination_lab.eval.selection import select_threshold
from grounded_hallucination_lab.research.counterfactual import training_records
from grounded_hallucination_lab.research.metrics import metrics
from grounded_hallucination_lab.research.protocol import (
    OUT,
    ROOT,
    digest,
    inner_split,
    load_parts,
    partition,
    preserve_code,
    provenance,
    read_json,
    write_json,
    write_rows,
)

DEST = ROOT / "models/phase2b_resume"
REPORT = ROOT / "reports/phase2b_resume"
PRED = ROOT / "predictions/phase2b_resume"


def data(mode):
    parts = load_parts()
    fit, val = inner_split(parts.train)
    manifest = read_json(ROOT / "configs/counterfactual_acceptance.json")
    rows, synthetic, audit = training_records(fit, manifest, mode)
    if {r.source_id for r in rows} & {r.source_id for r in val}:
        raise ValueError("synthetic/replay parents crossed inner-validation boundary")
    return rows, synthetic, audit


def prepare():
    for mode in ("C0", "C1"):
        rows, synthetic, audit = data(mode)
        audit["training_response_ids"] = [r.id for r in rows]
        audit["training_sources"] = sorted({r.source_id for r in rows})
        audit["positive_responses"] = sum(gold_label(r) for r in rows)
        audit["mean_answer_characters"] = float(np.mean([len(r.response) for r in rows]))
        audit["annotated_character_count"] = sum(
            a.end - a.start for r in rows for a in r.annotations
        )
        audit["provenance"] = provenance()
        write_json(REPORT / f"{mode}_data.json", audit)
        if mode == "C1":
            write_json(
                PRED / "counterfactual_training_examples.json", [asdict(r) for r in synthetic]
            )
    print(
        {
            "natural": audit["natural_rows"],
            "extra": audit["extra_rows"],
            "synthetic_parents": len(audit["parents"]),
            "sources": audit["synthetic_sources"],
        }
    )


def profile():
    import torch
    import torch.nn.functional as functional

    from grounded_hallucination_lab.research.span_model import collate, encode_records, load_model
    from grounded_hallucination_lab.research.verifiers import memory_report

    rows, synthetic, _ = data("C1")
    model, tokenizer = load_model(OUT / "span_inner_s42/best", seed=42)
    model.gradient_checkpointing_enable()
    model.train()
    encoded, metadata = encode_records(rows, tokenizer, 2048, False, training=True)
    syn, _ = encode_records(synthetic, tokenizer, 2048, False, training=True)
    longest = max(encoded, key=lambda r: len(r["input_ids"]))
    synthetic_longest = max(syn, key=lambda r: len(r["input_ids"]))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=0.01)
    positive_weight = read_json(OUT / "span_inner_s42/training_config.json")["positive_weight"]
    weights = torch.tensor([1.0, positive_weight], device="cuda")
    times = []
    torch.cuda.reset_peak_memory_stats()
    for example in [longest, longest, longest, synthetic_longest]:
        batch = collate([example], tokenizer.pad_token_id)
        torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(
                input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]
            ).logits
            loss_tokens = functional.cross_entropy(
                logits.float().reshape(-1, 2),
                batch["labels"].reshape(-1),
                weight=weights,
                ignore_index=-100,
                reduction="none",
            ).reshape_as(batch["labels"])
            loss = (loss_tokens * batch["loss_weights"]).sum() / batch["loss_weights"].sum()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        times.append(time.perf_counter() - start)
    result = {
        "status": "passed",
        "examples": len(rows),
        "windows": len(encoded),
        "maximum_window_tokens": len(longest["input_ids"]),
        "maximum_synthetic_window_tokens": len(synthetic_longest["input_ids"]),
        "answer_coverage_min": min(m["answer_coverage"] for m in metadata),
        "evidence_coverage_min": min(m["evidence_coverage"] for m in metadata),
        "step_seconds": times,
        "conservative_one_epoch_seconds": float(np.median(times[1:3])) * len(encoded),
        "provenance": provenance(),
        **memory_report(),
    }
    write_json(REPORT / "counterfactual_profile.json", result)
    print(result, flush=True)


def fit(mode):
    from grounded_hallucination_lab.research.span_model import train

    rows, _, _ = data(mode)
    previous = read_json(OUT / "span_inner_s42/training_config.json")
    result = train(
        rows,
        [],
        DEST / f"{mode}_inner_s42",
        epochs=1,
        seed=42,
        max_length=2048,
        microbatch=1,
        answer_only=False,
        select_epoch=False,
        initial_checkpoint=OUT / "span_inner_s42/best",
        learning_rate=1e-5,
        positive_weight_override=previous["positive_weight"],
    )
    print({"method": mode, "seconds": result["training_seconds_including_validation"]}, flush=True)


def score_inner(mode):
    from grounded_hallucination_lab.research.span_model import encode_records, load_model, score

    records = partition(load_parts(), "inner_val")
    folder = DEST / f"{mode}_inner_s42"
    config = read_json(folder / "training_config.json")
    if {r.source_id for r in records} & set(config["train_sources"]):
        raise ValueError("inner-validation sources entered training")
    model, tokenizer = load_model(folder / "best")
    examples, metadata = encode_records(records, tokenizer, 2048, False)
    rows, runtime = score(model, tokenizer, examples, metadata, batch_size=1)
    labels, risks = [gold_label(r) for r in records], [r["score"] for r in rows]
    selection = select_threshold(labels, risks)
    path = PRED / f"{mode}_inner_val.jsonl"
    write_rows(path, rows)
    report = {
        "method": mode,
        "partition": "inner_val",
        "row_count": len(rows),
        "inner_selection": selection,
        "inner_metrics": metrics(labels, risks, selection["threshold"]),
        "runtime": runtime,
        "provenance": provenance(),
        "prediction_sha256": digest(path),
        "checkpoint_sha256": digest(folder / "best/model.safetensors"),
    }
    write_json(PRED / f"{mode}_inner_val.meta.json", report)
    print(report["inner_metrics"], flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["prepare", "profile", "train", "score"])
    parser.add_argument("--method", choices=["C0", "C1"], default="C0")
    args = parser.parse_args()
    before = preserve_code()
    if args.command == "prepare":
        prepare()
    elif args.command == "profile":
        profile()
    elif args.command == "train":
        fit(args.method)
    else:
        score_inner(args.method)
    if provenance() != before:
        raise ValueError("research code/config changed during experiment")


if __name__ == "__main__":
    main()
