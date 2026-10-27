"""Explicit stages for the bounded research batch. No test mode and no remote inference."""

import argparse
import gc
import math
import time
from dataclasses import asdict

import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from grounded_hallucination_lab.baselines.lexical import LexicalDetector
from grounded_hallucination_lab.data.ragtruth import gold_label
from grounded_hallucination_lab.data.splits import Partitions
from grounded_hallucination_lab.eval.selection import select_threshold
from grounded_hallucination_lab.research.metrics import metrics, precision_point
from grounded_hallucination_lab.research.protocol import (
    OUT,
    PRED,
    REPORT,
    ROOT,
    InferenceInput,
    cv_split,
    digest,
    inner_split,
    load_parts,
    now,
    partition,
    preserve_code,
    provenance,
    read_json,
    read_rows,
    write_json,
    write_rows,
)
from grounded_hallucination_lab.research.segments import clean_segments

VERIFIERS = {
    "B1": ("nli", "old"),
    "B2": ("minicheck", "author"),
    "B3": ("hhem", "whole"),
    "B4": ("minicheck", "old"),
    "B5": ("minicheck", "joint"),
    "B6": ("minicheck", "clean"),
}


def inference_batch_size(method):
    if method == "B1":
        return 16
    if method in VERIFIERS:
        return 4
    if method.startswith(("B7", "B8")):
        return 2
    return None


def retriever(parts, inner):
    train, val = inner_split(parts.train) if inner else (parts.train, [])
    detector = LexicalDetector().fit_train(
        Partitions(train, parts.calibration, [*val, *parts.dev], parts.test)
    )
    return detector


def check_freeze():
    frozen = read_json(OUT / "batch_freeze.json")
    if frozen["provenance"] != provenance():
        raise ValueError("research freeze code/dependency/data mismatch; do not bypass")
    for name, value in frozen["artifacts"].items():
        if digest(ROOT / name) != value:
            raise ValueError(f"frozen artifact changed: {name}")
    return frozen


def score_stage(args, parts):
    started_provenance = provenance()
    if args.partition in ("calibration", "dev"):
        frozen = check_freeze()
        if args.method not in frozen["methods"]:
            raise ValueError("candidate not selected before calibration/dev")
        expected_batch = frozen["inference_batch_sizes"][args.method]
        if args.batch_size is not None and args.batch_size != expected_batch:
            raise ValueError("inference batch size differs from frozen configuration")
        if args.partition == "dev" and not (OUT / "thresholds.json").exists():
            raise ValueError("thresholds must be fixed before dev scoring")
    rows_input = partition(parts, args.partition)
    if args.limit:
        if args.partition not in ("inner_train", "inner_val", "train"):
            raise ValueError("cannot subsample held-out evaluation")
        rows_input = rows_input[: args.limit]
    name = args.method + (f"_smoke{args.limit}" if args.limit else "")
    target = PRED / f"{name}_{args.partition}.jsonl"
    if target.exists():
        raise FileExistsError(target)
    if args.method in VERIFIERS:
        from grounded_hallucination_lab.research.verifiers import Verifier, score_records

        kind, mode = VERIFIERS[args.method]
        detector = retriever(parts, args.partition.startswith("inner")) if mode == "old" else None
        model = Verifier(kind, args.batch_size)
        rows, runtime = score_records(
            model, [InferenceInput.from_record(r) for r in rows_input], mode, detector
        )
    elif args.method.startswith(("B7", "B8")):
        from grounded_hallucination_lab.research.span_model import encode_records, load_model, score

        seed = int(args.method.split("s")[-1])
        answer_only = args.method.startswith("B8")
        stage = "inner" if args.partition.startswith("inner") else "final"
        folder = OUT / f"{'answer' if answer_only else 'span'}_{stage}_s{seed}"
        config = read_json(folder / "training_config.json")
        model, tokenizer = load_model(folder / "best")
        encoding_start = time.perf_counter()
        examples, info = encode_records(rows_input, tokenizer, config["max_length"], answer_only)
        encoding_seconds = time.perf_counter() - encoding_start
        rows, runtime = score(model, tokenizer, examples, info, args.batch_size or 2)
        runtime["encoding_seconds"] = encoding_seconds
        runtime["pipeline_seconds_including_encoding"] = (
            runtime["pipeline_seconds"] + encoding_seconds
        )
        runtime["responses_per_second_including_encoding"] = (
            len(rows) / runtime["pipeline_seconds_including_encoding"]
        )
    elif args.method == "B9":
        stage = "inner" if args.partition.startswith("inner") else "final"
        model = joblib.load(OUT / f"{args.method}_{stage}.joblib")
        rows = [{"id": r.id, "source_id": r.source_id} for r in rows_input]
        features = [[math.log1p(len(r.response))] for r in rows_input]
        start = time.perf_counter()
        values = model.predict_proba(features)[:, 1]
        elapsed = time.perf_counter() - start
        for row, value in zip(rows, values, strict=True):
            row["score"] = float(value)
        import psutil

        mem = psutil.Process().memory_info()
        runtime = {
            "head_seconds": elapsed,
            "head_responses_per_second": len(rows) / elapsed,
            "process_peak_rss_bytes": getattr(mem, "peak_wset", mem.rss),
            "cuda_peak_allocated_bytes": 0,
        }
    else:
        raise ValueError(args.method)
    if provenance() != started_provenance:
        raise ValueError("source/dependencies changed during scoring; refusing artifact")
    write_rows(target, rows)
    labels = [gold_label(r) for r in rows_input]
    report = {
        "method": args.method,
        "partition": args.partition,
        "runtime": runtime,
        "row_count": len(rows),
        "provenance": provenance(),
        "prediction_sha256": digest(target),
    }
    if args.partition == "dev":
        report["thresholds_sha256"] = digest(OUT / "thresholds.json")
    if args.partition == "inner_val":
        selection = select_threshold(labels, [r["score"] for r in rows])
        report.update(
            {
                "inner_selection": selection,
                "inner_metrics": metrics(
                    labels, [r["score"] for r in rows], selection["threshold"]
                ),
            }
        )
    write_json(target.with_suffix(".meta.json"), report)
    print(
        {
            "method": args.method,
            "partition": args.partition,
            "runtime": runtime,
            "inner_metrics": report.get("inner_metrics"),
        },
        flush=True,
    )


def fit_simple(args, parts):
    rows_input = partition(parts, "inner_train" if args.stage == "inner" else "train")
    if args.method != "B9":
        raise ValueError("only B9 has a separate simple fitting stage")
    features = [[math.log1p(len(r.response))] for r in rows_input]
    target = OUT / f"{args.method}_{args.stage}.joblib"
    if target.exists():
        raise FileExistsError(target)
    model = make_pipeline(
        StandardScaler(), LogisticRegression(C=1.0, max_iter=1000, random_state=42)
    )
    model.fit(features, [gold_label(r) for r in rows_input])
    OUT.mkdir(exist_ok=True, parents=True)
    joblib.dump(model, target)
    write_json(
        target.with_suffix(".json"),
        {
            "train_ids": [r.id for r in rows_input],
            "method": args.method,
            "stage": args.stage,
            "provenance": provenance(),
            "sha256": digest(target),
        },
    )


def train_stage(args, parts):
    from grounded_hallucination_lab.research.span_model import train

    if args.stage == "inner":
        fit, val = inner_split(parts.train)
        epochs = args.epochs
    elif args.stage == "cv":
        fit, val = cv_split(parts.train, args.fold)
        epochs = args.epochs
    else:
        fit, val = parts.train, []
        config = read_json(OUT / "span_inner_s42/result.json")
        epochs = config["selected_epoch"]
    name = "answer" if args.answer_only else "span"
    suffix = f"cv{args.fold}" if args.stage == "cv" else args.stage
    result = train(
        sorted(fit, key=lambda r: r.id),
        sorted(val, key=lambda r: r.id),
        OUT / f"{name}_{suffix}_s{args.seed}",
        epochs=epochs,
        seed=args.seed,
        max_length=args.max_length,
        microbatch=args.batch_size or 1,
        answer_only=args.answer_only,
        select_epoch=args.stage == "inner" and not args.fixed_epochs,
    )
    print(
        {
            "selected_epoch": result["selected_epoch"],
            "seconds": result["training_seconds_including_validation"],
        }
    )


def prepare(parts):
    fit, val = inner_split(parts.train)
    write_json(
        REPORT / "split.json",
        {
            "created": now(),
            "outer": {
                name: {
                    "responses": len(rows),
                    "sources": len({r.source_id for r in rows}),
                    "response_ids": sorted(r.id for r in rows),
                    "source_ids": sorted({r.source_id for r in rows}),
                    "class_counts": {str(k): sum(gold_label(r) == k for r in rows) for k in (0, 1)}
                    if name != "test"
                    else "withheld",
                }
                for name, rows in parts.items()
            },
            "inner_train_ids": sorted(r.id for r in fit),
            "inner_val_ids": sorted(r.id for r in val),
            "inner_train_sources": sorted({r.source_id for r in fit}),
            "inner_val_sources": sorted({r.source_id for r in val}),
            "test_evaluated": False,
        },
    )
    detector = retriever(parts, True)
    # Fixed eight-source sample from inner-validation, independent of gold labels.
    sampled = sorted(val, key=lambda r: r.id)
    used, audit = set(), []
    for record in sampled:
        if record.source_id in used:
            continue
        used.add(record.source_id)
        view = InferenceInput.from_record(record)
        segments, dropped = clean_segments(view.response)
        audit.append(
            {
                **asdict(view),
                "retrieval": detector.score(view.lexical_record()),
                "clean_segments": [asdict(s) for s in segments],
                "dropped": dropped,
            }
        )
        if len(audit) == 8:
            break
    write_json(PRED / "train_retrieval_audit.json", audit)
    print({"inner_train": len(fit), "inner_validation": len(val), "manual_audit": len(audit)})


def validate_final_fits(parts, methods):
    selected = read_json(OUT / "span_inner_s42/result.json")["selected_epoch"]
    expected_ids = sorted(r.id for r in parts.train)
    expected_sources = sorted({r.source_id for r in parts.train})
    for method in methods:
        if not method.startswith(("B7", "B8")):
            continue
        seed = int(method.split("s")[-1])
        folder = OUT / f"{'answer' if method.startswith('B8') else 'span'}_final_s{seed}"
        for name in (
            "training_config.json",
            "result.json",
            "best/model.safetensors",
            "best/config.json",
        ):
            if not (folder / name).is_file():
                raise ValueError(f"incomplete final fit: {method}: {name}")
        config = read_json(folder / "training_config.json")
        result = read_json(folder / "result.json")
        valid = (
            config == result["config"]
            and config["seed"] == seed
            and config["answer_only"] == method.startswith("B8")
            and sorted(config["train_ids"]) == expected_ids
            and config["train_sources"] == expected_sources
            and not config["validation_ids"]
            and not config["select_epoch_on_validation"]
            and config["max_epochs"] == selected == result["selected_epoch"]
            and len(result["history"]) == selected
            and not any("inner_metrics" in h for h in result["history"])
        )
        if not valid:
            raise ValueError(f"final fit protocol mismatch: {method}")
    if "B9" in methods:
        metadata = read_json(OUT / "B9_final.json")
        if sorted(metadata["train_ids"]) != expected_ids or metadata["sha256"] != digest(
            OUT / "B9_final.joblib"
        ):
            raise ValueError("B9 final fit mismatch")


def freeze(args, parts):
    methods = args.methods.split(",")
    if not {"B1", "B2", "B3", "B4", "B5", "B6", "B7s42", "B7s43", "B7s44", "B8s42", "B9"} <= set(
        methods
    ):
        raise ValueError("missing prespecified controls or seed repeats")
    validate_final_fits(parts, methods)
    artifacts = {}
    for path in (ROOT / "models/pretrained").rglob("*"):
        if path.is_file() and path.suffix in (".json", ".safetensors", ".bin", ".model", ".py"):
            artifacts[str(path.relative_to(ROOT))] = digest(path)
    for path in OUT.rglob("*"):
        if path.is_file() and ("final" in str(path) or path.name.endswith("result.json")):
            artifacts[str(path.relative_to(ROOT))] = digest(path)
    for path in (ROOT / "models/nltk_data").rglob("*"):
        if path.is_file() and path.suffix != ".zip":
            artifacts[str(path.relative_to(ROOT))] = digest(path)
    write_json(
        OUT / "batch_freeze.json",
        {
            "created_at": now(),
            "methods": methods,
            "inference_batch_sizes": {m: inference_batch_size(m) for m in methods},
            "provenance": provenance(),
            "artifacts": artifacts,
            "selection_note": args.note,
            "label_policy": {"task": "QA", "implicit_true": "include", "quality": "all"},
            "decision_rule": "risk >= threshold; higher risk is more suspicious",
            "dev_evaluated": False,
            "test_evaluated": False,
        },
    )


def calibrate(parts):
    frozen = check_freeze()
    records = partition(parts, "calibration")
    labels = [gold_label(r) for r in records]
    result = {}
    historical = read_rows(ROOT / "predictions/phase2a/calibration.jsonl")
    if [r["id"] for r in historical] != [r.id for r in records]:
        raise ValueError("historical calibration IDs differ")
    historical_point = precision_point(labels, [r["score"] for r in historical])
    for method in frozen["methods"]:
        rows = read_rows(PRED / f"{method}_calibration.jsonl")
        meta = read_json(PRED / f"{method}_calibration.meta.json")
        if meta["provenance"] != frozen["provenance"] or meta["prediction_sha256"] != digest(
            PRED / f"{method}_calibration.jsonl"
        ):
            raise ValueError("calibration predictions do not match frozen code/artifact")
        if [r["id"] for r in rows] != [r.id for r in records]:
            raise ValueError("calibration IDs differ")
        scores = [r["score"] for r in rows]
        selection = select_threshold(labels, scores)
        result[method] = {
            "selection": selection,
            "selection_metrics": metrics(labels, scores, selection["threshold"]),
            "precision_point": precision_point(labels, scores),
            "prediction_hash": digest(PRED / f"{method}_calibration.jsonl"),
        }
    write_json(
        OUT / "thresholds.json",
        {
            "created_at": now(),
            "batch_freeze": digest(OUT / "batch_freeze.json"),
            "methods": result,
            "historical_precision_point": historical_point,
            "test_evaluated": False,
        },
    )
    print(
        {
            name: {
                "threshold": r["selection"]["threshold"],
                "calibration_macro_f1": r["selection"]["macro_f1"],
            }
            for name, r in result.items()
        }
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command",
        choices=["prepare", "score", "fit-simple", "train-span", "freeze", "calibrate", "evaluate"],
    )
    parser.add_argument("--method", default="B1")
    parser.add_argument(
        "--partition",
        choices=["inner_train", "inner_val", "train", "calibration", "dev"],
        default="inner_val",
    )
    parser.add_argument("--stage", choices=["inner", "final", "cv"], default="inner")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--answer-only", action="store_true")
    parser.add_argument(
        "--fixed-epochs",
        action="store_true",
        help="inner matched control: retain final epoch, no best-epoch selection",
    )
    parser.add_argument("--limit", type=int)
    parser.add_argument("--methods", default="")
    parser.add_argument("--note", default="")
    args = parser.parse_args()
    started_provenance = preserve_code()
    parts = load_parts()
    if args.command == "prepare":
        prepare(parts)
    elif args.command == "score":
        score_stage(args, parts)
    elif args.command == "fit-simple":
        fit_simple(args, parts)
    elif args.command == "train-span":
        train_stage(args, parts)
    elif args.command == "freeze":
        freeze(args, parts)
    elif args.command == "calibrate":
        calibrate(parts)
    elif args.command == "evaluate":
        from grounded_hallucination_lab.research.evaluate import evaluate

        evaluate(parts)
    gc.collect()
    if provenance() != started_provenance:
        raise ValueError("source/dependencies changed during command; run is not valid")


if __name__ == "__main__":
    main()
