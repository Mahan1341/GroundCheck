"""Two-stage lexical experiment: calibration writes a freeze, evaluation only reads it."""

import hashlib
import json
import time
import tomllib
from collections import Counter
from dataclasses import asdict
from pathlib import Path

import joblib

from grounded_hallucination_lab.baselines.lexical import LexicalConfig, LexicalDetector
from grounded_hallucination_lab.baselines.simple import MajorityBaseline
from grounded_hallucination_lab.data.ragtruth import LabelPolicy, gold_label
from grounded_hallucination_lab.data.splits import Partitions
from grounded_hallucination_lab.eval.scored import scored_metrics, source_bootstrap
from grounded_hallucination_lab.eval.selection import select_threshold


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _write_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".jsonl.part")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(path)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _settings(path: Path | None) -> dict:
    values = {
        **asdict(LexicalConfig()),
        "bootstrap_resamples": 1000,
        "bootstrap_seed": 42,
        "error_examples_per_class": 15,
    }
    if path:
        supplied = tomllib.loads(path.read_text(encoding="utf-8"))
        if set(supplied) - set(values):
            raise ValueError("unknown lexical config keys")
        values.update(supplied)
    for key in ("bootstrap_resamples", "bootstrap_seed", "error_examples_per_class"):
        if type(values[key]) is not int:
            raise ValueError(f"{key} must be integer")
    if values["bootstrap_resamples"] < 1000 or values["error_examples_per_class"] < 15:
        raise ValueError("require >=1000 resamples and >=15 error examples per class")
    LexicalConfig(**{key: values[key] for key in asdict(LexicalConfig())})
    return values


def _membership(parts: Partitions) -> dict:
    return {
        name: {
            "response_ids": sorted(r.id for r in rows),
            "source_ids": sorted({r.source_id for r in rows}),
        }
        for name, rows in parts.items()
    }


def _scored_rows(detector, records, policy):
    rows = []
    for record in sorted(records, key=lambda r: r.id):
        label = gold_label(record, policy)
        if label is not None:
            rows.append(
                {
                    **detector.score(record),
                    "gold": label,
                    "model": record.model,
                    "question": record.question,
                    "response": record.response,
                }
            )
    return rows


def _errors(rows: list[dict], records: list, count: int) -> dict:
    record_map = {r.id: r for r in records}
    result = {
        "selection": "SHA256(errors:42:id), one per source first, then fill; no cherry picking",
        "automatic_tags_are_not_causal_adjudication": True,
    }
    for name, gold, pred in (("false_positives", 0, 1), ("false_negatives", 1, 0)):
        errors = [r for r in rows if r["gold"] == gold and r["prediction"] == pred]
        ordered = sorted(
            errors, key=lambda r: hashlib.sha256(f"errors:42:{r['id']}".encode()).hexdigest()
        )
        selected, selected_ids, used_sources = [], set(), set()
        for row in ordered:
            if len(selected) < count and row["source_id"] not in used_sources:
                selected.append(row)
                selected_ids.add(row["id"])
                used_sources.add(row["source_id"])
        for row in ordered:
            if len(selected) < count and row["id"] not in selected_ids:
                selected.append(row)
                selected_ids.add(row["id"])
        examples = []
        tags = Counter()
        for row in errors:
            if any(s["numbers"] for s in row["sentences"]):
                tags["number_present"] += 1
            if any(s["negation_present"] for s in row["sentences"]):
                tags["negation_present"] += 1
            if any(s["numbers_absent_from_retrieved"] for s in row["sentences"]):
                tags["number_absent_from_retrieved"] += 1
            if any(s["oov_only"] for s in row["sentences"]):
                tags["oov_sentence"] += 1
        for row in selected:
            record = record_map[row["id"]]
            examples.append(
                {
                    **row,
                    "evidence": record.evidence,
                    "quality": record.quality,
                    "gold_spans": [asdict(a) for a in record.annotations],
                }
            )
        result[name] = {
            "available": len(errors),
            "selected": len(examples),
            "automatic_tags_all_errors": dict(tags),
            "examples": examples,
        }
    return result


def run_lexical(args, config: dict, parts: Partitions, report: dict) -> dict:
    started = time.perf_counter()
    expected = {
        "seed": 42,
        "dev_fraction": 0.2,
        "calibration_fraction": 0.2,
        "calibration_seed": 42,
        "implicit_true": "include",
        "quality": "all",
    }
    if config != expected:
        raise ValueError("Phase 2A uses the frozen Phase 1.1 split and include/all policy")
    if args.processed_output:
        raise ValueError("use audit for exports; lexical stages do not export training data")
    settings = _settings(args.lexical_config)
    frozen_path = args.artifacts_dir / "frozen.json"
    model_path = args.artifacts_dir / "tfidf.joblib"
    for path in (args.artifacts_dir, args.predictions_dir):
        if path.resolve().is_relative_to(args.data_dir.resolve()):
            raise ValueError("artifacts must not be written inside raw input directory")
    if args.output.resolve().is_relative_to(args.artifacts_dir.resolve()):
        raise ValueError("public report must be outside the frozen model directory")
    if args.artifacts_dir.resolve() == args.predictions_dir.resolve():
        raise ValueError("models and predictions require separate directories")
    report["lexical_config"] = settings
    policy = LabelPolicy(config["implicit_true"], config["quality"])
    membership = _membership(parts)
    train_labels = [gold_label(r, policy) for r in parts.train]
    t0 = time.perf_counter()
    majority = MajorityBaseline.fit(train_labels)
    majority_fit_seconds = time.perf_counter() - t0
    report["timing_scope"] = "CPU wall time; scoring includes chunk/sentence diagnostics; no GPU"

    if args.command == "lexical-calibrate":
        if frozen_path.exists():
            raise ValueError(
                "freeze already exists; use a fresh artifacts directory for reproduction"
            )
        detector = LexicalDetector(
            LexicalConfig(**{key: settings[key] for key in asdict(LexicalConfig())})
        )
        t0 = time.perf_counter()
        detector.fit_train(parts)
        fit_seconds = time.perf_counter() - t0
        t0 = time.perf_counter()
        rows = _scored_rows(detector, parts.calibration, policy)
        score_seconds = time.perf_counter() - t0
        labels, scores = [r["gold"] for r in rows], [r["score"] for r in rows]
        t0 = time.perf_counter()
        selection = select_threshold(labels, scores)
        select_seconds = time.perf_counter() - t0
        for row in rows:
            row["prediction"] = int(row["score"] >= selection["threshold"])
        report.update(
            {
                "stage": "calibration_selection_frozen_before_dev",
                "selection": selection,
                "selection_metrics": scored_metrics(
                    labels, [r["prediction"] for r in rows], scores
                ),
                "fit": {
                    "sources": len(detector.fit_source_ids),
                    "source_ids": detector.fit_source_ids,
                    "chunks": detector.fit_chunk_count,
                    "vocabulary_size": len(detector.vectorizer.vocabulary_),
                    "model_fingerprint": detector.fingerprint(),
                    "documents": "unique train evidence chunks",
                },
                "majority": asdict(majority),
                "membership": membership,
                "runtime": {
                    "fit_seconds": fit_seconds,
                    "calibration_scoring_seconds": score_seconds,
                    "calibration_responses": len(rows),
                    "calibration_responses_per_second": len(rows) / score_seconds,
                    "selection_seconds": select_seconds,
                    "majority_fit_seconds": majority_fit_seconds,
                },
                "dev_evaluated": False,
                "test_evaluated": False,
            }
        )
        detector.clear_cache()
        args.artifacts_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(detector, model_path)
        report["model_sha256"] = _sha(model_path)
        t0 = time.perf_counter()
        _write_rows(args.predictions_dir / "calibration.jsonl", rows)
        report["runtime"]["prediction_write_seconds"] = time.perf_counter() - t0
        report["runtime"]["stage_seconds_before_report_write"] = time.perf_counter() - started
        _write_json(frozen_path, report)
        _write_json(args.output, report)
        return report

    frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
    checks = {
        "dataset": frozen["dataset"]["sha256"] == report["dataset"]["sha256"],
        "split_config": frozen["config"] == config,
        "membership": frozen["membership"] == membership,
        "lexical_config": frozen["lexical_config"] == settings,
        "code": frozen["environment"]["source_sha256"] == report["environment"]["source_sha256"],
        "dependencies": all(
            frozen["environment"]["packages"][name] == report["environment"]["packages"][name]
            for name in ("numpy", "scipy", "scikit-learn", "joblib", "threadpoolctl")
        ),
        "model": frozen["model_sha256"] == _sha(model_path),
        "stage": frozen["stage"] == "calibration_selection_frozen_before_dev"
        and not frozen["dev_evaluated"],
    }
    if not all(checks.values()):
        raise ValueError(f"frozen evaluation mismatch: {checks}")
    t0 = time.perf_counter()
    detector = joblib.load(model_path)
    load_seconds = time.perf_counter() - t0
    if detector.fingerprint() != frozen["fit"]["model_fingerprint"]:
        raise ValueError("TF-IDF learned statistics changed after calibration")
    threshold = frozen["selection"]["threshold"]
    t0 = time.perf_counter()
    rows = _scored_rows(detector, parts.dev, policy)
    for row in rows:
        row["prediction"] = int(row["score"] >= threshold)
    scoring_seconds = time.perf_counter() - t0
    labels, scores = [r["gold"] for r in rows], [r["score"] for r in rows]
    pred = [r["prediction"] for r in rows]
    t0 = time.perf_counter()
    majority_predictions = majority.predict(len(rows))
    majority_scores = [majority.train_positive_count / majority.train_size] * len(rows)
    majority_scoring_seconds = time.perf_counter() - t0
    t0 = time.perf_counter()
    metrics = {
        "lexical": scored_metrics(labels, pred, scores),
        "majority": scored_metrics(labels, majority_predictions, majority_scores),
    }
    intervals = source_bootstrap(
        labels,
        [r["source_id"] for r in rows],
        {"majority": (majority_predictions, majority_scores), "lexical": (pred, scores)},
        resamples=settings["bootstrap_resamples"],
        seed=settings["bootstrap_seed"],
    )
    metrics_seconds = time.perf_counter() - t0
    error_report = _errors(rows, parts.dev, settings["error_examples_per_class"])
    t0 = time.perf_counter()
    _write_rows(args.predictions_dir / "dev.jsonl", rows)
    _write_json(args.predictions_dir / "errors.json", error_report)
    write_seconds = time.perf_counter() - t0
    report.update(
        {
            "stage": "frozen_dev_evaluation",
            "frozen_sha256": _sha(frozen_path),
            "freeze_created_at_utc": frozen["created_at_utc"],
            "freeze_checks": checks,
            "threshold": threshold,
            "evaluation_partition": "dev",
            "metrics": metrics,
            "bootstrap": intervals,
            "calibration_report": str(frozen_path),
            "calibration_selection_metrics_are_not_dev": True,
            "majority": asdict(majority),
            "fit": frozen["fit"],
            "runtime": {
                "model_load_seconds": load_seconds,
                "dev_scoring_seconds": scoring_seconds,
                "dev_responses": len(rows),
                "dev_sources": len({r["source_id"] for r in rows}),
                "dev_responses_per_second": len(rows) / scoring_seconds,
                "majority_fit_seconds": majority_fit_seconds,
                "majority_scoring_seconds": majority_scoring_seconds,
                "majority_responses_per_second": len(rows) / majority_scoring_seconds,
                "metrics_and_bootstrap_seconds": metrics_seconds,
                "prediction_write_seconds": write_seconds,
                "stage_seconds_before_report_write": time.perf_counter() - started,
            },
            "error_analysis": {
                name: {k: v for k, v in error_report[name].items() if k != "examples"}
                for name in ("false_positives", "false_negatives")
            },
            "diagnostics_path": str(args.predictions_dir / "dev.jsonl"),
            "error_examples_path": str(args.predictions_dir / "errors.json"),
            "test_evaluated": False,
        }
    )
    _write_json(args.output, report)
    return report
