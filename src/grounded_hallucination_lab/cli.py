"""Offline audit and dev-only majority evaluation. Network fetch is a separate script."""

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import time
import tomllib
from collections import Counter
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from grounded_hallucination_lab.baselines.simple import MajorityBaseline
from grounded_hallucination_lab.data.exports import EXPORT_PARTITIONS, export_labeled
from grounded_hallucination_lab.data.isolation import isolation_report
from grounded_hallucination_lab.data.ragtruth import (
    DataValidationError,
    LabelPolicy,
    Record,
    gold_label,
    load_ragtruth,
)
from grounded_hallucination_lab.data.splits import grouped_split
from grounded_hallucination_lab.eval.metrics import binary_metrics


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _provenance(directory: Path) -> dict[str, Any]:
    hashes = {name: _sha256(directory / name) for name in ("source_info.jsonl", "response.jsonl")}
    manifest_path = directory / "manifest.json"
    manifest = (
        json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else None
    )
    if manifest_path.exists():
        if not isinstance(manifest, dict) or not isinstance(manifest.get("files"), dict):
            raise ValueError("manifest must contain a files object")
        for name, digest in hashes.items():
            entry = manifest["files"].get(name)
            if not isinstance(entry, dict) or entry.get("sha256") != digest:
                raise ValueError(f"manifest SHA256 mismatch: {name}; fetch a consistent snapshot")
    fixture_path = directory / "dataset_metadata.json"
    fixture = (
        json.loads(fixture_path.read_text(encoding="utf-8")) if fixture_path.exists() else None
    )
    if fixture_path.exists() and not isinstance(fixture, dict):
        raise ValueError("dataset_metadata must be an object")
    kind = "synthetic_fixture" if fixture and fixture.get("synthetic") is True else "local_files"
    if manifest:
        kind = "ragtruth_snapshot_with_local_manifest"
    return {"kind": kind, "sha256": hashes, "manifest": manifest, "fixture": fixture}


def _environment() -> dict[str, Any]:
    root = Path(__file__).resolve().parents[2]
    package = Path(__file__).resolve().parent
    source_hashes = {
        str(path.relative_to(package.parent)).replace("\\", "/"): _sha256(path)
        for path in sorted(package.rglob("*.py"))
    }
    revision = None
    dirty = None
    try:
        command = ["git", "-c", f"safe.directory={root.as_posix()}", "-C", str(root)]
        result = subprocess.run(
            [*command, "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        revision = result.stdout.strip() if result.returncode == 0 else None
        status = subprocess.run(
            [*command, "status", "--porcelain"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        dirty = bool(status.stdout.strip()) if status.returncode == 0 else None
    except (OSError, subprocess.TimeoutExpired):
        pass
    packages = {}
    for name in (
        "grounded-hallucination-lab",
        "pytest",
        "ruff",
        "setuptools",
        "numpy",
        "scipy",
        "scikit-learn",
        "joblib",
        "threadpoolctl",
    ):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "logical_cpus": os.cpu_count(),
        "packages": packages,
        "git_revision": revision,
        "git_dirty": dirty,
        "source_sha256": source_hashes,
    }


def _summary(records: list[Record], policy: LabelPolicy) -> dict[str, Any]:
    included = [(r, gold_label(r, policy)) for r in records]
    return {
        "n_before_quality_filter": len(records),
        "n_included": sum(y is not None for _, y in included),
        "excluded_quality_count": sum(y is None for _, y in included),
        "class_counts": {str(label): sum(y == label for _, y in included) for label in (0, 1)},
        "sources": len({r.source_id for r in records}),
        "models": dict(Counter(r.model for r in records)),
        "quality": dict(Counter(r.quality for r in records)),
        "annotation_types": dict(Counter(a.label_type for r in records for a in r.annotations)),
        "implicit_true_spans": sum(a.implicit_true for r in records for a in r.annotations),
        "due_to_null_spans": sum(a.due_to_null for r in records for a in r.annotations),
        "implicit_only_responses": sum(
            bool(r.annotations) and all(a.implicit_true for a in r.annotations) for r in records
        ),
        "empty_responses": sum(not r.response.strip() for r in records),
        "response_characters": _length_summary([len(r.response) for r in records]),
        "evidence_characters": _length_summary([len(r.evidence) for r in records]),
    }


def _length_summary(values: list[int]) -> dict[str, int | float | None]:
    return {
        "n": len(values),
        "min": min(values) if values else None,
        "max": max(values) if values else None,
        "mean": sum(values) / len(values) if values else None,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    config = {
        "seed": 42,
        "dev_fraction": 0.2,
        "calibration_fraction": 0.2,
        "calibration_seed": 42,
        "implicit_true": "include",
        "quality": "all",
    }
    if args.config:
        supplied = tomllib.loads(args.config.read_text(encoding="utf-8"))
        if set(supplied) - set(config):
            raise ValueError(f"unknown config keys: {sorted(set(supplied) - set(config))}")
        config.update(supplied)
    for key in config:
        value = getattr(args, key)
        if value is not None:
            config[key] = value
    for key in ("seed", "calibration_seed"):
        if type(config[key]) is not int:
            raise ValueError(f"{key} must be an integer")
    for key in ("dev_fraction", "calibration_fraction"):
        if type(config[key]) not in (int, float):
            raise ValueError(f"{key} must be numeric")
    for output in (args.output, args.processed_output):
        if output and output.resolve().is_relative_to(args.data_dir.resolve()):
            raise ValueError("output paths must be outside the raw input directory")
    if args.processed_output and args.processed_output.resolve() == args.output.resolve():
        raise ValueError("report and processed output must be different files")
    policy = LabelPolicy(config["implicit_true"], config["quality"])
    provenance = _provenance(args.data_dir)
    dataset = load_ragtruth(args.data_dir)
    if dataset.diagnostics["official_source_overlap"]:
        raise ValueError("official split source overlap detected; no development access allowed")
    parts = grouped_split(
        dataset.records,
        config["dev_fraction"],
        config["seed"],
        calibration_fraction=config["calibration_fraction"],
        calibration_seed=config["calibration_seed"],
    )
    partition_summaries = {}
    for name, rows in parts.items():
        # Never map labels or expose annotation-derived quality statistics for test.
        summary = (
            _summary(rows, policy)
            if name != "test"
            else {
                "n_before_quality_filter": len(rows),
                "sources": len({r.source_id for r in rows}),
                "class_counts": None,
                "label_access": "withheld until authorized final evaluation",
            }
        )
        partition_summaries[name] = {
            **summary,
            "response_ids": sorted(r.id for r in rows),
            "source_ids": sorted({r.source_id for r in rows}),
        }
    report = {
        "schema_version": 2,
        "command": args.command,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "dataset": provenance,
        "environment": _environment(),
        "config": config,
        "audit": dataset.diagnostics,
        "official_train_summary": _summary(
            [r for r in dataset.records if r.split == "train"],
            policy,
        ),
        "split_protocol": {
            "version": "phase1.1",
            "dev": "unchanged Phase 1: SHA256(seed:source_id), ceil(n * dev_fraction)",
            "calibration": "SHA256(calibration:seed:source_id) on former train; ceil(n * fraction)",
            "roles": {
                "train": "fit",
                "calibration": "threshold and hyperparameter selection",
                "dev": "evaluate frozen configurations",
                "test": "authorized final only",
            },
        },
        "partitions": partition_summaries,
        "isolation": isolation_report(parts),
        "test_reserved": {
            "responses": len(parts.test),
            "sources": len({r.source_id for r in parts.test}),
            "class_counts": None,
            "labels_sealed": True,
        },
        "test_usage": "structural validation and source-text duplicate audit only; labels sealed",
    }
    if args.command.startswith("lexical-"):
        from grounded_hallucination_lab.lexical_experiment import run_lexical

        report["input_preparation_seconds"] = time.perf_counter() - started
        return run_lexical(args, config, parts, report)
    if args.command == "baseline":
        train_pairs = [(r, gold_label(r, policy)) for r in parts.train]
        dev_pairs = [(r, gold_label(r, policy)) for r in parts.dev]
        train = [(r, y) for r, y in train_pairs if y is not None]
        dev = [(r, y) for r, y in dev_pairs if y is not None]
        model = MajorityBaseline.fit([y for _, y in train])
        predictions = model.predict(len(dev))
        report.update(
            {
                "baseline": {
                    "name": "majority",
                    "tie_break": 0,
                    "fit_partition": "train",
                    **asdict(model),
                },
                "evaluation_partition": "dev",
                "metrics": binary_metrics([y for _, y in dev], predictions),
                "dev_metrics_by_model": {
                    model_name: binary_metrics(
                        [y for r, y in dev if r.model == model_name],
                        [model.label for r, _ in dev if r.model == model_name],
                    )
                    for model_name in sorted({r.model for r, _ in dev})
                },
                "dev_predictions": [
                    {"id": r.id, "source_id": r.source_id, "gold": y, "prediction": prediction}
                    for (r, y), prediction in zip(dev, predictions, strict=True)
                ],
            }
        )
    if args.processed_output:
        exported = export_labeled(parts, args.processed_output, policy, args.export_partition)
        report["processed_output"] = {
            "path": str(args.processed_output),
            "sha256": _sha256(args.processed_output),
            **exported,
        }
    _write_json(args.output, report)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command", choices=("audit", "baseline", "lexical-calibrate", "lexical-evaluate")
    )
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--processed-output", type=Path)
    parser.add_argument(
        "--export-partition",
        choices=EXPORT_PARTITIONS,
        default="train-calibration",
        help="default: train+calibration only; dev requires explicit selection",
    )
    parser.add_argument("--config", type=Path)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--dev-fraction", type=float)
    parser.add_argument("--calibration-fraction", type=float)
    parser.add_argument("--calibration-seed", type=int)
    parser.add_argument("--implicit-true", choices=("include", "exclude"))
    parser.add_argument("--quality", choices=("all", "good-only"))
    parser.add_argument("--lexical-config", type=Path)
    parser.add_argument("--artifacts-dir", type=Path, default=Path("models/phase2a"))
    parser.add_argument("--predictions-dir", type=Path, default=Path("predictions/phase2a"))
    args = parser.parse_args(argv)
    try:
        report = run(args)
    except (ValueError, OSError) as exc:
        diagnostics = (
            exc.diagnostics if isinstance(exc, DataValidationError) else {"error": str(exc)}
        )
        print(json.dumps(diagnostics, ensure_ascii=True), file=sys.stderr)
        return 2
    summary = {
        "report": str(args.output),
        "dataset_kind": report["dataset"]["kind"],
        "qa_responses": report["audit"]["selected_responses"],
    }
    if "metrics" in report:
        summary["metrics"] = report["metrics"]
    if "selection" in report:
        summary["selected_threshold"] = report["selection"]["threshold"]
        summary["calibration_macro_f1"] = report["selection"]["macro_f1"]
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
