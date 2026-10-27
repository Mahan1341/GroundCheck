"""Strict reader for the original ParticleMedia/RAGTruth JSONL schema."""

import json
import math
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

LABEL_TYPES = {
    "Evident Baseless Info",
    "Subtle Baseless Info",
    "Evident Conflict",
    "Subtle Conflict",
}
QUALITIES = {"good", "incorrect_refusal", "truncated"}
TASKS = {"QA", "Summary", "Data2txt"}


class DataValidationError(ValueError):
    """No partial dataset is returned when any row is invalid."""

    def __init__(self, diagnostics: dict[str, Any]):
        self.diagnostics = diagnostics
        super().__init__(json.dumps(diagnostics, ensure_ascii=False))


@dataclass(frozen=True)
class Annotation:
    start: int
    end: int
    text: str
    label_type: str
    implicit_true: bool
    due_to_null: bool
    raw: dict[str, Any]


@dataclass(frozen=True)
class Record:
    id: str
    source_id: str
    task_type: str
    model: str
    split: str
    quality: str
    response: str
    question: str
    evidence: str
    annotations: tuple[Annotation, ...]
    raw_response: dict[str, Any]
    raw_source: dict[str, Any]
    labels_available: bool = True


@dataclass(frozen=True)
class LabelPolicy:
    implicit_true: Literal["include", "exclude"] = "include"
    quality: Literal["all", "good-only"] = "all"

    def __post_init__(self) -> None:
        if self.implicit_true not in {"include", "exclude"}:
            raise ValueError("implicit_true must be include or exclude")
        if self.quality not in {"all", "good-only"}:
            raise ValueError("quality must be all or good-only")


def gold_label(record: Record, policy: LabelPolicy | None = None) -> int | None:
    """1 means annotated unfaithfulness; None is an explicit quality exclusion."""
    if record.split == "test" or not record.labels_available:
        raise ValueError("test labels are sealed; final evaluation requires separate authorization")
    policy = policy or LabelPolicy()
    if policy.quality == "good-only" and record.quality != "good":
        return None
    return int(
        any(
            policy.implicit_true == "include" or not span.implicit_true
            for span in record.annotations
        )
    )


@dataclass
class Dataset:
    records: list[Record]
    diagnostics: dict[str, Any] = field(default_factory=dict)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read JSON objects; blank lines and malformed rows are errors with line numbers."""
    rows = []
    errors = []
    line_count = 0
    with path.open(encoding="utf-8-sig") as stream:
        for line_count, line in enumerate(stream, 1):
            try:
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError("expected a JSON object")
                rows.append(row)
            except (ValueError, TypeError) as exc:
                errors.append(f"{path.name}:{line_count}: {exc}")
    if errors or not rows:
        raise DataValidationError(
            {
                "file": str(path),
                "rows_read": line_count,
                "valid_rows": len(rows),
                "error_count": len(errors) or 1,
                "errors": errors[:20] or ["empty file"],
            }
        )
    return rows


def _string(row: dict, key: str, *, nonempty: bool = True) -> str:
    value = row.get(key)
    if not isinstance(value, str) or (nonempty and not value.strip()):
        raise ValueError(f"{key}: expected {'nonempty ' if nonempty else ''}string")
    return value


def _source(row: dict) -> None:
    _string(row, "source_id")
    _string(row, "source")
    _string(row, "prompt")
    task = _string(row, "task_type")
    if task not in TASKS:
        raise ValueError(f"unknown task_type: {task}")
    info = row.get("source_info")
    if task == "Summary":
        _string(row, "source_info")
    elif not isinstance(info, dict):
        raise ValueError("source_info must be an object for QA/Data2txt")
    elif task == "QA":
        _string(info, "question")
        _string(info, "passages")


def _annotations(row: dict) -> tuple[Annotation, ...]:
    labels = row.get("labels")
    if not isinstance(labels, list):
        raise ValueError("labels must be a list; null is not a negative label")
    result = []
    for index, label in enumerate(labels):
        if not isinstance(label, dict):
            raise ValueError(f"labels[{index}] must be an object")
        start, end = label.get("start"), label.get("end")
        if type(start) is not int or type(end) is not int:
            raise ValueError(f"labels[{index}]: start/end must be integer character offsets")
        if not 0 <= start < end <= len(row["response"]):
            raise ValueError(f"labels[{index}]: invalid half-open span [{start}, {end})")
        text = _string(label, "text")
        if row["response"][start:end] != text:
            raise ValueError(f"labels[{index}]: text does not match response[start:end]")
        kind = _string(label, "label_type")
        if kind not in LABEL_TYPES:
            raise ValueError(f"labels[{index}]: unknown label_type {kind}")
        for flag in ("implicit_true", "due_to_null"):
            if flag in label and type(label[flag]) is not bool:
                raise ValueError(f"labels[{index}]: {flag} must be boolean")
        result.append(
            Annotation(
                start,
                end,
                text,
                kind,
                label.get("implicit_true", False),
                label.get("due_to_null", False),
                label,
            )
        )
    return tuple(result)


def load_ragtruth(directory: Path) -> Dataset:
    """Validate raw rows; return QA with official test annotations/quality sealed.

    This is the development reader, not a final-evaluation interface. Raw files
    remain intact for a future separately authorized evaluator.
    """
    sources = read_jsonl(directory / "source_info.jsonl")
    responses = read_jsonl(directory / "response.jsonl")
    source_map: dict[str, dict] = {}
    errors = []
    for line, row in enumerate(sources, 1):
        try:
            _source(row)
            if row["source_id"] in source_map:
                raise ValueError(f"duplicate source_id {row['source_id']}")
            source_map[row["source_id"]] = row
        except ValueError as exc:
            errors.append(f"source_info.jsonl:{line}: {exc}")
    seen = set()
    records = []
    split_sources: dict[str, set[str]] = {"train": set(), "test": set()}
    used_sources = set()
    valid_responses = 0
    span_count = 0
    for line, row in enumerate(responses, 1):
        try:
            rid = _string(row, "id")
            if rid in seen:
                raise ValueError(f"duplicate response id {rid}")
            seen.add(rid)
            sid = _string(row, "source_id")
            if sid not in source_map:
                raise ValueError(f"missing or invalid source_id {sid}")
            for key in ("model", "response", "split", "quality"):
                _string(row, key, nonempty=key != "response")
            if row["split"] not in split_sources:
                raise ValueError(f"unknown split {row['split']}")
            if row["quality"] not in QUALITIES:
                raise ValueError(f"unknown quality {row['quality']}")
            temperature = row.get("temperature")
            if (
                type(temperature) not in (int, float)
                or not math.isfinite(temperature)
                or temperature < 0
            ):
                raise ValueError("temperature must be a finite nonnegative number")
            annotations = _annotations(row)
            source = source_map[sid]
            used_sources.add(sid)
            split_sources[row["split"]].add(sid)
            span_count += len(annotations)
            valid_responses += 1
            if source["task_type"] == "QA":
                info = source["source_info"]
                is_test = row["split"] == "test"
                # Allowlist test metadata: deleting only `labels` would retain
                # annotation-derived fields or unknown nested review metadata.
                safe_response = (
                    {
                        key: row[key]
                        for key in ("id", "source_id", "model", "temperature", "split", "response")
                    }
                    if is_test
                    else row
                )
                safe_source = (
                    {
                        "source_id": sid,
                        "task_type": "QA",
                        "source": source["source"],
                        "prompt": source["prompt"],
                        "source_info": {"question": info["question"], "passages": info["passages"]},
                    }
                    if is_test
                    else source
                )
                records.append(
                    Record(
                        rid,
                        sid,
                        "QA",
                        row["model"],
                        row["split"],
                        "withheld" if is_test else row["quality"],
                        row["response"],
                        info["question"],
                        info["passages"],
                        () if is_test else annotations,
                        safe_response,
                        safe_source,
                        not is_test,
                    )
                )
        except ValueError as exc:
            errors.append(f"response.jsonl:{line}: {exc}")
    diagnostics = {
        "source_rows": len(sources),
        "response_rows": len(responses),
        "valid_response_rows": valid_responses,
        "validated_spans": span_count,
        "error_count": len(errors),
        "errors": errors[:20],
        "selected_task": "QA",
        "selected_responses": len(records),
        "non_qa_responses": valid_responses - len(records),
        "source_task_counts": dict(Counter(s["task_type"] for s in source_map.values())),
        "unused_source_ids": sorted(set(source_map) - used_sources),
        "official_source_overlap": sorted(split_sources["train"] & split_sources["test"]),
        "qa_official_split_counts": dict(Counter(r.split for r in records)),
        "qa_source_count": len({r.source_id for r in records}),
        "qa_evidence_coverage": sum(bool(r.evidence.strip()) for r in records),
        "test_label_access": "sealed after structural validation; no per-example test labels",
    }
    if errors:
        raise DataValidationError(diagnostics)
    if not records:
        raise DataValidationError({**diagnostics, "error_count": 1, "errors": ["no QA rows"]})
    return Dataset(records, diagnostics)
