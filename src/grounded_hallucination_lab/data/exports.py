"""Purpose-specific labeled exports; test is never an exportable partition."""

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from grounded_hallucination_lab.data.ragtruth import LabelPolicy, gold_label
from grounded_hallucination_lab.data.splits import Partitions

EXPORT_PARTITIONS = ("train-calibration", "train", "calibration", "dev")


def export_labeled(
    parts: Partitions,
    path: Path,
    policy: LabelPolicy,
    partition: str = "train-calibration",
) -> dict[str, Any]:
    """Export allowlisted model inputs and labels, with no nested raw annotations.

    Dev requires explicit selection and is for frozen evaluation only. Test is
    excluded altogether, including its response text and quality flags.
    """
    if partition not in EXPORT_PARTITIONS:
        raise ValueError("test is sealed; choose train-calibration, train, calibration or dev")
    names = ("train", "calibration") if partition == "train-calibration" else (partition,)
    rows = [(name, record) for name in names for record in getattr(parts, name)]
    if any(r.split != "train" or not r.labels_available for _, r in rows):
        raise ValueError("refusing export: official test or sealed labels in selected partition")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as stream:
            for name, record in rows:
                stream.write(
                    json.dumps(
                        {
                            "schema_version": 2,
                            "id": record.id,
                            "source_id": record.source_id,
                            "split": name,
                            "official_split": record.split,
                            "task_type": record.task_type,
                            "model": record.model,
                            "question": record.question,
                            "evidence": record.evidence,
                            "response": record.response,
                            "quality": record.quality,
                            "gold_label": gold_label(record, policy),
                            "label_policy": asdict(policy),
                        },
                        ensure_ascii=False,
                        allow_nan=False,
                    )
                    + "\n"
                )
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    return {
        "rows": len(rows),
        "partitions": list(names),
        "schema_version": 2,
        "official_test_rows": 0,
        "raw_annotation_fields_exported": False,
        "purpose": "frozen_evaluation_only" if partition == "dev" else "fit_or_calibration",
    }
