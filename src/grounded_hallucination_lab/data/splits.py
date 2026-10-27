"""Deterministic group split using only official train source IDs."""

import hashlib
import math
from dataclasses import dataclass

from grounded_hallucination_lab.data.ragtruth import Record


@dataclass
class Partitions:
    train: list[Record]
    calibration: list[Record]
    dev: list[Record]
    test: list[Record]

    def items(self) -> tuple[tuple[str, list[Record]], ...]:
        return tuple(
            (name, getattr(self, name)) for name in ("train", "calibration", "dev", "test")
        )


def grouped_split(
    records: list[Record],
    dev_fraction: float = 0.2,
    seed: int = 42,
    *,
    calibration_fraction: float = 0.2,
    calibration_seed: int = 42,
) -> Partitions:
    """Preserve Phase 1 dev, then reserve calibration from its remaining train groups."""
    if not 0 < dev_fraction < 1:
        raise ValueError("dev_fraction must be between 0 and 1")
    if not 0 < calibration_fraction < 1:
        raise ValueError("calibration_fraction must be between 0 and 1")
    if type(seed) is not int or type(calibration_seed) is not int:
        raise ValueError("split seeds must be integers")
    if len({r.id for r in records}) != len(records):
        raise ValueError("duplicate response IDs in split input")
    if any(r.split not in {"train", "test"} for r in records):
        raise ValueError("unknown official split")
    train_sources = {r.source_id for r in records if r.split == "train"}
    test_sources = {r.source_id for r in records if r.split == "test"}
    overlap = train_sources & test_sources
    if overlap:
        raise ValueError(f"official train/test source overlap: {sorted(overlap)}")
    if len(train_sources) < 2:
        raise ValueError("at least two official train source groups are required")
    ordered = sorted(
        train_sources,
        key=lambda s: (
            hashlib.sha256(f"{seed}:{s}".encode()).hexdigest(),
            s,
        ),
    )
    count = min(len(ordered) - 1, max(1, math.ceil(len(ordered) * dev_fraction)))
    dev_sources = set(ordered[:count])
    remaining = train_sources - dev_sources
    if len(remaining) < 2:
        raise ValueError(
            "at least two train groups must remain after reserving dev for calibration"
        )
    calibration_order = sorted(
        remaining,
        key=lambda s: (
            hashlib.sha256(f"calibration:{calibration_seed}:{s}".encode()).hexdigest(),
            s,
        ),
    )
    calibration_count = min(
        len(remaining) - 1,
        max(1, math.ceil(len(remaining) * calibration_fraction)),
    )
    calibration_sources = set(calibration_order[:calibration_count])
    return Partitions(
        train=[
            r
            for r in records
            if r.split == "train" and r.source_id not in dev_sources | calibration_sources
        ],
        calibration=[
            r for r in records if r.split == "train" and r.source_id in calibration_sources
        ],
        dev=[r for r in records if r.split == "train" and r.source_id in dev_sources],
        test=[r for r in records if r.split == "test"],
    )
