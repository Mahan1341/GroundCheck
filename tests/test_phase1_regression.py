"""Frozen Phase 1 membership/metric regression; no internet required."""

import hashlib
import json
import os
from dataclasses import replace
from pathlib import Path

import pytest

from grounded_hallucination_lab.cli import main
from grounded_hallucination_lab.data.splits import grouped_split

FIXTURE = Path(__file__).parent / "fixtures"
REFERENCE = json.loads((FIXTURE / "phase1_reference.json").read_text())


def test_real_membership_regression_from_id_only_snapshot(dataset):
    # IDs are real snapshot metadata; texts/labels are synthetic and never scored.
    template = dataset.records[0]
    records = [
        replace(template, id=rid, source_id=sid, split=group["split"])
        for sid, group in REFERENCE["source_groups"].items()
        for rid in group["response_ids"]
    ]
    parts = grouped_split(records)
    assert sorted(r.id for r in parts.dev) == REFERENCE["dev"]["response_ids"]
    assert sorted({r.source_id for r in parts.dev}) == REFERENCE["dev"]["source_ids"]
    assert (
        sorted(r.id for r in parts.train + parts.calibration)
        == (REFERENCE["legacy_train_response_ids"])
    )
    expected_test = {r.id for r in records if r.split == "test"}
    assert {r.id for r in parts.test} == expected_test
    assert [(len(rows), len({r.source_id for r in rows})) for _, rows in parts.items()] == [
        (3216, 536),
        (810, 135),
        (1008, 168),
        (900, 150),
    ]


def test_fixture_majority_metrics_remain_identical(tmp_path):
    out = tmp_path / "baseline.json"
    assert main(["baseline", "--data-dir", str(FIXTURE), "--output", str(out)]) == 0
    report = json.loads(out.read_text())
    assert report["metrics"] == REFERENCE["fixture"]["metrics"]
    assert report["partitions"]["dev"]["response_ids"] == REFERENCE["fixture"]["dev_response_ids"]


@pytest.mark.skipif(
    not os.environ.get("GROUNDCHECK_RAGTRUTH_DIR"),
    reason="opt-in real snapshot regression; offline CI uses metadata/fixtures",
)
def test_real_snapshot_metrics_and_membership(tmp_path):
    directory = Path(os.environ["GROUNDCHECK_RAGTRUTH_DIR"])
    for name, expected in REFERENCE["dataset_sha256"].items():
        assert hashlib.sha256((directory / name).read_bytes()).hexdigest() == expected
    out = tmp_path / "real-baseline.json"
    assert main(["baseline", "--data-dir", str(directory), "--output", str(out)]) == 0
    report = json.loads(out.read_text())
    assert report["metrics"] == REFERENCE["metrics"]
    assert report["dev_metrics_by_model"] == REFERENCE["dev_metrics_by_model"]
    assert report["partitions"]["dev"]["response_ids"] == REFERENCE["dev"]["response_ids"]
    assert report["isolation"]["all_ids_disjoint"]
    assert report["test_reserved"]["labels_sealed"]
