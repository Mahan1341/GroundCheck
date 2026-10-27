from dataclasses import replace
from itertools import combinations

import pytest

from grounded_hallucination_lab.data.splits import grouped_split


def test_disjoint_complete_deterministic(dataset):
    parts = grouped_split(dataset.records)
    groups = [{r.source_id for r in rows} for _, rows in parts.items()]
    assert all(not a & b for a, b in combinations(groups, 2))
    assert sum(len(rows) for _, rows in parts.items()) == len(dataset.records)
    assert all(r.split == "train" for r in parts.train + parts.calibration + parts.dev)
    assert {r.id for r in parts.test} == {r.id for r in dataset.records if r.split == "test"}
    reversed_parts = grouped_split(list(reversed(dataset.records)))
    for name, rows in parts.items():
        assert {r.id for r in rows} == {r.id for r in getattr(reversed_parts, name)}


def test_official_leak_is_rejected(dataset):
    records = dataset.records.copy()
    records[-1] = replace(records[-1], source_id=records[0].source_id)
    with pytest.raises(ValueError, match="official train/test source overlap"):
        grouped_split(records)


@pytest.mark.parametrize("fraction", [0, 1, -0.1, float("nan")])
def test_invalid_fraction(dataset, fraction):
    with pytest.raises(ValueError):
        grouped_split(dataset.records, fraction)


def test_too_few_groups(dataset):
    with pytest.raises(ValueError, match="at least two"):
        grouped_split(dataset.records[:2])


@pytest.mark.parametrize("fraction", [0, 1, -0.1, float("nan")])
def test_invalid_calibration_fraction(dataset, fraction):
    with pytest.raises(ValueError, match="calibration_fraction"):
        grouped_split(dataset.records, calibration_fraction=fraction)


def test_calibration_controls_never_change_dev_or_test(dataset):
    first = grouped_split(dataset.records)
    second = grouped_split(dataset.records, calibration_seed=99, calibration_fraction=0.5)
    for name in ("dev", "test"):
        assert {r.id for r in getattr(first, name)} == {r.id for r in getattr(second, name)}
    assert {r.id for r in first.train + first.calibration} == {
        r.id for r in second.train + second.calibration
    }


def test_two_train_groups_cannot_support_three_partitions(dataset):
    with pytest.raises(ValueError, match="after reserving dev"):
        grouped_split(dataset.records[:4])
