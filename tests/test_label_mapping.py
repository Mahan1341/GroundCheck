import pytest

from grounded_hallucination_lab.data.ragtruth import (
    DataValidationError,
    LabelPolicy,
    gold_label,
    load_ragtruth,
)


def test_empty_and_positive(dataset):
    records = {r.id: r for r in dataset.records}
    assert gold_label(records["r0a"]) == 0
    assert gold_label(records["r0b"]) == 1


def test_implicit_true_policy(dataset):
    record = next(r for r in dataset.records if r.id == "r1b")
    assert gold_label(record) == 1
    assert gold_label(record, LabelPolicy(implicit_true="exclude")) == 0


def test_due_to_null_is_not_a_null_annotation(dataset):
    record = next(r for r in dataset.records if r.id == "r2b")
    assert record.annotations[0].due_to_null
    assert gold_label(record) == 1


def test_quality_never_fabricates_positive(dataset):
    records = {r.id: r for r in dataset.records}
    assert records["r3a"].quality == "incorrect_refusal"
    assert gold_label(records["r3a"]) == 0
    assert gold_label(records["r3a"], LabelPolicy(quality="good-only")) is None
    assert records["r4b"].quality == "truncated"
    assert gold_label(records["r4b"]) == 1
    assert gold_label(records["r4b"], LabelPolicy(quality="good-only")) is None


@pytest.mark.parametrize(
    "field,value,match",
    [
        ("start", -1, "invalid half-open"),
        ("end", 999, "invalid half-open"),
        ("end", 0, "invalid half-open"),
        ("start", None, "integer character offsets"),
        ("start", True, "integer character offsets"),
        ("text", "wrong", "text does not match"),
        ("implicit_true", "false", "must be boolean"),
        ("due_to_null", None, "must be boolean"),
        ("label_type", "new category", "unknown label_type"),
    ],
)
def test_invalid_spans(mutable_data, field, value, match):
    rows, save = mutable_data
    rows["response.jsonl"][1]["labels"][0][field] = value
    with pytest.raises(DataValidationError, match=match):
        load_ragtruth(save())


def test_optional_flags_and_overlapping_unicode_spans(mutable_data):
    rows, save = mutable_data
    row = rows["response.jsonl"][1]
    row["response"] = "Я🙂 café"
    row["labels"] = [
        {"start": 0, "end": 2, "text": "Я🙂", "label_type": "Evident Conflict"},
        {"start": 1, "end": 3, "text": "🙂 ", "label_type": "Subtle Conflict"},
    ]
    record = load_ragtruth(save()).records[1]
    assert not record.annotations[0].implicit_true
    assert gold_label(record) == 1


def test_mixed_implicit_and_conflict(mutable_data):
    rows, save = mutable_data
    row = rows["response.jsonl"][1]
    row["labels"].append({**row["labels"][0], "implicit_true": True})
    record = load_ragtruth(save()).records[1]
    assert gold_label(record, LabelPolicy(implicit_true="exclude")) == 1


def test_invalid_policy():
    with pytest.raises(ValueError):
        LabelPolicy(implicit_true="unknown")
