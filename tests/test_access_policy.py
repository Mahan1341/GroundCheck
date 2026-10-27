import json

import pytest

from grounded_hallucination_lab.cli import main
from grounded_hallucination_lab.data.exports import export_labeled
from grounded_hallucination_lab.data.ragtruth import LabelPolicy, gold_label, load_ragtruth
from grounded_hallucination_lab.data.splits import grouped_split


def test_test_labels_and_nested_review_fields_are_sealed(mutable_data):
    rows, save = mutable_data
    test_sources = {r["source_id"] for r in rows["response.jsonl"] if r["split"] == "test"}
    for row in rows["response.jsonl"]:
        if row["split"] == "test":
            row["review"] = {"gold": "DO_NOT_EXPORT"}
    for source in rows["source_info.jsonl"]:
        if source["source_id"] in test_sources:
            source["review"] = {"gold": "DO_NOT_EXPORT"}
            source["source_info"]["review"] = "DO_NOT_EXPORT"
    dataset = load_ragtruth(save())
    for record in dataset.records:
        if record.split == "test":
            assert not record.labels_available
            assert record.annotations == ()
            assert record.quality == "withheld"
            assert "labels" not in record.raw_response
            assert "quality" not in record.raw_response
            assert "DO_NOT_EXPORT" not in json.dumps([record.raw_response, record.raw_source])
            with pytest.raises(ValueError, match="test labels are sealed"):
                gold_label(record)


@pytest.mark.parametrize("partition", ["train-calibration", "train", "calibration", "dev"])
def test_exports_are_purpose_specific_and_contain_no_test_rows(dataset, tmp_path, partition):
    parts = grouped_split(dataset.records)
    output = tmp_path / "export.jsonl"
    export_labeled(parts, output, LabelPolicy(), partition)
    rows = [json.loads(line) for line in output.read_text().splitlines()]
    names = {"train", "calibration"} if partition == "train-calibration" else {partition}
    assert {r["id"] for r in rows} == {
        r.id for name, records in parts.items() if name in names for r in records
    }
    assert {r["split"] for r in rows} == names
    assert all(r["official_split"] == "train" for r in rows)
    assert all(not {"annotations", "raw_response", "raw_source", "labels"} & r.keys() for r in rows)


def test_export_rejects_test_even_if_injected_into_train(dataset, tmp_path):
    parts = grouped_split(dataset.records)
    parts.train.append(parts.test[0])
    output = tmp_path / "existing.jsonl"
    output.write_text("existing safe content")
    with pytest.raises(ValueError, match="refusing export"):
        export_labeled(parts, output, LabelPolicy())
    assert output.read_text() == "existing safe content"


def test_test_export_is_not_available(dataset, tmp_path):
    with pytest.raises(ValueError, match="test is sealed"):
        export_labeled(
            grouped_split(dataset.records), tmp_path / "test.jsonl", LabelPolicy(), "test"
        )


def test_test_labels_cannot_affect_fit_or_dev_metrics(mutable_data, tmp_path):
    rows, save = mutable_data
    data_dir = save()
    out = tmp_path.parent / f"{tmp_path.name}-report.json"
    argv = ["baseline", "--data-dir", str(data_dir), "--output", str(out)]
    assert main(argv) == 0
    before = json.loads(out.read_text())
    for row in rows["response.jsonl"]:
        if row["split"] == "test":
            row["labels"] = []
            row["quality"] = "incorrect_refusal"
    save()
    assert main(argv) == 0
    after = json.loads(out.read_text())
    for key in ("baseline", "metrics", "dev_predictions", "partitions", "isolation"):
        assert before[key] == after[key]


def test_calibration_labels_are_not_used_to_fit_majority(mutable_data, tmp_path):
    rows, save = mutable_data
    data_dir = save()
    parts = grouped_split(load_ragtruth(data_dir).records)
    calibration_ids = {r.id for r in parts.calibration}
    out = tmp_path.parent / f"{tmp_path.name}-report.json"
    argv = ["baseline", "--data-dir", str(data_dir), "--output", str(out)]
    assert main(argv) == 0
    before = json.loads(out.read_text())
    for row in rows["response.jsonl"]:
        if row["id"] in calibration_ids:
            row["labels"] = [
                {
                    "start": 0,
                    "end": len(row["response"]),
                    "text": row["response"],
                    "label_type": "Evident Conflict",
                }
            ]
    save()
    assert main(argv) == 0
    after = json.loads(out.read_text())
    for key in ("baseline", "metrics", "dev_predictions"):
        assert before[key] == after[key]


@pytest.mark.parametrize("flag", ["--output", "--processed-output"])
def test_cli_cannot_overwrite_original_files(mutable_data, tmp_path, flag):
    _, save = mutable_data
    directory = save()
    original = directory / "response.jsonl"
    content = original.read_bytes()
    argv = ["audit", "--data-dir", str(directory), "--output", str(tmp_path.parent / "audit.json")]
    argv.extend([flag, str(original)])
    assert main(argv) == 2
    assert original.read_bytes() == content
