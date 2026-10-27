import pytest

from grounded_hallucination_lab.data.ragtruth import DataValidationError, load_ragtruth, read_jsonl


def test_fixture_join_and_exact_preservation(dataset):
    assert len(dataset.records) == 16
    assert dataset.diagnostics["error_count"] == 0
    assert dataset.diagnostics["official_source_overlap"] == []
    for record in dataset.records:
        assert record.evidence == record.raw_source["source_info"]["passages"]
        assert record.response == record.raw_response["response"]
        assert record.raw_source["source"] == "synthetic"
        for span in record.annotations:
            assert record.response[span.start : span.end] == span.text
            assert span.raw["meta"] == "Synthetic annotation for testing."


@pytest.mark.parametrize(
    "key,value,match",
    [
        ("source_id", "missing", "missing or invalid source_id"),
        ("id", 3, "id: expected"),
        ("labels", None, "labels must be a list"),
        ("labels", [None], "must be an object"),
        ("split", "dev", "unknown split"),
        ("quality", "unknown", "unknown quality"),
        ("temperature", float("nan"), "temperature"),
        ("response", None, "response: expected"),
    ],
)
def test_bad_response_fields(mutable_data, key, value, match):
    rows, save = mutable_data
    rows["response.jsonl"][0][key] = value
    with pytest.raises(DataValidationError, match=match) as error:
        load_ragtruth(save())
    assert error.value.diagnostics["error_count"] >= 1
    assert "response.jsonl:1" in str(error.value)


@pytest.mark.parametrize(
    "filename,key",
    [
        ("response.jsonl", "id"),
        ("source_info.jsonl", "source_id"),
    ],
)
def test_duplicate_ids(mutable_data, filename, key):
    rows, save = mutable_data
    rows[filename][1][key] = rows[filename][0][key]
    with pytest.raises(DataValidationError, match="duplicate"):
        load_ragtruth(save())


def test_evidence_not_silently_flattened(mutable_data):
    rows, save = mutable_data
    rows["source_info.jsonl"][0]["source_info"]["passages"] = ["passage"]
    with pytest.raises(DataValidationError, match="passages"):
        load_ragtruth(save())


@pytest.mark.parametrize("content", ["", "\n", "{}\n\n", "[]\n", "{broken}\n"])
def test_jsonl_errors(tmp_path, content):
    path = tmp_path / "bad.jsonl"
    path.write_text(content)
    with pytest.raises(DataValidationError):
        read_jsonl(path)


def test_valid_non_qa_is_counted_not_returned(mutable_data):
    rows, save = mutable_data
    rows["source_info.jsonl"][0].update(task_type="Summary", source_info="Source text.")
    data = load_ragtruth(save())
    assert data.diagnostics["non_qa_responses"] == 2
    assert len(data.records) == 14


def test_collects_multiple_errors(mutable_data):
    rows, save = mutable_data
    rows["response.jsonl"][0]["quality"] = "bad"
    rows["response.jsonl"][1]["split"] = "bad"
    with pytest.raises(DataValidationError) as exc:
        load_ragtruth(save())
    assert exc.value.diagnostics["error_count"] == 2
