"""Offline inference-contract tests; actual pretrained inference is explicitly opt-in."""

import json
import os
from pathlib import Path

import pytest

from grounded_hallucination_lab.inference import (
    B7Detector,
    file_hash,
    load_profile,
    main,
    predicted_spans,
    validate_checkpoint,
    validate_input,
)


def test_original_unicode_offsets_merge_only_whitespace():
    text = "α  twelve, cats"
    spans = predicted_spans(text, [[0, 1], [3, 9], [11, 15]], [.8, .7, .9], .5)
    assert spans == [{"start": 0, "end": 9, "score": .8, "text": "α  twelve"},
                     {"start": 11, "end": 15, "score": .9, "text": "cats"}]


def test_empty_no_positive_and_alignment_errors():
    assert predicted_spans("", [], [], .5) == []
    assert predicted_spans("ok", [[0, 2]], [.1], .5) == []
    with pytest.raises(ValueError, match="aligned"):
        predicted_spans("ok", [[0, 2]], [], .5)
    with pytest.raises(ValueError, match="Invalid"):
        predicted_spans("ok", [[0, 3]], [.8], .5)


def test_hash_guard_prevents_random_or_missing_weights(tmp_path):
    path = tmp_path / "model.safetensors"
    profile = {"files": {path.name: "bad"}}
    with pytest.raises(ValueError, match="missing"):
        validate_checkpoint(tmp_path, profile)
    path.write_bytes(b"offline dummy bytes, never loaded as model weights")
    with pytest.raises(ValueError, match="mismatch"):
        validate_checkpoint(tmp_path, profile)
    validate_checkpoint(tmp_path, {"files": {path.name: file_hash(path)}})


def test_invalid_input_and_missing_checkpoint_are_actionable(tmp_path, capsys):
    with pytest.raises(ValueError, match="evidence"):
        validate_input({"question": "q", "evidence": [], "answer": "a"})
    path = tmp_path / "input.json"
    path.write_text(json.dumps({"question": "q", "evidence": "e", "answer": "a"}))
    assert main(["--input", str(path), "--checkpoint", str(tmp_path / "absent")]) == 2
    assert "CHECKPOINT.md" in capsys.readouterr().err


def test_custom_profile_rejects_path_traversal_and_invalid_boundaries(tmp_path):
    profile = load_profile()
    path = tmp_path / "profile.json"
    profile["files"]["../private"] = "hash"
    path.write_text(json.dumps(profile))
    with pytest.raises(ValueError, match="five"):
        load_profile(path)
    profile = load_profile()
    profile["response_threshold"] = float("nan")
    path.write_text(json.dumps(profile))
    with pytest.raises(ValueError, match="response_threshold"):
        load_profile(path)


@pytest.mark.skipif(not os.environ.get("GROUNDCHECK_CHECKPOINT"), reason="explicit local B7 opt-in")
def test_real_checkpoint_original_offsets_and_batch_consistency():
    root = Path(__file__).resolve().parents[1]
    value = json.loads((root / "examples/input.json").read_text())
    detector = B7Detector(os.environ["GROUNDCHECK_CHECKPOINT"], "cpu")
    result = detector.predict(value, include_tokens=True)
    assert result == detector.predict(value, include_tokens=True)
    assert result["coverage"]["answer_coverage"] == 1
    for span in result["spans"]:
        assert value["answer"][span["start"]:span["end"]] == span["text"]
    assert detector.predict({**value, "answer": ""})["status"] == "no_answer"
