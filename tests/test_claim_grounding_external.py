"""Offline alignment tests: unresolved human spans never become negative gold."""

import importlib.util
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "experiments/scripts/claim_grounding_external.py"
SPEC = importlib.util.spec_from_file_location("claim_grounding_external", SCRIPT)
adapter = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(adapter)


def sentence(text, error=None, index=0):
    return {"index": index, "sentence": text, "annotation": [
        {"error_class": "No Error" if error is None else "EntE",
         "error_span": "" if error is None else error,
         "intrinsic/extrinsic": "n/a", "comment": "fixture"},
    ]}


def test_whitespace_normalization_preserves_original_offsets():
    response = "Alice\t has  12 boxes."
    result = adapter.align_summary(response, [sentence("Alice has 12 boxes.", "12 boxes")])
    assert result["localization_scorable"]
    assert result["label"] == 1
    start, end = result["gold_char_spans"][0]
    assert response[start:end] == "12 boxes"


def test_repeated_error_fragment_never_selects_first_occurrence():
    result = adapter.align_summary("Alice saw Alice.", [sentence("Alice saw Alice.", "Alice")])
    assert result["label"] == 1
    assert not result["localization_scorable"]
    assert result["gold_char_spans"] is None
    assert len(result["error_annotations"][0]["candidate_summary_offsets"]) == 2


def test_one_bad_span_makes_whole_summary_unscorable_but_detection_survives():
    result = adapter.align_summary("Alice came. Bob left.", [
        sentence("Alice came.", "Alice"), sentence("Bob left.", "Carol", 1),
    ])
    assert result["label"] == 1
    assert result["gold_char_spans"] is None
    assert not result["localization_scorable"]


def test_unannotated_summary_remainder_is_not_negative_token_gold():
    result = adapter.align_summary("Alice came. Extra claim.", [sentence("Alice came.")])
    assert result["label"] == 0
    assert not result["localization_scorable"]
    assert result["uncovered_nonwhitespace_characters"] > 0


def test_supported_complete_summary_has_explicit_empty_gold_spans():
    result = adapter.align_summary("Alice came.", [sentence("Alice came.")])
    assert result["label"] == 0
    assert result["localization_scorable"]
    assert result["gold_char_spans"] == []


def test_repeated_sentence_and_case_mismatch_remain_unresolved():
    result = adapter.align_summary("Alice came. Alice came.", [sentence("Alice came.", "Alice")])
    assert not result["localization_scorable"]
    assert adapter.match_fragment("alice", "Alice")["status"] == "unmatched"


def test_empty_annotation_is_unknown_not_supported():
    result = adapter.align_summary("Alice came.", [
        {"index": 0, "sentence": "Alice came.", "annotation": []},
    ])
    assert result["label"] is None
    assert result["gold_char_spans"] is None
