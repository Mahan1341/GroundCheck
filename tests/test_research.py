import os
import re
from dataclasses import replace
from types import SimpleNamespace

import pytest

from grounded_hallucination_lab.data.splits import grouped_split
from grounded_hallucination_lab.research.metrics import bootstrap, merge_ranges, precision_point
from grounded_hallucination_lab.research.protocol import InferenceInput, cv_split, inner_split
from grounded_hallucination_lab.research.segments import clean_segments
from grounded_hallucination_lab.research.span_model import token_targets, windows


class TinyTokenizer:
    cls_token_id, sep_token_id, pad_token_id = 1, 2, 0

    def __call__(self, text, **kwargs):
        offsets = [(m.start(), m.end()) for m in re.finditer(r"\S+", text)]
        return {"input_ids": list(range(3, len(offsets) + 3)), "offset_mapping": offsets}


def test_inner_sources_deterministic_no_outer_overlap(dataset):
    parts = grouped_split(dataset.records)
    fit, val = inner_split(parts.train)
    assert not {r.source_id for r in fit} & {r.source_id for r in val}
    fit2, val2 = inner_split(list(reversed(parts.train)))
    assert {r.id for r in fit} == {r.id for r in fit2}
    assert {r.id for r in val} == {r.id for r in val2}
    with pytest.raises(ValueError, match="test"):
        inner_split(parts.test)


def test_grouped_cv_membership(dataset):
    parts = grouped_split(dataset.records)
    seen = []
    for fold in range(3):
        fit, val = cv_split(parts.train, fold)
        assert not {r.source_id for r in fit} & {r.source_id for r in val}
        seen.extend(r.id for r in val)
    assert sorted(seen) == sorted(r.id for r in parts.train)


def test_inference_allowlist_and_test_guard(dataset):
    record = next(r for r in dataset.records if r.split != "test")
    view = InferenceInput.from_record(record)
    assert (
        InferenceInput.from_record(
            replace(record, annotations=(), model="SECRET", quality="truncated")
        )
        == view
    )
    assert not hasattr(view, "annotations") and not hasattr(view, "model")
    assert view.lexical_record().annotations == ()
    with pytest.raises(ValueError, match="test"):
        InferenceInput.from_record(replace(record, split="test"))


def test_extractive_cleaner_keeps_facts_in_mixed_refusal():
    text = "1. Paris is in France. Unable to answer. However, it opened in 1990; it did not close."
    spans, drops = clean_segments(text)
    assert spans[0].text == "Paris is in France."
    assert any("1990" in s.text for s in spans)
    assert any("did not close" in s.text for s in spans)
    assert all(s.text == text[s.start : s.end] for s in spans)
    assert len(drops) == 2


def test_token_targets_partial_unicode_and_empty_offsets():
    assert token_targets([(0, 0), (0, 3), (4, 7), (8, 10)], [(2, 5)]) == [-100, 1, 1, 0]
    assert token_targets([(0, 2)], []) == [0]


def test_cleaner_keeps_standalone_numeric_claim_and_mixed_refusal_fact():
    assert [s.text for s in clean_segments("5.")[0]] == ["5."]
    text = "Unable to answer based on given passages. The bridge opened in 1991."
    spans, drops = clean_segments(text)
    assert [s.text for s in spans] == ["The bridge opened in 1991."]
    assert len(drops) == 1


def test_answer_mask_windows_and_loss_multiplicity():
    view = InferenceInput("a", "s", "question", "trusted evidence", " ".join(["word"] * 180))
    result, info = windows(view, TinyTokenizer(), max_length=70, spans=[(0, 4)])
    assert len(result) > 1 and info["answer_coverage"] == info["evidence_coverage"] == 1
    counts = {}
    for row in result:
        assert len(row["input_ids"]) <= 70
        for index, label, weight in zip(
            row["answer_indices"], row["labels"], row["loss_weights"], strict=True
        ):
            if index < 0:
                assert label == -100 and weight == 0
            else:
                counts[index] = counts.get(index, 0) + weight
    assert len(counts) == 180
    assert all(value == pytest.approx(1) for value in counts.values())
    unlabelled, _ = windows(view, TinyTokenizer(), max_length=70)
    assert all(y == -100 for row in unlabelled for y in row["labels"])


def test_no_silent_evidence_truncation():
    view = InferenceInput("a", "s", "question", "word " * 100, "answer")
    with pytest.raises(ValueError, match="complete evidence does not fit"):
        windows(view, TinyTokenizer(), max_length=64)


def test_precision_point_rejects_empty_and_tiny_positive_sets():
    assert not precision_point([0, 1], [0.1, 0.9])["available"]
    point = precision_point([0] * 10 + [1] * 10, [0.1] * 10 + [0.9] * 10)
    assert point["available"] and point["threshold"] == 0.9
    assert point["calibration_metrics"]["positive_class"]["recall"] == 1


def test_span_merge_definition():
    assert merge_ranges([(0, 3), (4, 7), (9, 10)], "one two. x") == [(0, 7), (9, 10)]


def test_paired_cluster_bootstrap_identical_systems_have_zero_difference():
    systems = {name: ([0, 1, 1, 0], [0.1, 0.8, 0.9, 0.2]) for name in ("a", "b")}
    result = bootstrap([0, 1, 0, 1], ["s1", "s1", "s2", "s2"], systems, "a")
    assert result["resamples"] == 2000 and result["source_count"] == 2
    for value in result["difference_from_reference"]["b"].values():
        assert value["low"] == value["high"] == 0
        assert value["valid_resamples"] == 2000
    with pytest.raises(ValueError, match="2000"):
        bootstrap([0, 1], ["s1", "s2"], systems, "a", 100)


def test_frozen_research_rejects_changed_code_and_artifacts(tmp_path, monkeypatch):
    from grounded_hallucination_lab.research import run
    from grounded_hallucination_lab.research.protocol import digest, write_json

    monkeypatch.setattr(run, "OUT", tmp_path)
    monkeypatch.setattr(run, "ROOT", tmp_path)
    monkeypatch.setattr(run, "provenance", lambda: {"code": "original"})
    artifact = tmp_path / "weights.bin"
    artifact.write_bytes(b"original")
    write_json(
        tmp_path / "batch_freeze.json",
        {"provenance": {"code": "original"}, "artifacts": {"weights.bin": digest(artifact)}},
    )
    run.check_freeze()
    artifact.write_bytes(b"modified")
    with pytest.raises(ValueError, match="artifact changed"):
        run.check_freeze()

    monkeypatch.setattr(run, "provenance", lambda: {"code": "changed"})
    with pytest.raises(ValueError, match="mismatch"):
        run.check_freeze()


def test_heldout_scoring_rejects_unfrozen_batch_override(dataset, monkeypatch):
    from grounded_hallucination_lab.research import run

    monkeypatch.setattr(run, "provenance", lambda: {})
    monkeypatch.setattr(
        run, "check_freeze", lambda: {"methods": ["B1"], "inference_batch_sizes": {"B1": 16}}
    )
    args = SimpleNamespace(partition="calibration", method="B1", batch_size=8)
    with pytest.raises(ValueError, match="batch size"):
        run.score_stage(args, grouped_split(dataset.records))


@pytest.mark.skipif(
    os.environ.get("GROUNDCHECK_RESEARCH_MODELS") != "1", reason="opt-in local GPU models"
)
def test_real_nli_label_mapping_and_asymmetric_pairs():
    from grounded_hallucination_lab.research.verifiers import Verifier

    verifier = Verifier("nli", 2)
    rows, _ = verifier.predict(
        [
            ("Alice owns a blue car.", "Alice owns a car."),
            ("Alice owns a blue car.", "Alice does not own a car."),
        ]
    )
    assert set(rows[0]["probabilities"]) == {"entailment", "neutral", "contradiction"}
    assert rows[0]["support"] > rows[1]["support"]
