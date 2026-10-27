from dataclasses import replace

import numpy as np
import pytest

from grounded_hallucination_lab.baselines.lexical import (
    LexicalConfig,
    LexicalDetector,
    chunks,
    sentences,
    tokenize,
)
from grounded_hallucination_lab.data.splits import grouped_split


def test_sentence_offsets_decimals_and_abbreviations():
    text = ' Dr. Smith paid 3.14 dollars. "No!"\nNext line. '
    spans = sentences(text)
    assert [s.text for s in spans] == ["Dr. Smith paid 3.14 dollars.", '"No!"', "Next line."]
    assert all(text[s.start : s.end] == s.text for s in spans)


def test_chunks_trace_passages_and_cover_long_context():
    text = "passage 1: " + " ".join(f"w{i}" for i in range(260)) + "\npassage 2: Last fact."
    result = chunks(text, "source", LexicalConfig())
    assert len(result) == 4
    assert all(text[c.start : c.end] == c.text for c in result)
    assert result[-1].passage_label == "2"
    assert "w259" in result[-2].text
    assert all(len(c.text.split()) <= 100 for c in result)
    assert set(" ".join(c.text for c in result).split()) >= {f"w{i}" for i in range(260)}


def test_tokenizer_preserves_numbers_and_negations():
    tokens = tokenize("It isn't safe; can't pay 1,000 or 3.14; 25% won't disappear.")
    assert tokens.count("not") == 3
    assert {"1000", "3.14", "25%", "safe"} <= set(tokens)


def test_fit_train_only_and_deduplicates_sources(dataset):
    parts = grouped_split(dataset.records)
    parts.calibration = [replace(r, evidence="forbiddencaltoken") for r in parts.calibration]
    parts.dev = [replace(r, evidence="forbiddendevtoken") for r in parts.dev]
    detector = LexicalDetector().fit_train(parts)
    assert not {"forbiddencaltoken", "forbiddendevtoken"} & detector.vectorizer.vocabulary_.keys()
    assert detector.fit_source_ids == sorted({r.source_id for r in parts.train})
    before = detector.fingerprint()
    detector.score(parts.calibration[0])
    assert detector.fingerprint() == before
    repeated = LexicalDetector().fit_train(parts)
    assert repeated.fingerprint() == before


def test_refuses_fit_group_leak(dataset):
    parts = grouped_split(dataset.records)
    parts.train.append(parts.dev[0])
    with pytest.raises(ValueError, match="disjoint"):
        LexicalDetector().fit_train(parts)


def test_empty_oov_and_test_inputs(dataset):
    parts = grouped_split(dataset.records)
    detector = LexicalDetector().fit_train(parts)
    base = parts.dev[0]
    assert detector.score(replace(base, response="  "))["score"] == 0
    assert detector.score(replace(base, evidence=""))["score"] == 1
    unknown = detector.score(replace(base, response="zqxxyy"))
    assert unknown["score"] == 1 and unknown["sentences"][0]["oov_only"]
    assert detector.score(replace(base, response="!!!"))["sentences"][0]["tokenless"]
    with pytest.raises(ValueError, match="test scoring"):
        detector.score(parts.test[0])


def test_retrieval_offsets_diagnostics_and_label_blindness(dataset):
    parts = grouped_split(dataset.records)
    detector = LexicalDetector().fit_train(parts)
    record = replace(parts.dev[0], response="The marker is not blue. It costs 12345 dollars.")
    result = detector.score(record)
    assert 0 <= result["score"] <= 1
    assert result["sentences"][0]["negation_present"]
    assert "12345" in result["sentences"][1]["numbers_absent_from_context"]
    for sentence in result["sentences"]:
        assert record.response[sentence["start"] : sentence["end"]] == sentence["text"]
        for passage in sentence["retrieved"]:
            assert record.evidence[passage["start"] : passage["end"]] == passage["text"]
    assert detector.score(replace(record, annotations=(), quality="truncated")) == result


def test_similarity_matches_normalized_dot_product(dataset):
    parts = grouped_split(dataset.records)
    detector = LexicalDetector().fit_train(parts)
    record = replace(parts.dev[0], response="The marker is blue.")
    result = detector.score(record)
    docs = chunks(record.evidence, record.source_id, detector.config)
    q = detector.vectorizer.transform([record.response])
    expected = (detector.vectorizer.transform([d.text for d in docs]) @ q.T).toarray().max()
    assert result["score"] == pytest.approx(1 - expected)
    assert np.isfinite(expected)


def test_numeric_passage_marker_is_not_evidence(dataset):
    detector = LexicalDetector().fit_train(grouped_split(dataset.records))
    record = replace(
        dataset.records[0], evidence="passage 99: It is blue.", response="It costs 99."
    )
    assert "99" in detector.score(record)["sentences"][0]["numbers_absent_from_context"]
