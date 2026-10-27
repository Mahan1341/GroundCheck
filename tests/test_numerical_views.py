"""R5 target alignment, semantic controls and sealed-label regression tests."""

import os
from dataclasses import replace

import pytest

from grounded_hallucination_lab.data.ragtruth import Annotation, Record, gold_label
from grounded_hallucination_lab.research.invariance import suite as old_suite
from grounded_hallucination_lab.research.numerical_views import canonical_record
from grounded_hallucination_lab.research.numerical_views_suite import suite


def record(text, span):
    a, b = span
    return Record(
        "test",
        "source",
        "QA",
        "fixture",
        "train",
        "good",
        text,
        "How long?",
        "The test lasted 900 seconds.",
        (Annotation(a, b, text[a:b], "Evident Conflict", True, True, {}),),
        {},
        {},
    )


@pytest.mark.parametrize(
    "text,phrase,expected",
    [
        ("It lasted twenty-four minutes. Done.", "twenty-four minutes", "24 minutes"),
        ("Length was two kilometers.", "two kilometers", "2000 meters"),
        ("It lasted 900 seconds.", "900 seconds", "15 minutes"),
        ("It lasted about two minutes.", "two minutes", "two minutes"),
    ],
)
def test_annotation_remaps_with_policy(text, phrase, expected):
    start = text.index(phrase)
    original = record(text, (start, start + len(phrase)))
    changed, audit = canonical_record(original)
    span = changed.annotations[0]
    assert span.text == changed.response[span.start : span.end] == expected
    assert span.implicit_true and span.due_to_null
    assert gold_label(original) == gold_label(changed) == 1
    assert audit["expanded_annotation_intervals"] == 0


def test_partial_replacement_expansion_is_explicit():
    changed, audit = canonical_record(record("two kilometers", (0, 2)))
    assert changed.annotations[0].text == "2000"
    assert audit["expanded_annotation_intervals"] == 1


def test_sealed_and_malformed_targets_rejected():
    r = record("two minutes", (0, 11))
    with pytest.raises(ValueError, match="test labels"):
        canonical_record(replace(r, split="test"))
    with pytest.raises(ValueError, match="annotation text"):
        canonical_record(replace(r, response="other value"))


def test_controls_have_balanced_crossed_truth_and_disjoint_parents():
    records, design = suite()
    assert len(records) == 672 and len({r.source_id for r in records}) == 12
    assert sum(gold_label(r) for r in records) == 336
    assert not {r.source_id for r in records} & {r.source_id for r in old_suite("fit")[0]}
    pairs = {}
    for r, d in zip(records, design, strict=True):
        pairs.setdefault(d["pair_id"], []).append(r)
        assert gold_label(r) == d["gold"] == int(d["answer_index"] != d["evidence_index"])
        assert all(r.response[a.start : a.end] == a.text for a in r.annotations)
    for pair in pairs.values():
        assert len(pair) == 2 and {gold_label(r) for r in pair} == {0, 1}
        assert pair[0].response == pair[1].response and pair[0].evidence != pair[1].evidence
    assert suite() == (records, design)


@pytest.mark.skipif(
    os.getenv("GROUNDCHECK_RESEARCH_MODELS") != "1", reason="local tokenizer opt-in"
)
def test_actual_tokenizer_retargeting_and_loss_mask():
    from transformers import AutoTokenizer

    from grounded_hallucination_lab.research.protocol import ROOT
    from grounded_hallucination_lab.research.span_model import encode_records, token_targets

    tokenizer = AutoTokenizer.from_pretrained(
        ROOT / "models/phase2b/span_inner_s42/best", local_files_only=True
    )
    r = record("It lasted twenty-four minutes. Correct ending.", (10, 29))
    transformed, _ = canonical_record(r)
    rows, meta = encode_records([transformed], tokenizer, 2048, False, training=True)
    expected = token_targets(
        meta[0]["offsets"], [(a.start, a.end) for a in transformed.annotations]
    )
    assert [
        v for i, v in zip(rows[0]["answer_indices"], rows[0]["labels"], strict=True) if i >= 0
    ] == expected
    assert sum(expected) > 0
    assert all(
        v == -100 and w == 0
        for i, v, w in zip(
            rows[0]["answer_indices"], rows[0]["labels"], rows[0]["loss_weights"], strict=True
        )
        if i < 0
    )
