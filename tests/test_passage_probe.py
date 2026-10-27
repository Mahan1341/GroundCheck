"""R9 controls remain label-free, traceable and sentence-coherent."""

import pytest

from grounded_hallucination_lab.research.passage_probe import (
    evidence_view,
    passage_ranges,
    pool_passages,
)


def rows():
    return [
        {"id": "a", "offsets": [[0, 2], [3, 5]], "token_probabilities": p}
        for p in ([0.1, 0.9], [0.8, 0.2], [0.6, 0.6])
    ]


def test_all_bytes_and_original_passage_identity_preserved():
    text = "passage 1:Å.\n\npassage 2:B!\n\npassage 3:C?\n\n"
    parts = [text[a:b] for a, b in passage_ranges(text)]
    assert evidence_view(text, "full") == text
    assert evidence_view(text, "rotate1") == parts[1] + parts[2] + parts[0]
    assert evidence_view(text, "rotate2") == parts[2] + parts[0] + parts[1]
    assert "".join(evidence_view(text, f"single{i}") for i in range(3)) == text


@pytest.mark.parametrize("text", ["", "No headers", "passage 1:A\npassage 2:B"])
def test_incomplete_passages_refused(text):
    with pytest.raises(ValueError):
        passage_ranges(text)


def test_sentence_pool_does_not_stitch_token_support():
    coherent, token, choices = pool_passages(rows(), [(0, 5)], text="ab cd")
    assert coherent["score"] == 0.6
    assert token["score"] == 0.2
    assert choices[0]["passage_index"] == 2
    assert coherent["offsets"] == rows()[0]["offsets"]


def test_separate_sentences_can_use_separate_passages():
    coherent, token, choices = pool_passages(rows(), [(0, 3), (3, 5)], text="ab cd")
    assert coherent["token_probabilities"] == token["token_probabilities"] == [0.1, 0.2]
    assert [c["passage_index"] for c in choices] == [0, 1]


def test_alignment_and_coverage_are_mandatory():
    for spans in ([(0, 3)], [(0, 5), (0, 5)]):
        with pytest.raises(ValueError):
            pool_passages(rows(), spans, text="ab cd")
    changed = rows()
    changed[1]["offsets"] = [[1, 2], [3, 5]]
    with pytest.raises(ValueError):
        pool_passages(changed, [(0, 5)], text="ab cd")


def test_tie_uses_original_passage_order():
    same = [rows()[0]] * 3
    assert pool_passages(same, [(0, 5)], text="ab cd")[2][0]["passage_index"] == 0


def test_labels_cannot_affect_pooling():
    changed = [{**r, "gold": 1, "annotations": [(0, 5)]} for r in rows()]
    assert pool_passages(changed, [(0, 5)], text="ab cd")[0]["score"] == 0.6


def test_bpe_leading_space_belongs_to_next_sentence_content():
    predictions = [
        {"id": "a", "offsets": [[0, 4], [4, 10]], "token_probabilities": p}
        for p in ([0.1, 0.9], [0.9, 0.1], [0.7, 0.7])
    ]
    coherent, _, choices = pool_passages(predictions, [(0, 5), (5, 10)], text="One. Next.")
    assert coherent["token_probabilities"] == [0.1, 0.1]
    assert [c["passage_index"] for c in choices] == [0, 1]


def test_token_spanning_two_sentences_content_is_not_silently_assigned():
    predictions = [{"id": "a", "offsets": [[0, 6]], "token_probabilities": [0.1]}] * 3
    with pytest.raises(ValueError, match="crosses"):
        pool_passages(predictions, [(0, 3), (3, 6)], text="Ab.Cd.")


def test_closing_quote_token_can_cross_punkt_boundary_without_splitting_words():
    predictions = [
        {"id": "a", "offsets": [[0, 2], [2, 4]], "token_probabilities": p}
        for p in ([0.1, 0.9], [0.9, 0.1], [0.7, 0.7])
    ]
    coherent, _, _ = pool_passages(predictions, [(0, 1), (1, 4)], text='?"Hi')
    assert coherent["token_probabilities"] == [0.1, 0.1]
