"""Offline R4 semantic, offset, grouping and inference-view regression checks."""

import pytest

from grounded_hallucination_lab.research.numeric_local import (
    local_claims,
    normalize_numbers,
    project_probabilities,
)
from grounded_hallucination_lab.research.numeric_local_suite import suite


@pytest.mark.parametrize(
    ("original", "expected"),
    [
        ("Exactly twenty-four bolts.", "Exactly 24 bolts."),
        ("It lasted 900 seconds.", "It lasted 15 minutes."),
        ("Length: two kilometers.\n\nOK", "Length: 2000 meters.\n\nOK"),
        ("one minute", "1 minute"),
        ("61 seconds", "61 seconds"),
        ("", ""),
        ("Two entities.", "Two entities."),
    ],
)
def test_numeric_canonical(original, expected):
    mapped = normalize_numbers(original)
    assert mapped.text == expected
    assert len(mapped.origins) == len(mapped.text)
    assert all(0 <= a < b <= len(original) for a, b in mapped.origins)
    assert normalize_numbers(expected).text == expected


@pytest.mark.parametrize(
    "text",
    [
        "two to four kilometers",
        "about two kilometers",
        "2-4 kilometers",
        "between two and four kilometers",
        "2.5 kilometers",
        "1/2 kilometers",
        "-2 kilometers",
        "one million two kilometers",
        "one and a half kilometers",
        "either two or four kilometers",
        "02 minutes",
        "1,000 meters",
        "two hundred and five minutes",
        "two two minutes",
        "2 ± 1 kilometers",
    ],
)
def test_numeric_refuses_ambiguity(text):
    assert normalize_numbers(text).text == text


def test_according_to_is_not_a_numeric_range():
    assert normalize_numbers("According to the dispatch, two kilometers.").text == (
        "According to the dispatch, 2000 meters."
    )
    assert normalize_numbers("two through four kilometers").text == "two through four kilometers"


@pytest.mark.parametrize(
    "text",
    [
        "two - four kilometers",
        "2–four kilometers",
        "two/four kilometers",
        "twenty-four - five kilometers",
        "two-four kilometers",
        "2-four kilometers",
        "2...4 kilometers",
    ],
)
def test_mixed_range_and_fraction_guard(text):
    assert normalize_numbers(text).text == text


def test_offset_mapping_both_directions_and_invalid_ranges():
    mapped = normalize_numbers("Distance: twenty-four kilometers.")
    assert mapped.text == "Distance: 24000 meters."
    a, b = mapped.normalized_range(10, 21)
    assert mapped.text[a:b] == "24000"
    assert mapped.original_range(a, b) == (10, 21)
    with pytest.raises(ValueError):
        mapped.original_range(2, 2)
    with pytest.raises(ValueError):
        mapped.normalized_range(-1, 2)


def test_normalization_does_not_erase_entity_value_binding():
    first = normalize_numbers("Alpha: two kilometers. Beta: four kilometers.")
    second = normalize_numbers("Alpha: four kilometers. Beta: two kilometers.")
    assert first.text != second.text
    assert first.text == "Alpha: 2000 meters. Beta: 4000 meters."


def test_mapping_is_original_provenance_not_fake_character_alignment():
    text = "X: twenty-four bolts."
    mapped = normalize_numbers(text)
    assert mapped.original_range(3, 5) == (3, 14)
    assert project_probabilities(
        [(3, 9), (10, 14), (15, 20)], mapped, [(3, 5), (6, 11)], [0.8, 0.1]
    ) == [0.8, 0.8, 0.1]
    identity = normalize_numbers("A 17 bolts.")
    assert project_probabilities([(0, 1), (2, 4)], identity, [(0, 1), (2, 4)], [0.1, 0.9]) == [
        0.1,
        0.9,
    ]


def test_local_all_sentences_and_coreference_fallback():
    text = "A has 12 bolts. B has 14 bolts. C is red. D is blue."
    views, reason = local_claims(text)
    assert reason is None and len(views) == 4
    assert all(
        text[v["central_start"] : v["central_end"]] in text[v["start"] : v["end"]] for v in views
    )
    covered = {i for v in views for i in range(v["central_start"], v["central_end"])}
    assert all(i in covered for i, char in enumerate(text) if not char.isspace())
    text += " Its lid is green."
    views, reason = local_claims(text)
    assert reason == "anaphora" and views[0]["end"] == len(text) and len(views) == 1


def test_suite_pairs_offsets_and_disjoint_parents():
    from grounded_hallucination_lab.research.invariance import suite as previous

    records, design = suite()
    assert len(records) == 864 and len({r.source_id for r in records}) == 12
    assert not {r.source_id for r in records} & {r.source_id for r in previous("fit")[0]}
    pairs = {}
    for r, d in zip(records, design, strict=True):
        pairs.setdefault(d["pair_id"], []).append(r)
        assert bool(r.annotations) == bool(d["gold"])
        for a in r.annotations:
            assert r.response[a.start : a.end] == a.text
    assert len(pairs) == 432
    for pair in pairs.values():
        assert len(pair) == 2 and pair[0].response == pair[1].response
        assert pair[0].evidence != pair[1].evidence
