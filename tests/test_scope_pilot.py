"""CPU-only pilot alignment and conservation checks."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from scope_pilot import combine  # noqa: E402


def example():
    verifier = {"id": "x", "offsets": [[0, 2], [3, 5], [6, 8]], "score": .8,
                "claims": [{"pieces": [[0, 2], [6, 8]], "risk": .8}]}
    localizer = {"id": "x", "offsets": verifier["offsets"],
                 "token_probabilities": [.1, .9, .4]}
    return verifier, localizer


def test_relative_ranking_preserves_verdict_and_excludes_removed_gap():
    verifier, localizer = example()
    result = combine(verifier, localizer)
    assert result["token_probabilities"] == pytest.approx([.2, 0, .8])
    assert result["score"] == verifier["score"]
    assert result["claims"] == verifier["claims"]
    assert "token_probabilities" not in verifier


def test_zero_denominator_falls_back_only_inside_claim():
    verifier, localizer = example()
    localizer["token_probabilities"] = [0, .9, 0]
    result = combine(verifier, localizer)
    assert result["token_probabilities"] == [.8, 0, .8]
    assert result["zero_denominator_fallbacks"] == 1


def test_no_claim_has_no_localization():
    verifier, localizer = example()
    verifier.update(claims=[], score=0)
    assert combine(verifier, localizer)["token_probabilities"] == [0, 0, 0]


@pytest.mark.parametrize("field,value", [("id", "wrong"), ("offsets", [[1, 2]]),
                                          ("token_probabilities", [float("nan"), .9, .4])])
def test_misaligned_or_invalid_inputs_fail(field, value):
    verifier, localizer = example()
    localizer[field] = value
    with pytest.raises(ValueError):
        combine(verifier, localizer)
