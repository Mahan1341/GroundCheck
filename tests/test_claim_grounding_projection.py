"""Offline native-gradient readout checks; no transformer/GPU/data access."""

import importlib.util
import math
from copy import deepcopy
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "claim_grounding_projection",
    Path(__file__).resolve().parents[1] / "experiments/scripts/claim_grounding_projection.py",
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def fixture():
    row = {
        "id": "fixture",
        "response": "12 ab cd",
        "offsets": [[0, 2], [3, 5], [6, 8]],
        "score": 0.8,
        "claims": [
            {"text": "12 ab cd", "pieces": [[0, 8]], "risk": 0.8, "evidence": [{"key": "winner"}]}
        ],
    }
    cache = {
        "winner": {
            "claim_offsets": [[0, 1], [1, 2], [3, 5], [6, 8]],
            "L2values": [0.1, 0.3, 0.8, 0.4],
        }
    }
    return row, cache


def test_mean_projection_normalization_and_original_verdicts_unchanged():
    row, cache = fixture()
    before = deepcopy((row, cache))
    result = module.project(row, cache, ["winner"])
    assert result["native"]["token_probabilities"] == pytest.approx([0.2, 0.8, 0.4])
    assert result["native"]["score"] == result["null"]["score"] == row["score"]
    assert result["native"]["claims"] == result["null"]["claims"] == row["claims"]
    assert (row, cache) == before


def test_permutation_is_deterministic_and_preserves_each_digit_position_bin():
    values = list(map(float, range(1, 9)))
    texts = ["first", "12", "word", "34", "other", "56", "third", "last"]
    shuffled, stats = module.stratified_permutation(values, texts, "key", seed=42)
    assert (shuffled, stats) == module.stratified_permutation(values, texts, "key", seed=42)
    assert sorted(shuffled) == sorted(values)
    assert math.fsum(v * v for v in shuffled) == math.fsum(v * v for v in values)
    for group in ([0, 7], [1, 3, 5], [2, 4, 6]):
        assert sorted(shuffled[i] for i in group) == sorted(values[i] for i in group)
    assert stats["eligible_tokens"] == 8


def test_piecewise_projection_excludes_removed_original_gap():
    row = {
        "response": "a DROP b",
        "offsets": [[0, 1], [2, 6], [7, 8]],
        "score": 0.7,
        "claims": [
            {"text": "a\nb", "pieces": [[0, 1], [7, 8]], "risk": 0.7, "evidence": [{"key": "k"}]}
        ],
    }
    cache = {"k": {"claim_offsets": [[0, 1], [2, 3]], "L2values": [1.0, 2.0]}}
    result = module.project(row, cache, ["k"])
    assert result["native"]["token_probabilities"] == pytest.approx([0.35, 0, 0.7])
    assert result["null"]["token_probabilities"][1] == 0


def test_zero_magnitudes_use_logged_uniform_risk_not_nan():
    row, cache = fixture()
    cache["winner"]["L2values"] = [1e-14] * 4
    result = module.project(row, cache, ["winner"])
    assert result["native"]["token_probabilities"] == [0.8] * 3
    assert result["null"]["token_probabilities"] == [0.8] * 3
    assert result["stats"]["zero_denominator_fallbacks"] == 1


def test_overlapping_claims_take_max_without_mutating_claim_risk():
    row, cache = fixture()
    row["claims"].append(
        {"text": "cd", "pieces": [[6, 8]], "risk": 0.9, "evidence": [{"key": "second"}]}
    )
    row["score"] = 0.9
    cache["second"] = {"claim_offsets": [[0, 2]], "L2values": [0.1]}
    result = module.project(row, cache, ["winner", "second"])
    assert result["native"]["token_probabilities"] == pytest.approx([0.2, 0.8, 0.9])


def test_empty_claim_set_preserves_no_claim_response():
    row = {"response": "Unable to answer.", "offsets": [[0, 6]], "claims": [], "score": 0.0}
    result = module.project(row, {}, [])
    assert result["native"]["token_probabilities"] == [0]
    assert result["stats"]["claims"] == 0


@pytest.mark.parametrize("fault", ["missing_coverage", "misaligned", "negative", "nan", "text"])
def test_invalid_gradient_mapping_fails_instead_of_silent_uniform_fallback(fault):
    row, cache = fixture()
    if fault == "missing_coverage":
        cache["winner"]["claim_offsets"] = [[0, 0]] * 4
    elif fault == "misaligned":
        cache["winner"]["L2values"].pop()
    elif fault == "negative":
        cache["winner"]["L2values"][0] = -1
    elif fault == "nan":
        cache["winner"]["L2values"][0] = float("nan")
    else:
        row["claims"][0]["text"] = "different"
    with pytest.raises(ValueError):
        module.project(row, cache, ["winner"])
