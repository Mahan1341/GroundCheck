"""Offline regression tests for preregistered discovery operations."""

import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "discovery_core", Path(__file__).resolve().parents[1] / "scripts/discovery_core.py"
)
core = importlib.util.module_from_spec(spec)
spec.loader.exec_module(core)


def test_readout_retains_numbers_and_original_localization():
    row = {
        "offsets": [[0, 1], [1, 3], [3, 5]],
        "token_probabilities": [0.9, 0.7, 0.2],
        "score": 0.9,
    }
    out = core.content_readout("*12kg", row)
    assert out["score"] == 0.7
    assert out["offsets"] == row["offsets"]
    assert out["token_probabilities"] == row["token_probabilities"]
    assert row["score"] == 0.9


def test_readout_empty_and_unicode():
    row = {"offsets": [[0, 1]], "token_probabilities": [0.8], "score": 0.8}
    assert core.content_readout("$", row)["score"] == 0
    assert core.content_readout("я", row)["score"] == 0.8
    assert core.content_readout("", {"offsets": [], "token_probabilities": []})["score"] == 0


def test_readout_rejects_broken_alignment():
    with pytest.raises(ValueError):
        core.content_readout("x", {"offsets": [[0, 2]], "token_probabilities": [0.2]})


def test_claim_projection_uses_offsets_and_maximum_context_support():
    verifier_spec = importlib.util.spec_from_file_location(
        "discovery_verifier", Path(__file__).resolve().parents[1] / "scripts/discovery_verifier.py"
    )
    verifier = importlib.util.module_from_spec(verifier_spec)
    verifier_spec.loader.exec_module(verifier)
    rows = [
        {
            "offsets": [[0, 2], [3, 5], [6, 8]],
            "segments": [{"start": 3, "end": 5, "evidence": [{"key": "a"}, {"key": "b"}]}],
        }
    ]
    result = verifier.project(rows, {"a": {"support": 0.1}, "b": {"support": 0.8}})
    assert result[0]["score"] == pytest.approx(0.2)
    assert result[0]["token_probabilities"] == pytest.approx([0, 0.2, 0])
    assert "risk" not in rows[0]["segments"][0]
    assert verifier.pair_key("ab", "c") != verifier.pair_key("a", "bc")
