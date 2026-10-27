"""Offline metric semantics: claim AP and fair maximum ties."""

import importlib.util
import sys
from pathlib import Path

import pytest

scripts = Path(__file__).resolve().parents[1] / "experiments/scripts"
sys.path.insert(0, str(scripts))
spec = importlib.util.spec_from_file_location(
    "claim_grounding_analyze", scripts / "claim_grounding_analyze.py"
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_uniform_claim_ranking_gets_prevalence_not_argmax_luck():
    value = module.claim_rank([0, 1, 0, 1], [0.8] * 4)
    assert value["ap"] == 0.5
    assert value["pointing"] == 0.5


def test_positive_claim_ranking_and_all_positive_denominator():
    assert module.claim_rank([0, 1], [0.1, 0.9])["ap"] == 1
    assert module.claim_rank([1, 1], [0.1, 0.9])["ap"] == 1
    assert module.claim_rank([0, 0], [0.1, 0.9])["ap"] is None


def test_unaligned_claim_metrics_fail():
    with pytest.raises(ValueError):
        module.claim_rank([1], [])
