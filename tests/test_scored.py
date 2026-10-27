import math

import pytest

from grounded_hallucination_lab.eval.metrics import binary_metrics
from grounded_hallucination_lab.eval.scored import scored_metrics, source_bootstrap
from grounded_hallucination_lab.eval.selection import select_threshold


def test_exhaustive_threshold_candidates_and_tie():
    labels, scores = [0, 1], [0.5, 0.5]
    result = select_threshold(labels, scores)
    assert result["candidate_count"] == 2
    assert result["threshold"] == math.nextafter(0.5, math.inf)
    assert result["macro_f1"] == pytest.approx(1 / 3)


def test_threshold_confusions_match_direct_predictions():
    labels, scores = [0, 1, 0, 1, 0], [0.0, 0.4, 0.4, 1.0, 0.7]
    result = select_threshold(labels, scores)
    assert result["candidate_count"] == 5
    for candidate in result["candidates"]:
        pred = [int(s >= candidate["threshold"]) for s in scores]
        metrics = binary_metrics(labels, pred)
        assert candidate["confusion_matrix"] == metrics["confusion_matrix"]
        assert candidate["macro_f1"] == pytest.approx(metrics["macro_f1"])


@pytest.mark.parametrize("scores", [[float("nan")], [-0.1], [1.1], []])
def test_threshold_invalid_scores(scores):
    with pytest.raises(ValueError):
        select_threshold([1], scores)


def test_rank_metrics_known_values_and_ties():
    result = scored_metrics([0, 1, 0, 1], [0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8])
    assert result["auroc"] == 1.0
    assert result["average_precision"] == 1.0
    tied = scored_metrics([0, 0, 1, 1], [0, 0, 0, 0], [0.2] * 4)
    assert tied["auroc"] == 0.5 and tied["average_precision"] == 0.5
    assert scored_metrics([0], [0], [0.2])["auroc"] is None


def test_bootstrap_is_grouped_paired_and_deterministic():
    y, groups = [0, 0, 1, 1], ["a", "a", "b", "b"]
    systems = {"majority": ([0] * 4, [0.5] * 4), "lexical": ([0, 0, 1, 1], [0, 0, 1, 1])}
    first = source_bootstrap(y, groups, systems)
    second = source_bootstrap(y, groups, systems)
    assert first == second
    assert first["source_groups"] == 2
    # Both responses in a source always travel together. All-negative/all-positive
    # cluster draws cause undefined AUROC; perfect predictions keep accuracy=1.
    assert first["systems"]["lexical"]["accuracy"]["low"] == 1
    assert first["systems"]["lexical"]["auroc"]["undefined_resamples"] > 0
    assert first["systems"]["lexical"]["auroc"]["valid_resamples"] > 0
    assert first["paired_differences"]["lexical_minus_majority"]["accuracy"]["low"] >= 0


def test_bootstrap_requires_1000_resamples():
    with pytest.raises(ValueError, match="1000"):
        source_bootstrap([0, 1], ["a", "b"], {"majority": ([0, 0], [0.5, 0.5])}, resamples=999)
