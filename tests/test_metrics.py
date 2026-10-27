import pytest

from grounded_hallucination_lab.baselines.simple import MajorityBaseline
from grounded_hallucination_lab.eval.metrics import binary_metrics


def test_hand_computed_confusion_and_metrics():
    result = binary_metrics([0, 0, 0, 1, 1, 1], [0, 0, 1, 0, 1, 1])
    assert result["confusion_matrix"] == [[2, 1], [1, 2]]
    assert result["positive_class"]["precision"] == pytest.approx(2 / 3)
    assert result["positive_class"]["recall"] == pytest.approx(2 / 3)
    assert result["macro_f1"] == pytest.approx(2 / 3)
    assert result["accuracy"] == pytest.approx(2 / 3)
    assert result["n"] == 6


def test_asymmetric_class_metrics():
    result = binary_metrics([0, 0, 0, 1, 1], [0, 0, 1, 1, 1])
    assert result["confusion_matrix"] == [[2, 1], [0, 2]]
    assert result["positive_class"]["precision"] == pytest.approx(2 / 3)
    assert result["positive_class"]["recall"] == 1
    assert result["positive_class"]["f1"] == pytest.approx(0.8)


def test_absent_class_and_undefined_denominators():
    result = binary_metrics([0, 0], [0, 0])
    assert result["accuracy"] == 1
    assert result["macro_f1"] == 0.5
    assert result["positive_class"]["f1"] == 0
    assert result["positive_class"]["precision_defined"] is False
    assert result["positive_class"]["recall_defined"] is False
    assert result["auroc"] is None


@pytest.mark.parametrize("actual,pred", [([], []), ([0], []), ([2], [0]), ([0.0], [0])])
def test_invalid_arrays(actual, pred):
    with pytest.raises(ValueError):
        binary_metrics(actual, pred)


def test_majority_training_only_and_tie():
    assert MajorityBaseline.fit([0, 0, 1]).predict(2) == [0, 0]
    assert MajorityBaseline.fit([1, 1, 0]).predict(2) == [1, 1]
    assert MajorityBaseline.fit([0, 1]).label == 0
    with pytest.raises(ValueError):
        MajorityBaseline.fit([])
