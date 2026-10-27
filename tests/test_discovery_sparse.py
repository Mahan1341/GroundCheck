"""Offline algebraic checks for Cycle3; no torch/models/data/network required."""

import importlib.util
import math
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "discovery_sparse", Path(__file__).resolve().parents[1] / "scripts/discovery_sparse_training.py"
)
sparse = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sparse)


def test_global_positive_mass_preserved_and_all_negative_coefficients_unchanged():
    rows = [[-100, 1, 0, 0], [0, 1, 1, 1, -100], [0, 0], [1, 1]]
    constant = sparse.positive_constant(rows)
    totals = {"A": 0.0, "B": 0.0}
    for labels in rows:
        arms = {arm: sparse.token_coefficients(labels, constant, arm) for arm in totals}
        for y, a, b in zip(labels, arms["A"], arms["B"], strict=True):
            if y in (-100, 0):
                assert a == b
            if y == -100:
                assert a == 0
            if y == 1:
                totals["A"] += a
                totals["B"] += b
        if 1 in labels:
            assert sum(
                c for y, c in zip(labels, arms["B"], strict=True) if y == 1
            ) == pytest.approx(constant)
    assert totals["A"] == pytest.approx(totals["B"])


def test_sparse_positive_gains_mass_without_changing_negative_only():
    rows = [[1] + [0] * 99, [1] * 50 + [0] * 50, [0] * 10]
    constant = sparse.positive_constant(rows)
    assert (
        sparse.token_coefficients(rows[0], constant, "B")[0]
        > 20 * sparse.token_coefficients(rows[0], constant, "A")[0]
    )
    assert sparse.token_coefficients(rows[-1], constant, "B") == [0.1] * 10


def test_duplicating_response_preserves_per_row_objective():
    labels = [-100, 1, 0, 0]
    losses = [99, 0.7, 0.2, 0.1]
    coefficients = sparse.token_coefficients(labels, 0.4, "B")
    single = sum(a * b for a, b in zip(losses, coefficients, strict=True))
    duplicated_mean = sum(a * b / 2 for a, b in zip(losses * 2, coefficients * 2, strict=True))
    assert duplicated_mean == pytest.approx(single)


def test_logit_mass_matches_finite_difference_binary_cross_entropy():
    logits, labels, coefficients = [(0.3, -0.4), (1.0, 0.2)], [1, 0], [0.4, 0.2]
    probabilities = [math.exp(b) / (math.exp(a) + math.exp(b)) for a, b in logits]
    result = sparse.logit_gradient_mass(probabilities, labels, coefficients)
    eps = 1e-6
    numerical = {"positive": 0.0, "negative": 0.0}
    for pair, label, coefficient in zip(logits, labels, coefficients, strict=True):

        def loss(values, coefficient=coefficient, label=label):
            return coefficient * (math.log(sum(math.exp(v) for v in values)) - values[label])

        for index in (0, 1):
            plus, minus = list(pair), list(pair)
            plus[index] += eps
            minus[index] -= eps
            numerical["positive" if label else "negative"] += abs(
                (loss(plus) - loss(minus)) / (2 * eps)
            )
    assert result["positive"] == pytest.approx(numerical["positive"])
    assert result["negative"] == pytest.approx(numerical["negative"])


@pytest.mark.parametrize("rows", [[], [[0, 0]], [[-100]], [[2, 1]]])
def test_invalid_target_populations_rejected(rows):
    with pytest.raises(ValueError):
        sparse.positive_constant(rows)


def test_masked_logits_do_not_contribute():
    assert sparse.logit_gradient_mass([0.99, 0.2], [-100, 0], [0, 0.5]) == {
        "positive": 0,
        "negative": 0.2,
        "positive_share": 0,
    }
    with pytest.raises(ValueError):
        sparse.logit_gradient_mass([0.99], [-100], [0.5])


def test_gate_uses_ratio_of_median_response_shares():
    rows = [
        {"A": {"positive_share": a}, "B": {"positive_share": b}}
        for a, b in [(0.1, 0.3), (0.15, 0.4), (0.9, 0.99)]
    ]
    gate = sparse.derivative_gate(rows)
    assert gate["historical_median_positive_derivative_fraction"] == 0.15
    assert gate["median_fraction_ratio"] == pytest.approx(0.4 / 0.15)
    assert gate["passed"]
    assert not sparse.derivative_gate([{"A": {"positive_share": 0}, "B": {"positive_share": 0}}])[
        "passed"
    ]
