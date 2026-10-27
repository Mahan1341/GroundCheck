"""CPU checks for source-aggregate bootstrap arithmetic, without data/model access."""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
spec = importlib.util.spec_from_file_location(
    "scope_analysis", Path(__file__).resolve().parents[1] / "scripts/scope_analysis.py"
)
analysis = importlib.util.module_from_spec(spec)
spec.loader.exec_module(analysis)


def test_confusion_and_token_mass_aggregation():
    counts = analysis.confusion_rows([1, 0, 1, 0], [1, 1, 0, 0], [{"tp": 1, "fp": 2, "fn": 3}] * 4)
    assert counts.sum(0).tolist() == [1, 1, 1, 1, 4, 8, 12]
    values = analysis.fast_values(counts.sum(0))
    assert values["macro_f1"] == 0.5
    assert values["precision"] == 0.5
    assert values["token_f1"] == pytest.approx(8 / 28)


def test_vectorized_metric_zero_denominators_and_batch_axis():
    values = analysis.fast_values(np.array([[0, 0, 1, 3], [2, 0, 0, 4]]))
    assert values["recall"].tolist() == [0, 1]
    assert values["macro_f1"].tolist() == pytest.approx([3 / 7, 1])


def test_source_bootstrap_identical_systems_have_exact_zero_paired_deltas():
    predictions = [0, 1, 1, 0, 1, 0]
    response = {
        "methods": {
            name: {op: {"predictions": predictions} for op in ("historical", "crossfit")}
            for name in ("B7", *analysis.METHODS)
        }
    }
    result = analysis.fast_bootstrap([0, 1, 0, 1, 1, 0], ["a", "a", "b", "c", "d", "e"], response)
    assert result["resamples"] == 2000
    assert result["source_clusters"] == 5
    for op in ("historical", "crossfit"):
        for values in result[op]["differences"].values():
            assert all(interval == {"low": 0.0, "high": 0.0} for interval in values.values())
