"""No real data: grouped operating-point isolation and exact selection boundaries."""

import ast
import importlib.util
import math
from fractions import Fraction
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/scope_eval.py"
SPEC = importlib.util.spec_from_file_location("scope_eval", SCRIPT)
scope = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(scope)


def test_macro_f1_exact_tie_prefers_highest_boundary():
    result = scope.response_threshold([0, 1], [0.8, 0.2])
    assert result["threshold"] == math.nextafter(0.8, math.inf)
    assert result["macro_f1"] == pytest.approx(1 / 3)


def test_source_groups_share_fence_and_fold_assignment_is_order_invariant():
    ids = ["14325", "15545", "a", "a", "b", "c", "d", "e"]
    folds = scope.make_folds(ids)
    assert folds["fold_by_row"][0] == folds["fold_by_row"][1]
    assert folds["fold_by_row"][2] == folds["fold_by_row"][3]
    assert folds["group_to_fold"] == scope.make_folds(list(reversed(ids)))["group_to_fold"]
    for fold in folds["folds"]:
        assert not set(fold["train_groups"]) & set(fold["held_groups"])
    external = scope.make_folds(ids, natural=False)
    assert "14325" in external["group_to_fold"] and "15545" in external["group_to_fold"]
    assert "merged_14325_15545" not in external["group_to_fold"]


def test_held_labels_cannot_change_their_operating_point_or_predictions():
    sources = [f"source{i}" for i in range(15)]
    labels = [i % 2 for i in range(15)]
    scores = {"m": [(i + 1) / 16 for i in range(15)]}
    first = scope.evaluate_op(labels, sources, scores, {"m": 0.5})
    held = first["folds"]["folds"][0]["held_indices"]
    changed = [1 - y if i in held else y for i, y in enumerate(labels)]
    second = scope.evaluate_op(changed, sources, scores, {"m": 0.5})
    a, b = first["methods"]["m"]["crossfit"], second["methods"]["m"]["crossfit"]
    assert a["fold_selections"][0] == b["fold_selections"][0]
    for index in held:
        assert a["threshold_by_row"][index] == b["threshold_by_row"][index]
        assert a["predictions"][index] == b["predictions"][index]


def test_token_positive_f1_exact_tie_and_empty_predictions():
    selection = scope.token_f1_threshold([1, 0, 0, 1], [0.1, 0.2, 0.3, 0.4])
    assert selection["threshold"] == 0.4
    assert selection["token_f1"] == pytest.approx(2 / 3)
    empty = scope.token_f1_threshold([0, 0], [0.2, 0.8])
    assert empty["threshold"] > 0.8
    assert empty["precision"] == empty["token_f1"] == 0.0


@pytest.mark.parametrize("labels", [[0, 0, 0], [1, 1, 1], [0, 1, 0], [1, 0, 1]])
def test_efficient_token_selection_matches_fraction_exhaustive(labels):
    scores = [0.2, 0.2, 0.8]
    candidates = [0.2, 0.8, math.nextafter(0.8, math.inf)]
    expected = []
    for threshold in candidates:
        pred = [s >= threshold for s in scores]
        tp = sum(y == 1 and p for y, p in zip(labels, pred, strict=True))
        fp = sum(y == 0 and p for y, p in zip(labels, pred, strict=True))
        fn = sum(y == 1 and not p for y, p in zip(labels, pred, strict=True))
        expected.append((Fraction(2 * tp, 2 * tp + fp + fn or 1), threshold))
    actual = scope.token_f1_threshold(labels, scores)
    assert actual["threshold"] == max(expected)[1]
    assert actual["token_f1"] == float(max(expected)[0])


def test_token_held_label_fence_and_separate_historical_threshold():
    sources = [f"source{i}" for i in range(10)]
    targets = [[i % 2, -100, 1 - i % 2] for i in range(10)]
    rows = [{"id": str(i), "offsets": [[0, 1], [0, 0], [2, 3]],
             "token_probabilities": [0.1 + 0.07 * i, 1.0, 0.8 - 0.05 * i]}
            for i in range(10)]
    ids = [str(i) for i in range(10)]
    first = scope.evaluate_token_op(targets, sources, {"m": rows}, ids=ids)
    held = first["folds"]["folds"][0]["held_indices"]
    changed = [[1 - y if y != -100 else -100 for y in row] if i in held else row
               for i, row in enumerate(targets)]
    second = scope.evaluate_token_op(changed, sources, {"m": rows}, ids=ids)
    a, b = first["methods"]["m"], second["methods"]["m"]
    assert a["historical"]["threshold"] == 0.5
    assert a["crossfit"]["fold_selections"][0] == b["crossfit"]["fold_selections"][0]
    for index in held:
        assert a["crossfit"]["predictions"][index] == b["crossfit"]["predictions"][index]
        assert a["crossfit"]["predictions"][index][1] == -100
    assert a["historical"]["metrics"]["tp"] + a["historical"]["metrics"]["fn"] == 10


def test_missing_token_gold_is_not_inferred_as_entailment():
    with pytest.raises(ValueError):
        scope.evaluate_token_op([], [], {})
    with pytest.raises(ValueError):
        scope.token_f1_threshold([-100], [0.2])
    with pytest.raises(ValueError):
        scope.make_folds(["a", "b"])


def test_module_has_no_io_model_or_partition_loader_dependency():
    tree = ast.parse(SCRIPT.read_text(encoding="utf-8"))
    imported = [node.module or "" for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)]
    assert not any("data." in module or "transformers" in module for module in imported)
    forbidden = {"open", "read_text", "read_bytes", "load_ragtruth", "load_dataset",
                 "from_pretrained", "read_rows", "write_json", "read_json"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            name = node.func.id if isinstance(node.func, ast.Name) else (
                node.func.attr if isinstance(node.func, ast.Attribute) else ""
            )
            assert name not in forbidden
