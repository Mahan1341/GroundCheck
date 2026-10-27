"""Operating-point cross-fit on caller-supplied, already reused development data.

Pure CPU/data functions: no file access, dataset loaders, model inference or weights.
Cross-fit limits threshold-selection leakage; it does not undo earlier checkpoint,
architecture or research selection using these same development sources.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from grounded_hallucination_lab.eval.scored import scored_metrics
from grounded_hallucination_lab.eval.selection import select_threshold

NAMESPACE = "groundcheck-operating-point-audit-v1:42"
FOLDS = 5


def make_folds(source_ids: Sequence[str], *, natural: bool = True) -> dict[str, Any]:
    """Five label-blind hash-order/round-robin folds of complete source clusters.

Pass natural=False for external group IDs. Original dataset partitions never change.
Only raw natural source IDs 14325/15545 share the known overlapping-passage cluster.
"""
    if not source_ids or any(not isinstance(s, str) or not s for s in source_ids):
        raise ValueError("Nonempty source ID strings required")
    clusters = ["merged_14325_15545" if natural and s in {"14325", "15545"} else s
                for s in source_ids]
    groups = sorted(set(clusters), key=lambda s: hashlib.sha256(
        f"{NAMESPACE}:{s}".encode()
    ).hexdigest())
    if len(groups) < FOLDS:
        raise ValueError("At least five source clusters required")
    assignment = {group: i % FOLDS for i, group in enumerate(groups)}
    by_row = [assignment[group] for group in clusters]
    folds = []
    for fold in range(FOLDS):
        held = [i for i, f in enumerate(by_row) if f == fold]
        train = [i for i, f in enumerate(by_row) if f != fold]
        held_groups = sorted({clusters[i] for i in held})
        train_groups = sorted({clusters[i] for i in train})
        if set(held_groups) & set(train_groups):
            raise AssertionError("Source cluster crossed its selection/evaluation fence")
        folds.append({"fold": fold, "train_indices": train, "held_indices": held,
                      "train_groups": train_groups, "held_groups": held_groups})
    return {"namespace": NAMESPACE, "n_folds": FOLDS, "natural": natural,
            "cluster_by_row": clusters, "fold_by_row": by_row,
            "group_to_fold": assignment, "folds": folds,
            "scope": "Operating-point cross-fit on reused development data; not independent test"}


def _scores(values: Sequence[float], length: int) -> list[float]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or len(array) != length:
        raise ValueError("Score length must match labels")
    if not np.isfinite(array).all() or np.any((array < 0) | (array > 1)):
        raise ValueError("Finite risk scores in [0,1] required")
    return array.tolist()


def _labels(values: Sequence[int]) -> list[int]:
    if len(values) == 0 or any(type(y) is not int or y not in (0, 1) for y in values):
        raise ValueError("Nonempty explicit binary annotation labels required")
    return list(values)


def response_threshold(labels: Sequence[int], scores: Sequence[float]) -> dict[str, Any]:
    """Exact macro-F1 across every risk>=threshold boundary; highest exact tie."""
    labels = _labels(labels)
    result = select_threshold(labels, _scores(scores, len(labels)))
    result["selection_partition"] = "crossfit_other_source_groups_only"
    return result


def _row_ids(ids: Sequence[str] | None, count: int) -> list[str]:
    result = [str(i) for i in range(count)] if ids is None else list(ids)
    if len(result) != count or len(set(result)) != count:
        raise ValueError("Unique response IDs aligned with rows required")
    return result


def _metrics(labels: list[int], predictions: list[int], scores: list[float]) -> dict[str, Any]:
    result = scored_metrics(labels, predictions, scores)
    (tn, fp), (fn, tp) = result["confusion_matrix"]
    result["fpr"] = fp / (fp + tn) if fp + tn else None
    result["fnr"] = fn / (fn + tp) if fn + tp else None
    return result


def evaluate_op(
    labels: Sequence[int], source_ids: Sequence[str],
    score_arrays: Mapping[str, Sequence[float]], historical_thresholds: Mapping[str, float],
    *, ids: Sequence[str] | None = None, natural: bool = True,
) -> dict[str, Any]:
    """Historical and OOF response decisions for each frozen score vector.

Returns prediction arrays and thresholds per row for caller-owned paired bootstrap.
No final deployment threshold is fitted. A bootstrap of returned fixed predictions
is conditional on these cross-fit selections, not selection-adjusted uncertainty.
"""
    labels = _labels(labels)
    if len(source_ids) != len(labels) or not score_arrays:
        raise ValueError("Sources/models must align with nonempty labels")
    if set(score_arrays) != set(historical_thresholds):
        raise ValueError("Each model needs exactly one historical threshold")
    row_ids = _row_ids(ids, len(labels))
    fold_info = make_folds(source_ids, natural=natural)
    methods = {}
    for name, values in score_arrays.items():
        scores = _scores(values, len(labels))
        historical = historical_thresholds[name]
        if not math.isfinite(historical):
            raise ValueError("Historical threshold must be finite")
        fixed = [int(score >= historical) for score in scores]
        selected_predictions = [-1] * len(labels)
        thresholds = [None] * len(labels)
        selections = []
        for fold in fold_info["folds"]:
            train, held = fold["train_indices"], fold["held_indices"]
            selection = response_threshold([labels[i] for i in train], [scores[i] for i in train])
            threshold = selection["threshold"]
            winner = next(c for c in selection["candidates"] if c["threshold"] == threshold)
            selections.append({
                "fold": fold["fold"], "threshold": threshold,
                "selection_objective": "response_macro_f1", "tie_break": selection["tie_break"],
                "candidate_count": selection["candidate_count"],
                "training_macro_f1": selection["macro_f1"],
                "training_positive_predictions": winner["positive_predictions"],
                "training_responses": len(train), "held_response_ids": [row_ids[i] for i in held],
                "training_groups": fold["train_groups"], "held_groups": fold["held_groups"],
            })
            for index in held:
                selected_predictions[index] = int(scores[index] >= threshold)
                thresholds[index] = threshold
        if any(p < 0 for p in selected_predictions):
            raise AssertionError("Every response must be evaluated exactly once")
        methods[name] = {
            "historical": {"threshold": historical, "predictions": fixed,
                           "metrics": _metrics(labels, fixed, scores)},
            "crossfit": {"predictions": selected_predictions, "threshold_by_row": thresholds,
                         "metrics": _metrics(labels, selected_predictions, scores),
                         "fold_selections": selections},
        }
    return {"ids": row_ids, "folds": fold_info, "methods": methods,
            "prediction_rule": "risk >= threshold", "final_deployment_threshold_fitted": False}


def token_f1_threshold(labels: Sequence[int], scores: Sequence[float]) -> dict[str, Any]:
    """Exhaustive positive-token F1 using sorted cumulative counts and exact integer ties.

Zero means not annotated hallucination, never an inference of entailment.
An empty predicted-positive set has precision/F1 zero, not precision one.
"""
    labels = _labels(labels)
    probabilities = np.asarray(_scores(scores, len(labels)), dtype=np.float64)
    targets = np.asarray(labels, dtype=np.int64)
    order = np.argsort(probabilities, kind="stable")
    unique, starts = np.unique(probabilities[order], return_index=True)
    positives_per_value = np.add.reduceat(targets[order], starts)
    removed_positive = np.cumsum(positives_per_value) - positives_per_value
    positive, negative = int(targets.sum()), len(targets) - int(targets.sum())
    true_positive = positive - removed_positive
    false_positive = negative - (starts - removed_positive)
    thresholds = [*unique.tolist(), math.nextafter(float(unique[-1]), math.inf)]
    true_positive = [*true_positive.tolist(), 0]
    false_positive = [*false_positive.tolist(), 0]
    best_num, best_den, best = -1, 1, None
    for threshold, tp, fp in zip(thresholds, true_positive, false_positive, strict=True):
        tp, fp = int(tp), int(fp)
        fn = positive - tp
        num, den = 2 * tp, 2 * tp + fp + fn
        if not den:
            den = 1
        # Ascending thresholds let equality choose the greatest boundary exactly.
        if num * best_den >= best_num * den:
            best_num, best_den = num, den
            best = {"threshold": threshold, "token_f1": num / den,
                    "tp": tp, "fp": fp, "fn": fn,
                    "precision": tp / (tp + fp) if tp + fp else 0.0,
                    "recall": tp / positive if positive else 0.0}
    return {**best, "objective": "positive_token_f1", "candidate_count": len(thresholds),
            "n_tokens": len(labels), "prediction_rule": "token risk >= threshold",
            "tie_break": "highest threshold; exact integer cross-multiplication",
            "zero_label_semantics": "not annotated hallucination; not entailed"}


def _token_counts(labels: list[list[int]], predictions: list[list[int]]) -> dict[str, Any]:
    tp = fp = fn = 0
    by_row = []
    for target, predicted in zip(labels, predictions, strict=True):
        counts = [sum(y == 1 and p == 1 for y, p in zip(target, predicted, strict=True)),
                  sum(y == 0 and p == 1 for y, p in zip(target, predicted, strict=True)),
                  sum(y == 1 and p == 0 for y, p in zip(target, predicted, strict=True))]
        by_row.append(dict(zip(("tp", "fp", "fn"), counts, strict=True)))
        tp += counts[0]
        fp += counts[1]
        fn += counts[2]
    return {"tp": tp, "fp": fp, "fn": fn, "precision": tp / (tp + fp) if tp + fp else 0.0,
            "recall": tp / (tp + fn) if tp + fn else 0.0,
            "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
            "counts_by_row": by_row}


def evaluate_token_op(
    gold_token_labels: Sequence[Sequence[int]], source_ids: Sequence[str],
    systems: Mapping[str, Sequence[Mapping[str, Any]]],
    *, ids: Sequence[str] | None = None, natural: bool = True,
) -> dict[str, Any]:
    """Separate historical .5 and OOF token-F1 operating points.

Caller supplies explicit annotation-derived targets aligned to offsets. -100 masks
ignored positions; missing gold is an error. No label is inferred from a model's
score, response-level label or missing extraction. Same source folds as evaluate_op.
"""
    if not gold_token_labels or len(gold_token_labels) != len(source_ids) or not systems:
        raise ValueError("Explicit aligned token annotation labels/sources/models required")
    targets = [list(row) for row in gold_token_labels]
    for row in targets:
        if any(type(y) is not int or y not in (-100, 0, 1) for y in row):
            raise ValueError("Token annotations must be explicit 0/1 or masked -100")
    row_ids = _row_ids(ids, len(targets))
    fold_info = make_folds(source_ids, natural=natural)
    methods, reference_offsets = {}, None
    for name, rows in systems.items():
        if len(rows) != len(targets):
            raise ValueError("Model token rows must align with targets")
        offsets = [row["offsets"] for row in rows]
        if reference_offsets is None:
            reference_offsets = offsets
        elif offsets != reference_offsets:
            raise ValueError("All models must use identical token offsets for comparison")
        scores = []
        for index, (row, target) in enumerate(zip(rows, targets, strict=True)):
            if ids is not None and row.get("id") != row_ids[index]:
                raise ValueError("Model IDs must align with provided response IDs")
            if len(row["offsets"]) != len(target):
                raise ValueError("Token offsets must align with explicit labels")
            if any(b <= a and y != -100 for (a, b), y in zip(
                row["offsets"], target, strict=True
            )):
                raise ValueError("Empty/special offsets must be masked, not supervised")
            scores.append(_scores(row["token_probabilities"], len(target)))
        fixed = [[int(s >= 0.5) if y != -100 else -100 for s, y in zip(p, t, strict=True)]
                 for p, t in zip(scores, targets, strict=True)]
        predicted = [None] * len(targets)
        thresholds, selections = [None] * len(targets), []
        for fold in fold_info["folds"]:
            train, held = fold["train_indices"], fold["held_indices"]
            values = [(y, s) for i in train for y, s in zip(targets[i], scores[i], strict=True)
                      if y != -100]
            selection = token_f1_threshold([y for y, _ in values], [s for _, s in values])
            threshold = selection["threshold"]
            selections.append({**selection, "fold": fold["fold"],
                               "held_response_ids": [row_ids[i] for i in held],
                               "training_groups": fold["train_groups"],
                               "held_groups": fold["held_groups"]})
            for index in held:
                predicted[index] = [int(s >= threshold) if y != -100 else -100 for s, y in zip(
                    scores[index], targets[index], strict=True
                )]
                thresholds[index] = threshold
        methods[name] = {
            "historical": {"threshold": 0.5, "predictions": fixed,
                           "metrics": _token_counts(targets, fixed)},
            "crossfit": {"predictions": predicted, "threshold_by_row": thresholds,
                         "metrics": _token_counts(targets, predicted),
                         "fold_selections": selections},
        }
    return {"ids": row_ids, "folds": fold_info, "methods": methods,
            "official_scorer": False, "criterion_independent_of_response_threshold": True,
            "zero_label_semantics": "not annotated hallucination; not entailed"}
