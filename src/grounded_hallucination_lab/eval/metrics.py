"""Binary response metrics; positive class is annotated unfaithfulness."""

from typing import Any


def binary_metrics(y_true: list[int], y_pred: list[int]) -> dict[str, Any]:
    if not y_true or len(y_true) != len(y_pred):
        raise ValueError("metrics require nonempty, equal-length arrays")
    if any(type(y) is not int or y not in (0, 1) for y in [*y_true, *y_pred]):
        raise ValueError("labels must be binary integers")
    matrix = [[0, 0], [0, 0]]
    for actual, predicted in zip(y_true, y_pred, strict=True):
        matrix[actual][predicted] += 1
    per_class = {}
    for label in (0, 1):
        tp = matrix[label][label]
        support = sum(matrix[label])
        predicted = sum(row[label] for row in matrix)
        precision = tp / predicted if predicted else 0.0
        recall = tp / support if support else 0.0
        f1 = 2 * tp / (support + predicted) if support + predicted else 0.0
        per_class[str(label)] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
            "predicted_count": predicted,
            "true_positive_count": tp,
            "precision_defined": predicted > 0,
            "recall_defined": support > 0,
            "f1_defined": support + predicted > 0,
        }
    counts = {str(k): sum(matrix[k]) for k in (0, 1)}
    n = len(y_true)
    return {
        "n": n,
        "positive_label": 1,
        "class_counts": counts,
        "class_proportions": {k: v / n for k, v in counts.items()},
        "confusion_matrix": matrix,
        "confusion_matrix_axes": "rows=true, columns=pred; [0,1]",
        "per_class": per_class,
        "positive_class": per_class["1"],
        "macro_f1": sum(v["f1"] for v in per_class.values()) / 2,
        "accuracy": (matrix[0][0] + matrix[1][1]) / n,
        "accuracy_denominator": n,
        "macro_f1_class_denominator": 2,
        "zero_division": "undefined precision/recall/F1 set to 0; flags retained",
        "auroc": None,
        "average_precision": None,
        "ranking_metrics_reason": "majority baseline has no discriminative continuous score",
    }
