"""Calibration-only exhaustive decision boundaries with exact tie comparison."""

import math
from fractions import Fraction


def select_threshold(labels: list[int], scores: list[float]) -> dict:
    if not labels or len(labels) != len(scores):
        raise ValueError("nonempty equal-length calibration inputs required")
    if any(type(y) is not int or y not in (0, 1) for y in labels):
        raise ValueError("binary labels required")
    if any(not math.isfinite(s) or not 0 <= s <= 1 for s in scores):
        raise ValueError("risk scores must be finite in [0,1]")
    groups = {}
    for y, score in zip(labels, scores, strict=True):
        counts = groups.setdefault(score, [0, 0])
        counts[y] += 1
    positives, negatives = sum(labels), len(labels) - sum(labels)
    tp, fp = positives, negatives
    candidates = []
    best_value, best_threshold = Fraction(-1), -1.0
    for threshold in [*sorted(groups), math.nextafter(max(scores), math.inf)]:
        fn, tn = positives - tp, negatives - fp
        pos_denom, neg_denom = 2 * tp + fp + fn, 2 * tn + fp + fn
        objective = (Fraction(tp, pos_denom) if pos_denom else Fraction(0)) + (
            Fraction(tn, neg_denom) if neg_denom else Fraction(0)
        )
        candidates.append(
            {
                "threshold": threshold,
                "macro_f1": float(objective),
                "positive_predictions": tp + fp,
                "confusion_matrix": [[tn, fp], [fn, tp]],
            }
        )
        if objective > best_value or (objective == best_value and threshold > best_threshold):
            best_value, best_threshold = objective, threshold
        if threshold in groups:
            fp -= groups[threshold][0]
            tp -= groups[threshold][1]
    return {
        "threshold": best_threshold,
        "macro_f1": float(best_value),
        "prediction_rule": "risk >= threshold",
        "objective": "macro_f1",
        "tie_break": "highest candidate threshold; exact rational objective comparison",
        "candidate_count": len(candidates),
        "candidates": candidates,
        "selection_partition": "calibration",
        "n": len(labels),
    }
