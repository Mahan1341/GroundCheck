"""Scored metrics and paired cluster bootstrap for frozen dev predictions."""

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from grounded_hallucination_lab.eval.metrics import binary_metrics


def _validate(labels, predictions, scores):
    binary_metrics(list(labels), list(predictions))
    if len(scores) != len(labels) or not np.isfinite(scores).all():
        raise ValueError("scores must be finite and match label length")


def scored_metrics(labels: list[int], predictions: list[int], scores: list[float]) -> dict:
    _validate(labels, predictions, scores)
    result = binary_metrics(labels, predictions)
    both = len(set(labels)) == 2
    result.update(
        {
            "auroc": float(roc_auc_score(labels, scores)) if both else None,
            "average_precision": float(average_precision_score(labels, scores)) if both else None,
            "ranking_metrics_reason": None if both else "both classes required by this protocol",
            "ranking_metrics_denominator": len(labels),
        }
    )
    return result


def _weighted_metrics(y, pred, scores, weight) -> dict:
    tp = float(weight[(y == 1) & (pred == 1)].sum())
    fp = float(weight[(y == 0) & (pred == 1)].sum())
    fn = float(weight[(y == 1) & (pred == 0)].sum())
    tn = float(weight[(y == 0) & (pred == 0)].sum())
    def ratio(a, b):
        return a / b if b else 0.0
    pos_f1, neg_f1 = ratio(2 * tp, 2 * tp + fp + fn), ratio(2 * tn, 2 * tn + fp + fn)
    both = tp + fn > 0 and tn + fp > 0
    return {
        "accuracy": ratio(tp + tn, tp + fp + fn + tn),
        "macro_f1": (pos_f1 + neg_f1) / 2,
        "precision": ratio(tp, tp + fp),
        "recall": ratio(tp, tp + fn),
        "f1": pos_f1,
        "auroc": float(roc_auc_score(y, scores, sample_weight=weight)) if both else None,
        "average_precision": float(average_precision_score(y, scores, sample_weight=weight))
        if both
        else None,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    }


def source_bootstrap(
    labels: list[int],
    source_ids: list[str],
    systems: dict[str, tuple[list[int], list[float]]],
    *,
    resamples: int = 1000,
    seed: int = 42,
) -> dict:
    """Identical source draws for every system; percentile CIs, no refit/recalibration."""
    if type(resamples) is not int or resamples < 1000:
        raise ValueError("at least 1000 source bootstrap resamples required")
    if len(source_ids) != len(labels) or not systems:
        raise ValueError("source IDs must match nonempty labeled predictions")
    for pred, scores in systems.values():
        _validate(labels, pred, scores)
    groups = sorted(set(source_ids))
    mapping = {sid: i for i, sid in enumerate(groups)}
    indices = np.array([mapping[sid] for sid in source_ids])
    y = np.array(labels)
    arrays = {name: (np.array(p), np.array(s)) for name, (p, s) in systems.items()}
    draws = {name: {} for name in systems}
    differences = {}
    reference = "majority" if "majority" in systems else next(iter(systems))
    rng = np.random.default_rng(seed)
    for _ in range(resamples):
        counts = np.bincount(rng.integers(0, len(groups), len(groups)), minlength=len(groups))
        weights = counts[indices]
        values = {
            name: _weighted_metrics(y, pred, scores, weights)
            for name, (pred, scores) in arrays.items()
        }
        for name, metrics in values.items():
            for metric, value in metrics.items():
                if value is not None:
                    draws[name].setdefault(metric, []).append(value)
                else:
                    draws[name].setdefault(metric, [])
                if name != reference:
                    delta = differences.setdefault(f"{name}_minus_{reference}", {})
                    delta.setdefault(metric, [])
                    other = values[reference][metric]
                    if value is not None and other is not None:
                        delta[metric].append(value - other)

    def summarize(table):
        return {
            name: {
                metric: {
                    "low": float(np.percentile(values, 2.5)) if values else None,
                    "high": float(np.percentile(values, 97.5)) if values else None,
                    "valid_resamples": len(values),
                    "undefined_resamples": resamples - len(values),
                }
                for metric, values in metrics.items()
            }
            for name, metrics in table.items()
        }

    return {
        "method": "paired source-group percentile bootstrap",
        "confidence": 0.95,
        "resamples": resamples,
        "seed": seed,
        "source_groups": len(groups),
        "response_count": len(labels),
        "systems": summarize(draws),
        "paired_differences": summarize(differences),
        "scope": "fixed model/threshold; excludes training and calibration uncertainty",
    }
