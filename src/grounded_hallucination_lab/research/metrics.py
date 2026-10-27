"""Research metrics: paired clusters, nonempty precision point and separate span task."""

import numpy as np

from grounded_hallucination_lab.eval.scored import _weighted_metrics, scored_metrics
from grounded_hallucination_lab.eval.selection import select_threshold


def metrics(labels, scores, threshold):
    predictions = [int(s >= threshold) for s in scores]
    result = scored_metrics(labels, predictions, scores)
    (tn, fp), (fn, tp) = result["confusion_matrix"]
    result.update(
        {
            "fpr": fp / (fp + tn) if fp + tn else None,
            "fnr": fn / (fn + tp) if fn + tp else None,
            "threshold": threshold,
        }
    )
    return result


def precision_point(labels: list[int], scores: list[float]) -> dict:
    selected = select_threshold(labels, scores)
    feasible = []
    for candidate in selected["candidates"]:
        (_, fp), (_, tp) = candidate["confusion_matrix"]
        if tp + fp >= 10 and 5 * tp >= 4 * (tp + fp):
            feasible.append((tp, candidate["threshold"]))
    if not feasible:
        return {"available": False, "reason": "no precision>=0.80 boundary with >=10 positives"}
    _, threshold = max(feasible)
    return {
        "available": True,
        "threshold": threshold,
        "selection_rule": "max recall subject to precision>=.80 and >=10 positives; highest tie",
        "calibration_metrics": metrics(labels, scores, threshold),
    }


def bootstrap(labels, sources, systems: dict, reference: str, resamples: int = 2000) -> dict:
    if resamples < 2000 or reference not in systems:
        raise ValueError("need >=2000 resamples and an explicit clean reference")
    groups = sorted(set(sources))
    mapping = {s: i for i, s in enumerate(groups)}
    group_index = np.array([mapping[s] for s in sources])
    y = np.array(labels)
    arrays = {name: (np.array(pred), np.array(score)) for name, (pred, score) in systems.items()}
    rng = np.random.default_rng(42)
    draws = {name: {} for name in systems}
    differences = {name: {} for name in systems if name != reference}
    for _ in range(resamples):
        count = np.bincount(rng.integers(0, len(groups), len(groups)), minlength=len(groups))
        values = {}
        for name, (pred, score) in arrays.items():
            value = _weighted_metrics(y, pred, score, count[group_index])
            value["fpr"] = (
                value["fp"] / (value["fp"] + value["tn"]) if value["fp"] + value["tn"] else None
            )
            value["fnr"] = (
                value["fn"] / (value["fn"] + value["tp"]) if value["fn"] + value["tp"] else None
            )
            values[name] = value
            for metric, number in value.items():
                draws[name].setdefault(metric, [])
                if number is not None:
                    draws[name][metric].append(number)
        for name, value in values.items():
            if name == reference:
                continue
            for metric, number in value.items():
                differences[name].setdefault(metric, [])
                other = values[reference][metric]
                if number is not None and other is not None:
                    differences[name][metric].append(number - other)

    def summarize(table):
        return {
            name: {
                metric: {
                    "low": float(np.percentile(v, 2.5)) if v else None,
                    "high": float(np.percentile(v, 97.5)) if v else None,
                    "valid_resamples": len(v),
                    "undefined_resamples": resamples - len(v),
                }
                for metric, v in values.items()
            }
            for name, values in table.items()
        }

    return {
        "resamples": resamples,
        "seed": 42,
        "source_count": len(groups),
        "method": "paired source percentile95; fixed models/thresholds; no selection correction",
        "reference": reference,
        "absolute": summarize(draws),
        "difference_from_reference": summarize(differences),
    }


def _prf(tp, fp, fn):
    p, r = tp / (tp + fp) if tp + fp else 0.0, tp / (tp + fn) if tp + fn else 0.0
    return {
        "precision": p,
        "recall": r,
        "f1": 2 * p * r / (p + r) if p + r else 0.0,
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }


def merge_ranges(ranges, text):
    result = []
    for start, end in sorted(ranges):
        while start < end and text[start].isspace():
            start += 1
        while end > start and text[end - 1].isspace():
            end -= 1
        if end <= start:
            continue
        if result and (start <= result[-1][1] or text[result[-1][1] : start].isspace()):
            result[-1] = (result[-1][0], max(result[-1][1], end))
        else:
            result.append((start, end))
    return result


def span_metrics(records, rows):
    """Custom micro metrics at fixed .5 token probability, NOT RAGTruth official scorer."""
    from grounded_hallucination_lab.research.span_model import token_targets

    counts = {kind: np.zeros(3, dtype=int) for kind in ("token", "character", "exact_span")}
    mapping = {r.id: r for r in records}
    for row in rows:
        record = mapping[row["id"]]
        gold = [(a.start, a.end) for a in record.annotations]
        labels = np.array(token_targets(row["offsets"], gold))
        predictions = np.array(row["token_probabilities"]) >= 0.5
        counts["token"] += [
            sum(predictions & (labels == 1)),
            sum(predictions & (labels == 0)),
            sum(~predictions & (labels == 1)),
        ]
        predicted_ranges = merge_ranges(
            [tuple(offset) for offset, p in zip(row["offsets"], predictions, strict=True) if p],
            record.response,
        )
        gold_ranges = merge_ranges(gold, record.response)
        gold_chars = {
            i for a, b in gold_ranges for i in range(a, b) if not record.response[i].isspace()
        }
        pred_chars = {
            i for a, b in predicted_ranges for i in range(a, b) if not record.response[i].isspace()
        }
        for key, predicted, expected in (
            ("character", pred_chars, gold_chars),
            ("exact_span", set(predicted_ranges), set(gold_ranges)),
        ):
            counts[key] += [
                len(predicted & expected),
                len(predicted - expected),
                len(expected - predicted),
            ]
    return {
        "threshold": 0.5,
        "official_scorer": False,
        "definitions": (
            "token:any gold char overlap; char:non-whitespace; "
            "exact_span:trimmed, overlapping/whitespace-adjacent merged ranges; micro"
        ),
        **{key: _prf(*map(int, value)) for key, value in counts.items()},
    }
