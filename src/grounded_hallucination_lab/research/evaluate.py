"""One evaluation of the preselected batch, with no model/threshold changes."""

import hashlib

import numpy as np

from grounded_hallucination_lab.data.ragtruth import gold_label
from grounded_hallucination_lab.research.metrics import bootstrap, metrics, span_metrics
from grounded_hallucination_lab.research.protocol import (
    OUT,
    PRED,
    REPORT,
    ROOT,
    digest,
    now,
    partition,
    read_json,
    read_rows,
    write_json,
)


def evaluate(parts):
    from grounded_hallucination_lab.research.run import check_freeze

    frozen = check_freeze()
    thresholds = read_json(OUT / "thresholds.json")
    if thresholds["batch_freeze"] != digest(OUT / "batch_freeze.json"):
        raise ValueError("thresholds do not belong to this frozen batch")
    records = partition(parts, "dev")
    labels = [gold_label(r) for r in records]
    sources = [r.source_id for r in records]
    ids = [r.id for r in records]
    methods = {}
    all_rows, systems = {}, {}
    for method in ["B0", *frozen["methods"]]:
        if method == "B0":
            rows = read_rows(ROOT / "predictions/phase2a/dev.jsonl")
            threshold = read_json(ROOT / "reports/lexical_dev.json")["threshold"]
            runtime = read_json(ROOT / "reports/lexical_dev.json")["runtime"]
            point = thresholds["historical_precision_point"]
        else:
            rows = read_rows(PRED / f"{method}_dev.jsonl")
            meta = read_json(PRED / f"{method}_dev.meta.json")
            if meta["provenance"] != frozen["provenance"]:
                raise ValueError("prediction metadata does not match frozen environment/code")
            if meta["prediction_sha256"] != digest(PRED / f"{method}_dev.jsonl"):
                raise ValueError("predictions changed")
            if meta["thresholds_sha256"] != digest(OUT / "thresholds.json"):
                raise ValueError("thresholds changed after dev scoring")
            threshold = thresholds["methods"][method]["selection"]["threshold"]
            point = thresholds["methods"][method]["precision_point"]
            runtime = meta["runtime"]
        if [r["id"] for r in rows] != ids or [r["source_id"] for r in rows] != sources:
            raise ValueError(f"response/source IDs differ for {method}")
        scores = [r["score"] for r in rows]
        report = {
            "metrics": metrics(labels, scores, threshold),
            "runtime": runtime,
            "eligible_main_comparison": method != "B3",
            "provenance_status": "insufficient training provenance"
            if method == "B3"
            else "no known overlap; pretraining contamination not excluded",
            "precision80_point": {
                "calibration": point,
                "dev_metrics": metrics(labels, scores, point["threshold"])
                if point["available"]
                else None,
            },
        }
        report["by_generator"] = {}
        for generator in sorted({r.model for r in records}):
            index = [i for i, r in enumerate(records) if r.model == generator]
            report["by_generator"][generator] = metrics(
                [labels[i] for i in index], [scores[i] for i in index], threshold
            )
        report["by_answer_length_characters"] = {}
        for low, high in ((0, 250), (250, 500), (500, 1000), (1000, 100000)):
            index = [i for i, r in enumerate(records) if low <= len(r.response) < high]
            if index:
                report["by_answer_length_characters"][f"{low}:{high}"] = metrics(
                    [labels[i] for i in index], [scores[i] for i in index], threshold
                )
        report["positive_type_recall_nonexclusive"] = {}
        kinds = sorted({a.label_type for r in records for a in r.annotations})
        for kind in [*kinds, "implicit_true"]:
            index = [
                i
                for i, r in enumerate(records)
                if any(
                    a.implicit_true if kind == "implicit_true" else a.label_type == kind
                    for a in r.annotations
                )
            ]
            report["positive_type_recall_nonexclusive"][kind] = {
                "n_positive_responses": len(index),
                "detected": sum(scores[i] >= threshold for i in index),
                "recall": sum(scores[i] >= threshold for i in index) / len(index)
                if index
                else None,
            }
        if method.startswith(("B7", "B8")):
            report["span_task"] = span_metrics(records, rows)
        methods[method] = report
        all_rows[method] = rows
        systems[method] = ([int(s >= threshold) for s in scores], scores)
    reference = max(
        (m for m in ("B2", "B4", "B5", "B6") if m in methods),
        key=lambda name: methods[name]["metrics"]["macro_f1"],
    )
    intervals = bootstrap(labels, sources, systems, reference, 2000)
    seeds = {key: value["metrics"] for key, value in methods.items() if key.startswith("B7")}
    seed_summary = {
        metric: {
            "mean": float(np.mean([v[metric] for v in seeds.values()])),
            "sample_std": float(np.std([v[metric] for v in seeds.values()], ddof=1)),
            "all_values": [v[metric] for v in seeds.values()],
        }
        for metric in ("macro_f1", "accuracy", "auroc", "average_precision")
    }
    result = {
        "created_at": now(),
        "evaluation_role": "development benchmark, already inspected",
        "n": len(records),
        "source_count": len(set(sources)),
        "methods": methods,
        "strongest_clean_specialized_baseline_in_batch": reference,
        "reference_selection_note": (
            "descriptive dev max across clean MiniCheck pipelines, including improvements; "
            "not a confirmatory test or a global SOTA comparison"
        ),
        "bootstrap": intervals,
        "own_training_seeds": seeds,
        "seed_summary": seed_summary,
        "batch_freeze_sha256": digest(OUT / "batch_freeze.json"),
        "thresholds_sha256": digest(OUT / "thresholds.json"),
        "test_evaluated": False,
    }
    write_json(REPORT / "dev_results.json", result)
    examples = {}
    reference_pred = systems[reference][0]
    order = sorted(
        range(len(records)),
        key=lambda i: hashlib.sha256(f"errors:42:{ids[i]}".encode()).hexdigest(),
    )
    for method, (predictions, scores) in systems.items():
        categories = {
            "corrected_vs_baseline": [
                i for i in order if predictions[i] == labels[i] != reference_pred[i]
            ],
            "new_errors_vs_baseline": [
                i for i in order if predictions[i] != labels[i] == reference_pred[i]
            ],
            "false_positive": [i for i in order if predictions[i] == 1 and labels[i] == 0],
            "false_negative": [i for i in order if predictions[i] == 0 and labels[i] == 1],
        }
        examples[method] = {
            category: [
                {
                    "id": ids[i],
                    "source_id": sources[i],
                    "gold": labels[i],
                    "prediction": predictions[i],
                    "score": scores[i],
                    "reference_prediction": reference_pred[i],
                    "question": records[i].question,
                    "response": records[i].response,
                    "evidence": records[i].evidence,
                    "gold_spans": [
                        {
                            "start": a.start,
                            "end": a.end,
                            "type": a.label_type,
                            "implicit_true": a.implicit_true,
                        }
                        for a in records[i].annotations
                    ],
                    "diagnostics": all_rows[method][i],
                }
                for i in selected[:15]
            ]
            for category, selected in categories.items()
        }
    write_json(PRED / "error_examples.json", examples)
    print({name: value["metrics"] for name, value in methods.items()}, flush=True)
    return result
