"""CPU-only matched-scope analysis on explicit development allowlists; no locked loader."""

import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from discovery_analyze import pair_statistics
from discovery_core import short_error
from research_r10 import records
from scope_eval import evaluate_op, evaluate_token_op, response_threshold

from grounded_hallucination_lab.data.ragtruth import gold_label
from grounded_hallucination_lab.eval.scored import scored_metrics
from grounded_hallucination_lab.research.metrics import metrics, span_metrics
from grounded_hallucination_lab.research.span_model import token_targets

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "models/scope_diagnosis_20261026"
PRED = ROOT / "predictions/scope_diagnosis_20261026"
REPORT = ROOT / "reports/scope_diagnosis_20261026"
OLD = ROOT / "predictions/discovery_loop_20261026"
METHODS = ("seg_all", "seg_filtered", "whole_all", "whole_filtered")
THRESHOLDS = {"B7": 0.37612172961235046, **dict.fromkeys(METHODS, 0.5)}
DATA = {
    "natural": ROOT / "models/phase2b_r10/natural_records.jsonl",
    "external_dev": ROOT
    / "models/discovery_loop_20261026/external/development_eligible_records.jsonl",
    "r1": ROOT / "models/phase2b_r10/r1_records.jsonl",
    "r6": ROOT / "predictions/phase2b_r6/transfer_records.jsonl",
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def rows(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def b7_path(dataset):
    return (
        OLD / "external_development_B7.jsonl"
        if dataset == "external_dev"
        else ROOT / "predictions/phase2b_r10" / f"B7_{dataset}.jsonl"
    )


def load_systems(dataset, ids):
    result = {"B7": rows(b7_path(dataset))}
    result.update({method: rows(PRED / f"{dataset}_{method}.jsonl") for method in METHODS})
    for predictions in result.values():
        assert [p["id"] for p in predictions] == ids
        assert [p["offsets"] for p in predictions] == [p["offsets"] for p in result["B7"]]
    for policy in ("all", "filtered"):
        assert [r["retained"] for r in result[f"seg_{policy}"]] == [
            r["retained"] for r in result[f"whole_{policy}"]
        ], "retained content must be identical"
    return result


def fast_values(counts):
    """Vectorized confusion-derived metrics; undefined ratios use zero, as point scorer."""
    tp, fp, fn, tn = np.moveaxis(np.asarray(counts, dtype=float)[..., :4], -1, 0)

    def divide(a, b):
        return np.divide(a, b, out=np.zeros_like(a, dtype=float), where=b != 0)

    result = {
        "macro_f1": (divide(2 * tp, 2 * tp + fp + fn) + divide(2 * tn, 2 * tn + fp + fn)) / 2,
        "precision": divide(tp, tp + fp),
        "recall": divide(tp, tp + fn),
        "positive_f1": divide(2 * tp, 2 * tp + fp + fn),
        "fpr": divide(fp, fp + tn),
        "fnr": divide(fn, fn + tp),
    }
    if np.shape(counts)[-1] == 7:
        a, b, c = np.moveaxis(np.asarray(counts, dtype=float)[..., 4:], -1, 0)
        result.update(
            token_f1=divide(2 * a, 2 * a + b + c),
            token_precision=divide(a, a + b),
            token_recall=divide(a, a + c),
        )
    return result


def confusion_rows(labels, predictions, token_counts=None):
    y, p = np.asarray(labels), np.asarray(predictions)
    result = np.column_stack(
        ((y == 1) & (p == 1), (y == 0) & (p == 1), (y == 1) & (p == 0), (y == 0) & (p == 0))
    ).astype(int)
    if token_counts is not None:
        result = np.column_stack((result, [[c["tp"], c["fp"], c["fn"]] for c in token_counts]))
    return result


def contrasts(methods):
    return [(name, "B7") for name in methods if name != "B7"] + [
        ("whole_all", "seg_all"),
        ("whole_filtered", "seg_filtered"),
        ("seg_filtered", "seg_all"),
        ("whole_filtered", "whole_all"),
    ]


def fast_bootstrap(labels, groups, response, token=None):
    """One 2000-draw source count matrix for every method and both operating points."""
    unique = sorted(set(groups))
    lookup = {value: i for i, value in enumerate(unique)}
    index = np.asarray([lookup[g] for g in groups])
    rng = np.random.default_rng(42)
    draws = np.stack(
        [
            np.bincount(rng.integers(0, len(unique), len(unique)), minlength=len(unique))
            for _ in range(2000)
        ]
    )

    def interval(values):
        return {
            key: {
                "low": float(np.percentile(value, 2.5)),
                "high": float(np.percentile(value, 97.5)),
            }
            for key, value in values.items()
        }

    output = {}
    for op in ("historical", "crossfit"):
        simulated, absolute = {}, {}
        for name, method in response["methods"].items():
            token_counts = token["methods"][name][op]["metrics"]["counts_by_row"] if token else None
            counts = confusion_rows(labels, method[op]["predictions"], token_counts)
            grouped = np.zeros((len(unique), counts.shape[1]), dtype=np.int64)
            np.add.at(grouped, index, counts)
            simulated[name] = fast_values(draws @ grouped)
            absolute[name] = interval(simulated[name])
        differences = {
            f"{a} - {b}": interval(
                {metric: simulated[a][metric] - simulated[b][metric] for metric in simulated[a]}
            )
            for a, b in contrasts(response["methods"])
        }
        output[op] = {"absolute": absolute, "differences": differences}
    return {
        "resamples": 2000,
        "seed": 42,
        "source_clusters": len(unique),
        "draws_sha256": hashlib.sha256(draws.tobytes()).hexdigest(),
        "scope": (
            "paired source percentile95, conditional on frozen OOF decisions; no threshold refit"
        ),
        "ranking": "AUROC/AP points only, not bootstrapped",
        **output,
    }


def point_summary(value):
    return {
        key: value[key]
        for key in (
            "n",
            "class_counts",
            "accuracy",
            "macro_f1",
            "positive_class",
            "auroc",
            "average_precision",
            "confusion_matrix",
        )
        if key in value
    } | {
        "fpr": value["confusion_matrix"][0][1] / sum(value["confusion_matrix"][0])
        if sum(value["confusion_matrix"][0])
        else None,
        "fnr": value["confusion_matrix"][1][0] / sum(value["confusion_matrix"][1])
        if sum(value["confusion_matrix"][1])
        else None,
    }


def coverage(records_, systems):
    counts, cases = Counter(), []
    for r, all_, filtered in zip(
        records_, systems["seg_all"], systems["seg_filtered"], strict=True
    ):
        gold = {
            i for a in r.annotations for i in range(a.start, a.end) if not r.response[i].isspace()
        }
        kept = {
            i for a, b in filtered["retained"] for i in range(a, b) if not r.response[i].isspace()
        }
        meta = {
            i
            for d in filtered["dropped"]
            if d["reason"] == "standalone_meta_or_refusal"
            for i in range(d["start"], d["end"])
            if not r.response[i].isspace()
        }
        formatting = {
            i
            for d in filtered["dropped"]
            if d["reason"] == "list_marker"
            for i in range(d["start"], d["end"])
            if not r.response[i].isspace()
        }
        mixed = bool(meta) and bool(filtered["retained"])
        refusal = any(
            re.search(
                r"\b(?:unable|cannot|can.t|can not)\b", r.response[d["start"] : d["end"]], re.I
            )
            for d in filtered["dropped"]
            if d["reason"] == "standalone_meta_or_refusal"
        )
        counts.update(
            rows=1,
            gold_chars=len(gold),
            gold_kept=len(gold & kept),
            gold_meta=len(gold & meta),
            gold_list_markers=len(gold & formatting),
            gold_other_uncovered=len(gold - kept - meta - formatting),
            zero_filtered=int(filtered["no_claim"]),
            mixed_meta=int(mixed),
            mixed_refusal=int(mixed and refusal),
            mixed_refusal_positive=int(mixed and refusal and bool(gold)),
        )
        if mixed or gold - kept or filtered["no_claim"]:
            cases.append(
                {
                    "id": r.id,
                    "source_id": r.source_id,
                    "response": r.response,
                    "annotations": [
                        {"start": a.start, "end": a.end, "label_type": a.label_type}
                        for a in r.annotations
                    ],
                    "retained": filtered["retained"],
                    "dropped": filtered["dropped"],
                    "mixed_refusal": mixed and refusal,
                    "all_units": all_["retained"],
                }
            )
    path = BASE / "coverage_cases.json"
    write(
        path,
        {"status": "annotation coverage, no independent human gold or relabeling", "cases": cases},
    )
    return {**dict(counts), "cases_path": str(path.relative_to(ROOT)), "cases_sha256": sha(path)}


def summarize_dataset(dataset, records_, systems, labels, source_ids, response, token):
    natural = dataset == "natural"
    result = {
        "n": len(labels),
        "source_clusters": len(set(response["folds"]["cluster_by_row"])),
        "methods": {},
        "strata": {},
    }
    for name, method in response["methods"].items():
        result["methods"][name] = {}
        for op in ("historical", "crossfit"):
            value = {"response": point_summary(method[op]["metrics"])}
            if op == "crossfit":
                value["response_fold_thresholds"] = [
                    s["threshold"] for s in method[op]["fold_selections"]
                ]
            else:
                value["response_threshold"] = method[op]["threshold"]
            if natural:
                tp = token["methods"][name][op]
                projected = [
                    {**p, "token_probabilities": [int(y == 1) for y in binary]}
                    for p, binary in zip(systems[name], tp["predictions"], strict=True)
                ]
                localization = span_metrics(records_, projected)
                for key in ("tp", "fp", "fn"):
                    assert localization["token"][key] == tp["metrics"][key]
                value["localization"] = localization
                value["localization"]["threshold"] = (
                    "binary projection of selected original-token threshold"
                )
                value["token_fold_thresholds" if op == "crossfit" else "token_threshold"] = (
                    [s["threshold"] for s in tp["fold_selections"]] if op == "crossfit" else 0.5
                )
                indices = [i for i, r in enumerate(records_) if short_error(r)]
                value["short_error"] = {
                    "n": len(indices),
                    "tp": sum(method[op]["predictions"][i] for i in indices),
                    "gold_token_hit": sum(
                        tp["metrics"]["counts_by_row"][i]["tp"] > 0 for i in indices
                    ),
                }
            result["methods"][name][op] = value
    dimensions = {"answer_tokens": defaultdict(list), "category": defaultdict(list)}
    if natural:
        dimensions["annotation_type"] = defaultdict(list)
        dimensions["generator"] = defaultdict(list)
    for i, r in enumerate(records_):
        size = len(systems["B7"][i]["offsets"])
        dimensions["answer_tokens"][
            "lt100" if size < 100 else "100_199" if size < 200 else "ge200"
        ].append(i)
        if natural:
            layout = systems["seg_filtered"][i]
            meta = any(d["reason"] == "standalone_meta_or_refusal" for d in layout["dropped"])
            category = "zero_kept" if layout["no_claim"] else "mixed_meta" if meta else "no_meta"
            dimensions["generator"][r.model].append(i)
            for category_ in {a.label_type for a in r.annotations} or {"no_annotated_error"}:
                dimensions["annotation_type"][category_].append(i)
        else:
            category = r["category"]
        dimensions["category"][category].append(i)
    for dimension, groups in dimensions.items():
        result["strata"][dimension] = {}
        for category, indices in groups.items():
            entry = {
                "n": len(indices),
                "sources": len({source_ids[i] for i in indices}),
                "methods": {},
            }
            for name, method in response["methods"].items():
                entry["methods"][name] = {}
                for op in ("historical", "crossfit"):
                    scores = [systems[name][i]["score"] for i in indices]
                    pred = [method[op]["predictions"][i] for i in indices]
                    value = point_summary(
                        scored_metrics([labels[i] for i in indices], pred, scores)
                    )
                    if natural:
                        tc = token["methods"][name][op]["metrics"]["counts_by_row"]
                        sums = {key: sum(tc[i][key] for i in indices) for key in ("tp", "fp", "fn")}
                        denominator = 2 * sums["tp"] + sums["fp"] + sums["fn"]
                        value["token_f1"] = 2 * sums["tp"] / denominator if denominator else 0.0
                    entry["methods"][name][op] = value
            result["strata"][dimension][category] = entry
    result["bootstrap"] = fast_bootstrap(
        labels, response["folds"]["cluster_by_row"], response, token
    )
    return result


def historical_references(dataset, labels, rr):
    paths = (
        {
            "old_segmented": OLD / "cycle2_FactCG_natural.jsonl",
            "old_whole_original_answer": OLD / "cycle4_FactCG_natural.jsonl",
        }
        if dataset == "natural"
        else {
            "old_segmented": OLD / "external_development_FactCG.jsonl",
            "old_whole_original_answer": OLD / "external_development_Whole.jsonl",
        }
    )
    result = {}
    for name, path in paths.items():
        predictions = rows(path)
        ids = [r.id for r in rr] if dataset == "natural" else [r["id"] for r in rr]
        assert [p["id"] for p in predictions] == ids
        result[name] = {
            "response": metrics(labels, [p["score"] for p in predictions], 0.5),
            "status": "historical reference only; outside matched factorial contrasts",
        }
        if dataset == "natural":
            result[name]["localization"] = span_metrics(rr, predictions)
    return result


def main():
    assert not (REPORT / "diagnostics.json").exists()
    expected = read(BASE / "freeze.json")
    for group in ("source", "input_files"):
        for path, digest in expected[group].items():
            assert sha(ROOT / path) == digest, path
    files = [
        Path(__file__),
        ROOT / "scripts/scope_eval.py",
        ROOT / "docs/SCOPE_DIAGNOSIS_PLAN.md",
        BASE / "freeze.json",
        *DATA.values(),
    ]
    for dataset in DATA:
        files += [b7_path(dataset), *[PRED / f"{dataset}_{m}.jsonl" for m in METHODS]]
    files += [
        ROOT / "models/phase2b_r10/r1_design.json",
        ROOT / "predictions/phase2b_r6/transfer_design.json",
        OLD / "cycle2_FactCG_natural.jsonl",
        OLD / "cycle4_FactCG_natural.jsonl",
        OLD / "external_development_FactCG.jsonl",
        OLD / "external_development_Whole.jsonl",
        ROOT / "scripts/discovery_analyze.py",
        ROOT / "scripts/discovery_core.py",
        ROOT / "src/grounded_hallucination_lab/research/metrics.py",
        ROOT / "src/grounded_hallucination_lab/research/span_model.py",
        ROOT / "src/grounded_hallucination_lab/eval/scored.py",
        ROOT / "src/grounded_hallucination_lab/eval/selection.py",
    ]
    freeze = {str(p.relative_to(ROOT)): sha(p) for p in files}
    write(
        BASE / "analysis_freeze.json",
        {
            "files": freeze,
            "bootstrap": 2000,
            "scope": "natural/explicit external development and existing controlled suites only",
        },
    )
    snapshot = BASE / "analysis_source"
    snapshot.mkdir(exist_ok=False)
    for p in files[:3]:
        (snapshot / p.name).write_bytes(p.read_bytes())
    results, natural_thresholds = {}, None
    for dataset in ("natural", "external_dev"):
        natural = dataset == "natural"
        rr = records(DATA[dataset]) if natural else rows(DATA[dataset])
        ids = [r.id for r in rr] if natural else [r["id"] for r in rr]
        sources = [r.source_id for r in rr] if natural else [r["source_id"] for r in rr]
        labels = [gold_label(r) for r in rr] if natural else [int(r["label"]) for r in rr]
        assert len(rr) == (642 if natural else 274)
        if not natural:
            assert all(
                r["partition"] == "development" and not r["span_labels_available"] for r in rr
            )
        systems = load_systems(dataset, ids)
        scores = {m: [p["score"] for p in values] for m, values in systems.items()}
        response = evaluate_op(labels, sources, scores, THRESHOLDS, ids=ids, natural=natural)
        write(BASE / f"{dataset}_response_operating_points.json", response)
        token = None
        if natural:
            selections = {m: response_threshold(labels, values) for m, values in scores.items()}
            for selection in selections.values():
                selection["selection_partition"] = (
                    "all_allowed_natural_development_for_controlled_transfer"
                )
            natural_thresholds = {m: s["threshold"] for m, s in selections.items()}
            write(
                BASE / "controlled_response_thresholds.json",
                {
                    "thresholds": natural_thresholds,
                    "selections": selections,
                    "controlled_labels_used": False,
                    "natural_ids_sha256": hashlib.sha256(json.dumps(ids).encode()).hexdigest(),
                },
            )
            target = [
                token_targets(p["offsets"], [(a.start, a.end) for a in r.annotations])
                for r, p in zip(rr, systems["B7"], strict=True)
            ]
            token = evaluate_token_op(target, sources, systems, ids=ids, natural=True)
            write(BASE / "natural_token_operating_points.json", token)
        early = {
            m: {
                op: {
                    "response": point_summary(v[op]["metrics"]),
                    "token_f1": token["methods"][m][op]["metrics"]["f1"] if token else None,
                }
                for op in ("historical", "crossfit")
            }
            for m, v in response["methods"].items()
        }
        write(BASE / f"{dataset}_early_points.json", early)
        print(dataset, "operating-point metrics saved", flush=True)
        results[dataset] = summarize_dataset(dataset, rr, systems, labels, sources, response, token)
        results[dataset]["historical_references"] = historical_references(dataset, labels, rr)
        if natural:
            results[dataset]["coverage_audit"] = coverage(rr, systems)
        print(
            dataset,
            {
                m: {
                    op: round(v[op]["response"]["macro_f1"], 4) for op in ("historical", "crossfit")
                }
                for m, v in results[dataset]["methods"].items()
            },
            flush=True,
        )
    assert (
        natural_thresholds is not None and (BASE / "controlled_response_thresholds.json").exists()
    )
    for dataset in ("r1", "r6"):
        rr = records(DATA[dataset])  # First controlled-label read: AFTER natural threshold freeze.
        labels, ids = [gold_label(r) for r in rr], [r.id for r in rr]
        systems = load_systems(dataset, ids)
        design = read(
            ROOT / "models/phase2b_r10/r1_design.json"
            if dataset == "r1"
            else ROOT / "predictions/phase2b_r6/transfer_design.json"
        )
        results[dataset] = {}
        for op, thresholds in (("historical", THRESHOLDS), ("natural_global", natural_thresholds)):
            results[dataset][op] = {
                "response": {
                    m: metrics(labels, [p["score"] for p in values], thresholds[m])
                    for m, values in systems.items()
                },
                "paired": pair_statistics(rr, systems, thresholds, design),
                "thresholds": thresholds,
            }
        print(
            dataset,
            {
                op: {m: v["pair_accuracy"] for m, v in values["paired"]["models"].items()}
                for op, values in results[dataset].items()
            },
            flush=True,
        )
    assert all(sha(ROOT / path) == digest for path, digest in freeze.items())
    write(
        REPORT / "diagnostics.json",
        {
            "datasets": results,
            "analysis_freeze_sha256": sha(BASE / "analysis_freeze.json"),
            "limitations": [
                "Reused development; cross-fit does not undo checkpoint/model selection.",
                "Bootstrap conditional on fixed OOF decisions; no threshold-refit uncertainty.",
                "External response labels derive from reviewed triplets; no external span scoring.",
                "Type strata overlap; annotation absence is not evidence entailment.",
            ],
            "supporting_arrays_directory": str(BASE.relative_to(ROOT)),
            "gpu_seconds": 0,
        },
    )


if __name__ == "__main__":
    main()
