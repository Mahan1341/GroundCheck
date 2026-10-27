"""R1 train-side diagnostics only. No calibration/dev/test mode."""

import argparse
import hashlib
from dataclasses import asdict, replace

import numpy as np

from grounded_hallucination_lab.baselines.lexical import sentences
from grounded_hallucination_lab.data.ragtruth import gold_label
from grounded_hallucination_lab.eval.selection import select_threshold
from grounded_hallucination_lab.research.controls import controls
from grounded_hallucination_lab.research.metrics import bootstrap, metrics, span_metrics
from grounded_hallucination_lab.research.protocol import (
    OUT,
    PRED,
    ROOT,
    InferenceInput,
    digest,
    load_parts,
    partition,
    preserve_code,
    provenance,
    read_json,
    read_rows,
    write_json,
    write_rows,
)

REPORT = ROOT / "reports/phase2b_resume"
PREDICTIONS = ROOT / "predictions/phase2b_resume"


def score_model(method, records):
    if method == "B6":
        from grounded_hallucination_lab.research.verifiers import Verifier, score_records

        return score_records(
            Verifier("minicheck", 4), [InferenceInput.from_record(r) for r in records], "clean"
        )
    from grounded_hallucination_lab.research.span_model import encode_records, load_model, score

    folder = (
        ROOT / f"models/phase2b_resume/{method}_inner_s42"
        if method in ("C0", "C1")
        else OUT / ("span_inner_s42" if method == "B7" else "answer_inner_s42")
    )
    config = read_json(folder / "training_config.json")
    heldout = partition(load_parts(), "inner_val")
    forbidden = {r.id for r in heldout}
    forbidden_sources = {r.source_id for r in heldout}
    if set(config["train_ids"]) & forbidden or set(config["train_sources"]) & forbidden_sources:
        raise ValueError("train-side validation model has seen inner-validation labels")
    model, tokenizer = load_model(folder / "best")
    examples, metadata = encode_records(records, tokenizer, config["max_length"], method == "B8")
    rows, runtime = score(model, tokenizer, examples, metadata, batch_size=1)
    runtime["checkpoint_sha256"] = digest(folder / "best/model.safetensors")
    return rows, runtime


def control_score(method):
    records, design = controls()
    rows, runtime = score_model(method, records)
    point = read_json(
        PREDICTIONS / f"{method}_inner_val.meta.json"
        if method in ("C0", "C1")
        else PRED / f"{method if method == 'B6' else method + 's42'}_inner_val.meta.json"
    )
    threshold = point["inner_selection"]["threshold"]
    paired = {}
    for row, info in zip(rows, design, strict=True):
        assert row["id"] == info["id"]
        if "offsets" in row:
            row["claim_risk_diagnostic"] = max(
                (
                    p
                    for (a, b), p in zip(row["offsets"], row["token_probabilities"], strict=True)
                    if a < info["claim_end"] and b > info["claim_start"]
                ),
                default=0.0,
            )
        key = (info["group"], info["padding_sentences"], info["position"])
        paired.setdefault(key, {})[info["condition"]] = row
    pairs = []
    for (group, padding, position), conditions in paired.items():
        good, bad = conditions["supported"], conditions["contradicted"]
        pairs.append(
            {
                "group": group,
                "kind": group.split(":")[0],
                "padding": padding,
                "position": position,
                "supported_risk": good["score"],
                "contradicted_risk": bad["score"],
                "delta": bad["score"] - good["score"],
                "correct_pair_at_natural_threshold": good["score"] < threshold <= bad["score"],
                "claim_delta": bad.get("claim_risk_diagnostic", bad["score"])
                - good.get("claim_risk_diagnostic", good["score"]),
                "paraphrase_delta": conditions["paraphrase"]["score"] - good["score"],
                "reordered_delta": conditions["reordered"]["score"] - good["score"],
            }
        )
    summaries = {}
    for key in ["all", *sorted({p["kind"] for p in pairs}), "padding0", "padding8", "padding32"]:
        selected = [
            p for p in pairs if key == "all" or p["kind"] == key or key == f"padding{p['padding']}"
        ]
        summaries[key] = {
            "paired_cases": len(selected),
            "fraction_risk_increased": float(np.mean([p["delta"] > 1e-6 for p in selected])),
            "correct_pair_fraction": float(
                np.mean([p["correct_pair_at_natural_threshold"] for p in selected])
            ),
            **{
                field: float(np.mean([p[field] for p in selected]))
                for field in ("delta", "claim_delta", "paraphrase_delta", "reordered_delta")
            },
        }
    report = {
        "method": method,
        "n": len(rows),
        "underlying_fact_groups": 15,
        "threshold_from_natural_inner": threshold,
        "runtime": runtime,
        "summary": summaries,
        "paired_results": pairs,
        "metrics_on_authored_suite": metrics(
            [d["gold"] for d in design], [r["score"] for r in rows], threshold
        ),
        "interpretation": "controlled stress suite, not a population benchmark; no tuning",
        "provenance": provenance(),
    }
    write_rows(PREDICTIONS / f"controls_{method}.jsonl", rows)
    write_json(PREDICTIONS / "controls_design.json", design) if not (
        PREDICTIONS / "controls_design.json"
    ).exists() else None
    write_json(REPORT / f"controls_{method}.json", report)
    print(summaries, flush=True)


def perturb(method):
    records = partition(load_parts(), "inner_val")
    selected, used = [], set()
    for r in sorted(records, key=lambda r: hashlib.sha256(f"R1:42:{r.id}".encode()).hexdigest()):
        if r.source_id not in used:
            selected.append(r)
            used.add(r.source_id)
        if len(selected) == 32:
            break
    conditions = {
        "original": selected,
        "blank": [replace(r, evidence="") for r in selected],
        "cyclic": [
            replace(r, evidence=selected[(i + 1) % len(selected)].evidence)
            for i, r in enumerate(selected)
        ],
        "duplicate": [replace(r, evidence=r.evidence + "\n" + r.evidence) for r in selected],
    }
    all_rows, runtimes = {}, {}
    for condition, data in conditions.items():
        all_rows[condition], runtimes[condition] = score_model(method, data)
    base = np.array([r["score"] for r in all_rows["original"]])
    summary = {}
    for condition, rows in all_rows.items():
        delta = np.array([r["score"] for r in rows]) - base
        summary[condition] = {
            "mean_risk": float(np.mean([r["score"] for r in rows])),
            "mean_delta": float(delta.mean()),
            "median_delta": float(np.median(delta)),
            "fraction_increased": float(np.mean(delta > 1e-6)),
        }
    write_json(PREDICTIONS / f"perturb_{method}.json", all_rows)
    write_json(
        REPORT / f"perturb_{method}.json",
        {
            "method": method,
            "partition": "inner_val",
            "n_sources": 32,
            "ids": [r.id for r in selected],
            "summary": summary,
            "runtime": runtimes,
            "labels_reassigned": False,
            "provenance": provenance(),
            "caveat": "arbitrary evidence substitution is sensitivity, not new verified gold",
        },
    )
    print(summary, flush=True)


def natural_analysis(include_cf=False):
    records = partition(load_parts(), "inner_val")
    labels = [gold_label(r) for r in records]
    methods, systems, saved = {}, {}, {}
    method_names = ["B7s42", "B8s42", "B6", "B9"] + (["C0", "C1"] if include_cf else [])
    for method in method_names:
        directory = PREDICTIONS if method in ("C0", "C1") else PRED
        rows = read_rows(directory / f"{method}_inner_val.jsonl")
        assert [r["id"] for r in rows] == [r.id for r in records]
        scores = [r["score"] for r in rows]
        threshold = select_threshold(labels, scores)["threshold"]
        predictions = [int(s >= threshold) for s in scores]
        methods[method] = {"metrics": metrics(labels, scores, threshold)}
        if "offsets" in rows[0]:
            methods[method]["custom_span_metrics"] = span_metrics(records, rows)
        systems[method] = (predictions, scores)
        saved[method] = rows
    token_lengths = [len(r["offsets"]) for r in saved["B7s42"]]
    sentence_counts = [len(sentences(r.response)) for r in records]
    for name, value in methods.items():
        pred, scores = systems[name]
        threshold = value["metrics"]["threshold"]
        groups = {}
        for lo, hi in ((0, 64), (64, 128), (128, 10000)):
            groups[f"answer_tokens:{lo}:{hi}"] = [
                i for i, n in enumerate(token_lengths) if lo <= n < hi
            ]
        for lo, hi in ((0, 3), (3, 6), (6, 10000)):
            groups[f"sentences:{lo}:{hi}"] = [
                i for i, n in enumerate(sentence_counts) if lo <= n < hi
            ]
        for generator in sorted({r.model for r in records}):
            groups[f"generator:{generator}"] = [
                i for i, r in enumerate(records) if r.model == generator
            ]
        value["positive_type_recall"] = {}
        for kind in sorted({a.label_type for r in records for a in r.annotations}):
            indices = [
                i for i, r in enumerate(records) if any(a.label_type == kind for a in r.annotations)
            ]
            value["positive_type_recall"][kind] = {
                "n": len(indices),
                "detected": sum(pred[i] for i in indices),
                "recall": sum(pred[i] for i in indices) / len(indices),
                "caveat": "positive-only overlapping annotation groups; not separate binary tasks",
            }
        value["subgroups"] = {
            group: metrics([labels[i] for i in idx], [scores[i] for i in idx], threshold)
            for group, idx in groups.items()
            if idx
        }
        short = [
            i
            for i, r in enumerate(records)
            if labels[i]
            and len(r.response) >= 500
            and sum(a.end - a.start for a in r.annotations) / len(r.response) <= 0.05
        ]
        isolated = [i for i, r in enumerate(records) if r.source_id != "15181"]
        value["passage_overlap_sensitivity"] = metrics(
            [labels[i] for i in isolated], [scores[i] for i in isolated], threshold
        )
        value["long_answer_short_error"] = {
            "n_positive": len(short),
            "detected": sum(pred[i] for i in short),
            "ids": [records[i].id for i in short],
        }
        if "offsets" in saved[name][0]:
            peaks = [
                max(
                    (
                        probability
                        for (start, end), probability in zip(
                            saved[name][i]["offsets"],
                            saved[name][i]["token_probabilities"],
                            strict=True,
                        )
                        if any(start < a.end and end > a.start for a in records[i].annotations)
                    ),
                    default=0.0,
                )
                for i in short
            ]
            value["long_answer_short_error"].update(
                {
                    "localized_at_fixed_token_threshold_0_5": sum(p >= 0.5 for p in peaks),
                    "mean_gold_overlap_token_peak": float(np.mean(peaks)) if peaks else None,
                    "interpretation": "post-hoc gold overlap; not inference input",
                }
            )
    intervals = bootstrap(labels, [r.source_id for r in records], systems, "B7s42", 2000)
    suffix = "_cf" if include_cf else ""
    write_json(
        REPORT / f"natural_inner_comparison{suffix}.json",
        {
            "n": len(records),
            "sources": len({r.source_id for r in records}),
            "methods": methods,
            "paired_bootstrap": intervals,
            "paired_bootstrap_external": bootstrap(
                labels, [r.source_id for r in records], systems, "B6", 2000
            ),
            "paired_bootstrap_control": (
                bootstrap(labels, [r.source_id for r in records], systems, "C0", 2000)
                if include_cf
                else None
            ),
            "role": "inner selection; CIs do not correct threshold/model selection",
            "provenance": provenance(),
        },
    )
    examples = []
    for i, r in enumerate(records):
        if any(systems["B7s42"][0][i] != systems[m][0][i] for m in method_names) or (
            labels[i] and sum(a.end - a.start for a in r.annotations) <= 20
        ):
            examples.append(
                {
                    "record": asdict(r),
                    "gold": labels[i],
                    "scores": {m: systems[m][1][i] for m in methods},
                    "predictions": {m: systems[m][0][i] for m in methods},
                }
            )
    write_json(PREDICTIONS / f"natural_inner_audit{suffix}.json", examples)
    print({k: v["metrics"] for k, v in methods.items()}, flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["controls", "perturb", "analyze", "analyze-cf"])
    parser.add_argument("--method", choices=["B7", "B8", "B6", "C0", "C1"], default="B7")
    args = parser.parse_args()
    state = preserve_code()
    if args.command == "controls":
        control_score(args.method)
    elif args.command == "perturb":
        perturb(args.method)
    else:
        natural_analysis(args.command == "analyze-cf")
    if provenance() != state:
        raise ValueError("research source changed during command")


if __name__ == "__main__":
    main()
