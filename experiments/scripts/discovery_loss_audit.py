"""CPU-only target-mass and localization audit; never changes annotations or fits."""

import hashlib
import json
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer

from grounded_hallucination_lab.research.span_model import token_targets
from grounded_hallucination_lab.research.verifiers import punkt_spans

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "models/discovery_loop_20261026"
REPORT = ROOT / "reports/discovery_loop_20261026/loss_audit.json"
OLD = ROOT / "models/phase2b_r10"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def normalized(text):
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def quantiles(values):
    if not values:
        return {"n": 0}
    return {
        "n": len(values),
        **dict(
            zip(
                ("minimum", "p25", "median", "p75", "maximum"),
                map(float, np.percentile(values, [0, 25, 50, 75, 100])),
                strict=True,
            )
        ),
    }


def short_error(record):
    chars = {
        i
        for annotation in record["annotations"]
        for i in range(annotation["start"], annotation["end"])
        if not record["response"][i].isspace()
    }
    return (
        bool(chars)
        and len(record["response"]) >= 500
        and len(chars) <= 0.05 * sum(not c.isspace() for c in record["response"])
    )


def coefficients(positive, negative, positive_weight):
    """Sum of CE term coefficients, NOT measured loss or gradient magnitude."""
    total = positive + negative
    assert total > 0 and min(positive, negative) >= 0
    present = int(positive > 0) + int(negative > 0)
    return {
        "unweighted_mean_ce": [positive / total, negative / total],
        "historical_weighted_mean_ce": [positive_weight * positive / total, negative / total],
        "hypothetical_present_class_mean_ce": [
            int(positive > 0) / present,
            int(negative > 0) / present,
        ],
    }


def summarize(rows):
    out = {
        "responses": len(rows),
        "sources": len({r["source_id"] for r in rows}),
        "positive_responses": sum(r["positive_tokens"] > 0 for r in rows),
        "answer_tokens": sum(r["tokens"] for r in rows),
        "positive_tokens": sum(r["positive_tokens"] for r in rows),
        "positive_token_fraction_per_response": quantiles([r["fraction"] for r in rows]),
        "objective_coefficient_mass": {},
    }
    for objective in rows[0]["coefficients"]:
        pos = sum(r["coefficients"][objective][0] for r in rows)
        neg = sum(r["coefficients"][objective][1] for r in rows)
        out["objective_coefficient_mass"][objective] = {
            "positive": pos,
            "negative": neg,
            "positive_share": pos / (pos + neg),
            "positive_coefficient_per_response": quantiles(
                [r["coefficients"][objective][0] for r in rows]
            ),
        }
    if all(r["positive_tokens"] for r in rows):
        out["hypothetical_to_historical_positive_coefficient_ratio"] = quantiles(
            [
                r["coefficients"]["hypothetical_present_class_mean_ce"][0]
                / r["coefficients"]["historical_weighted_mean_ce"][0]
                for r in rows
            ]
        )
    return out


def main():
    assert not REPORT.exists(), "preserve prior audit"
    manifest = read(OLD / "R10_artifact_manifest.json")["files"]
    paths = {
        "train": OLD / "train_records.jsonl",
        "natural": OLD / "natural_records.jsonl",
        "b7_predictions": ROOT / "predictions/phase2b_r10/B7_natural.jsonl",
        "training_config": OLD / "R10As42/training_config.json",
    }
    hashes = {}
    for name, path in paths.items():
        sha = digest(path)
        assert sha == manifest[path.relative_to(ROOT).as_posix()]["sha256"], name
        hashes[name] = {"path": path.relative_to(ROOT).as_posix(), "sha256": sha}
    data = {
        split: [json.loads(line) for line in paths[split].read_text(encoding="utf-8").splitlines()]
        for split in ("train", "natural", "b7_predictions")
    }
    train, natural = data["train"], data["natural"]
    assert len(train) == 2568 and len(natural) == 642
    assert all(r["split"] == "train" and r["labels_available"] for r in train + natural)
    assert not {r["source_id"] for r in train} & {r["source_id"] for r in natural}
    weight = read(paths["training_config"])["positive_weight"]
    tokenizer_path = ROOT / "models/pretrained/backbone"
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
    targets = []
    for record in train:
        offsets = tokenizer(
            record["response"], add_special_tokens=False, return_offsets_mapping=True
        )["offset_mapping"]
        labels = token_targets(offsets, [(a["start"], a["end"]) for a in record["annotations"]])
        assert all(t in (0, 1) for t in labels)
        pos = sum(labels)
        targets.append(
            {
                "id": record["id"],
                "source_id": record["source_id"],
                "tokens": len(labels),
                "positive_tokens": pos,
                "fraction": pos / len(labels),
                "short_error": short_error(record),
                "coefficients": coefficients(pos, len(labels) - pos, weight),
            }
        )
    assert max(r["tokens"] for r in targets) <= 512, "audit assumes one R10 answer window"
    subsets = {
        "all": targets,
        "positive_only": [r for r in targets if r["positive_tokens"]],
        "negative_only": [r for r in targets if not r["positive_tokens"]],
        "long_short_error": [r for r in targets if r["short_error"]],
        "tokens_lt100": [r for r in targets if r["tokens"] < 100],
        "tokens_100_199": [r for r in targets if 100 <= r["tokens"] < 200],
        "tokens_ge200": [r for r in targets if r["tokens"] >= 200],
    }
    natural_by_id = {r["id"]: r for r in natural}
    counts, bins = Counter(), defaultdict(Counter)
    peak_examples = []
    for prediction in data["b7_predictions"]:
        record = natural_by_id[prediction["id"]]
        values = np.asarray(prediction["token_probabilities"])
        k = int(np.argmax(values))
        a, b = prediction["offsets"][k]
        gold = bool(record["annotations"])
        positive = prediction["score"] >= 0.37612172961235046
        labels = token_targets(
            prediction["offsets"], [(a["start"], a["end"]) for a in record["annotations"]]
        )
        punctuation = not re.search(r"\w", record["response"][a:b])
        counts["responses"] += 1
        counts["predicted_positive"] += int(positive)
        counts["positive_without_any_token_ge_point5"] += int(positive and max(values) < 0.5)
        counts["gold_positive_detected"] += int(gold and positive)
        counts["detected_positive_peak_off_gold"] += int(gold and positive and not labels[k])
        counts["predicted_positive_nonword_peak"] += int(positive and punctuation)
        if positive and punctuation:
            peak_examples.append(
                {"id": record["id"], "offset": [a, b], "gold": gold, "score": float(max(values))}
            )
        key = "lt100" if len(values) < 100 else "100_199" if len(values) < 200 else "ge200"
        c = bins[key]
        for name, value in {
            "responses": 1,
            "positive": gold,
            "negative": not gold,
            "fp": positive and not gold,
            "tp": positive and gold,
        }.items():
            c[name] += int(value)
    sentences, passages = defaultdict(list), defaultdict(set)
    for split in ("train", "natural"):
        for record in data[split]:
            for text in re.split(r"passage\s+\d+:", record["evidence"], flags=re.I)[1:]:
                if len(normalized(text)) >= 80:
                    passages[normalized(text)].add((split, record["source_id"]))
            for sentence in punkt_spans(record["response"]):
                key = normalized(sentence.text)
                if len(key) < 35:
                    continue
                overlap = [
                    annotation
                    for annotation in record["annotations"]
                    if annotation["start"] < sentence.end and annotation["end"] > sentence.start
                ]
                sentences[(record["source_id"], key)].append(
                    {
                        "id": record["id"],
                        "partition": split,
                        "annotated_overlap": bool(overlap),
                        "sentence_offset": [sentence.start, sentence.end],
                        "overlapping_annotations": overlap,
                        "question": record["question"],
                        "evidence": record["evidence"],
                        "response": record["response"],
                    }
                )
    disagreements = [
        {"source_id": sid, "normalized_sentence": text, "occurrences": occurrences}
        for (sid, text), occurrences in sentences.items()
        if len({o["annotated_overlap"] for o in occurrences}) > 1
    ]
    case_path = BASE / "loss_annotation_cases.json"
    write(
        case_path,
        {
            "status": "automatic disagreement flags; NOT independent human annotations",
            "warning": "Same sentence can have different truth under headings/attribution.",
            "cases": disagreements,
        },
    )
    duplicates = [
        {"normalized_chars": len(text), "source_groups": sorted(groups)}
        for text, groups in passages.items()
        if len(groups) > 1
    ]
    # Formula-only tests; no logits, losses, gradients, or fitting.
    assert coefficients(0, 100, weight)["hypothetical_present_class_mean_ce"] == [0.0, 1.0]
    assert coefficients(1, 99, weight)["hypothetical_present_class_mean_ce"] == [0.5, 0.5]
    assert coefficients(100, 0, weight)["hypothetical_present_class_mean_ce"] == [1.0, 0.0]
    report = {
        "status": "CPU-only descriptive audit; no fit, no relabeling, no new predictions",
        "inputs": hashes,
        "script_sha256": digest(Path(__file__)),
        "tokenizer_hashes": {
            p.name: digest(p)
            for p in tokenizer_path.iterdir()
            if p.is_file() and p.suffix == ".json"
        },
        "definitions": {
            "historical_weight": weight,
            "historical_objective": "sum(weight[class] * token CE) / answer token count",
            "unweighted_objective": "mean token CE per response",
            "hypothetical_objective": (
                "mean CE within each present class, then mean over present classes per response; "
                "negative-only remains ordinary mean CE"
            ),
            "mass": "Sum of scalar CE coefficients, NOT actual CE loss/gradient magnitude",
            "batch_caveat": (
                "Exact for R10 raw+raw duplicate pairs: equal answer lengths, one window. "
                "For mixed-length multiresponse batches or overlap windows, "
                "recompute actual weights."
            ),
            "short_error": ">=500 chars and gold nonspace chars <=5% of response nonspace chars",
            "scope": "Include/all policy; no source/doc/special tokens in targets",
        },
        "training": {name: summarize(rows) for name, rows in subsets.items()},
        "natural_B7": {
            "counts": dict(counts),
            "by_answer_token_length": {
                key: {
                    **dict(c),
                    "prevalence": c["positive"] / c["responses"],
                    "fpr": c["fp"] / c["negative"],
                    "recall": c["tp"] / c["positive"],
                }
                for key, c in bins.items()
            },
            "nonword_peak_offsets": peak_examples,
            "nonword_caveat": "Non-word does not mean erroneous: currency/unit symbols matter.",
        },
        "annotation_disagreement": {
            "sentence_groups": len(disagreements),
            "sources": len({d["source_id"] for d in disagreements}),
            "cases_path": case_path.relative_to(ROOT).as_posix(),
            "cases_sha256": digest(case_path),
            "interpretation": (
                "Not an annotation error rate. 12362 changes magnesium/calcium heading; "
                "12681 changes passage attribution. 17627 repeated foil instruction is a "
                "candidate omission, not adjudicated human gold. No span propagation justified."
            ),
        },
        "shared_passages": duplicates,
        "cross_partition_shared_passages": sum(
            len({part for part, _ in d["source_groups"]}) > 1 for d in duplicates
        ),
        "diagnostic_verdict": (
            "There is an auditable target-coefficient dilution mechanism for short errors, "
            "not causal evidence that it limits training. Balanced per-response CE would also "
            "change large-span weighting and global class mass. It warrants a small matched "
            "pilot only after a gradient/loss audit and retained-factual-sensitivity gate; "
            "these counts do not establish superiority over a claim verifier. Context-aware "
            "claim evaluation is needed because identical clauses can change meaning "
            "under headings."
        ),
        "formula_tests_passed": 3,
        "gpu_seconds": 0,
    }
    write(REPORT, report)
    print(
        {
            "train_tokens": report["training"]["all"]["answer_tokens"],
            "positive_tokens": report["training"]["all"]["positive_tokens"],
            "long_short_error_responses": report["training"]["long_short_error"]["responses"],
            "sentence_disagreements_not_gold_errors": len(disagreements),
            "report": REPORT.relative_to(ROOT).as_posix(),
        }
    )


if __name__ == "__main__":
    main()
