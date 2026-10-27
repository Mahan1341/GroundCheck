"""Train-only evidence sensitivity; no new labels assigned to natural perturbations."""

import argparse
import hashlib
from dataclasses import replace

import numpy as np

from grounded_hallucination_lab.research.protocol import (
    OUT,
    PRED,
    REPORT,
    InferenceInput,
    load_parts,
    read_json,
    write_json,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("method", choices=["B2", "B7s42"])
    args = parser.parse_args()
    parts = load_parts()
    ordered = sorted(
        parts.train, key=lambda r: hashlib.sha256(f"sensitivity:42:{r.id}".encode()).hexdigest()
    )
    records, used = [], set()
    for record in ordered:
        if record.source_id not in used:
            records.append(record)
            used.add(record.source_id)
        if len(records) == 32:
            break
    conditions = {
        "original": records,
        "blank_evidence": [replace(r, evidence="") for r in records],
        "cyclic_different_source": [
            replace(r, evidence=records[(i + 1) % len(records)].evidence)
            for i, r in enumerate(records)
        ],
    }
    toy_pairs = [
        ("The bridge opened in 1990.", "The bridge opened in 1991.", "The bridge opened in 1990."),
        ("Alice owns a blue car.", "Alice does not own a car.", "Alice owns a car."),
        ("Alice owns blue; Bob owns red.", "Alice owns red; Bob owns blue.", "Alice owns blue."),
    ]
    for name, evidence_index in (("toy_supported", 0), ("toy_contradicted", 1)):
        conditions[name] = [
            replace(
                records[0],
                id=f"toy{i}",
                source_id=f"toy{i}",
                question="",
                evidence=pair[evidence_index],
                response=pair[2],
                annotations=(),
            )
            for i, pair in enumerate(toy_pairs)
        ]
    rows, timings = {}, {}
    if args.method == "B2":
        from grounded_hallucination_lab.research.verifiers import Verifier, score_records

        verifier = Verifier("minicheck")
        for name, changed in conditions.items():
            rows[name], timings[name] = score_records(
                verifier, [InferenceInput.from_record(r) for r in changed], "author"
            )
    else:
        from grounded_hallucination_lab.research.span_model import encode_records, load_model, score

        folder = OUT / "span_final_s42"
        config = read_json(folder / "training_config.json")
        model, tokenizer = load_model(folder / "best")
        for name, changed in conditions.items():
            examples, info = encode_records(changed, tokenizer, config["max_length"], False)
            rows[name], timings[name] = score(model, tokenizer, examples, info)
    original = np.array([r["score"] for r in rows["original"]])
    summary = {}
    for name, scored in rows.items():
        if name.startswith("toy_"):
            continue
        scores = np.array([r["score"] for r in scored])
        delta = scores - original
        summary[name] = {
            "mean_risk": float(scores.mean()),
            "mean_delta": float(delta.mean()),
            "median_delta": float(np.median(delta)),
            "fraction_risk_increased": float((delta > 0).mean()),
        }
    write_json(
        REPORT / f"sensitivity_{args.method}.json",
        {
            "partition": "train",
            "n": len(records),
            "response_ids": [r.id for r in records],
            "source_ids": [r.source_id for r in records],
            "cyclic_donor_source_ids": [
                records[(i + 1) % len(records)].source_id for i in range(len(records))
            ],
            "labels_reassigned": False,
            "interpretation": (
                "sensitivity only; arbitrary substituted evidence is not a new verified gold label"
            ),
            "summary": summary,
            "controlled_toy_checks": [
                {
                    "kind": kind,
                    "supported_risk": a["score"],
                    "contradicted_risk": b["score"],
                    "expected_direction_passed": b["score"] > a["score"],
                }
                for kind, a, b in zip(
                    ("number", "negation", "entity_relation"),
                    rows["toy_supported"],
                    rows["toy_contradicted"],
                    strict=True,
                )
            ],
            "blank_evidence_caveat": (
                "MiniCheck author adapter has zero chunks for blank evidence and returns risk=1 "
                "by missing-support policy; this is not learned evidence sensitivity."
            ),
            "timings": timings,
        },
    )
    write_json(PRED / f"sensitivity_{args.method}.json", rows)
    print(summary)


if __name__ == "__main__":
    main()
