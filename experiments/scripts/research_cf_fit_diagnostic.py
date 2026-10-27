"""In-sample fit diagnostic, never a generalization benchmark or selection loop."""

from collections import defaultdict

import numpy as np

from grounded_hallucination_lab.data.ragtruth import gold_label
from grounded_hallucination_lab.research.counterfactual import make_pairs
from grounded_hallucination_lab.research.metrics import metrics, span_metrics
from grounded_hallucination_lab.research.protocol import (
    OUT,
    PRED,
    ROOT,
    digest,
    inner_split,
    load_parts,
    preserve_code,
    provenance,
    read_json,
    write_json,
    write_rows,
)
from grounded_hallucination_lab.research.span_model import encode_records, load_model, score


def main():
    before = preserve_code()
    fit, validation = inner_split(load_parts().train)
    records, _ = make_pairs(fit, read_json(ROOT / "configs/counterfactual_acceptance.json"))
    reports = {}
    for method in ("B7", "C0", "C1"):
        folder = (
            OUT / "span_inner_s42"
            if method == "B7"
            else (ROOT / f"models/phase2b_resume/{method}_inner_s42")
        )
        config = read_json(folder / "training_config.json")
        assert set(config["train_sources"]) <= {r.source_id for r in fit}
        assert not set(config["train_sources"]) & {r.source_id for r in validation}
        point = read_json(
            PRED / "B7s42_inner_val.meta.json"
            if method == "B7"
            else (ROOT / f"predictions/phase2b_resume/{method}_inner_val.meta.json")
        )
        threshold = point["inner_selection"]["threshold"]
        model, tokenizer = load_model(folder / "best")
        examples, metadata = encode_records(records, tokenizer, 2048, False)
        rows, runtime = score(model, tokenizer, examples, metadata, batch_size=1)
        groups = defaultdict(list)
        for record, row in zip(records, rows, strict=True):
            assert record.id == row["id"]
            groups[(record.source_id, record.response)].append((gold_label(record), row["score"]))
        paired = []
        for group in groups.values():
            assert sorted(label for label, _ in group) == [0, 1]
            ordered = dict(group)
            paired.append((ordered[0], ordered[1]))
        reports[method] = {
            "metrics": metrics(
                [gold_label(r) for r in records], [r["score"] for r in rows], threshold
            ),
            "custom_span_metrics": span_metrics(records, rows),
            "paired_cases": len(paired),
            "mean_evidence_delta": float(np.mean([bad - good for good, bad in paired])),
            "correct_pair_fraction": float(
                np.mean([good < threshold <= bad for good, bad in paired])
            ),
            "runtime": runtime,
            "checkpoint_sha256": digest(folder / "best/model.safetensors"),
        }
        write_rows(ROOT / f"predictions/phase2b_resume/cf_training_fit_{method}.jsonl", rows)
        del model
    write_json(
        ROOT / "reports/phase2b_resume/cf_training_fit.json",
        {
            "role": "IN-SAMPLE synthetic training fit, not held-out quality; no tuning",
            "n": len(records),
            "parent_sources": len({r.source_id for r in records}),
            "methods": reports,
            "provenance": before,
        },
    )
    if provenance() != before:
        raise ValueError("source changed during fit diagnostic")
    print(
        {
            name: {
                "macro_f1": r["metrics"]["macro_f1"],
                "auroc": r["metrics"]["auroc"],
                "pair_delta": r["mean_evidence_delta"],
            }
            for name, r in reports.items()
        }
    )


if __name__ == "__main__":
    main()
