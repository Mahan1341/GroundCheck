"""Read-only same-format probe on inner-validation parents, with frozen thresholds."""

import unicodedata
from collections import defaultdict
from dataclasses import asdict

import numpy as np

from grounded_hallucination_lab.data.ragtruth import gold_label
from grounded_hallucination_lab.research.counterfactual import make_pairs, require_unseen_sources
from grounded_hallucination_lab.research.metrics import bootstrap, metrics, span_metrics
from grounded_hallucination_lab.research.protocol import (
    OUT,
    PRED,
    ROOT,
    InferenceInput,
    digest,
    inner_split,
    load_parts,
    preserve_code,
    provenance,
    read_json,
    write_json,
    write_rows,
)


def normalized(text):
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def main():
    before = preserve_code()
    fit, val = inner_split(load_parts().train)
    manifest = read_json(ROOT / "configs/counterfactual_validation_acceptance.json")
    records, audit = make_pairs(val, manifest)
    require_unseen_sources(records, {r.source_id for r in fit})
    sources = {r.source_id: r for r in val}
    fit_evidence = {normalized(r.evidence) for r in fit}
    for parent in audit:
        text = sources[parent["source_id"]].evidence[parent["source_start"] : parent["source_end"]]
        if any(normalized(text) in document for document in fit_evidence):
            raise ValueError("probe parent text appears in normalized inner-fit evidence")
    labels = [gold_label(r) for r in records]
    reports, systems = {}, {}
    for method in ("B7", "B8", "C0", "C1", "B6"):
        if method == "B6":
            from grounded_hallucination_lab.research.verifiers import Verifier, score_records

            rows, runtime = score_records(
                Verifier("minicheck", 4), [InferenceInput.from_record(r) for r in records], "clean"
            )
            point = read_json(PRED / "B6_inner_val.meta.json")
            model_hash = runtime["checkpoint"]
        else:
            from grounded_hallucination_lab.research.span_model import (
                encode_records,
                load_model,
                score,
            )

            folder = (
                OUT / ("span_inner_s42" if method == "B7" else "answer_inner_s42")
                if method in ("B7", "B8")
                else (ROOT / f"models/phase2b_resume/{method}_inner_s42")
            )
            config = read_json(folder / "training_config.json")
            require_unseen_sources(records, set(config["train_sources"]))
            point = read_json(
                PRED / f"{method}s42_inner_val.meta.json"
                if method in ("B7", "B8")
                else (ROOT / f"predictions/phase2b_resume/{method}_inner_val.meta.json")
            )
            model, tokenizer = load_model(folder / "best")
            examples, metadata = encode_records(records, tokenizer, 2048, method == "B8")
            rows, runtime = score(model, tokenizer, examples, metadata, batch_size=1)
            model_hash = digest(folder / "best/model.safetensors")
            del model
        threshold = point["inner_selection"]["threshold"]
        groups = defaultdict(list)
        for record, row in zip(records, rows, strict=True):
            if record.id != row["id"]:
                raise ValueError("probe predictions changed row order")
            groups[(record.source_id, record.response)].append((gold_label(record), row["score"]))
        pairs = []
        for (source, _), group in groups.items():
            if sorted(label for label, _ in group) != [0, 1]:
                raise ValueError("each fixed answer must have one supported/unsupported premise")
            ordered = dict(group)
            pairs.append(
                {
                    "source_id": source,
                    "supported": ordered[0],
                    "unsupported": ordered[1],
                    "delta": ordered[1] - ordered[0],
                    "correct_pair": ordered[0] < threshold <= ordered[1],
                }
            )
        risks = [r["score"] for r in rows]
        systems[method] = ([int(s >= threshold) for s in risks], risks)
        reports[method] = {
            "metrics": metrics(labels, risks, threshold),
            "custom_span_metrics": span_metrics(records, rows) if "offsets" in rows[0] else None,
            "pairs": pairs,
            "mean_evidence_delta": float(np.mean([p["delta"] for p in pairs])),
            "correct_pair_fraction": float(np.mean([p["correct_pair"] for p in pairs])),
            "runtime": runtime,
            "checkpoint_sha256_or_revision": model_hash,
        }
        write_rows(ROOT / f"predictions/phase2b_resume/cf_unseen_{method}.jsonl", rows)
    write_json(
        ROOT / "predictions/phase2b_resume/cf_unseen_design.json", [asdict(r) for r in records]
    )
    write_json(
        ROOT / "reports/phase2b_resume/cf_unseen.json",
        {
            "role": "post-diagnostic internal development probe; not confirmatory; no tuning",
            "n": len(records),
            "source_count": len({r.source_id for r in records}),
            "methods": reports,
            "provenance": before,
            "parent_audit": audit,
            "paired_bootstrap": bootstrap(
                labels, [r.source_id for r in records], systems, "C0", 2000
            ),
            "caveat": "few parents, one transformation family; CIs do not imply broad transfer",
        },
    )
    if provenance() != before:
        raise ValueError("source changed during probe")
    print(
        {
            m: {
                "macro_f1": r["metrics"]["macro_f1"],
                "auroc": r["metrics"]["auroc"],
                "delta": r["mean_evidence_delta"],
            }
            for m, r in reports.items()
        }
    )


if __name__ == "__main__":
    main()
