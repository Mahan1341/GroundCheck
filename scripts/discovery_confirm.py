"""Freeze then explicitly unlock the external confirmation; never RAGTruth outer data."""

import argparse

from discovery_external_data import export_partition

from grounded_hallucination_lab.research.protocol import ROOT, digest, write_json

BASE = ROOT / "models/discovery_loop_20261026"
FREEZE = BASE / "external_final_freeze.json"


def artifacts(paths):
    return [{"path": p.relative_to(ROOT).as_posix(), "sha256": digest(p)} for p in paths]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["freeze", "unlock"])
    args = parser.parse_args()
    if args.command == "unlock":
        export_partition("locked_confirmation", FREEZE, digest(FREEZE))
        return
    b7 = sorted((ROOT / "models/phase2b/span_inner_s42/best").glob("*"))
    factcg = [p for p in sorted((BASE / "factcg").glob("*")) if p.is_file()]
    source = [
        ROOT / name
        for name in (
            "scripts/discovery_external_eval.py",
            "scripts/discovery_whole_external.py",
            "scripts/discovery_verifier.py",
            "scripts/discovery_whole.py",
            "scripts/discovery_external_data.py",
            "scripts/discovery_confirm.py",
            "src/grounded_hallucination_lab/research/span_model.py",
            "src/grounded_hallucination_lab/research/stable_inference.py",
            "src/grounded_hallucination_lab/research/segments.py",
            "src/grounded_hallucination_lab/research/verifiers.py",
            "docs/DISCOVERY_CONFIRMATION_PREREG.md",
        )
    ]
    write_json(
        FREEZE,
        {
            "stage": "external_confirmation",
            "selection_complete": True,
            "no_adaptation_after_confirmation": True,
            "external_split_sha256": digest(BASE / "external/frozen_split.json"),
            "external_eligibility_sha256": digest(BASE / "external/frozen_eligibility.json"),
            "primary": "FactCG segmented minus B7, all eligible response macro-F1",
            "secondary": "Whole ablation; categories reported separately; no multiplicity correction",
            "metrics": [
                "accuracy",
                "macro_f1",
                "positive_precision_recall_f1",
                "auroc",
                "average_precision",
                "confusion",
                "fpr",
                "fnr",
                "source_bootstrap2000",
            ],
            "candidates": [
                {
                    "name": "B7",
                    "threshold": 0.37612172961235046,
                    "artifacts": artifacts([p for p in b7 if p.is_file()] + source),
                },
                {"name": "FactCG", "threshold": 0.5, "artifacts": artifacts(factcg + source)},
                {"name": "Whole", "threshold": 0.5, "artifacts": artifacts(factcg + source)},
            ],
        },
    )
    print({"freeze_sha256": digest(FREEZE), "locked_labels_still_closed": True}, flush=True)


if __name__ == "__main__":
    main()
