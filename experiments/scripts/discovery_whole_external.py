"""Same whole-answer ablation on external data; independent confirmation remains gated."""

import argparse

from discovery_external_eval import data, metadata
from discovery_verifier import BASE, PRED, project, read_cache, run
from discovery_whole import whole_prepare

from grounded_hallucination_lab.research.metrics import bootstrap, metrics
from grounded_hallucination_lab.research.protocol import (
    ROOT,
    digest,
    read_json,
    read_rows,
    write_json,
    write_rows,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["prepare", "run", "project", "analyze"])
    parser.add_argument(
        "--partition", choices=["development", "confirmation"], default="development"
    )
    args = parser.parse_args()
    rr, path = data(args.partition)
    tag = "external_" + args.partition
    name = tag + "_whole"
    if args.mode == "prepare":
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(
            ROOT / "models/phase2b/span_inner_s42/best", local_files_only=True
        )
        _, old = metadata(rr, tok)
        views = [{k: r[k] for k in ("id", "source_id", "response", "evidence")} for r in rr]
        pairs, layouts = whole_prepare(views, old)
        write_json(BASE / f"{name}_pairs.json", pairs)
        write_json(BASE / f"{name}_layout.json", layouts)
        write_json(
            BASE / f"{name}_freeze.json",
            {
                "data_sha256": digest(path),
                "pairs_sha256": digest(BASE / f"{name}_pairs.json"),
                "source_sha256": digest(ROOT / "scripts/discovery_whole_external.py"),
                "prereg_sha256": digest(ROOT / "docs/DISCOVERY_CYCLE4_PREREG.md"),
            },
        )
    elif args.mode == "run":
        run(name, BASE / f"{name}_pairs.json", minutes=5)
    elif args.mode == "project":
        result = project(
            read_json(BASE / f"{name}_layout.json"), read_cache(PRED / "factcg_pair_cache.jsonl")
        )
        write_rows(PRED / f"{tag}_Whole.jsonl", result)
    else:
        thresholds = {"B7": 0.37612172961235046, "FactCG": 0.5, "Whole": 0.5}
        systems = {m: read_rows(PRED / f"{tag}_{m}.jsonl") for m in thresholds}
        result = {}
        for subset in ("all", "closed_qa", "information_extraction", "summarization"):
            indices = [i for i, r in enumerate(rr) if subset == "all" or r["category"] == subset]
            if not indices:
                continue
            labels, groups = [rr[i]["label"] for i in indices], [rr[i]["group_id"] for i in indices]
            scores = {}
            for m, rows in systems.items():
                assert [r["id"] for r in rr] == [p["id"] for p in rows]
                scores[m] = [rows[i]["score"] for i in indices]
            boot_systems = {m: ([int(p >= thresholds[m]) for p in s], s) for m, s in scores.items()}
            result[subset] = {
                "n": len(indices),
                "sources": len(set(groups)),
                "metrics": {m: metrics(labels, s, thresholds[m]) for m, s in scores.items()},
                "bootstrap": bootstrap(labels, groups, boot_systems, "B7"),
                "bootstrap_FactCG_reference": bootstrap(labels, groups, boot_systems, "FactCG"),
            }
        write_json(ROOT / "reports/discovery_loop_20261026" / f"{tag}_three_analysis.json", result)
        print(
            {k: {m: v["macro_f1"] for m, v in r["metrics"].items()} for k, r in result.items()},
            flush=True,
        )


if __name__ == "__main__":
    main()
