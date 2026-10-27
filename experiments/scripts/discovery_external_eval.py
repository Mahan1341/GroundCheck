"""Frozen external response evaluation; no gold triplets or spans used as inputs."""

import argparse
import time

from discovery_verifier import prepare, project, read_cache, run

from grounded_hallucination_lab.research.metrics import bootstrap, metrics
from grounded_hallucination_lab.research.protocol import (
    ROOT,
    InferenceInput,
    digest,
    read_json,
    read_rows,
    write_json,
    write_rows,
)

BASE = ROOT / "models/discovery_loop_20261026"
PRED = ROOT / "predictions/discovery_loop_20261026"
REPORT = ROOT / "reports/discovery_loop_20261026"


def data(partition):
    if partition not in ("development", "confirmation"):
        raise ValueError("external partitions only")
    if partition == "confirmation":
        freeze = read_json(BASE / "external_final_freeze.json")
        assert freeze["selection_complete"] and freeze["no_adaptation_after_confirmation"]
    filename = (
        "development_eligible_records.jsonl"
        if partition == "development"
        else "confirmation_records.jsonl"
    )
    path = BASE / "external" / filename
    return read_rows(path), path


def metadata(rr, tok):
    from grounded_hallucination_lab.research.span_model import windows

    examples, meta = [], []
    for r in rr:
        view = InferenceInput(
            **{k: r[k] for k in ("id", "source_id", "question", "evidence", "response")}
        )
        ex, m = windows(view, tok, max_length=2048)
        assert m["answer_coverage"] == m["evidence_coverage"] == 1
        examples.extend(ex)
        meta.append(m)
    return examples, meta


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["prepare", "b7", "factcg", "project", "analyze"])
    parser.add_argument(
        "--partition", choices=["development", "confirmation"], default="development"
    )
    args = parser.parse_args()
    rr, path = data(args.partition)
    tag = "external_" + args.partition
    if args.command == "prepare":
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(
            ROOT / "models/phase2b/span_inner_s42/best", local_files_only=True
        )
        _, meta = metadata(rr, tok)
        views = [{k: r[k] for k in ("id", "source_id", "response", "evidence")} for r in rr]
        pairs, layout = prepare(views, meta)
        write_json(BASE / f"{tag}_pairs.json", pairs)
        write_json(BASE / f"{tag}_layout.json", layout)
        write_json(
            BASE / f"{tag}_inputs.json",
            {
                "data_sha256": digest(path),
                "pairs": len(pairs),
                "n": len(rr),
                "pairs_sha256": digest(BASE / f"{tag}_pairs.json"),
                "code_sha256": digest(ROOT / "scripts/discovery_external_eval.py"),
            },
        )
    elif args.command == "b7":
        from grounded_hallucination_lab.research.pair_training import INITIAL, INITIAL_SHA
        from grounded_hallucination_lab.research.span_model import load_model
        from grounded_hallucination_lab.research.stable_inference import score

        assert digest(INITIAL / "model.safetensors") == INITIAL_SHA
        start = time.perf_counter()
        model, tok = load_model(INITIAL, 42)
        examples, meta = metadata(rr, tok)
        rows, resources = score(model, tok, examples, meta)
        write_rows(PRED / f"{tag}_B7.jsonl", rows)
        write_json(
            REPORT / f"{tag}_B7_resources.json",
            {
                **resources,
                "wall_seconds_with_loading": time.perf_counter() - start,
                "gpu_minutes_conservative": (time.perf_counter() - start) / 60,
                "checkpoint_sha256": INITIAL_SHA,
            },
        )
    elif args.command == "factcg":
        run(tag, BASE / f"{tag}_pairs.json", minutes=10)
    elif args.command == "project":
        rows = project(
            read_json(BASE / f"{tag}_layout.json"), read_cache(PRED / "factcg_pair_cache.jsonl")
        )
        write_rows(PRED / f"{tag}_FactCG.jsonl", rows)
    else:
        systems = {m: read_rows(PRED / f"{tag}_{m}.jsonl") for m in ("B7", "FactCG")}
        thresholds = {"B7": 0.37612172961235046, "FactCG": 0.5}
        result = {}
        for subset in ("all", "closed_qa", "information_extraction", "summarization"):
            indices = [i for i, r in enumerate(rr) if subset == "all" or r["category"] == subset]
            if not indices:
                continue
            labels = [rr[i]["label"] for i in indices]
            groups = [rr[i]["group_id"] for i in indices]
            scores = {}
            for m, rows in systems.items():
                assert [r["id"] for r in rr] == [p["id"] for p in rows]
                scores[m] = [rows[i]["score"] for i in indices]
            result[subset] = {
                "n": len(indices),
                "sources": len(set(groups)),
                "metrics": {m: metrics(labels, s, thresholds[m]) for m, s in scores.items()},
                "bootstrap": bootstrap(
                    labels,
                    groups,
                    {m: ([int(p >= thresholds[m]) for p in s], s) for m, s in scores.items()},
                    "B7",
                ),
            }
        write_json(REPORT / f"{tag}_analysis.json", result)
        print(
            {k: {m: v["macro_f1"] for m, v in r["metrics"].items()} for k, r in result.items()},
            flush=True,
        )


if __name__ == "__main__":
    main()
