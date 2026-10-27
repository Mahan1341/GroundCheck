"""R7 frozen graph pilot. No training and no outer partition loading."""

import argparse
import time
from dataclasses import asdict, is_dataclass, replace

from grounded_hallucination_lab.data.ragtruth import Annotation, Record, gold_label
from grounded_hallucination_lab.research.evidence_graph import (
    extract,
    graph_override,
    keyed,
    retrieval_text,
)
from grounded_hallucination_lab.research.graph_suite import oracle_graphs, suite
from grounded_hallucination_lab.research.metrics import metrics, span_metrics
from grounded_hallucination_lab.research.protocol import (
    ROOT,
    digest,
    preserve_code,
    read_json,
    read_rows,
    write_json,
    write_rows,
)
from grounded_hallucination_lab.research.relations import content_hash

BASE = ROOT / "models/phase2b_r7"
PRED = ROOT / "predictions/phase2b_r7"
REPORT = ROOT / "reports/phase2b_r7"
MODEL = ROOT / "models/phase2b_r6/R6Bs42/best"
MODEL_SHA = "86929f0f77031c7d79ca64f45cb0746c04555f712724875c63951909de48cf0e"
THRESHOLD = 0.37612172961235046
DATASETS = ("natural", "controls")
METHODS = ("baseline", "keyed_auto", "graph_auto", "keyed_oracle", "graph_oracle")


def data(name):
    if name == "controls":
        return suite()
    if name != "natural":
        raise ValueError("only frozen inner data allowed")
    path = ROOT / "predictions/phase2b_r6/natural_records.jsonl"
    manifest = read_json(ROOT / "models/phase2b_r6/R6_artifact_manifest.json")
    assert digest(path) == manifest["files"][path.relative_to(ROOT).as_posix()]["sha256"]
    rows = read_rows(path)
    return [
        Record(**{**r, "annotations": tuple(Annotation(**a) for a in r["annotations"])})
        for r in rows
    ], []


def state():
    return {"source": preserve_code(), "plan_sha256": digest(ROOT / "EXPERIMENT_R7_PLAN.md")}


def checked():
    frozen = read_json(BASE / "freeze.json")
    assert frozen["state"] == state(), "source/plan changed; never bypass hashes"
    assert digest(MODEL / "model.safetensors") == MODEL_SHA
    return frozen


def freeze():
    previous = read_json(ROOT / "models/phase2b_r6/R6_artifact_manifest.json")
    for path, spec in previous["files"].items():
        assert digest(ROOT / path) == spec["sha256"], path
    assert digest(MODEL / "model.safetensors") == MODEL_SHA
    train_sources = set(read_json(MODEL.parent / "training_config.json")["train_sources"])
    frozen = {
        "state": state(),
        "old_R6_verified": len(previous["files"]),
        "checkpoint_sha256": MODEL_SHA,
        "datasets": {},
    }
    for name in DATASETS:
        records, design = data(name)
        assert not train_sources & {r.source_id for r in records}
        assert all(r.split == "train" and r.labels_available for r in records)
        frozen["datasets"][name] = {
            "sha256": content_hash(records, design),
            "n": len(records),
            "positive": sum(gold_label(r) for r in records),
            "sources": sorted({r.source_id for r in records}),
        }
        write_rows(PRED / f"{name}_records.jsonl", [asdict(r) for r in records])
        write_json(PRED / f"{name}_design.json", design)
    write_json(BASE / "freeze.json", frozen)
    print("frozen", {k: v["n"] for k, v in frozen["datasets"].items()}, flush=True)


def serial(value):
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, dict):
        return {k: serial(v) for k, v in value.items()}
    if isinstance(value, tuple | list):
        return [serial(v) for v in value]
    return value


def evaluate(method, dataset):
    from grounded_hallucination_lab.research.span_model import encode_records, load_model
    from grounded_hallucination_lab.research.stable_inference import score
    from grounded_hallucination_lab.research.verifiers import memory_report

    if method not in METHODS or dataset not in DATASETS:
        raise ValueError("unknown experiment")
    oracle = method.endswith("_oracle")
    if oracle and dataset != "controls":
        raise ValueError("oracle only for manually authored controls")
    frozen = checked()
    records, design = data(dataset)
    assert content_hash(records, design) == frozen["datasets"][dataset]["sha256"]
    model, tokenizer = load_model(MODEL)
    # Each method really runs its complete pipeline: baseline is not a free cached oracle.
    started = time.perf_counter()
    graphs = []
    if method != "baseline":
        for i, r in enumerate(records):
            graphs.append(
                oracle_graphs(design[i])
                if oracle
                else (extract(r.evidence, r.source_id), extract(r.response, "answer"))
            )
    extraction_seconds = time.perf_counter() - started
    examples, metadata = encode_records(records, tokenizer, 2048, False)
    raw, raw_runtime = score(model, tokenizer, examples, metadata)
    rows = raw
    diagnostics = []
    selected_runtime = None
    if method.startswith("keyed"):
        selected = []
        indices = []
        intervals = []
        for i, (r, (e, c)) in enumerate(zip(records, graphs, strict=True)):
            facts = []
            spans = []
            for claim in c.facts:
                candidates = keyed(claim, e, c)
                if candidates:
                    facts.extend(candidates)
                    spans.append((claim.provenance.start, claim.provenance.end))
            text, refs = retrieval_text(r.evidence, facts, e) if facts else ("", [])
            diagnostics.append(
                {
                    "id": r.id,
                    "selected_references": serial(refs),
                    "claim_intervals": spans,
                    "fallback": not bool(facts),
                }
            )
            if facts:
                selected.append(replace(r, evidence=text))
                indices.append(i)
                intervals.append(spans)
        rows = [{**row, "token_probabilities": list(row["token_probabilities"])} for row in raw]
        if selected:
            enc, meta = encode_records(selected, tokenizer, 2048, False)
            predictions, selected_runtime = score(model, tokenizer, enc, meta)
            for i, spans, pred in zip(indices, intervals, predictions, strict=True):
                assert pred["offsets"] == rows[i]["offsets"]
                for j, (a, b) in enumerate(pred["offsets"]):
                    if any(a < end and b > start for start, end in spans):
                        rows[i]["token_probabilities"][j] = pred["token_probabilities"][j]
                rows[i]["score"] = max(rows[i]["token_probabilities"], default=0)
    elif method.startswith("graph"):
        rows = []
        for row, (e, c) in zip(raw, graphs, strict=True):
            changed, details = graph_override(row, c, e)
            rows.append(changed)
            diagnostics.append({"id": row["id"], "verdicts": serial(details)})
    seconds = time.perf_counter() - started
    assert [r["id"] for r in rows] == [r.id for r in records]
    runtime = {
        "end_to_end_seconds": seconds,
        "end_to_end_ms_per_response": seconds / len(records) * 1000,
        "responses_per_second": len(records) / seconds,
        "extraction_seconds": extraction_seconds,
        "scope": (
            "resident model; extraction, encoding, all neural passes and fusion; "
            "excludes load/disk/analysis"
        ),
        "raw_neural": raw_runtime,
        "selected_neural": selected_runtime,
        **memory_report(),
    }
    runtime["cuda_peak_allocated_bytes"] = max(
        raw_runtime["cuda_peak_allocated_bytes"],
        (selected_runtime or raw_runtime)["cuda_peak_allocated_bytes"],
    )
    write_rows(PRED / f"{method}_{dataset}.jsonl", rows)
    write_rows(PRED / f"{method}_{dataset}_diagnostics.jsonl", diagnostics)
    if graphs:
        write_rows(
            PRED / f"{method}_{dataset}_graphs.jsonl",
            [
                {"id": r.id, "evidence": asdict(e), "claims": asdict(c)}
                for r, (e, c) in zip(records, graphs, strict=True)
            ],
        )
    report = {
        "method": method,
        "dataset": dataset,
        "checkpoint_sha256": MODEL_SHA,
        "metrics": metrics([gold_label(r) for r in records], [r["score"] for r in rows], THRESHOLD),
        "localization": span_metrics(records, rows),
        "runtime": runtime,
        "oracle": oracle,
    }
    write_json(REPORT / f"{method}_{dataset}.json", report)
    assert checked() == frozen
    print(method, dataset, report["metrics"]["macro_f1"], flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("freeze", "evaluate"))
    parser.add_argument("--method", choices=METHODS, default="baseline")
    parser.add_argument("--dataset", choices=DATASETS, default="controls")
    args = parser.parse_args()
    if args.command == "freeze":
        freeze()
    else:
        evaluate(args.method, args.dataset)
