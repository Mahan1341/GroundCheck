"""R8 train-side natural extraction pilot. No training, oracle graphs or outer loading."""

import argparse
import re
import time
from dataclasses import asdict, replace

from research_r7 import MODEL, MODEL_SHA, THRESHOLD
from research_r7 import data as inner_records

from grounded_hallucination_lab.data.ragtruth import gold_label
from grounded_hallucination_lab.research.evidence_graph import extract_natural, natural_override
from grounded_hallucination_lab.research.metrics import metrics, span_metrics
from grounded_hallucination_lab.research.protocol import (
    ROOT,
    digest,
    preserve_code,
    read_json,
    write_json,
    write_rows,
)
from grounded_hallucination_lab.research.relations import content_hash

BASE = ROOT / "models/phase2b_r8"
PRED = ROOT / "predictions/phase2b_r8"
REPORT = ROOT / "reports/phase2b_r8"
METHODS = ("baseline", "sentence", "flat", "graph")
PARTITIONS = ("development", "evaluation")


def data(partition):
    if partition not in PARTITIONS:
        raise ValueError("only R8 train-side partitions allowed")
    split = read_json(BASE / "source_split.json")
    path = ROOT / "predictions/phase2b_r6/natural_records.jsonl"
    assert digest(path) == split["input_sha256"]
    records, _ = inner_records("natural")
    return [r for r in records if r.source_id in split["sources"][partition]]


def state():
    spec = read_json(ROOT / "configs/r8-nlp.json")
    return {
        "source": preserve_code(),
        "plan": digest(ROOT / "EXPERIMENT_R8_PLAN.md"),
        "split": digest(BASE / "source_split.json"),
        "nlp_config": digest(ROOT / "configs/r8-nlp.json"),
        "nlp_files": {n: digest(ROOT / n) for n in spec["files"]},
    }


def checked():
    frozen = read_json(BASE / "freeze.json")
    assert state() == frozen["state"], "R8 source/resource change: never bypass freeze"
    assert digest(MODEL / "model.safetensors") == MODEL_SHA
    return frozen


def freeze():
    previous = read_json(ROOT / "models/phase2b_r7/R7_artifact_manifest.json")
    for name, spec in previous["files"].items():
        assert digest(ROOT / name) == spec["sha256"], name
    train = set(read_json(MODEL.parent / "training_config.json")["train_sources"])
    result = {"state": state(), "checkpoint": MODEL_SHA, "old_verified": len(previous["files"])}
    sources = []
    for partition in PARTITIONS:
        records = data(partition)
        ss = {r.source_id for r in records}
        sources.append(ss)
        assert not ss & train
        assert all(r.split == "train" and r.labels_available for r in records)
        result[partition] = {
            "n": len(records),
            "sources": sorted(ss),
            "positive": sum(gold_label(r) for r in records),
            "hash": content_hash(records, []),
        }
        write_rows(PRED / f"{partition}_records.jsonl", [asdict(r) for r in records])
    assert not sources[0] & sources[1]
    write_json(BASE / "freeze.json", result)
    print({p: result[p]["n"] for p in PARTITIONS}, flush=True)


STOP = set("a an the is are was were of in on to and or for as with by it this that".split())


def terms(text):
    return set(re.findall(r"[a-z0-9]+", text.casefold())) - STOP


def sentence_selection(record, evidence, claims):
    """Fixed top2 overlap sentences + same-passage neighbors, no learned statistics."""
    selected = set()
    intervals = []
    for fact in claims.facts:
        query = terms(record.response[fact.provenance.start : fact.provenance.end])
        ranked = []
        for j, prov in enumerate(evidence.clauses):
            words = terms(record.evidence[prov.start : prov.end])
            value = len(query & words) / max(1, len(query | words))
            ranked.append((-value, j))
        hits = [j for score, j in sorted(ranked)[:2] if score < 0]
        if hits:
            intervals.append((fact.provenance.start, fact.provenance.end))
        for j in hits:
            selected.add(j)
            for neighbor in (j - 1, j + 1):
                if 0 <= neighbor < len(evidence.clauses):
                    if evidence.clauses[neighbor].passage_id == evidence.clauses[j].passage_id:
                        selected.add(neighbor)
    refs = [evidence.clauses[j] for j in sorted(selected)]
    return "\n\n".join(record.evidence[p.start : p.end] for p in refs), refs, intervals


def evaluate(method, partition):
    from grounded_hallucination_lab.research.span_model import encode_records, load_model
    from grounded_hallucination_lab.research.stable_inference import score
    from grounded_hallucination_lab.research.verifiers import memory_report

    frozen = checked()
    records = data(partition)
    assert content_hash(records, []) == frozen[partition]["hash"]
    model, tokenizer = load_model(MODEL)
    started = time.perf_counter()
    graphs = []
    if method != "baseline":
        graphs = [
            (extract_natural(r.evidence, r.source_id), extract_natural(r.response, "answer"))
            for r in records
        ]
    extraction_seconds = time.perf_counter() - started
    enc, meta = encode_records(records, tokenizer, 2048, False)
    assert all(m["answer_coverage"] == m["evidence_coverage"] == 1 for m in meta)
    raw, raw_time = score(model, tokenizer, enc, meta)
    rows = raw
    diagnostics = []
    selected_time = None
    if method == "sentence":
        selected, indices, spans = [], [], []
        rows = [{**r, "token_probabilities": list(r["token_probabilities"])} for r in raw]
        for i, (record, (e, c)) in enumerate(zip(records, graphs, strict=True)):
            text, refs, intervals = sentence_selection(record, e, c)
            diagnostics.append(
                {"id": record.id, "references": [asdict(p) for p in refs], "intervals": intervals}
            )
            if refs:
                selected.append(replace(record, evidence=text))
                indices.append(i)
                spans.append(intervals)
        if selected:
            enc, meta2 = encode_records(selected, tokenizer, 2048, False)
            assert all(m["answer_coverage"] == m["evidence_coverage"] == 1 for m in meta2)
            predictions, selected_time = score(model, tokenizer, enc, meta2)
            for i, intervals, pred in zip(indices, spans, predictions, strict=True):
                assert pred["offsets"] == rows[i]["offsets"]
                for j, (a, b) in enumerate(pred["offsets"]):
                    if any(a < end and b > start for start, end in intervals):
                        rows[i]["token_probabilities"][j] = pred["token_probabilities"][j]
                rows[i]["score"] = max(rows[i]["token_probabilities"], default=0)
    elif method in ("flat", "graph"):
        rows = []
        for row, (e, c) in zip(raw, graphs, strict=True):
            pred, verdicts = natural_override(row, c, e, links=method == "graph")
            rows.append(pred)
            diagnostics.append(
                {
                    "id": row["id"],
                    "verdicts": [
                        {
                            **v,
                            "claim": asdict(v["claim"]),
                            "references": [asdict(f) for f in v["references"]],
                        }
                        for v in verdicts
                    ],
                }
            )
    seconds = time.perf_counter() - started
    runtime = {
        "seconds": seconds,
        "ms_per_response": seconds / len(records) * 1000,
        "responses_per_second": len(records) / seconds,
        "extraction_seconds": extraction_seconds,
        "raw_neural": raw_time,
        "selected_neural": selected_time,
        **memory_report(),
    }
    runtime["cuda_peak_allocated_bytes"] = max(
        raw_time["cuda_peak_allocated_bytes"],
        (selected_time or raw_time)["cuda_peak_allocated_bytes"],
    )
    write_rows(PRED / f"{method}_{partition}.jsonl", rows)
    write_rows(PRED / f"{method}_{partition}_diagnostics.jsonl", diagnostics)
    if graphs:
        write_rows(
            PRED / f"{method}_{partition}_graphs.jsonl",
            [
                {"id": r.id, "evidence": asdict(e), "claims": asdict(c)}
                for r, (e, c) in zip(records, graphs, strict=True)
            ],
        )
    report = {
        "method": method,
        "partition": partition,
        "checkpoint": MODEL_SHA,
        "metrics": metrics([gold_label(r) for r in records], [p["score"] for p in rows], THRESHOLD),
        "localization": span_metrics(records, rows),
        "runtime": runtime,
        "independently_reviewed_graph_gold": False,
    }
    write_json(REPORT / f"{method}_{partition}.json", report)
    assert checked() == frozen
    print(method, partition, report["metrics"]["macro_f1"], flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("freeze", "evaluate"))
    parser.add_argument("--method", choices=METHODS, default="baseline")
    parser.add_argument("--partition", choices=PARTITIONS, default="evaluation")
    args = parser.parse_args()
    if args.command == "freeze":
        freeze()
    else:
        evaluate(args.method, args.partition)
