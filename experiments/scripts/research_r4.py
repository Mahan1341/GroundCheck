"""Frozen R4 inference packet; no training or outer-evaluation entry point."""

import argparse
import re
import time
from dataclasses import asdict, replace

from grounded_hallucination_lab.data.ragtruth import gold_label
from grounded_hallucination_lab.research.invariance import suite as old_suite
from grounded_hallucination_lab.research.metrics import metrics, span_metrics
from grounded_hallucination_lab.research.numeric_local import (
    local_claims,
    normalize_numbers,
    project_probabilities,
)
from grounded_hallucination_lab.research.numeric_local_suite import suite
from grounded_hallucination_lab.research.protocol import (
    ROOT,
    digest,
    load_parts,
    partition,
    preserve_code,
    read_json,
    read_rows,
    write_json,
    write_rows,
)
from grounded_hallucination_lab.research.relations import content_hash

BASE = ROOT / "models/phase2b_r4"
PRED = ROOT / "predictions/phase2b_r4"
REPORT = ROOT / "reports/phase2b_r4"
THRESHOLD = 0.37612172961235046
MODELS = {
    "B7": (
        "models/phase2b/span_inner_s42",
        "7c6fb6f5f842863175f6422a84acd9c85d5056c3144ec73738b3e3a83464b4f4",
    ),
    "R2CE": (
        "models/phase2b_r2/R2C0s42",
        "7fe60c82e32af79c3c31cd845736a656ca3aa0a003a23e477f125f084fd594a6",
    ),
    "R3A": (
        "models/phase2b_r3/R3As42",
        "a0116cda0d54be0a97fa70940b59c2ef1c734229c7fdea856dd52c64078e1d26",
    ),
}
DATASETS = ("old", "fresh", "natural")
MODES = ("raw", "evidence", "symmetric", "local")


def data(name):
    if name == "old":
        records, design = old_suite("diagnostic")
        expected = read_json(ROOT / "models/phase2b_r3/suite_manifest.json")
        assert content_hash(records, design) == expected["splits"]["diagnostic"]["content_sha256"]
        return records, design
    if name == "fresh":
        return suite()
    if name == "natural":
        return partition(load_parts(), "inner_val"), []
    raise ValueError("only fixed inner datasets are allowed")


def state():
    return {"source": preserve_code(), "plan_sha256": digest(ROOT / "EXPERIMENT_R4_PLAN.md")}


def freeze():
    manifest = {"state": state(), "datasets": {}, "models": {}, "threshold": THRESHOLD}
    old = read_json(ROOT / "models/phase2b_r3/R3_artifact_manifest.json")
    for name, spec in old["files"].items():
        assert digest(ROOT / name) == spec["sha256"], name
    manifest["R3_verified_files"] = len(old["files"])
    train_sources = set()
    for method, (folder, expected) in MODELS.items():
        path = ROOT / folder
        assert digest(path / "best/model.safetensors") == expected
        cfg = read_json(path / "training_config.json")
        train_sources.update(cfg["train_sources"])
        manifest["models"][method] = {
            "checkpoint_sha256": expected,
            "config_sha256": digest(path / "training_config.json"),
        }
    for name in DATASETS:
        records, design = data(name)
        assert not train_sources & {r.source_id for r in records}
        manifest["datasets"][name] = {
            "sha256": content_hash(records, design),
            "n": len(records),
            "positive": sum(gold_label(r) for r in records),
            "sources": sorted({r.source_id for r in records}),
        }
        write_rows(PRED / f"{name}_records.jsonl", [asdict(r) for r in records])
        write_json(PRED / f"{name}_design.json", design)
    write_json(BASE / "freeze.json", manifest)
    print(
        {
            "frozen": {k: (v["n"], len(v["sources"])) for k, v in manifest["datasets"].items()},
            "old_artifacts_verified": len(old["files"]),
        },
        flush=True,
    )


def run(method):
    from grounded_hallucination_lab.research.span_model import encode_records, load_model, score

    frozen = read_json(BASE / "freeze.json")
    current = state()
    assert current == frozen["state"], "source/plan changed after freeze"
    folder, expected = MODELS[method]
    assert digest(ROOT / folder / "best/model.safetensors") == expected
    model, tokenizer = load_model(ROOT / folder / "best")
    for dataset in DATASETS:
        records, design = data(dataset)
        assert content_hash(records, design) == frozen["datasets"][dataset]["sha256"]
        _, original_meta = encode_records(records, tokenizer, 2048, False)
        original_meta = {m["id"]: m for m in original_meta}
        for mode in MODES:
            tag = f"{method}_{dataset}_{mode}"
            report_path = REPORT / f"{tag}.json"
            if report_path.exists():
                report = read_json(report_path)
                assert report["state"] == current
                assert report["predictions_sha256"] == digest(PRED / f"{tag}.jsonl")
                print(f"verified completed {tag}", flush=True)
                continue
            start = time.perf_counter()
            views, mapping, audit, fallback_rows = [], [], [], {}
            raw = (
                {r["id"]: r for r in read_rows(PRED / f"{method}_{dataset}_raw.jsonl")}
                if mode == "local"
                else {}
            )
            for r in records:
                if mode == "local":
                    claims, fallback = local_claims(r.response)
                    audit.append({"id": r.id, "views": claims, "fallback": fallback})
                    if fallback:
                        fallback_rows[r.id] = raw[r.id]
                        continue
                    for i, claim in enumerate(claims):
                        views.append(
                            replace(
                                r,
                                id=f"{r.id}::local{i}",
                                annotations=(),
                                response=r.response[claim["start"] : claim["end"]],
                            )
                        )
                        mapping.append((r, claim))
                else:
                    evidence = normalize_numbers(r.evidence) if mode != "raw" else None
                    answer = normalize_numbers(r.response) if mode == "symmetric" else None
                    views.append(
                        replace(
                            r,
                            annotations=(),
                            response=answer.text if answer else r.response,
                            evidence=evidence.text if evidence else r.evidence,
                        )
                    )
                    mapping.append((r, answer))
                    audit.append(
                        {
                            "id": r.id,
                            "answer_edits": answer.edits if answer else [],
                            "evidence_edits": evidence.edits if evidence else [],
                        }
                    )
            encoded, meta = encode_records(views, tokenizer, 2048, False)
            if views:
                native, runtime = score(model, tokenizer, encoded, meta, batch_size=2)
            else:
                native, runtime = [], {"pipeline_seconds": 0, "responses": 0}
            rows = {
                r.id: {
                    **original_meta[r.id],
                    "token_probabilities": [0.0] * original_meta[r.id]["answer_tokens"],
                }
                for r in records
            }
            for row, (record, transform) in zip(native, mapping, strict=True):
                offsets = original_meta[record.id]["offsets"]
                if mode == "local":
                    translated = [
                        (a + transform["start"], b + transform["start"]) for a, b in row["offsets"]
                    ]
                    central = [
                        (a, b, p)
                        for (a, b), p in zip(translated, row["token_probabilities"], strict=True)
                        if a < transform["central_end"] and b > transform["central_start"]
                    ]
                    values = [
                        max(
                            (
                                p
                                for a, b, p in central
                                if a < end
                                and b > start
                                and start < transform["central_end"]
                                and end > transform["central_start"]
                            ),
                            default=0.0,
                        )
                        for start, end in offsets
                    ]
                    rows[record.id]["token_probabilities"] = [
                        max(a, b)
                        for a, b in zip(rows[record.id]["token_probabilities"], values, strict=True)
                    ]
                else:
                    values = (
                        project_probabilities(
                            offsets, transform, row["offsets"], row["token_probabilities"]
                        )
                        if transform
                        else row["token_probabilities"]
                    )
                    rows[record.id].update(token_probabilities=values, score=row["score"])
            for identifier, row in fallback_rows.items():
                rows[identifier] = row
            for row in rows.values():
                if mode == "local":
                    row["score"] = max(row["token_probabilities"], default=0.0)
            result = [rows[r.id] for r in records]
            runtime.update(
                total_seconds_including_encode_projection=time.perf_counter() - start,
                original_responses=len(records),
                inference_views=len(views),
                full_fallback_responses=len(fallback_rows),
                max_input_tokens=max((m["max_window_tokens"] for m in meta), default=0),
                truncated_tokens=sum(m["truncated_tokens"] for m in meta),
            )
            runtime["original_responses_per_second_end_to_end"] = (
                len(records) / runtime["total_seconds_including_encode_projection"]
            )
            write_rows(PRED / f"{tag}.jsonl", result)
            write_rows(PRED / f"{tag}_views.jsonl", audit)
            write_rows(PRED / f"{tag}_native.jsonl", native)
            write_json(
                report_path,
                {
                    "state": current,
                    "method": method,
                    "dataset": dataset,
                    "mode": mode,
                    "checkpoint_sha256": expected,
                    "predictions_sha256": digest(PRED / f"{tag}.jsonl"),
                    "runtime": runtime,
                    "metrics": metrics(
                        [gold_label(r) for r in records], [r["score"] for r in result], THRESHOLD
                    ),
                    "localization": span_metrics(records, result),
                },
            )
            assert state() == current, "source changed during inference"
            print(
                f"completed {tag}: {runtime['total_seconds_including_encode_projection']:.2f}s",
                flush=True,
            )
    assert digest(ROOT / folder / "best/model.safetensors") == expected


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["freeze", *MODELS])
    parser.add_argument("--output-name", default="phase2b_r4")
    args = parser.parse_args()
    if not re.fullmatch(r"phase2b_r4(?:_[a-z0-9_]+)?", args.output_name):
        parser.error("output-name must be a simple phase2b_r4 name without path separators")
    BASE = ROOT / "models" / args.output_name
    PRED = ROOT / "predictions" / args.output_name
    REPORT = ROOT / "reports" / args.output_name
    freeze() if args.command == "freeze" else run(args.command)
