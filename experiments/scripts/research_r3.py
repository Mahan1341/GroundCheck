"""Bounded R3 inner-only experiments. No outer calibration/dev/test command."""

import argparse
import hashlib
import json
from dataclasses import asdict, replace

import numpy as np
from research_r2 import add_diagnostics
from research_r2 import predict as old_predict
from research_r2 import threshold as old_threshold

from grounded_hallucination_lab.data.ragtruth import gold_label
from grounded_hallucination_lab.research.invariance import normalize_evidence, suite
from grounded_hallucination_lab.research.metrics import metrics, span_metrics
from grounded_hallucination_lab.research.protocol import (
    ROOT,
    digest,
    load_parts,
    partition,
    preserve_code,
    read_json,
    write_json,
    write_rows,
)
from grounded_hallucination_lab.research.relations import content_hash

BASE = ROOT / "models/phase2b_r3"
PRED = ROOT / "predictions/phase2b_r3"
REPORT = ROOT / "reports/phase2b_r3"


def state():
    result = preserve_code()
    result["R3_plan"] = digest(ROOT / "EXPERIMENT_R3_PLAN.md")
    key = hashlib.sha256(json.dumps(result, sort_keys=True).encode()).hexdigest()
    path = BASE / "snapshots" / key
    path.mkdir(parents=True, exist_ok=True)
    if not (path / "provenance.json").exists():
        write_json(path / "provenance.json", result)
        (path / "EXPERIMENT_R3_PLAN.md").write_bytes((ROOT / "EXPERIMENT_R3_PLAN.md").read_bytes())
    return result


def freeze():
    manifest = {"splits": {}, "provenance": state()}
    sources, families, values = set(), set(), set()
    for split in ("fit", "selection", "diagnostic"):
        records, design = suite(split)
        ss = {r.source_id for r in records}
        fs = {d["family"] for d in design}
        vs = {v for d in design for v in d["values"]}
        assert not sources & ss and not families & fs and not values & vs
        sources |= ss
        families |= fs
        values |= vs
        manifest["splits"][split] = {
            "n": len(records),
            "sources": sorted(ss),
            "families": sorted(fs),
            "positive": sum(gold_label(r) for r in records),
            "content_sha256": content_hash(records, design),
        }
        write_rows(PRED / f"{split}_records.jsonl", [asdict(r) for r in records])
        write_json(PRED / f"{split}_design.json", design)
    write_json(BASE / "suite_manifest.json", manifest)
    print(manifest["splits"], flush=True)


def frozen(split):
    records, design = suite(split)
    if (
        content_hash(records, design)
        != read_json(BASE / "suite_manifest.json")["splits"][split]["content_sha256"]
    ):
        raise ValueError("frozen R3 suite changed")
    return records, design


def threshold(method):
    return old_threshold("B7" if method.startswith("R3") else method.removesuffix("N"))


def predict(method, records):
    normalized = method.endswith("N")
    method = method.removesuffix("N")
    if normalized:
        records = [replace(r, evidence=normalize_evidence(r.evidence)) for r in records]
    if not method.startswith("R3"):
        return old_predict(method, records)
    from grounded_hallucination_lab.research.span_model import encode_records, load_model, score

    folder = BASE / method
    cfg = read_json(folder / "training_config.json")
    if {r.source_id for r in records} & set(cfg["train_sources"]):
        raise ValueError("heldout source used for fit")
    expected = read_json(folder / "result.json")["checkpoint_sha256"]
    if digest(folder / "best/model.safetensors") != expected:
        raise ValueError("checkpoint hash changed")
    model, tokenizer = load_model(folder / "best")
    examples, meta = encode_records(records, tokenizer, 2048, False)
    rows, runtime = score(model, tokenizer, examples, meta, batch_size=2)
    runtime["checkpoint_sha256"] = expected
    return rows, runtime


def summarize(records, design, rows, point):
    lookup = {
        (d["source_id"], d["variant"], d["order"], d["answer_index"], d["evidence_index"]): r
        for d, r in zip(design, rows, strict=True)
    }
    pairs = {}
    for d, row in zip(design, rows, strict=True):
        pairs.setdefault(d["pair_id"], {})[d["gold"]] = (d, row)
    paired = []
    for pair in pairs.values():
        d, good = pair[0]
        bad = pair[1][1]
        paired.append(
            {
                "source_id": d["source_id"],
                "variant": d["variant"],
                "kind": d["kind"],
                "pair_id": d["pair_id"],
                "correct": int(good["score"] < point <= bad["score"]),
                "directed": int(bad["score"] > good["score"] + 1e-6),
                "response_delta": bad["score"] - good["score"],
                "claim_delta": bad["claim_risk_diagnostic"] - good["claim_risk_diagnostic"],
            }
        )
    categories = {}
    for key in [
        "all",
        *sorted({d["variant"] for d in design}),
        *sorted({d["kind"] for d in design}),
    ]:
        idx = [i for i, d in enumerate(design) if key in ("all", d["variant"], d["kind"])]
        pp = [p for p in paired if key in ("all", p["variant"], p["kind"])]
        cat = {
            "metrics": metrics(
                [design[i]["gold"] for i in idx], [rows[i]["score"] for i in idx], point
            ),
            "pairs": len(pp),
            "sources": len({design[i]["source_id"] for i in idx}),
        }
        for field in ("correct", "directed", "response_delta", "claim_delta"):
            cat[field] = float(np.mean([p[field] for p in pp]))
        for label, name in [(0, "supported"), (1, "unsupported_or_contradicted")]:
            ii = [i for i in idx if design[i]["gold"] == label]
            for field in ("score", "claim_risk_diagnostic"):
                cat[f"{name}_mean_{field}"] = float(np.mean([rows[i][field] for i in ii]))
        cat["evidence_reorder_mean_absolute_delta"] = float(
            np.mean(
                [
                    abs(
                        rows[i]["score"]
                        - lookup[
                            (
                                design[i]["source_id"],
                                design[i]["variant"],
                                1,
                                design[i]["answer_index"],
                                design[i]["evidence_index"],
                            )
                        ]["score"]
                    )
                    for i in idx
                    if design[i]["order"] == 0
                ]
            )
        )
        if "offsets" in rows[0]:
            cat["localization"] = span_metrics([records[i] for i in idx], [rows[i] for i in idx])
        categories[key] = cat
    invariance = []
    for d, row in zip(design, rows, strict=True):
        if d["gold"] or d["variant"] in ("exact", "unsupported"):
            continue
        base = lookup[(d["source_id"], "exact", d["order"], d["answer_index"], d["evidence_index"])]
        invariance.append(
            {
                "source_id": d["source_id"],
                "variant": d["variant"],
                "absolute_delta": abs(row["score"] - base["score"]),
                "delta": row["score"] - base["score"],
                "introduced_fp": int(base["score"] < point <= row["score"]),
            }
        )
    return {"categories": categories, "pairs": paired, "supported_invariance": invariance}


def score_command(method, split):
    if split == "natural":
        records, design = partition(load_parts(), "inner_val"), None
    elif split == "natural_audit":
        records, design = [], None
        for r in partition(load_parts(), "inner_val"):
            if r.id not in ("11958", "12482", "13847"):
                continue
            if r.id == "11958":
                text = r.evidence.replace("15 minutes", "900 seconds")
            elif r.id == "12482":
                text = r.evidence.replace("two shades", "2 shades")
            else:
                text = "\n\n".join(reversed(r.evidence.strip().split("\n\n")))
            assert text != r.evidence
            records += [r, replace(r, id=r.id + ":equivalent", evidence=text)]
    elif split == "r2_normalization":
        from research_r2 import frozen as old_frozen

        records, design = old_frozen("diagnostic")
        selected = [
            i
            for i, d in enumerate(design)
            if d["variant"] in ("parallel", "number_words", "unit_conversion")
        ]
        records, design = [records[i] for i in selected], [design[i] for i in selected]
    else:
        records, design = frozen(split)
    path = REPORT / f"{method}_{split}.json"
    if path.exists() or (PRED / f"{method}_{split}.jsonl").exists():
        raise FileExistsError(path)
    rows, runtime = predict(method, records)
    point = threshold(method)
    report = {
        "method": method,
        "split": split,
        "threshold": point,
        "runtime": runtime,
        "provenance": state(),
    }
    if design:
        add_diagnostics(rows, design)
        if split == "r2_normalization":
            # R2 subset lacks all canonical variants needed by its full invariance scorer.
            report["categories"] = {}
            for variant in ("parallel", "number_words", "unit_conversion"):
                ii = [i for i, d in enumerate(design) if d["variant"] == variant]
                report["categories"][variant] = metrics(
                    [design[i]["gold"] for i in ii], [rows[i]["score"] for i in ii], point
                )
        else:
            report.update(summarize(records, design, rows, point))
    else:
        report["metrics"] = metrics(
            [gold_label(r) for r in records], [r["score"] for r in rows], point
        )
        report["localization"] = span_metrics(records, rows)
    write_rows(PRED / f"{method}_{split}.jsonl", rows)
    write_json(path, report)
    print(
        {
            "method": method,
            "split": split,
            "runtime": runtime,
            "metrics": report.get("metrics", report.get("categories", {}).get("all")),
        },
        flush=True,
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("command", choices=["freeze", "score", "profile", "train"])
    p.add_argument("--method", default="B7")
    p.add_argument(
        "--split",
        choices=["selection", "diagnostic", "natural", "natural_audit", "r2_normalization"],
        default="selection",
    )
    p.add_argument("--seed", type=int, choices=[42, 43, 44], default=42)
    p.add_argument("--invariance-weight", type=float, choices=[0.0, 1.0], default=0.0)
    a = p.parse_args()
    before = state()
    if a.command == "freeze":
        freeze()
    elif a.command == "score":
        score_command(a.method, a.split)
    else:
        from grounded_hallucination_lab.research.invariance_training import run

        run(a.command, a.seed, a.invariance_weight)
    if state() != before:
        raise ValueError("R3 source changed during command")


if __name__ == "__main__":
    main()
