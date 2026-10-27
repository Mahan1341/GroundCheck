"""R2 only: authored diagnostics and natural INNER validation; no outer mode."""

import argparse
import hashlib
import json
from dataclasses import asdict

import numpy as np

from grounded_hallucination_lab.data.ragtruth import gold_label
from grounded_hallucination_lab.research.metrics import metrics, span_metrics
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
from grounded_hallucination_lab.research.relations import content_hash, relation_suite

BASE = ROOT / "models/phase2b_r2"
PRED = ROOT / "predictions/phase2b_r2"
REPORT = ROOT / "reports/phase2b_r2"


def state():
    result = preserve_code()
    result["R2_plan"] = digest(ROOT / "EXPERIMENT_R2_PLAN.md")
    result["amendments"] = {p.name: digest(p) for p in ROOT.glob("EXPERIMENT_R2_*AMENDMENT*.md")}
    key = hashlib.sha256(json.dumps(result, sort_keys=True).encode()).hexdigest()
    path = BASE / "snapshots" / key
    path.mkdir(parents=True, exist_ok=True)
    for plan in [ROOT / "EXPERIMENT_R2_PLAN.md", *ROOT.glob("EXPERIMENT_R2_*AMENDMENT*.md")]:
        saved = path / plan.name
        if not saved.exists():
            saved.write_bytes(plan.read_bytes())
    if not (path / "provenance.json").exists():
        write_json(path / "provenance.json", result)
    return result


def freeze():
    manifest = {"splits": {}, "provenance": state(), "semantic_spec": "research/relations.py"}
    all_sources = set()
    for split in ("fit", "selection", "diagnostic"):
        records, design = relation_suite(split)
        sources = {r.source_id for r in records}
        assert not sources & all_sources
        all_sources |= sources
        manifest["splits"][split] = {
            "n": len(records),
            "sources": sorted(sources),
            "positive": sum(d["gold"] for d in design),
            "families": sorted({d["family"] for d in design}),
            "content_sha256": content_hash(records, design),
        }
        write_rows(PRED / f"{split}_records.jsonl", [asdict(r) for r in records])
        write_json(PRED / f"{split}_design.json", design)
    write_json(BASE / "suite_manifest.json", manifest)
    print(manifest["splits"], flush=True)


def frozen(split):
    records, design = relation_suite(split)
    expected = read_json(BASE / "suite_manifest.json")["splits"][split]["content_sha256"]
    if content_hash(records, design) != expected:
        raise ValueError("R2 suite content changed after freeze")
    return records, design


def threshold(method):
    if method.startswith("R2"):
        method = "B7"
    folder = ROOT / (
        "predictions/phase2b_resume" if method in ("C0", "C1") else "predictions/phase2b"
    )
    name = method if method in ("C0", "C1", "B6") else method + "s42"
    return read_json(folder / f"{name}_inner_val.meta.json")["inner_selection"]["threshold"]


def predict(method, records):
    allowed = {"B7", "B8", "B6", "C0", "C1"} | {
        f"R2C{arm}s{seed}" for arm in (0, 1) for seed in (42, 43, 44)
    }
    if method not in allowed:
        raise ValueError("unknown R2 model identifier")
    if not method.startswith("R2"):
        from research_resume import score_model

        return score_model(method, records)
    from grounded_hallucination_lab.research.span_model import encode_records, load_model, score

    folder = BASE / method
    cfg = read_json(folder / "training_config.json")
    if {r.source_id for r in records} & set(cfg["train_sources"]):
        raise ValueError("prediction source was used in supervised fit")
    model, tokenizer = load_model(folder / "best")
    ex, meta = encode_records(records, tokenizer, 2048, False)
    rows, runtime = score(model, tokenizer, ex, meta, batch_size=2)
    runtime["checkpoint_sha256"] = digest(folder / "best/model.safetensors")
    return rows, runtime


def add_diagnostics(rows, design):
    for row, d in zip(rows, design, strict=True):
        assert row["id"] == d["id"]
        if "offsets" in row:
            for name in ("claim", "slot"):
                row[f"{name}_risk_diagnostic"] = max(
                    (
                        p
                        for (a, b), p in zip(
                            row["offsets"], row["token_probabilities"], strict=True
                        )
                        if a < d[f"{name}_end"] and b > d[f"{name}_start"]
                    ),
                    default=0.0,
                )
        else:
            row["claim_risk_diagnostic"] = max(
                (
                    1 - s["support"]
                    for s in row["segments"]
                    if s["start"] < d["claim_end"] and s["end"] > d["claim_start"]
                ),
                default=0.0,
            )
            row["slot_risk_diagnostic"] = None  # MiniCheck has no token-level head.


def summarize(records, design, rows, point):
    pairs, lookup = {}, {}
    for d, r in zip(design, rows, strict=True):
        pairs.setdefault(d["pair_id"], {})[d["gold"]] = (d, r)
        lookup[
            (d["source_id"], d["variant"], d["order"], d["answer_index"], d["evidence_index"])
        ] = r
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
                "slot_delta": (
                    None
                    if good["slot_risk_diagnostic"] is None
                    else bad["slot_risk_diagnostic"] - good["slot_risk_diagnostic"]
                ),
            }
        )
    categories = {}
    keys = ["all", *sorted({d["variant"] for d in design}), *sorted({d["kind"] for d in design})]
    for key in keys:
        idx = [i for i, d in enumerate(design) if key in ("all", d["kind"], d["variant"])]
        selected = [p for p in paired if key in ("all", p["kind"], p["variant"])]
        cat = {
            "metrics": metrics(
                [design[i]["gold"] for i in idx], [rows[i]["score"] for i in idx], point
            ),
            "pairs": len(selected),
            "sources": len({design[i]["source_id"] for i in idx}),
        }
        for field in ("correct", "directed", "response_delta", "claim_delta", "slot_delta"):
            values = [p[field] for p in selected if p[field] is not None]
            cat[field] = float(np.mean(values)) if values else None
        reorder = []
        for i in idx:
            d, r = design[i], rows[i]
            if d["order"] == 0:
                twin = lookup[
                    (d["source_id"], d["variant"], 1, d["answer_index"], d["evidence_index"])
                ]
                reorder.append(abs(twin["score"] - r["score"]))
        cat["evidence_reorder_mean_absolute_delta"] = float(np.mean(reorder))
        if "offsets" in rows[0]:
            cat["localization"] = span_metrics([records[i] for i in idx], [rows[i] for i in idx])
        categories[key] = cat
    invariance = []
    for d, r in zip(design, rows, strict=True):
        if d["gold"] or d["variant"] in ("record", "parallel", "novel", "known_values_novel"):
            continue
        baseline = "novel" if d["padding"] else "parallel"
        base = lookup[
            (d["source_id"], baseline, d["order"], d["answer_index"], d["evidence_index"])
        ]
        invariance.append(
            {
                "source_id": d["source_id"],
                "variant": d["variant"],
                "delta": r["score"] - base["score"],
                "absolute_delta": abs(r["score"] - base["score"]),
                "introduced_fp": int(base["score"] < point <= r["score"]),
            }
        )
    return {"categories": categories, "pairs": paired, "supported_invariance": invariance}


def score_command(method, split):
    records, design = (
        (partition(load_parts(), "inner_val"), None) if split == "natural" else frozen(split)
    )
    rows, runtime = predict(method, records)
    point = threshold(method)
    report = {
        "method": method,
        "split": split,
        "threshold": point,
        "runtime": runtime,
        "provenance": state(),
    }
    if design is not None:
        add_diagnostics(rows, design)
        report.update(summarize(records, design, rows, point))
    else:
        report["metrics"] = metrics(
            [gold_label(r) for r in records], [r["score"] for r in rows], point
        )
        report["localization"] = span_metrics(records, rows)
    write_rows(PRED / f"{method}_{split}.jsonl", rows)
    write_json(REPORT / f"{method}_{split}.json", report)
    print(
        {
            "method": method,
            "split": split,
            "summary": report.get("metrics", report.get("categories", {}).get("all")),
            "runtime": runtime,
        },
        flush=True,
    )


def compare(split, methods, reference):
    from grounded_hallucination_lab.research.metrics import bootstrap

    if split == "natural":
        records, design = partition(load_parts(), "inner_val"), None
    else:
        records, design = frozen(split)
    labels, sources = [gold_label(r) for r in records], [r.source_id for r in records]
    systems, reports = {}, {}
    for method in methods:
        rows = read_rows(PRED / f"{method}_{split}.jsonl")
        assert [r["id"] for r in rows] == [r.id for r in records]
        scores = [r["score"] for r in rows]
        systems[method] = ([int(s >= threshold(method)) for s in scores], scores)
        reports[method] = read_json(REPORT / f"{method}_{split}.json")
    result = {
        "bootstrap": bootstrap(labels, sources, systems, reference, 2000),
        "reference": reference,
        "methods": methods,
        "resamples": 2000,
        "role": "internal authored diagnostic or natural inner; not outer test",
    }
    if design:
        result["paired_grounding_intervals"] = pair_intervals(reports, reference)
    suffix = "_".join(methods)
    write_json(REPORT / f"comparison_{split}_{suffix}.json", result)


def pair_intervals(reports, reference):
    """Same source draws for absolute and paired differences, including category strata."""
    output = {}
    all_sources = sorted({p["source_id"] for p in reports[reference]["pairs"]})
    rng = np.random.default_rng(42)
    draws = rng.integers(0, len(all_sources), size=(2000, len(all_sources)))
    for category in reports[reference]["categories"]:
        output[category] = {}
        for field in ("correct", "directed", "response_delta", "claim_delta"):
            samples = {}
            for method, report in reports.items():
                pairs = [p for p in report["pairs"] if category in ("all", p["kind"], p["variant"])]
                sums = np.array(
                    [sum(p[field] for p in pairs if p["source_id"] == s) for s in all_sources]
                )
                counts = np.array([sum(p["source_id"] == s for p in pairs) for s in all_sources])
                denom = counts[draws].sum(1)
                samples[method] = np.divide(
                    sums[draws].sum(1), denom, out=np.full(2000, np.nan), where=denom > 0
                )
            output[category][field] = {
                m: {
                    "absolute95": np.nanpercentile(v, [2.5, 97.5]).tolist(),
                    "difference95": np.nanpercentile(v - samples[reference], [2.5, 97.5]).tolist(),
                }
                for m, v in samples.items()
            }
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["freeze", "score", "compare", "profile", "train"])
    parser.add_argument("--method", default="B7")
    parser.add_argument(
        "--split", choices=["selection", "diagnostic", "natural"], default="diagnostic"
    )
    parser.add_argument("--methods", nargs="+", default=["B7", "B8", "C0", "C1", "B6"])
    parser.add_argument("--reference", default="B6")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pair-weight", type=float, choices=[0.0, 0.1], default=0.0)
    args = parser.parse_args()
    before = state()
    if args.command == "freeze":
        freeze()
    elif args.command == "score":
        score_command(args.method, args.split)
    elif args.command == "compare":
        compare(args.split, args.methods, args.reference)
    else:
        from grounded_hallucination_lab.research.pair_training import run

        run(args.command, args.seed, args.pair_weight)
    if state() != before:
        raise ValueError("R2 source changed during command")


if __name__ == "__main__":
    main()
