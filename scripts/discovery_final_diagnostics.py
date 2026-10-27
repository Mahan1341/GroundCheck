"""Frozen Cycle2/4 contrasts, descriptive strata and annotation-risk audit; CPU only."""

import hashlib
import json
from collections import Counter
from pathlib import Path

from discovery_analyze import local, pair_statistics
from discovery_core import short_error
from research_r10 import records

from grounded_hallucination_lab.data.ragtruth import gold_label
from grounded_hallucination_lab.research.metrics import bootstrap, metrics, span_metrics

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "models/discovery_loop_20261026"
REPORT = ROOT / "reports/discovery_loop_20261026"
PRED = ROOT / "predictions/discovery_loop_20261026"
THRESHOLDS = {"B7": 0.37612172961235046, "Segmented": 0.5, "Whole": 0.5}
INPUTS = {}


def capture(path):
    INPUTS[path.relative_to(ROOT).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    return path


def read(path):
    return json.loads(capture(path).read_text(encoding="utf-8"))


def rows(path):
    return [json.loads(line) for line in capture(path).read_text(encoding="utf-8").splitlines()]


def boot(rr, systems, merge=False):
    groups = [
        "merged_14325_15545" if merge and r.source_id in ("14325", "15545") else r.source_id
        for r in rr
    ]
    return bootstrap(
        [gold_label(r) for r in rr],
        groups,
        {
            name: (
                [int(p["score"] >= THRESHOLDS[name]) for p in predictions],
                [p["score"] for p in predictions],
            )
            for name, predictions in systems.items()
        },
        "Segmented",
    )


def concise(rr, systems):
    result = {"n": len(rr), "sources": len({r.source_id for r in rr}), "models": {}}
    y = [gold_label(r) for r in rr]
    for name, predictions in systems.items():
        value = metrics(y, [p["score"] for p in predictions], THRESHOLDS[name])
        result["models"][name] = {
            key: value[key]
            for key in (
                "class_counts",
                "macro_f1",
                "positive_class",
                "fpr",
                "fnr",
                "auroc",
                "average_precision",
                "confusion_matrix",
            )
        }
        result["models"][name]["token_f1"] = span_metrics(rr, predictions)["token"]["f1"]
    return result


def gate(condition, actual, rule):
    return {"passed": bool(condition), "actual": actual, "rule": rule}


def frozen_gates(c2, c4, direct):
    a, b = c2["natural"], c4["natural"]
    am, bm = a["metrics"]["FactCG"], b["metrics"]["FactCG"]
    baseline = a["metrics"]["B7"]
    delta2 = am["macro_f1"] - baseline["macro_f1"]
    low2 = a["bootstrap"]["difference_from_reference"]["FactCG"]["macro_f1"]["low"]
    delta4 = bm["macro_f1"] - am["macro_f1"]
    low4 = direct["difference_from_reference"]["Whole"]["macro_f1"]["low"]
    c2g = {
        "natural_gain": gate(delta2 >= 0.01, delta2, ">=.01 vs B7"),
        "natural_lower_ci": gate(low2 > 0, low2, ">0"),
        "token_f1": gate(
            a["localization"]["FactCG"]["token"]["f1"]
            >= a["localization"]["B7"]["token"]["f1"] - 0.02,
            a["localization"]["FactCG"]["token"]["f1"] - a["localization"]["B7"]["token"]["f1"],
            ">=-.02 vs B7",
        ),
    }
    for key in ("tp", "gold_token_hit"):
        delta = a["short_error"]["FactCG"][key] - a["short_error"]["B7"][key]
        c2g[f"short_{key}"] = gate(delta >= 0, delta, ">=0 vs B7")
    c4g = {
        "natural_gain": gate(delta4 >= 0.05, delta4, ">=.05 vs Segmented"),
        "natural_lower_ci": gate(low4 > 0, low4, ">0"),
        "fpr_drop": gate(am["fpr"] - bm["fpr"] >= 0.10, am["fpr"] - bm["fpr"], ">=.10"),
        "natural_vs_B7": gate(
            bm["macro_f1"] >= baseline["macro_f1"] - 0.01,
            bm["macro_f1"] - baseline["macro_f1"],
            ">=-.01",
        ),
        "short_tp": gate(
            b["short_error"]["FactCG"]["tp"] >= a["short_error"]["FactCG"]["tp"] - 1,
            b["short_error"]["FactCG"]["tp"] - a["short_error"]["FactCG"]["tp"],
            ">=-1",
        ),
        "token_f1": gate(
            b["localization"]["FactCG"]["token"]["f1"]
            >= a["localization"]["FactCG"]["token"]["f1"] - 0.02,
            b["localization"]["FactCG"]["token"]["f1"] - a["localization"]["FactCG"]["token"]["f1"],
            ">=-.02 vs Segmented",
        ),
    }
    for suite in ("r1", "r6"):
        old, new = c2[suite]["paired"]["models"], c4[suite]["paired"]["models"]
        for key, bound in (("pair_accuracy", 0.10), ("local_gap", 0.05)):
            delta = old["FactCG"][key] - old["B7"][key]
            c2g[f"{suite}_{key}"] = gate(delta >= bound, delta, f">={bound} vs B7")
        delta = new["FactCG"]["pair_accuracy"] - old["FactCG"]["pair_accuracy"]
        c4g[f"{suite}_pair_accuracy"] = gate(delta >= -0.02, delta, ">=-.02 vs Segmented")
    return {
        name: {"criteria": criteria, "passed": all(v["passed"] for v in criteria.values())}
        for name, criteria in (("cycle2", c2g), ("cycle4", c4g))
    }


def main():
    destination = REPORT / "final_diagnostics.json"
    assert not destination.exists(), "preserve previous diagnostics"
    c2, c4 = read(REPORT / "cycle2_analysis.json"), read(REPORT / "cycle4_analysis.json")
    for name in ("DISCOVERY_CYCLE2_PREREG.md", "DISCOVERY_CYCLE4_PREREG.md"):
        capture(ROOT / "docs" / name)
    direct, datasets = {}, {}
    for suite, path in {
        "natural": ROOT / "models/phase2b_r10/natural_records.jsonl",
        "r1": ROOT / "models/phase2b_r10/r1_records.jsonl",
        "r6": ROOT / "predictions/phase2b_r6/transfer_records.jsonl",
    }.items():
        rr = records(capture(path))
        systems = {
            "B7": rows(ROOT / "predictions/phase2b_r10" / f"B7_{suite}.jsonl"),
            "Segmented": rows(PRED / f"cycle2_FactCG_{suite}.jsonl"),
            "Whole": rows(PRED / f"cycle4_FactCG_{suite}.jsonl"),
        }
        for predictions in systems.values():
            assert [r.id for r in rr] == [p["id"] for p in predictions]
        datasets[suite] = rr, systems
        direct[suite] = {"bootstrap": boot(rr, systems)}
        if suite != "natural":
            design = read(
                ROOT / "models/phase2b_r10/r1_design.json"
                if suite == "r1"
                else ROOT / "predictions/phase2b_r6/transfer_design.json"
            )
            paired = pair_statistics(
                rr,
                {"B7": systems["Segmented"], "Whole": systems["Whole"]},
                {"B7": 0.5, "Whole": 0.5},
                design,
            )
            whole = paired["models"]["Whole"]
            whole["delta_Segmented_ci95"] = whole.pop("delta_B7_ci95")
            paired["models"]["Segmented"] = paired["models"].pop("B7")
            direct[suite]["paired"] = paired
        print(f"paired source bootstrap {suite} complete", flush=True)
    rr, systems = datasets["natural"]
    layout = read(BASE / "cycle2_natural_layout.json")
    whole_layout = read(BASE / "cycle4_natural_layout.json")
    assert [p["id"] for p in layout] == [r.id for r in rr]
    assert [p["id"] for p in whole_layout] == [r.id for r in rr]
    strata = {"answer_tokens": {}, "segments": {}, "source_chunks": {}}
    for index, row in enumerate(layout):
        length, segments = len(row["offsets"]), len(row["segments"])
        chunk_counts = {len(s["evidence"]) for s in whole_layout[index]["segments"]}
        assert len(chunk_counts) == 1, "whole-answer layout exposes unchanged evidence chunks"
        chunks = next(iter(chunk_counts))
        assert all(len(s["evidence"]) == chunks for s in row["segments"])
        keys = {
            "answer_tokens": "lt100" if length < 100 else "100_199" if length < 200 else "ge200",
            "segments": str(segments) if segments <= 1 else "2_4" if segments <= 4 else "ge5",
            "source_chunks": str(chunks),
        }
        for dimension, key in keys.items():
            strata[dimension].setdefault(key, []).append(index)
    for groups in strata.values():
        for key, indices in groups.items():
            groups[key] = concise(
                [rr[i] for i in indices], {m: [p[i] for i in indices] for m, p in systems.items()}
            )
    by_id = {r.id: (r, {m: p[i] for m, p in systems.items()}) for i, r in enumerate(rr)}
    audit = read(BASE / "factcg_fp_audit.json")
    fixed = []
    for case in audit["cases"]:
        record, predictions = by_id[case["id"]]
        assert gold_label(record) == 0
        category = case["assistant_review"]["primary_category"]
        fixed.append(
            {
                "id": record.id,
                "source_id": record.source_id,
                "gold": 0,
                "assistant_provisional_category": category,
                "scores": {m: p["score"] for m, p in predictions.items()},
                "predictions": {
                    m: int(p["score"] >= THRESHOLDS[m]) for m, p in predictions.items()
                },
                "suppressed_by_Whole": predictions["Segmented"]["score"]
                >= 0.5
                > predictions["Whole"]["score"],
                "suspected_annotation_issue": category == "potential_unannotated_error",
            }
        )
    short = []
    for record, predictions in by_id.values():
        if short_error(record):
            short.append(
                {
                    "id": record.id,
                    "source_id": record.source_id,
                    "models": {
                        m: {
                            "score": p["score"],
                            "tp": p["score"] >= THRESHOLDS[m],
                            "gold_token_hit": any(
                                local(p, a.start, a.end) >= 0.5 for a in record.annotations
                            ),
                        }
                        for m, p in predictions.items()
                    },
                }
            )
    result = {
        "scope": "reused natural inner development; fixed models/thresholds; no new inference",
        "direct_Whole_vs_Segmented": direct,
        "frozen_gates": frozen_gates(c2, c4, direct["natural"]["bootstrap"]),
        "natural_merged_source_sensitivity": {
            "note": (
                "14325+15545 share one normalized passage. Bootstrap grouping only; "
                "no split/ID change."
            ),
            "bootstrap": boot(rr, systems, merge=True),
        },
        "natural_strata": strata,
        "strata_warning": (
            "Descriptive posthoc strata, correlated dimensions; no causal identification. "
            "Whole includes regions dropped by clean_segments: this also changes checkworthiness "
            "filtering, not only claim granularity and max aggregation."
        ),
        "strata_correction": (
            "Initial final_diagnostics_v0_superseded_strata.json preserved: zero segments were "
            "incorrectly labeled zero source chunks. Corrected from whole-answer evidence layout; "
            "bootstrap, predictions and gates unchanged. Zero-segment answers reported separately."
        ),
        "short_error_cases": short,
        "fixed_12_FP_audit": {
            "n": len(fixed),
            "cases": fixed,
            "suppressed": sum(c["suppressed_by_Whole"] for c in fixed),
            "potential_unannotated_suppressed": sum(
                c["suppressed_by_Whole"] and c["suspected_annotation_issue"] for c in fixed
            ),
            "category_counts": dict(Counter(c["assistant_provisional_category"] for c in fixed)),
            "warning": (
                "Assistant provisional judgments, NOT independent gold; suppression is not "
                "proof of a correction. Labels unchanged."
            ),
        },
        "inputs_sha256": INPUTS,
        "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "gpu_seconds": 0,
    }
    with destination.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
    print({name: value["passed"] for name, value in result["frozen_gates"].items()}, flush=True)


if __name__ == "__main__":
    main()
