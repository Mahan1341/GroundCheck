"""Frozen C2 CPU analysis: raw margins, strict zero boundary, no threshold fitting."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
from claim_grounding_analyze import cluster, token_counts
from discovery_core import short_error
from research_r10 import records
from scope_analysis import DATA, ROOT, b7_path, fast_values, read, rows, sha, write
from sklearn.metrics import average_precision_score

from grounded_hallucination_lab.data.ragtruth import gold_label
from grounded_hallucination_lab.eval.scored import scored_metrics
from grounded_hallucination_lab.research.metrics import span_metrics

ORIGINAL_BASE = ROOT / "models/claim_grounding_20261026"
ORIGINAL_PRED = ROOT / "predictions/claim_grounding_20261026"
BASE = ORIGINAL_BASE / "c2_ready"
PRED = ORIGINAL_PRED / "c2_ready"
REPORT = ROOT / "reports/claim_grounding_20261026/c2_ready"
ARMS = ("original", "joint_window", "delta")
DATASETS = ("natural", "r1", "r6", "factorial", "external")
B7_THRESHOLD = 0.37612172961235046


def response_metrics(labels, scores, *, b7=False):
    """Exact-zero margin is negative; B7 retains its historical operating point."""
    predicted = [int(s >= B7_THRESHOLD) if b7 else int(s > 0) for s in scores]
    value = scored_metrics(labels, predicted, scores)
    (tn, fp), (fn, tp) = value["confusion_matrix"]
    value.update(
        fpr=fp / (fp + tn) if fp + tn else None,
        fnr=fn / (fn + tp) if fn + tp else None,
        rule=f"score >= {B7_THRESHOLD}" if b7 else "margin > 0",
    )
    return value, predicted


def interval(value):
    finite = np.asarray(value)[np.isfinite(value)]
    return {
        "low": float(np.percentile(finite, 2.5)) if len(finite) else None,
        "high": float(np.percentile(finite, 97.5)) if len(finite) else None,
        "valid_resamples": len(finite),
    }


def claim_table(rr, systems, sources):
    """Natural labels denote annotation overlap, never independently supported claims."""
    table = []
    for i, record in enumerate(rr):
        if record is None:  # Incomplete external alignment is excluded for every method.
            continue
        layout = systems["original"][i]
        for j, claim in enumerate(layout["claims"]):
            indices = [
                k
                for k, (a, b) in enumerate(layout["offsets"])
                if a < b and any(a < y and b > x for x, y in claim["pieces"])
            ]
            if not indices:
                raise ValueError("Target has no original answer tokens")
            gold = int(
                any(
                    a.start < end and a.end > start
                    for a in record.annotations
                    for start, end in claim["pieces"]
                )
            )
            scores = {m: pp[i]["claims"][j]["margin"] for m, pp in systems.items() if m != "B7"}
            if "B7" in systems:
                scores["B7"] = max(systems["B7"][i]["token_probabilities"][k] for k in indices)
            table.append(
                {
                    "id": record.id,
                    "claim_index": j,
                    "source": sources[i],
                    "label": gold,
                    "context": not claim["no_context_fallback"],
                    "scores": scores,
                }
            )
    return table


def claim_summary(table, names):
    result = {"semantics": "any annotation overlap proxy; unannotated is not verified support"}
    for kind, subset in (("all", table), ("context_bearing", [r for r in table if r["context"]])):
        result[kind] = {
            "n": len(subset),
            "positive": sum(r["label"] for r in subset),
            "clusters": len({r["source"] for r in subset}),
            "methods": {},
        }
        if not subset:
            continue
        for name in names:
            # B7 token readout .5; historical response threshold has no claim interpretation.
            scores = [r["scores"][name] for r in subset]
            if name == "B7":
                value = scored_metrics(
                    [r["label"] for r in subset], [int(s >= 0.5) for s in scores], scores
                )
                value["rule"] = "maximum original B7 token probability >= .5"
            else:
                value, _ = response_metrics([r["label"] for r in subset], scores)
            result[kind]["methods"][name] = value
    return result


def paired_bootstrap(sources, labels, decisions, token_rows, table, resamples=2000):
    """Common source draws for response/token counts and context-bearing claim AP."""
    groups = sorted(set(sources))
    index = {s: i for i, s in enumerate(groups)}
    rng = np.random.default_rng(42)
    weights = np.stack(
        [
            np.bincount(rng.integers(0, len(groups), len(groups)), minlength=len(groups))
            for _ in range(resamples)
        ]
    )
    simulations, ap_simulations = {}, {}
    context = [r for r in table if r["context"]]
    cy = np.array([r["label"] for r in context])
    ci = np.array([index[r["source"]] for r in context], dtype=int)
    for name, predictions in decisions.items():
        counts = np.zeros((len(groups), 7), dtype=int)
        for source, y, p, tok in zip(sources, labels, predictions, token_rows[name], strict=True):
            counts[index[source]] += [
                int(y == 1 and p == 1),
                int(y == 0 and p == 1),
                int(y == 1 and p == 0),
                int(y == 0 and p == 0),
                *tok,
            ]
        simulations[name] = fast_values(weights @ counts)
        ap = []
        scores = np.array([r["scores"][name] for r in context])
        for draw in weights:
            cw = draw[ci]
            valid = len(cy) and cw[cy == 1].sum() > 0 and cw[cy == 0].sum() > 0
            ap.append(
                float(average_precision_score(cy, scores, sample_weight=cw)) if valid else np.nan
            )
        ap_simulations[name] = np.array(ap)
    return {
        "resamples": resamples,
        "seed": 42,
        "source_clusters": len(groups),
        "scope": "fixed model/decisions; common draws; no refit or selection uncertainty",
        "absolute": {m: {k: interval(v) for k, v in d.items()} for m, d in simulations.items()},
        "differences_from_original": {
            m: {k: interval(v - simulations["original"][k]) for k, v in d.items()}
            for m, d in simulations.items()
            if m != "original"
        },
        "context_claim_ap_absolute": {m: interval(v) for m, v in ap_simulations.items()},
        "context_claim_ap_difference_from_original": {
            m: interval(v - ap_simulations["original"])
            for m, v in ap_simulations.items()
            if m != "original"
        },
    }


def coverage(systems):
    claims = [c for r in systems["original"] for c in r["claims"]]
    context = [c for c in claims if not c["no_context_fallback"]]
    return {
        "claims": len(claims),
        "context_bearing_claims": len(context),
        "no_context_fallback_claims": len(claims) - len(context),
        "zero_claim_responses": sum(not r["claims"] for r in systems["original"]),
        "exact_zero_margins": {
            m: sum(c["margin"] == 0 for r in systems[m] for c in r["claims"]) for m in ARMS
        },
        "delta_absolute_margin_lt_1e_minus_6": sum(
            abs(c["margin"]) < 1e-6 for r in systems["delta"] for c in r["claims"]
        ),
        "delta_tiny_positive_margin_lt_1e_minus_6": sum(
            0 < c["margin"] < 1e-6 for r in systems["delta"] for c in r["claims"]
        ),
        "near_zero_scope": "descriptive only; strict margin > 0 remains unchanged",
        "branch_switch_semantics": "compare evidence source pieces; pair keys include hypothesis",
        "evidence_chunk_switch_joint_context": sum(
            winner_pieces(c, "joint_window") != winner_pieces(c, "context_only") for c in context
        ),
        "evidence_chunk_switch_original_joint": sum(
            winner_pieces(c, "original") != winner_pieces(c, "joint_window") for c in context
        ),
    }


def winner_pieces(claim, arm):
    key = claim["winning_evidence_keys"][arm]
    return next(e["source_pieces"] for e in claim["hypotheses"][arm]["evidence"] if e["key"] == key)


def factorial_metrics(design, systems):
    """Both target roles; paired counterfactuals hold neighbor truth and answer order fixed."""
    result = {}
    for name in ARMS:
        examples, pairs = [], {}
        for meta, row in zip(design, systems[name], strict=True):
            assert meta["id"] == row["id"] and len(meta["targets"]) == 2
            for target in meta["targets"]:
                selected = [
                    c
                    for c in row["claims"]
                    if any(a < target["end"] and b > target["start"] for a, b in c["pieces"])
                ]
                assert len(selected) == 1, "factorial atomic target must remain a single claim"
                other = next(t for t in meta["targets"] if t["parent"] != target["parent"])
                value = {
                    "source": meta["source_id"],
                    "label": target["gold"],
                    "neighbor_label": other["gold"],
                    "score": selected[0]["margin"],
                }
                examples.append(value)
                key = (meta["source_id"], target["parent"], other["gold"], meta["order"])
                pairs.setdefault(key, {})[target["gold"]] = {
                    **value,
                    "text": row["response"][target["start"] : target["end"]],
                }
        conditions = {}
        for neighbor in (0, 1):
            ee = [v for v in examples if v["neighbor_label"] == neighbor]
            metric, _ = response_metrics([e["label"] for e in ee], [e["score"] for e in ee])
            pp = [p for key, p in pairs.items() if key[2] == neighbor]
            assert all(set(p) == {0, 1} for p in pp)
            if not all(p[0]["text"] == p[1]["text"] for p in pp):
                raise ValueError("Factorial target text changed across evidence worlds")
            entries = [
                {
                    "source": p[0]["source"],
                    "correct": int(p[0]["score"] <= 0 < p[1]["score"]),
                    "separation": p[1]["score"] - p[0]["score"],
                }
                for p in pp
            ]
            conditions[str(neighbor)] = {
                "target_metrics": metric,
                "pairs": len(pp),
                "pair_accuracy": sum(e["correct"] for e in entries) / len(entries),
                "paired_target_separation": float(np.mean([e["separation"] for e in entries])),
                "pair_bootstrap": pair_bootstrap(entries),
            }
        result[name] = {
            "target_role_observations": len(examples),
            "neighbor_conditions": conditions,
            "wrong_minus_supported_neighbor_pair_accuracy": conditions["1"]["pair_accuracy"]
            - conditions["0"]["pair_accuracy"],
        }
    return result


def pair_bootstrap(entries):
    groups = sorted({e["source"] for e in entries})
    table = np.array(
        [
            [
                sum(e["correct"] for e in entries if e["source"] == source),
                sum(e["separation"] for e in entries if e["source"] == source),
                sum(e["source"] == source for e in entries),
            ]
            for source in groups
        ]
    )
    rng = np.random.default_rng(42)
    draws = np.stack(
        [
            np.bincount(rng.integers(0, len(groups), len(groups)), minlength=len(groups))
            for _ in range(2000)
        ]
    )
    correct, separation, n = (draws @ table).T
    return {
        "clusters": len(groups),
        "resamples": 2000,
        "seed": 42,
        "pair_accuracy": interval(correct / n),
        "separation": interval(separation / n),
    }


def analyze_dataset(dataset):
    systems = {m: rows(PRED / f"c2_{dataset}_{m}.jsonl") for m in ARMS}
    ids = [r["id"] for r in systems["original"]]
    assert len(ids) == len(set(ids))
    assert ids == [r["id"] for r in rows(BASE / f"c2_{dataset}_layout.jsonl")]
    assert len(ids) == read(BASE / "c2_freeze.json")["counts"][dataset]
    if dataset == "external":
        inputs = {r["id"]: r for r in rows(ORIGINAL_BASE / "external/samsum_inputs.jsonl")}
        annotations = {r["id"]: r for r in rows(ORIGINAL_BASE / "external/samsum_labels.jsonl")}
        assert set(ids) == set(inputs) == set(annotations)
        rr = [
            SimpleNamespace(
                id=i,
                source_id=annotations[i]["parent_id"],
                response=inputs[i]["response"],
                annotations=[
                    SimpleNamespace(start=a, end=b) for a, b in annotations[i]["gold_char_spans"]
                ],
            )
            if annotations[i]["localization_scorable"]
            else None
            for i in ids
        ]
        labels = [int(annotations[i]["label"]) for i in ids]
        sources = [annotations[i]["parent_id"] for i in ids]
        bpath = ORIGINAL_PRED / "external_B7.jsonl"
    else:
        path = (
            ORIGINAL_BASE / "c2_factorial_records.jsonl"
            if dataset == "factorial"
            else DATA[dataset]
        )
        lookup = {r.id: r for r in records(path)}
        rr = [lookup[i] for i in ids]
        labels = [gold_label(r) for r in rr]
        sources = [cluster(r.source_id) for r in rr]
        bpath = b7_path(dataset) if dataset != "factorial" else None
    if bpath:
        lookup = {r["id"]: r for r in rows(bpath)}
        systems["B7"] = [lookup[i] for i in ids]
    for name, pp in systems.items():
        assert [r["id"] for r in pp] == ids
        assert [r["offsets"] for r in pp] == [r["offsets"] for r in systems["original"]]
        if name != "B7":
            assert all(r["prediction"] == int(r["score"] > 0) for r in pp)
            assert all(v in (0, 1) for r in pp for v in r["token_probabilities"])
            assert [r["retained"] for r in pp] == [r["retained"] for r in systems["original"]]
            assert [[c["pieces"] for c in r["claims"]] for r in pp] == [
                [c["pieces"] for c in r["claims"]] for r in systems["original"]
            ]
    scorable = [i for i, r in enumerate(rr) if r is not None]
    short = [i for i in scorable if short_error(rr[i])]
    table = claim_table(rr, systems, sources)
    counts, decisions, methods = {}, {}, {}
    for name, pp in systems.items():
        counts[name] = [
            token_counts(r, p) if r is not None else [0, 0, 0] for r, p in zip(rr, pp, strict=True)
        ]
        response, decisions[name] = response_metrics(
            labels, [p["score"] for p in pp], b7=name == "B7"
        )
        methods[name] = {
            "response": response,
            "localization": span_metrics([rr[i] for i in scorable], [pp[i] for i in scorable]),
            "short_error": {
                "n": len(short),
                "gold_token_hit": sum(counts[name][i][0] > 0 for i in short),
            },
            "negative_response_token_alarms": {
                "fp_tokens": sum(counts[name][i][1] for i in scorable if not labels[i]),
                "negative_responses_with_alarm": sum(
                    counts[name][i][1] > 0 for i in scorable if not labels[i]
                ),
                "negative_responses": sum(not labels[i] for i in scorable),
            },
        }
    result = {
        "n": len(ids),
        "positive_responses": sum(labels),
        "clusters": len(set(sources)),
        "localization_n": len(scorable),
        "localization_excluded": len(ids) - len(scorable),
        "localization_excluded_positive": sum(labels[i] for i, r in enumerate(rr) if r is None),
        "coverage": coverage(systems),
        "methods": methods,
        "claim_proxy": claim_summary(table, systems),
        "bootstrap": paired_bootstrap(sources, labels, decisions, counts, table),
    }
    if dataset == "factorial":
        result["factorial"] = factorial_metrics(
            read(ORIGINAL_BASE / "c2_factorial_design.json"), systems
        )
    errors = []
    for row in table:
        a, d = row["scores"]["original"], row["scores"]["delta"]
        if (a > 0) != (d > 0):
            idx = ids.index(row["id"])
            errors.append(
                {
                    **row,
                    "response": systems["original"][idx]["response"],
                    "claim": systems["original"][idx]["claims"][row["claim_index"]],
                    "warning": "annotation-overlap proxy; no independent claim adjudication",
                }
            )
    write(BASE / f"c2_{dataset}_casepack.json", errors[:24])
    return result


def gates(results):
    n = results["natural"]
    methods = n["methods"]
    ca = n["claim_proxy"]["context_bearing"]["methods"]
    rules = {
        "natural_context_claim_AP_gain": (
            ca["delta"]["average_precision"] - ca["original"]["average_precision"],
            0.03,
        ),
        "natural_response_macro_f1_loss": (
            methods["delta"]["response"]["macro_f1"] - methods["original"]["response"]["macro_f1"],
            -0.02,
        ),
        "natural_token_recall_loss": (
            methods["delta"]["localization"]["token"]["recall"]
            - methods["original"]["localization"]["token"]["recall"],
            -0.05,
        ),
        "natural_short_hit_loss": (
            methods["delta"]["short_error"]["gold_token_hit"]
            - methods["original"]["short_error"]["gold_token_hit"],
            -1,
        ),
    }
    for dataset in ("r1", "r6"):
        m = results[dataset]["methods"]
        rules[f"{dataset}_token_recall_retention"] = (
            m["delta"]["localization"]["token"]["recall"]
            - 0.9 * m["original"]["localization"]["token"]["recall"],
            0,
        )
    factorial = results["factorial"]["factorial"]
    for neighbor in ("0", "1"):
        rules[f"factorial_neighbor_{neighbor}_recall_retention"] = (
            factorial["delta"]["neighbor_conditions"][neighbor]["target_metrics"]["positive_class"][
                "recall"
            ]
            - 0.9
            * factorial["original"]["neighbor_conditions"][neighbor]["target_metrics"][
                "positive_class"
            ]["recall"],
            0,
        )
    rules["factorial_wrong_neighbor_pair_accuracy_drop"] = (
        factorial["delta"]["wrong_minus_supported_neighbor_pair_accuracy"],
        -0.05,
    )
    evaluated = {
        key: {"value": value, "minimum": minimum, "passed": value >= minimum}
        for key, (value, minimum) in rules.items()
    }
    low = n["bootstrap"]["context_claim_ap_difference_from_original"]["delta"]["low"]
    evaluated["natural_claim_AP_paired_lowerCI_positive"] = {
        "value": low,
        "passed": low is not None and low > 0,
    }
    return {
        "criteria": evaluated,
        "passed": all(v["passed"] for v in evaluated.values()),
        "external_used_for_gate": False,
        "fit_performed": False,
    }


def main():
    assert not (REPORT / "c2_results.json").exists()
    for folder in (BASE, ORIGINAL_BASE):
        for path, digest in read(folder / "c2_freeze.json")["files"].items():
            assert sha(ROOT / path) == digest, path
    source = [
        Path(__file__),
        ROOT / "tests/test_claim_grounding_c2_analyze.py",
        ROOT / "scripts/claim_grounding_analyze.py",
        ROOT / "scripts/scope_analysis.py",
        ROOT / "scripts/discovery_core.py",
        ROOT / "scripts/research_r10.py",
        ROOT / "src/grounded_hallucination_lab/research/metrics.py",
        ROOT / "src/grounded_hallucination_lab/research/span_model.py",
        ROOT / "src/grounded_hallucination_lab/eval/scored.py",
        ROOT / "src/grounded_hallucination_lab/eval/metrics.py",
        ROOT / "docs/CLAIM_GROUNDING_C2_PLAN.md",
        ROOT / "docs/CLAIM_GROUNDING_C2_RESOURCE.md",
    ]
    files = source + [
        BASE / "c2_freeze.json",
        ORIGINAL_BASE / "c2_freeze.json",
        ORIGINAL_BASE / "c1_membership.json",
        ORIGINAL_BASE / "c2_factorial_records.jsonl",
        ORIGINAL_BASE / "c2_factorial_design.json",
        ORIGINAL_BASE / "external/samsum_inputs.jsonl",
        ORIGINAL_BASE / "external/samsum_labels.jsonl",
        ORIGINAL_BASE / "external/samsum_manifest.json",
        ORIGINAL_PRED / "external_B7.jsonl",
    ]
    files += [DATA[d] for d in ("natural", "r1", "r6")]
    files += [b7_path(d) for d in ("natural", "r1", "r6")]
    files += [PRED / f"c2_{d}_{m}.jsonl" for d in DATASETS for m in ARMS]
    freeze = {str(p.relative_to(ROOT)): sha(p) for p in files}
    write(BASE / "c2_analysis_freeze.json", {"files": freeze, "new_threshold_selection": False})
    snapshot = BASE / "c2_analysis_source"
    snapshot.mkdir(exist_ok=False)
    for path in source:
        target = snapshot / path.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(path.read_bytes())
    results = {}
    for dataset in DATASETS:
        results[dataset] = analyze_dataset(dataset)
        print(
            dataset,
            {
                m: {
                    "macro_f1": round(v["response"]["macro_f1"], 5),
                    "token_f1": round(v["localization"]["token"]["f1"], 5),
                }
                for m, v in results[dataset]["methods"].items()
            },
            flush=True,
        )
    report = {
        "datasets": results,
        "gates": gates(results),
        "gpu_seconds_analysis": 0,
        "analysis_freeze_sha256": sha(BASE / "c2_analysis_freeze.json"),
        "warnings": [
            "Reused natural inner is not independent confirmation",
            "Any-overlap natural claim labels are a proxy, not supported claim gold",
            "All C2 arms are coarse claim localizers, without within-claim resolution",
            "Finite context erasure includes deletion and nonadditivity effects",
            "Context is only immediate previous/next kept claims, not complete discourse",
            "External localization excludes 11 incompletely aligned positive summaries",
        ],
    }
    assert all(sha(ROOT / path) == digest for path, digest in freeze.items())
    write(REPORT / "c2_results.json", report)
    print("C2 gate", report["gates"]["passed"], flush=True)


if __name__ == "__main__":
    main()
