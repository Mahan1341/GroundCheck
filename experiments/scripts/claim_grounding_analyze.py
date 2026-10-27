"""C1 bounded CPU analysis: immutable scores, overlap-proxy claims, no threshold search."""

import json
import math
from pathlib import Path

import numpy as np
from claim_grounding_projection import project
from discovery_analyze import pair_statistics
from discovery_core import short_error
from research_r10 import records
from scipy.stats import spearmanr
from scope_analysis import DATA, ROOT, b7_path, read, rows, sha, write
from sklearn.metrics import average_precision_score

from grounded_hallucination_lab.data.ragtruth import gold_label
from grounded_hallucination_lab.eval.scored import scored_metrics
from grounded_hallucination_lab.research.metrics import metrics, span_metrics
from grounded_hallucination_lab.research.span_model import token_targets

BASE = ROOT / "models/claim_grounding_20261026/c1_bounded"
PRED = ROOT / "predictions/claim_grounding_20261026/c1_bounded"
REPORT = ROOT / "reports/claim_grounding_20261026/c1_bounded"
OLD = ROOT / "predictions/scope_diagnosis_20261026"
NAMES = ("B7", "uniform", "old_pilot", "native", "null")


def cluster(value):
    return "14325+15545" if value in {"14325", "15545"} else value


def claim_rank(labels, scores):
    """Positive-claim AP; tied maxima receive fractional gold credit, not first-index credit."""
    if not labels or len(labels) != len(scores):
        raise ValueError("Nonempty aligned claim targets/scores required")
    if not any(labels):
        return {"ap": None, "pointing": None, "positive_tokens": 0, "tokens": len(labels)}
    maximum = max(scores)
    maxima = [i for i, value in enumerate(scores) if value == maximum]
    return {
        "ap": float(average_precision_score(labels, scores)),
        "pointing": sum(labels[i] for i in maxima) / len(maxima),
        "positive_tokens": sum(labels),
        "tokens": len(labels),
    }


def token_counts(record, row):
    target = token_targets(row["offsets"], [(a.start, a.end) for a in record.annotations])
    predicted = [int(p >= 0.5) for p in row["token_probabilities"]]
    return [
        sum(y == 1 and p == 1 for y, p in zip(target, predicted, strict=True)),
        sum(y == 0 and p == 1 for y, p in zip(target, predicted, strict=True)),
        sum(y == 1 and p == 0 for y, p in zip(target, predicted, strict=True)),
    ]


def rank_and_proxy(rr, layout, systems):
    labels, claim_sources = [], []
    scores = {m: [] for m in NAMES}
    ranked = {m: [] for m in NAMES}
    mapped_missing, positive_count, mixed_count = 0, 0, 0
    for i, (record, row) in enumerate(zip(rr, layout, strict=True)):
        for claim in row["claims"]:
            positive = int(
                any(
                    a.start < end and a.end > start
                    for a in record.annotations
                    for start, end in claim["pieces"]
                )
            )
            indices = [
                j
                for j, (a, b) in enumerate(row["offsets"])
                if a < b and any(a < end and b > start for start, end in claim["pieces"])
            ]
            if not indices:
                raise ValueError("Claim has no original token indices")
            targets = [
                int(
                    any(
                        max(a.start, start, row["offsets"][j][0])
                        < min(a.end, end, row["offsets"][j][1])
                        for a in record.annotations
                        for start, end in claim["pieces"]
                    )
                )
                for j in indices
            ]
            labels.append(positive)
            claim_sources.append(cluster(record.source_id))
            positive_count += positive
            mixed_count += int(positive and 0 < sum(targets) < len(targets))
            mapped_missing += int(positive and not any(targets))
            for name, predictions in systems.items():
                values = [predictions[i]["token_probabilities"][j] for j in indices]
                # Native/null/old pilot retain the original FactCG claim verdict exactly.
                scores[name].append(max(values) if name == "B7" else claim["risk"])
                if positive:
                    value = claim_rank(targets, values)
                    ranked[name].append(
                        {"source": cluster(record.source_id), "id": record.id, **value}
                    )
    result = {
        "n_claims": len(labels),
        "sources": len(set(claim_sources)),
        "positive_proxy_claims": positive_count,
        "mixed_positive_claims": mixed_count,
        "positive_claims_without_mapped_gold": mapped_missing,
        "semantics": "natural any-annotation-overlap proxy; zero is NOT independently supported",
        "methods": {},
    }
    for name in NAMES:
        values = ranked[name]
        result["methods"][name] = {
            "claim_discrimination": metrics(labels, scores[name], 0.5),
            "claim_score": "max B7 token score" if name == "B7" else "unchanged FactCG claim risk",
            "within_positive_claim_AP": sum(v["ap"] or 0 for v in values) / positive_count
            if positive_count
            else None,
            "pointing_tie_fraction": sum(v["pointing"] or 0 for v in values) / positive_count
            if positive_count
            else None,
            "AP_denominator": positive_count,
            "AP_defined": sum(v["ap"] is not None for v in values),
        }
    return result, ranked


def bootstrap(rr, counts, rankings):
    groups = sorted({cluster(r.source_id) for r in rr})
    idx = {g: i for i, g in enumerate(groups)}
    rng = np.random.default_rng(42)
    weights = np.stack(
        [
            np.bincount(rng.integers(0, len(groups), len(groups)), minlength=len(groups))
            for _ in range(2000)
        ]
    )
    simulated, ap_simulated = {}, {}
    for name in NAMES:
        table = np.zeros((len(groups), 3), dtype=np.int64)
        for record, values in zip(rr, counts[name], strict=True):
            table[idx[cluster(record.source_id)]] += values
        tp, fp, fn = (weights @ table).T
        simulated[name] = 2 * tp / np.maximum(2 * tp + fp + fn, 1)
        ap_table = np.zeros((len(groups), 2), dtype=float)
        for row in rankings[name]:
            ap_table[idx[row["source"]]] += [row["ap"] or 0, 1]
        totals, denominators = (weights @ ap_table).T
        ap_simulated[name] = np.divide(
            totals, denominators, out=np.full_like(totals, np.nan), where=denominators > 0
        )

    def ci(value):
        finite = value[np.isfinite(value)]
        return {
            "low": float(np.percentile(finite, 2.5)) if len(finite) else None,
            "high": float(np.percentile(finite, 97.5)) if len(finite) else None,
            "valid_resamples": len(finite),
        }

    return {
        "resamples": 2000,
        "seed": 42,
        "source_clusters": len(groups),
        "warning": (
            "Conditional fixed-model source resamples; only four natural clusters, not confirmation"
        ),
        "token_f1_absolute": {m: ci(value) for m, value in simulated.items()},
        "native_token_f1_differences": {
            m: ci(simulated["native"] - simulated[m])
            for m in ("B7", "uniform", "old_pilot", "null")
        },
        "native_minus_null_positive_claim_AP": ci(ap_simulated["native"] - ap_simulated["null"]),
    }


def analyze_dataset(dataset, cache, membership):
    layout = rows(BASE / f"c1_{dataset}_layout.jsonl")
    ids = [r["id"] for r in layout]
    assert ids == membership[dataset]["ids"]
    lookup = {r.id: r for r in records(DATA[dataset])}
    rr = [lookup[i] for i in ids]
    systems = {"uniform": layout}
    for name, path in (("B7", b7_path(dataset)), ("old_pilot", OLD / f"pilot_{dataset}.jsonl")):
        mapping = {r["id"]: r for r in rows(path)}
        systems[name] = [mapping[i] for i in ids]
    projected = [project(row, cache, row["winnerkeys"]) for row in layout]
    systems.update({name: [value[name] for value in projected] for name in ("native", "null")})
    systems = {name: systems[name] for name in NAMES}
    for name, predictions in systems.items():
        assert [p["id"] for p in predictions] == ids
        assert [p["offsets"] for p in predictions] == [p["offsets"] for p in layout]
        if name != "B7":
            assert [p["score"] for p in predictions] == [p["score"] for p in layout]
    for name in ("native", "null"):
        with (PRED / f"c1_{name}_{dataset}.jsonl").open("x", encoding="utf-8") as stream:
            for value in systems[name]:
                stream.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")
    labels = [gold_label(r) for r in rr]
    counts = {
        name: [token_counts(r, p) for r, p in zip(rr, pp, strict=True)]
        for name, pp in systems.items()
    }
    proxy, rankings = rank_and_proxy(rr, layout, systems)
    result = {
        "n": len(rr),
        "sources": len({r.source_id for r in rr}),
        "source_clusters": len({cluster(r.source_id) for r in rr}),
        "positive_responses": sum(labels),
        "response_identity": True,
        "token_threshold": 0.5,
        "methods": {},
        "claim_proxy": proxy,
        "bootstrap": bootstrap(rr, counts, rankings),
    }
    for name, pp in systems.items():
        short = [i for i, r in enumerate(rr) if short_error(r)]
        negatives = [i for i, y in enumerate(labels) if not y]
        token_total = sum(len(pp[i]["offsets"]) for i in negatives)
        false_tokens = sum(sum(p >= 0.5 for p in pp[i]["token_probabilities"]) for i in negatives)
        result["methods"][name] = {
            "localization": span_metrics(rr, pp),
            "response_historical": metrics(
                labels, [p["score"] for p in pp], 0.37612172961235046 if name == "B7" else 0.5
            ),
            "short_error": {
                "n": len(short),
                "gold_token_hit": sum(counts[name][i][0] > 0 for i in short),
            },
            "negative_response_token_alarms": {
                "positive_tokens": false_tokens,
                "tokens": token_total,
                "rate": false_tokens / token_total if token_total else None,
            },
        }
    if dataset == "natural":
        saved = read(
            ROOT / "models/scope_diagnosis_20261026/natural_response_operating_points.json"
        )
        order = {identity: i for i, identity in enumerate(saved["ids"])}
        for name, pp in systems.items():
            method = saved["methods"]["B7" if name == "B7" else "seg_filtered"]["crossfit"]
            decisions = [method["predictions"][order[i]] for i in ids]
            result["methods"][name]["response_cached_source_OOF"] = scored_metrics(
                labels, decisions, [p["score"] for p in pp]
            )
        result["OOF_scope"] = (
            "Reuse previous 642-row source-fold decisions; no fitting on this subset"
        )
        changes = []
        for i, r in enumerate(rr):
            if counts["native"][i] != counts["old_pilot"][i]:
                changes.append(
                    {
                        "id": r.id,
                        "source_id": r.source_id,
                        "response": r.response,
                        "evidence": r.evidence,
                        "gold": [[a.start, a.end] for a in r.annotations],
                        "counts": {m: c[i] for m, c in counts.items()},
                    }
                )
        write(
            BASE / "c1_error_changes.json",
            {"status": "existing spans, not new human adjudication", "cases": changes},
        )
    else:
        design = read(
            ROOT / "models/phase2b_r10/r1_design.json"
            if dataset == "r1"
            else ROOT / "predictions/phase2b_r6/transfer_design.json"
        )
        by_id = {r["id"]: r for r in design}
        result["paired"] = pair_statistics(
            rr,
            systems,
            {m: 0.37612172961235046 if m == "B7" else 0.5 for m in systems},
            [by_id[i] for i in ids],
        )
    projection_stats = [value["stats"] for value in projected]
    result["projection"] = {
        key: sum(s[key] for s in projection_stats)
        for key in (
            "claims",
            "zero_denominator_fallbacks",
            "unmapped_whitespace_tokens",
            "tokens_crossing_removed_nonspace",
            "null_eligible_tokens",
            "null_permuted_positions",
            "null_changed_values",
        )
    }
    write(
        BASE / f"c1_{dataset}_analysis_arrays.json",
        {
            "ids": ids,
            "token_counts": counts,
            "claim_rankings": rankings,
            "projection_stats": projection_stats,
        },
    )
    return result


def gates(results, cache, head):
    n = results["natural"]
    t = {m: v["localization"]["token"] for m, v in n["methods"].items()}
    lower = n["bootstrap"]["native_token_f1_differences"]["old_pilot"]["low"]
    rules = {
        "natural_f1_over_old_pilot": (t["native"]["f1"] - t["old_pilot"]["f1"], 0.03),
        "natural_precision_floor": (t["native"]["precision"] - t["B7"]["precision"], -0.03),
        "natural_recall_floor": (t["native"]["recall"] - t["B7"]["recall"], -0.03),
        "short_hit_floor": (
            n["methods"]["native"]["short_error"]["gold_token_hit"]
            - n["methods"]["uniform"]["short_error"]["gold_token_hit"],
            -1,
        ),
    }
    for dataset, value in results.items():
        ap = value["claim_proxy"]["methods"]
        native_ap = ap["native"]["within_positive_claim_AP"]
        null_ap = ap["null"]["within_positive_claim_AP"]
        rules[f"{dataset}_positive_claim_AP_over_null"] = (
            native_ap - null_ap if native_ap is not None and null_ap is not None else None,
            0.05,
        )
        if dataset != "natural":
            loc = value["methods"]
            rules[f"{dataset}_recall_retention"] = (
                loc["native"]["localization"]["token"]["recall"]
                - 0.9 * loc["uniform"]["localization"]["token"]["recall"],
                0,
            )
    evaluated = {
        key: {"value": value, "minimum": minimum, "passed": value is not None and value >= minimum}
        for key, (value, minimum) in rules.items()
    }
    evaluated["paired_lower_ci_positive"] = {
        "value": lower,
        "passed": lower is not None and lower > 0,
    }
    evaluated["response_identity"] = {
        "passed": all(d["response_identity"] for d in results.values())
    }
    evaluated["zero_truncation"] = {
        "passed": all(r["truncated_tokens"] == 0 for r in cache.values())
    }
    evaluated["positive_claim_target_coverage"] = {
        "passed": all(
            not d["claim_proxy"]["positive_claims_without_mapped_gold"] for d in results.values()
        )
    }
    return {
        "criteria": evaluated,
        "passed": all(v["passed"] for v in evaluated.values()),
        "learned_followup_allowed": False,
        "reason": (
            "Bounded four-cluster natural diagnostic cannot authorize training; "
            "separate decision required"
        ),
        "head_warning": head["any_rank_similarity_gt_0_9"],
    }


def main():
    assert not (REPORT / "c1_results.json").exists()
    frozen = read(BASE / "c1_freeze.json")
    for path, digest in frozen["files"].items():
        assert sha(ROOT / path) == digest, path
    files = [
        Path(__file__),
        ROOT / "scripts/claim_grounding_projection.py",
        ROOT / "docs/CLAIM_GROUNDING_C1_PLAN.md",
        BASE / "c1_membership.json",
        BASE / "c1_pairs.json",
        BASE / "c1_freeze.json",
        PRED / "c1_gradients.jsonl",
    ]
    files += [ROOT / "docs/CLAIM_GROUNDING_RESOURCE_AMENDMENT.md"]
    for dataset in ("natural", "r1", "r6"):
        files += [
            DATA[dataset],
            b7_path(dataset),
            BASE / f"c1_{dataset}_layout.jsonl",
            OLD / f"pilot_{dataset}.jsonl",
        ]
    files += [
        ROOT / "models/scope_diagnosis_20261026/natural_response_operating_points.json",
        ROOT / "scripts/discovery_analyze.py",
        ROOT / "scripts/discovery_core.py",
        ROOT / "src/grounded_hallucination_lab/research/metrics.py",
        ROOT / "src/grounded_hallucination_lab/research/span_model.py",
        ROOT / "scripts/scope_analysis.py",
        ROOT / "models/phase2b_r10/r1_design.json",
        ROOT / "predictions/phase2b_r6/transfer_design.json",
    ]
    freeze = {str(p.relative_to(ROOT)): sha(p) for p in files}
    write(
        BASE / "c1_analysis_freeze.json",
        {"files": freeze, "token_threshold": 0.5, "new_threshold_selection": False},
    )
    snapshot = BASE / "c1_analysis_source"
    snapshot.mkdir(exist_ok=False)
    for path in files[:3]:
        (snapshot / path.name).write_bytes(path.read_bytes())
    cache_rows = rows(PRED / "c1_gradients.jsonl")
    cache = {r["key"]: r for r in cache_rows}
    assert len(cache) == len(cache_rows)
    expected = set(read(BASE / "c1_pairs.json"))
    assert set(cache) == expected, "gradient run incomplete or unscoped cache"
    membership = read(BASE / "c1_membership.json")
    head_values = []
    for key, row in cache.items():
        if "random_head_L2values" in row:
            corr = float(spearmanr(row["L2values"], row["random_head_L2values"]).statistic)
            head_values.append(
                {
                    "key": key,
                    "spearman": corr if math.isfinite(corr) else None,
                    "hypothesis_tokens": len(row["L2values"]),
                }
            )
    finite = [v["spearman"] for v in head_values if v["spearman"] is not None]
    head = {
        "n": len(head_values),
        "valid": len(finite),
        "pairs": head_values,
        "median": float(np.median(finite)) if finite else None,
        "any_rank_similarity_gt_0_9": any(v > 0.9 for v in finite),
        "scope": "optional bounded-cache profile examples; no gold or benchmark selection",
        "warning": (
            "Any >.9 conservatively warns against learned followup; undefined/no pairs inconclusive"
        ),
    }
    results = {}
    for dataset in ("natural", "r1", "r6"):
        results[dataset] = analyze_dataset(dataset, cache, membership)
        print(
            dataset,
            {
                m: round(v["localization"]["token"]["f1"], 5)
                for m, v in results[dataset]["methods"].items()
            },
            flush=True,
        )
    result = {
        "datasets": results,
        "head_randomization": head,
        "gates": gates(results, cache, head),
        "analysis_freeze_sha256": sha(BASE / "c1_analysis_freeze.json"),
        "gpu_seconds_analysis": 0,
        "scope": (
            "Resource-amended bounded four-source-cluster pilot; not full C1 or independent test"
        ),
        "ranking_scope": (
            "Macro AP/pointing across positive overlap-proxy claims; all-positive claims included"
        ),
        "warnings": [
            "Gradients measure score sensitivity, not error probabilities or causal explanations",
            "Natural unannotated claims/tokens are not independently established supported facts",
        ],
    }
    assert all(sha(ROOT / path) == digest for path, digest in freeze.items())
    write(REPORT / "c1_results.json", result)
    print("C1 gates", result["gates"]["passed"], flush=True)


if __name__ == "__main__":
    main()
