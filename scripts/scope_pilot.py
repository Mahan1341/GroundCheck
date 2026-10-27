"""Frozen claim-verdict / relative-token-ranking falsifier; zero GPU forwards."""

import json

import numpy as np
from discovery_analyze import pair_statistics
from discovery_core import short_error
from research_r10 import records
from scope_analysis import BASE, DATA, PRED, REPORT, ROOT, b7_path, read, rows, sha, write
from scope_eval import evaluate_token_op, token_f1_threshold

from grounded_hallucination_lab.research.metrics import span_metrics
from grounded_hallucination_lab.research.span_model import token_targets


def combine(verifier, localizer):
    """Retain response verdict exactly, redistribute each claim risk over original tokens."""
    if verifier["id"] != localizer["id"] or verifier["offsets"] != localizer["offsets"]:
        raise ValueError("IDs and original token offsets must match")
    offsets, source = verifier["offsets"], localizer["token_probabilities"]
    if len(source) != len(offsets) or any(not 0 <= p <= 1 for p in source):
        raise ValueError("Finite aligned localizer probabilities required")
    output, fallbacks = [0.0] * len(offsets), 0
    for claim in verifier["claims"]:
        risk = claim["risk"]
        if not 0 <= risk <= 1:
            raise ValueError("Claim risk out of range")
        indices = [i for i, (a, b) in enumerate(offsets) if b > a and any(
            a < end and b > start for start, end in claim["pieces"]
        )]
        if not indices:
            raise ValueError("Nonempty claim has no covered answer tokens")
        denominator = max(source[i] for i in indices)
        fallbacks += int(denominator <= 1e-12)
        for i in indices:
            score = risk if denominator <= 1e-12 else risk * source[i] / denominator
            output[i] = max(output[i], score)
    return {**verifier, "token_probabilities": output, "localizer": "B7_relative_within_claim",
            "zero_denominator_fallbacks": fallbacks}


def token_bootstrap(token):
    groups = token["folds"]["cluster_by_row"]
    unique = sorted(set(groups))
    index = np.asarray([unique.index(g) for g in groups])
    rng = np.random.default_rng(42)
    draws = np.stack([np.bincount(rng.integers(0, len(unique), len(unique)),
                                  minlength=len(unique)) for _ in range(2000)])

    def ci(value):
        return dict(zip(("low", "high"), np.percentile(value, [2.5, 97.5]).tolist(), strict=True))

    output = {"resamples": 2000, "seed": 42, "source_clusters": len(unique),
              "scope": "conditional paired source bootstrap; "
              "no threshold refit/selection correction"
              }
    for op in ("historical", "crossfit"):
        simulation = {}
        for name, method in token["methods"].items():
            grouped = np.zeros((len(unique), 3), dtype=np.int64)
            counts = [[c[k] for k in ("tp", "fp", "fn")]
                      for c in method[op]["metrics"]["counts_by_row"]]
            np.add.at(grouped, index, counts)
            tp, fp, fn = (draws @ grouped).T
            simulation[name] = 2 * tp / np.maximum(2 * tp + fp + fn, 1)
        output[op] = {"absolute": {m: ci(v) for m, v in simulation.items()},
                      "differences": {f"pilot - {m}": ci(simulation["pilot"] - simulation[m])
                                      for m in ("B7", "seg_filtered")}}
    return output


def load(dataset):
    rr = records(DATA[dataset])
    systems = {"B7": rows(b7_path(dataset)),
               "seg_filtered": rows(PRED / f"{dataset}_seg_filtered.jsonl")}
    systems["pilot"] = [combine(a, b) for a, b in zip(
        systems["seg_filtered"], systems["B7"], strict=True)]
    assert all([r.id for r in rr] == [p["id"] for p in pp] for pp in systems.values())
    with (PRED / f"pilot_{dataset}.jsonl").open("x", encoding="utf-8") as stream:
        for row in systems["pilot"]:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    return rr, systems


def main():
    sources = [ROOT / p for p in ("scripts/scope_pilot.py", "scripts/scope_eval.py",
                                  "docs/SCOPE_PILOT_PLAN.md")]
    inputs = [DATA[d] for d in ("natural", "r1", "r6")]
    inputs += [p for d in ("natural", "r1", "r6")
               for p in (b7_path(d), PRED / f"{d}_seg_filtered.jsonl")]
    write(BASE / "pilot_freeze.json", {"files": {
        p.relative_to(ROOT).as_posix(): sha(p) for p in sources + inputs},
        "gpu_forwards": 0, "plan_frozen_before_predictions": True})
    snapshot = BASE / "pilot_source"
    snapshot.mkdir(exist_ok=False)
    for path in sources:
        (snapshot / path.name).write_bytes(path.read_bytes())
    rr, systems = load("natural")
    targets = [token_targets(p["offsets"], [(a.start, a.end) for a in r.annotations])
               for r, p in zip(rr, systems["B7"], strict=True)]
    token = evaluate_token_op(targets, [r.source_id for r in rr], systems,
                              ids=[r.id for r in rr], natural=True)
    write(BASE / "pilot_natural_token_operating_points.json", token)
    thresholds = {}
    for name, pp in systems.items():
        values = [(y, s) for tt, p in zip(targets, pp, strict=True)
                  for y, s in zip(tt, p["token_probabilities"], strict=True) if y != -100]
        thresholds[name] = token_f1_threshold([y for y, _ in values], [s for _, s in values])
    write(BASE / "pilot_controlled_token_thresholds.json", {
        "selection": "all reused natural development; frozen BEFORE control labels",
        "selections": thresholds})
    result = {"natural": {}, "controlled": {}, "bootstrap": token_bootstrap(token)}
    short = [i for i, r in enumerate(rr) if short_error(r)]
    for name, method in token["methods"].items():
        result["natural"][name] = {}
        for op in ("historical", "crossfit"):
            binary = method[op]["predictions"]
            projected = [{**p, "token_probabilities": [int(y == 1) for y in pred]}
                         for p, pred in zip(systems[name], binary, strict=True)]
            result["natural"][name][op] = {
                "localization": span_metrics(rr, projected),
                "short_n": len(short), "short_gold_hit": sum(
                    method[op]["metrics"]["counts_by_row"][i]["tp"] > 0 for i in short),
                "thresholds": [s["threshold"] for s in method[op]["fold_selections"]]
                if op == "crossfit" else [0.5]}
    audit = []
    for i, r in enumerate(rr):
        counts = {m: token["methods"][m]["crossfit"]["metrics"]["counts_by_row"][i]
                  for m in systems}
        delta = {k: counts["pilot"][k] - counts["seg_filtered"][k] for k in ("tp", "fp", "fn")}
        if any(delta.values()):
            audit.append({"id": r.id, "source_id": r.source_id, "answer": r.response,
                          "annotations": [(a.start, a.end) for a in r.annotations],
                          "counts": counts, "delta": delta})
    write(BASE / "pilot_error_changes.json", audit)
    # First control-label access only after all-natural token thresholds are saved.
    for dataset in ("r1", "r6"):
        rr, control = load(dataset)
        design = read(ROOT / ("models/phase2b_r10/r1_design.json" if dataset == "r1"
                             else "predictions/phase2b_r6/transfer_design.json"))
        paired = pair_statistics(rr, control, {"B7": 0.37612172961235046,
                                               "seg_filtered": 0.5, "pilot": 0.5}, design)
        result["controlled"][dataset] = {"paired": paired, "localization": {}}
        for name, pp in control.items():
            result["controlled"][dataset]["localization"][name] = {
                "historical": span_metrics(rr, pp),
                "natural_threshold": span_metrics(rr, [
                    {**p, "token_probabilities": [int(s >= thresholds[name]["threshold"])
                                                  for s in p["token_probabilities"]]} for p in pp])}
        assert [p["score"] for p in control["pilot"]] == [
            p["score"] for p in control["seg_filtered"]]
    loc = {m: result["natural"][m]["crossfit"]["localization"]["token"] for m in systems}
    checks = {
        "token_f1_gain_ge_03": loc["pilot"]["f1"] >= loc["seg_filtered"]["f1"] + 0.03,
        "token_f1_ge_B7_minus_02": loc["pilot"]["f1"] >= loc["B7"]["f1"] - 0.02,
        "paired_lower_ci_positive": result["bootstrap"]["crossfit"]["differences"][
            "pilot - seg_filtered"]["low"] > 0,
        "precision_gain_ge_10": loc["pilot"]["precision"] >= loc["seg_filtered"]["precision"] + .1,
        "recall_loss_le_05": loc["pilot"]["recall"] >= loc["seg_filtered"]["recall"] - .05,
        "short_hit_loss_le_one": result["natural"]["pilot"]["crossfit"]["short_gold_hit"] >=
        result["natural"]["seg_filtered"]["crossfit"]["short_gold_hit"] - 1,
        "response_scores_unchanged": all(a["score"] == b["score"] for a, b in zip(
            systems["pilot"], systems["seg_filtered"], strict=True)),
    }
    for dataset, value in result["controlled"].items():
        checks[f"{dataset}_recall_retained_90pct"] = value["localization"]["pilot"][
            "historical"]["token"]["recall"] >= .9 * value["localization"]["seg_filtered"][
                "historical"]["token"]["recall"]
        checks[f"{dataset}_local_gap_retained_80pct"] = value["paired"]["models"]["pilot"][
            "local_gap"] >= .8 * value["paired"]["models"]["seg_filtered"]["local_gap"]
    result["gate"] = {"checks": checks, "passed": all(checks.values())}
    result["zero_denominator_fallbacks_natural"] = sum(
        p["zero_denominator_fallbacks"] for p in systems["pilot"])
    write(REPORT / "pilot_results.json", result)
    print(json.dumps(result["gate"], indent=2), flush=True)


if __name__ == "__main__":
    main()
