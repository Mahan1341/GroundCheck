"""Paired source diagnostics shared by frozen discovery cycles."""

import argparse
from collections import defaultdict

import numpy as np
from discovery_core import short_error
from research_r10 import records

from grounded_hallucination_lab.data.ragtruth import gold_label
from grounded_hallucination_lab.research.metrics import bootstrap, metrics, span_metrics
from grounded_hallucination_lab.research.protocol import ROOT, read_json, read_rows, write_json

BASE = ROOT / "models/discovery_loop_20261026"
PRED = ROOT / "predictions/discovery_loop_20261026"
REPORT = ROOT / "reports/discovery_loop_20261026"
THRESHOLDS = {"B7": 0.37612172961235046, "FactCG": 0.5}


def local(row, start, end):
    return max(
        (
            p
            for (a, b), p in zip(row["offsets"], row["token_probabilities"], strict=True)
            if a < end and b > start
        ),
        default=0,
    )


def pair_statistics(rr, systems, thresholds, design):
    pairs = defaultdict(dict)
    for i, (r, d) in enumerate(zip(rr, design, strict=True)):
        assert r.id == d["id"]
        pairs[d["pair_id"]][gold_label(r)] = i
    groups, values = [], {name: [] for name in systems}
    for pair in pairs.values():
        assert set(pair) == {0, 1}
        a, b = pair[0], pair[1]
        assert rr[a].response == rr[b].response
        assert rr[a].source_id == rr[b].source_id
        groups.append(rr[a].source_id)
        d = design[a]
        start, end = d.get("slot_start", d["claim_start"]), d.get("slot_end", d["claim_end"])
        for name, rows in systems.items():
            threshold = thresholds[name]
            values[name].append(
                [
                    float(rows[a]["score"] < threshold <= rows[b]["score"]),
                    rows[b]["score"] - rows[a]["score"],
                    local(rows[b], start, end) - local(rows[a], start, end),
                ]
            )
    arrays = {m: np.array(v) for m, v in values.items()}
    keys = ["pair_accuracy", "response_gap", "local_gap"]
    result = {
        m: dict(zip(keys, np.mean(v, axis=0).tolist(), strict=True)) for m, v in arrays.items()
    }
    unique = sorted(set(groups))
    idx = np.array([unique.index(g) for g in groups])
    draws = {m: [] for m in arrays}
    differences = {m: [] for m in arrays if m != "B7"}
    rng = np.random.default_rng(42)
    for _ in range(2000):
        counts = np.bincount(rng.integers(0, len(unique), len(unique)), minlength=len(unique))
        weighted = {m: np.average(a, axis=0, weights=counts[idx]) for m, a in arrays.items()}
        for m, v in weighted.items():
            draws[m].append(v)
            if m != "B7":
                differences[m].append(v - weighted["B7"])
    for m in result:
        result[m]["ci95"] = dict(
            zip(keys, np.percentile(draws[m], [2.5, 97.5], axis=0).T.tolist(), strict=True)
        )
        if m != "B7":
            result[m]["delta_B7_ci95"] = dict(
                zip(
                    keys, np.percentile(differences[m], [2.5, 97.5], axis=0).T.tolist(), strict=True
                )
            )
    return {"pairs": len(pairs), "sources": len(unique), "models": result}


def analyze(rr, systems, thresholds, design=None):
    y = [gold_label(r) for r in rr]
    sources = [r.source_id for r in rr]
    for rows in systems.values():
        assert [r.id for r in rr] == [p["id"] for p in rows]
    result = {
        "n": len(rr),
        "sources": len(set(sources)),
        "positive": sum(y),
        "metrics": {},
        "localization": {},
        "short_error": {},
    }
    for m, rows in systems.items():
        result["metrics"][m] = metrics(y, [p["score"] for p in rows], thresholds[m])
        result["localization"][m] = span_metrics(rr, rows)
        short = [(r, p) for r, p in zip(rr, rows, strict=True) if short_error(r)]
        result["short_error"][m] = {
            "n": len(short),
            "tp": sum(p["score"] >= thresholds[m] for r, p in short),
            "gold_token_hit": sum(
                any(local(p, a.start, a.end) >= 0.5 for a in r.annotations) for r, p in short
            ),
        }
    result["bootstrap"] = bootstrap(
        y,
        sources,
        {
            m: ([int(p["score"] >= thresholds[m]) for p in rows], [p["score"] for p in rows])
            for m, rows in systems.items()
        },
        "B7",
    )
    if design is not None:
        result["paired"] = pair_statistics(rr, systems, thresholds, design)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cycle", default="cycle2")
    args = parser.parse_args()
    results = {}
    for label, path in {
        "natural": ROOT / "models/phase2b_r10/natural_records.jsonl",
        "r1": ROOT / "models/phase2b_r10/r1_records.jsonl",
        "r6": ROOT / "predictions/phase2b_r6/transfer_records.jsonl",
    }.items():
        rr = records(path)
        old = read_rows(ROOT / "predictions/phase2b_r10" / f"B7_{label}.jsonl")
        new = read_rows(PRED / f"{args.cycle}_FactCG_{label}.jsonl")
        design = (
            None
            if label == "natural"
            else read_json(
                ROOT / "models/phase2b_r10/r1_design.json"
                if label == "r1"
                else ROOT / "predictions/phase2b_r6/transfer_design.json"
            )
        )
        results[label] = analyze(rr, {"B7": old, "FactCG": new}, THRESHOLDS, design)
        print(label, {m: v["macro_f1"] for m, v in results[label]["metrics"].items()}, flush=True)
    write_json(REPORT / f"{args.cycle}_analysis.json", results)


if __name__ == "__main__":
    main()
