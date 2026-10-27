"""Post-run paired localization CIs, short-error diagnostics and saved error transitions."""

import argparse
from dataclasses import asdict

import numpy as np

from grounded_hallucination_lab.data.ragtruth import gold_label
from grounded_hallucination_lab.research.metrics import span_metrics
from grounded_hallucination_lab.research.protocol import (
    ROOT,
    load_parts,
    partition,
    read_rows,
    write_json,
)
from grounded_hallucination_lab.research.relations import relation_suite


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["natural", "diagnostic"], required=True)
    parser.add_argument("--methods", nargs="+", required=True)
    parser.add_argument("--reference", required=True)
    a = parser.parse_args()
    if any(m != "B7" and not m.startswith("R2C") for m in a.methods):
        raise ValueError("this audit uses the fixed B7 threshold; only B7 and R2 arms are allowed")
    records = (
        partition(load_parts(), "inner_val")
        if a.split == "natural"
        else relation_suite("diagnostic")[0]
    )
    sources = sorted({r.source_id for r in records})
    mapping = {s: i for i, s in enumerate(sources)}
    rng = np.random.default_rng(42)
    draws = rng.integers(0, len(sources), size=(2000, len(sources)))
    samples = {}
    short = {}
    saved = {}
    for method in a.methods:
        if method in ("B7", "B8", "C0", "C1") and a.split == "natural":
            folder = "phase2b_resume" if method in ("C0", "C1") else "phase2b"
            label = method if method in ("C0", "C1") else method + "s42"
            path = ROOT / f"predictions/{folder}/{label}_inner_val.jsonl"
        else:
            path = ROOT / f"predictions/phase2b_r2/{method}_{a.split}.jsonl"
        rows = read_rows(path)
        assert [r["id"] for r in rows] == [r.id for r in records]
        saved[method] = rows
        counts = {k: np.zeros((len(sources), 3)) for k in ("token", "character", "exact_span")}
        selected = []
        for record, row in zip(records, rows, strict=True):
            m = span_metrics([record], [row])
            for k in counts:
                counts[k][mapping[record.source_id]] += [m[k][s] for s in ("tp", "fp", "fn")]
            if (
                gold_label(record)
                and len(record.response) >= 500
                and sum(x.end - x.start for x in record.annotations) / len(record.response) <= 0.05
            ):
                peak = max(
                    (
                        p
                        for (x, y), p in zip(
                            row["offsets"], row["token_probabilities"], strict=True
                        )
                        if any(x < t.end and y > t.start for t in record.annotations)
                    ),
                    default=0.0,
                )
                selected.append(
                    {
                        "id": record.id,
                        "response_detected": row["score"] >= 0.37612172961235046,
                        "gold_token_hit": peak >= 0.37612172961235046,
                    }
                )
        short[method] = {
            "n": len(selected),
            "detected": sum(x["response_detected"] for x in selected),
            "localized": sum(x["gold_token_hit"] for x in selected),
            "rows": selected,
            "threshold": 0.37612172961235046,
            "role": "same B7 threshold for new arms/B7; no optimizing on this subgroup",
        }
        samples[method] = {}
        for k, table in counts.items():
            values = table[draws].sum(1)
            tp, fp, fn = values.T
            samples[method][k] = np.divide(
                2 * tp, 2 * tp + fp + fn, out=np.zeros(2000), where=(2 * tp + fp + fn) > 0
            )
    intervals = {
        m: {
            k: {
                "absolute95": np.percentile(v, [2.5, 97.5]).tolist(),
                "difference95": np.percentile(v - samples[a.reference][k], [2.5, 97.5]).tolist(),
            }
            for k, v in s.items()
        }
        for m, s in samples.items()
    }
    transitions = []
    if a.split == "natural":
        for i, r in enumerate(records):
            pred = {m: int(rows[i]["score"] >= 0.37612172961235046) for m, rows in saved.items()}
            if len(set(pred.values())) > 1:
                transitions.append(
                    {
                        "record": asdict(r),
                        "gold": gold_label(r),
                        "predictions": pred,
                        "scores": {m: rows[i]["score"] for m, rows in saved.items()},
                    }
                )
    suffix = "_".join(a.methods)
    write_json(
        ROOT / f"reports/phase2b_r2/localization_{a.split}_{suffix}.json",
        {
            "intervals": intervals,
            "reference": a.reference,
            "resamples": 2000,
            "sources": len(sources),
            "short_error_long_answer": short,
        },
    )
    if transitions:
        write_json(ROOT / f"predictions/phase2b_r2/errors_{suffix}.json", transitions)


if __name__ == "__main__":
    main()
