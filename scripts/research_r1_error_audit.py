"""Read-only R1 error export and localization at matched response thresholds."""

from dataclasses import asdict

from grounded_hallucination_lab.data.ragtruth import gold_label
from grounded_hallucination_lab.research.protocol import (
    PRED,
    ROOT,
    load_parts,
    partition,
    provenance,
    read_json,
    read_rows,
    write_json,
)


def main():
    records = partition(load_parts(), "inner_val")
    report = read_json(ROOT / "reports/phase2b_resume/natural_inner_comparison_cf.json")
    by_id = {r.id: r for r in records}
    results, errors = {}, {}
    for method in ("B7s42", "B8s42", "C0", "C1"):
        base = ROOT / "predictions/phase2b_resume" if method in ("C0", "C1") else PRED
        rows = read_rows(base / f"{method}_inner_val.jsonl")
        if {r["id"] for r in rows} != set(by_id) or len(rows) != len(records):
            raise ValueError("inner prediction membership mismatch")
        threshold = report["methods"][method]["metrics"]["threshold"]
        short = []
        fp, fn = [], []
        for row in rows:
            record = by_id[row["id"]]
            gold = gold_label(record)
            predicted = int(row["score"] >= threshold)
            if (
                gold
                and len(record.response) >= 500
                and (
                    sum(a.end - a.start for a in record.annotations) / len(record.response) <= 0.05
                )
            ):
                hit = any(
                    p >= threshold
                    and any(a < span.end and b > span.start for span in record.annotations)
                    for (a, b), p in zip(row["offsets"], row["token_probabilities"], strict=True)
                )
                short.append(
                    {
                        "id": record.id,
                        "detected": bool(predicted),
                        "gold_token_hit_at_response_threshold": hit,
                    }
                )
            if predicted != gold:
                error = {
                    "record": asdict(record),
                    "gold": gold,
                    "prediction": predicted,
                    "threshold": threshold,
                    "diagnostics": row,
                }
                (fp if predicted else fn).append(error)
        fp.sort(key=lambda x: (-x["diagnostics"]["score"], x["record"]["id"]))
        fn.sort(key=lambda x: (x["diagnostics"]["score"], x["record"]["id"]))
        results[method] = {
            "threshold": threshold,
            "long_short_n": len(short),
            "response_detected": sum(x["detected"] for x in short),
            "gold_token_hit_same_threshold": sum(
                x["gold_token_hit_at_response_threshold"] for x in short
            ),
            "detected_without_gold_token_hit": sum(
                x["detected"] and not x["gold_token_hit_at_response_threshold"] for x in short
            ),
            "false_positives": len(fp),
            "false_negatives": len(fn),
        }
        errors[method] = {"top15_fp": fp[:15], "top15_fn": fn[:15], "long_short": short}
    write_json(ROOT / "predictions/phase2b_resume/error_export_R1.json", errors)
    write_json(
        ROOT / "reports/phase2b_resume/error_localization_R1.json",
        {
            "role": "descriptive inner diagnostic; no new threshold/aggregation selection",
            "selection": "15 highest-risk FP and 15 lowest-risk FN per method; not random sample",
            "note": "gold-token hit is any overlap, not complete or accurate localization",
            "methods": results,
            "provenance": provenance(),
        },
    )
    print(results)


if __name__ == "__main__":
    main()
