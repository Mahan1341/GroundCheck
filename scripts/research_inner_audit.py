"""Descriptive inner-validation diagnostics, never an inference feature or new candidate."""

import re

import numpy as np
from scipy.stats import spearmanr

from grounded_hallucination_lab.data.ragtruth import gold_label
from grounded_hallucination_lab.research.protocol import (
    PRED,
    REPORT,
    load_parts,
    partition,
    read_json,
    read_rows,
    write_json,
)
from grounded_hallucination_lab.research.segments import clean_segments


def main():
    records = partition(load_parts(), "inner_val")
    by_id = {r.id: r for r in records}
    result = {}
    for method in ("B1", "B2", "B4", "B5"):
        rows = read_rows(PRED / f"{method}_inner_val.jsonl")
        threshold = read_json(PRED / f"{method}_inner_val.meta.json")["inner_selection"][
            "threshold"
        ]
        false_alarms = []
        short_misses = []
        for row in rows:
            record = by_id[row["id"]]
            label, pred = gold_label(record), row["score"] >= threshold
            if pred and not label:
                worst = min(row["segments"], key=lambda s: s["support"])
                _, all_drops = clean_segments(record.response)
                drops = [
                    d for d in all_drops if d["start"] < worst["end"] and d["end"] > worst["start"]
                ]
                false_alarms.append(
                    {
                        "id": row["id"],
                        "bare_list_marker": bool(re.fullmatch(r"\s*\d+[.)]\s*", worst["text"])),
                        "cleaner_would_drop_any_part": bool(drops),
                        "segment_characters": len(worst["text"]),
                        "fragments": len(row["segments"]),
                    }
                )
            if label and not pred:
                short_misses.append(
                    {
                        "id": row["id"],
                        "shortest_gold_span_characters": min(
                            a.end - a.start for a in record.annotations
                        ),
                        "answer_characters": len(record.response),
                    }
                )
        result[method] = {
            "false_positives": len(false_alarms),
            "fp_worst_is_bare_list_marker": sum(r["bare_list_marker"] for r in false_alarms),
            "fp_worst_triggers_diagnostic_cleaner": sum(
                r["cleaner_would_drop_any_part"] for r in false_alarms
            ),
            "false_negatives": len(short_misses),
            "fn_with_gold_span_at_most20_characters": sum(
                r["shortest_gold_span_characters"] <= 20 for r in short_misses
            ),
            "risk_fragment_count_spearman": float(
                spearmanr([r["score"] for r in rows], [len(r["segments"]) for r in rows]).statistic
            ),
            "mean_fragments": float(np.mean([len(r["segments"]) for r in rows])),
            "false_alarm_diagnostics": false_alarms,
            "miss_diagnostics": short_misses,
            "caveat": "descriptive triggers, not manually assigned causal error categories",
        }
    write_json(REPORT / "inner_error_audit_full_context.json", result)
    print({k: {a: b for a, b in v.items() if not isinstance(b, list)} for k, v in result.items()})


if __name__ == "__main__":
    main()
