"""Offline end-to-end reporting test; all texts and labels are synthetic fixtures."""

from grounded_hallucination_lab.data.ragtruth import gold_label
from grounded_hallucination_lab.data.splits import grouped_split
from grounded_hallucination_lab.research import evaluate as module
from grounded_hallucination_lab.research import run
from grounded_hallucination_lab.research.protocol import digest, partition, write_json, write_rows


def test_frozen_batch_report_end_to_end(dataset, tmp_path, monkeypatch):
    parts = grouped_split(dataset.records)
    records = partition(parts, "dev")
    out, pred, report = tmp_path / "models", tmp_path / "predictions/phase2b", tmp_path / "reports"
    for key, value in (("ROOT", tmp_path), ("OUT", out), ("PRED", pred), ("REPORT", report)):
        monkeypatch.setattr(module, key, value)
    methods = ["B2", "B4", "B6", "B7s42", "B7s43", "B7s44", "B8s42", "B9"]
    frozen = {"methods": methods, "provenance": {"synthetic": True}}
    monkeypatch.setattr(run, "check_freeze", lambda: frozen)
    write_json(out / "batch_freeze.json", frozen)
    point = {"available": False, "reason": "synthetic test"}
    thresholds = {
        "batch_freeze": digest(out / "batch_freeze.json"),
        "historical_precision_point": point,
        "methods": {
            m: {"selection": {"threshold": 0.5}, "precision_point": point} for m in methods
        },
    }
    write_json(out / "thresholds.json", thresholds)
    rows = [
        {
            "id": r.id,
            "source_id": r.source_id,
            "score": 0.9 if gold_label(r) else 0.1,
            "offsets": [[0, len(r.response)]],
            "token_probabilities": [0.9 if gold_label(r) else 0.1],
        }
        for r in records
    ]
    write_rows(tmp_path / "predictions/phase2a/dev.jsonl", rows)
    write_json(report / "lexical_dev.json", {"threshold": 0.5, "runtime": {"synthetic": True}})
    for method in methods:
        values = [dict(r, score=1 - r["score"]) for r in rows] if method == "B4" else rows
        target = pred / f"{method}_dev.jsonl"
        write_rows(target, values)
        write_json(
            target.with_suffix(".meta.json"),
            {
                "provenance": frozen["provenance"],
                "prediction_sha256": digest(target),
                "thresholds_sha256": digest(out / "thresholds.json"),
                "runtime": {"synthetic": True},
            },
        )
    result = module.evaluate(parts)
    assert result["n"] == len(records) and result["test_evaluated"] is False
    assert result["bootstrap"]["resamples"] == 2000
    assert result["methods"]["B2"]["metrics"]["accuracy"] == 1
    assert result["methods"]["B4"]["metrics"]["accuracy"] == 0
    assert result["methods"]["B7s42"]["span_task"]["official_scorer"] is False
    assert (pred / "error_examples.json").exists()
