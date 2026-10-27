from collections import Counter
from types import SimpleNamespace

import pytest
from test_research import TinyTokenizer

from grounded_hallucination_lab.research.controls import controls, fact_cases
from grounded_hallucination_lab.research.protocol import InferenceInput, write_json
from grounded_hallucination_lab.research.span_model import windows


def test_counterfactual_pairs_keep_answer_and_bag_of_evidence_words():
    for fact in fact_cases():
        assert Counter(fact["supported"].split()) == Counter(fact["contradicted"].split())
    records, design = controls()
    assert len(records) == len(design) == 300
    assert len({r.source_id for r in records}) == 15
    by_condition = {}
    for r, d in zip(records, design, strict=True):
        key = (d["group"], d["padding_sentences"], d["position"])
        by_condition.setdefault(key, []).append(r)
        assert r.response[d["claim_start"] : d["claim_end"]]
        assert bool(r.annotations) == (d["condition"] == "contradicted")
        for a in r.annotations:
            assert r.response[a.start : a.end] == a.text
            assert d["claim_start"] <= a.start < a.end <= d["claim_end"]
    assert all(len({r.response for r in group}) == 1 for group in by_condition.values())


def test_answer_only_removes_prefix_not_supervised_targets():
    view = InferenceInput("r", "s", "question SECRET", "evidence SECRET", "Mira owns ten crates.")
    full, _ = windows(view, TinyTokenizer(), spans=[(10, 13)])
    answer, _ = windows(view, TinyTokenizer(), answer_only=True, spans=[(10, 13)])

    def supervised(rows):
        return [
            (y, w)
            for row in rows
            for y, w in zip(row["labels"], row["loss_weights"], strict=True)
            if y != -100
        ]

    assert supervised(full) == supervised(answer)
    assert len(answer[0]["input_ids"]) < len(full[0]["input_ids"])


def test_freeze_rejects_incomplete_or_heldout_trained_checkpoint(tmp_path, monkeypatch):
    from grounded_hallucination_lab.research import run

    monkeypatch.setattr(run, "OUT", tmp_path)
    write_json(tmp_path / "span_inner_s42/result.json", {"selected_epoch": 3})
    parts = SimpleNamespace(train=[SimpleNamespace(id="a", source_id="s")])
    with pytest.raises(ValueError, match="incomplete final fit"):
        run.validate_final_fits(parts, ["B7s42"])
    folder = tmp_path / "span_final_s42"
    config = {
        "seed": 42,
        "answer_only": False,
        "train_ids": ["a"],
        "train_sources": ["s"],
        "validation_ids": [],
        "select_epoch_on_validation": False,
        "max_epochs": 3,
    }
    write_json(folder / "training_config.json", config)
    write_json(
        folder / "result.json",
        {"config": config, "selected_epoch": 3, "history": [{"epoch": n} for n in (1, 2, 3)]},
    )
    write_json(folder / "best/config.json", {})
    (folder / "best/model.safetensors").write_bytes(b"fixture")
    run.validate_final_fits(parts, ["B7s42"])
    config["validation_ids"] = ["heldout"]
    write_json(folder / "training_config.json", config, overwrite=True)
    with pytest.raises(ValueError, match="protocol mismatch"):
        run.validate_final_fits(parts, ["B7s42"])


@pytest.mark.parametrize("fixed_epochs", [False, True])
def test_inner_epoch_selection_is_explicit(dataset, monkeypatch, tmp_path, fixed_epochs):
    from grounded_hallucination_lab.data.splits import grouped_split
    from grounded_hallucination_lab.research import run, span_model

    captured = {}

    def fake_train(fit, val, destination, **kwargs):
        captured.update(kwargs)
        assert {r.source_id for r in fit}.isdisjoint(r.source_id for r in val)
        return {"selected_epoch": 3, "training_seconds_including_validation": 0}

    monkeypatch.setattr(run, "OUT", tmp_path)
    monkeypatch.setattr(span_model, "train", fake_train)
    args = SimpleNamespace(
        stage="inner",
        epochs=3,
        answer_only=True,
        seed=42,
        max_length=2048,
        batch_size=1,
        fixed_epochs=fixed_epochs,
    )
    run.train_stage(args, grouped_split(dataset.records))
    assert captured["select_epoch"] is not fixed_epochs
    assert captured["epochs"] == 3
    assert captured["answer_only"] is True
